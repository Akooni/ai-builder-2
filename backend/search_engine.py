"""
State-space search for PC builds.

State: a 6-tuple (cpu_i, mb_i, ram_i, storage_i, gpu_i, psu_i) using row indices
into each DataFrame. None means not yet assigned. gpu_i == NO_GPU means an
office-style build using only integrated graphics (no discrete GPU).
Expansion order is strictly CPU → Motherboard → RAM → Storage → GPU → PSU.

Before search: hard-pruned pools (gaming: ≥8 GB VRAM, CPUs ≥ $150, plus budget-aware CPU/GPU price bands so mid budgets
do not pair halo CPUs with entry GPUs; high-end: ≥8 GB VRAM and CPUs ≥ $150 only; office: iGPU-only, no dGPU rows),
then budget caps, then purpose-sorted tables. ``expand`` re-sorts successors by purpose (guided search).
Demo ``print`` lines (throttled) narrate partial paths and prunes. Budget UCS = cheapest (A*). Office UCS = cheapest.
All other UCS uses priority (budget − spent) to favor builds that use the budget (best goal by $ then perf score).
"""

from __future__ import annotations

import hashlib
import heapq
import logging
import time
import random
import re
from collections import deque
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from data_loader import yes_no

logger = logging.getLogger(__name__)

# Changes whenever this file is saved — use /api/health or build JSON to confirm the server reloaded your edits.
_SEARCH_ENGINE_PATH = Path(__file__).resolve()
try:
    SEARCH_ENGINE_FINGERPRINT = f"{_SEARCH_ENGINE_PATH.name}@{int(_SEARCH_ENGINE_PATH.stat().st_mtime)}"
except OSError:
    SEARCH_ENGINE_FINGERPRINT = "search_engine@unknown"
SEARCH_ENGINE_FILE = str(_SEARCH_ENGINE_PATH)

NO_GPU = -1
State6 = tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]

ORDER_KEYS = ("CPUs", "MBs", "RAMs", "Storage", "GPUs", "PSUs")

# Throttle demo stdout so logs look "deep" without flooding the console.
SMARTSEARCH_DEMO_INTERVAL = 900

# Limits scale with purpose; higher caps + performance-ordered candidates reduce "always $470" behavior.
# Budget needs a wide successor cap at storage/GPU layers: cheap SATA ranks before NVMe by price sort alone.
LIMITS_BUDGET = {
    "candidates": 32,
    "successors": 30,
    "expansions": 220_000,
    "visited": 220_000,
    "pops": 180_000,
}
LIMITS_DEFAULT = {"candidates": 20, "successors": 12, "expansions": 85_000, "visited": 65_000, "pops": 65_000}
LIMITS_HIGH_END = {"candidates": 32, "successors": 18, "expansions": 120_000, "visited": 95_000, "pops": 95_000}
# Graph search is capped; Monte Carlo fallback (_random_feasible_state) completes typical gaming queries quickly.
LIMITS_GAMING = {"candidates": 40, "successors": 32, "expansions": 120_000, "visited": 150_000, "pops": 45_000}

# Gaming builds above this budget get CPU max-price and GPU min-price bands (relaxed if the sheet would go empty).
_GAMING_BALANCE_MIN_BUDGET_USD = 750.0
# (max CPU as fraction of budget, min GPU as fraction). Later rows relax if no CPUs/GPUs survive both cuts.
_GAMING_PRICE_TIERS: tuple[tuple[float, float], ...] = (
    (0.35, 0.27),
    (0.38, 0.24),
    (0.42, 0.21),
    (0.48, 0.18),
    (0.55, 0.14),
)


def _select_gaming_price_caps(
    budget: float, cpus: pd.DataFrame, gpus: pd.DataFrame
) -> tuple[float, float] | None:
    """
    Pick (max_cpu_usd, min_gpu_usd) so mid/high gaming budgets reserve spend for the video card.
    Returns None if budget is below threshold or either sheet lacks prices / rows.
    """
    b = float(budget)
    if b < _GAMING_BALANCE_MIN_BUDGET_USD or len(cpus) == 0 or len(gpus) == 0:
        return None
    if "price_usd" not in cpus.columns or "price_usd" not in gpus.columns:
        return None
    cp = pd.to_numeric(cpus["price_usd"], errors="coerce")
    gp = pd.to_numeric(gpus["price_usd"], errors="coerce")
    for max_f, min_f in _GAMING_PRICE_TIERS:
        max_cpu = min(580.0, max(290.0, b * max_f))
        min_gpu = max(240.0, b * min_f)
        c_ok = cpus.loc[cp <= max_cpu + 1e-9]
        g_ok = gpus.loc[gp >= min_gpu - 1e-9]
        if len(c_ok) > 0 and len(g_ok) > 0:
            return (max_cpu, min_gpu)
    return None


class Purpose(str, Enum):
    GAMING = "gaming"
    OFFICE = "office"
    CONTENT_CREATION = "content_creation"
    AI_ML = "ai_ml"
    BUDGET = "budget"
    HIGH_END = "high_end"


class SearchAlgorithm(str, Enum):
    BFS = "bfs"
    DFS = "dfs"
    UCS = "ucs"
    ASTAR = "astar"


class SearchTimeoutError(RuntimeError):
    """Raised when a search exceeds the configured wall-clock time limit."""


def purpose_prefilter_catalog(
    tables: dict[str, pd.DataFrame], purpose: Purpose, budget: float | None = None
) -> tuple[dict[str, pd.DataFrame], tuple[float, float] | None]:
    """
    Purpose-specific row drops before budget / ordering (smaller search tree).

    High-End: GPUs with >= 8 GB VRAM; CPUs priced at >= $150.
    Gaming: same VRAM / CPU floor, then optional budget-aware CPU cap + GPU floor (see _select_gaming_price_caps).
    Office: only CPUs with integrated graphics; drop all discrete GPU rows (iGPU-only builds).

    Returns (stripped tables, gaming_balance_caps or None). Caps are (max_cpu_usd, min_gpu_usd) for validate/candidates.
    """
    out: dict[str, pd.DataFrame] = {name: df.copy().reset_index(drop=True) for name, df in tables.items()}
    gaming_caps: tuple[float, float] | None = None

    if purpose == Purpose.HIGH_END:
        g = out["GPUs"]
        if len(g) > 0 and "vram_gb" in g.columns:
            v = pd.to_numeric(g["vram_gb"], errors="coerce").fillna(0)
            keep = g.loc[v >= 8].copy().reset_index(drop=True)
            if len(keep) == 0 and len(g) > 0:
                keep = g.copy().reset_index(drop=True)
            out["GPUs"] = keep
        c = out["CPUs"]
        if len(c) > 0 and "price_usd" in c.columns:
            p = pd.to_numeric(c["price_usd"], errors="coerce")
            keep_c = c.loc[p >= 150.0].copy().reset_index(drop=True)
            if len(keep_c) == 0 and len(c) > 0:
                keep_c = c.copy().reset_index(drop=True)
            out["CPUs"] = keep_c

    elif purpose == Purpose.GAMING:
        g = out["GPUs"]
        if len(g) > 0 and "vram_gb" in g.columns:
            v = pd.to_numeric(g["vram_gb"], errors="coerce").fillna(0)
            keep = g.loc[v >= 8].copy().reset_index(drop=True)
            if len(keep) == 0 and len(g) > 0:
                keep = g.copy().reset_index(drop=True)
            out["GPUs"] = keep
        c = out["CPUs"]
        if len(c) > 0 and "price_usd" in c.columns:
            p = pd.to_numeric(c["price_usd"], errors="coerce")
            keep_c = c.loc[p >= 150.0].copy().reset_index(drop=True)
            if len(keep_c) == 0 and len(c) > 0:
                keep_c = c.copy().reset_index(drop=True)
            out["CPUs"] = keep_c
        if budget is not None:
            gaming_caps = _select_gaming_price_caps(float(budget), out["CPUs"], out["GPUs"])
            if gaming_caps is not None:
                mx, mn = gaming_caps
                c2 = out["CPUs"]
                g2 = out["GPUs"]
                if len(c2) > 0 and "price_usd" in c2.columns:
                    pc = pd.to_numeric(c2["price_usd"], errors="coerce")
                    c_f = c2.loc[pc <= mx + 1e-9].copy().reset_index(drop=True)
                    if len(c_f) > 0:
                        out["CPUs"] = c_f
                if len(g2) > 0 and "price_usd" in g2.columns:
                    pg = pd.to_numeric(g2["price_usd"], errors="coerce")
                    g_f = g2.loc[pg >= mn - 1e-9].copy().reset_index(drop=True)
                    if len(g_f) > 0:
                        out["GPUs"] = g_f

    elif purpose == Purpose.OFFICE:
        c = out["CPUs"]
        if len(c) > 0:
            ig = []
            for i in range(len(c)):
                if _cpu_has_igpu(c.iloc[i]):
                    ig.append(i)
            keep_c = c.iloc[ig].copy().reset_index(drop=True) if ig else c.iloc[0:0].copy()
            if len(keep_c) == 0 and len(c) > 0:
                keep_c = c.copy().reset_index(drop=True)
            out["CPUs"] = keep_c
        g = out["GPUs"]
        if len(g) > 0:
            out["GPUs"] = g.iloc[0:0].copy().reset_index(drop=True)

    return out, gaming_caps


def prefilter_tables_by_budget(
    tables: dict[str, pd.DataFrame],
    budget: float,
    max_fraction: float = 0.6,
    purpose: Purpose | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Remove parts priced above a per-budget cap (keeps rows if a sheet would go empty).

    Gaming / high-end / AI-ML use a higher cap so a single GPU can use most of the budget
    (e.g. $1500 gaming builds are not capped at $900 for the GPU row).
    """
    eff = max_fraction
    if purpose == Purpose.GAMING:
        eff = min(0.92, max_fraction + 0.32)
    elif purpose in (Purpose.HIGH_END, Purpose.AI_ML):
        eff = min(0.9, max_fraction + 0.28)
    cap = max(float(budget) * eff, 1.0)
    out: dict[str, pd.DataFrame] = {}
    for name, df in tables.items():
        if len(df) == 0 or "price_usd" not in df.columns:
            out[name] = df.copy().reset_index(drop=True)
            continue
        s = pd.to_numeric(df["price_usd"], errors="coerce")
        keep = df.loc[s <= cap].copy().reset_index(drop=True)
        if len(keep) == 0 and len(df) > 0:
            keep = df.copy().reset_index(drop=True)
        out[name] = keep
    return out


def order_tables_for_purpose(tables: dict[str, pd.DataFrame], purpose: Purpose) -> dict[str, pd.DataFrame]:
    """
    Reorder rows so successor generation follows purpose-aware ordering (CPU → … → PSU expansion order unchanged).
    """
    out: dict[str, pd.DataFrame] = {name: df.copy().reset_index(drop=True) for name, df in tables.items()}

    def price_asc(key: str) -> None:
        d = out[key]
        if len(d) == 0 or "price_usd" not in d.columns:
            return
        out[key] = d.sort_values(by=["price_usd"], ascending=[True], kind="mergesort").reset_index(drop=True)

    def price_desc(key: str) -> None:
        d = out[key]
        if len(d) == 0 or "price_usd" not in d.columns:
            return
        out[key] = d.sort_values(by=["price_usd"], ascending=[False], kind="mergesort").reset_index(drop=True)

    if purpose in (Purpose.OFFICE, Purpose.BUDGET):
        for key in ORDER_KEYS:
            price_asc(key)
        return out

    if purpose == Purpose.HIGH_END:
        for key in ORDER_KEYS:
            price_desc(key)
        return out

    if purpose == Purpose.CONTENT_CREATION:
        c = out["CPUs"]
        if len(c) > 0:
            by = [x for x in ("threads", "cores", "boost_clock_ghz", "price_usd") if x in c.columns]
            if by:
                out["CPUs"] = c.sort_values(by=by, ascending=[False] * len(by), kind="mergesort").reset_index(drop=True)
        return out

    if purpose == Purpose.GAMING:
        # Strong GPUs first; cheap supporting parts + adequate PSUs first so budget goes to the GPU.
        for key in ("CPUs", "MBs", "RAMs", "Storage"):
            price_asc(key)
        g = out["GPUs"]
        if len(g) > 0 and "price_usd" in g.columns:
            by = ["price_usd"]
            if "vram_gb" in g.columns:
                by.append("vram_gb")
            out["GPUs"] = g.sort_values(by=by, ascending=[False] * len(by), kind="mergesort").reset_index(drop=True)
        price_asc("PSUs")
        return out

    if purpose == Purpose.AI_ML:
        g = out["GPUs"]
        if len(g) > 0 and "vram_gb" in g.columns:
            by = ["vram_gb"]
            if "price_usd" in g.columns:
                by.append("price_usd")
            out["GPUs"] = g.sort_values(by=by, ascending=[False] * len(by), kind="mergesort").reset_index(drop=True)
        ps = out["PSUs"]
        if len(ps) > 0 and "wattage" in ps.columns:
            byp = ["wattage"]
            if "price_usd" in ps.columns:
                byp.append("price_usd")
            out["PSUs"] = ps.sort_values(by=byp, ascending=[False, False], kind="mergesort").reset_index(drop=True)
        return out

    return out


@dataclass(frozen=True)
class BuildResult:
    cpu: dict
    motherboard: dict
    ram: dict
    storage: dict
    gpu: Optional[dict]
    psu: dict
    total_price: float
    required_psu_watts: int
    psu_headroom_watts: int
    algorithm: str
    purpose: str
    notes: list[str]


def _row_to_dict(df: pd.DataFrame, idx: int) -> dict:
    row = df.iloc[idx]
    out: dict = {}
    for k, v in row.items():
        if pd.isna(v):
            out[k] = None
        elif hasattr(v, "item"):
            try:
                out[k] = v.item()
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def _state_depth(state: State6) -> int:
    for i, v in enumerate(state):
        if v is None:
            return i
    return 6


def _total_price(state: State6, tables: dict[str, pd.DataFrame]) -> float:
    total = 0.0
    for k, idx in zip(ORDER_KEYS, state):
        if idx is None or idx == NO_GPU:
            continue
        total += float(tables[k].iloc[idx]["price_usd"])
    return total


def _cpu_has_igpu(cpu_row: pd.Series) -> bool:
    return yes_no(cpu_row.get("integrated_graphics", False))


def _storage_ok_for_mb(st_row: pd.Series, mb_row: pd.Series) -> bool:
    iface = str(st_row.get("interface", "")).strip().upper()
    if iface in ("EXTERNAL", "USB", "THUNDERBOLT"):
        return True
    if iface in ("NVME", "M.2", "PCIE NVME"):
        m2 = int(mb_row.get("m2_slots", 0) or 0)
        nvme = yes_no(mb_row.get("nvme_support", False))
        return m2 >= 1 and nvme
    if iface == "SATA":
        return int(mb_row.get("sata_ports", 0) or 0) >= 1
    return True


def _pcie_ok(gpu_row: pd.Series, mb_row: pd.Series) -> bool:
    try:
        gv = int(float(gpu_row.get("pcie_version", 0) or 0))
        mv = int(float(mb_row.get("pcie_version", 0) or 0))
    except (TypeError, ValueError):
        return True
    return mv >= gv


def _psu_meets_load(psu_row: pd.Series, cpu_row: pd.Series, gpu_row: Optional[pd.Series]) -> bool:
    """PSU must satisfy (cpu_tdp + gpu_tdp + 20W safety buffer) <= rated wattage."""
    cpu_tdp = int(float(cpu_row.get("tdp_watts", 0) or 0))
    gpu_tdp = int(float(gpu_row.get("tdp_watts", 0) or 0)) if gpu_row is not None else 0
    required = cpu_tdp + gpu_tdp + 20
    watt = int(float(psu_row.get("wattage", 0) or 0))
    return watt >= required


# --- Performance scoring (higher = better). Used for high-end / gaming / content / AI / office UCS & ordering ---


def _row_perf_cpu(row: pd.Series) -> float:
    return (
        float(row.get("cores", 0) or 0) * 22.0
        + float(row.get("threads", 0) or 0) * 3.5
        + float(row.get("boost_clock_ghz", 0) or 0) * 14.0
        + float(row.get("base_clock_ghz", 0) or 0) * 5.0
        + (40.0 if _cpu_has_igpu(row) else 0.0)
    )


def _row_perf_mb(row: pd.Series) -> float:
    pcie = float(row.get("pcie_version", 0) or 0)
    return (
        pcie * 45.0
        + float(row.get("m2_slots", 0) or 0) * 18.0
        + float(row.get("sata_ports", 0) or 0) * 2.0
        + (35.0 if yes_no(row.get("nvme_support", False)) else 0.0)
        + float(row.get("ram_slots", 0) or 0) * 6.0
    )


def _row_perf_ram(row: pd.Series) -> float:
    return float(row.get("capacity_gb", 0) or 0) * 1.35 + float(row.get("speed_mhz", 0) or 0) * 0.022


def _row_perf_storage(row: pd.Series) -> float:
    return float(row.get("read_mbps", 0) or 0) * 0.018 + float(row.get("capacity_gb", 0) or 0) * 0.06


def _row_perf_gpu(row: pd.Series) -> float:
    return (
        float(row.get("vram_gb", 0) or 0) * 58.0
        + float(row.get("tdp_watts", 0) or 0) * 0.55
        + float(row.get("pcie_version", 0) or 0) * 28.0
    )


def _row_perf_psu(row: pd.Series) -> float:
    eff = str(row.get("efficiency", "")).lower()
    tier = 0.0
    if "titanium" in eff:
        tier = 220.0
    elif "platinum" in eff:
        tier = 160.0
    elif "gold" in eff:
        tier = 110.0
    elif "bronze" in eff:
        tier = 40.0
    return float(row.get("wattage", 0) or 0) * 0.18 + tier


def _row_perf_for_key(key: str, row: pd.Series) -> float:
    if key == "CPUs":
        return _row_perf_cpu(row)
    if key == "MBs":
        return _row_perf_mb(row)
    if key == "RAMs":
        return _row_perf_ram(row)
    if key == "Storage":
        return _row_perf_storage(row)
    if key == "GPUs":
        return _row_perf_gpu(row)
    if key == "PSUs":
        return _row_perf_psu(row)
    return 0.0


def _purpose_budget_minimizes_price(purpose: Purpose) -> bool:
    """Only the Budget purpose optimizes for lowest total price."""
    return purpose == Purpose.BUDGET


def _high_end_value_name(name: object) -> bool:
    """Heuristic: skip obvious entry-tier product names for high-end builds."""
    n = str(name).lower()
    patterns = (
        r"valueram",
        r"\bvalue\b",
        r"basics\b",
        r"\ba400\b",
        r"\bh610\b",
        r"prime a520",
        r"a520m",
        r"elite 600",
        r"\bvs600\b",
        r"cv450",
        r"cv550",
        r"500 w1",
    )
    return any(re.search(p, n) for p in patterns)


def _limits(purpose: Purpose) -> dict[str, int]:
    if purpose == Purpose.BUDGET:
        return LIMITS_BUDGET
    if purpose == Purpose.HIGH_END:
        return LIMITS_HIGH_END
    if purpose == Purpose.GAMING:
        return LIMITS_GAMING
    return LIMITS_DEFAULT


class SearchEngine:
    """Purpose-ordered component tables and BFS / DFS / UCS over compatible partial builds."""

    ORDER = ORDER_KEYS

    def __init__(
        self,
        tables: dict[str, pd.DataFrame],
        diversity_seed: int = 0,
        gaming_balance_caps: tuple[float, float] | None = None,
    ) -> None:
        self.tables = tables
        self._diversity_seed = int(diversity_seed) & 0x7FFFFFFF
        self._gaming_max_cpu_usd: float | None = None
        self._gaming_min_gpu_usd: float | None = None
        if gaming_balance_caps is not None:
            self._gaming_max_cpu_usd, self._gaming_min_gpu_usd = (
                float(gaming_balance_caps[0]),
                float(gaming_balance_caps[1]),
            )
        self.cpus = tables["CPUs"]
        self.mbs = tables["MBs"]
        self.rams = tables["RAMs"]
        self.storage = tables["Storage"]
        self.gpus = tables["GPUs"]
        self.psus = tables["PSUs"]
        # Admissible tail for budget A*: sum of global minimum USD per remaining slot (underestimates true remaining cost).
        self._budget_tail_floor: list[float] = [0.0] * 7
        for d in range(5, -1, -1):
            key = self.ORDER[d]
            df = self.tables[key]
            if len(df) == 0 or "price_usd" not in df.columns:
                mn = 0.0
            else:
                mn = float(df["price_usd"].min())
            self._budget_tail_floor[d] = mn + self._budget_tail_floor[d + 1]
        self._demo_prune_budget = 0
        self._demo_prune_projected = 0

    def _row_short_name(self, key: str, idx: int) -> str:
        if idx == NO_GPU:
            return "iGPU (no dGPU)"
        row = self.tables[key].iloc[int(idx)]
        for col in ("name", "cpu_id", "mb_id", "ram_id", "storage_id", "gpu_id", "psu_id"):
            if col in row.index and row[col] is not None and str(row[col]).strip() != "":
                return str(row[col])[:48]
        return f"{key}[{idx}]"

    def _demo_path_line(self, state: State6, max_parts: int = 4) -> str:
        parts: list[str] = []
        for depth in range(min(6, max_parts)):
            idx = state[depth]
            if idx is None:
                break
            key = self.ORDER[depth]
            parts.append(self._row_short_name(key, int(idx)))
        return " + ".join(parts) if parts else "(empty)"

    def _sort_successors_for_purpose(self, states: list[State6], depth: int, purpose: Purpose) -> None:
        """Guided search: order expanded children by purpose (not raw CSV order)."""
        if len(states) < 2:
            return
        key = self.ORDER[depth]

        def price_at(s: State6) -> float:
            li = s[depth]
            if li is None or li == NO_GPU:
                return 0.0
            return float(self.tables[key].iloc[int(li)].get("price_usd", 0) or 0)

        reverse = False
        if purpose in (Purpose.OFFICE, Purpose.BUDGET):
            reverse = False
        elif purpose == Purpose.HIGH_END:
            reverse = True
        elif purpose == Purpose.GAMING:
            reverse = key == "GPUs"
        elif purpose in (Purpose.CONTENT_CREATION, Purpose.AI_ML):
            reverse = key == "GPUs"
        states.sort(key=price_at, reverse=reverse)

    def _budget_astar_priority(self, state: State6) -> float:
        """f = g + h with g=spent USD and h=sum of global per-category price floors for unfilled slots."""
        d = _state_depth(state)
        g = _total_price(state, self.tables)
        if d >= 6:
            return g
        return g + self._budget_tail_floor[d]

    def _min_remaining_spend(self, assign_at_depth: int, purpose: Purpose) -> float:
        """Lower bound on USD still needed to complete the build from slot ``assign_at_depth + 1`` onward."""
        if assign_at_depth >= 5:
            return 0.0
        if purpose == Purpose.OFFICE:
            total = 0.0
            for jd in range(assign_at_depth + 1, 6):
                key = self.ORDER[jd]
                if key == "GPUs":
                    continue
                df = self.tables[key]
                if len(df) > 0 and "price_usd" in df.columns:
                    total += float(df["price_usd"].min())
            return total
        return self._budget_tail_floor[assign_at_depth + 1]

    def _shuffle_successor_blocks(self, states: list[State6], depth: int, purpose: Purpose) -> None:
        """Shuffle only within equal rounded-price buckets so purpose ordering stays mostly intact."""
        if len(states) < 2 or self._diversity_seed == 0:
            return
        key = self.ORDER[depth]
        rng = random.Random(self._diversity_seed ^ (depth * 1_009_033) ^ sum(map(ord, purpose.value)))

        def bucket(s: State6) -> float:
            li = s[depth]
            if li is None or li == NO_GPU:
                return float("-inf")
            row = self.tables[key].iloc[int(li)]
            if key == "PSUs" and purpose == Purpose.AI_ML:
                return float(round(float(row.get("wattage", 0) or 0), -1))
            p = row.get("price_usd")
            if p is not None and not (isinstance(p, float) and pd.isna(p)):
                return float(round(float(p), -1))
            return 0.0

        states.sort(key=bucket)
        i = 0
        while i < len(states):
            j = i + 1
            bi = bucket(states[i])
            while j < len(states) and bucket(states[j]) == bi:
                j += 1
            block = states[i:j]
            if len(block) > 1:
                rng.shuffle(block)
            states[i:j] = block
            i = j

    def _state_performance_total(self, state: State6) -> float:
        s = 0.0
        for depth, key in enumerate(self.ORDER):
            idx = state[depth]
            if idx is None:
                break
            if key == "GPUs" and idx == NO_GPU:
                continue
            s += _row_perf_for_key(key, self.tables[key].iloc[int(idx)])
        return s

    def _select_candidates(self, key: str, df: pd.DataFrame, idxs: list[int], purpose: Purpose) -> list[int]:
        lim = _limits(purpose)["candidates"]
        if not idxs:
            return []
        # DataFrames are pre-sorted for the active purpose (order_tables_for_purpose); preserve row order.
        return idxs[:lim]

    # --- purpose-based pruning (candidate filtering before compatibility) ---

    def _cpu_candidates(self, purpose: Purpose) -> list[int]:
        idxs = list(range(len(self.cpus)))
        s = self.cpus["price_usd"]

        def pct(p: float) -> float:
            return float(pd.Series(s).quantile(p))

        if purpose == Purpose.OFFICE:
            idxs = [i for i in idxs if _cpu_has_igpu(self.cpus.iloc[i])]
        elif purpose == Purpose.CONTENT_CREATION:
            idxs = [i for i in idxs if int(self.cpus.iloc[i].get("cores", 0) or 0) >= 6]
        elif purpose == Purpose.BUDGET:
            cap = pct(0.78)
            idxs = [i for i in idxs if float(self.cpus.iloc[i]["price_usd"]) <= cap]
        elif purpose == Purpose.HIGH_END:
            idxs = [
                i
                for i in idxs
                if (float(self.cpus.iloc[i]["price_usd"]) >= 200 or int(self.cpus.iloc[i].get("cores", 0) or 0) >= 10)
                and not _high_end_value_name(self.cpus.iloc[i].get("name", ""))
            ]
        elif purpose == Purpose.GAMING:
            idxs = [i for i in idxs if int(self.cpus.iloc[i].get("cores", 0) or 0) >= 6]
            if self._gaming_max_cpu_usd is not None:
                cap = self._gaming_max_cpu_usd
                idxs = [i for i in idxs if float(self.cpus.iloc[i].get("price_usd", 0) or 0) <= cap + 1e-9]
        elif purpose == Purpose.AI_ML:
            idxs = [i for i in idxs if int(self.cpus.iloc[i].get("cores", 0) or 0) >= 8]
        return self._select_candidates("CPUs", self.cpus, idxs, purpose)

    def _mb_candidates(self, purpose: Purpose) -> list[int]:
        idxs = list(range(len(self.mbs)))
        s = self.mbs["price_usd"]

        def pct(p: float) -> float:
            return float(pd.Series(s).quantile(p))

        if purpose == Purpose.BUDGET:
            cap = pct(0.82)
            idxs = [i for i in idxs if float(self.mbs.iloc[i]["price_usd"]) <= cap]
        elif purpose == Purpose.HIGH_END:
            idxs = [
                i
                for i in idxs
                if float(self.mbs.iloc[i]["price_usd"]) >= 180
                and not _high_end_value_name(self.mbs.iloc[i].get("name", ""))
            ]
        return self._select_candidates("MBs", self.mbs, idxs, purpose)

    def _ram_candidates(self, purpose: Purpose) -> list[int]:
        idxs = list(range(len(self.rams)))
        s = self.rams["price_usd"]

        def pct(p: float) -> float:
            return float(pd.Series(s).quantile(p))

        if purpose == Purpose.CONTENT_CREATION:
            idxs = [i for i in idxs if int(self.rams.iloc[i].get("capacity_gb", 0) or 0) >= 16]
        elif purpose == Purpose.AI_ML:
            idxs = [i for i in idxs if int(self.rams.iloc[i].get("capacity_gb", 0) or 0) >= 32]
        elif purpose == Purpose.BUDGET:
            cap = pct(0.82)
            idxs = [i for i in idxs if float(self.rams.iloc[i]["price_usd"]) <= cap]
        elif purpose == Purpose.HIGH_END:
            idxs = [
                i
                for i in idxs
                if int(self.rams.iloc[i].get("capacity_gb", 0) or 0) >= 32 or float(self.rams.iloc[i]["price_usd"]) >= 160
            ]
            idxs = [i for i in idxs if not _high_end_value_name(self.rams.iloc[i].get("name", ""))]
        elif purpose == Purpose.GAMING:
            idxs = [i for i in idxs if int(self.rams.iloc[i].get("capacity_gb", 0) or 0) >= 16]
        return self._select_candidates("RAMs", self.rams, idxs, purpose)

    def _storage_candidates(self, purpose: Purpose) -> list[int]:
        idxs = list(range(len(self.storage)))
        s = self.storage["price_usd"]

        def pct(p: float) -> float:
            return float(pd.Series(s).quantile(p))

        idxs = [
            i
            for i in idxs
            if str(self.storage.iloc[i].get("interface", "")).strip().upper() not in ("EXTERNAL",)
        ]
        if purpose == Purpose.BUDGET:
            cap = pct(0.85)
            idxs = [i for i in idxs if float(self.storage.iloc[i]["price_usd"]) <= cap]
        elif purpose == Purpose.HIGH_END:
            idxs = [
                i
                for i in idxs
                if float(self.storage.iloc[i]["price_usd"]) >= 100 or int(self.storage.iloc[i].get("read_mbps", 0) or 0) >= 3000
            ]
            idxs = [i for i in idxs if not _high_end_value_name(self.storage.iloc[i].get("name", ""))]
        elif purpose == Purpose.GAMING:
            idxs = [
                i
                for i in idxs
                if int(self.storage.iloc[i].get("read_mbps", 0) or 0) >= 400 or float(self.storage.iloc[i]["price_usd"]) >= 55
            ]
        return self._select_candidates("Storage", self.storage, idxs, purpose)

    def _gpu_candidates(self, purpose: Purpose, cpu_idx: int) -> list[int]:
        idxs = list(range(len(self.gpus)))
        s = self.gpus["price_usd"]

        def pct(p: float) -> float:
            return float(pd.Series(s).quantile(p))

        if purpose == Purpose.GAMING:
            idxs = [i for i in idxs if int(self.gpus.iloc[i].get("vram_gb", 0) or 0) >= 8]
            if self._gaming_min_gpu_usd is not None:
                lo = self._gaming_min_gpu_usd
                idxs = [i for i in idxs if float(self.gpus.iloc[i].get("price_usd", 0) or 0) >= lo - 1e-9]
        elif purpose == Purpose.AI_ML:
            idxs = [i for i in idxs if int(self.gpus.iloc[i].get("vram_gb", 0) or 0) >= 12]
        elif purpose == Purpose.BUDGET:
            cap = pct(0.78)
            idxs = [i for i in idxs if float(self.gpus.iloc[i]["price_usd"]) <= cap]
        elif purpose == Purpose.HIGH_END:
            idxs = [
                i
                for i in idxs
                if float(self.gpus.iloc[i]["price_usd"]) >= 500 or int(self.gpus.iloc[i].get("vram_gb", 0) or 0) >= 12
            ]
            idxs = [i for i in idxs if not _high_end_value_name(self.gpus.iloc[i].get("name", ""))]
        elif purpose == Purpose.OFFICE:
            return []
        elif purpose == Purpose.CONTENT_CREATION:
            idxs = [i for i in idxs if int(self.gpus.iloc[i].get("vram_gb", 0) or 0) >= 6]
        return self._select_candidates("GPUs", self.gpus, idxs, purpose)

    def _psu_candidates(self, purpose: Purpose) -> list[int]:
        idxs = list(range(len(self.psus)))
        s = self.psus["price_usd"]

        def pct(p: float) -> float:
            return float(pd.Series(s).quantile(p))

        if purpose == Purpose.AI_ML:
            idxs = [i for i in idxs if int(float(self.psus.iloc[i].get("wattage", 0) or 0)) >= 750]
        if purpose == Purpose.BUDGET:
            cap = pct(0.82)
            idxs = [i for i in idxs if float(self.psus.iloc[i]["price_usd"]) <= cap]
        elif purpose == Purpose.HIGH_END:
            idxs = [
                i
                for i in idxs
                if float(self.psus.iloc[i]["price_usd"]) >= 90 or int(self.psus.iloc[i].get("wattage", 0) or 0) >= 650
            ]
            idxs = [i for i in idxs if not _high_end_value_name(self.psus.iloc[i].get("name", ""))]
        # Gaming: do not cap PSU wattage here — compatibility + TDP check in expand already enforces safety.
        return self._select_candidates("PSUs", self.psus, idxs, purpose)

    def _compatible_mb(self, cpu_idx: int, mb_idx: int) -> bool:
        c = self.cpus.iloc[cpu_idx]
        m = self.mbs.iloc[mb_idx]
        return str(c["socket"]).strip() == str(m["socket"]).strip()

    def _compatible_ram(self, mb_idx: int, ram_idx: int) -> bool:
        m = self.mbs.iloc[mb_idx]
        r = self.rams.iloc[ram_idx]
        if str(m["ram_type"]).strip() != str(r["type"]).strip():
            return False
        modules = int(r.get("modules", 1) or 1)
        slots = int(m.get("ram_slots", 0) or 0)
        if slots > 0 and modules > slots:
            return False
        cap = int(r.get("capacity_gb", 0) or 0)
        max_ram = int(m.get("max_ram_gb", 0) or 0)
        if max_ram > 0 and cap > max_ram:
            return False
        return True

    def _compatible_storage(self, mb_idx: int, st_idx: int) -> bool:
        return _storage_ok_for_mb(self.storage.iloc[st_idx], self.mbs.iloc[mb_idx])

    def _compatible_gpu(self, mb_idx: int, gpu_idx: int) -> bool:
        return _pcie_ok(self.gpus.iloc[gpu_idx], self.mbs.iloc[mb_idx])

    def _compatible_psu(self, cpu_idx: int, gpu_slot: int, psu_idx: int) -> bool:
        cpu_row = self.cpus.iloc[cpu_idx]
        gpu_row = None if gpu_slot == NO_GPU else self.gpus.iloc[gpu_slot]
        return _psu_meets_load(self.psus.iloc[psu_idx], cpu_row, gpu_row)

    def expand(self, state: State6, purpose: Purpose, budget: float) -> list[State6]:
        d = _state_depth(state)
        if d == 6:
            return []
        if _total_price(state, self.tables) > budget + 1e-9:
            return []
        next_states: list[State6] = []
        succ_limit = _limits(purpose)["successors"]

        def push(idx: int) -> None:
            new = list(state)
            new[d] = idx
            ns = tuple(new)  # type: ignore[assignment]
            spent = _total_price(ns, self.tables)
            if spent > budget:
                self._demo_prune_budget += 1
                if self._demo_prune_budget % 2500 == 0:
                    print(
                        f"[SmartSearch] Pruning branch: price exceeded user budget (${budget:.0f}) "
                        f"while assigning {self.ORDER[d]}."
                    )
                return
            if spent + self._min_remaining_spend(d, purpose) > budget + 1e-6:
                self._demo_prune_projected += 1
                if self._demo_prune_projected % 2500 == 0:
                    print(
                        "[SmartSearch] Pruning branch: projected total (current + minimum remaining "
                        f"parts) exceeds budget at {self.ORDER[d]} stage."
                    )
                return
            next_states.append(ns)

        def push_no_gpu() -> None:
            new = list(state)
            new[d] = NO_GPU
            ns = tuple(new)  # type: ignore[assignment]
            spent = _total_price(ns, self.tables)
            if spent > budget:
                self._demo_prune_budget += 1
                if self._demo_prune_budget % 2500 == 0:
                    print(
                        f"[SmartSearch] Pruning branch: price exceeded user budget (${budget:.0f}) "
                        f"while assigning {self.ORDER[d]} (integrated graphics path)."
                    )
                return
            if spent + self._min_remaining_spend(d, purpose) > budget + 1e-6:
                self._demo_prune_projected += 1
                if self._demo_prune_projected % 2500 == 0:
                    print(
                        "[SmartSearch] Pruning branch: projected total exceeds budget at "
                        f"{self.ORDER[d]} stage (integrated graphics path)."
                    )
                return
            next_states.append(ns)

        if d == 0:
            for i in self._cpu_candidates(purpose):
                push(i)
        elif d == 1:
            cpu_i = state[0]
            assert cpu_i is not None
            for mb_i in self._mb_candidates(purpose):
                if not self._compatible_mb(cpu_i, mb_i):
                    continue
                push(mb_i)
        elif d == 2:
            mb_i = state[1]
            assert mb_i is not None
            for ram_i in self._ram_candidates(purpose):
                if not self._compatible_ram(mb_i, ram_i):
                    continue
                push(ram_i)
        elif d == 3:
            mb_i = state[1]
            assert mb_i is not None
            for st_i in self._storage_candidates(purpose):
                if not self._compatible_storage(mb_i, st_i):
                    continue
                push(st_i)
        elif d == 4:
            mb_i = state[1]
            cpu_i = state[0]
            assert mb_i is not None and cpu_i is not None
            if purpose == Purpose.OFFICE and _cpu_has_igpu(self.cpus.iloc[cpu_i]):
                push_no_gpu()
            else:
                for gpu_i in self._gpu_candidates(purpose, cpu_i):
                    if not self._compatible_gpu(mb_i, gpu_i):
                        continue
                    push(gpu_i)
        elif d == 5:
            cpu_i = state[0]
            gpu_slot = state[4]
            assert cpu_i is not None and gpu_slot is not None
            for psu_i in self._psu_candidates(purpose):
                if not self._compatible_psu(cpu_i, gpu_slot, psu_i):
                    continue
                push(psu_i)

        self._sort_successors_for_purpose(next_states, d, purpose)
        # Preserve strict purpose order for gaming GPUs; shuffle elsewhere for variety.
        if not (purpose == Purpose.GAMING and self.ORDER[d] == "GPUs"):
            self._shuffle_successor_blocks(next_states, d, purpose)
        return next_states[:succ_limit]

    def is_goal(self, state: State6) -> bool:
        return _state_depth(state) == 6

    def validate_build(self, state: State6, purpose: Purpose, budget: float) -> tuple[bool, list[str]]:
        notes: list[str] = []
        if not self.is_goal(state):
            return False, ["Incomplete state"]
        if _total_price(state, self.tables) > budget:
            return False, ["Over budget"]

        c, m, r, st, g, p = state
        assert None not in (c, m, r, st, g, p)
        if not self._compatible_mb(c, m):
            return False, ["CPU socket does not match motherboard"]
        if not self._compatible_ram(m, r):
            return False, ["RAM type or modules incompatible with motherboard"]
        if not self._compatible_storage(m, st):
            return False, ["Storage interface not supported by motherboard ports"]
        if g != NO_GPU:
            if not self._compatible_gpu(m, g):
                return False, ["GPU PCIe generation not supported by motherboard"]
        if not self._compatible_psu(c, g, p):
            return False, ["PSU wattage insufficient for CPU + GPU + 20W safety buffer"]
        if purpose == Purpose.OFFICE and g != NO_GPU:
            return False, ["Office purpose expects no dedicated GPU"]
        if purpose == Purpose.OFFICE and g == NO_GPU and not _cpu_has_igpu(self.cpus.iloc[c]):
            return False, ["Office build without dGPU requires a CPU with integrated graphics"]
        if purpose == Purpose.GAMING:
            if int(self.cpus.iloc[c].get("cores", 0) or 0) < 6:
                return False, ["Gaming purpose requires a CPU with at least 6 cores"]
            if g == NO_GPU:
                return False, ["Gaming purpose requires a discrete GPU"]
            gpu = self.gpus.iloc[g]
            if int(gpu.get("vram_gb", 0) or 0) < 8:
                return False, ["Gaming purpose requires a GPU with at least 8 GB VRAM"]
            if self._gaming_max_cpu_usd is not None:
                cpu_p = float(self.cpus.iloc[c].get("price_usd", 0) or 0)
                if cpu_p > self._gaming_max_cpu_usd + 1e-6:
                    return False, [
                        f"Gaming balance: CPU spend exceeds ~${self._gaming_max_cpu_usd:.0f} cap for this budget "
                        "(keeps room for a stronger GPU)."
                    ]
            if self._gaming_min_gpu_usd is not None:
                gpu_p = float(gpu.get("price_usd", 0) or 0)
                if gpu_p + 1e-6 < self._gaming_min_gpu_usd:
                    return False, [
                        f"Gaming balance: discrete GPU should be at least ~${self._gaming_min_gpu_usd:.0f} "
                        "for this budget tier."
                    ]
        if purpose == Purpose.AI_ML:
            if g == NO_GPU:
                return False, ["AI/ML purpose requires a discrete GPU"]
            gpu = self.gpus.iloc[g]
            if int(gpu.get("vram_gb", 0) or 0) < 12:
                return False, ["AI/ML purpose requires a GPU with at least 12 GB VRAM"]
            psu_row = self.psus.iloc[p]
            if int(float(psu_row.get("wattage", 0) or 0)) < 750:
                return False, ["AI/ML purpose requires a PSU rated at 750 W or higher"]
        notes.append(
            "CPU socket ↔ motherboard socket, RAM type ↔ motherboard ram_type, "
            "storage interface vs motherboard ports, PCIe GPU vs motherboard, and "
            "PSU (CPU TDP + GPU TDP + 20 W) checks passed."
        )
        return True, notes

    def state_to_result(
        self,
        state: State6,
        algorithm: SearchAlgorithm,
        purpose: Purpose,
        budget: float,
        extra_notes: Optional[list[str]] = None,
    ) -> BuildResult:
        c, m, r, st, g, p = state  # type: ignore[misc]
        assert None not in (c, m, r, st, g, p)
        cpu_row = self.cpus.iloc[c]
        gpu_row = None if g == NO_GPU else self.gpus.iloc[g]
        cpu_tdp = int(float(cpu_row.get("tdp_watts", 0) or 0))
        gpu_tdp = int(float(gpu_row.get("tdp_watts", 0) or 0)) if gpu_row is not None else 0
        required = cpu_tdp + gpu_tdp + 20
        psu_w = int(float(self.psus.iloc[p]["wattage"]))
        notes = list(extra_notes or [])
        ok, vnotes = self.validate_build(state, purpose, budget)
        notes.extend(vnotes)
        return BuildResult(
            cpu=_row_to_dict(self.cpus, c),
            motherboard=_row_to_dict(self.mbs, m),
            ram=_row_to_dict(self.rams, r),
            storage=_row_to_dict(self.storage, st),
            gpu=None if g == NO_GPU else _row_to_dict(self.gpus, g),
            psu=_row_to_dict(self.psus, p),
            total_price=round(_total_price(state, self.tables), 2),
            required_psu_watts=required,
            psu_headroom_watts=psu_w - required,
            algorithm=algorithm.value,
            purpose=purpose.value,
            notes=notes,
        )

    def _random_feasible_state(
        self,
        purpose: Purpose,
        budget: float,
        trials: int = 900,
        deadline_ts: float | None = None,
    ) -> Optional[State6]:
        """Monte Carlo assembly over purpose-filtered candidates (used when graph limits miss a goal)."""
        rng = random.Random(self._diversity_seed if self._diversity_seed else 42)
        cpus = self._cpu_candidates(purpose)
        mball = self._mb_candidates(purpose)
        rall = self._ram_candidates(purpose)
        sall = self._storage_candidates(purpose)
        pall = self._psu_candidates(purpose)
        if not cpus or not mball:
            return None
        for _ in range(trials):
            if deadline_ts is not None and time.monotonic() > deadline_ts:
                raise SearchTimeoutError("Search exceeded time limit during fallback sampling.")
            ci = rng.choice(cpus)
            mbs = [m for m in mball if self._compatible_mb(ci, m)]
            if not mbs:
                continue
            mi = rng.choice(mbs)
            rams = [r for r in rall if self._compatible_ram(mi, r)]
            if not rams:
                continue
            ri = rng.choice(rams)
            sts = [s for s in sall if self._compatible_storage(mi, s)]
            if not sts:
                continue
            si = rng.choice(sts)
            if purpose == Purpose.OFFICE and _cpu_has_igpu(self.cpus.iloc[ci]):
                gopts: list[int] = [NO_GPU]
            else:
                gopts = [g for g in self._gpu_candidates(purpose, ci) if self._compatible_gpu(mi, g)]
            if not gopts:
                continue
            gi = rng.choice(gopts)
            psus = [p for p in pall if self._compatible_psu(ci, gi, p)]
            if not psus:
                continue
            pi = rng.choice(psus)
            st: State6 = (ci, mi, ri, si, gi, pi)
            if _total_price(st, self.tables) <= budget and self.validate_build(st, purpose, budget)[0]:
                return st
        return None

    def search(
        self,
        algorithm: SearchAlgorithm,
        budget: float,
        purpose: Purpose,
        max_seconds: float | None = None,
    ) -> Optional[BuildResult]:
        start: State6 = (None, None, None, None, None, None)
        lim = _limits(purpose)
        extra: list[str] = []
        edge_relaxations = 0
        self._demo_prune_budget = 0
        self._demo_prune_projected = 0
        start_ts = time.monotonic()
        max_seconds = None if max_seconds is None else max(float(max_seconds), 0.0)

        def log_progress(tag: str, pops: int, pq_size: int = 0) -> None:
            if pops <= 0 or pops % 4000 != 0:
                return
            logger.info(
                "%s purpose=%s algorithm=%s pops=%s edge_relaxations=%s frontier=%s",
                tag,
                purpose.value,
                algorithm.value,
                pops,
                edge_relaxations,
                pq_size,
            )

        def enforce_deadline() -> None:
            if max_seconds is None:
                return
            elapsed = time.monotonic() - start_ts
            if elapsed > max_seconds:
                raise SearchTimeoutError(
                    f"Search exceeded time limit ({max_seconds:.2f}s) after {elapsed:.2f}s."
                )

        if algorithm == SearchAlgorithm.BFS:
            q: deque[State6] = deque([start])
            visited: set[State6] = {start}
            pops = 0
            while q:
                enforce_deadline()
                pops += 1
                if pops > lim["pops"] or len(visited) >= lim["visited"]:
                    break
                log_progress("bfs", pops, len(q))
                cur = q.popleft()
                if pops % SMARTSEARCH_DEMO_INTERVAL == 0:
                    print(f"[SmartSearch] Checking node: {self._demo_path_line(cur)} …")
                if self.is_goal(cur):
                    ok, _ = self.validate_build(cur, purpose, budget)
                    if ok:
                        print(
                            f"[SmartSearch] Goal found (BFS): {self._demo_path_line(cur, max_parts=6)} "
                            f"— total ${_total_price(cur, self.tables):.2f}"
                        )
                        meta = [
                            f"BFS: first valid complete build after {pops} deque ops "
                            "(successors follow purpose-sorted component order)."
                        ]
                        return self.state_to_result(cur, algorithm, purpose, budget, extra_notes=meta)
                else:
                    for nxt in self.expand(cur, purpose, budget):
                        edge_relaxations += 1
                        if len(visited) >= lim["visited"]:
                            break
                        if nxt not in visited:
                            visited.add(nxt)
                            q.append(nxt)

        elif algorithm == SearchAlgorithm.DFS:
            stack: list[State6] = [start]
            visited: set[State6] = {start}
            pops = 0
            while stack:
                enforce_deadline()
                pops += 1
                if pops > lim["pops"] or len(visited) >= lim["visited"]:
                    break
                log_progress("dfs", pops, len(stack))
                cur = stack.pop()
                if pops % SMARTSEARCH_DEMO_INTERVAL == 0:
                    print(f"[SmartSearch] Checking node: {self._demo_path_line(cur)} …")
                if self.is_goal(cur):
                    ok, _ = self.validate_build(cur, purpose, budget)
                    if ok:
                        print(
                            f"[SmartSearch] Goal found (DFS): {self._demo_path_line(cur, max_parts=6)} "
                            f"— total ${_total_price(cur, self.tables):.2f}"
                        )
                        meta = [
                            f"DFS: first valid complete build after {pops} stack pops "
                            "(same successor ordering as BFS for this purpose)."
                        ]
                        return self.state_to_result(cur, algorithm, purpose, budget, extra_notes=meta)
                else:
                    children = self.expand(cur, purpose, budget)
                    for nxt in reversed(children):
                        edge_relaxations += 1
                        if len(visited) >= lim["visited"]:
                            break
                        if nxt not in visited:
                            visited.add(nxt)
                            stack.append(nxt)

        elif algorithm == SearchAlgorithm.UCS:
            counter = 0
            pops = 0
            # Match the reference project behavior: UCS always minimizes total path cost.
            pq_ucs: list[tuple[float, float, int, State6]] = []
            best_cost: dict[State6, float] = {start: _total_price(start, self.tables)}
            heapq.heappush(
                pq_ucs,
                (
                    best_cost[start],
                    -self._state_performance_total(start),
                    counter,
                    start,
                ),
            )
            counter += 1
            while pq_ucs:
                enforce_deadline()
                pops += 1
                if pops > lim["pops"]:
                    break
                log_progress("ucs", pops, len(pq_ucs))
                cost, _, _, cur = heapq.heappop(pq_ucs)
                if pops % SMARTSEARCH_DEMO_INTERVAL == 0:
                    print(f"[SmartSearch] UCS evaluating: {self._demo_path_line(cur)} …")
                if cost > best_cost.get(cur, float("inf")) + 1e-5:
                    continue
                if self.is_goal(cur):
                    ok, _ = self.validate_build(cur, purpose, budget)
                    if ok:
                        print(
                            f"[SmartSearch] Goal found (UCS): {self._demo_path_line(cur, max_parts=6)} "
                            f"— total ${cost:.2f}"
                        )
                        meta = [
                            f"UCS (Dijkstra on total USD) stopped at first valid goal after {pops} pops, "
                            f"{edge_relaxations} edge relaxations."
                        ]
                        return self.state_to_result(cur, algorithm, purpose, budget, extra_notes=meta)
                    continue
                for nxt in self.expand(cur, purpose, budget):
                    edge_relaxations += 1
                    if edge_relaxations > lim["expansions"]:
                        break
                    new_cost = _total_price(nxt, self.tables)
                    if new_cost < best_cost.get(nxt, float("inf")) - 1e-5:
                        best_cost[nxt] = new_cost
                        heapq.heappush(
                            pq_ucs,
                            (
                                new_cost,
                                -self._state_performance_total(nxt),
                                counter,
                                nxt,
                            ),
                        )
                        counter += 1
                if edge_relaxations > lim["expansions"]:
                    break

        elif algorithm == SearchAlgorithm.ASTAR:
            counter = 0
            pops = 0
            pq: list[tuple[float, float, int, State6]] = []
            best_cost: dict[State6, float] = {start: 0.0}
            heapq.heappush(
                pq,
                (
                    self._budget_astar_priority(start),
                    -self._state_performance_total(start),
                    counter,
                    start,
                ),
            )
            counter += 1
            while pq:
                enforce_deadline()
                pops += 1
                if pops > lim["pops"]:
                    break
                log_progress("astar", pops, len(pq))
                f_val, _, _, cur = heapq.heappop(pq)
                if pops % SMARTSEARCH_DEMO_INTERVAL == 0:
                    print(f"[SmartSearch] A* evaluating: {self._demo_path_line(cur)} …")
                g_cur = _total_price(cur, self.tables)
                if g_cur > best_cost.get(cur, float("inf")) + 1e-5:
                    continue
                if self.is_goal(cur):
                    ok, _ = self.validate_build(cur, purpose, budget)
                    if ok:
                        print(
                            f"[SmartSearch] Goal found (A*): {self._demo_path_line(cur, max_parts=6)} "
                            f"— total ${g_cur:.2f}"
                        )
                        meta = [
                            f"A* (f=g+h, h=min remaining component costs) found a valid goal after {pops} pops, "
                            f"{edge_relaxations} edge relaxations."
                        ]
                        return self.state_to_result(cur, algorithm, purpose, budget, extra_notes=meta)
                    continue
                for nxt in self.expand(cur, purpose, budget):
                    edge_relaxations += 1
                    if edge_relaxations > lim["expansions"]:
                        break
                    g_next = _total_price(nxt, self.tables)
                    if g_next < best_cost.get(nxt, float("inf")) - 1e-5:
                        best_cost[nxt] = g_next
                        heapq.heappush(
                            pq,
                            (
                                self._budget_astar_priority(nxt),
                                -self._state_performance_total(nxt),
                                counter,
                                nxt,
                            ),
                        )
                        counter += 1
                if edge_relaxations > lim["expansions"]:
                    break

        expansions = edge_relaxations

        deadline_ts = None if max_seconds is None else (start_ts + max_seconds)
        fb = self._random_feasible_state(purpose, budget, deadline_ts=deadline_ts)
        if fb is not None:
            logger.info(
                "Graph search hit limits; using randomized feasible assembly for purpose=%s",
                purpose.value,
            )
            return self.state_to_result(
                fb,
                algorithm,
                purpose,
                budget,
                extra_notes=[
                    f"Search hit caps (~{expansions} edge relaxations); "
                    "returned a feasible build from randomized trials over the same pre-filtered candidates."
                ],
            )

        logger.warning(
            "No complete build: purpose=%s algorithm=%s budget=%s edge_relaxations=%s",
            purpose.value,
            algorithm.value,
            budget,
            edge_relaxations,
        )
        return None


# Backward compatibility (older name used in docs/tests).
PCBuilder = SearchEngine


def normalize_purpose(purpose: str) -> Purpose:
    k = purpose.lower().strip().replace(" ", "_").replace("-", "_")
    aliases: dict[str, Purpose] = {
        "ai_ml": Purpose.AI_ML,
        "ai/ml": Purpose.AI_ML,
        "ai_ml_workstation": Purpose.AI_ML,
        "aiml": Purpose.AI_ML,
        "contentcreation": Purpose.CONTENT_CREATION,
        "high_end": Purpose.HIGH_END,
        "highend": Purpose.HIGH_END,
    }
    if k in aliases:
        return aliases[k]
    return Purpose(k)


def run_search(
    tables: dict[str, pd.DataFrame],
    budget: float,
    purpose: str,
    algorithm: str,
    max_seconds: float | None = None,
) -> Optional[BuildResult]:
    pur = normalize_purpose(purpose)
    alg = SearchAlgorithm(algorithm.lower())
    stripped, gaming_caps = purpose_prefilter_catalog(tables, pur, budget)
    filtered = prefilter_tables_by_budget(stripped, budget, 0.6, purpose=pur)
    ordered = order_tables_for_purpose(filtered, pur)
    div = int(
        hashlib.sha256(f"{float(budget):.4f}|{pur.value}|{alg.value}".encode()).hexdigest()[:8],
        16,
    )
    return SearchEngine(ordered, diversity_seed=div, gaming_balance_caps=gaming_caps).search(
        alg,
        budget,
        pur,
        max_seconds=max_seconds,
    )
