"""FastAPI application for the AI PC Builder search API."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from data_loader import load_components
from search_engine import (
    BuildResult,
    Purpose,
    SEARCH_ENGINE_FINGERPRINT,
    SEARCH_ENGINE_FILE,
    SearchAlgorithm,
    SearchTimeoutError,
    normalize_purpose,
    per_component_prefilter_cap_usd,
    run_search,
    try_random_fallback_build,
)

# Search progress (node counts, incumbent metrics) logs at INFO for this package.
logging.getLogger("search_engine").setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

_tables = None
_tables_lock = threading.Lock()


def get_tables():
    """Load dataset once; lock avoids duplicate loads under concurrency."""
    global _tables
    if _tables is not None:
        return _tables
    with _tables_lock:
        if _tables is None:
            _tables = load_components()
        return _tables


class BuildRequest(BaseModel):
    budget: float = Field(gt=0, description="Maximum total build price (USD)")
    purpose: str = Field(
        examples=["gaming"],
        description="gaming | office | content_creation | ai_ml | budget | high_end",
    )
    algorithm: str = Field(
        examples=["bfs"],
        description="bfs | dfs | ucs | astar",
    )


def _result_to_payload(res: BuildResult, request_budget: float | None = None) -> dict:
    out = {
        "cpu": res.cpu,
        "motherboard": res.motherboard,
        "ram": res.ram,
        "storage": res.storage,
        "gpu": res.gpu,
        "psu": res.psu,
        "total_price": res.total_price,
        "required_psu_watts": res.required_psu_watts,
        "psu_headroom_watts": res.psu_headroom_watts,
        "algorithm": res.algorithm,
        "purpose": res.purpose,
        "notes": res.notes,
    }
    if request_budget is not None and request_budget > 0:
        pur = normalize_purpose(res.purpose)
        out["request_budget_usd"] = round(request_budget, 2)
        out["prefilter_component_price_cap_usd"] = per_component_prefilter_cap_usd(request_budget, pur)
        out["budget_utilization"] = round(100.0 * float(res.total_price) / request_budget, 1)
    out["engine_fingerprint"] = SEARCH_ENGINE_FINGERPRINT
    out["search_engine_file"] = SEARCH_ENGINE_FILE
    return out


app = FastAPI(title="AI PC Builder", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    """If ``engine_fingerprint`` does not change after you save ``search_engine.py``, the API process did not reload."""
    return {
        "status": "ok",
        "engine_fingerprint": SEARCH_ENGINE_FINGERPRINT,
        "search_engine_file": SEARCH_ENGINE_FILE,
        "process_cwd": os.getcwd(),
        "hint": "Run uvicorn from the backend folder so --reload watches search_engine.py (see project README).",
    }


@app.get("/api/options")
def options():
    return {
        "purposes": [p.value for p in Purpose],
        "algorithms": [a.value for a in SearchAlgorithm],
    }


_NO_STORE = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}


def _build_max_seconds() -> float | None:
    """Render and similar hosts need a bounded request time; local dev has no cap unless set."""
    raw = os.environ.get("SEARCH_MAX_SECONDS", "").strip()
    if raw:
        try:
            v = float(raw)
            return v if v > 0 else None
        except ValueError:
            return None
    if os.environ.get("RENDER", "").lower() == "true":
        # Tune with SEARCH_MAX_SECONDS; extra randomized fallback runs after if needed.
        return 88.0
    return None


@app.post("/api/build")
async def build_pc(body: BuildRequest):
    try:
        purpose = normalize_purpose(body.purpose)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid purpose: {body.purpose}") from e
    try:
        alg = SearchAlgorithm(body.algorithm.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid algorithm: {body.algorithm}") from e

    # First Excel/CSV load can take many seconds on small hosts; never block the event loop.
    tables = await asyncio.to_thread(get_tables)
    max_s = _build_max_seconds()
    try:
        result = await asyncio.to_thread(
            run_search,
            tables,
            body.budget,
            purpose.value,
            alg.value,
            max_s,
        )
    except SearchTimeoutError:
        fb_secs = 22.0
        raw_fb = os.environ.get("SEARCH_FALLBACK_SECONDS", "").strip()
        if raw_fb:
            try:
                fb_secs = max(4.0, float(raw_fb))
            except ValueError:
                pass
        result = await asyncio.to_thread(
            try_random_fallback_build,
            tables,
            body.budget,
            purpose.value,
            alg.value,
            fb_secs,
        )
        if result is not None:
            return JSONResponse(
                content={
                    "found": True,
                    "build": _result_to_payload(result, body.budget),
                    "message": None,
                },
                headers=_NO_STORE,
            )
        return JSONResponse(
            content={
                "found": False,
                "build": None,
                "message": (
                    "Search hit the host time limit and no quick randomized build was found. "
                    "Try a higher budget, purpose “budget” or “office”, or algorithm A*."
                ),
            },
            headers=_NO_STORE,
        )
    if result is None:
        return JSONResponse(
            content={
                "found": False,
                "build": None,
                "message": "No valid build found within budget and constraints. Try raising the budget or switching purpose/algorithm.",
            },
            headers=_NO_STORE,
        )
    return JSONResponse(
        content={
            "found": True,
            "build": _result_to_payload(result, body.budget),
            "message": None,
        },
        headers=_NO_STORE,
    )


if FRONTEND_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")


@app.get("/")
def root_index():
    index = FRONTEND_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"message": "AI PC Builder API", "docs": "/docs", "frontend": "/ui/index.html"}
