"""Load PC component sheets from Excel using pandas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

SHEET_NAMES = ("CPUs", "MBs", "RAMs", "Storage", "GPUs", "PSUs")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_dataset_path() -> Path:
    return _project_root() / "PC_Components_Dataset_small.xlsx"


def default_csv_dir() -> Path:
    """Directory containing CPUs.csv, MBs.csv, … (optional alternative to Excel)."""
    return _project_root() / "data"


def _load_from_csv_dir(csv_dir: Path) -> dict[str, pd.DataFrame] | None:
    if not csv_dir.is_dir():
        return None
    paths = {name: csv_dir / f"{name}.csv" for name in SHEET_NAMES}
    if not all(p.is_file() for p in paths.values()):
        return None
    return {name: pd.read_csv(paths[name]) for name in SHEET_NAMES}


def load_components(excel_path: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Load the six component tables.

    If ``excel_path`` is omitted: use ``./data/<Sheet>.csv`` when all six CSVs exist,
    otherwise the default Excel workbook.
    """
    if excel_path is not None:
        path = Path(excel_path)
        if path.is_dir():
            loaded = _load_from_csv_dir(path)
            if loaded is None:
                raise FileNotFoundError(f"Not all component CSVs found under directory: {path}")
            return loaded
        if not path.is_file():
            raise FileNotFoundError(f"Component dataset not found: {path}")
        xl = pd.ExcelFile(path)
        missing = [s for s in SHEET_NAMES if s not in xl.sheet_names]
        if missing:
            raise ValueError(f"Excel file missing sheets: {missing}. Found: {xl.sheet_names}")
        return {name: pd.read_excel(path, sheet_name=name) for name in SHEET_NAMES}

    csv_dir = default_csv_dir()
    from_csv = _load_from_csv_dir(csv_dir)
    if from_csv is not None:
        return from_csv

    path = default_dataset_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Component dataset not found: {path}. "
            f"Either add the Excel file or place CPUs.csv … PSUs.csv under {csv_dir}."
        )
    xl = pd.ExcelFile(path)
    missing = [s for s in SHEET_NAMES if s not in xl.sheet_names]
    if missing:
        raise ValueError(f"Excel file missing sheets: {missing}. Found: {xl.sheet_names}")
    return {name: pd.read_excel(path, sheet_name=name) for name in SHEET_NAMES}


def yes_no(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("yes", "true", "1", "y")
