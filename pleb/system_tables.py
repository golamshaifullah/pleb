"""Load editable system tables (jumps, bandwidths, overlaps, PTA labels).

These tables are declarative configuration assets and are intentionally loaded
with permissive fallbacks so pipeline behavior remains stable when a table is
missing or malformed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _tables_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "configs" / "system_tables"


def load_table(name: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Load a JSON system table or return a default mapping.

    Parameters
    ----------
    name : str
        Table basename under ``configs/system_tables`` (without extension).
    default : dict
        Fallback mapping returned when file is absent or unreadable.

    Returns
    -------
    dict
        Loaded table content or ``default``.
    """
    path = _tables_dir() / f"{name}.json"
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
