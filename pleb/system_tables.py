"""Load editable system tables (jumps, bandwidths, overlaps, PTA labels)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _tables_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "configs" / "system_tables"


def load_table(name: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Load a JSON table from configs/system_tables or return default."""
    path = _tables_dir() / f"{name}.json"
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
