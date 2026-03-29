"""Preset utilities for UX wrapper commands."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

try:  # py3.11+
    import tomllib  # type: ignore
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def preset_dir() -> Path:
    """Return repository preset directory path."""
    return Path(__file__).resolve().parents[2] / "configs" / "presets"


def list_presets() -> List[str]:
    """List available preset names."""
    out: List[str] = []
    for p in sorted(preset_dir().glob("*.toml")):
        out.append(p.stem)
    return out


def load_preset(name: str) -> Dict[str, Any]:
    """Load a named preset TOML file.

    Parameters
    ----------
    name : str
        Preset basename without ``.toml``.

    Returns
    -------
    dict
        Parsed preset dictionary.
    """
    path = preset_dir() / f"{name}.toml"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return tomllib.loads(path.read_text(encoding="utf-8"))
