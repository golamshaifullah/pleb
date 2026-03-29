"""Load and write UX-facing ``pleb.toml`` files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .models import UXConfig

try:  # py3.11+
    import tomllib  # type: ignore
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from ..config_io import _dump_toml_no_nulls


def load_ux_config(path: Path) -> UXConfig:
    """Load UX config from TOML path.

    Parameters
    ----------
    path : pathlib.Path
        Path to ``pleb.toml``.

    Returns
    -------
    UXConfig
        Parsed UX model with known sections extracted.
    """
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    known = {"paths", "data", "run", "policy", "workflow", "pipeline"}
    return UXConfig(
        paths=_as_dict(raw.get("paths")),
        data=_as_dict(raw.get("data")),
        run=_as_dict(raw.get("run")),
        policy=_as_dict(raw.get("policy")),
        workflow=_as_dict(raw.get("workflow")),
        pipeline=_as_dict(raw.get("pipeline")),
        extra={k: v for k, v in raw.items() if k not in known},
    )


def write_ux_config(path: Path, cfg: Dict[str, Any]) -> None:
    """Write a UX config dictionary to TOML.

    Parameters
    ----------
    path : pathlib.Path
        Output path.
    cfg : dict
        Config dictionary to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_toml_no_nulls(cfg), encoding="utf-8")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries.

    Parameters
    ----------
    base : dict
        Base mapping.
    override : dict
        Override mapping.

    Returns
    -------
    dict
        Deep-merged mapping.
    """
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _as_dict(v: Any) -> Dict[str, Any]:
    return dict(v) if isinstance(v, dict) else {}
