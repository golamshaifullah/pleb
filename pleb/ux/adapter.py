"""Translate UX config sections into existing PLEB flat config keys."""

from __future__ import annotations

from typing import Any, Dict

from .models import UXConfig

_POLICY_SECTION_PREFIX = {
    "fix": "fix_",
    "pqc": "pqc_",
    "report": "qc_report_",
    "qc_report": "qc_report_",
    "ingest": "ingest_",
}


def ux_to_legacy_dict(cfg: UXConfig) -> Dict[str, Any]:
    """Compile UX model into legacy flat config dictionary.

    Parameters
    ----------
    cfg : UXConfig
        UX-facing structured configuration.

    Returns
    -------
    dict
        Flat configuration dictionary understood by existing PLEB loaders.
    """
    out: Dict[str, Any] = {}

    # Highest-priority direct pass-through block.
    out.update(cfg.pipeline)

    # Common user sections map 1:1 for overlapping keys.
    out.update(cfg.paths)
    out.update(cfg.data)

    # run section maps 1:1 except UX-only keys.
    for k, v in cfg.run.items():
        if k in {"mode", "profile"}:
            continue
        out[k] = v

    # workflow section contains mode-specific keys.
    if "file" in cfg.workflow and "workflow_file" not in out:
        out["workflow_file"] = cfg.workflow["file"]
    for k, v in cfg.workflow.items():
        if k == "file":
            continue
        out[k] = v

    # policy can contain scalar keys and/or nested named sections.
    for k, v in cfg.policy.items():
        if isinstance(v, dict):
            pref = _POLICY_SECTION_PREFIX.get(k)
            if pref is None:
                # Unknown subgroup: pass through nested keys as-is.
                for sk, sv in v.items():
                    out[sk] = sv
                continue
            for sk, sv in v.items():
                if sk.startswith(pref):
                    out[sk] = sv
                else:
                    out[f"{pref}{sk}"] = sv
        else:
            out[k] = v

    out.update(cfg.extra)
    return out
