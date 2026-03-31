"""Central key catalog for UX generation, validation, and docs."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..config import IngestConfig, PipelineConfig, QCReportConfig, WorkflowRunConfig


@dataclass(frozen=True, slots=True)
class KeySpec:
    """Metadata for one flat configuration key."""

    name: str
    type_name: str
    default: Any
    modes: Sequence[str]
    group: str
    level: str
    required: bool = False
    choices: Optional[Sequence[str]] = None
    description: str = ""


_CHOICES: Dict[str, Sequence[str]] = {
    "pqc_structure_mode": ("none", "detrend", "test", "both"),
    "pqc_step_scope": ("global", "backend", "both"),
    "pqc_dm_step_scope": ("global", "backend", "both"),
    "pqc_robust_scope": ("global", "backend", "both"),
    "fix_qc_action": ("comment", "delete"),
    "fix_qc_solar_action": ("comment", "delete"),
    "fix_qc_orbital_phase_action": ("comment", "delete"),
    "fix_wsrt_p2_action": ("comment", "delete"),
    "fix_coord_convert": ("equatorial_to_ecliptic", "ecliptic_to_equatorial"),
    "whitenoise_single_toa_mode": ("combined", "equad0", "ecorr0"),
}


def _group_for_key(name: str) -> str:
    if name in {"home_dir", "dataset_name", "results_dir", "singularity_image"}:
        return "paths"
    if name in {"branches", "reference_branch", "pulsars", "jobs"}:
        return "data"
    if name.startswith("run_") or name in {
        "run_tempo2",
        "make_plots",
        "make_reports",
        "make_covmat",
        "qc_report",
    }:
        return "run"
    if name.startswith("fix_"):
        return "policy.fix"
    if name.startswith("pqc_") or name.startswith("qc_cross_pulsar_"):
        return "policy.pqc"
    if name.startswith("qc_report_"):
        return "policy.report"
    if name.startswith("ingest_"):
        return "policy.ingest"
    if name.startswith("compare_public_"):
        return "policy.compare_public"
    if name.startswith("whitenoise_") or name == "run_whitenoise":
        return "policy.whitenoise"
    return "pipeline"


def _modes_for_key(name: str) -> Sequence[str]:
    if name.startswith("ingest_"):
        return ("ingest", "pipeline", "workflow")
    if name.startswith("qc_report_"):
        return ("qc-report", "pipeline", "workflow")
    if name.startswith("compare_public_"):
        return ("workflow",)
    if name.startswith("whitenoise_") or name == "run_whitenoise":
        return ("pipeline", "workflow")
    if (
        name.startswith("fix_")
        or name.startswith("pqc_")
        or name.startswith("qc_cross_pulsar_")
    ):
        return ("pipeline", "workflow")
    return ("pipeline", "workflow")


def _level_for_key(name: str) -> str:
    advanced_prefixes = (
        "pqc_",
        "qc_cross_pulsar_",
        "fix_qc_",
        "fix_jump_reference_",
        "fix_overlap_",
        "fix_relabel_",
        "fix_wsrt_",
        "compare_public_",
        "whitenoise_",
    )
    if name.startswith(advanced_prefixes):
        return "full"
    if name.startswith("fix_"):
        return "balanced"
    return "minimal"


def _spec_from_field(
    name: str, type_name: str, default: Any, required: bool
) -> KeySpec:
    return KeySpec(
        name=name,
        type_name=type_name,
        default=default,
        required=required,
        modes=_modes_for_key(name),
        group=_group_for_key(name),
        level=_level_for_key(name),
        choices=_CHOICES.get(name),
    )


def all_key_specs() -> List[KeySpec]:
    """Return full key catalog for PLEB configuration surfaces."""
    out: List[KeySpec] = []
    required_pipeline = {"home_dir", "singularity_image"}
    for f in dc_fields(PipelineConfig):
        required = f.name in required_pipeline
        out.append(_spec_from_field(f.name, str(f.type), f.default, required))

    # Workflow and mode-only keys.
    out.append(
        KeySpec(
            name="workflow_file",
            type_name="Path",
            default=None,
            required=False,
            modes=("workflow",),
            group="workflow",
            level="minimal",
        )
    )
    out.append(
        KeySpec(
            name="run_dir",
            type_name="Path",
            default=None,
            required=False,
            modes=("qc-report",),
            group="policy.report",
            level="minimal",
        )
    )
    for f in dc_fields(QCReportConfig):
        if f.name == "run_dir":
            continue
        out.append(
            KeySpec(
                name=f.name,
                type_name=str(f.type),
                default=f.default,
                required=False,
                modes=("qc-report",),
                group="policy.report",
                level="balanced",
            )
        )
    for f in dc_fields(IngestConfig):
        if f.name in {"home_dir", "dataset_name"}:
            # also valid in pipeline
            continue
        out.append(
            KeySpec(
                name=f.name,
                type_name=str(f.type),
                default=f.default,
                required=False,
                modes=("ingest",),
                group="policy.ingest",
                level="balanced",
            )
        )
    for f in dc_fields(WorkflowRunConfig):
        out.append(
            KeySpec(
                name=f.name,
                type_name=str(f.type),
                default=f.default,
                required=True,
                modes=("workflow",),
                group="workflow",
                level="minimal",
            )
        )
    # deterministic
    return sorted({s.name: s for s in out}.values(), key=lambda x: x.name)


def specs_for_mode(mode: str) -> List[KeySpec]:
    """Return catalog entries relevant for a run mode."""
    m = (mode or "pipeline").strip().lower()
    if m == "qc":
        m = "qc-report"
    return [s for s in all_key_specs() if m in s.modes]


def specs_for_mode_level(mode: str, level: str) -> List[KeySpec]:
    """Return catalog entries filtered by mode and verbosity level."""
    ordered = {"minimal": 0, "balanced": 1, "full": 2}
    lvl = (level or "minimal").strip().lower()
    if lvl not in ordered:
        lvl = "minimal"
    max_rank = ordered[lvl]
    return [
        s
        for s in specs_for_mode(mode)
        if ordered.get(s.level, 2) <= max_rank or s.required
    ]


def default_for_spec(spec: KeySpec) -> Any:
    """Return a TOML-friendly default value for a key spec."""
    if spec.required and spec.name == "home_dir":
        return "/path/to/repo"
    if spec.required and spec.name == "singularity_image":
        return "/path/to/tempo2.sif"
    if spec.required and spec.name in {"workflow_file", "run_dir"}:
        return ""
    d = spec.default
    if str(d).startswith("<dataclasses._MISSING_TYPE"):
        return None
    if isinstance(d, Path):
        return str(d)
    if type(d).__name__ in {"_MISSING_TYPE"}:
        return None
    # dataclass field default_factory values are represented as non-serializable sentinels.
    if callable(d):
        try:
            return d()
        except Exception:
            return None
    return d


def grouped_specs(specs: Iterable[KeySpec]) -> Dict[str, List[KeySpec]]:
    """Group specs by logical UX section name."""
    out: Dict[str, List[KeySpec]] = {}
    for s in specs:
        out.setdefault(s.group, []).append(s)
    for k in list(out):
        out[k] = sorted(out[k], key=lambda x: x.name)
    return out
