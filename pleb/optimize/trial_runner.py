"""Execution adapter from optimization trials to existing PLEB runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import tempfile

from ..config import PipelineConfig
from ..config_io import _load_config_dict, _set_dotted_key
from ..pipeline import run_pipeline
from ..workflow import run_workflow
from .models import OptimizationConfig, TrialResult
from .search_space import parameters_to_set_overrides


def run_trial(
    cfg: OptimizationConfig, trial_id: int, params: Dict[str, Any]
) -> TrialResult:
    """Execute one optimization trial through pipeline or workflow mode."""
    try:
        if cfg.execution_mode == "pipeline":
            run_dir = _run_pipeline_trial(cfg, trial_id, params)
        elif cfg.execution_mode == "workflow":
            run_dir = _run_workflow_trial(cfg, trial_id, params)
        else:
            raise ValueError(
                f"Unsupported optimization execution_mode={cfg.execution_mode!r}"
            )
        return TrialResult(
            trial_id=trial_id,
            status="ok",
            params=dict(params),
            score=None,
            metrics={},
            run_dir=run_dir,
        )
    except Exception as exc:
        return TrialResult(
            trial_id=trial_id,
            status="failed",
            params=dict(params),
            score=None,
            metrics={},
            run_dir=None,
            error=str(exc),
        )


def run_fold_trial(
    cfg: OptimizationConfig,
    trial_id: int,
    params: Dict[str, Any],
    *,
    fold_label: str,
    home_dir: Path,
    dataset_name: str,
) -> Path:
    """Execute one fold-specific held-in rerun."""
    if cfg.execution_mode == "pipeline":
        raw = _load_config_dict(str(cfg.base_config_path))
        _apply_flat_overrides(raw, params)
        if cfg.fixed_overrides:
            _apply_flat_overrides(raw, cfg.fixed_overrides)
        raw["home_dir"] = str(home_dir)
        raw["dataset_name"] = dataset_name
        raw["outdir_name"] = f"{cfg.study_name}_trial_{trial_id:04d}_{fold_label}"
        raw["force_rerun"] = True
        pcfg = PipelineConfig.from_dict(raw)
        out_paths = run_pipeline(pcfg)
        return Path(out_paths["tag"])
    if cfg.execution_mode == "workflow":
        wf_path = cfg.workflow_file or cfg.base_config_path
        raw = _load_config_dict(str(wf_path))
        existing_set = list(raw.get("set", []) or [])
        raw["set"] = existing_set + parameters_to_set_overrides(params)
        if cfg.fixed_overrides:
            raw["set"] = list(raw["set"]) + parameters_to_set_overrides(
                cfg.fixed_overrides
            )
        raw["set"].append(f'home_dir="{str(home_dir)}"')
        raw["set"].append(f'dataset_name="{dataset_name}"')
        raw["set"].append(
            f'outdir_name="{cfg.study_name}_trial_{trial_id:04d}_{fold_label}"'
        )
        raw["set"].append("force_rerun=true")
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(raw, handle, indent=2)
        try:
            ctx = run_workflow(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        run_dir = ctx.last_run_dir or ctx.last_pipeline_run_dir
        if run_dir is None:
            raise RuntimeError("workflow fold trial produced no run directory")
        return Path(run_dir)
    raise ValueError(f"Unsupported optimization execution_mode={cfg.execution_mode!r}")


def _run_pipeline_trial(
    cfg: OptimizationConfig, trial_id: int, params: Dict[str, Any]
) -> Path:
    raw = _load_config_dict(str(cfg.base_config_path))
    _apply_flat_overrides(raw, params)
    if cfg.fixed_overrides:
        _apply_flat_overrides(raw, cfg.fixed_overrides)
    raw["outdir_name"] = f"{cfg.study_name}_trial_{trial_id:04d}"
    raw["force_rerun"] = True
    pcfg = PipelineConfig.from_dict(raw)
    out_paths = run_pipeline(pcfg)
    return Path(out_paths["tag"])


def _run_workflow_trial(
    cfg: OptimizationConfig, trial_id: int, params: Dict[str, Any]
) -> Path:
    wf_path = cfg.workflow_file or cfg.base_config_path
    raw = _load_config_dict(str(wf_path))
    existing_set = list(raw.get("set", []) or [])
    raw["set"] = existing_set + parameters_to_set_overrides(params)
    if cfg.fixed_overrides:
        raw["set"] = list(raw["set"]) + parameters_to_set_overrides(cfg.fixed_overrides)
    raw["set"].append(f'outdir_name="{cfg.study_name}_trial_{trial_id:04d}"')
    raw["set"].append("force_rerun=true")
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        tmp_path = Path(handle.name)
        json.dump(raw, handle, indent=2)
    try:
        ctx = run_workflow(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    run_dir = ctx.last_run_dir or ctx.last_pipeline_run_dir
    if run_dir is None:
        raise RuntimeError("workflow trial produced no run directory")
    return Path(run_dir)


def _apply_flat_overrides(target: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        _set_dotted_key(target, str(key), value)
