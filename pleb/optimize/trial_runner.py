"""Execution adapter from optimization trials to existing PLEB runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import tempfile

from ..config import PipelineConfig
from ..config_io import _dump_toml_no_nulls, _load_config_dict, _set_dotted_key
from ..pipeline import _discover_pulsars_at_ref, _materialize_pulsar_snapshot
from ..pipeline import run_pipeline
from ..workflow import run_workflow
from .models import OptimizationConfig, TrialResult
from .search_space import parameters_to_set_overrides, split_backend_profile_parameters


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
    selected_variant: str | None = None,
) -> Path:
    """Execute one fold-specific held-in rerun."""
    if cfg.execution_mode == "pipeline":
        raw = _load_config_dict(str(cfg.base_config_path))
        _apply_flat_overrides(raw, params)
        if cfg.fixed_overrides:
            _apply_flat_overrides(raw, cfg.fixed_overrides)
        raw["home_dir"] = str(home_dir)
        raw["dataset_name"] = dataset_name
        raw["readonly_materialized_dataset"] = True
        if selected_variant not in (None, ""):
            # The selected candidate variant has already been materialized as the
            # base <PSR>_all.tim in the temporary fold dataset.
            raw["pqc_run_variants"] = False
        raw["outdir_name"] = f"{cfg.study_name}_trial_{trial_id:04d}_{fold_label}"
        raw["force_rerun"] = True
        tmp_dir = _materialize_backend_profiles_for_run(raw, params, cfg.fixed_overrides)
        try:
            pcfg = PipelineConfig.from_dict(raw)
            out_paths = run_pipeline(pcfg)
            return Path(out_paths["tag"])
        finally:
            _cleanup_tmp_dir(tmp_dir)
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
        raw["set"].append("readonly_materialized_dataset=true")
        if selected_variant not in (None, ""):
            raw["set"].append("pqc_run_variants=false")
        raw["set"].append(
            f'outdir_name="{cfg.study_name}_trial_{trial_id:04d}_{fold_label}"'
        )
        raw["set"].append("force_rerun=true")
        tmp_dir = _materialize_backend_profiles_for_workflow(
            raw, params, cfg.study_name, trial_id, cfg.fixed_overrides
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(raw, handle, indent=2)
        try:
            ctx = run_workflow(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
            _cleanup_tmp_dir(tmp_dir)
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
    snapshot_dir = _prepare_materialized_trial_dataset(
        raw,
        study_name=cfg.study_name,
        label=f"trial_{trial_id:04d}",
    )
    tmp_dir = _materialize_backend_profiles_for_run(raw, params, cfg.fixed_overrides)
    try:
        pcfg = PipelineConfig.from_dict(raw)
        out_paths = run_pipeline(pcfg)
        return Path(out_paths["tag"])
    finally:
        _cleanup_tmp_dir(tmp_dir)
        _cleanup_tmp_dir(snapshot_dir)


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
    tmp_dir = _materialize_backend_profiles_for_workflow(
        raw, params, cfg.study_name, trial_id, cfg.fixed_overrides
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        tmp_path = Path(handle.name)
        json.dump(raw, handle, indent=2)
    try:
        ctx = run_workflow(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
        _cleanup_tmp_dir(tmp_dir)
    run_dir = ctx.last_run_dir or ctx.last_pipeline_run_dir
    if run_dir is None:
        raise RuntimeError("workflow trial produced no run directory")
    return Path(run_dir)


def _apply_flat_overrides(target: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    flat, _ = split_backend_profile_parameters(overrides)
    for key, value in flat.items():
        _set_dotted_key(target, str(key), value)


def _materialize_backend_profiles_for_run(
    raw: Dict[str, Any],
    params: Dict[str, Any],
    fixed_overrides: Dict[str, Any] | None = None,
) -> Path | None:
    backend_profiles = _merged_backend_profile_parameters(fixed_overrides, params)
    if not backend_profiles:
        return None
    tmp_dir = _make_backend_profile_tmp_dir(raw)
    target = tmp_dir / "backend_profiles.optimized.toml"
    base_path = _profile_source_path(raw)
    profile_data = _load_backend_profile_doc(base_path)
    profile_root = profile_data.setdefault("backend_profiles", {})
    for pattern, values in backend_profiles.items():
        existing = profile_root.get(pattern, {})
        if not isinstance(existing, dict):
            existing = {}
        merged = dict(existing)
        merged.update(values)
        profile_root[pattern] = merged
    target.write_text(_dump_toml_no_nulls(profile_data), encoding="utf-8")
    raw["pqc_backend_profiles_path"] = str(target)
    return tmp_dir


def _materialize_backend_profiles_for_workflow(
    raw: Dict[str, Any],
    params: Dict[str, Any],
    study_name: str,
    trial_id: int,
    fixed_overrides: Dict[str, Any] | None = None,
) -> Path | None:
    backend_profiles = _merged_backend_profile_parameters(fixed_overrides, params)
    if not backend_profiles:
        return None
    base_path = _profile_source_path(raw)
    profile_data = _load_backend_profile_doc(base_path)
    profile_root = profile_data.setdefault("backend_profiles", {})
    for pattern, values in backend_profiles.items():
        existing = profile_root.get(pattern, {})
        if not isinstance(existing, dict):
            existing = {}
        merged = dict(existing)
        merged.update(values)
        profile_root[pattern] = merged
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f"pleb_optimize_profiles_{study_name}_{trial_id:04d}_")
    )
    target = tmp_dir / "backend_profiles.toml"
    target.write_text(_dump_toml_no_nulls(profile_data), encoding="utf-8")
    raw["set"] = list(raw.get("set", []) or [])
    raw["set"].append(f'pqc_backend_profiles_path="{str(target)}"')
    return tmp_dir


def _merged_backend_profile_parameters(
    *sources: Dict[str, Any] | None,
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        if not source:
            continue
        _, profiles = split_backend_profile_parameters(source)
        for pattern, values in profiles.items():
            current = merged.setdefault(pattern, {})
            current.update(values)
    return merged


def _profile_source_path(raw: Dict[str, Any]) -> Path | None:
    value = raw.get("pqc_backend_profiles_path")
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    home_dir = raw.get("home_dir")
    if home_dir not in (None, ""):
        return (Path(str(home_dir)).expanduser() / path).resolve()
    return path.resolve()


def _make_backend_profile_tmp_dir(raw: Dict[str, Any]) -> Path:
    outdir_name = str(raw.get("outdir_name") or "pqc_optimize_trial").strip() or "pqc_optimize_trial"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in outdir_name)
    return Path(tempfile.mkdtemp(prefix=f"pleb_optimize_profiles_{safe}_"))


def _cleanup_tmp_dir(path: Path | None) -> None:
    if path is None:
        return
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        except FileNotFoundError:
            continue
    try:
        path.rmdir()
    except FileNotFoundError:
        return


def _load_backend_profile_doc(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {"backend_profiles": {}}
    data = _load_config_dict(str(path))
    if not isinstance(data, dict):
        return {"backend_profiles": {}}
    root = data.get("backend_profiles", {})
    if not isinstance(root, dict):
        data["backend_profiles"] = {}
    return data


def _prepare_materialized_trial_dataset(
    raw: Dict[str, Any],
    *,
    study_name: str,
    label: str,
) -> Path | None:
    snapshot_root: Path | None = None
    try:
        pcfg = PipelineConfig.from_dict(raw).resolved()
        repo_root = Path(pcfg.home_dir).resolve()
        dataset_root = Path(pcfg.dataset_name).resolve()
        source_branch = _single_trial_source_branch(pcfg)
        if source_branch is None:
            return None
        pulsars = _trial_pulsars(repo_root, dataset_root, pcfg, source_branch)
        snapshot_root = repo_root / ".pleb_optimize_trial_datasets" / study_name / label
        dataset_copy_root = snapshot_root / dataset_root.name
        for pulsar in pulsars:
            _materialize_pulsar_snapshot(
                repo_root,
                dataset_root,
                ref=source_branch,
                pulsar=pulsar,
                snapshot_dataset_root=dataset_copy_root,
            )
        raw["home_dir"] = str(repo_root)
        raw["dataset_name"] = dataset_copy_root.relative_to(repo_root).as_posix()
        raw["readonly_materialized_dataset"] = True
        return snapshot_root
    except Exception:
        _cleanup_tmp_dir(snapshot_root)
        return None


def _single_trial_source_branch(pcfg: PipelineConfig) -> str | None:
    branches = [str(b).strip() for b in list(getattr(pcfg, "branches", []) or []) if str(b).strip()]
    reference = str(getattr(pcfg, "reference_branch", "") or "").strip()
    branch_set = list(dict.fromkeys([*branches, *([reference] if reference else [])]))
    if len(branch_set) != 1:
        return None
    return branch_set[0]


def _trial_pulsars(
    repo_root: Path,
    dataset_root: Path,
    pcfg: PipelineConfig,
    source_branch: str,
) -> list[str]:
    selected = getattr(pcfg, "pulsars", "ALL")
    if selected == "ALL":
        return _discover_pulsars_at_ref(repo_root, dataset_root, source_branch)
    if isinstance(selected, str):
        return [selected]
    return [str(item) for item in list(selected)]
