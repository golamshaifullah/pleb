"""CLI helpers for optimization mode."""

from __future__ import annotations

from pathlib import Path
import json

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from ..config import _find_nearest_git_root, _resolve_declared_path
from .models import OptimizationConfig


def _normalize_fixed_overrides(
    raw: object,
    *,
    base_dir: Path,
    repo_root: Path | None,
) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    out = dict(raw)
    home_dir = out.get("home_dir")
    if home_dir not in (None, ""):
        resolved_home = _resolve_declared_path(
            str(home_dir), base_dir=base_dir, repo_root=None
        )
        if resolved_home is not None:
            out["home_dir"] = str(resolved_home)
    return out


def load_optimization_config(path: Path) -> OptimizationConfig:
    """Load optimization settings from TOML or JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    resolved_path = path.expanduser().resolve()
    base_dir = resolved_path.parent
    repo_root = _find_nearest_git_root(base_dir)

    def _path(raw):
        if raw in (None, ""):
            return None
        return _resolve_declared_path(str(raw), base_dir=base_dir, repo_root=repo_root)

    if resolved_path.suffix.lower() == ".json":
        data = json.loads(resolved_path.read_text(encoding="utf-8"))
    elif resolved_path.suffix.lower() in (".toml", ".tml"):
        data = tomllib.loads(resolved_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config file type: {resolved_path.suffix}")
    if "optimize" in data and isinstance(data["optimize"], dict):
        data = data["optimize"]
    return OptimizationConfig(
        base_config_path=_path(data["base_config_path"]) or Path("."),
        execution_mode=str(data.get("execution_mode", "pipeline")),
        workflow_file=(
            None
            if data.get("workflow_file") in (None, "")
            else _path(data["workflow_file"])
        ),
        search_space_path=(
            None
            if data.get("search_space_path") in (None, "")
            else _path(data["search_space_path"])
        ),
        objective_path=(
            None
            if data.get("objective_path") in (None, "")
            else _path(data["objective_path"])
        ),
        folds_path=(
            None
            if data.get("folds_path") in (None, "")
            else _path(data["folds_path"])
        ),
        out_dir=_path(data.get("out_dir", "results/optimize")) or Path("results/optimize"),
        study_name=str(data.get("study_name", "pqc_optimize")),
        baseline_run_dir=(
            None
            if data.get("baseline_run_dir") in (None, "")
            else _path(data["baseline_run_dir"])
        ),
        n_trials=int(data.get("n_trials", 20)),
        sampler=str(data.get("sampler", "random")),
        seed=int(data.get("seed", 12345)),
        jobs=int(data.get("jobs", 1)),
        keep_trial_runs=bool(data.get("keep_trial_runs", True)),
        fail_fast=bool(data.get("fail_fast", False)),
        write_best_config=bool(data.get("write_best_config", True)),
        variant_strategy=str(data.get("variant_strategy", "auto")),
        fixed_overrides=_normalize_fixed_overrides(
            data.get("fixed_overrides"), base_dir=base_dir, repo_root=repo_root
        ),
        post_apply_eval=bool(data.get("post_apply_eval", False)),
        post_apply_source_branch=(
            None
            if data.get("post_apply_source_branch") in (None, "")
            else str(data.get("post_apply_source_branch"))
        ),
        post_apply_qc_branch=(
            None
            if data.get("post_apply_qc_branch") in (None, "")
            else str(data.get("post_apply_qc_branch"))
        ),
        post_apply_qc_action=str(data.get("post_apply_qc_action", "delete")),
    )
