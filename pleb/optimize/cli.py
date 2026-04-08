"""CLI helpers for optimization mode."""

from __future__ import annotations

from pathlib import Path
import json

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from .models import OptimizationConfig


def load_optimization_config(path: Path) -> OptimizationConfig:
    """Load optimization settings from TOML or JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in (".toml", ".tml"):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config file type: {path.suffix}")
    if "optimize" in data and isinstance(data["optimize"], dict):
        data = data["optimize"]
    return OptimizationConfig(
        base_config_path=Path(str(data["base_config_path"])),
        execution_mode=str(data.get("execution_mode", "pipeline")),
        workflow_file=(
            None
            if data.get("workflow_file") in (None, "")
            else Path(str(data["workflow_file"]))
        ),
        search_space_path=(
            None
            if data.get("search_space_path") in (None, "")
            else Path(str(data["search_space_path"]))
        ),
        objective_path=(
            None
            if data.get("objective_path") in (None, "")
            else Path(str(data["objective_path"]))
        ),
        folds_path=(
            None
            if data.get("folds_path") in (None, "")
            else Path(str(data["folds_path"]))
        ),
        out_dir=Path(str(data.get("out_dir", "results/optimize"))),
        study_name=str(data.get("study_name", "pqc_optimize")),
        baseline_run_dir=(
            None
            if data.get("baseline_run_dir") in (None, "")
            else Path(str(data["baseline_run_dir"]))
        ),
        n_trials=int(data.get("n_trials", 20)),
        sampler=str(data.get("sampler", "random")),
        seed=int(data.get("seed", 12345)),
        jobs=int(data.get("jobs", 1)),
        keep_trial_runs=bool(data.get("keep_trial_runs", True)),
        fail_fast=bool(data.get("fail_fast", False)),
        write_best_config=bool(data.get("write_best_config", True)),
        fixed_overrides=(
            dict(data.get("fixed_overrides", {}))
            if isinstance(data.get("fixed_overrides"), dict)
            else None
        ),
    )
