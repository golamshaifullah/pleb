"""Core data models for optimization-driven PQC tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..compat import dataclass


@dataclass(slots=True)
class ParameterSpec:
    """Search-space definition for one tunable parameter."""

    name: str
    kind: str
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    log: bool = False
    choices: Optional[List[Any]] = None
    fixed: Any = None
    depends_on: Optional[str] = None
    enabled_values: Optional[List[Any]] = None


@dataclass(slots=True)
class SearchSpace:
    """Validated search-space specification."""

    parameters: List[ParameterSpec]


@dataclass(slots=True)
class ObjectiveConfig:
    """Weighted objective definition plus optional hard metric constraints."""

    weights: Dict[str, float]
    maximize: bool = True
    score_offset: float = 0.0
    constraints: Dict[str, float] | None = None
    constraint_penalty: float = 1.0e12


@dataclass(slots=True)
class FoldConfig:
    """Evaluation-fold specification."""

    mode: str = "none"
    n_splits: int = 1
    time_col: str = "mjd"
    backend_col: str = "sys"
    rerun_mode: str = "held_in"


@dataclass(slots=True)
class OptimizationConfig:
    """Top-level configuration for an optimization run."""

    base_config_path: Path
    execution_mode: str = "pipeline"
    workflow_file: Optional[Path] = None
    search_space_path: Optional[Path] = None
    objective_path: Optional[Path] = None
    folds_path: Optional[Path] = None
    out_dir: Path = Path("results/optimize")
    study_name: str = "pqc_optimize"
    baseline_run_dir: Optional[Path] = None
    n_trials: int = 20
    sampler: str = "random"
    seed: int = 12345
    jobs: int = 1
    keep_trial_runs: bool = True
    fail_fast: bool = False
    write_best_config: bool = True
    variant_strategy: str = "auto"
    fixed_overrides: Dict[str, Any] | None = None


@dataclass(slots=True)
class FoldSummary:
    """Metrics evaluated on a single fold."""

    label: str
    metrics: Dict[str, float]
    run_dir: Optional[Path] = None


@dataclass(slots=True)
class TrialResult:
    """Single optimization-trial result."""

    trial_id: int
    status: str
    params: Dict[str, Any]
    score: Optional[float]
    metrics: Dict[str, float]
    run_dir: Optional[Path] = None
    error: Optional[str] = None
    fold_summaries: Optional[List[FoldSummary]] = None


@dataclass(slots=True)
class OptimizationResult:
    """Completed optimization study summary."""

    config: OptimizationConfig
    trials: List[TrialResult]
    best_trial: Optional[TrialResult]
    out_dir: Path
