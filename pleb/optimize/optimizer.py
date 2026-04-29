"""Optimization driver for PQC configuration search."""

from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any, Dict, List

from ..config import PipelineConfig
from ..config_io import _load_config_dict, _set_dotted_key
from .folds import load_fold_config
from .fold_datasets import build_fold_dataset
from .models import OptimizationConfig, OptimizationResult, TrialResult
from .objectives import compute_score, load_objective_config, violated_constraints
from .results import write_results
from .report import write_markdown_report, write_pdf_report
from .models import FoldSummary
from .scorers import (
    list_variant_labels,
    score_run_dir,
    score_run_dir_consensus,
    score_run_dir_variants,
    write_bad_toa_masks,
    write_variant_consensus_artifacts,
    write_variant_selection_table,
)
from .search_space import (
    active_parameter_count,
    load_search_space,
    sample_parameters,
)
from .trial_runner import run_fold_trial, run_trial
from ..utils import remove_tree_if_exists


def run_optimization(cfg: OptimizationConfig) -> OptimizationResult:
    """Run an optimization study and write result artifacts."""
    if cfg.jobs != 1:
        raise ValueError(
            "Optimization-level parallel jobs are not yet supported; use jobs=1."
        )
    if cfg.search_space_path is None or cfg.objective_path is None:
        raise ValueError("Optimization requires search_space_path and objective_path.")
    space = load_search_space(Path(cfg.search_space_path))
    objective = load_objective_config(Path(cfg.objective_path))
    fold_cfg = load_fold_config(cfg.folds_path)
    if cfg.sampler == "random":
        trials = _run_random_trials(cfg, space, objective, fold_cfg)
    elif cfg.sampler == "optuna_tpe":
        trials = _run_optuna_trials(cfg, space, objective, fold_cfg)
    else:
        raise ValueError(f"Unsupported optimization sampler: {cfg.sampler!r}")
    successful = [
        trial for trial in trials if trial.status == "ok" and trial.score is not None
    ]
    best = (
        None if not successful else max(successful, key=lambda item: float(item.score))
    )
    if not cfg.keep_trial_runs:
        best_run_dir = None if best is None else best.run_dir
        for trial in trials:
            if trial.run_dir is not None and trial.run_dir != best_run_dir:
                remove_tree_if_exists(Path(trial.run_dir))
    result = OptimizationResult(
        config=cfg, trials=trials, best_trial=best, out_dir=cfg.out_dir
    )
    write_results(result)
    write_markdown_report(result)
    write_pdf_report(result)
    if cfg.write_best_config and best is not None:
        write_best_overrides(result.out_dir / "best_overrides.toml", best.params)
    return result


def write_best_overrides(path: Path, params: Dict[str, Any]) -> None:
    """Write best parameters as a flat TOML snippet."""
    lines = [f"{key} = {_toml_literal(value)}" for key, value in sorted(params.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_random_trials(cfg, space, objective, fold_cfg) -> List[TrialResult]:
    rng = Random(int(cfg.seed))
    trials: List[TrialResult] = []
    for trial_id in range(1, int(cfg.n_trials) + 1):
        params = sample_parameters(space, rng)
        trial = run_trial(cfg, trial_id, params)
        try:
            _score_trial(cfg, trial, fold_cfg, objective, space)
        except Exception as exc:
            trial.status = "failed"
            trial.error = str(exc)
            trial.score = None
        trials.append(trial)
        if cfg.fail_fast and trial.status != "ok":
            break
    return trials


def _run_optuna_trials(cfg, space, objective, fold_cfg) -> List[TrialResult]:
    try:
        import optuna  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "sampler='optuna_tpe' requested but Optuna is not installed."
        ) from exc

    trials: List[TrialResult] = []

    def objective_fn(study_trial) -> float:
        params = _suggest_with_optuna(space, study_trial)
        trial_id = len(trials) + 1
        trial = run_trial(cfg, trial_id, params)
        try:
            _score_trial(cfg, trial, fold_cfg, objective, space)
        except Exception as exc:
            trial.status = "failed"
            trial.error = str(exc)
            trial.score = None
        trials.append(trial)
        if trial.status != "ok" or trial.score is None:
            raise optuna.TrialPruned()
        return float(trial.score)

    sampler = optuna.samplers.TPESampler(seed=int(cfg.seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_fn, n_trials=int(cfg.n_trials))
    return trials


def _suggest_with_optuna(space, trial) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for spec in space.parameters:
        from .search_space import is_parameter_active

        if not is_parameter_active(spec, params):
            continue
        if spec.kind == "bool":
            params[spec.name] = bool(
                trial.suggest_categorical(spec.name, [False, True])
            )
        elif spec.kind == "categorical":
            params[spec.name] = trial.suggest_categorical(
                spec.name, list(spec.choices or [])
            )
        elif spec.kind == "float":
            params[spec.name] = trial.suggest_float(
                spec.name,
                float(spec.low),
                float(spec.high),
                step=spec.step,
                log=bool(spec.log),
            )
        elif spec.kind == "int":
            step = None if spec.step is None else int(spec.step)
            params[spec.name] = int(
                trial.suggest_int(
                    spec.name, int(spec.low), int(spec.high), step=step or 1
                )
            )
        elif spec.kind == "fixed":
            params[spec.name] = spec.fixed
    return params


def _score_trial(cfg, trial, fold_cfg, objective, space) -> None:
    if trial.status != "ok" or trial.run_dir is None:
        return
    pipeline_cfg = None
    backend_col = _resolve_trial_backend_col(None, fold_cfg)
    try:
        pipeline_cfg = _build_pipeline_config_for_trial(cfg, trial.params)
    except Exception:
        if fold_cfg.mode != "none" and fold_cfg.n_splits > 1:
            raise
    if pipeline_cfg is not None:
        backend_col = _resolve_trial_backend_col(pipeline_cfg, fold_cfg)
    parameter_complexity_penalty = active_parameter_count(
        space, trial.params
    ) / max(len(space.parameters), 1)
    variant_strategy = _resolve_trial_variant_strategy(
        cfg, trial.run_dir, pipeline_cfg
    )
    selected_variant: str | None = None
    if variant_strategy == "select_best":
        selected_variant, full_metrics = _select_trial_variant(
            trial.run_dir,
            objective=objective,
            parameter_complexity_penalty=parameter_complexity_penalty,
            backend_col=backend_col,
        )
    elif variant_strategy == "consensus":
        full_metrics = _score_trial_consensus(
            trial.run_dir,
            objective=objective,
            parameter_complexity_penalty=parameter_complexity_penalty,
            backend_col=backend_col,
        )
    else:
        full_metrics, _ = score_run_dir(
            trial.run_dir,
            fold_cfg=load_fold_config(None),
            parameter_complexity_penalty=parameter_complexity_penalty,
            backend_col=backend_col,
        )
    if fold_cfg.mode != "none" and fold_cfg.n_splits > 1:
        fold_summaries = _run_true_fold_reruns(
            cfg,
            trial,
            pipeline_cfg,
            fold_cfg,
            space,
            objective,
            selected_variant=selected_variant,
            variant_strategy=variant_strategy,
            backend_col=backend_col,
        )
        averaged = _average_fold_metrics(fold_summaries)
        for key, value in full_metrics.items():
            averaged[f"full_{key}"] = value
        averaged["stability"] = _fold_metric_stability(fold_summaries, "bad_fraction")
        averaged["event_stability"] = _fold_metric_stability(
            fold_summaries, "event_fraction"
        )
        trial.metrics = averaged
        trial.fold_summaries = fold_summaries
        trial.score = compute_score(averaged, objective)
        return
    trial.metrics = full_metrics
    trial.fold_summaries = []
    trial.score = compute_score(full_metrics, objective)


def _select_trial_variant(
    run_dir: Path,
    *,
    objective,
    parameter_complexity_penalty: float,
    backend_col: str,
) -> tuple[str, Dict[str, float]]:
    """Pick one candidate bad-TOA variant for trial scoring.

    Each PQC variant QC CSV is scored independently. The highest-scoring
    constraint-satisfying candidate (or least-bad penalized candidate) is
    selected and recorded with mask/selection artifacts in the run directory.
    """
    variant_metrics = score_run_dir_variants(
        run_dir,
        fold_cfg=load_fold_config(None),
        parameter_complexity_penalty=parameter_complexity_penalty,
        backend_col=backend_col,
    )
    selected_variant, (selected_metrics, _selected_folds) = max(
        variant_metrics.items(),
        key=lambda item: compute_score(item[1][0], objective),
    )
    write_bad_toa_masks(run_dir, backend_col=backend_col)
    write_variant_selection_table(
        run_dir,
        variant_metrics,
        objective,
        selected_variant=selected_variant,
    )
    return selected_variant, selected_metrics


def _score_trial_consensus(
    run_dir: Path,
    *,
    objective,
    parameter_complexity_penalty: float,
    backend_col: str,
) -> Dict[str, float]:
    """Score one trial by collapsing variants into one support-aware QC table."""
    variant_metrics = score_run_dir_variants(
        run_dir,
        fold_cfg=load_fold_config(None),
        parameter_complexity_penalty=parameter_complexity_penalty,
        backend_col=backend_col,
    )
    metrics, _folds, consensus = score_run_dir_consensus(
        run_dir,
        fold_cfg=load_fold_config(None),
        parameter_complexity_penalty=parameter_complexity_penalty,
        backend_col=backend_col,
    )
    write_bad_toa_masks(run_dir, backend_col=backend_col)
    write_variant_consensus_artifacts(
        run_dir,
        backend_col=backend_col,
        consensus_df=consensus,
    )
    write_variant_selection_table(
        run_dir,
        variant_metrics,
        objective,
        selected_variant=None,
    )
    return metrics


def _resolve_trial_variant_strategy(
    cfg: OptimizationConfig,
    run_dir: Path,
    pipeline_cfg: PipelineConfig | None,
) -> str:
    strategy = str(getattr(cfg, "variant_strategy", "auto") or "auto").strip().lower()
    if strategy in {"single", "consensus", "select_best"}:
        return strategy
    labels = [label for label in list_variant_labels(run_dir) if label != "base"]
    if labels:
        return "consensus"
    if pipeline_cfg is not None and bool(getattr(pipeline_cfg, "pqc_run_variants", False)):
        return "consensus"
    return "single"


def _run_true_fold_reruns(
    cfg: OptimizationConfig,
    trial: TrialResult,
    pipeline_cfg: PipelineConfig,
    fold_cfg,
    space,
    objective,
    *,
    selected_variant: str | None,
    variant_strategy: str,
    backend_col: str,
) -> List[FoldSummary]:
    del objective
    dataset_name = Path(pipeline_cfg.resolved().dataset_name).name
    work_root = Path(cfg.out_dir) / "_fold_datasets" / f"trial_{trial.trial_id:04d}"
    out: List[FoldSummary] = []
    try:
        for fold_index in range(int(fold_cfg.n_splits)):
            tmp_home, held_out_label = build_fold_dataset(
                pipeline_cfg.resolved(),
                fold_cfg=fold_cfg,
                fold_index=fold_index,
                out_root=work_root,
                variant_label=(
                    selected_variant if variant_strategy == "select_best" else None
                ),
            )
            fold_run_dir = run_fold_trial(
                cfg,
                trial.trial_id,
                trial.params,
                fold_label=f"fold_{fold_index:02d}",
                home_dir=tmp_home,
                dataset_name=dataset_name,
                selected_variant=(
                    selected_variant if variant_strategy == "select_best" else None
                ),
            )
            if variant_strategy == "consensus":
                fold_metrics, _folds, _consensus = score_run_dir_consensus(
                    fold_run_dir,
                    fold_cfg=load_fold_config(None),
                    parameter_complexity_penalty=(
                        active_parameter_count(space, trial.params)
                        / max(len(space.parameters), 1)
                    ),
                    backend_col=backend_col,
                )
            else:
                fold_metrics, _ = score_run_dir(
                    fold_run_dir,
                    fold_cfg=load_fold_config(None),
                    parameter_complexity_penalty=(
                        active_parameter_count(space, trial.params)
                        / max(len(space.parameters), 1)
                    ),
                    backend_col=backend_col,
                )
            out.append(
                FoldSummary(
                    label=str(held_out_label),
                    metrics=fold_metrics,
                    run_dir=fold_run_dir,
                )
            )
    finally:
        remove_tree_if_exists(work_root)
    return out


def _average_fold_metrics(folds: List[FoldSummary]) -> Dict[str, float]:
    if not folds:
        return {}
    keys = sorted({key for fold in folds for key in fold.metrics})
    averaged: Dict[str, float] = {}
    for key in keys:
        values = [float(fold.metrics[key]) for fold in folds if key in fold.metrics]
        if values:
            averaged[key] = sum(values) / float(len(values))
    return averaged


def _fold_metric_stability(folds: List[FoldSummary], key: str) -> float:
    values = [float(f.metrics.get(key, 0.0)) for f in folds]
    if len(values) <= 1:
        return 1.0
    mean = sum(values) / float(len(values))
    var = sum((val - mean) ** 2 for val in values) / float(len(values))
    return 1.0 / (1.0 + var**0.5)


def _build_pipeline_config_for_trial(
    cfg: OptimizationConfig, params: Dict[str, Any]
) -> PipelineConfig:
    raw = _load_config_dict(str(cfg.base_config_path))
    for key, value in params.items():
        _set_dotted_key(raw, str(key), value)
    for key, value in (cfg.fixed_overrides or {}).items():
        _set_dotted_key(raw, str(key), value)
    return PipelineConfig.from_dict(raw)


def _resolve_trial_backend_col(
    pipeline_cfg: PipelineConfig | None, fold_cfg
) -> str:
    if pipeline_cfg is not None:
        backend_col = str(getattr(pipeline_cfg, "pqc_backend_col", "") or "").strip()
        if backend_col:
            return backend_col
    fold_backend_col = str(getattr(fold_cfg, "backend_col", "") or "").strip()
    if fold_backend_col:
        return fold_backend_col
    return "group"


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
