"""Persist optimization study outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json

import pandas as pd

from .models import OptimizationResult, TrialResult


def write_results(result: OptimizationResult) -> Dict[str, Path]:
    """Write optimization tables and metadata to disk."""
    out_dir = Path(result.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_csv = out_dir / "trials.csv"
    summary_json = out_dir / "summary.json"
    best_json = out_dir / "best_trial.json"
    pd.DataFrame([_trial_row(t) for t in result.trials]).to_csv(trials_csv, index=False)
    summary_json.write_text(
        json.dumps(_summary_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result.best_trial is not None:
        best_json.write_text(
            json.dumps(_trial_payload(result.best_trial), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "out_dir": out_dir,
        "trials_csv": trials_csv,
        "summary_json": summary_json,
        "best_json": best_json,
    }


def _trial_row(trial: TrialResult) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "trial_id": trial.trial_id,
        "status": trial.status,
        "score": trial.score,
        "run_dir": str(trial.run_dir) if trial.run_dir else "",
        "error": trial.error or "",
    }
    for key, value in sorted(trial.params.items()):
        row[f"param.{key}"] = value
    for key, value in sorted(trial.metrics.items()):
        row[f"metric.{key}"] = value
    return row


def _trial_payload(trial: TrialResult) -> Dict[str, Any]:
    return {
        "trial_id": trial.trial_id,
        "status": trial.status,
        "score": trial.score,
        "params": trial.params,
        "metrics": trial.metrics,
        "run_dir": str(trial.run_dir) if trial.run_dir else None,
        "error": trial.error,
        "fold_summaries": [
            {
                "label": fold.label,
                "metrics": fold.metrics,
                "run_dir": None if fold.run_dir is None else str(fold.run_dir),
            }
            for fold in (trial.fold_summaries or [])
        ],
    }


def _summary_payload(result: OptimizationResult) -> Dict[str, Any]:
    return {
        "study_name": result.config.study_name,
        "n_trials": len(result.trials),
        "best_trial_id": None if result.best_trial is None else result.best_trial.trial_id,
        "best_score": None if result.best_trial is None else result.best_trial.score,
    }
