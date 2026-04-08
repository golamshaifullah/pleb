"""Fold generation for optimization metrics."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from .models import FoldConfig, FoldSummary


def load_fold_config(path: Path | None) -> FoldConfig:
    """Load a fold specification from TOML or return defaults."""
    if path is None:
        return FoldConfig()
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if "folds" in data and isinstance(data["folds"], dict):
        data = data["folds"]
    return FoldConfig(
        mode=str(data.get("mode", "none")),
        n_splits=int(data.get("n_splits", 1)),
        time_col=str(data.get("time_col", "mjd")),
        backend_col=str(data.get("backend_col", "sys")),
        rerun_mode=str(data.get("rerun_mode", "held_in")),
    )


def make_fold_frames(
    df: pd.DataFrame, cfg: FoldConfig
) -> List[tuple[str, pd.DataFrame]]:
    """Split a QC dataframe into evaluation folds."""
    if df.empty or cfg.mode == "none" or cfg.n_splits <= 1:
        return [("all", df.copy())]
    if cfg.mode == "time_blocks":
        return _time_block_folds(df, cfg)
    if cfg.mode == "backend_holdout":
        return _backend_holdout_folds(df, cfg)
    raise ValueError(f"Unsupported fold mode: {cfg.mode!r}")


def summarize_fold_stability(folds: List[FoldSummary], metric_name: str) -> float:
    """Return an inverse-dispersion stability score for one metric."""
    values = [float(f.metrics.get(metric_name, 0.0)) for f in folds]
    if len(values) <= 1:
        return 1.0
    series = pd.Series(values, dtype=float)
    std = float(series.std(ddof=0))
    return 1.0 / (1.0 + std)


def _time_block_folds(
    df: pd.DataFrame, cfg: FoldConfig
) -> List[tuple[str, pd.DataFrame]]:
    time_col = cfg.time_col if cfg.time_col in df.columns else "mjd"
    if time_col not in df.columns:
        return [("all", df.copy())]
    work = df.copy()
    work["_fold_time"] = pd.to_numeric(work[time_col], errors="coerce")
    work = work.loc[work["_fold_time"].notna()].sort_values("_fold_time")
    if work.empty:
        return [("all", df.copy())]
    bins = min(cfg.n_splits, len(work))
    work["_fold_id"] = pd.qcut(
        work["_fold_time"], q=bins, labels=False, duplicates="drop"
    )
    out: List[tuple[str, pd.DataFrame]] = []
    for fold_id, frame in work.groupby("_fold_id", sort=True):
        out.append((f"time_block_{int(fold_id)}", frame.drop(columns="_fold_id")))
    return out or [("all", df.copy())]


def _backend_holdout_folds(
    df: pd.DataFrame, cfg: FoldConfig
) -> List[tuple[str, pd.DataFrame]]:
    backend_col = cfg.backend_col if cfg.backend_col in df.columns else None
    if backend_col is None:
        return [("all", df.copy())]
    out: List[tuple[str, pd.DataFrame]] = []
    for value, frame in df.groupby(backend_col, sort=True):
        out.append((f"backend_{value}", frame.copy()))
    return out or [("all", df.copy())]
