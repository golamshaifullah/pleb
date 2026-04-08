"""Score optimization trials from existing PLEB QC artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
import math

import pandas as pd

from .folds import FoldConfig, make_fold_frames, summarize_fold_stability
from .models import FoldSummary

DEFAULT_OUTLIER_COLS = (
    "bad_point",
    "robust_outlier",
    "robust_global_outlier",
    "bad_mad",
    "bad",
)


def score_run_dir(
    run_dir: Path,
    *,
    fold_cfg: FoldConfig,
    parameter_complexity_penalty: float,
    backend_col: str = "sys",
) -> tuple[Dict[str, float], List[FoldSummary]]:
    """Compute optimization metrics from a pipeline or workflow run directory."""
    frames = _load_qc_frames(run_dir)
    if not frames:
        raise RuntimeError(f"No *_qc.csv files found under {run_dir}")
    df = pd.concat(frames, ignore_index=True, sort=False)
    metrics = _compute_metrics(
        df,
        backend_col=backend_col,
        parameter_complexity_penalty=parameter_complexity_penalty,
    )
    fold_summaries: List[FoldSummary] = []
    for label, frame in make_fold_frames(df, fold_cfg):
        fold_metrics = _compute_metrics(
            frame,
            backend_col=backend_col,
            parameter_complexity_penalty=parameter_complexity_penalty,
        )
        fold_summaries.append(FoldSummary(label=label, metrics=fold_metrics))
    metrics["stability"] = summarize_fold_stability(fold_summaries, "bad_fraction")
    metrics["event_stability"] = summarize_fold_stability(
        fold_summaries, "event_fraction"
    )
    return metrics, fold_summaries


def _load_qc_frames(run_dir: Path) -> List[pd.DataFrame]:
    csvs = sorted(Path(run_dir).rglob("*_qc.csv"))
    frames: List[pd.DataFrame] = []
    for path in csvs:
        frame = pd.read_csv(path)
        frame["_source_csv"] = str(path)
        frames.append(frame)
    return frames


def _compute_metrics(
    df: pd.DataFrame, *, backend_col: str, parameter_complexity_penalty: float
) -> Dict[str, float]:
    n_toas = float(len(df))
    bad_mask = _combined_bad_mask(df)
    event_mask = _combined_event_mask(df)
    resid = pd.to_numeric(
        df.get("resid_us", df.get("resid", pd.Series(dtype=float))), errors="coerce"
    )
    sigma = pd.to_numeric(
        df.get("sigma_us", df.get("sigma", pd.Series(dtype=float))), errors="coerce"
    )
    clean_resid = resid.loc[~bad_mask & resid.notna()]

    metrics: Dict[str, float] = {
        "n_toas": n_toas,
        "n_bad": float(bad_mask.sum()),
        "n_events": float(_event_count(df)),
        "n_event_members": float(event_mask.sum()),
        "bad_fraction": float(bad_mask.mean()) if n_toas else 0.0,
        "event_fraction": float(event_mask.mean()) if n_toas else 0.0,
        "event_coherence": _event_coherence(df, event_mask, backend_col=backend_col),
        "residual_cleanliness": _residual_cleanliness(clean_resid),
        "residual_whiteness": _residual_whiteness(clean_resid),
        "overfragmentation_penalty": _overfragmentation_penalty(df),
        "backend_inconsistency_penalty": _backend_inconsistency_penalty(
            df, bad_mask, backend_col=backend_col
        ),
        "parameter_complexity_penalty": float(parameter_complexity_penalty),
    }
    if sigma.notna().any() and resid.notna().any():
        valid = resid.notna() & sigma.notna() & (sigma > 0)
        if valid.any():
            z = (resid.loc[valid].abs() / sigma.loc[valid]).astype(float)
            metrics["scaled_residual_cleanliness"] = 1.0 / (1.0 + float(z.median()))
    return metrics


def _combined_bad_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in DEFAULT_OUTLIER_COLS:
        if col in df.columns:
            mask = mask | df[col].fillna(False).astype(bool)
    return mask


def _combined_event_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if "event_member" in df.columns:
        mask = mask | df["event_member"].fillna(False).astype(bool)
    if "solar_event_member" in df.columns:
        mask = mask | df["solar_event_member"].fillna(False).astype(bool)
    if "orbital_phase_bad" in df.columns:
        mask = mask | df["orbital_phase_bad"].fillna(False).astype(bool)
    for col in _event_id_columns(df.columns):
        values = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
        mask = mask | (values != -1)
    return mask


def _event_id_columns(columns: Iterable[str]) -> List[str]:
    return [
        col
        for col in columns
        if col.endswith("_id")
        and col not in {"row_id"}
        and not col.startswith("backend_")
    ]


def _event_count(df: pd.DataFrame) -> int:
    total = 0
    for col in _event_id_columns(df.columns):
        values = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
        total += int(values.loc[values != -1].nunique())
    if (
        "solar_event_member" in df.columns
        and df["solar_event_member"].fillna(False).any()
    ):
        total += 1
    if (
        "orbital_phase_bad" in df.columns
        and df["orbital_phase_bad"].fillna(False).any()
    ):
        total += 1
    return total


def _event_coherence(
    df: pd.DataFrame, event_mask: pd.Series, *, backend_col: str
) -> float:
    if not event_mask.any():
        return 0.0
    if backend_col not in df.columns:
        return float(event_mask.mean())
    event_df = df.loc[event_mask].copy()
    counts = event_df[backend_col].astype(str).value_counts(dropna=False)
    if counts.empty:
        return 0.0
    top = float(counts.iloc[0])
    return top / float(counts.sum())


def _residual_cleanliness(clean_resid: pd.Series) -> float:
    if clean_resid.empty:
        return 0.0
    med = float(clean_resid.median())
    mad = float((clean_resid - med).abs().median())
    return 1.0 / (1.0 + mad)


def _residual_whiteness(clean_resid: pd.Series) -> float:
    if len(clean_resid) < 3:
        return 0.0
    corr = float(clean_resid.astype(float).autocorr(lag=1))
    if math.isnan(corr):
        return 0.0
    return 1.0 / (1.0 + abs(corr))


def _overfragmentation_penalty(df: pd.DataFrame) -> float:
    small = 0
    total = 0
    for col in _event_id_columns(df.columns):
        values = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
        if (values != -1).any():
            sizes = values.loc[values != -1].value_counts()
            total += int(len(sizes))
            small += int((sizes <= 1).sum())
    if total == 0:
        return 0.0
    return float(small) / float(total)


def _backend_inconsistency_penalty(
    df: pd.DataFrame, bad_mask: pd.Series, *, backend_col: str
) -> float:
    if backend_col not in df.columns or not bad_mask.any():
        return 0.0
    series = df.loc[bad_mask, backend_col].astype(str)
    counts = series.value_counts(dropna=False)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts.astype(float) / total
    entropy = -float((probs * probs.map(math.log)).sum()) if len(probs) > 1 else 0.0
    max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0
