"""Score optimization trials from existing PLEB QC artifacts.

The optimize layer supports two distinct variant semantics:

- ``score_run_dir_variants``: treat each PQC variant QC CSV as a competing
  candidate bad-TOA selection.
- ``score_run_dir_consensus``: treat variants as evidence views over the same
  TOAs and collapse them into one support-aware QC table before scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import hashlib
import math

import pandas as pd

from .folds import FoldConfig, make_fold_frames, summarize_fold_stability
from .models import FoldSummary, ObjectiveConfig
from .objectives import compute_score, violated_constraints

DEFAULT_OUTLIER_COLS = (
    "bad_point",
    "robust_outlier",
    "robust_global_outlier",
    "bad_mad",
    "bad",
)

IDENTITY_PREFERRED_COLS = (
    "pulsar",
    "branch",
    "variant",
    "_timfile",
    "filename",
    "row_id",
    "mjd",
    "freq",
    "sigma_us",
    "sigma",
    "site",
    "be",
    "sys",
    "group",
)

CONSENSUS_MERGE_TOL_SECONDS = 2.0


def score_run_dir(
    run_dir: Path,
    *,
    fold_cfg: FoldConfig,
    parameter_complexity_penalty: float,
    backend_col: str = "sys",
) -> tuple[Dict[str, float], List[FoldSummary]]:
    """Compute aggregate optimization metrics from all QC artifacts.

    This is retained for compatibility.  New optimize ranking should use
    :func:`score_run_dir_variants` so each variant/candidate bad mask is scored
    independently.
    """
    frames = _load_qc_frames(run_dir)
    if not frames:
        raise RuntimeError(f"No *_qc.csv files found under {run_dir}")
    return _score_frames(
        frames,
        fold_cfg=fold_cfg,
        parameter_complexity_penalty=parameter_complexity_penalty,
        backend_col=backend_col,
    )


def score_run_dir_variants(
    run_dir: Path,
    *,
    fold_cfg: FoldConfig,
    parameter_complexity_penalty: float,
    backend_col: str = "sys",
) -> Dict[str, Tuple[Dict[str, float], List[FoldSummary]]]:
    """Compute metrics independently for each PQC variant QC output.

    Variant runs are written as ``<PSR>.<variant>_qc.csv`` by the pipeline.
    Treating those CSVs independently prevents optimize from scoring the
    concatenation of multiple candidate TOA selections as if it were one data
    set.  Base runs are reported under the ``base`` label.
    """
    frames = _load_qc_frames(run_dir)
    if not frames:
        raise RuntimeError(f"No *_qc.csv files found under {run_dir}")

    grouped: Dict[str, List[pd.DataFrame]] = {}
    for frame in frames:
        label = str(frame.get("_optimize_variant", pd.Series(["base"])).iloc[0])
        grouped.setdefault(label or "base", []).append(frame)

    return {
        label: _score_frames(
            group_frames,
            fold_cfg=fold_cfg,
            parameter_complexity_penalty=parameter_complexity_penalty,
            backend_col=backend_col,
        )
        for label, group_frames in sorted(grouped.items())
    }


def score_run_dir_consensus(
    run_dir: Path,
    *,
    fold_cfg: FoldConfig,
    parameter_complexity_penalty: float,
    backend_col: str = "sys",
) -> tuple[Dict[str, float], List[FoldSummary], pd.DataFrame]:
    """Collapse overlapping variant rows into one support-aware QC table."""
    frames = _load_qc_frames(run_dir)
    if not frames:
        raise RuntimeError(f"No *_qc.csv files found under {run_dir}")
    consensus = _build_variant_consensus_frame(frames, backend_col=backend_col)
    metrics, folds = _score_frames(
        [consensus],
        fold_cfg=fold_cfg,
        parameter_complexity_penalty=parameter_complexity_penalty,
        backend_col=backend_col,
    )
    return metrics, folds, consensus


def write_bad_toa_masks(
    run_dir: Path,
    *,
    backend_col: str = "sys",
    out_dirname: str = "optimize_bad_masks",
) -> List[Path]:
    """Write explicit per-variant keep/bad mask artifacts for review.

    The mask is not a timing truth label; it records what the candidate PQC run
    selected.  Each row has a deterministic ``toa_id`` built from stable QC
    columns where available, plus ``keep`` / ``bad`` and human-readable reasons.
    """
    frames = _load_qc_frames(run_dir)
    if not frames:
        return []
    root = Path(run_dir) / out_dirname
    root.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[pd.DataFrame]] = {}
    for frame in frames:
        variant = str(frame.get("_optimize_variant", pd.Series(["base"])).iloc[0]) or "base"
        grouped.setdefault(variant, []).append(
            _mask_frame(frame, variant=variant, backend_col=backend_col)
        )

    paths: List[Path] = []
    for variant, masks in sorted(grouped.items()):
        safe_variant = _safe_filename_part(variant)
        path = root / f"{safe_variant}_bad_mask.csv"
        pd.concat(masks, ignore_index=True, sort=False).to_csv(path, index=False)
        paths.append(path)
    return paths


def write_variant_consensus_artifacts(
    run_dir: Path,
    *,
    backend_col: str = "sys",
    out_dirname: str = "optimize_bad_masks",
    consensus_df: pd.DataFrame | None = None,
) -> List[Path]:
    """Write one aggregated cross-variant QC table plus review masks."""
    if consensus_df is None:
        frames = _load_qc_frames(run_dir)
        if not frames:
            return []
        consensus_df = _build_variant_consensus_frame(frames, backend_col=backend_col)
    if consensus_df.empty:
        return []

    root = Path(run_dir) / out_dirname
    root.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    combined_path = root / "variant_consensus.csv"
    consensus_df.to_csv(combined_path, index=False)
    paths.append(combined_path)

    mask_path = root / "consensus_bad_mask.csv"
    _mask_frame(consensus_df, variant="consensus", backend_col=backend_col).to_csv(
        mask_path, index=False
    )
    paths.append(mask_path)

    if "pulsar" in consensus_df.columns:
        for pulsar, frame in consensus_df.groupby("pulsar", sort=True, dropna=False):
            label = str(pulsar) or "unknown"
            path = root / f"{label}.consensus_qc.csv"
            frame.to_csv(path, index=False)
            paths.append(path)
    else:
        path = root / "consensus_qc.csv"
        consensus_df.to_csv(path, index=False)
        paths.append(path)
    return paths


def write_variant_selection_table(
    run_dir: Path,
    variant_metrics: Dict[str, Tuple[Dict[str, float], List[FoldSummary]]],
    objective: ObjectiveConfig,
    *,
    selected_variant: str | None = None,
    out_dirname: str = "optimize_bad_masks",
) -> Path:
    """Write per-variant scores, constraints, and core selection metrics."""
    root = Path(run_dir) / out_dirname
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, (metrics, _folds) in sorted(variant_metrics.items()):
        violations = violated_constraints(metrics, objective)
        row: Dict[str, object] = {
            "variant": label,
            "score": compute_score(metrics, objective),
            "selected": str(label) == str(selected_variant),
            "constraint_violations": len(violations),
            "constraint_violation_keys": ",".join(violations),
        }
        for key in sorted(metrics):
            row[f"metric.{key}"] = metrics[key]
        rows.append(row)
    path = root / "variant_selection_scores.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _score_frames(
    frames: List[pd.DataFrame],
    *,
    fold_cfg: FoldConfig,
    parameter_complexity_penalty: float,
    backend_col: str,
) -> tuple[Dict[str, float], List[FoldSummary]]:
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
    csvs = sorted(
        path
        for path in Path(run_dir).rglob("*_qc.csv")
        if not path.name.endswith(".consensus_qc.csv")
        and "optimize_bad_masks" not in path.parts
    )
    frames: List[pd.DataFrame] = []
    for path in csvs:
        frame = pd.read_csv(path)
        frame["_source_csv"] = str(path)
        frame["_optimize_variant"] = _variant_label_from_qc_path(path, frame)
        frames.append(frame)
    return frames


def list_variant_labels(run_dir: Path) -> List[str]:
    """Return discovered QC variant labels under one run directory."""
    labels = []
    for frame in _load_qc_frames(run_dir):
        label = str(frame.get("_optimize_variant", pd.Series(["base"])).iloc[0]) or "base"
        labels.append(label)
    return sorted(dict.fromkeys(labels))


def _variant_label_from_qc_path(path: Path, frame: pd.DataFrame) -> str:
    if "variant" in frame.columns:
        values = frame["variant"].dropna().astype(str)
        values = values.loc[(values != "") & (values.str.lower() != "nan")]
        if not values.empty:
            return str(values.iloc[0])
    name = path.name
    if name.endswith("_qc.csv"):
        name = name[: -len("_qc.csv")]
    if "." in name:
        return name.split(".", 1)[1] or "base"
    return "base"


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
    n_bad = float(bad_mask.sum())
    n_clean = float(n_toas - n_bad)

    metrics: Dict[str, float] = {
        "n_toas": n_toas,
        "n_bad": n_bad,
        "n_clean": n_clean,
        "n_events": float(_event_count(df)),
        "n_event_members": float(event_mask.sum()),
        "bad_fraction": float(bad_mask.mean()) if n_toas else 0.0,
        "clean_fraction": float(n_clean / n_toas) if n_toas else 0.0,
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
    metrics.update(_backend_bad_fraction_metrics(df, bad_mask, backend_col=backend_col))
    if "variant_support_present" in df.columns:
        support = pd.to_numeric(df["variant_support_present"], errors="coerce")
        metrics["variant_support_mean_present"] = float(support.mean()) if support.notna().any() else 0.0
    if "variant_support_any_bad_fraction" in df.columns:
        support = pd.to_numeric(
            df["variant_support_any_bad_fraction"], errors="coerce"
        )
        if support.notna().any():
            metrics["variant_bad_support_mean"] = float(support.mean())
            metrics["variant_bad_support_among_bad"] = float(
                support.loc[bad_mask & support.notna()].mean()
            ) if (bad_mask & support.notna()).any() else 0.0
    if "variant_support_any_event_fraction" in df.columns:
        support = pd.to_numeric(
            df["variant_support_any_event_fraction"], errors="coerce"
        )
        if support.notna().any():
            metrics["variant_event_support_mean"] = float(support.mean())
    if sigma.notna().any() and resid.notna().any():
        valid = resid.notna() & sigma.notna() & (sigma > 0)
        if valid.any():
            z = (resid.loc[valid].abs() / sigma.loc[valid]).astype(float)
            metrics["scaled_residual_cleanliness"] = 1.0 / (1.0 + float(z.median()))
    return metrics


def _mask_frame(df: pd.DataFrame, *, variant: str, backend_col: str) -> pd.DataFrame:
    bad_mask = _combined_bad_mask(df)
    event_mask = _combined_event_mask(df)
    out = pd.DataFrame(index=df.index)
    out["variant"] = variant
    out["source_csv"] = df.get("_source_csv", pd.Series([""] * len(df), index=df.index))
    out["row_index"] = range(len(df))
    if "toa_id" in df.columns:
        out["toa_id"] = df["toa_id"].astype(str)
    else:
        out["toa_id"] = [_toa_id(row, idx) for idx, row in df.iterrows()]
    out["keep"] = ~bad_mask
    out["bad"] = bad_mask
    out["event_member"] = event_mask
    out["bad_reason"] = [_bad_reason(df, idx) for idx in df.index]
    for col in (
        "pulsar",
        "branch",
        "mjd",
        "freq",
        "sigma_us",
        "sigma",
        "resid_us",
        "resid",
        "_timfile",
        "_timfile_base",
        "filename",
        backend_col,
        "variant_support_present",
        "variant_support_labels",
        "variant_support_any_bad_count",
        "variant_support_any_bad_fraction",
        "variant_support_any_event_count",
        "variant_support_any_event_fraction",
    ):
        if col in df.columns and col not in out.columns:
            out[col] = df[col]
    return out.reset_index(drop=True)


def _toa_id(row: pd.Series, row_index: int) -> str:
    parts = []
    for col in IDENTITY_PREFERRED_COLS:
        if col in row.index:
            parts.append(f"{col}={row.get(col)}")
    if not parts:
        parts.append(f"row_index={row_index}")
    payload = "|".join(parts)
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def _bad_reason(df: pd.DataFrame, idx) -> str:
    reasons = []
    for col in DEFAULT_OUTLIER_COLS:
        if col in df.columns and bool(_truthy_mask(df[col]).loc[idx]):
            reasons.append(col)
    for col in _event_id_columns(df.columns):
        try:
            value = int(float(df.at[idx, col]))
        except Exception:
            value = -1
        if value != -1:
            reasons.append(col)
    for col in ("event_member", "solar_event_member", "orbital_phase_bad"):
        if col in df.columns and bool(_truthy_mask(df[col]).loc[idx]):
            reasons.append(col)
    return ",".join(sorted(set(reasons)))


def _build_variant_consensus_frame(
    frames: List[pd.DataFrame], *, backend_col: str
) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined.empty:
        return combined

    work = combined.copy()
    if "pulsar" not in work.columns:
        work["pulsar"] = ""
    if "_optimize_variant" not in work.columns:
        work["_optimize_variant"] = "base"
    work["_optimize_variant"] = work["_optimize_variant"].fillna("base").astype(str)
    work["_timfile_base"] = _canonical_timfile_keys(work)
    work["__mjd__"] = pd.to_numeric(work.get("mjd", pd.Series(dtype=float)), errors="coerce")
    work["__freq__"] = pd.to_numeric(
        work.get("freq", pd.Series(dtype=float)), errors="coerce"
    )
    work["__site__"] = work.get("site", pd.Series([""] * len(work), index=work.index)).fillna("").astype(str)
    work["__group_id__"] = -1

    next_group_id = 0
    group_keys = ["pulsar", "_timfile_base", "__site__"]
    if "__freq__" in work.columns:
        work["__freq_key__"] = work["__freq__"].round(6)
        group_keys.append("__freq_key__")
    for _group_key, idx in work.groupby(group_keys, sort=False, dropna=False).indices.items():
        sub = work.loc[idx].copy()
        mjd = pd.to_numeric(sub["__mjd__"], errors="coerce")
        if mjd.notna().any():
            order = mjd.sort_values(kind="mergesort").index.tolist()
            current: List[int] = []
            current_center: float | None = None
            for row_idx in order:
                row_mjd = work.at[row_idx, "__mjd__"]
                if pd.isna(row_mjd):
                    if current:
                        work.loc[current, "__group_id__"] = next_group_id
                        next_group_id += 1
                        current = []
                        current_center = None
                    work.at[row_idx, "__group_id__"] = next_group_id
                    next_group_id += 1
                    continue
                row_mjd = float(row_mjd)
                if current_center is None:
                    current = [int(row_idx)]
                    current_center = row_mjd
                    continue
                if abs(row_mjd - current_center) <= (CONSENSUS_MERGE_TOL_SECONDS / 86400.0):
                    current.append(int(row_idx))
                    current_center = float(work.loc[current, "__mjd__"].astype(float).mean())
                else:
                    work.loc[current, "__group_id__"] = next_group_id
                    next_group_id += 1
                    current = [int(row_idx)]
                    current_center = row_mjd
            if current:
                work.loc[current, "__group_id__"] = next_group_id
                next_group_id += 1
        else:
            for row_idx in sub.index:
                work.at[row_idx, "__group_id__"] = next_group_id
                next_group_id += 1

    bool_cols = _consensus_bool_columns(work.columns)
    event_id_cols = _event_id_columns(work.columns)
    rows: List[Dict[str, object]] = []
    for group_id, sub in work.groupby("__group_id__", sort=True):
        rows.append(
            _aggregate_consensus_group(
                sub,
                group_id=int(group_id),
                backend_col=backend_col,
                bool_cols=bool_cols,
                event_id_cols=event_id_cols,
            )
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if "mjd" in out.columns:
        out = out.sort_values(["pulsar", "_timfile_base", "mjd"], kind="mergesort")
    return out.reset_index(drop=True)


def _aggregate_consensus_group(
    df: pd.DataFrame,
    *,
    group_id: int,
    backend_col: str,
    bool_cols: List[str],
    event_id_cols: List[str],
) -> Dict[str, object]:
    variants = sorted(
        {
            str(value).strip() or "base"
            for value in df["_optimize_variant"].dropna().astype(str).tolist()
        }
    )
    present_count = max(len(variants), 1)
    row: Dict[str, object] = {
        "variant": "consensus",
        "variant_support_present": present_count,
        "variant_support_labels": ",".join(variants),
        "variant_support_source_csvs": ",".join(
            sorted(
                {
                    str(value)
                    for value in df.get("_source_csv", pd.Series(dtype=str))
                    .dropna()
                    .astype(str)
                    .tolist()
                    if str(value)
                }
            )
        ),
    }

    preferred_cols = [
        "pulsar",
        "branch",
        "_timfile",
        "_timfile_base",
        "filename",
        "row_id",
        "mjd",
        "freq",
        "sigma_us",
        "sigma",
        "resid_us",
        "resid",
        "site",
        "be",
        "sys",
        "group",
        backend_col,
    ]
    for col in preferred_cols:
        if col in row or col not in df.columns:
            continue
        row[col] = _representative_value(df[col], prefer_numeric=col in {"mjd", "freq", "sigma_us", "sigma", "resid_us", "resid"})

    bad_mask = _combined_bad_mask(df)
    event_mask = _combined_event_mask(df)
    row["variant_support_any_bad_count"] = _unique_variant_truth_count(
        df, bad_mask
    )
    row["variant_support_any_bad_fraction"] = float(
        row["variant_support_any_bad_count"]
    ) / float(present_count)
    row["variant_support_any_event_count"] = _unique_variant_truth_count(
        df, event_mask
    )
    row["variant_support_any_event_fraction"] = float(
        row["variant_support_any_event_count"]
    ) / float(present_count)

    for col in bool_cols:
        truth_count = _unique_variant_truth_count(df, _truthy_mask(df[col]))
        row[f"variant_support_{col}_count"] = truth_count
        row[f"variant_support_{col}_fraction"] = float(truth_count) / float(present_count)
        row[col] = truth_count > 0

    for col in event_id_cols:
        row[col] = _representative_event_id(df[col])

    row_series = pd.Series(row)
    row["toa_id"] = _toa_id(row_series, group_id)
    return row


def _canonical_timfile_keys(df: pd.DataFrame) -> pd.Series:
    if "_timfile_base" in df.columns:
        return df["_timfile_base"].fillna("").astype(str)
    if "_timfile" in df.columns:
        return df["_timfile"].fillna("").astype(str).map(
            lambda x: Path(x).name if x else ""
        )
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def _consensus_bool_columns(columns: Iterable[str]) -> List[str]:
    known = {
        "bad",
        "bad_day",
        "bad_hard",
        "bad_mad",
        "bad_ou",
        "bad_point",
        "event_member",
        "orbital_phase_bad",
        "outlier_any",
        "robust_global_outlier",
        "robust_outlier",
        "solar_event_member",
    }
    out = []
    for col in columns:
        if col.startswith("_") or col in {"variant"}:
            continue
        if col in known or col.endswith("_member"):
            out.append(str(col))
    return sorted(dict.fromkeys(out))


def _truthy_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(float) != 0.0
    return series.fillna("").astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})


def _unique_variant_truth_count(df: pd.DataFrame, mask: pd.Series) -> int:
    if not mask.any():
        return 0
    return int(
        df.loc[mask, "_optimize_variant"].fillna("base").astype(str).nunique(dropna=True)
    )


def _representative_value(series: pd.Series, *, prefer_numeric: bool) -> object:
    clean = series.dropna()
    if clean.empty:
        return None
    if prefer_numeric:
        numeric = pd.to_numeric(clean, errors="coerce")
        if numeric.notna().any():
            return float(numeric.median())
    text = clean.astype(str)
    text = text.loc[(text != "") & (text.str.lower() != "nan")]
    if text.empty:
        return None
    counts = text.value_counts(dropna=False)
    return str(counts.index[0])


def _representative_event_id(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce").fillna(-1).astype(int)
    active = numeric.loc[numeric != -1]
    if active.empty:
        return -1
    counts = active.value_counts()
    return int(counts.index[0])


def _combined_bad_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in DEFAULT_OUTLIER_COLS:
        if col in df.columns:
            mask = mask | _truthy_mask(df[col])
    return mask


def _combined_event_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if "event_member" in df.columns:
        mask = mask | _truthy_mask(df["event_member"])
    if "solar_event_member" in df.columns:
        mask = mask | _truthy_mask(df["solar_event_member"])
    if "orbital_phase_bad" in df.columns:
        mask = mask | _truthy_mask(df["orbital_phase_bad"])
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
        if (values != -1).any():
            total += int(values.loc[values != -1].nunique())
    if (
        "solar_event_member" in df.columns
        and _truthy_mask(df["solar_event_member"]).any()
    ):
        total += 1
    if (
        "orbital_phase_bad" in df.columns
        and _truthy_mask(df["orbital_phase_bad"]).any()
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


def _backend_bad_fraction_metrics(
    df: pd.DataFrame, bad_mask: pd.Series, *, backend_col: str
) -> Dict[str, float]:
    if backend_col not in df.columns or len(df) == 0:
        return {
            "max_backend_bad_fraction": 0.0,
            "backend_bad_fraction_std": 0.0,
            "min_backend_n_clean": 0.0,
        }
    work = pd.DataFrame({"backend": df[backend_col].astype(str), "bad": bad_mask})
    grouped = work.groupby("backend", dropna=False)["bad"]
    fractions = grouped.mean().astype(float)
    n_clean = grouped.apply(lambda s: float((~s.astype(bool)).sum()))
    return {
        "max_backend_bad_fraction": float(fractions.max()) if not fractions.empty else 0.0,
        "backend_bad_fraction_std": float(fractions.std(ddof=0)) if len(fractions) > 1 else 0.0,
        "min_backend_n_clean": float(n_clean.min()) if not n_clean.empty else 0.0,
    }


def _safe_filename_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned or "base"
