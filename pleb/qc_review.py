"""Expert review layer for PQC outlier decisions.

This module is intentionally additive: it does not mutate raw PQC CSV files.
It loads one or more ``*_qc.csv`` files, attaches stable per-row review IDs,
merges optional expert overrides, and writes reviewed QC tables that downstream
PLEB stages can consume explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

import pandas as pd

REVIEW_ACTIONS = (
    "mark_bad",
    "mark_event",
    "keep",
    "clear_auto_bad",
    "clear_manual",
)

OVERRIDE_COLUMNS = [
    "override_id",
    "review_id",
    "qc_csv",
    "row_index",
    "pulsar",
    "variant",
    "timfile",
    "mjd",
    "freq",
    "backend",
    "manual_action",
    "manual_reason",
    "reviewer",
    "reviewed_at",
    "source",
]

BAD_COLUMNS = (
    "bad_point",
    "bad_mad",
    "robust_outlier",
    "robust_global_outlier",
    "outlier_any",
)

EVENT_COLUMNS = (
    "event_member",
    "event_any",
    "solar_event_member",
    "orbital_phase_bad",
    "eclipse_member",
    "gaussian_bump_member",
    "glitch_member",
    "transient_member",
    "step_member",
    "dm_step_member",
)

TIME_COLUMNS = (
    "mjd",
    "MJD",
    "sat",
    "toa_mjd",
    "bary_mjd",
    "mjd_float",
)

RESIDUAL_COLUMNS = (
    "resid_us",
    "residual_us",
    "resid",
    "residual",
    "post",
    "postfit",
    "post_fit",
)

UNCERTAINTY_COLUMNS = (
    "sigma_us",
    "toa_err_us",
    "err_us",
    "toa_uncertainty_us",
    "sigma",
    "toa_err",
    "error",
    "err",
)

FREQUENCY_COLUMNS = (
    "freq",
    "frequency",
    "obs_freq",
    "freq_MHz",
    "freq_mhz",
)

BACKEND_COLUMNS = (
    "backend",
    "sys",
    "system",
    "group",
    "telescope",
    "obs",
)

TIMFILE_COLUMNS = (
    "_timfile",
    "timfile",
    "filename",
    "file",
)


@dataclass(frozen=True)
class ReviewColumns:
    """Resolved column names used by the review UI/export path."""

    time: str | None
    residual: str | None
    uncertainty: str | None
    frequency: str | None
    backend: str | None
    timfile: str | None


def utc_now_iso() -> str:
    """Return a UTC timestamp in a compact ISO-8601 form."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def is_review_artifact_path(path: str | Path) -> bool:
    """Return ``True`` for generated review artifacts, not raw QC inputs."""

    p = Path(path)
    if p.name.lower() == "reviewed_qc.csv":
        return True
    return any(part.lower() == "qc_review" for part in p.parts)


def find_qc_csvs(run_dir: str | Path) -> list[Path]:
    """Find raw QC CSVs below a run directory.

    Generated review exports such as ``qc_review/reviewed_qc.csv`` are excluded
    so the reviewer never ingests its own output on reload.
    """

    root = Path(run_dir).expanduser()
    if root.is_file():
        return [] if is_review_artifact_path(root) else [root]
    return sorted(
        p for p in root.rglob("*_qc.csv") if p.is_file() and not is_review_artifact_path(p)
    )


def choose_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Return the first matching column from ``candidates``."""

    cols = set(str(c) for c in df.columns)
    for col in candidates:
        if col in cols:
            return col
    lower = {str(c).lower(): str(c) for c in df.columns}
    for col in candidates:
        hit = lower.get(str(col).lower())
        if hit is not None:
            return hit
    return None


def infer_columns(df: pd.DataFrame) -> ReviewColumns:
    """Infer important plotting/review columns from a QC dataframe."""

    return ReviewColumns(
        time=choose_column(df, TIME_COLUMNS),
        residual=choose_column(df, RESIDUAL_COLUMNS),
        uncertainty=choose_column(df, UNCERTAINTY_COLUMNS),
        frequency=choose_column(df, FREQUENCY_COLUMNS),
        backend=choose_column(df, BACKEND_COLUMNS),
        timfile=choose_column(df, TIMFILE_COLUMNS),
    )


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0) != 0
    text = s.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y", "bad", "outlier"})


def infer_auto_decision(df: pd.DataFrame) -> pd.Series:
    """Infer compact automatic decisions from common PQC columns.

    If a ``decision`` column already exists, it is used as the automatic
    decision. Otherwise common outlier/event columns are reduced to one of
    ``KEEP``, ``BAD_TOA``, ``EVENT``, or ``REVIEW_EVENT``.
    """

    if "decision" in df.columns:
        out = df["decision"].fillna("KEEP").astype(str).str.strip().str.upper()
        return out.replace({"": "KEEP"})

    bad = pd.Series(False, index=df.index)
    for col in BAD_COLUMNS:
        bad |= _bool_series(df, col)

    event = pd.Series(False, index=df.index)
    for col in EVENT_COLUMNS:
        event |= _bool_series(df, col)
    if "transient_id" in df.columns:
        tid = pd.to_numeric(df["transient_id"], errors="coerce").fillna(-1)
        event |= tid >= 0

    decision = pd.Series("KEEP", index=df.index, dtype=object)
    decision.loc[event] = "EVENT"
    decision.loc[bad] = "BAD_TOA"
    decision.loc[bad & event] = "REVIEW_EVENT"
    return decision


def _guess_pulsar_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_qc"):
        stem = stem[:-3]
    for sep in ("__", "."):
        if sep in stem:
            return stem.split(sep, 1)[0]
    return stem


def _normalise_scalar(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def make_review_id(row: Mapping[str, object]) -> str:
    """Create a stable row identifier for a QC row.

    The identifier prefers source identity over physics values. This prevents
    accidental cross-matching of duplicate or nearly duplicate MJDs.
    """

    parts = [
        _normalise_scalar(row.get("qc_csv")),
        _normalise_scalar(row.get("row_index")),
        _normalise_scalar(row.get("pulsar")),
        _normalise_scalar(row.get("variant")),
        _normalise_scalar(row.get("timfile")),
        _normalise_scalar(row.get("mjd")),
        _normalise_scalar(row.get("freq")),
        _normalise_scalar(row.get("backend")),
    ]
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    prefix = parts[2] or "toa"
    prefix = "".join(ch if ch.isalnum() else "_" for ch in prefix).strip("_") or "toa"
    return f"{prefix}_{digest}"


def _series_or_default(df: pd.DataFrame, col: str | None, default: object = "") -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return df[col]


def load_qc_csv(path: str | Path, *, root: str | Path | None = None) -> pd.DataFrame:
    """Load a single QC CSV and attach review metadata."""

    path = Path(path).expanduser()
    df = pd.read_csv(path, low_memory=False)
    cols = infer_columns(df)
    out = df.copy()

    try:
        qc_csv = path.resolve().relative_to(Path(root).expanduser().resolve()).as_posix() if root else path.resolve().as_posix()
    except Exception:
        qc_csv = path.resolve().as_posix()

    if "pulsar" not in out.columns:
        out["pulsar"] = _guess_pulsar_from_path(path)
    if "variant" not in out.columns:
        out["variant"] = "base"

    out["qc_csv"] = qc_csv
    out["row_index"] = range(len(out))
    out["auto_decision"] = infer_auto_decision(out)

    out["mjd"] = pd.to_numeric(_series_or_default(out, cols.time), errors="coerce")
    out["residual"] = pd.to_numeric(_series_or_default(out, cols.residual), errors="coerce")
    out["uncertainty"] = pd.to_numeric(_series_or_default(out, cols.uncertainty), errors="coerce")
    out["freq"] = pd.to_numeric(_series_or_default(out, cols.frequency), errors="coerce")
    out["backend"] = _series_or_default(out, cols.backend).fillna("").astype(str)
    out["timfile"] = _series_or_default(out, cols.timfile).fillna("").astype(str)

    # Use row-wise apply here for clarity; QC review tables are interactive-sized
    # after filtering, and this is not part of the tempo-fitting hot path.
    out["review_id"] = [make_review_id(rec) for rec in out.to_dict("records")]
    return out


def load_qc_frames(
    run_dir: str | Path | None = None,
    *,
    csvs: Iterable[str | Path] | None = None,
) -> pd.DataFrame:
    """Load all requested QC CSVs into one review dataframe."""

    paths = [Path(p).expanduser() for p in csvs] if csvs else []
    explicit_review_artifacts = [p for p in paths if is_review_artifact_path(p)]
    if explicit_review_artifacts:
        listed = ", ".join(str(p) for p in explicit_review_artifacts)
        raise ValueError(
            f"Refusing to load reviewed QC artifact(s) as raw QC input: {listed}"
        )
    if run_dir is not None:
        paths.extend(find_qc_csvs(run_dir))
    paths = sorted(dict.fromkeys(p.resolve() for p in paths))
    if not paths:
        raise FileNotFoundError("No *_qc.csv files found for QC review.")

    root = Path(run_dir).expanduser() if run_dir is not None else None
    frames = [load_qc_csv(p, root=root) for p in paths]
    return pd.concat(frames, ignore_index=True, sort=False)


def empty_overrides() -> pd.DataFrame:
    """Return an empty override table with the canonical schema."""

    return pd.DataFrame(columns=OVERRIDE_COLUMNS)


def load_overrides(path: str | Path | None) -> pd.DataFrame:
    """Load manual overrides, returning an empty table if absent."""

    if path is None:
        return empty_overrides()
    p = Path(path).expanduser()
    if not p.exists() or p.stat().st_size == 0:
        return empty_overrides()
    df = pd.read_csv(p, dtype={"review_id": str, "qc_csv": str})
    for col in OVERRIDE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[OVERRIDE_COLUMNS].copy()


def validate_overrides(overrides: pd.DataFrame) -> None:
    """Validate manual override actions."""

    if overrides.empty:
        return
    actions = overrides.get("manual_action", pd.Series(dtype=object)).fillna("").astype(str)
    invalid = sorted(set(actions) - set(REVIEW_ACTIONS) - {""})
    if invalid:
        allowed = ", ".join(REVIEW_ACTIONS)
        raise ValueError(f"Unsupported manual_action values {invalid}; allowed: {allowed}")


def write_overrides(overrides: pd.DataFrame, path: str | Path) -> Path:
    """Write overrides using the canonical column order."""

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    df = overrides.copy() if overrides is not None else empty_overrides()
    for col in OVERRIDE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df[OVERRIDE_COLUMNS].to_csv(out, index=False)
    return out


def make_override_rows(
    rows: pd.DataFrame,
    *,
    action: str,
    reason: str = "",
    reviewer: str = "",
    source: str = "manual_review_ui",
    timestamp: str | None = None,
) -> pd.DataFrame:
    """Create override rows for selected review rows."""

    if action not in REVIEW_ACTIONS:
        raise ValueError(f"Unsupported manual_action {action!r}")
    if rows.empty:
        return empty_overrides()
    ts = timestamp or utc_now_iso()
    out = empty_overrides()
    for _, row in rows.iterrows():
        out.loc[len(out)] = {
            "override_id": str(uuid4()),
            "review_id": _normalise_scalar(row.get("review_id")),
            "qc_csv": _normalise_scalar(row.get("qc_csv")),
            "row_index": _normalise_scalar(row.get("row_index")),
            "pulsar": _normalise_scalar(row.get("pulsar")),
            "variant": _normalise_scalar(row.get("variant")),
            "timfile": _normalise_scalar(row.get("timfile")),
            "mjd": _normalise_scalar(row.get("mjd")),
            "freq": _normalise_scalar(row.get("freq")),
            "backend": _normalise_scalar(row.get("backend")),
            "manual_action": action,
            "manual_reason": reason,
            "reviewer": reviewer,
            "reviewed_at": ts,
            "source": source,
        }
    return out


def append_overrides(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Append override rows while preserving audit history."""

    if existing is None or existing.empty:
        base = empty_overrides()
    else:
        base = existing.copy()
    if new_rows is None or new_rows.empty:
        return base
    merged = pd.concat([base, new_rows], ignore_index=True, sort=False)
    for col in OVERRIDE_COLUMNS:
        if col not in merged.columns:
            merged[col] = ""
    return merged[OVERRIDE_COLUMNS].copy()


def _latest_effective_overrides(overrides: pd.DataFrame) -> pd.DataFrame:
    """Return the latest non-empty override row per review_id.

    ``clear_manual`` is retained as the latest row so the merge layer can fall
    back to automatic labels while still recording the clearing action.
    """

    if overrides is None or overrides.empty:
        return empty_overrides()
    validate_overrides(overrides)
    df = overrides.copy()
    df["_order"] = range(len(df))
    df["reviewed_at"] = df.get("reviewed_at", "").fillna("").astype(str)
    df = df[df["review_id"].fillna("").astype(str).str.len() > 0]
    if df.empty:
        return empty_overrides()
    df = df.sort_values(["review_id", "reviewed_at", "_order"])
    latest = df.groupby("review_id", as_index=False, sort=False).tail(1)
    return latest.drop(columns=["_order"], errors="ignore")


def apply_overrides(qc_df: pd.DataFrame, overrides: pd.DataFrame | None) -> pd.DataFrame:
    """Merge manual overrides into a QC dataframe.

    Manual columns are additive; raw automatic columns are preserved. The final
    reviewed decision is exposed in ``reviewed_decision`` and boolean helper
    columns ``reviewed_bad_point``, ``reviewed_event_member``, and
    ``reviewed_keep``.
    """

    if "review_id" not in qc_df.columns:
        raise ValueError("qc_df must contain review_id; load via load_qc_frames/load_qc_csv")

    out = qc_df.copy()
    if "auto_decision" not in out.columns:
        out["auto_decision"] = infer_auto_decision(out)
    out["reviewed_decision"] = out["auto_decision"].fillna("KEEP").astype(str).str.upper()
    out["manual_action"] = ""
    out["manual_reason"] = ""
    out["manual_reviewer"] = ""
    out["manual_reviewed_at"] = ""
    out["manual_override_id"] = ""

    latest = _latest_effective_overrides(overrides if overrides is not None else empty_overrides())
    if not latest.empty:
        cols = [
            "review_id",
            "override_id",
            "manual_action",
            "manual_reason",
            "reviewer",
            "reviewed_at",
        ]
        override_latest = latest[cols].rename(
            columns={
                "override_id": "_override_id",
                "manual_action": "_override_action",
                "manual_reason": "_override_reason",
                "reviewer": "_override_reviewer",
                "reviewed_at": "_override_reviewed_at",
            }
        )
        merged = out.merge(override_latest, on="review_id", how="left")
        action = merged["_override_action"].fillna("").astype(str)
        merged["manual_action"] = action
        merged["manual_reason"] = merged["_override_reason"].fillna("").astype(str)
        merged["manual_reviewer"] = merged["_override_reviewer"].fillna("").astype(str)
        merged["manual_reviewed_at"] = merged["_override_reviewed_at"].fillna("").astype(str)
        merged["manual_override_id"] = merged["_override_id"].fillna("").astype(str)

        reviewed = merged["reviewed_decision"].copy()
        reviewed.loc[action == "mark_bad"] = "BAD_TOA"
        reviewed.loc[action == "mark_event"] = "EVENT"
        reviewed.loc[action == "keep"] = "KEEP"
        reviewed.loc[action == "clear_auto_bad"] = "KEEP"
        # clear_manual intentionally leaves reviewed decision as auto_decision.
        merged["reviewed_decision"] = reviewed
        out = merged.drop(
            columns=[
                "_override_id",
                "_override_action",
                "_override_reason",
                "_override_reviewer",
                "_override_reviewed_at",
            ],
            errors="ignore",
        )

    decision = out["reviewed_decision"].fillna("KEEP").astype(str).str.upper()
    out["reviewed_bad_point"] = decision.isin({"BAD_TOA", "REVIEW_EVENT"})
    out["reviewed_event_member"] = decision.isin({"EVENT", "REVIEW_EVENT"})
    out["reviewed_keep"] = decision.eq("KEEP")
    return out


def write_reviewed_qc(
    qc_df: pd.DataFrame,
    out_path: str | Path,
    *,
    overrides: pd.DataFrame | None = None,
) -> Path:
    """Apply overrides and write a reviewed QC CSV."""

    reviewed = apply_overrides(qc_df, overrides)
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    reviewed.to_csv(out, index=False)
    return out


def write_reviewed_from_paths(
    *,
    run_dir: str | Path | None = None,
    csvs: Iterable[str | Path] | None = None,
    overrides_csv: str | Path | None = None,
    out_csv: str | Path,
) -> Path:
    """Load QC paths, merge overrides, and write one reviewed CSV."""

    qc = load_qc_frames(run_dir, csvs=csvs)
    overrides = load_overrides(overrides_csv)
    return write_reviewed_qc(qc, out_csv, overrides=overrides)


def selection_frame(df: pd.DataFrame, review_ids: Sequence[str]) -> pd.DataFrame:
    """Return rows matching selected review IDs, preserving input order."""

    wanted = [str(x) for x in review_ids if str(x)]
    if not wanted:
        return df.iloc[0:0].copy()
    order = {rid: i for i, rid in enumerate(wanted)}
    sel = df[df["review_id"].astype(str).isin(order)].copy()
    sel["_selection_order"] = sel["review_id"].map(order)
    return sel.sort_values("_selection_order").drop(columns=["_selection_order"])


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge manual QC overrides into PLEB/PQC CSVs.")
    p.add_argument("--run-dir", type=Path, help="Run directory to search for *_qc.csv files.")
    p.add_argument("--qc-csv", type=Path, action="append", default=[], help="Explicit QC CSV; may be repeated.")
    p.add_argument("--overrides", type=Path, help="Manual overrides CSV.")
    p.add_argument("--out", type=Path, required=True, help="Reviewed QC CSV output path.")
    p.add_argument("--init-overrides", type=Path, help="Create an empty overrides CSV and exit.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for non-interactive merge/export."""

    args = _build_arg_parser().parse_args(argv)
    if args.init_overrides is not None:
        write_overrides(empty_overrides(), args.init_overrides)
        return 0
    write_reviewed_from_paths(
        run_dir=args.run_dir,
        csvs=args.qc_csv,
        overrides_csv=args.overrides,
        out_csv=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
