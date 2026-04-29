#!/usr/bin/env python3
"""Matplotlib/table expert-review UI for PLEB/PQC outputs.

This version deliberately avoids Plotly. It is optimized for large QC outputs by:

- loading one QC CSV at a time;
- reading only columns needed for plotting/review during normal interaction;
- using a static Matplotlib scatter plot via st.pyplot;
- using st.dataframe row selection for review actions;
- writing manual overrides separately from raw PQC CSVs;
- reading the full QC CSV only when explicitly writing a reviewed full CSV.

Run from the repository root, for example:

    python -m streamlit run scripts/pleb_qc_review_matplotlib.py -- --run-dir outputs/my_run

Artifacts written by default:

- <run-dir>/qc_review/manual_qc_overrides.csv
- <run-dir>/qc_review/reviewed_current_qc.csv

Notes
-----
The plot does not subtract backend medians or otherwise fake JUMP removal.
If JUMPs were fitted upstream, choose the post-fit/JUMP-fitted residual column
from the sidebar. The selected residual column is shown explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Any

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

TIME_COLUMNS = (
    "mjd",
    "MJD",
    "toa_mjd",
    "sat",
    "bary_mjd",
    "barycentric_mjd",
)

# Prefer post-fit/JUMP-fitted residuals first. Raw-ish residuals are fallback.
RESIDUAL_PREFERENCE = (
    "postfit_us",
    "post_fit_us",
    "postfit_resid_us",
    "post_fit_resid_us",
    "post_resid_us",
    "post_us",
    "postfit",
    "post_fit",
    "post",
    "resid_postfit_us",
    "resid_post_fit_us",
    "residual_postfit_us",
    "residual_post_fit_us",
    "resid_postfit",
    "resid_post_fit",
    "residual_postfit",
    "residual_post_fit",
    "clean_resid_us",
    "clean_residual_us",
    "resid_us",
    "residual_us",
    "resid",
    "residual",
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
    "freq_mhz",
    "frequency_mhz",
    "f",
)

BACKEND_COLUMNS = (
    "sys",
    "-sys",
    "backend",
    "system",
    "group",
    "-group",
    "instrument",
)

TIMFILE_COLUMNS = (
    "_timfile",
    "timfile",
    "filename",
    "file",
    "source_tim",
)

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
    "eclipse_event_member",
    "gaussian_bump_member",
    "glitch_member",
    "exp_dip_member",
    "step_member",
    "dm_step_member",
)

IMPORTANT_DECISIONS = {"BAD_TOA", "REVIEW_EVENT", "EVENT"}

DEFAULT_MAX_KEEP_PLOT = 3000
DEFAULT_TABLE_ROWS = 800


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--run-dir", default="")
    p.add_argument("--overrides", default="")
    p.add_argument("--reviewed-out", default="")
    return p.parse_known_args()[0]


def _default_paths(run_dir: str) -> tuple[Path, Path]:
    root = Path(run_dir or ".").expanduser()
    review_dir = root / "qc_review"
    return review_dir / "manual_qc_overrides.csv", review_dir / "reviewed_current_qc.csv"


def _normalise_scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def make_review_id(row: Mapping[str, object]) -> str:
    """Stable row identifier compatible with the earlier qc_review submodule."""

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


def _guess_pulsar_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_qc"):
        stem = stem[:-3]
    for part in path.parts[::-1]:
        if part.startswith("J") and ("+" in part or "-" in part):
            return part
    return stem


def _guess_variant_from_path(path: Path) -> str:
    stem = path.stem.replace("_qc", "")
    # Common patterns: Jxxxx_variant_qc.csv or variant folder names.
    bits = stem.split("_", 1)
    if len(bits) == 2 and bits[1]:
        return bits[1]
    return "base"


def find_qc_csvs(run_dir: str | Path) -> list[Path]:
    """Find raw QC CSVs, excluding generated review outputs."""

    root = Path(run_dir).expanduser()
    if root.is_file():
        return [root]

    out: list[Path] = []
    for path in root.rglob("*_qc.csv"):
        if not path.is_file():
            continue
        parts = {p.lower() for p in path.parts}
        name = path.name.lower()
        if "qc_review" in parts:
            continue
        if name.startswith("reviewed") or "reviewed" in name:
            continue
        out.append(path)
    return sorted(out)


def choose_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    cols = {str(c): str(c) for c in columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    lower = {str(c).lower(): str(c) for c in columns}
    for c in candidates:
        hit = lower.get(str(c).lower())
        if hit is not None:
            return hit
    return None


def residual_column_options(columns: Sequence[str]) -> list[str]:
    options: list[str] = []
    colset = {str(c) for c in columns}
    lower_to_actual = {str(c).lower(): str(c) for c in columns}

    for c in RESIDUAL_PREFERENCE:
        if c in colset and c not in options:
            options.append(c)
        elif c.lower() in lower_to_actual:
            actual = lower_to_actual[c.lower()]
            if actual not in options:
                options.append(actual)

    keywords = ("resid", "residual", "post", "postfit", "clean")
    for c in columns:
        name = str(c).lower()
        if c in options:
            continue
        if any(k in name for k in keywords):
            options.append(str(c))

    return options


def _read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


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

    out = pd.Series("KEEP", index=df.index, dtype=object)
    out.loc[bad & ~event] = "BAD_TOA"
    out.loc[~bad & event] = "EVENT"
    out.loc[bad & event] = "REVIEW_EVENT"
    return out


def _series_or_default(df: pd.DataFrame, col: str | None, default: object = "") -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return df[col]


def _columns_to_read(header: Sequence[str], residual_col: str) -> list[str]:
    """Return a narrow usecols list for normal interactive use."""

    candidates = set(
        list(TIME_COLUMNS)
        + [residual_col]
        + list(UNCERTAINTY_COLUMNS)
        + list(FREQUENCY_COLUMNS)
        + list(BACKEND_COLUMNS)
        + list(TIMFILE_COLUMNS)
        + list(BAD_COLUMNS)
        + list(EVENT_COLUMNS)
        + ["decision", "pulsar", "variant", "transient_id"]
    )
    header_set = {str(c) for c in header}
    return [c for c in header if str(c) in candidates and str(c) in header_set]


@st.cache_data(show_spinner=False)
def load_qc_narrow(
    csv_path: str,
    root_dir: str,
    residual_col: str,
    file_size: int,
    file_mtime_ns: int,
) -> pd.DataFrame:
    """Load one QC CSV with a narrow set of columns."""

    del file_size, file_mtime_ns  # cache-key inputs only

    path = Path(csv_path)
    root = Path(root_dir) if root_dir else None
    header = _read_header(path)
    usecols = _columns_to_read(header, residual_col)
    if residual_col not in usecols and residual_col in header:
        usecols.append(residual_col)

    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    time_col = choose_column(df.columns, TIME_COLUMNS)
    unc_col = choose_column(df.columns, UNCERTAINTY_COLUMNS)
    freq_col = choose_column(df.columns, FREQUENCY_COLUMNS)
    backend_col = choose_column(df.columns, BACKEND_COLUMNS)
    timfile_col = choose_column(df.columns, TIMFILE_COLUMNS)

    out = df.copy()

    try:
        qc_csv = (
            path.resolve().relative_to(root.expanduser().resolve()).as_posix()
            if root is not None
            else path.resolve().as_posix()
        )
    except Exception:
        qc_csv = path.resolve().as_posix()

    if "pulsar" not in out.columns:
        out["pulsar"] = _guess_pulsar_from_path(path)
    if "variant" not in out.columns:
        out["variant"] = _guess_variant_from_path(path)

    out["qc_csv"] = qc_csv
    out["row_index"] = np.arange(len(out), dtype=int)
    out["auto_decision"] = infer_auto_decision(out)

    out["mjd"] = pd.to_numeric(_series_or_default(out, time_col), errors="coerce")
    out["residual"] = pd.to_numeric(_series_or_default(out, residual_col), errors="coerce")
    out["uncertainty"] = pd.to_numeric(_series_or_default(out, unc_col), errors="coerce")
    out["freq"] = pd.to_numeric(_series_or_default(out, freq_col), errors="coerce")
    out["backend"] = _series_or_default(out, backend_col).fillna("").astype(str)
    out["timfile"] = _series_or_default(out, timfile_col).fillna("").astype(str)

    # Vectorizing the hash is awkward; this still avoids the old full-row dict
    # conversion and hashes only narrow identity fields.
    records = out[
        ["qc_csv", "row_index", "pulsar", "variant", "timfile", "mjd", "freq", "backend"]
    ].to_dict("records")
    out["review_id"] = [make_review_id(r) for r in records]
    return out


def empty_overrides() -> pd.DataFrame:
    return pd.DataFrame(columns=OVERRIDE_COLUMNS)


def load_overrides(path: Path | None) -> pd.DataFrame:
    if path is None:
        return empty_overrides()
    if not path.exists() or path.stat().st_size == 0:
        return empty_overrides()
    df = pd.read_csv(path, dtype={"review_id": str, "qc_csv": str})
    for col in OVERRIDE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[OVERRIDE_COLUMNS].copy()


def validate_overrides(overrides: pd.DataFrame) -> None:
    if overrides.empty:
        return
    actions = overrides.get("manual_action", pd.Series(dtype=object)).fillna("").astype(str)
    invalid = sorted(set(actions) - set(REVIEW_ACTIONS) - {""})
    if invalid:
        raise ValueError(f"Unsupported manual_action values {invalid}; allowed: {', '.join(REVIEW_ACTIONS)}")


def write_overrides(overrides: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = overrides.copy()
    for col in OVERRIDE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df[OVERRIDE_COLUMNS].to_csv(path, index=False)
    return path


def _latest_effective_overrides(overrides: pd.DataFrame) -> pd.DataFrame:
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
        cols = ["review_id", "override_id", "manual_action", "manual_reason", "reviewer", "reviewed_at"]
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
        reviewed.loc[action == "clear_manual"] = merged.loc[action == "clear_manual", "auto_decision"]
        merged["reviewed_decision"] = reviewed.fillna("KEEP").astype(str).str.upper()

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

    out["reviewed_bad_point"] = out["reviewed_decision"].eq("BAD_TOA") | out["reviewed_decision"].eq("REVIEW_EVENT")
    out["reviewed_event_member"] = out["reviewed_decision"].eq("EVENT") | out["reviewed_decision"].eq("REVIEW_EVENT")
    out["reviewed_keep"] = out["reviewed_decision"].eq("KEEP")
    return out


def make_override_rows(
    rows: pd.DataFrame,
    *,
    action: str,
    reason: str,
    reviewer: str,
    source: str = "streamlit_matplotlib_review",
) -> pd.DataFrame:
    if action not in REVIEW_ACTIONS:
        raise ValueError(f"Unsupported action: {action}")

    timestamp = datetime.now(timezone.utc).isoformat()
    out_rows: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        override_id = hashlib.sha1(
            f"{row.get('review_id','')}\x1f{action}\x1f{timestamp}\x1f{len(out_rows)}".encode("utf-8")
        ).hexdigest()[:16]
        out_rows.append(
            {
                "override_id": override_id,
                "review_id": row.get("review_id", ""),
                "qc_csv": row.get("qc_csv", ""),
                "row_index": row.get("row_index", ""),
                "pulsar": row.get("pulsar", ""),
                "variant": row.get("variant", ""),
                "timfile": row.get("timfile", ""),
                "mjd": row.get("mjd", ""),
                "freq": row.get("freq", ""),
                "backend": row.get("backend", ""),
                "manual_action": action,
                "manual_reason": reason,
                "reviewer": reviewer,
                "reviewed_at": timestamp,
                "source": source,
            }
        )
    return pd.DataFrame(out_rows, columns=OVERRIDE_COLUMNS)


def selection_frame(df: pd.DataFrame, review_ids: Sequence[str]) -> pd.DataFrame:
    wanted = [str(x) for x in review_ids if str(x)]
    if not wanted:
        return df.iloc[0:0].copy()
    order = {rid: i for i, rid in enumerate(wanted)}
    sel = df[df["review_id"].astype(str).isin(order)].copy()
    sel["_selection_order"] = sel["review_id"].map(order)
    return sel.sort_values("_selection_order").drop(columns=["_selection_order"])


def _extract_selected_rows(event: Any) -> list[int]:
    """Extract selected positional rows from Streamlit dataframe event."""

    if not event:
        return []
    try:
        selection = event.get("selection", {})
    except AttributeError:
        selection = getattr(event, "selection", {}) or {}
    try:
        rows = selection.get("rows", [])
    except AttributeError:
        rows = getattr(selection, "rows", []) or []
    return [int(x) for x in rows]


def _compact_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "review_id",
        "row_index",
        "pulsar",
        "variant",
        "mjd",
        "residual",
        "backend",
        "freq",
        "timfile",
        "auto_decision",
        "reviewed_decision",
        "manual_action",
        "manual_reason",
        "uncertainty",
        "qc_csv",
    ]
    return [c for c in cols if c in df.columns]


def _filter_view(
    df: pd.DataFrame,
    *,
    decisions: list[str],
    backends: list[str],
    only_suspicious: bool,
    abs_resid_min: float | None,
) -> pd.DataFrame:
    view = df
    if decisions:
        view = view[view["reviewed_decision"].astype(str).isin(decisions)]
    if backends:
        view = view[view["backend"].astype(str).isin(backends)]
    if only_suspicious:
        view = view[view["reviewed_decision"].astype(str).str.upper().isin(IMPORTANT_DECISIONS) | (view["manual_action"].astype(str) != "")]
    if abs_resid_min is not None and abs_resid_min > 0:
        view = view[pd.to_numeric(view["residual"], errors="coerce").abs() >= float(abs_resid_min)]
    return view


def _plot_sample(
    df: pd.DataFrame,
    *,
    max_keep_points: int,
    plot_all_keep: bool,
) -> pd.DataFrame:
    plot_df = df.dropna(subset=["mjd", "residual"]).copy()
    if plot_df.empty:
        return plot_df

    decision = plot_df["reviewed_decision"].fillna("KEEP").astype(str).str.upper()
    important = decision.isin(IMPORTANT_DECISIONS) | (plot_df["manual_action"].fillna("").astype(str) != "")

    important_df = plot_df[important].copy()
    keep_df = plot_df[~important].copy()

    if not plot_all_keep and len(keep_df) > max_keep_points:
        keep_df = keep_df.sample(int(max_keep_points), random_state=260408373)

    return pd.concat([important_df, keep_df], ignore_index=True, sort=False)


def _make_matplotlib_figure(plot_df: pd.DataFrame, *, residual_col: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)

    if plot_df.empty:
        ax.text(0.5, 0.5, "No plottable rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("MJD")
        ax.set_ylabel(residual_col)
        return fig

    order = ["KEEP", "EVENT", "REVIEW_EVENT", "BAD_TOA"]
    colors = {
        "KEEP": "0.55",
        "EVENT": "#1f77b4",
        "REVIEW_EVENT": "#ff7f0e",
        "BAD_TOA": "#d62728",
    }
    sizes = {"KEEP": 8, "EVENT": 18, "REVIEW_EVENT": 26, "BAD_TOA": 26}
    alphas = {"KEEP": 0.35, "EVENT": 0.75, "REVIEW_EVENT": 0.85, "BAD_TOA": 0.85}

    decision = plot_df["reviewed_decision"].fillna("KEEP").astype(str).str.upper()
    for dec in order + sorted(set(decision) - set(order)):
        g = plot_df[decision == dec]
        if g.empty:
            continue
        ax.scatter(
            pd.to_numeric(g["mjd"], errors="coerce"),
            pd.to_numeric(g["residual"], errors="coerce"),
            s=sizes.get(dec, 16),
            alpha=alphas.get(dec, 0.7),
            label=dec,
            c=colors.get(dec, None),
            edgecolors="none",
        )

    manual = plot_df[plot_df["manual_action"].fillna("").astype(str) != ""]
    if not manual.empty:
        ax.scatter(
            pd.to_numeric(manual["mjd"], errors="coerce"),
            pd.to_numeric(manual["residual"], errors="coerce"),
            s=60,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            label="manual",
        )

    ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
    ax.set_xlabel("MJD")
    ax.set_ylabel(residual_col)
    ax.set_title("Residual vs MJD")
    ax.grid(True, alpha=0.18, lw=0.6)
    ax.legend(loc="best", fontsize=8, frameon=True, ncol=2)
    return fig


def _safe_stat(path: Path) -> tuple[int, int]:
    try:
        s = path.stat()
        return int(s.st_size), int(s.st_mtime_ns)
    except OSError:
        return -1, -1


def _load_full_current_and_apply(
    csv_path: Path,
    root_dir: Path,
    residual_col: str,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    """Read full current CSV only on demand and attach reviewed columns."""

    full = pd.read_csv(csv_path, low_memory=False)
    header = list(full.columns)

    time_col = choose_column(header, TIME_COLUMNS)
    unc_col = choose_column(header, UNCERTAINTY_COLUMNS)
    freq_col = choose_column(header, FREQUENCY_COLUMNS)
    backend_col = choose_column(header, BACKEND_COLUMNS)
    timfile_col = choose_column(header, TIMFILE_COLUMNS)

    out = full.copy()

    try:
        qc_csv = csv_path.resolve().relative_to(root_dir.resolve()).as_posix()
    except Exception:
        qc_csv = csv_path.resolve().as_posix()

    if "pulsar" not in out.columns:
        out["pulsar"] = _guess_pulsar_from_path(csv_path)
    if "variant" not in out.columns:
        out["variant"] = _guess_variant_from_path(csv_path)

    out["qc_csv"] = qc_csv
    out["row_index"] = np.arange(len(out), dtype=int)
    out["auto_decision"] = infer_auto_decision(out)
    out["mjd"] = pd.to_numeric(_series_or_default(out, time_col), errors="coerce")
    out["residual"] = pd.to_numeric(_series_or_default(out, residual_col), errors="coerce")
    out["uncertainty"] = pd.to_numeric(_series_or_default(out, unc_col), errors="coerce")
    out["freq"] = pd.to_numeric(_series_or_default(out, freq_col), errors="coerce")
    out["backend"] = _series_or_default(out, backend_col).fillna("").astype(str)
    out["timfile"] = _series_or_default(out, timfile_col).fillna("").astype(str)

    records = out[
        ["qc_csv", "row_index", "pulsar", "variant", "timfile", "mjd", "freq", "backend"]
    ].to_dict("records")
    out["review_id"] = [make_review_id(r) for r in records]
    return apply_overrides(out, overrides)


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="PLEB QC Review — Matplotlib", layout="wide")
    st.title("PLEB QC Expert Review — Matplotlib/table mode")
    st.caption(
        "No Plotly. Static Matplotlib plot plus selectable table. "
        "Manual overrides are separate; raw PQC CSVs are not modified."
    )

    with st.sidebar:
        st.header("Inputs")
        run_dir_text = st.text_input("Run directory or QC CSV", value=args.run_dir or "")
        if not run_dir_text:
            st.info("Enter a run directory containing `*_qc.csv` files, or a single QC CSV.")
            return

        run_path = Path(run_dir_text).expanduser()
        root_dir = run_path.parent if run_path.is_file() else run_path

        default_overrides, default_reviewed = _default_paths(str(root_dir))
        overrides_path = Path(st.text_input("Overrides CSV", value=args.overrides or str(default_overrides))).expanduser()
        reviewed_out = Path(st.text_input("Reviewed current QC output", value=args.reviewed_out or str(default_reviewed))).expanduser()

        reload_clicked = st.button("Reload file list / clear data cache")
        if reload_clicked:
            load_qc_narrow.clear()

    try:
        csvs = find_qc_csvs(run_path)
    except Exception as e:
        st.error(f"Failed to find QC CSVs: {e}")
        return

    if not csvs:
        st.error("No raw `*_qc.csv` files found.")
        return

    labels = []
    for p in csvs:
        try:
            labels.append(p.resolve().relative_to(root_dir.resolve()).as_posix())
        except Exception:
            labels.append(p.name)

    with st.sidebar:
        selected_label = st.selectbox("QC CSV", labels, index=0)
        csv_path = csvs[labels.index(selected_label)]

    try:
        header = _read_header(csv_path)
    except Exception as e:
        st.error(f"Failed to read CSV header: {csv_path}: {e}")
        return

    residual_options = residual_column_options(header)
    if not residual_options:
        st.error("No residual-like column found in this QC CSV.")
        st.write("Columns:", header)
        return

    with st.sidebar:
        st.header("Residual")
        residual_col = st.selectbox(
            "Residual column to plot",
            residual_options,
            index=0,
            help="Choose the post-fit/JUMP-fitted residual column if present.",
        )
        if residual_col in {"resid_us", "residual_us", "resid", "residual"}:
            st.warning(
                "This looks like a raw-ish residual column. If a post/postfit column exists, prefer that."
            )

        st.header("Review action")
        reviewer = st.text_input("Reviewer", value="")
        reason = st.text_input("Reason", value="manual review")
        action = st.selectbox("Manual action", REVIEW_ACTIONS, index=0)

        st.header("Plot / table")
        plot_all_keep = st.checkbox("Plot all KEEP rows", value=False)
        max_keep_plot = int(
            st.number_input(
                "Max sampled KEEP rows in plot",
                min_value=100,
                max_value=250000,
                value=DEFAULT_MAX_KEEP_PLOT,
                step=500,
                disabled=plot_all_keep,
            )
        )
        table_limit = int(
            st.number_input(
                "Max table rows",
                min_value=50,
                max_value=50000,
                value=DEFAULT_TABLE_ROWS,
                step=50,
            )
        )
        only_suspicious = st.checkbox("Table: suspicious/manual rows only", value=True)
        abs_resid_min = float(
            st.number_input(
                "Table: min |residual|",
                min_value=0.0,
                value=0.0,
                step=0.1,
            )
        )

    size, mtime = _safe_stat(csv_path)
    try:
        qc = load_qc_narrow(
            str(csv_path.resolve()),
            str(root_dir.resolve()),
            residual_col,
            size,
            mtime,
        )
    except Exception as e:
        st.error(f"Failed to load narrow QC table: {e}")
        return

    try:
        overrides = load_overrides(overrides_path)
    except Exception as e:
        st.error(f"Failed to load overrides: {e}")
        return

    reviewed = apply_overrides(qc, overrides)

    with st.sidebar:
        decisions = sorted(str(x) for x in reviewed["reviewed_decision"].dropna().unique())
        selected_decisions = st.multiselect("Decision filter", decisions, default=decisions)
        backends = sorted(str(x) for x in reviewed["backend"].dropna().unique())
        selected_backends = st.multiselect("Backend filter", backends, default=[])

    view = _filter_view(
        reviewed,
        decisions=selected_decisions,
        backends=selected_backends,
        only_suspicious=False,
        abs_resid_min=None,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows in CSV", len(reviewed))
    c2.metric("Visible rows", len(view))
    c3.metric("Bad", int(view["reviewed_bad_point"].sum()))
    c4.metric("Event", int(view["reviewed_event_member"].sum()))
    c5.metric("Manual overrides", len(overrides))

    st.caption(
        f"CSV: `{selected_label}` · plotting residual column: `{residual_col}` · "
        "no display-time backend/JUMP centering is applied."
    )

    plot_df = _plot_sample(view, max_keep_points=max_keep_plot, plot_all_keep=plot_all_keep)
    if len(plot_df) < len(view.dropna(subset=["mjd", "residual"])):
        st.caption(
            f"Plotting {len(plot_df):,} sampled/important rows. "
            "All BAD/EVENT/manual rows are retained; KEEP rows may be sampled."
        )

    fig = _make_matplotlib_figure(plot_df, residual_col=residual_col)
    st.pyplot(fig, clear_figure=True)

    table_view = _filter_view(
        view,
        decisions=selected_decisions,
        backends=selected_backends,
        only_suspicious=only_suspicious,
        abs_resid_min=abs_resid_min,
    )
    table_view = table_view.sort_values(["reviewed_decision", "mjd"], ascending=[True, True])
    table_show = table_view[_compact_columns(table_view)].head(table_limit).reset_index(drop=True)

    st.subheader("Select rows to review")
    st.caption(
        "Select rows in the table, choose an action in the sidebar, then click Apply. "
        "If your Streamlit version does not support row selection, paste review IDs below."
    )

    event = st.dataframe(
        table_show,
        use_container_width=True,
        hide_index=True,
        height=360,
        on_select="rerun",
        selection_mode="multi-row",
    )
    selected_positions = _extract_selected_rows(event)
    selected_ids_from_table = []
    if selected_positions:
        selected_ids_from_table = [
            str(table_show.iloc[i]["review_id"])
            for i in selected_positions
            if 0 <= i < len(table_show)
        ]

    manual_ids_text = st.text_area(
        "Selected review IDs",
        value="\n".join(selected_ids_from_table),
        height=90,
        help="One review_id per line. This is also a fallback if table selection is unavailable.",
    )
    selected_ids = [x.strip() for x in manual_ids_text.splitlines() if x.strip()]
    selected = selection_frame(reviewed, selected_ids)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Selected TOAs")
        st.dataframe(
            selected[_compact_columns(selected)] if not selected.empty else selected,
            use_container_width=True,
            height=220,
        )

    with right:
        st.subheader("Apply")
        if st.button("Apply action to selected", disabled=selected.empty):
            new_rows = make_override_rows(
                selected,
                action=action,
                reason=reason,
                reviewer=reviewer,
            )
            merged = pd.concat([overrides, new_rows], ignore_index=True, sort=False)
            write_overrides(merged, overrides_path)
            load_qc_narrow.clear()
            st.success(f"Wrote {len(new_rows)} override row(s) to {overrides_path}")
            st.rerun()

        if st.button("Write reviewed current QC CSV"):
            with st.spinner("Reading full current QC CSV and applying overrides..."):
                full_reviewed = _load_full_current_and_apply(
                    csv_path=csv_path,
                    root_dir=root_dir,
                    residual_col=residual_col,
                    overrides=overrides,
                )
                reviewed_out.parent.mkdir(parents=True, exist_ok=True)
                full_reviewed.to_csv(reviewed_out, index=False)
            st.success(f"Wrote reviewed current QC CSV to {reviewed_out}")

    with st.expander("Manual override audit table", expanded=False):
        st.dataframe(overrides, use_container_width=True, height=280)

    with st.expander("Debug / columns", expanded=False):
        st.write("Raw CSV columns:")
        st.write(header)
        st.write("Loaded narrow columns:")
        st.write(list(qc.columns))


if __name__ == "__main__":
    main()
