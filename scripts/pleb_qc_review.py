#!/usr/bin/env python3
"""Streamlit expert-review UI for PLEB/PQC outputs.

Run from the repository root, for example:

    python -m streamlit run scripts/pleb_qc_review.py -- --run-dir outputs/my_run

The app writes two artifacts by default:

- manual_qc_overrides.csv: append-only expert decisions
- qc_review/<selected *_qc.csv basename>: raw QC rows plus reviewed decision
  columns, using a FixDataset-discoverable filename

This version deliberately avoids st.cache_data for QC loading. Plotly selection
callbacks rerun the script frequently, so QC data is kept in ``st.session_state``
and only the currently selected QC CSV is loaded. The app reloads that file
only when the selected pulsar/QC CSV changes or when the user clicks reload.

Residual-column notes
---------------------
The plot does not subtract backend medians or otherwise fake JUMP removal.
If JUMPs were fitted upstream, the GUI should plot the post-fit/JUMP-fitted
residual column produced by that stage. The sidebar exposes the residual column
being plotted so users can verify and override it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:  # Plotly is in the ``gui`` extra for this patch.
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - handled in UI
    go = None  # type: ignore

from pleb.qc_review import (
    REVIEW_ACTIONS,
    append_overrides,
    apply_overrides,
    find_qc_csvs,
    load_qc_csv,
    load_overrides,
    make_override_rows,
    selection_frame,
    write_overrides,
    write_reviewed_qc,
)

DEFAULT_MAX_KEEP_POINTS = 3000
DEFAULT_TABLE_PREVIEW_ROWS = 500
IMPORTANT_DECISIONS = {"BAD_TOA", "REVIEW_EVENT", "EVENT"}
PLOT_KEY = "qc_review_residual_plot_fast"
PLOT_SELECTION_MODES = ("box", "lasso")
PLOT_DEFAULT_DRAGMODE = "select"
PLOT_MODEBAR_BUTTONS_TO_ADD = ("select2d", "lasso2d")
X_AXIS_PREFERENCE = (
    "mjd",
    "uncertainty",
    "tempo2_err_us",
    "tempo2_err",
    "sigma_us",
    "toa_err_us",
    "err_us",
    "freq",
    "frequency",
    "freq_mhz",
    "obs_freq",
    "orbital_phase",
    "binphase",
    "post_phase",
    "pre_phase",
    "solar_elongation_deg",
    "solarangle",
    "elev",
)
X_AXIS_KEYWORDS = (
    "mjd",
    "time",
    "err",
    "sigma",
    "uncert",
    "phase",
    "freq",
    "solar",
    "elong",
    "angle",
    "elev",
)

# Prefer fitted/postfit residuals first. Raw-ish residual columns are fallback.
RESIDUAL_PREFERENCE = (
    "tempo2_post_us",
    "tempo2_postfit_us",
    "tempo2_post",
    "tempo2_postfit",
    "tempo2_pre_us",
    "tempo2_pre",
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
    "resid_us",
    "residual_us",
    "clean_resid_us",
    "clean_residual_us",
    "clean_resid",
    "clean_residual",
    "resid",
    "residual",
)

POSTFIT_COLUMNS = (
    "tempo2_post_us",
    "tempo2_postfit_us",
    "tempo2_post",
    "tempo2_postfit",
)

FEATURE_COLUMN_PREFERENCE = (
    "bad_point",
    "bad_mad",
    "bad_ou",
    "bad_hard",
    "robust_outlier",
    "robust_global_outlier",
    "outlier_any",
    "event_member",
    "event_any",
    "transient_id",
    "solar_event_member",
    "orbital_phase",
    "orbital_phase_bad",
    "eclipse_member",
    "eclipse_event_member",
    "gaussian_bump_member",
    "glitch_member",
    "step_id",
    "dm_step_id",
    "step_member",
    "dm_step_member",
    "solar_elongation_deg",
    "freq_bin",
)

FEATURE_VALUE_COLUMNS = (
    "orbital_phase",
    "solar_elongation_deg",
    "freq_bin",
)

COLOR_COLUMN_PREFERENCE = (
    "reviewed_decision",
    "backend",
    "sys",
    "-sys",
    "system",
    "tel",
    "telescope",
    "observatory",
    "pta",
    "group",
    "-group",
    "fe",
    "be",
    "freq",
    "frequency",
    "freq_mhz",
    "frequency_mhz",
    "freq_meta",
    "cenfreq",
    "freq_bin",
)
MAX_COLOR_CATEGORIES = 40
DECISION_MARKER_SYMBOLS = {
    "KEEP": "circle",
    "BAD_TOA": "x",
    "EVENT": "diamond",
    "REVIEW_EVENT": "square",
}
DEFAULT_MARKER_SYMBOL = "circle-open"
PLOTLY_CATEGORY_COLORS = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--run-dir", default="")
    p.add_argument("--overrides", default="")
    p.add_argument("--reviewed-out", default="")
    return p.parse_known_args()[0]


def _default_paths(run_dir: str) -> tuple[Path, Path]:
    root = Path(run_dir or ".").expanduser()
    review_dir = root / "qc_review"
    return review_dir / "manual_qc_overrides.csv", review_dir / "reviewed_qc.csv"


def _default_reviewed_path(run_dir: str, qc_csv_path: str | Path | None) -> Path:
    root = Path(run_dir or ".").expanduser()
    review_dir = root / "qc_review"
    if qc_csv_path is not None:
        try:
            name = Path(qc_csv_path).name.strip()
        except Exception:
            name = str(qc_csv_path).strip()
        if name.lower().endswith(".csv") and name:
            return review_dir / name
    return review_dir / "reviewed_qc.csv"


def _sync_reviewed_output_default(
    run_dir: str,
    qc_csv_path: str | Path | None,
    *,
    cli_reviewed_out: str = "",
) -> Path:
    auto_default = _default_reviewed_path(run_dir, qc_csv_path)
    if cli_reviewed_out:
        chosen = Path(cli_reviewed_out).expanduser()
        st.session_state["reviewed_out_path"] = str(chosen)
        st.session_state["_reviewed_out_auto_default"] = str(auto_default)
        return chosen

    current = str(st.session_state.get("reviewed_out_path", "")).strip()
    previous_auto = str(st.session_state.get("_reviewed_out_auto_default", "")).strip()
    if not current or current == previous_auto:
        st.session_state["reviewed_out_path"] = str(auto_default)
    st.session_state["_reviewed_out_auto_default"] = str(auto_default)
    return Path(
        str(st.session_state.get("reviewed_out_path", auto_default))
    ).expanduser()


def _resolved_path_key(path: str | Path) -> str:
    """Return a stable string key for one local path."""

    try:
        return str(Path(path).expanduser().resolve())
    except Exception:
        return str(Path(path).expanduser())


def _guess_pulsar_from_qc_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_qc"):
        stem = stem[:-3]
    for sep in ("__", "."):
        if sep in stem:
            return stem.split(sep, 1)[0]
    return stem


def _guess_variant_from_qc_path(path: Path, pulsar: str) -> str:
    stem = path.stem
    if stem.endswith("_qc"):
        stem = stem[:-3]
    prefixes = (f"{pulsar}.", f"{pulsar}_")
    for prefix in prefixes:
        if stem.startswith(prefix):
            variant = stem[len(prefix) :].strip("._")
            if variant:
                return variant
    return "base"


def _label_qc_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.name


def _manifest_key(run_dir: str) -> dict[str, str]:
    return {"run_dir": _resolved_path_key(run_dir)}


def _load_manifest_once(
    run_dir: str,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Load the QC CSV manifest into session_state.

    Plotly selections rerun the Streamlit script. This function therefore does
    not use filesystem-fingerprint cache keys. It reloads only when the run
    directory changes or when the user explicitly asks for reload.
    """

    key = _manifest_key(run_dir)
    must_load = (
        force
        or st.session_state.get("_loaded_manifest_key") != key
        or "qc_manifest_df" not in st.session_state
    )
    if must_load:
        root = Path(run_dir).expanduser()
        with st.spinner("Scanning QC CSVs..."):
            rows: list[dict[str, str]] = []
            for path in find_qc_csvs(root):
                pulsar = _guess_pulsar_from_qc_path(path)
                rows.append(
                    {
                        "path": _resolved_path_key(path),
                        "label": _label_qc_path(path, root),
                        "pulsar": pulsar,
                        "variant": _guess_variant_from_qc_path(path, pulsar),
                    }
                )
            manifest = pd.DataFrame(rows)
            if not manifest.empty:
                manifest = manifest.sort_values(
                    ["pulsar", "variant", "label"], kind="stable"
                ).reset_index(drop=True)
            st.session_state["qc_manifest_df"] = manifest
            st.session_state["_loaded_manifest_key"] = key
            _reset_plot_selection()
            st.session_state.pop("_loaded_qc_key", None)

    manifest = st.session_state["qc_manifest_df"]
    if manifest.empty:
        return manifest

    valid_paths = set(manifest["path"].astype(str))
    current_csv = str(st.session_state.get("current_qc_csv", ""))
    if current_csv not in valid_paths:
        current_csv = str(manifest.iloc[0]["path"])
        st.session_state["current_qc_csv"] = current_csv

    current_rows = manifest[manifest["path"].astype(str) == current_csv]
    if not current_rows.empty:
        st.session_state["current_pulsar"] = str(current_rows.iloc[0]["pulsar"])
    else:
        st.session_state["current_pulsar"] = str(manifest.iloc[0]["pulsar"])
    return manifest


def _load_overrides_once(
    overrides_path: Path,
    *,
    force: bool = False,
) -> pd.DataFrame:
    key = {"overrides_path": _resolved_path_key(overrides_path)}
    must_load = (
        force
        or st.session_state.get("_loaded_overrides_key") != key
        or "overrides_df" not in st.session_state
    )
    if must_load:
        st.session_state["overrides_df"] = load_overrides(overrides_path)
        st.session_state["_loaded_overrides_key"] = key
    return st.session_state["overrides_df"]


def _load_qc_once(
    csv_path: Path,
    *,
    root_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    key = {
        "csv_path": _resolved_path_key(csv_path),
        "root_dir": _resolved_path_key(root_dir),
    }
    must_load = (
        force
        or st.session_state.get("_loaded_qc_key") != key
        or "qc_df" not in st.session_state
    )
    if must_load:
        with st.spinner(f"Loading QC CSV: {csv_path.name}"):
            st.session_state["qc_df"] = load_qc_csv(csv_path, root=root_dir)
            st.session_state["_loaded_qc_key"] = key
            _reset_plot_selection()
    return st.session_state["qc_df"]


def _extract_selected_review_ids(event: Any) -> list[str]:
    """Extract review IDs from a Streamlit Plotly selection event."""

    if not event:
        return []
    try:
        selection = event.get("selection", {})
    except AttributeError:
        selection = getattr(event, "selection", {}) or {}
    try:
        points = selection.get("points", [])
    except AttributeError:
        points = getattr(selection, "points", []) or []

    ids: list[str] = []
    lookup = st.session_state.get("_plot_review_id_lookup", [])
    for point in points or []:
        custom = None
        if isinstance(point, dict):
            custom = point.get("customdata")
        else:
            custom = getattr(point, "customdata", None)
        if isinstance(custom, (list, tuple, np.ndarray)) and len(custom):
            ids.append(str(custom[0]))
            continue
        if custom is None:
            continue
        try:
            rid_i = int(custom)
        except (TypeError, ValueError):
            ids.append(str(custom))
            continue
        if 0 <= rid_i < len(lookup):
            ids.append(str(lookup[rid_i]))

    # Keep order, drop duplicates.
    return list(dict.fromkeys(ids))


def _sync_plot_selection() -> None:
    """Copy the chart selection into session state on selection callbacks."""

    event_key = st.session_state.get("_plot_event_key", PLOT_KEY)
    st.session_state["selected_review_ids"] = _extract_selected_review_ids(
        st.session_state.get(event_key)
    )


def _reset_plot_selection() -> None:
    st.session_state["selected_review_ids"] = []
    st.session_state["_plot_reset_token"] = (
        int(st.session_state.get("_plot_reset_token", 0)) + 1
    )
    st.session_state.pop("_plot_review_id_lookup", None)


def _compact_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "review_id",
        "pulsar",
        "variant",
        "qc_csv",
        "row_index",
        "timfile",
        "mjd",
        "freq",
        "backend",
        "residual",
        "_plot_residual",
        "_plot_residual_column",
        "uncertainty",
        "auto_decision",
        "reviewed_decision",
        "manual_action",
        "manual_reason",
    ]
    cols.extend(_feature_columns(df))
    return [c for c in cols if c in df.columns]


def _normalised_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _residual_column_options(df: pd.DataFrame) -> list[str]:
    """Return numeric-looking residual columns, preferred order first."""

    options: list[str] = []
    for col in RESIDUAL_PREFERENCE:
        if col in df.columns and col not in options:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                options.append(col)

    # Include any other residual/postfit-looking numeric columns for inspection.
    keywords = ("resid", "residual", "post", "postfit", "clean")
    for col in df.columns:
        name = str(col).lower()
        if col in options:
            continue
        if not any(k in name for k in keywords):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            options.append(str(col))

    return options


def _numeric_axis_options(df: pd.DataFrame) -> list[str]:
    """Return numeric columns suitable for use on the x-axis."""

    actual_cols = [str(c) for c in df.columns]
    lower_to_actual = {str(c).lower(): str(c) for c in df.columns}
    options: list[str] = []

    def _usable_axis_col(col: str) -> bool:
        lower = col.lower()
        if lower == "row_index" or lower.startswith("_plot_") or col.startswith("_"):
            return False
        if lower.startswith(("reviewed_", "auto_")):
            return False
        if lower.endswith(("_bad", "_member", "_any")):
            return False
        if col not in df.columns:
            return False

        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            return False

        numeric = pd.to_numeric(series, errors="coerce")
        values = numeric.dropna()
        if values.empty:
            return False

        # Hide simple 0/1 flag columns from the main axis chooser.
        if values.nunique(dropna=True) <= 2 and values.isin([0.0, 1.0]).all():
            return False
        return True

    for candidate in X_AXIS_PREFERENCE:
        actual = lower_to_actual.get(candidate.lower())
        if actual and actual not in options and _usable_axis_col(actual):
            options.append(actual)

    for col in actual_cols:
        lower = col.lower()
        if col in options or not _usable_axis_col(col):
            continue
        if any(keyword in lower for keyword in X_AXIS_KEYWORDS):
            options.append(col)

    for col in actual_cols:
        if col in options or not _usable_axis_col(col):
            continue
        options.append(col)

    return options


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return review-relevant PQC feature columns present in a table."""

    cols: list[str] = []
    for col in FEATURE_COLUMN_PREFERENCE:
        if col in df.columns and col not in cols:
            cols.append(col)

    keywords = (
        "bad",
        "outlier",
        "event",
        "transient",
        "glitch",
        "bump",
        "eclipse",
        "orbital",
        "solar",
        "step",
        "freq_bin",
    )
    for col in df.columns:
        name = str(col).lower()
        if col in cols or name.startswith(("reviewed_", "manual_", "auto_")):
            continue
        if any(keyword in name for keyword in keywords):
            cols.append(str(col))
    return cols


def _color_column_options(df: pd.DataFrame) -> list[str]:
    """Return low-cardinality columns useful for trace coloring."""

    options: list[str] = []
    skip_names = {
        "review_id",
        "qc_csv",
        "timfile",
        "filename",
        "file",
        "_timfile",
        "_timfile_base",
    }

    def add(col: str, *, force: bool = False) -> None:
        if col in options or col not in df.columns:
            return
        lower = str(col).lower()
        if lower in skip_names or lower.startswith("_plot_"):
            return
        values = df[col].dropna()
        if values.empty:
            return
        n_unique = int(values.astype(str).nunique(dropna=True))
        if force or 1 <= n_unique <= MAX_COLOR_CATEGORIES:
            options.append(str(col))

    for col in COLOR_COLUMN_PREFERENCE:
        add(col, force=True)
    for col in _feature_columns(df):
        add(col)

    keywords = (
        "sys",
        "backend",
        "tel",
        "telescope",
        "observatory",
        "group",
        "flag",
        "event",
        "bad",
        "outlier",
        "step",
        "glitch",
        "bump",
        "freq",
        "frequency",
        "cenfreq",
        "freq_bin",
    )
    for col in df.columns:
        lower = str(col).lower()
        if any(keyword in lower for keyword in keywords):
            add(str(col))
    return options


def _filterable_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns that represent active flags or event IDs."""

    value_cols = set(FEATURE_VALUE_COLUMNS)
    return [col for col in _feature_columns(df) if col not in value_cols]


def _boolish_feature_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0) != 0
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y", "bad", "outlier", "event"})


def _id_feature_mask(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.fillna("").astype(str).str.strip().str.lower()
    return (numeric.fillna(-1) >= 0) | (
        numeric.isna() & ~text.isin({"", "nan", "none", "null", "-1"})
    )


def _feature_mask(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    if col.endswith("_id"):
        return _id_feature_mask(df[col])
    return _boolish_feature_mask(df[col])


def _active_feature_labels(row: pd.Series, feature_cols: list[str]) -> str:
    labels: list[str] = []
    for col in feature_cols:
        value = row.get(col)
        if col.endswith("_id"):
            text = "" if pd.isna(value) else str(value).strip()
            if text and text.lower() not in {"nan", "none", "null", "-1"}:
                labels.append(f"{col}={text}")
            continue
        if bool(_boolish_feature_mask(pd.Series([value])).iloc[0]):
            labels.append(col)
    return ", ".join(labels)


def _filter_view(
    reviewed: pd.DataFrame,
    *,
    selected_pulsars: list[str],
    selected_variants: list[str],
    selected_decisions: list[str],
    selected_backends: list[str],
    selected_features: list[str] | None = None,
) -> pd.DataFrame:
    view = reviewed
    if selected_pulsars and "pulsar" in view.columns:
        view = view[view["pulsar"].astype(str).isin(selected_pulsars)]
    if selected_variants and "variant" in view.columns:
        view = view[view["variant"].astype(str).isin(selected_variants)]
    if selected_decisions and "reviewed_decision" in view.columns:
        view = view[view["reviewed_decision"].astype(str).isin(selected_decisions)]
    if selected_backends and "backend" in view.columns:
        view = view[view["backend"].astype(str).isin(selected_backends)]
    if selected_features:
        mask = pd.Series(False, index=view.index)
        for feature in selected_features:
            mask |= _feature_mask(view, feature)
        view = view[mask]
    return view


def _attach_plot_columns(
    view: pd.DataFrame,
    residual_col: str,
    x_axis_col: str,
) -> pd.DataFrame:
    out = view.copy()
    out["_plot_x"] = pd.to_numeric(out[x_axis_col], errors="coerce")
    out["_plot_x_column"] = x_axis_col
    out["_plot_residual"] = pd.to_numeric(out[residual_col], errors="coerce")
    out["_plot_residual_column"] = residual_col

    err_col = None
    if residual_col.endswith("_us") and "tempo2_err_us" in out.columns:
        err_col = "tempo2_err_us" * 1e-6
    elif "tempo2_err" in out.columns:
        err_col = "tempo2_err" * 1e-6
    elif "uncertainty" in out.columns:
        err_col = "uncertainty"

    if err_col:
        out["_plot_error"] = pd.to_numeric(out[err_col], errors="coerce")

    return out


def _apply_display_median_centering(
    view: pd.DataFrame,
    *,
    group_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Subtract KEEP-row group medians from plot residuals only."""

    meta = {
        "enabled": False,
        "group_col": group_col,
        "n_groups": 0,
        "n_reference_rows": 0,
    }
    if group_col not in view.columns or "_plot_residual" not in view.columns:
        return view, meta

    out = view.copy()
    decisions = (
        out.get("reviewed_decision", pd.Series("KEEP", index=out.index))
        .fillna("KEEP")
        .astype(str)
        .str.upper()
    )
    reference = (
        decisions.eq("KEEP")
        & pd.to_numeric(out["_plot_residual"], errors="coerce").notna()
        & out[group_col].notna()
    )
    if "manual_action" in out.columns:
        reference &= _normalised_text(out["manual_action"]) == ""
    if "manual_reason" in out.columns:
        reference &= _normalised_text(out["manual_reason"]) == ""

    medians = (
        out.loc[reference].groupby(group_col, dropna=False)["_plot_residual"].median()
    )
    if medians.empty:
        return out, meta

    offsets = out[group_col].map(medians)
    valid_offsets = pd.to_numeric(offsets, errors="coerce").notna()
    out["_plot_residual_raw"] = out["_plot_residual"]
    out.loc[valid_offsets, "_plot_residual"] = pd.to_numeric(
        out.loc[valid_offsets, "_plot_residual"], errors="coerce"
    ) - pd.to_numeric(offsets.loc[valid_offsets], errors="coerce")
    out["_plot_center_offset"] = pd.to_numeric(offsets, errors="coerce")
    meta.update(
        {
            "enabled": True,
            "n_groups": int(len(medians)),
            "n_reference_rows": int(reference.sum()),
        }
    )
    return out, meta


def _plot_subset(
    view: pd.DataFrame,
    *,
    selected_ids: list[str],
    max_keep_points: int,
    plot_all_keep: bool,
) -> pd.DataFrame:
    """Return rows to plot, preserving suspicious/manual/selected rows."""

    plot_df = view.dropna(subset=["_plot_x", "_plot_residual"]).copy()
    if plot_df.empty:
        return plot_df

    selected = plot_df["review_id"].astype(str).isin(selected_ids)
    decision = (
        plot_df.get("reviewed_decision", pd.Series("KEEP", index=plot_df.index))
        .fillna("KEEP")
        .astype(str)
        .str.upper()
    )
    important = decision.isin(IMPORTANT_DECISIONS) | selected

    if "manual_action" in plot_df.columns:
        important |= _normalised_text(plot_df["manual_action"]) != ""
    if "manual_reason" in plot_df.columns:
        important |= _normalised_text(plot_df["manual_reason"]) != ""

    important_df = plot_df.loc[important].copy()
    keep_df = plot_df.loc[~important].copy()

    if not plot_all_keep and len(keep_df) > max_keep_points:
        sample_idx = np.linspace(
            0, len(keep_df) - 1, int(max_keep_points), dtype=np.int64
        )
        keep_df = keep_df.iloc[sample_idx].copy()
        keep_df["_plot_sampled"] = True
    else:
        keep_df["_plot_sampled"] = False
    important_df["_plot_sampled"] = False

    out = pd.concat([important_df, keep_df], ignore_index=True, sort=False)
    out["_selected"] = out["review_id"].astype(str).isin(selected_ids)
    out["_plot_rid_i"] = np.arange(len(out), dtype=np.int32)
    return out


def _bool_col(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[col].fillna(False).astype(bool)


def _color_series(frame: pd.DataFrame, color_col: str) -> pd.Series:
    if color_col not in frame.columns:
        color_col = "reviewed_decision"
    raw = frame[color_col]
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().any() and numeric.nunique(dropna=True) > MAX_COLOR_CATEGORIES:
        try:
            binned = pd.qcut(
                numeric,
                q=min(12, int(numeric.nunique(dropna=True))),
                duplicates="drop",
            )
            series = binned.astype(str).replace({"nan": "(missing)"})
        except Exception:
            series = raw.fillna("(missing)").astype(str).replace({"": "(missing)"})
    else:
        series = raw.fillna("(missing)").astype(str).replace({"": "(missing)"})
    counts = series.value_counts(dropna=False)
    if len(counts) > MAX_COLOR_CATEGORIES:
        keep = set(counts.head(MAX_COLOR_CATEGORIES - 1).index.astype(str))
        series = series.where(series.isin(keep), other="(other)")
    return series


def _decision_symbol_array(frame: pd.DataFrame) -> np.ndarray:
    decision = (
        frame.get("reviewed_decision", pd.Series("KEEP", index=frame.index))
        .fillna("KEEP")
        .astype(str)
        .str.upper()
    )
    return (
        decision.map(DECISION_MARKER_SYMBOLS).fillna(DEFAULT_MARKER_SYMBOL).to_numpy()
    )


def _color_map(values: list[str]) -> dict[str, str]:
    return {
        str(value): PLOTLY_CATEGORY_COLORS[idx % len(PLOTLY_CATEGORY_COLORS)]
        for idx, value in enumerate(values)
    }


def _add_trace(
    fig: Any,
    frame: pd.DataFrame,
    *,
    name: str,
    size: int,
    opacity: float,
    symbol: str | np.ndarray,
    color: str | None,
    showlegend: bool,
    x_axis_col: str,
    residual_col: str,
    rich_hover: bool,
    color_col: str,
    show_error_bars: bool,
) -> None:
    """Add one WebGL scatter trace with a deliberately small payload."""

    if frame.empty:
        return

    custom = frame["_plot_rid_i"].to_numpy(dtype=np.int32, copy=False)
    hovertemplate = f"{x_axis_col}=%{{x:.6g}}<br>{residual_col}=%{{y:.4g}}"

    if rich_hover:
        custom_cols = ["review_id"]
        for optional in (
            "backend",
            "timfile",
            "freq",
            "auto_decision",
            "reviewed_decision",
            "_active_features",
            color_col,
        ):
            if optional in frame.columns and optional not in custom_cols:
                custom_cols.append(optional)

        custom = frame[custom_cols].astype(str).to_numpy()
        hovertemplate += "<br>review_id=%{customdata[0]}"
        for idx, col in enumerate(custom_cols[1:], start=1):
            hovertemplate += f"<br>{col}=%{{customdata[{idx}]}}"
    hovertemplate += "<extra></extra>"

    error_y = None
    if (
        show_error_bars
        and "_plot_error" in frame.columns
        # and frame["_plot_error"].notna().any()
    ):
        error_y = {
            "type": "data",
            "array": pd.to_numeric(frame["_plot_error"], errors="coerce").to_numpy(
                dtype=np.float32, copy=False
            ),
            "visible": True,
            "width": 0,
            "thickness": 0.5,
        }

    fig.add_trace(
        go.Scattergl(
            x=pd.to_numeric(frame["_plot_x"], errors="coerce").to_numpy(
                dtype=np.float32, copy=False
            ),
            y=pd.to_numeric(frame["_plot_residual"], errors="coerce").to_numpy(
                dtype=np.float32, copy=False
            ),
            error_y=error_y,
            mode="markers",
            name=name,
            customdata=custom,
            marker={
                "size": size,
                "opacity": opacity,
                "symbol": symbol,
                **({"color": color} if color is not None else {}),
            },
            hovertemplate=hovertemplate,
            showlegend=showlegend,
        )
    )


def _make_fast_scatter(
    plot_df: pd.DataFrame,
    *,
    title: str,
    x_axis_col: str,
    residual_col: str,
    rich_hover: bool,
    color_col: str,
    show_error_bars: bool,
) -> Any:
    fig = go.Figure()
    if plot_df.empty:
        return fig

    decision = (
        plot_df.get("reviewed_decision", pd.Series("KEEP", index=plot_df.index))
        .fillna("KEEP")
        .astype(str)
    )
    plot_df = plot_df.assign(
        _decision=decision, _color_value=_color_series(plot_df, color_col)
    )

    selected_mask = _bool_col(plot_df, "_selected")

    color_values = sorted(
        plot_df["_color_value"].unique(), key=lambda x: (x != "KEEP", x)
    )
    colors = _color_map([str(value) for value in color_values])
    for value in color_values:
        group = plot_df[(plot_df["_color_value"] == value) & (~selected_mask)]
        if group.empty:
            continue
        value_text = str(value)
        fig.add_trace(
            go.Scattergl(
                x=[None],
                y=[None],
                mode="markers",
                name=value_text,
                marker={
                    "size": 8,
                    "symbol": "circle",
                    "color": colors[value_text],
                },
                hoverinfo="skip",
                showlegend=True,
            )
        )
        decisions = group["_decision"].astype(str).str.upper()
        has_important = decisions.isin(IMPORTANT_DECISIONS).any()
        _add_trace(
            fig,
            group,
            name=value_text,
            size=7 if has_important else 4,
            opacity=0.82 if has_important else 0.42,
            symbol=_decision_symbol_array(group),
            color=colors[value_text],
            showlegend=False,
            x_axis_col=x_axis_col,
            residual_col=residual_col,
            rich_hover=rich_hover,
            color_col=color_col,
            show_error_bars=show_error_bars,
        )

    selected = plot_df[selected_mask]
    _add_trace(
        fig,
        selected,
        name="selected",
        size=11,
        opacity=0.96,
        symbol=_decision_symbol_array(selected),
        color="rgba(30, 30, 30, 0.95)",
        showlegend=False,
        x_axis_col=x_axis_col,
        residual_col=residual_col,
        rich_hover=rich_hover,
        color_col=color_col,
        show_error_bars=show_error_bars,
    )

    for decision_value, symbol in DECISION_MARKER_SYMBOLS.items():
        fig.add_trace(
            go.Scattergl(
                x=[None],
                y=[None],
                mode="markers",
                name=decision_value,
                marker={
                    "size": 8,
                    "symbol": symbol,
                    "color": "rgba(70, 70, 70, 0.85)",
                },
                hoverinfo="skip",
                showlegend=True,
                legend="legend2",
            )
        )

    fig.add_annotation(
        text=title,
        x=0,
        y=1.28,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 16},
    )
    fig.update_layout(
        dragmode=PLOT_DEFAULT_DRAGMODE,
        height=570,
        hovermode="closest",
        uirevision="pleb-qc-review-fast-session-state",
        selectionrevision="pleb-qc-review-selection-state",
        newselection={"line": {"color": "#d62728", "width": 1.5}},
        activeselection={"fillcolor": "rgba(214, 39, 40, 0.12)"},
        legend={
            "title": {"text": color_col},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.08,
            "xanchor": "left",
            "x": 0,
        },
        legend2={
            "title": {"text": "review decision"},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.08,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 45, "r": 20, "t": 135, "b": 40},
        xaxis_title=x_axis_col,
        yaxis_title=residual_col,
    )
    return fig


def _safe_sum_bool(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].fillna(False).astype(bool).sum())


def _select_current_pulsar(manifest: pd.DataFrame) -> str:
    """Sidebar controls for one-pulsar-at-a-time review."""

    pulsars = list(dict.fromkeys(manifest["pulsar"].astype(str)))
    current = str(st.session_state.get("current_pulsar", ""))
    if current not in pulsars:
        current = pulsars[0]
        st.session_state["current_pulsar"] = current

    idx = pulsars.index(current)
    prev_col, next_col = st.columns(2)
    if prev_col.button("← Prev", disabled=idx <= 0):
        current = pulsars[idx - 1]
        st.session_state["current_pulsar"] = current
        st.session_state["current_qc_csv"] = str(
            manifest.loc[manifest["pulsar"].astype(str) == current, "path"].iloc[0]
        )
        _reset_plot_selection()
        st.rerun()
    if next_col.button("Next →", disabled=idx >= len(pulsars) - 1):
        current = pulsars[idx + 1]
        st.session_state["current_pulsar"] = current
        st.session_state["current_qc_csv"] = str(
            manifest.loc[manifest["pulsar"].astype(str) == current, "path"].iloc[0]
        )
        _reset_plot_selection()
        st.rerun()

    chosen_pulsar = st.selectbox("Pulsar", pulsars, index=pulsars.index(current))
    if chosen_pulsar != str(st.session_state.get("current_pulsar", "")):
        st.session_state["current_pulsar"] = chosen_pulsar
        st.session_state["current_qc_csv"] = str(
            manifest.loc[manifest["pulsar"].astype(str) == chosen_pulsar, "path"].iloc[
                0
            ]
        )
        _reset_plot_selection()
    return str(st.session_state["current_pulsar"])


def _variant_availability_once(
    subset: pd.DataFrame,
    *,
    root_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    key = {
        "root_dir": _resolved_path_key(root_dir),
        "paths": tuple(subset["path"].astype(str).tolist()),
    }
    must_load = (
        force
        or st.session_state.get("_loaded_variant_availability_key") != key
        or "variant_availability_df" not in st.session_state
    )
    if must_load:
        rows: list[dict[str, object]] = []
        with st.spinner("Scanning variant residual availability..."):
            for _, row in subset.iterrows():
                path = Path(str(row["path"])).expanduser()
                variant = str(row.get("variant", "base") or "base")
                label = str(row.get("label", path.name) or path.name)
                try:
                    df = load_qc_csv(path, root=root_dir)
                    residual_cols = _residual_column_options(df)
                except Exception as e:
                    residual_cols = []
                    rows.append(
                        {
                            "path": _resolved_path_key(path),
                            "variant": variant,
                            "label": label,
                            "postfit_available": False,
                            "postfit_columns": [],
                            "residual_columns": [],
                            "load_error": str(e),
                        }
                    )
                    continue
                postfit_cols = [c for c in POSTFIT_COLUMNS if c in residual_cols]
                rows.append(
                    {
                        "path": _resolved_path_key(path),
                        "variant": variant,
                        "label": label,
                        "postfit_available": bool(postfit_cols),
                        "postfit_columns": postfit_cols,
                        "residual_columns": residual_cols,
                        "load_error": "",
                    }
                )
        st.session_state["variant_availability_df"] = pd.DataFrame(rows)
        st.session_state["_loaded_variant_availability_key"] = key
    return st.session_state["variant_availability_df"]


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="PLEB QC Review", layout="wide")
    st.title("PLEB QC Expert Review")
    st.caption("Manual overrides are stored separately; raw PQC CSVs are not modified.")

    if go is None:
        st.error(
            "Plotly is not installed. Install with `pip install -e .[gui]` "
            "after applying this patch."
        )
        return

    with st.sidebar:
        st.header("Inputs")
        run_dir = st.text_input("Run directory", value=args.run_dir or "")
        default_overrides, _default_reviewed = _default_paths(run_dir or ".")
        overrides_path = Path(
            st.text_input(
                "Overrides CSV", value=args.overrides or str(default_overrides)
            )
        ).expanduser()
        reviewer = st.text_input("Reviewer", value="")
        reason = st.text_input("Reason", value="manual review")
        action = st.selectbox("Manual action", REVIEW_ACTIONS, index=0)
        reload_clicked = st.button("Reload QC data")

    if not run_dir:
        st.info("Enter a run directory containing one or more `*_qc.csv` files.")
        return

    try:
        manifest = _load_manifest_once(run_dir, force=bool(reload_clicked))
    except Exception as e:
        st.error(f"Failed to scan QC data: {e}")
        return
    if manifest.empty:
        st.error("No raw `*_qc.csv` files found.")
        return

    root_dir = Path(run_dir).expanduser()

    with st.sidebar:
        st.header("Filters")
        current_pulsar = _select_current_pulsar(manifest)

    subset = manifest[manifest["pulsar"].astype(str) == current_pulsar].reset_index(
        drop=True
    )
    if subset.empty:
        st.error(f"No QC CSVs found for pulsar {current_pulsar}.")
        return

    availability = _variant_availability_once(
        subset,
        root_dir=root_dir,
        force=bool(reload_clicked),
    )
    availability_by_path = {
        str(row["path"]): row for row in availability.to_dict("records")
    }
    variant_counts = subset["variant"].fillna("base").astype(str).value_counts()

    current_csv = str(st.session_state.get("current_qc_csv", ""))
    valid_paths = subset["path"].astype(str).tolist()
    if current_csv not in valid_paths:
        preferred = availability[
            availability["postfit_available"].fillna(False).astype(bool)
        ]
        current_csv = (
            str(preferred.iloc[0]["path"])
            if not preferred.empty
            else str(subset.iloc[0]["path"])
        )
        st.session_state["current_qc_csv"] = current_csv

    variant_labels: list[str] = []
    path_by_variant_label: dict[str, str] = {}
    for _, row in subset.iterrows():
        variant = str(row.get("variant", "base") or "base")
        path_str = str(row["path"])
        status = availability_by_path.get(path_str, {})
        tag = (
            "post-fit"
            if bool(status.get("postfit_available", False))
            else "no post-fit"
        )
        if int(variant_counts.get(variant, 0)) > 1:
            base_label = f"{variant} | {Path(path_str).name}"
        else:
            base_label = variant
        label = f"{base_label} [{tag}]"
        variant_labels.append(label)
        path_by_variant_label[label] = path_str

    current_variant_label = next(
        (label for label, path in path_by_variant_label.items() if path == current_csv),
        variant_labels[0],
    )
    with st.sidebar:
        selected_variant_label = st.selectbox(
            "Variant",
            variant_labels,
            index=variant_labels.index(current_variant_label),
            help="Select the QC variant to review and plot.",
        )
    selected_csv = str(path_by_variant_label[selected_variant_label])
    if selected_csv != current_csv:
        st.session_state["current_qc_csv"] = selected_csv
        _reset_plot_selection()
    csv_path = Path(str(st.session_state["current_qc_csv"])).expanduser()

    _sync_reviewed_output_default(
        run_dir or ".",
        csv_path,
        cli_reviewed_out=str(args.reviewed_out or ""),
    )
    with st.sidebar:
        reviewed_out = Path(
            st.text_input(
                "Reviewed QC output",
                key="reviewed_out_path",
                help=(
                    "Default follows the selected raw QC CSV basename, saved under "
                    "`qc_review/`, so a later FixDataset apply step can discover it."
                ),
            )
        ).expanduser()

    try:
        qc_df = _load_qc_once(csv_path, root_dir=root_dir, force=bool(reload_clicked))
        overrides_df = _load_overrides_once(overrides_path, force=bool(reload_clicked))
    except Exception as e:
        st.error(f"Failed to load QC data: {e}")
        return

    reviewed = apply_overrides(qc_df, overrides_df)
    filterable_feature_cols = _filterable_feature_columns(reviewed)
    color_options = _color_column_options(reviewed)
    if not color_options:
        color_options = ["reviewed_decision"]
    residual_options = _residual_column_options(reviewed)
    if not residual_options:
        st.error(
            "No numeric residual-like column found in the selected variant. "
            "Expected one of: " + ", ".join(RESIDUAL_PREFERENCE)
        )
        return
    x_axis_options = _numeric_axis_options(reviewed)
    if not x_axis_options:
        st.error("No numeric x-axis column found in the selected variant.")
        return

    current_path_key = _resolved_path_key(csv_path)
    current_status = availability_by_path.get(current_path_key, {})
    current_variant = str(current_status.get("variant", "base") or "base")
    current_postfit_cols = [c for c in POSTFIT_COLUMNS if c in residual_options]
    other_postfit_variants = sorted(
        {
            str(row.get("variant", "base") or "base")
            for row in availability.to_dict("records")
            if bool(row.get("postfit_available", False))
            and str(row.get("path")) != current_path_key
        }
    )

    with st.sidebar:
        st.header("Residual")
        residual_col = st.selectbox(
            "Residual column to plot",
            residual_options,
            index=0,
            help=(
                "Prefer the post-fit/JUMP-fitted residual column. "
                "The GUI does not subtract backend offsets cosmetically."
            ),
        )
        x_axis_col = st.selectbox(
            "X-axis column",
            x_axis_options,
            index=x_axis_options.index("mjd") if "mjd" in x_axis_options else 0,
            help=(
                "Choose any numeric QC column for the horizontal axis, for example "
                "`mjd`, uncertainty/error, `orbital_phase`, frequency, or other "
                "structure features present in the table."
            ),
        )
        color_col = st.selectbox(
            "Color by",
            color_options,
            index=(
                color_options.index("backend")
                if "backend" in color_options
                else (
                    color_options.index("reviewed_decision")
                    if "reviewed_decision" in color_options
                    else 0
                )
            ),
            help=(
                "Choose a low-cardinality TOA column such as backend/system/telescope "
                "or a PQC feature flag/id column. Very high-cardinality columns are hidden."
            ),
        )

        if current_postfit_cols:
            st.caption(
                "Post-fit residual columns here: " + ", ".join(current_postfit_cols)
            )
        else:
            msg = (
                f"No numeric post-fit residual is available for variant `{current_variant}`. "
                "Available residual columns here: " + ", ".join(residual_options) + "."
            )
            if other_postfit_variants:
                msg += (
                    " Variants with post-fit residuals: "
                    + ", ".join(other_postfit_variants)
                    + "."
                )
            st.warning(msg)

        load_error = str(current_status.get("load_error", "") or "").strip()
        if load_error:
            st.warning("Variant availability scan error: " + load_error)

        if residual_col in {"resid_us", "residual_us", "resid", "residual"}:
            if current_postfit_cols:
                st.warning(
                    "This variant has post-fit residuals available. Choose one of "
                    + ", ".join(current_postfit_cols)
                    + " if that is what you want to inspect."
                )
            else:
                st.caption(
                    "The dropdown hides all-NaN `tempo2_post*` columns by design."
                )

        decisions = sorted(
            str(x) for x in reviewed["reviewed_decision"].dropna().unique()
        )
        selected_decisions = st.multiselect(
            "Reviewed decision",
            decisions,
            default=decisions,
        )

        backend_values = (
            sorted(str(x) for x in reviewed["backend"].dropna().unique())
            if "backend" in reviewed
            else []
        )
        selected_backends = st.multiselect("Backend", backend_values, default=[])
        selected_features = st.multiselect(
            "Require feature flag",
            filterable_feature_cols,
            default=[],
            help=(
                "Filter to rows where at least one selected PQC feature/event flag "
                "is active. ID columns such as `step_id` and `dm_step_id` count as "
                "active when they are non-negative/non-empty."
            ),
        )
        if filterable_feature_cols:
            with st.expander("Feature counts", expanded=False):
                counts = pd.DataFrame(
                    [
                        {
                            "feature": col,
                            "active_rows": int(_feature_mask(reviewed, col).sum()),
                        }
                        for col in filterable_feature_cols
                    ]
                )
                st.dataframe(counts, width="stretch", hide_index=True)

        st.header("Performance")
        plot_all_keep = st.checkbox("Plot all KEEP rows", value=False)
        center_plot = st.checkbox(
            "Center plot by KEEP median",
            value=False,
            help=(
                "Display-only: subtract each group's median residual computed from "
                "KEEP rows after filters. Raw QC values and outputs are unchanged."
            ),
        )
        center_options = [col for col in color_options if col in reviewed.columns]
        if "backend" in reviewed.columns and "backend" not in center_options:
            center_options.insert(0, "backend")
        center_group_col = st.selectbox(
            "Center group",
            center_options or ["reviewed_decision"],
            index=(
                (center_options or ["reviewed_decision"]).index("backend")
                if "backend" in (center_options or ["reviewed_decision"])
                else 0
            ),
            disabled=not center_plot,
            help="Column whose KEEP-row medians define the display-only offsets.",
        )
        show_error_bars = st.checkbox(
            "Show error bars",
            value=True,
            help="Use the same uncertainty scaling as the older Plotly review path.",
        )
        rich_hover = st.checkbox(
            "Rich hover metadata",
            value=False,
            help=(
                "Show per-row backend/timfile/frequency/feature hover details. "
                "Off keeps the Plotly payload small for large QC files."
            ),
        )
        max_keep_points = int(
            st.number_input(
                "Max sampled KEEP rows",
                min_value=100,
                max_value=200000,
                value=DEFAULT_MAX_KEEP_POINTS,
                step=500,
                disabled=plot_all_keep,
            )
        )
        show_visible_table = st.checkbox("Show visible-row preview", value=False)
        table_preview_rows = int(
            st.number_input(
                "Preview rows",
                min_value=50,
                max_value=10000,
                value=DEFAULT_TABLE_PREVIEW_ROWS,
                step=50,
                disabled=not show_visible_table,
            )
        )

    view = _filter_view(
        reviewed,
        selected_pulsars=[],
        selected_variants=[],
        selected_decisions=selected_decisions,
        selected_backends=selected_backends,
        selected_features=selected_features,
    )
    active_feature_cols = _filterable_feature_columns(view)
    if rich_hover and active_feature_cols:
        view = view.copy()
        view["_active_features"] = [
            _active_feature_labels(row, active_feature_cols)
            for _, row in view.iterrows()
        ]
    view = _attach_plot_columns(view, residual_col, x_axis_col)
    center_meta = {
        "enabled": False,
        "group_col": center_group_col,
        "n_groups": 0,
        "n_reference_rows": 0,
    }
    if center_plot:
        view, center_meta = _apply_display_median_centering(
            view,
            group_col=center_group_col,
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Visible TOAs", len(view))
    c2.metric("Reviewed bad", _safe_sum_bool(view, "reviewed_bad_point"))
    c3.metric("Reviewed event", _safe_sum_bool(view, "reviewed_event_member"))
    c4.metric("Manual rows", len(overrides_df))

    selected_ids_state = st.session_state.get("selected_review_ids", [])
    plottable = view.dropna(subset=["_plot_x", "_plot_residual"])
    plot_df = _plot_subset(
        view,
        selected_ids=selected_ids_state,
        max_keep_points=max_keep_points,
        plot_all_keep=plot_all_keep,
    )
    c5.metric("Plotted rows", len(plot_df))

    st.caption(
        f"QC CSV: `{csv_path.name}`. "
        f"X-axis column: `{x_axis_col}`. "
        f"Plotting residual column: `{residual_col}`. "
        f"Coloring by: `{color_col}`. "
        + (
            "Display-only median centering is applied."
            if center_meta["enabled"]
            else "No display-time backend/JUMP centering is applied."
        )
    )
    if center_meta["enabled"]:
        st.caption(
            "Display-only median centering is enabled: "
            f"`{center_meta['group_col']}` medians from "
            f"{center_meta['n_reference_rows']:,} KEEP rows across "
            f"{center_meta['n_groups']:,} group(s) were subtracted from the plot. "
            "Raw QC values, selected-row table, overrides, and exports are unchanged."
        )
    elif center_plot:
        st.warning(
            f"Display centering requested, but no KEEP-row medians were available for `{center_group_col}`."
        )

    if plottable.empty:
        st.warning(
            f"No plottable rows found. Need numeric `{x_axis_col}` and `{residual_col}` values."
        )
    else:
        if len(plot_df) < len(plottable):
            st.caption(
                f"Fast mode is plotting {len(plot_df):,} of "
                f"{len(plottable):,} plottable rows. "
                "All BAD/EVENT/manual/selected rows are retained; KEEP rows may be sampled."
            )

        fig = _make_fast_scatter(
            plot_df,
            title=f"Residual vs {x_axis_col} — click, box-select, or lasso points",
            x_axis_col=x_axis_col,
            residual_col=residual_col,
            rich_hover=rich_hover,
            color_col=color_col,
            show_error_bars=show_error_bars,
        )
        st.session_state["_plot_review_id_lookup"] = (
            plot_df["review_id"].astype(str).tolist()
        )
        plot_event_key = (
            f"{PLOT_KEY}_{int(st.session_state.get('_plot_reset_token', 0))}"
        )
        st.session_state["_plot_event_key"] = plot_event_key
        st.plotly_chart(
            fig,
            key=plot_event_key,
            on_select=_sync_plot_selection,
            selection_mode=PLOT_SELECTION_MODES,
            config={
                "displayModeBar": True,
                "doubleClick": "reset+autosize",
                "modeBarButtonsToAdd": list(PLOT_MODEBAR_BUTTONS_TO_ADD),
            },
            width="stretch",
        )
        st.caption(
            "Box selection is the default. Use the always-visible mode bar to switch "
            "between box, lasso, pan, and zoom."
        )

    clear_sel_col, _spacer = st.columns([1, 5])
    if clear_sel_col.button("Clear selection"):
        _reset_plot_selection()
        st.rerun()

    manual_ids = st.text_area(
        "Selected review IDs",
        value="\n".join(st.session_state.get("selected_review_ids", [])),
        help="You can paste review IDs here manually; one per line.",
        height=90,
    )
    selected_ids = [x.strip() for x in manual_ids.splitlines() if x.strip()]
    selected = selection_frame(reviewed, selected_ids)
    if not selected.empty:
        selected_feature_cols = _filterable_feature_columns(selected)
        if selected_feature_cols:
            selected = selected.copy()
            selected["_active_features"] = [
                _active_feature_labels(row, selected_feature_cols)
                for _, row in selected.iterrows()
            ]
    if not selected.empty and residual_col in selected.columns:
        selected = _attach_plot_columns(selected, residual_col, x_axis_col)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Selected TOAs")
        st.dataframe(
            selected[_compact_columns(selected)] if not selected.empty else selected,
            width="stretch",
            height=220,
        )

    with right:
        st.subheader("Apply action")
        if st.button("Apply to selected", disabled=selected.empty):
            new_rows = make_override_rows(
                selected,
                action=action,
                reason=reason,
                reviewer=reviewer,
                source="streamlit_qc_review",
            )
            st.session_state["overrides_df"] = append_overrides(overrides_df, new_rows)
            write_overrides(st.session_state["overrides_df"], overrides_path)
            _reset_plot_selection()
            st.success(f"Saved {len(new_rows)} override row(s) to {overrides_path}")
            st.rerun()

        if st.button("Write reviewed QC CSV"):
            write_reviewed_qc(qc_df, reviewed_out, overrides=overrides_df)
            st.success(f"Wrote reviewed QC to {reviewed_out}")

    if show_visible_table:
        with st.expander("Visible rows preview", expanded=True):
            shown = view[_compact_columns(view)].head(table_preview_rows)
            st.dataframe(shown, width="stretch", height=360)
            if len(view) > len(shown):
                st.caption(
                    f"Showing first {len(shown):,} of {len(view):,} visible rows."
                )
    else:
        st.caption(
            "Visible-row preview is hidden for speed. Enable it in the sidebar if needed."
        )

    with st.expander("Manual override audit table", expanded=False):
        st.dataframe(overrides_df, width="stretch", height=280)


if __name__ == "__main__":
    main()
