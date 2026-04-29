#!/usr/bin/env python3
"""Streamlit expert-review UI for PLEB/PQC outputs.

Run from the repository root, for example:

    python -m streamlit run scripts/pleb_qc_review.py -- --run-dir outputs/my_run

The app writes two artifacts by default:

- manual_qc_overrides.csv: append-only expert decisions
- reviewed_qc.csv: raw QC rows plus reviewed decision columns

This version deliberately avoids st.cache_data for QC loading. Plotly selection
uses ``on_select=\"rerun\"``, so the script reruns frequently. QC data is kept
in ``st.session_state`` and is reloaded only when the run directory / overrides
path changes or when the user clicks reload.

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
    empty_overrides,
    load_overrides,
    load_qc_frames,
    make_override_rows,
    selection_frame,
    write_overrides,
    write_reviewed_qc,
)


DEFAULT_MAX_KEEP_POINTS = 3000
DEFAULT_TABLE_PREVIEW_ROWS = 500
IMPORTANT_DECISIONS = {"BAD_TOA", "REVIEW_EVENT", "EVENT"}

# Prefer fitted/postfit residuals first. Raw-ish residual columns are fallback.
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
    "resid_us",
    "residual_us",
    "clean_resid_us",
    "clean_residual_us",
    "clean_resid",
    "clean_residual",
    "resid",
    "residual",
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


def _resolved_key(run_dir: str, overrides_path: Path) -> dict[str, str]:
    """Build a stable identity key for the currently loaded inputs."""

    try:
        run_dir_key = str(Path(run_dir).expanduser().resolve())
    except Exception:
        run_dir_key = str(Path(run_dir).expanduser())
    try:
        overrides_key = str(overrides_path.expanduser().resolve())
    except Exception:
        overrides_key = str(overrides_path.expanduser())
    return {"run_dir": run_dir_key, "overrides_path": overrides_key}


def _load_inputs_once(
    run_dir: str,
    overrides_path: Path,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load QC and override inputs into session_state.

    Plotly selections rerun the Streamlit script. This function therefore does
    not use filesystem-fingerprint cache keys. It reloads only when the input
    identity changes or when the user explicitly asks for reload.
    """

    key = _resolved_key(run_dir, overrides_path)
    must_load = (
        force
        or st.session_state.get("_loaded_input_key") != key
        or "qc_df" not in st.session_state
        or "overrides_df" not in st.session_state
    )
    if must_load:
        with st.spinner("Loading QC CSVs..."):
            st.session_state["qc_df"] = load_qc_frames(Path(run_dir).expanduser())
            st.session_state["overrides_df"] = load_overrides(overrides_path)
            st.session_state["_loaded_input_key"] = key
            st.session_state["selected_review_ids"] = []
    return st.session_state["qc_df"], st.session_state["overrides_df"]


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
    for point in points or []:
        custom = None
        if isinstance(point, dict):
            custom = point.get("customdata")
        else:
            custom = getattr(point, "customdata", None)
        if isinstance(custom, (list, tuple)) and custom:
            ids.append(str(custom[0]))
        elif custom:
            ids.append(str(custom))

    # Keep order, drop duplicates.
    return list(dict.fromkeys(ids))


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


def _filter_view(
    reviewed: pd.DataFrame,
    *,
    selected_pulsars: list[str],
    selected_variants: list[str],
    selected_decisions: list[str],
    selected_backends: list[str],
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
    return view


def _attach_plot_residual(view: pd.DataFrame, residual_col: str) -> pd.DataFrame:
    out = view.copy()
    out["_plot_residual"] = pd.to_numeric(out[residual_col], errors="coerce")
    out["_plot_residual_column"] = residual_col
    return out


def _plot_subset(
    view: pd.DataFrame,
    *,
    selected_ids: list[str],
    max_keep_points: int,
    plot_all_keep: bool,
) -> pd.DataFrame:
    """Return rows to plot, preserving suspicious/manual/selected rows."""

    plot_df = view.dropna(subset=["mjd", "_plot_residual"]).copy()
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
        keep_df = keep_df.sample(int(max_keep_points), random_state=260408373)
        keep_df["_plot_sampled"] = True
    else:
        keep_df["_plot_sampled"] = False
    important_df["_plot_sampled"] = False

    out = pd.concat([important_df, keep_df], ignore_index=True, sort=False)
    out["_selected"] = out["review_id"].astype(str).isin(selected_ids)
    return out


def _bool_col(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[col].fillna(False).astype(bool)


def _add_trace(
    fig: Any,
    frame: pd.DataFrame,
    *,
    name: str,
    size: int,
    opacity: float,
    residual_col: str,
) -> None:
    """Add one WebGL scatter trace with a deliberately small payload."""

    if frame.empty:
        return

    custom_cols = ["review_id"]
    for optional in ("backend", "timfile", "freq", "auto_decision", "reviewed_decision"):
        if optional in frame.columns:
            custom_cols.append(optional)

    custom = frame[custom_cols].astype(str).to_numpy()

    hovertemplate = (
        "MJD=%{x:.6f}<br>"
        f"{residual_col}=%{{y:.4g}}<br>"
        "review_id=%{customdata[0]}"
    )
    for idx, col in enumerate(custom_cols[1:], start=1):
        hovertemplate += f"<br>{col}=%{{customdata[{idx}]}}"
    hovertemplate += "<extra></extra>"

    fig.add_trace(
        go.Scattergl(
            x=pd.to_numeric(frame["mjd"], errors="coerce"),
            y=pd.to_numeric(frame["_plot_residual"], errors="coerce"),
            mode="markers",
            name=name,
            customdata=custom,
            marker={"size": size, "opacity": opacity},
            hovertemplate=hovertemplate,
        )
    )


def _make_fast_scatter(
    plot_df: pd.DataFrame,
    *,
    title: str,
    residual_col: str,
    enable_bulk_select: bool,
) -> Any:
    fig = go.Figure()
    if plot_df.empty:
        return fig

    decision = (
        plot_df.get("reviewed_decision", pd.Series("KEEP", index=plot_df.index))
        .fillna("KEEP")
        .astype(str)
    )
    plot_df = plot_df.assign(_decision=decision)

    selected_mask = _bool_col(plot_df, "_selected")

    # Put KEEP first so flagged/selected points remain visible on top.
    for dec in sorted(plot_df["_decision"].unique(), key=lambda x: (x != "KEEP", x)):
        group = plot_df[(plot_df["_decision"] == dec) & (~selected_mask)]
        size = 4 if str(dec).upper() == "KEEP" else 7
        opacity = 0.42 if str(dec).upper() == "KEEP" else 0.82
        _add_trace(
            fig,
            group,
            name=str(dec),
            size=size,
            opacity=opacity,
            residual_col=residual_col,
        )

    selected = plot_df[selected_mask]
    _add_trace(
        fig,
        selected,
        name="selected",
        size=11,
        opacity=0.96,
        residual_col=residual_col,
    )

    fig.update_layout(
        title=title,
        dragmode="lasso" if enable_bulk_select else "select",
        height=540,
        hovermode="closest",
        uirevision="pleb-qc-review-fast-session-state",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        margin={"l": 45, "r": 20, "t": 75, "b": 40},
        xaxis_title="MJD",
        yaxis_title=residual_col,
    )
    return fig


def _safe_sum_bool(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].fillna(False).astype(bool).sum())


def _pulsar_selection(reviewed: pd.DataFrame) -> list[str]:
    """Sidebar controls for one-pulsar-at-a-time review."""

    pulsars = (
        sorted(str(x) for x in reviewed["pulsar"].dropna().unique())
        if "pulsar" in reviewed
        else []
    )
    show_all_pulsars = st.checkbox("Show all pulsars", value=False)
    if show_all_pulsars or not pulsars:
        return pulsars

    current = st.session_state.get("current_pulsar")
    if current not in pulsars:
        current = pulsars[0]
    idx = pulsars.index(current)

    prev_col, next_col = st.columns(2)
    if prev_col.button("← Prev", disabled=idx <= 0):
        current = pulsars[idx - 1]
        st.session_state["current_pulsar"] = current
        st.rerun()
    if next_col.button("Next →", disabled=idx >= len(pulsars) - 1):
        current = pulsars[idx + 1]
        st.session_state["current_pulsar"] = current
        st.rerun()

    current = st.selectbox("Pulsar", pulsars, index=pulsars.index(current))
    st.session_state["current_pulsar"] = current
    return [current]


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
        default_overrides, default_reviewed = _default_paths(run_dir or ".")
        overrides_path = Path(
            st.text_input("Overrides CSV", value=args.overrides or str(default_overrides))
        ).expanduser()
        reviewed_out = Path(
            st.text_input("Reviewed QC output", value=args.reviewed_out or str(default_reviewed))
        ).expanduser()
        reviewer = st.text_input("Reviewer", value="")
        reason = st.text_input("Reason", value="manual review")
        action = st.selectbox("Manual action", REVIEW_ACTIONS, index=0)
        reload_clicked = st.button("Reload QC data")

    if not run_dir:
        st.info("Enter a run directory containing one or more `*_qc.csv` files.")
        return

    try:
        qc_df, overrides_df = _load_inputs_once(
            run_dir,
            overrides_path,
            force=bool(reload_clicked),
        )
    except Exception as e:
        st.error(f"Failed to load QC data: {e}")
        return

    reviewed = apply_overrides(qc_df, overrides_df)

    residual_options = _residual_column_options(reviewed)
    if not residual_options:
        st.error(
            "No numeric residual-like column found. Expected one of: "
            + ", ".join(RESIDUAL_PREFERENCE)
        )
        return

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

        if residual_col in {"resid_us", "residual_us", "resid", "residual"}:
            st.warning(
                "You are plotting a raw-ish residual column. If the QC CSV also "
                "contains a post/postfit residual, choose that instead."
            )

        st.header("Filters")
        selected_pulsars = _pulsar_selection(reviewed)

        variants = (
            sorted(str(x) for x in reviewed["variant"].dropna().unique())
            if "variant" in reviewed
            else []
        )
        selected_variants = st.multiselect("Variant", variants, default=variants)

        decisions = sorted(str(x) for x in reviewed["reviewed_decision"].dropna().unique())
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

        st.header("Performance")
        plot_all_keep = st.checkbox("Plot all KEEP rows", value=False)
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
        enable_bulk_select = st.checkbox("Enable box/lasso selection", value=False)
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
        selected_pulsars=selected_pulsars,
        selected_variants=selected_variants,
        selected_decisions=selected_decisions,
        selected_backends=selected_backends,
    )
    view = _attach_plot_residual(view, residual_col)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Visible TOAs", len(view))
    c2.metric("Reviewed bad", _safe_sum_bool(view, "reviewed_bad_point"))
    c3.metric("Reviewed event", _safe_sum_bool(view, "reviewed_event_member"))
    c4.metric("Manual rows", len(overrides_df))

    selected_ids_state = st.session_state.get("selected_review_ids", [])
    plottable = view.dropna(subset=["mjd", "_plot_residual"])
    plot_df = _plot_subset(
        view,
        selected_ids=selected_ids_state,
        max_keep_points=max_keep_points,
        plot_all_keep=plot_all_keep,
    )
    c5.metric("Plotted rows", len(plot_df))

    st.caption(
        f"Plotting residual column: `{residual_col}`. "
        "No display-time backend/JUMP centering is applied."
    )

    if plottable.empty:
        st.warning("No plottable rows found. Need numeric MJD and residual columns.")
    else:
        if len(plot_df) < len(plottable):
            st.caption(
                f"Fast mode is plotting {len(plot_df):,} of "
                f"{len(plottable):,} plottable rows. "
                "All BAD/EVENT/manual/selected rows are retained; KEEP rows may be sampled."
            )

        fig = _make_fast_scatter(
            plot_df,
            title="Residual vs MJD — select points, then apply a manual action",
            residual_col=residual_col,
            enable_bulk_select=enable_bulk_select,
        )
        event = st.plotly_chart(
            fig,
            key="qc_review_residual_plot_fast",
            on_select="rerun",
            selection_mode=("points", "box", "lasso")
            if enable_bulk_select
            else ("points",),
            use_container_width=True,
        )
        selected_ids = _extract_selected_review_ids(event)
        if event is not None:
            st.session_state["selected_review_ids"] = selected_ids

    clear_sel_col, _spacer = st.columns([1, 5])
    if clear_sel_col.button("Clear selection"):
        st.session_state["selected_review_ids"] = []
        st.rerun()

    manual_ids = st.text_area(
        "Selected review IDs",
        value="\n".join(st.session_state.get("selected_review_ids", [])),
        help="You can paste review IDs here manually; one per line.",
        height=90,
    )
    selected_ids = [x.strip() for x in manual_ids.splitlines() if x.strip()]
    selected = selection_frame(reviewed, selected_ids)
    if not selected.empty and residual_col in selected.columns:
        selected = _attach_plot_residual(selected, residual_col)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Selected TOAs")
        st.dataframe(
            selected[_compact_columns(selected)] if not selected.empty else selected,
            use_container_width=True,
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
            st.session_state["selected_review_ids"] = []
            write_overrides(st.session_state["overrides_df"], overrides_path)
            st.success(f"Saved {len(new_rows)} override row(s) to {overrides_path}")
            st.rerun()

        if st.button("Write reviewed QC CSV"):
            write_reviewed_qc(qc_df, reviewed_out, overrides=overrides_df)
            st.success(f"Wrote reviewed QC to {reviewed_out}")

    if show_visible_table:
        with st.expander("Visible rows preview", expanded=True):
            shown = view[_compact_columns(view)].head(table_preview_rows)
            st.dataframe(shown, use_container_width=True, height=360)
            if len(view) > len(shown):
                st.caption(f"Showing first {len(shown):,} of {len(view):,} visible rows.")
    else:
        st.caption(
            "Visible-row preview is hidden for speed. Enable it in the sidebar if needed."
        )

    with st.expander("Manual override audit table", expanded=False):
        st.dataframe(overrides_df, use_container_width=True, height=280)


if __name__ == "__main__":
    main()
