#!/usr/bin/env python3
"""Streamlit fast QC plotting benchmark for large review CSVs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pleb.qc_review import attach_tempo2_general2_residuals

X_CANDIDATES = ("mjd", "freq")
Y_CANDIDATES = (
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
UNCERTAINTY_SOURCE_CANDIDATES = (
    "sigma_us",
    "toa_err_us",
    "err_us",
    "toa_uncertainty_us",
    "sigma",
    "toa_err",
    "error",
    "err",
)
FLAG_CANDIDATES = {
    "BAD": (
        "BAD",
        "bad",
        "is_bad",
        "qc_bad",
        "bad_qc",
        "flag_bad",
        "bad_point",
        "bad_mad",
        "bad_ou",
        "bad_hard",
        "robust_outlier",
        "robust_global_outlier",
        "outlier_any",
    ),
    "EVENT": (
        "EVENT",
        "event",
        "is_event",
        "qc_event",
        "flag_event",
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
        "transient_id",
        "step_id",
        "dm_step_id",
    ),
    "MANUAL": (
        "MANUAL",
        "manual",
        "is_manual",
        "manual_review",
        "flag_manual",
        "manual_action",
        "reviewed_decision",
        "review_decision",
    ),
}
DEFAULT_SYNTHETIC_N = 200_000
DEFAULT_MAX_MAIN_POINTS = 50_000
EXPORT_PATH = Path("qc_review") / "fast_test_selected_ids.csv"


def _fragment(func):
    fragment = getattr(st, "fragment", None)
    if callable(fragment):
        return fragment(func)
    return func


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    exact = {str(col): str(col) for col in columns}
    lower = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        found = lower.get(candidate.lower())
        if found is not None:
            return found
    return None


def _residual_column_options(df: pd.DataFrame) -> list[str]:
    options: list[str] = []
    for col in Y_CANDIDATES:
        actual = _find_column(df.columns, (col,))
        if actual is None or actual in options:
            continue
        converted = pd.to_numeric(df[actual], errors="coerce")
        if converted.notna().any():
            options.append(actual)

    keywords = ("resid", "residual", "post", "postfit", "clean")
    for col in df.columns:
        name = str(col).lower()
        if col in options or not any(keyword in name for keyword in keywords):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            options.append(str(col))
    return options


def _truthy_mask(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).to_numpy() != 0

    values = series.astype("string").str.strip().str.lower().fillna("")
    return values.isin(
        ("1", "true", "t", "yes", "y", "bad", "outlier", "event", "manual")
    ).to_numpy()


def _id_member_mask(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.lower().fillna("")
    return (numeric.fillna(-1).to_numpy() >= 0) | (
        numeric.isna().to_numpy()
        & ~text.isin(("", "nan", "none", "null", "-1")).to_numpy()
    )


def _flag_mask(df: pd.DataFrame, label: str, col: str) -> np.ndarray:
    if label == "EVENT" and col in {"transient_id", "step_id", "dm_step_id"}:
        return _id_member_mask(df[col])
    if label == "MANUAL" and pd.api.types.is_object_dtype(df[col]):
        text = df[col].astype("string").str.strip().str.upper().fillna("")
        return (~text.isin(("", "KEEP", "NONE", "NAN", "NULL"))).to_numpy()
    return _truthy_mask(df[col])


def _plot_error_column(df: pd.DataFrame, residual_col: str) -> str | None:
    if residual_col.endswith("_us") and "tempo2_err_us" in df.columns:
        return "tempo2_err_us"
    if "tempo2_err" in df.columns:
        return "tempo2_err"
    if "uncertainty" in df.columns:
        return "uncertainty"
    return None


def _synthetic_qc_data(n_rows: int, seed: int = 1729) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_rows = max(1, int(n_rows))
    mjd = 58_000.0 + np.sort(rng.random(n_rows, dtype=np.float64) * 3_650.0)
    freq = rng.choice(
        np.array([700.0, 820.0, 1_400.0, 2_300.0], dtype=np.float32), size=n_rows
    )
    trend = 0.25 * np.sin((mjd - mjd.min()) / 37.0)
    noise = rng.normal(0.0, 1.8, size=n_rows)
    sigma = rng.uniform(0.4, 2.2, size=n_rows)
    event = rng.random(n_rows) < 0.01
    bad = rng.random(n_rows) < 0.02
    manual = rng.random(n_rows) < 0.005
    residual = trend + noise
    residual[event] += rng.normal(6.0, 1.3, size=int(event.sum()))
    residual[bad] += rng.normal(-5.0, 1.8, size=int(bad.sum()))

    return pd.DataFrame(
        {
            "review_id": [f"synthetic_{idx}" for idx in range(n_rows)],
            "mjd": mjd,
            "freq": freq,
            "tempo2_post_us": residual,
            "sigma_us": sigma,
            "BAD": bad,
            "EVENT": event,
            "MANUAL": manual,
        }
    )


def _read_or_synthesize(
    csv_path: str, synthetic_n: int
) -> tuple[pd.DataFrame, str, str, Path | None]:
    clean_path = csv_path.strip()
    if not clean_path:
        return (
            _synthetic_qc_data(synthetic_n),
            "synthetic",
            "No CSV path provided.",
            None,
        )

    try:
        path = Path(clean_path).expanduser()
        df = pd.read_csv(path, low_memory=False)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - the UI should fall back for any read failure.
        message = f"CSV read failed: {exc}. Using synthetic data."
        return _synthetic_qc_data(synthetic_n), "synthetic", message, None

    x_col = _find_column(df.columns, X_CANDIDATES)
    residual_options = _residual_column_options(df)
    y_col = residual_options[0] if residual_options else None
    if x_col is None:
        message = "CSV is missing a supported x column. Using synthetic data."
        return _synthetic_qc_data(synthetic_n), "synthetic", message, None

    message = (
        "Loaded CSV."
        if y_col is not None
        else "Loaded CSV; waiting for TEMPO2 general2 residuals."
    )
    return df, str(path), message, path


@st.cache_data(show_spinner=False)
def load_and_prepare(
    csv_path: str, synthetic_n: int, general2_root: str = ""
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    raw_df, source, message, source_path = _read_or_synthesize(csv_path, synthetic_n)

    attached_general2 = False
    if source_path is not None:
        try:
            root = Path(general2_root).expanduser() if general2_root.strip() else None
            if root is not None and root.name == "general2":
                root = root.parent
            before_cols = set(str(col) for col in raw_df.columns)
            raw_df = attach_tempo2_general2_residuals(raw_df, source_path, root=root)
            attached_general2 = any(
                col not in before_cols
                for col in ("tempo2_post_us", "tempo2_postfit_us", "tempo2_pre_us")
                if col in raw_df.columns
            )
            if attached_general2:
                message = f"{message} Attached TEMPO2 general2 residuals."
        except Exception as exc:  # noqa: BLE001 - optional TEMPO2 attachment.
            message = f"{message} TEMPO2 general2 attachment failed: {exc}"

    x_col = _find_column(raw_df.columns, X_CANDIDATES)
    residual_options = _residual_column_options(raw_df)
    y_col = residual_options[0] if residual_options else None
    if x_col is None or y_col is None:
        raw_df = _synthetic_qc_data(synthetic_n)
        source = "synthetic"
        message = "Prepared synthetic data because x/y detection failed."
        x_col = "mjd"
        y_col = "tempo2_post_us"
        residual_options = [y_col]

    df = raw_df.reset_index(drop=True).copy()
    x_values = pd.to_numeric(df[x_col], errors="coerce")
    y_values = pd.to_numeric(df[y_col], errors="coerce")
    valid = x_values.notna() & y_values.notna()
    dropped_rows = int((~valid).sum())
    if dropped_rows:
        df = df.loc[valid].reset_index(drop=True).copy()
        x_values = pd.to_numeric(df[x_col], errors="coerce")
        y_values = pd.to_numeric(df[y_col], errors="coerce")

    if df.empty:
        df = _synthetic_qc_data(synthetic_n)
        source = "synthetic"
        message = "Prepared synthetic data because no finite x/y rows remained."
        x_col = "mjd"
        y_col = "tempo2_post_us"
        x_values = pd.to_numeric(df[x_col], errors="coerce")
        y_values = pd.to_numeric(df[y_col], errors="coerce")
        dropped_rows = 0

    if len(df) > np.iinfo(np.int32).max:
        raise ValueError(
            "This test app requires fewer than 2,147,483,647 rows for int32 row IDs."
        )

    review_col = _find_column(df.columns, ("review_id", "reviewid", "id"))
    if review_col is None:
        review_ids = pd.Series([f"row_{idx}" for idx in range(len(df))], dtype="string")
    else:
        review_ids = df[review_col].astype("string").fillna("")
        missing = review_ids.eq("")
        if bool(missing.any()):
            review_ids = review_ids.mask(
                missing, [f"row_{idx}" for idx in np.flatnonzero(missing)]
            )

    df["_x_f32"] = x_values.to_numpy(dtype=np.float32, copy=False)
    df["_y_f32"] = y_values.to_numpy(dtype=np.float32, copy=False)
    df["_rid_i"] = np.arange(len(df), dtype=np.int32)
    df["_review_id_str"] = review_ids.to_numpy(dtype=str, copy=False)
    uncertainty_source_col = _find_column(df.columns, UNCERTAINTY_SOURCE_CANDIDATES)
    if uncertainty_source_col is not None and "uncertainty" not in df.columns:
        df["uncertainty"] = pd.to_numeric(df[uncertainty_source_col], errors="coerce")

    uncertainty_col = _plot_error_column(df, y_col)
    if uncertainty_col is not None:
        err_values = pd.to_numeric(df[uncertainty_col], errors="coerce")
        df["_err_f32"] = np.abs(err_values.to_numpy(dtype=np.float32, copy=False))

    flag_cols: dict[str, str | None] = {}
    flag_counts: dict[str, int] = {}
    for label, candidates in FLAG_CANDIDATES.items():
        col = _find_column(df.columns, candidates)
        flag_cols[label] = col
        flag_counts[label] = (
            int(_flag_mask(df, label, col).sum()) if col is not None else 0
        )

    meta = {
        "source": source,
        "message": message,
        "x_col": x_col,
        "y_col": y_col,
        "uncertainty_col": uncertainty_col,
        "uncertainty_source_col": uncertainty_source_col,
        "residual_options": residual_options,
        "attached_general2": attached_general2,
        "review_col": review_col,
        "row_count": int(len(df)),
        "dropped_rows": dropped_rows,
        "flag_cols": flag_cols,
        "flag_counts": flag_counts,
        "load_prep_s": time.perf_counter() - started,
    }
    return df, meta


def _values_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        df["_x_f32"].to_numpy(dtype=np.float32, copy=False),
        df["_y_f32"].to_numpy(dtype=np.float32, copy=False),
        df["_rid_i"].to_numpy(dtype=np.int32, copy=False),
    )


def _errors_from_df(df: pd.DataFrame) -> np.ndarray:
    if "_err_f32" not in df.columns:
        return np.array([], dtype=np.float32)
    return df["_err_f32"].to_numpy(dtype=np.float32, copy=False)


def _build_plot_payloads(
    df: pd.DataFrame,
    meta: Mapping[str, Any],
    *,
    plot_all: bool,
    max_main_points: int,
    selected_ids: list[str],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray, float]:
    started = time.perf_counter()
    n_rows = len(df)
    if plot_all or n_rows <= max_main_points:
        main_df = df
    else:
        sample_idx = np.linspace(0, n_rows - 1, max_main_points, dtype=np.int64)
        main_df = df.iloc[sample_idx]

    payloads: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "Main": _values_from_df(main_df)
    }
    main_errors = _errors_from_df(main_df)

    flag_cols = meta.get("flag_cols", {})
    for label in ("BAD", "EVENT", "MANUAL"):
        col = flag_cols.get(label)
        if col is None:
            payloads[label] = (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int32),
            )
            continue
        payloads[label] = _values_from_df(df.loc[_flag_mask(df, label, col)])

    if selected_ids:
        selected_mask = df["_review_id_str"].isin(selected_ids).to_numpy()
        payloads["Selected"] = _values_from_df(df.loc[selected_mask])
    else:
        payloads["Selected"] = (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int32),
        )

    return payloads, main_errors, time.perf_counter() - started


def _add_scattergl(
    fig: go.Figure,
    name: str,
    payload: tuple[np.ndarray, np.ndarray, np.ndarray],
    marker: dict[str, Any],
    error_y: np.ndarray | None = None,
) -> None:
    x_values, y_values, row_ids = payload
    if len(x_values) == 0:
        return
    error_config = None
    if error_y is not None and len(error_y) == len(y_values):
        error_config = {
            "type": "data",
            "array": error_y,
            "visible": True,
            "thickness": 0.6,
            "width": 0,
            "color": "rgba(58, 58, 58, 0.18)",
        }
    fig.add_trace(
        go.Scattergl(
            x=x_values,
            y=y_values,
            customdata=row_ids,
            mode="markers",
            marker=marker,
            name=name,
            error_y=error_config,
            hovertemplate="x=%{x}<br>y=%{y}<extra></extra>",
        )
    )


def build_figure(
    payloads: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    x_label: str,
    y_label: str,
    source_key: str,
    show_error_bars: bool,
    main_errors: np.ndarray,
) -> tuple[go.Figure, float]:
    started = time.perf_counter()
    fig = go.Figure()
    _add_scattergl(
        fig,
        "Main",
        payloads["Main"],
        {
            "size": 4,
            "color": "rgba(58, 111, 166, 0.42)",
        },
        main_errors if show_error_bars else None,
    )
    _add_scattergl(
        fig,
        "BAD",
        payloads["BAD"],
        {
            "size": 7,
            "color": "rgba(214, 67, 67, 0.9)",
        },
    )
    _add_scattergl(
        fig,
        "EVENT",
        payloads["EVENT"],
        {
            "size": 7,
            "color": "rgba(219, 149, 39, 0.9)",
            "symbol": "diamond",
        },
    )
    _add_scattergl(
        fig,
        "MANUAL",
        payloads["MANUAL"],
        {
            "size": 7,
            "color": "rgba(72, 151, 93, 0.9)",
            "symbol": "square",
        },
    )
    _add_scattergl(
        fig,
        "Selected",
        payloads["Selected"],
        {
            "size": 10,
            "color": "rgba(0, 0, 0, 0)",
            "line": {"width": 2, "color": "rgba(26, 26, 26, 1.0)"},
        },
    )
    fig.update_layout(
        height=650,
        margin={"l": 55, "r": 20, "t": 35, "b": 55},
        dragmode="select",
        hovermode="closest",
        uirevision=source_key,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )
    return fig, time.perf_counter() - started


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _selection_box_ranges(selection: Any) -> list[tuple[float, float, float, float]]:
    boxes = _get_value(selection, "box", []) or []
    ranges: list[tuple[float, float, float, float]] = []
    for box in boxes:
        x_range = _get_value(box, "x")
        y_range = _get_value(box, "y")
        if x_range is None or y_range is None:
            continue
        if len(x_range) != 2 or len(y_range) != 2:
            continue
        x0, x1 = sorted(float(value) for value in x_range)
        y0, y1 = sorted(float(value) for value in y_range)
        ranges.append((x0, x1, y0, y1))
    return ranges


def _ids_from_box_selection(
    df: pd.DataFrame,
    boxes: list[tuple[float, float, float, float]],
    review_lookup: list[str],
) -> tuple[list[str], float]:
    started = time.perf_counter()
    if not boxes:
        return [], time.perf_counter() - started

    x_values = df["_x_f32"].to_numpy(dtype=np.float32, copy=False)
    y_values = df["_y_f32"].to_numpy(dtype=np.float32, copy=False)
    exact_mask = np.zeros(len(df), dtype=bool)
    for x0, x1, y0, y1 in boxes:
        exact_mask |= (
            (x_values >= x0) & (x_values <= x1) & (y_values >= y0) & (y_values <= y1)
        )

    row_ids = df.loc[exact_mask, "_rid_i"].to_numpy(dtype=np.int32, copy=False)
    selected = [review_lookup[int(row_id)] for row_id in row_ids]
    return selected, time.perf_counter() - started


def _ids_from_point_selection(
    selection: Any, review_lookup: list[str]
) -> tuple[list[str], float]:
    started = time.perf_counter()
    points = _get_value(selection, "points", []) or []
    selected: list[str] = []
    for point in points:
        customdata = _get_value(point, "customdata")
        if customdata is None:
            continue
        if isinstance(customdata, (list, tuple, np.ndarray)):
            if len(customdata) == 0:
                continue
            customdata = customdata[0]
        row_id = int(customdata)
        if 0 <= row_id < len(review_lookup):
            selected.append(review_lookup[row_id])
    return selected, time.perf_counter() - started


def _merge_selected_ids(existing: list[str], new_ids: list[str]) -> list[str]:
    seen = set(existing)
    merged = list(existing)
    for review_id in new_ids:
        if review_id not in seen:
            merged.append(review_id)
            seen.add(review_id)
    return merged


def _process_selection_event(
    event: Any,
    df: pd.DataFrame,
    review_lookup: list[str],
) -> tuple[list[str], float, str]:
    selection = _get_value(event, "selection", None)
    if not selection:
        return [], 0.0, "none"

    boxes = _selection_box_ranges(selection)
    if boxes:
        selected, elapsed = _ids_from_box_selection(df, boxes, review_lookup)
        return selected, elapsed, "box"

    selected, elapsed = _ids_from_point_selection(selection, review_lookup)
    mode = "points" if selected else "none"
    return selected, elapsed, mode


def _selected_rows(df: pd.DataFrame, selected_ids: list[str]) -> pd.DataFrame:
    if not selected_ids:
        return pd.DataFrame()
    mask = df["_review_id_str"].isin(selected_ids)
    display_cols = [col for col in df.columns if not col.startswith("_")]
    return df.loc[mask, display_cols].head(200)


def _export_selected_ids(selected_ids: list[str]) -> Path:
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"review_id": selected_ids}).to_csv(EXPORT_PATH, index=False)
    return EXPORT_PATH


@_fragment
def render_plot_section(
    df: pd.DataFrame,
    meta: Mapping[str, Any],
    *,
    plot_all: bool,
    max_main_points: int,
    show_error_bars: bool,
    source_key: str,
) -> None:
    review_lookup = st.session_state["review_id_lookup"]
    selected_ids = st.session_state["selected_ids"]

    payloads, main_errors, plot_build_s = _build_plot_payloads(
        df,
        meta,
        plot_all=plot_all,
        max_main_points=max_main_points,
        selected_ids=selected_ids,
    )
    fig, figure_build_s = build_figure(
        payloads,
        x_label=str(meta["x_col"]),
        y_label=str(meta["y_col"]),
        source_key=source_key,
        show_error_bars=show_error_bars,
        main_errors=main_errors,
    )

    event = st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode=("box", "points"),
        key=f"qc_fast_plot_{st.session_state['plot_reset_token']}",
        config={"displaylogo": False, "scrollZoom": True},
    )

    new_ids, selection_mask_s, selection_mode = _process_selection_event(
        event, df, review_lookup
    )
    if new_ids:
        st.session_state["selected_ids"] = _merge_selected_ids(selected_ids, new_ids)
        selected_ids = st.session_state["selected_ids"]

    st.session_state["timings"] = {
        "load_prep_s": float(meta["load_prep_s"]),
        "plot_df_build_s": plot_build_s,
        "figure_build_s": figure_build_s,
        "selection_mask_s": selection_mask_s,
        "selection_mode": selection_mode,
    }

    c0, c1, c2, c3 = st.columns(4)
    timings = st.session_state["timings"]
    c0.metric("load/prep", f"{timings['load_prep_s']:.3f}s")
    c1.metric("plot df build", f"{timings['plot_df_build_s']:.3f}s")
    c2.metric("figure build", f"{timings['figure_build_s']:.3f}s")
    c3.metric("selection mask", f"{timings['selection_mask_s']:.3f}s")

    st.caption(f"Selection mode: {timings['selection_mode']}")
    selected_rows = _selected_rows(df, selected_ids)
    st.subheader(f"Selected IDs: {len(selected_ids)}")
    if selected_rows.empty:
        st.info("No selected rows.")
    else:
        st.dataframe(selected_rows, width="stretch", height=320)


def main() -> None:
    st.set_page_config(page_title="PLEB QC Fast Plot Test", layout="wide")
    st.title("PLEB QC Fast Plot Test")

    with st.sidebar:
        csv_path = st.text_input("Input CSV path", value="")
        general2_root = st.text_input("Optional general2 root", value="")
        synthetic_n = int(
            st.number_input(
                "Synthetic rows",
                min_value=1_000,
                max_value=2_000_000,
                value=DEFAULT_SYNTHETIC_N,
                step=10_000,
            )
        )
        plot_all = st.checkbox("Plot all main points", value=False)
        show_error_bars = st.checkbox(
            "Show main error bars",
            value=False,
            disabled=False,
            help="Off by default for the fast benchmark path.",
        )
        max_main_points = int(
            st.number_input(
                "Max sampled main points",
                min_value=1_000,
                max_value=500_000,
                value=DEFAULT_MAX_MAIN_POINTS,
                step=5_000,
                disabled=plot_all,
            )
        )

        if st.button("Clear selection", width="stretch"):
            st.session_state["selected_ids"] = []
            st.session_state["plot_reset_token"] = (
                st.session_state.get("plot_reset_token", 0) + 1
            )
            st.rerun()

        if st.button("Export selected IDs", width="stretch"):
            exported = _export_selected_ids(st.session_state.get("selected_ids", []))
            st.success(f"Wrote {exported}")

    if "selected_ids" not in st.session_state:
        st.session_state["selected_ids"] = []
    if "plot_reset_token" not in st.session_state:
        st.session_state["plot_reset_token"] = 0
    if "timings" not in st.session_state:
        st.session_state["timings"] = {
            "load_prep_s": 0.0,
            "plot_df_build_s": 0.0,
            "figure_build_s": 0.0,
            "selection_mask_s": 0.0,
            "selection_mode": "none",
        }

    with st.spinner("Loading and preparing data..."):
        df, meta = load_and_prepare(csv_path, synthetic_n, general2_root)

    source_key = "|".join(
        [
            str(meta["source"]),
            str(meta["row_count"]),
            str(meta["x_col"]),
            str(meta["y_col"]),
        ]
    )
    if st.session_state.get("source_key") != source_key:
        st.session_state["source_key"] = source_key
        st.session_state["selected_ids"] = []
        st.session_state["plot_reset_token"] = (
            st.session_state.get("plot_reset_token", 0) + 1
        )

    st.session_state["review_id_lookup"] = df["_review_id_str"].tolist()

    st.caption(
        f"{meta['message']} Source: {meta['source']} | rows: {meta['row_count']:,} | "
        f"x: {meta['x_col']} | y: {meta['y_col']} | dropped x/y rows: {meta['dropped_rows']:,}"
    )
    if meta.get("residual_options"):
        st.caption("Residual candidates: " + ", ".join(meta["residual_options"][:12]))
    st.caption(f"Uncertainty column: `{meta.get('uncertainty_col') or 'none'}`")
    st.caption(
        "Flags: "
        + ", ".join(
            f"{label}={meta['flag_counts'][label]:,}"
            for label in ("BAD", "EVENT", "MANUAL")
        )
    )

    render_plot_section(
        df,
        meta,
        plot_all=plot_all,
        max_main_points=max_main_points,
        show_error_bars=show_error_bars,
        source_key=source_key,
    )


if __name__ == "__main__":
    main()
