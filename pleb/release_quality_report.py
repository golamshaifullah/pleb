"""Generate release-facing quality reports from PLEB QC artifacts.

The compact QC report is optimized for operator triage.  This module builds a
higher-level release scorecard that lets a reader quickly assess whether final
TOA products look ready, marginal, or blocked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from .qc_report import _build_compact_decisions, _find_qc_csvs

_A4_FIGSIZE = (8.27, 11.69)
_DEFAULT_OUTLIER_COLS = [
    "outlier_any",
    "bad_point",
    "bad_hard",
    "robust_outlier",
    "robust_global_outlier",
    "bad_mad",
    "bad_ou",
    "bad",
    "bad_day",
]
_DECISION_ORDER = ["KEEP", "EVENT", "REVIEW_EVENT", "BAD_TOA"]
_STATUS_COLORS = {
    "GREEN": "#2ca02c",
    "YELLOW": "#ffbf00",
    "RED": "#d62728",
    "NO_QC": "#7f7f7f",
}
_DECISION_COLORS = {
    "KEEP": "#2ca02c",
    "EVENT": "#1f77b4",
    "REVIEW_EVENT": "#ff7f0e",
    "BAD_TOA": "#d62728",
}


@dataclass(slots=True)
class ReleaseQualityThresholds:
    """Thresholds used to grade release-quality scorecard rows.

    Fractions are computed per pulsar/variant as counts divided by total TOAs in
    that QC table.  ``BAD_TOA`` and ``REVIEW_EVENT`` are intentionally separated
    because an event-like outlier is often not an automatic deletion candidate,
    but it is still a release-review risk.
    """

    yellow_bad_fraction: float = 0.01
    red_bad_fraction: float = 0.05
    yellow_review_fraction: float = 0.005
    red_review_fraction: float = 0.02
    yellow_event_fraction: float = 0.10


@dataclass(slots=True)
class ReleaseQualityReportResult:
    """Paths written by :func:`generate_release_quality_report`."""

    report_dir: Path
    pdf_path: Path
    scorecard_path: Path
    backend_risks_path: Path
    flagged_toas_path: Path
    summary_json_path: Path


def _coerce_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0) != 0
    return s.fillna(False).astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})


def _load_qc_summary(run_dir: Path) -> dict[str, dict[str, Any]]:
    summary_path = run_dir / "qc" / "qc_summary.tsv"
    if not summary_path.exists():
        return {}
    try:
        df = pd.read_csv(summary_path, sep="\t")
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    if "qc_csv" not in df.columns:
        return out
    for _, row in df.iterrows():
        raw = str(row.get("qc_csv", "")).strip()
        if not raw:
            continue
        candidates = [Path(raw)]
        if not Path(raw).is_absolute():
            candidates.append((run_dir / raw).resolve())
        for path in candidates:
            out[str(path.resolve())] = row.to_dict()
    return out


def _infer_pulsar_variant_from_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if stem.endswith("_qc"):
        stem = stem[:-3]
    if "." in stem:
        pulsar, variant = stem.split(".", 1)
        return pulsar, variant or "base"
    return stem, "base"


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    present = set(columns)
    for candidate in candidates:
        if candidate in present:
            return candidate
    return None


def _resolve_backend_col(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    if requested and requested in df.columns:
        return requested
    return _first_present(
        list(df.columns),
        [
            "group",
            "sys",
            "-sys",
            "backend",
            "backend_key",
            "system",
            "tel_backend",
            "_timfile_base",
            "_timfile",
        ],
    )


def _read_qc_frames(
    run_dir: Path,
    *,
    backend_col: Optional[str],
    outlier_cols: Optional[list[str]],
) -> pd.DataFrame:
    csvs = _find_qc_csvs(run_dir)
    summary = _load_qc_summary(run_dir)
    frames: list[pd.DataFrame] = []
    cols = outlier_cols if outlier_cols else _DEFAULT_OUTLIER_COLS
    for csv_path in csvs:
        try:
            raw = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue
        if raw.empty:
            continue
        d = _build_compact_decisions(raw, outlier_cols=cols)
        meta = summary.get(str(csv_path.resolve()), {})
        inferred_pulsar, inferred_variant = _infer_pulsar_variant_from_name(csv_path)
        pulsar = str(
            meta.get("pulsar")
            or raw.get("pulsar", pd.Series([inferred_pulsar])).iloc[0]
        )
        variant = str(
            meta.get("variant")
            or raw.get("variant", pd.Series([inferred_variant])).iloc[0]
        )
        branch = str(meta.get("branch") or raw.get("branch", pd.Series([""])).iloc[0])
        chosen_backend_col = _resolve_backend_col(d, backend_col)
        if chosen_backend_col:
            backend = d[chosen_backend_col].fillna("unknown").astype(str)
        else:
            backend = pd.Series(["unknown"] * len(d), index=d.index)
            chosen_backend_col = ""
        d["__pulsar"] = pulsar
        d["__variant"] = variant if variant else "base"
        d["__branch"] = branch
        d["__qc_csv"] = str(csv_path)
        d["__backend_col"] = chosen_backend_col
        d["__backend"] = backend
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _fraction(n: int, total: int) -> float:
    return 0.0 if total <= 0 else float(n) / float(total)


def _grade_row(row: pd.Series, thresholds: ReleaseQualityThresholds) -> str:
    n_toas = int(row.get("n_toas", 0) or 0)
    if n_toas <= 0:
        return "RED"
    bad_fraction = float(row.get("bad_fraction", 0.0) or 0.0)
    review_fraction = float(row.get("review_fraction", 0.0) or 0.0)
    event_fraction = float(row.get("event_fraction", 0.0) or 0.0)
    if (
        bad_fraction >= thresholds.red_bad_fraction
        or review_fraction >= thresholds.red_review_fraction
    ):
        return "RED"
    if (
        bad_fraction >= thresholds.yellow_bad_fraction
        or review_fraction >= thresholds.yellow_review_fraction
        or event_fraction >= thresholds.yellow_event_fraction
    ):
        return "YELLOW"
    return "GREEN"


def _build_scorecard(
    all_qc: pd.DataFrame, thresholds: ReleaseQualityThresholds
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if all_qc.empty:
        return pd.DataFrame()
    group_cols = ["__pulsar", "__variant", "__branch", "__qc_csv"]
    for keys, g in all_qc.groupby(group_cols, dropna=False):
        pulsar, variant, branch, qc_csv = keys
        n_toas = int(len(g))
        counts = g["decision"].value_counts().to_dict()
        bad_toa = int(counts.get("BAD_TOA", 0))
        review = int(counts.get("REVIEW_EVENT", 0))
        event = int(counts.get("EVENT", 0))
        keep = int(counts.get("KEEP", 0))
        residual_col = _first_present(
            list(g.columns),
            [
                "residual_us",
                "resid_us",
                "postfit_residual_us",
                "prefit_residual_us",
                "residual",
                "resid",
            ],
        )
        rms_us = ""
        if residual_col:
            vals = pd.to_numeric(g[residual_col], errors="coerce").dropna()
            if not vals.empty:
                rms_us = float((vals.pow(2).mean()) ** 0.5)
        row = {
            "pulsar": pulsar,
            "variant": variant,
            "branch": branch,
            "n_toas": n_toas,
            "keep": keep,
            "event": event,
            "review_event": review,
            "bad_toa": bad_toa,
            "bad_plus_review": bad_toa + review,
            "bad_fraction": _fraction(bad_toa, n_toas),
            "review_fraction": _fraction(review, n_toas),
            "event_fraction": _fraction(event, n_toas),
            "bad_plus_review_fraction": _fraction(bad_toa + review, n_toas),
            "residual_rms_us": rms_us,
            "qc_csv": qc_csv,
        }
        row["grade"] = _grade_row(pd.Series(row), thresholds)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["__grade_rank"] = (
        out["grade"].map({"RED": 0, "YELLOW": 1, "GREEN": 2}).fillna(3)
    )
    out = out.sort_values(
        ["__grade_rank", "bad_plus_review_fraction", "pulsar", "variant"],
        ascending=[True, False, True, True],
    )
    return out.drop(columns=["__grade_rank"])


def _build_backend_risks(all_qc: pd.DataFrame) -> pd.DataFrame:
    if all_qc.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, g in all_qc.groupby(["__backend", "__backend_col"], dropna=False):
        backend, backend_col = keys
        n_toas = int(len(g))
        decision_counts = g["decision"].value_counts().to_dict()
        bad_toa = int(decision_counts.get("BAD_TOA", 0))
        review = int(decision_counts.get("REVIEW_EVENT", 0))
        event = int(decision_counts.get("EVENT", 0))
        pulsars = sorted(str(x) for x in g["__pulsar"].dropna().astype(str).unique())
        reason_counts = (
            g.loc[g["decision"].isin(["BAD_TOA", "REVIEW_EVENT"]), "decision_reason"]
            .fillna("n/a")
            .astype(str)
            .value_counts()
        )
        rows.append(
            {
                "backend": backend,
                "backend_col": backend_col,
                "n_toas": n_toas,
                "bad_toa": bad_toa,
                "review_event": review,
                "event": event,
                "bad_plus_review": bad_toa + review,
                "bad_plus_review_fraction": _fraction(bad_toa + review, n_toas),
                "n_pulsars": len(pulsars),
                "pulsars": ",".join(pulsars[:12])
                + ("..." if len(pulsars) > 12 else ""),
                "top_reasons": "; ".join(
                    f"{idx}={val}" for idx, val in reason_counts.head(5).items()
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["bad_plus_review_fraction", "bad_plus_review", "n_toas"], ascending=False
    )


def _build_flagged_toas(all_qc: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if all_qc.empty:
        return pd.DataFrame()
    keep_cols = [
        "__pulsar",
        "__variant",
        "__branch",
        "__backend",
        "mjd",
        "freq",
        "toaerr",
        "decision",
        "decision_reason",
        "__qc_csv",
    ]
    available = [c for c in keep_cols if c in all_qc.columns]
    flagged = all_qc.loc[
        all_qc["decision"].isin(["BAD_TOA", "REVIEW_EVENT"]), available
    ].copy()
    if flagged.empty:
        return flagged
    rename = {
        "__pulsar": "pulsar",
        "__variant": "variant",
        "__branch": "branch",
        "__backend": "backend",
        "__qc_csv": "qc_csv",
    }
    flagged = flagged.rename(columns=rename)
    if "decision" in flagged.columns:
        flagged["decision_rank"] = flagged["decision"].map(
            {"BAD_TOA": 0, "REVIEW_EVENT": 1}
        )
        sort_cols = ["decision_rank"]
        if "pulsar" in flagged.columns:
            sort_cols.append("pulsar")
        if "mjd" in flagged.columns:
            sort_cols.append("mjd")
        flagged = flagged.sort_values(sort_cols).drop(columns=["decision_rank"])
    return flagged.head(max(0, int(top_n)))


def _overall_grade(scorecard: pd.DataFrame) -> str:
    if scorecard.empty:
        return "NO_QC"
    grades = set(scorecard["grade"].astype(str))
    if "RED" in grades:
        return "RED"
    if "YELLOW" in grades:
        return "YELLOW"
    return "GREEN"


def _format_fraction(x: Any) -> str:
    try:
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return "n/a"


def _draw_cover_page(
    pdf: Any,
    *,
    title: str,
    run_dir: Path,
    scorecard: pd.DataFrame,
    all_qc: pd.DataFrame,
    thresholds: ReleaseQualityThresholds,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=_A4_FIGSIZE)
    ax = fig.add_subplot(111)
    ax.axis("off")
    overall = _overall_grade(scorecard)
    color = _STATUS_COLORS.get(overall, _STATUS_COLORS["NO_QC"])
    ax.text(0.03, 0.96, title, fontsize=18, fontweight="bold", va="top")
    ax.text(
        0.03, 0.925, "Release-quality data product scorecard", fontsize=10, va="top"
    )
    ax.add_patch(plt.Rectangle((0.03, 0.80), 0.31, 0.095, color=color, alpha=0.95))
    ax.text(
        0.185,
        0.846,
        overall,
        color="white",
        fontsize=24,
        fontweight="bold",
        ha="center",
        va="center",
    )

    if scorecard.empty:
        metrics = [
            ("QC tables", "0"),
            ("Pulsars", "0"),
            ("TOAs", "0"),
            ("Flagged", "0"),
        ]
    else:
        n_toas = int(scorecard["n_toas"].sum())
        n_pulsars = int(scorecard["pulsar"].nunique())
        n_tables = int(len(scorecard))
        n_flagged = int(scorecard["bad_plus_review"].sum())
        metrics = [
            ("QC tables", f"{n_tables}"),
            ("Pulsars", f"{n_pulsars}"),
            ("TOAs", f"{n_toas}"),
            ("Bad/review", f"{n_flagged}"),
        ]
    x0 = 0.39
    for i, (label, value) in enumerate(metrics):
        x = x0 + (i % 2) * 0.29
        y = 0.85 - (i // 2) * 0.075
        ax.text(x, y, value, fontsize=18, fontweight="bold", va="top")
        ax.text(x, y - 0.028, label, fontsize=8.5, va="top")

    ax.text(
        0.03,
        0.74,
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        fontsize=8.5,
        family="monospace",
    )
    ax.text(0.03, 0.715, f"Run directory: {run_dir}", fontsize=8.5, family="monospace")
    ax.text(
        0.03,
        0.685,
        "Thresholds: "
        f"yellow_bad={_format_fraction(thresholds.yellow_bad_fraction)}, "
        f"red_bad={_format_fraction(thresholds.red_bad_fraction)}, "
        f"yellow_review={_format_fraction(thresholds.yellow_review_fraction)}, "
        f"red_review={_format_fraction(thresholds.red_review_fraction)}",
        fontsize=8.2,
        family="monospace",
    )

    if not scorecard.empty:
        grade_counts = (
            scorecard["grade"]
            .value_counts()
            .reindex(["GREEN", "YELLOW", "RED"])
            .fillna(0)
        )
        ax.text(0.03, 0.62, "Pulsar/variant grades", fontsize=12, fontweight="bold")
        bx = fig.add_axes([0.10, 0.48, 0.80, 0.11])
        bx.bar(
            grade_counts.index.tolist(),
            grade_counts.values.tolist(),
            color=[_STATUS_COLORS[k] for k in grade_counts.index],
        )
        bx.set_ylabel("count")
        bx.set_ylim(0, max(1, float(grade_counts.max())) * 1.25)

    if not all_qc.empty:
        decision_counts = (
            all_qc["decision"].value_counts().reindex(_DECISION_ORDER).fillna(0)
        )
        ax.text(0.03, 0.42, "TOA decisions", fontsize=12, fontweight="bold")
        dx = fig.add_axes([0.10, 0.27, 0.80, 0.11])
        dx.bar(
            decision_counts.index.tolist(),
            decision_counts.values.tolist(),
            color=[_DECISION_COLORS[k] for k in decision_counts.index],
        )
        dx.set_ylabel("TOAs")
        dx.tick_params(axis="x", labelrotation=15)
        dx.set_ylim(0, max(1, float(decision_counts.max())) * 1.25)

    if not scorecard.empty:
        risk = scorecard.sort_values("bad_plus_review_fraction", ascending=False).head(
            5
        )
        lines = []
        for _, row in risk.iterrows():
            lines.append(
                f"{row['pulsar']} {row['variant']}: grade={row['grade']}, "
                f"bad+review={int(row['bad_plus_review'])}/{int(row['n_toas'])} "
                f"({_format_fraction(row['bad_plus_review_fraction'])})"
            )
        ax.text(
            0.03,
            0.20,
            "Highest-risk pulsar/variant rows",
            fontsize=12,
            fontweight="bold",
        )
        for idx, line in enumerate(lines or ["No high-risk rows found."]):
            ax.text(0.05, 0.17 - idx * 0.026, line, fontsize=8.5, family="monospace")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _draw_table_page(
    pdf: Any,
    title: str,
    df: pd.DataFrame,
    *,
    rows_per_page: int = 24,
    font_size: float = 7.0,
) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        fig = plt.figure(figsize=_A4_FIGSIZE)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title(title, fontsize=16, loc="left")
        ax.text(0.03, 0.92, "No rows available.", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    for start in range(0, len(df), rows_per_page):
        chunk = df.iloc[start : start + rows_per_page].copy()
        for col in chunk.columns:
            if pd.api.types.is_float_dtype(chunk[col]):
                chunk[col] = chunk[col].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
        fig = plt.figure(figsize=_A4_FIGSIZE)
        ax = fig.add_subplot(111)
        ax.axis("off")
        suffix = (
            ""
            if len(df) <= rows_per_page
            else f" ({start + 1}-{start + len(chunk)} of {len(df)})"
        )
        ax.set_title(f"{title}{suffix}", fontsize=16, loc="left")
        table = ax.table(
            cellText=chunk.astype(str).values.tolist(),
            colLabels=[str(c) for c in chunk.columns],
            bbox=[0.0, 0.0, 1.0, 0.92],
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        if "grade" in chunk.columns:
            grade_idx = list(chunk.columns).index("grade")
            for r, grade in enumerate(chunk["grade"].astype(str), start=1):
                cell = table[(r, grade_idx)]
                cell.set_facecolor(_STATUS_COLORS.get(grade, "#ffffff"))
                cell.set_text_props(
                    color="white" if grade in {"GREEN", "RED"} else "black",
                    weight="bold",
                )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _draw_per_pulsar_pages(
    pdf: Any,
    all_qc: pd.DataFrame,
    *,
    max_pages: int,
) -> None:
    import matplotlib.pyplot as plt

    if all_qc.empty or "mjd" not in all_qc.columns:
        return
    residual_col = _first_present(
        list(all_qc.columns),
        [
            "residual_us",
            "resid_us",
            "postfit_residual_us",
            "prefit_residual_us",
            "residual",
            "resid",
        ],
    )
    if residual_col is None:
        return
    score = (
        all_qc.assign(
            __flag=all_qc["decision"].isin(["BAD_TOA", "REVIEW_EVENT"]).astype(int)
        )
        .groupby(["__pulsar", "__variant"], dropna=False)["__flag"]
        .sum()
        .sort_values(ascending=False)
    )
    for (pulsar, variant), _ in score.head(max_pages).items():
        g = all_qc[
            (all_qc["__pulsar"] == pulsar) & (all_qc["__variant"] == variant)
        ].copy()
        if g.empty:
            continue
        x = pd.to_numeric(g["mjd"], errors="coerce")
        y = pd.to_numeric(g[residual_col], errors="coerce")
        valid = x.notna() & y.notna()
        if not bool(valid.any()):
            continue
        fig = plt.figure(figsize=_A4_FIGSIZE)
        ax = fig.add_subplot(111)
        for decision in _DECISION_ORDER:
            mask = valid & (g["decision"].astype(str) == decision)
            if bool(mask.any()):
                ax.scatter(
                    x.loc[mask],
                    y.loc[mask],
                    s=14 if decision == "KEEP" else 26,
                    label=decision,
                    alpha=0.85,
                    color=_DECISION_COLORS[decision],
                )
        ax.axhline(0.0, linewidth=0.8, color="#555555", alpha=0.6)
        ax.set_title(
            f"{pulsar} / {variant}: residual decisions", loc="left", fontsize=14
        )
        ax.set_xlabel("MJD")
        ax.set_ylabel(residual_col)
        ax.legend(loc="best", fontsize=8)
        text = (
            f"TOAs={len(g)}  BAD_TOA={(g['decision'] == 'BAD_TOA').sum()}  "
            f"REVIEW_EVENT={(g['decision'] == 'REVIEW_EVENT').sum()}  "
            f"EVENT={(g['decision'] == 'EVENT').sum()}"
        )
        ax.text(
            0.01, 0.01, text, transform=ax.transAxes, fontsize=8, family="monospace"
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def generate_release_quality_report(
    run_dir: Path,
    *,
    report_dir: Optional[Path] = None,
    output_name: str = "release_quality_report.pdf",
    title: Optional[str] = None,
    backend_col: Optional[str] = None,
    outlier_cols: Optional[list[str]] = None,
    thresholds: Optional[ReleaseQualityThresholds] = None,
    include_per_pulsar_pages: bool = True,
    per_pulsar_page_limit: int = 30,
    top_n: int = 50,
) -> Optional[ReleaseQualityReportResult]:
    """Write a release-quality PDF and companion machine-readable tables.

    Parameters
    ----------
    run_dir
        Existing PLEB run directory containing ``qc`` outputs.
    report_dir
        Destination directory. Defaults to ``run_dir / 'release_quality_report'``.
    output_name
        PDF filename written inside ``report_dir``.
    title
        Human-readable report title.
    backend_col
        Preferred backend grouping column. If absent in a QC CSV, common
        fallbacks such as ``group``, ``sys`` and ``_timfile`` are tried.
    outlier_cols
        Optional outlier columns to feed into the compact decision policy.
    thresholds
        Quality thresholds used to grade pulsar/variant rows.
    include_per_pulsar_pages
        Add residual-vs-MJD pages for highest-risk pulsar/variant rows when the
        QC CSVs contain MJD and residual columns.
    per_pulsar_page_limit
        Maximum number of per-pulsar pages.
    top_n
        Number of flagged TOA rows written to the compact PDF table and TSV.

    Returns
    -------
    ReleaseQualityReportResult or None
        ``None`` if matplotlib is unavailable.
    """
    try:
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        return None

    run_dir = Path(run_dir).expanduser().resolve()
    report_dir = (
        Path(report_dir)
        if report_dir is not None
        else run_dir / "release_quality_report"
    )
    if not report_dir.is_absolute():
        report_dir = run_dir / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    thresholds = thresholds or ReleaseQualityThresholds()

    all_qc = _read_qc_frames(
        run_dir, backend_col=backend_col, outlier_cols=outlier_cols
    )
    scorecard = _build_scorecard(all_qc, thresholds)
    backend_risks = _build_backend_risks(all_qc)
    flagged_toas = _build_flagged_toas(all_qc, top_n=top_n)

    scorecard_path = report_dir / "release_quality_scorecard.tsv"
    backend_risks_path = report_dir / "release_quality_backend_risks.tsv"
    flagged_toas_path = report_dir / "release_quality_flagged_toas.tsv"
    summary_json_path = report_dir / "release_quality_summary.json"
    pdf_path = report_dir / output_name

    scorecard.to_csv(scorecard_path, sep="\t", index=False)
    backend_risks.to_csv(backend_risks_path, sep="\t", index=False)
    flagged_toas.to_csv(flagged_toas_path, sep="\t", index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "overall_grade": _overall_grade(scorecard),
        "n_qc_tables": int(len(scorecard)),
        "n_pulsars": int(scorecard["pulsar"].nunique()) if not scorecard.empty else 0,
        "n_toas": int(scorecard["n_toas"].sum()) if not scorecard.empty else 0,
        "n_bad_toas": int(scorecard["bad_toa"].sum()) if not scorecard.empty else 0,
        "n_review_event_toas": (
            int(scorecard["review_event"].sum()) if not scorecard.empty else 0
        ),
        "thresholds": asdict(thresholds),
        "artifacts": {
            "pdf": str(pdf_path),
            "scorecard": str(scorecard_path),
            "backend_risks": str(backend_risks_path),
            "flagged_toas": str(flagged_toas_path),
        },
    }
    summary_json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with PdfPages(pdf_path) as pdf:
        _draw_cover_page(
            pdf,
            title=title or "PLEB Release Quality Report",
            run_dir=run_dir,
            scorecard=scorecard,
            all_qc=all_qc,
            thresholds=thresholds,
        )
        show_cols = [
            "grade",
            "pulsar",
            "variant",
            "branch",
            "n_toas",
            "bad_toa",
            "review_event",
            "event",
            "bad_plus_review_fraction",
            "residual_rms_us",
        ]
        _draw_table_page(
            pdf,
            "Pulsar/variant release scorecard",
            (
                scorecard[[c for c in show_cols if c in scorecard.columns]]
                if not scorecard.empty
                else scorecard
            ),
        )
        backend_cols = [
            "backend",
            "backend_col",
            "n_toas",
            "bad_toa",
            "review_event",
            "event",
            "bad_plus_review_fraction",
            "n_pulsars",
            "top_reasons",
        ]
        _draw_table_page(
            pdf,
            "Backend risk ranking",
            (
                backend_risks[[c for c in backend_cols if c in backend_risks.columns]]
                if not backend_risks.empty
                else backend_risks
            ),
            rows_per_page=20,
            font_size=6.6,
        )
        _draw_table_page(
            pdf,
            "Top flagged TOAs",
            flagged_toas,
            rows_per_page=22,
            font_size=6.4,
        )
        if include_per_pulsar_pages:
            _draw_per_pulsar_pages(
                pdf,
                all_qc,
                max_pages=max(0, int(per_pulsar_page_limit)),
            )

    return ReleaseQualityReportResult(
        report_dir=report_dir,
        pdf_path=pdf_path,
        scorecard_path=scorecard_path,
        backend_risks_path=backend_risks_path,
        flagged_toas_path=flagged_toas_path,
        summary_json_path=summary_json_path,
    )
