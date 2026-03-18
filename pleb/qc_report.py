"""Generate PQC report artifacts from pipeline outputs.

This module wraps helper scripts that produce diagnostic text, transient plots,
feature plots, and structure summaries from PQC CSV outputs.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Optional
import subprocess
import sys
import re

from .logging_utils import get_logger

try:
    from pqc.utils.diagnostics import export_structure_table
except Exception:  # pragma: no cover
    export_structure_table = None  # type: ignore[assignment]

logger = get_logger("pleb.qc_report")


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) != 0.0
    return s.fillna(False).astype(str).str.lower().isin(
        {"1", "true", "t", "yes", "y"}
    )


def _build_compact_decisions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    outlier_any = pd.Series([False] * len(out), index=out.index)
    for c in (
        "outlier_any",
        "bad_point",
        "bad_hard",
        "robust_outlier",
        "robust_global_outlier",
        "bad_mad",
        "bad_ou",
        "bad",
        "bad_day",
    ):
        outlier_any |= _bool_series(out, c)
    event_any = pd.Series([False] * len(out), index=out.index)
    for c in (
        "transient_member",
        "solar_event_member",
        "orbital_phase_bad",
        "eclipse_member",
        "gaussian_bump_member",
        "glitch_member",
    ):
        event_any |= _bool_series(out, c)
    if "transient_id" in out.columns:
        tid = pd.to_numeric(out["transient_id"], errors="coerce").fillna(-1).astype(int)
        event_any |= tid >= 0

    decision = pd.Series(["KEEP"] * len(out), index=out.index)
    decision[outlier_any & (~event_any)] = "BAD_TOA"
    decision[outlier_any & event_any] = "REVIEW_EVENT"
    decision[(~outlier_any) & event_any] = "EVENT"

    outlier_cols = [
        c
        for c in (
            "outlier_any",
            "bad_point",
            "bad_hard",
            "robust_outlier",
            "robust_global_outlier",
            "bad_mad",
            "bad_ou",
            "bad",
            "bad_day",
        )
        if c in out.columns
    ]
    event_cols = [
        c
        for c in (
            "transient_member",
            "solar_event_member",
            "orbital_phase_bad",
            "eclipse_member",
            "gaussian_bump_member",
            "glitch_member",
        )
        if c in out.columns
    ]

    reason = pd.Series([""] * len(out), index=out.index, dtype=object)
    if outlier_cols:
        reason_out = (
            pd.DataFrame({c: _bool_series(out, c) for c in outlier_cols})
            .apply(lambda r: ",".join([c for c, v in r.items() if bool(v)]), axis=1)
            .astype(str)
        )
    else:
        reason_out = pd.Series([""] * len(out), index=out.index, dtype=object)
    if event_cols:
        reason_evt = (
            pd.DataFrame({c: _bool_series(out, c) for c in event_cols})
            .apply(lambda r: ",".join([c for c, v in r.items() if bool(v)]), axis=1)
            .astype(str)
        )
    else:
        reason_evt = pd.Series([""] * len(out), index=out.index, dtype=object)
    if "transient_id" in out.columns:
        tid = pd.to_numeric(out["transient_id"], errors="coerce").fillna(-1).astype(int)
        has_tid = tid >= 0
        reason_evt.loc[has_tid] = reason_evt.loc[has_tid].apply(
            lambda s: f"{s},transient_id" if s else "transient_id"
        )
    reason[decision == "BAD_TOA"] = reason_out[decision == "BAD_TOA"]
    reason[decision == "REVIEW_EVENT"] = (
        reason_out[decision == "REVIEW_EVENT"] + " |event| " + reason_evt[decision == "REVIEW_EVENT"]
    )
    reason[decision == "EVENT"] = reason_evt[decision == "EVENT"]
    reason = reason.str.strip().str.strip("|").replace("", "n/a")

    out["decision"] = decision
    out["outlier_any_compact"] = outlier_any
    out["event_any_compact"] = event_any
    out["decision_reason"] = reason
    return out


def _write_compact_pdf(
    csvs: list[Path],
    pdf_path: Path,
    backend_col: str = "group",
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Compact PDF report requires matplotlib. Install extras with `pip install .[plot]`."
        ) from e

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    action_root = pdf_path.parent / "action_lists"
    action_root.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        # Cover / aggregate page
        totals = []
        for p in csvs:
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            d = _build_compact_decisions(df)
            totals.append(
                {
                    "pulsar": p.stem.replace("_qc", ""),
                    "n_toa": int(len(d)),
                    "bad_toa": int((d["decision"] == "BAD_TOA").sum()),
                    "review_event": int((d["decision"] == "REVIEW_EVENT").sum()),
                    "event": int((d["decision"] == "EVENT").sum()),
                }
            )
        tdf = pd.DataFrame(totals)
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title("PQC Compact Report", fontsize=16, pad=12)
        if tdf.empty:
            ax.text(0.02, 0.90, "No QC CSV files found.", fontsize=11, va="top")
        else:
            cols = ["pulsar", "n_toa", "bad_toa", "review_event", "event"]
            show = tdf[cols].sort_values(["bad_toa", "review_event"], ascending=False)
            table = ax.table(
                cellText=show.values.tolist(),
                colLabels=cols,
                loc="upper left",
                cellLoc="left",
                colLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.2)
            ax.text(
                0.02,
                0.04,
                "Policy: BAD_TOA = outlier and not event; REVIEW_EVENT = outlier+event; EVENT = event only.",
                fontsize=9,
            )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Per-pulsar pages
        for p in csvs:
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            d = _build_compact_decisions(df)
            psr = p.stem.replace("_qc", "")
            mjd = pd.to_numeric(d.get("mjd", pd.Series([])), errors="coerce")
            resid = pd.to_numeric(d.get("resid_us", d.get("resid", pd.Series([]))), errors="coerce")

            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), height_ratios=[2.6, 1.4])
            ax = axes[0]
            mask_valid = mjd.notna() & resid.notna()
            dd = d.loc[mask_valid].copy()
            m = mjd.loc[mask_valid]
            r = resid.loc[mask_valid]
            colors = {
                "KEEP": "#5a5a5a",
                "EVENT": "#1b9e77",
                "REVIEW_EVENT": "#d95f02",
                "BAD_TOA": "#d62728",
            }
            for label in ["KEEP", "EVENT", "REVIEW_EVENT", "BAD_TOA"]:
                sel = dd["decision"] == label
                if int(sel.sum()) == 0:
                    continue
                ax.scatter(
                    m[sel],
                    r[sel],
                    s=10 if label == "KEEP" else 14,
                    alpha=0.55 if label == "KEEP" else 0.8,
                    c=colors[label],
                    label=f"{label} ({int(sel.sum())})",
                )
            ax.set_title(f"{psr}: Residuals with compact decisions")
            ax.set_xlabel("MJD")
            ax.set_ylabel("Residual")
            ax.legend(loc="best", fontsize=8, frameon=False)

            # Top suspicious backends
            ax2 = axes[1]
            bcol = backend_col if backend_col in d.columns else ("sys" if "sys" in d.columns else None)
            if bcol is not None and len(d):
                bad = d[d["decision"].isin(["BAD_TOA", "REVIEW_EVENT"])].copy()
                if len(bad):
                    vc = bad[bcol].fillna("NA").astype(str).value_counts().head(12)
                    ax2.barh(vc.index[::-1], vc.values[::-1], color="#444444")
                    ax2.set_title(f"Top flagged {bcol} (BAD_TOA + REVIEW_EVENT)")
                    ax2.set_xlabel("Count")
                else:
                    ax2.axis("off")
                    ax2.text(0.02, 0.5, "No flagged TOAs for this pulsar.", fontsize=10)
            else:
                ax2.axis("off")
                ax2.text(0.02, 0.5, "Backend column not available.", fontsize=10)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Per-backend action list page with artifact pointers.
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.set_title(f"{psr}: Per-backend action list", fontsize=14, pad=10)
            bcol2 = bcol
            if bcol2 is None:
                ax.text(0.02, 0.90, "No backend/system column available.", fontsize=10)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                continue
            flagged = d[d["decision"].isin(["BAD_TOA", "REVIEW_EVENT"])].copy()
            if flagged.empty:
                ax.text(0.02, 0.90, "No BAD_TOA/REVIEW_EVENT rows for this pulsar.", fontsize=10)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                continue

            rows = []
            stem = p.stem
            diag_path = (pdf_path.parent / "diagnostics" / f"{stem}.txt").as_posix()
            summary_png = (pdf_path.parent / "summary" / f"{stem}_summary.png").as_posix()
            for be, g in flagged.groupby(bcol2):
                be_str = str(be)
                bad_toa = int((g["decision"] == "BAD_TOA").sum())
                review = int((g["decision"] == "REVIEW_EVENT").sum())
                reasons = (
                    g.loc[g["decision"] == "BAD_TOA", "decision_reason"]
                    .fillna("n/a")
                    .astype(str)
                    .value_counts()
                )
                top_reason = reasons.index[0] if len(reasons) else "n/a"
                be_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", be_str)
                out_csv = action_root / f"{psr}__{be_slug}.csv"
                cols_keep = [
                    c
                    for c in (
                        "mjd",
                        "freq",
                        bcol2,
                        "_timfile",
                        "filename",
                        "decision",
                        "decision_reason",
                        "resid",
                        "resid_us",
                    )
                    if c in g.columns
                ]
                gg = g[cols_keep].copy() if cols_keep else g.copy()
                gg.to_csv(out_csv, index=False)
                rows.append(
                    [
                        be_str,
                        bad_toa,
                        review,
                        top_reason[:70],
                        out_csv.as_posix(),
                    ]
                )

            if not rows:
                ax.text(0.02, 0.90, "No backend rows after grouping.", fontsize=10)
            else:
                col_labels = [
                    bcol2,
                    "BAD_TOA",
                    "REVIEW_EVENT",
                    "Top BAD reason",
                    "Action CSV",
                ]
                t = ax.table(
                    cellText=rows,
                    colLabels=col_labels,
                    loc="upper left",
                    cellLoc="left",
                    colLoc="left",
                )
                t.auto_set_font_size(False)
                t.set_fontsize(8)
                t.scale(1, 1.2)
                ax.text(
                    0.02,
                    0.08,
                    f"QC CSV: {p.as_posix()}\nDiagnostics: {diag_path}\nSummary plot: {summary_png}",
                    fontsize=8.5,
                    va="bottom",
                )
                ax.text(
                    0.02,
                    0.02,
                    "Use Action CSV per backend to review exact TOAs and reasons before editing/commenting.",
                    fontsize=9,
                    va="bottom",
                )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _find_qc_csvs(run_dir: Path) -> list[Path]:
    """Find pqc CSVs under a run directory."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))
    candidates: list[Path] = []
    qc_dir = run_dir / "qc"
    if qc_dir.exists():
        candidates.extend(qc_dir.rglob("*_qc.csv"))
    candidates.extend(run_dir.rglob("*_qc.csv"))
    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return sorted(out, key=lambda x: str(x))


def _run_script(
    script: Path, args: list[str], capture: bool = False
) -> subprocess.CompletedProcess:
    """Run a helper script as a subprocess."""
    cmd = [sys.executable, str(script)] + args
    return subprocess.run(cmd, capture_output=capture, text=True)


def generate_qc_report(
    run_dir: Path,
    backend_col: str = "group",
    backend: Optional[str] = None,
    report_dir: Optional[Path] = None,
    no_plots: bool = False,
    structure_group_cols: Optional[str] = None,
    no_feature_plots: bool = False,
    compact_pdf: bool = False,
    compact_pdf_name: str = "qc_compact_report.pdf",
) -> Path:
    """Generate diagnostics and transient plots for pqc outputs.

    Args:
        run_dir: Pipeline run directory that contains pqc outputs.
        backend_col: Column name used for backend grouping.
        backend: Optional backend key filter for plotting.
        report_dir: Output directory (default: <run_dir>/qc_report).
        no_plots: If True, skip transient plots.
        structure_group_cols: Optional grouping columns for structure summaries.
        no_feature_plots: If True, skip feature plots.
        compact_pdf: If True, generate a compact composite PDF report.
        compact_pdf_name: Output PDF filename under report_dir.

    Returns:
        Path to the report directory.

    Raises:
        FileNotFoundError: If the run directory does not exist.
        RuntimeError: If required PQC CSVs or helper scripts are missing.

    Examples:
        Generate a QC report from a run directory::

            report_dir = generate_qc_report(Path("results/run_2024-01-01"))
    """
    if export_structure_table is None:
        raise RuntimeError(
            "qc_report requires `pqc` to be installed. Install extras with `pip install .[qc]`."
        )

    run_dir = Path(run_dir).expanduser().resolve()
    report_dir = (
        Path(report_dir).expanduser().resolve() if report_dir else run_dir / "qc_report"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    csvs = _find_qc_csvs(run_dir)
    if not csvs:
        raise RuntimeError(f"No *_qc.csv files found under {run_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    diag_script = scripts_dir / "diagnose_qc.py"
    plot_script = scripts_dir / "plot_transients.py"
    feature_script = scripts_dir / "plot_features.py"
    summary_script = scripts_dir / "plot_qc_summary.py"
    if not diag_script.exists():
        raise RuntimeError(f"Missing script: {diag_script}")
    if not plot_script.exists():
        raise RuntimeError(f"Missing script: {plot_script}")
    if not feature_script.exists():
        raise RuntimeError(f"Missing script: {feature_script}")
    if not summary_script.exists():
        raise RuntimeError(f"Missing script: {summary_script}")

    diag_dir = report_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    plots_root = report_dir / "plots"
    if not no_plots:
        plots_root.mkdir(parents=True, exist_ok=True)
    feature_root = report_dir / "features"
    if not no_feature_plots:
        feature_root.mkdir(parents=True, exist_ok=True)
    summary_root = report_dir / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    for csv_path in csvs:
        stem = csv_path.stem
        diag_out = diag_dir / f"{stem}.txt"
        proc = _run_script(
            diag_script,
            ["--csv", str(csv_path), "--backend-col", str(backend_col)]
            + (
                ["--structure-group-cols", str(structure_group_cols)]
                if structure_group_cols
                else []
            ),
            capture=True,
        )
        diag_out.write_text(
            (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else ""),
            encoding="utf-8",
        )
        if proc.returncode != 0:
            logger.warning(
                "diagnose_qc.py failed for %s (code %s)", csv_path, proc.returncode
            )

        if not no_plots:
            outdir = plots_root / stem
            plot_args = [
                "--csv",
                str(csv_path),
                "--backend-col",
                str(backend_col),
                "--outdir",
                str(outdir),
            ]
            if backend is not None:
                plot_args += ["--backend", str(backend)]
            proc = _run_script(plot_script, plot_args, capture=True)
            if proc.returncode != 0:
                msg = (proc.stdout or "") + (
                    ("\n" + proc.stderr) if proc.stderr else ""
                )
                msg = msg.strip()
                if any(
                    token in msg
                    for token in (
                        "No transients found in CSV.",
                        "CSV has no transient_id",
                        "No transients for backend",
                    )
                ):
                    logger.info("No transients to plot for %s", csv_path)
                else:
                    logger.warning(
                        "plot_transients.py failed for %s (code %s): %s",
                        csv_path,
                        proc.returncode,
                        msg,
                    )

        if not no_feature_plots:
            outdir = feature_root / stem
            feat_args = [
                "--csv",
                str(csv_path),
                "--backend-col",
                str(backend_col),
                "--outdir",
                str(outdir),
            ]
            if structure_group_cols:
                feat_args += ["--structure-group-cols", str(structure_group_cols)]
            proc = _run_script(feature_script, feat_args, capture=True)
            if proc.returncode != 0:
                msg = (proc.stdout or "") + (
                    ("\n" + proc.stderr) if proc.stderr else ""
                )
                msg = msg.strip()
                logger.warning(
                    "plot_features.py failed for %s (code %s): %s",
                    csv_path,
                    proc.returncode,
                    msg,
                )

        # Summary residual plot per pulsar
        summary_out = summary_root / f"{stem}_summary.png"
        summary_feat_dir = summary_root / stem
        proc = _run_script(
            summary_script,
            [
                "--csv",
                str(csv_path),
                "--out",
                str(summary_out),
                "--backend-col",
                str(backend_col),
            ]
            + (
                []
                if no_feature_plots
                else ["--feature-plots", "--feature-outdir", str(summary_feat_dir)]
            ),
            capture=True,
        )
        if proc.returncode != 0:
            msg = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
            msg = msg.strip()
            logger.warning(
                "plot_qc_summary.py failed for %s (code %s): %s",
                csv_path,
                proc.returncode,
                msg,
            )

        # Export structure summary table for this CSV
        try:
            df = pd.read_csv(csv_path)
            groupings = []
            if structure_group_cols:
                for raw in str(structure_group_cols).split(";"):
                    cols = tuple([c.strip() for c in raw.split(",") if c.strip()])
                    if cols:
                        groupings.append(cols)
            else:
                groupings = [(backend_col,)]
            for cols in groupings:
                label = "_".join(cols) if cols else "none"
                struct = export_structure_table(df, group_cols=cols)
                if struct.empty:
                    continue
                out = report_dir / f"structure_{label}.tsv"
                struct.to_csv(out, sep="	", index=False)
        except Exception as e:
            logger.warning("Failed to export structure table for %s: %s", csv_path, e)

    if compact_pdf:
        pdf_path = report_dir / str(compact_pdf_name)
        _write_compact_pdf(csvs, pdf_path, backend_col=backend_col)
        logger.info("Wrote compact QC PDF report: %s", pdf_path)

    return report_dir
