"""Generate report artifacts from existing ``pqc`` CSV outputs.

This module is a post-processing/reporting layer. It does not re-run ``pqc``;
instead it reads ``*_qc.csv`` files, renders helper-script diagnostics, and can
assemble a compact PDF with actionable per-backend tables.

Notes
-----
Compact decisions are derived from two logical sets:

- outlier set (by default union of ``outlier_any``, ``bad_point``,
  robust/bad-mad columns, etc.)
- event set (transient, solar, eclipse, Gaussian bump, glitch, orbital flags)

Decision rules:

- ``BAD_TOA``: outlier and not event
- ``REVIEW_EVENT``: outlier and event
- ``EVENT``: event and not outlier
- ``KEEP``: neither set

References
----------
- PQC docs: https://golamshaifullah.github.io/pqc/index.html
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
    return s.fillna(False).astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})


def _build_compact_decisions(
    df: pd.DataFrame, outlier_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """Build compact decision labels from raw QC columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-TOA QC table from ``pqc``.
    outlier_cols : list of str, optional
        Explicit outlier columns to union. If omitted, a compatibility default
        set is used.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with additional columns:
        ``decision``, ``outlier_any_compact``, ``event_any_compact``,
        ``decision_reason``.

    Notes
    -----
    This function performs Boolean set logic, not probabilistic inference.
    ``REVIEW_EVENT`` is intentionally conservative: it highlights rows that are
    simultaneously outlier-like and event-like for human inspection.
    """
    out = df.copy()
    outlier_any = pd.Series([False] * len(out), index=out.index)
    cols = (
        outlier_cols
        if outlier_cols
        else [
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
    )
    for c in cols:
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

    outlier_cols = [c for c in cols if c in out.columns]
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
        reason_out[decision == "REVIEW_EVENT"]
        + " |event| "
        + reason_evt[decision == "REVIEW_EVENT"]
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
    run_dir: Path,
    backend_col: str = "group",
    outlier_cols: Optional[list[str]] = None,
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

    def _parse_psr_variant(csv_path: Path) -> tuple[str, str, str]:
        stem = csv_path.stem
        core = stem[: -len("_qc")] if stem.endswith("_qc") else stem
        if "." in core:
            psr, variant = core.split(".", 1)
        else:
            psr, variant = core, "base"
        label = psr if variant == "base" else f"{psr} [{variant}]"
        return psr, variant, label

    with PdfPages(pdf_path) as pdf:
        # Cover / aggregate page
        totals = []
        for p in csvs:
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            d = _build_compact_decisions(df, outlier_cols=outlier_cols)
            psr, variant, label = _parse_psr_variant(p)
            totals.append(
                {
                    "target": label,
                    "pulsar": psr,
                    "variant": variant,
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
            cols = ["target", "n_toa", "bad_toa", "review_event", "event"]
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
            d = _build_compact_decisions(df, outlier_cols=outlier_cols)
            psr, variant, label = _parse_psr_variant(p)
            mjd = pd.to_numeric(d.get("mjd", pd.Series([])), errors="coerce")
            resid = pd.to_numeric(
                d.get("resid_us", d.get("resid", pd.Series([]))), errors="coerce"
            )

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
                if label == "REVIEW_EVENT":
                    ax.scatter(
                        m[sel],
                        r[sel],
                        s=18,
                        alpha=0.9,
                        facecolors="none",
                        edgecolors=colors[label],
                        linewidths=0.9,
                        marker="o",
                        label=f"{label} ({int(sel.sum())})",
                    )
                else:
                    ax.scatter(
                        m[sel],
                        r[sel],
                        s=10 if label == "KEEP" else 14,
                        alpha=0.55 if label == "KEEP" else 0.8,
                        c=colors[label],
                        label=f"{label} ({int(sel.sum())})",
                    )
            ax.set_title(f"{label}: Residuals with compact decisions")
            ax.set_xlabel("MJD")
            ax.set_ylabel("Residual")
            ax.legend(loc="best", fontsize=8, frameon=False)

            # Top suspicious backends
            ax2 = axes[1]
            bcol = (
                backend_col
                if backend_col in d.columns
                else ("sys" if "sys" in d.columns else None)
            )
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

            # Residual-vs-uncertainty diagnostics (colored by backend and PQC class).
            def _uncert_series(frame: pd.DataFrame) -> pd.Series:
                for c in (
                    "sigma_us",
                    "toa_err_us",
                    "err_us",
                    "toa_uncertainty_us",
                    "sigma",
                    "toa_err",
                    "error",
                    "err",
                ):
                    if c in frame.columns:
                        return pd.to_numeric(frame[c], errors="coerce")
                return pd.Series([pd.NA] * len(frame), index=frame.index, dtype=float)

            def _pqc_class_series(frame: pd.DataFrame) -> pd.Series:
                out = pd.Series(["good"] * len(frame), index=frame.index, dtype=object)
                event_cols = [
                    ("step_member", "step"),
                    ("dm_step_member", "dm_step"),
                    ("transient_member", "transient"),
                    ("solar_event_member", "solar"),
                    ("orbital_phase_bad", "orbital_phase"),
                    ("eclipse_member", "eclipse"),
                    ("gaussian_bump_member", "gaussian_bump"),
                    ("glitch_member", "glitch"),
                ]
                for col, label in event_cols:
                    sel = _bool_series(frame, col)
                    out.loc[sel] = f"event_{label}"
                if "transient_id" in frame.columns:
                    tid = (
                        pd.to_numeric(frame["transient_id"], errors="coerce")
                        .fillna(-1)
                        .astype(int)
                    )
                    out.loc[tid >= 0] = "event_transient"
                out.loc[frame["decision"] == "BAD_TOA"] = "bad"
                out.loc[frame["decision"] == "REVIEW_EVENT"] = "event_review"
                return out.astype(str)

            resid2 = pd.to_numeric(
                d.get("resid_us", d.get("resid", pd.Series([]))), errors="coerce"
            )
            unc = _uncert_series(d)
            valid2 = resid2.notna() & unc.notna()
            dd2 = d.loc[valid2].copy()
            if not dd2.empty:
                dd2["_resid"] = resid2.loc[valid2].astype(float)
                dd2["_unc"] = unc.loc[valid2].astype(float)
                dd2["_pqc_class"] = _pqc_class_series(d).loc[valid2]
                bcol3 = (
                    backend_col
                    if backend_col in dd2.columns
                    else ("sys" if "sys" in dd2.columns else None)
                )

                fig2, axarr = plt.subplots(1, 2, figsize=(11, 4.6))
                axb, axc = axarr[0], axarr[1]

                if bcol3 is not None:
                    for be, g in dd2.groupby(bcol3):
                        axb.scatter(
                            g["_unc"],
                            g["_resid"],
                            s=11,
                            alpha=0.75,
                            label=str(be),
                        )
                    axb.legend(loc="best", fontsize=6, frameon=False, ncol=2)
                else:
                    axb.scatter(dd2["_unc"], dd2["_resid"], s=11, alpha=0.75)
                axb.set_title(f"{label}: Residual vs TOA uncertainty (by backend)")
                axb.set_xlabel("TOA uncertainty")
                axb.set_ylabel("Residual")

                cls_palette = {
                    "good": "#6c757d",
                    "bad": "#d62728",
                    "event_review": "#d95f02",
                }
                for cls, g in dd2.groupby("_pqc_class"):
                    axc.scatter(
                        g["_unc"],
                        g["_resid"],
                        s=11 if cls == "good" else 14,
                        alpha=0.62 if cls == "good" else 0.84,
                        c=cls_palette.get(str(cls), None),
                        label=str(cls),
                    )
                axc.legend(loc="best", fontsize=6, frameon=False, ncol=2)
                axc.set_title(f"{label}: Residual vs TOA uncertainty (by -pqc class)")
                axc.set_xlabel("TOA uncertainty")
                axc.set_ylabel("Residual")

                pdf.savefig(fig2, bbox_inches="tight")
                plt.close(fig2)

            # Per-backend action list page with artifact pointers.
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.set_title(f"{label}: Per-backend action list", fontsize=14, pad=10)
            bcol2 = bcol
            if bcol2 is None:
                ax.text(0.02, 0.90, "No backend/system column available.", fontsize=10)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                continue
            flagged = d[d["decision"].isin(["BAD_TOA", "REVIEW_EVENT"])].copy()
            if flagged.empty:
                ax.text(
                    0.02,
                    0.90,
                    "No BAD_TOA/REVIEW_EVENT rows for this pulsar.",
                    fontsize=10,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                continue

            rows = []
            stem = p.stem
            diag_path = (pdf_path.parent / "diagnostics" / f"{stem}.txt").as_posix()
            summary_png = (
                pdf_path.parent / "summary" / f"{stem}_summary.png"
            ).as_posix()
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
                var_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(variant))
                out_csv = action_root / f"{psr}.{var_slug}__{be_slug}.csv"
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

            # JUMP summary table page (best effort par-file discovery).
            def _find_par() -> Optional[Path]:
                for c in ("_parfile", "parfile"):
                    if c in d.columns:
                        vals = d[c].dropna().astype(str).unique().tolist()
                        for v in vals:
                            pp = Path(v).expanduser()
                            if pp.exists():
                                return pp
                roots = [run_dir, run_dir.parent, run_dir.parent.parent]
                seen = set()
                for root in roots:
                    rr = Path(root).resolve()
                    if rr in seen or not rr.exists():
                        continue
                    seen.add(rr)
                    if variant != "base":
                        direct_variant = rr / psr / f"{psr}_{variant}.par"
                        if direct_variant.exists():
                            return direct_variant
                        direct_variant = rr / psr / f"{psr}.{variant}.par"
                        if direct_variant.exists():
                            return direct_variant
                    direct = rr / psr / f"{psr}.par"
                    if direct.exists():
                        return direct
                    try:
                        if variant != "base":
                            cvar = list(rr.glob(f"**/{psr}/{psr}_{variant}.par"))
                            if cvar:
                                return sorted(cvar, key=lambda x: len(str(x)))[0]
                            cvar = list(rr.glob(f"**/{psr}/{psr}.{variant}.par"))
                            if cvar:
                                return sorted(cvar, key=lambda x: len(str(x)))[0]
                        candidates = list(rr.glob(f"**/{psr}/{psr}.par"))
                    except Exception:
                        candidates = []
                    if candidates:
                        return sorted(candidates, key=lambda x: len(str(x)))[0]
                return None

            def _read_jumps(par: Path) -> pd.DataFrame:
                rows_j = []
                try:
                    lines = par.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()
                except Exception:
                    return pd.DataFrame([])
                for ln in lines:
                    s = ln.strip()
                    if not s or s.startswith(("C ", "#")):
                        continue
                    if not s.startswith("JUMP"):
                        continue
                    parts = s.split()
                    if len(parts) < 5:
                        continue
                    jump_flag = parts[1]
                    system = parts[2]
                    value = parts[3]
                    fit_flag = parts[4]
                    rows_j.append(
                        {
                            "jump_flag": jump_flag,
                            "system": system,
                            "value": value,
                            "fit_flag": fit_flag,
                        }
                    )
                return pd.DataFrame(rows_j)

            par_path = _find_par()
            figj = plt.figure(figsize=(11, 8.5))
            axj = figj.add_subplot(111)
            axj.axis("off")
            axj.set_title(f"{label}: JUMP summary", fontsize=14, pad=10)
            if par_path is None:
                axj.text(
                    0.02,
                    0.90,
                    "Could not locate par file for JUMP summary.",
                    fontsize=10,
                )
            else:
                jdf = _read_jumps(par_path)
                axj.text(
                    0.02, 0.95, f"par: {par_path.as_posix()}", fontsize=8, va="top"
                )
                if jdf.empty:
                    axj.text(
                        0.02, 0.90, "No JUMP lines found in par file.", fontsize=10
                    )
                else:
                    show_cols = ["jump_flag", "system", "value", "fit_flag"]
                    table = axj.table(
                        cellText=jdf[show_cols].values.tolist(),
                        colLabels=show_cols,
                        loc="upper left",
                        cellLoc="left",
                        colLoc="left",
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1, 1.2)
            pdf.savefig(figj, bbox_inches="tight")
            plt.close(figj)


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
    compact_outlier_cols: Optional[list[str]] = None,
) -> Path:
    """Generate diagnostics, plots, and optional compact PDF from QC CSVs.

    Parameters
    ----------
    run_dir : pathlib.Path
        Pipeline run directory containing PQC outputs.
    backend_col : str, optional
        Column name used for backend grouping.
    backend : str, optional
        Optional backend key filter for plotting.
    report_dir : pathlib.Path, optional
        Output directory. Defaults to ``<run_dir>/qc_report``.
    no_plots : bool, optional
        If ``True``, skip transient plots.
    structure_group_cols : str, optional
        Grouping columns for structure summaries.
    no_feature_plots : bool, optional
        If ``True``, skip feature plots.
    compact_pdf : bool, optional
        If ``True``, generate a compact composite PDF report.
    compact_pdf_name : str, optional
        Output PDF filename under ``report_dir``.
    compact_outlier_cols : list of str, optional
        QC columns used to define outliers in compact decisions/reporting.

    Returns
    -------
    pathlib.Path
        Report directory path.

    Raises
    ------
    FileNotFoundError
        If ``run_dir`` does not exist.
    RuntimeError
        If required PQC CSVs or helper scripts are missing.

    Notes
    -----
    Statistical interpretation:

    - Report plots are descriptive diagnostics, not hypothesis tests by
      themselves.
    - Counts in compact pages are useful for triage but should be interpreted
      with cadence/system context.
    - If ``compact_outlier_cols`` is set, compact decision logic is driven only
      by those columns.

    References
    ----------
    - PQC docs: https://golamshaifullah.github.io/pqc/index.html

    Examples
    --------
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
        _write_compact_pdf(
            csvs,
            pdf_path,
            run_dir=run_dir,
            backend_col=backend_col,
            outlier_cols=compact_outlier_cols,
        )
        logger.info("Wrote compact QC PDF report: %s", pdf_path)

    return report_dir


def _choose_time_col(
    df: pd.DataFrame, preferred: Optional[str] = None
) -> Optional[str]:
    """Choose the best-available MJD-like time column."""
    if preferred and preferred in df.columns:
        return preferred
    for cand in ("mjd", "mjd_sat", "sat_mjd", "mjd_bat", "bat_mjd", "bat"):
        if cand in df.columns:
            return cand
    return None


def generate_cross_pulsar_coincidence_report(
    run_dir: Path,
    report_dir: Optional[Path] = None,
    *,
    time_col: Optional[str] = None,
    window_days: float = 1.0,
    min_pulsars: int = 2,
    include_outliers: bool = True,
    include_events: bool = True,
    outlier_cols: Optional[list[str]] = None,
    event_cols: Optional[list[str]] = None,
) -> Optional[Path]:
    """Generate cross-pulsar coincidence tables from PQC CSV outputs.

    This stage is purely post-processing and does not rerun PQC.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    csvs = _find_qc_csvs(run_dir)
    if not csvs:
        return None

    report_dir = (
        Path(report_dir).expanduser().resolve()
        if report_dir
        else run_dir / "qc_cross_pulsar"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[pd.DataFrame] = []
    selected_counts: list[dict[str, object]] = []
    fallback_event_cols = (
        event_cols
        if event_cols
        else [
            "transient_member",
            "solar_event_member",
            "orbital_phase_bad",
            "eclipse_member",
            "gaussian_bump_member",
            "glitch_member",
        ]
    )

    for csv in csvs:
        pulsar = csv.stem.replace("_qc", "")
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        tcol = _choose_time_col(df, preferred=time_col)
        if tcol is None:
            continue
        mjd = pd.to_numeric(df[tcol], errors="coerce")
        valid = mjd.notna()
        if not bool(valid.any()):
            continue
        out = _build_compact_decisions(df, outlier_cols=outlier_cols)
        is_out = _bool_series(out, "outlier_any_compact")
        is_evt = pd.Series([False] * len(out), index=out.index)
        for c in fallback_event_cols:
            is_evt |= _bool_series(out, c)
        if "transient_id" in out.columns:
            tid = (
                pd.to_numeric(out["transient_id"], errors="coerce")
                .fillna(-1)
                .astype(int)
            )
            is_evt |= tid >= 0
        if "event_any_compact" in out.columns:
            is_evt |= _bool_series(out, "event_any_compact")

        keep = pd.Series([False] * len(out), index=out.index)
        if include_outliers:
            keep |= is_out
        if include_events:
            keep |= is_evt
        keep &= valid
        n_keep = int(keep.sum())
        selected_counts.append(
            {"pulsar": pulsar, "selected_rows": n_keep, "qc_csv": str(csv)}
        )
        if n_keep == 0:
            continue

        part = pd.DataFrame(
            {
                "pulsar": pulsar,
                "mjd": mjd[keep].astype(float),
                "kind": [
                    (
                        "outlier+event"
                        if (bool(is_out.loc[i]) and bool(is_evt.loc[i]))
                        else ("outlier" if bool(is_out.loc[i]) else "event")
                    )
                    for i in out.index[keep]
                ],
                "qc_csv": str(csv),
                "row_index": out.index[keep].astype(int),
                "decision": out.loc[keep, "decision"].astype(str),
                "decision_reason": out.loc[keep, "decision_reason"].astype(str),
            }
        )
        rows.append(part)

    pd.DataFrame(selected_counts).to_csv(
        report_dir / "selected_row_counts.tsv", sep="\t", index=False
    )
    if not rows:
        return report_dir

    points = (
        pd.concat(rows, ignore_index=True).sort_values("mjd").reset_index(drop=True)
    )
    points.to_csv(report_dir / "coincident_points.tsv", sep="\t", index=False)

    # Build coincidence clusters as contiguous MJD runs where consecutive points
    # are closer than window_days.
    window = max(0.0, float(window_days))
    labels = []
    cluster = 0
    prev = None
    for m in points["mjd"].to_numpy():
        if prev is None or (float(m) - float(prev)) > window:
            cluster += 1
        labels.append(cluster)
        prev = float(m)
    points["cluster_id"] = labels

    clusters = []
    for cid, sub in points.groupby("cluster_id", sort=True):
        pulsar_set = sorted(set(sub["pulsar"].astype(str)))
        npulsars = len(pulsar_set)
        if npulsars < int(min_pulsars):
            continue
        kind_counts = sub["kind"].value_counts().to_dict()
        clusters.append(
            {
                "cluster_id": int(cid),
                "mjd_start": float(sub["mjd"].min()),
                "mjd_end": float(sub["mjd"].max()),
                "span_days": float(sub["mjd"].max() - sub["mjd"].min()),
                "n_points": int(len(sub)),
                "n_pulsars": int(npulsars),
                "pulsars": ",".join(pulsar_set),
                "outlier_points": int(kind_counts.get("outlier", 0)),
                "event_points": int(kind_counts.get("event", 0)),
                "outlier_event_points": int(kind_counts.get("outlier+event", 0)),
            }
        )

    cluster_df = pd.DataFrame(clusters)
    if not cluster_df.empty:
        cluster_df = cluster_df.sort_values(
            ["n_pulsars", "n_points", "mjd_start"], ascending=[False, False, True]
        )
        cluster_df.to_csv(
            report_dir / "coincidence_clusters.tsv", sep="\t", index=False
        )
        keep_ids = set(cluster_df["cluster_id"].astype(int).tolist())
        points[points["cluster_id"].isin(keep_ids)].to_csv(
            report_dir / "coincidence_cluster_points.tsv", sep="\t", index=False
        )
    else:
        (report_dir / "coincidence_clusters.tsv").write_text(
            "cluster_id\tmjd_start\tmjd_end\tspan_days\tn_points\tn_pulsars\tpulsars\toutlier_points\tevent_points\toutlier_event_points\n",
            encoding="utf-8",
        )

    return report_dir
