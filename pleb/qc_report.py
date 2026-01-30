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

from .logging_utils import get_logger
from pqc.utils.diagnostics import export_structure_table

logger = get_logger("pleb.qc_report")


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


def _run_script(script: Path, args: list[str], capture: bool = False) -> subprocess.CompletedProcess:
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

    Returns:
        Path to the report directory.

    Raises:
        FileNotFoundError: If the run directory does not exist.
        RuntimeError: If required PQC CSVs or helper scripts are missing.

    Examples:
        Generate a QC report from a run directory::

            report_dir = generate_qc_report(Path("results/run_2024-01-01"))
    """
    run_dir = Path(run_dir).expanduser().resolve()
    report_dir = Path(report_dir).expanduser().resolve() if report_dir else run_dir / "qc_report"
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
            ["--csv", str(csv_path), "--backend-col", str(backend_col)] + (["--structure-group-cols", str(structure_group_cols)] if structure_group_cols else []),
            capture=True,
        )
        diag_out.write_text((proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else ""), encoding="utf-8")
        if proc.returncode != 0:
            logger.warning("diagnose_qc.py failed for %s (code %s)", csv_path, proc.returncode)

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
                msg = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
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
                    logger.warning("plot_transients.py failed for %s (code %s): %s", csv_path, proc.returncode, msg)

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
                msg = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
                msg = msg.strip()
                logger.warning("plot_features.py failed for %s (code %s): %s", csv_path, proc.returncode, msg)

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
            + ([] if no_feature_plots else ["--feature-plots", "--feature-outdir", str(summary_feat_dir)]),
            capture=True,
        )
        if proc.returncode != 0:
            msg = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
            msg = msg.strip()
            logger.warning("plot_qc_summary.py failed for %s (code %s): %s", csv_path, proc.returncode, msg)

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

    return report_dir
