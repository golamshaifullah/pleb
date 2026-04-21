"""Generate consolidated run-level PDF reports.

This module synthesizes existing pipeline/workflow artifacts into a single PDF
summary. It does not attempt to merge existing PDFs byte-for-byte; instead it
reads structured artifacts (TSV/JSON/config text) and renders a compact run
report that points to the detailed stage outputs.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from textwrap import wrap
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

_A4_FIGSIZE = (8.27, 11.69)


def _find_run_settings_dir(run_dir: Path) -> Optional[Path]:
    direct = run_dir / "run_settings"
    if direct.exists():
        return direct
    ingest = run_dir / "ingest_reports" / "run_settings"
    if ingest.exists():
        return ingest
    return None


def _read_lines(path: Path, limit: int = 40) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) <= limit:
        return lines
    return lines[:limit] + [f"... ({len(lines) - limit} more lines omitted)"]


def _draw_text_page(
    pdf: Any, title: str, sections: Sequence[tuple[str, Sequence[str]]]
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=_A4_FIGSIZE)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, fontsize=16, loc="left")
    y = 0.97
    for heading, lines in sections:
        ax.text(0.03, y, heading, fontsize=12, fontweight="bold", va="top")
        y -= 0.035
        for line in lines or [""]:
            for chunk in wrap(str(line), width=112) or [""]:
                ax.text(0.05, y, chunk, fontsize=8.5, family="monospace", va="top")
                y -= 0.022
                if y < 0.05:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    fig = plt.figure(figsize=_A4_FIGSIZE)
                    ax = fig.add_subplot(111)
                    ax.axis("off")
                    ax.set_title(title, fontsize=16, loc="left")
                    y = 0.97
        y -= 0.02
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _draw_table_pages(
    pdf: Any,
    title: str,
    df: pd.DataFrame,
    *,
    rows_per_page: int = 28,
    font_size: float = 7.5,
) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        _draw_text_page(pdf, title, [("Summary", ["No rows available."])])
        return

    for start in range(0, len(df), rows_per_page):
        chunk = df.iloc[start : start + rows_per_page]
        fig = plt.figure(figsize=_A4_FIGSIZE)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title(title, fontsize=16, loc="left")
        table = ax.table(
            cellText=chunk.astype(str).values.tolist(),
            colLabels=[str(c) for c in chunk.columns],
            bbox=[0.0, 0.0, 1.0, 0.93],
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _summarize_fix_reports(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = run_dir / "fix_dataset"
    if not root.exists():
        return pd.DataFrame()
    for branch_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        json_path = branch_dir / "fix_dataset_report.json"
        tsv_path = branch_dir / "fix_dataset_summary.tsv"
        pdf_path = branch_dir / "fix_dataset_report.pdf"
        reports: list[dict[str, Any]] = []
        if json_path.exists():
            try:
                reports = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                reports = []
        error_count = sum(
            1 for r in reports if isinstance(r, dict) and bool(r.get("error"))
        )
        step_counts: Counter[str] = Counter()
        for report in reports:
            if not isinstance(report, dict):
                continue
            for step in report.get("steps", []) or []:
                if isinstance(step, dict):
                    step_counts.update(step.keys())
        added_includes = ""
        missing_jumps = ""
        removed_lines = ""
        if tsv_path.exists():
            try:
                df = pd.read_csv(tsv_path, sep="\t")
                if "added_includes" in df.columns:
                    added_includes = str(
                        int(
                            pd.to_numeric(
                                df["added_includes"], errors="coerce"
                            ).fillna(0).sum()
                        )
                    )
                if "missing_jumps" in df.columns:
                    missing_jumps = str(
                        int(
                            pd.to_numeric(
                                df["missing_jumps"], errors="coerce"
                            ).fillna(0).sum()
                        )
                    )
                if "removed_lines" in df.columns:
                    removed_lines = str(
                        int(
                            pd.to_numeric(
                                df["removed_lines"], errors="coerce"
                            ).fillna(0).sum()
                        )
                    )
            except Exception:
                pass
        rows.append(
            {
                "branch": branch_dir.name,
                "pulsars": len(reports),
                "errors": error_count,
                "added_includes": added_includes,
                "missing_jumps": missing_jumps,
                "removed_lines": removed_lines,
                "steps_seen": ", ".join(
                    f"{k}={v}" for k, v in sorted(step_counts.items())
                ),
                "json": str(json_path) if json_path.exists() else "",
                "pdf": str(pdf_path) if pdf_path.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def _summarize_qc(run_dir: Path) -> pd.DataFrame:
    summary_path = run_dir / "qc" / "qc_summary.tsv"
    if not summary_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(summary_path, sep="\t")
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if "qc_status" in df.columns:
        grouped = (
            df.groupby(["variant", "qc_status"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["variant", "qc_status"])
        )
        grouped["variant"] = grouped["variant"].fillna("base")
        return grouped

    rows = []
    for col in ("bad_point", "outlier_any", "solar_event_member", "orbital_phase_bad"):
        if col in df.columns:
            rows.append(
                {
                    "metric": col,
                    "count": int(df[col].fillna(False).astype(bool).sum()),
                }
            )
    return pd.DataFrame(rows)


def _artifact_inventory(run_dir: Path) -> pd.DataFrame:
    candidates = [
        run_dir / "run_report.pdf",
        run_dir / "workflow_report.pdf",
        run_dir / "ingest_reports" / "ingest_report.pdf",
        run_dir / "ingest_reports" / "ingest_breakdown.csv",
        run_dir / "qc" / "qc_summary.tsv",
        run_dir / "qc_report" / "qc_compact_report.pdf",
        run_dir / "binary_analysis" / "binary_analysis.tsv",
    ]
    if (run_dir / "fix_dataset").exists():
        for path in sorted((run_dir / "fix_dataset").glob("*/*")):
            if path.name in {
                "fix_dataset_report.pdf",
                "fix_dataset_report.json",
                "fix_dataset_summary.tsv",
            }:
                candidates.append(path)

    rows = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            rel = path.relative_to(run_dir)
        except Exception:
            rel = path
        rows.append(
            {
                "artifact": str(rel),
                "kind": "dir" if path.is_dir() else path.suffix.lstrip("."),
                "size_bytes": path.stat().st_size if path.is_file() else "",
            }
        )
    return pd.DataFrame(rows)


def generate_run_report(
    run_dir: Path,
    *,
    title: Optional[str] = None,
    output_name: str = "run_report.pdf",
    workflow_steps: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Optional[Path]:
    """Write a consolidated PDF report for one run directory."""
    try:
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        return None

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = run_dir / output_name
    settings_dir = _find_run_settings_dir(run_dir)
    command_path = settings_dir / "command.txt" if settings_dir else None
    config_path = settings_dir / "pipeline_config.resolved.toml" if settings_dir else None

    fix_df = _summarize_fix_reports(run_dir)
    qc_df = _summarize_qc(run_dir)
    artifact_df = _artifact_inventory(run_dir)
    workflow_df = (
        pd.DataFrame(list(workflow_steps or []))
        if workflow_steps
        else pd.DataFrame()
    )

    with PdfPages(pdf_path) as pdf:
        _draw_text_page(
            pdf,
            title or "PLEB Run Report",
            [
                (
                    "Summary",
                    [
                        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
                        f"Run directory: {run_dir}",
                        f"FixDataset branches summarized: {len(fix_df)}",
                        f"QC summary present: {bool(not qc_df.empty)}",
                        f"Workflow step records: {len(workflow_df)}",
                    ],
                ),
                (
                    "Command",
                    _read_lines(command_path, limit=12)
                    if command_path
                    else ["No command.txt found."],
                ),
            ],
        )

        if config_path and config_path.exists():
            _draw_text_page(
                pdf,
                "Resolved Config",
                [("pipeline_config.resolved.toml", _read_lines(config_path, limit=80))],
            )

        if not fix_df.empty:
            _draw_table_pages(pdf, "FixDataset Summary", fix_df)

        if not qc_df.empty:
            _draw_table_pages(pdf, "QC Summary", qc_df)

        if not workflow_df.empty:
            keep_cols = [
                c
                for c in ("step", "kind", "run_dir", "fix_summary", "qc_summary")
                if c in workflow_df.columns
            ]
            _draw_table_pages(pdf, "Workflow Steps", workflow_df[keep_cols])

        if not artifact_df.empty:
            _draw_table_pages(pdf, "Artifacts", artifact_df)

    return pdf_path
