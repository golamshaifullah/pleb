#!/usr/bin/env python3
"""Generate a reviewer-facing synthesis package for one pulsar workflow.

This is intentionally a read-only synthesis layer. It does not rerun science
or mutate the dataset repo. It inspects existing branches, run artifacts, and
QC CSVs, then writes a compact review package with:

- provenance and artifact manifests;
- dataset/TIM change summaries across workflow branches;
- post-fit residual review plots built from tempo2 ``general2`` outputs;
- concrete tables for surviving suspicious TOAs and backend concentrations;
- a human-readable Markdown index and decision sheet.

The QC plotting path reuses ``pleb.qc_review.load_qc_csv`` so post-fit residual
attachment follows the same deterministic join logic as the review UIs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table
import numpy as np
import pandas as pd

from pleb.qc_review import (
    apply_overrides,
    find_qc_csvs,
    load_overrides,
    load_qc_csv,
)


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    creates_branch: bool
    optional: bool
    default_branch_template: Optional[str]
    default_run_templates: tuple[str, ...]


@dataclass
class StageRuntime:
    spec: StageSpec
    branch: str
    branch_exists: bool
    commit: str
    run_dir: Optional[Path]
    run_dir_exists: bool
    notes: str


@dataclass(frozen=True)
class ResidualChoice:
    column: str | None
    error_column: str | None
    units: str
    n_numeric: int
    n_error_numeric: int
    available_residual_columns: tuple[str, ...]
    warning: str


STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        key="ingest",
        label="Stage 0 ingest",
        creates_branch=True,
        optional=False,
        default_branch_template="raw_ingest",
        default_run_templates=(),
    ),
    StageSpec(
        key="step1_fix",
        label="Step 1 fix dataset",
        creates_branch=True,
        optional=False,
        default_branch_template="{slug}_step1_fix",
        default_run_templates=("{slug}_step1_fix/{slug}_step1_fix",),
    ),
    StageSpec(
        key="step2_detect_variants",
        label="Step 2 detect variants",
        creates_branch=True,
        optional=False,
        default_branch_template="{slug}_step2_detect_variants",
        default_run_templates=(
            "{slug}_step2_detect_variants/{slug}_step2_detect_variants",
        ),
    ),
    StageSpec(
        key="step4_detect_selected",
        label="Step 4 detect selected",
        creates_branch=True,
        optional=False,
        default_branch_template="{slug}_step4_detect_selected",
        default_run_templates=(
            "{slug}_step4_detect_selected/{slug}_step4_detect_selected",
        ),
    ),
    StageSpec(
        key="step5_apply_comments",
        label="Step 5 apply comments",
        creates_branch=True,
        optional=False,
        default_branch_template="{slug}_step5_apply_comments",
        default_run_templates=(
            "{slug}_step5_apply_comments/{slug}_step5_apply_comments",
        ),
    ),
    StageSpec(
        key="step6_apply_delete",
        label="Step 6 apply delete",
        creates_branch=True,
        optional=False,
        default_branch_template="{slug}_step6_apply_delete",
        default_run_templates=("{slug}_step6_apply_delete/{slug}_step6_apply_delete",),
    ),
    StageSpec(
        key="step6_param_scan",
        label="Step 6 param scan",
        creates_branch=False,
        optional=True,
        default_branch_template=None,
        default_run_templates=(
            "{slug}_step6_apply_delete_param_scan/{slug}_step6_apply_delete",
        ),
    ),
    StageSpec(
        key="step7_whitenoise",
        label="Step 7 whitenoise",
        creates_branch=False,
        optional=True,
        default_branch_template=None,
        default_run_templates=(
            "{slug}_step7_whitenoise/{slug}_step6_apply_delete",
            "{slug}_step7_whitenoise/{final_branch}",
            "{slug}_step7_whitenoise/*",
        ),
    ),
    StageSpec(
        key="step8_compare_public",
        label="Step 8 compare public",
        creates_branch=False,
        optional=True,
        default_branch_template=None,
        default_run_templates=(
            "public_compare/{slug}",
            "public_compare/{psr}",
            "public_compare/*{slug}*",
        ),
    ),
)

PDF_TEXT_DOCS: tuple[tuple[str, str], ...] = (
    ("index.md", "Package overview"),
    ("00_decision/final_data_quality_decision_sheet.md", "Final decision sheet"),
    ("03_postfit_review/postfit_review.md", "Post-fit residual review"),
    ("10_model_checks/model_checks.md", "Model checks"),
    ("08_noise_model/whitenoise_not_found.md", "Whitenoise synthesis"),
)

PDF_TABLE_DOCS: tuple[tuple[str, str, tuple[str, ...], int], ...] = (
    (
        "03_postfit_review/variant_postfit_summary.tsv",
        "Variant post-fit summary",
        (
            "variant",
            "n_rows",
            "n_postfit",
            "n_bad_toa",
            "n_event",
            "n_review_event",
            "max_abs_sigma",
            "max_abs_sigma_keep",
            "worst_backend_by_flagged_fraction",
            "worst_backend_flagged_fraction",
        ),
        24,
    ),
    (
        "05_qc_and_outliers/backend_flag_summary.tsv",
        "Backend flag summary",
        (
            "variant",
            "backend",
            "n_rows",
            "n_flagged",
            "flagged_fraction",
            "median_abs_postfit_us",
            "max_abs_postfit_us",
        ),
        40,
    ),
    (
        "03_postfit_review/surviving_keep_outliers.tsv",
        "Surviving keep outliers",
        (
            "variant",
            "backend",
            "mjd",
            "freq_mhz",
            "postfit_us",
            "abs_sigma",
            "manual_action",
        ),
        32,
    ),
    (
        "09_public_comparison/public_synthesis.tsv",
        "Public comparison summary",
        ("metric", "value"),
        40,
    ),
    (
        "09_public_comparison/public_parameter_tension_table.tsv",
        "Public comparison tension rows",
        ("parameter", "sigma_tension", "provider_pair", "agreement_class"),
        40,
    ),
    (
        "10_model_checks/param_scan_summary.tsv",
        "Parameter scan summary",
        (
            "stage",
            "candidate",
            "redchisq",
            "lrt_delta_chisq",
            "lrt_p_value",
            "max_param_z",
        ),
        32,
    ),
    (
        "10_model_checks/change_report_model_summary.tsv",
        "Model comparison summary",
        (
            "stage",
            "reference",
            "delta_redchisq",
            "delta_wrms_post",
            "lrt_delta_chisq",
            "lrt_p_value",
        ),
        32,
    ),
)

PRIMARY_ARTIFACT_NAMES = {
    "binary_analysis.tsv",
    "ingest_report.pdf",
    "run_report.pdf",
    "workflow_report.pdf",
    "qc_compact_report.pdf",
    "qc_summary.tsv",
    "residual_summary.tsv",
    "fix_dataset_summary.tsv",
    "whitenoise_summary.tsv",
    "public_release_parameters.summary.tsv",
    "public_release_parameters.report.md",
    "public_release_parameters.comparison.tsv",
}

SKIP_TIM_DIRECTIVES = {
    "FORMAT",
    "MODE",
    "INCLUDE",
    "END",
    "EFAC",
    "EQUAD",
    "JUMP",
    "TIME",
    "SKIP",
    "NOSKIP",
}

POSTFIT_RESIDUAL_PREFERENCE = (
    "tempo2_post_us",
    "tempo2_postfit_us",
    "tempo2_post",
    "tempo2_postfit",
)

ALL_RESIDUAL_PREFERENCE = POSTFIT_RESIDUAL_PREFERENCE + (
    "tempo2_pre_us",
    "tempo2_pre",
    "postfit_us",
    "post_fit_us",
    "postfit",
    "post_fit",
    "post",
    "resid_us",
    "residual_us",
    "resid",
    "residual",
    "resid_detrended",
)

KEEP_DECISIONS = {"KEEP"}
FLAGGED_DECISIONS = {"BAD_TOA", "EVENT", "REVIEW_EVENT"}
DECISION_ORDER = ("KEEP", "BAD_TOA", "EVENT", "REVIEW_EVENT")
DECISION_COLORS = {
    "KEEP": "#9ca3af",
    "BAD_TOA": "#b91c1c",
    "EVENT": "#0f766e",
    "REVIEW_EVENT": "#b45309",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def clean_rel(path: Path, base: Optional[Path] = None) -> str:
    try:
        if base is not None:
            return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        pass
    return path.as_posix()


def mkdirs(root: Path) -> dict[str, Path]:
    subdirs = {
        "decision": root / "00_decision",
        "provenance": root / "01_provenance",
        "dataset": root / "02_dataset_state",
        "postfit": root / "03_postfit_review",
        "lineage": root / "04_branch_lineage",
        "qc": root / "05_qc_and_outliers",
        "toa": root / "06_toa_actions",
        "timing": root / "07_timing_fit_quality",
        "noise": root / "08_noise_model",
        "public": root / "09_public_comparison",
        "model_checks": root / "10_model_checks",
        "machine": root / "11_machine_readable",
        "raw_links": root / "raw_links",
    }
    root.mkdir(parents=True, exist_ok=True)
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)
    (subdirs["postfit"] / "plots").mkdir(parents=True, exist_ok=True)
    return subdirs


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_filename(value: str, *, default: str = "review_synthesis") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or default


def _normalize_markdown_line(raw: str) -> str:
    line = raw.rstrip()
    line = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"[image: \1]", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = line.replace("**", "").replace("__", "").replace("`", "")
    if line.lstrip().startswith("#"):
        line = line.lstrip("#").strip()
    return line


def _is_markdown_table_separator(line: str) -> bool:
    return bool(re.match(r"^\|\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$", line.strip()))


def _looks_like_markdown_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and "|" in stripped[1:]


def _split_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip().replace("\\|", "|") for cell in stripped.split("|")]


def _markdown_to_pdf_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = _normalize_markdown_line(raw)
        if _is_markdown_table_separator(line):
            continue
        lines.append(line)
    return lines


def _parse_markdown_pdf_blocks(text: str) -> list[dict[str, Any]]:
    raw_lines = [_normalize_markdown_line(raw) for raw in text.splitlines()]
    blocks: list[dict[str, Any]] = []
    text_lines: list[str] = []
    idx = 0
    while idx < len(raw_lines):
        line = raw_lines[idx]
        next_line = raw_lines[idx + 1] if idx + 1 < len(raw_lines) else ""
        if _looks_like_markdown_table_row(line) and _is_markdown_table_separator(
            next_line
        ):
            if text_lines:
                blocks.append({"kind": "text", "lines": text_lines})
                text_lines = []
            fields = _split_markdown_table_row(line)
            idx += 2
            row_dicts: list[dict[str, Any]] = []
            while idx < len(raw_lines) and _looks_like_markdown_table_row(
                raw_lines[idx]
            ):
                cells = _split_markdown_table_row(raw_lines[idx])
                padded = cells + [""] * max(0, len(fields) - len(cells))
                row_dicts.append(
                    {
                        field: padded[pos] if pos < len(padded) else ""
                        for pos, field in enumerate(fields)
                    }
                )
                idx += 1
            blocks.append({"kind": "table", "fields": tuple(fields), "rows": row_dicts})
            continue
        text_lines.append(line)
        idx += 1
    if text_lines:
        blocks.append({"kind": "text", "lines": text_lines})
    return blocks


def _wrap_pdf_lines(lines: Sequence[str], *, width: int = 110) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            wrapped.append("")
            continue
        if stripped.startswith(("- ", "* ")):
            body = stripped[2:].strip()
            wrapped.extend(
                textwrap.wrap(
                    body,
                    width=max(10, width - 2),
                    initial_indent="- ",
                    subsequent_indent="  ",
                )
                or ["-"]
            )
            continue
        wrapped.extend(textwrap.wrap(stripped, width=width) or [""])
    return wrapped


def _write_pdf_text_pages(
    pdf: PdfPages,
    *,
    title: str,
    lines: Sequence[str],
    source_label: str = "",
    page_height_lines: int = 52,
) -> None:
    wrapped = _wrap_pdf_lines(lines)
    if not wrapped:
        wrapped = ["No content."]
    chunks = [
        wrapped[i : i + page_height_lines]
        for i in range(0, len(wrapped), page_height_lines)
    ]
    for page_no, chunk in enumerate(chunks, start=1):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.05, 0.972, title, fontsize=15, fontweight="bold", va="top")
        if source_label:
            fig.text(0.05, 0.948, source_label, fontsize=8.5, color="#4b5563", va="top")
        fig.text(
            0.05,
            0.925,
            "\n".join(chunk),
            fontsize=8.7,
            family="DejaVu Sans Mono",
            va="top",
        )
        fig.text(
            0.95,
            0.02,
            f"Page {page_no}/{len(chunks)}",
            fontsize=8,
            ha="right",
            color="#6b7280",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _read_pdf_table_rows(
    path: Path, *, fields: Sequence[str], max_rows: int
) -> tuple[list[dict[str, Any]], str]:
    rows = read_delimited(path)
    if not rows:
        return [], f"No rows found in {path.name}."
    selected = [
        {field: row.get(field, "") for field in fields} for row in rows[:max_rows]
    ]
    if len(rows) > max_rows:
        note = f"Showing {len(selected)} of {len(rows)} rows from {path.name}."
    else:
        note = f"Rows shown: {len(selected)} from {path.name}."
    return selected, note


def _pdf_table_display_text(value: Any) -> str:
    return scalar(value).replace("\n", " ").strip()


def _wrap_pdf_table_cell(text: str, *, width: int) -> list[str]:
    content = text or ""
    wrapped = textwrap.wrap(
        content,
        width=max(4, width),
        break_long_words=True,
        break_on_hyphens=True,
    )
    return wrapped or [""]


def _compute_pdf_table_column_layout(
    fields: Sequence[str],
    rows: Sequence[dict[str, Any]],
    *,
    total_char_budget: int = 92,
) -> tuple[list[float], list[int]]:
    weights: list[int] = []
    for field in fields:
        observed = [len(str(field))]
        for row in rows[:40]:
            observed.append(min(len(_pdf_table_display_text(row.get(field, ""))), 60))
        weights.append(min(max(max(observed), 8), 24))
    total_weight = max(sum(weights), 1)
    char_widths = [max(6, int(total_char_budget * w / total_weight)) for w in weights]
    current_total = sum(char_widths)
    if current_total < total_char_budget:
        char_widths[-1] += total_char_budget - current_total
    elif current_total > total_char_budget:
        char_widths[-1] = max(6, char_widths[-1] - (current_total - total_char_budget))
    total_chars = max(sum(char_widths), 1)
    col_fracs = [width / total_chars for width in char_widths]
    return col_fracs, char_widths


def _paginate_pdf_table_rows(
    fields: Sequence[str],
    rows: Sequence[dict[str, Any]],
    *,
    char_widths: Sequence[int],
    max_row_units: float = 34.0,
) -> tuple[list[list[list[str]]], list[list[float]], list[list[str]], float]:
    wrapped_header = [
        _wrap_pdf_table_cell(str(field), width=char_widths[idx])
        for idx, field in enumerate(fields)
    ]
    header_units = max(len(cell) for cell in wrapped_header) + 0.5
    wrapped_rows = [
        [
            "\n".join(
                _wrap_pdf_table_cell(
                    _pdf_table_display_text(row.get(field, "")),
                    width=char_widths[idx],
                )
            )
            for idx, field in enumerate(fields)
        ]
        for row in rows
    ]
    row_units = [
        max(cell.count("\n") + 1 for cell in wrapped_row) + 0.35
        for wrapped_row in wrapped_rows
    ]
    pages: list[list[list[str]]] = []
    page_units: list[list[float]] = []
    cur_rows: list[list[str]] = []
    cur_units: list[float] = []
    used = header_units
    for wrapped_row, units in zip(wrapped_rows, row_units):
        if cur_rows and used + units > max_row_units:
            pages.append(cur_rows)
            page_units.append(cur_units)
            cur_rows = []
            cur_units = []
            used = header_units
        cur_rows.append(wrapped_row)
        cur_units.append(units)
        used += units
    if cur_rows or not pages:
        pages.append(cur_rows)
        page_units.append(cur_units)
    return pages, page_units, wrapped_header, header_units


def _write_pdf_table_pages(
    pdf: PdfPages,
    *,
    title: str,
    fields: Sequence[str],
    rows: Sequence[dict[str, Any]],
    source_label: str = "",
    footer_note: str = "",
) -> None:
    if not rows:
        _write_pdf_text_pages(
            pdf,
            title=title,
            lines=[footer_note or "No rows."],
            source_label=source_label,
        )
        return
    col_fracs, char_widths = _compute_pdf_table_column_layout(fields, rows)
    pages, page_units, wrapped_header, header_units = _paginate_pdf_table_rows(
        fields, rows, char_widths=char_widths
    )
    total_pages = len(pages)
    for page_idx, (page_rows, row_units) in enumerate(zip(pages, page_units), start=1):
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        fig.text(0.05, 0.972, title, fontsize=15, fontweight="bold", va="top")
        if source_label:
            fig.text(0.05, 0.948, source_label, fontsize=8.5, color="#4b5563", va="top")
        table_top = 0.915 if source_label else 0.935
        table_bottom = 0.08
        table_bbox = [0.05, table_bottom, 0.90, table_top - table_bottom]
        table = Table(ax, bbox=table_bbox)
        total_units = header_units + sum(row_units) if page_rows else header_units
        header_height = header_units / total_units
        for col_idx, header_lines in enumerate(wrapped_header):
            cell = table.add_cell(
                0,
                col_idx,
                width=col_fracs[col_idx],
                height=header_height,
                text="\n".join(header_lines),
                loc="left",
                facecolor="#e5e7eb",
                edgecolor="#9ca3af",
            )
            cell.PAD = 0.03
            cell.get_text().set_fontsize(7.1)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("left")
            cell.get_text().set_va("center")
        for row_idx, wrapped_row in enumerate(page_rows, start=1):
            row_height = row_units[row_idx - 1] / total_units
            for col_idx, cell_text in enumerate(wrapped_row):
                cell = table.add_cell(
                    row_idx,
                    col_idx,
                    width=col_fracs[col_idx],
                    height=row_height,
                    text=cell_text,
                    loc="left",
                    facecolor="white",
                    edgecolor="#d1d5db",
                )
                cell.PAD = 0.03
                cell.get_text().set_fontsize(6.8)
                cell.get_text().set_ha("left")
                cell.get_text().set_va("center")
        ax.add_table(table)
        if footer_note:
            fig.text(0.05, 0.038, footer_note, fontsize=8, color="#4b5563", va="bottom")
        fig.text(
            0.95,
            0.02,
            f"Page {page_idx}/{total_pages}",
            fontsize=8,
            ha="right",
            color="#6b7280",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _write_pdf_image_page(pdf: PdfPages, *, image_path: Path, title: str) -> None:
    image = plt.imread(image_path)
    if image.ndim >= 2:
        height, width = image.shape[:2]
    else:
        height, width = 1, max(1, image.shape[0])
    figsize = (11.69, 8.27) if width >= height else (8.27, 11.69)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.text(0.5, 0.02, image_path.as_posix(), ha="center", fontsize=8, color="#4b5563")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf_with_matplotlib(
    *,
    args: argparse.Namespace,
    out_root: Path,
    decision: str,
) -> Path:
    """Build a single reviewer PDF from generated review-package artifacts."""
    pdf_name = (
        getattr(args, "pdf_name", "")
        or f"{safe_filename(args.psr)}_review_synthesis.pdf"
    )
    dest = out_root / pdf_name
    with PdfPages(dest) as pdf:
        meta = pdf.infodict()
        meta["Title"] = f"Review synthesis package: {args.psr}"
        meta["Author"] = "PLEB workflow synthesis"
        meta["Subject"] = "Reviewer-facing synthesis package"
        meta["Keywords"] = "pleb review synthesis pulsar timing"
        meta["CreationDate"] = dt.datetime.now()

        overview_lines = [
            f"Pulsar: {args.psr}",
            f"Generated: {now_iso()}",
            f"Automatic synthesis decision: {decision}",
        ]
        _write_pdf_text_pages(
            pdf, title=f"Review synthesis package: {args.psr}", lines=overview_lines
        )

        for rel_path, title in PDF_TEXT_DOCS:
            doc_path = out_root / rel_path
            if not doc_path.exists():
                continue
            text = doc_path.read_text(encoding="utf-8", errors="replace")
            blocks = _parse_markdown_pdf_blocks(text)
            for block in blocks:
                if block["kind"] == "text":
                    _write_pdf_text_pages(
                        pdf,
                        title=title,
                        lines=block["lines"],
                        source_label=doc_path.relative_to(out_root).as_posix(),
                    )
                    continue
                _write_pdf_table_pages(
                    pdf,
                    title=title,
                    fields=block["fields"],
                    rows=block["rows"],
                    source_label=doc_path.relative_to(out_root).as_posix(),
                )

        for rel_path, title, fields, max_rows in PDF_TABLE_DOCS:
            table_path = out_root / rel_path
            if not table_path.exists():
                continue
            rows, footer_note = _read_pdf_table_rows(
                table_path, fields=fields, max_rows=max_rows
            )
            _write_pdf_table_pages(
                pdf,
                title=title,
                fields=fields,
                rows=rows,
                source_label=table_path.relative_to(out_root).as_posix(),
                footer_note=footer_note,
            )

        plots_dir = out_root / "03_postfit_review" / "plots"
        for image_path in sorted(plots_dir.glob("*.png")):
            plot_title = image_path.stem.replace("_", " ")
            _write_pdf_image_page(
                pdf, image_path=image_path, title=f"Post-fit review plot: {plot_title}"
            )

    if not dest.exists() or dest.stat().st_size <= 0:
        raise RuntimeError(
            f"Native PDF build completed but no PDF was written to {dest}."
        )
    return dest


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: scalar(row.get(k, "")) for k in fields})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha1_text(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:length]


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def safe_float(value: Any) -> Optional[float]:
    try:
        s = str(value).strip()
        if not s or s.upper() in {"NA", "NAN", "NONE", "NULL"}:
            return None
        return float(s)
    except Exception:
        return None


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def choose_col(
    columns: Iterable[str], candidates: Iterable[str], contains: Iterable[str] = ()
) -> Optional[str]:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        c = lower.get(cand.lower())
        if c:
            return c
    for needle in contains:
        needle_l = needle.lower()
        for c in cols:
            if needle_l in c.lower():
                return c
    return None


def read_delimited(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.is_file():
        return []
    try:
        sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
    except Exception:
        return []
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        delimiter = dialect.delimiter
    except Exception:
        pass
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            return list(csv.DictReader(f, delimiter=delimiter))
    except Exception:
        return []


def row_count(path: Path) -> str:
    if path.suffix.lower() not in {".tsv", ".csv", ".txt", ".md"}:
        return ""
    try:
        if path.suffix.lower() in {".tsv", ".csv"}:
            n = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
            return str(max(0, n - 1))
        return str(sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")))
    except Exception:
        return ""


def parse_key_value_args(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"Expected KEY=VALUE argument, got: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def run_cmd(
    args: list[str], cwd: Path, text: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        check=False,
    )


def git_available(repo_root: Path) -> bool:
    cp = run_cmd(["git", "rev-parse", "--is-inside-work-tree"], repo_root)
    return cp.returncode == 0 and cp.stdout.strip() == "true"


def git_text(repo_root: Path, args: list[str]) -> Optional[str]:
    cp = run_cmd(["git", *args], repo_root)
    if cp.returncode != 0:
        return None
    return cp.stdout


def git_bytes(repo_root: Path, args: list[str]) -> Optional[bytes]:
    cp = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        check=False,
    )
    if cp.returncode != 0:
        return None
    return cp.stdout


def git_commit(repo_root: Path, ref: str) -> str:
    return (
        git_text(repo_root, ["rev-parse", "--verify", ref]) or ""
    ).strip() or "UNKNOWN"


def git_ref_exists(repo_root: Path, ref: str) -> bool:
    return bool((git_text(repo_root, ["rev-parse", "--verify", ref]) or "").strip())


def git_is_dirty(repo_root: Path) -> str:
    status = git_text(repo_root, ["status", "--porcelain"])
    if status is None:
        return "UNKNOWN"
    return "dirty" if status.strip() else "clean"


def git_ls_files_at_ref(repo_root: Path, ref: str, prefix: str) -> list[str]:
    if not ref or ref == "UNKNOWN":
        return []
    out = git_text(repo_root, ["ls-tree", "-r", "--name-only", ref, "--", prefix])
    if out is None:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def git_show_file(repo_root: Path, ref: str, path_in_repo: str) -> Optional[bytes]:
    if not ref or ref == "UNKNOWN":
        return None
    return git_bytes(repo_root, ["show", f"{ref}:{path_in_repo}"])


def path_in_repo(repo_root: Path, path: Path) -> Optional[str]:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return None


def format_template(template: str, *, slug: str, psr: str, final_branch: str) -> str:
    return template.format(slug=slug, psr=psr, final_branch=final_branch)


def git_path_has_files_at_ref(repo_root: Path, ref: str, path: Path) -> bool:
    rel = path_in_repo(repo_root, path)
    if not rel:
        return False
    return bool(git_ls_files_at_ref(repo_root, ref, rel))


def discover_run_dir(
    results_root: Path,
    spec: StageSpec,
    *,
    slug: str,
    psr: str,
    final_branch: str,
    override: Optional[str],
) -> tuple[Optional[Path], str]:
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = results_root / p
        return p, "override"
    candidates: list[Path] = []
    for template in spec.default_run_templates:
        pattern = format_template(
            template, slug=slug, psr=psr, final_branch=final_branch
        )
        if "*" in pattern:
            candidates.extend(
                sorted(p for p in results_root.glob(pattern) if p.is_dir())
            )
        else:
            candidates.append(results_root / pattern)
    for p in candidates:
        if p.exists() and p.is_dir():
            return p, "discovered"
    if candidates:
        return candidates[0], "expected_missing"
    return None, "not_applicable"


def build_stage_runtimes(
    args: argparse.Namespace, repo_root: Path, results_root: Path
) -> list[StageRuntime]:
    branch_overrides = parse_key_value_args(args.stage_branch)
    run_overrides = parse_key_value_args(args.stage_run)
    runtimes: list[StageRuntime] = []
    for spec in STAGES:
        if spec.default_branch_template:
            default_branch = format_template(
                spec.default_branch_template,
                slug=args.slug,
                psr=args.psr,
                final_branch=args.final_branch,
            )
        else:
            default_branch = ""
        branch = branch_overrides.get(spec.key, default_branch)
        if spec.key == "step6_apply_delete" and args.final_branch:
            branch = branch_overrides.get(spec.key, args.final_branch)
        branch_exists = bool(branch) and git_ref_exists(repo_root, branch)
        commit = git_commit(repo_root, branch) if branch_exists else "UNKNOWN"
        run_dir, note = discover_run_dir(
            results_root,
            spec,
            slug=args.slug,
            psr=args.psr,
            final_branch=args.final_branch,
            override=run_overrides.get(spec.key),
        )
        run_dir_exists = bool(run_dir and run_dir.exists() and run_dir.is_dir())
        if not run_dir_exists and run_dir and branch_exists:
            if git_path_has_files_at_ref(repo_root, branch, run_dir):
                run_dir_exists = True
                note = "branch_only"
        runtimes.append(
            StageRuntime(
                spec=spec,
                branch=branch,
                branch_exists=branch_exists,
                commit=commit,
                run_dir=run_dir,
                run_dir_exists=run_dir_exists,
                notes=note,
            )
        )
    return runtimes


def artifact_priority(path: Path) -> str:
    if path.name in PRIMARY_ARTIFACT_NAMES:
        return "primary"
    if "diagnostics" in path.parts or path.suffix.lower() in {
        ".log",
        ".covmat",
        ".general2",
    }:
        return "forensic"
    if path.suffix.lower() in {".png", ".pdf", ".tsv", ".csv", ".json", ".md"}:
        return "secondary"
    return "forensic"


def artifact_type(path: Path) -> str:
    name = path.name
    if name == "binary_analysis.tsv":
        return "binary_analysis"
    if name == "run_report.pdf":
        return "run_report"
    if name == "workflow_report.pdf":
        return "workflow_report"
    if name == "ingest_report.pdf":
        return "ingest_report"
    if name == "qc_compact_report.pdf":
        return "qc_report"
    if name == "qc_summary.tsv":
        return "qc_summary"
    if name == "residual_summary.tsv":
        return "residual_summary"
    if "Outliers" in name:
        return "outlier_summary"
    if name.startswith("public_release_parameters"):
        return "public_compare"
    if name.startswith("MODEL_COMPARISON_"):
        return "change_report_model"
    if name.startswith("NEW_PARAM_SIGNIFICANCE_"):
        return "change_report_new_params"
    if name.startswith("param_scan_") and path.suffix.lower() == ".tsv":
        return "param_scan"
    if "_change_" in name and path.suffix.lower() == ".tsv":
        return "change_report_detail"
    if name.endswith("fix_dataset_summary.tsv"):
        return "fix_dataset_summary"
    return path.suffix.lower().lstrip(".") or "file"


def add_artifact_rows(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    category: str,
    base: Path,
    patterns: Iterable[str],
    repo_root: Path,
) -> None:
    seen: set[Path] = set()
    for pattern in patterns:
        for p in sorted(base.glob(pattern)):
            if not p.is_file() or p in seen:
                continue
            seen.add(p)
            rows.append(
                {
                    "stage": stage,
                    "category": category,
                    "artifact_type": artifact_type(p),
                    "priority": artifact_priority(p),
                    "path": clean_rel(p, repo_root),
                    "exists": "yes",
                    "size_bytes": p.stat().st_size,
                    "row_count": row_count(p),
                }
            )


def row_count_from_bytes(path: Path, data: bytes) -> str:
    suffix = path.suffix.lower()
    if suffix not in {".tsv", ".csv", ".txt", ".md"}:
        return ""
    text = data.decode("utf-8", errors="replace")
    n = text.count("\n")
    if text and not text.endswith("\n"):
        n += 1
    if suffix in {".tsv", ".csv"}:
        n = max(0, n - 1)
    return str(n)


def add_branch_artifact_rows(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    branch: str,
    base: Path,
    patterns: Iterable[str],
    repo_root: Path,
) -> None:
    base_rel = path_in_repo(repo_root, base)
    if not base_rel:
        return
    files = git_ls_files_at_ref(repo_root, branch, base_rel)
    if not files:
        return
    seen: set[str] = set()
    for rel_path in sorted(files):
        if rel_path in seen:
            continue
        p = Path(rel_path)
        if not any(p.match(f"{base_rel}/{pattern}") for pattern in patterns):
            continue
        data = git_show_file(repo_root, branch, rel_path)
        if data is None:
            continue
        seen.add(rel_path)
        rows.append(
            {
                "stage": stage,
                "category": "run_artifact",
                "artifact_type": artifact_type(p),
                "priority": artifact_priority(p),
                "path": rel_path,
                "exists": "yes",
                "size_bytes": len(data),
                "row_count": row_count_from_bytes(p, data),
            }
        )


def discover_artifacts(
    runtimes: list[StageRuntime], dataset_root: Path, repo_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ingest_dir = dataset_root / "ingest_reports"
    if ingest_dir.exists():
        add_artifact_rows(
            rows,
            stage="ingest",
            category="ingest",
            base=ingest_dir,
            patterns=("*", "summary/*", "diagnostics/*"),
            repo_root=repo_root,
        )
    run_patterns = (
        "run_report.pdf",
        "workflow_report.pdf",
        "residual_summary.tsv",
        "*_summary.tsv",
        "*_all_summary.tsv",
        "binary_analysis/**/*.tsv",
        "change_report/**/*.tsv",
        "plk/*.log",
        "covmat/*",
        "general2/*",
        "param_scan/**/*.tsv",
        "png/*.png",
        "fix_dataset/**/*.json",
        "fix_dataset/**/*.pdf",
        "fix_dataset/**/*.tsv",
        "qc/qc_summary.tsv",
        "qc/**/*.csv",
        "qc_report/**/*.pdf",
        "qc_report/**/*.tsv",
        "qc_report/**/*.csv",
        "qc_report/**/*.txt",
        "qc_report/**/*.png",
        "OutlierSummary/**/*.tsv",
        "whitenoise/**/*.tsv",
        "run_settings/**/*",
        "optimize_bad_masks/**/*.csv",
    )
    public_patterns = (
        "public_release_parameters.*",
        "resolved_assets.tsv",
        "downloads/**/*",
        "run_settings/**/*",
        "workflow_report.pdf",
    )
    for rt in runtimes:
        if not rt.run_dir_exists or not rt.run_dir:
            continue
        patterns = (
            public_patterns if rt.spec.key == "step8_compare_public" else run_patterns
        )
        if rt.run_dir.exists() and rt.run_dir.is_dir():
            add_artifact_rows(
                rows,
                stage=rt.spec.key,
                category="run_artifact",
                base=rt.run_dir,
                patterns=patterns,
                repo_root=repo_root,
            )
        elif rt.branch_exists:
            add_branch_artifact_rows(
                rows,
                stage=rt.spec.key,
                branch=rt.branch,
                base=rt.run_dir,
                patterns=patterns,
                repo_root=repo_root,
            )
    return rows


def make_raw_links(
    runtimes: list[StageRuntime],
    dataset_root: Path,
    out_raw_links: Path,
    repo_root: Path,
) -> None:
    def link_or_note(name: str, target: Path) -> None:
        link = out_raw_links / name
        if link.exists() or link.is_symlink():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        try:
            rel = os.path.relpath(target.resolve(), start=link.parent.resolve())
            link.symlink_to(rel, target_is_directory=target.is_dir())
        except Exception:
            write_text(
                link.with_suffix(".path.txt"), clean_rel(target, repo_root) + "\n"
            )

    ingest_dir = dataset_root / "ingest_reports"
    if ingest_dir.exists():
        link_or_note("ingest_reports", ingest_dir)
    for rt in runtimes:
        if rt.run_dir_exists and rt.run_dir:
            if rt.run_dir.exists() and rt.run_dir.is_dir():
                link_or_note(rt.spec.key, rt.run_dir)
            else:
                write_text(
                    (out_raw_links / f"{rt.spec.key}.path.txt"),
                    f"branch={rt.branch or 'NA'}\npath={clean_rel(rt.run_dir, repo_root)}\n",
                )


@dataclass
class TimLine:
    file_path: str
    raw: str
    normalized: str
    active: bool
    commented: bool
    toa_id: str
    freq: str
    mjd: str
    err: str
    observatory: str
    backend: str
    telescope: str
    system: str


def strip_tim_comment(line: str) -> tuple[str, bool]:
    s = line.strip()
    if s.startswith("C "):
        return s[2:].strip(), True
    if s.startswith("C\t"):
        return s[1:].strip(), True
    return s, False


def normalize_tim_line(line: str) -> str:
    s, _ = strip_tim_comment(line)
    return re.sub(r"\s+", " ", s.strip())


def extract_flag(tokens: list[str], names: Iterable[str]) -> str:
    name_set = {n.lower() for n in names}
    for i, tok in enumerate(tokens[:-1]):
        if tok.lower() in name_set:
            return tokens[i + 1]
    return ""


def looks_like_toa(tokens: list[str]) -> bool:
    if len(tokens) < 4:
        return False
    if tokens[0].upper() in SKIP_TIM_DIRECTIVES:
        return False
    numeric_positions = sum(1 for x in tokens[:5] if is_number(x))
    if numeric_positions < 2:
        return False
    for x in tokens[:6]:
        v = safe_float(x)
        if v is not None and 30000 <= v <= 90000:
            return True
    return False


def parse_tim_line(file_path: str, line: str) -> Optional[TimLine]:
    stripped = line.strip()
    if not stripped:
        return None
    normalized = normalize_tim_line(line)
    if not normalized or normalized.startswith("#"):
        return None
    tokens = normalized.split()
    if not looks_like_toa(tokens):
        return None
    _, commented = strip_tim_comment(line)
    freq = tokens[1] if len(tokens) > 1 else ""
    mjd = tokens[2] if len(tokens) > 2 else ""
    err = tokens[3] if len(tokens) > 3 else ""
    observatory = tokens[4] if len(tokens) > 4 else ""
    if safe_float(mjd) is None or not (
        30000 <= float(mjd) <= 90000 if is_number(mjd) else False
    ):
        for i, tok in enumerate(tokens[:6]):
            v = safe_float(tok)
            if v is not None and 30000 <= v <= 90000:
                mjd = tok
                freq = tokens[i - 1] if i - 1 >= 0 else freq
                err = tokens[i + 1] if i + 1 < len(tokens) else err
                observatory = tokens[i + 2] if i + 2 < len(tokens) else observatory
                break
    backend = extract_flag(tokens, ("-be", "-backend", "-f", "-fe", "-receiver"))
    telescope = (
        extract_flag(tokens, ("-tel", "-telescope", "-obs", "-observatory"))
        or observatory
    )
    system = extract_flag(tokens, ("-sys", "-system", "-group", "-pta", "-name"))
    if not system:
        if telescope and backend:
            system = f"{telescope}/{backend}"
        else:
            system = telescope or backend or observatory or "UNKNOWN"
    return TimLine(
        file_path=file_path,
        raw=line.rstrip("\n"),
        normalized=normalized,
        active=not commented,
        commented=commented,
        toa_id=sha1_text(normalized),
        freq=freq,
        mjd=mjd,
        err=err,
        observatory=observatory,
        backend=backend,
        telescope=telescope,
        system=system,
    )


def branch_file_bytes(
    repo_root: Path, dataset_root: Path, branch: str, path_in_repo_: str
) -> Optional[bytes]:
    data = git_show_file(repo_root, branch, path_in_repo_)
    if data is not None:
        return data
    fallback = repo_root / path_in_repo_
    if fallback.exists() and fallback.is_file():
        try:
            return fallback.read_bytes()
        except Exception:
            return None
    fallback = dataset_root / path_in_repo_
    if fallback.exists() and fallback.is_file():
        try:
            return fallback.read_bytes()
        except Exception:
            return None
    return None


def list_dataset_files(
    repo_root: Path, dataset_root: Path, psr: str, branch: str
) -> list[str]:
    prefix = path_in_repo(repo_root, dataset_root / psr)
    if prefix:
        files = git_ls_files_at_ref(repo_root, branch, prefix)
        if files:
            return files
    fs_psr = dataset_root / psr
    if fs_psr.exists():
        out: list[str] = []
        for p in sorted(fs_psr.rglob("*")):
            if p.is_file():
                pir = path_in_repo(repo_root, p)
                out.append(pir or p.as_posix())
        return out
    return []


def collect_dataset_inventory(
    repo_root: Path, dataset_root: Path, psr: str, final_branch: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = list_dataset_files(repo_root, dataset_root, psr, final_branch)
    for f in files:
        data = branch_file_bytes(repo_root, dataset_root, final_branch, f)
        if data is None:
            continue
        rows.append(
            {
                "branch": final_branch,
                "path": f,
                "file_name": Path(f).name,
                "suffix": Path(f).suffix.lower(),
                "size_bytes": len(data),
                "line_count": data.count(b"\n")
                + (1 if data and not data.endswith(b"\n") else 0),
                "sha256": sha256_bytes(data),
            }
        )
    return rows


def read_tim_snapshot(
    repo_root: Path, dataset_root: Path, psr: str, branch: str
) -> dict[str, list[TimLine]]:
    snapshot: dict[str, list[TimLine]] = {}
    files = [
        f
        for f in list_dataset_files(repo_root, dataset_root, psr, branch)
        if f.lower().endswith(".tim")
    ]
    for f in files:
        data = branch_file_bytes(repo_root, dataset_root, branch, f)
        if data is None:
            continue
        lines: list[TimLine] = []
        text = data.decode("utf-8", errors="replace")
        for raw_line in text.splitlines():
            parsed = parse_tim_line(f, raw_line)
            if parsed:
                lines.append(parsed)
        snapshot[f] = lines
    return snapshot


def snapshot_counts(snapshot: dict[str, list[TimLine]]) -> dict[str, int]:
    active = sum(1 for lines in snapshot.values() for x in lines if x.active)
    commented = sum(1 for lines in snapshot.values() for x in lines if x.commented)
    return {
        "tim_files": len(snapshot),
        "active_toas": active,
        "commented_toas": commented,
        "total_toa_like_lines": active + commented,
    }


def representative_line(lines: list[TimLine]) -> TimLine:
    active = [x for x in lines if x.active]
    return active[0] if active else lines[0]


def diff_tim_snapshots(
    before: dict[str, list[TimLine]],
    after: dict[str, list[TimLine]],
    *,
    transition: str,
    from_branch: str,
    to_branch: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted(set(before) | set(after))
    for f in files:
        b_lines = before.get(f, [])
        a_lines = after.get(f, [])
        b_by_norm: dict[str, list[TimLine]] = defaultdict(list)
        a_by_norm: dict[str, list[TimLine]] = defaultdict(list)
        for x in b_lines:
            b_by_norm[x.normalized].append(x)
        for x in a_lines:
            a_by_norm[x.normalized].append(x)
        norms = sorted(set(b_by_norm) | set(a_by_norm))
        for norm in norms:
            b = b_by_norm.get(norm, [])
            a = a_by_norm.get(norm, [])
            b_active = sum(1 for x in b if x.active)
            b_comment = sum(1 for x in b if x.commented)
            a_active = sum(1 for x in a if x.active)
            a_comment = sum(1 for x in a if x.commented)
            active_lost = max(0, b_active - a_active)
            comment_gained = max(0, a_comment - b_comment)
            n_commented = min(active_lost, comment_gained)
            n_removed = max(0, (b_active + b_comment) - (a_active + a_comment))
            rep = representative_line(b or a)
            evidence = f"{transition}:{f}"
            for _ in range(n_commented):
                rows.append(
                    {
                        "toa_id": rep.toa_id,
                        "transition": transition,
                        "from_branch": from_branch,
                        "to_branch": to_branch,
                        "relative_tim_file": f,
                        "mjd": rep.mjd,
                        "freq": rep.freq,
                        "observatory": rep.observatory,
                        "system": rep.system,
                        "backend": rep.backend,
                        "telescope": rep.telescope,
                        "action": "commented",
                        "final_state": "present_commented",
                        "normalized_toa_line_sha1": rep.toa_id,
                        "evidence": evidence,
                    }
                )
            for _ in range(n_removed):
                rows.append(
                    {
                        "toa_id": rep.toa_id,
                        "transition": transition,
                        "from_branch": from_branch,
                        "to_branch": to_branch,
                        "relative_tim_file": f,
                        "mjd": rep.mjd,
                        "freq": rep.freq,
                        "observatory": rep.observatory,
                        "system": rep.system,
                        "backend": rep.backend,
                        "telescope": rep.telescope,
                        "action": "deleted",
                        "final_state": "removed",
                        "normalized_toa_line_sha1": rep.toa_id,
                        "evidence": evidence,
                    }
                )
    return rows


def branch_file_hashes(
    repo_root: Path, dataset_root: Path, psr: str, branch: str, suffix: str
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for f in list_dataset_files(repo_root, dataset_root, psr, branch):
        if not f.lower().endswith(suffix.lower()):
            continue
        data = branch_file_bytes(repo_root, dataset_root, branch, f)
        if data is None:
            continue
        key = f.split(f"/{psr}/", 1)[-1]
        hashes[key] = sha256_bytes(data)
    return hashes


def changed_file_count(a: dict[str, str], b: dict[str, str]) -> int:
    keys = set(a) | set(b)
    return sum(1 for k in keys if a.get(k) != b.get(k))


def collect_branch_diffs(
    repo_root: Path,
    dataset_root: Path,
    psr: str,
    runtimes: list[StageRuntime],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, list[TimLine]]]
]:
    branches = {rt.spec.key: rt.branch for rt in runtimes if rt.branch}
    transitions = [
        ("ingest_to_step1", "ingest", "step1_fix"),
        ("step1_to_step2", "step1_fix", "step2_detect_variants"),
        ("step2_to_step4", "step2_detect_variants", "step4_detect_selected"),
        ("step4_to_step5", "step4_detect_selected", "step5_apply_comments"),
        ("step5_to_step6", "step5_apply_comments", "step6_apply_delete"),
    ]
    snapshots: dict[str, dict[str, list[TimLine]]] = {}
    for key, branch in branches.items():
        if branch and git_ref_exists(repo_root, branch):
            snapshots[key] = read_tim_snapshot(repo_root, dataset_root, psr, branch)
    diff_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for transition, before_key, after_key in transitions:
        before_branch = branches.get(before_key, "")
        after_branch = branches.get(after_key, "")
        before_snapshot = snapshots.get(before_key, {})
        after_snapshot = snapshots.get(after_key, {})
        before_counts = snapshot_counts(before_snapshot)
        after_counts = snapshot_counts(after_snapshot)
        actions = (
            diff_tim_snapshots(
                before_snapshot,
                after_snapshot,
                transition=transition,
                from_branch=before_branch,
                to_branch=after_branch,
            )
            if before_snapshot or after_snapshot
            else []
        )
        action_rows.extend(actions)
        par_before = (
            branch_file_hashes(repo_root, dataset_root, psr, before_branch, ".par")
            if before_branch
            else {}
        )
        par_after = (
            branch_file_hashes(repo_root, dataset_root, psr, after_branch, ".par")
            if after_branch
            else {}
        )
        tim_before = (
            branch_file_hashes(repo_root, dataset_root, psr, before_branch, ".tim")
            if before_branch
            else {}
        )
        tim_after = (
            branch_file_hashes(repo_root, dataset_root, psr, after_branch, ".tim")
            if after_branch
            else {}
        )
        diff_rows.append(
            {
                "transition": transition,
                "from_branch": before_branch or "NA",
                "to_branch": after_branch or "NA",
                "from_active_toas": before_counts["active_toas"],
                "to_active_toas": after_counts["active_toas"],
                "from_commented_toas": before_counts["commented_toas"],
                "to_commented_toas": after_counts["commented_toas"],
                "active_toa_delta": after_counts["active_toas"]
                - before_counts["active_toas"],
                "comments_added_detected": sum(
                    1 for r in actions if r["action"] == "commented"
                ),
                "deletions_detected": sum(
                    1 for r in actions if r["action"] == "deleted"
                ),
                "par_files_changed": changed_file_count(par_before, par_after),
                "tim_files_changed": changed_file_count(tim_before, tim_after),
                "status": "ok" if before_branch and after_branch else "missing_branch",
            }
        )
    return diff_rows, action_rows, snapshots


def collect_qc_index(artifact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a in artifact_rows:
        if (
            a.get("artifact_type") in {"qc_summary", "outlier_summary"}
            or "qc" in str(a.get("path", "")).lower()
        ):
            rows.append(
                {
                    "stage": a.get("stage", ""),
                    "artifact_type": a.get("artifact_type", ""),
                    "priority": a.get("priority", ""),
                    "path": a.get("path", ""),
                    "row_count": a.get("row_count", ""),
                }
            )
    return rows


def collect_system_quality_matrix(
    final_snapshot: dict[str, list[TimLine]], action_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    systems: dict[str, Counter] = defaultdict(Counter)
    for lines in final_snapshot.values():
        for x in lines:
            systems[x.system][
                "final_active_toas" if x.active else "final_commented_toas"
            ] += 1
    for r in action_rows:
        sysname = r.get("system") or "UNKNOWN"
        if r.get("action") == "commented":
            systems[sysname]["commented_actions"] += 1
        elif r.get("action") == "deleted":
            systems[sysname]["deleted_actions"] += 1
    rows: list[dict[str, Any]] = []
    for sysname, c in sorted(systems.items()):
        denominator = c["final_active_toas"] + c["deleted_actions"]
        frac = (c["deleted_actions"] / denominator) if denominator else 0.0
        status = (
            "red"
            if c["deleted_actions"] >= 5 and frac >= 0.20
            else "amber" if c["deleted_actions"] > 0 else "green"
        )
        rows.append(
            {
                "system": sysname,
                "final_active_toas": c["final_active_toas"],
                "final_commented_toas": c["final_commented_toas"],
                "commented_actions": c["commented_actions"],
                "deleted_actions": c["deleted_actions"],
                "deletion_fraction_vs_final_plus_deleted": f"{frac:.6g}",
                "auto_status": status,
            }
        )
    return rows


def collect_residual_synthesis(
    runtimes: list[StageRuntime], repo_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rt in runtimes:
        if not rt.run_dir_exists or not rt.run_dir:
            continue
        path = rt.run_dir / "residual_summary.tsv"
        if not path.exists():
            continue
        data = read_delimited(path)
        for i, row in enumerate(data):
            cols = row.keys()
            n_col = choose_col(cols, ("n_toas", "ntoa", "n", "num_toas"), ("toa",))
            rms_col = choose_col(
                cols, ("rms_residual", "rms", "wrms", "weighted_rms"), ("rms",)
            )
            chi_col = choose_col(
                cols, ("reduced_chi2", "red_chi2", "chisq", "chi2"), ("chi",)
            )
            status_col = choose_col(cols, ("fit_status", "status"), ("status",))
            variant_col = choose_col(
                cols, ("variant", "dataset_variant", "kind"), ("variant",)
            )
            rows.append(
                {
                    "stage": rt.spec.key,
                    "branch": rt.branch or "NA",
                    "source_path": clean_rel(path, repo_root),
                    "row_index": i,
                    "variant": row.get(variant_col, "") if variant_col else "",
                    "n_toas": row.get(n_col, "") if n_col else "",
                    "rms_or_wrms": row.get(rms_col, "") if rms_col else "",
                    "reduced_chi2": row.get(chi_col, "") if chi_col else "",
                    "fit_status": row.get(status_col, "") if status_col else "",
                    "raw_columns_json": json.dumps(row, sort_keys=True),
                }
            )
    return rows


def _iter_stage_files(
    runtimes: list[StageRuntime],
    pattern: str,
) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for rt in runtimes:
        if not rt.run_dir_exists or not rt.run_dir:
            continue
        for path in sorted(rt.run_dir.glob(pattern)):
            if path.is_file():
                found.append((rt.spec.key, path))
    return found


def collect_model_check_index(
    artifact_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in artifact_rows:
        path = str(row.get("path", ""))
        if not any(
            part in path
            for part in ("binary_analysis/", "param_scan/", "change_report/")
        ):
            continue
        rows.append(
            {
                "stage": row.get("stage", ""),
                "artifact_type": row.get("artifact_type", ""),
                "priority": row.get("priority", ""),
                "path": path,
                "row_count": row.get("row_count", ""),
            }
        )
    return rows


def extract_binary_analysis_rows(
    runtimes: list[StageRuntime], repo_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage, path in _iter_stage_files(
        runtimes, "binary_analysis/binary_analysis.tsv"
    ):
        for row in read_delimited(path):
            rows.append(
                {
                    "stage": stage,
                    "pulsar": row.get("pulsar", ""),
                    "branch": row.get("branch", ""),
                    "BINARY": row.get("BINARY", ""),
                    "PB": row.get("PB", ""),
                    "A1": row.get("A1", ""),
                    "ECC": row.get("ECC", ""),
                    "EPS1": row.get("EPS1", ""),
                    "EPS2": row.get("EPS2", ""),
                    "T0": row.get("T0", ""),
                    "TASC": row.get("TASC", ""),
                    "source_path": clean_rel(path, repo_root),
                    "raw_columns_json": json.dumps(row, sort_keys=True),
                }
            )
    return rows


def extract_param_scan_summary_rows(
    runtimes: list[StageRuntime], repo_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scan_files: dict[tuple[str, Path], None] = {}
    for item in _iter_stage_files(runtimes, "param_scan/param_scan_*.tsv"):
        scan_files[item] = None
    for item in _iter_stage_files(runtimes, "param_scan/*/param_scan_*.tsv"):
        scan_files[item] = None
    for stage, path in sorted(scan_files):
        data = read_delimited(path)
        if not data:
            continue
        by_pulsar: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in data:
            by_pulsar[str(row.get("pulsar", "") or Path(path).stem).strip()].append(row)
        for pulsar, pulsar_rows in sorted(by_pulsar.items()):
            best = sorted(
                pulsar_rows,
                key=lambda row: (
                    (
                        safe_float(row.get("lrt_p_value"))
                        if safe_float(row.get("lrt_p_value")) is not None
                        else math.inf
                    ),
                    -(
                        safe_float(row.get("lrt_delta_chisq"))
                        if safe_float(row.get("lrt_delta_chisq")) is not None
                        else -math.inf
                    ),
                ),
            )[0]
            rows.append(
                {
                    "stage": stage,
                    "pulsar": pulsar,
                    "branch": best.get("branch", ""),
                    "candidate": best.get("candidate", ""),
                    "redchisq": best.get("redchisq", ""),
                    "delta_k_fit": best.get("delta_k_fit", ""),
                    "lrt_delta_chisq": best.get("lrt_delta_chisq", ""),
                    "lrt_p_value": best.get("lrt_p_value", ""),
                    "max_param_z": best.get("max_param_z", ""),
                    "source_path": clean_rel(path, repo_root),
                    "raw_columns_json": json.dumps(best, sort_keys=True),
                }
            )
    return rows


def extract_change_report_model_rows(
    runtimes: list[StageRuntime], repo_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage, path in _iter_stage_files(
        runtimes, "change_report/MODEL_COMPARISON_*.tsv"
    ):
        for row in read_delimited(path):
            rows.append(
                {
                    "stage": stage,
                    "pulsar": row.get("pulsar", ""),
                    "branch": row.get("branch", ""),
                    "reference": row.get("reference", ""),
                    "ref_k_fit": row.get("ref_k_fit", ""),
                    "br_k_fit": row.get("br_k_fit", ""),
                    "delta_redchisq": row.get("delta_redchisq", ""),
                    "delta_wrms_post": row.get("delta_wrms_post", ""),
                    "delta_aic": row.get("delta_aic", ""),
                    "delta_bic": row.get("delta_bic", ""),
                    "delta_k_fit": row.get("delta_k_fit", ""),
                    "lrt_delta_chisq": row.get("lrt_delta_chisq", ""),
                    "lrt_p_value": row.get("lrt_p_value", ""),
                    "source_path": clean_rel(path, repo_root),
                    "raw_columns_json": json.dumps(row, sort_keys=True),
                }
            )
    return rows


def extract_new_param_significance_rows(
    runtimes: list[StageRuntime], repo_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage, path in _iter_stage_files(
        runtimes, "change_report/NEW_PARAM_SIGNIFICANCE_*.tsv"
    ):
        for row in read_delimited(path):
            threshold_col = next(
                (key for key in row if str(key).startswith("n_new_sig_z>=")),
                "",
            )
            threshold = ""
            if threshold_col:
                match = re.search(r"z>=(.+)$", threshold_col)
                threshold = match.group(1) if match else ""
            rows.append(
                {
                    "stage": stage,
                    "pulsar": row.get("pulsar", ""),
                    "branch": row.get("branch", ""),
                    "reference": row.get("reference", ""),
                    "n_new_params": row.get("n_new_params", ""),
                    "n_new_with_numeric_sigma": row.get("n_new_with_numeric_sigma", ""),
                    "n_new_sig_z": row.get(threshold_col, "") if threshold_col else "",
                    "n_new_sig_threshold": threshold,
                    "max_new_param_z": row.get("max_new_param_z", ""),
                    "max_new_param": row.get("max_new_param", ""),
                    "source_path": clean_rel(path, repo_root),
                    "raw_columns_json": json.dumps(row, sort_keys=True),
                }
            )
    return rows


def build_model_check_package(
    *,
    runtimes: list[StageRuntime],
    artifact_rows: list[dict[str, Any]],
    repo_root: Path,
    model_dir: Path,
) -> dict[str, Any]:
    artifact_index_rows = collect_model_check_index(artifact_rows)
    binary_rows = extract_binary_analysis_rows(runtimes, repo_root)
    param_scan_rows = extract_param_scan_summary_rows(runtimes, repo_root)
    change_model_rows = extract_change_report_model_rows(runtimes, repo_root)
    new_param_rows = extract_new_param_significance_rows(runtimes, repo_root)

    binary_models = sorted(
        {
            str(row.get("BINARY", "")).strip()
            for row in binary_rows
            if str(row.get("BINARY", "")).strip()
        }
    )
    review_lines = [
        "# Model checks",
        "",
        "This section indexes the workflow outputs that are relevant to timing-model sanity checks but were previously only left as raw artifacts.",
        "",
        "## Summary",
        "",
        f"- Binary-analysis rows: **{len(binary_rows)}**",
        f"- Param-scan summary rows: **{len(param_scan_rows)}**",
        f"- Change-report model-comparison rows: **{len(change_model_rows)}**",
        f"- New-parameter significance rows: **{len(new_param_rows)}**",
        "",
        f"- Binary models seen: **{', '.join(binary_models) if binary_models else 'none'}**",
        "",
        "## Strongest param-scan candidates",
        "",
        md_table(
            sorted(
                param_scan_rows,
                key=lambda row: (
                    (
                        safe_float(row.get("lrt_p_value"))
                        if safe_float(row.get("lrt_p_value")) is not None
                        else math.inf
                    ),
                    -(
                        safe_float(row.get("lrt_delta_chisq"))
                        if safe_float(row.get("lrt_delta_chisq")) is not None
                        else -math.inf
                    ),
                ),
            ),
            [
                "stage",
                "pulsar",
                "candidate",
                "lrt_p_value",
                "lrt_delta_chisq",
                "max_param_z",
                "source_path",
            ],
            max_rows=12,
        ),
        "## Change-report model comparison",
        "",
        md_table(
            sorted(
                change_model_rows,
                key=lambda row: (
                    (
                        safe_float(row.get("lrt_p_value"))
                        if safe_float(row.get("lrt_p_value")) is not None
                        else math.inf
                    ),
                    -(
                        safe_float(row.get("lrt_delta_chisq"))
                        if safe_float(row.get("lrt_delta_chisq")) is not None
                        else -math.inf
                    ),
                ),
            ),
            [
                "stage",
                "pulsar",
                "branch",
                "reference",
                "delta_redchisq",
                "lrt_delta_chisq",
                "lrt_p_value",
                "source_path",
            ],
            max_rows=12,
        ),
        "## New-parameter significance",
        "",
        md_table(
            sorted(
                new_param_rows,
                key=lambda row: -(
                    safe_float(row.get("max_new_param_z"))
                    if safe_float(row.get("max_new_param_z")) is not None
                    else -math.inf
                ),
            ),
            [
                "stage",
                "pulsar",
                "branch",
                "n_new_params",
                "n_new_sig_z",
                "max_new_param_z",
                "max_new_param",
                "source_path",
            ],
            max_rows=12,
        ),
        "## Binary analysis",
        "",
        md_table(
            binary_rows,
            [
                "stage",
                "pulsar",
                "branch",
                "BINARY",
                "PB",
                "A1",
                "ECC",
                "T0",
                "TASC",
                "source_path",
            ],
            max_rows=12,
        ),
    ]
    write_text(model_dir / "model_checks.md", "\n".join(review_lines) + "\n")
    return {
        "artifact_index_rows": artifact_index_rows,
        "binary_rows": binary_rows,
        "param_scan_rows": param_scan_rows,
        "change_model_rows": change_model_rows,
        "new_param_rows": new_param_rows,
    }


def extract_public_metrics(
    public_dir: Optional[Path], repo_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not public_dir or not public_dir.exists():
        return [], [], {}
    metrics: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    tension_rows: list[dict[str, Any]] = []
    summary = public_dir / "public_release_parameters.summary.tsv"
    if summary.exists():
        rows = read_delimited(summary)
        for row in rows:
            lower = {k.lower(): k for k in row.keys()}
            if "metric" in lower and "value" in lower:
                metrics[row[lower["metric"]]] = row[lower["value"]]
            else:
                for k, v in row.items():
                    if k in {
                        "sigma_tension_max",
                        "sigma_tension_mean",
                        "reduced_chi2",
                        "worst_provider_pair",
                        "agreement_class",
                    }:
                        metrics[k] = v
        for k in sorted(metrics):
            metric_rows.append(
                {
                    "metric": k,
                    "value": metrics[k],
                    "source_path": clean_rel(summary, repo_root),
                }
            )
    comparison = public_dir / "public_release_parameters.comparison.tsv"
    if comparison.exists():
        rows = read_delimited(comparison)
        for row in rows:
            cols = row.keys()
            param_col = choose_col(
                cols, ("parameter", "param", "name"), ("parameter", "param")
            )
            sigma_col = choose_col(cols, ("sigma_tension",), ("sigma",))
            pair_col = choose_col(
                cols, ("provider_pair", "worst_provider_pair"), ("pair",)
            )
            class_col = choose_col(
                cols,
                ("agreement_class", "status", "class"),
                ("agreement", "class", "status"),
            )
            parameter = row.get(param_col, "") if param_col else ""
            sigma = row.get(sigma_col, "") if sigma_col else ""
            if parameter in {"RA_ICRS_DEG", "DEC_ICRS_DEG"} or sigma:
                tension_rows.append(
                    {
                        "parameter": parameter,
                        "sigma_tension": sigma,
                        "provider_pair": row.get(pair_col, "") if pair_col else "",
                        "agreement_class": row.get(class_col, "") if class_col else "",
                        "source_path": clean_rel(comparison, repo_root),
                        "raw_columns_json": json.dumps(row, sort_keys=True),
                    }
                )
        sigmas = [safe_float(r.get("sigma_tension")) for r in tension_rows]
        sigmas_f = [x for x in sigmas if x is not None]
        if sigmas_f and "sigma_tension_max" not in metrics:
            metrics["sigma_tension_max"] = max(sigmas_f)
            metric_rows.append(
                {
                    "metric": "sigma_tension_max",
                    "value": f"{max(sigmas_f):.6g}",
                    "source_path": clean_rel(comparison, repo_root),
                }
            )
    return metric_rows, tension_rows, metrics


def _guess_variant(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_qc"):
        stem = stem[:-3]
    for sep in (".", "__", "_"):
        if sep in stem:
            head, tail = stem.split(sep, 1)
            if head.startswith("J") and tail:
                return tail.strip("._") or "base"
    return "base"


def _numeric_series(
    df: pd.DataFrame, col: str | None, *, microseconds: bool = False
) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    out = pd.to_numeric(df[col], errors="coerce")
    if microseconds:
        out = out * 1.0e6
    return out


def choose_postfit_residual_column(df: pd.DataFrame) -> ResidualChoice:
    available: list[str] = []
    for col in ALL_RESIDUAL_PREFERENCE:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                available.append(col)
    seen = set(available)
    for col in df.columns:
        name = str(col).lower()
        if col in seen:
            continue
        if any(
            k in name for k in ("resid", "residual", "post", "postfit", "detrended")
        ):
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                available.append(str(col))
                seen.add(str(col))

    for col in POSTFIT_RESIDUAL_PREFERENCE:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if not values.notna().any():
            continue
        err_col = None
        n_err = 0
        # TEMPO2 general2 ``post`` is in seconds, but in real outputs the
        # general2 ``err`` column is already in microseconds. Prefer the raw
        # ``tempo2_err`` field here to avoid double-scaling.
        if "tempo2_err" in df.columns:
            err_col = "tempo2_err"
            n_err = int(pd.to_numeric(df[err_col], errors="coerce").notna().sum())
        elif "tempo2_err_us" in df.columns:
            err_col = "tempo2_err_us"
            n_err = int(pd.to_numeric(df[err_col], errors="coerce").notna().sum())
        units = "us" if col.endswith("_us") else "s"
        return ResidualChoice(
            column=col,
            error_column=err_col,
            units=units,
            n_numeric=int(values.notna().sum()),
            n_error_numeric=n_err,
            available_residual_columns=tuple(available),
            warning="",
        )
    warning = "No numeric post-fit residual column was found."
    return ResidualChoice(
        column=None,
        error_column=None,
        units="us",
        n_numeric=0,
        n_error_numeric=0,
        available_residual_columns=tuple(available),
        warning=warning,
    )


def decision_series(df: pd.DataFrame) -> pd.Series:
    base = df.get(
        "reviewed_decision", df.get("auto_decision", pd.Series("KEEP", index=df.index))
    )
    return base.fillna("KEEP").astype(str).str.upper()


def manual_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df.get("manual_action", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    )


def build_plot_subset(df: pd.DataFrame, *, max_keep_points: int) -> pd.DataFrame:
    plot_df = df.dropna(subset=["mjd", "plot_residual_us"]).copy()
    if plot_df.empty:
        return plot_df
    decision = decision_series(plot_df)
    important = decision.isin(FLAGGED_DECISIONS) | manual_mask(plot_df)
    important_df = plot_df.loc[important].copy()
    keep_df = plot_df.loc[~important].copy()
    keep_df["_plot_sampled"] = False
    important_df["_plot_sampled"] = False
    if len(keep_df) > max_keep_points:
        keep_df = keep_df.sample(int(max_keep_points), random_state=260408373).copy()
        keep_df["_plot_sampled"] = True
    out = pd.concat([important_df, keep_df], ignore_index=True, sort=False)
    out["_decision"] = decision_series(out)
    out["_manual"] = manual_mask(out)
    return out


def _draw_vertical_errorbars(
    ax: plt.Axes,
    xvals: Any,
    yvals: Any,
    yerr: Any,
    *,
    color: Any,
    alpha: float,
    linewidth: float,
    zorder: float = 1.0,
) -> None:
    xv = pd.to_numeric(pd.Series(xvals), errors="coerce").to_numpy(dtype=float)
    yv = pd.to_numeric(pd.Series(yvals), errors="coerce").to_numpy(dtype=float)
    ev = pd.to_numeric(pd.Series(yerr), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(ev) & (ev > 0.0)
    if not np.any(mask):
        return
    ax.errorbar(
        xv[mask],
        yv[mask],
        yerr=ev[mask],
        fmt="none",
        ecolor=color,
        elinewidth=linewidth,
        alpha=alpha,
        capsize=1.8,
        capthick=linewidth,
        zorder=zorder,
    )


def _choose_residual_display_limit(frame: pd.DataFrame) -> float | None:
    if "plot_residual_us" not in frame.columns:
        return None
    residual = pd.to_numeric(frame["plot_residual_us"], errors="coerce")
    finite = residual[np.isfinite(residual)]
    if finite.empty:
        return None
    decision = frame.get("_decision")
    if decision is None:
        decision = decision_series(frame)
    keep_mask = decision.astype(str).eq("KEEP")
    core = residual[keep_mask & np.isfinite(residual)]
    if len(core) < 5:
        core = finite
    core_abs = np.abs(core.to_numpy(dtype=float))
    finite_abs = np.abs(finite.to_numpy(dtype=float))
    bulk_abs = core_abs if core_abs.size else finite_abs
    err_abs = np.array([], dtype=float)
    if "plot_err_us" in frame.columns:
        err = pd.to_numeric(frame["plot_err_us"], errors="coerce")
        err_abs = np.abs(err[np.isfinite(err)].to_numpy(dtype=float))
    core_p95 = float(np.nanpercentile(core_abs, 95)) if core_abs.size else 0.0
    bulk_p90 = float(np.nanpercentile(bulk_abs, 90)) if bulk_abs.size else 0.0
    err_p95 = float(np.nanpercentile(err_abs, 95)) if err_abs.size else 0.0
    limit = max(core_p95 * 4.0, bulk_p90 * 2.0, err_p95 * 8.0, 5.0)
    max_abs = float(np.nanmax(finite_abs))
    if max_abs <= limit * 1.2:
        return max(limit, max_abs * 1.05, 5.0)
    return min(limit, max_abs * 0.98)


def _draw_clipped_markers(
    ax: plt.Axes,
    xvals: Any,
    yvals: Any,
    *,
    limit: float | None,
    color: Any,
    alpha: float,
    size: float,
    zorder: float = 4.0,
) -> None:
    if limit is None or limit <= 0:
        return
    xv = pd.to_numeric(pd.Series(xvals), errors="coerce").to_numpy(dtype=float)
    yv = pd.to_numeric(pd.Series(yvals), errors="coerce").to_numpy(dtype=float)
    top = np.isfinite(xv) & np.isfinite(yv) & (yv > limit)
    bottom = np.isfinite(xv) & np.isfinite(yv) & (yv < -limit)
    if np.any(top):
        ax.scatter(
            xv[top],
            np.full(int(np.count_nonzero(top)), limit),
            s=size,
            alpha=alpha,
            c=[color],
            marker="^",
            edgecolors="none",
            rasterized=True,
            zorder=zorder,
        )
    if np.any(bottom):
        ax.scatter(
            xv[bottom],
            np.full(int(np.count_nonzero(bottom)), -limit),
            s=size,
            alpha=alpha,
            c=[color],
            marker="v",
            edgecolors="none",
            rasterized=True,
            zorder=zorder,
        )


def _scatter_by_decision(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    xlabel: str,
    ylabel: str,
    yerr: str | None = None,
    display_limit: float | None = None,
) -> None:
    ax.axhline(0.0, color="#d1d5db", lw=1.0, zorder=0)
    for dec in DECISION_ORDER:
        group = frame[frame["_decision"] == dec]
        if group.empty:
            continue
        size = 7 if dec == "KEEP" else 16
        alpha = 0.35 if dec == "KEEP" else 0.9
        yraw = pd.to_numeric(group[y], errors="coerce")
        yplot = (
            yraw.clip(-display_limit, display_limit)
            if display_limit is not None
            else yraw
        )
        if yerr and yerr in group.columns:
            err_visible = group
            if display_limit is not None:
                err_visible = group.loc[np.abs(yraw) <= display_limit]
            _draw_vertical_errorbars(
                ax,
                err_visible[x],
                err_visible[y],
                err_visible[yerr],
                color=DECISION_COLORS.get(dec, "#111827"),
                alpha=0.55 if dec == "KEEP" else 0.85,
                linewidth=0.55 if dec == "KEEP" else 0.9,
                zorder=3.0,
            )
        ax.scatter(
            pd.to_numeric(group[x], errors="coerce"),
            yplot,
            s=size,
            alpha=alpha,
            c=DECISION_COLORS.get(dec, "#111827"),
            edgecolors="none",
            label=dec,
            rasterized=True,
        )
        _draw_clipped_markers(
            ax,
            group[x],
            yraw,
            limit=display_limit,
            color=DECISION_COLORS.get(dec, "#111827"),
            alpha=alpha,
            size=size + 6,
        )
    manual = frame[frame["_manual"]]
    if not manual.empty:
        my = pd.to_numeric(manual[y], errors="coerce")
        my_plot = (
            my.clip(-display_limit, display_limit) if display_limit is not None else my
        )
        ax.scatter(
            pd.to_numeric(manual[x], errors="coerce"),
            my_plot,
            s=34,
            facecolors="none",
            edgecolors="#111827",
            linewidths=0.7,
            label="manual override",
            rasterized=True,
        )
        _draw_clipped_markers(
            ax,
            manual[x],
            my,
            limit=display_limit,
            color="#111827",
            alpha=0.95,
            size=34,
        )
    if display_limit is not None:
        ax.set_ylim(-display_limit, display_limit)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _backend_palette(values: Sequence[str]) -> dict[str, Any]:
    uniq = list(dict.fromkeys([str(v) for v in values if str(v)]))
    cmap = plt.get_cmap("tab20")
    return {name: cmap(i % 20) for i, name in enumerate(uniq)}


def _add_axis_legend(
    ax: plt.Axes,
    *,
    loc: str,
    ncol: int = 1,
    fontsize: float = 8.0,
    title: str | None = None,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    seen: set[str] = set()
    uniq_handles: list[Any] = []
    uniq_labels: list[str] = []
    for handle, label in zip(handles, labels):
        if not label or label in seen:
            continue
        seen.add(label)
        uniq_handles.append(handle)
        uniq_labels.append(label)
    if not uniq_handles:
        return
    ax.legend(
        uniq_handles,
        uniq_labels,
        loc=loc,
        ncol=ncol,
        fontsize=fontsize,
        title=title,
        title_fontsize=fontsize,
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        borderpad=0.35,
        labelspacing=0.3,
        handletextpad=0.4,
        columnspacing=0.8,
        facecolor="white",
        edgecolor="#d1d5db",
    )


def _plot_backend_strip(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    ylabel: str,
    display_limit: float | None = None,
) -> None:
    xvals = (
        frame.get("backend_group", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
    )
    uniq = [x for x in dict.fromkeys(xvals.tolist()) if x]
    if not uniq:
        ax.text(
            0.5,
            0.5,
            "No backend labels",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return
    xpos = {name: i for i, name in enumerate(uniq)}
    decision = frame["_decision"].astype(str)
    jitter_rng = np.random.default_rng(260408373)
    ax.axhline(0.0, color="#d1d5db", lw=1.0, zorder=0)
    for dec in DECISION_ORDER:
        group = frame[decision == dec]
        if group.empty:
            continue
        gx = (
            group["backend_group"]
            .fillna("")
            .astype(str)
            .map(xpos)
            .to_numpy(dtype=float)
        )
        gx = gx + jitter_rng.uniform(-0.22, 0.22, size=len(group))
        gy = pd.to_numeric(group["plot_residual_us"], errors="coerce")
        gy_plot = (
            gy.clip(-display_limit, display_limit) if display_limit is not None else gy
        )
        if "plot_err_us" in group.columns:
            err_visible = group
            if display_limit is not None:
                err_visible = group.loc[np.abs(gy) <= display_limit]
            _draw_vertical_errorbars(
                ax,
                (
                    gx[np.abs(gy.to_numpy(dtype=float)) <= display_limit]
                    if display_limit is not None
                    else gx
                ),
                err_visible["plot_residual_us"],
                err_visible["plot_err_us"],
                color=DECISION_COLORS.get(dec, "#111827"),
                alpha=0.55 if dec == "KEEP" else 0.85,
                linewidth=0.55 if dec == "KEEP" else 0.9,
                zorder=3.0,
            )
        ax.scatter(
            gx,
            gy_plot,
            s=6 if dec == "KEEP" else 15,
            alpha=0.35 if dec == "KEEP" else 0.9,
            c=DECISION_COLORS.get(dec, "#111827"),
            edgecolors="none",
            label=dec,
            rasterized=True,
        )
        _draw_clipped_markers(
            ax,
            gx,
            gy,
            limit=display_limit,
            color=DECISION_COLORS.get(dec, "#111827"),
            alpha=0.35 if dec == "KEEP" else 0.9,
            size=12 if dec == "KEEP" else 18,
        )
    manual = frame[frame["_manual"]]
    if not manual.empty:
        mx = (
            manual["backend_group"]
            .fillna("")
            .astype(str)
            .map(xpos)
            .to_numpy(dtype=float)
        )
        mx = mx + jitter_rng.uniform(-0.22, 0.22, size=len(manual))
        my = pd.to_numeric(manual["plot_residual_us"], errors="coerce")
        my_plot = (
            my.clip(-display_limit, display_limit) if display_limit is not None else my
        )
        ax.scatter(
            mx,
            my_plot,
            s=28,
            facecolors="none",
            edgecolors="#111827",
            linewidths=0.7,
            label="manual override",
            rasterized=True,
        )
        _draw_clipped_markers(
            ax,
            mx,
            my,
            limit=display_limit,
            color="#111827",
            alpha=0.95,
            size=28,
        )
    if display_limit is not None:
        ax.set_ylim(-display_limit, display_limit)
    ax.set_xticks(range(len(uniq)))
    ax.set_xticklabels(uniq, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("backend")


def _plot_backend_time(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    ylabel: str,
    display_limit: float | None = None,
) -> None:
    ax.axhline(0.0, color="#d1d5db", lw=1.0, zorder=0)
    palette = _backend_palette(frame["backend_group"].fillna("").astype(str).tolist())
    for backend, group in frame.groupby("backend_group", sort=False):
        if not str(backend):
            continue
        flagged = (
            decision_series(group).isin(FLAGGED_DECISIONS).any()
            or manual_mask(group).any()
        )
        alpha = 0.75 if flagged else 0.25
        size = 10 if flagged else 5
        color = palette.get(str(backend), "#6b7280")
        gy = pd.to_numeric(group["plot_residual_us"], errors="coerce")
        gy_plot = (
            gy.clip(-display_limit, display_limit) if display_limit is not None else gy
        )
        if "plot_err_us" in group.columns:
            err_visible = group
            if display_limit is not None:
                err_visible = group.loc[np.abs(gy) <= display_limit]
            _draw_vertical_errorbars(
                ax,
                err_visible["mjd"],
                err_visible["plot_residual_us"],
                err_visible["plot_err_us"],
                color=color,
                alpha=0.85 if flagged else 0.55,
                linewidth=0.9 if flagged else 0.55,
                zorder=3.0,
            )
        ax.scatter(
            pd.to_numeric(group["mjd"], errors="coerce"),
            gy_plot,
            s=size,
            alpha=alpha,
            c=[color],
            edgecolors="none",
            label=str(backend),
            rasterized=True,
        )
        _draw_clipped_markers(
            ax,
            group["mjd"],
            gy,
            limit=display_limit,
            color=color,
            alpha=alpha,
            size=size + 6,
        )
    manual = frame[frame["_manual"]]
    if not manual.empty:
        my = pd.to_numeric(manual["plot_residual_us"], errors="coerce")
        my_plot = (
            my.clip(-display_limit, display_limit) if display_limit is not None else my
        )
        ax.scatter(
            pd.to_numeric(manual["mjd"], errors="coerce"),
            my_plot,
            s=30,
            facecolors="none",
            edgecolors="#111827",
            linewidths=0.7,
            rasterized=True,
        )
        _draw_clipped_markers(
            ax,
            manual["mjd"],
            my,
            limit=display_limit,
            color="#111827",
            alpha=0.95,
            size=30,
        )
    if display_limit is not None:
        ax.set_ylim(-display_limit, display_limit)
    ax.set_xlabel("MJD")
    ax.set_ylabel(ylabel)


def savefig(fig: plt.Figure, path: Path, *, dpi: int = 170) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_variant_overview_plot(
    frame: pd.DataFrame,
    *,
    title: str,
    residual_label: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    display_limit = _choose_residual_display_limit(frame)
    _scatter_by_decision(
        axes[0, 0],
        frame,
        "mjd",
        "plot_residual_us",
        xlabel="MJD",
        ylabel=residual_label,
        yerr="plot_err_us",
        display_limit=display_limit,
    )
    axes[0, 0].set_title("Residual vs MJD by decision")
    if (
        "freq" in frame.columns
        and pd.to_numeric(frame["freq"], errors="coerce").notna().any()
    ):
        _scatter_by_decision(
            axes[0, 1],
            frame,
            "freq",
            "plot_residual_us",
            xlabel="Frequency [MHz]",
            ylabel=residual_label,
            yerr="plot_err_us",
            display_limit=display_limit,
        )
        axes[0, 1].set_title("Residual vs frequency by decision")
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No numeric frequency column",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_axis_off()
    _plot_backend_strip(
        axes[1, 0], frame, ylabel=residual_label, display_limit=display_limit
    )
    axes[1, 0].set_title("Residual vs backend by decision")
    _plot_backend_time(
        axes[1, 1], frame, ylabel=residual_label, display_limit=display_limit
    )
    axes[1, 1].set_title("Residual vs MJD by backend")
    _add_axis_legend(axes[0, 0], loc="upper left", fontsize=8.0, title="decision")
    backend_n = max(
        1,
        len(
            frame.get("backend_group", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .replace("", np.nan)
            .dropna()
            .unique()
        ),
    )
    _add_axis_legend(
        axes[1, 1],
        loc="upper left",
        ncol=1 if backend_n <= 6 else 2,
        fontsize=7.0,
        title="backend",
    )
    if display_limit is not None and np.isfinite(display_limit):
        fig.text(
            0.99,
            0.01,
            f"bulk display clipped at +/-{display_limit:.1f} us",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#4b5563",
        )
    fig.suptitle(title, fontsize=14)
    savefig(fig, out_path)


def make_signed_sigma_plot(frame: pd.DataFrame, *, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="#d1d5db", lw=1.0, zorder=0)
    ax.axhline(3.0, color="#9ca3af", lw=0.9, ls="--")
    ax.axhline(-3.0, color="#9ca3af", lw=0.9, ls="--")
    ax.axhline(5.0, color="#b91c1c", lw=0.9, ls=":")
    ax.axhline(-5.0, color="#b91c1c", lw=0.9, ls=":")
    _scatter_by_decision(
        ax,
        frame,
        "mjd",
        "plot_sigma",
        xlabel="MJD",
        ylabel="post-fit residual / tempo2 error",
    )
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels, loc="upper right", frameon=False, ncol=min(5, len(labels))
        )
    savefig(fig, out_path)


def choose_qc_review_stage(runtimes: list[StageRuntime]) -> StageRuntime | None:
    preferred = ("step4_detect_selected", "step2_detect_variants")
    for key in preferred:
        rt = next((x for x in runtimes if x.spec.key == key and x.run_dir_exists), None)
        if rt is not None:
            return rt
    return None


def _guess_qc_override_csv(qc_run_dir: Path) -> Path:
    return qc_run_dir / "qc_review" / "manual_qc_overrides.csv"


def _qc_csvs_for_pulsar(run_dir: Path, psr: str) -> list[Path]:
    out: list[Path] = []
    for path in find_qc_csvs(run_dir):
        stem = path.stem
        if stem.startswith(psr):
            out.append(path)
    if out:
        return sorted(out)
    fallback_patterns = (
        f"{psr}.*_qc.csv",
        f"{psr}*.new_qc.csv",
        f"{psr}*.legacy_qc.csv",
        f"{psr}*.combined_qc.csv",
    )
    seen: set[Path] = set()
    for pattern in fallback_patterns:
        for path in sorted(run_dir.rglob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                out.append(path)
    return sorted(out)


def _group_backend_names(series: pd.Series, *, max_backends: int = 12) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    counts = text[text != ""].value_counts()
    keep = set(counts.head(max_backends).index.tolist())
    grouped = text.where(text.isin(keep), other="OTHER")
    grouped = grouped.where(grouped != "", other="UNKNOWN")
    return grouped


def _round_stat(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def build_qc_review_package(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    qc_stage: StageRuntime | None,
    postfit_dir: Path,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "stage": qc_stage.spec.key if qc_stage else "",
        "run_dir": (
            clean_rel(qc_stage.run_dir, repo_root)
            if qc_stage and qc_stage.run_dir
            else ""
        ),
        "warnings": [],
        "variant_summary_rows": [],
        "backend_rows": [],
        "worst_rows": [],
        "surviving_keep_rows": [],
        "plot_rows": [],
        "availability_rows": [],
    }
    if qc_stage is None or not qc_stage.run_dir:
        output["warnings"].append("No QC run directory was found for step4 or step2.")
        return output

    qc_run_dir = qc_stage.run_dir
    csvs = _qc_csvs_for_pulsar(qc_run_dir, args.psr)
    if not csvs:
        output["warnings"].append(f"No raw *_qc.csv files were found for {args.psr}.")
        return output

    overrides_path = (
        Path(args.overrides).expanduser()
        if args.overrides
        else _guess_qc_override_csv(qc_run_dir)
    )
    overrides = load_overrides(overrides_path if overrides_path.exists() else None)
    if overrides_path.exists():
        output["overrides_path"] = clean_rel(overrides_path, repo_root)
    else:
        output["overrides_path"] = ""

    review_lines = [
        f"# Post-fit residual review: {args.psr}",
        "",
        f"QC stage: `{qc_stage.spec.key}`",
        f"Run directory: `{clean_rel(qc_run_dir, repo_root)}`",
        f"Manual overrides: `{clean_rel(overrides_path, repo_root) if overrides_path.exists() else 'not found'}`",
        "",
        "This section uses only tempo2 post-fit residuals for the primary reviewer plots. If a variant has no numeric post-fit residual column attached, the package warns and does not silently substitute a raw residual as the main plot.",
        "",
    ]

    for qc_csv in csvs:
        df = load_qc_csv(qc_csv, root=qc_run_dir)
        reviewed = apply_overrides(df, overrides)
        reviewed["decision"] = decision_series(reviewed)
        choice = choose_postfit_residual_column(reviewed)
        guessed_variant = _guess_variant(qc_csv)
        qc_variant = str(
            reviewed.get("variant", pd.Series([guessed_variant])).iloc[0] or ""
        ).strip()
        variant = (
            guessed_variant
            if qc_variant.lower() in {"", "base"} and guessed_variant
            else qc_variant or guessed_variant or "base"
        )
        available_cols = ",".join(choice.available_residual_columns)
        availability_row = {
            "variant": variant,
            "qc_csv": clean_rel(qc_csv, repo_root),
            "postfit_available": "yes" if choice.column else "no",
            "selected_residual_column": choice.column or "",
            "selected_error_column": choice.error_column or "",
            "n_rows": len(reviewed),
            "n_numeric_postfit": choice.n_numeric,
            "n_numeric_error": choice.n_error_numeric,
            "available_residual_columns": available_cols,
            "warning": choice.warning,
        }
        output["availability_rows"].append(availability_row)

        if choice.column is None:
            output["warnings"].append(
                f"{variant}: no numeric post-fit residual column was found."
            )
            review_lines.extend(
                [
                    f"## Variant `{variant}`",
                    "",
                    f"- QC CSV: `{clean_rel(qc_csv, repo_root)}`",
                    "- Post-fit residuals: **missing**",
                    f"- Available residual-like columns: `{available_cols or 'none'}`",
                    "",
                ]
            )
            continue

        if choice.units == "us":
            reviewed["plot_residual_us"] = _numeric_series(
                reviewed, choice.column, microseconds=False
            )
        else:
            reviewed["plot_residual_us"] = _numeric_series(
                reviewed, choice.column, microseconds=True
            )
        if choice.error_column == "tempo2_err":
            reviewed["plot_err_us"] = _numeric_series(
                reviewed, choice.error_column, microseconds=False
            )
        elif choice.error_column == "tempo2_err_us":
            reviewed["plot_err_us"] = _numeric_series(
                reviewed, choice.error_column, microseconds=False
            )
        else:
            reviewed["plot_err_us"] = np.nan
        reviewed["plot_sigma"] = reviewed["plot_residual_us"] / reviewed["plot_err_us"]
        reviewed.loc[~np.isfinite(reviewed["plot_sigma"]), "plot_sigma"] = np.nan
        reviewed["backend_group"] = _group_backend_names(
            reviewed.get("backend", pd.Series("", index=reviewed.index))
        )

        n_postfit = int(reviewed["plot_residual_us"].notna().sum())
        if n_postfit < len(reviewed):
            frac = n_postfit / max(1, len(reviewed))
            warning = f"{variant}: post-fit residual coverage is {n_postfit}/{len(reviewed)} ({frac:.1%})."
            output["warnings"].append(warning)

        plot_df = build_plot_subset(reviewed, max_keep_points=args.max_keep_points)
        decision = reviewed["decision"]
        n_keep = int((decision == "KEEP").sum())
        n_bad = int((decision == "BAD_TOA").sum())
        n_event = int((decision == "EVENT").sum())
        n_review_event = int((decision == "REVIEW_EVENT").sum())
        n_manual = int(manual_mask(reviewed).sum())
        abs_resid = reviewed["plot_residual_us"].abs()
        abs_sigma = reviewed["plot_sigma"].abs()
        keep_abs_sigma = reviewed.loc[decision == "KEEP", "plot_sigma"].abs()

        backend_summary = (
            reviewed.assign(
                _is_flagged=decision.isin(FLAGGED_DECISIONS),
                _is_manual=manual_mask(reviewed),
            )
            .groupby("backend_group", dropna=False)
            .agg(
                n_rows=("backend_group", "size"),
                n_postfit=(
                    "plot_residual_us",
                    lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum()),
                ),
                n_flagged=("_is_flagged", "sum"),
                n_manual=("_is_manual", "sum"),
                median_abs_postfit_us=(
                    "plot_residual_us",
                    lambda s: (
                        float(pd.to_numeric(s, errors="coerce").abs().median())
                        if pd.to_numeric(s, errors="coerce").notna().any()
                        else math.nan
                    ),
                ),
                max_abs_postfit_us=(
                    "plot_residual_us",
                    lambda s: (
                        float(pd.to_numeric(s, errors="coerce").abs().max())
                        if pd.to_numeric(s, errors="coerce").notna().any()
                        else math.nan
                    ),
                ),
            )
            .reset_index()
            .rename(columns={"backend_group": "backend"})
        )
        for _, row in backend_summary.iterrows():
            n_rows = int(row["n_rows"])
            n_flagged = int(row["n_flagged"])
            output["backend_rows"].append(
                {
                    "variant": variant,
                    "backend": row["backend"],
                    "n_rows": n_rows,
                    "n_postfit": int(row["n_postfit"]),
                    "n_flagged": n_flagged,
                    "n_manual": int(row["n_manual"]),
                    "flagged_fraction": _round_stat(
                        n_flagged / n_rows if n_rows else None
                    ),
                    "median_abs_postfit_us": _round_stat(
                        float(row["median_abs_postfit_us"])
                        if math.isfinite(float(row["median_abs_postfit_us"]))
                        else None
                    ),
                    "max_abs_postfit_us": _round_stat(
                        float(row["max_abs_postfit_us"])
                        if math.isfinite(float(row["max_abs_postfit_us"]))
                        else None
                    ),
                }
            )

        worst_backend = ""
        worst_backend_frac = None
        if not backend_summary.empty:
            tmp = backend_summary.copy()
            tmp["_frac"] = tmp["n_flagged"] / tmp["n_rows"].replace({0: np.nan})
            tmp = tmp.sort_values(
                ["_frac", "n_flagged"], ascending=[False, False], kind="stable"
            )
            if len(tmp):
                worst_backend = str(tmp.iloc[0]["backend"])
                worst_backend_frac = (
                    float(tmp.iloc[0]["_frac"])
                    if pd.notna(tmp.iloc[0]["_frac"])
                    else None
                )

        output["variant_summary_rows"].append(
            {
                "variant": variant,
                "qc_csv": clean_rel(qc_csv, repo_root),
                "selected_postfit_column": choice.column,
                "selected_error_column": choice.error_column or "",
                "n_rows": len(reviewed),
                "n_postfit": n_postfit,
                "n_keep": n_keep,
                "n_bad_toa": n_bad,
                "n_event": n_event,
                "n_review_event": n_review_event,
                "n_manual_override_rows": n_manual,
                "median_abs_postfit_us": _round_stat(
                    float(abs_resid.median()) if abs_resid.notna().any() else None
                ),
                "p95_abs_postfit_us": _round_stat(
                    float(abs_resid.quantile(0.95)) if abs_resid.notna().any() else None
                ),
                "max_abs_postfit_us": _round_stat(
                    float(abs_resid.max()) if abs_resid.notna().any() else None
                ),
                "median_abs_sigma": _round_stat(
                    float(abs_sigma.median()) if abs_sigma.notna().any() else None
                ),
                "p95_abs_sigma": _round_stat(
                    float(abs_sigma.quantile(0.95)) if abs_sigma.notna().any() else None
                ),
                "max_abs_sigma": _round_stat(
                    float(abs_sigma.max()) if abs_sigma.notna().any() else None
                ),
                "max_abs_sigma_keep": _round_stat(
                    float(keep_abs_sigma.max())
                    if keep_abs_sigma.notna().any()
                    else None
                ),
                "worst_backend_by_flagged_fraction": worst_backend,
                "worst_backend_flagged_fraction": _round_stat(worst_backend_frac),
            }
        )

        scored = reviewed.copy()
        scored["_abs_residual_us"] = scored["plot_residual_us"].abs()
        scored["_abs_sigma"] = scored["plot_sigma"].abs()
        scored["_manual"] = manual_mask(scored)
        sort_col = (
            "_abs_sigma" if scored["_abs_sigma"].notna().any() else "_abs_residual_us"
        )
        top_rows = scored.sort_values(sort_col, ascending=False, kind="stable").head(
            args.top_n_rows
        )
        keep_rows = (
            scored[decision_series(scored) == "KEEP"]
            .sort_values(sort_col, ascending=False, kind="stable")
            .head(args.top_n_rows)
        )
        for _, row in top_rows.iterrows():
            output["worst_rows"].append(
                {
                    "variant": variant,
                    "review_id": row.get("review_id", ""),
                    "decision": row.get("decision", ""),
                    "manual_action": row.get("manual_action", ""),
                    "backend": row.get("backend", ""),
                    "timfile": row.get("timfile", ""),
                    "mjd": _round_stat(safe_float(row.get("mjd"))),
                    "freq_mhz": _round_stat(safe_float(row.get("freq"))),
                    "postfit_us": _round_stat(safe_float(row.get("plot_residual_us"))),
                    "tempo2_err_us": _round_stat(safe_float(row.get("plot_err_us"))),
                    "abs_sigma": _round_stat(safe_float(row.get("_abs_sigma"))),
                    "bad_point": row.get("bad_point", ""),
                    "event_member": row.get("event_member", ""),
                    "transient_id": row.get("transient_id", ""),
                    "qc_csv": clean_rel(qc_csv, repo_root),
                }
            )
        for _, row in keep_rows.iterrows():
            output["surviving_keep_rows"].append(
                {
                    "variant": variant,
                    "review_id": row.get("review_id", ""),
                    "backend": row.get("backend", ""),
                    "timfile": row.get("timfile", ""),
                    "mjd": _round_stat(safe_float(row.get("mjd"))),
                    "freq_mhz": _round_stat(safe_float(row.get("freq"))),
                    "postfit_us": _round_stat(safe_float(row.get("plot_residual_us"))),
                    "tempo2_err_us": _round_stat(safe_float(row.get("plot_err_us"))),
                    "abs_sigma": _round_stat(safe_float(row.get("_abs_sigma"))),
                    "manual_action": row.get("manual_action", ""),
                    "bad_point": row.get("bad_point", ""),
                    "event_member": row.get("event_member", ""),
                    "transient_id": row.get("transient_id", ""),
                    "qc_csv": clean_rel(qc_csv, repo_root),
                }
            )

        overview_path = (
            postfit_dir / "plots" / f"{args.psr}.{variant}.postfit_overview.png"
        )
        make_variant_overview_plot(
            plot_df,
            title=f"{args.psr} {variant}: post-fit residual review",
            residual_label="tempo2 post-fit residual [us]",
            out_path=overview_path,
        )
        output["plot_rows"].append(
            {
                "variant": variant,
                "plot_type": "overview",
                "path": (Path("plots") / overview_path.name).as_posix(),
                "source_qc_csv": clean_rel(qc_csv, repo_root),
            }
        )
        sigma_path = None
        if reviewed["plot_sigma"].notna().any():
            sigma_path = (
                postfit_dir / "plots" / f"{args.psr}.{variant}.postfit_sigma_vs_mjd.png"
            )
            make_signed_sigma_plot(
                plot_df.dropna(subset=["plot_sigma"]).copy(),
                title=f"{args.psr} {variant}: signed post-fit sigma residuals",
                out_path=sigma_path,
            )
            output["plot_rows"].append(
                {
                    "variant": variant,
                    "plot_type": "signed_sigma_vs_mjd",
                    "path": (Path("plots") / sigma_path.name).as_posix(),
                    "source_qc_csv": clean_rel(qc_csv, repo_root),
                }
            )

        max_keep_sigma = (
            float(keep_abs_sigma.max()) if keep_abs_sigma.notna().any() else None
        )
        review_lines.extend(
            [
                f"## Variant `{variant}`",
                "",
                f"- QC CSV: `{clean_rel(qc_csv, repo_root)}`",
                f"- Post-fit column used: `{choice.column}`",
                f"- Tempo2 error column used: `{choice.error_column or 'missing'}`",
                f"- Numeric post-fit rows: **{n_postfit}/{len(reviewed)}**",
                f"- Decision counts: KEEP={n_keep}, BAD_TOA={n_bad}, EVENT={n_event}, REVIEW_EVENT={n_review_event}",
                f"- Manual override rows: **{n_manual}**",
                f"- Largest surviving KEEP |sigma|: **{_round_stat(max_keep_sigma) or 'NA'}**",
                f"- Worst backend by flagged fraction: `{worst_backend or 'NA'}` ({_round_stat(worst_backend_frac) or 'NA'})",
                "",
                f"![{variant} overview](plots/{overview_path.name})",
                "",
            ]
        )
        if sigma_path is not None:
            review_lines.extend(
                [
                    f"![{variant} signed sigma](plots/{sigma_path.name})",
                    "",
                ]
            )
        if max_keep_sigma is not None and max_keep_sigma >= 5:
            output["warnings"].append(
                f"{variant}: surviving KEEP rows reach |post-fit sigma|={max_keep_sigma:.3g}."
            )

    output["warnings"] = list(dict.fromkeys(output["warnings"]))
    if output["warnings"]:
        review_lines.extend(["## Automatic warnings", ""])
        for warning in output["warnings"]:
            review_lines.append(f"- {warning}")
        review_lines.append("")
    write_text(postfit_dir / "postfit_review.md", "\n".join(review_lines) + "\n")
    return output


def severity_rank(sev: str) -> int:
    return {
        "info": 0,
        "green": 0,
        "warning": 1,
        "amber": 1,
        "critical": 2,
        "red": 2,
    }.get(sev.lower(), 0)


def add_risk(
    rows: list[dict[str, Any]],
    severity: str,
    area: str,
    description: str,
    evidence: str,
    required_action: str,
    status: str = "open",
) -> None:
    rows.append(
        {
            "risk_id": f"R{len(rows) + 1:03d}",
            "severity": severity,
            "area": area,
            "description": description,
            "evidence": evidence,
            "required_action": required_action,
            "status": status,
        }
    )


def build_risk_register(
    runtimes: list[StageRuntime],
    dataset_inventory: list[dict[str, Any]],
    diff_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    system_rows: list[dict[str, Any]],
    public_metrics: dict[str, Any],
    qc_package: dict[str, Any],
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    del diff_rows  # retained for symmetry with the original structure

    for rt in runtimes:
        if rt.spec.creates_branch and not rt.branch_exists:
            add_risk(
                risks,
                "critical" if not rt.spec.optional else "warning",
                "provenance",
                f"Expected branch is missing for {rt.spec.key}: {rt.branch or 'NA'}",
                "01_provenance/branch_run_manifest.tsv",
                "verify branch naming or rerun stage",
            )
        if not rt.spec.optional and rt.spec.key != "ingest" and not rt.run_dir_exists:
            add_risk(
                risks,
                "warning",
                "provenance",
                f"Expected run directory is missing for {rt.spec.key}",
                "01_provenance/branch_run_manifest.tsv",
                "verify results root or run-stage output path",
            )

    if not dataset_inventory:
        add_risk(
            risks,
            "critical",
            "dataset",
            "Final dataset inventory is empty; final branch or dataset path may be wrong.",
            "02_dataset_state/final_dataset_inventory.tsv",
            "fix --dataset-root/--psr/--final-branch and regenerate synthesis",
        )

    total_deleted = sum(1 for r in action_rows if r.get("action") == "deleted")
    total_commented = sum(1 for r in action_rows if r.get("action") == "commented")
    if total_deleted == 0 and total_commented == 0:
        add_risk(
            risks,
            "warning",
            "toa_actions",
            "No commented or deleted TOA-like rows were detected by branch diffing. This may be valid, but it may also mean TIM parsing or branch mapping failed.",
            "06_toa_actions/toa_action_ledger.tsv",
            "check Step 5/6 reports and TIM diff assumptions",
        )

    for row in system_rows:
        frac = safe_float(row.get("deletion_fraction_vs_final_plus_deleted")) or 0.0
        n_deleted = safe_int(row.get("deleted_actions"))
        if n_deleted >= 5 and frac >= 0.20:
            add_risk(
                risks,
                "warning",
                "system_concentration",
                f"Deleted TOAs are concentrated in system {row.get('system')} ({n_deleted} deleted; fraction {frac:.3g}).",
                "05_qc_and_outliers/system_quality_matrix.tsv",
                "review system-specific deletion rationale",
            )

    sigma_max = safe_float(public_metrics.get("sigma_tension_max"))
    agreement_class = str(public_metrics.get("agreement_class", "")).lower()
    if sigma_max is not None:
        if sigma_max >= 5:
            add_risk(
                risks,
                "critical",
                "public_comparison",
                f"Public comparison has sigma_tension_max={sigma_max:.3g}, above the critical threshold of 5.",
                "09_public_comparison/public_synthesis.tsv",
                "resolve parameter disagreement before acceptance",
            )
        elif sigma_max >= 3:
            add_risk(
                risks,
                "warning",
                "public_comparison",
                f"Public comparison has sigma_tension_max={sigma_max:.3g}, above the warning threshold of 3.",
                "09_public_comparison/public_synthesis.tsv",
                "review public comparison tension",
            )
    if any(word in agreement_class for word in ("fail", "serious", "bad")):
        add_risk(
            risks,
            "critical",
            "public_comparison",
            f"Public comparison agreement_class is {public_metrics.get('agreement_class')!r}.",
            "09_public_comparison/public_synthesis.tsv",
            "resolve public comparison failure",
        )

    for row in qc_package.get("availability_rows", []):
        variant = str(row.get("variant", ""))
        n_rows = safe_int(row.get("n_rows"))
        n_numeric = safe_int(row.get("n_numeric_postfit"))
        if row.get("postfit_available") != "yes":
            severity = "critical" if variant == "combined" else "warning"
            add_risk(
                risks,
                severity,
                "postfit_review",
                f"{variant}: no numeric post-fit residual column was attached to the QC CSV.",
                "03_postfit_review/postfit_residual_availability.tsv",
                "fix tempo2/general2 attachment before trusting reviewer plots",
            )
            continue
        if n_rows and n_numeric < n_rows:
            frac = n_numeric / n_rows
            sev = "critical" if variant == "combined" and frac < 0.5 else "warning"
            add_risk(
                risks,
                sev,
                "postfit_review",
                f"{variant}: post-fit residual coverage is only {n_numeric}/{n_rows} ({frac:.1%}).",
                "03_postfit_review/postfit_residual_availability.tsv",
                "check variant/general2 matching before sign-off",
            )

    keep_rows = qc_package.get("surviving_keep_rows", [])
    if keep_rows:
        max_keep_sigma = max(
            [
                x
                for x in (safe_float(r.get("abs_sigma")) for r in keep_rows)
                if x is not None
            ],
            default=None,
        )
        if max_keep_sigma is not None:
            if max_keep_sigma >= 10:
                add_risk(
                    risks,
                    "critical",
                    "surviving_keep",
                    f"At least one reviewed KEEP row survives with |post-fit sigma|={max_keep_sigma:.3g}.",
                    "03_postfit_review/surviving_keep_outliers.tsv",
                    "inspect the worst surviving KEEP rows in the post-fit review tables and plots",
                )
            elif max_keep_sigma >= 5:
                add_risk(
                    risks,
                    "warning",
                    "surviving_keep",
                    f"At least one reviewed KEEP row survives with |post-fit sigma|={max_keep_sigma:.3g}.",
                    "03_postfit_review/surviving_keep_outliers.tsv",
                    "inspect the worst surviving KEEP rows in the post-fit review tables and plots",
                )

    if not risks:
        add_risk(
            risks,
            "green",
            "overall",
            "No automatic risks were detected by the synthesis script.",
            "all generated manifests",
            "human reviewer should still inspect the post-fit plots and decision sheet",
            status="closed",
        )
    return risks


def build_acceptance_checklist(
    runtimes: list[StageRuntime],
    dataset_inventory: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
    public_metric_rows: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    qc_package: dict[str, Any],
    model_package: dict[str, Any],
) -> list[dict[str, Any]]:
    criticals = [r for r in risks if severity_rank(str(r.get("severity", ""))) >= 2]
    warnings = [r for r in risks if severity_rank(str(r.get("severity", ""))) == 1]
    branch_ok = all(
        rt.branch_exists
        for rt in runtimes
        if rt.spec.creates_branch and not rt.spec.optional
    )
    run_ok = all(
        rt.run_dir_exists
        for rt in runtimes
        if not rt.spec.optional and rt.spec.key != "ingest"
    )
    final_rt = next(
        (rt for rt in runtimes if rt.spec.key == "step6_apply_delete"), None
    )
    postfit_rows = qc_package.get("availability_rows", [])
    postfit_ok = bool(postfit_rows) and all(
        r.get("postfit_available") == "yes" for r in postfit_rows
    )

    return [
        {
            "check": "Expected dataset branches exist",
            "status": "pass" if branch_ok else "fail",
            "evidence": "01_provenance/branch_run_manifest.tsv",
            "comment": (
                "Required branch refs resolved by git rev-parse."
                if branch_ok
                else "At least one required branch is missing."
            ),
        },
        {
            "check": "Expected run directories exist",
            "status": "pass" if run_ok else "warn",
            "evidence": "01_provenance/branch_run_manifest.tsv",
            "comment": (
                "Required run directories were found."
                if run_ok
                else "At least one required run directory was not found."
            ),
        },
        {
            "check": "Final branch resolved",
            "status": "pass" if bool(final_rt and final_rt.branch_exists) else "fail",
            "evidence": "01_provenance/branch_run_manifest.tsv",
            "comment": f"Final branch: {final_rt.branch if final_rt else 'UNKNOWN'}",
        },
        {
            "check": "Final dataset inventory is non-empty",
            "status": "pass" if bool(dataset_inventory) else "fail",
            "evidence": "02_dataset_state/final_dataset_inventory.tsv",
            "comment": f"Inventory rows: {len(dataset_inventory)}",
        },
        {
            "check": "TOA comment/delete ledger generated",
            "status": "pass" if bool(action_rows) else "warn",
            "evidence": "06_toa_actions/toa_action_ledger.tsv",
            "comment": f"Action rows: {len(action_rows)}",
        },
        {
            "check": "Post-fit reviewer plots generated",
            "status": "pass" if bool(qc_package.get("plot_rows")) else "fail",
            "evidence": "03_postfit_review/postfit_review.md",
            "comment": f"Plot rows: {len(qc_package.get('plot_rows', []))}",
        },
        {
            "check": "All reviewed variants have numeric post-fit residuals",
            "status": "pass" if postfit_ok else "warn",
            "evidence": "03_postfit_review/postfit_residual_availability.tsv",
            "comment": f"Variant availability rows: {len(postfit_rows)}",
        },
        {
            "check": "Residual summary available",
            "status": "pass" if bool(residual_rows) else "warn",
            "evidence": "07_timing_fit_quality/residual_comparison_across_branches.tsv",
            "comment": f"Residual rows: {len(residual_rows)}",
        },
        {
            "check": "Public comparison available or explicitly absent",
            "status": "pass" if public_metric_rows else "warn",
            "evidence": "09_public_comparison/public_synthesis.tsv",
            "comment": (
                "Public comparison metrics found."
                if public_metric_rows
                else "Public comparison output not found; this may be valid if Step 8 was not run."
            ),
        },
        {
            "check": "Binary / param-scan / change-report artifacts indexed",
            "status": "pass" if model_package.get("artifact_index_rows") else "warn",
            "evidence": "10_model_checks/model_check_artifact_index.tsv",
            "comment": (
                f"Binary rows: {len(model_package.get('binary_rows', []))}; "
                f"param-scan rows: {len(model_package.get('param_scan_rows', []))}; "
                f"change-report rows: {len(model_package.get('change_model_rows', [])) + len(model_package.get('new_param_rows', []))}"
            ),
        },
        {
            "check": "No automatic critical risks",
            "status": "pass" if not criticals else "fail",
            "evidence": "00_decision/risk_register.tsv",
            "comment": f"Critical risks: {len(criticals)}; warnings: {len(warnings)}",
        },
    ]


def auto_decision(risks: list[dict[str, Any]]) -> str:
    if any(severity_rank(str(r.get("severity", ""))) >= 2 for r in risks):
        return "REJECT_OR_RERUN_REQUIRED"
    if any(severity_rank(str(r.get("severity", ""))) == 1 for r in risks):
        return "NEEDS_MANUAL_REVIEW"
    return "ACCEPT_CANDIDATE_FOR_HUMAN_SIGNOFF"


def md_table(rows: list[dict[str, Any]], fields: list[str], max_rows: int = 20) -> str:
    if not rows:
        return "_No rows._\n"
    shown = rows[:max_rows]
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [header, sep]
    for row in shown:
        vals = [str(row.get(f, "")).replace("|", "\\|") for f in fields]
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(rows)} rows._")
    return "\n".join(lines) + "\n"


def generate_decision_sheet(
    *,
    args: argparse.Namespace,
    runtimes: list[StageRuntime],
    repo_root: Path,
    dataset_root: Path,
    decision: str,
    dataset_inventory: list[dict[str, Any]],
    diff_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    system_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
    public_metrics: dict[str, Any],
    risks: list[dict[str, Any]],
    checklist: list[dict[str, Any]],
    qc_package: dict[str, Any],
    model_package: dict[str, Any],
) -> str:
    final_rt = next(
        (rt for rt in runtimes if rt.spec.key == "step6_apply_delete"), None
    )
    total_deleted = sum(1 for r in action_rows if r.get("action") == "deleted")
    total_commented = sum(1 for r in action_rows if r.get("action") == "commented")
    criticals = [r for r in risks if severity_rank(str(r.get("severity", ""))) >= 2]
    warnings = [r for r in risks if severity_rank(str(r.get("severity", ""))) == 1]
    variant_rows = qc_package.get("variant_summary_rows", [])

    public_lines = []
    for k in (
        "agreement_class",
        "sigma_tension_max",
        "sigma_tension_mean",
        "reduced_chi2",
        "worst_provider_pair",
    ):
        if k in public_metrics:
            public_lines.append(f"- **{k}:** {public_metrics[k]}")
    if not public_lines:
        public_lines.append(
            "- Public comparison metrics were not found in the expected Step 8 location."
        )

    warning_lines = qc_package.get("warnings", [])
    warning_block = (
        "\n".join(f"- {line}" for line in warning_lines) if warning_lines else "- None"
    )

    return f"""# Final data-quality decision sheet

Generated: {now_iso()}  
Pulsar: **{args.psr}**  
Workflow config: `{args.workflow_config or 'NA'}`  
Dataset root: `{clean_rel(dataset_root, repo_root)}`  
Results root: `{clean_rel(Path(args.results_root), repo_root)}`  
Git working tree status at synthesis time: **{git_is_dirty(repo_root)}**

## Automatic synthesis decision

**{decision}**

This is not a scientific sign-off. It is an automatic reviewer-facing synthesis based on discovered branches, artifacts, TIM diffs, QC tables, and post-fit residual plots.

## Final branch

- Final branch: `{final_rt.branch if final_rt else args.final_branch}`
- Final commit: `{final_rt.commit if final_rt else 'UNKNOWN'}`
- Final dataset inventory rows: **{len(dataset_inventory)}**

## Reviewer starting points

1. `03_postfit_review/postfit_review.md`
2. `03_postfit_review/surviving_keep_outliers.tsv`
3. `06_toa_actions/toa_action_ledger.tsv`
4. `05_qc_and_outliers/backend_flag_summary.tsv`
5. `10_model_checks/model_checks.md`

## Post-fit residual review summary

{md_table(variant_rows, ['variant', 'selected_postfit_column', 'n_postfit', 'n_keep', 'n_bad_toa', 'n_event', 'n_review_event', 'max_abs_sigma_keep', 'worst_backend_by_flagged_fraction'], max_rows=10)}

### QC warnings

{warning_block}

## TOA actions detected by branch diffing

- Commented TOA-like rows detected: **{total_commented}**
- Deleted TOA-like rows detected: **{total_deleted}**

See `06_toa_actions/toa_action_ledger.tsv` for row-level evidence.

## Public comparison summary

{chr(10).join(public_lines)}

## Model-check summary

- Binary-analysis rows: **{len(model_package.get('binary_rows', []))}**
- Param-scan summary rows: **{len(model_package.get('param_scan_rows', []))}**
- Change-report model rows: **{len(model_package.get('change_model_rows', []))}**
- New-parameter significance rows: **{len(model_package.get('new_param_rows', []))}**

See `10_model_checks/model_checks.md`.

## Risk summary

- Critical risks: **{len(criticals)}**
- Warning risks: **{len(warnings)}**
- Total risk-register rows: **{len(risks)}**

{md_table(risks, ['risk_id', 'severity', 'area', 'description', 'required_action'], max_rows=10)}

## Acceptance checklist

{md_table(checklist, ['check', 'status', 'comment', 'evidence'], max_rows=20)}

## Dataset transition summary

{md_table(diff_rows, ['transition', 'from_branch', 'to_branch', 'from_active_toas', 'to_active_toas', 'comments_added_detected', 'deletions_detected', 'par_files_changed', 'tim_files_changed'], max_rows=10)}

## Systems with TOA actions

{md_table([r for r in system_rows if safe_int(r.get('commented_actions')) or safe_int(r.get('deleted_actions'))], ['system', 'final_active_toas', 'commented_actions', 'deleted_actions', 'deletion_fraction_vs_final_plus_deleted', 'auto_status'], max_rows=15)}

## Residual synthesis availability

Residual summary rows collected: **{len(residual_rows)}**.  
See `07_timing_fit_quality/residual_comparison_across_branches.tsv`.

## Reviewer sign-off

Reviewer:  
Date:  
Decision: ACCEPT / ACCEPT WITH CAVEATS / NEEDS RERUN / REJECT  
Notes:
"""


def generate_index(
    *,
    args: argparse.Namespace,
    decision: str,
    runtimes: list[StageRuntime],
    risks: list[dict[str, Any]],
) -> str:
    stage_rows = [
        {
            "stage": rt.spec.key,
            "branch": rt.branch or "NA",
            "branch_exists": "yes" if rt.branch_exists else "no",
            "run_dir_exists": "yes" if rt.run_dir_exists else "no",
            "commit": rt.commit[:12] if rt.commit != "UNKNOWN" else "UNKNOWN",
        }
        for rt in runtimes
    ]
    return f"""# Review synthesis package: {args.psr}

Generated: {now_iso()}  
Automatic synthesis decision: **{decision}**

Start here:

1. `03_postfit_review/postfit_review.md`
2. `00_decision/final_data_quality_decision_sheet.md`
3. `03_postfit_review/surviving_keep_outliers.tsv`
4. `10_model_checks/model_checks.md`
5. `00_decision/risk_register.tsv`
6. `06_toa_actions/toa_action_ledger.tsv`
7. `01_provenance/artifact_manifest.tsv`

## Stage map

{md_table(stage_rows, ['stage', 'branch', 'branch_exists', 'run_dir_exists', 'commit'], max_rows=20)}

## Open risks

{md_table([r for r in risks if r.get('status') == 'open'], ['risk_id', 'severity', 'area', 'description', 'required_action'], max_rows=20)}

## Raw evidence links

The `raw_links/` directory contains symlinks, or `.path.txt` fallbacks, pointing back to existing ingest reports and run directories.
"""


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a reviewer-facing synthesis package for one pulsar workflow."
    )
    p.add_argument("--psr", required=True, help="Pulsar name, e.g. J1909-3744")
    p.add_argument(
        "--slug", required=True, help="Workflow slug used in results dirs, e.g. j1909"
    )
    p.add_argument(
        "--workflow-config",
        default="",
        help="Path to workflow TOML config, recorded for provenance",
    )
    p.add_argument("--repo-root", default=".", help="Dataset Git repository root")
    p.add_argument(
        "--dataset-root", required=True, help="Path to EPTA-DR3/epta-dr3-data"
    )
    p.add_argument("--results-root", required=True, help="Path to results/")
    p.add_argument(
        "--final-branch",
        required=True,
        help="Final dataset branch, usually <slug>_step6_apply_delete",
    )
    p.add_argument("--out", required=True, help="Output review package directory")
    p.add_argument(
        "--overrides",
        default="",
        help="Optional manual_qc_overrides.csv; defaults to <qc-run>/qc_review/manual_qc_overrides.csv",
    )
    p.add_argument(
        "--max-keep-points",
        type=int,
        default=4000,
        help="Maximum KEEP points to render per plot before sampling.",
    )
    p.add_argument(
        "--top-n-rows",
        type=int,
        default=50,
        help="Number of top suspicious rows to keep in TSV summaries.",
    )
    p.add_argument(
        "--stage-branch",
        action="append",
        default=[],
        help="Override stage branch as STAGE=BRANCH. Can be repeated.",
    )
    p.add_argument(
        "--stage-run",
        action="append",
        default=[],
        help="Override stage run directory as STAGE=PATH. Can be repeated.",
    )
    p.add_argument("--no-pdf", action="store_true", help="Skip the PDF build.")
    p.add_argument(
        "--pdf-name",
        default="",
        help="PDF filename written inside --out. Defaults to <psr>_review_synthesis.pdf.",
    )
    p.add_argument(
        "--pandoc",
        default="pandoc",
        help="Ignored legacy option; native PDF generation no longer uses pandoc.",
    )
    p.add_argument(
        "--pdf-engine",
        default="xelatex",
        help="Ignored legacy option; native PDF generation no longer uses an external PDF engine.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = (repo_root / dataset_root).resolve()
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = (repo_root / results_root).resolve()
    out_root = Path(args.out).resolve()
    dirs = mkdirs(out_root)

    if not git_available(repo_root):
        print(
            f"WARNING: {repo_root} is not a Git working tree. Git-derived provenance will be UNKNOWN.",
            file=sys.stderr,
        )

    runtimes = build_stage_runtimes(args, repo_root, results_root)
    artifact_rows = discover_artifacts(runtimes, dataset_root, repo_root)
    make_raw_links(runtimes, dataset_root, dirs["raw_links"], repo_root)

    branch_manifest_rows = []
    for rt in runtimes:
        branch_manifest_rows.append(
            {
                "stage": rt.spec.key,
                "label": rt.spec.label,
                "creates_dataset_branch": "yes" if rt.spec.creates_branch else "no",
                "optional": "yes" if rt.spec.optional else "no",
                "branch": rt.branch or "NA",
                "branch_exists": "yes" if rt.branch_exists else "no",
                "commit": rt.commit,
                "run_dir": clean_rel(rt.run_dir, repo_root) if rt.run_dir else "NA",
                "run_dir_exists": "yes" if rt.run_dir_exists else "no",
                "run_report": (
                    clean_rel(rt.run_dir / "run_report.pdf", repo_root)
                    if rt.run_dir
                    else "NA"
                ),
                "workflow_report": (
                    clean_rel(rt.run_dir / "workflow_report.pdf", repo_root)
                    if rt.run_dir
                    else "NA"
                ),
                "notes": rt.notes,
            }
        )

    dataset_inventory = collect_dataset_inventory(
        repo_root, dataset_root, args.psr, args.final_branch
    )
    diff_rows, action_rows, snapshots = collect_branch_diffs(
        repo_root, dataset_root, args.psr, runtimes
    )
    final_snapshot = snapshots.get("step6_apply_delete", {})
    system_rows = collect_system_quality_matrix(final_snapshot, action_rows)
    qc_index_rows = collect_qc_index(artifact_rows)
    residual_rows = collect_residual_synthesis(runtimes, repo_root)
    model_package = build_model_check_package(
        runtimes=runtimes,
        artifact_rows=artifact_rows,
        repo_root=repo_root,
        model_dir=dirs["model_checks"],
    )

    public_rt = next(
        (rt for rt in runtimes if rt.spec.key == "step8_compare_public"), None
    )
    public_metric_rows, public_tension_rows, public_metrics = extract_public_metrics(
        public_rt.run_dir if public_rt else None, repo_root
    )

    qc_stage = choose_qc_review_stage(runtimes)
    qc_package = build_qc_review_package(
        args=args, repo_root=repo_root, qc_stage=qc_stage, postfit_dir=dirs["postfit"]
    )

    risks = build_risk_register(
        runtimes,
        dataset_inventory,
        diff_rows,
        action_rows,
        system_rows,
        public_metrics,
        qc_package,
    )
    checklist = build_acceptance_checklist(
        runtimes,
        dataset_inventory,
        action_rows,
        residual_rows,
        public_metric_rows,
        risks,
        qc_package,
        model_package,
    )
    decision = auto_decision(risks)

    write_tsv(
        dirs["provenance"] / "branch_run_manifest.tsv",
        branch_manifest_rows,
        [
            "stage",
            "label",
            "creates_dataset_branch",
            "optional",
            "branch",
            "branch_exists",
            "commit",
            "run_dir",
            "run_dir_exists",
            "run_report",
            "workflow_report",
            "notes",
        ],
    )
    write_tsv(
        dirs["provenance"] / "artifact_manifest.tsv",
        artifact_rows,
        [
            "stage",
            "category",
            "artifact_type",
            "priority",
            "path",
            "exists",
            "size_bytes",
            "row_count",
        ],
    )
    write_tsv(
        dirs["dataset"] / "final_dataset_inventory.tsv",
        dataset_inventory,
        ["branch", "path", "file_name", "suffix", "size_bytes", "line_count", "sha256"],
    )
    write_tsv(
        dirs["lineage"] / "dataset_diff_summary.tsv",
        diff_rows,
        [
            "transition",
            "from_branch",
            "to_branch",
            "from_active_toas",
            "to_active_toas",
            "from_commented_toas",
            "to_commented_toas",
            "active_toa_delta",
            "comments_added_detected",
            "deletions_detected",
            "par_files_changed",
            "tim_files_changed",
            "status",
        ],
    )
    write_tsv(
        dirs["toa"] / "toa_action_ledger.tsv",
        action_rows,
        [
            "toa_id",
            "transition",
            "from_branch",
            "to_branch",
            "relative_tim_file",
            "mjd",
            "freq",
            "observatory",
            "system",
            "backend",
            "telescope",
            "action",
            "final_state",
            "normalized_toa_line_sha1",
            "evidence",
        ],
    )
    write_tsv(
        dirs["qc"] / "qc_artifact_index.tsv",
        qc_index_rows,
        ["stage", "artifact_type", "priority", "path", "row_count"],
    )
    write_tsv(
        dirs["qc"] / "system_quality_matrix.tsv",
        system_rows,
        [
            "system",
            "final_active_toas",
            "final_commented_toas",
            "commented_actions",
            "deleted_actions",
            "deletion_fraction_vs_final_plus_deleted",
            "auto_status",
        ],
    )
    write_tsv(
        dirs["timing"] / "residual_comparison_across_branches.tsv",
        residual_rows,
        [
            "stage",
            "branch",
            "source_path",
            "row_index",
            "variant",
            "n_toas",
            "rms_or_wrms",
            "reduced_chi2",
            "fit_status",
            "raw_columns_json",
        ],
    )
    write_tsv(
        dirs["public"] / "public_synthesis.tsv",
        public_metric_rows,
        ["metric", "value", "source_path"],
    )
    write_tsv(
        dirs["public"] / "public_parameter_tension_table.tsv",
        public_tension_rows,
        [
            "parameter",
            "sigma_tension",
            "provider_pair",
            "agreement_class",
            "source_path",
            "raw_columns_json",
        ],
    )
    write_tsv(
        dirs["model_checks"] / "model_check_artifact_index.tsv",
        model_package.get("artifact_index_rows", []),
        ["stage", "artifact_type", "priority", "path", "row_count"],
    )
    write_tsv(
        dirs["model_checks"] / "binary_analysis_summary.tsv",
        model_package.get("binary_rows", []),
        [
            "stage",
            "pulsar",
            "branch",
            "BINARY",
            "PB",
            "A1",
            "ECC",
            "EPS1",
            "EPS2",
            "T0",
            "TASC",
            "source_path",
            "raw_columns_json",
        ],
    )
    write_tsv(
        dirs["model_checks"] / "param_scan_summary.tsv",
        model_package.get("param_scan_rows", []),
        [
            "stage",
            "pulsar",
            "branch",
            "candidate",
            "redchisq",
            "delta_k_fit",
            "lrt_delta_chisq",
            "lrt_p_value",
            "max_param_z",
            "source_path",
            "raw_columns_json",
        ],
    )
    write_tsv(
        dirs["model_checks"] / "change_report_model_summary.tsv",
        model_package.get("change_model_rows", []),
        [
            "stage",
            "pulsar",
            "branch",
            "reference",
            "ref_k_fit",
            "br_k_fit",
            "delta_redchisq",
            "delta_wrms_post",
            "delta_aic",
            "delta_bic",
            "delta_k_fit",
            "lrt_delta_chisq",
            "lrt_p_value",
            "source_path",
            "raw_columns_json",
        ],
    )
    write_tsv(
        dirs["model_checks"] / "change_report_new_param_summary.tsv",
        model_package.get("new_param_rows", []),
        [
            "stage",
            "pulsar",
            "branch",
            "reference",
            "n_new_params",
            "n_new_with_numeric_sigma",
            "n_new_sig_z",
            "n_new_sig_threshold",
            "max_new_param_z",
            "max_new_param",
            "source_path",
            "raw_columns_json",
        ],
    )
    write_tsv(
        dirs["postfit"] / "postfit_residual_availability.tsv",
        qc_package.get("availability_rows", []),
        [
            "variant",
            "qc_csv",
            "postfit_available",
            "selected_residual_column",
            "selected_error_column",
            "n_rows",
            "n_numeric_postfit",
            "n_numeric_error",
            "available_residual_columns",
            "warning",
        ],
    )
    write_tsv(
        dirs["postfit"] / "variant_postfit_summary.tsv",
        qc_package.get("variant_summary_rows", []),
        [
            "variant",
            "qc_csv",
            "selected_postfit_column",
            "selected_error_column",
            "n_rows",
            "n_postfit",
            "n_keep",
            "n_bad_toa",
            "n_event",
            "n_review_event",
            "n_manual_override_rows",
            "median_abs_postfit_us",
            "p95_abs_postfit_us",
            "max_abs_postfit_us",
            "median_abs_sigma",
            "p95_abs_sigma",
            "max_abs_sigma",
            "max_abs_sigma_keep",
            "worst_backend_by_flagged_fraction",
            "worst_backend_flagged_fraction",
        ],
    )
    write_tsv(
        dirs["qc"] / "backend_flag_summary.tsv",
        qc_package.get("backend_rows", []),
        [
            "variant",
            "backend",
            "n_rows",
            "n_postfit",
            "n_flagged",
            "n_manual",
            "flagged_fraction",
            "median_abs_postfit_us",
            "max_abs_postfit_us",
        ],
    )
    write_tsv(
        dirs["postfit"] / "worst_postfit_rows.tsv",
        qc_package.get("worst_rows", []),
        [
            "variant",
            "review_id",
            "decision",
            "manual_action",
            "backend",
            "timfile",
            "mjd",
            "freq_mhz",
            "postfit_us",
            "tempo2_err_us",
            "abs_sigma",
            "bad_point",
            "event_member",
            "transient_id",
            "qc_csv",
        ],
    )
    write_tsv(
        dirs["postfit"] / "surviving_keep_outliers.tsv",
        qc_package.get("surviving_keep_rows", []),
        [
            "variant",
            "review_id",
            "backend",
            "timfile",
            "mjd",
            "freq_mhz",
            "postfit_us",
            "tempo2_err_us",
            "abs_sigma",
            "manual_action",
            "bad_point",
            "event_member",
            "transient_id",
            "qc_csv",
        ],
    )
    write_tsv(
        dirs["postfit"] / "plot_manifest.tsv",
        qc_package.get("plot_rows", []),
        ["variant", "plot_type", "path", "source_qc_csv"],
    )
    write_tsv(
        dirs["decision"] / "risk_register.tsv",
        risks,
        [
            "risk_id",
            "severity",
            "area",
            "description",
            "evidence",
            "required_action",
            "status",
        ],
    )
    write_tsv(
        dirs["decision"] / "acceptance_checklist.tsv",
        checklist,
        ["check", "status", "evidence", "comment"],
    )

    metrics = {
        "generated_at": now_iso(),
        "psr": args.psr,
        "slug": args.slug,
        "workflow_config": args.workflow_config,
        "final_branch": args.final_branch,
        "final_commit": next(
            (rt.commit for rt in runtimes if rt.spec.key == "step6_apply_delete"),
            "UNKNOWN",
        ),
        "auto_decision": decision,
        "dataset_inventory_rows": len(dataset_inventory),
        "toa_actions_detected": len(action_rows),
        "toas_commented_detected": sum(
            1 for r in action_rows if r.get("action") == "commented"
        ),
        "toas_deleted_detected": sum(
            1 for r in action_rows if r.get("action") == "deleted"
        ),
        "risk_counts": dict(Counter(str(r.get("severity", "")) for r in risks)),
        "public_metrics": public_metrics,
        "qc_review_warnings": qc_package.get("warnings", []),
        "model_check_counts": {
            "artifact_index_rows": len(model_package.get("artifact_index_rows", [])),
            "binary_rows": len(model_package.get("binary_rows", [])),
            "param_scan_rows": len(model_package.get("param_scan_rows", [])),
            "change_model_rows": len(model_package.get("change_model_rows", [])),
            "new_param_rows": len(model_package.get("new_param_rows", [])),
        },
    }
    write_json(dirs["machine"] / "metrics.json", metrics)
    write_json(
        dirs["machine"] / "review_package_manifest.json",
        {"stages": branch_manifest_rows, "artifacts": artifact_rows},
    )

    decision_sheet = generate_decision_sheet(
        args=args,
        runtimes=runtimes,
        repo_root=repo_root,
        dataset_root=dataset_root,
        decision=decision,
        dataset_inventory=dataset_inventory,
        diff_rows=diff_rows,
        action_rows=action_rows,
        system_rows=system_rows,
        residual_rows=residual_rows,
        public_metrics=public_metrics,
        risks=risks,
        checklist=checklist,
        qc_package=qc_package,
        model_package=model_package,
    )
    write_text(
        dirs["decision"] / "final_data_quality_decision_sheet.md", decision_sheet
    )
    write_text(
        out_root / "index.md",
        generate_index(args=args, decision=decision, runtimes=runtimes, risks=risks),
    )

    step7 = next((rt for rt in runtimes if rt.spec.key == "step7_whitenoise"), None)
    if not step7 or not step7.run_dir_exists:
        write_text(
            dirs["noise"] / "whitenoise_not_found.md",
            "# Whitenoise synthesis\n\nStep 7 whitenoise output was not found in the expected location.\n",
        )

    pdf_path: Optional[Path] = None
    if not args.no_pdf:
        try:
            pdf_path = build_pdf_with_matplotlib(
                args=args, out_root=out_root, decision=decision
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    print(f"Wrote review synthesis package: {out_root}")
    if pdf_path is not None:
        print(f"Wrote PDF: {pdf_path}")
    print(f"Automatic synthesis decision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
