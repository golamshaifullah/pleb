#!/usr/bin/env python3
"""Build a compact release tree from PLEB output directories.

Key behaviors:
- stage into a fresh release directory (never edits source tree)
- copy ``data_source`` as ``raw_data_source``
- copy ``EPTA-DR3`` while skipping nested ``results`` folders
- copy selected run folders from ``results``
- harvest existing PDFs into ``results/reports`` with hash deduplication
- prune bulky intermediates (PNG/general2/covmat/log by default)
- emit audit tables:
  - ``results/reports/REPORT_INDEX.tsv``
  - ``results/manifests/release_manifest.tsv``
  - ``results/manifests/size_report.tsv``

Dry-run is the default. Use ``--apply`` to write files.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_EXCLUDE_RUN_REGEX = r"(debug|retry|test|trial)"
DEFAULT_DROP_GLOBS = ("**/*.png", "**/*.general2", "**/*.covmat", "**/*.log")


@dataclass(frozen=True)
class FileStat:
    files: int
    bytes: int


@dataclass(frozen=True)
class RunSelection:
    selected: list[Path]
    skipped_excluded: list[Path]
    skipped_family: list[Path]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a compact release from PLEB outputs (dry-run by default)."
    )
    p.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root containing data_source, EPTA-DR3, and results.",
    )
    p.add_argument(
        "--output-parent",
        type=Path,
        default=None,
        help="Parent directory where release directory is created (default: source parent).",
    )
    p.add_argument(
        "--release-name",
        type=str,
        default=None,
        help="Release directory name (default: release_prod_YYYYmmdd_HHMMSS).",
    )
    p.add_argument(
        "--results-rel",
        type=str,
        default="results",
        help="Relative results directory under source-root (default: results).",
    )
    p.add_argument(
        "--data-source-rel",
        type=str,
        default="data_source",
        help="Relative data source directory under source-root (default: data_source).",
    )
    p.add_argument(
        "--epta-rel",
        type=str,
        default="EPTA-DR3",
        help="Relative EPTA-DR3 directory under source-root (default: EPTA-DR3).",
    )
    p.add_argument(
        "--selection-mode",
        choices=("latest_per_family", "all"),
        default="latest_per_family",
        help="Run-folder selection policy for results subdirs.",
    )
    p.add_argument(
        "--keep-per-family",
        type=int,
        default=1,
        help="When in latest_per_family mode, retain N newest per normalized family.",
    )
    p.add_argument(
        "--include-run-glob",
        action="append",
        default=[],
        help="Optional run name glob(s) to include (can repeat).",
    )
    p.add_argument(
        "--exclude-run-regex",
        type=str,
        default=DEFAULT_EXCLUDE_RUN_REGEX,
        help="Regex used to exclude run names before selection.",
    )
    p.add_argument(
        "--drop-glob",
        action="append",
        default=list(DEFAULT_DROP_GLOBS),
        help="Glob pattern(s) to prune inside copied run dirs (can repeat).",
    )
    p.add_argument(
        "--merge-report-pdf",
        action="store_true",
        help="Attempt to merge harvested PDFs into results/reports/00_release_summary.pdf.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Perform the copy/build actions. Default is dry-run only.",
    )
    return p.parse_args(argv)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.1f}{u}"
        x /= 1024.0
    return f"{n}B"


def compute_tree_stats(root: Path, exclude_dirs: set[str] | None = None) -> FileStat:
    files = 0
    total = 0
    exclude_dirs = exclude_dirs or set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            files += 1
            total += int(st.st_size)
    return FileStat(files=files, bytes=total)


def normalize_family(run_name: str) -> str:
    x = run_name
    x = re.sub(r"_\d{8}T\d{4,6}$", "", x)
    x = re.sub(r"_v\d+(?:_\d+)*$", "", x)
    return x


def discover_run_dirs(results_root: Path) -> list[Path]:
    if not results_root.is_dir():
        return []
    return sorted([p for p in results_root.iterdir() if p.is_dir()])


def select_runs(
    run_dirs: Sequence[Path],
    include_globs: Sequence[str],
    exclude_re: re.Pattern[str],
    mode: str,
    keep_per_family: int,
) -> RunSelection:
    filtered: list[Path] = []
    skipped_excluded: list[Path] = []
    skipped_family: list[Path] = []

    for run_dir in run_dirs:
        name = run_dir.name
        if include_globs:
            if not any(fnmatch.fnmatch(name, pat) for pat in include_globs):
                skipped_excluded.append(run_dir)
                continue
        elif exclude_re.search(name):
            skipped_excluded.append(run_dir)
            continue
        filtered.append(run_dir)

    if mode == "all":
        return RunSelection(
            selected=sorted(filtered),
            skipped_excluded=sorted(skipped_excluded),
            skipped_family=[],
        )

    buckets: dict[str, list[Path]] = {}
    for run_dir in filtered:
        buckets.setdefault(normalize_family(run_dir.name), []).append(run_dir)

    selected: list[Path] = []
    for _, items in sorted(buckets.items()):
        ordered = sorted(items, key=lambda p: p.stat().st_mtime, reverse=True)
        selected.extend(ordered[:keep_per_family])
        skipped_family.extend(ordered[keep_per_family:])

    return RunSelection(
        selected=sorted(selected),
        skipped_excluded=sorted(skipped_excluded),
        skipped_family=sorted(skipped_family),
    )


def path_matches_any_glob(rel: Path, globs: Sequence[str]) -> bool:
    text = rel.as_posix()
    return any(fnmatch.fnmatch(text, pat) for pat in globs)


def copy_tree(src: Path, dst: Path, ignore_names: set[str] | None = None) -> None:
    ignore_names = ignore_names or set()

    def ignore_fn(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names if n in ignore_names}

    shutil.copytree(src, dst, ignore=ignore_fn, dirs_exist_ok=True)


def gather_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            out.append(Path(dirpath) / name)
    return out


def prune_files(root: Path, drop_globs: Sequence[str]) -> tuple[int, int]:
    removed_files = 0
    removed_bytes = 0
    for p in gather_files(root):
        rel = p.relative_to(root)
        if path_matches_any_glob(rel, drop_globs):
            try:
                size = p.stat().st_size
            except FileNotFoundError:
                continue
            p.unlink(missing_ok=True)
            removed_files += 1
            removed_bytes += int(size)

    # clean empty dirs bottom-up
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if dirnames or filenames:
            continue
        d = Path(dirpath)
        try:
            d.rmdir()
        except OSError:
            pass
    return removed_files, removed_bytes


def write_tsv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def safe_pdf_name(rel_pdf: Path) -> str:
    stem = "__".join(rel_pdf.parts)
    stem = re.sub(r"[^A-Za-z0-9._+-]", "_", stem)
    if len(stem) > 220:
        stem = stem[:220]
    return stem


def harvest_pdfs(results_root: Path) -> tuple[list[dict[str, object]], dict[str, Path]]:
    reports_dir = results_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, Path] = {}
    rows: list[dict[str, object]] = []

    pdfs = sorted(
        p
        for p in results_root.rglob("*.pdf")
        if reports_dir not in p.parents and p.parent != reports_dir
    )
    for pdf in pdfs:
        rel = pdf.relative_to(results_root)
        digest = sha256_file(pdf)
        size = pdf.stat().st_size
        duplicate_of = ""
        if digest in hashes:
            canonical = hashes[digest]
            duplicate_of = canonical.relative_to(results_root).as_posix()
        else:
            out_name = safe_pdf_name(rel)
            if not out_name.lower().endswith(".pdf"):
                out_name += ".pdf"
            out_path = reports_dir / out_name
            suffix = 1
            while out_path.exists():
                out_path = reports_dir / f"{Path(out_name).stem}_{suffix}.pdf"
                suffix += 1
            shutil.copy2(pdf, out_path)
            hashes[digest] = out_path

        rows.append(
            {
                "source_rel": rel.as_posix(),
                "sha256": digest,
                "size_bytes": int(size),
                "duplicate_of": duplicate_of,
                "canonical_report_rel": hashes[digest]
                .relative_to(results_root)
                .as_posix(),
            }
        )
    return rows, hashes


def merge_pdf_reports(results_root: Path, canonical_map: dict[str, Path]) -> str:
    if not canonical_map:
        return "skip:no_reports"
    try:
        from pypdf import PdfMerger  # type: ignore
    except Exception:
        return "skip:pypdf_missing"
    out = results_root / "reports" / "00_release_summary.pdf"
    merger = PdfMerger()
    for p in sorted(canonical_map.values()):
        merger.append(str(p))
    with out.open("wb") as f:
        merger.write(f)
    merger.close()
    return "ok"


def write_release_manifest(release_root: Path) -> Path:
    rows = []
    for p in sorted(gather_files(release_root)):
        rel = p.relative_to(release_root)
        rows.append(
            {
                "rel_path": rel.as_posix(),
                "size_bytes": int(p.stat().st_size),
                "sha256": sha256_file(p),
            }
        )
    out = release_root / "results" / "manifests" / "release_manifest.tsv"
    write_tsv(out, ("rel_path", "size_bytes", "sha256"), rows)
    return out


def write_size_report(
    out_path: Path,
    sections: Sequence[tuple[str, FileStat]],
) -> None:
    rows = [
        {
            "section": name,
            "files": stat.files,
            "bytes": stat.bytes,
            "human_bytes": human_bytes(stat.bytes),
        }
        for name, stat in sections
    ]
    write_tsv(out_path, ("section", "files", "bytes", "human_bytes"), rows)


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"Missing required directory for {label}: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    source_root = args.source_root.expanduser().resolve()
    output_parent = (
        args.output_parent.expanduser().resolve()
        if args.output_parent is not None
        else source_root.parent
    )
    release_name = args.release_name or f"release_prod_{now_stamp()}"
    release_root = output_parent / release_name

    data_source_src = source_root / args.data_source_rel
    epta_src = source_root / args.epta_rel
    results_src = source_root / args.results_rel
    if not results_src.is_dir():
        alt = epta_src / args.results_rel
        if alt.is_dir():
            results_src = alt

    require_dir(data_source_src, "data_source")
    require_dir(epta_src, "EPTA-DR3")
    require_dir(results_src, "results")

    run_dirs = discover_run_dirs(results_src)
    exclude_re = re.compile(args.exclude_run_regex, re.IGNORECASE)
    selection = select_runs(
        run_dirs=run_dirs,
        include_globs=args.include_run_glob,
        exclude_re=exclude_re,
        mode=args.selection_mode,
        keep_per_family=max(1, int(args.keep_per_family)),
    )

    # Planning stats from source tree.
    data_stat = compute_tree_stats(data_source_src)
    epta_stat = compute_tree_stats(epta_src, exclude_dirs={"results"})
    selected_run_stats = {run: compute_tree_stats(run) for run in selection.selected}
    selected_total = FileStat(
        files=sum(s.files for s in selected_run_stats.values()),
        bytes=sum(s.bytes for s in selected_run_stats.values()),
    )

    drop_globs = tuple(dict.fromkeys(args.drop_glob))
    drop_files = 0
    drop_bytes = 0
    for run in selection.selected:
        for p in gather_files(run):
            rel = p.relative_to(run)
            if path_matches_any_glob(rel, drop_globs):
                drop_files += 1
                drop_bytes += int(p.stat().st_size)

    print("== Release Plan ==")
    print(f"source_root:     {source_root}")
    print(f"results_source:  {results_src}")
    print(f"release_root:    {release_root}")
    print(f"mode:            {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"selection_mode:  {args.selection_mode}")
    print(f"selected_runs:   {len(selection.selected)}")
    print(f"excluded_runs:   {len(selection.skipped_excluded)}")
    print(f"family_skipped:  {len(selection.skipped_family)}")
    print(f"drop_globs:      {', '.join(drop_globs)}")
    print()
    print("Source footprint to stage:")
    print(
        f"- raw_data_source (from data_source): files={data_stat.files} size={human_bytes(data_stat.bytes)}"
    )
    print(
        f"- EPTA-DR3 (excluding nested results): files={epta_stat.files} size={human_bytes(epta_stat.bytes)}"
    )
    print(
        f"- selected results runs: files={selected_total.files} size={human_bytes(selected_total.bytes)}"
    )
    print(
        f"- planned prune inside copied runs: files={drop_files} size={human_bytes(drop_bytes)}"
    )

    if selection.selected:
        print("\nSelected run directories:")
        for run in selection.selected:
            print(f"- {run.name}")
    else:
        print(
            "\nNo run directories selected; adjust --include-run-glob / --exclude-run-regex."
        )

    if not args.apply:
        print("\nDry-run complete. Re-run with --apply to build the release.")
        return 0

    if release_root.exists():
        raise SystemExit(
            f"Release root already exists, refusing to overwrite: {release_root}"
        )

    # Stage copy.
    (release_root / "results").mkdir(parents=True, exist_ok=False)
    copy_tree(data_source_src, release_root / "raw_data_source")
    copy_tree(epta_src, release_root / "EPTA-DR3", ignore_names={"results"})

    staged_results_root = release_root / "results"
    for run in selection.selected:
        copy_tree(run, staged_results_root / run.name)

    # Keep release metadata if present.
    rel_info_src = source_root / "RELEASE_INFO.txt"
    if rel_info_src.exists():
        shutil.copy2(rel_info_src, release_root / "RELEASE_INFO.txt")

    # Harvest + dedupe existing PDFs before pruning.
    report_rows, canonical_reports = harvest_pdfs(staged_results_root)
    report_index = staged_results_root / "reports" / "REPORT_INDEX.tsv"
    write_tsv(
        report_index,
        ("source_rel", "sha256", "size_bytes", "duplicate_of", "canonical_report_rel"),
        report_rows,
    )

    merge_status = "skip:not_requested"
    if args.merge_report_pdf:
        merge_status = merge_pdf_reports(staged_results_root, canonical_reports)

    # Prune bulky intermediates from copied run dirs.
    removed_files = 0
    removed_bytes = 0
    for run in selection.selected:
        rf, rb = prune_files(staged_results_root / run.name, drop_globs)
        removed_files += rf
        removed_bytes += rb

    # Emit manifests.
    manifest_path = write_release_manifest(release_root)
    size_report = release_root / "results" / "manifests" / "size_report.tsv"
    final_stat = compute_tree_stats(release_root)
    write_size_report(
        size_report,
        sections=[
            ("source_raw_data_source", data_stat),
            ("source_EPTA-DR3_no_results", epta_stat),
            ("source_selected_runs", selected_total),
            ("pruned_from_runs", FileStat(files=removed_files, bytes=removed_bytes)),
            ("release_final", final_stat),
        ],
    )

    print("\n== Release Build Complete ==")
    print(f"release_root:          {release_root}")
    print(f"report_index:          {report_index}")
    print(f"release_manifest:      {manifest_path}")
    print(f"size_report:           {size_report}")
    print(f"reports_harvested:     {len(report_rows)}")
    print(f"reports_unique:        {len(canonical_reports)}")
    print(f"merge_report_status:   {merge_status}")
    print(f"pruned_files:          {removed_files}")
    print(f"pruned_size:           {human_bytes(removed_bytes)}")
    print(f"release_files:         {final_stat.files}")
    print(f"release_size:          {human_bytes(final_stat.bytes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
