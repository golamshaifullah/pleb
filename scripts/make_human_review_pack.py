#!/usr/bin/env python3
"""Create a compact human-review entry point for PLEB outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

PULSAR_ALIASES = ("pulsar", "psr", "PSR", "name", "source", "source_psr")
BAD_ALIASES = (
    "bad_points",
    "num_bad",
    "n_bad",
    "auto_bad",
    "bad",
    "n_auto_bad",
    "selected_bad",
)
EVENT_ALIASES = (
    "event_points",
    "num_events",
    "n_events",
    "event",
    "events",
    "n_event",
)
STATUS_ORDER = {"FAIL": 0, "REVIEW": 1, "PASS": 2}


def norm_key(value: str) -> str:
    return str(value or "").strip().lower()


def slug(value: str) -> str:
    out = []
    for ch in str(value).lower():
        if ch == "+":
            out.append("p")
        elif ch == "-":
            out.append("m")
        elif ch.isalnum():
            out.append(ch)
    return "".join(out)


def safe_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in value)
    return cleaned.strip("_") or "report"


def report_name(path: Path, root: Path) -> str:
    parts = list(path.parts)
    name = path.name
    if name == "qc_compact_report.pdf":
        return "qc_compact_report.pdf"
    if name == "REPORT_INDEX.tsv":
        return "REPORT_INDEX.tsv"
    if name in {"report.pdf", "report.md"} and path.parent.name:
        return f"optimize__{safe_part(path.parent.name)}__{name}"
    if name in {
        "fix_dataset_report.pdf",
        "run_report.pdf",
        "workflow_report.pdf",
        "ingest_report.pdf",
    }:
        pulsar = ""
        stage = ""
        for part in parts:
            low = part.lower()
            if not pulsar and low.startswith("j") and any(ch.isdigit() for ch in low):
                pulsar = safe_part(low)
            if "step6" in low:
                stage = "step6"
            elif "step5" in low and not stage:
                stage = "step5"
            elif "step4" in low and not stage:
                stage = "step4"
            elif "step2" in low and not stage:
                stage = "step2"
            elif "step1" in low and not stage:
                stage = "step1"
        bits = [x for x in (pulsar, stage) if x]
        if bits:
            return "__".join(bits + [name])
    try:
        rel = path.resolve().relative_to(root.resolve())
    except Exception:
        rel = Path(path.name)
    out_parts = []
    for part in rel.parts:
        out_parts.append(safe_part(part))
    return "__".join(out_parts)


def read_table(path: Path) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return []
    if not text.strip():
        return []
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
    except Exception:
        dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
    try:
        return [
            {str(k or "").strip(): str(v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(text.splitlines(), dialect=dialect)
        ]
    except Exception:
        return []


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def header_lookup(row: dict[str, str]) -> dict[str, str]:
    return {norm_key(k): k for k in row}


def first_value(row: dict[str, str], aliases: tuple[str, ...]) -> str:
    lookup = header_lookup(row)
    for alias in aliases:
        key = lookup.get(norm_key(alias))
        if key is not None:
            return row.get(key, "")
    return ""


def parse_number(value: str) -> int:
    raw = str(value or "").strip().lower()
    if raw in ("", "none", "nan", "n/a", "na", "false", "no"):
        return 0
    if raw in ("true", "yes"):
        return 1
    try:
        return int(float(raw))
    except Exception:
        return 0


def row_pulsars(row: dict[str, str]) -> list[str]:
    value = first_value(row, PULSAR_ALIASES)
    if not value:
        lookup = header_lookup(row)
        plural = lookup.get("pulsars")
        if plural is not None:
            value = row.get(plural, "")
    if not value:
        return []
    chunks = []
    for item in value.replace(";", ",").split(","):
        for part in item.split():
            for piece in Path(part.strip()).parts:
                piece = piece.strip()
                if is_pulsar_token(piece):
                    chunks.append(piece)
    return chunks


def is_pulsar_token(value: str) -> bool:
    value = str(value or "").strip()
    return (
        bool(value)
        and value[0].upper() == "J"
        and any(ch.isdigit() for ch in value)
        and all(ch.isalnum() or ch in "+-" for ch in value)
    )


def failure_evidence(row: dict[str, str]) -> list[str]:
    evidence = []
    for key, value in row.items():
        lk = norm_key(key)
        lv = norm_key(value)
        if not lv or lv in (
            "0",
            "false",
            "none",
            "nan",
            "n/a",
            "na",
            "ok",
            "pass",
            "passed",
            "success",
            "done",
        ):
            continue
        if "status" in lk and any(token in lv for token in ("fail", "failed", "error")):
            evidence.append(f"{key}={value}")
        elif "failed" in lk or "fail" in lk:
            if lv not in ("no", "false", "0"):
                evidence.append(f"{key}={value}")
        elif "error" in lk:
            evidence.append(f"{key}={value}")
    return evidence


def default_record(pulsar: str) -> dict[str, object]:
    return {
        "pulsar": pulsar,
        "qc_rows": 0,
        "bad_points": 0,
        "event_points": 0,
        "cross_pulsar_hits": 0,
        "fix_dataset_rows": 0,
        "failures": [],
        "reports": set(),
    }


def get_record(records: dict[str, dict[str, object]], pulsar: str) -> dict[str, object]:
    key = str(pulsar).strip()
    if key not in records:
        records[key] = default_record(key)
    return records[key]


def add_report_match(
    report_by_pulsar: dict[str, set[str]],
    pulsars: set[str],
    source: Path,
    dest_rel: str,
) -> None:
    hay = f"{source.as_posix()} {dest_rel}".lower()
    for pulsar in pulsars:
        ps = slug(pulsar)
        if ps and (ps in slug(hay) or pulsar.lower() in hay):
            report_by_pulsar.setdefault(pulsar, set()).add(dest_rel)


def discover_existing(root: Path, rels: list[str]) -> list[Path]:
    out = []
    seen = set()
    for rel in rels:
        p = root / rel
        if p.exists() and p.resolve() not in seen:
            out.append(p)
            seen.add(p.resolve())
    return out


def discover_named(root: Path, name: str) -> list[Path]:
    out = []
    seen = set()
    for p in root.rglob(name):
        if "human_review" in p.parts:
            continue
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp not in seen:
            out.append(p)
            seen.add(rp)
    return sorted(out)


def load_json_failure(path: Path) -> tuple[list[str], list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return [], []
    pulsars = []
    failures = []
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            row = {
                str(k): str(v)
                for k, v in item.items()
                if not isinstance(v, (dict, list))
            }
            pulsars.extend(row_pulsars(row))
            failures.extend(failure_evidence(row))
            stack.extend(v for v in item.values() if isinstance(v, (dict, list)))
        elif isinstance(item, list):
            stack.extend(item)
    if not pulsars:
        for part in path.parts:
            if is_pulsar_token(part):
                pulsars.append(part)
    return sorted(set(pulsars)), sorted(set(failures))


def actionable_fix_row(row: dict[str, str]) -> bool:
    return bool(failure_evidence(row))


def link_or_copy(src: Path, dst: Path, copy_reports: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_reports:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dst)


def build_pack(root: Path, out_dir: Path, copy_reports: bool, verbose: bool) -> None:
    root = root.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    reports_dir = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    for old in reports_dir.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()

    records: dict[str, dict[str, object]] = {}
    missing: list[str] = []

    qc_summary = root / "qc" / "qc_summary.tsv"
    if qc_summary.exists():
        for row in read_table(qc_summary):
            for pulsar in row_pulsars(row):
                rec = get_record(records, pulsar)
                rec["qc_rows"] = int(rec["qc_rows"]) + 1
                rec["bad_points"] = int(rec["bad_points"]) + parse_number(
                    first_value(row, BAD_ALIASES)
                )
                rec["event_points"] = int(rec["event_points"]) + parse_number(
                    first_value(row, EVENT_ALIASES)
                )
                rec["failures"].extend(failure_evidence(row))
    else:
        missing.append("qc/qc_summary.tsv")

    cross_files = [
        root / "qc_cross_pulsar" / "coincidence_cluster_points.tsv",
        root / "qc_cross_pulsar" / "coincident_points.tsv",
        root / "qc_cross_pulsar" / "coincidence_clusters.tsv",
        root / "qc_cross_pulsar" / "selected_row_counts.tsv",
    ]
    found_cross = [p for p in cross_files if p.exists()]
    if found_cross:
        source = found_cross[0]
        for row in read_table(source):
            hits = (
                parse_number(row.get("selected_rows", ""))
                if source.name == "selected_row_counts.tsv"
                else 1
            )
            hits = max(1, hits)
            for pulsar in row_pulsars(row):
                rec = get_record(records, pulsar)
                rec["cross_pulsar_hits"] = int(rec["cross_pulsar_hits"]) + hits
                rec["failures"].extend(failure_evidence(row))
    else:
        missing.extend(str(p.relative_to(root)) for p in cross_files)

    fix_summaries = discover_existing(root, ["fix_dataset_summary.tsv"])
    fix_summaries.extend(
        p
        for p in discover_named(root, "fix_dataset_summary.tsv")
        if p not in fix_summaries
    )
    if fix_summaries:
        for path in fix_summaries:
            for row in read_table(path):
                for pulsar in row_pulsars(row):
                    rec = get_record(records, pulsar)
                    failures = failure_evidence(row)
                    rec["failures"].extend(failures)
                    if actionable_fix_row(row):
                        rec["fix_dataset_rows"] = int(rec["fix_dataset_rows"]) + 1
    else:
        missing.append("fix_dataset_summary.tsv")

    json_inputs = []
    json_inputs.extend((root / "results" / "optimize").glob("*/*summary.json"))
    json_inputs.extend((root / "results" / "optimize").glob("*/*best_trial.json"))
    for path in sorted(p for p in json_inputs if p.exists()):
        pulsars, failures = load_json_failure(path)
        if not failures:
            continue
        if not pulsars and failures:
            pulsars = [path.parent.name]
        for pulsar in pulsars:
            rec = get_record(records, pulsar)
            rec["failures"].extend(failures)

    report_sources = []
    report_sources.extend(
        discover_existing(
            root,
            [
                "qc_report/qc_compact_report.pdf",
                "fix_dataset_report.pdf",
                "results/reports/REPORT_INDEX.tsv",
            ],
        )
    )
    report_sources.extend(discover_named(root, "fix_dataset_report.pdf"))
    report_sources.extend(discover_named(root, "run_report.pdf"))
    report_sources.extend(discover_named(root, "workflow_report.pdf"))
    report_sources.extend(discover_named(root, "ingest_report.pdf"))
    report_sources.extend(discover_named(root, "qc_compact_report.pdf"))
    report_sources.extend((root / "results" / "optimize").glob("*/report.pdf"))
    report_sources.extend((root / "results" / "optimize").glob("*/report.md"))
    report_sources.extend((root / "results" / "reports").glob("*.pdf"))
    seen_reports = set()
    seen_report_hashes = set()
    unique_reports = []
    for src in report_sources:
        if not src.exists() or "human_review" in src.parts:
            continue
        rp = src.resolve()
        if rp in seen_reports:
            continue
        try:
            digest = file_digest(src)
        except Exception:
            digest = ""
        if digest and digest in seen_report_hashes:
            continue
        seen_reports.add(rp)
        if digest:
            seen_report_hashes.add(digest)
        unique_reports.append(src)

    report_by_pulsar: dict[str, set[str]] = {}
    all_pulsars = set(records)
    for src in unique_reports:
        name = report_name(src, root)
        dest = reports_dir / name
        link_or_copy(src, dest, copy_reports)
        rel = dest.relative_to(out_dir).as_posix()
        add_report_match(report_by_pulsar, all_pulsars, src, rel)
        if src.name == "qc_compact_report.pdf":
            for pulsar in all_pulsars:
                report_by_pulsar.setdefault(pulsar, set()).add(rel)

    rows = []
    for pulsar, rec in records.items():
        failures = sorted(set(str(x) for x in rec["failures"] if str(x).strip()))
        bad = int(rec["bad_points"])
        events = int(rec["event_points"])
        cross = int(rec["cross_pulsar_hits"])
        fix_rows = int(rec["fix_dataset_rows"])
        if failures:
            status = "FAIL"
        elif bad > 0 or events > 0 or cross > 0 or fix_rows > 0:
            status = "REVIEW"
        else:
            status = "PASS"
        priority = cross * 100 + bad * 10 + events * 5 + fix_rows
        reasons = []
        if failures:
            reasons.append("failure evidence: " + "; ".join(failures[:3]))
        if cross:
            reasons.append(f"cross_pulsar_hits={cross}")
        if bad:
            reasons.append(f"bad_points={bad}")
        if events:
            reasons.append(f"event_points={events}")
        if fix_rows:
            reasons.append(f"fix_dataset_rows={fix_rows}")
        if not reasons:
            reasons.append("no review triggers found")
        hint_bits = []
        reports = sorted(report_by_pulsar.get(pulsar, set()))
        if reports:
            hint_bits.append("open " + reports[0])
        if cross:
            hint_bits.append("check cross-pulsar coincidence rows")
        if bad or events:
            hint_bits.append("check QC CSV/report decisions")
        if fix_rows:
            hint_bits.append("check fix-dataset summary")
        rows.append(
            {
                "pulsar": pulsar,
                "status": status,
                "priority": str(priority),
                "reason": "; ".join(reasons),
                "qc_rows": str(rec["qc_rows"]),
                "bad_points": str(bad),
                "event_points": str(events),
                "cross_pulsar_hits": str(cross),
                "fix_dataset_rows": str(fix_rows),
                "report_paths": ";".join(reports),
                "review_hint": (
                    "; ".join(hint_bits) if hint_bits else "no action needed"
                ),
            }
        )

    rows.sort(
        key=lambda r: (
            STATUS_ORDER.get(r["status"], 9),
            -int(r["priority"]),
            r["pulsar"],
        )
    )

    fields = [
        "pulsar",
        "status",
        "priority",
        "reason",
        "qc_rows",
        "bad_points",
        "event_points",
        "cross_pulsar_hits",
        "fix_dataset_rows",
        "report_paths",
        "review_hint",
    ]
    with (out_dir / "REVIEW_INDEX.tsv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    first_lines = [
        "\t".join([r["status"], r["pulsar"], r["reason"], r["review_hint"]])
        for r in rows
        if r["status"] in ("FAIL", "REVIEW")
    ]
    (out_dir / "OPEN_THESE_FIRST.txt").write_text(
        "\n".join(first_lines) + ("\n" if first_lines else ""),
        encoding="utf-8",
    )

    readme = [
        "# PLEB Human Review Pack",
        "",
        "This directory is generated by `scripts/make_human_review_pack.py` as a compact entry point for manual review.",
        "",
        "Original PLEB outputs are untouched. This pack only writes files under the selected `human_review` output directory.",
        "",
        "Start with `OPEN_THESE_FIRST.txt` for FAIL/REVIEW items, then use `REVIEW_INDEX.tsv` for the full sortable table.",
        "If `OPEN_THESE_FIRST.txt` is empty, no explicit review trigger was inferred from the available summaries; browse `reports/` for the human PDFs.",
        "",
        "`reports/` contains symlinks or copies of obvious human-facing PDFs and report indices found in the source tree, with duplicate report files removed by content hash.",
        "",
        "Missing input files are tolerated; absent numeric fields are treated as zero and rows without an inferred pulsar are skipped.",
    ]
    if missing and verbose:
        readme.extend(["", "Missing inputs observed while generating this pack:"])
        readme.extend(f"- `{item}`" for item in sorted(set(missing)))
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"human_review_dir: {out_dir}")
    print(f"review_index:     {out_dir / 'REVIEW_INDEX.tsv'}")
    print(f"open_first:       {out_dir / 'OPEN_THESE_FIRST.txt'}")
    print(f"reports_dir:      {reports_dir}")
    if verbose and missing:
        print(f"missing_inputs:   {len(set(missing))}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a compact human-review pack from a PLEB output root."
    )
    parser.add_argument("root", type=Path, help="PLEB output root to scan")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/human_review)",
    )
    parser.add_argument(
        "--copy-reports",
        action="store_true",
        help="Copy reports instead of symlinking them",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra details")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = args.out_dir if args.out_dir else root / "human_review"
    build_pack(root, out_dir, args.copy_reports, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
