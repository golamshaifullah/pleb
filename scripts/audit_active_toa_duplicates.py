#!/usr/bin/env python3
"""Read-only audit of active duplicate candidates and PLEB's current checks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import csv
import json
from pathlib import Path
import tomllib

from pleb.dataset_fix import (
    dedupe_timfile_toas,
    remove_overlaps_exact,
    find_duplicate_backend_timfiles,
    _default_overlap_exact_catalog_path,
)
from pleb.tim_utils import is_toa_line, parse_tim_flags_from_line


@dataclass
class Row:
    psr: str
    tim: str
    line: int
    name: str
    freq_text: str
    mjd_text: str
    err_text: str
    site: str
    offset: Decimal
    flags: dict
    combined: bool

    @property
    def epoch(self):
        return Decimal(self.mjd_text) + self.offset / Decimal(86400)

    @property
    def freq(self):
        return Decimal(self.freq_text)

    @property
    def full_key(self):
        return self.name, self.freq_text, self.mjd_text, self.err_text


def include_paths(path: Path, stack=()):
    path = path.resolve()
    if path in stack:
        raise ValueError(f"INCLUDE cycle: {path}")
    result = []
    for raw in path.read_text().splitlines():
        p = raw.split()
        if p and p[0] == "INCLUDE":
            child = (path.parent / p[1]).resolve()
            result.append(child)
            result.extend(include_paths(child, stack + (path,)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-catalog", type=Path, required=True)
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    overlap_map = tomllib.loads(args.overlap_catalog.read_text())["overlap_map"]
    known_pairs = {
        frozenset([keep, drop]) for keep, drops in overlap_map.items() for drop in drops
    }
    allrows, dedupe, overlaps, repeated, identical = [], [], [], [], []
    offsets = Counter()
    for psrdir in sorted(data.glob("J*")):
        if not psrdir.is_dir():
            continue
        used = set(include_paths(psrdir / f"{psrdir.name}_combined_all.tim"))
        files = sorted((psrdir / "tims").glob("*.tim"))
        active_files = []
        for wrapper in psrdir.glob("*_all.tim"):
            for path, count in Counter(include_paths(wrapper)).items():
                if count > 1:
                    repeated.append(
                        {
                            "wrapper": str(wrapper.relative_to(data)),
                            "included": str(path.relative_to(data)),
                            "times": count,
                        }
                    )
        for path in files:
            offset = Decimal(0)
            active = 0
            for lineno, raw in enumerate(path.read_text().splitlines(), 1):
                p = raw.split()
                if p and p[0] == "TIME":
                    offset += Decimal(p[1])  # Tempo2 TIME is cumulative.
                    offsets[p[1]] += 1
                    continue
                if not is_toa_line(raw) or len(p) < 5:
                    continue
                try:
                    f, m, e = map(Decimal, p[1:4])
                    if not (
                        f.is_finite()
                        and m.is_finite()
                        and e.is_finite()
                        and Decimal(30000) < m < Decimal(100000)
                    ):
                        continue
                except InvalidOperation:
                    continue
                allrows.append(
                    Row(
                        psrdir.name,
                        str(path.relative_to(data)),
                        lineno,
                        p[0],
                        p[1],
                        p[2],
                        p[3],
                        p[4],
                        offset,
                        parse_tim_flags_from_line(raw),
                        path.resolve() in used,
                    )
                )
                active += 1
            if active:
                active_files.append(path)
            report = dedupe_timfile_toas(path, apply=False, backup=False, dry_run=True)
            if report.get("duplicates", report["removed"]):
                dedupe.append(report)
        for group in find_duplicate_backend_timfiles(active_files):
            identical.append([str(p.relative_to(data)) for p in group])
        report = remove_overlaps_exact(
            psrdir, overlap_map=overlap_map, apply=False, backup=False, dry_run=True
        )
        if report["commented"]:
            overlaps.append({"pulsar": psrdir.name, **report})
        print(f"Audited {psrdir.name}", flush=True)

    raw_groups, numeric_groups, physical_groups = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for i, row in enumerate(allrows):
        raw_groups[(row.psr, *row.full_key)].append(i)
        numeric_groups[
            (row.psr, row.name, row.freq, Decimal(row.mjd_text), Decimal(row.err_text))
        ].append(i)
        physical_groups[
            (row.psr, Path(row.name).name, row.site, row.freq, row.epoch)
        ].append(i)
    candidate_categories = defaultdict(set)

    def add_groups(groups, label):
        for indices in groups.values():
            if len(indices) > 1:
                for k, i in enumerate(indices):
                    for j in indices[k + 1 :]:
                        candidate_categories[(min(i, j), max(i, j))].add(label)

    add_groups(raw_groups, "exact_first_four_columns")
    add_groups(numeric_groups, "numeric_first_four_columns")
    add_groups(physical_groups, "same_observation_frequency_site_effective_epoch")

    def proximity(groups, tolerance, label, cross_only=False):
        for indices in groups.values():
            indices.sort(key=lambda i: allrows[i].epoch)
            for k, i in enumerate(indices):
                a = allrows[i]
                for j in indices[k + 1 :]:
                    b = allrows[j]
                    if (b.epoch - a.epoch) * 86400 > tolerance:
                        break
                    if cross_only and a.tim == b.tim:
                        continue
                    candidate_categories[(min(i, j), max(i, j))].add(label)

    by_obs, by_channel = defaultdict(list), defaultdict(list)
    for i, row in enumerate(allrows):
        frequency = row.freq.quantize(Decimal("0.00001"))
        by_obs[(row.psr, Path(row.name).name, row.site, frequency)].append(i)
        by_channel[(row.psr, row.site, frequency)].append(i)
    proximity(by_obs, Decimal("0.000001"), "same_observation_within_1us")
    proximity(
        by_channel,
        Decimal("0.000001"),
        "coincident_channel_within_1us_cross_file",
        True,
    )
    proximity(by_obs, Decimal("1"), "same_observation_within_1s_review_candidate")

    pair_rows, counts, per_psr = [], Counter(), defaultdict(Counter)
    for (i, j), categories in sorted(candidate_categories.items()):
        a, b = allrows[i], allrows[j]
        scope = "within_file" if a.tim == b.tim else "across_files"
        combined = a.combined and b.combined
        if "exact_first_four_columns" in categories:
            reason = "exact_key_remaining"
        elif a.name != b.name:
            reason = "different_filename_tokens"
        elif a.err_text != b.err_text:
            reason = "different_uncertainty_or_formatting"
        elif a.freq_text != b.freq_text or a.mjd_text != b.mjd_text:
            reason = "different_numeric_tokens"
        else:
            reason = "near_epoch_candidate"
        record = {
            "pulsar": a.psr,
            "scope": scope,
            "both_in_combined": combined,
            "categories": ";".join(sorted(categories)),
            "reason_exact_key_misses": reason,
            "configured_overlap_pair": frozenset([Path(a.tim).name, Path(b.tim).name])
            in known_pairs,
            "delta_time_us": str(abs(a.epoch - b.epoch) * 86400 * 1000000),
            "delta_frequency_mhz": str(abs(a.freq - b.freq)),
        }
        for label, row in [("a", a), ("b", b)]:
            record.update(
                {
                    f"{label}_tim": row.tim,
                    f"{label}_line": row.line,
                    f"{label}_filename": row.name,
                    f"{label}_frequency": row.freq_text,
                    f"{label}_raw_mjd": row.mjd_text,
                    f"{label}_time_offset_sec": str(row.offset),
                    f"{label}_site": row.site,
                    f"{label}_error_us": row.err_text,
                    f"{label}_sys": row.flags.get("-sys", ""),
                    f"{label}_bw": row.flags.get("-bw", ""),
                }
            )
        pair_rows.append(record)
        for category in categories:
            key = f"{category}/{scope}"
            counts[key] += 1
            per_psr[a.psr][key] += 1
    with (output / "candidate_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(pair_rows[0]) if pair_rows else ["pulsar", "scope"]
        )
        writer.writeheader()
        writer.writerows(pair_rows)
    default_catalog = _default_overlap_exact_catalog_path()
    summary = {
        "active_backend_rows": len(allrows),
        "active_rows_in_combined_includes": sum(r.combined for r in allrows),
        "candidate_pairs": len(pair_rows),
        "category_pair_counts": dict(counts),
        "per_pulsar": {k: dict(v) for k, v in per_psr.items()},
        "default_within_file_strategy_would_remove": sum(r["removed"] for r in dedupe),
        "default_within_file_strategy_would_exclude": sum(
            r.get("duplicates", r["removed"]) for r in dedupe
        ),
        "default_within_file_reports": dedupe,
        "explicit_exact_overlap_catalog_would_comment": sum(
            r["commented"] for r in overlaps
        ),
        "exact_overlap_reports": overlaps,
        "installed_default_overlap_catalog": str(default_catalog),
        "installed_default_overlap_catalog_exists": default_catalog.exists(),
        "whole_backend_duplicate_groups": identical,
        "repeated_include_paths": repeated,
        "time_directives": dict(offsets),
        "notes": [
            "Categories overlap; do not add pair counts.",
            "1-second neighbours are review candidates, not automatic duplicates.",
            "Separate legacy/new/combined wrappers are not counted as duplicates of each other.",
            "No data files were changed.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k
                not in {
                    "per_pulsar",
                    "default_within_file_reports",
                    "exact_overlap_reports",
                    "time_directives",
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
