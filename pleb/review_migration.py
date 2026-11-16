"""One-time import of an approved selection into portable source decisions.

This command may read an existing release to capture its decisions. Normal
production runs use only raw inputs and the resulting decision catalog.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import argparse
import json

from .observations import read_observations
from .release import (
    ReleaseError,
    _bindings_read,
    bound_observations,
    digest,
    file_digest,
    json_write,
    run_release,
    load_release_config,
    observation_digest,
    timing_policy_digest,
)


def reference_observations(dataset, variant="combined"):
    """Follow the reference's actual INCLUDE/END/SKIP selection."""
    dataset = Path(dataset).resolve()

    def walk(path, stack=(), offset=Decimal(0)):
        path = path.resolve()
        if not path.is_relative_to(dataset) or path in stack:
            raise ReleaseError(f"Unsafe or cyclic reference INCLUDE: {path}")
        lines = path.read_text().splitlines()
        includes = any(line.split()[:1] == ["INCLUDE"] for line in lines)
        if not includes:
            return [
                replace(row, time=row.time + offset)
                for row in read_observations(path)
                if row.state == "active"
            ]
        result = []
        skip = False
        time = offset
        for line in lines:
            parts = line.split()
            if not parts or parts[0] in {"C", "c"} or parts[0].startswith("#"):
                continue
            command = parts[0].upper()
            if command == "END":
                break
            if command in {"SKIP", "NOSKIP"}:
                skip = command == "SKIP"
            elif not skip and command == "TIME":
                time += Decimal(parts[1])
            elif not skip and command == "INCLUDE":
                result.extend(walk(path.parent / parts[1], stack + (path,), time))
        return result

    for parent in sorted(dataset.glob("J*")):
        if not parent.is_dir():
            continue
        wrapper = parent / f"{parent.name}_{variant}_all.tim"
        if not wrapper.exists():
            wrapper = parent / f"{parent.name}_all.tim"
        if not wrapper.exists():
            raise ReleaseError(f"Missing reference wrapper for {parent.name}")
        for row in walk(wrapper):
            yield parent.name, row


def import_selection(config, reference, output, expected, workspace, *, tolerance_ns=1):
    """Capture inherited decisions without pretending they were newly reviewed.

    Source row values are never replaced by values from the reference. Matching
    allows only explicitly bounded print precision; missing measurements fail.
    """
    workspace = Path(workspace).resolve()
    _, original_config = load_release_config(config)
    overrides = [
        'review.mode="none"',
        'review.catalog=""',
        'review.sha256=""',
        'verify.reference=""',
        'verify.sha256=""',
        'source.inventory_sha256=""',
        'qc.mode="disabled"',
        "clocks.include_container=false",
        "timing.fit=false",
        "timing.validate=false",
        "output.directory=" + json.dumps(str(workspace)),
    ]
    prepared = run_release(config, overrides=overrides, stop_after="normalise")
    bindings = _bindings_read(workspace)
    bound = list(bound_observations(workspace / "dataset", bindings))
    index = defaultdict(list)
    for number, (record, row) in enumerate(bound):
        index[
            (record["pulsar"], Path(row.filename).name, row.frequency, row.uncertainty)
        ].append(number)
    chosen = set()
    counts = Counter()
    unmatched, contexts = [], []
    tolerance = Decimal(str(tolerance_ns)) / Decimal(86400 * 10**9)
    maximum = Decimal(0)
    aliases = {
        "g": "eff",
        "8": "jb",
        "t": "lofar",
        "f": "ncy",
        "i": "wsrt",
        "y": "jb",
        "z": "srt",
    }
    target_rows = list(reference_observations(reference))
    for psr, target in target_rows:
        candidates = [
            n
            for n in index[
                (psr, Path(target.filename).name, target.frequency, target.uncertainty)
            ]
            if n not in chosen and abs(bound[n][1].mjd - target.mjd) <= tolerance
        ]
        if not candidates:
            unmatched.append(
                dict(
                    pulsar=psr,
                    filename=target.filename,
                    mjd=str(target.mjd),
                    frequency=str(target.frequency),
                )
            )
            continue
        # Prefer the existing active copy, then the closest epoch and stable
        # native ingest order. Distinct duplicate copies retain distinct IDs.
        number = min(
            candidates,
            key=lambda n: (
                bound[n][1].state != "active",
                abs(bound[n][1].mjd - target.mjd),
                n,
            ),
        )
        record, source = bound[number]
        a, b = source.site.lower(), target.site.lower()
        if aliases.get(a, a) != aliases.get(b, b):
            contexts.append(
                dict(
                    source_id=record["source_id"],
                    pulsar=psr,
                    observatory_source=source.site,
                    observatory_reference=target.site,
                )
            )
        if abs(source.epoch - target.epoch) > tolerance:
            contexts.append(
                dict(
                    source_id=record["source_id"],
                    pulsar=psr,
                    source_time=str(source.time),
                    reference_time=str(target.time),
                    source_addsat=source.flags.get("-addsat"),
                    reference_addsat=target.flags.get("-addsat"),
                )
            )
        for flag in ("-padd", "-to", "-dmo", "-dm", "-efac", "-equad"):
            default = "1" if flag == "-efac" else "0"
            a, b = source.flags.get(flag, default), target.flags.get(flag, default)
            if Decimal(a or "0") != Decimal(b or "0"):
                contexts.append(
                    dict(
                        source_id=record["source_id"],
                        pulsar=psr,
                        timing_flag=flag,
                        source=a,
                        reference=b,
                    )
                )
        if source.efac != target.efac or source.equad != target.equad:
            contexts.append(
                dict(
                    source_id=record["source_id"],
                    pulsar=psr,
                    error_scaling="EFAC/EQUAD differ",
                )
            )
        maximum = max(maximum, abs(source.mjd - target.mjd) * Decimal(86400 * 10**9))
        chosen.add(number)
        counts[psr] += 1
    audit = dict(
        reference=str(Path(reference).resolve()),
        reference_active=len(target_rows),
        matched=len(chosen),
        tolerance_ns=tolerance_ns,
        maximum_print_difference_ns=str(maximum),
        unmatched=unmatched,
        timing_context_differences=contexts,
    )
    json_write(workspace / "reports/decision_import.json", audit)
    if unmatched or contexts:
        raise ReleaseError(
            f"Decision import failed: {len(unmatched)} unmatched measurements, {len(contexts)} timing-context differences; see decision_import.json"
        )
    decisions = []
    active_ids = defaultdict(list)
    for number, (record, row) in enumerate(bound):
        keep = number in chosen
        if keep:
            active_ids[record["pulsar"]].append(record["source_id"])
        if keep != (row.state == "active"):
            decisions.append(
                dict(
                    source_id=record["source_id"],
                    action="keep" if keep else "comment",
                    reason="Inherited approved selection",
                    origin="inherited_selection",
                )
            )
    reference_root = Path(reference).resolve()
    reference_digest = digest(
        {
            p.relative_to(reference_root).as_posix(): file_digest(p)
            for p in sorted(reference_root.rglob("*.tim"))
        }
    )
    catalog = dict(
        format_version=1,
        identities_sha256=prepared["input"]["identities_sha256"],
        provenance={
            "kind": "selection_import",
            "reference_timfiles_sha256": reference_digest,
            "reference_active": len(target_rows),
            "matching_tolerance_ns": tolerance_ns,
            "note": "These are inherited selection decisions; unknown reviewer or automated provenance is not relabelled as a new manual inspection.",
        },
        decisions=decisions,
    )
    reference_summary = dict(
        format_version=1,
        active=sum(counts.values()),
        pulsars={
            p: dict(active=len(ids), selection_sha256=digest(sorted(ids)))
            for p, ids in sorted(active_ids.items())
        },
        selection_sha256=digest(sorted(i for ids in active_ids.values() for i in ids)),
        measurements_sha256=digest(
            sorted(
                [bound[n][0]["source_id"], observation_digest(bound[n][1])]
                for n in chosen
            )
        ),
        timing_policy_sha256=timing_policy_digest(original_config),
    )
    json_write(output, catalog)
    json_write(expected, reference_summary)
    return dict(
        catalog=str(output),
        catalog_sha256=file_digest(output),
        reference=str(expected),
        reference_sha256=file_digest(expected),
        inventory_sha256=digest(prepared["input"]),
        decisions=len(decisions),
        active=reference_summary["active"],
        audit=str(workspace / "reports/decision_import.json"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--tolerance-ns", type=float, default=1)
    args = parser.parse_args(argv)
    for path in (args.output, args.expected):
        if path.exists():
            parser.error(f"Output exists: {path}")
    if args.tolerance_ns < 0:
        parser.error("Tolerance must be nonnegative")
    try:
        result = import_selection(
            args.config,
            args.reference,
            args.output,
            args.expected,
            args.workspace,
            tolerance_ns=args.tolerance_ns,
        )
    except (ReleaseError, OSError, KeyError) as error:
        parser.exit(1, f"release-decisions: {error}\n")
    print(json.dumps(result, indent=2))
    return 0
