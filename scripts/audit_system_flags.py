#!/usr/bin/env python3
"""Audit existing -sys/-group/-pta flags against PLEB inference.

This script does not modify any timfiles. It compares the flags currently
present in backend timfiles to what ``pleb.system_flag_inference`` would infer
under the same mapping/rules inputs that Step 1 uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from pleb.system_flag_inference import (
    SystemInferenceConfig,
    infer_sys_group_pta,
    load_flag_sys_freq_rules,
    load_system_flag_mapping,
    parse_tim_toa_table,
)


def _discover_pulsars(dataset_root: Path, pulsars_arg: str) -> list[str]:
    if pulsars_arg.strip().upper() == "ALL":
        return sorted(
            p.name
            for p in dataset_root.iterdir()
            if p.is_dir() and p.name.startswith("J") and not p.name.startswith(".")
        )
    return [p.strip() for p in pulsars_arg.split(",") if p.strip()]


def _load_optional_mapping(path: Path | None) -> Optional[Dict[str, object]]:
    if path is None or not path.exists():
        return None
    return load_system_flag_mapping(path)


def _mapping_cfg(mapping: Optional[Dict[str, object]]) -> SystemInferenceConfig:
    if not mapping:
        return SystemInferenceConfig()
    base_cfg = SystemInferenceConfig()
    return SystemInferenceConfig(
        backend_allowlist=mapping.get("backend_allowlist"),
        telescope_allowlist=(
            mapping.get("telescope_allowlist")
            if mapping.get("telescope_allowlist") is not None
            else base_cfg.telescope_allowlist
        ),
        backend_aliases=mapping.get("backend_aliases", {}),
        telescope_aliases=mapping.get("telescope_aliases", {}),
    )


def _override_for_timfile(
    mapping: Optional[Dict[str, object]],
    timfile: Path,
    key: str,
) -> Optional[str]:
    if not mapping:
        return None
    values = mapping.get(key, {}) or {}
    if not isinstance(values, dict):
        return None
    return values.get(timfile.name) or values.get(str(timfile))


def _join_distinct(values: Iterable[object]) -> str:
    uniq = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if text in seen:
            continue
        seen.add(text)
        uniq.append(text)
    return ";".join(sorted(uniq))


def _series_from_flags(df: pd.DataFrame, flag: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    return df["flags"].apply(
        lambda flags: ""
        if not isinstance(flags, dict)
        else str(flags.get(flag, "") or "").strip()
    )


def audit_timfile(
    timfile: Path,
    *,
    pulsar: str,
    cfg: SystemInferenceConfig,
    mapping: Optional[Dict[str, object]],
    flag_sys_freq_rules: Optional[Dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = parse_tim_toa_table(timfile, cfg=cfg).copy()
    if current.empty:
        return (
            pd.DataFrame(
                [
                    {
                        "pulsar": pulsar,
                        "timfile": timfile.name,
                        "n_toas": 0,
                        "status": "empty",
                    }
                ]
            ),
            pd.DataFrame(),
        )

    override_backend = _override_for_timfile(mapping, timfile, "backend_by_timfile")
    override_telescope = _override_for_timfile(mapping, timfile, "telescope_by_timfile")
    inferred = infer_sys_group_pta(
        timfile,
        cfg=cfg,
        override_backend=override_backend,
        override_telescope=override_telescope,
        pulsar=pulsar,
        flag_sys_freq_rules=flag_sys_freq_rules,
    ).copy()

    current["current_sys"] = _series_from_flags(current, "-sys")
    current["current_group"] = _series_from_flags(current, "-group")
    current["current_pta"] = _series_from_flags(current, "-pta")

    merged = current.merge(
        inferred[
            [
                "line_idx",
                "sys",
                "group",
                "pta",
                "backend",
                "tel",
                "centre_mhz",
                "bw_mhz",
                "nband",
            ]
        ],
        on="line_idx",
        how="left",
        suffixes=("", "_inferred"),
    )
    merged = merged.rename(
        columns={
            "sys": "inferred_sys",
            "group": "inferred_group",
            "pta": "inferred_pta",
        }
    )

    merged["sys_match"] = merged["current_sys"] == merged["inferred_sys"]
    merged["group_match"] = merged["current_group"] == merged["inferred_group"]
    merged["pta_match"] = merged["current_pta"] == merged["inferred_pta"]
    merged["sys_missing"] = merged["current_sys"].eq("")
    merged["group_missing"] = merged["current_group"].eq("")
    merged["pta_missing"] = merged["current_pta"].eq("")
    merged["any_mismatch"] = ~(
        merged["sys_match"] & merged["group_match"] & merged["pta_match"]
    )

    summary = pd.DataFrame(
        [
            {
                "pulsar": pulsar,
                "timfile": timfile.name,
                "n_toas": int(len(merged)),
                "backend": str(merged["backend"].dropna().iloc[0])
                if merged["backend"].notna().any()
                else "",
                "tel": str(merged["tel"].dropna().iloc[0])
                if merged["tel"].notna().any()
                else "",
                "current_sys_values": _join_distinct(merged["current_sys"]),
                "inferred_sys_values": _join_distinct(merged["inferred_sys"]),
                "current_group_values": _join_distinct(merged["current_group"]),
                "inferred_group_values": _join_distinct(merged["inferred_group"]),
                "current_pta_values": _join_distinct(merged["current_pta"]),
                "inferred_pta_values": _join_distinct(merged["inferred_pta"]),
                "sys_missing_count": int(merged["sys_missing"].sum()),
                "group_missing_count": int(merged["group_missing"].sum()),
                "pta_missing_count": int(merged["pta_missing"].sum()),
                "sys_mismatch_count": int((~merged["sys_match"]).sum()),
                "group_mismatch_count": int((~merged["group_match"]).sum()),
                "pta_mismatch_count": int((~merged["pta_match"]).sum()),
                "status": "ok",
            }
        ]
    )

    detail = merged.loc[
        merged["any_mismatch"],
        [
            "line_idx",
            "freq_mhz",
            "current_sys",
            "inferred_sys",
            "current_group",
            "inferred_group",
            "current_pta",
            "inferred_pta",
            "backend",
            "tel",
            "centre_mhz",
            "bw_mhz",
            "nband",
            "line",
        ],
    ].copy()
    if not detail.empty:
        detail.insert(0, "timfile", timfile.name)
        detail.insert(0, "pulsar", pulsar)

    return summary, detail


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit existing -sys/-group/-pta flags against PLEB inference."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Canonical dataset root, e.g. /work/git_projects/epta-dr3-optimize/EPTA-DR3/epta-dr3-data",
    )
    parser.add_argument(
        "--pulsars",
        default="ALL",
        help='Comma-separated pulsars or "ALL".',
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help="Optional system_flag_mapping.json path. Defaults to <dataset-root>/system_flag_mapping.json if present.",
    )
    parser.add_argument(
        "--rules",
        default=None,
        help="Optional flag_sys_freq_rules.yaml path. Defaults to <dataset-root>/flag_sys_freq_rules.yaml if present.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for summary and mismatch CSVs.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = (
        Path(args.mapping).expanduser().resolve()
        if args.mapping
        else (dataset_root / "system_flag_mapping.json")
    )
    rules_path = (
        Path(args.rules).expanduser().resolve()
        if args.rules
        else (dataset_root / "flag_sys_freq_rules.yaml")
    )

    mapping = _load_optional_mapping(mapping_path)
    inference_cfg = _mapping_cfg(mapping)
    flag_sys_freq_rules = (
        load_flag_sys_freq_rules(rules_path) if rules_path.exists() else None
    )

    summaries: list[pd.DataFrame] = []
    mismatches: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []

    pulsars = _discover_pulsars(dataset_root, args.pulsars)
    for pulsar in pulsars:
        psr_dir = dataset_root / pulsar
        tim_dir = psr_dir / "tims"
        timfiles = sorted(tim_dir.glob("*.tim")) if tim_dir.exists() else []
        for timfile in timfiles:
            try:
                summary, detail = audit_timfile(
                    timfile,
                    pulsar=pulsar,
                    cfg=inference_cfg,
                    mapping=mapping,
                    flag_sys_freq_rules=flag_sys_freq_rules,
                )
                summaries.append(summary)
                if not detail.empty:
                    mismatches.append(detail)
            except Exception as exc:
                failures.append(
                    {
                        "pulsar": pulsar,
                        "timfile": timfile.name,
                        "status": "error",
                        "error": str(exc),
                    }
                )

    summary_df = (
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    )
    mismatch_df = (
        pd.concat(mismatches, ignore_index=True) if mismatches else pd.DataFrame()
    )
    failures_df = pd.DataFrame(failures)

    summary_path = out_dir / "system_flag_audit_summary.csv"
    mismatch_path = out_dir / "system_flag_audit_mismatches.csv"
    failures_path = out_dir / "system_flag_audit_failures.csv"
    manifest_path = out_dir / "system_flag_audit_manifest.json"

    summary_df.to_csv(summary_path, index=False)
    mismatch_df.to_csv(mismatch_path, index=False)
    failures_df.to_csv(failures_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "pulsars": pulsars,
                "mapping": str(mapping_path) if mapping_path.exists() else None,
                "rules": str(rules_path) if rules_path.exists() else None,
                "n_summary_rows": int(len(summary_df)),
                "n_mismatch_rows": int(len(mismatch_df)),
                "n_failures": int(len(failures_df)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    n_bad_timfiles = 0
    if not summary_df.empty:
        bad_mask = (
            (summary_df["sys_mismatch_count"] > 0)
            | (summary_df["group_mismatch_count"] > 0)
            | (summary_df["pta_mismatch_count"] > 0)
        )
        n_bad_timfiles = int(bad_mask.sum())

    print(f"Summary   : {summary_path}")
    print(f"Mismatches: {mismatch_path}")
    print(f"Failures  : {failures_path}")
    print(f"Manifest  : {manifest_path}")
    print(f"Timfiles audited: {len(summary_df)}")
    print(f"Timfiles with flag mismatches: {n_bad_timfiles}")
    print(f"Audit failures: {len(failures_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
