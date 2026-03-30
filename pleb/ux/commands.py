"""CLI commands for the UX wrapper interface."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from ..config_io import _parse_value_as_toml_literal, _set_dotted_key, _dump_toml_no_nulls
from .adapter import ux_to_legacy_dict
from .loader import deep_merge, load_ux_config, write_ux_config
from .models import UXConfig
from .presets import list_presets, load_preset

_DEFAULT_UX_PATH = Path("pleb.toml")


def run_ux_cli(argv: list[str] | None) -> int:
    """Execute UX subcommands.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector including the UX subcommand.

    Returns
    -------
    int
        Process exit code.
    """
    p = _build_parser()
    args = p.parse_args(argv)

    if args.ux_cmd == "init":
        return _cmd_init(args)
    if args.ux_cmd == "run":
        return _cmd_run(args)
    if args.ux_cmd == "profile":
        return _cmd_profile(args)
    if args.ux_cmd == "doctor":
        return _cmd_doctor(args)
    if args.ux_cmd == "explain":
        return _cmd_explain(args)
    raise SystemExit(f"Unknown UX command: {args.ux_cmd}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UX wrapper commands for pleb.toml")
    sp = p.add_subparsers(dest="ux_cmd", required=True)

    p_init = sp.add_parser("init", help="Create a starter pleb.toml")
    p_init.add_argument("--config", default=str(_DEFAULT_UX_PATH), help="Output UX config path")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing file")

    p_run = sp.add_parser("run", help="Run pleb using UX config")
    p_run.add_argument("--config", default=str(_DEFAULT_UX_PATH), help="UX config path")
    p_run.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override UX key (dotted notation supported).",
    )

    p_prof = sp.add_parser("profile", help="Manage UX presets")
    p_prof_sp = p_prof.add_subparsers(dest="profile_cmd", required=True)
    p_prof_sp.add_parser("list", help="List available presets")
    p_use = p_prof_sp.add_parser("use", help="Merge a preset into UX config")
    p_use.add_argument("name", help="Preset name")
    p_use.add_argument("--config", default=str(_DEFAULT_UX_PATH), help="UX config path")

    p_doc = sp.add_parser("doctor", help="Validate and summarize resolved config")
    p_doc.add_argument("--config", default=str(_DEFAULT_UX_PATH), help="UX config path")

    p_exp = sp.add_parser("explain", help="Explain UX->legacy key mapping")
    p_exp.add_argument("--config", default=str(_DEFAULT_UX_PATH), help="UX config path")
    return p


def _cmd_init(args: argparse.Namespace) -> int:
    out = Path(args.config)
    if out.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing file: {out}. Use --force.")
    cfg = {
        "paths": {
            "home_dir": "/path/to/repo",
            "dataset_name": ".",
            "results_dir": "results",
            "singularity_image": "/path/to/tempo2.sif",
        },
        "data": {
            "branches": ["main"],
            "reference_branch": "main",
            "pulsars": "ALL",
            "jobs": 4,
        },
        "run": {
            "mode": "pipeline",
            "run_tempo2": True,
            "run_fix_dataset": False,
            "run_pqc": False,
            "qc_report": False,
        },
        "policy": {
            "fix": {
                "apply": False,
                "base_branch": "main",
                "branch_name": "",
                "commit_message": "",
            },
            "pqc": {
                "backend_col": "sys",
            },
        },
    }
    write_ux_config(out, cfg)
    print(str(out))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    ux = load_ux_config(cfg_path)
    ux_dict = _ux_to_dict(ux)

    for item in args.overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item!r}")
        k, vraw = item.split("=", 1)
        _set_dotted_key(ux_dict, k.strip(), _parse_value_as_toml_literal(vraw))

    ux = _dict_to_ux(ux_dict)
    legacy = ux_to_legacy_dict(ux)

    mode = str(ux.run.get("mode", "pipeline")).strip().lower() or "pipeline"
    return _dispatch_legacy(mode, legacy)


def _cmd_profile(args: argparse.Namespace) -> int:
    if args.profile_cmd == "list":
        names = list_presets()
        for n in names:
            print(n)
        return 0

    if args.profile_cmd == "use":
        cfg_path = Path(args.config)
        if cfg_path.exists():
            base = _ux_to_dict(load_ux_config(cfg_path))
        else:
            base = {}
        preset = load_preset(args.name)
        merged = deep_merge(base, preset)
        write_ux_config(cfg_path, merged)
        print(str(cfg_path))
        return 0

    raise SystemExit(f"Unknown profile command: {args.profile_cmd}")


def _cmd_doctor(args: argparse.Namespace) -> int:
    ux = load_ux_config(Path(args.config))
    legacy = ux_to_legacy_dict(ux)

    mode = str(ux.run.get("mode", "pipeline"))
    if mode in {"qc", "qc-report", "qc_report"}:
        qc_run_dir = (
            legacy.get("run_dir")
            or legacy.get("qc_report_run_dir")
            or legacy.get("qc_report_dir")
        )
        missing = [] if qc_run_dir else ["run_dir"]
    else:
        required = ["home_dir", "singularity_image"]
        missing = [k for k in required if not legacy.get(k)]

    print(f"mode={mode}")
    print(f"resolved_keys={len(legacy)}")
    if missing:
        print("missing_required=" + ",".join(missing))
    else:
        print("missing_required=none")
    return 0 if not missing else 1


def _cmd_explain(args: argparse.Namespace) -> int:
    ux = load_ux_config(Path(args.config))
    legacy = ux_to_legacy_dict(ux)
    print("UX sections -> legacy flat keys")
    print(f"  paths keys: {sorted(ux.paths.keys())}")
    print(f"  data keys: {sorted(ux.data.keys())}")
    print(f"  run keys: {sorted(ux.run.keys())}")
    print(f"  policy keys: {sorted(ux.policy.keys())}")
    print(f"  workflow keys: {sorted(ux.workflow.keys())}")
    print(f"  total legacy keys: {len(legacy)}")
    return 0


def _dispatch_legacy(mode: str, cfg_data: Dict[str, Any]) -> int:
    """Dispatch into existing CLI commands using a temporary legacy config."""
    from ..cli import main as legacy_main

    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        tmp_path = tmp.name
        tmp.write(_dump_toml_no_nulls(cfg_data))
        tmp.close()

        if mode == "pipeline":
            return legacy_main(["--config", str(tmp_path)])
        if mode == "ingest":
            return legacy_main(["ingest", "--config", str(tmp_path)])
        if mode == "workflow":
            return legacy_main(["workflow", "--config", str(tmp_path)])
        if mode in {"qc", "qc-report", "qc_report"}:
            run_dir = (
                cfg_data.get("run_dir")
                or cfg_data.get("qc_report_run_dir")
                or cfg_data.get("qc_report_dir")
            )
            if not run_dir:
                raise SystemExit(
                    "qc-report mode requires run_dir. "
                    "Set [policy.report].run_dir (preferred) or run_dir."
                )
            qc_cfg = {
                "qc_report": {
                    "run_dir": run_dir,
                    "backend_col": cfg_data.get(
                        "qc_report_backend_col", cfg_data.get("backend_col", "group")
                    ),
                    "backend": cfg_data.get("qc_report_backend", cfg_data.get("backend")),
                    "report_dir": cfg_data.get("qc_report_report_dir"),
                    "no_plots": cfg_data.get("qc_report_no_plots", False),
                    "structure_group_cols": cfg_data.get(
                        "qc_report_structure_group_cols"
                    ),
                    "no_feature_plots": cfg_data.get(
                        "qc_report_no_feature_plots", False
                    ),
                    "compact_pdf": cfg_data.get("qc_report_compact_pdf", False),
                    "compact_pdf_name": cfg_data.get(
                        "qc_report_compact_pdf_name", "qc_compact_report.pdf"
                    ),
                }
            }
            with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as qt:
                qtmp = qt.name
                qt.write(_dump_toml_no_nulls(qc_cfg))
            try:
                return legacy_main(["qc-report", "--config", str(qtmp)])
            finally:
                if os.path.exists(qtmp):
                    os.unlink(qtmp)
        raise SystemExit(
            f"Unsupported run.mode={mode!r}. Use one of: pipeline, ingest, workflow, qc-report"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _ux_to_dict(cfg: UXConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if cfg.paths:
        out["paths"] = dict(cfg.paths)
    if cfg.data:
        out["data"] = dict(cfg.data)
    if cfg.run:
        out["run"] = dict(cfg.run)
    if cfg.policy:
        out["policy"] = dict(cfg.policy)
    if cfg.workflow:
        out["workflow"] = dict(cfg.workflow)
    if cfg.pipeline:
        out["pipeline"] = dict(cfg.pipeline)
    out.update(cfg.extra)
    return out


def _dict_to_ux(d: Dict[str, Any]) -> UXConfig:
    known = {"paths", "data", "run", "policy", "workflow", "pipeline"}
    return UXConfig(
        paths=dict(d.get("paths", {})) if isinstance(d.get("paths"), dict) else {},
        data=dict(d.get("data", {})) if isinstance(d.get("data"), dict) else {},
        run=dict(d.get("run", {})) if isinstance(d.get("run"), dict) else {},
        policy=dict(d.get("policy", {})) if isinstance(d.get("policy"), dict) else {},
        workflow=dict(d.get("workflow", {})) if isinstance(d.get("workflow"), dict) else {},
        pipeline=dict(d.get("pipeline", {})) if isinstance(d.get("pipeline"), dict) else {},
        extra={k: v for k, v in d.items() if k not in known},
    )
