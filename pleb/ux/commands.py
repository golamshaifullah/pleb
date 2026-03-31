"""CLI commands for the UX wrapper interface."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

from ..config import IngestConfig, PipelineConfig, QCReportConfig, WorkflowRunConfig
from ..config_io import _dump_toml_no_nulls, _parse_value_as_toml_literal, _set_dotted_key
from .adapter import ux_to_legacy_dict
from .key_catalog import default_for_spec, grouped_specs, specs_for_mode_level
from .loader import deep_merge, load_ux_config, write_ux_config
from .models import UXConfig
from .presets import list_presets, load_preset

_DEFAULT_UX_PATH = Path("configs/runs/pipeline/pleb.pipeline.toml")
_MODE_FILE_BASENAME = {
    "pipeline": "pleb.pipeline.toml",
    "ingest": "pleb.ingest.toml",
    "workflow": "pleb.workflow.toml",
    "qc-report": "pleb.qc-report.toml",
}
_MODE_SUBDIR = {
    "pipeline": Path("runs/pipeline"),
    "ingest": Path("runs/ingest"),
    "workflow": Path("runs/workflow"),
    "qc-report": Path("runs/qc_report"),
}


def _mode_path(mode_name: str, root_dir: Path) -> Path:
    return root_dir / _MODE_SUBDIR[mode_name] / _MODE_FILE_BASENAME[mode_name]


def run_ux_cli(argv: list[str] | None) -> int:
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

    p_init = sp.add_parser("init", help="Create starter config files")
    p_init.add_argument("--config", default=None)
    p_init.add_argument(
        "--mode",
        choices=["pipeline", "ingest", "workflow", "qc-report"],
        default=None,
    )
    p_init.add_argument(
        "--level",
        choices=["minimal", "balanced", "full"],
        default="minimal",
        help="Template verbosity level.",
    )
    p_init.add_argument(
        "--workflow-template",
        choices=["3pass-clean", "golden-path"],
        default=None,
        help="Generate a ready-made multi-pass workflow blueprint.",
    )
    p_init.add_argument("--all-modes", action="store_true")
    p_init.add_argument("--outdir", default="configs")
    p_init.add_argument("--force", action="store_true")

    p_run = sp.add_parser("run", help="Run pleb using UX config")
    p_run.add_argument(
        "journey",
        nargs="?",
        choices=["detect", "apply", "publish"],
        help="Golden-path journey shortcut.",
    )
    p_run.add_argument("--config", default=str(_DEFAULT_UX_PATH))
    p_run.add_argument(
        "--profile",
        dest="profiles",
        action="append",
        default=[],
        help="Preset profile name to merge before --set overrides.",
    )
    p_run.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p_run.add_argument(
        "--plan",
        action="store_true",
        help="Print resolved execution plan and exit.",
    )
    p_run.add_argument(
        "--confirm",
        action="store_true",
        help="Acknowledge mutating actions for journey shortcuts.",
    )
    p_run.add_argument(
        "--expert",
        action="store_true",
        help="Bypass journey safety checks.",
    )

    p_prof = sp.add_parser("profile", help="Manage UX presets")
    p_prof_sp = p_prof.add_subparsers(dest="profile_cmd", required=True)
    p_prof_sp.add_parser("list", help="List available presets")
    p_use = p_prof_sp.add_parser("use", help="Merge a preset into UX config")
    p_use.add_argument("name")
    p_use.add_argument("--config", default=str(_DEFAULT_UX_PATH))

    p_doc = sp.add_parser("doctor", help="Validate and summarize resolved config")
    p_doc.add_argument("--config", default=str(_DEFAULT_UX_PATH))

    p_exp = sp.add_parser("explain", help="Explain UX->legacy key mapping")
    p_exp.add_argument("--config", default=str(_DEFAULT_UX_PATH))
    return p


def _render_cfg_for_mode(mode: str, level: str) -> Dict[str, Any]:
    specs = specs_for_mode_level(mode, level)
    blocks = grouped_specs(specs)
    cfg: Dict[str, Any] = {"run": {"mode": mode}}
    for block, items in blocks.items():
        vals = {s.name: default_for_spec(s) for s in items}
        if block == "pipeline":
            cfg["pipeline"] = vals
        elif block in {"paths", "data", "run", "workflow"}:
            cfg.setdefault(block, {}).update(vals)
        elif block.startswith("policy."):
            sec = block.split(".", 1)[1]
            cfg.setdefault("policy", {})
            cfg["policy"].setdefault(sec, {})
            cfg["policy"][sec].update(vals)
        else:
            cfg.setdefault("pipeline", {}).update(vals)

    # Keep UX mode explicit and practical defaults.
    cfg.setdefault("run", {})["mode"] = mode
    cfg.setdefault("paths", {}).setdefault("home_dir", "/path/to/repo")
    cfg.setdefault("paths", {}).setdefault("dataset_name", ".")
    cfg.setdefault("paths", {}).setdefault("results_dir", "results")
    cfg.setdefault("paths", {}).setdefault("singularity_image", "/path/to/tempo2.sif")
    cfg.setdefault("data", {}).setdefault("branches", ["main"])
    cfg.setdefault("data", {}).setdefault("reference_branch", "main")
    cfg.setdefault("data", {}).setdefault("pulsars", "ALL")
    cfg.setdefault("data", {}).setdefault("jobs", 4)
    return cfg


def _write_cfg(path: Path, cfg: Dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing file: {path}. Use --force.")
    write_ux_config(path, cfg)


def _init_workflow_template_3pass(root: Path, level: str, force: bool) -> list[Path]:
    run_cfg = root / "runs" / "pipeline" / "pleb.3pass-clean.pipeline.toml"
    wf_cfg = root / "workflows" / "pleb.3pass-clean.toml"
    ux_runner_cfg = root / "runs" / "workflow" / "pleb.3pass-clean.workflow.toml"

    run_payload = _render_cfg_for_mode("pipeline", max(level, "balanced"))
    run_payload.setdefault("data", {}).update({"pulsars": ["J1713+0747", "J1022+1001"], "jobs": 2})
    run_payload.setdefault("policy", {}).setdefault("fix", {}).update(
        {
            "apply": True,
            "infer_system_flags": True,
            "system_flag_overwrite_existing": True,
            "insert_missing_jumps": True,
            "remove_overlaps_exact": False,
        }
    )
    run_payload.setdefault("policy", {}).setdefault("pqc", {}).update(
        {
            "backend_col": "sys",
            "merge_tol_seconds": 1.0,
            "backend_profiles_path": "configs/rules/pqc/backend_profiles.example.toml",
            "step_delta_chi2_thresh": 10.0,
            "dm_step_delta_chi2_thresh": 10.0,
            "gaussian_bump_delta_chi2_thresh": 10.0,
            "glitch_delta_chi2_thresh": 10.0,
            "glitch_noise_k": 0.7,
        }
    )
    _write_cfg(run_cfg, run_payload, force)

    wf_text = """
workflow_version = 1
config = "configs/runs/pipeline/pleb.3pass-clean.pipeline.toml"
mode = "serial"

[[groups]]
name = "pass1_detect"
mode = "serial"
[[groups.steps]]
name = "pipeline"
[groups.steps.overrides]
outdir_name = "run_3pass_detect"
run_tempo2 = true
run_pqc = true
run_fix_dataset = true
fix_apply = true
fix_base_branch = "main"
fix_branch_name = "pass1_detect_features"
fix_qc_remove_outliers = false
fix_qc_remove_bad = false
fix_qc_remove_transients = false
fix_qc_remove_solar = false
fix_qc_remove_orbital_phase = false

[[groups]]
name = "pass2_apply_comments"
mode = "serial"
[[groups.steps]]
name = "fix_apply"
[groups.steps.overrides]
outdir_name = "run_3pass_apply"
run_tempo2 = false
run_pqc = false
run_fix_dataset = true
fix_apply = true
fix_base_branch = "pass1_detect_features"
fix_branch_name = "pass2_cleaned"
fix_qc_results_dir = "results/run_3pass_detect/main/qc"
fix_qc_branch = "main"
fix_qc_remove_outliers = true
fix_qc_action = "comment"
fix_qc_outlier_cols = ["bad_point", "robust_outlier", "robust_global_outlier", "bad_mad"]
fix_qc_remove_bad = true
fix_qc_remove_transients = false
fix_qc_remove_solar = false
fix_qc_remove_orbital_phase = false
fix_generate_alltim_variants = true
fix_jump_reference_variants = true

[[groups]]
name = "pass3_postclean_products"
mode = "serial"
[[groups.steps]]
name = "whitenoise"
[groups.steps.overrides]
outdir_name = "run_3pass_postclean"
fix_base_branch = "pass2_cleaned"
run_whitenoise = true
whitenoise_source_path = "/work/git_projects/whitenoise/src"

[[groups.steps]]
name = "compare_public"
[groups.steps.overrides]
compare_public_out_dir = "results/public_compare_3pass"
compare_public_providers_path = "configs/catalogs/public_releases/providers.toml"
"""
    if wf_cfg.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing file: {wf_cfg}. Use --force.")
    wf_cfg.parent.mkdir(parents=True, exist_ok=True)
    wf_cfg.write_text(wf_text.strip() + "\n", encoding="utf-8")

    ux_runner_payload = _render_cfg_for_mode("workflow", "minimal")
    ux_runner_payload.setdefault("workflow", {})["file"] = str(wf_cfg)
    _write_cfg(ux_runner_cfg, ux_runner_payload, force)
    return [run_cfg, wf_cfg, ux_runner_cfg]


def _init_workflow_template_golden_path(root: Path, force: bool) -> list[Path]:
    project_cfg = root / "project.toml"
    policy_cfg = root / "policy.toml"
    workflow_cfg = root / "workflows" / "workflow.toml"
    runner_cfg = root / "runs" / "workflow" / "pleb.workflow.toml"

    project = {
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
    }
    policy = {
        "run": {"mode": "pipeline", "run_tempo2": True, "run_fix_dataset": True, "run_pqc": True, "qc_report": True},
        "policy": {
            "fix": {
                "apply": True,
                "infer_system_flags": True,
                "system_flag_overwrite_existing": True,
                "insert_missing_jumps": True,
                "remove_overlaps_exact": False,
                "qc_action": "comment",
                "qc_remove_outliers": True,
                "qc_outlier_cols": ["bad_point", "robust_outlier", "robust_global_outlier", "bad_mad"],
            },
            "pqc": {"backend_col": "sys", "merge_tol_seconds": 1.0},
            "whitenoise": {"single_toa_mode": "combined", "fit_timing_model_first": True},
        },
    }
    for p, payload in ((project_cfg, project), (policy_cfg, policy)):
        _write_cfg(p, payload, force)

    wf_text = """
workflow_version = 1
config = "configs/runs/pipeline/pleb.3pass-clean.pipeline.toml"
mode = "serial"

[[groups]]
name = "detect"
mode = "serial"
[[groups.steps]]
name = "pipeline"
[groups.steps.overrides]
outdir_name = "run_detect"
fix_base_branch = "main"
fix_branch_name = "detect_branch"
fix_qc_remove_outliers = false
fix_qc_remove_bad = false

[[groups]]
name = "apply"
mode = "serial"
[[groups.steps]]
name = "fix_apply"
[groups.steps.overrides]
outdir_name = "run_apply"
fix_base_branch = "detect_branch"
fix_branch_name = "clean_branch"
run_pqc = false
run_tempo2 = false

[[groups]]
name = "publish"
mode = "serial"
[[groups.steps]]
name = "whitenoise"
[groups.steps.overrides]
outdir_name = "run_publish"
fix_base_branch = "clean_branch"
run_whitenoise = true

[[groups.steps]]
name = "compare_public"
[groups.steps.overrides]
compare_public_out_dir = "results/public_compare_publish"
compare_public_providers_path = "configs/catalogs/public_releases/providers.toml"
"""
    if workflow_cfg.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing file: {workflow_cfg}. Use --force.")
    workflow_cfg.parent.mkdir(parents=True, exist_ok=True)
    workflow_cfg.write_text(wf_text.strip() + "\n", encoding="utf-8")

    runner = _render_cfg_for_mode("workflow", "minimal")
    runner.setdefault("workflow", {})["file"] = str(workflow_cfg)
    _write_cfg(runner_cfg, runner, force)
    return [project_cfg, policy_cfg, workflow_cfg, runner_cfg]


def _cmd_init(args: argparse.Namespace) -> int:
    if args.all_modes and args.config:
        raise SystemExit("--all-modes cannot be used with --config.")

    if args.workflow_template:
        outdir = Path(args.outdir)
        if args.workflow_template == "3pass-clean":
            paths = _init_workflow_template_3pass(outdir, args.level, args.force)
        else:
            paths = _init_workflow_template_golden_path(outdir, args.force)
        for p in paths:
            print(str(p))
        return 0

    if args.all_modes:
        outdir = Path(args.outdir)
        out: list[Path] = []
        for m in ("pipeline", "ingest", "workflow", "qc-report"):
            p = _mode_path(m, outdir)
            _write_cfg(p, _render_cfg_for_mode(m, args.level), args.force)
            out.append(p)
        for p in out:
            print(str(p))
        return 0

    mode = args.mode or "pipeline"
    if args.config:
        out_path = Path(args.config)
    elif args.mode:
        out_path = _mode_path(mode, Path("configs"))
    else:
        out_path = _DEFAULT_UX_PATH
    _write_cfg(out_path, _render_cfg_for_mode(mode, args.level), args.force)
    print(str(out_path))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    ux = load_ux_config(Path(args.config))
    ux_dict = _ux_to_dict(ux)

    profs: list[str] = []
    run_profile = ux_dict.get("run", {}).get("profile")
    if isinstance(run_profile, str):
        profs.extend([x.strip() for x in run_profile.split(",") if x.strip()])
    elif isinstance(run_profile, list):
        profs.extend([str(x).strip() for x in run_profile if str(x).strip()])
    profs.extend([str(x).strip() for x in (args.profiles or []) if str(x).strip()])
    for p in profs:
        ux_dict = deep_merge(ux_dict, load_preset(p))

    for item in args.overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item!r}")
        k, vraw = item.split("=", 1)
        _set_dotted_key(ux_dict, k.strip(), _parse_value_as_toml_literal(vraw))

    ux = _dict_to_ux(ux_dict)
    legacy = ux_to_legacy_dict(ux)

    journey = getattr(args, "journey", None)
    if journey:
        if journey == "detect":
            legacy.update(
                {
                    "run_tempo2": True,
                    "run_fix_dataset": True,
                    "run_pqc": True,
                    "qc_report": True,
                    "fix_apply": True,
                    "fix_qc_remove_outliers": False,
                    "fix_qc_remove_bad": False,
                    "fix_qc_remove_transients": False,
                    "fix_qc_remove_solar": False,
                    "fix_qc_remove_orbital_phase": False,
                }
            )
        elif journey == "apply":
            legacy.update(
                {
                    "run_tempo2": False,
                    "run_fix_dataset": True,
                    "run_pqc": False,
                    "qc_report": False,
                    "fix_apply": True,
                    "fix_qc_remove_outliers": True,
                    "fix_qc_action": "comment",
                    "fix_qc_outlier_cols": [
                        "bad_point",
                        "robust_outlier",
                        "robust_global_outlier",
                        "bad_mad",
                    ],
                }
            )
        elif journey == "publish":
            legacy.update(
                {
                    "run_tempo2": True,
                    "run_fix_dataset": False,
                    "run_pqc": False,
                    "qc_report": True,
                    "run_whitenoise": True,
                }
            )
        if journey in {"detect", "apply"} and not (args.confirm or args.expert):
            raise SystemExit(
                f"Journey '{journey}' is mutating. Re-run with --confirm (or --expert)."
            )

    if args.plan:
        mode_plan = str(ux.run.get("mode", "pipeline")).strip().lower() or "pipeline"
        print("Execution plan")
        print(f"  mode={mode_plan}")
        print(f"  run_tempo2={legacy.get('run_tempo2')}")
        print(f"  run_fix_dataset={legacy.get('run_fix_dataset')}")
        print(f"  fix_apply={legacy.get('fix_apply')}")
        print(f"  run_pqc={legacy.get('run_pqc')}")
        print(f"  run_whitenoise={legacy.get('run_whitenoise', False)}")
        print(f"  qc_report={legacy.get('qc_report')}")
        return 0

    mode = str(ux.run.get("mode", "pipeline")).strip().lower() or "pipeline"
    return _dispatch_legacy(mode, legacy)


def _cmd_profile(args: argparse.Namespace) -> int:
    if args.profile_cmd == "list":
        for n in list_presets():
            print(n)
        return 0
    if args.profile_cmd == "use":
        path = Path(args.config)
        base = _ux_to_dict(load_ux_config(path)) if path.exists() else {}
        merged = deep_merge(base, load_preset(args.name))
        write_ux_config(path, merged)
        print(str(path))
        return 0
    raise SystemExit(f"Unknown profile command: {args.profile_cmd}")


def _cmd_doctor(args: argparse.Namespace) -> int:
    ux = load_ux_config(Path(args.config))
    legacy = ux_to_legacy_dict(ux)
    mode = str(ux.run.get("mode", "pipeline")).strip().lower()

    issues: list[str] = []
    invalid_paths: list[str] = []
    invalid_values: list[str] = []
    undefined_keys: list[str] = []

    known_keys = {
        *(f.name for f in fields(PipelineConfig)),
        *(f.name for f in fields(IngestConfig)),
        *(f.name for f in fields(WorkflowRunConfig)),
        "run_dir",
        "backend_col",
        "backend",
        "report_dir",
        "no_plots",
        "structure_group_cols",
        "no_feature_plots",
        "compact_pdf",
        "compact_pdf_name",
        "qc_report_run_dir",
        "qc_report_dir",
    }
    undefined_keys = sorted([k for k in legacy.keys() if k not in known_keys])

    placeholder_invalid_keys: set[str] = set()
    for k, v in legacy.items():
        if isinstance(v, str):
            s = v.strip()
            if ("/path/to/" in s) or ("<run_tag>" in s):
                placeholder_invalid_keys.add(k)

    missing: list[str]
    try:
        if mode in {"qc", "qc-report", "qc_report"}:
            qrun = legacy.get("run_dir") or legacy.get("qc_report_run_dir") or legacy.get("qc_report_dir")
            QCReportConfig.from_dict(
                {
                    "run_dir": qrun,
                    "backend_col": legacy.get("qc_report_backend_col", legacy.get("backend_col", "group")),
                    "backend": legacy.get("qc_report_backend", legacy.get("backend")),
                    "report_dir": legacy.get("qc_report_report_dir"),
                    "no_plots": legacy.get("qc_report_no_plots", False),
                    "structure_group_cols": legacy.get("qc_report_structure_group_cols"),
                    "no_feature_plots": legacy.get("qc_report_no_feature_plots", False),
                    "compact_pdf": legacy.get("qc_report_compact_pdf", False),
                    "compact_pdf_name": legacy.get("qc_report_compact_pdf_name", "qc_compact_report.pdf"),
                }
            )
            missing = [] if qrun else ["run_dir"]
            if qrun and not Path(str(qrun)).expanduser().exists():
                invalid_paths.append("run_dir")
        elif mode == "ingest":
            icfg = IngestConfig.from_dict(legacy)
            icfg.resolved_output_root()
            missing = [k for k in ("ingest_mapping_file",) if not legacy.get(k)]
            if legacy.get("ingest_mapping_file") and not Path(str(legacy["ingest_mapping_file"])).expanduser().exists():
                invalid_paths.append("ingest_mapping_file")
        elif mode == "workflow":
            WorkflowRunConfig.from_dict({"workflow_file": legacy.get("workflow_file")})
            missing = [] if legacy.get("workflow_file") else ["workflow_file"]
            wf_path = Path(str(legacy.get("workflow_file", ""))).expanduser()
            if legacy.get("workflow_file") and not wf_path.exists():
                invalid_paths.append("workflow_file")
            if wf_path.exists():
                try:
                    import tomllib  # py3.11+
                except Exception:
                    tomllib = None  # type: ignore
                if tomllib is not None and wf_path.suffix.lower() in {".toml", ".tml"}:
                    data = tomllib.loads(wf_path.read_text(encoding="utf-8"))
                    groups = list(data.get("groups", []) or [])
                    prev_branch = None
                    for gi, g in enumerate(groups):
                        steps = list(g.get("steps", []) or [])
                        for si, step in enumerate(steps):
                            name = str(step.get("name", "")).strip()
                            ov = dict(step.get("overrides", {}) or {})
                            if name in {"pipeline", "fix_apply", "fix_dataset"} and ov.get("fix_apply", False):
                                bname = ov.get("fix_branch_name")
                                if not bname:
                                    invalid_values.append(f"groups[{gi}].steps[{si}].overrides.fix_branch_name")
                                base = ov.get("fix_base_branch")
                                if prev_branch and base and str(base) != str(prev_branch):
                                    invalid_values.append(f"groups[{gi}].steps[{si}].overrides.fix_base_branch")
                                if bname:
                                    prev_branch = bname
        elif mode == "pipeline":
            PipelineConfig.from_dict(legacy)
            missing = [k for k in ("home_dir", "singularity_image") if not legacy.get(k)]
            if legacy.get("home_dir") and not Path(str(legacy["home_dir"])).expanduser().exists():
                invalid_paths.append("home_dir")
            if legacy.get("singularity_image") and not Path(str(legacy["singularity_image"])).expanduser().exists():
                invalid_paths.append("singularity_image")
            if legacy.get("run_whitenoise", False):
                wsp = legacy.get("whitenoise_source_path")
                if wsp and not Path(str(wsp)).expanduser().exists():
                    invalid_paths.append("whitenoise_source_path")
                if str(legacy.get("whitenoise_single_toa_mode", "combined")) not in {"combined", "equad0", "ecorr0"}:
                    invalid_values.append("whitenoise_single_toa_mode")
        else:
            missing = []
            invalid_values.append("run.mode")
    except Exception as e:
        missing = []
        issues.append(str(e))

    if invalid_paths:
        placeholder_invalid_keys.difference_update(set(invalid_paths))
    invalid_values.extend(sorted(placeholder_invalid_keys))

    print(f"mode={mode}")
    print(f"resolved_keys={len(legacy)}")
    print("missing_required=" + (",".join(missing) if missing else "none"))
    print("undefined_keys=" + (",".join(undefined_keys) if undefined_keys else "none"))
    print("invalid_paths=" + (",".join(sorted(set(invalid_paths))) if invalid_paths else "none"))
    print("invalid_values=" + (",".join(sorted(set(invalid_values))) if invalid_values else "none"))
    print("validation_errors=" + (" | ".join(issues) if issues else "none"))
    ok = not missing and not undefined_keys and not invalid_paths and not invalid_values and not issues
    return 0 if ok else 1


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
            run_dir = cfg_data.get("run_dir") or cfg_data.get("qc_report_run_dir") or cfg_data.get("qc_report_dir")
            if not run_dir:
                raise SystemExit("qc-report mode requires run_dir. Set [policy.report].run_dir or run_dir.")
            qc_cfg = {
                "qc_report": {
                    "run_dir": run_dir,
                    "backend_col": cfg_data.get("qc_report_backend_col", cfg_data.get("backend_col", "group")),
                    "backend": cfg_data.get("qc_report_backend", cfg_data.get("backend")),
                    "report_dir": cfg_data.get("qc_report_report_dir"),
                    "no_plots": cfg_data.get("qc_report_no_plots", False),
                    "structure_group_cols": cfg_data.get("qc_report_structure_group_cols"),
                    "no_feature_plots": cfg_data.get("qc_report_no_feature_plots", False),
                    "compact_pdf": cfg_data.get("qc_report_compact_pdf", False),
                    "compact_pdf_name": cfg_data.get("qc_report_compact_pdf_name", "qc_compact_report.pdf"),
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
