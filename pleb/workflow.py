"""Workflow runner for multi-step pipeline sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import copy

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

import pandas as pd

from .config import PipelineConfig
from .pipeline import run_pipeline
from .param_scan import run_param_scan
from .qc_report import generate_qc_report
from .ingest import ingest_dataset
from .logging_utils import get_logger, set_log_dir
from .config_io import _load_config_dict, _parse_value_as_toml_literal, _set_dotted_key

logger = get_logger("pleb.workflow")

try:
    from git import Repo, InvalidGitRepositoryError  # type: ignore
except Exception:  # pragma: no cover
    Repo = None  # type: ignore
    InvalidGitRepositoryError = Exception  # type: ignore


@dataclass
class WorkflowContext:
    last_run_dir: Optional[Path] = None
    last_pipeline_run_dir: Optional[Path] = None
    last_qc_summary: Optional[Path] = None
    last_fix_summary: Optional[Path] = None


def _load_workflow(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in (".toml", ".tml"):
        return tomllib.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported workflow file type: {path.suffix}")


def _normalize_step(step: Any) -> Dict[str, Any]:
    if isinstance(step, str):
        return {"name": step, "set": [], "overrides": {}}
    if isinstance(step, dict):
        if "name" in step:
            return {
                "name": str(step["name"]),
                "set": list(step.get("set", []) or []),
                "overrides": dict(step.get("overrides", {}) or {}),
                "run_dir": step.get("run_dir"),
            }
        if "step" in step:
            return {
                "name": str(step["step"]),
                "set": list(step.get("set", []) or []),
                "overrides": dict(step.get("overrides", {}) or {}),
                "run_dir": step.get("run_dir"),
            }
        if len(step) == 1:
            name, payload = next(iter(step.items()))
            payload = payload or {}
            return {
                "name": str(name),
                "set": list(payload.get("set", []) or []),
                "overrides": dict(payload.get("overrides", {}) or {}),
                "run_dir": payload.get("run_dir"),
            }
    raise ValueError(f"Invalid step format: {step!r}")


def _normalize_loop(loop: Any) -> Dict[str, Any]:
    if not isinstance(loop, dict):
        raise ValueError(f"Invalid loop format: {loop!r}")
    name = loop.get("name") or loop.get("loop") or ""
    return {
        "name": str(name) if name else None,
        "max_iters": int(loop.get("max_iters", 1) or 1),
        "steps": [_normalize_step(s) for s in loop.get("steps", [])],
        "stop_if": list(loop.get("stop_if", []) or []),
        "set": list(loop.get("set", []) or []),
        "overrides": dict(loop.get("overrides", {}) or {}),
    }


def _apply_overrides(cfg_dict: Dict[str, Any], set_list: List[str], overrides: Dict[str, Any]) -> Dict[str, Any]:
    d = copy.deepcopy(cfg_dict)
    for item in set_list or []:
        if "=" not in item:
            raise ValueError(f"workflow set expects KEY=VALUE, got: {item!r}")
        k, vraw = item.split("=", 1)
        v = _parse_value_as_toml_literal(vraw)
        _set_dotted_key(d, k.strip(), v)
    for k, v in (overrides or {}).items():
        d[k] = v
    return d


def _build_cfg(base_dict: Dict[str, Any], set_list: List[str], overrides: Dict[str, Any]) -> PipelineConfig:
    d = _apply_overrides(base_dict, set_list, overrides)
    return PipelineConfig.from_dict(d)


def _find_latest_fix_summary(out_paths: Dict[str, Path]) -> Optional[Path]:
    root = out_paths.get("fix_dataset")
    if not root or not root.exists():
        return None
    candidates = list(root.rglob("fix_dataset_summary.tsv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_qc_summary(out_paths: Dict[str, Path]) -> Optional[Path]:
    qc_dir = out_paths.get("qc")
    if not qc_dir:
        return None
    path = qc_dir / "qc_summary.tsv"
    return path if path.exists() else None


def _stop_no_changes(ctx: WorkflowContext) -> bool:
    if not ctx.last_fix_summary or not ctx.last_fix_summary.exists():
        return False
    df = pd.read_csv(ctx.last_fix_summary, sep="\t")
    cols = [c for c in df.columns if c in ("added_includes", "missing_jumps", "removed_lines")]
    if not cols:
        return False
    return int(df[cols].fillna(0).sum().sum()) == 0


def _stop_qc_ok(ctx: WorkflowContext) -> bool:
    if not ctx.last_qc_summary or not ctx.last_qc_summary.exists():
        return False
    df = pd.read_csv(ctx.last_qc_summary, sep="\t")
    cols = [
        c
        for c in (
            "n_bad",
            "n_bad_days",
            "n_transient_toas",
            "n_solar_bad",
            "n_orbital_phase_bad",
        )
        if c in df.columns
    ]
    if not cols:
        return False
    return int(df[cols].fillna(0).sum().sum()) == 0


def _should_stop(stop_if: List[Any], ctx: WorkflowContext) -> bool:
    for cond in stop_if or []:
        if isinstance(cond, dict):
            if cond.get("no_changes"):
                if _stop_no_changes(ctx):
                    return True
            if cond.get("qc_ok"):
                if _stop_qc_ok(ctx):
                    return True
        if isinstance(cond, str):
            if cond == "no_changes" and _stop_no_changes(ctx):
                return True
            if cond == "qc_ok" and _stop_qc_ok(ctx):
                return True
    return False


def _run_step(step: Dict[str, Any], base_dict: Dict[str, Any], ctx: WorkflowContext) -> None:
    name = step["name"]
    cfg = _build_cfg(base_dict, step.get("set", []), step.get("overrides", {}))
    if name in ("pipeline", "fix_dataset", "fix_apply", "param_scan"):
        try:
            home_dir = Path(cfg.home_dir)
        except Exception:
            home_dir = None
        if home_dir is not None and not home_dir.exists():
            ingest_root = getattr(cfg, "ingest_output_dir", None)
            if ingest_root:
                cfg.home_dir = Path(ingest_root)
                home_dir = Path(cfg.home_dir)
            home_dir.mkdir(parents=True, exist_ok=True)
            logger.warning("home_dir did not exist; created %s.", home_dir)
        if name in ("fix_dataset", "fix_apply") and getattr(cfg, "fix_apply", False):
            if Repo is None:
                raise RuntimeError("GitPython is required for fix_apply workflows.")
            try:
                repo = Repo(str(home_dir), search_parent_directories=False)
            except InvalidGitRepositoryError:
                repo = Repo.init(str(home_dir))
                base_branch = str(getattr(cfg, "fix_base_branch", "") or "main").strip() or "main"
                repo.git.checkout("-b", base_branch)
                try:
                    repo.git.commit("--allow-empty", "-m", "Initialize ingest repository")
                except Exception as e:
                    raise RuntimeError(
                        "Failed to create initial git commit for fix_apply. "
                        "Configure git user.name/user.email and retry."
                    ) from e

    if name == "ingest":
        if not getattr(cfg, "ingest_mapping_file", None) or not getattr(cfg, "ingest_output_dir", None):
            raise RuntimeError("ingest step requires ingest_mapping_file and ingest_output_dir in config.")
        set_log_dir(Path(cfg.ingest_output_dir) / "logs")
        ingest_dataset(
            Path(cfg.ingest_mapping_file),
            Path(cfg.ingest_output_dir),
            verify=bool(getattr(cfg, "ingest_verify", False)),
        )
        from .ingest import commit_ingest_changes

        commit_ingest_changes(
            Path(cfg.ingest_output_dir),
            branch_name=getattr(cfg, "ingest_commit_branch_name", None),
            base_branch=getattr(cfg, "ingest_commit_base_branch", None),
            commit_message=getattr(cfg, "ingest_commit_message", None),
        )
        return

    if name == "pipeline":
        out_paths = run_pipeline(cfg)
        ctx.last_run_dir = out_paths.get("tag")
        ctx.last_pipeline_run_dir = out_paths.get("tag")
        ctx.last_fix_summary = _find_latest_fix_summary(out_paths)
        ctx.last_qc_summary = _find_qc_summary(out_paths)
        return

    if name == "fix_dataset":
        cfg.run_fix_dataset = True
        # Honor fix_apply/fix_* settings provided in the config/overrides.
        cfg.fix_apply = bool(getattr(cfg, "fix_apply", False))
        out_paths = run_pipeline(cfg)
        ctx.last_run_dir = out_paths.get("tag")
        ctx.last_pipeline_run_dir = out_paths.get("tag")
        ctx.last_fix_summary = _find_latest_fix_summary(out_paths)
        ctx.last_qc_summary = _find_qc_summary(out_paths)
        return

    if name == "fix_apply":
        cfg.run_fix_dataset = True
        cfg.fix_apply = True
        out_paths = run_pipeline(cfg)
        ctx.last_run_dir = out_paths.get("tag")
        ctx.last_pipeline_run_dir = out_paths.get("tag")
        ctx.last_fix_summary = _find_latest_fix_summary(out_paths)
        ctx.last_qc_summary = _find_qc_summary(out_paths)
        return

    if name == "param_scan":
        out_paths = run_param_scan(
            cfg,
            scan_typical=bool(getattr(cfg, "param_scan_typical", False)),
            dm_redchisq_threshold=getattr(cfg, "param_scan_dm_redchisq_threshold", None),
            dm_max_order=getattr(cfg, "param_scan_dm_max_order", None),
            btx_max_fb=getattr(cfg, "param_scan_btx_max_fb", None),
        )
        ctx.last_run_dir = out_paths.get("tag")
        return

    if name == "qc_report":
        run_dir = step.get("run_dir")
        if run_dir:
            run_dir = Path(run_dir)
        else:
            run_dir = ctx.last_pipeline_run_dir or ctx.last_run_dir
        if not run_dir:
            raise RuntimeError("qc_report step requires a prior pipeline run or explicit run_dir.")
        generate_qc_report(
            run_dir=run_dir,
            backend_col=str(getattr(cfg, "qc_report_backend_col", None) or getattr(cfg, "pqc_backend_col", "group") or "group"),
            backend=str(cfg.qc_report_backend) if getattr(cfg, "qc_report_backend", None) else None,
            report_dir=(Path(cfg.qc_report_dir) if getattr(cfg, "qc_report_dir", None) else None),
            no_plots=bool(getattr(cfg, "qc_report_no_plots", False)),
            structure_group_cols=str(getattr(cfg, "qc_report_structure_group_cols", None))
            if getattr(cfg, "qc_report_structure_group_cols", None)
            else None,
            no_feature_plots=bool(getattr(cfg, "qc_report_no_feature_plots", False)),
        )
        return

    raise ValueError(f"Unknown workflow step: {name}")


def run_workflow(path: Path) -> WorkflowContext:
    wf = _load_workflow(Path(path))
    config_path = wf.get("config")
    if not config_path:
        raise ValueError("Workflow must specify 'config' (path to pipeline config).")
    base_dict = _load_config_dict(str(config_path))
    base_dict = _apply_overrides(base_dict, list(wf.get("set", []) or []), dict(wf.get("overrides", {}) or {}))

    ctx = WorkflowContext()

    # Top-level steps
    steps = wf.get("steps", [])
    if steps:
        logger.info("Workflow: running %s top-level steps", len(steps))
        for idx, step in enumerate(steps, start=1):
            s = _normalize_step(step)
            logger.info("Step %s/%s: %s", idx, len(steps), s["name"])
            _run_step(s, base_dict, ctx)

    loops = wf.get("loops", [])
    for li, loop in enumerate(loops, start=1):
        lp = _normalize_loop(loop)
        lname = lp.get("name") or f"loop_{li}"
        for it in range(1, lp["max_iters"] + 1):
            logger.info("Loop %s (%s/%s)", lname, it, lp["max_iters"])
            loop_base = _apply_overrides(base_dict, lp.get("set", []), lp.get("overrides", {}))
            for si, step in enumerate(lp["steps"], start=1):
                logger.info("  Step %s/%s: %s", si, len(lp["steps"]), step["name"])
                _run_step(step, loop_base, ctx)
            if _should_stop(lp.get("stop_if", []), ctx):
                logger.info("Loop %s stopping early (stop_if condition met).", lname)
                break

    return ctx
