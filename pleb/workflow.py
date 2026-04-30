"""Workflow runner for multi-step pipeline sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import copy

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

import pandas as pd

from .config import IngestConfig, PipelineConfig
from .pipeline import run_pipeline
from .param_scan import run_param_scan
from .qc_report import generate_qc_report
from .public_release_compare import compare_public_releases
from .ingest import ingest_dataset
from .logging_utils import get_logger, set_log_dir
from .run_report import generate_run_report
from .config_io import _load_config_dict, _parse_value_as_toml_literal, _set_dotted_key

logger = get_logger("pleb.workflow")

try:
    from git import Repo, InvalidGitRepositoryError  # type: ignore
except Exception:  # pragma: no cover
    Repo = None  # type: ignore
    InvalidGitRepositoryError = Exception  # type: ignore


@dataclass
class WorkflowContext:
    """Mutable state shared across workflow steps.

    Attributes
    ----------
    last_run_dir : pathlib.Path or None
        Most recent pipeline run directory produced by a step.
    last_pipeline_run_dir : pathlib.Path or None
        Alias for the latest run directory from a full pipeline invocation.
    last_qc_summary : pathlib.Path or None
        Path to the latest QC summary artifact, when generated.
    last_fix_summary : pathlib.Path or None
        Path to the latest FixDataset summary artifact, when generated.
    """

    last_run_dir: Optional[Path] = None
    last_pipeline_run_dir: Optional[Path] = None
    last_qc_summary: Optional[Path] = None
    last_fix_summary: Optional[Path] = None
    step_records: List[Dict[str, Any]] = field(default_factory=list)


def _load_workflow(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in (".toml", ".tml"):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported workflow file type: {path.suffix}")
    wf_ver = int(data.get("workflow_version", 1) or 1)
    if wf_ver != 1:
        raise ValueError(
            f"Unsupported workflow_version={wf_ver}. Supported: workflow_version=1."
        )
    return data


def _resolve_config_path(config_arg: str | Path, *, base_dir: Path | None = None) -> Path:
    """Resolve a workflow-referenced config path against the workflow directory."""
    path = Path(config_arg).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


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
                "config": step.get("config"),
            }
        if "step" in step:
            return {
                "name": str(step["step"]),
                "set": list(step.get("set", []) or []),
                "overrides": dict(step.get("overrides", {}) or {}),
                "run_dir": step.get("run_dir"),
                "config": step.get("config"),
            }
        if len(step) == 1:
            name, payload = next(iter(step.items()))
            payload = payload or {}
            return {
                "name": str(name),
                "set": list(payload.get("set", []) or []),
                "overrides": dict(payload.get("overrides", {}) or {}),
                "run_dir": payload.get("run_dir"),
                "config": payload.get("config"),
            }
    raise ValueError(f"Invalid step format: {step!r}")


def _normalize_loop(loop: Any) -> Dict[str, Any]:
    if not isinstance(loop, dict):
        raise ValueError(f"Invalid loop format: {loop!r}")
    name = loop.get("name") or loop.get("loop") or ""
    return {
        "name": str(name) if name else None,
        "max_iters": int(loop.get("max_iters", 1) or 1),
        "mode": str(loop.get("mode", "serial") or "serial").lower(),
        "parallel_workers": int(loop.get("parallel_workers", 0) or 0),
        "steps": [_normalize_step(s) for s in loop.get("steps", [])],
        "groups": [_normalize_group(g) for g in loop.get("groups", [])],
        "stop_if": list(loop.get("stop_if", []) or []),
        "set": list(loop.get("set", []) or []),
        "overrides": dict(loop.get("overrides", {}) or {}),
    }


def _normalize_group(group: Any) -> Dict[str, Any]:
    if not isinstance(group, dict):
        raise ValueError(f"Invalid group format: {group!r}")
    name = group.get("name") or group.get("group") or ""
    mode = str(group.get("mode", "serial") or "serial").lower()
    return {
        "name": str(name) if name else None,
        "mode": mode,
        "parallel_workers": int(group.get("parallel_workers", 0) or 0),
        "steps": [_normalize_step(s) for s in group.get("steps", [])],
    }


def _apply_overrides(
    cfg_dict: Dict[str, Any], set_list: List[str], overrides: Dict[str, Any]
) -> Dict[str, Any]:
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


def _build_cfg(
    step_name: str,
    base_dict: Dict[str, Any],
    set_list: List[str],
    overrides: Dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> PipelineConfig | IngestConfig:
    d = _apply_overrides(base_dict, set_list, overrides)
    if step_name == "ingest":
        return IngestConfig.from_dict(d, base_dir=base_dir)
    return PipelineConfig.from_dict(d, base_dir=base_dir)


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
    cols = [
        c
        for c in df.columns
        if c in ("added_includes", "missing_jumps", "removed_lines")
    ]
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


def _run_step(
    step: Dict[str, Any],
    base_dict: Dict[str, Any],
    ctx: WorkflowContext,
    *,
    workflow_base_dir: Path | None = None,
    cfg_base_dir: Path | None = None,
) -> None:
    name = step["name"]
    step_cfg_path = step.get("config")
    step_base_dict = base_dict
    step_cfg_base_dir = cfg_base_dir
    resolved_step_cfg_path: Path | None = None
    if step_cfg_path:
        resolved_step_cfg_path = _resolve_config_path(
            str(step_cfg_path), base_dir=workflow_base_dir
        )
        if name == "optimize":
            from .optimize.cli import load_optimization_config
            from .optimize.optimizer import run_optimization

            ocfg = load_optimization_config(resolved_step_cfg_path)
            result = run_optimization(ocfg)
            ctx.last_run_dir = result.out_dir
            ctx.step_records.append(
                {
                    "step": name,
                    "kind": name,
                    "run_dir": str(result.out_dir),
                    "fix_summary": "",
                    "qc_summary": "",
                }
            )
            return
        step_base_dict = _load_config_dict(str(resolved_step_cfg_path))
        step_cfg_base_dir = resolved_step_cfg_path.parent
    cfg = _build_cfg(
        name,
        step_base_dict,
        step.get("set", []),
        step.get("overrides", {}),
        base_dir=step_cfg_base_dir,
    )
    if name in (
        "pipeline",
        "fix_dataset",
        "fix_apply",
        "param_scan",
        "whitenoise",
        "compare_public",
    ):
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
                base_branch = (
                    str(getattr(cfg, "fix_base_branch", "") or "main").strip() or "main"
                )
                repo.git.checkout("-b", base_branch)
                try:
                    repo.git.commit(
                        "--allow-empty", "-m", "Initialize ingest repository"
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to create initial git commit for fix_apply. "
                        "Configure git user.name/user.email and retry."
                    ) from e

    if name == "ingest":
        if not getattr(cfg, "ingest_mapping_file", None) or not getattr(
            cfg, "ingest_output_dir", None
        ):
            raise RuntimeError(
                "ingest step requires ingest_mapping_file and ingest_output_dir in config."
            )
        set_log_dir(Path(cfg.ingest_output_dir) / "logs")
        ingest_dataset(
            Path(cfg.ingest_mapping_file),
            Path(cfg.ingest_output_dir),
            verify=bool(getattr(cfg, "ingest_verify", False)),
            report_metadata={
                "fix_ensure_ephem": getattr(cfg, "fix_ensure_ephem", None),
                "fix_ensure_clk": getattr(cfg, "fix_ensure_clk", None),
                "fix_ensure_ne_sw": getattr(cfg, "fix_ensure_ne_sw", None),
                "ingest_commit_branch_name": getattr(
                    cfg, "ingest_commit_branch_name", None
                ),
                "ingest_commit_base_branch": getattr(
                    cfg, "ingest_commit_base_branch", None
                ),
            },
        )
        from .ingest import commit_ingest_changes

        commit_ingest_changes(
            Path(cfg.ingest_output_dir),
            branch_name=getattr(cfg, "ingest_commit_branch_name", None),
            base_branch=getattr(cfg, "ingest_commit_base_branch", None),
            commit_message=getattr(cfg, "ingest_commit_message", None),
        )
        ctx.step_records.append(
            {
                "step": name,
                "kind": name,
                "run_dir": str(Path(cfg.ingest_output_dir)),
                "fix_summary": "",
                "qc_summary": "",
            }
        )
        return

    if name == "pipeline":
        out_paths = run_pipeline(cfg)
        ctx.last_run_dir = out_paths.get("tag")
        ctx.last_pipeline_run_dir = out_paths.get("tag")
        ctx.last_fix_summary = _find_latest_fix_summary(out_paths)
        ctx.last_qc_summary = _find_qc_summary(out_paths)
        ctx.step_records.append(
            {
                "step": name,
                "kind": name,
                "run_dir": str(out_paths.get("tag") or ""),
                "fix_summary": str(ctx.last_fix_summary or ""),
                "qc_summary": str(ctx.last_qc_summary or ""),
            }
        )
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
        ctx.step_records.append(
            {
                "step": name,
                "kind": name,
                "run_dir": str(out_paths.get("tag") or ""),
                "fix_summary": str(ctx.last_fix_summary or ""),
                "qc_summary": str(ctx.last_qc_summary or ""),
            }
        )
        return

    if name == "fix_apply":
        cfg.run_fix_dataset = True
        cfg.fix_apply = True
        out_paths = run_pipeline(cfg)
        ctx.last_run_dir = out_paths.get("tag")
        ctx.last_pipeline_run_dir = out_paths.get("tag")
        ctx.last_fix_summary = _find_latest_fix_summary(out_paths)
        ctx.last_qc_summary = _find_qc_summary(out_paths)
        ctx.step_records.append(
            {
                "step": name,
                "kind": name,
                "run_dir": str(out_paths.get("tag") or ""),
                "fix_summary": str(ctx.last_fix_summary or ""),
                "qc_summary": str(ctx.last_qc_summary or ""),
            }
        )
        return

    if name == "param_scan":
        out_paths = run_param_scan(
            cfg,
            scan_typical=bool(getattr(cfg, "param_scan_typical", False)),
            dm_redchisq_threshold=getattr(
                cfg, "param_scan_dm_redchisq_threshold", None
            ),
            dm_max_order=getattr(cfg, "param_scan_dm_max_order", None),
            btx_max_fb=getattr(cfg, "param_scan_btx_max_fb", None),
        )
        ctx.last_run_dir = out_paths.get("tag")
        ctx.step_records.append(
            {
                "step": name,
                "kind": name,
                "run_dir": str(out_paths.get("tag") or ""),
                "fix_summary": "",
                "qc_summary": "",
            }
        )
        return

    if name == "whitenoise":
        # Whitenoise-only step: run pipeline harness with only white-noise stage enabled.
        cfg.run_whitenoise = True
        cfg.run_tempo2 = False
        cfg.run_pqc = False
        cfg.qc_report = False
        cfg.run_fix_dataset = False
        cfg.make_plots = False
        cfg.make_reports = False
        cfg.make_covmat = False
        out_paths = run_pipeline(cfg)
        ctx.last_run_dir = out_paths.get("tag")
        ctx.last_pipeline_run_dir = out_paths.get("tag")
        ctx.step_records.append(
            {
                "step": name,
                "kind": name,
                "run_dir": str(out_paths.get("tag") or ""),
                "fix_summary": "",
                "qc_summary": "",
            }
        )
        return

    if name == "compare_public":
        out_dir = getattr(cfg, "compare_public_out_dir", None)
        if out_dir is None:
            if ctx.last_pipeline_run_dir:
                out_dir = Path(ctx.last_pipeline_run_dir) / "public_release_compare"
            else:
                out_dir = Path(cfg.results_dir) / "public_release_compare"
        providers_path = getattr(cfg, "compare_public_providers_path", None)
        out = compare_public_releases(
            out_dir=Path(out_dir),
            providers_path=(Path(providers_path) if providers_path else None),
        )
        ctx.last_run_dir = Path(out["out_dir"])
        ctx.step_records.append(
            {
                "step": name,
                "kind": name,
                "run_dir": str(ctx.last_run_dir or ""),
                "fix_summary": "",
                "qc_summary": "",
            }
        )
        return

    if name == "qc_report":
        run_dir = step.get("run_dir")
        if run_dir:
            run_dir = Path(run_dir)
        else:
            run_dir = ctx.last_pipeline_run_dir or ctx.last_run_dir
        if not run_dir:
            raise RuntimeError(
                "qc_report step requires a prior pipeline run or explicit run_dir."
            )
        generate_qc_report(
            run_dir=run_dir,
            backend_col=str(
                getattr(cfg, "qc_report_backend_col", None)
                or getattr(cfg, "pqc_backend_col", "group")
                or "group"
            ),
            backend=(
                str(cfg.qc_report_backend)
                if getattr(cfg, "qc_report_backend", None)
                else None
            ),
            report_dir=(
                Path(cfg.qc_report_dir) if getattr(cfg, "qc_report_dir", None) else None
            ),
            no_plots=bool(getattr(cfg, "qc_report_no_plots", False)),
            structure_group_cols=(
                str(getattr(cfg, "qc_report_structure_group_cols", None))
                if getattr(cfg, "qc_report_structure_group_cols", None)
                else None
            ),
            no_feature_plots=bool(getattr(cfg, "qc_report_no_feature_plots", False)),
            compact_pdf=bool(getattr(cfg, "qc_report_compact_pdf", False)),
            compact_pdf_name=str(
                getattr(cfg, "qc_report_compact_pdf_name", "qc_compact_report.pdf")
            ),
            compact_outlier_cols=getattr(cfg, "qc_report_compact_outlier_cols", None),
        )
        return

    raise ValueError(f"Unknown workflow step: {name}")


def _merge_context(dst: WorkflowContext, src: WorkflowContext) -> None:
    if src.last_run_dir is not None:
        dst.last_run_dir = src.last_run_dir
    if src.last_pipeline_run_dir is not None:
        dst.last_pipeline_run_dir = src.last_pipeline_run_dir
    if src.last_qc_summary is not None:
        dst.last_qc_summary = src.last_qc_summary
    if src.last_fix_summary is not None:
        dst.last_fix_summary = src.last_fix_summary
    if src.step_records:
        dst.step_records.extend(src.step_records)


def _run_steps_serial(
    steps: List[Dict[str, Any]],
    base_dict: Dict[str, Any],
    ctx: WorkflowContext,
    *,
    workflow_base_dir: Path | None = None,
    cfg_base_dir: Path | None = None,
    label_prefix: str = "",
) -> None:
    for idx, s in enumerate(steps, start=1):
        logger.info("%sStep %s/%s: %s", label_prefix, idx, len(steps), s["name"])
        _run_step(
            s,
            base_dict,
            ctx,
            workflow_base_dir=workflow_base_dir,
            cfg_base_dir=cfg_base_dir,
        )


def _run_steps_parallel(
    steps: List[Dict[str, Any]],
    base_dict: Dict[str, Any],
    ctx: WorkflowContext,
    *,
    workflow_base_dir: Path | None = None,
    cfg_base_dir: Path | None = None,
    workers: int = 0,
    label_prefix: str = "",
) -> None:
    if not steps:
        return
    max_workers = int(workers or 0)
    if max_workers <= 0:
        max_workers = len(steps)
    max_workers = max(1, min(max_workers, len(steps)))
    logger.info(
        "%sRunning %d steps in parallel (workers=%d)",
        label_prefix,
        len(steps),
        max_workers,
    )
    results: Dict[int, WorkflowContext] = {}
    errors: List[str] = []

    def _worker(i: int, st: Dict[str, Any]) -> tuple[int, WorkflowContext]:
        local_ctx = WorkflowContext(
            last_run_dir=ctx.last_run_dir,
            last_pipeline_run_dir=ctx.last_pipeline_run_dir,
            last_qc_summary=ctx.last_qc_summary,
            last_fix_summary=ctx.last_fix_summary,
        )
        logger.info(
            "%s[parallel %d/%d] %s", label_prefix, i + 1, len(steps), st["name"]
        )
        _run_step(
            st,
            base_dict,
            local_ctx,
            workflow_base_dir=workflow_base_dir,
            cfg_base_dir=cfg_base_dir,
        )
        return i, local_ctx

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_worker, i, s): i for i, s in enumerate(steps)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                idx, local_ctx = fut.result()
                results[idx] = local_ctx
            except Exception as e:  # pragma: no cover
                errors.append(f"step[{i}] {steps[i].get('name')}: {e}")

    if errors:
        raise RuntimeError("Parallel workflow step failure(s): " + " | ".join(errors))

    # Deterministic merge in declared order.
    for i in range(len(steps)):
        if i in results:
            _merge_context(ctx, results[i])


def _run_step_sequence(
    steps: List[Dict[str, Any]],
    base_dict: Dict[str, Any],
    ctx: WorkflowContext,
    *,
    workflow_base_dir: Path | None = None,
    cfg_base_dir: Path | None = None,
    mode: str = "serial",
    parallel_workers: int = 0,
    label_prefix: str = "",
) -> None:
    m = str(mode or "serial").strip().lower()
    if m not in {"serial", "parallel"}:
        raise ValueError(
            f"Unsupported workflow mode: {mode!r}; use 'serial' or 'parallel'."
        )
    if m == "serial":
        _run_steps_serial(
            steps,
            base_dict,
            ctx,
            workflow_base_dir=workflow_base_dir,
            cfg_base_dir=cfg_base_dir,
            label_prefix=label_prefix,
        )
        return
    _run_steps_parallel(
        steps,
        base_dict,
        ctx,
        workflow_base_dir=workflow_base_dir,
        cfg_base_dir=cfg_base_dir,
        workers=int(parallel_workers or 0),
        label_prefix=label_prefix,
    )


def run_workflow(path: Path) -> WorkflowContext:
    """Execute a workflow file with optional serial/parallel stage control.

    Parameters
    ----------
    path : pathlib.Path
        Workflow definition file (TOML or JSON).

    Returns
    -------
    WorkflowContext
        Final context populated with latest run and summary artifact paths.

    Raises
    ------
    FileNotFoundError
        If the workflow or referenced config file does not exist.
    ValueError
        If workflow schema is invalid (missing config, unsupported mode, etc.).

    Notes
    -----
    Execution model:

    - top-level steps and groups are executed first,
    - groups create barriers between internal serial/parallel blocks,
    - loop sections then execute with per-loop overrides and the same
      serial/parallel semantics.

    Parallel execution here coordinates independent workflow steps; it does not
    change per-step internal parallelism (e.g., pulsar-level workers inside a
    pipeline step).
    """
    path = Path(path).expanduser().resolve()
    wf = _load_workflow(path)
    workflow_base_dir = path.parent
    config_path = wf.get("config")
    if not config_path:
        raise ValueError("Workflow must specify 'config' (path to pipeline config).")
    resolved_config_path = _resolve_config_path(
        str(config_path), base_dir=workflow_base_dir
    )
    base_dict = _load_config_dict(str(resolved_config_path))
    base_cfg_base_dir = resolved_config_path.parent
    base_dict = _apply_overrides(
        base_dict, list(wf.get("set", []) or []), dict(wf.get("overrides", {}) or {})
    )

    ctx = WorkflowContext()
    global_mode = str(wf.get("mode", "serial") or "serial").lower()
    global_parallel_workers = int(wf.get("parallel_workers", 0) or 0)

    # Top-level steps
    steps = wf.get("steps", [])
    if steps:
        logger.info("Workflow: running %s top-level steps", len(steps))
        norm_steps = [_normalize_step(step) for step in steps]
        _run_step_sequence(
            norm_steps,
            base_dict,
            ctx,
            workflow_base_dir=workflow_base_dir,
            cfg_base_dir=base_cfg_base_dir,
            mode=global_mode,
            parallel_workers=global_parallel_workers,
        )

    # Top-level grouped stages (barrier between groups).
    groups = [_normalize_group(g) for g in list(wf.get("groups", []) or [])]
    if groups:
        logger.info("Workflow: running %s top-level groups", len(groups))
        for gi, gp in enumerate(groups, start=1):
            gname = gp.get("name") or f"group_{gi}"
            logger.info("Group %s/%s: %s", gi, len(groups), gname)
            _run_step_sequence(
                gp["steps"],
                base_dict,
                ctx,
                workflow_base_dir=workflow_base_dir,
                cfg_base_dir=base_cfg_base_dir,
                mode=str(gp.get("mode") or global_mode),
                parallel_workers=int(
                    gp.get("parallel_workers") or global_parallel_workers
                ),
                label_prefix=f"[{gname}] ",
            )

    loops = wf.get("loops", [])
    for li, loop in enumerate(loops, start=1):
        lp = _normalize_loop(loop)
        lname = lp.get("name") or f"loop_{li}"
        for it in range(1, lp["max_iters"] + 1):
            logger.info("Loop %s (%s/%s)", lname, it, lp["max_iters"])
            loop_base = _apply_overrides(
                base_dict, lp.get("set", []), lp.get("overrides", {})
            )
            if lp.get("groups"):
                loop_groups = list(lp.get("groups") or [])
                logger.info("  Loop %s: running %d groups", lname, len(loop_groups))
                for gi, gp in enumerate(loop_groups, start=1):
                    gname = gp.get("name") or f"{lname}_group_{gi}"
                    logger.info("  Group %s/%s: %s", gi, len(loop_groups), gname)
                    _run_step_sequence(
                        gp["steps"],
                        loop_base,
                        ctx,
                        workflow_base_dir=workflow_base_dir,
                        cfg_base_dir=base_cfg_base_dir,
                        mode=str(gp.get("mode") or lp.get("mode") or "serial"),
                        parallel_workers=int(
                            gp.get("parallel_workers")
                            or lp.get("parallel_workers")
                            or 0
                        ),
                        label_prefix=f"[{gname}] ",
                    )
            else:
                _run_step_sequence(
                    lp["steps"],
                    loop_base,
                    ctx,
                    workflow_base_dir=workflow_base_dir,
                    cfg_base_dir=base_cfg_base_dir,
                    mode=str(lp.get("mode") or "serial"),
                    parallel_workers=int(lp.get("parallel_workers") or 0),
                    label_prefix="  ",
                )
            if _should_stop(lp.get("stop_if", []), ctx):
                logger.info("Loop %s stopping early (stop_if condition met).", lname)
                break

    report_enabled = bool(base_dict.get("consolidated_report", True))
    report_stages = base_dict.get("consolidated_report_stages", None)
    report_title = base_dict.get("consolidated_report_title", None)
    report_name = base_dict.get("consolidated_report_name", None)

    report_run_dir = ctx.last_pipeline_run_dir or ctx.last_run_dir
    if report_enabled and report_run_dir is not None:
        try:
            report_path = generate_run_report(
                report_run_dir,
                title=(
                    str(report_title)
                    if report_title not in (None, "")
                    else f"PLEB Workflow Report: {Path(path).name}"
                ),
                output_name=(
                    str(report_name)
                    if report_name not in (None, "")
                    else "workflow_report.pdf"
                ),
                workflow_steps=ctx.step_records,
                include_stages=report_stages,
            )
            if report_path is not None:
                logger.info("Workflow report written to: %s", report_path)
        except Exception as e:
            logger.warning("Workflow report generation failed: %s", e)

    return ctx
