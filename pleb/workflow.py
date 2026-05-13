"""Workflow runner for multi-step pipeline sequences."""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import copy
import shutil
import subprocess

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

import pandas as pd

from .config import IngestConfig, PipelineConfig
from .pipeline import run_pipeline, _git_add_pathspecs
from .param_scan import run_param_scan
from .qc_report import generate_qc_report
from .public_release_compare import compare_public_releases
from .review_synthesis import run_review_synthesis
from .ingest import ingest_dataset
from .git_tools import branch_checked_out_in_worktree, temporary_detached_worktree
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


@dataclass(frozen=True)
class _WorktreePlan:
    start_ref: str
    commit_branch: Optional[str]
    commit_message: Optional[str]


def _git_branch_exists(repo_root: Path, branch: str) -> bool:
    branch = str(branch or "").strip()
    if not branch:
        return False
    res = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/heads/{branch}"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return res.returncode == 0


def _is_git_repo_root(repo_root: Path) -> bool:
    if not repo_root.exists() or not repo_root.is_dir():
        return False
    res = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return False
    try:
        return Path((res.stdout or "").strip()).resolve() == repo_root.resolve()
    except Exception:
        return False


def _path_rel_to_repo(path: Path, repo_root: Path) -> Optional[Path]:
    try:
        return path.expanduser().resolve().relative_to(repo_root.expanduser().resolve())
    except Exception:
        return None


def _remap_repo_local_path(
    path: Path, old_repo_root: Path, new_repo_root: Path
) -> Path:
    rel = _path_rel_to_repo(path, old_repo_root)
    if rel is None:
        return path.expanduser().resolve()
    return (new_repo_root / rel).resolve()


def _remap_repo_local_string_path(
    raw: Any, old_repo_root: Path, new_repo_root: Path
) -> Any:
    if raw in (None, ""):
        return raw
    text = str(raw).strip()
    if not text:
        return raw
    try:
        p = Path(text).expanduser()
    except Exception:
        return raw
    if not p.is_absolute():
        return raw
    rel = _path_rel_to_repo(p, old_repo_root)
    if rel is None:
        return raw
    return str((new_repo_root / rel).resolve())


def _rebase_cfg_for_worktree(
    cfg: PipelineConfig | IngestConfig, worktree_root: Path
) -> None:
    old_repo_root = (
        Path(cfg.home_dir).expanduser().resolve()
        if getattr(cfg, "home_dir", None) not in (None, "")
        else None
    )
    if old_repo_root is None:
        return
    new_repo_root = Path(worktree_root).expanduser().resolve()
    for fld in dataclass_fields(cfg):
        name = fld.name
        value = getattr(cfg, name)
        if name == "home_dir":
            setattr(cfg, name, new_repo_root)
            continue
        if name == "dataset_name" and value not in (None, ""):
            p = Path(value).expanduser()
            if p.is_absolute():
                rel = _path_rel_to_repo(p, old_repo_root)
                if rel is not None:
                    setattr(cfg, name, rel)
            continue
        if name == "compare_public_cache_dir":
            continue
        if isinstance(value, Path) and value.is_absolute():
            rel = _path_rel_to_repo(value, old_repo_root)
            if rel is None:
                continue
            setattr(cfg, name, (new_repo_root / rel).resolve())


def _rebase_workflow_overrides_for_worktree(
    step_effective_dict: Dict[str, Any],
    old_repo_root: Path,
    new_repo_root: Path,
) -> Dict[str, Any]:
    out = copy.deepcopy(step_effective_dict)
    for key in ("review_out_dir", "review_overrides", "compare_public_out_dir"):
        if key in out:
            out[key] = _remap_repo_local_string_path(
                out.get(key), old_repo_root, new_repo_root
            )
    if "review_stage_run" in out and isinstance(out["review_stage_run"], list):
        remapped: list[str] = []
        for item in out["review_stage_run"]:
            text = str(item)
            if "=" not in text:
                remapped.append(text)
                continue
            k, v = text.split("=", 1)
            remapped.append(
                f"{k}={_remap_repo_local_string_path(v, old_repo_root, new_repo_root)}"
            )
        out["review_stage_run"] = remapped
    return out


def _map_ctx_paths_from_worktree(
    ctx: WorkflowContext,
    *,
    worktree_root: Path,
    repo_root: Path,
    record_start: int,
) -> None:
    def _map_path_obj(path: Optional[Path]) -> Optional[Path]:
        if path is None:
            return None
        rel = _path_rel_to_repo(path, worktree_root)
        if rel is None:
            return path
        return (repo_root / rel).resolve()

    def _map_path_text(text: Any) -> Any:
        if text in (None, ""):
            return text
        try:
            p = Path(str(text)).expanduser()
        except Exception:
            return text
        rel = _path_rel_to_repo(p, worktree_root)
        if rel is None:
            return text
        return str((repo_root / rel).resolve())

    ctx.last_run_dir = _map_path_obj(ctx.last_run_dir)
    ctx.last_pipeline_run_dir = _map_path_obj(ctx.last_pipeline_run_dir)
    ctx.last_qc_summary = _map_path_obj(ctx.last_qc_summary)
    ctx.last_fix_summary = _map_path_obj(ctx.last_fix_summary)
    for rec in ctx.step_records[record_start:]:
        for key in ("run_dir", "fix_summary", "qc_summary"):
            if key in rec:
                rec[key] = _map_path_text(rec.get(key))


def _copy_tree_without_git(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name == ".git":
            continue
        target = dst / child.name
        if child.is_dir() and not child.is_symlink():
            _copy_tree_without_git(child, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)


def _collect_changed_and_untracked_paths(repo_root: Path) -> List[str]:
    def _run(*args: str) -> List[str]:
        res = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=False,
            check=False,
        )
        if res.returncode != 0:
            return []
        return [
            item.decode("utf-8", errors="replace")
            for item in res.stdout.split(b"\0")
            if item
        ]

    changed = _run("diff", "--name-only", "-z")
    untracked = _run("ls-files", "--others", "--exclude-standard", "-z")
    out: List[str] = []
    seen: set[str] = set()
    for item in [*changed, *untracked]:
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _commit_generated_changes_to_branch(
    *,
    source_worktree_root: Path,
    repo_root: Path,
    branch: str,
    commit_message: str,
) -> None:
    changed_paths = _collect_changed_and_untracked_paths(source_worktree_root)
    if not changed_paths:
        return
    checked_out = branch_checked_out_in_worktree(repo_root, branch)
    if checked_out is not None:
        raise RuntimeError(
            f"Cannot commit generated outputs onto branch '{branch}' because it is checked out in {checked_out}."
        )
    with temporary_detached_worktree(repo_root, branch) as target_root:
        if Repo is None:
            raise RuntimeError(
                "GitPython is required for branch-owned workflow commits."
            )
        wt_repo = Repo(str(target_root), search_parent_directories=False)
        wt_repo.git.checkout(branch)
        to_stage: List[str] = []
        for rel_path in changed_paths:
            src = source_worktree_root / rel_path
            dst = target_root / rel_path
            if src.is_dir():
                _copy_tree_without_git(src, dst)
                to_stage.append(rel_path)
            elif src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                to_stage.append(rel_path)
        if not to_stage:
            return
        _git_add_pathspecs(target_root.resolve(), to_stage)
        if wt_repo.is_dirty(untracked_files=True):
            wt_repo.index.commit(commit_message)


def _single_branch_from_cfg(cfg: PipelineConfig) -> Optional[str]:
    branches = [str(b).strip() for b in getattr(cfg, "branches", []) if str(b).strip()]
    branches = list(dict.fromkeys(branches))
    return branches[0] if len(branches) == 1 else None


def _single_source_branch_from_cfg(cfg: PipelineConfig) -> Optional[str]:
    branches = [str(b).strip() for b in getattr(cfg, "branches", []) if str(b).strip()]
    reference = str(getattr(cfg, "reference_branch", "") or "").strip()
    branch_set = list(dict.fromkeys([*branches, *([reference] if reference else [])]))
    return branch_set[0] if len(branch_set) == 1 else None


def _apply_optimize_fixed_overrides(
    target: Dict[str, Any], overrides: Dict[str, Any] | None
) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        dotted = str(key).strip()
        if not dotted:
            continue
        if dotted.startswith("backend_profile.") or dotted.startswith(
            "backend_profiles."
        ):
            continue
        _set_dotted_key(target, dotted, value)


def _infer_optimize_worktree_plan(
    optimize_cfg_path: Path,
) -> tuple[_WorktreePlan, Path]:
    from .optimize.cli import load_optimization_config

    ocfg = load_optimization_config(optimize_cfg_path)
    raw = _load_config_dict(str(ocfg.base_config_path))
    _apply_optimize_fixed_overrides(raw, ocfg.fixed_overrides)
    pcfg = PipelineConfig.from_dict(raw, base_dir=ocfg.base_config_path.parent)
    resolved = pcfg.resolved()
    repo_root = Path(resolved.home_dir).expanduser().resolve()
    branch = _single_source_branch_from_cfg(resolved)
    if not branch:
        raise RuntimeError(
            "Workflow optimize step requires a single source branch in the base "
            "pipeline config."
        )
    if not _git_branch_exists(repo_root, branch):
        raise RuntimeError(
            f"Workflow optimize step source branch does not exist: {branch}"
        )
    return (
        _WorktreePlan(
            start_ref=branch,
            commit_branch=branch,
            commit_message="Workflow optimize: generated outputs",
        ),
        repo_root,
    )


def _infer_worktree_plan(
    name: str,
    cfg: PipelineConfig | IngestConfig,
    step_effective_dict: Dict[str, Any],
    *,
    workflow_path: Path | None = None,
) -> Optional[_WorktreePlan]:
    home_dir = getattr(cfg, "home_dir", None)
    if home_dir in (None, ""):
        return None
    repo_root = Path(home_dir).expanduser().resolve()
    if not _is_git_repo_root(repo_root):
        return None

    if name == "ingest":
        target = (
            str(getattr(cfg, "ingest_commit_branch_name", "") or "raw_ingest").strip()
            or "raw_ingest"
        )
        base = (
            str(getattr(cfg, "ingest_commit_base_branch", "") or "main").strip()
            or "main"
        )
        start_ref = target if _git_branch_exists(repo_root, target) else base
        return _WorktreePlan(
            start_ref=start_ref, commit_branch=None, commit_message=None
        )

    if name in {"pipeline", "fix_dataset", "fix_apply"}:
        if bool(getattr(cfg, "fix_apply", False)):
            base = str(
                getattr(cfg, "fix_base_branch", "")
                or getattr(cfg, "reference_branch", "")
                or ""
            ).strip()
            if not base:
                return None
            return _WorktreePlan(
                start_ref=base, commit_branch=None, commit_message=None
            )
        branch = _single_branch_from_cfg(cfg)
        if not branch:
            return None
        return _WorktreePlan(
            start_ref=branch,
            commit_branch=branch,
            commit_message=f"Workflow {name}: generated outputs",
        )

    if name in {"param_scan", "whitenoise"}:
        branch = str(getattr(cfg, "reference_branch", "") or "").strip() or (
            _single_branch_from_cfg(cfg) or ""
        )
        if not branch:
            return None
        return _WorktreePlan(
            start_ref=branch,
            commit_branch=branch,
            commit_message=f"Workflow {name}: generated outputs",
        )

    if name == "compare_public":
        branch = str(
            step_effective_dict.get("compare_public_local_branch", "") or ""
        ).strip()
        if not branch:
            branch = str(getattr(cfg, "reference_branch", "") or "").strip()
        if not branch:
            branch = _single_branch_from_cfg(cfg) or ""
        if not branch:
            return None
        return _WorktreePlan(
            start_ref=branch,
            commit_branch=branch,
            commit_message="Workflow compare_public: generated outputs",
        )

    if name == "review_synthesis":
        review_final_branch = (
            str(step_effective_dict.get("review_final_branch", "") or "").strip()
            or f"{_infer_review_slug(step_effective_dict.get('review_slug'), step_effective_dict.get('review_final_branch'), workflow_path)}_step6_apply_delete"
        )
        if not review_final_branch.strip():
            return None
        return _WorktreePlan(
            start_ref=review_final_branch,
            commit_branch=review_final_branch,
            commit_message="Workflow review_synthesis: generated outputs",
        )

    return None


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


def _resolve_config_path(
    config_arg: str | Path, *, base_dir: Path | None = None
) -> Path:
    """Resolve a workflow-referenced config path against the workflow directory."""
    path = Path(config_arg).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _normalize_step(step: Any) -> Dict[str, Any]:
    def _normalize_set_from_toml(raw: Any) -> List[str]:
        if raw in (None, ""):
            return []
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            return [str(item) for item in raw if item not in (None, "")]
        raise ValueError(f"Invalid set_from_toml value: {raw!r}")

    if isinstance(step, str):
        return {"name": step, "set": [], "overrides": {}, "set_from_toml": []}
    if isinstance(step, dict):
        if "name" in step:
            return {
                "name": str(step["name"]),
                "set": list(step.get("set", []) or []),
                "overrides": dict(step.get("overrides", {}) or {}),
                "run_dir": step.get("run_dir"),
                "config": step.get("config"),
                "set_from_toml": _normalize_set_from_toml(step.get("set_from_toml")),
            }
        if "step" in step:
            return {
                "name": str(step["step"]),
                "set": list(step.get("set", []) or []),
                "overrides": dict(step.get("overrides", {}) or {}),
                "run_dir": step.get("run_dir"),
                "config": step.get("config"),
                "set_from_toml": _normalize_set_from_toml(step.get("set_from_toml")),
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
                "set_from_toml": _normalize_set_from_toml(payload.get("set_from_toml")),
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
    cfg_dict: Dict[str, Any],
    set_list: List[str],
    overrides: Dict[str, Any],
    *,
    file_overrides: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    d = copy.deepcopy(cfg_dict)
    for mapping in file_overrides or []:
        for k, v in mapping.items():
            _set_dotted_key(d, k, v)
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
    file_overrides: List[Dict[str, Any]] | None = None,
    *,
    base_dir: Path | None = None,
) -> PipelineConfig | IngestConfig:
    d = _apply_overrides(
        base_dict,
        set_list,
        overrides,
        file_overrides=file_overrides,
    )
    if step_name == "ingest":
        return IngestConfig.from_dict(d, base_dir=base_dir)
    return PipelineConfig.from_dict(d, base_dir=base_dir)


def _resolve_step_artifact_path(
    path_arg: str | Path,
    *,
    workflow_base_dir: Path | None = None,
    cfg_base_dir: Path | None = None,
) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates: List[Path] = []
    if workflow_base_dir is not None:
        candidates.append((workflow_base_dir / path).resolve())
    if cfg_base_dir is not None:
        candidate = (cfg_base_dir / path).resolve()
        if candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        return path.resolve()
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _flatten_override_mapping(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_override_mapping(value, dotted))
        else:
            out[dotted] = value
    return out


def _load_step_override_file(
    path_arg: str | Path,
    *,
    workflow_base_dir: Path | None = None,
    cfg_base_dir: Path | None = None,
) -> Dict[str, Any]:
    path = _resolve_step_artifact_path(
        path_arg,
        workflow_base_dir=workflow_base_dir,
        cfg_base_dir=cfg_base_dir,
    )
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in (".toml", ".tml"):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported override file type: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Override file must contain a top-level table/object: {path}")
    return _flatten_override_mapping(data)


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


def _normalize_step_values(raw: Any) -> List[str]:
    if raw in (None, ""):
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    return [text] if text else []


def _select_single_pulsar(raw: Any) -> Optional[str]:
    if raw in (None, ""):
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text.upper() == "ALL":
            return None
        return text
    if isinstance(raw, (list, tuple)):
        items = [str(item).strip() for item in raw if str(item).strip()]
        return items[0] if len(items) == 1 else None
    return None


def _resolve_repo_path(home_dir: Path, raw: Any) -> Optional[Path]:
    if raw in (None, ""):
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (home_dir / path).resolve()


def _infer_review_slug(
    explicit_slug: Any,
    explicit_final_branch: Any,
    workflow_path: Path | None,
) -> str:
    slug = str(explicit_slug or "").strip()
    if slug:
        return slug
    final_branch = str(explicit_final_branch or "").strip()
    if final_branch.endswith("_step6_apply_delete"):
        return final_branch[: -len("_step6_apply_delete")]
    if workflow_path is not None:
        stem = workflow_path.stem.strip()
        if stem:
            return stem.split("_", 1)[0]
    return ""


def _execute_step_body(
    *,
    name: str,
    cfg: PipelineConfig | IngestConfig,
    step_effective_dict: Dict[str, Any],
    ctx: WorkflowContext,
    step_run_dir: Any = None,
    workflow_path: Path | None = None,
) -> None:
    if name in (
        "pipeline",
        "fix_dataset",
        "fix_apply",
        "param_scan",
        "whitenoise",
        "compare_public",
        "review_synthesis",
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
            pulsars=step_effective_dict.get("pulsars"),
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
        cfg.run_whitenoise = True
        cfg.run_tempo2 = False
        cfg.run_pqc = False
        cfg.qc_report = False
        cfg.run_fix_dataset = False
        cfg.fix_apply = False
        cfg.make_binary_analysis = False
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
        local_dataset_root = None
        if getattr(cfg, "dataset_name", None):
            local_dataset_root = Path(cfg.dataset_name)
            if not local_dataset_root.is_absolute():
                local_dataset_root = Path(cfg.home_dir) / local_dataset_root
        local_branch = str(
            step_effective_dict.get("compare_public_local_branch", "") or ""
        ).strip()
        if not local_branch:
            local_branch = str(getattr(cfg, "reference_branch", "") or "").strip()
        if not local_branch:
            branches = [
                str(b).strip() for b in getattr(cfg, "branches", []) if str(b).strip()
            ]
            if len(branches) == 1:
                local_branch = branches[0]
        out = compare_public_releases(
            out_dir=Path(out_dir),
            providers_path=(Path(providers_path) if providers_path else None),
            cache_dir=(
                Path(cfg.compare_public_cache_dir)
                if getattr(cfg, "compare_public_cache_dir", None)
                else None
            ),
            local_dataset_root=local_dataset_root,
            local_branch=(local_branch or None),
            local_pulsars=getattr(cfg, "pulsars", None),
            alias_mapping_path=(
                Path(cfg.ingest_mapping_file)
                if getattr(cfg, "ingest_mapping_file", None)
                else None
            ),
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

    if name == "review_synthesis":
        home_dir = Path(cfg.home_dir).resolve()
        dataset_name = getattr(cfg, "dataset_name", None)
        if dataset_name in (None, ""):
            raise RuntimeError(
                "review_synthesis step requires dataset_name in the base pipeline config."
            )
        dataset_root = (home_dir / str(dataset_name)).resolve()
        results_root = Path(cfg.results_dir).resolve()
        review_psr = str(step_effective_dict.get("review_psr", "") or "").strip()
        if not review_psr:
            review_psr = _select_single_pulsar(getattr(cfg, "pulsars", None)) or ""
        if not review_psr:
            raise RuntimeError(
                "review_synthesis step requires review_psr, or cfg.pulsars must select a single pulsar."
            )
        review_slug = _infer_review_slug(
            step_effective_dict.get("review_slug"),
            step_effective_dict.get("review_final_branch"),
            workflow_path,
        )
        if not review_slug:
            raise RuntimeError(
                "review_synthesis step requires review_slug, review_final_branch, or a workflow filename that starts with the slug."
            )
        review_final_branch = (
            str(step_effective_dict.get("review_final_branch", "") or "").strip()
            or f"{review_slug}_step6_apply_delete"
        )
        review_workflow_config = str(
            step_effective_dict.get("review_workflow_config", "") or ""
        ).strip() or (str(workflow_path.resolve()) if workflow_path is not None else "")
        review_out_dir = _resolve_repo_path(
            home_dir, step_effective_dict.get("review_out_dir")
        )
        if review_out_dir is None:
            package_name = (
                Path(review_workflow_config).stem
                if review_workflow_config
                else review_slug
            )
            review_out_dir = (
                results_root / "review_packages" / review_psr / package_name
            ).resolve()
        review_overrides = _resolve_repo_path(
            home_dir, step_effective_dict.get("review_overrides")
        )
        result = run_review_synthesis(
            psr=review_psr,
            slug=review_slug,
            workflow_config=review_workflow_config or None,
            repo_root=home_dir,
            dataset_root=dataset_root,
            results_root=results_root,
            final_branch=review_final_branch,
            out=review_out_dir,
            overrides=review_overrides,
            max_keep_points=int(
                step_effective_dict.get("review_max_keep_points", 4000)
            ),
            top_n_rows=int(step_effective_dict.get("review_top_n_rows", 50)),
            stage_branch=_normalize_step_values(
                step_effective_dict.get("review_stage_branch")
            ),
            stage_run=_normalize_step_values(
                step_effective_dict.get("review_stage_run")
            ),
        )
        ctx.last_run_dir = result.out_dir
        ctx.last_pipeline_run_dir = result.out_dir
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

    if name == "qc_report":
        run_dir = (
            Path(step_run_dir)
            if step_run_dir not in (None, "")
            else (ctx.last_pipeline_run_dir or ctx.last_run_dir)
        )
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


def _run_step(
    step: Dict[str, Any],
    base_dict: Dict[str, Any],
    ctx: WorkflowContext,
    *,
    workflow_base_dir: Path | None = None,
    cfg_base_dir: Path | None = None,
    workflow_path: Path | None = None,
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

            worktree_plan, original_repo_root = _infer_optimize_worktree_plan(
                resolved_step_cfg_path
            )
            with temporary_detached_worktree(
                original_repo_root, worktree_plan.start_ref
            ) as worktree_root:
                worktree_cfg_path = _remap_repo_local_path(
                    resolved_step_cfg_path, original_repo_root, worktree_root
                )
                ocfg = load_optimization_config(worktree_cfg_path)
                result = run_optimization(ocfg)
                if worktree_plan.commit_branch:
                    _commit_generated_changes_to_branch(
                        source_worktree_root=worktree_root,
                        repo_root=original_repo_root,
                        branch=worktree_plan.commit_branch,
                        commit_message=str(
                            worktree_plan.commit_message
                            or "Workflow optimize: generated outputs"
                        ),
                    )
                mapped_out_dir = _remap_repo_local_path(
                    result.out_dir, worktree_root, original_repo_root
                )
                ctx.last_run_dir = mapped_out_dir
                ctx.step_records.append(
                    {
                        "step": name,
                        "kind": name,
                        "run_dir": str(mapped_out_dir),
                        "fix_summary": "",
                        "qc_summary": "",
                    }
                )
            return
        step_base_dict = _load_config_dict(str(resolved_step_cfg_path))
        step_cfg_base_dir = resolved_step_cfg_path.parent
    provisional_effective_dict = _apply_overrides(
        step_base_dict,
        step.get("set", []),
        step.get("overrides", {}),
        file_overrides=None,
    )
    provisional_cfg = _build_cfg(
        name,
        step_base_dict,
        step.get("set", []),
        step.get("overrides", {}),
        file_overrides=None,
        base_dir=step_cfg_base_dir,
    )
    worktree_plan = _infer_worktree_plan(
        name, provisional_cfg, provisional_effective_dict, workflow_path=workflow_path
    )
    if worktree_plan is not None:
        original_repo_root = (
            Path(getattr(provisional_cfg, "home_dir")).expanduser().resolve()
        )
        with temporary_detached_worktree(
            original_repo_root, worktree_plan.start_ref
        ) as worktree_root:
            cfg = copy.deepcopy(provisional_cfg)
            worktree_base_dict = step_base_dict
            worktree_cfg_base_dir = step_cfg_base_dir
            if resolved_step_cfg_path is not None:
                worktree_cfg_path = _remap_repo_local_path(
                    resolved_step_cfg_path, original_repo_root, worktree_root
                )
                worktree_base_dict = _load_config_dict(str(worktree_cfg_path))
                worktree_cfg_base_dir = worktree_cfg_path.parent
            worktree_workflow_base_dir = workflow_base_dir
            if workflow_base_dir is not None:
                worktree_workflow_base_dir = _remap_repo_local_path(
                    Path(workflow_base_dir), original_repo_root, worktree_root
                )
            file_overrides: List[Dict[str, Any]] = []
            for override_path in step.get("set_from_toml", []) or []:
                file_overrides.append(
                    _load_step_override_file(
                        override_path,
                        workflow_base_dir=worktree_workflow_base_dir,
                        cfg_base_dir=worktree_cfg_base_dir,
                    )
                )
            step_effective_dict = _apply_overrides(
                worktree_base_dict,
                step.get("set", []),
                step.get("overrides", {}),
                file_overrides=file_overrides,
            )
            cfg = _build_cfg(
                name,
                worktree_base_dict,
                step.get("set", []),
                step.get("overrides", {}),
                file_overrides=file_overrides,
                base_dir=worktree_cfg_base_dir,
            )
            step_effective_dict = _rebase_workflow_overrides_for_worktree(
                step_effective_dict,
                original_repo_root,
                worktree_root,
            )
            _rebase_cfg_for_worktree(cfg, worktree_root)
            record_start = len(ctx.step_records)
            _execute_step_body(
                name=name,
                cfg=cfg,
                step_effective_dict=step_effective_dict,
                ctx=ctx,
                step_run_dir=step.get("run_dir"),
                workflow_path=workflow_path,
            )
            if worktree_plan.commit_branch:
                _commit_generated_changes_to_branch(
                    source_worktree_root=worktree_root,
                    repo_root=original_repo_root,
                    branch=worktree_plan.commit_branch,
                    commit_message=str(
                        worktree_plan.commit_message
                        or f"Workflow {name}: generated outputs"
                    ),
                )
            _map_ctx_paths_from_worktree(
                ctx,
                worktree_root=worktree_root,
                repo_root=original_repo_root,
                record_start=record_start,
            )
            return
    file_overrides: List[Dict[str, Any]] = []
    for override_path in step.get("set_from_toml", []) or []:
        file_overrides.append(
            _load_step_override_file(
                override_path,
                workflow_base_dir=workflow_base_dir,
                cfg_base_dir=step_cfg_base_dir,
            )
        )
    step_effective_dict = _apply_overrides(
        step_base_dict,
        step.get("set", []),
        step.get("overrides", {}),
        file_overrides=file_overrides,
    )
    cfg = _build_cfg(
        name,
        step_base_dict,
        step.get("set", []),
        step.get("overrides", {}),
        file_overrides=file_overrides,
        base_dir=step_cfg_base_dir,
    )
    _execute_step_body(
        name=name,
        cfg=cfg,
        step_effective_dict=step_effective_dict,
        ctx=ctx,
        step_run_dir=step.get("run_dir"),
        workflow_path=workflow_path,
    )


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
    workflow_path: Path | None = None,
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
            workflow_path=workflow_path,
        )


def _run_steps_parallel(
    steps: List[Dict[str, Any]],
    base_dict: Dict[str, Any],
    ctx: WorkflowContext,
    *,
    workflow_base_dir: Path | None = None,
    cfg_base_dir: Path | None = None,
    workflow_path: Path | None = None,
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
            workflow_path=workflow_path,
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
    workflow_path: Path | None = None,
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
            workflow_path=workflow_path,
            label_prefix=label_prefix,
        )
        return
    _run_steps_parallel(
        steps,
        base_dict,
        ctx,
        workflow_base_dir=workflow_base_dir,
        cfg_base_dir=cfg_base_dir,
        workflow_path=workflow_path,
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
            workflow_path=path,
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
                workflow_path=path,
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
                        workflow_path=path,
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
                    workflow_path=path,
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
