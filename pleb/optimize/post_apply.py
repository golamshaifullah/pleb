"""Post-apply fit evaluation for optimization trials.

This module materializes a temporary dataset snapshot from a source branch,
applies one trial's QC-driven TOA deletions/comments with FixDataset, reruns
TEMPO2 on the cleaned dataset, and extracts fit-quality metrics that can be
fed back into optimization scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import json

import numpy as np
import pandas as pd

from ..dataset_fix import FixDatasetConfig, fix_pulsar_dataset
from ..parsers import read_general2
from ..reports import summarize_run
from ..tempo2 import run_tempo2_for_pulsar
from ..tim_utils import is_toa_line, parse_tim_flags_from_line
from ..utils import make_output_tree
from .models import OptimizationConfig, TrialResult


def run_post_apply_evaluation(
    cfg: OptimizationConfig,
    trial: TrialResult,
    *,
    pipeline_cfg,
    backend_col: str,
) -> Dict[str, float]:
    """Evaluate one trial after applying its QC decisions to a temp dataset."""
    if trial.run_dir is None:
        return {}
    resolved = pipeline_cfg.resolved()
    repo_root = Path(resolved.home_dir).resolve()
    dataset_root = _resolve_source_dataset_root(cfg, resolved)
    pulsars = _selected_pulsars(resolved.pulsars)
    source_branch = _resolve_source_branch(cfg, resolved)
    qc_branch = _resolve_qc_branch(cfg, resolved)

    eval_root = Path(trial.run_dir) / "post_apply_eval"
    dataset_copy_root = eval_root / "materialized_dataset" / dataset_root.name
    out_paths = make_output_tree(eval_root, ["post_apply_eval"], "fit", lazy=False)
    branch_label = "post_apply_eval"

    _materialize_dataset_subset(
        repo_root=repo_root,
        source_branch=source_branch,
        dataset_root=dataset_root,
        dataset_copy_root=dataset_copy_root,
        pulsars=pulsars,
    )

    rel_dataset_name = dataset_copy_root.relative_to(repo_root).as_posix()
    fix_cfg = FixDatasetConfig(
        apply=True,
        backup=False,
        dry_run=False,
        update_alltim_includes=True,
        generate_alltim_variants=False,
        jump_reference_variants=False,
        qc_remove_outliers=True,
        qc_action=str(cfg.post_apply_qc_action or "delete"),
        qc_backend_col=str(backend_col or "sys"),
        qc_remove_bad=True,
        qc_remove_transients=False,
        qc_remove_solar=False,
        qc_remove_orbital_phase=False,
        qc_results_dir=Path(trial.run_dir) / "qc",
        qc_branch=qc_branch,
        qc_require_csv=True,
    )
    _apply_fixdataset(dataset_copy_root, pulsars, fix_cfg)

    metric_rows: list[dict[str, float]] = []
    for pulsar in pulsars:
        run_tempo2_for_pulsar(
            home_dir=repo_root,
            dataset_name=Path(rel_dataset_name),
            singularity_image=Path(resolved.singularity_image),
            native=bool(getattr(resolved, "tempo2_native", False)),
            out_paths=out_paths,
            pulsar=pulsar,
            branch=branch_label,
            epoch=str(resolved.epoch),
            force_rerun=True,
        )
        base = summarize_run(out_paths, pulsar, branch_label)
        extra = _compute_backend_alignment_metrics(
            timfile=dataset_copy_root / pulsar / f"{pulsar}_all.tim",
            general2_path=out_paths["general2"] / f"{pulsar}_{branch_label}.general2",
            backend_flag=backend_col,
        )
        row: Dict[str, float] = {}
        if base.get("chisq") is not None:
            row["post_apply_chisq"] = float(base["chisq"])
        if base.get("redchisq") is not None:
            red = float(base["redchisq"])
            row["post_apply_redchisq"] = red
            row["post_apply_fit_quality"] = 1.0 / (1.0 + max(red, 0.0))
        if base.get("n_toas") is not None:
            row["post_apply_n_toas"] = float(base["n_toas"])
        if base.get("k_fit") is not None:
            row["post_apply_k_fit"] = float(base["k_fit"])
        if base.get("wrms_post") is not None:
            wrms_s = float(base["wrms_post"])
            wrms_us = wrms_s * 1.0e6
            row["post_apply_wrms_post"] = wrms_s
            row["post_apply_wrms_us"] = wrms_us
            row["post_apply_wrms_quality"] = 1.0 / (1.0 + max(wrms_us, 0.0))
        row.update(extra)
        metric_rows.append(row)

    metrics = _aggregate_metric_rows(metric_rows)
    metrics_path = eval_root / "post_apply_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _selected_pulsars(values: Iterable[str] | str | None) -> List[str]:
    if values in (None, "", "ALL"):
        raise ValueError(
            "post_apply_eval requires an explicit pulsar list; optimize cannot use ALL here."
        )
    if isinstance(values, str):
        return [values]
    return [str(v) for v in values if str(v).strip()]


def _resolve_source_branch(cfg: OptimizationConfig, resolved_cfg) -> str:
    if cfg.post_apply_source_branch:
        return str(cfg.post_apply_source_branch)
    fixed = cfg.fixed_overrides or {}
    ref = str(fixed.get("reference_branch", "") or "").strip()
    if ref:
        return ref
    branches = fixed.get("branches", None)
    if isinstance(branches, (list, tuple)) and branches:
        return str(branches[0])
    if getattr(resolved_cfg, "reference_branch", None):
        return str(resolved_cfg.reference_branch)
    if getattr(resolved_cfg, "branches", None):
        return str(list(resolved_cfg.branches)[0])
    raise ValueError("Unable to resolve post_apply source branch for optimize trial.")


def _resolve_qc_branch(cfg: OptimizationConfig, resolved_cfg) -> str:
    if cfg.post_apply_qc_branch:
        return str(cfg.post_apply_qc_branch)
    return _resolve_source_branch(cfg, resolved_cfg)


def _resolve_source_dataset_root(cfg: OptimizationConfig, resolved_cfg) -> Path:
    fixed = cfg.fixed_overrides or {}
    raw = fixed.get("dataset_name", None)
    if raw not in (None, ""):
        path = Path(str(raw)).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (Path(resolved_cfg.home_dir).resolve() / path).resolve()
    return Path(resolved_cfg.dataset_name).resolve()


def _materialize_dataset_subset(
    *,
    repo_root: Path,
    source_branch: str,
    dataset_root: Path,
    dataset_copy_root: Path,
    pulsars: List[str],
) -> None:
    dataset_rel = _path_in_repo_required(repo_root, dataset_root)
    dataset_copy_root.mkdir(parents=True, exist_ok=True)
    for pulsar in pulsars:
        prefix = f"{dataset_rel}/{pulsar}"
        files = _git_ls_files_at_ref(repo_root, source_branch, prefix)
        if not files:
            raise FileNotFoundError(
                f"No dataset files found for {pulsar} on branch {source_branch} under {dataset_rel}."
            )
        for rel_path in files:
            data = _git_show_file(repo_root, source_branch, rel_path)
            if data is None:
                raise FileNotFoundError(
                    f"Unable to read {rel_path} from branch {source_branch}."
                )
            rel_to_dataset = Path(rel_path).relative_to(dataset_rel)
            dest = dataset_copy_root / rel_to_dataset
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)


def _apply_fixdataset(
    dataset_copy_root: Path,
    pulsars: List[str],
    cfg: FixDatasetConfig,
) -> None:
    errors: list[str] = []
    for pulsar in pulsars:
        try:
            rep = fix_pulsar_dataset(dataset_copy_root / pulsar, cfg)
        except Exception as exc:
            errors.append(f"{pulsar}: {exc}")
            continue
        err = str(rep.get("error", "") or "").strip()
        if err:
            errors.append(f"{pulsar}: {err}")
    if errors:
        raise RuntimeError("post_apply_eval FixDataset failed. " + "; ".join(errors))


def _expand_tim_backends(
    timfile: Path,
    *,
    backend_flag: str,
) -> List[str]:
    out: List[str] = []

    def _walk(path: Path) -> None:
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = raw.strip()
            if not s:
                continue
            if s.startswith("INCLUDE"):
                parts = s.split(maxsplit=1)
                if len(parts) == 2:
                    _walk((path.parent / parts[1].strip()).resolve())
                continue
            if not is_toa_line(raw):
                continue
            flags = parse_tim_flags_from_line(raw)
            backend = (
                flags.get(backend_flag)
                or flags.get("-sys")
                or flags.get("-group")
                or flags.get("-be")
                or path.stem
            )
            out.append(str(backend))

    _walk(timfile.resolve())
    return out


def _compute_backend_alignment_metrics(
    *,
    timfile: Path,
    general2_path: Path,
    backend_flag: str,
) -> Dict[str, float]:
    if not timfile.exists() or not general2_path.exists():
        return {}
    gen = read_general2(general2_path)
    if "post" not in gen.columns:
        return {}
    backends = _expand_tim_backends(timfile, backend_flag=backend_flag)
    post = pd.to_numeric(gen["post"], errors="coerce") * 1.0e6
    if post.notna().sum() == 0 or not backends:
        return {}
    n = min(len(backends), int(len(post)))
    if n <= 0:
        return {}
    frame = pd.DataFrame(
        {
            "backend": backends[:n],
            "post_us": post.iloc[:n].to_numpy(dtype=float),
        }
    )
    frame = frame.loc[frame["backend"].astype(str).str.strip() != ""].copy()
    frame = frame.loc[np.isfinite(frame["post_us"])].copy()
    if frame.empty:
        return {}
    medians = frame.groupby("backend", dropna=False)["post_us"].median().astype(float)
    if medians.empty:
        return {}
    center = float(medians.median())
    centered = medians - center
    max_abs = float(centered.abs().max())
    span = float(centered.max() - centered.min()) if len(centered) > 1 else 0.0
    std = float(centered.std(ddof=0)) if len(centered) > 1 else 0.0
    return {
        "post_apply_backend_count": float(len(medians)),
        "post_apply_max_backend_abs_offset_us": max_abs,
        "post_apply_backend_offset_span_us": span,
        "post_apply_backend_offset_std_us": std,
        "post_apply_backend_alignment": 1.0 / (1.0 + max(max_abs, 0.0)),
    }


def _aggregate_metric_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    out: Dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        if key in {
            "post_apply_redchisq",
            "post_apply_chisq",
            "post_apply_wrms_post",
            "post_apply_wrms_us",
            "post_apply_max_backend_abs_offset_us",
            "post_apply_backend_offset_span_us",
            "post_apply_backend_offset_std_us",
        }:
            out[key] = max(values)
        elif key in {"post_apply_fit_quality", "post_apply_wrms_quality", "post_apply_backend_alignment"}:
            out[key] = min(values)
        else:
            out[key] = sum(values) / float(len(values))
    return out


def _path_in_repo_required(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception as exc:
        raise RuntimeError(f"Path {path} is not under repo root {repo_root}.") from exc


def _git_ls_files_at_ref(repo_root: Path, ref: str, prefix: str) -> List[str]:
    import subprocess

    res = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "--", prefix],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return []
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def _git_show_file(repo_root: Path, ref: str, path_in_repo: str) -> bytes | None:
    import subprocess

    res = subprocess.run(
        ["git", "show", f"{ref}:{path_in_repo}"],
        cwd=str(repo_root),
        capture_output=True,
        text=False,
        check=False,
    )
    if res.returncode != 0:
        return None
    return res.stdout
