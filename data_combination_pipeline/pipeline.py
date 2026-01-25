"""Main orchestration logic for the data-combination pipeline.

This module coordinates git branch management, tempo2 runs, report generation,
and optional quality-control steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util

import pandas as pd

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover

    def tqdm(x, **kwargs):
        return x


from .config import PipelineConfig
from .git_tools import checkout, require_clean_repo
from .logging_utils import get_logger
from .plotting import (
    plot_covmat_heatmaps,
    plot_pulsars_per_system,
    plot_residuals,
    plot_systems_per_pulsar,
)
from .reports import (
    write_change_reports,
    write_model_comparison_summary,
    write_new_param_significance,
    write_outlier_tables,
)
from .tempo2 import run_tempo2_for_pulsar
from .utils import discover_pulsars, make_output_tree, which_or_raise

# Add-ons from FixDataset.ipynb / AnalysePulsars.ipynb
from .dataset_fix import FixDatasetConfig, fix_pulsar_dataset, write_fix_report
from .outlier_qc import PTAQCConfig, run_pta_qc_for_parfile_subprocess, summarize_pta_qc
from .pulsar_analysis import analyse_binary_from_par, BinaryAnalysisConfig

logger = get_logger("data_combination_pipeline")


def _cfg_get(cfg, name: str, default=None):
    """Safely read a config value from an object or environment.

    Args:
        cfg: Config object (typically :class:`PipelineConfig`).
        name: Attribute name to read.
        default: Fallback value when missing.

    Returns:
        The resolved config value or ``default``.

    Notes:
        This keeps :class:`PipelineConfig` schema changes optional for
        notebook-driven workflows by allowing environment overrides.
    """
    try:
        return getattr(cfg, name)
    except Exception:
        pass

    env_key = {
        "fix_branch_name": "FIXDATASET_BRANCH_NAME",
        "fix_commit_message": "FIXDATASET_COMMIT_MESSAGE",
        "fix_base_branch": "FIXDATASET_BASE_BRANCH",
    }.get(name)
    if env_key:
        v = os.environ.get(env_key, "")
        if v != "":
            return v
    return default


def _cfg_get_bool(cfg, name: str, default: bool = False) -> bool:
    """Resolve a config value as a boolean.

    Args:
        cfg: Config object (typically :class:`PipelineConfig`).
        name: Attribute name to read.
        default: Fallback value when missing.

    Returns:
        Boolean interpretation of the configuration value.

    Notes:
        This avoids Python's ``bool("0")`` pitfall for environment overrides.
    """
    v = None
    try:
        v = getattr(cfg, name)
    except Exception:
        v = None

    if v is None:
        env_key = {
            "fix_apply": "FIXDATASET_APPLY",
            "run_fix_dataset": "RUN_FIX_DATASET",
        }.get(name)
        if env_key:
            v = os.environ.get(env_key)

    if v is None:
        return bool(default)

    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))

    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    # Fall back to Python truthiness for odd values
    return bool(s)

def _fix_cfg_fields() -> set[str]:
    """Return the supported :class:`FixDatasetConfig` field names."""
    try:
        from dataclasses import fields as dc_fields

        return {f.name for f in dc_fields(FixDatasetConfig)}
    except Exception:  # pragma: no cover
        # fallback for non-dataclass implementations (unlikely)
        return set(getattr(FixDatasetConfig, "__annotations__", {}).keys())


def _build_fixdataset_config(cfg, *, apply: bool) -> FixDatasetConfig:
    """Create a :class:`FixDatasetConfig` from a :class:`PipelineConfig`.

    Args:
        cfg: Pipeline configuration.
        apply: Whether FixDataset should apply and commit changes.

    Returns:
        A :class:`FixDatasetConfig` with only supported fields populated.

    Notes:
        The pipeline config may contain more ``fix_*`` fields than are supported
        by a given FixDataset version; unsupported fields are ignored.
    """
    # Apply-mode safety: avoid leaving backup artifacts that dirty the repo.
    dry_run = bool(_cfg_get(cfg, "fix_dry_run", False))
    backup_default = False if apply else True
    backup = bool(_cfg_get(cfg, "fix_backup", backup_default))

    if apply:
        dry_run = False

    # canonical knobs (supported by the dataset_fix.py shipped with this repo)
    kwargs = dict(
        apply=bool(apply),
        backup=bool(backup),
        dry_run=bool(dry_run),
        update_alltim_includes=bool(_cfg_get(cfg, "fix_update_alltim_includes", True)),
        min_toas_per_backend_tim=int(_cfg_get(cfg, "fix_min_toas_per_backend_tim", 10) or 10),
        required_tim_flags=dict(_cfg_get(cfg, "fix_required_tim_flags", {}) or {}),
        infer_system_flags=bool(_cfg_get(cfg, "fix_infer_system_flags", False)),
        system_flag_table_path=_cfg_get(cfg, "fix_system_flag_table_path", None),
        system_flag_overwrite_existing=bool(_cfg_get(cfg, "fix_system_flag_overwrite_existing", False)),
        backend_overrides=dict(_cfg_get(cfg, "fix_backend_overrides", {}) or {}),
        raise_on_backend_missing=bool(_cfg_get(cfg, "fix_raise_on_backend_missing", False)),
        dedupe_toas_within_tim=bool(_cfg_get(cfg, "fix_dedupe_toas_within_tim", False)),
        check_duplicate_backend_tims=bool(_cfg_get(cfg, "fix_check_duplicate_backend_tims", False)),
        remove_overlaps_exact=bool(_cfg_get(cfg, "fix_remove_overlaps_exact", False)),
        insert_missing_jumps=bool(_cfg_get(cfg, "fix_insert_missing_jumps", True)),
        jump_flag=str(_cfg_get(cfg, "fix_jump_flag", "-sys") or "-sys"),
        ensure_ephem=_cfg_get(cfg, "fix_ensure_ephem", None),
        ensure_clk=_cfg_get(cfg, "fix_ensure_clk", None),
        ensure_ne_sw=_cfg_get(cfg, "fix_ensure_ne_sw", None),
        remove_patterns=list(_cfg_get(cfg, "fix_remove_patterns", ["NRT.NUPPI.", "NRT.NUXPI."]) or []),
        coord_convert=_cfg_get(cfg, "fix_coord_convert", None),
    )

    # Extended knobs (present in pipelineb; only applied if FixDatasetConfig supports them)
    kwargs.update(
        dict(
            prune_missing_includes=bool(_cfg_get(cfg, "fix_prune_missing_includes", True)),
            drop_small_backend_includes=bool(_cfg_get(cfg, "fix_drop_small_backend_includes", True)),
            system_flag_update_table=bool(_cfg_get(cfg, "fix_system_flag_update_table", True)),
            default_backend=_cfg_get(cfg, "fix_default_backend", None),
            group_flag=str(_cfg_get(cfg, "fix_group_flag", "-group") or "-group"),
            pta_flag=str(_cfg_get(cfg, "fix_pta_flag", "-pta") or "-pta"),
            pta_value=_cfg_get(cfg, "fix_pta_value", None),
            standardize_par_values=bool(_cfg_get(cfg, "fix_standardize_par_values", True)),
            prune_small_system_toas=bool(_cfg_get(cfg, "fix_prune_small_system_toas", False)),
            prune_small_system_flag=str(_cfg_get(cfg, "fix_prune_small_system_flag", "-sys") or "-sys"),
            qc_remove_outliers=bool(_cfg_get(cfg, "fix_qc_remove_outliers", False)),
            qc_backend_col=str(_cfg_get(cfg, "fix_qc_backend_col", "sys") or "sys"),
            qc_comment_prefix=str(_cfg_get(cfg, "fix_qc_comment_prefix", "C QC_OUTLIER") or "C QC_OUTLIER"),
            qc_remove_bad=bool(_cfg_get(cfg, "fix_qc_remove_bad", True)),
            qc_remove_transients=bool(_cfg_get(cfg, "fix_qc_remove_transients", False)),
            qc_bad_tau_corr_days=float(_cfg_get(cfg, "fix_qc_bad_tau_corr_days", 0.02) or 0.02),
            qc_bad_fdr_q=float(_cfg_get(cfg, "fix_qc_bad_fdr_q", 0.01) or 0.01),
            qc_bad_mark_only_worst_per_day=bool(_cfg_get(cfg, "fix_qc_bad_mark_only_worst_per_day", True)),
            qc_tr_tau_rec_days=float(_cfg_get(cfg, "fix_qc_tr_tau_rec_days", 7.0) or 7.0),
            qc_tr_window_mult=float(_cfg_get(cfg, "fix_qc_tr_window_mult", 5.0) or 5.0),
            qc_tr_min_points=int(_cfg_get(cfg, "fix_qc_tr_min_points", 6) or 6),
            qc_tr_delta_chi2_thresh=float(_cfg_get(cfg, "fix_qc_tr_delta_chi2_thresh", 25.0) or 25.0),
            qc_tr_suppress_overlap=bool(_cfg_get(cfg, "fix_qc_tr_suppress_overlap", True)),
            qc_merge_tol_days=float(_cfg_get(cfg, "fix_qc_merge_tol_days", 2.0 / 86400.0) or (2.0 / 86400.0)),
        )
    )

    supported = _fix_cfg_fields()
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return FixDatasetConfig(**filtered)


def _pta_qc_available() -> bool:
    try:
        return importlib.util.find_spec("pta_qc") is not None
    except Exception:
        return False


def _apply_fixdataset_and_commit(
    repo,
    cfg,
    pulsars: List[str],
    out_paths: Dict[str, Path],
    *,
    base_branch: str,
    new_branch: str,
    commit_message: str,
) -> str:
    """Create a new branch from base_branch, apply FixDataset, and commit changed .par/.tim (+ sys table if present)."""
    require_clean_repo(repo)
    checkout(repo, base_branch)

    existing = {h.name for h in getattr(repo, "heads", [])}
    if new_branch in existing:
        raise RuntimeError(f"Requested fix branch '{new_branch}' already exists. Choose a different name.")

    repo.git.checkout("-b", new_branch)

    fcfg = _build_fixdataset_config(cfg, apply=True)

    reports = []
    for pulsar in tqdm(pulsars, desc=f"fix-dataset (apply on {new_branch})"):
        rep = fix_pulsar_dataset(cfg.home_dir / cfg.dataset_name / pulsar, fcfg)
        rep["branch"] = new_branch
        reports.append(rep)

    write_fix_report(reports, out_paths["fix_dataset"] / new_branch)

    dataset_prefix = str(cfg.dataset_name).strip("/")

    changed = [p for p in repo.git.diff("--name-only").splitlines() if p.strip()]
    untracked = list(getattr(repo, "untracked_files", []) or [])
    paths = list(dict.fromkeys(changed + untracked))

    def _want(p: str) -> bool:
        pp = p.replace("\\", "/")
        if dataset_prefix and not pp.startswith(dataset_prefix + "/"):
            return False
        # Backups / scratch artifacts should not block cleanliness checks:
        if pp.endswith(".orig"):
            return False
        if pp.endswith(".par") or pp.endswith(".tim"):
            return True
        # system flag table (if created/updated)
        if pp.endswith("system_flag_table.json") or pp.endswith("system_flag_table.toml"):
            return True
        return False

    to_stage = [p for p in paths if _want(p)]

    if to_stage:
        repo.git.add("--", *to_stage)
        repo.index.commit(commit_message)
    else:
        repo.git.commit("--allow-empty", "-m", commit_message + " (no changes)")

    # If backups were produced anyway, they will keep the repo dirty. Delete them to preserve pipeline invariants.
    for p in list(getattr(repo, "untracked_files", []) or []):
        if p.endswith(".orig"):
            try:
                (cfg.home_dir / p).unlink()
            except Exception:
                pass

    require_clean_repo(repo)
    return new_branch


def run_pipeline(config: PipelineConfig) -> Dict[str, Path]:
    """Run the full diagnostics pipeline.

    Concurrency model:
        * Git branch checkouts are single-threaded.
        * Within each branch, pulsars can be processed concurrently using ``cfg.jobs``.
        * Each pulsar uses its own work directory to avoid tempo2 output collisions.

    Args:
        config: Pipeline configuration.

    Returns:
        Mapping of output path labels to their filesystem paths.

    Raises:
        FileNotFoundError: If required paths (home_dir, singularity image) are missing.
        RuntimeError: For missing dependencies or invalid configuration.
    """
    cfg = config.resolved()

    run_fix_dataset = _cfg_get_bool(cfg, "run_fix_dataset", False)
    fix_apply = _cfg_get_bool(cfg, "fix_apply", False)
    run_pta_qc = bool(getattr(cfg, "run_pta_qc", False))

    logger.info(
        "Config flags: run_fix_dataset=%s fix_apply=%s run_pta_qc=%s",
        run_fix_dataset,
        fix_apply,
        run_pta_qc,
    )
    if fix_apply:
        logger.info(
            "FixDataset apply config: branch=%s base=%s commit_message=%s",
            str(_cfg_get(cfg, "fix_branch_name", "") or "").strip() or "<missing>",
            str(_cfg_get(cfg, "fix_base_branch", "") or "").strip() or "<auto>",
            str(_cfg_get(cfg, "fix_commit_message", "") or "").strip() or "<default>",
        )
    logger.info("pta_qc available: %s", _pta_qc_available())

    if not cfg.home_dir.exists():
        raise FileNotFoundError(f"home_dir does not exist: {cfg.home_dir}")
    if not cfg.dataset_name.exists():
        warnings.warn(
            f"Dataset {cfg.dataset_name} does not exist. Assuming the pulsar folders live in {cfg.home_dir}.",
            stacklevel=2,
        )
    if not cfg.singularity_image.exists():
        raise FileNotFoundError(f"singularity_image does not exist: {cfg.singularity_image}")

    which_or_raise("singularity", hint="Install Singularity/Apptainer or load it in your environment.")

    try:
        from git import Repo  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("GitPython is required to run the pipeline (branch checkouts). Install GitPython.") from e

    repo = Repo(str(cfg.home_dir))
    require_clean_repo(repo)
    current_branch = repo.active_branch.name
    logger.info("Current git branch: %s", current_branch)

    # Pulsar selection
    if cfg.pulsars == "ALL":
        pulsars = discover_pulsars(cfg.home_dir / cfg.dataset_name)
    else:
        pulsars = list(cfg.pulsars)  # type: ignore[arg-type]
    if not pulsars:
        raise RuntimeError("No pulsars selected/found.")

    # Branch selection
    compare_branches: List[str] = list(dict.fromkeys(list(cfg.branches)))  # preserve order
    reference_branch = str(cfg.reference_branch) if cfg.reference_branch else ""

    branches_to_run = compare_branches.copy()
    change_reports_enabled = bool(cfg.make_change_reports) and bool(reference_branch)
    if getattr(cfg, "testing_mode", False):
        logger.info("Testing mode enabled: change reports will be skipped.")
        change_reports_enabled = False

    if reference_branch and reference_branch not in branches_to_run and change_reports_enabled:
        branches_to_run.append(reference_branch)

    out_paths = make_output_tree(cfg.results_dir, compare_branches, cfg.outdir_name)
    logger.info("Writing outputs to: %s", out_paths["tag"])

    # Ensure fix-dataset output path exists even if the output tree helper doesn't pre-create it
    if "fix_dataset" not in out_paths:
        out_paths["fix_dataset"] = out_paths["tag"] / "fix_dataset"
    out_paths["fix_dataset"].mkdir(parents=True, exist_ok=True)

    binary_rows: List[Dict[str, object]] = []
    qc_rows: List[Dict[str, object]] = []
    qc_enabled = run_pta_qc
    if run_pta_qc and not _pta_qc_available():
        logger.error(
            "run_pta_qc=true but pta_qc is not importable. QC stage will be skipped. "
            "Install pta_qc (and libstempo) to enable QC."
        )
        qc_enabled = False

    # If requested, apply FixDataset on a new branch and commit the resulting .par/.tim files.
    if fix_apply:
        fix_branch_name = str(_cfg_get(cfg, "fix_branch_name", "") or "").strip()
        if not fix_branch_name:
            raise RuntimeError(
                "fix_apply=true requested but no fix_branch_name was provided. "
                "Set 'fix_branch_name' in the config or export FIXDATASET_BRANCH_NAME."
            )

        base_branch = str(_cfg_get(cfg, "fix_base_branch", "") or "").strip()
        if not base_branch:
            base_branch = str(reference_branch or current_branch)

        commit_message = str(_cfg_get(cfg, "fix_commit_message", "") or "").strip() or "FixDataset: apply automated dataset fixes"

        logger.info(
            "Applying FixDataset on new branch '%s' (base: %s) and committing .par/.tim changes.",
            fix_branch_name,
            base_branch,
        )
        _apply_fixdataset_and_commit(
            repo,
            cfg,
            pulsars,
            out_paths,
            base_branch=base_branch,
            new_branch=fix_branch_name,
            commit_message=commit_message,
        )
        checkout(repo, current_branch)

    try:
        for branch in branches_to_run:
            logger.info("=== Branch: %s ===", branch)
            checkout(repo, branch)

            # Forced fix_dataset reporting per branch (report-only; never modifies the repo in this loop)
            fcfg = _build_fixdataset_config(cfg, apply=False)
            reports = []
            if run_fix_dataset:
                for pulsar in tqdm(pulsars, desc=f"fix-dataset ({branch})"):
                    rep = fix_pulsar_dataset(cfg.home_dir / cfg.dataset_name / pulsar, fcfg)
                    rep["branch"] = branch
                    reports.append(rep)
                write_fix_report(reports, out_paths["fix_dataset"] / branch)
            else:
                logger.info("FixDataset report-only stage skipped (run_fix_dataset=false).")

            # tempo2 runs (parallelizable across pulsars)
            if cfg.run_tempo2:
                n_jobs = max(1, int(getattr(cfg, "jobs", 1) or 1))
                # pipelineb feature: if we just applied fixes and we're only running one branch,
                # force tempo2 rerun unless user explicitly disabled it.
                force_rerun = bool(cfg.force_rerun) or (
                    _cfg_get_bool(cfg, "fix_apply", False) and len(branches_to_run) == 1 and bool(getattr(cfg, "run_fix_dataset", True))
                )

                if n_jobs == 1:
                    for pulsar in tqdm(pulsars, desc=f"tempo2 ({branch})"):
                        run_tempo2_for_pulsar(
                            home_dir=cfg.home_dir,
                            dataset_name=cfg.dataset_name,
                            singularity_image=cfg.singularity_image,
                            out_paths=out_paths,
                            pulsar=pulsar,
                            branch=branch,
                            epoch=str(cfg.epoch),
                            force_rerun=force_rerun,
                        )
                else:
                    futures = []
                    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                        for pulsar in pulsars:
                            futures.append(
                                ex.submit(
                                    run_tempo2_for_pulsar,
                                    home_dir=cfg.home_dir,
                                    dataset_name=cfg.dataset_name,
                                    singularity_image=cfg.singularity_image,
                                    out_paths=out_paths,
                                    pulsar=pulsar,
                                    branch=branch,
                                    epoch=str(cfg.epoch),
                                    force_rerun=force_rerun,
                                )
                            )
                        for fut in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=f"tempo2 ({branch})",
                        ):
                            fut.result()

            if qc_enabled:
                qc_cfg = PTAQCConfig(
                    backend_col=str(getattr(cfg, "pta_qc_backend_col", "group")),
                    drop_unmatched=bool(getattr(cfg, "pta_qc_drop_unmatched", False)),
                    merge_tol_seconds=float(getattr(cfg, "pta_qc_merge_tol_seconds", 2.0)),
                    tau_corr_minutes=float(getattr(cfg, "pta_qc_tau_corr_minutes", 30.0)),
                    fdr_q=float(getattr(cfg, "pta_qc_fdr_q", 0.01)),
                    mark_only_worst_per_day=bool(getattr(cfg, "pta_qc_mark_only_worst_per_day", True)),
                    tau_rec_days=float(getattr(cfg, "pta_qc_tau_rec_days", 7.0)),
                    window_mult=float(getattr(cfg, "pta_qc_window_mult", 5.0)),
                    min_points=int(getattr(cfg, "pta_qc_min_points", 6)),
                    delta_chi2_thresh=float(getattr(cfg, "pta_qc_delta_chi2_thresh", 25.0)),
                )
                qc_out_dir = out_paths["qc"] / branch
                qc_out_dir.mkdir(parents=True, exist_ok=True)
                for pulsar in tqdm(pulsars, desc=f"pta_qc ({branch})"):
                    parfile = cfg.home_dir / cfg.dataset_name / pulsar / f"{pulsar}.par"
                    out_csv = qc_out_dir / f"{pulsar}_qc.csv"
                    try:
                        df = run_pta_qc_for_parfile_subprocess(parfile, out_csv, qc_cfg)
                    except Exception as e:
                        logger.warning("pta_qc failed for %s (%s); skipping QC for this pulsar: %s", pulsar, branch, e)
                        qc_rows.append({"pulsar": pulsar, "branch": branch, "qc_csv": str(out_csv), "qc_error": str(e)})
                        continue
                    row = {"pulsar": pulsar, "branch": branch, "qc_csv": str(out_csv)}
                    row.update(summarize_pta_qc(df))
                    qc_rows.append(row)

            # Branch-level plots and tables (only for compare_branches, not the optional reference-only branch)
            if branch in compare_branches:
                if cfg.make_toa_coverage_plots:
                    plot_systems_per_pulsar(cfg.home_dir, cfg.dataset_name, out_paths, pulsars, branch, dpi=int(cfg.dpi))
                    plot_pulsars_per_system(cfg.home_dir, cfg.dataset_name, out_paths, pulsars, branch, dpi=int(cfg.dpi))

                if cfg.make_outlier_reports:
                    write_outlier_tables(cfg.home_dir, cfg.dataset_name, out_paths, pulsars, [branch])

            # Binary analysis per branch
            if cfg.make_binary_analysis:
                bcfg = BinaryAnalysisConfig(only_models=cfg.binary_only_models)
                for pulsar in pulsars:
                    parfile = cfg.home_dir / cfg.dataset_name / pulsar / f"{pulsar}.par"
                    row = analyse_binary_from_par(parfile)
                    if bcfg.only_models and row.get("BINARY") not in set(bcfg.only_models):
                        continue
                    row["pulsar"] = pulsar
                    row["branch"] = branch
                    binary_rows.append(row)

        # Cross-branch reports
        if change_reports_enabled:
            branches_for_reports = list(compare_branches)
            if reference_branch not in branches_for_reports:
                branches_for_reports.append(reference_branch)
            write_change_reports(out_paths, pulsars, branches_for_reports, reference_branch)
            write_model_comparison_summary(out_paths, pulsars, branches_for_reports, reference_branch)
            write_new_param_significance(out_paths, pulsars, branches_for_reports, reference_branch)

        if cfg.make_covariance_heatmaps:
            plot_covmat_heatmaps(
                out_paths,
                pulsars,
                compare_branches,
                dpi=int(cfg.dpi),
                max_params=cfg.max_covmat_params,
            )

        if cfg.make_residual_plots:
            plot_residuals(out_paths, pulsars, compare_branches, dpi=int(cfg.dpi))

        if cfg.make_binary_analysis and binary_rows:
            df = pd.DataFrame(binary_rows)
            df.to_csv(out_paths["binary_analysis"] / "binary_analysis.tsv", sep="\t", index=False)

        if getattr(cfg, "run_pta_qc", False) and qc_rows:
            dfq = pd.DataFrame(qc_rows)
            dfq.to_csv(out_paths["qc"] / "qc_summary.tsv", sep="\t", index=False)

        logger.info("Pipeline complete.")
        return out_paths

    finally:
        try:
            checkout(repo, current_branch)
        except Exception:
            pass
