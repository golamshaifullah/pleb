from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from git import Repo

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from .config import PipelineConfig
from .logging_utils import get_logger
from .utils import which_or_raise, discover_pulsars, make_output_tree
from .git_tools import checkout, require_clean_repo
from .tempo2 import run_tempo2_for_pulsar
from .plotting import (
    plot_systems_per_pulsar,
    plot_pulsars_per_system,
    plot_covmat_heatmaps,
    plot_residuals,
)
from .reports import write_change_reports, write_model_comparison_summary, write_outlier_tables

logger = get_logger("data_combination_pipeline")

def run_pipeline(config: PipelineConfig) -> Dict[str, Path]:
    cfg = config.resolved()

    if not cfg.home_dir.exists():
        raise FileNotFoundError(f"home_dir does not exist: {cfg.home_dir}")
    if not cfg.singularity_image.exists():
        raise FileNotFoundError(f"singularity_image does not exist: {cfg.singularity_image}")

    which_or_raise("singularity", hint="Install Singularity/Apptainer or load it in your environment.")

    repo = Repo(str(cfg.home_dir))
    require_clean_repo(repo)
    current_branch = repo.active_branch.name
    logger.info("Current git branch: %s", current_branch)

    if cfg.pulsars == "ALL":
        pulsars = discover_pulsars(cfg.home_dir)
    else:
        pulsars = list(cfg.pulsars)  # type: ignore[arg-type]

    if not pulsars:
        raise RuntimeError("No pulsars selected/found.")

    compare_branches: List[str] = list(dict.fromkeys(list(cfg.branches)))  # preserve order
    reference_branch = cfg.reference_branch

    branches_to_run = compare_branches.copy()
    if reference_branch and reference_branch not in branches_to_run and cfg.make_change_reports:
        branches_to_run.append(reference_branch)

    out_paths = make_output_tree(cfg.results_dir, compare_branches, cfg.outdir_name)
    logger.info("Writing outputs to: %s", out_paths["tag"])

    try:
        for branch in branches_to_run:
            logger.info("=== Branch: %s ===", branch)
            checkout(repo, branch)

            if cfg.run_tempo2:
                for pulsar in tqdm(pulsars, desc=f"tempo2 ({branch})"):
                    run_tempo2_for_pulsar(
                        home_dir=cfg.home_dir,
                        singularity_image=cfg.singularity_image,
                        out_paths=out_paths,
                        pulsar=pulsar,
                        branch=branch,
                        epoch=str(cfg.epoch),
                        force_rerun=bool(cfg.force_rerun),
                    )

            if branch in compare_branches and cfg.make_toa_coverage_plots:
                plot_systems_per_pulsar(cfg.home_dir, out_paths, pulsars, branch, dpi=int(cfg.dpi))
                plot_pulsars_per_system(cfg.home_dir, out_paths, pulsars, branch, dpi=int(cfg.dpi))

            if branch in compare_branches and cfg.make_outlier_reports:
                write_outlier_tables(cfg.home_dir, out_paths, pulsars, [branch])

        if cfg.make_change_reports and reference_branch:
            branches_for_reports = compare_branches + ([reference_branch] if reference_branch else [])
            write_change_reports(out_paths, pulsars, branches_for_reports, reference_branch)
            # high-level fit-quality comparison (chisq/AIC/BIC/WRMS) vs reference
            write_model_comparison_summary(out_paths, pulsars, branches_for_reports, reference_branch)

        if cfg.make_covariance_heatmaps:
            plot_covmat_heatmaps(out_paths, pulsars, compare_branches, dpi=int(cfg.dpi), max_params=cfg.max_covmat_params)

        if cfg.make_residual_plots:
            plot_residuals(out_paths, pulsars, compare_branches, dpi=int(cfg.dpi))

        logger.info("Pipeline complete.")
        return out_paths

    finally:
        try:
            checkout(repo, current_branch)
        except Exception:
            pass
