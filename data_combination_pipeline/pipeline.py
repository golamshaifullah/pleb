from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings

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
from .reports import (
    write_change_reports,
    write_model_comparison_summary,
    write_new_param_significance,
    write_outlier_tables,
)

# Add-ons from FixDataset.ipynb / AnalysePulsars.ipynb
from .dataset_fix import FixDatasetConfig, fix_pulsar_dataset, write_fix_report
from .pulsar_analysis import analyse_binary_from_par, BinaryAnalysisConfig
from .outlier_qc import PTAQCConfig, run_pta_qc_for_parfile, summarize_pta_qc

logger = get_logger("data_combination_pipeline")


def run_pipeline(config: PipelineConfig) -> Dict[str, Path]:
    """Run the full diagnostics pipeline.

    Concurrency model:
      * Git branch checkouts are always single-threaded.
      * Within each branch, pulsars can be processed concurrently using cfg.jobs.
      * Each pulsar uses its own work directory to avoid tempo2 output collisions.
    """
    cfg = config.resolved()

    if not cfg.home_dir.exists():
        raise FileNotFoundError(f"home_dir does not exist: {cfg.home_dir}")
    if not cfg.dataset_name.exists():
        warnings.warn(
            f"Dataset {cfg.dataset_name} does not exist. "
            f"Assuming the pulsar folders live in {cfg.home_dir}.",
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
    if reference_branch and reference_branch not in branches_to_run and cfg.make_change_reports:
        branches_to_run.append(reference_branch)

    out_paths = make_output_tree(cfg.results_dir, compare_branches, cfg.outdir_name)
    logger.info("Writing outputs to: %s", out_paths["tag"])

    # Ensure fix-dataset output path exists even if the output tree helper doesn't pre-create it
    if "fix_dataset" not in out_paths:
        out_paths["fix_dataset"] = out_paths["tag"] / "fix_dataset"
    out_paths["fix_dataset"].mkdir(parents=True, exist_ok=True)


    # Collect binary analysis rows as we iterate branches
    binary_rows: List[Dict[str, object]] = []
    qc_rows: List[Dict[str, object]] = []

    try:
        for branch in branches_to_run:
            logger.info("=== Branch: %s ===", branch)
            checkout(repo, branch)

            # Optional dataset-fix reporting (report-only inside the multi-branch pipeline)
            
            # Dataset-fix reporting (forced; report-only inside the multi-branch pipeline)
            # NOTE: We intentionally do NOT apply fixes here because it would dirty the git working tree and break branch switching.
            if getattr(cfg, "fix_apply", False):
                logger.warning(
                    "fix_apply=true was requested, but run_pipeline forces the FixDataset step to be report-only "
                    "so branch switching remains safe. Ignoring fix_apply and running with apply=False."
                )

            fcfg = FixDatasetConfig(
                apply=False,
                backup=bool(getattr(cfg, "fix_backup", False)),
                dry_run=bool(getattr(cfg, "fix_dry_run", False)),
                update_alltim_includes=bool(getattr(cfg, "fix_update_alltim_includes", False)),
                min_toas_per_backend_tim=int(getattr(cfg, "fix_min_toas_per_backend_tim", 0) or 0),
                required_tim_flags=dict(getattr(cfg, "fix_required_tim_flags", {}) or {}),
                insert_missing_jumps=bool(getattr(cfg, "fix_insert_missing_jumps", False)),
                jump_flag=str(getattr(cfg, "fix_jump_flag", "")),
                ensure_ephem=getattr(cfg, "fix_ensure_ephem", None),
                ensure_clk=getattr(cfg, "fix_ensure_clk", None),
                ensure_ne_sw=getattr(cfg, "fix_ensure_ne_sw", None),
                remove_patterns=list(getattr(cfg, "fix_remove_patterns", []) or []),
                coord_convert=getattr(cfg, "fix_coord_convert", None),
            )

            reports = []
            for pulsar in tqdm(pulsars, desc=f"fix-dataset ({branch})"):
                rep = fix_pulsar_dataset(cfg.home_dir / cfg.dataset_name / pulsar, fcfg)
                rep["branch"] = branch
                reports.append(rep)

            write_fix_report(reports, out_paths["fix_dataset"] / branch)



            # tempo2 runs (parallelizable across pulsars)
            if cfg.run_tempo2:
                n_jobs = max(1, int(getattr(cfg, "jobs", 1) or 1))
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
                            force_rerun=bool(cfg.force_rerun),
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
                                    force_rerun=bool(cfg.force_rerun),
                                )
                            )
                        for fut in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=f"tempo2 ({branch})",
                        ):
                            fut.result()  # propagate exceptions

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
        if cfg.make_change_reports and reference_branch:
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
