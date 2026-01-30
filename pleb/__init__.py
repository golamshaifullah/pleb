"""Run the data-combination diagnostics pipeline.

This package exposes a small public API for running the full pipeline,
running parameter scans, and applying FixDataset operations programmatically.
The implementation is refactored from the original notebooks and designed to
import quickly; heavy dependencies (GitPython, libstempo/pqc) are imported
lazily by entry points.

Examples:
    Run the pipeline programmatically::

        from pathlib import Path
        from pleb import PipelineConfig, run_pipeline

        cfg = PipelineConfig(
            home_dir=Path("/data/epta"),
            singularity_image=Path("/images/tempo2.sif"),
            dataset_name="EPTA",
        )
        outputs = run_pipeline(cfg)

    Run a parameter scan::

        from pleb import PipelineConfig, run_param_scan

        cfg = PipelineConfig(
            home_dir=Path("/data/epta"),
            singularity_image=Path("/images/tempo2.sif"),
            dataset_name="EPTA",
            param_scan_typical=True,
        )
        results = run_param_scan(cfg)

See Also:
    pleb.pipeline.run_pipeline: Full pipeline implementation.
    pleb.param_scan.run_param_scan: Parameter scan runner.
    pleb.dataset_fix: FixDataset helpers.
"""

from __future__ import annotations

from .config import PipelineConfig
from .dataset_fix import FixDatasetConfig, fix_pulsar_dataset, write_fix_report
from .pulsar_analysis import BinaryAnalysisConfig, write_binary_analysis


def run_pipeline(cfg: PipelineConfig):
    """Run the full data-combination pipeline.

    This is a lightweight wrapper that lazily imports the heavy pipeline module.

    Args:
        cfg: Pipeline configuration.

    Returns:
        A dictionary of output paths as returned by
        :func:`pleb.pipeline.run_pipeline`.

    See Also:
        pleb.pipeline.run_pipeline: Full pipeline implementation.
    """
    from .pipeline import run_pipeline as _run

    return _run(cfg)


def run_param_scan(cfg: PipelineConfig, **kwargs):
    """Run a parameter scan (fit-only) workflow.

    This wrapper lazily imports the parameter scan module.

    Args:
        cfg: Pipeline configuration.
        **kwargs: Forwarded to
            :func:`pleb.param_scan.run_param_scan`.

    Returns:
        A dictionary of output paths produced by the scan.

    See Also:
        pleb.param_scan.run_param_scan: Full scan implementation.
    """
    from .param_scan import run_param_scan as _run

    return _run(cfg, **kwargs)


__all__ = [
    "PipelineConfig",
    "run_pipeline",
    "run_param_scan",
    "FixDatasetConfig",
    "fix_pulsar_dataset",
    "write_fix_report",
    "BinaryAnalysisConfig",
    "write_binary_analysis",
]
