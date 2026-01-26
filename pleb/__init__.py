"""Data Combination Diagnostics Pipeline.

This package provides the refactored pipeline formerly implemented in notebooks.
It exposes a small public API for running the full pipeline, running parameter
scans, and applying FixDataset operations programmatically.

The package is intentionally light to import. Heavy dependencies (for example
GitPython or libstempo/pqc) are imported lazily by the entry points.

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
