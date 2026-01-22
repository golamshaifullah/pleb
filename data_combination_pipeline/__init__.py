"""Data Combination Diagnostics Pipeline.

Refactored from the original Jupyter notebook into a reusable Python package.

This package is intentionally light to import. Heavy dependencies (e.g. GitPython)
are only imported when you call the corresponding entry points.
"""

from __future__ import annotations

from .config import PipelineConfig
from .dataset_fix import FixDatasetConfig, fix_pulsar_dataset, write_fix_report
from .pulsar_analysis import BinaryAnalysisConfig, write_binary_analysis


def run_pipeline(cfg: PipelineConfig):
    """Lazy import wrapper for :func:`data_combination_pipeline.pipeline.run_pipeline`."""
    from .pipeline import run_pipeline as _run

    return _run(cfg)


__all__ = [
    "PipelineConfig",
    "run_pipeline",
    "FixDatasetConfig",
    "fix_pulsar_dataset",
    "write_fix_report",
    "BinaryAnalysisConfig",
    "write_binary_analysis",
]
