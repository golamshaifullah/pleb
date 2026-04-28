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

from .config import (
    IngestConfig,
    ParamScanConfig,
    PipelineConfig,
    QCReportConfig,
    WorkflowRunConfig,
)
from .dataset_fix import FixDatasetConfig, fix_pulsar_dataset, write_fix_report
from .pulsar_analysis import BinaryAnalysisConfig, write_binary_analysis
from .optimize import OptimizationConfig, run_optimization


def run_pipeline(cfg: PipelineConfig):
    """Run the full data-combination pipeline.

    This is a lightweight wrapper that lazily imports the heavy pipeline module.

    Parameters
    ----------
    cfg : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    dict
        Output-path mapping as returned by
        :func:`pleb.pipeline.run_pipeline`.

    See Also
    --------
        pleb.pipeline.run_pipeline: Full pipeline implementation.
    """
    from .pipeline import run_pipeline as _run

    return _run(cfg)


_LAZY_EXPORTS = {
    "FixDatasetConfig": (".dataset_fix", "FixDatasetConfig"),
    "fix_pulsar_dataset": (".dataset_fix", "fix_pulsar_dataset"),
    "write_fix_report": (".dataset_fix", "write_fix_report"),
    "BinaryAnalysisConfig": (".pulsar_analysis", "BinaryAnalysisConfig"),
    "write_binary_analysis": (".pulsar_analysis", "write_binary_analysis"),
    "OptimizationConfig": (".optimize", "OptimizationConfig"),
    "run_optimization": (".optimize", "run_optimization"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def run_param_scan(cfg: PipelineConfig, **kwargs):
    """Run a parameter scan (fit-only) workflow.

    This wrapper lazily imports the parameter scan module.

    Parameters
    ----------
    cfg : PipelineConfig
        Pipeline configuration.
    ``**kwargs``
        Forwarded to :func:`pleb.param_scan.run_param_scan`.

    Returns
    -------
    dict
        Output-path mapping produced by the scan.

    See Also
    --------
        pleb.param_scan.run_param_scan: Full scan implementation.
    """
    from .param_scan import run_param_scan as _run

    return _run(cfg, **kwargs)


__all__ = [
    "PipelineConfig",
    "IngestConfig",
    "ParamScanConfig",
    "QCReportConfig",
    "WorkflowRunConfig",
    "run_pipeline",
    "run_param_scan",
    "FixDatasetConfig",
    "fix_pulsar_dataset",
    "write_fix_report",
    "BinaryAnalysisConfig",
    "write_binary_analysis",
    "OptimizationConfig",
    "run_optimization",
]
