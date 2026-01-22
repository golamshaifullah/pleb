"""Data Combination Diagnostics Pipeline.

Refactored from the original Jupyter notebook into a reusable Python package.

Extras included from the additional notebooks:
- dataset fixing utilities (FixDataset.ipynb)
- Kepler/orbit helpers and binary parameter analysis (AnalysePulsars.ipynb)
"""

from .config import PipelineConfig
from .pipeline import run_pipeline

from .dataset_fix import FixDatasetConfig, fix_pulsar_dataset, write_fix_report
from .pulsar_analysis import BinaryAnalysisConfig, write_binary_analysis

__all__ = [
    "PipelineConfig",
    "run_pipeline",
    "FixDatasetConfig",
    "fix_pulsar_dataset",
    "write_fix_report",
    "BinaryAnalysisConfig",
    "write_binary_analysis",
]
