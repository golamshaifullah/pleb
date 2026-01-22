"""Data Combination Diagnostics Pipeline.

Refactored from the original Jupyter notebook into a reusable Python package.
"""

from .config import PipelineConfig
from .pipeline import run_pipeline

__all__ = ["PipelineConfig", "run_pipeline"]
