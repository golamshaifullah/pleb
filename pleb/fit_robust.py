"""Legacy robust `.tim` parsing import path.

This module is kept for backward compatibility and re-exports the canonical
reader from :mod:`pleb.tim_reader`.
"""

from __future__ import annotations

from .tim_reader import read_tim_file_robust

__all__ = ["read_tim_file_robust"]
