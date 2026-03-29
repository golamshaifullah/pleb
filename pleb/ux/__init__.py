"""User-friendly UX wrapper for PLEB configuration and execution.

This package provides a single-file UX layer (``pleb.toml``) and adapter
helpers that translate UX-oriented sections into the existing PLEB
configuration model.
"""

from .commands import run_ux_cli

__all__ = ["run_ux_cli"]
