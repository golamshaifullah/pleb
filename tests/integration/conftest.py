"""Shared helpers for integration scaffolding tests."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def require_flag(flag_name: str, reason: str) -> None:
    """Skip unless an explicit opt-in environment flag is set."""
    if not _env_truthy(flag_name):
        pytest.skip(f"{reason} (set {flag_name}=1 to enable)")


def require_binary(binary: str, reason: str | None = None) -> None:
    """Skip unless a required executable is available on PATH."""
    if shutil.which(binary) is None:
        msg = reason or f"requires binary '{binary}' on PATH"
        pytest.skip(msg)


def require_existing_path(path: str | Path, reason: str | None = None) -> None:
    """Skip unless a required path exists."""
    p = Path(path)
    if not p.exists():
        msg = reason or f"required path missing: {p}"
        pytest.skip(msg)
