"""Compatibility helpers for older Python versions."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import inspect


_DATACLASS_SUPPORTS_SLOTS = "slots" in inspect.signature(_dataclass).parameters


def dataclass(*args, **kwargs):
    """dataclass() wrapper that ignores slots on older Python versions."""
    if not _DATACLASS_SUPPORTS_SLOTS:
        kwargs.pop("slots", None)
    return _dataclass(*args, **kwargs)
