"""Compatibility helpers for older Python versions."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import inspect

_DATACLASS_SUPPORTS_SLOTS = "slots" in inspect.signature(_dataclass).parameters


def dataclass(*args, **kwargs):
    """Return a ``dataclasses.dataclass`` compatible across Python versions.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :func:`dataclasses.dataclass`.
    **kwargs
        Keyword arguments forwarded to :func:`dataclasses.dataclass`. If the
        running interpreter does not support ``slots=``, that keyword is
        removed before delegation.

    Returns
    -------
    callable
        Decorator returned by :func:`dataclasses.dataclass`.

    Notes
    -----
    This helper keeps one code path for ``@dataclass(slots=True)`` usage while
    preserving runtime compatibility on older Python versions.
    """
    if not _DATACLASS_SUPPORTS_SLOTS:
        kwargs.pop("slots", None)
    return _dataclass(*args, **kwargs)
