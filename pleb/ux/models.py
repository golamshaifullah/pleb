"""Data models for the UX-facing ``pleb.toml`` format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class UXConfig:
    """Parsed UX config structure.

    Attributes
    ----------
    paths : dict
        Filesystem/path-oriented settings.
    data : dict
        Dataset scope settings (branches, pulsars, jobs).
    run : dict
        Run mode and stage toggles.
    policy : dict
        Policy group settings (fix/pqc/report/etc.).
    workflow : dict
        Workflow mode options (e.g. workflow file path).
    pipeline : dict
        Optional direct pass-through keys in legacy flat format.
    extra : dict
        Any additional top-level keys not mapped to known UX sections.
    """

    paths: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)
    workflow: Dict[str, Any] = field(default_factory=dict)
    pipeline: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
