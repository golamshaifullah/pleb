"""Tests for workflow version contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from pleb.workflow import _load_workflow


def test_workflow_version_default_ok(tmp_path: Path) -> None:
    p = tmp_path / "wf.toml"
    p.write_text('config = "x.toml"\nmode = "serial"\n', encoding="utf-8")
    data = _load_workflow(p)
    assert data.get("config") == "x.toml"


def test_workflow_version_rejects_unsupported(tmp_path: Path) -> None:
    p = tmp_path / "wf.toml"
    p.write_text(
        'workflow_version = 2\nconfig = "x.toml"\nmode = "serial"\n', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        _load_workflow(p)
