"""Integration smoke tests for the UX wrapper commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pleb.cli", *args],
        text=True,
        capture_output=True,
        check=False,
        cwd=str(cwd),
    )


def test_ux_init_default_creates_pipeline_config_tree(tmp_path: Path) -> None:
    proc = _run(tmp_path, "init")
    assert proc.returncode == 0, proc.stderr
    out = tmp_path / "configs" / "runs" / "pipeline" / "pleb.pipeline.toml"
    assert out.exists(), f"missing expected file: {out}"


def test_ux_init_all_modes_creates_mode_specific_files(tmp_path: Path) -> None:
    proc = _run(tmp_path, "init", "--all-modes", "--outdir", "configs", "--force")
    assert proc.returncode == 0, proc.stderr

    expected = [
        tmp_path / "configs" / "runs" / "pipeline" / "pleb.pipeline.toml",
        tmp_path / "configs" / "runs" / "ingest" / "pleb.ingest.toml",
        tmp_path / "configs" / "runs" / "workflow" / "pleb.workflow.toml",
        tmp_path / "configs" / "runs" / "qc_report" / "pleb.qc-report.toml",
    ]
    for p in expected:
        assert p.exists(), f"missing expected file: {p}"
