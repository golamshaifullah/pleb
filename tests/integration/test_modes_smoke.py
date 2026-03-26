"""Integration scaffolding for PLEB run modes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import require_binary, require_existing_path, require_flag


pytestmark = pytest.mark.integration


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pleb.cli", *args],
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        ("--help",),
        ("ingest", "--help"),
        ("workflow", "--help"),
        ("qc-report", "--help"),
    ],
)
def test_cli_mode_help_surfaces(args: tuple[str, ...]) -> None:
    proc = _run(*args)
    assert proc.returncode == 0, proc.stderr


def test_workflow_mode_scaffold_file_is_loadable() -> None:
    wf = Path("configs/workflows/branch_chained_fix_pqc_variants.toml")
    require_existing_path(wf)
    proc = _run("workflow", "--file", str(wf), "--help")
    assert proc.returncode == 0, proc.stderr


def test_ingest_mode_runtime_scaffold() -> None:
    """Starter scaffold for real ingest integration.

    This test is opt-in because it needs local data roots and write targets.
    """
    require_flag(
        "PLEB_INTEGRATION_INGEST",
        "ingest integration disabled by default",
    )
    cfg = Path("configs/runs/ingest/ingest_epta_data.toml")
    require_existing_path(cfg)
    proc = _run("ingest", "--config", str(cfg))
    assert proc.returncode == 0, proc.stderr


def test_pipeline_mode_runtime_scaffold() -> None:
    """Starter scaffold for real pipeline integration.

    This test is opt-in and expects a valid tempo2 runtime/container path.
    """
    require_flag(
        "PLEB_INTEGRATION_PIPELINE",
        "pipeline integration disabled by default",
    )
    require_binary("singularity", "pipeline integration requires singularity/apptainer")
    cfg = Path("configs/runs/pipeline/example.toml")
    require_existing_path(cfg)
    proc = _run("--config", str(cfg))
    assert proc.returncode == 0, proc.stderr


def test_qc_report_mode_runtime_scaffold() -> None:
    """Starter scaffold for report integration from existing run outputs."""
    require_flag(
        "PLEB_INTEGRATION_QC_REPORT",
        "qc-report integration disabled by default",
    )
    run_dir = Path("results")
    require_existing_path(run_dir, "qc-report integration expects existing results/")
    proc = _run("qc-report", "--run-dir", str(run_dir))
    assert proc.returncode == 0, proc.stderr
