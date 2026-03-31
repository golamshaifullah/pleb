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


def test_ux_init_full_contains_pipeline_catalog(tmp_path: Path) -> None:
    proc = _run(
        tmp_path,
        "init",
        "--mode",
        "pipeline",
        "--level",
        "full",
        "--force",
    )
    assert proc.returncode == 0, proc.stderr
    out = tmp_path / "configs" / "runs" / "pipeline" / "pleb.pipeline.toml"
    txt = out.read_text(encoding="utf-8")
    assert "run_whitenoise" in txt
    assert "pqc_glitch_noise_k" in txt


def test_ux_init_workflow_template_3pass(tmp_path: Path) -> None:
    proc = _run(
        tmp_path,
        "init",
        "--workflow-template",
        "3pass-clean",
        "--outdir",
        "configs",
        "--force",
    )
    assert proc.returncode == 0, proc.stderr
    run_cfg = tmp_path / "configs" / "runs" / "pipeline" / "pleb.3pass-clean.pipeline.toml"
    wf_cfg = tmp_path / "configs" / "workflows" / "pleb.3pass-clean.toml"
    ux_wf_cfg = tmp_path / "configs" / "runs" / "workflow" / "pleb.3pass-clean.workflow.toml"
    assert run_cfg.exists(), f"missing expected file: {run_cfg}"
    assert wf_cfg.exists(), f"missing expected file: {wf_cfg}"
    assert ux_wf_cfg.exists(), f"missing expected file: {ux_wf_cfg}"
    wf_text = wf_cfg.read_text(encoding="utf-8")
    assert 'name = "whitenoise"' in wf_text
    assert 'name = "compare_public"' in wf_text


def test_ux_init_workflow_template_golden_path(tmp_path: Path) -> None:
    proc = _run(
        tmp_path,
        "init",
        "--workflow-template",
        "golden-path",
        "--outdir",
        "configs",
        "--force",
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "configs" / "project.toml").exists()
    assert (tmp_path / "configs" / "policy.toml").exists()
    assert (tmp_path / "configs" / "workflows" / "workflow.toml").exists()
