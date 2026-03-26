"""Tests for workflow step-specific config support."""

from __future__ import annotations

from pathlib import Path

from pleb.workflow import WorkflowContext, _run_step


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_workflow_step_uses_its_own_config(monkeypatch, tmp_path: Path) -> None:
    base_home = tmp_path / "base_home"
    step_home = tmp_path / "step_home"
    image = tmp_path / "tempo2.sif"
    image.write_text("", encoding="utf-8")

    base_cfg = {
        "home_dir": str(base_home),
        "singularity_image": str(image),
        "dataset_name": ".",
        "results_dir": str(tmp_path / "results_base"),
    }
    step_cfg_path = tmp_path / "step.toml"
    _write(
        step_cfg_path,
        f"""
home_dir = "{step_home}"
singularity_image = "{image}"
dataset_name = "."
results_dir = "{tmp_path / "results_step"}"
""",
    )

    captured = {}

    def _fake_run_pipeline(cfg):
        captured["home_dir"] = str(cfg.home_dir)
        tag = tmp_path / "run_tag"
        tag.mkdir(parents=True, exist_ok=True)
        return {"tag": tag, "qc": tag / "qc", "fix_dataset": tag / "fix_dataset"}

    monkeypatch.setattr("pleb.workflow.run_pipeline", _fake_run_pipeline)

    step = {
        "name": "pipeline",
        "config": str(step_cfg_path),
        "set": [],
        "overrides": {},
    }
    ctx = WorkflowContext()
    _run_step(step, base_cfg, ctx)

    assert captured["home_dir"] == str(step_home.resolve())
