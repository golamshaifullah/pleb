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


def test_workflow_whitenoise_step_dispatches_pipeline(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    image = tmp_path / "tempo2.sif"
    image.write_text("", encoding="utf-8")

    base_cfg = {
        "home_dir": str(home),
        "singularity_image": str(image),
        "dataset_name": ".",
        "results_dir": str(tmp_path / "results"),
    }
    captured = {}

    def _fake_run_pipeline(cfg):
        captured["run_whitenoise"] = bool(getattr(cfg, "run_whitenoise", False))
        captured["run_tempo2"] = bool(getattr(cfg, "run_tempo2", True))
        tag = tmp_path / "wn_tag"
        tag.mkdir(parents=True, exist_ok=True)
        return {"tag": tag, "qc": tag / "qc", "fix_dataset": tag / "fix_dataset"}

    monkeypatch.setattr("pleb.workflow.run_pipeline", _fake_run_pipeline)
    step = {"name": "whitenoise", "set": [], "overrides": {}}
    ctx = WorkflowContext()
    _run_step(step, base_cfg, ctx)

    assert captured["run_whitenoise"] is True
    assert captured["run_tempo2"] is False
    assert ctx.last_pipeline_run_dir is not None


def test_workflow_compare_public_step_dispatches(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    image = tmp_path / "tempo2.sif"
    image.write_text("", encoding="utf-8")
    out_dir = tmp_path / "public_out"
    providers = tmp_path / "providers.toml"
    providers.write_text("[providers]\n", encoding="utf-8")

    base_cfg = {
        "home_dir": str(home),
        "singularity_image": str(image),
        "dataset_name": ".",
        "results_dir": str(tmp_path / "results"),
        "compare_public_out_dir": str(out_dir),
        "compare_public_providers_path": str(providers),
    }
    captured = {}

    def _fake_compare_public_releases(out_dir, providers_path=None):
        captured["out_dir"] = str(out_dir)
        captured["providers_path"] = str(providers_path) if providers_path else None
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return {"out_dir": Path(out_dir)}

    monkeypatch.setattr(
        "pleb.workflow.compare_public_releases", _fake_compare_public_releases
    )
    step = {"name": "compare_public", "set": [], "overrides": {}}
    ctx = WorkflowContext()
    _run_step(step, base_cfg, ctx)

    assert captured["out_dir"] == str(out_dir.resolve())
    assert captured["providers_path"] == str(providers.resolve())
