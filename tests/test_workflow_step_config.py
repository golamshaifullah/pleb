"""Tests for workflow step-specific config support."""

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

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


def test_workflow_step_uses_top_level_config_base_dir(
    monkeypatch, tmp_path: Path
) -> None:
    cfg_dir = tmp_path / "configs" / "runs"
    step_home = cfg_dir / "repo"
    step_home.mkdir(parents=True, exist_ok=True)
    image = tmp_path / "tempo2.sif"
    image.write_text("", encoding="utf-8")

    base_cfg = {
        "home_dir": "repo",
        "singularity_image": str(image),
        "dataset_name": ".",
        "results_dir": str(tmp_path / "results_base"),
    }
    captured = {}

    def _fake_run_pipeline(cfg):
        captured["home_dir"] = str(cfg.home_dir)
        tag = tmp_path / "run_tag_base"
        tag.mkdir(parents=True, exist_ok=True)
        return {"tag": tag, "qc": tag / "qc", "fix_dataset": tag / "fix_dataset"}

    monkeypatch.setattr("pleb.workflow.run_pipeline", _fake_run_pipeline)

    step = {"name": "pipeline", "set": [], "overrides": {}}
    ctx = WorkflowContext()
    _run_step(step, base_cfg, ctx, cfg_base_dir=cfg_dir)

    assert captured["home_dir"] == str(step_home.resolve())


def test_workflow_step_loads_override_toml_from_workflow_dir(
    monkeypatch, tmp_path: Path
) -> None:
    workflow_dir = tmp_path / "output_repo" / "workflows"
    step_cfg = tmp_path / "pleb_repo" / "configs" / "runs" / "pipeline.toml"
    repo_home = tmp_path / "dataset_repo"
    repo_home.mkdir(parents=True, exist_ok=True)
    (repo_home / ".git").mkdir()
    image = tmp_path / "tempo2.sif"
    image.write_text("", encoding="utf-8")

    _write(
        step_cfg,
        f"""
home_dir = "{repo_home}"
singularity_image = "{image}"
dataset_name = "."
results_dir = "{tmp_path / "results"}"
pqc_delta_chi2_thresh = 25.0
pqc_glitch_enabled = true
""",
    )
    _write(
        workflow_dir / "artifacts" / "best_overrides.toml",
        """
pqc_delta_chi2_thresh = 17.0
pqc_glitch_enabled = false
""",
    )

    captured = {}

    def _fake_run_pipeline(cfg):
        captured["delta_chi2"] = float(getattr(cfg, "pqc_delta_chi2_thresh"))
        captured["glitch_enabled"] = bool(getattr(cfg, "pqc_glitch_enabled"))
        tag = tmp_path / "run_tag_override_file"
        tag.mkdir(parents=True, exist_ok=True)
        return {"tag": tag, "qc": tag / "qc", "fix_dataset": tag / "fix_dataset"}

    monkeypatch.setattr("pleb.workflow.run_pipeline", _fake_run_pipeline)

    step = {
        "name": "pipeline",
        "config": str(step_cfg),
        "set_from_toml": ["artifacts/best_overrides.toml"],
        "set": ["pqc_delta_chi2_thresh=19.0"],
        "overrides": {"pqc_delta_chi2_thresh": 23.0},
    }
    ctx = WorkflowContext()
    _run_step(step, {}, ctx, workflow_base_dir=workflow_dir, cfg_base_dir=workflow_dir)

    assert captured["delta_chi2"] == 23.0
    assert captured["glitch_enabled"] is False


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
        captured["run_fix_dataset"] = bool(getattr(cfg, "run_fix_dataset", True))
        captured["fix_apply"] = bool(getattr(cfg, "fix_apply", True))
        tag = tmp_path / "wn_tag"
        tag.mkdir(parents=True, exist_ok=True)
        return {"tag": tag, "qc": tag / "qc", "fix_dataset": tag / "fix_dataset"}

    monkeypatch.setattr("pleb.workflow.run_pipeline", _fake_run_pipeline)
    step = {"name": "whitenoise", "set": [], "overrides": {}}
    ctx = WorkflowContext()
    _run_step(step, base_cfg, ctx)

    assert captured["run_whitenoise"] is True
    assert captured["run_tempo2"] is False
    assert captured["run_fix_dataset"] is False
    assert captured["fix_apply"] is False
    assert ctx.last_pipeline_run_dir is not None


def test_workflow_compare_public_step_dispatches(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / ".git").mkdir()
    (home / "dataset").mkdir()
    image = tmp_path / "tempo2.sif"
    image.write_text("", encoding="utf-8")
    out_dir = tmp_path / "public_out"
    cache_dir = tmp_path / "public_cache"
    providers = tmp_path / "providers.toml"
    providers.write_text("[providers]\n", encoding="utf-8")
    ingest_mapping = tmp_path / "ingest_mapping.json"
    ingest_mapping.write_text("{}", encoding="utf-8")

    base_cfg = {
        "home_dir": str(home),
        "singularity_image": str(image),
        "dataset_name": "dataset",
        "results_dir": str(tmp_path / "results"),
        "compare_public_out_dir": str(out_dir),
        "compare_public_cache_dir": str(cache_dir),
        "compare_public_providers_path": str(providers),
        "ingest_mapping_file": str(ingest_mapping),
        "pulsars": ["J1909-3744"],
        "reference_branch": "j1909_step6_apply_delete",
    }
    captured = {}

    def _fake_compare_public_releases(
        out_dir,
        providers_path=None,
        cache_dir=None,
        local_dataset_root=None,
        local_branch=None,
        local_pulsars=None,
        alias_mapping_path=None,
    ):
        captured["out_dir"] = str(out_dir)
        captured["providers_path"] = str(providers_path) if providers_path else None
        captured["cache_dir"] = str(cache_dir) if cache_dir is not None else None
        captured["local_dataset_root"] = (
            str(local_dataset_root) if local_dataset_root is not None else None
        )
        captured["local_branch"] = str(local_branch) if local_branch is not None else None
        captured["local_pulsars"] = list(local_pulsars) if local_pulsars is not None else None
        captured["alias_mapping_path"] = (
            str(alias_mapping_path) if alias_mapping_path is not None else None
        )
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
    assert captured["cache_dir"] == str(cache_dir.resolve())
    assert captured["local_dataset_root"] == str((home / "dataset").resolve())
    assert captured["local_branch"] == "j1909_step6_apply_delete"
    assert captured["local_pulsars"] == ["J1909-3744"]
    assert captured["alias_mapping_path"] == str(ingest_mapping.resolve())


def test_workflow_compare_public_step_uses_isolated_worktree_and_keeps_main_clean(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=home, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=home, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tester@example.com"], cwd=home, check=True
    )
    (home / "dataset").mkdir()
    (home / "dataset" / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=home, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=home, check=True)
    subprocess.run(
        ["git", "branch", "j1909_step6_apply_delete"], cwd=home, check=True
    )

    image = tmp_path / "tempo2.sif"
    image.write_text("", encoding="utf-8")
    providers = tmp_path / "providers.toml"
    providers.write_text("[providers]\n", encoding="utf-8")
    ingest_mapping = tmp_path / "ingest_mapping.json"
    ingest_mapping.write_text("{}", encoding="utf-8")
    requested_out_dir = home / "results" / "public_compare" / "j1909"
    requested_cache_dir = home / "public_release_cache"

    base_cfg = {
        "home_dir": str(home),
        "singularity_image": str(image),
        "dataset_name": "dataset",
        "results_dir": str(home / "results"),
        "compare_public_out_dir": str(requested_out_dir),
        "compare_public_cache_dir": str(requested_cache_dir),
        "compare_public_providers_path": str(providers),
        "ingest_mapping_file": str(ingest_mapping),
        "pulsars": ["J1909-3744"],
        "reference_branch": "j1909_step6_apply_delete",
    }
    captured = {}

    def _fake_compare_public_releases(
        out_dir,
        providers_path=None,
        cache_dir=None,
        local_dataset_root=None,
        local_branch=None,
        local_pulsars=None,
        alias_mapping_path=None,
    ):
        captured["out_dir"] = str(out_dir)
        captured["cache_dir"] = str(cache_dir) if cache_dir is not None else None
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "done.txt").write_text("ok\n", encoding="utf-8")
        return {"out_dir": Path(out_dir)}

    monkeypatch.setattr(
        "pleb.workflow.compare_public_releases", _fake_compare_public_releases
    )

    step = {"name": "compare_public", "set": [], "overrides": {}}
    ctx = WorkflowContext()
    _run_step(step, base_cfg, ctx)

    assert "pleb_worktree_" in captured["out_dir"]
    assert captured["cache_dir"] == str(requested_cache_dir.resolve())
    assert "pleb_worktree_" not in captured["cache_dir"]
    assert ctx.last_run_dir == requested_out_dir.resolve()

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=home,
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout.strip() == ""

    subprocess.run(["git", "checkout", "j1909_step6_apply_delete"], cwd=home, check=True)
    assert (requested_out_dir / "done.txt").exists()


def test_workflow_optimize_step_uses_isolated_worktree_and_commits_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=home, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=home, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tester@example.com"], cwd=home, check=True
    )
    image = tmp_path / "tempo2.sif"
    image.write_text("", encoding="utf-8")
    (home / "configs").mkdir(parents=True, exist_ok=True)
    _write(
        home / "configs" / "base.toml",
        f"""
home_dir = ".."
singularity_image = "{image}"
dataset_name = "."
results_dir = "results"
branches = ["j1909_step2_detect_variants"]
reference_branch = "j1909_step2_detect_variants"
""",
    )
    _write(
        home / "configs" / "optimize.toml",
        """
[optimize]
base_config_path = "configs/base.toml"
out_dir = "results/optimize"
study_name = "study"
""",
    )
    subprocess.run(["git", "add", "."], cwd=home, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=home, check=True)
    subprocess.run(
        ["git", "branch", "j1909_step2_detect_variants"], cwd=home, check=True
    )

    captured = {}

    def _fake_run_optimization(cfg):
        captured["out_dir"] = str(cfg.out_dir)
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.out_dir) / "done.txt").write_text("ok\n", encoding="utf-8")
        return SimpleNamespace(out_dir=Path(cfg.out_dir))

    monkeypatch.setattr("pleb.optimize.optimizer.run_optimization", _fake_run_optimization)

    step = {"name": "optimize", "config": str(home / "configs" / "optimize.toml")}
    ctx = WorkflowContext()
    _run_step(step, {}, ctx, workflow_base_dir=home, cfg_base_dir=home)

    assert "pleb_worktree_" in captured["out_dir"]
    assert ctx.last_run_dir == (home / "results" / "optimize").resolve()

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=home,
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout.strip() == ""

    subprocess.run(
        ["git", "checkout", "j1909_step2_detect_variants"], cwd=home, check=True
    )
    assert (home / "results" / "optimize" / "done.txt").exists()


def test_workflow_set_from_toml_loads_branch_local_artifact_in_worktree(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=home, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=home, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tester@example.com"], cwd=home, check=True
    )
    image = tmp_path / "tempo2.sif"
    image.write_text("", encoding="utf-8")
    (home / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=home, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=home, check=True)
    subprocess.run(
        ["git", "checkout", "-b", "j1909_step2_detect_variants"], cwd=home, check=True
    )
    _write(
        home / "results" / "optimize" / "best_overrides.toml",
        """
pqc_delta_chi2_thresh = 17.0
""",
    )
    subprocess.run(["git", "add", "."], cwd=home, check=True)
    subprocess.run(["git", "commit", "-m", "best overrides"], cwd=home, check=True)
    subprocess.run(["git", "checkout", "main"], cwd=home, check=True)

    captured = {}

    def _fake_run_pipeline(cfg):
        captured["delta_chi2"] = float(getattr(cfg, "pqc_delta_chi2_thresh"))
        tag = Path(cfg.results_dir) / "tag"
        tag.mkdir(parents=True, exist_ok=True)
        return {"tag": tag, "qc": tag / "qc", "fix_dataset": tag / "fix_dataset"}

    monkeypatch.setattr("pleb.workflow.run_pipeline", _fake_run_pipeline)

    base_cfg = {
        "home_dir": str(home),
        "singularity_image": str(image),
        "dataset_name": ".",
        "results_dir": str(home / "results"),
        "branches": ["j1909_step2_detect_variants"],
        "pqc_delta_chi2_thresh": 25.0,
    }
    step = {
        "name": "pipeline",
        "set_from_toml": ["results/optimize/best_overrides.toml"],
        "set": [],
        "overrides": {},
    }
    ctx = WorkflowContext()
    _run_step(step, base_cfg, ctx, workflow_base_dir=home, cfg_base_dir=home)

    assert captured["delta_chi2"] == 17.0

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=home,
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout.strip() == ""


def test_workflow_review_synthesis_step_dispatches(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / ".git").mkdir()
    (home / "dataset").mkdir()
    image = tmp_path / "tempo2.sif"
    image.write_text("", encoding="utf-8")
    workflow_path = tmp_path / "j1909_full_runthrough.toml"
    workflow_path.write_text("workflow_version = 1\n", encoding="utf-8")
    captured = {}

    class _Result:
        out_dir = tmp_path / "review_package"

    def _fake_run_review_synthesis(
        *,
        psr,
        slug,
        workflow_config,
        repo_root,
        dataset_root,
        results_root,
        final_branch,
        out,
        overrides=None,
        max_keep_points=4000,
        top_n_rows=50,
        stage_branch=None,
        stage_run=None,
    ):
        captured["psr"] = psr
        captured["slug"] = slug
        captured["workflow_config"] = str(workflow_config) if workflow_config else None
        captured["repo_root"] = str(repo_root)
        captured["dataset_root"] = str(dataset_root)
        captured["results_root"] = str(results_root)
        captured["final_branch"] = final_branch
        captured["out"] = str(out)
        captured["overrides"] = str(overrides) if overrides is not None else None
        captured["max_keep_points"] = max_keep_points
        captured["top_n_rows"] = top_n_rows
        captured["stage_branch"] = list(stage_branch or [])
        captured["stage_run"] = list(stage_run or [])
        _Result.out_dir.mkdir(parents=True, exist_ok=True)
        return _Result()

    monkeypatch.setattr("pleb.workflow.run_review_synthesis", _fake_run_review_synthesis)

    base_cfg = {
        "home_dir": str(home),
        "singularity_image": str(image),
        "dataset_name": "dataset",
        "results_dir": str(tmp_path / "results"),
        "pulsars": ["J1909-3744"],
    }
    step = {
        "name": "review_synthesis",
        "set": [],
        "overrides": {
            "review_max_keep_points": 123,
            "review_top_n_rows": 9,
            "review_stage_branch": ["step5_apply_comments=j1909_step5_apply_comments"],
            "review_stage_run": ["step8_compare_public=results/public_compare/j1909"],
        },
    }
    ctx = WorkflowContext()
    _run_step(step, base_cfg, ctx, workflow_path=workflow_path)

    assert captured["psr"] == "J1909-3744"
    assert captured["slug"] == "j1909"
    assert captured["workflow_config"] == str(workflow_path.resolve())
    assert captured["repo_root"] == str(home.resolve())
    assert captured["dataset_root"] == str((home / "dataset").resolve())
    assert captured["results_root"] == str((tmp_path / "results").resolve())
    assert captured["final_branch"] == "j1909_step6_apply_delete"
    assert captured["out"] == str(
        (tmp_path / "results" / "review_packages" / "J1909-3744" / "j1909_full_runthrough").resolve()
    )
    assert captured["max_keep_points"] == 123
    assert captured["top_n_rows"] == 9
    assert captured["stage_branch"] == ["step5_apply_comments=j1909_step5_apply_comments"]
    assert captured["stage_run"] == ["step8_compare_public=results/public_compare/j1909"]
    assert ctx.last_run_dir == _Result.out_dir
    assert ctx.last_pipeline_run_dir == _Result.out_dir


def test_workflow_optimize_step_dispatches(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=home, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=home, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tester@example.com"], cwd=home, check=True
    )
    image = tmp_path / "tempo2.sif"
    image.write_text("", encoding="utf-8")
    workflow_dir = home / "configs" / "workflows"
    optimize_cfg = home / "configs" / "optimize" / "runs" / "optimize.toml"
    _write(
        home / "configs" / "base.toml",
        f"""
home_dir = ".."
singularity_image = "{image}"
dataset_name = "."
results_dir = "results"
branches = ["j1909_step2_detect_variants"]
reference_branch = "j1909_step2_detect_variants"
""",
    )
    _write(
        optimize_cfg,
        """
[optimize]
base_config_path = "configs/base.toml"
out_dir = "results/out"
""",
    )
    subprocess.run(["git", "add", "."], cwd=home, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=home, check=True)
    subprocess.run(
        ["git", "branch", "j1909_step2_detect_variants"], cwd=home, check=True
    )

    captured = {}

    def _fake_run_optimization(cfg):
        captured["out_dir"] = str(cfg.out_dir)
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(out_dir=Path(cfg.out_dir))

    monkeypatch.setattr(
        "pleb.optimize.optimizer.run_optimization", _fake_run_optimization
    )

    step = {
        "name": "optimize",
        "config": "../optimize/runs/optimize.toml",
        "set": [],
        "overrides": {},
    }
    ctx = WorkflowContext()
    _run_step(
        step,
        {},
        ctx,
        workflow_base_dir=workflow_dir,
        cfg_base_dir=workflow_dir,
    )

    assert "pleb_worktree_" in captured["out_dir"]
    assert ctx.last_run_dir == (home / "results" / "out").resolve()


def test_workflow_ingest_step_passes_report_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    image = tmp_path / "tempo2.sif"
    image.write_text("", encoding="utf-8")
    mapping = tmp_path / "mapping.json"
    mapping.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "dataset"

    base_cfg = {
        "home_dir": str(home),
        "singularity_image": str(image),
        "dataset_name": ".",
        "results_dir": str(tmp_path / "results"),
        "ingest_mapping_file": str(mapping),
        "ingest_output_dir": str(out_dir),
        "ingest_verify": True,
        "ingest_commit_branch_name": "ingest/from-workflow",
        "ingest_commit_base_branch": "main",
        "fix_ensure_ephem": "DE440",
        "fix_ensure_clk": "TT(BIPM2023)",
        "fix_ensure_ne_sw": "1",
        "pulsars": ["J1909-3744"],
    }
    captured = {}

    def _fake_ingest_dataset(
        mapping_file,
        output_root,
        *,
        verify=False,
        pulsars=None,
        report_metadata=None,
    ):
        captured["mapping_file"] = str(mapping_file)
        captured["output_root"] = str(output_root)
        captured["verify"] = verify
        captured["pulsars"] = pulsars
        captured["report_metadata"] = dict(report_metadata or {})
        return {"output_root": str(output_root)}

    def _fake_commit_ingest_changes(
        output_root,
        *,
        branch_name=None,
        base_branch=None,
        commit_message=None,
    ):
        captured["commit_output_root"] = str(output_root)
        captured["commit_branch_name"] = branch_name
        captured["commit_base_branch"] = base_branch
        captured["commit_message"] = commit_message
        return branch_name or "ingest/generated"

    monkeypatch.setattr("pleb.workflow.ingest_dataset", _fake_ingest_dataset)
    monkeypatch.setattr(
        "pleb.ingest.commit_ingest_changes", _fake_commit_ingest_changes
    )

    step = {"name": "ingest", "set": [], "overrides": {}}
    ctx = WorkflowContext()
    _run_step(step, base_cfg, ctx)

    assert captured["mapping_file"] == str(mapping.resolve())
    assert captured["output_root"] == str(out_dir.resolve())
    assert captured["verify"] is True
    assert list(captured["pulsars"]) == ["J1909-3744"]
    assert captured["report_metadata"] == {
        "fix_ensure_ephem": "DE440",
        "fix_ensure_clk": "TT(BIPM2023)",
        "fix_ensure_ne_sw": "1",
        "ingest_commit_branch_name": "ingest/from-workflow",
        "ingest_commit_base_branch": "main",
    }


def test_workflow_ingest_step_uses_its_own_config_file(
    monkeypatch, tmp_path: Path
) -> None:
    workflow_dir = tmp_path / "configs" / "workflows"
    ingest_cfg_dir = tmp_path / "configs" / "runs" / "ingest"
    mapping = ingest_cfg_dir / "mapping.json"
    out_dir = tmp_path / "dataset"
    mapping.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_text("{}", encoding="utf-8")

    step_cfg_path = ingest_cfg_dir / "ingest.toml"
    _write(
        step_cfg_path,
        f"""
ingest_mapping_file = "mapping.json"
ingest_output_dir = "{out_dir}"
ingest_verify = true
ingest_commit_branch_name = "ingest/from-step-config"
ingest_commit_base_branch = "main"
""",
    )

    captured = {}

    def _fake_ingest_dataset(
        mapping_file,
        output_root,
        *,
        verify=False,
        pulsars=None,
        report_metadata=None,
    ):
        captured["mapping_file"] = str(mapping_file)
        captured["output_root"] = str(output_root)
        captured["verify"] = verify
        captured["report_metadata"] = dict(report_metadata or {})
        return {"output_root": str(output_root)}

    def _fake_commit_ingest_changes(
        output_root,
        *,
        branch_name=None,
        base_branch=None,
        commit_message=None,
    ):
        captured["commit_output_root"] = str(output_root)
        captured["commit_branch_name"] = branch_name
        captured["commit_base_branch"] = base_branch
        captured["commit_message"] = commit_message
        return branch_name or "ingest/generated"

    monkeypatch.setattr("pleb.workflow.ingest_dataset", _fake_ingest_dataset)
    monkeypatch.setattr(
        "pleb.ingest.commit_ingest_changes", _fake_commit_ingest_changes
    )

    step = {
        "name": "ingest",
        "config": "../runs/ingest/ingest.toml",
        "set": [],
        "overrides": {},
    }
    ctx = WorkflowContext()
    _run_step(
        step,
        {},
        ctx,
        workflow_base_dir=workflow_dir,
        cfg_base_dir=workflow_dir,
    )

    assert captured["mapping_file"] == str(mapping.resolve())
    assert captured["output_root"] == str(out_dir.resolve())
    assert captured["verify"] is True
    assert captured["commit_branch_name"] == "ingest/from-step-config"
    assert captured["commit_base_branch"] == "main"
