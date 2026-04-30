"""Tests for pipeline configuration serialization and parsing."""

from __future__ import annotations

from pathlib import Path

from pleb.config import IngestConfig, PipelineConfig


def _make_git_root(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".git").mkdir(exist_ok=True)
    return path


def test_config_json_roundtrip(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        home_dir=tmp_path / "repo",
        singularity_image=tmp_path / "tempo2.sif",
        results_dir=tmp_path / "results",
        branches=["master", "feature"],
        reference_branch="master",
        pulsars=["J0000+0000"],
        epoch="55000",
        force_rerun=True,
        run_tempo2=False,
        make_toa_coverage_plots=False,
        make_change_reports=True,
    )

    out = tmp_path / "cfg.json"
    cfg.save_json(out)

    loaded = PipelineConfig.load(out)
    assert loaded.home_dir == cfg.home_dir
    assert loaded.singularity_image == cfg.singularity_image
    assert loaded.branches == ["master", "feature"]
    assert loaded.reference_branch == "master"
    assert loaded.pulsars == ["J0000+0000"]
    assert loaded.force_rerun is True
    assert loaded.run_tempo2 is False


def test_config_toml_pipeline_table(tmp_path: Path) -> None:
    # The loader accepts either top-level keys or [pipeline] table.
    toml_text = """
[pipeline]
home_dir = "/tmp/repo"
singularity_image = "/tmp/tempo2.sif"
results_dir = "/tmp/results"
branches = ["master", "b1"]
reference_branch = "master"
pulsars = ["J1234+5678"]
"""
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(toml_text, encoding="utf-8")

    cfg = PipelineConfig.load(cfg_path)
    assert str(cfg.home_dir) == "/tmp/repo"
    assert str(cfg.singularity_image) == "/tmp/tempo2.sif"
    assert str(cfg.results_dir) == "/tmp/results"
    assert cfg.branches == ["master", "b1"]
    assert cfg.pulsars == ["J1234+5678"]


def test_config_parses_new_cross_pulsar_and_flag_rule_keys(tmp_path: Path) -> None:
    toml_text = """
home_dir = "/tmp/repo"
singularity_image = "/tmp/tempo2.sif"
dataset_name = "."
results_dir = "/tmp/results"
qc_cross_pulsar_enabled = true
qc_cross_pulsar_window_days = 0.5
qc_cross_pulsar_min_pulsars = 3
fix_flag_sys_freq_rules_enabled = true
fix_flag_sys_freq_rules_path = "/tmp/flag_sys_freq_rules.yaml"
"""
    cfg_path = tmp_path / "cfg_new.toml"
    cfg_path.write_text(toml_text, encoding="utf-8")

    cfg = PipelineConfig.load(cfg_path)
    assert cfg.qc_cross_pulsar_enabled is True
    assert cfg.qc_cross_pulsar_window_days == 0.5
    assert cfg.qc_cross_pulsar_min_pulsars == 3
    assert cfg.fix_flag_sys_freq_rules_enabled is True
    assert str(cfg.fix_flag_sys_freq_rules_path) == "/tmp/flag_sys_freq_rules.yaml"


def test_config_parses_whitenoise_keys(tmp_path: Path) -> None:
    toml_text = """
home_dir = "/tmp/repo"
singularity_image = "/tmp/tempo2.sif"
dataset_name = "."
results_dir = "/tmp/results"
run_whitenoise = true
whitenoise_source_path = "/tmp/whitenoise/src"
whitenoise_epoch_tolerance_seconds = 2.5
whitenoise_single_toa_mode = "equad0"
whitenoise_fit_timing_model_first = false
whitenoise_timfile_name = "{pulsar}_all.new.tim"
"""
    cfg_path = tmp_path / "cfg_wn.toml"
    cfg_path.write_text(toml_text, encoding="utf-8")

    cfg = PipelineConfig.load(cfg_path)
    assert cfg.run_whitenoise is True
    assert str(cfg.whitenoise_source_path) == "/tmp/whitenoise/src"
    assert cfg.whitenoise_epoch_tolerance_seconds == 2.5
    assert cfg.whitenoise_single_toa_mode == "equad0"
    assert cfg.whitenoise_fit_timing_model_first is False
    assert cfg.whitenoise_timfile_name == "{pulsar}_all.new.tim"


def test_ingest_config_roundtrip_preserves_timing_defaults(tmp_path: Path) -> None:
    cfg = IngestConfig(
        ingest_mapping_file=tmp_path / "mapping.json",
        ingest_output_dir=tmp_path / "dataset",
        home_dir=tmp_path / "home",
        dataset_name="DR3single",
        ingest_verify=True,
        ingest_commit_branch_name="ingest/demo",
        ingest_commit_base_branch="main",
        ingest_commit_message="demo ingest",
        fix_ensure_ephem="DE440",
        fix_ensure_clk="TT(BIPM2023)",
        fix_ensure_ne_sw="USE_DEFAULT",
    )

    loaded = IngestConfig.from_dict(cfg.to_dict())
    assert loaded.ingest_mapping_file == cfg.ingest_mapping_file
    assert loaded.ingest_output_dir == cfg.ingest_output_dir
    assert loaded.ingest_commit_branch_name == "ingest/demo"
    assert loaded.ingest_commit_base_branch == "main"
    assert loaded.fix_ensure_ephem == "DE440"
    assert loaded.fix_ensure_clk == "TT(BIPM2023)"
    assert loaded.fix_ensure_ne_sw == "USE_DEFAULT"


def test_pipeline_resolved_uses_repo_relative_dataset_path(tmp_path: Path) -> None:
    repo_root = _make_git_root(tmp_path / "repo")
    cfg = PipelineConfig(
        home_dir=repo_root,
        singularity_image=tmp_path / "tempo2.sif",
        dataset_name="EPTA-DR3/epta-dr3-data-v0",
        results_dir=Path("results"),
    )

    resolved = cfg.resolved()
    assert resolved.home_dir == repo_root.resolve()
    assert resolved.dataset_name == (repo_root / "EPTA-DR3/epta-dr3-data-v0").resolve()
    assert resolved.results_dir == (repo_root / "results").resolve()


def test_pipeline_resolved_rejects_absolute_dataset_path(tmp_path: Path) -> None:
    repo_root = _make_git_root(tmp_path / "repo")
    cfg = PipelineConfig(
        home_dir=repo_root,
        singularity_image=tmp_path / "tempo2.sif",
        dataset_name=str(tmp_path / "outside"),
        results_dir=Path("results"),
    )

    try:
        cfg.resolved()
    except ValueError as exc:
        assert "dataset_name must be a path relative to home_dir" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for absolute dataset_name")


def test_pipeline_resolved_rejects_dataset_path_outside_repo(tmp_path: Path) -> None:
    repo_root = _make_git_root(tmp_path / "repo")
    cfg = PipelineConfig(
        home_dir=repo_root,
        singularity_image=tmp_path / "tempo2.sif",
        dataset_name="../outside",
        results_dir=Path("results"),
    )

    try:
        cfg.resolved()
    except ValueError as exc:
        assert "must resolve inside home_dir" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for escaping dataset_name")


def test_pipeline_resolved_rejects_non_repo_home_dir(tmp_path: Path) -> None:
    home_dir = tmp_path / "not_a_repo"
    home_dir.mkdir()
    cfg = PipelineConfig(
        home_dir=home_dir,
        singularity_image=tmp_path / "tempo2.sif",
        dataset_name="EPTA-DR3/epta-dr3-data-v0",
        results_dir=Path("results"),
    )

    try:
        cfg.resolved()
    except ValueError as exc:
        assert "home_dir must be the git repo root" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-repo home_dir")


def test_pipeline_config_repo_resources_resolve_from_declared_config_repo(
    tmp_path: Path,
) -> None:
    tool_repo = _make_git_root(tmp_path / "tool-repo")
    cfg_base_dir = tool_repo / "configs" / "runs" / "pqc"
    cfg_base_dir.mkdir(parents=True, exist_ok=True)
    resource = (
        tool_repo
        / "configs"
        / "rules"
        / "pqc"
        / "epta_dr_optimize_single_pulsar_backend_profiles.toml"
    )
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_text("[backend_profiles]\n", encoding="utf-8")

    data_repo = _make_git_root(tmp_path / "data-repo")
    cfg = PipelineConfig.from_dict(
        {
            "home_dir": str(data_repo),
            "dataset_name": "EPTA-DR3/epta-dr3-data",
            "singularity_image": str(tmp_path / "tempo2.sif"),
            "results_dir": "results",
            "pqc_backend_profiles_path": "configs/rules/pqc/epta_dr_optimize_single_pulsar_backend_profiles.toml",
        },
        base_dir=cfg_base_dir,
    )

    resolved = cfg.resolved()
    assert Path(str(resolved.pqc_backend_profiles_path)) == resource.resolve()


def test_ingest_resolved_output_root_uses_repo_relative_dataset_path(tmp_path: Path) -> None:
    repo_root = _make_git_root(tmp_path / "repo")
    cfg = IngestConfig(
        home_dir=repo_root,
        dataset_name="EPTA-DR3/epta-dr3-data-v0",
    )

    assert cfg.resolved_output_root() == (
        repo_root / "EPTA-DR3/epta-dr3-data-v0"
    ).resolve()


def test_ingest_resolved_output_root_rejects_mismatched_explicit_output(
    tmp_path: Path,
) -> None:
    repo_root = _make_git_root(tmp_path / "repo")
    cfg = IngestConfig(
        home_dir=repo_root,
        dataset_name="EPTA-DR3/epta-dr3-data-v0",
        ingest_output_dir=tmp_path / "different-output",
    )

    try:
        cfg.resolved_output_root()
    except ValueError as exc:
        assert "ingest_output_dir disagrees with home_dir + dataset_name" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for mismatched ingest_output_dir")
