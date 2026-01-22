from __future__ import annotations

from pathlib import Path

from data_combination_pipeline.config import PipelineConfig


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
