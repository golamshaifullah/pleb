"""Tests for the optimization module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pleb.optimize.cli import load_optimization_config
from pleb.optimize.fold_datasets import build_fold_dataset
from pleb.optimize.folds import FoldConfig, load_fold_config
from pleb.optimize.objectives import compute_score, load_objective_config
from pleb.optimize.optimizer import run_optimization
from pleb.optimize.scorers import score_run_dir
from pleb.optimize.search_space import (
    active_parameter_count,
    load_search_space,
    parameters_to_set_overrides,
    sample_parameters,
)


def test_load_search_space_and_sample_conditional(tmp_path: Path) -> None:
    path = tmp_path / "space.toml"
    path.write_text(
        """
[parameters.pqc_step_enabled]
type = "bool"

[parameters.pqc_step_delta_chi2_thresh]
type = "float"
low = 10.0
high = 20.0
depends_on = "pqc_step_enabled"
enabled_values = [true]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    space = load_search_space(path)
    assert len(space.parameters) == 2
    seen_true = False
    seen_false = False
    for _ in range(50):
        params = sample_parameters(space, __import__("random").Random(_))
        if params["pqc_step_enabled"]:
            seen_true = True
            assert "pqc_step_delta_chi2_thresh" in params
        else:
            seen_false = True
            assert "pqc_step_delta_chi2_thresh" not in params
    assert seen_true and seen_false


def test_objective_and_fold_loaders(tmp_path: Path) -> None:
    objective_path = tmp_path / "objective.toml"
    objective_path.write_text(
        """
[weights]
bad_fraction = -1.0
residual_cleanliness = 2.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    objective = load_objective_config(objective_path)
    assert compute_score(
        {"bad_fraction": 0.1, "residual_cleanliness": 0.5}, objective
    ) == 0.9

    folds_path = tmp_path / "folds.toml"
    folds_path.write_text(
        """
[folds]
mode = "time_blocks"
n_splits = 3
""".strip()
        + "\n",
        encoding="utf-8",
    )
    folds = load_fold_config(folds_path)
    assert folds == FoldConfig(mode="time_blocks", n_splits=3, time_col="mjd", backend_col="sys")


def test_score_run_dir_reads_qc_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    qc_dir = run_dir / "qc"
    qc_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "mjd": [1.0, 2.0, 3.0, 4.0],
            "sys": ["A", "A", "B", "B"],
            "bad_point": [False, True, False, False],
            "transient_id": [-1, 2, 2, -1],
            "resid_us": [0.1, 5.0, 0.2, 0.25],
            "sigma_us": [1.0, 1.0, 1.0, 1.0],
        }
    )
    df.to_csv(qc_dir / "J0000+0000_qc.csv", index=False)
    metrics, folds = score_run_dir(
        run_dir,
        fold_cfg=FoldConfig(mode="time_blocks", n_splits=2),
        parameter_complexity_penalty=0.25,
    )
    assert metrics["n_toas"] == 4.0
    assert metrics["n_bad"] == 1.0
    assert metrics["n_events"] == 1.0
    assert metrics["parameter_complexity_penalty"] == 0.25
    assert len(folds) == 2


def test_build_fold_dataset_rewrites_backend_tims(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    psr_dir = dataset_root / "J0000+0000"
    tims_dir = psr_dir / "tims"
    tims_dir.mkdir(parents=True)
    (psr_dir / "J0000+0000.par").write_text("PSRJ J0000+0000\n", encoding="utf-8")
    (psr_dir / "J0000+0000_all.tim").write_text(
        "INCLUDE tims/A.tim\nINCLUDE tims/B.tim\n", encoding="utf-8"
    )
    (tims_dir / "A.tim").write_text(
        "FORMAT 1\nfileA 1400 58000.0 1.0 ao -sys A\n",
        encoding="utf-8",
    )
    (tims_dir / "B.tim").write_text(
        "FORMAT 1\nfileB 1400 59000.0 1.0 ao -sys B\n",
        encoding="utf-8",
    )

    from pleb.config import PipelineConfig

    cfg = PipelineConfig(
        home_dir=tmp_path,
        singularity_image=tmp_path / "tempo2.sif",
        dataset_name=dataset_root,
        pulsars=["J0000+0000"],
    )
    tmp_home, held_out = build_fold_dataset(
        cfg,
        fold_cfg=FoldConfig(mode="backend_holdout", n_splits=2, backend_col="sys"),
        fold_index=0,
        out_root=tmp_path / "folds",
    )
    assert held_out in {"A", "B"}
    copied = tmp_home / dataset_root.name / "J0000+0000" / "J0000+0000_all.tim"
    all_text = copied.read_text(encoding="utf-8")
    assert "INCLUDE tims/" in all_text
    if held_out == "A":
        assert "INCLUDE tims/A.tim" not in all_text
    else:
        assert "INCLUDE tims/B.tim" not in all_text


def test_run_optimization_with_pipeline_monkeypatch(
    tmp_path: Path, monkeypatch
) -> None:
    base_cfg = tmp_path / "pipeline.toml"
    base_cfg.write_text(
        """
home_dir = "."
dataset_name = "."
singularity_image = "tempo2.sif"
results_dir = "results"
branches = ["main"]
reference_branch = "main"
pulsars = ["J0000+0000"]
run_tempo2 = true
run_pqc = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    search_space = tmp_path / "space.toml"
    search_space.write_text(
        """
[parameters.pqc_fdr_q]
type = "float"
low = 0.001
high = 0.01
""".strip()
        + "\n",
        encoding="utf-8",
    )
    objective = tmp_path / "objective.toml"
    objective.write_text(
        """
[weights]
residual_cleanliness = 1.0
bad_fraction = -1.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    optimize_cfg = tmp_path / "optimize.toml"
    optimize_cfg.write_text(
        f"""
[optimize]
base_config_path = "{base_cfg}"
execution_mode = "pipeline"
search_space_path = "{search_space}"
objective_path = "{objective}"
out_dir = "{tmp_path / 'optimize_out'}"
study_name = "unit"
n_trials = 2
sampler = "random"
seed = 7
jobs = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    from pleb.optimize import trial_runner as trial_runner_module

    def fake_run_pipeline(_cfg):
        out_dir = tmp_path / "run_out" / getattr(_cfg, "outdir_name", "trial")
        qc_dir = out_dir / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "mjd": [1.0, 2.0],
                "sys": ["A", "B"],
                "bad_point": [False, True],
                "resid_us": [0.1, 5.0],
                "sigma_us": [1.0, 1.0],
            }
        ).to_csv(qc_dir / "J0000+0000_qc.csv", index=False)
        return {"tag": out_dir}

    monkeypatch.setattr(trial_runner_module, "run_pipeline", fake_run_pipeline)
    cfg = load_optimization_config(optimize_cfg)
    result = run_optimization(cfg)
    assert len(result.trials) == 2
    assert result.best_trial is not None
    assert (cfg.out_dir / "trials.csv").exists()
    assert (cfg.out_dir / "best_overrides.toml").exists()


def test_parameter_override_helpers() -> None:
    params = {"pqc_fdr_q": 0.01, "pqc_step_enabled": True, "pulsars": ["J1713+0747"]}
    overrides = parameters_to_set_overrides(params)
    assert 'pqc_fdr_q=0.01' in overrides
    assert 'pqc_step_enabled=true' in overrides
    assert 'pulsars=["J1713+0747"]' in overrides
    space = load_search_space(
        Path("/work/git_projects/pleb/configs/optimize/search_spaces/pqc_balanced_v1.toml")
    )
    assert active_parameter_count(space, {"pqc_step_enabled": False}) >= 1
