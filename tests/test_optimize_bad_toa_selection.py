from __future__ import annotations

from pathlib import Path

import pandas as pd

from pleb.config import PipelineConfig
from pleb.optimize.folds import FoldConfig, load_fold_config
from pleb.optimize.fold_datasets import build_fold_dataset
from pleb.optimize.models import (
    ObjectiveConfig,
    OptimizationConfig,
    SearchSpace,
    TrialResult,
)
from pleb.optimize.objectives import compute_score, load_objective_config, violated_constraints
from pleb.optimize.optimizer import _score_trial
from pleb.optimize.scorers import (
    score_run_dir_consensus,
    score_run_dir_variants,
    write_bad_toa_masks,
    write_variant_consensus_artifacts,
    write_variant_selection_table,
)
from pleb.optimize.trial_runner import run_fold_trial


def _write_qc(path: Path, *, variant: str, bad, resid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(bad)
    pd.DataFrame(
        {
            "pulsar": ["J0000+0000"] * n,
            "variant": [variant] * n,
            "mjd": [58000.0 + i for i in range(n)],
            "freq": [1400.0] * n,
            "sys": ["A"] * n,
            "_timfile": [f"tims/{variant}.tim"] * n,
            "bad_point": bad,
            "resid_us": resid,
            "sigma_us": [1.0] * n,
        }
    ).to_csv(path, index=False)


def test_objective_constraints_penalize_destructive_bad_masks(tmp_path: Path) -> None:
    objective_path = tmp_path / "objective.toml"
    objective_path.write_text(
        """
[weights]
residual_cleanliness = 10.0
bad_fraction = -1.0

[constraints]
max_bad_fraction = 0.25
min_n_clean = 3
""".strip()
        + "\n",
        encoding="utf-8",
    )
    objective = load_objective_config(objective_path)

    acceptable = {"residual_cleanliness": 1.0, "bad_fraction": 0.25, "n_clean": 3.0}
    destructive = {"residual_cleanliness": 100.0, "bad_fraction": 0.75, "n_clean": 1.0}

    assert violated_constraints(acceptable, objective) == ()
    assert set(violated_constraints(destructive, objective)) == {
        "max_bad_fraction",
        "min_n_clean",
    }
    assert compute_score(acceptable, objective) > compute_score(destructive, objective)


def test_variants_are_scored_independently_and_constraints_select_safe_candidate(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_qc(
        run_dir / "qc" / "J0000+0000.aggressive_qc.csv",
        variant="aggressive",
        bad=[True, True, True, False],
        resid=[100.0, 50.0, 25.0, 0.01],
    )
    _write_qc(
        run_dir / "qc" / "J0000+0000.conservative_qc.csv",
        variant="conservative",
        bad=[False, True, False, False],
        resid=[0.10, 20.0, 0.20, 0.30],
    )

    by_variant = score_run_dir_variants(
        run_dir,
        fold_cfg=load_fold_config(None),
        parameter_complexity_penalty=0.0,
    )
    assert set(by_variant) == {"aggressive", "conservative"}
    assert by_variant["aggressive"][0]["n_toas"] == 4.0
    assert by_variant["conservative"][0]["n_toas"] == 4.0

    objective = ObjectiveConfig(
        weights={"residual_cleanliness": 100.0, "bad_fraction": -1.0},
        constraints={"max_bad_fraction": 0.50, "min_n_clean": 2.0},
        constraint_penalty=1.0e6,
    )
    selected, (metrics, _folds) = max(
        by_variant.items(),
        key=lambda item: compute_score(item[1][0], objective),
    )
    assert selected == "conservative"
    assert metrics["bad_fraction"] == 0.25

    table = write_variant_selection_table(
        run_dir, by_variant, objective, selected_variant=selected
    )
    tdf = pd.read_csv(table)
    assert set(tdf["variant"]) == {"aggressive", "conservative"}
    assert tdf.loc[tdf["variant"] == "conservative", "selected"].iloc[0] in {True, "True"}


def test_bad_mask_artifacts_have_stable_ids_keep_flags_and_reasons(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_qc(
        run_dir / "qc" / "J0000+0000.safe_qc.csv",
        variant="safe",
        bad=[False, True, False],
        resid=[0.1, 8.0, 0.2],
    )

    paths = write_bad_toa_masks(run_dir)
    assert len(paths) == 1
    mask = pd.read_csv(paths[0])
    assert list(mask["variant"].unique()) == ["safe"]
    assert mask["toa_id"].is_unique
    assert mask["bad"].tolist() == [False, True, False]
    assert mask["keep"].tolist() == [True, False, True]
    assert "bad_point" in str(mask.loc[mask["bad"], "bad_reason"].iloc[0])


def test_consensus_scoring_collapses_overlapping_variant_rows_and_tracks_support(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    qc_dir = run_dir / "qc"
    qc_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "pulsar": ["J0000+0000", "J0000+0000"],
            "variant": ["legacy", "legacy"],
            "mjd": [58000.0, 58001.0],
            "freq": [1400.0, 1400.0],
            "sys": ["A", "A"],
            "_timfile": ["tims/SHARED.tim", "tims/LEGACY.tim"],
            "bad_point": [True, False],
            "resid_us": [8.0, 0.2],
            "sigma_us": [1.0, 1.0],
        }
    ).to_csv(qc_dir / "J0000+0000.legacy_qc.csv", index=False)
    pd.DataFrame(
        {
            "pulsar": ["J0000+0000", "J0000+0000"],
            "variant": ["new", "new"],
            "mjd": [58000.0, 58002.0],
            "freq": [1400.0, 1400.0],
            "sys": ["A", "B"],
            "_timfile": ["tims/SHARED.tim", "tims/NEW.tim"],
            "bad_point": [False, True],
            "resid_us": [7.5, 9.0],
            "sigma_us": [1.0, 1.0],
        }
    ).to_csv(qc_dir / "J0000+0000.new_qc.csv", index=False)

    metrics, _folds, consensus = score_run_dir_consensus(
        run_dir,
        fold_cfg=load_fold_config(None),
        parameter_complexity_penalty=0.0,
    )

    assert metrics["n_toas"] == 3.0
    assert metrics["n_bad"] == 2.0
    shared = consensus.loc[consensus["_timfile_base"] == "SHARED.tim"].iloc[0]
    assert shared["bad_point"] in {True, "True"}
    assert shared["variant_support_present"] == 2
    assert shared["variant_support_any_bad_count"] == 1
    assert shared["variant_support_any_bad_fraction"] == 0.5

    written = write_variant_consensus_artifacts(run_dir, consensus_df=consensus)
    assert (run_dir / "optimize_bad_masks" / "variant_consensus.csv").exists()
    assert (run_dir / "optimize_bad_masks" / "consensus_bad_mask.csv").exists()
    assert (run_dir / "optimize_bad_masks" / "J0000+0000.consensus_qc.csv").exists()
    assert len(written) >= 3


def test_score_trial_selects_best_variant_and_writes_selection_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_qc(
        run_dir / "qc" / "J0000+0000.aggressive_qc.csv",
        variant="aggressive",
        bad=[True, True, True, False],
        resid=[100.0, 50.0, 25.0, 0.01],
    )
    _write_qc(
        run_dir / "qc" / "J0000+0000.conservative_qc.csv",
        variant="conservative",
        bad=[False, True, False, False],
        resid=[0.10, 20.0, 0.20, 0.30],
    )
    trial = TrialResult(
        trial_id=1,
        status="ok",
        params={},
        score=None,
        metrics={},
        run_dir=run_dir,
    )
    cfg = OptimizationConfig(
        base_config_path=tmp_path / "base.toml",
        variant_strategy="select_best",
    )
    objective = ObjectiveConfig(
        weights={"residual_cleanliness": 100.0, "bad_fraction": -1.0},
        constraints={"max_bad_fraction": 0.50, "min_n_clean": 2.0},
        constraint_penalty=1.0e6,
    )

    _score_trial(cfg, trial, FoldConfig(), objective, SearchSpace(parameters=[]))

    assert trial.metrics["bad_fraction"] == 0.25
    assert trial.score == compute_score(trial.metrics, objective)
    selection = pd.read_csv(
        run_dir / "optimize_bad_masks" / "variant_selection_scores.csv"
    )
    assert set(selection["variant"]) == {"aggressive", "conservative"}
    assert (
        selection.loc[selection["selected"].astype(str) == "True", "variant"].iloc[0]
        == "conservative"
    )
    assert (run_dir / "optimize_bad_masks" / "aggressive_bad_mask.csv").exists()
    assert (run_dir / "optimize_bad_masks" / "conservative_bad_mask.csv").exists()


def test_score_trial_auto_uses_consensus_when_variant_qc_files_exist(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    qc_dir = run_dir / "qc"
    qc_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "pulsar": ["J0000+0000", "J0000+0000"],
            "variant": ["legacy", "legacy"],
            "mjd": [58000.0, 58001.0],
            "freq": [1400.0, 1400.0],
            "sys": ["A", "A"],
            "_timfile": ["tims/SHARED.tim", "tims/LEGACY.tim"],
            "bad_point": [True, True],
            "resid_us": [8.0, 9.0],
            "sigma_us": [1.0, 1.0],
        }
    ).to_csv(qc_dir / "J0000+0000.legacy_qc.csv", index=False)
    pd.DataFrame(
        {
            "pulsar": ["J0000+0000", "J0000+0000"],
            "variant": ["new", "new"],
            "mjd": [58000.0, 58002.0],
            "freq": [1400.0, 1400.0],
            "sys": ["A", "B"],
            "_timfile": ["tims/SHARED.tim", "tims/NEW.tim"],
            "bad_point": [False, False],
            "resid_us": [7.0, 0.2],
            "sigma_us": [1.0, 1.0],
        }
    ).to_csv(qc_dir / "J0000+0000.new_qc.csv", index=False)

    trial = TrialResult(
        trial_id=1,
        status="ok",
        params={},
        score=None,
        metrics={},
        run_dir=run_dir,
    )
    cfg = OptimizationConfig(base_config_path=tmp_path / "base.toml")
    objective = ObjectiveConfig(weights={"bad_fraction": -1.0})

    _score_trial(cfg, trial, FoldConfig(), objective, SearchSpace(parameters=[]))

    assert trial.metrics["n_toas"] == 3.0
    assert trial.metrics["bad_fraction"] == 2.0 / 3.0
    selection = pd.read_csv(
        run_dir / "optimize_bad_masks" / "variant_selection_scores.csv"
    )
    assert set(selection["selected"].astype(str)) == {"False"}
    assert (run_dir / "optimize_bad_masks" / "variant_consensus.csv").exists()
    assert (run_dir / "optimize_bad_masks" / "J0000+0000.consensus_qc.csv").exists()


def test_score_trial_uses_pipeline_backend_col_for_variant_selection(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    qc_dir = run_dir / "qc"
    qc_dir.mkdir(parents=True)
    base = {
        "pulsar": ["J0000+0000"] * 4,
        "mjd": [58000.0, 58001.0, 58002.0, 58003.0],
        "freq": [1400.0] * 4,
        "sys": ["A"] * 4,
        "bad_point": [False] * 4,
        "resid_us": [0.1] * 4,
        "sigma_us": [1.0] * 4,
    }
    pd.DataFrame(
        {
            **base,
            "variant": ["safe"] * 4,
            "group": ["G1", "G1", "G2", "G2"],
            "event_member": [True, True, False, False],
            "_timfile": ["tims/safe.tim"] * 4,
        }
    ).to_csv(qc_dir / "J0000+0000.safe_qc.csv", index=False)
    pd.DataFrame(
        {
            **base,
            "variant": ["risky"] * 4,
            "group": ["G1", "G2", "G1", "G2"],
            "event_member": [True, True, False, False],
            "_timfile": ["tims/risky.tim"] * 4,
        }
    ).to_csv(qc_dir / "J0000+0000.risky_qc.csv", index=False)

    base_cfg = tmp_path / "pipeline.toml"
    base_cfg.write_text(
        "\n".join(
            [
                'home_dir = "."',
                'dataset_name = "."',
                'singularity_image = "tempo2.sif"',
                'results_dir = "results"',
                'branches = ["main"]',
                'reference_branch = "main"',
                'pulsars = ["J0000+0000"]',
                'pqc_backend_col = "group"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trial = TrialResult(
        trial_id=1,
        status="ok",
        params={},
        score=None,
        metrics={},
        run_dir=run_dir,
    )
    cfg = OptimizationConfig(
        base_config_path=base_cfg,
        variant_strategy="select_best",
    )
    objective = ObjectiveConfig(weights={"event_coherence": 1.0})

    _score_trial(cfg, trial, FoldConfig(), objective, SearchSpace(parameters=[]))

    assert trial.metrics["event_coherence"] == 1.0
    selection = pd.read_csv(
        run_dir / "optimize_bad_masks" / "variant_selection_scores.csv"
    )
    assert (
        selection.loc[selection["selected"].astype(str) == "True", "variant"].iloc[0]
        == "safe"
    )


def test_fold_dataset_can_be_built_from_selected_variant_only(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    psr_dir = dataset_root / "J0000+0000"
    tims_dir = psr_dir / "tims"
    tims_dir.mkdir(parents=True)
    (psr_dir / "J0000+0000.par").write_text("PSRJ J0000+0000\n", encoding="utf-8")
    (psr_dir / "J0000+0000_all.tim").write_text(
        "INCLUDE tims/A.tim\nINCLUDE tims/B.tim\nINCLUDE tims/stale.tim\n",
        encoding="utf-8",
    )
    (psr_dir / "J0000+0000_safe_all.tim").write_text(
        "INCLUDE tims/A.tim\n", encoding="utf-8"
    )
    (psr_dir / "J0000+0000_risky_all.tim").write_text(
        "INCLUDE tims/B.tim\n", encoding="utf-8"
    )
    (tims_dir / "A.tim").write_text(
        "FORMAT 1\nfileA 1400 58000.0 1.0 ao -sys A\n", encoding="utf-8"
    )
    (tims_dir / "B.tim").write_text(
        "FORMAT 1\nfileB 1400 59000.0 1.0 ao -sys B\n", encoding="utf-8"
    )
    (tims_dir / "stale.tim").write_text(
        "FORMAT 1\nstale 1400 60000.0 1.0 ao -sys STALE\n", encoding="utf-8"
    )

    cfg = PipelineConfig(
        home_dir=tmp_path,
        singularity_image=tmp_path / "tempo2.sif",
        dataset_name=dataset_root,
        pulsars=["J0000+0000"],
        pqc_run_variants=True,
    )
    tmp_home, held_out = build_fold_dataset(
        cfg,
        fold_cfg=FoldConfig(mode="backend_holdout", n_splits=1, backend_col="sys"),
        fold_index=0,
        out_root=tmp_path / "folds",
        variant_label="safe",
    )

    assert held_out == "A"
    copied = tmp_home / dataset_root.name / "J0000+0000"
    base_all = (copied / "J0000+0000_all.tim").read_text(encoding="utf-8")
    safe_all = (copied / "J0000+0000_safe_all.tim").read_text(encoding="utf-8")
    risky_all = (copied / "J0000+0000_risky_all.tim").read_text(encoding="utf-8")
    assert "INCLUDE tims/A.tim" not in base_all
    assert "INCLUDE tims/A.tim" not in safe_all
    assert "INCLUDE tims/B.tim" in risky_all
    assert "STALE" not in held_out


def test_fold_dataset_without_variant_uses_union_of_active_variant_includes(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    psr_dir = dataset_root / "J0000+0000"
    tims_dir = psr_dir / "tims"
    tims_dir.mkdir(parents=True)
    (psr_dir / "J0000+0000.par").write_text("PSRJ J0000+0000\n", encoding="utf-8")
    (psr_dir / "J0000+0000_left_all.tim").write_text(
        "INCLUDE tims/A.tim\n", encoding="utf-8"
    )
    (psr_dir / "J0000+0000_right_all.tim").write_text(
        "INCLUDE tims/B.tim\n", encoding="utf-8"
    )
    (tims_dir / "A.tim").write_text(
        "FORMAT 1\nfileA 1400 58000.0 1.0 ao -sys A\n", encoding="utf-8"
    )
    (tims_dir / "B.tim").write_text(
        "FORMAT 1\nfileB 1400 59000.0 1.0 ao -sys B\n", encoding="utf-8"
    )
    (tims_dir / "stale.tim").write_text(
        "FORMAT 1\nstale 1400 60000.0 1.0 ao -sys STALE\n", encoding="utf-8"
    )
    cfg = PipelineConfig(
        home_dir=tmp_path,
        singularity_image=tmp_path / "tempo2.sif",
        dataset_name=dataset_root,
        pulsars=["J0000+0000"],
        pqc_run_variants=True,
    )
    _tmp_home, held_out = build_fold_dataset(
        cfg,
        fold_cfg=FoldConfig(mode="backend_holdout", n_splits=2, backend_col="sys"),
        fold_index=1,
        out_root=tmp_path / "folds",
    )
    assert held_out in {"A", "B"}
    assert held_out != "STALE"


def test_fold_rerun_disables_variant_discovery_when_variant_is_selected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_cfg = tmp_path / "base.toml"
    base_cfg.write_text(
        """
home_dir = "."
singularity_image = "tempo2.sif"
dataset_name = "dataset"
pulsars = ["J0000+0000"]
pqc_run_variants = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    seen = {}

    def fake_run_pipeline(cfg):
        seen["cfg"] = cfg
        out = tmp_path / "trial_out"
        out.mkdir(parents=True, exist_ok=True)
        return {"tag": out}

    monkeypatch.setattr("pleb.optimize.trial_runner.run_pipeline", fake_run_pipeline)
    cfg = OptimizationConfig(base_config_path=base_cfg, execution_mode="pipeline")

    run_dir = run_fold_trial(
        cfg,
        1,
        {},
        fold_label="fold_00",
        home_dir=tmp_path / "fold_home",
        dataset_name="dataset",
        selected_variant="safe",
    )

    assert run_dir == tmp_path / "trial_out"
    assert seen["cfg"].pqc_run_variants is False


def test_fold_rerun_preserves_variant_discovery_without_selected_variant(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_cfg = tmp_path / "base.toml"
    base_cfg.write_text(
        """
home_dir = "."
singularity_image = "tempo2.sif"
dataset_name = "dataset"
pulsars = ["J0000+0000"]
pqc_run_variants = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    seen = {}

    def fake_run_pipeline(cfg):
        seen["cfg"] = cfg
        out = tmp_path / "trial_out"
        out.mkdir(parents=True, exist_ok=True)
        return {"tag": out}

    monkeypatch.setattr("pleb.optimize.trial_runner.run_pipeline", fake_run_pipeline)
    cfg = OptimizationConfig(base_config_path=base_cfg, execution_mode="pipeline")

    run_dir = run_fold_trial(
        cfg,
        1,
        {},
        fold_label="fold_00",
        home_dir=tmp_path / "fold_home",
        dataset_name="dataset",
        selected_variant=None,
    )

    assert run_dir == tmp_path / "trial_out"
    assert seen["cfg"].pqc_run_variants is True
