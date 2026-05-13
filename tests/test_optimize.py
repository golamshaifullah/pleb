"""Tests for the optimization module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pleb.config import PipelineConfig
from pleb.optimize.cli import load_optimization_config
from pleb.optimize.fold_datasets import build_fold_dataset
from pleb.optimize.folds import FoldConfig, load_fold_config
from pleb.optimize.models import OptimizationConfig, OptimizationResult, TrialResult
from pleb.optimize.objectives import compute_score, load_objective_config
from pleb.optimize.optimizer import _run_true_fold_reruns, run_optimization
from pleb.optimize.post_apply import _compute_backend_alignment_metrics
from pleb.optimize.report import write_pdf_report
from pleb.optimize.scorers import score_run_dir
from pleb.optimize.search_space import (
    active_parameter_count,
    load_search_space,
    parse_backend_profile_parameter_name,
    parameters_to_set_overrides,
    sample_parameters,
    split_backend_profile_parameters,
)
from pleb.optimize.trial_runner import run_fold_trial, run_trial


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
""".strip() + "\n",
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
""".strip() + "\n",
        encoding="utf-8",
    )
    objective = load_objective_config(objective_path)
    assert (
        compute_score({"bad_fraction": 0.1, "residual_cleanliness": 0.5}, objective)
        == 0.9
    )

    folds_path = tmp_path / "folds.toml"
    folds_path.write_text(
        """
[folds]
mode = "time_blocks"
n_splits = 3
""".strip() + "\n",
        encoding="utf-8",
    )
    folds = load_fold_config(folds_path)
    assert folds == FoldConfig(
        mode="time_blocks", n_splits=3, time_col="mjd", backend_col="sys"
    )


def test_load_optimization_config_reads_variant_strategy(tmp_path: Path) -> None:
    base_cfg = tmp_path / "pipeline.toml"
    base_cfg.write_text('home_dir = "."\n', encoding="utf-8")
    optimize_cfg = tmp_path / "optimize.toml"
    optimize_cfg.write_text(
        f"""
[optimize]
base_config_path = "{base_cfg}"
variant_strategy = "consensus"
""".strip() + "\n",
        encoding="utf-8",
    )

    cfg = load_optimization_config(optimize_cfg)

    assert cfg.variant_strategy == "consensus"


def test_load_optimization_config_reads_post_apply_settings(tmp_path: Path) -> None:
    base_cfg = tmp_path / "pipeline.toml"
    base_cfg.write_text('home_dir = "."\n', encoding="utf-8")
    optimize_cfg = tmp_path / "optimize.toml"
    optimize_cfg.write_text(
        f"""
[optimize]
base_config_path = "{base_cfg}"
post_apply_eval = true
post_apply_source_branch = "step2_branch"
post_apply_qc_branch = "step4_branch"
post_apply_qc_action = "delete"
""".strip() + "\n",
        encoding="utf-8",
    )

    cfg = load_optimization_config(optimize_cfg)

    assert cfg.post_apply_eval is True
    assert cfg.post_apply_source_branch == "step2_branch"
    assert cfg.post_apply_qc_branch == "step4_branch"
    assert cfg.post_apply_qc_action == "delete"


def test_load_optimization_config_resolves_repo_relative_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / ".git").mkdir(parents=True)
    optimize_dir = repo_root / "configs" / "optimize" / "runs"
    optimize_dir.mkdir(parents=True)
    optimize_cfg = optimize_dir / "optimize.toml"
    optimize_cfg.write_text(
        """
[optimize]
base_config_path = "configs/runs/pqc/base.toml"
search_space_path = "configs/optimize/search_spaces/space.toml"
objective_path = "configs/optimize/objectives/objective.toml"
folds_path = "configs/optimize/folds/folds.toml"
out_dir = "results/optimize/example"
""".strip() + "\n",
        encoding="utf-8",
    )

    cfg = load_optimization_config(optimize_cfg)

    assert (
        cfg.base_config_path
        == (repo_root / "configs" / "runs" / "pqc" / "base.toml").resolve()
    )
    assert (
        cfg.search_space_path
        == (
            repo_root / "configs" / "optimize" / "search_spaces" / "space.toml"
        ).resolve()
    )
    assert (
        cfg.objective_path
        == (
            repo_root / "configs" / "optimize" / "objectives" / "objective.toml"
        ).resolve()
    )
    assert (
        cfg.folds_path
        == (repo_root / "configs" / "optimize" / "folds" / "folds.toml").resolve()
    )
    assert cfg.out_dir == (repo_root / "results" / "optimize" / "example").resolve()


def test_load_optimization_config_resolves_fixed_override_home_dir(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / ".git").mkdir(parents=True)
    optimize_dir = repo_root / "configs" / "optimize" / "runs"
    optimize_dir.mkdir(parents=True)
    optimize_cfg = optimize_dir / "optimize.toml"
    optimize_cfg.write_text(
        """
[optimize]
base_config_path = "configs/runs/pqc/base.toml"

[optimize.fixed_overrides]
home_dir = "../../../"
dataset_name = "EPTA-DR3/epta-dr3-data"
""".strip() + "\n",
        encoding="utf-8",
    )

    cfg = load_optimization_config(optimize_cfg)

    assert cfg.fixed_overrides is not None
    assert cfg.fixed_overrides["home_dir"] == str(repo_root.resolve())
    assert cfg.fixed_overrides["dataset_name"] == "EPTA-DR3/epta-dr3-data"


def test_compute_backend_alignment_metrics_reads_alltim_order(tmp_path: Path) -> None:
    psr_dir = tmp_path / "J0000+0000"
    tims_dir = psr_dir / "tims"
    tims_dir.mkdir(parents=True)
    all_tim = psr_dir / "J0000+0000_all.tim"
    all_tim.write_text(
        "INCLUDE tims/A.tim\nINCLUDE tims/B.tim\n",
        encoding="utf-8",
    )
    (tims_dir / "A.tim").write_text(
        "FORMAT 1\nA1 1400 58000.0 1.0 ao -sys A\nA2 1400 58001.0 1.0 ao -sys A\n",
        encoding="utf-8",
    )
    (tims_dir / "B.tim").write_text(
        "FORMAT 1\nB1 1400 58002.0 1.0 ao -sys B\nB2 1400 58003.0 1.0 ao -sys B\n",
        encoding="utf-8",
    )
    gen = tmp_path / "J0000+0000_post_apply.general2"
    gen.write_text(
        "\n".join(
            [
                "Starting general2 plugin",
                "58000 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1400 0 0 0.000001 0 1 0 0",
                "58001 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1400 0 0 0.000002 0 1 0 0",
                "58002 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1400 0 0 0.000009 0 1 0 0",
                "58003 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1400 0 0 0.000010 0 1 0 0",
                "Finished general2 plugin",
                "",
            ]
        ),
        encoding="utf-8",
    )

    metrics = _compute_backend_alignment_metrics(
        timfile=all_tim,
        general2_path=gen,
        backend_flag="-sys",
    )

    assert metrics["post_apply_backend_count"] == 2.0
    assert metrics["post_apply_max_backend_abs_offset_us"] > 0.0
    assert metrics["post_apply_backend_alignment"] < 1.0


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
    base_profiles = tmp_path / "base_backend_profiles.toml"
    base_profiles.write_text(
        """
[backend_profiles]
BASE = { robust_z_thresh = 5.5 }
""".strip() + "\n",
        encoding="utf-8",
    )
    base_cfg = tmp_path / "pipeline.toml"
    base_cfg.write_text(
        f"""
home_dir = "{tmp_path}"
dataset_name = "."
singularity_image = "{tmp_path / 'tempo2.sif'}"
results_dir = "{tmp_path / 'results'}"
branches = ["main"]
reference_branch = "main"
pulsars = ["J0000+0000"]
run_tempo2 = true
run_pqc = true
pqc_backend_profiles_path = "{base_profiles}"
""".strip() + "\n",
        encoding="utf-8",
    )
    search_space = tmp_path / "space.toml"
    search_space.write_text(
        """
[parameters.pqc_fdr_q]
type = "float"
low = 0.001
high = 0.01

[parameters."backend_profile::A::robust_z_thresh"]
type = "float"
low = 4.0
high = 6.0
""".strip() + "\n",
        encoding="utf-8",
    )
    objective = tmp_path / "objective.toml"
    objective.write_text(
        """
[weights]
residual_cleanliness = 1.0
bad_fraction = -1.0
""".strip() + "\n",
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
out_dir = "{tmp_path / 'results' / 'optimize' / 'unit'}"
study_name = "unit"
n_trials = 1
sampler = "random"
seed = 7
jobs = 1

[optimize.fixed_overrides]
"backend_profile::B::fdr_q" = 0.02
""".strip() + "\n",
        encoding="utf-8",
    )

    from pleb.optimize import trial_runner as trial_runner_module

    seen_profile_docs: list[str] = []
    seen_profile_paths: list[Path] = []

    def fake_run_pipeline(_cfg):
        out_dir = tmp_path / "run_out" / getattr(_cfg, "outdir_name", "trial")
        qc_dir = out_dir / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        assert _cfg.pqc_backend_profiles_path is not None
        profile_path = Path(str(_cfg.pqc_backend_profiles_path))
        seen_profile_paths.append(profile_path)
        seen_profile_docs.append(profile_path.read_text(encoding="utf-8"))
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
    assert len(result.trials) == 1
    assert result.best_trial is not None
    assert (cfg.out_dir / "trials.csv").exists()
    assert (cfg.out_dir / "best_overrides.toml").exists()
    assert (cfg.out_dir / "best_backend_profiles.toml").exists()
    best_overrides = (cfg.out_dir / "best_overrides.toml").read_text(encoding="utf-8")
    assert "backend_profile::A::robust_z_thresh" not in best_overrides
    assert (
        'pqc_backend_profiles_path = "results/optimize/unit/best_backend_profiles.toml"'
        in best_overrides
    )
    best_profiles = (cfg.out_dir / "best_backend_profiles.toml").read_text(
        encoding="utf-8"
    )
    assert "BASE" in best_profiles
    assert "A" in best_profiles
    assert "robust_z_thresh" in best_profiles
    assert "B" in best_profiles
    assert "fdr_q = 0.02" in best_profiles
    assert len(seen_profile_docs) == 1
    assert "B" in seen_profile_docs[0]
    assert len(seen_profile_paths) == 1
    assert not seen_profile_paths[0].exists()
    assert tmp_path / "results" not in seen_profile_paths[0].parents
    assert not list((tmp_path / "results").rglob("backend_profiles.optimized.toml"))


def test_run_fold_trial_materializes_backend_profiles(
    tmp_path: Path, monkeypatch
) -> None:
    base_profiles = tmp_path / "base_backend_profiles.toml"
    base_profiles.write_text(
        """
[backend_profiles]
BASE = { robust_z_thresh = 5.5 }
""".strip() + "\n",
        encoding="utf-8",
    )
    base_cfg = tmp_path / "pipeline.toml"
    base_cfg.write_text(
        f"""
home_dir = "{tmp_path}"
dataset_name = "."
singularity_image = "{tmp_path / 'tempo2.sif'}"
results_dir = "{tmp_path / 'results'}"
branches = ["main"]
reference_branch = "main"
pulsars = ["J0000+0000"]
run_tempo2 = true
run_pqc = true
pqc_backend_profiles_path = "{base_profiles}"
""".strip() + "\n",
        encoding="utf-8",
    )
    cfg = OptimizationConfig(
        base_config_path=base_cfg,
        execution_mode="pipeline",
        out_dir=tmp_path / "results" / "optimize" / "study",
        study_name="study",
        fixed_overrides={"backend_profile::B::fdr_q": 0.02},
    )

    seen_profile_doc: dict[str, str] = {}
    seen_profile_path: dict[str, Path] = {}

    def fake_run_pipeline(_cfg):
        assert _cfg.pqc_backend_profiles_path is not None
        assert _cfg.readonly_materialized_dataset is True
        profile_path = Path(str(_cfg.pqc_backend_profiles_path))
        seen_profile_path["path"] = profile_path
        seen_profile_doc["text"] = profile_path.read_text(encoding="utf-8")
        out_dir = tmp_path / "run_out" / getattr(_cfg, "outdir_name", "trial")
        out_dir.mkdir(parents=True, exist_ok=True)
        return {"tag": out_dir}

    from pleb.optimize import trial_runner as trial_runner_module

    monkeypatch.setattr(trial_runner_module, "run_pipeline", fake_run_pipeline)

    run_dir = run_fold_trial(
        cfg,
        1,
        {"backend_profile::A::robust_z_thresh": 4.5},
        fold_label="fold_00",
        home_dir=tmp_path,
        dataset_name="dataset",
        selected_variant=None,
    )

    assert run_dir.name == "study_trial_0001_fold_00"
    assert "BASE" in seen_profile_doc["text"]
    assert "A" in seen_profile_doc["text"]
    assert "robust_z_thresh = 4.5" in seen_profile_doc["text"]
    assert "B" in seen_profile_doc["text"]
    assert "fdr_q = 0.02" in seen_profile_doc["text"]
    assert not seen_profile_path["path"].exists()
    assert tmp_path / "results" not in seen_profile_path["path"].parents


def test_run_trial_materializes_single_branch_source_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    import subprocess

    def _git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )

    _git("init", "-b", "main")
    psr_dir = repo_root / "dataset" / "J0000+0000"
    tims_dir = psr_dir / "tims"
    tims_dir.mkdir(parents=True)
    par_path = psr_dir / "J0000+0000.par"
    all_tim = psr_dir / "J0000+0000_all.tim"
    tim_path = tims_dir / "A.tim"
    par_path.write_text("PSRJ J0000+0000\nF0 1.0 1\n", encoding="utf-8")
    all_tim.write_text("INCLUDE tims/A.tim\n", encoding="utf-8")
    tim_path.write_text("FORMAT 1\nA 1400 58000.0 1.0 ao\n", encoding="utf-8")
    _git("add", ".")
    _git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "main",
    )

    _git("checkout", "-b", "scanbranch")
    par_path.write_text("PSRJ J0000+0000\nF0 2.0 1\n", encoding="utf-8")
    _git("add", "dataset/J0000+0000/J0000+0000.par")
    _git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "scanbranch",
    )
    _git("checkout", "main")
    par_path.write_text("PSRJ J0000+0000\nF0 9.0 1\n", encoding="utf-8")

    base_cfg = repo_root / "pipeline.toml"
    base_cfg.write_text(
        f"""
home_dir = "{repo_root}"
dataset_name = "dataset"
singularity_image = "{repo_root / 'tempo2.sif'}"
results_dir = "{repo_root / 'results'}"
branches = ["scanbranch"]
reference_branch = "scanbranch"
pulsars = ["J0000+0000"]
run_tempo2 = true
run_pqc = false
""".strip() + "\n",
        encoding="utf-8",
    )
    cfg = OptimizationConfig(
        base_config_path=base_cfg,
        execution_mode="pipeline",
        out_dir=repo_root / "results" / "optimize" / "study",
        study_name="study",
    )

    seen: dict[str, str] = {}

    def fake_run_pipeline(_cfg):
        assert _cfg.readonly_materialized_dataset is True
        psr_snapshot = Path(_cfg.home_dir) / Path(_cfg.dataset_name) / "J0000+0000"
        par_text = (psr_snapshot / "J0000+0000.par").read_text(encoding="utf-8")
        seen["dataset_name"] = Path(_cfg.dataset_name).as_posix()
        seen["par_text"] = par_text
        seen["snapshot_root"] = str(psr_snapshot.parent.parent)
        assert "F0 2.0 1" in par_text
        assert "F0 9.0 1" not in par_text
        return {"tag": repo_root / "results" / "trial_0001"}

    from pleb.optimize import trial_runner as trial_runner_module

    monkeypatch.setattr(trial_runner_module, "run_pipeline", fake_run_pipeline)

    trial = run_trial(cfg, 1, {})

    assert trial.status == "ok"
    assert seen["dataset_name"].startswith(
        ".pleb_optimize_trial_datasets/study/trial_0001/"
    )
    assert not Path(seen["snapshot_root"]).exists()


def test_run_fold_trial_workflow_marks_materialized_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    workflow_cfg = tmp_path / "workflow.toml"
    workflow_cfg.write_text(
        """
config = "pipeline.toml"
mode = "serial"
""".strip() + "\n",
        encoding="utf-8",
    )
    cfg = OptimizationConfig(
        base_config_path=workflow_cfg,
        execution_mode="workflow",
        out_dir=tmp_path / "results" / "optimize" / "study",
        study_name="study",
    )

    seen: dict[str, object] = {}

    class _Ctx:
        last_run_dir = tmp_path / "workflow_run"
        last_pipeline_run_dir = None

    def fake_run_workflow(path: Path):
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        seen["set"] = list(payload.get("set", []) or [])
        return _Ctx()

    from pleb.optimize import trial_runner as trial_runner_module

    monkeypatch.setattr(trial_runner_module, "run_workflow", fake_run_workflow)

    run_dir = run_fold_trial(
        cfg,
        1,
        {"pqc_fdr_q": 0.02},
        fold_label="fold_00",
        home_dir=tmp_path / "repo",
        dataset_name=".pleb_optimize_fold_datasets/study/trial_0001/dataset",
        selected_variant=None,
    )

    assert run_dir == _Ctx.last_run_dir
    assert "readonly_materialized_dataset=true" in seen["set"]
    assert 'home_dir="' + str(tmp_path / "repo") + '"' in seen["set"]


def test_true_fold_reruns_keep_repo_root_as_home_dir(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / ".git").mkdir(parents=True)
    dataset_root = repo_root / "dataset"
    (dataset_root / "J0000+0000").mkdir(parents=True)
    pipeline_cfg = PipelineConfig(
        home_dir=repo_root,
        singularity_image=repo_root / "tempo2.sif",
        dataset_name="dataset",
        results_dir=repo_root / "results",
        branches=["main"],
        reference_branch="main",
        pulsars=["J0000+0000"],
        run_tempo2=True,
        run_pqc=True,
    )
    cfg = OptimizationConfig(
        base_config_path=repo_root / "base.toml",
        out_dir=repo_root / "results" / "optimize" / "study",
        study_name="study",
    )
    trial = TrialResult(
        trial_id=1,
        status="ok",
        params={"pqc_fdr_q": 0.01},
        score=1.0,
        metrics={},
        run_dir=repo_root / "results" / "optimize" / "study" / "trial_0001",
    )
    fold_cfg = FoldConfig(
        mode="time_blocks", n_splits=1, time_col="mjd", backend_col="sys"
    )
    seen: dict[str, object] = {}

    def fake_build_fold_dataset(*_args, **kwargs):
        out_root = Path(kwargs["out_root"])
        tmp_home = out_root / "fold_00"
        tmp_dataset = tmp_home / dataset_root.name
        tmp_dataset.mkdir(parents=True, exist_ok=True)
        return tmp_home, "held_out"

    def fake_run_fold_trial(*_args, **kwargs):
        seen["home_dir"] = kwargs["home_dir"]
        seen["dataset_name"] = kwargs["dataset_name"]
        return repo_root / "results" / "optimize" / "study" / "trial_0001_fold_00"

    def fake_score_run_dir(*_args, **_kwargs):
        return {"n_toas": 1.0}, []

    monkeypatch.setattr(
        "pleb.optimize.optimizer.build_fold_dataset", fake_build_fold_dataset
    )
    monkeypatch.setattr("pleb.optimize.optimizer.run_fold_trial", fake_run_fold_trial)
    monkeypatch.setattr("pleb.optimize.optimizer.score_run_dir", fake_score_run_dir)

    folds = _run_true_fold_reruns(
        cfg,
        trial,
        pipeline_cfg,
        fold_cfg,
        load_search_space(
            Path(
                "/work/git_projects/pleb/configs/optimize/search_spaces/pqc_balanced_v1.toml"
            )
        ),
        load_objective_config(
            Path(
                "/work/git_projects/pleb/configs/optimize/objectives/single_pulsar_variant_consensus_production.toml"
            )
        ),
        selected_variant=None,
        variant_strategy="single",
        backend_col="sys",
    )

    assert len(folds) == 1
    assert seen["home_dir"] == repo_root.resolve()
    assert (
        seen["dataset_name"]
        == ".pleb_optimize_fold_datasets/study/trial_0001/fold_00/dataset"
    )


def test_parameter_override_helpers() -> None:
    params = {
        "pqc_fdr_q": 0.01,
        "pqc_step_enabled": True,
        "pulsars": ["J1713+0747"],
        "backend_profile::NRT.NUPPI.*::robust_z_thresh": 5.5,
    }
    overrides = parameters_to_set_overrides(params)
    assert "pqc_fdr_q=0.01" in overrides
    assert "pqc_step_enabled=true" in overrides
    assert 'pulsars=["J1713+0747"]' in overrides
    assert all("backend_profile::" not in item for item in overrides)
    assert parse_backend_profile_parameter_name(
        "backend_profile::NRT.NUPPI.*::robust_z_thresh"
    ) == ("NRT.NUPPI.*", "robust_z_thresh")
    flat, profiles = split_backend_profile_parameters(params)
    assert "backend_profile::NRT.NUPPI.*::robust_z_thresh" not in flat
    assert profiles == {"NRT.NUPPI.*": {"robust_z_thresh": 5.5}}
    space = load_search_space(
        Path(
            "/work/git_projects/pleb/configs/optimize/search_spaces/pqc_balanced_v1.toml"
        )
    )
    assert active_parameter_count(space, {"pqc_step_enabled": False}) >= 1


def test_write_pdf_report_with_baseline_and_trials(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline" / "main"
    trial_root = tmp_path / "trial_0001" / "main"
    baseline_run = baseline_root / "qc" / "main"
    trial_run = trial_root / "qc" / "main"
    baseline_run.mkdir(parents=True)
    trial_run.mkdir(parents=True)
    pd.DataFrame(
        {
            "mjd": [1.0, 2.0, 3.0],
            "resid_us": [0.1, 2.0, 0.3],
            "sigma_us": [1.0, 1.0, 1.0],
            "bad_point": [False, True, False],
            "event_type": [None, "step", None],
        }
    ).to_csv(baseline_run / "J0000+0000_qc.csv", index=False)
    pd.DataFrame(
        {
            "mjd": [1.0, 2.0, 3.0],
            "resid_us": [0.1, 0.2, 0.15],
            "sigma_us": [1.0, 1.0, 1.0],
            "bad_point": [False, False, False],
            "event_type": [None, None, None],
        }
    ).to_csv(trial_run / "J0000+0000_qc.csv", index=False)
    cfg = OptimizationConfig(
        base_config_path=tmp_path / "base.toml",
        out_dir=tmp_path / "out",
        study_name="unit",
        baseline_run_dir=baseline_root,
    )
    trial = TrialResult(
        trial_id=1,
        status="ok",
        params={"pqc_fdr_q": 0.01},
        score=1.2,
        metrics={},
        run_dir=trial_root,
    )
    result = OptimizationResult(
        config=cfg,
        trials=[trial],
        best_trial=trial,
        out_dir=cfg.out_dir,
    )
    pdf_path = write_pdf_report(result)
    assert pdf_path is not None
    assert pdf_path.exists()
