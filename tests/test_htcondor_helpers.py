from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = REPO_ROOT / "scripts/htcondor/prepare_optimize_submit.py"
RUN_SCRIPT = REPO_ROOT / "scripts/htcondor/run_optimize_one.sh"


def _init_dataset_repo(repo: Path, pulsars: list[str]) -> None:
    for pulsar in pulsars:
        psr_dir = repo / "EPTA-DR3" / "epta-dr3-data" / pulsar
        psr_dir.mkdir(parents=True, exist_ok=True)
        (psr_dir / f"{pulsar}.par").write_text("PSRJ test\n", encoding="utf-8")
        (psr_dir / f"{pulsar}_all.tim").write_text("FORMAT 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "EPTA-DR3"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    subprocess.run(["git", "branch", "step2_pqc_variant_smoke"], cwd=repo, check=True)


def test_prepare_optimize_submit_writes_queue_files(tmp_path: Path) -> None:
    dataset_home = tmp_path / "dataset"
    dataset_home.mkdir()
    _init_dataset_repo(dataset_home, ["J1713+0747", "J1909-3744"])
    out_dir = tmp_path / "out"
    results_root = dataset_home / "results"

    subprocess.run(
        [
            sys.executable,
            str(PREPARE_SCRIPT),
            "--dataset-home",
            str(dataset_home),
            "--results-root",
            str(results_root),
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    pulsars = (out_dir / "pulsars.txt").read_text(encoding="utf-8").strip().splitlines()
    submit_text = (out_dir / "optimize.sub").read_text(encoding="utf-8")

    assert pulsars == ["J1713+0747", "J1909-3744"]
    assert "queue pulsar from" in submit_text
    assert str(RUN_SCRIPT) in submit_text
    assert "PLEB_SOURCE_BRANCH=step2_pqc_variant_smoke" in submit_text


def test_run_optimize_one_dry_run_creates_isolated_worktree(tmp_path: Path) -> None:
    dataset_home = tmp_path / "dataset"
    dataset_home.mkdir()
    _init_dataset_repo(dataset_home, ["J1713+0747"])
    results_root = dataset_home / "results"
    results_root.mkdir(exist_ok=True)
    fake_sif = tmp_path / "fake.sif"
    fake_sif.write_text("", encoding="utf-8")

    env = {
        **os.environ,
        "PLEB_DATASET_HOME": str(dataset_home),
        "PLEB_DATASET_NAME": "EPTA-DR3/epta-dr3-data",
        "PLEB_RESULTS_ROOT": str(results_root),
        "PLEB_SOURCE_BRANCH": "step2_pqc_variant_smoke",
        "PLEB_OUTER_SIF": str(fake_sif),
        "PLEB_DRY_RUN": "1",
        "PLEB_KEEP_WORKTREE": "1",
    }

    subprocess.run(
        ["bash", str(RUN_SCRIPT), "J1713+0747"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    config_root = results_root / "htcondor" / "generated_configs" / "j1713_0747"
    cfgs = sorted(config_root.rglob("*.toml"))
    worktrees = sorted((results_root / "htcondor" / "worktrees").iterdir())

    assert len(cfgs) == 2
    assert worktrees
    base_text = next(
        p.read_text(encoding="utf-8") for p in cfgs if p.name.endswith(".base.toml")
    )
    opt_text = next(
        p.read_text(encoding="utf-8") for p in cfgs if p.name.endswith(".optimize.toml")
    )

    assert 'pulsars = ["J1713+0747"]' in base_text
    assert 'variant_strategy = "auto"' in opt_text
    assert 'execution_mode = "pipeline"' in opt_text
