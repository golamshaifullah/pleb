from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd

from pleb.config import PipelineConfig
from pleb.pipeline import _warn_backend_tim_drift, run_pipeline


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def test_run_pipeline_readonly_branches_use_branch_snapshots(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-b", "main")

    psr_dir = repo_root / "dataset" / "J0000+0000"
    tims_dir = psr_dir / "tims"
    tims_dir.mkdir(parents=True)
    par_path = psr_dir / "J0000+0000.par"
    all_tim = psr_dir / "J0000+0000_all.tim"
    tim_path = tims_dir / "A.tim"

    par_path.write_text("PSRJ J0000+0000\nF0 1.0 1\n", encoding="utf-8")
    all_tim.write_text("INCLUDE tims/A.tim\n", encoding="utf-8")
    tim_path.write_text("FORMAT 1\nA 1400 58000.0 1.0 ao\n", encoding="utf-8")
    _git(repo_root, "add", ".")
    _git(
        repo_root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "main",
    )

    _git(repo_root, "checkout", "-b", "scanbranch")
    par_path.write_text("PSRJ J0000+0000\nF0 2.0 1\n", encoding="utf-8")
    tim_path.write_text("FORMAT 1\nA 1400 58001.0 1.0 ao\n", encoding="utf-8")
    _git(
        repo_root,
        "add",
        "dataset/J0000+0000/J0000+0000.par",
        "dataset/J0000+0000/tims/A.tim",
    )
    _git(
        repo_root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "scanbranch",
    )

    _git(repo_root, "checkout", "main")
    # Dirty current worktree with content that would block checkout to scanbranch.
    par_path.write_text("PSRJ J0000+0000\nF0 9.0 1\n", encoding="utf-8")

    seen: dict[str, str] = {}

    def _assert_snapshot_par(path: Path, *, key: str) -> None:
        text = path.read_text(encoding="utf-8")
        seen[key] = text
        seen[f"{key}_path"] = str(path)
        assert "F0 2.0 1" in text
        assert "F0 9.0 1" not in text
        assert "__branch_snapshot" in str(path)

    def _fake_fix_pulsar_dataset(psr_dir: Path, _cfg):
        _assert_snapshot_par(psr_dir / f"{psr_dir.name}.par", key="fix")
        return {"psr": psr_dir.name}

    def _fake_write_fix_report(_reports, _out_dir):
        return None

    def _fake_run_tempo2_for_pulsar(
        home_dir: Path,
        dataset_name: Path,
        singularity_image: Path,
        *,
        native: bool,
        out_paths,
        pulsar: str,
        branch: str,
        epoch: str,
        force_rerun: bool = False,
    ) -> None:
        _ = singularity_image, native, branch, epoch, force_rerun
        psr_dir = Path(home_dir) / Path(dataset_name) / pulsar
        _assert_snapshot_par(psr_dir / f"{pulsar}.par", key="tempo2")

    def _fake_run_pqc_for_parfile_subprocess(
        parfile: Path,
        out_csv: Path,
        _cfg,
        **kwargs,
    ):
        _ = kwargs
        _assert_snapshot_par(parfile, key="pqc")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "filename": ["A.tim"],
                "mjd": [58001.0],
                "freq": [1400.0],
                "bad": [False],
            }
        )
        df.to_csv(out_csv, index=False)
        return df

    def _fake_plot_systems_per_pulsar(
        home_dir: Path,
        dataset_name: Path,
        out_paths,
        pulsars,
        branch: str,
        dpi: int,
    ) -> None:
        _ = out_paths, pulsars, branch, dpi
        dataset_path = Path(home_dir) / Path(dataset_name)
        seen["plot_systems_dataset"] = str(dataset_path)
        assert "__branch_snapshot" in str(dataset_path)

    def _fake_plot_pulsars_per_system(
        home_dir: Path,
        dataset_name: Path,
        out_paths,
        pulsars,
        branch: str,
        dpi: int,
    ) -> None:
        _ = out_paths, pulsars, branch, dpi
        dataset_path = Path(home_dir) / Path(dataset_name)
        seen["plot_pulsars_dataset"] = str(dataset_path)
        assert "__branch_snapshot" in str(dataset_path)

    def _fake_write_outlier_tables(
        home_dir: Path,
        dataset_name: Path,
        out_paths,
        pulsars,
        branches,
    ) -> None:
        _ = out_paths, pulsars, branches
        dataset_path = Path(home_dir) / Path(dataset_name)
        seen["outlier_dataset"] = str(dataset_path)
        assert "__branch_snapshot" in str(dataset_path)

    def _fake_analyse_binary_from_par(parfile: Path):
        _assert_snapshot_par(parfile, key="binary")
        return {"BINARY": "ELL1"}

    monkeypatch.setattr("pleb.pipeline._pqc_available", lambda: True)
    monkeypatch.setattr("pleb.pipeline.fix_pulsar_dataset", _fake_fix_pulsar_dataset)
    monkeypatch.setattr("pleb.pipeline.write_fix_report", _fake_write_fix_report)
    monkeypatch.setattr(
        "pleb.pipeline.run_tempo2_for_pulsar", _fake_run_tempo2_for_pulsar
    )
    monkeypatch.setattr(
        "pleb.pipeline.run_pqc_for_parfile_subprocess",
        _fake_run_pqc_for_parfile_subprocess,
    )
    monkeypatch.setattr(
        "pleb.pipeline.plot_systems_per_pulsar",
        _fake_plot_systems_per_pulsar,
    )
    monkeypatch.setattr(
        "pleb.pipeline.plot_pulsars_per_system",
        _fake_plot_pulsars_per_system,
    )
    monkeypatch.setattr(
        "pleb.pipeline.write_outlier_tables", _fake_write_outlier_tables
    )
    monkeypatch.setattr(
        "pleb.pipeline.analyse_binary_from_par", _fake_analyse_binary_from_par
    )

    cfg = PipelineConfig(
        home_dir=repo_root,
        singularity_image=repo_root / "tempo2.sif",
        dataset_name="dataset",
        results_dir=repo_root / "results",
        branches=["scanbranch"],
        reference_branch="scanbranch",
        pulsars=["J0000+0000"],
        tempo2_native=True,
        jobs=1,
        run_fix_dataset=True,
        run_tempo2=True,
        run_pqc=True,
        make_toa_coverage_plots=True,
        make_outlier_reports=True,
        make_binary_analysis=True,
        make_covariance_heatmaps=False,
        make_residual_plots=False,
        make_change_reports=False,
        qc_report=False,
        consolidated_report=False,
        outdir_name="readonly_snapshot_test",
    )

    out = run_pipeline(cfg)

    assert "F0 9.0 1" in par_path.read_text(encoding="utf-8")
    assert "__branch_snapshot" in seen["fix_path"]
    assert "__branch_snapshot" in seen["tempo2_path"]
    assert "__branch_snapshot" in seen["pqc_path"]
    assert "__branch_snapshot" in seen["binary_path"]
    assert "__branch_snapshot" in seen["plot_systems_dataset"]
    assert "__branch_snapshot" in seen["plot_pulsars_dataset"]
    assert "__branch_snapshot" in seen["outlier_dataset"]

    assert (out["qc"] / "qc_summary.tsv").exists()
    assert (out["binary_analysis"] / "binary_analysis.tsv").exists()


def test_run_pipeline_fix_apply_reads_new_branch_via_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-b", "main")

    psr_dir = repo_root / "dataset" / "J0000+0000"
    psr_dir.mkdir(parents=True)
    par_path = psr_dir / "J0000+0000.par"
    tim_path = psr_dir / "J0000+0000_all.tim"
    par_path.write_text("PSRJ J0000+0000\nF0 1.0 1\n", encoding="utf-8")
    tim_path.write_text("FORMAT 1\n", encoding="utf-8")
    _git(repo_root, "add", ".")
    _git(
        repo_root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "base",
    )

    seen: dict[str, str] = {}

    def _fake_fix_pulsar_dataset(psr_dir: Path, _cfg):
        (psr_dir / f"{psr_dir.name}.par").write_text(
            "PSRJ J0000+0000\nF0 2.0 1\n", encoding="utf-8"
        )
        return {"psr": psr_dir.name}

    def _fake_run_tempo2_for_pulsar(
        home_dir: Path,
        dataset_name: Path,
        singularity_image: Path,
        *,
        native: bool,
        out_paths,
        pulsar: str,
        branch: str,
        epoch: str,
        force_rerun: bool = False,
    ) -> None:
        _ = singularity_image, native, out_paths, branch, epoch, force_rerun
        psr_par = Path(home_dir) / Path(dataset_name) / pulsar / f"{pulsar}.par"
        seen["tempo2_path"] = str(psr_par)
        text = psr_par.read_text(encoding="utf-8")
        assert "__branch_snapshot" in str(psr_par)
        assert "F0 2.0 1" in text
        assert "F0 1.0 1" not in text

    monkeypatch.setattr("pleb.pipeline.fix_pulsar_dataset", _fake_fix_pulsar_dataset)
    monkeypatch.setattr("pleb.pipeline.write_fix_report", lambda reports, out_dir: None)
    monkeypatch.setattr(
        "pleb.pipeline._validate_fixdataset_qc_inputs",
        lambda pulsars, cfg, *, branch: None,
    )
    monkeypatch.setattr(
        "pleb.pipeline.run_tempo2_for_pulsar", _fake_run_tempo2_for_pulsar
    )

    cfg = PipelineConfig(
        home_dir=repo_root,
        singularity_image=repo_root / "tempo2.sif",
        dataset_name="dataset",
        results_dir=repo_root / "results",
        branches=["apply_branch"],
        reference_branch="main",
        pulsars=["J0000+0000"],
        jobs=1,
        tempo2_native=True,
        run_fix_dataset=True,
        fix_apply=True,
        fix_base_branch="main",
        fix_branch_name="apply_branch",
        run_tempo2=True,
        make_change_reports=False,
        qc_report=False,
        consolidated_report=False,
        outdir_name="fix_apply_snapshot_test",
    )

    run_pipeline(cfg)

    assert "F0 1.0 1" in par_path.read_text(encoding="utf-8")
    assert "__branch_snapshot" in seen["tempo2_path"]

    show = subprocess.run(
        ["git", "show", "apply_branch:dataset/J0000+0000/J0000+0000.par"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "F0 2.0 1" in show.stdout

    worktrees = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "refs/heads/apply_branch" not in worktrees.stdout


def test_warn_backend_tim_drift_uses_snapshots(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-b", "main")

    psr_dir = repo_root / "dataset" / "J0000+0000" / "tims"
    psr_dir.mkdir(parents=True)
    (psr_dir.parent / "J0000+0000.par").write_text(
        "PSRJ J0000+0000\n", encoding="utf-8"
    )
    (psr_dir / "A.tim").write_text("FORMAT 1\n", encoding="utf-8")
    _git(repo_root, "add", ".")
    _git(
        repo_root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "main",
    )

    _git(repo_root, "checkout", "-b", "compare_branch")
    (psr_dir / "B.tim").write_text("FORMAT 1\n", encoding="utf-8")
    _git(repo_root, "add", "dataset/J0000+0000/tims/B.tim")
    _git(
        repo_root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "compare",
    )
    _git(repo_root, "checkout", "main")

    monkeypatch.setattr(
        "pleb.pipeline.checkout",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("live checkout should not be used")
        ),
    )

    cfg = PipelineConfig(
        home_dir=repo_root,
        singularity_image=repo_root / "tempo2.sif",
        dataset_name="dataset",
        results_dir=repo_root / "results",
        branches=["main"],
        reference_branch="main",
        pulsars=["J0000+0000"],
        jobs=1,
    ).resolved()

    from git import Repo  # type: ignore

    repo = Repo(str(repo_root), search_parent_directories=False)
    _warn_backend_tim_drift(
        repo,
        cfg,
        ["J0000+0000"],
        baseline_branch="main",
        compare_branch="compare_branch",
        return_branch="main",
    )
