"""Tests for optional whitenoise integration helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess

from pleb.config import PipelineConfig
from pleb.pipeline import run_pipeline

from pleb.whitenoise_integration import (
    WhiteNoiseStageConfig,
    estimate_white_noise_for_pulsar,
    resolve_timfile_for_pulsar,
)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


def test_resolve_timfile_for_pulsar_prefers_template(tmp_path: Path) -> None:
    psr_dir = tmp_path / "J1713+0747"
    psr_dir.mkdir()
    explicit = psr_dir / "J1713+0747_all.new.tim"
    explicit.write_text("", encoding="utf-8")
    (psr_dir / "J1713+0747_all.tim").write_text("", encoding="utf-8")

    got = resolve_timfile_for_pulsar(psr_dir, "J1713+0747", "{pulsar}_all.new.tim")
    assert got == explicit


def test_resolve_timfile_for_pulsar_fallbacks(tmp_path: Path) -> None:
    psr_dir = tmp_path / "J1022+1001"
    psr_dir.mkdir()
    default_all = psr_dir / "J1022+1001_all.tim"
    default_all.write_text("", encoding="utf-8")
    assert resolve_timfile_for_pulsar(psr_dir, "J1022+1001", None) == default_all


def test_estimate_white_noise_for_pulsar_maps_result(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeResult:
        n_toas = 10
        n_epochs = 6
        has_multi_toa_epochs = True
        efac = 1.2
        efac_err = 0.1
        equad = 1e-6
        equad_err = 2e-7
        ecorr = 5e-7
        ecorr_err = 1e-7
        extra_variance_floor = 0.0
        extra_variance_floor_err = 0.0
        single_toa_mode = "combined"
        warning = ""
        success = True
        message = "ok"
        fun = 1.0

    def _fake_estimator(**kwargs):  # noqa: ARG001
        return _FakeResult()

    monkeypatch.setattr(
        "pleb.whitenoise_integration._resolve_estimator",
        lambda source_path=None: _fake_estimator,
    )

    par = tmp_path / "J1713+0747.par"
    tim = tmp_path / "J1713+0747_all.tim"
    par.write_text("", encoding="utf-8")
    tim.write_text("", encoding="utf-8")
    cfg = WhiteNoiseStageConfig()

    row = estimate_white_noise_for_pulsar(par, tim, cfg)
    assert row["n_toas"] == 10
    assert row["n_epochs"] == 6
    assert row["success"] is True
    assert row["efac"] == 1.2
    assert row["ecorr"] == 5e-7


def test_run_pipeline_whitenoise_uses_branch_snapshot_without_checkout(
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
    _git(repo_root, "add", "dataset/J0000+0000/J0000+0000.par")
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

    def _fake_estimate(parfile: Path, timfile: Path, cfg: WhiteNoiseStageConfig):
        seen["par"] = parfile.read_text(encoding="utf-8")
        seen["tim"] = timfile.read_text(encoding="utf-8")
        seen["par_path"] = str(parfile)
        assert "F0 2.0 1" in seen["par"]
        assert "F0 9.0 1" not in seen["par"]
        assert "INCLUDE tims/A.tim" in seen["tim"]
        assert "__branch_snapshot" in seen["par_path"]
        return {
            "n_toas": 1,
            "n_epochs": 1,
            "has_multi_toa_epochs": False,
            "efac": 1.0,
            "efac_err": 0.1,
            "equad": 0.0,
            "equad_err": 0.0,
            "ecorr": 0.0,
            "ecorr_err": 0.0,
            "extra_variance_floor": 0.0,
            "extra_variance_floor_err": 0.0,
            "single_toa_mode": cfg.single_toa_mode,
            "warning": "",
            "success": True,
            "message": "ok",
            "fun": 0.0,
        }

    monkeypatch.setattr("pleb.pipeline.estimate_white_noise_for_pulsar", _fake_estimate)

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
        run_tempo2=False,
        run_pqc=False,
        run_whitenoise=True,
        run_fix_dataset=False,
        make_plots=False,
        make_reports=False,
        make_covmat=False,
        make_binary_analysis=False,
        qc_report=False,
        consolidated_report=False,
        outdir_name="wn_snapshot_test",
    )

    out = run_pipeline(cfg)

    summary = out["whitenoise"] / "scanbranch" / "whitenoise_summary.tsv"
    assert summary.exists()
    assert "F0 9.0 1" in par_path.read_text(encoding="utf-8")
    assert seen["par_path"].endswith("/dataset/J0000+0000/J0000+0000.par")
