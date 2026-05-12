"""Tests for parameter scan utilities."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd

from pleb.config import PipelineConfig
from pleb.param_scan import (
    parse_candidate_specs,
    apply_candidate_to_par_text,
    build_typical_candidates,
    run_param_scan,
)


def test_parse_candidate_specs_group_and_values() -> None:
    cands = parse_candidate_specs(["F2", "F3=0", "F2+F3", "raw:JUMP -sys P200 0 1"])
    assert len(cands) == 4
    assert cands[0].params[0][0] == "F2"
    assert cands[1].params[0] == ("F3", "0")
    assert {p[0] for p in cands[2].params} == {"F2", "F3"}
    assert cands[3].raw_lines[0].startswith("JUMP")


def test_apply_candidate_sets_fit_flag_and_inserts_missing_param() -> None:
    base = """
C example par
F0 1.0 0
F1 -1e-15
RAJ 12:00:00.0 0
""".strip() + "\n"

    cand = parse_candidate_specs(["F0+F2=0"])[0]
    out = apply_candidate_to_par_text(base, cand)
    # F0 fit flag should become 1
    assert "F0 1.0 1" in out
    # F2 should be appended as a simple scalar fitted param
    assert "F2 0 1" in out


def test_apply_candidate_preserves_inline_comment() -> None:
    base = "F0 1.0 0  # fixed\n"
    cand = parse_candidate_specs(["F0"])[0]
    out = apply_candidate_to_par_text(base, cand)
    assert "# fixed" in out
    assert "F0 1.0 1" in out


def test_build_typical_candidates_includes_px_and_ell1_derivs() -> None:
    par = """
PSRJ J1234+5678
RAJ 12:00:00.0 0
DECJ 01:00:00.0 0
BINARY ELL1
PB 10.0 1
A1 1.0 1
EPS1 0 1
EPS2 0 1
TASC 55000 1
""".strip() + "\n"
    cands = build_typical_candidates(par, {"redchisq": 1.0}, dm_redchisq_threshold=2.0)
    labels = {c.label for c in cands}
    assert "PX" in labels
    # ELL1 profile includes these
    assert "PBDOT" in labels
    assert "XDOT" in labels
    assert "EPS1DOT+EPS2DOT" in labels


def test_build_typical_candidates_dm_derivs_trigger_only_when_no_binary_and_high_chisq() -> (
    None
):
    par = """
PSRJ J1234+5678
F0 1.0 1
DM 10.0 1
""".strip() + "\n"
    cands_low = build_typical_candidates(
        par, {"redchisq": 1.1}, dm_redchisq_threshold=2.0, dm_max_order=3
    )
    labels_low = {c.label for c in cands_low}
    # Still includes PX by default (missing)
    assert "PX" in labels_low
    assert all(not lbl.startswith("DM") for lbl in labels_low)

    cands_high = build_typical_candidates(
        par, {"redchisq": 3.0}, dm_redchisq_threshold=2.0, dm_max_order=3
    )
    labels_high = {c.label for c in cands_high}
    assert "DM1" in labels_high
    assert "DM1+DM2" in labels_high
    assert "DM1+DM2+DM3" in labels_high


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


def test_run_param_scan_uses_branch_snapshot_without_checkout(
    tmp_path: Path, monkeypatch
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
    # Leave the current worktree dirty with content that differs from the scan branch.
    par_path.write_text("PSRJ J0000+0000\nF0 9.0 1\n", encoding="utf-8")

    seen_texts: list[str] = []
    seen_snapshots: list[Path] = []

    def fake_run_fit_only_plk(
        *,
        cfg,
        pulsar,
        par_host_path,
        work_dir,
        plk_out,
        fit_home_dir=None,
        fit_dataset_name=None,
    ) -> None:
        text = Path(par_host_path).read_text(encoding="utf-8")
        seen_texts.append(text)
        assert "F0 2.0 1" in text
        assert "F0 9.0 1" not in text
        assert fit_home_dir is not None
        assert fit_dataset_name is not None
        snap_psr_dir = Path(fit_home_dir) / Path(fit_dataset_name) / pulsar
        seen_snapshots.append(snap_psr_dir)
        assert (snap_psr_dir / f"{pulsar}_all.tim").exists()
        plk_out.write_text(
            "chisq = 10\nredchisq = 1\nnumber of TOAs = 5\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("pleb.param_scan._run_fit_only_plk", fake_run_fit_only_plk)

    cfg = PipelineConfig(
        home_dir=repo_root,
        singularity_image=repo_root / "tempo2.sif",
        dataset_name="dataset",
        results_dir=repo_root / "results",
        pulsars=["J0000+0000"],
        reference_branch="scanbranch",
        tempo2_native=True,
        jobs=1,
        force_rerun=True,
    )

    out = run_param_scan(
        cfg,
        branch="scanbranch",
        candidate_specs=["F1"],
        outdir_name="param_scan_unit",
    )

    combined = out["param_scan"] / "param_scan_scanbranch.tsv"
    assert combined.exists()
    df = pd.read_csv(combined, sep="\t")
    assert set(df["candidate"]) == {"BASE", "F1"}
    assert set(df["branch"]) == {"scanbranch"}
    assert seen_texts
    assert all("F0 2.0 1" in text for text in seen_texts)
    assert seen_snapshots
    expected_snapshot_root = (
        repo_root
        / "results"
        / "param_scan_unit"
        / "scanbranch"
        / "work"
        / "__branch_snapshot"
        / "scanbranch"
        / "dataset"
        / "J0000+0000"
    )
    assert all(snap == expected_snapshot_root for snap in seen_snapshots)
    assert "F0 9.0 1" in par_path.read_text(encoding="utf-8")
