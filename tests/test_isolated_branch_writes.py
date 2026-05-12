from __future__ import annotations

from pathlib import Path
import subprocess

from git import Repo  # type: ignore

from pleb.config import PipelineConfig
from pleb.ingest import commit_ingest_changes
from pleb.pipeline import _apply_fixdataset_and_commit, _commit_branch_artifacts


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def test_commit_ingest_changes_keeps_main_worktree_branch(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-b", "main")
    (repo_root / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo_root, "add", "README.md")
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

    output_root = repo_root / "dataset"
    psr_dir = output_root / "J0000+0000"
    (psr_dir / "tims").mkdir(parents=True)
    (psr_dir / "J0000+0000.par").write_text("PSRJ J0000+0000\n", encoding="utf-8")
    (psr_dir / "J0000+0000_all.tim").write_text(
        "INCLUDE tims/A.tim\n", encoding="utf-8"
    )
    (psr_dir / "tims" / "A.tim").write_text("FORMAT 1\n", encoding="utf-8")

    new_branch = commit_ingest_changes(
        output_root,
        branch_name="raw_ingest",
        base_branch="main",
        commit_message="Ingest: test",
    )

    repo = Repo(str(repo_root), search_parent_directories=False)
    assert new_branch == "raw_ingest"
    assert repo.active_branch.name == "main"

    show = _git(
        repo_root,
        "show",
        "raw_ingest:dataset/J0000+0000/J0000+0000.par",
    )
    assert "PSRJ J0000+0000" in show.stdout


def test_apply_fixdataset_and_commit_keeps_main_worktree_branch(
    tmp_path: Path, monkeypatch
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

    def fake_fix_pulsar_dataset(psr_path: Path, _cfg):
        (psr_path / f"{psr_path.name}.par").write_text(
            "PSRJ J0000+0000\nF0 2.0 1\n", encoding="utf-8"
        )
        return {"psr": psr_path.name}

    monkeypatch.setattr("pleb.pipeline.fix_pulsar_dataset", fake_fix_pulsar_dataset)
    monkeypatch.setattr("pleb.pipeline.write_fix_report", lambda reports, out_dir: None)
    monkeypatch.setattr(
        "pleb.pipeline._validate_fixdataset_qc_inputs",
        lambda pulsars, cfg, *, branch: None,
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
        fix_apply=True,
    ).resolved()

    repo = Repo(str(repo_root), search_parent_directories=False)
    out_paths = {"fix_dataset": repo_root / "results" / "fix_dataset"}

    new_branch = _apply_fixdataset_and_commit(
        repo,
        cfg,
        ["J0000+0000"],
        out_paths,
        base_branch="main",
        new_branch="apply_branch",
        commit_message="FixDataset: test",
    )

    repo = Repo(str(repo_root), search_parent_directories=False)
    assert new_branch == "apply_branch"
    assert repo.active_branch.name == "main"
    assert "F0 1.0 1" in par_path.read_text(encoding="utf-8")

    show = _git(
        repo_root,
        "show",
        "apply_branch:dataset/J0000+0000/J0000+0000.par",
    )
    assert "F0 2.0 1" in show.stdout
    worktrees = _git(repo_root, "worktree", "list", "--porcelain")
    assert "refs/heads/apply_branch" not in worktrees.stdout


def test_commit_branch_artifacts_scopes_to_current_run_outputs(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-b", "main")
    (repo_root / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo_root, "add", "README.md")
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
    _git(repo_root, "branch", "apply_branch", "main")

    run_tag = repo_root / "results" / "current_run" / "apply_branch"
    run_tag.mkdir(parents=True)
    (run_tag / "run_report.pdf").write_text("current\n", encoding="utf-8")

    unrelated = repo_root / "results" / "other_run" / "other_branch" / "noise.tsv"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("unrelated\n", encoding="utf-8")

    cfg = PipelineConfig(
        home_dir=repo_root,
        singularity_image=repo_root / "tempo2.sif",
        dataset_name="dataset",
        results_dir=repo_root / "results",
        branches=["apply_branch"],
        reference_branch="apply_branch",
        pulsars=["J0000+0000"],
        jobs=1,
    ).resolved()
    repo = Repo(str(repo_root), search_parent_directories=False)

    _commit_branch_artifacts(
        repo,
        cfg,
        branch="apply_branch",
        out_paths={"base": repo_root / "results" / "current_run", "tag": run_tag},
        commit_message="Artifacts: test",
        cleanup_source_worktree=True,
    )

    show_current = _git(
        repo_root, "show", "apply_branch:results/current_run/apply_branch/run_report.pdf"
    )
    assert "current" in show_current.stdout
    assert not (run_tag / "run_report.pdf").exists()
    assert unrelated.exists()

    show_unrelated = subprocess.run(
        ["git", "show", "apply_branch:results/other_run/other_branch/noise.tsv"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert show_unrelated.returncode != 0
