"""Tests for ingest report artifacts and metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pleb.ingest import commit_ingest_changes, ingest_dataset


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_ingest_writes_summary_breakdown_and_optional_pdf(tmp_path: Path) -> None:
    src = tmp_path / "src"
    par_root = src / "pars"
    tim_root = src / "tim"
    template_root = src / "templates"

    _write(
        par_root / "J1234+5678.par",
        "\n".join(
            [
                "PSRJ J1234+5678",
                "EPHEM DE440",
                "CLK TT(BIPM2023)",
                "NE_SW 1",
                "JUMP -sys EFF",
                "JUMP -sys NRT",
                "",
            ]
        ),
    )
    _write(tim_root / "J1234+5678_eff.tim", "FORMAT 1\n")
    _write(tim_root / "J2345+6789_eff.tim", "FORMAT 1\n")
    _write(tim_root / "J1234+5678_all.tim", "INCLUDE tims/should_not_be_used.tim\n")
    _write(template_root / "J1234+5678.std", "template")
    _write(src / "clockfiles" / "time_gbt.clk", "clock")

    mapping = {
        "sources": [str(src)],
        "par_roots": [str(par_root)],
        "template_roots": [str(template_root)],
        "backends": {
            "EFF.P200.1400": {
                "root": str(tim_root),
            }
        },
    }
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    output_root = tmp_path / "dataset"
    report = ingest_dataset(
        mapping_path,
        output_root,
    )

    ingest_reports = output_root / "ingest_reports"
    manifest = ingest_reports / "ingest_manifest_tim.csv"
    breakdown = ingest_reports / "ingest_pulsar_breakdown.csv"

    assert manifest.exists()
    assert breakdown.exists()
    assert report["summary"]["n_pulsars_total"] == 2
    assert report["summary"]["n_valid_parfiles"] == 1
    assert report["summary"]["n_missing_parfiles"] == 1
    assert report["summary"]["n_timfiles_added_total"] == 2
    assert report["summary"]["n_templates_added_total"] == 1
    assert report["summary"]["n_all_tim_present"] == 2
    assert report["summary"]["n_clockfiles"] == 1

    per_pulsar = {row["pulsar"]: row for row in report["per_pulsar"]}
    assert per_pulsar["J1234+5678"]["n_jump_lines"] == 2
    assert per_pulsar["J1234+5678"]["timfiles_added"] == ["EFF.P200.1400.tim"]
    assert per_pulsar["J1234+5678"]["templates_added"] == ["J1234+5678.std"]
    assert per_pulsar["J1234+5678"]["ephem"] == "DE440"
    assert per_pulsar["J1234+5678"]["clk"] == "TT(BIPM2023)"
    assert per_pulsar["J2345+6789"]["missing_parfile"] is True
    assert per_pulsar["J2345+6789"]["missing_templates"] is True

    manifest_text = manifest.read_text(encoding="utf-8")
    breakdown_text = breakdown.read_text(encoding="utf-8")
    assert "J1234+5678, EFF.P200.1400" not in manifest_text
    assert "J1234+5678,EFF.P200.1400" in manifest_text
    assert "J1234+5678" in breakdown_text
    assert "J2345+6789" in breakdown_text

    if "pdf_report" in report:
        pdf_path = Path(str(report["pdf_report"]))
        assert pdf_path.exists()
        assert pdf_path.name == "ingest_report.pdf"


def test_ingest_verify_respects_pulsar_filter(tmp_path: Path) -> None:
    src = tmp_path / "src"
    par_root = src / "pars"
    tim_root = src / "tim"

    _write(par_root / "J1234+5678.par", "PSRJ J1234+5678\n")
    _write(tim_root / "J1234+5678_eff.tim", "FORMAT 1\n")
    _write(par_root / "J9999+0000.par", "PSRJ J9999+0000\n")
    _write(tim_root / "J9999+0000_eff.tim", "FORMAT 1\n")

    mapping = {
        "sources": [str(src)],
        "par_roots": [str(par_root)],
        "backends": {
            "EFF.P200.1400": {
                "root": str(tim_root),
            }
        },
    }
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    output_root = tmp_path / "dataset"
    stale_dir = output_root / "J9999+0000"
    stale_dir.mkdir(parents=True)
    stale_all_tim = stale_dir / "J9999+0000_all.tim"
    stale_all_tim.write_text("INCLUDE tims/not_the_new_file.tim\n", encoding="utf-8")
    (stale_dir / "tims").mkdir()
    (stale_dir / "tims" / "EFF.P200.1400.tim").write_text("FORMAT 1\n", encoding="utf-8")

    ingest_dataset(
        mapping_path,
        output_root,
        verify=True,
        pulsars=["J1234+5678"],
    )

    assert (output_root / "J1234+5678" / "J1234+5678_all.tim").exists()
    assert stale_all_tim.read_text(encoding="utf-8") == "INCLUDE tims/not_the_new_file.tim\n"


def test_commit_ingest_changes_isolated_and_scoped(tmp_path: Path) -> None:
    git = pytest.importorskip("git")  # provided by GitPython

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    repo = git.Repo.init(str(repo_root))
    if repo.head.is_valid():
        current = repo.active_branch.name
        if current != "main":
            repo.git.checkout("-b", "main")
    else:
        repo.git.checkout("-b", "main")

    output_root = repo_root / "EPTA-DR3" / "epta-dr3-data-v1_5"
    output_root.mkdir(parents=True)
    (output_root / "_campaign").mkdir()
    (repo_root / "README.md").write_text("seed\n", encoding="utf-8")
    repo.git.add("-A")
    repo.index.commit("seed")

    psr_dir = output_root / "J1234+5678"
    psr_dir.mkdir(parents=True)
    (psr_dir / "J1234+5678.par").write_text("PSRJ J1234+5678\n", encoding="utf-8")
    psr2_dir = output_root / "J9999+0000"
    psr2_dir.mkdir(parents=True)
    (psr2_dir / "J9999+0000.par").write_text("PSRJ J9999+0000\n", encoding="utf-8")
    (output_root / "clockfiles").mkdir(exist_ok=True)
    (output_root / "clockfiles" / "test.clk").write_text("clock\n", encoding="utf-8")
    (output_root / "ingest_reports").mkdir(exist_ok=True)
    (output_root / "ingest_reports" / "ingest_manifest_tim.csv").write_text(
        "pulsar,backend,src_backend,src,dst\n", encoding="utf-8"
    )
    (output_root / "ingest_reports" / "ingest_commit_manifest.json").write_text(
        json.dumps(
            {
                "paths": ["J1234+5678", "clockfiles", "ingest_reports"],
                "pulsars": ["J1234+5678"],
                "clockfiles": ["test.clk"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    new_branch = commit_ingest_changes(
        output_root,
        branch_name="raw_ingest_v1_5",
        base_branch="main",
        commit_message="Ingest: test",
    )

    repo = git.Repo(str(repo_root))
    assert new_branch == "raw_ingest_v1_5"
    assert repo.active_branch.name == "main"
    assert "raw_ingest_v1_5" in {h.name for h in repo.heads}
    assert (
        repo.git.show(
            "raw_ingest_v1_5:EPTA-DR3/epta-dr3-data-v1_5/J1234+5678/J1234+5678.par"
        ).strip()
        == "PSRJ J1234+5678"
    )
    assert (
        repo.git.show(
            "raw_ingest_v1_5:EPTA-DR3/epta-dr3-data-v1_5/clockfiles/test.clk"
        ).strip()
        == "clock"
    )
    with pytest.raises(Exception):
        repo.git.show(
            "raw_ingest_v1_5:EPTA-DR3/epta-dr3-data-v1_5/J9999+0000/J9999+0000.par"
        )
