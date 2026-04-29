from __future__ import annotations

import json
from pathlib import Path

from pleb.reference_dataset_tools import (
    stage_expected_outputs,
    stage_reference_input,
    write_reference_manifest,
)


def test_stage_reference_input_copies_requested_pulsars_and_extra_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "J0001+0001").mkdir(parents=True)
    (source / "J0001+0001" / "J0001+0001.par").write_text("PSR J0001+0001\n", encoding="utf-8")
    (source / "J0001+0001" / "J0001+0001_all.tim").write_text("FORMAT 1\n", encoding="utf-8")
    (source / "shared").mkdir()
    (source / "shared" / "note.txt").write_text("hello\n", encoding="utf-8")

    dest = tmp_path / "reference" / "input"
    copied = stage_reference_input(
        source,
        dest,
        pulsars=["J0001+0001"],
        extra_paths=["shared/note.txt"],
    )

    assert len(copied) == 3
    assert (dest / "J0001+0001" / "J0001+0001.par").read_text(encoding="utf-8") == "PSR J0001+0001\n"
    assert (dest / "shared" / "note.txt").read_text(encoding="utf-8") == "hello\n"


def test_stage_expected_outputs_preserves_relative_paths(tmp_path: Path) -> None:
    source = tmp_path / "run"
    (source / "qc").mkdir(parents=True)
    (source / "qc" / "qc_summary.tsv").write_text("backend\tcount\n", encoding="utf-8")

    dest = tmp_path / "reference" / "expected"
    copied = stage_expected_outputs(source, dest, ["qc/qc_summary.tsv"])

    assert len(copied) == 1
    assert (dest / "qc" / "qc_summary.tsv").read_text(encoding="utf-8") == "backend\tcount\n"


def test_write_reference_manifest_records_input_and_expected_files(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    (reference_root / "input" / "J0001+0001").mkdir(parents=True)
    (reference_root / "input" / "J0001+0001" / "J0001+0001.par").write_text(
        "PSR J0001+0001\n", encoding="utf-8"
    )
    (reference_root / "expected" / "qc").mkdir(parents=True)
    (reference_root / "expected" / "qc" / "qc_summary.tsv").write_text(
        "backend\tcount\n", encoding="utf-8"
    )

    manifest_path = write_reference_manifest(reference_root, metadata={"source": "unit-test"})
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["metadata"] == {"source": "unit-test"}
    paths = {entry["path"] for entry in payload["files"]}
    assert paths == {
        "expected/qc/qc_summary.tsv",
        "input/J0001+0001/J0001+0001.par",
    }
    for entry in payload["files"]:
        assert len(entry["sha256"]) == 64
        assert entry["size_bytes"] > 0
