"""Tests for PLK log parsing."""

from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd

from pleb.parsers import read_plklog
from pleb.reports import write_change_reports


def _write_plk(path: Path, header: str, rows: list[str]) -> None:
    path.write_text("\n".join([header] + rows) + "\n", encoding="utf-8")


def test_read_plklog_header_variants(tmp_path: Path) -> None:
    plk = tmp_path / "test_plk.log"
    header = "PARAMETER   Pre-fit   Post-fit   Uncertainty   Difference   Fit"
    rows = [
        "F0 1 2 3 4 Y",
        "F1 -1 -2 0.1 0.0 N",
    ]
    _write_plk(plk, header, rows)
    df = read_plklog(plk)
    assert list(df.columns) == [
        "Param",
        "Prefit",
        "Postfit",
        "Uncertainty",
        "Difference",
        "Fit",
    ]
    assert len(df) == 2


def test_write_change_reports_skips_unparseable(tmp_path: Path, caplog) -> None:
    out_paths = {
        "plk": tmp_path / "plk",
        "change_report": tmp_path / "change_report",
    }
    out_paths["plk"].mkdir(parents=True, exist_ok=True)
    out_paths["change_report"].mkdir(parents=True, exist_ok=True)

    ref = "main"
    branch = "b1"
    pulsars = ["J0000+0000", "J0001+0001"]

    # Parseable logs for J0000+0000
    header = "Param   Prefit   Postfit   Uncertainty   Difference   Fit"
    _write_plk(
        out_paths["plk"] / f"{pulsars[0]}_{ref}_plk.log", header, ["F0 1 2 3 4 Y"]
    )
    _write_plk(
        out_paths["plk"] / f"{pulsars[0]}_{branch}_plk.log", header, ["F0 1 2 3 4 Y"]
    )

    # Unparseable logs for J0001+0001 (empty)
    (out_paths["plk"] / f"{pulsars[1]}_{ref}_plk.log").write_text("", encoding="utf-8")
    (out_paths["plk"] / f"{pulsars[1]}_{branch}_plk.log").write_text(
        "", encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="pleb.reports"):
        write_change_reports(out_paths, pulsars, [ref, branch], ref)

    assert any("Skipping change report" in rec.message for rec in caplog.records)
    # Ensure change report exists for the parseable pulsar
    out_file = out_paths["change_report"] / f"{pulsars[0]}_change_{ref}_to_{branch}.tsv"
    assert out_file.exists()
    assert pd.read_csv(out_file, sep="\t").shape[0] == 1
