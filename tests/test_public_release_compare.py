"""Tests for public release comparison utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pleb.public_release_compare import (
    _build_comparison,
    _parse_parfile,
    normalize_astrometry,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_parse_parfile_extracts_value_and_error(tmp_path: Path) -> None:
    p = tmp_path / "J0000+0000.par"
    _write(
        p,
        "\n".join(
            [
                "RAJ 12:00:00 1 0.0001",
                "DECJ +01:00:00 1 0.0002",
                "F0 100.0 1 1e-9",
                "JUMP -sys EFF.P200.1380 0 1",
            ]
        )
        + "\n",
    )
    df = _parse_parfile(p)
    assert "JUMP" not in set(df["param"].tolist())
    assert "RAJ" in set(df["param"].tolist())
    f0 = df[df["param"] == "F0"].iloc[0]
    assert str(f0["value_raw"]) == "100.0"
    assert float(f0["error_raw"]) == 1e-9


def test_normalize_astrometry_converts_to_icrs_degrees() -> None:
    df = pd.DataFrame(
        [
            {
                "provider": "TEST",
                "pulsar": "J0000+0000",
                "param": "RAJ",
                "value_raw": "12:00:00",
                "error_raw": 0.1,
                "parfile": "/tmp/a.par",
            },
            {
                "provider": "TEST",
                "pulsar": "J0000+0000",
                "param": "DECJ",
                "value_raw": "+00:00:00",
                "error_raw": 0.1,
                "parfile": "/tmp/a.par",
            },
        ]
    )
    out = normalize_astrometry(df)
    raj = out[out["param"] == "RAJ"].iloc[0]
    decj = out[out["param"] == "DECJ"].iloc[0]
    assert abs(float(raj["value_num"]) - 180.0) < 1e-8
    assert abs(float(decj["value_num"]) - 0.0) < 1e-8
    assert raj["frame_norm"] == "icrs"


def test_build_comparison_aggregates_provider_spread() -> None:
    norm = pd.DataFrame(
        [
            {
                "provider": "A",
                "pulsar": "J1",
                "param": "F0",
                "value_num": 1.0,
                "error_raw": 0.1,
            },
            {
                "provider": "B",
                "pulsar": "J1",
                "param": "F0",
                "value_num": 1.2,
                "error_raw": 0.2,
            },
        ]
    )
    cmp = _build_comparison(norm)
    row = cmp.iloc[0]
    assert int(row["n_providers"]) == 2
    assert abs(float(row["span"]) - 0.2) < 1e-12
