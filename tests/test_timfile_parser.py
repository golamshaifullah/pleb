"""Tests for timfile parsing."""

import pandas as pd
from pathlib import Path
from pta_qc.io.timfile import parse_all_timfiles

def _write(p: Path, txt: str):
    p.write_text(txt, encoding="utf-8")

def test_negative_numeric_flag_value_parses(tmp_path: Path):
    allf = tmp_path / "X_all.tim"
    inc = tmp_path / "NRT.NUPPI.1484.tim"

    _write(allf, f"INCLUDE {inc.name}\n")
    _write(
        inc,
        '''
        TIME 10
        nuppi_x.F4T 1294.31 55812.817258838058272 0.178 ncyobs -padd -0.193655 -i NUPPI -r ROACH -misc NUPPI1p4
        '''
    )

    res = parse_all_timfiles(allf)
    df = res.df
    assert len(df) == 1
    assert abs(df.loc[0, "mjd"] - (55812.817258838058272 + 10/86400.0)) < 1e-12
    assert df.loc[0, "padd"] == "-0.193655"
    assert df.loc[0, "i"] == "NUPPI"
    assert df.loc[0, "r"] == "ROACH"
    assert df.loc[0, "misc"] == "NUPPI1p4"

def test_ignores_comment_lines(tmp_path: Path):
    allf = tmp_path / "X_all.tim"
    _write(allf, '''
    C this is a comment
    # also comment
    MODE 1
    FORMAT 1
    nuppi_x.F4T 1000 58000.0 1.0 ncyobs -i NUPPI
    ''')
    res = parse_all_timfiles(allf)
    assert len(res.df) == 1
