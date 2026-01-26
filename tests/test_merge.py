"""Tests for time/metadata merging logic."""

import pandas as pd
from pta_qc.io.merge import merge_time_and_meta

def test_merge_asof_with_tolerance():
    df_time = pd.DataFrame({
        "mjd": [58000.00000, 58001.00000],
        "resid": [0.0, 1.0],
        "sigma": [1.0, 1.0],
        "freq": [1400.0, 1400.0],
        "day": [58000, 58001],
    })
    df_meta = pd.DataFrame({
        "mjd": [58000.00001, 58001.00002],
        "filename": ["a", "b"],
        "freq": [1400.0, 1400.0],
        "toaerr_tim": [1.0, 1.0],
        "tel": ["X", "X"],
        "_timfile": ["NRT.BON.1400.tim", "NRT.BON.1400.tim"],
        "_time_offset_sec": [0.0, 0.0],
    })
    merged = merge_time_and_meta(df_time, df_meta, tol_days=5/86400.0)
    assert merged["filename"].isna().sum() == 0
