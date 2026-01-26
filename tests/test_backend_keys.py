"""Tests for backend key normalization utilities."""

import pandas as pd
from pta_qc.features.backend_keys import ensure_sys_group

def test_sys_group_from_timfile_name_and_freq():
    df = pd.DataFrame({
        "freq": [1294.31, 1400.1],
        "_timfile": ["NRT.NUPPI.1484.tim", "NRT.BON.1400.tim"],
        "cenfreq": [1484.0, None],
        "sys": [None, "NRT.BON.1400"],
        "group": [None, "NRT.BON.1400"]
    })
    out = ensure_sys_group(df)

    assert out.loc[0, "sys"].startswith("NRT.NUPPI.")
    assert out.loc[0, "group"] == "NRT.NUPPI.1484"
    assert out.loc[1, "sys"] == "NRT.BON.1400"
    assert out.loc[1, "group"] == "NRT.BON.1400"
