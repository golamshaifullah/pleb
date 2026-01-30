"""Tests for bad-measurement detection logic."""

import pytest

try:
    from pqc.detect.bad_measurements import detect_bad
except Exception as e:  # pragma: no cover
    pytest.skip(f"pqc bad_measurements unavailable: {e}", allow_module_level=True)

import numpy as np
import pandas as pd


def test_bad_measurement_day_flagging():
    rng = np.random.default_rng(0)
    n = 300
    mjd = 58000.0 + np.sort(rng.random(n) * 30)
    day = np.floor(mjd).astype(int)
    sigma = np.full(n, 1.0)
    resid = rng.normal(0, 1.0, size=n)

    target_day = day[n // 2]
    idx = np.where(day == target_day)[0][0]
    resid[idx] = 12.0

    df = pd.DataFrame({"mjd": mjd, "day": day, "resid": resid, "sigma": sigma})
    out = detect_bad(df, tau_corr_days=0.02, fdr_q=0.05)

    assert out["bad"].sum() >= 1
    assert out.loc[idx, "bad"] or out.loc[out["day"] == target_day, "bad"].any()
