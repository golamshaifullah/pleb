"""Tests for transient detection."""

import numpy as np
import pandas as pd
from pqc.detect.transients import scan_transients

def test_transient_detection_simple():
    rng = np.random.default_rng(1)
    n = 120
    t0 = 58010.0
    tau = 5.0

    mjd = np.linspace(58000.0, 58030.0, n)
    sigma = np.full(n, 0.2)

    A = 2.0
    resid = rng.normal(0, sigma, size=n)
    resid += np.where(mjd >= t0, A * np.exp(-(mjd - t0)/tau), 0.0)

    df = pd.DataFrame({"mjd": mjd, "resid": resid, "sigma": sigma, "bad": False})
    out = scan_transients(df, tau_rec_days=tau, window_mult=5, min_points=10, delta_chi2_thresh=25)

    assert out["transient_id"].max() >= 0
