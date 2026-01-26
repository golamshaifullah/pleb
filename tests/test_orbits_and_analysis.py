"""Tests for orbital utilities and binary analysis."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np

from pleb.kepler_orbits import (
    eccentric_from_mean,
    true_from_eccentric,
    btx_parameters,
    kepler_2d,
    Kepler2DParameters,
)
from pleb.pulsar_analysis import read_parfile, analyse_binary_from_par
from pleb.tempo2 import build_singularity_prefix, tempo2_paths_in_container


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_eccentric_from_mean_solves_kepler() -> None:
    e = 0.3
    M = 1.0
    E, dEde, dEdM = eccentric_from_mean(e, M)
    # Kepler equation
    assert abs(E - e * math.sin(E) - M) < 1e-10
    assert np.isfinite(dEde)
    assert np.isfinite(dEdM)


def test_true_from_eccentric_basic() -> None:
    e = 0.1
    E = 0.2
    f, dfde, dfdE = true_from_eccentric(e, E)
    assert np.isfinite(f)
    assert np.isfinite(dfde)
    assert np.isfinite(dfdE)


def test_btx_parameters_sanity() -> None:
    asini, pb, eps1, eps2, tasc = 1.0, 10.0, 0.0, 0.1, 50000.0
    a1, pb2, e, om, t0 = btx_parameters(asini, pb, eps1, eps2, tasc)
    assert a1 == asini
    assert pb2 == pb
    assert math.isclose(e, 0.1, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(om, 0.0, rel_tol=0, abs_tol=1e-12)
    # for om=0 -> mean anomaly at ascending node is 0 -> t0==tasc
    assert math.isclose(t0, tasc, rel_tol=0, abs_tol=1e-10)


def test_kepler_2d_returns_finite_state() -> None:
    params = Kepler2DParameters(a=1.0, pb=10.0, eps1=0.0, eps2=0.1, t0=50000.0)
    st = kepler_2d(params, t=50001.0)
    assert st.shape == (4,)
    assert np.isfinite(st).all()


def test_parfile_reader_and_binary_analysis(tmp_path: Path) -> None:
    par = tmp_path /  "test_dataset/J0000+0000.par"
    _write(
        par,
        """
BINARY ELL1
A1 1.0
PB 10.0
EPS1 0.0
EPS2 0.1
TASC 50000.0
""",
    )

    d = read_parfile(par)
    assert d["BINARY"] == "ELL1"

    out = analyse_binary_from_par(par)
    assert out["BINARY"] == "ELL1"
    assert math.isclose(float(out["ELL1_e"]), 0.1, abs_tol=1e-12)
    assert math.isclose(float(out["ELL1_t0"]), 50000.0, abs_tol=1e-10)


def test_tempo2_command_helpers(tmp_path: Path) -> None:
    prefix = build_singularity_prefix(tmp_path / "repo", "test_dataset", tmp_path / "tempo2.sif")
    assert prefix[:2] == ["singularity", "exec"]
    assert "/data" in " ".join(prefix)

    par, tim = tempo2_paths_in_container("J0000+0000")
    assert par.endswith("/data/J0000+0000/J0000+0000.par")
    assert tim.endswith("/data/J0000+0000/J0000+0000_all.tim")
