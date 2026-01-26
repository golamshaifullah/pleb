"""Kepler/orbit helper functions.

This module is adapted from the AnalysePulsars.ipynb notebook you provided.

Notes
-----
- The original notebook mixed unit-aware calculations (astropy) and raw floats.
  Here we keep the core conversions in plain floats (tempo2-style), and expose
  the more advanced orbit solvers behind optional SciPy/Astropy imports.
- Functions raise clear errors if optional dependencies are required but missing.
"""

from __future__ import annotations
from typing import NamedTuple, Tuple
import math

import numpy as np


def true_from_eccentric(e: float, eccentric_anomaly: float) -> Tuple[float, float, float]:
    """Compute the true anomaly from the eccentric anomaly.

    Args:
        e: Orbital eccentricity.
        eccentric_anomaly: Eccentric anomaly ``E``.

    Returns:
        Tuple ``(true_anomaly, d(true)/d(e), d(true)/d(E))``.
    """
    E = eccentric_anomaly
    beta = math.sqrt(1 - e * e)
    # true anomaly
    f = math.atan2(beta * math.sin(E), math.cos(E) - e)

    # derivatives
    # df/dE
    denom = (1 - e * math.cos(E))
    fprime = beta / denom

    # df/de
    # derivative derived from atan2 form
    # x = cosE - e, y = beta sinE
    x = math.cos(E) - e
    y = beta * math.sin(E)
    dxde = -1.0
    dyde = (-e / beta) * math.sin(E) if beta != 0 else 0.0
    dfde = (x * dyde - y * dxde) / (x * x + y * y)

    return f, dfde, fprime


def eccentric_from_mean(e: float, mean_anomaly: float) -> Tuple[float, float, float]:
    """Solve Kepler's equation for eccentric anomaly.

    Args:
        e: Orbital eccentricity.
        mean_anomaly: Mean anomaly ``M``.

    Returns:
        Tuple ``(E, dE/de, dE/dM)``.
    """
    # Lazy import SciPy if available; otherwise use a simple Newton loop.
    try:
        from scipy.optimize import newton  # type: ignore
    except Exception:
        newton = None  # type: ignore

    M = mean_anomaly

    def f(E):
        return E - e * math.sin(E) - M

    def fp(E):
        return 1 - e * math.cos(E)

    if newton is not None:
        E = float(newton(f, M, fprime=fp, tol=1e-13, maxiter=50))
    else:
        # fallback Newton
        E = float(M)
        for _ in range(50):
            step = f(E) / fp(E)
            E -= step
            if abs(step) < 1e-13:
                break

    dEdM = 1.0 / fp(E)
    dEde = math.sin(E) / fp(E)
    return E, dEde, dEdM


def btx_parameters(asini: float, pb: float, eps1: float, eps2: float, tasc: float) -> Tuple[float, float, float, float, float]:
    """Convert ELL1 parameters to BTX-like values (a1, pb, e, om, t0).

    Args:
        asini: Projected semi-major axis (A1) in light-seconds.
        pb: Binary period.
        eps1: ELL1 eccentricity parameter ``e*sin(om)``.
        eps2: ELL1 eccentricity parameter ``e*cos(om)``.
        tasc: Time of ascending node.

    Returns:
        Tuple ``(a1, pb, e, om, t0)``.
    """
    e = float(np.hypot(eps1, eps2))
    om = float(np.arctan2(eps1, eps2))
    true_anomaly = -om
    eccentric_anomaly = float(
        np.arctan2(np.sqrt(1 - e * e) * np.sin(true_anomaly), e + np.cos(true_anomaly))
    )
    mean_anomaly = eccentric_anomaly - e * np.sin(eccentric_anomaly)
    t0 = float(tasc - mean_anomaly * pb / (2 * np.pi))
    return float(asini), float(pb), float(e), float(om), t0


class Kepler2DParameters(NamedTuple):
    """Parameter bundle for :func:`kepler_2d`."""
    a: float
    pb: float
    eps1: float
    eps2: float
    t0: float


def kepler_2d(params: Kepler2DParameters, t: float) -> np.ndarray:
    """Position and velocity of a particle in a Kepler orbit (2D).

    Returns a length-4 array [x, y, vx, vy].

    This is a simplified version of the notebook implementation.

    Args:
        params: Keplerian parameters.
        t: Time value for evaluation.

    Returns:
        Array ``[x, y, vx, vy]`` in orbital-plane coordinates.
    """
    a, pb, eps1, eps2, t0 = params
    e = float(np.hypot(eps1, eps2))
    om = float(np.arctan2(eps1, eps2))

    # mean anomaly
    n = 2 * np.pi / pb
    M = n * (t - t0)

    E, _, _ = eccentric_from_mean(e, M)
    f, _, _ = true_from_eccentric(e, E)

    r = a * (1 - e * np.cos(E))
    x = r * np.cos(f + om)
    y = r * np.sin(f + om)

    # velocity in orbital plane
    # standard relations
    mu = 1.0  # scaled; absolute scale depends on GM which isn't carried here
    h = math.sqrt(mu * a * (1 - e * e))
    vx = -(mu / h) * (math.sin(f + om) + e * math.sin(om))
    vy = (mu / h) * (math.cos(f + om) + e * math.cos(om))

    return np.array([x, y, vx, vy], dtype=float)
