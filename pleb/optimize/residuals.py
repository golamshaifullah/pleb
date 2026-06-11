"""Residual-column selection for optimization scoring and reports."""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd


class ResidualChoice(NamedTuple):
    values: pd.Series
    column: str
    units: str
    jump_corrected: bool


_RESIDUAL_CANDIDATES: tuple[tuple[str, float, bool], ...] = (
    ("tempo2_post_us", 1.0, True),
    ("tempo2_postfit_us", 1.0, True),
    ("postfit_us", 1.0, True),
    ("post_us", 1.0, True),
    ("resid_detrended_us", 1.0, True),
    ("residual_detrended_us", 1.0, True),
    ("resid_detrended", 1.0e6, True),
    ("residual_detrended", 1.0e6, True),
    ("tempo2_post", 1.0e6, True),
    ("tempo2_postfit", 1.0e6, True),
    ("postfit", 1.0e6, True),
    ("post", 1.0e6, True),
    ("resid_us", 1.0, False),
    ("residual_us", 1.0, False),
    ("resid", 1.0e6, False),
    ("residual", 1.0e6, False),
)


def choose_residual_us(df: pd.DataFrame) -> ResidualChoice:
    """Return the best available residual series, normalized to microseconds.

    Optimisation must not silently prefer raw backend-offset residuals when a
    post-fit or detrended residual is available.  Raw ``resid`` columns are kept
    only as a compatibility fallback and marked as not jump-corrected.
    """
    for col, scale, jump_corrected in _RESIDUAL_CANDIDATES:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce") * float(scale)
        if values.notna().any():
            return ResidualChoice(
                values=values,
                column=col,
                units="us",
                jump_corrected=jump_corrected,
            )
    return ResidualChoice(
        values=pd.Series(dtype=float),
        column="",
        units="us",
        jump_corrected=False,
    )
