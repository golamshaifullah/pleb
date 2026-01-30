"""Parse tempo2 outputs and pipeline text formats.

This module provides small, resilient parsers for tempo2 logs and output
artifacts such as ``plk`` logs, covariance matrices, and general2 output.

See Also:
    pleb.reports: Utilities that consume parsed outputs.
    pleb.tim_reader.read_tim_file_robust: Robust `.tim` reader used here.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from io import StringIO
import re

import numpy as np
import pandas as pd

from .tim_reader import read_tim_file_robust

class PlkParseError(ValueError):
    """Raised when a tempo2 plk log cannot be parsed."""

    def __init__(self, path: Path, reason: str) -> None:
        """Initialize the error with a path and reason."""
        super().__init__(f"{reason} in {path}")
        self.path = path
        self.reason = reason

def _find_header_line(lines: List[str], startswith: str) -> Optional[int]:
    """Return the line index whose stripped content starts with ``startswith``."""
    for i, line in enumerate(lines):
        if line.strip().startswith(startswith):
            return i
    return None

def read_plklog(file: Path) -> pd.DataFrame:
    """Parse a tempo2 ``plk`` log into a parameter table.

    Args:
        file: Path to a ``*_plk.log`` file.

    Returns:
        DataFrame with columns: ``Param``, ``Prefit``, ``Postfit``,
        ``Uncertainty``, ``Difference``, ``Fit``.

    Raises:
        FileNotFoundError: If the log file does not exist.
        PlkParseError: If the log is empty or the table cannot be parsed.

    Examples:
        Parse a tempo2 plk log::

            df = read_plklog(Path("J1234+5678_plk.log"))
    """
    if not file.exists():
        raise FileNotFoundError(str(file))

    text = file.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        raise PlkParseError(file, "Empty plk log")

    lines = text.splitlines(True)

    header_i = _find_header_line(lines, "Param")
    if header_i is None:
        header_i = next((i for i, l in enumerate(lines) if "Param" in l and "Postfit" in l), None)
    if header_i is None:
        param_rx = re.compile(r"\bparam(?:eter)?\b", re.IGNORECASE)
        pre_rx = re.compile(r"\bpre[- ]?fit\b", re.IGNORECASE)
        post_rx = re.compile(r"\bpost[- ]?fit\b", re.IGNORECASE)
        for i, l in enumerate(lines):
            s = l.strip()
            if not s:
                continue
            if param_rx.search(s) and pre_rx.search(s) and post_rx.search(s):
                header_i = i
                break
    if header_i is None:
        raise PlkParseError(file, "Could not find plk table header")

    rows = []
    for line in lines[header_i + 1:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("---") or s.lower().startswith("finishing"):
            continue
        parts = s.split()
        if len(parts) < 6:
            continue
        rows.append(parts[:6])

    if not rows:
        raise PlkParseError(file, "Plk table header found but no data rows parsed")

    df = pd.DataFrame(rows, columns=["Param", "Prefit", "Postfit", "Uncertainty", "Difference", "Fit"])
    return df

def read_covmat(file: Path) -> pd.DataFrame:
    """Parse a tempo2 covariance matrix text file.

    Args:
        file: Path to a covariance matrix file.

    Returns:
        DataFrame indexed by parameter name with numeric values.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the header or table cannot be parsed.
    """
    if not file.exists():
        raise FileNotFoundError(str(file))

    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    header_i = _find_header_line(lines, "Param")
    if header_i is None:
        header_i = next((i for i, l in enumerate(lines) if l.strip().startswith("Param")), None)
    if header_i is None:
        raise ValueError(f"Could not find covmat header in {file}")

    data = "".join(lines[header_i:])
    from io import StringIO
    df = pd.read_csv(StringIO(data), sep=r"\s+", engine="python")
    if "Param" not in df.columns:
        raise ValueError(f"Could not parse covmat into a table in {file}")
    df = df.set_index("Param")
    if "Finishing" in df.index:
        df = df.drop(index="Finishing")
    return df.apply(pd.to_numeric, errors="coerce")

GENERAL2_FORMAT = (
    "{sat} {bat} {clock0} {clock1} {clock2} {clock3} {clock4} {shapiro} {shapiroJ} "
    "{shapiroS} {shapiroV} {shapiroU} {shapiroN} {tropo} {roemer} {tt} {tt2tb} "
    "{earth_ssb} {earth_ssb1} {earth_ssb2} {earth_ssb3} {sun_earth1} {sun_earth2} "
    "{sun_earth3} {ism} {elev} {npulse} {clock} {ipm} {freq} {pre} {pre_phase} "
    "{post} {post_phase} {err} {solarangle} {binphase}"
)
GENERAL2_COLUMNS = re.findall(r"\{([^}]+)\}", GENERAL2_FORMAT)

def _is_number(tok: str) -> bool:
    """Return True if token can be parsed as a float."""
    try:
        float(tok)
        return True
    except Exception:
        return False

def read_general2(file: Path) -> pd.DataFrame:
    """Parse tempo2 general2 plugin output embedded in a log file.

    Args:
        file: Path to a log containing general2 plugin output.

    Returns:
        DataFrame of parsed general2 rows. Empty if no data rows exist.

    Raises:
        ValueError: If general2 start/end markers are missing.

    Examples:
        Parse a general2 log file::

            df = read_general2(Path("J1234+5678_general2.log"))
    """
    start_marker = "Starting general2 plugin"
    end_marker = "Finished general2 plugin"
    n = len(GENERAL2_COLUMNS)

    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if start_marker in ln) + 1
        end = next(i for i, ln in enumerate(lines[start:], start=start) if end_marker in ln)
    except StopIteration as e:
        raise ValueError(f"Missing general2 markers in {file}") from e

    block = [ln.strip() for ln in lines[start:end] if ln.strip()]
    if not block:
        return pd.DataFrame(columns=GENERAL2_COLUMNS)

    rows: list[list[float]] = []
    columns = GENERAL2_COLUMNS

    # Detect optional header line like: "sat post err"
    header_tokens = block[0].split()
    if header_tokens and not all(_is_number(tok) for tok in header_tokens):
        # Accept headers that are a subset of known GENERAL2 columns
        if all(tok in GENERAL2_COLUMNS for tok in header_tokens):
            columns = header_tokens
            block = block[1:]

    for ln in block:
        parts = ln.split()

        # Must have at least as many fields as we expect; skip junk lines like "-nan"
        if len(parts) < len(columns):
            continue

        # Take only the expected fields; ignore any trailing "ERROR: ..." etc
        parts = parts[:len(columns)]

        # Convert to float; if any token isn't numeric, skip the line
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue

        rows.append(vals)

    df = pd.DataFrame(rows, columns=columns).astype("float64")

    # Convenience integer day bucket for joins/grouping downstream
    df["mjd_int"] = np.floor(df["sat"]).astype("Int64")

    return df

def read_tim_file(timfile: Path) -> pd.DataFrame:
    """Read a tempo2 `.tim` file using the robust reader.

    Args:
        timfile: Path to a `.tim` file.

    Returns:
        DataFrame of parsed TOA rows.
    """
    return read_tim_file_robust(timfile)
