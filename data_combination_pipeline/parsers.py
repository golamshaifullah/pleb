from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from io import StringIO
import re

import numpy as np
import pandas as pd

from .tim_reader import read_tim_file_robust

def _find_header_line(lines: List[str], startswith: str) -> Optional[int]:
    for i, line in enumerate(lines):
        if line.strip().startswith(startswith):
            return i
    return None

def read_plklog(file: Path) -> pd.DataFrame:
    if not file.exists():
        raise FileNotFoundError(str(file))

    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    header_i = _find_header_line(lines, "Param")
    if header_i is None:
        header_i = next((i for i, l in enumerate(lines) if "Param" in l and "Postfit" in l), None)
    if header_i is None:
        raise ValueError(f"Could not find plk table header in {file}")

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

    df = pd.DataFrame(rows, columns=["Param", "Prefit", "Postfit", "Uncertainty", "Difference", "Fit"])
    return df

def read_covmat(file: Path) -> pd.DataFrame:
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

# If you truly want hard dtypes, set them here.
# Safer default: float64 everywhere (general2 is numeric-only; ints get stored as floats).
GENERAL2_DTYPES = {c: "float64" for c in GENERAL2_COLUMNS}

def read_general2(file: Path) -> pd.DataFrame:
    start_marker = "Starting general2 plugin"
    end_marker = "Finished general2 plugin"

    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if start_marker in ln) + 1
        end = next(i for i, ln in enumerate(lines[start:], start=start) if end_marker in ln)
    except StopIteration as e:
        raise ValueError(f"Missing general2 markers in {file}") from e

    block = [ln.strip() for ln in lines[start:end] if ln.strip()]
    if not block:
        return pd.DataFrame(columns=GENERAL2_COLUMNS)

    # Hard contract: every row must match the general2 -s format width
    expected = len(GENERAL2_COLUMNS)
    nfields0 = len(block[0].split())
    if nfields0 != expected:
        raise ValueError(
            f"{file}: expected {expected} fields per row from GENERAL2_FORMAT, got {nfields0} "
            f"on first data row. Check your general2 -s format vs output."
        )

    df = pd.read_csv(
        StringIO("\n".join(block) + "\n"),
        sep=r"\s+",
        header=None,
        names=GENERAL2_COLUMNS,
        dtype=GENERAL2_DTYPES,
        engine="python",
    )

    # Convenience day bucket (this is derived, so created after read)
    df["mjd_int"] = np.floor(df["sat"]).astype("Int64")
    return df

def read_tim_file(timfile: Path) -> pd.DataFrame:
    return read_tim_file_robust(timfile)
