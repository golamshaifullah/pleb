from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import pandas as pd
from io import StringIO
import re

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

# Must match the -s string used in the tempo2 general2 call
GENERAL2_FORMAT = (
    "{sat} {bat} {clock0} {clock1} {clock2} {clock3} {clock4} {shapiro} {shapiroJ} "
    "{shapiroS} {shapiroV} {shapiroU} {shapiroN} {tropo} {roemer} {tt} {tt2tb} "
    "{earth_ssb} {earth_ssb1} {earth_ssb2} {earth_ssb3} {sun_earth1} {sun_earth2} "
    "{sun_earth3} {ism} {elev} {npulse} {clock} {ipm} {freq} {pre} {pre_phase} "
    "{post} {post_phase} {err} {solarangle} {binphase}"
)
GENERAL2_COLUMNS = re.findall(r"\{([^}]+)\}", GENERAL2_FORMAT)

def read_general2(file: Path) -> pd.DataFrame:
    start_marker = "Starting general2 plugin"
    end_marker = "Finished general2 plugin"

    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Find block bounds
    try:
        start_idx = next(i for i, ln in enumerate(lines) if start_marker in ln) + 1
        end_idx = next(i for i, ln in enumerate(lines[start_idx:], start=start_idx) if end_marker in ln)
    except StopIteration as e:
        raise ValueError(
            f"Could not find general2 markers in {file} "
            f"(need '{start_marker}' and '{end_marker}')."
        ) from e

    block = [ln.strip() for ln in lines[start_idx:end_idx] if ln.strip()]
    if not block:
        return pd.DataFrame()

    # If the first line looks like a header, drop it and use it as column names
    first = block[0].split()
    header_like = (len(first) >= 2) and all(any(ch.isalpha() for ch in tok) for tok in first)
    if header_like and ("sat" in first or "post" in first or "err" in first):
        cols = first
        block = block[1:]
        if not block:
            return pd.DataFrame(columns=cols)
    else:
        cols = GENERAL2_COLUMNS

    # Read raw table (don’t trust pandas’ header inference)
    df = pd.read_csv(
        StringIO("\n".join(block) + "\n"),
        sep=r"\s+",
        header=None,
        engine="python",
    )

    # Assign names robustly even if column counts differ slightly
    ncol = df.shape[1]
    if len(cols) < ncol:
        cols = cols + [f"col{i}" for i in range(len(cols), ncol)]
    df.columns = cols[:ncol]

    # Numeric conversion
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def read_tim_file(timfile: Path) -> pd.DataFrame:
    return read_tim_file_robust(timfile)
