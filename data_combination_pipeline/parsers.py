from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import pandas as pd
from io import StringIO


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

from io import StringIO  # make sure this import exists near the top


def read_general2(file: Path) -> pd.DataFrame:
    """
    Robust general2 reader.

    Supports:
      - header present (any header line with alphabetic tokens)
      - headerless numeric-only files
      - leading comment/blank lines

    For headerless files, creates minimal canonical columns:
      - sat = first column
      - err = last column
      - post = second-last column (if present)
    """
    raw_lines = file.read_text().splitlines()
    if not raw_lines:
        raise ValueError(f"general2 file is empty: {file}")

    def is_comment(ln: str) -> bool:
        s = ln.strip()
        return (not s) or s.startswith("#") or s.startswith("//") or s.startswith("C ")

    # Remove comment/blank lines
    lines = [ln.strip() for ln in raw_lines if not is_comment(ln)]
    if not lines:
        raise ValueError(f"general2 file has no data rows after comments: {file}")

    # Find a header-like line: contains any alphabetic tokens
    header_idx = None
    for i, ln in enumerate(lines):
        toks = ln.split()
        if any(any(ch.isalpha() for ch in tok) for tok in toks):
            header_idx = i
            break

    if header_idx is not None:
        header = lines[header_idx].split()
        data_lines = lines[header_idx + 1 :]
        if not data_lines:
            raise ValueError(f"general2 header found but no data rows: {file}")
        df = pd.read_csv(
            StringIO("\n".join(data_lines) + "\n"),
            sep=r"\s+",
            names=header,
            engine="python",
        )
    else:
        # Headerless numeric-only
        df = pd.read_csv(
            StringIO("\n".join(lines) + "\n"),
            sep=r"\s+",
            header=None,
            engine="python",
        )
        ncol = df.shape[1]
        rename = {}
        if ncol >= 1:
            rename[0] = "sat"
        if ncol >= 2:
            rename[ncol - 1] = "err"
        if ncol >= 3:
            rename[ncol - 2] = "post"
        df = df.rename(columns=rename)

    # Numeric conversion that works on pandas>=2.2 (no errors="ignore")
    def to_numeric_maybe(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s
        converted = pd.to_numeric(s, errors="coerce")
        # if conversion yields all NaN but original had data, keep original
        if converted.notna().sum() == 0 and s.notna().sum() > 0:
            return s
        return converted

    for col in df.columns:
        df[col] = to_numeric_maybe(df[col])

    return df

def read_general2(file: Path) -> pd.DataFrame:
    """
    Robust general2 reader.

    Supports:
      - header present (any header line with alphabetic tokens)
      - headerless numeric-only files
      - leading comment/blank lines

    For headerless files, creates minimal canonical columns:
      - sat = first column
      - err = last column
      - post = second-last column (if present)
    """
    raw_lines = file.read_text().splitlines()
    if not raw_lines:
        raise ValueError(f"general2 file is empty: {file}")

    def is_comment(ln: str) -> bool:
        s = ln.strip()
        return (not s) or s.startswith("#") or s.startswith("//") or s.startswith("C ")

    # Remove comment/blank lines
    lines = [ln.strip() for ln in raw_lines if not is_comment(ln)]
    if not lines:
        raise ValueError(f"general2 file has no data rows after comments: {file}")

    # Find a header-like line: contains any alphabetic tokens
    header_idx = None
    for i, ln in enumerate(lines):
        toks = ln.split()
        if any(any(ch.isalpha() for ch in tok) for tok in toks):
            header_idx = i
            break

    if header_idx is not None:
        header = lines[header_idx].split()
        data_lines = lines[header_idx + 1 :]
        if not data_lines:
            raise ValueError(f"general2 header found but no data rows: {file}")
        df = pd.read_csv(
            StringIO("\n".join(data_lines) + "\n"),
            sep=r"\s+",
            names=header,
            engine="python",
        )
    else:
        # Headerless numeric-only
        df = pd.read_csv(
            StringIO("\n".join(lines) + "\n"),
            sep=r"\s+",
            header=None,
            engine="python",
        )
        ncol = df.shape[1]
        rename = {}
        if ncol >= 1:
            rename[0] = "sat"
        if ncol >= 2:
            rename[ncol - 1] = "err"
        if ncol >= 3:
            rename[ncol - 2] = "post"
        df = df.rename(columns=rename)

    # Numeric conversion that works on pandas>=2.2 (no errors="ignore")
    def to_numeric_maybe(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s
        converted = pd.to_numeric(s, errors="coerce")
        # if conversion yields all NaN but original had data, keep original
        if converted.notna().sum() == 0 and s.notna().sum() > 0:
            return s
        return converted

    for col in df.columns:
        df[col] = to_numeric_maybe(df[col])

    return df

def read_tim_file(timfile: Path) -> pd.DataFrame:
    rows = []
    with open(timfile, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("C", "#")):
                continue
            if line.startswith(("FORMAT", "MODE", "TIME", "EFAC", "EQUAD", "JUMP", "INCLUDE", "SKIP", "TRACK")):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            rows.append(parts)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in range(min(4, df.shape[1])):
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df
