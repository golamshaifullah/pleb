from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import re

import pandas as pd


def _to_numeric_maybe(s: pd.Series) -> pd.Series:
    """
    Best-effort numeric conversion compatible with pandas>=2.2.

    - convert where possible (errors='coerce')
    - if conversion yields all-NaN but original had values, keep original
    """
    if pd.api.types.is_numeric_dtype(s):
        return s
    converted = pd.to_numeric(s, errors="coerce")
    if converted.notna().sum() == 0 and s.notna().sum() > 0:
        return s
    return converted


_BAD_STARTS = (
    "TIME", "MODE", "FORMAT",
    "Er", "END", "end",
    "C", "CC", "#",
)

_BAD_CONTAINS = (
    "below", "SKIP", "EFAC",
)

# Lines starting with 'c ' are comments in many files, but some data lines can start with c0..c9
# (your notebook explicitly kept those). We implement the same rule:
_c_digit = re.compile(r"^c\d\b", re.IGNORECASE)


def _is_skippable_tim_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True

    # comments / common junk
    if s.startswith(("#",)):
        return True

    # C / c comment lines, but keep c0..c9 style data rows
    if s.startswith(("C ", "c ", "C\t", "c\t", "C\\ ", "c\\ ")):
        return True
    if s.lower().startswith("c") and not _c_digit.match(s):
        # catches "c ..." and "c\\ " etc without killing c0..c9
        return True

    # TIME / MODE / FORMAT etc
    for p in _BAD_STARTS:
        if s.startswith(p):
            return True
        if s.startswith(" " + p):
            return True

    # other patterns anywhere in line
    for frag in _BAD_CONTAINS:
        if frag in s:
            return True

    return False


def _normalize_tim_line(line: str) -> str:
    # strip newline + trailing spaces
    line = line.rstrip("\n").rstrip(" ")

    # normalize tabs and repeated whitespace
    line = line.replace("\t", " ")
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _fix_padd_continuations(lines: List[str]) -> List[str]:
    """
    Merge '-padd ...' lines that have been broken onto their own line.

    Common failure mode:
        <toa line ...>
        -padd  12.3

    We join "-padd ..." onto the previous non-skipped line.
    """
    out: List[str] = []
    prev_kept_idx: Optional[int] = None

    for raw in lines:
        s = raw.strip()
        if not s:
            out.append(raw)
            continue

        if s.startswith("-padd"):
            # attach to previous kept line if possible
            if prev_kept_idx is not None:
                out[prev_kept_idx] = (out[prev_kept_idx].rstrip() + " " + s).strip()
            else:
                # nothing to attach to; keep it (it'll likely be skipped/filtered later)
                out.append(raw)
            continue

        out.append(raw)

        # track previous "kept-ish" line for attachment; don't attach to obvious junk
        if not _is_skippable_tim_line(raw):
            prev_kept_idx = len(out) - 1

    return out


def read_tim_file_robust(
    timfile: Path,
    *,
    prune_by_mjd: bool = True,
    mjd_min: float = 10_000.0,
) -> pd.DataFrame:
    """
    Robust tim reader for messy, hand-edited tim files.

    Returns a DataFrame with columns 0..N-1 holding tokens.
    Numeric columns are converted where possible without destroying string columns.
    """
    timfile = Path(timfile)
    if not timfile.exists():
        return pd.DataFrame()

    # Read raw text (be forgiving with encoding)
    raw_lines = timfile.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Normalize whitespace first
    norm = [_normalize_tim_line(ln) for ln in raw_lines]

    # Fix common continuation bug: '-padd' split onto its own line
    norm = _fix_padd_continuations(norm)

    # Drop blank + skippable lines
    kept: List[Tuple[int, List[str]]] = []
    for i, ln in enumerate(norm, start=1):
        if _is_skippable_tim_line(ln):
            continue
        toks = ln.split()
        if not toks:
            continue
        kept.append((i, toks))

    if not kept:
        return pd.DataFrame()

    # Build ragged table
    max_len = max(len(toks) for _, toks in kept)
    rows: List[List[object]] = []
    for lineno, toks in kept:
        row: List[object] = list(toks) + [pd.NA] * (max_len - len(toks))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Convert columns safely (don’t nuke filename/site columns)
    for col in df.columns:
        df[col] = _to_numeric_maybe(df[col])

    # Heuristic: identify MJD column and prune non-TOA rows
    if prune_by_mjd and df.shape[1] >= 3:
        mjd_col = _guess_mjd_column(df)
        if mjd_col is not None:
            mjd = pd.to_numeric(df[mjd_col], errors="coerce")
            df = df.loc[mjd.notna() & (mjd > mjd_min)].reset_index(drop=True)

    return df


def _guess_mjd_column(df: pd.DataFrame) -> Optional[int]:
    """
    Find the column that most likely contains MJD.
    We look for a mostly-numeric column with median in a plausible MJD range.
    """
    best: Optional[int] = None
    best_score = -1.0

    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        frac = float(s.notna().mean())
        if frac < 0.5:
            continue
        med = float(s.dropna().median())
        # very loose MJD bounds
        if not (30_000.0 < med < 90_000.0):
            continue

        # score favors higher numeric fraction and “more MJD-like” values
        score = frac
        if score > best_score:
            best_score = score
            best = int(col)

    return best
