"""Shared helpers for tempo2 .tim parsing.

This module intentionally stays dependency-light so other modules can import it
without creating circular dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set, Tuple

# Common tempo2 directive words in .tim files.
TIM_DIRECTIVES = {
    "FORMAT",
    "MODE",
    "TIME",
    "EFAC",
    "EQUAD",
    "ECORR",
    "JUMP",
    "INCLUDE",
    "SKIP",
    "TRACK",
    "PHASE",
    "FREQ",
    "SCALE",
    "T2EFAC",
    "T2EQUAD",
}


def is_toa_line(line: str) -> bool:
    """Return True if a line appears to be a TOA data line."""
    s = line.strip()
    if not s:
        return False
    if s.startswith(("C", "#")):
        return False
    head = s.split()[0]
    return head not in TIM_DIRECTIVES


def count_toa_lines(timfile: Path) -> int:
    """Count TOA-like lines in a .tim file."""
    if not timfile.exists():
        return 0
    n = 0
    for raw in timfile.read_text(encoding="utf-8", errors="ignore").splitlines():
        if is_toa_line(raw):
            n += 1
    return n


def parse_include_lines(alltim: Path) -> Set[str]:
    """Parse INCLUDE lines from an all-tim file."""
    inc: Set[str] = set()
    if not alltim.exists():
        return inc
    for raw in alltim.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("INCLUDE"):
            parts = s.split(maxsplit=1)
            if len(parts) == 2:
                inc.add(parts[1].strip())
    return inc


def cleanline(line: str) -> str:
    """Normalize line endings and trailing spaces."""
    return line.rstrip("\n").rstrip(" ")


def list_backend_timfiles(psr_dir: Path) -> List[Path]:
    """List per-backend tim files for a pulsar."""
    tims_dir = psr_dir / "tims"
    if not tims_dir.exists():
        return []
    out: List[Path] = []
    for p in sorted(tims_dir.glob("*.tim")):
        if p.name.endswith("_all.tim"):
            continue
        if p.name.startswith("."):
            continue
        out.append(p)
    return out


def extract_flag_values(timfile: Path, flag: str) -> Set[str]:
    """Collect unique values for a flag from TOA lines."""
    vals: Set[str] = set()
    if not timfile.exists():
        return vals
    for raw in timfile.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not is_toa_line(raw):
            continue
        parts = raw.split()
        for i, tok in enumerate(parts[:-1]):
            if tok == flag:
                vals.add(parts[i + 1])
    return vals


def mjd_from_toa_line(line: str, time_offset_sec: float = 0.0) -> Optional[float]:
    """Parse an MJD from a TOA line, applying TIME offset (seconds)."""
    if not is_toa_line(line):
        return None
    parts = line.strip().split()
    if len(parts) < 3:
        return None
    try:
        mjd = float(parts[2])
    except Exception:
        return None
    return mjd + (float(time_offset_sec) / 86400.0)


def toa_key_from_line(line: str) -> Optional[Tuple[str, str, str, str]]:
    """Cheap TOA identity key for FORMAT 1: first 4 columns as strings."""
    if not is_toa_line(line):
        return None
    parts = line.strip().split()
    if len(parts) < 4:
        return None
    return (parts[0], parts[1], parts[2], parts[3])
