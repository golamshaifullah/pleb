"""
Infer system flags for EPTA-style tempo2 FORMAT 1 `.tim` files.

Goal:

- When -sys/-group/-pta are missing (and sometimes -be missing), infer them cheaply and consistently.
- If bandwidth (-bw) and number-of-bands (-nchan/-nband) are available, assign sub-band systems by binning
  frequencies into equal-width sub-bands.
- Keep system format:  <TEL>.<BACKEND>.<CENTRE_MHZ>  (used with "-sys" flag)

Design choices (cheap + robust):

- Only TOA lines are processed; directives/comments are preserved.
- We never try to infer a header; we assume FORMAT 1 and use the 2nd column as frequency (MHz).
- We drop/ignore any TOA lines whose frequency is non-numeric.
- Backend inference:
    1) per-TOA "-be" flag if present
    2) filename stem heuristic: <TEL>.<BACKEND>....tim
    3) otherwise raise BackendMissingError with a sample TOA line for the UI to show the user
- Second pass canonicalisation across pulsars:
    Use canonicalise_centres() on a combined table of inferred centres to "snap" them across pulsars
    within a tolerance (default 1 MHz).
See Also:
    pleb.dataset_fix.infer_and_apply_system_flags: Integration point for FixDataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import re

import numpy as np
import pandas as pd

TELESCOPE_CODES = {"EFF", "JBO", "WSRT", "NRT", "SRT", "LEAP", "LOFAR"}

DEFAULT_PTA_BY_TEL = {
    "EFF": "EPTA",
    "JBO": "EPTA",
    "WSRT": "EPTA",
    "NRT": "EPTA",
    "SRT": "EPTA",
    "LEAP": "EPTA",
    "LOFAR": "EPTA",
}

# Common directive words in tempo2 tim files.
_TIM_DIRECTIVES = {
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

# Regex that finds "-key value" pairs (value = next non-space token).
_FLAG_RE = re.compile(r"(?P<k>-\w+)\s+(?P<v>[^\s]+)")


class BackendMissingError(RuntimeError):
    """Raised when a backend cannot be inferred automatically.

    Catch this in the GUI and prompt the user for the backend name.
    """

    def __init__(self, timfile: Path, sample_toa_line: str):
        super().__init__(
            f"Could not infer backend for {timfile}. "
            f"Provide a backend name; sample TOA line:\n{sample_toa_line}"
        )
        self.timfile = timfile
        self.sample_toa_line = sample_toa_line


@dataclass(frozen=True, slots=True)
class SystemInferenceConfig:
    """Configuration for system flag inference from `.tim` files.

    Attributes:
        backend_flag: Flag key used for backend inference (default: ``-be``).
        bandwidth_flags: Flag keys containing bandwidth values.
        nband_flags: Flag keys containing sub-band counts.
        canonical_tol_mhz: Frequency tolerance for snapping centers.
        round_mhz: Decimal rounding for inferred centers.
    """

    # flag keys to consult for backend/bw/nband if present on TOA lines
    backend_flag: str = "-be"
    bandwidth_flags: Tuple[str, ...] = ("-bw", "-BW", "-bandwidth", "-bwidth")
    nband_flags: Tuple[str, ...] = ("-nchan", "-nband", "-nsub", "-nch")
    # if inferred centres differ slightly across pulsars, snap within this tolerance (MHz)
    canonical_tol_mhz: float = 1.0
    # how to round centre frequencies (MHz)
    round_mhz: int = 1


def is_toa_line(raw: str) -> bool:
    """Return True if a raw line appears to be a TOA data line.

    Args:
        raw: Raw line from a `.tim` file.

    Returns:
        True if the line looks like a TOA row.
    """
    s = raw.strip()
    if not s:
        return False
    if s.startswith(("C", "#")):
        return False
    head = s.split()[0]
    return head not in _TIM_DIRECTIVES


def _extract_flags(line: str) -> Dict[str, str]:
    """Extract ``-flag value`` pairs from a TOA line."""
    # Only scan the part after the first 4 columns to reduce false positives.
    parts = line.split()
    if len(parts) <= 4:
        return {}
    tail = " ".join(parts[4:])
    return {m.group("k"): m.group("v") for m in _FLAG_RE.finditer(tail)}


def _infer_telescope_code(timfile: Path) -> Optional[str]:
    """Infer telescope code from a .tim filename."""
    stem = timfile.name
    # common: TEL.BE.xxx.tim
    toks = stem.split(".")
    if toks and toks[0] in TELESCOPE_CODES:
        return toks[0]
    return None


def _infer_backend_from_filename(timfile: Path, tel: Optional[str]) -> Optional[str]:
    """Infer backend name from a .tim filename."""
    toks = timfile.name.split(".")
    if tel and toks and toks[0] == tel and len(toks) >= 2:
        return toks[1]
    # fallback: if first token looks like a telescope anyway
    if toks and toks[0] in TELESCOPE_CODES and len(toks) >= 2:
        return toks[1]
    return None


def parse_tim_toa_table(
    timfile: Path, cfg: SystemInferenceConfig = SystemInferenceConfig()
) -> pd.DataFrame:
    """Parse TOA lines into a table.

    Returns columns:
      - line_idx: original line index in file (0-based)
      - line: raw line (stripped)
      - freq_mhz: float (TOA frequency)
      - flags: dict of parsed flags for that TOA line
      - be: backend if present on that TOA line via cfg.backend_flag
      - bw_mhz: bandwidth value if present
      - nband: number of bands if present

    Args:
        timfile: Path to a `.tim` file.
        cfg: System inference configuration.

    Returns:
        DataFrame with TOA metadata extracted from the `.tim` file.

    Examples:
        Parse TOA metadata::

            df = parse_tim_toa_table(Path("EFF.P200.1380.tim"))
    """
    lines = timfile.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = []
    for i, raw in enumerate(lines):
        if not is_toa_line(raw):
            continue
        line = raw.strip()
        parts = line.split()
        if len(parts) < 4:
            continue
        # FORMAT 1: second column is frequency
        try:
            freq = float(parts[1])
        except Exception:
            continue

        flags = _extract_flags(line)
        be = flags.get(cfg.backend_flag)

        bw = None
        for k in cfg.bandwidth_flags:
            if k in flags:
                try:
                    bw = float(flags[k])
                except Exception:
                    bw = None
                break

        nb = None
        for k in cfg.nband_flags:
            if k in flags:
                try:
                    nb = int(float(flags[k]))
                except Exception:
                    nb = None
                break

        rows.append((i, line, freq, flags, be, bw, nb))

    return pd.DataFrame(
        rows, columns=["line_idx", "line", "freq_mhz", "flags", "be", "bw_mhz", "nband"]
    )


def infer_backend(
    timfile: Path,
    df: pd.DataFrame,
    cfg: SystemInferenceConfig = SystemInferenceConfig(),
    override_backend: Optional[str] = None,
) -> str:
    """Infer backend name for a tim file.

    Args:
        timfile: Path to the tim file.
        df: Parsed TOA table from :func:`parse_tim_toa_table`.
        cfg: System inference configuration.
        override_backend: Optional explicit backend name.

    Returns:
        Backend name.

    Raises:
        BackendMissingError: If the backend cannot be inferred.
    """
    if override_backend:
        return override_backend

    # 1) from -be if present anywhere
    if "be" in df.columns:
        uniq = [x for x in df["be"].dropna().unique().tolist() if str(x).strip()]
        if len(uniq) == 1:
            return str(uniq[0])
        if len(uniq) > 1:
            # If multiple, keep as-is: backend varies => treat each distinct as a backend group.
            # Downstream caller can handle grouping.
            return str(uniq[0])

    tel = _infer_telescope_code(timfile)
    be2 = _infer_backend_from_filename(timfile, tel)
    if be2:
        return be2

    # 3) can't infer -> raise with a sample TOA line
    sample = df["line"].iloc[0] if len(df) else "(no TOA lines found)"
    raise BackendMissingError(timfile, sample)


def infer_subband_centres(
    freqs_mhz: np.ndarray,
    bw_mhz: float,
    nband: int,
    round_mhz: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each frequency to a subband index and center.

    Uses:
        sub_bw = bw/nband
        f_lo = min(freqs) - sub_bw/2
        idx = floor((freq - f_lo)/sub_bw), clipped to [0, nband-1]
        centre = f_lo + (idx+0.5)*sub_bw

    Args:
        freqs_mhz: Frequency array in MHz.
        bw_mhz: Total bandwidth in MHz.
        nband: Number of subbands.
        round_mhz: Decimal rounding for centers.

    Returns:
        Tuple of (subband_index, centre_mhz_rounded).

    Raises:
        ValueError: If ``bw_mhz`` or ``nband`` are non-positive.
    """
    if nband <= 0 or bw_mhz <= 0:
        raise ValueError("bw_mhz and nband must be positive")

    sub_bw = bw_mhz / float(nband)
    f_lo = float(np.nanmin(freqs_mhz)) - 0.5 * sub_bw

    idx = np.floor((freqs_mhz - f_lo) / sub_bw).astype(int)
    idx = np.clip(idx, 0, nband - 1)

    centre = f_lo + (idx + 0.5) * sub_bw
    centre = (
        np.rint(centre).astype(int) if round_mhz == 1 else np.round(centre, round_mhz)
    )
    return idx, centre


def infer_sys_group_pta(
    timfile: Path,
    cfg: SystemInferenceConfig = SystemInferenceConfig(),
    override_backend: Optional[str] = None,
    override_telescope: Optional[str] = None,
    override_pta: Optional[str] = None,
) -> pd.DataFrame:
    """Infer -sys/-group/-pta values for each TOA row.

    Args:
        timfile: Path to the tim file.
        cfg: System inference configuration.
        override_backend: Optional backend override.
        override_telescope: Optional telescope override.
        override_pta: Optional PTA override.

    Returns:
        DataFrame with columns: ``line_idx``, ``sys``, ``group``, ``pta``,
        ``backend``, ``tel``, ``centre_mhz``, ``bw_mhz``, ``nband``.

    Examples:
        Infer flags for a tim file::

            df = infer_sys_group_pta(Path("EFF.P200.1380.tim"))
    """
    df = parse_tim_toa_table(timfile, cfg=cfg)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "line_idx",
                "sys",
                "group",
                "pta",
                "backend",
                "tel",
                "centre_mhz",
                "bw_mhz",
                "nband",
            ]
        )

    tel = override_telescope or _infer_telescope_code(timfile) or "UNKNOWN"
    backend = infer_backend(timfile, df, cfg=cfg, override_backend=override_backend)

    pta = override_pta or DEFAULT_PTA_BY_TEL.get(tel, "EPTA")

    # If bandwidth/nband available (prefer scalar per file), use binning.
    bw_vals = df["bw_mhz"].dropna().unique()
    nb_vals = df["nband"].dropna().unique()

    use_binning = (
        len(bw_vals) == 1
        and len(nb_vals) == 1
        and float(bw_vals[0]) > 0
        and int(nb_vals[0]) > 0
    )

    if use_binning:
        bw = float(bw_vals[0])
        nb = int(nb_vals[0])

        _, centre = infer_subband_centres(
            df["freq_mhz"].to_numpy(), bw, nb, round_mhz=cfg.round_mhz
        )
        centre = centre.astype(int)
    else:
        # No bw/nband -> cheap fallback: centre = rounded TOA frequency itself.
        bw = float(bw_vals[0]) if len(bw_vals) == 1 else np.nan
        nb = int(nb_vals[0]) if len(nb_vals) == 1 else np.nan
        centre = np.rint(df["freq_mhz"].to_numpy()).astype(int)

    # Vectorised string build
    centre_s = pd.Series(centre, index=df.index, dtype="int64").astype(str)
    sys_val = tel + "." + backend + "." + centre_s
    group_val = (
        sys_val  # cheap: keep group identical unless you have a receiver naming rule
    )
    pta_val = pd.Series([pta] * len(df), index=df.index)

    out = pd.DataFrame(
        {
            "line_idx": df["line_idx"].to_numpy(),
            "sys": sys_val.to_numpy(),
            "group": group_val.to_numpy(),
            "pta": pta_val.to_numpy(),
            "backend": backend,
            "tel": tel,
            "centre_mhz": centre,
            "bw_mhz": bw,
            "nband": nb,
        }
    )
    return out


def apply_flags_to_timfile(
    timfile: Path,
    inferred: pd.DataFrame,
    apply: bool = False,
    backup: bool = True,
    dry_run: bool = False,
    overwrite_existing: bool = False,
) -> Dict[str, object]:
    """Rewrite timfile so each TOA line has -sys/-group/-pta.

    Args:
        timfile: Path to tim file to update.
        inferred: Inferred flags from :func:`infer_sys_group_pta`.
        apply: If True, write changes to disk.
        backup: If True, create a backup before writing.
        dry_run: If True, do not write but return stats.
        overwrite_existing: If True, replace existing flag values.

    Returns:
        Stats dictionary containing counts and file path.

    Examples:
        Apply inferred flags to a tim file::

            inferred = infer_sys_group_pta(Path("EFF.P200.1380.tim"))
            stats = apply_flags_to_timfile(Path("EFF.P200.1380.tim"), inferred, apply=True)
    """
    if inferred.empty:
        return {"timfile": str(timfile), "changed": False, "added": 0, "overwritten": 0}

    lines = timfile.read_text(encoding="utf-8", errors="ignore").splitlines()
    idx_to_row = {int(r.line_idx): r for r in inferred.itertuples(index=False)}

    changed = False
    added = 0
    overwritten = 0
    new_lines: List[str] = []

    def _set_flag(
        parts: List[str], flag: str, value: str
    ) -> Tuple[List[str], int, int]:
        nonlocal overwrite_existing
        if flag in parts:
            if not overwrite_existing:
                return parts, 0, 0
            i = parts.index(flag)
            if i + 1 < len(parts) and parts[i + 1] != value:
                parts[i + 1] = value
                return parts, 0, 1
            return parts, 0, 0
        parts.extend([flag, value])
        return parts, 1, 0

    for i, raw in enumerate(lines):
        if i not in idx_to_row:
            new_lines.append(raw.rstrip("\n"))
            continue
        row = idx_to_row[i]
        parts = raw.strip().split()
        before = " ".join(parts)

        parts, a1, o1 = _set_flag(parts, "-sys", str(row.sys))
        parts, a2, o2 = _set_flag(parts, "-group", str(row.group))
        parts, a3, o3 = _set_flag(parts, "-pta", str(row.pta))

        added += a1 + a2 + a3
        overwritten += o1 + o2 + o3

        after = " ".join(parts)
        if after != before:
            changed = True
        new_lines.append(after)

    if dry_run or not apply or not changed:
        return {
            "timfile": str(timfile),
            "changed": bool(changed),
            "added": added,
            "overwritten": overwritten,
        }

    if backup:
        b = timfile.with_suffix(timfile.suffix + ".orig")
        if not b.exists():
            b.write_text("\n".join(lines) + "\n", encoding="utf-8")

    timfile.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return {
        "timfile": str(timfile),
        "changed": True,
        "added": added,
        "overwritten": overwritten,
    }


def canonicalise_centres(
    assignments: pd.DataFrame, tol_mhz: float = 1.0
) -> pd.DataFrame:
    """Snap center frequencies across pulsars within a tolerance.

    Args:
        assignments: Assignment table with columns ``tel``, ``backend``,
            ``bw_mhz``, ``nband``, ``centre_mhz``, ``sys``, ``group``.
        tol_mhz: Frequency tolerance for clustering.

    Returns:
        Updated assignments with canonicalized center frequencies.

    Examples:
        Snap centers across pulsars::

            snapped = canonicalise_centres(assignments, tol_mhz=1.0)
    """
    required = {"tel", "backend", "bw_mhz", "nband", "centre_mhz", "sys", "group"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"assignments missing columns: {sorted(missing)}")

    out = assignments.copy()

    key_cols = ["tel", "backend", "bw_mhz", "nband"]
    for key, sub in out.groupby(key_cols, dropna=False):
        centres = np.sort(pd.Series(sub["centre_mhz"].astype(float)).dropna().unique())
        if centres.size == 0:
            continue

        # cluster
        clusters: List[np.ndarray] = []
        current = [centres[0]]
        for c in centres[1:]:
            if abs(c - current[-1]) <= tol_mhz:
                current.append(c)
            else:
                clusters.append(np.array(current))
                current = [c]
        clusters.append(np.array(current))

        mapping: Dict[float, int] = {}
        for cl in clusters:
            canon = int(np.rint(np.median(cl)))
            for v in cl:
                mapping[float(v)] = canon

        idx = sub.index
        new_centres = out.loc[idx, "centre_mhz"].astype(float).map(mapping).astype(int)

        # rebuild sys/group
        tel = str(key[0])
        backend = str(key[1])
        sys_new = tel + "." + backend + "." + new_centres.astype(str)
        out.loc[idx, "centre_mhz"] = new_centres
        out.loc[idx, "sys"] = sys_new
        out.loc[idx, "group"] = sys_new

    return out


def update_mapping_table(
    mapping_path: Path, inferred: pd.DataFrame
) -> Dict[str, List[str]]:
    """Persist a mapping table for timfile-name -> list of sys values.

    Args:
        mapping_path: JSON file path to write/update.
        inferred: Inferred assignments with ``timfile`` and ``sys`` columns.

    Returns:
        Mapping dictionary written to disk.

    Examples:
        Update a mapping JSON table::

            table = update_mapping_table(Path("system_flag_table.json"), inferred)
    """
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    if mapping_path.exists():
        table = json.loads(mapping_path.read_text(encoding="utf-8"))
    else:
        table = {}

    # inferred must include timfile_name + sys
    if "timfile" not in inferred.columns:
        raise ValueError(
            "inferred must have column 'timfile' with the tim filename/key"
        )

    for tname, sub in inferred.groupby("timfile"):
        sys_vals = sorted(pd.Series(sub["sys"]).dropna().unique().tolist())
        if sys_vals:
            table[tname] = sys_vals

    mapping_path.write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return table
