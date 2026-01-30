"""FixDataset utilities for cleaning and normalizing EPTA datasets.

This module provides lightweight, file-based cleanup utilities for `.par` and
`.tim` data, adapted from the FixDataset notebook workflow. It can run in a
report-only mode (default) or apply edits with backups.

See Also:
    pleb.config.PipelineConfig: Pipeline settings for FixDataset integration.
    pleb.pipeline.run_pipeline: Orchestrates FixDataset in the full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import re
import shutil
import json
import hashlib
import numpy as np
import pandas as pd

from .logging_utils import get_logger

logger = get_logger("pleb.dataset_fix")


# A pragmatic set of header/directive prefixes commonly seen in tempo2 .tim files.
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
}


@dataclass(slots=True)
class FixDatasetConfig:
    """Controls for dataset fixing utilities.

    These features are adapted from the FixDataset notebook. They are disabled
    by default because automatic edits can be repo-specific.

    Notes:
        Use ``apply=False`` for report-only runs. When ``apply=True``, this
        module will modify `.par`/`.tim` files and optionally create backups.

    Attributes:
        apply: Apply changes when True; otherwise run report-only.
        backup: Create ``.orig`` backups when applying.
        dry_run: Compute changes without writing to disk.
        update_alltim_includes: Update ``INCLUDE`` lines in ``*_all.tim``.
        min_toas_per_backend_tim: Minimum TOAs for a backend tim to be included.
        required_tim_flags: Flags to ensure on each TOA line.
        infer_system_flags: Infer ``-sys``/``-group``/``-pta`` flags.
        system_flag_table_path: Path to the system-flag table (JSON/TOML).
        system_flag_overwrite_existing: Overwrite existing system flags.
        backend_overrides: Map tim basename to backend name override.
        raise_on_backend_missing: Raise when backend cannot be inferred.
        dedupe_toas_within_tim: Remove duplicate TOAs within each tim.
        check_duplicate_backend_tims: Detect duplicated backend tims.
        remove_overlaps_exact: Remove known overlapping TOAs across backends.
        insert_missing_jumps: Insert missing JUMP lines into par files.
        jump_flag: Flag used to label inserted jumps.
        ensure_ephem: Ensure EPHEM param exists (optional value).
        ensure_clk: Ensure CLK param exists (optional value).
        ensure_ne_sw: Ensure NE_SW param exists (optional value).
        remove_patterns: Remove lines matching these patterns.
        coord_convert: Coordinate conversion mode (``equ2ecl`` or ``ecl2equ``).
        qc_remove_outliers: Apply/remove outliers flagged by PQC.
        qc_action: ``comment`` or ``delete`` for flagged TOAs.
        qc_comment_prefix: Prefix for commented-out TOAs.
        qc_backend_col: Backend column for matching QC results.
        qc_remove_bad: Apply bad/bad_day flags from QC.
        qc_remove_transients: Apply transient flags from QC.
        qc_remove_solar: Apply solar-elongation flags from QC.
        qc_solar_action: ``comment`` or ``delete`` for solar-flagged TOAs.
        qc_solar_comment_prefix: Prefix for solar-flagged TOA comments.
        qc_remove_orbital_phase: Apply orbital-phase flags from QC.
        qc_orbital_phase_action: ``comment`` or ``delete`` for orbital-phase TOAs.
        qc_orbital_phase_comment_prefix: Prefix for orbital-phase TOA comments.
        qc_merge_tol_days: MJD tolerance for QC matching.
        qc_results_dir: Directory containing QC CSV outputs.
        qc_branch: Subdirectory for QC results (optional).

    Examples:
        Run a report-only pass for a pulsar::

            cfg = FixDatasetConfig()
            report = fix_pulsar_dataset(Path("/data/epta/J1234+5678"), cfg)
    """

    apply: bool = False
    backup: bool = True
    dry_run: bool = False

    # all.tim maintenance
    update_alltim_includes: bool = True
    min_toas_per_backend_tim: int = 10

    # tim flag insertion (applies to per-backend tims under <psr>/tims/)
    # Example: {"-pta": "EPTA", "-be": "P200", "-sys": "SomeSys"}
    required_tim_flags: Dict[str, str] = field(default_factory=dict)

    # System flag inference (smart -sys/-group/-pta creation)
    infer_system_flags: bool = False
    system_flag_table_path: Optional[str] = None  # JSON mapping stored at dataset root if None
    system_flag_overwrite_existing: bool = False
    backend_overrides: Dict[str, str] = field(default_factory=dict)  # tim basename -> backend
    raise_on_backend_missing: bool = False

    # TIM hygiene
    dedupe_toas_within_tim: bool = False
    check_duplicate_backend_tims: bool = False

    # Overlap handling (cheap: exact TOA duplicate removal across known overlapping backends)
    remove_overlaps_exact: bool = False

    # parfile maintenance
    insert_missing_jumps: bool = True
    jump_flag: str = "-sys"
    ensure_ephem: Optional[str] = None
    ensure_clk: Optional[str] = None
    ensure_ne_sw: Optional[str] = None

    # remove known-bad backend strings (used by remove_nuppi_big in the notebook)
    remove_patterns: List[str] = field(default_factory=lambda: ["NRT.NUPPI.", "NRT.NUXPI."])

    # Coordinate conversion of par files
    # None | "equ2ecl" | "ecl2equ"
    coord_convert: Optional[str] = None

    # ---- Optional PQC-driven TOA removal/commenting ----
    qc_remove_outliers: bool = False
    qc_action: str = "comment"  # "comment" | "delete"
    qc_comment_prefix: str = "C QC_OUTLIER"
    qc_backend_col: str = "sys"
    qc_remove_bad: bool = True
    qc_remove_transients: bool = False
    qc_remove_solar: bool = False
    qc_solar_action: str = "comment"
    qc_solar_comment_prefix: str = "# QC_SOLAR"
    qc_remove_orbital_phase: bool = False
    qc_orbital_phase_action: str = "comment"
    qc_orbital_phase_comment_prefix: str = "# QC_BIANRY_ECLIPSE"
    qc_merge_tol_days: float = 2.0 / 86400.0
    qc_results_dir: Optional[Path] = None
    qc_branch: Optional[str] = None


# -----------------------------
# tim file helpers
# -----------------------------

def _cleanline(line: str) -> str:
    """Normalize line endings and trailing spaces."""
    return line.rstrip("\n").rstrip(" ")


def _is_comment_or_blank(line: str) -> bool:
    """Return True if a line is blank or a comment."""
    s = line.strip()
    return (not s) or s.startswith(("C", "#"))


def _is_directive(line: str) -> bool:
    """Return True if a line is a tempo2 directive."""
    s = line.strip()
    if not s:
        return False
    head = s.split()[0]
    return head in _TIM_DIRECTIVES


def is_toa_line(line: str) -> bool:
    """Return True if a line appears to be a TOA data line.

    Args:
        line: Raw line from a .tim file.

    Returns:
        True if the line looks like a TOA record.
    """
    if _is_comment_or_blank(line):
        return False
    if _is_directive(line):
        return False
    # Remaining lines are usually TOAs.
    return True


def count_toa_lines(timfile: Path) -> int:
    """Count TOA lines in a .tim file.

    Args:
        timfile: Path to the .tim file.

    Returns:
        Number of lines that look like TOA records.
    """
    n = 0
    if not timfile.exists():
        return 0
    for raw in timfile.read_text(encoding="utf-8", errors="ignore").splitlines():
        if is_toa_line(raw):
            n += 1
    return n


def parse_include_lines(alltim: Path) -> Set[str]:
    """Parse INCLUDE lines from an all.tim file.

    Args:
        alltim: Path to the ``<psr>_all.tim`` file.

    Returns:
        Set of included relative paths (e.g., ``"tims/foo.tim"``).
    """
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


def list_backend_timfiles(psr_dir: Path) -> List[Path]:
    """List per-backend tim files for a pulsar.

    Args:
        psr_dir: Pulsar directory containing ``tims/``.

    Returns:
        List of backend tim files under ``tims/``.
    """
    tims_dir = psr_dir / "tims"
    if not tims_dir.exists():
        return []
    out = []
    for p in sorted(tims_dir.glob("*.tim")):
        if p.name.endswith("_all.tim"):
            continue
        if p.name.startswith("."):
            continue
        out.append(p)
    return out


def _backup_file(path: Path) -> None:
    """Create a one-time backup of a file."""
    if not path.exists():
        return
    b = path.with_suffix(path.suffix + ".orig")
    if b.exists():
        return
    shutil.copy2(path, b)


def update_alltim_includes(psr_dir: Path, min_toas: int = 10, apply: bool = False, backup: bool = True, dry_run: bool = False) -> Dict[str, object]:
    """Ensure <psr>_all.tim contains INCLUDE lines for each backend tim file.

    This is a refactor of insert_missing_timfiles/update_alltims from FixDataset.ipynb.

    Args:
        psr_dir: Pulsar directory.
        min_toas: Minimum TOA count required to include a backend tim file.
        apply: If True, write changes to disk.
        backup: If True, create a backup before writing.
        dry_run: If True, do not write but return planned changes.

    Returns:
        Stats dictionary summarizing added/dropped includes.
    """
    psr = psr_dir.name
    alltim = psr_dir / f"{psr}_all.tim"
    tims = list_backend_timfiles(psr_dir)
    existing = parse_include_lines(alltim)

    to_add: List[str] = []
    dropped: List[str] = []

    for tim in tims:
        rel = f"tims/{tim.name}"
        if rel in existing:
            continue
        good = count_toa_lines(tim)
        if good >= min_toas:
            to_add.append(rel)
        else:
            dropped.append(rel)

    if not to_add:
        return {"psr": psr, "alltim": str(alltim), "added": 0, "to_add": [], "dropped": dropped}

    logger.info("%s: %d missing INCLUDE lines in %s", psr, len(to_add), alltim.name)

    if dry_run or not apply:
        return {"psr": psr, "alltim": str(alltim), "added": 0, "to_add": to_add, "dropped": dropped}

    if not alltim.exists():
        raise FileNotFoundError(str(alltim))

    if backup:
        _backup_file(alltim)

    text = alltim.read_text(encoding="utf-8", errors="ignore").splitlines()
    new_lines = list(text)
    for rel in to_add:
        new_lines.append(f"INCLUDE {rel}")

    alltim.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return {"psr": psr, "alltim": str(alltim), "added": len(to_add), "to_add": to_add, "dropped": dropped}


def extract_flag_values(timfile: Path, flag: str) -> Set[str]:
    """Collect unique values for a flag from TOA lines.

    Args:
        timfile: Path to the .tim file.
        flag: Flag key (e.g., ``"-sys"``).

    Returns:
        Set of unique flag values seen in TOA lines.
    """
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


def ensure_timfile_flags(timfile: Path, required: Dict[str, str], apply: bool = False, backup: bool = True, dry_run: bool = False) -> Dict[str, object]:
    """Insert missing flags into TOA lines.

    Conservative behavior: only adds a flag if it is missing in that TOA line.

    Args:
        timfile: Path to the .tim file.
        required: Mapping of flag keys to required values.
        apply: If True, write changes to disk.
        backup: If True, create a backup before writing.
        dry_run: If True, do not write but return planned changes.

    Returns:
        Stats dictionary summarizing changes.
    """
    if not timfile.exists():
        raise FileNotFoundError(str(timfile))

    added_counts: Dict[str, int] = {k: 0 for k in required}
    changed = False

    lines = timfile.read_text(encoding="utf-8", errors="ignore").splitlines()
    new_lines: List[str] = []

    for raw in lines:
        line = _cleanline(raw)
        if not is_toa_line(line):
            new_lines.append(line)
            continue

        parts = line.split()
        # append flags that are absent
        for flag, value in required.items():
            if flag not in parts:
                parts.extend([flag, str(value)])
                added_counts[flag] += 1
                changed = True
        new_lines.append(" ".join(parts))

    if (dry_run or not apply) or (not changed):
        return {
            "timfile": str(timfile),
            "changed": bool(changed),
            "added": {k: v for k, v in added_counts.items() if v},
        }

    if backup:
        _backup_file(timfile)
    timfile.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return {
        "timfile": str(timfile),
        "changed": True,
        "added": {k: v for k, v in added_counts.items() if v},
    }


# -----------------------------
# par file helpers
# -----------------------------

def update_parfile_jumps(
    parfile: Path,
    jump_flag: str,
    jump_values: Sequence[str],
    ensure_ephem: Optional[str] = None,
    ensure_clk: Optional[str] = None,
    ensure_ne_sw: Optional[str] = None,
    apply: bool = False,
    backup: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Ensure EPHEM/CLK/NE_SW values and append missing JUMP lines.

    Adapted from insert_missing_jumps/update_parfiles in FixDataset.ipynb.

    Args:
        parfile: Path to the .par file.
        jump_flag: Flag used for JUMP insertion (e.g., ``"-sys"``).
        jump_values: Sequence of jump flag values to ensure.
        ensure_ephem: Optional EPHEM value to enforce.
        ensure_clk: Optional CLK value to enforce.
        ensure_ne_sw: Optional NE_SW value to enforce.
        apply: If True, write changes to disk.
        backup: If True, create a backup before writing.
        dry_run: If True, do not write but return planned changes.

    Returns:
        Stats dictionary summarizing changes.
    """
    if not parfile.exists():
        raise FileNotFoundError(str(parfile))

    jump_values = [j for j in jump_values if j is not None]
    wanted: Set[str] = set(jump_values)
    found: Set[str] = set()

    lines = parfile.read_text(encoding="utf-8", errors="ignore").splitlines()
    new_lines: List[str] = []

    changed = False
    for raw in lines:
        line = _cleanline(raw)
        if not line.strip():
            new_lines.append(line)
            continue
        parts = line.split()
        key = parts[0]

        if ensure_ephem is not None and key == "EPHEM" and len(parts) > 1 and parts[1] != ensure_ephem:
            parts[1] = ensure_ephem
            changed = True
        if ensure_clk is not None and key == "CLK" and len(parts) > 1 and parts[1] != ensure_clk:
            parts[1] = ensure_clk
            changed = True
        if ensure_ne_sw is not None and key == "NE_SW" and len(parts) > 1 and parts[1] != ensure_ne_sw:
            parts[1] = ensure_ne_sw
            changed = True

        # consume already-present JUMPs
        if key == "JUMP" and len(parts) >= 3 and parts[1] == jump_flag:
            found.add(parts[2])

        new_lines.append(" ".join(parts))

    
    # If ensure_* requested but the key is missing entirely, insert it (tempo2 accepts anywhere; keep near top).
    present_keys = {(_cleanline(l).split()[0] if _cleanline(l).strip() else '') for l in lines if _cleanline(l).strip()}
    to_insert: List[str] = []
    if ensure_ephem is not None and "EPHEM" not in present_keys:
        to_insert.append(f"EPHEM {ensure_ephem}")
    if ensure_clk is not None and "CLK" not in present_keys:
        to_insert.append(f"CLK {ensure_clk}")
    if ensure_ne_sw is not None and "NE_SW" not in present_keys:
        to_insert.append(f"NE_SW {ensure_ne_sw}")
    if to_insert:
        changed = True
        # Insert after any leading comments/blanks, otherwise at start.
        insert_at = 0
        while insert_at < len(new_lines) and (not new_lines[insert_at].strip() or new_lines[insert_at].lstrip().startswith(("C", "#"))):
            insert_at += 1
        new_lines[insert_at:insert_at] = to_insert + [""]

    missing = sorted(list(wanted - found))
    if missing:
        changed = True

    if dry_run or not apply:
        return {
            "parfile": str(parfile),
            "changed": bool(changed),
            "missing_jumps": missing,
        }

    if not changed:
        return {"parfile": str(parfile), "changed": False, "missing_jumps": []}

    if backup:
        _backup_file(parfile)

    if missing:
        new_lines.append("")
        for v in missing:
            new_lines.append(f"JUMP {jump_flag} {v} 0 0")

    parfile.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return {"parfile": str(parfile), "changed": True, "missing_jumps": missing}




# -----------------------------
# PQC outlier application (optional)
# -----------------------------

def _mjd_from_toa_line(line: str, time_offset_sec: float = 0.0) -> Optional[float]:
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


def _find_qc_csv(psr: str, cfg: FixDatasetConfig) -> Optional[Path]:
    if cfg.qc_results_dir is None:
        return None
    base = Path(cfg.qc_results_dir)
    if cfg.qc_branch:
        cand = base / str(cfg.qc_branch) / f"{psr}_qc.csv"
        if cand.exists():
            return cand
    matches = sorted(base.rglob(f"{psr}_qc.csv"))
    if not matches:
        return None
    return matches[0]


def _collect_qc_mjds(df: pd.DataFrame, cfg: FixDatasetConfig) -> Dict[str, Dict[Optional[str], list[float]]]:
    standard = np.zeros(len(df), dtype=bool)
    if cfg.qc_remove_bad:
        for col in ("bad", "bad_day"):
            if col in df.columns:
                standard |= df[col].fillna(False).astype(bool).to_numpy()
    if cfg.qc_remove_transients and "transient_id" in df.columns:
        standard |= df["transient_id"].fillna(-1).astype(int).to_numpy() >= 0

    solar = np.zeros(len(df), dtype=bool)
    if cfg.qc_remove_solar and "solar_bad" in df.columns:
        solar |= df["solar_bad"].fillna(False).astype(bool).to_numpy()

    orbital = np.zeros(len(df), dtype=bool)
    if cfg.qc_remove_orbital_phase and "orbital_phase_bad" in df.columns:
        orbital |= df["orbital_phase_bad"].fillna(False).astype(bool).to_numpy()

    out: Dict[str, Dict[Optional[str], list[float]]] = {"standard": {}, "solar": {}, "orbital": {}}

    def _build(mask: np.ndarray) -> Dict[Optional[str], list[float]]:
        if not mask.any():
            return {}
        if "_timfile" in df.columns:
            mapping: Dict[Optional[str], list[float]] = {}
            for timfile, sub in df.loc[mask, ["_timfile", "mjd"]].groupby("_timfile"):
                mapping[str(timfile)] = [float(x) for x in sub["mjd"].to_numpy()]
            return mapping
        return {None: [float(x) for x in df.loc[mask, "mjd"].to_numpy()]}

    out["standard"] = _build(standard)
    out["solar"] = _build(solar)
    out["orbital"] = _build(orbital)
    return out


def apply_pqc_outliers(psr_dir: Path, cfg: FixDatasetConfig) -> Dict[str, object]:
    """Comment out or delete TOAs based on pqc outputs for this pulsar."""
    psr = psr_dir.name
    qc_csv = _find_qc_csv(psr, cfg)
    if qc_csv is None:
        return {"pulsar": psr, "qc_csv": None, "matched": 0, "changed": False}

    try:
        df = pd.read_csv(qc_csv)
    except Exception as e:
        return {"pulsar": psr, "qc_csv": str(qc_csv), "error": str(e)}

    mjd_maps = _collect_qc_mjds(df, cfg)
    if not mjd_maps.get("standard") and not mjd_maps.get("solar") and not mjd_maps.get("orbital"):
        return {"pulsar": psr, "qc_csv": str(qc_csv), "matched": 0, "changed": False}

    action = str(cfg.qc_action or "comment").strip().lower()
    if action not in {"comment", "delete"}:
        return {"pulsar": psr, "qc_csv": str(qc_csv), "error": f"Unsupported qc_action: {cfg.qc_action}"}
    solar_action = str(cfg.qc_solar_action or "comment").strip().lower()
    if solar_action not in {"comment", "delete"}:
        return {"pulsar": psr, "qc_csv": str(qc_csv), "error": f"Unsupported qc_solar_action: {cfg.qc_solar_action}"}
    orbital_action = str(cfg.qc_orbital_phase_action or "comment").strip().lower()
    if orbital_action not in {"comment", "delete"}:
        return {"pulsar": psr, "qc_csv": str(qc_csv), "error": f"Unsupported qc_orbital_phase_action: {cfg.qc_orbital_phase_action}"}

    tol = float(cfg.qc_merge_tol_days or (2.0 / 86400.0))
    comment_prefix = str(cfg.qc_comment_prefix or "C QC_OUTLIER").strip()
    solar_prefix = str(cfg.qc_solar_comment_prefix or "# QC_SOLAR").strip()
    orbital_prefix = str(cfg.qc_orbital_phase_comment_prefix or "# QC_BINARY_ECLIPSE").strip()

    tims = list_backend_timfiles(psr_dir)
    total_matched = 0
    changed_files = 0
    file_reports: list[Dict[str, object]] = []

    for tim in tims:
        key = tim.name
        std_map = mjd_maps.get("standard", {})
        solar_map = mjd_maps.get("solar", {})
        orbital_map = mjd_maps.get("orbital", {})
        if (
            (key not in std_map) and (None not in std_map)
            and (key not in solar_map) and (None not in solar_map)
            and (key not in orbital_map) and (None not in orbital_map)
        ):
            continue
        target_mjds = np.asarray(std_map.get(key, std_map.get(None, [])), dtype=float)
        target_mjds_solar = np.asarray(solar_map.get(key, solar_map.get(None, [])), dtype=float)
        target_mjds_orbital = np.asarray(orbital_map.get(key, orbital_map.get(None, [])), dtype=float)
        if target_mjds.size == 0 and target_mjds_solar.size == 0 and target_mjds_orbital.size == 0:
            continue

        lines = tim.read_text(encoding="utf-8", errors="ignore").splitlines()
        new_lines: list[str] = []
        removed = 0
        commented = 0
        time_offset_sec = 0.0

        for raw in lines:
            s = raw.strip()
            if s.startswith("TIME"):
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        time_offset_sec = float(parts[1])
                    except Exception:
                        pass
                new_lines.append(raw)
                continue

            mjd = _mjd_from_toa_line(raw, time_offset_sec=time_offset_sec)
            if mjd is None:
                new_lines.append(raw)
                continue

            is_solar = target_mjds_solar.size > 0 and np.any(np.abs(target_mjds_solar - mjd) <= tol)
            is_orbital = target_mjds_orbital.size > 0 and np.any(np.abs(target_mjds_orbital - mjd) <= tol)
            is_std = target_mjds.size > 0 and np.any(np.abs(target_mjds - mjd) <= tol)

            if is_solar:
                total_matched += 1
                if solar_action == "delete":
                    removed += 1
                    continue
                if solar_prefix and raw.lstrip().startswith(solar_prefix):
                    new_lines.append(raw)
                else:
                    new_lines.append(f"{solar_prefix} {raw}" if solar_prefix else raw)
                    commented += 1
                continue

            if is_orbital:
                total_matched += 1
                if orbital_action == "delete":
                    removed += 1
                    continue
                if orbital_prefix and raw.lstrip().startswith(orbital_prefix):
                    new_lines.append(raw)
                else:
                    new_lines.append(f"{orbital_prefix} {raw}" if orbital_prefix else raw)
                    commented += 1
                continue

            if is_std:
                total_matched += 1
                if action == "delete":
                    removed += 1
                    continue
                if comment_prefix and raw.lstrip().startswith(comment_prefix):
                    new_lines.append(raw)
                else:
                    new_lines.append(f"{comment_prefix} {raw}" if comment_prefix else raw)
                    commented += 1
                continue

            new_lines.append(raw)

        changed = (removed + commented) > 0
        file_reports.append(
            {
                "timfile": str(tim),
                "removed": int(removed),
                "commented": int(commented),
                "changed": bool(changed),
            }
        )

        if changed:
            changed_files += 1
            if cfg.apply and not cfg.dry_run:
                if cfg.backup:
                    _backup_file(tim)
                tim.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return {
        "pulsar": psr,
        "qc_csv": str(qc_csv),
        "matched": int(total_matched),
        "changed_files": int(changed_files),
        "files": file_reports,
    }

def remove_patterns_from_par_tim(
    parfile: Path,
    timfile: Path,
    patterns: Sequence[str],
    apply: bool = False,
    backup: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Remove lines containing any of the patterns from parfile and timfile.

    Adapted from remove_nuppi_big in FixDataset.ipynb.

    Args:
        parfile: Path to the .par file.
        timfile: Path to the .tim file.
        patterns: Sequence of substrings to remove.
        apply: If True, write changes to disk.
        backup: If True, create a backup before writing.
        dry_run: If True, do not write but return planned changes.

    Returns:
        Stats dictionary with counts of removed lines.
    """
    patterns = [p for p in patterns if p]
    if not patterns:
        return {"par_removed": 0, "tim_removed": 0}

    def filt(path: Path) -> Tuple[List[str], int]:
        if not path.exists():
            return ([], 0)
        out = []
        removed = 0
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if any(p in raw for p in patterns):
                removed += 1
            else:
                out.append(raw)
        return out, removed

    par_lines, par_removed = filt(parfile)
    tim_lines, tim_removed = filt(timfile)

    if dry_run or not apply:
        return {"parfile": str(parfile), "timfile": str(timfile), "par_removed": par_removed, "tim_removed": tim_removed}

    if par_removed and backup:
        _backup_file(parfile)
    if tim_removed and backup:
        _backup_file(timfile)

    if par_removed:
        parfile.write_text("\n".join(par_lines) + "\n", encoding="utf-8")
    if tim_removed:
        timfile.write_text("\n".join(tim_lines) + "\n", encoding="utf-8")

    return {"parfile": str(parfile), "timfile": str(timfile), "par_removed": par_removed, "tim_removed": tim_removed}


# -----------------------------
# Coordinate conversion (optional: requires astropy)
# -----------------------------

def _require_astropy():
    """Import astropy or raise a helpful error."""
    try:
        from astropy.coordinates import SkyCoord  # noqa
        from astropy import units  # noqa
        return True
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Coordinate conversion requires optional dependency 'astropy'. "
            "Install with: pip install 'data-combination-pipeline[astro]'"
        ) from e


def equatorial_to_ecliptic_par(parfile: Path) -> Tuple[str, str, str, str, List[str]]:
    """Convert equatorial coordinates in a .par file to ecliptic values.

    Args:
        parfile: Path to the .par file.

    Returns:
        Tuple ``(ELONG, ELAT, PMELONG, PMELAT, original_lines)``.

    Raises:
        RuntimeError: If astropy is not available.
        ValueError: If required keys are missing.
    """
    _require_astropy()
    from astropy.coordinates import SkyCoord
    from astropy import units

    lines = parfile.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    def _get(key: str) -> Optional[List[str]]:
        rows = [l.rstrip("\n").split() for l in lines if l.strip().startswith(key)]
        return rows[0] if rows else None

    raj = _get("RAJ")
    decj = _get("DECJ")
    pmra = _get("PMRA")
    pmdec = _get("PMDEC")

    if not (raj and decj):
        raise ValueError(f"Missing RAJ/DECJ in {parfile}")

    radec = SkyCoord(ra=raj[1], dec=decj[1], unit=(units.hourangle, units.deg), frame="icrs")
    ecl = radec.barycentrictrueecliptic

    # Proper motions are tricky; keep notebook behavior: just transform numeric components if present.
    pmelong = "0"
    pmelat = "0"
    if pmra and pmdec:
        try:
            pmc = SkyCoord(
                ra=raj[1],
                dec=decj[1],
                pm_ra_cosdec=float(pmra[1]) * units.mas / units.yr,
                pm_dec=float(pmdec[1]) * units.mas / units.yr,
                unit=(units.hourangle, units.deg),
                frame="icrs",
            )
            pm_ecl = pmc.barycentrictrueecliptic
            # astropy keeps pm in representation; but values can be extracted
            # This is a best-effort; exact definition depends on convention.
            pmelong = str(getattr(pm_ecl, "pm_lon_coslat", 0 * units.mas / units.yr).to_value(units.mas / units.yr))
            pmelat = str(getattr(pm_ecl, "pm_lat", 0 * units.mas / units.yr).to_value(units.mas / units.yr))
        except Exception:
            pmelong = "0"
            pmelat = "0"

    elong = str(ecl.lon.to_value(units.deg))
    elat = str(ecl.lat.to_value(units.deg))
    return elong, elat, pmelong, pmelat, lines


def ecliptic_to_equatorial_par(parfile: Path) -> Tuple[str, str, str, str, List[str]]:
    """Convert ecliptic coordinates in a .par file to equatorial values.

    Args:
        parfile: Path to the .par file.

    Returns:
        Tuple ``(RAJ, DECJ, PMRA, PMDEC, original_lines)``.

    Raises:
        RuntimeError: If astropy is not available.
        ValueError: If required keys are missing.
    """
    _require_astropy()
    from astropy.coordinates import SkyCoord
    from astropy import units

    lines = parfile.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    def _get(key: str) -> Optional[List[str]]:
        rows = [l.rstrip("\n").split() for l in lines if l.strip().startswith(key)]
        return rows[0] if rows else None

    elong = _get("ELONG")
    elat = _get("ELAT")
    pmelong = _get("PMELONG")
    pmelat = _get("PMELAT")

    if not (elong and elat):
        raise ValueError(f"Missing ELONG/ELAT in {parfile}")

    ecl = SkyCoord(lon=float(elong[1]) * units.deg, lat=float(elat[1]) * units.deg, frame="barycentrictrueecliptic")
    radec = ecl.icrs

    pmra = "0"
    pmdec = "0"
    if pmelong and pmelat:
        try:
            ecl_pm = SkyCoord(
                lon=float(elong[1]) * units.deg,
                lat=float(elat[1]) * units.deg,
                pm_lon_coslat=float(pmelong[1]) * units.mas / units.yr,
                pm_lat=float(pmelat[1]) * units.mas / units.yr,
                frame="barycentrictrueecliptic",
            )
            icrs_pm = ecl_pm.icrs
            pmra = str(getattr(icrs_pm, "pm_ra_cosdec", 0 * units.mas / units.yr).to_value(units.mas / units.yr))
            pmdec = str(getattr(icrs_pm, "pm_dec", 0 * units.mas / units.yr).to_value(units.mas / units.yr))
        except Exception:
            pmra = "0"
            pmdec = "0"

    raj = radec.ra.to_string(unit=units.hourangle, sep=":", pad=True)
    decj = radec.dec.to_string(unit=units.deg, sep=":", pad=True, alwayssign=True)
    return raj, decj, pmra, pmdec, lines


def convert_par_coordinates(parfile: Path, mode: str, apply: bool = False, backup: bool = True, dry_run: bool = False) -> Dict[str, object]:
    """Convert a par file between equatorial (RAJ/DECJ) and ecliptic (ELONG/ELAT).

    mode: 'equ2ecl' or 'ecl2equ'

    Args:
        parfile: Path to the .par file.
        mode: Conversion mode (``"equ2ecl"`` or ``"ecl2equ"``).
        apply: If True, write changes to disk.
        backup: If True, create a backup before writing.
        dry_run: If True, do not write but return planned changes.

    Returns:
        Stats dictionary summarizing the conversion.
    """
    if mode not in {"equ2ecl", "ecl2equ"}:
        raise ValueError("mode must be 'equ2ecl' or 'ecl2equ'")

    if mode == "equ2ecl":
        elong, elat, pmelong, pmelat, lines = equatorial_to_ecliptic_par(parfile)
        replacements = {
            "RAJ": f"ELONG {elong}",
            "DECJ": f"ELAT {elat}",
            "PMRA": f"PMELONG {pmelong}",
            "PMDEC": f"PMELAT {pmelat}",
        }
        drop_keys = {"RAJ", "DECJ", "PMRA", "PMDEC"}
    else:
        raj, decj, pmra, pmdec, lines = ecliptic_to_equatorial_par(parfile)
        replacements = {
            "ELONG": f"RAJ {raj}",
            "ELAT": f"DECJ {decj}",
            "PMELONG": f"PMRA {pmra}",
            "PMELAT": f"PMDEC {pmdec}",
        }
        drop_keys = {"ELONG", "ELAT", "PMELONG", "PMELAT"}

    new_lines: List[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            new_lines.append(line)
            continue
        parts = line.split()
        key = parts[0]
        if key in replacements:
            new_lines.append(replacements[key])
        elif key in drop_keys:
            # drop the opposite-frame key
            continue
        else:
            new_lines.append(line)

    if dry_run or not apply:
        return {"parfile": str(parfile), "changed": True, "mode": mode}

    if backup:
        _backup_file(parfile)
    parfile.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return {"parfile": str(parfile), "changed": True, "mode": mode}


# -----------------------------
# High-level entry point
# -----------------------------

def fix_pulsar_dataset(psr_dir: Path, cfg: FixDatasetConfig) -> Dict[str, object]:
    """Apply or report dataset fixes for a single pulsar directory.

    Args:
        psr_dir: Pulsar directory containing .par/.tim files.
        cfg: FixDataset configuration.

    Returns:
        Report dictionary with per-step results.

    Examples:
        Run a report-only pass::

            report = fix_pulsar_dataset(Path("/data/epta/J1234+5678"), FixDatasetConfig())

    See Also:
        write_fix_report: Write aggregated reports to disk.
    """
    psr = psr_dir.name
    parfile = psr_dir / f"{psr}.par"
    alltim = psr_dir / f"{psr}_all.tim"

    report: Dict[str, object] = {"psr": psr, "steps": []}

    if cfg.remove_patterns and parfile.exists() and alltim.exists():
        rep = remove_patterns_from_par_tim(
            parfile,
            alltim,
            cfg.remove_patterns,
            apply=cfg.apply,
            backup=cfg.backup,
            dry_run=cfg.dry_run,
        )
        report["steps"].append({"remove_patterns": rep})

    if cfg.update_alltim_includes:
        rep = update_alltim_includes(
            psr_dir,
            min_toas=int(cfg.min_toas_per_backend_tim),
            apply=cfg.apply,
            backup=cfg.backup,
            dry_run=cfg.dry_run,
        )
        report["steps"].append({"update_alltim_includes": rep})


    # TIM hygiene: dedupe within each backend tim
    if cfg.dedupe_toas_within_tim:
        tims = list_backend_timfiles(psr_dir)
        reps = []
        for t in tims:
            try:
                reps.append(dedupe_timfile_toas(t, apply=cfg.apply, backup=cfg.backup, dry_run=cfg.dry_run))
            except Exception as e:
                reps.append({"timfile": str(t), "error": str(e)})
        report["steps"].append({"dedupe_toas_within_tim": reps})

    # Smart system flag inference: ensure -sys/-group/-pta in every TOA line
    if cfg.infer_system_flags:
        tims = list_backend_timfiles(psr_dir)
        reps = []
        for t in tims:
            try:
                reps.append(infer_and_apply_system_flags(t, cfg))
            except Exception as e:
                reps.append({"timfile": str(t), "error": str(e)})
        report["steps"].append({"infer_system_flags": reps})

    # Cheap overlap removal (exact duplicates across known overlapping backend tims)
    if cfg.remove_overlaps_exact:
        try:
            rep = remove_overlaps_exact(psr_dir, apply=cfg.apply, backup=cfg.backup, dry_run=cfg.dry_run)
        except Exception as e:
            rep = {"error": str(e)}
        report["steps"].append({"remove_overlaps_exact": rep})

    # Duplicate backend timfile detection (content-identical TOA sets)
    if cfg.check_duplicate_backend_tims:
        try:
            dups = find_duplicate_backend_timfiles(list_backend_timfiles(psr_dir))
            rep = {"groups": [[str(p) for p in g] for g in dups]}
        except Exception as e:
            rep = {"error": str(e)}
        report["steps"].append({"duplicate_backend_timfiles": rep})


    if cfg.required_tim_flags:
        tims = list_backend_timfiles(psr_dir)
        tim_reports = []
        for t in tims:
            try:
                tim_reports.append(
                    ensure_timfile_flags(
                        t,
                        cfg.required_tim_flags,
                        apply=cfg.apply,
                        backup=cfg.backup,
                        dry_run=cfg.dry_run,
                    )
                )
            except Exception as e:
                tim_reports.append({"timfile": str(t), "error": str(e)})
        report["steps"].append({"ensure_timfile_flags": tim_reports})

    if cfg.insert_missing_jumps and parfile.exists():
        # gather jump values across backend tim files
        vals: Set[str] = set()
        for t in list_backend_timfiles(psr_dir):
            vals |= extract_flag_values(t, cfg.jump_flag)

        rep = update_parfile_jumps(
            parfile,
            jump_flag=str(cfg.jump_flag),
            jump_values=sorted(vals),
            ensure_ephem=cfg.ensure_ephem,
            ensure_clk=cfg.ensure_clk,
            ensure_ne_sw=cfg.ensure_ne_sw,
            apply=cfg.apply,
            backup=cfg.backup,
            dry_run=cfg.dry_run,
        )
        report["steps"].append({"update_parfile_jumps": rep})

    if cfg.coord_convert and parfile.exists():
        rep = convert_par_coordinates(
            parfile,
            mode=cfg.coord_convert,
            apply=cfg.apply,
            backup=cfg.backup,
            dry_run=cfg.dry_run,
        )
        report["steps"].append({"coord_convert": rep})

    if cfg.qc_remove_outliers:
        try:
            rep = apply_pqc_outliers(psr_dir, cfg)
        except Exception as e:
            rep = {"error": str(e)}
        report["steps"].append({"qc_outliers": rep})

    return report


def write_fix_report(reports: List[Dict[str, object]], out_dir: Path) -> Path:
    """Write FixDataset reports to disk.

    Args:
        reports: List of per-pulsar report dictionaries.
        out_dir: Output directory.

    Returns:
        Path to the detailed JSON report file.

    Examples:
        Save reports for multiple pulsars::

            detail_path = write_fix_report(reports, Path("results/fix_dataset"))
    """
    import json
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / "fix_dataset_report.json"
    detail_path.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")

    # lightweight TSV summary
    rows = []
    for r in reports:
        psr = r.get("psr")
        steps = r.get("steps", [])
        # flatten some common stats
        added_includes = 0
        missing_jumps = 0
        removed = 0
        for s in steps:
            if "update_alltim_includes" in s:
                rep = s["update_alltim_includes"]
                added_includes += int(rep.get("added", 0) or 0)
                missing = rep.get("to_add", [])
                if missing and int(rep.get("added", 0) or 0) == 0:
                    # dry-run suggestion
                    added_includes += 0
            if "update_parfile_jumps" in s:
                rep = s["update_parfile_jumps"]
                missing_jumps += len(rep.get("missing_jumps", []) or [])
            if "remove_patterns" in s:
                rep = s["remove_patterns"]
                removed += int(rep.get("par_removed", 0) or 0) + int(rep.get("tim_removed", 0) or 0)
        rows.append((psr, added_includes, missing_jumps, removed))

    summary_path = out_dir / "fix_dataset_summary.tsv"
    header = "pulsar\tadded_includes\tmissing_jumps\tremoved_lines\n"
    summary_path.write_text(header + "\n".join([f"{a}\t{b}\t{c}\t{d}" for a, b, c, d in rows]) + "\n", encoding="utf-8")

    return detail_path


# Known overlapping backend .tim basenames (from legacy FixDataset notebook).
# These are NOT pulsar-specific; they describe backends that can contain overlapping TOAs.
OVERLAPPED_TIMFILES: Dict[str, List[str]] = {
    "EFF.P200.1380.tim": ["EFF.EBPP.1360.tim", "EFF.EBPP.1410.tim"],
    "EFF.P217.1380.tim": ["EFF.EBPP.1360.tim", "EFF.EBPP.1410.tim"],
    "EFF.S110.2487.tim": ["EFF.EBPP.2639.tim", "EFF.EBPP.2639.tim"],
    "JBO.ROACH.1520.tim": ["JBO.DFB.1400.tim", "JBO.DFB.1520.tim"],
    "JBO.DFB.1520.tim": ["JBO.DFB.1400.tim"],
    "NRT.NUPPI.1484.tim": ["NRT.BON.1400.tim", "NRT.BON.1600.tim"],
    "NRT.NUPPI.1854.tim": ["NRT.BON.1600.tim", "NRT.BON.2000.tim"],
    "NRT.NUPPI.2539.tim": ["NRT.BON.2000.tim"],
}

def _toa_key_from_line(line: str) -> Optional[Tuple[str, str, str, str]]:
    """Cheap TOA identity key for FORMAT 1: first 4 columns as strings."""
    if not is_toa_line(line):
        return None
    parts = line.strip().split()
    if len(parts) < 4:
        return None
    return (parts[0], parts[1], parts[2], parts[3])

def dedupe_timfile_toas(
    timfile: Path,
    apply: bool = False,
    backup: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Remove exact duplicate TOA lines within a single .tim file (FORMAT 1).

    This is intentionally conservative: only exact duplicates on first 4 columns are removed.
    Directives/comments are preserved in place.

    Args:
        timfile: Path to the .tim file.
        apply: If True, write changes to disk.
        backup: If True, create a backup before writing.
        dry_run: If True, do not write but return planned changes.

    Returns:
        Stats dictionary summarizing removed duplicates.
    """
    if not timfile.exists():
        return {"timfile": str(timfile), "changed": False, "removed": 0}

    lines = timfile.read_text(encoding="utf-8", errors="ignore").splitlines()
    seen: Set[Tuple[str, str, str, str]] = set()
    removed = 0
    new_lines: List[str] = []
    changed = False

    for raw in lines:
        key = _toa_key_from_line(raw)
        if key is None:
            new_lines.append(_cleanline(raw))
            continue
        if key in seen:
            removed += 1
            changed = True
            # drop duplicate
            continue
        seen.add(key)
        new_lines.append(_cleanline(raw))

    if dry_run or not apply or not changed:
        return {"timfile": str(timfile), "changed": bool(changed), "removed": int(removed)}

    if backup:
        _backup_file(timfile)
    timfile.write_text("\\n".join(new_lines) + "\\n", encoding="utf-8")
    return {"timfile": str(timfile), "changed": True, "removed": int(removed)}


def _infer_backend_override(cfg: FixDatasetConfig, timfile: Path) -> Optional[str]:
    """Return an override backend name for this timfile if provided."""
    base = timfile.name
    if cfg.backend_overrides and base in cfg.backend_overrides:
        return str(cfg.backend_overrides[base])
    return None


def infer_and_apply_system_flags(
    timfile: Path,
    cfg: FixDatasetConfig,
) -> Dict[str, object]:
    """Infer -sys/-group/-pta using system_flag_inference and apply to the timfile.

    If backend cannot be inferred and no override exists, this records an error (and optionally raises).

    Args:
        timfile: Path to the .tim file.
        cfg: FixDataset configuration.

    Returns:
        Stats dictionary summarizing applied flags and mapping table updates.
    """
    try:
        from .system_flag_inference import (
            BackendMissingError,
            infer_sys_group_pta,
            apply_flags_to_timfile,
            update_mapping_table,
        )
    except Exception as e:
        return {"timfile": str(timfile), "error": f"system_flag_inference import failed: {e}"}

    override_backend = _infer_backend_override(cfg, timfile)

    try:
        inferred = infer_sys_group_pta(timfile, override_backend=override_backend)
    except BackendMissingError as e:
        msg = str(e)
        if cfg.raise_on_backend_missing:
            raise
        return {"timfile": str(timfile), "error": msg, "sample_toa_line": e.sample_toa_line}

    stats = apply_flags_to_timfile(
        timfile,
        inferred,
        apply=cfg.apply,
        backup=cfg.backup,
        dry_run=cfg.dry_run,
        overwrite_existing=cfg.system_flag_overwrite_existing,
    )

    # Update a global mapping table at dataset root (keyed by timfile basename).
    try:
        # Dataset root is assumed to be parent of psr_dir; fall back to tim's grandparent.
        dataset_root = timfile.parent.parent
        mapping_path = Path(cfg.system_flag_table_path) if cfg.system_flag_table_path else (dataset_root / "system_flag_table.json")
        inferred2 = inferred.copy()
        inferred2["timfile"] = timfile.name
        update_mapping_table(mapping_path, inferred2)
        stats["mapping_table"] = str(mapping_path)
    except Exception as e:
        stats["mapping_error"] = str(e)

    return stats


def _timfile_signature(timfile: Path) -> str:
    """Hash signature of TOA keys for duplicate-tim detection (order-independent)."""
    keys: List[str] = []
    for raw in timfile.read_text(encoding="utf-8", errors="ignore").splitlines():
        k = _toa_key_from_line(raw)
        if k is None:
            continue
        keys.append("|".join(k))
    keys.sort()
    h = hashlib.sha1("\\n".join(keys).encode("utf-8", errors="ignore")).hexdigest()
    return h


def find_duplicate_backend_timfiles(timfiles: Sequence[Path]) -> List[List[Path]]:
    """Find backend timfiles that contain exactly the same set of TOAs.

    Args:
        timfiles: Sequence of backend tim files.

    Returns:
        List of groups, each containing duplicate tim file paths.
    """
    by_sig: Dict[str, List[Path]] = {}
    for t in timfiles:
        try:
            sig = _timfile_signature(t)
        except Exception:
            continue
        by_sig.setdefault(sig, []).append(t)
    return [grp for grp in by_sig.values() if len(grp) > 1]


def remove_overlaps_exact(
    psr_dir: Path,
    overlap_map: Dict[str, List[str]] = OVERLAPPED_TIMFILES,
    apply: bool = False,
    backup: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Cheap overlap remover: for known overlapping backend pairs, comment out exact duplicate TOAs in 'drop' files.

    This will NOT attempt fuzzy time/freq matching; it only removes exact duplicates based on first 4 columns.

    Args:
        psr_dir: Pulsar directory.
        overlap_map: Mapping of keep -> drop backend tim basenames.
        apply: If True, write changes to disk.
        backup: If True, create a backup before writing.
        dry_run: If True, do not write but return planned changes.

    Returns:
        Stats dictionary summarizing commented duplicates.
    """
    tims_by_name = {t.name: t for t in list_backend_timfiles(psr_dir)}
    changed_files = []
    total_commented = 0

    for retain_name, drop_list in overlap_map.items():
        retain = tims_by_name.get(retain_name)
        if retain is None:
            continue

        # Build retain key set
        retain_keys: Set[Tuple[str, str, str, str]] = set()
        for raw in retain.read_text(encoding="utf-8", errors="ignore").splitlines():
            k = _toa_key_from_line(raw)
            if k is not None:
                retain_keys.add(k)

        for drop_name in drop_list:
            drop = tims_by_name.get(drop_name)
            if drop is None:
                continue

            lines = drop.read_text(encoding="utf-8", errors="ignore").splitlines()
            new_lines: List[str] = []
            commented = 0
            file_changed = False

            for raw in lines:
                k = _toa_key_from_line(raw)
                if k is None:
                    new_lines.append(_cleanline(raw))
                    continue
                if k in retain_keys and not raw.lstrip().startswith("C"):
                    # comment this TOA line out
                    new_lines.append("C " + _cleanline(raw))
                    commented += 1
                    file_changed = True
                else:
                    new_lines.append(_cleanline(raw))

            if file_changed:
                total_commented += commented
                changed_files.append(str(drop))
                if not dry_run and apply:
                    if backup:
                        _backup_file(drop)
                    drop.write_text("\\n".join(new_lines) + "\\n", encoding="utf-8")

    return {"changed_files": changed_files, "commented": int(total_commented)}
