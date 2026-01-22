from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import re
import shutil

from .logging_utils import get_logger

logger = get_logger("data_combination_pipeline.dataset_fix")


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

    These features are adapted from the FixDataset.ipynb notebook. They are *disabled by default*
    because automatic edits can be repo-specific.
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


# -----------------------------
# tim file helpers
# -----------------------------

def _cleanline(line: str) -> str:
    return line.rstrip("\n").rstrip(" ")


def _is_comment_or_blank(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith(("C", "#"))


def _is_directive(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    head = s.split()[0]
    return head in _TIM_DIRECTIVES


def is_toa_line(line: str) -> bool:
    """Heuristic: a data line (TOA) in a .tim file."""
    if _is_comment_or_blank(line):
        return False
    if _is_directive(line):
        return False
    # Remaining lines are usually TOAs.
    return True


def count_toa_lines(timfile: Path) -> int:
    n = 0
    if not timfile.exists():
        return 0
    for raw in timfile.read_text(encoding="utf-8", errors="ignore").splitlines():
        if is_toa_line(raw):
            n += 1
    return n


def parse_include_lines(alltim: Path) -> Set[str]:
    """Return set of included relative paths (e.g., 'tims/foo.tim')"""
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
    """List per-backend tim files under <psr>/tims/*.tim (excluding <psr>_all.tim)."""
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
    if not path.exists():
        return
    b = path.with_suffix(path.suffix + ".orig")
    if b.exists():
        return
    shutil.copy2(path, b)


def update_alltim_includes(psr_dir: Path, min_toas: int = 10, apply: bool = False, backup: bool = True, dry_run: bool = False) -> Dict[str, object]:
    """Ensure <psr>_all.tim contains INCLUDE lines for each backend tim file.

    This is a refactor of insert_missing_timfiles/update_alltims from FixDataset.ipynb.
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
    """Collect unique values for a flag from TOA lines (e.g. flag='-sys')."""
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
    """Return (ELONG, ELAT, PMELONG, PMELAT, original_lines)."""
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
    """Return (RAJ, DECJ, PMRA, PMDEC, original_lines)."""
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
    """Apply (or report) dataset fixes for a single pulsar directory."""
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

    return report


def write_fix_report(reports: List[Dict[str, object]], out_dir: Path) -> Path:
    """Write a JSON-ish TSV summary and a detailed JSON file."""
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
