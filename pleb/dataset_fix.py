"""FixDataset utilities for cleaning and normalizing EPTA datasets.

This module provides lightweight, file-based cleanup utilities for `.par` and
`.tim` data, adapted from the FixDataset notebook workflow. It can run in a
report-only mode (default) or apply edits with backups.

See Also:
    pleb.config.PipelineConfig: Pipeline settings for FixDataset integration.
    pleb.pipeline.run_pipeline: Orchestrates FixDataset in the full pipeline.
"""

from __future__ import annotations

from dataclasses import field
from .compat import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from fnmatch import fnmatch
import shutil
import json
import hashlib
import re
import numpy as np
import pandas as pd

try:  # Python 3.11+
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore

from .logging_utils import get_logger
from .system_tables import load_table
from .tim_utils import (
    TIM_DIRECTIVES,
    cleanline,
    count_toa_lines,
    extract_flag_values,
    is_toa_line,
    list_backend_timfiles,
    mjd_from_toa_line,
    parse_include_lines,
    toa_key_from_line,
)
from .tempo2 import build_singularity_prefix, run_subprocess

logger = get_logger("pleb.dataset_fix")

_BACKEND_BW_TABLE: Dict[str, float] = load_table("backend_bw", {})

_LEGACY_SYS_ALLOWLIST = {
    "EFF.EBPP.1360",
    "EFF.EBPP.1410",
    "EFF.EBPP.2639",
    "EFF.P200.1365",
    "EFF.P200.1380",
    "EFF.P200.1425",
    "EFF.P217.1365",
    "EFF.P217.1380",
    "EFF.P217.1425",
    "EFF.S110.2487",
    "EFF.S60.4857",
    "JBO.DFB.1400",
    "JBO.DFB.1520",
    "JBO.MK2.1520",
    "JBO.ROACH.1420",
    "JBO.ROACH.1620",
    "LEAP.1396",
    "NRT.BON.1400",
    "NRT.BON.1600",
    "NRT.BON.2000",
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
        generate_alltim_variants: Generate additional ``*_all.<variant>.tim`` files.
        backend_classifications_path: TOML with class->system mappings.
        alltim_variants_path: TOML with variant selection rules.
        relabel_rules_path: TOML with declarative TOA relabel rules.
        overlap_rules_path: TOML with declarative overlap/dedup preference rules.
        overlap_exact_catalog_path: TOML keep->drop map used by remove_overlaps_exact.
        jump_reference_variants: Build per-variant reference-system JUMP parfiles.
        required_tim_flags: Flags to ensure on each TOA line.
        infer_system_flags: Infer ``-sys``/``-group``/``-pta`` flags.
        system_flag_table_path: Path to the system-flag table (JSON/TOML).
        system_flag_mapping_path: Path to editable mapping/allowlist JSON.
        system_flag_overwrite_existing: Overwrite existing system flags.
        backend_overrides: Map tim basename to backend name override.
        raise_on_backend_missing: Raise when backend cannot be inferred.
        dedupe_toas_within_tim: Remove duplicate TOAs within each tim.
        check_duplicate_backend_tims: Detect duplicated backend tims.
        remove_overlaps_exact: Remove known overlapping TOAs across backends.
        insert_missing_jumps: Insert missing JUMP lines into par files.
        jump_flag: Flag used to label inserted jumps.
        prune_stale_jumps: Drop JUMP lines not present in timfile flags.
        ensure_ephem: Ensure EPHEM param exists (optional value).
        ensure_clk: Ensure CLK param exists (optional value).
        ensure_ne_sw: Ensure NE_SW param exists (optional value).
        force_ne_sw_overwrite: Overwrite existing NE_SW only when True.
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
    generate_alltim_variants: bool = False
    backend_classifications_path: Optional[str] = None
    alltim_variants_path: Optional[str] = None
    relabel_rules_path: Optional[str] = None
    overlap_rules_path: Optional[str] = None
    overlap_exact_catalog_path: Optional[str] = None
    jump_reference_variants: bool = False
    jump_reference_keep_tmp: bool = False
    jump_reference_jump_flag: str = "-sys"
    jump_reference_csv_dir: Optional[str] = None
    tempo2_home_dir: Optional[str] = None
    tempo2_dataset_name: Optional[str] = None
    tempo2_singularity_image: Optional[str] = None

    # tim flag insertion (applies to per-backend tims under <psr>/tims/)
    # Example: {"-pta": "EPTA", "-be": "P200", "-sys": "SomeSys"}
    required_tim_flags: Dict[str, str] = field(default_factory=dict)

    # System flag inference (smart -sys/-group/-pta creation)
    infer_system_flags: bool = False
    system_flag_table_path: Optional[str] = (
        None  # JSON mapping stored at dataset root if None
    )
    system_flag_mapping_path: Optional[str] = None
    system_flag_overwrite_existing: bool = False
    wsrt_p2_force_sys_by_freq: bool = False
    wsrt_p2_prefer_dual_channel: bool = False
    wsrt_p2_mjd_tol_sec: float = 0.99e-6
    wsrt_p2_action: str = "comment"  # "comment" | "delete"
    wsrt_p2_comment_prefix: str = "C WSRT_P2_PREFER_DUAL"
    backend_overrides: Dict[str, str] = field(
        default_factory=dict
    )  # tim basename -> backend
    raise_on_backend_missing: bool = False

    # TIM hygiene
    dedupe_toas_within_tim: bool = True
    dedupe_mjd_tol_sec: float = 0.0
    dedupe_freq_tol_mhz: Optional[float] = None
    dedupe_freq_tol_auto: bool = False
    check_duplicate_backend_tims: bool = False

    # Overlap handling (cheap: exact TOA duplicate removal across known overlapping backends)
    remove_overlaps_exact: bool = True

    # parfile maintenance
    insert_missing_jumps: bool = True
    jump_flag: str = "-sys"
    prune_stale_jumps: bool = False
    ensure_ephem: Optional[str] = None
    ensure_clk: Optional[str] = None
    ensure_ne_sw: Optional[str] = None
    force_ne_sw_overwrite: bool = False

    # remove known-bad backend strings (used by remove_nuppi_big in the notebook)
    remove_patterns: List[str] = field(
        default_factory=lambda: ["NRT.NUPPI.", "NRT.NUXPI."]
    )

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


def _backup_file(path: Path) -> None:
    """Create a one-time backup of a file."""
    if not path.exists():
        return
    b = path.with_suffix(path.suffix + ".orig")
    if b.exists():
        return
    shutil.copy2(path, b)


def update_alltim_includes(
    psr_dir: Path,
    min_toas: int = 10,
    apply: bool = False,
    backup: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
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
        return {
            "psr": psr,
            "alltim": str(alltim),
            "added": 0,
            "to_add": [],
            "dropped": dropped,
        }

    logger.info("%s: %d missing INCLUDE lines in %s", psr, len(to_add), alltim.name)

    if dry_run or not apply:
        return {
            "psr": psr,
            "alltim": str(alltim),
            "added": 0,
            "to_add": to_add,
            "dropped": dropped,
        }

    if not alltim.exists():
        raise FileNotFoundError(str(alltim))

    if backup:
        _backup_file(alltim)

    text = alltim.read_text(encoding="utf-8", errors="ignore").splitlines()
    new_lines = list(text)
    for rel in to_add:
        new_lines.append(f"INCLUDE {rel}")

    alltim.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return {
        "psr": psr,
        "alltim": str(alltim),
        "added": len(to_add),
        "to_add": to_add,
        "dropped": dropped,
    }


def _load_toml(path: Path) -> Dict[str, object]:
    if tomllib is None:
        raise RuntimeError(
            "tomllib unavailable; Python 3.11+ required for TOML support."
        )
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _default_overlap_exact_catalog_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "configs" / "system_tables" / "overlapped_timfiles.toml"


def _load_overlap_exact_catalog(path: Path) -> Dict[str, List[str]]:
    """Load keep->drop overlap mapping from TOML.

    Supported forms:
    - [overlap_map] with keys as keep timfile names and list[str] values
    - top-level keys keep->list[str]
    """
    data = _load_toml(path)
    raw = data.get("overlap_map", data)
    out: Dict[str, List[str]] = {}
    if not isinstance(raw, dict):
        return out
    for keep_name, drop_vals in raw.items():
        keep = str(keep_name).strip()
        if not keep:
            continue
        if isinstance(drop_vals, str):
            vals = [drop_vals.strip()] if str(drop_vals).strip() else []
        elif isinstance(drop_vals, list):
            vals = [str(v).strip() for v in drop_vals if str(v).strip()]
        else:
            vals = []
        if vals:
            out[keep] = vals
    return out


def _collect_timfile_sys_values(timfile: Path) -> Set[str]:
    vals: Set[str] = set()
    for raw in timfile.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not is_toa_line(raw):
            continue
        parts = raw.strip().split()
        if "-sys" not in parts:
            continue
        i = parts.index("-sys")
        if i + 1 < len(parts):
            vals.add(str(parts[i + 1]).strip())
    return vals


def _load_backend_classifications(
    path: Path,
) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    data = _load_toml(path)
    raw_cls = data.get("classifications", {})
    classes: Dict[str, Set[str]] = {}
    if isinstance(raw_cls, dict):
        for cname, vals in raw_cls.items():
            key = str(cname).strip()
            if not key or not isinstance(vals, list):
                continue
            classes[key] = {str(v).strip() for v in vals if str(v).strip()}

    raw_over = data.get("overrides", {})
    tim_over: Dict[str, str] = {}
    if isinstance(raw_over, dict):
        raw_tim = raw_over.get("timfile_class", {})
        if isinstance(raw_tim, dict):
            for tname, cname in raw_tim.items():
                tn = str(tname).strip()
                cn = str(cname).strip()
                if tn and cn:
                    tim_over[tn] = cn
    return classes, tim_over


def _load_alltim_variants(path: Path) -> Dict[str, Dict[str, object]]:
    data = _load_toml(path)
    out: Dict[str, Dict[str, object]] = {}
    raw = data.get("variants", {})
    if not isinstance(raw, dict):
        return out
    for vname, conf in raw.items():
        name = str(vname).strip()
        if not name or not isinstance(conf, dict):
            continue
        include = conf.get("include_classes", [])
        exclude = conf.get("exclude_classes", [])
        mixed = str(conf.get("mixed_policy", "any")).strip().lower()
        if mixed not in {"any", "all", "error"}:
            mixed = "any"
        out[name] = {
            "include_classes": (
                [str(x).strip() for x in include if str(x).strip()]
                if isinstance(include, list)
                else []
            ),
            "exclude_classes": (
                [str(x).strip() for x in exclude if str(x).strip()]
                if isinstance(exclude, list)
                else []
            ),
            "mixed_policy": mixed,
        }
    return out


def _timfile_classes(
    timfile: Path,
    class_to_systems: Dict[str, Set[str]],
    timfile_class_overrides: Dict[str, str],
) -> Set[str]:
    if timfile.name in timfile_class_overrides:
        return {timfile_class_overrides[timfile.name]}
    systems = _collect_timfile_sys_values(timfile)
    out: Set[str] = set()
    for cname, sysvals in class_to_systems.items():
        if systems & sysvals:
            out.add(cname)
    return out


def _variant_selects(
    tim_classes: Set[str],
    include_classes: Set[str],
    exclude_classes: Set[str],
    mixed_policy: str,
) -> Tuple[bool, Optional[str]]:
    if tim_classes & exclude_classes:
        return False, None
    if mixed_policy == "error" and len(tim_classes) > 1:
        return False, f"mixed classes: {sorted(tim_classes)}"
    if mixed_policy == "all":
        if not tim_classes:
            return False, None
        return tim_classes.issubset(include_classes), None
    return bool(tim_classes & include_classes), None


def _parse_tempo2_redchisq(stdout_path: Path) -> Optional[float]:
    text = (
        stdout_path.read_text(encoding="utf-8", errors="ignore")
        if stdout_path.exists()
        else ""
    )
    if not text:
        return None
    for pat in (r"reduced\s*chisq\s*=\s*([0-9.+\-eE]+)", r"red\s*chisq\s*=\s*([0-9.+\-eE]+)"):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def _variant_name_from_alltim(psr: str, alltim: Path) -> str:
    base = alltim.name
    if base == f"{psr}_all.tim":
        return "base"
    pref = f"{psr}_all."
    if base.startswith(pref) and base.endswith(".tim"):
        return base[len(pref) : -len(".tim")]
    return alltim.stem


def _parse_tim_system_rows(timfile: Path) -> Tuple[Dict[str, List[str]], List[str], Dict[str, List[float]]]:
    lines = timfile.read_text(encoding="utf-8", errors="ignore").splitlines()
    systems: Set[str] = set()
    errs: Dict[str, List[float]] = {}
    for raw in lines:
        if not is_toa_line(raw):
            continue
        parts = raw.strip().split()
        if len(parts) < 4:
            continue
        if "-sys" not in parts:
            continue
        i = parts.index("-sys")
        if i + 1 >= len(parts):
            continue
        sys_val = parts[i + 1]
        systems.add(sys_val)
        try:
            errs.setdefault(sys_val, []).append(float(parts[3]))
        except Exception:
            pass

    by_sys: Dict[str, List[str]] = {s: [] for s in systems}
    for raw in lines:
        if is_toa_line(raw):
            parts = raw.strip().split()
            if "-sys" in parts:
                i = parts.index("-sys")
                if i + 1 < len(parts):
                    s = parts[i + 1]
                    if s in by_sys:
                        by_sys[s].append(raw)
                        continue
            continue
        for s in by_sys:
            by_sys[s].append(raw)

    return by_sys, sorted(systems), errs


def build_variant_reference_jump_pars(psr_dir: Path, cfg: FixDatasetConfig) -> Dict[str, object]:
    psr = psr_dir.name
    par = psr_dir / f"{psr}.par"
    if not par.exists():
        return {"psr": psr, "error": f"Missing par file: {par}"}
    if not (cfg.tempo2_home_dir and cfg.tempo2_dataset_name and cfg.tempo2_singularity_image):
        return {
            "psr": psr,
            "error": "tempo2 context missing; set tempo2_home_dir/tempo2_dataset_name/tempo2_singularity_image",
        }

    all_variants = sorted(psr_dir.glob(f"{psr}_all.*.tim"))
    if not all_variants:
        return {"psr": psr, "variants": [], "message": "No _all.variant.tim files found"}

    tmp_root = psr_dir / ".pleb_jump_reference_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    dataset_root = psr_dir.parent
    if cfg.jump_reference_csv_dir:
        csv_base = Path(str(cfg.jump_reference_csv_dir))
        if not csv_base.is_absolute():
            csv_base = dataset_root / csv_base
    else:
        csv_base = dataset_root / "results" / "jump_reference"
    csv_base = csv_base / psr
    csv_base.mkdir(parents=True, exist_ok=True)

    # Build no-JUMP par once.
    par_lines = par.read_text(encoding="utf-8", errors="ignore").splitlines()
    no_jump_lines = []
    for ln in par_lines:
        s = ln.strip()
        if not s:
            no_jump_lines.append(ln)
            continue
        if s.split(maxsplit=1)[0] == "JUMP":
            continue
        no_jump_lines.append(ln)
    no_jump_par = tmp_root / f"{psr}.nojump.par"
    no_jump_par.write_text("\n".join(no_jump_lines) + "\n", encoding="utf-8")

    prefix = build_singularity_prefix(
        Path(str(cfg.tempo2_home_dir)),
        Path(str(cfg.tempo2_dataset_name)),
        Path(str(cfg.tempo2_singularity_image)),
    )

    out_variants: Dict[str, object] = {}
    for alltim in all_variants:
        vname = _variant_name_from_alltim(psr, alltim)
        include_paths = sorted(parse_include_lines(alltim))
        system_rows: Dict[str, Dict[str, object]] = {}
        temp_sys_files: Dict[str, List[Path]] = {}

        for rel in include_paths:
            src_tim = psr_dir / rel
            if not src_tim.exists():
                continue
            by_sys, systems, errs = _parse_tim_system_rows(src_tim)
            if not systems:
                continue
            for sysv in systems:
                buf = by_sys.get(sysv, [])
                if not any(is_toa_line(x) for x in buf):
                    continue
                out_tim = tmp_root / f"{src_tim.stem}__{sysv}.tim"
                out_tim.write_text("\n".join(buf) + "\n", encoding="utf-8")
                temp_sys_files.setdefault(sysv, []).append(out_tim)
                r = system_rows.setdefault(
                    sysv,
                    {
                        "system": sysv,
                        "n_toa": 0,
                        "toa_err_us": [],
                        "source_timfiles": set(),
                        "reduced_chisq": None,
                    },
                )
                r["n_toa"] = int(r["n_toa"]) + int(count_toa_lines(out_tim))
                r["toa_err_us"].extend(errs.get(sysv, []))
                r["source_timfiles"].add(src_tim.name)

        # Evaluate each system with no-JUMP par.
        for sysv, files in temp_sys_files.items():
            sys_all = tmp_root / f"{psr}_all.{vname}.{sysv}.tim"
            lines = ["FORMAT 1"]
            for f in files:
                lines.append(f"INCLUDE {f.name}")
            sys_all.write_text("\n".join(lines) + "\n", encoding="utf-8")

            par_container = f"/data/{psr}/.pleb_jump_reference_tmp/{no_jump_par.name}"
            tim_container = f"/data/{psr}/.pleb_jump_reference_tmp/{sys_all.name}"
            log_path = tmp_root / f"{psr}.{vname}.{sysv}.tempo2.log"
            rc = run_subprocess(prefix + ["tempo2", "-f", par_container, tim_container], log_path)
            red = _parse_tempo2_redchisq(log_path) if rc == 0 else None
            system_rows[sysv]["reduced_chisq"] = red

        # Finalize metrics and choose reference system.
        rows = []
        for sysv, r in system_rows.items():
            errs = np.asarray(r.pop("toa_err_us", []), dtype=float)
            med = float(np.median(errs)) if errs.size else np.nan
            r["median_toa_err_us"] = med
            r["source_timfiles"] = ",".join(sorted(r["source_timfiles"]))
            rows.append(r)

        if not rows:
            out_variants[vname] = {"variant_alltim": str(alltim), "error": "No systems found"}
            continue

        # smallest median err first, then largest n_toa, then lexical system.
        rows_sorted = sorted(
            rows,
            key=lambda x: (
                np.inf if np.isnan(float(x.get("median_toa_err_us", np.nan))) else float(x.get("median_toa_err_us", np.nan)),
                -int(x.get("n_toa", 0)),
                str(x.get("system", "")),
            ),
        )
        ref_system = str(rows_sorted[0]["system"])

        csv_path = csv_base / f"{psr}_jump_reference.{vname}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("variant,system,n_toa,median_toa_err_us,reduced_chisq,source_timfiles,reference_system\n")
            for r in rows_sorted:
                f.write(
                    f"{vname},{r['system']},{r['n_toa']},{r['median_toa_err_us']},{r.get('reduced_chisq')},{r.get('source_timfiles')},{ref_system}\n"
                )

        # Build variant par with jumps for all systems in this variant:
        # reference fixed at 0 0, others start at 0 with fit flag 1.
        par_out = psr_dir / f"{psr}.{vname}.par"
        jump_flag = str(cfg.jump_reference_jump_flag or "-sys")
        out_lines = []
        for ln in no_jump_lines:
            out_lines.append(ln)
        for r in rows_sorted:
            sysv = str(r["system"])
            fit_flag = 0 if sysv == ref_system else 1
            out_lines.append(f"JUMP {jump_flag} {sysv} 0 {fit_flag}")
        if cfg.apply and not cfg.dry_run:
            if cfg.backup and par_out.exists():
                _backup_file(par_out)
            par_out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

        out_variants[vname] = {
            "variant_alltim": str(alltim),
            "reference_system": ref_system,
            "systems": rows_sorted,
            "csv": str(csv_path),
            "par_out": str(par_out),
            "written": bool(cfg.apply and not cfg.dry_run),
        }

    if not bool(cfg.jump_reference_keep_tmp):
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass

    return {"psr": psr, "variants": out_variants}


def generate_alltim_variants(psr_dir: Path, cfg: FixDatasetConfig) -> Dict[str, object]:
    psr = psr_dir.name
    alltim = psr_dir / f"{psr}_all.tim"
    if not alltim.exists():
        return {"psr": psr, "error": f"Missing {alltim.name}"}
    if not cfg.backend_classifications_path or not cfg.alltim_variants_path:
        return {"psr": psr, "error": "Missing backend/variant TOML paths"}

    cls_path = Path(cfg.backend_classifications_path)
    var_path = Path(cfg.alltim_variants_path)
    if not cls_path.is_absolute():
        cls_path = psr_dir.parent / cls_path
    if not var_path.is_absolute():
        var_path = psr_dir.parent / var_path
    if not cls_path.exists() or not var_path.exists():
        return {
            "psr": psr,
            "error": "Classification or variant TOML does not exist",
            "classification_path": str(cls_path),
            "variants_path": str(var_path),
        }

    class_to_systems, tim_overrides = _load_backend_classifications(cls_path)
    variants = _load_alltim_variants(var_path)
    if not variants:
        return {"psr": psr, "error": "No variants defined"}

    tims = list_backend_timfiles(psr_dir)
    tim_classes: Dict[str, Set[str]] = {
        t.name: _timfile_classes(t, class_to_systems, tim_overrides) for t in tims
    }

    base_lines = alltim.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_lines = [ln for ln in base_lines if not ln.strip().startswith("INCLUDE")]
    out_variants: Dict[str, object] = {}

    for vname, vcfg in variants.items():
        include = set(vcfg.get("include_classes", []))
        exclude = set(vcfg.get("exclude_classes", []))
        mixed_policy = str(vcfg.get("mixed_policy", "any"))
        selected: List[str] = []
        skipped_mixed: List[str] = []
        for t in tims:
            ok, err = _variant_selects(
                tim_classes.get(t.name, set()), include, exclude, mixed_policy
            )
            if err:
                skipped_mixed.append(t.name)
            if ok:
                selected.append(f"tims/{t.name}")

        target = psr_dir / f"{psr}_all.{vname}.tim"
        content = list(header_lines)
        for rel in sorted(selected):
            content.append(f"INCLUDE {rel}")

        if cfg.dry_run or not cfg.apply:
            out_variants[vname] = {
                "path": str(target),
                "selected_includes": sorted(selected),
                "skipped_mixed": skipped_mixed,
                "written": False,
            }
            continue

        if cfg.backup and target.exists():
            _backup_file(target)
        target.write_text("\n".join(content) + "\n", encoding="utf-8")
        out_variants[vname] = {
            "path": str(target),
            "selected_includes": sorted(selected),
            "skipped_mixed": skipped_mixed,
            "written": True,
        }

    return {
        "psr": psr,
        "classification_path": str(cls_path),
        "variants_path": str(var_path),
        "variants": out_variants,
    }


def ensure_timfile_flags(
    timfile: Path,
    required: Dict[str, str],
    apply: bool = False,
    backup: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
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
        line = cleanline(raw)
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


def _token_get_flag(parts: Sequence[str], flag: str) -> Optional[str]:
    if flag not in parts:
        return None
    i = parts.index(flag)
    if i + 1 >= len(parts):
        return None
    return str(parts[i + 1])


def _token_set_flag(parts: List[str], flag: str, value: str) -> bool:
    if flag in parts:
        i = parts.index(flag)
        if i + 1 < len(parts) and parts[i + 1] != value:
            parts[i + 1] = value
            return True
        if i + 1 >= len(parts):
            parts.append(value)
            return True
        return False
    parts.extend([flag, value])
    return True


def _load_relabel_rules(path: Path) -> List[Dict[str, object]]:
    data = _load_toml(path)
    raw = data.get("rules", [])
    if not isinstance(raw, list):
        return []

    rules: List[Dict[str, object]] = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            continue
        name = str(r.get("name", f"rule_{i+1}")).strip() or f"rule_{i+1}"
        enabled = bool(r.get("enabled", True))
        tim_glob = r.get("tim_glob", "*.tim")
        if isinstance(tim_glob, str):
            tim_globs = [tim_glob.strip()] if tim_glob.strip() else ["*.tim"]
        elif isinstance(tim_glob, list):
            tim_globs = [str(x).strip() for x in tim_glob if str(x).strip()]
            if not tim_globs:
                tim_globs = ["*.tim"]
        else:
            tim_globs = ["*.tim"]

        def _list_str(key: str) -> Set[str]:
            v = r.get(key)
            if v is None:
                return set()
            if isinstance(v, str):
                s = v.strip()
                return {s} if s else set()
            if isinstance(v, list):
                return {str(x).strip() for x in v if str(x).strip()}
            return set()

        def _opt_float(key: str) -> Optional[float]:
            v = r.get(key)
            if v in (None, ""):
                return None
            return float(v)

        required_flags: Dict[str, str] = {}
        rf = r.get("require_flags", {})
        if isinstance(rf, dict):
            for k, v in rf.items():
                ks = str(k).strip()
                vs = str(v).strip()
                if ks and vs:
                    required_flags[ks] = vs

        rules.append(
            {
                "name": name,
                "enabled": enabled,
                "tim_globs": tim_globs,
                "match_sys": _list_str("match_sys"),
                "match_group": _list_str("match_group"),
                "match_pta": _list_str("match_pta"),
                "sat_regex": str(r.get("sat_regex", "")).strip(),
                "line_regex": str(r.get("line_regex", "")).strip(),
                "mjd_min": _opt_float("mjd_min"),
                "mjd_max": _opt_float("mjd_max"),
                "freq_min_mhz": _opt_float("freq_min_mhz"),
                "freq_max_mhz": _opt_float("freq_max_mhz"),
                "require_flags": required_flags,
                "set_sys": (
                    None
                    if r.get("set_sys") in (None, "")
                    else str(r.get("set_sys")).strip()
                ),
                "set_group": (
                    None
                    if r.get("set_group") in (None, "")
                    else str(r.get("set_group")).strip()
                ),
                "set_pta": (
                    None
                    if r.get("set_pta") in (None, "")
                    else str(r.get("set_pta")).strip()
                ),
            }
        )
    return rules


def apply_relabel_rules(psr_dir: Path, cfg: FixDatasetConfig) -> Dict[str, object]:
    psr = psr_dir.name
    if not cfg.relabel_rules_path:
        return {"psr": psr, "skipped": True, "reason": "no rules path"}

    rules_path = Path(cfg.relabel_rules_path)
    if not rules_path.is_absolute():
        rules_path = psr_dir.parent / rules_path
    if not rules_path.exists():
        return {
            "psr": psr,
            "error": f"Missing relabel rules file: {rules_path}",
        }

    rules = _load_relabel_rules(rules_path)
    if not rules:
        return {"psr": psr, "rules_path": str(rules_path), "rules": []}

    tims = list_backend_timfiles(psr_dir)
    out_rules: List[Dict[str, object]] = []

    for rule in rules:
        if not bool(rule.get("enabled", True)):
            out_rules.append(
                {"name": str(rule.get("name", "unnamed")), "enabled": False}
            )
            continue

        tim_globs = list(rule.get("tim_globs", ["*.tim"]))
        files = [
            t
            for t in tims
            if any(fnmatch(t.name, str(g)) for g in tim_globs if str(g).strip())
        ]

        sat_re = None
        if str(rule.get("sat_regex", "")).strip():
            sat_re = re.compile(str(rule.get("sat_regex")))
        line_re = None
        if str(rule.get("line_regex", "")).strip():
            line_re = re.compile(str(rule.get("line_regex")))

        match_sys = set(rule.get("match_sys", set()))
        match_group = set(rule.get("match_group", set()))
        match_pta = set(rule.get("match_pta", set()))
        req_flags = dict(rule.get("require_flags", {}))
        mjd_min = rule.get("mjd_min")
        mjd_max = rule.get("mjd_max")
        freq_min = rule.get("freq_min_mhz")
        freq_max = rule.get("freq_max_mhz")
        set_sys = rule.get("set_sys")
        set_group = rule.get("set_group")
        set_pta = rule.get("set_pta")

        file_reports: List[Dict[str, object]] = []
        for tim in files:
            lines = tim.read_text(encoding="utf-8", errors="ignore").splitlines()
            changed = False
            matched = 0
            changed_lines = 0
            new_lines: List[str] = []
            for raw in lines:
                line = cleanline(raw)
                if not is_toa_line(line):
                    new_lines.append(line)
                    continue
                parts = line.split()
                if len(parts) < 4:
                    new_lines.append(line)
                    continue

                try:
                    freq = float(parts[1])
                    mjd = float(parts[2])
                except Exception:
                    new_lines.append(line)
                    continue

                sat = parts[0]
                sys_val = _token_get_flag(parts, "-sys")
                grp_val = _token_get_flag(parts, "-group")
                pta_val = _token_get_flag(parts, "-pta")

                if match_sys and (sys_val not in match_sys):
                    new_lines.append(line)
                    continue
                if match_group and (grp_val not in match_group):
                    new_lines.append(line)
                    continue
                if match_pta and (pta_val not in match_pta):
                    new_lines.append(line)
                    continue
                if sat_re is not None and not sat_re.search(sat):
                    new_lines.append(line)
                    continue
                if line_re is not None and not line_re.search(line):
                    new_lines.append(line)
                    continue
                if mjd_min is not None and mjd < float(mjd_min):
                    new_lines.append(line)
                    continue
                if mjd_max is not None and mjd > float(mjd_max):
                    new_lines.append(line)
                    continue
                if freq_min is not None and freq < float(freq_min):
                    new_lines.append(line)
                    continue
                if freq_max is not None and freq > float(freq_max):
                    new_lines.append(line)
                    continue
                flag_miss = False
                for fk, fv in req_flags.items():
                    if _token_get_flag(parts, str(fk)) != str(fv):
                        flag_miss = True
                        break
                if flag_miss:
                    new_lines.append(line)
                    continue

                matched += 1
                local_change = False
                if set_sys is not None:
                    local_change = (
                        _token_set_flag(parts, "-sys", str(set_sys)) or local_change
                    )
                if set_group is not None:
                    local_change = (
                        _token_set_flag(parts, "-group", str(set_group)) or local_change
                    )
                if set_pta is not None:
                    local_change = (
                        _token_set_flag(parts, "-pta", str(set_pta)) or local_change
                    )
                if local_change:
                    changed = True
                    changed_lines += 1
                new_lines.append(" ".join(parts))

            if changed and cfg.apply and not cfg.dry_run:
                if cfg.backup:
                    _backup_file(tim)
                tim.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

            file_reports.append(
                {
                    "timfile": str(tim),
                    "matched_toas": int(matched),
                    "changed_toas": int(changed_lines),
                    "changed": bool(changed),
                }
            )

        out_rules.append(
            {
                "name": str(rule.get("name", "unnamed")),
                "enabled": True,
                "tim_globs": tim_globs,
                "files": file_reports,
            }
        )

    return {
        "psr": psr,
        "rules_path": str(rules_path),
        "rules": out_rules,
    }


def _load_overlap_rules(path: Path) -> List[Dict[str, object]]:
    """Load declarative overlap rules from TOML.

    Supported kinds:
    - ``prefer_multichannel``: between a primary/secondary timfile pair, keep
      the file that contains dual-channel epochs and comment/delete duplicates
      from the paired file.
    """
    data = _load_toml(path)
    raw = data.get("rules", [])
    if not isinstance(raw, list):
        return []

    rules: List[Dict[str, object]] = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            continue
        kind = str(r.get("kind", "prefer_multichannel")).strip().lower()
        if kind != "prefer_multichannel":
            continue
        name = str(r.get("name", f"overlap_rule_{i+1}")).strip() or f"overlap_rule_{i+1}"
        primary_glob = str(r.get("primary_glob", "")).strip()
        secondary_glob = str(r.get("secondary_glob", "")).strip()
        if not primary_glob or not secondary_glob:
            continue
        rules.append(
            {
                "name": name,
                "enabled": bool(r.get("enabled", True)),
                "kind": kind,
                "primary_glob": primary_glob,
                "secondary_glob": secondary_glob,
                "mjd_tol_sec": float(r.get("mjd_tol_sec", 0.99e-6)),
                "action": str(r.get("action", "comment")).strip().lower(),
                "comment_prefix": str(
                    r.get("comment_prefix", "C RULE_PREFER_MULTICHANNEL")
                ).strip(),
            }
        )
    return rules


def apply_overlap_rules(psr_dir: Path, cfg: FixDatasetConfig) -> Dict[str, object]:
    """Apply declarative overlap rules to per-backend tim files."""
    psr = psr_dir.name
    if not cfg.overlap_rules_path:
        return {"psr": psr, "skipped": True, "reason": "no overlap rules path"}

    rules_path = Path(cfg.overlap_rules_path)
    if not rules_path.is_absolute():
        rules_path = psr_dir.parent / rules_path
    if not rules_path.exists():
        return {"psr": psr, "error": f"Missing overlap rules file: {rules_path}"}

    rules = _load_overlap_rules(rules_path)
    if not rules:
        return {"psr": psr, "rules_path": str(rules_path), "rules": []}

    out_rules: List[Dict[str, object]] = []
    for rule in rules:
        if not bool(rule.get("enabled", True)):
            out_rules.append(
                {"name": str(rule.get("name", "unnamed")), "enabled": False}
            )
            continue

        rep = prefer_multichannel_pair_rule(
            psr_dir,
            primary_glob=str(rule.get("primary_glob")),
            secondary_glob=str(rule.get("secondary_glob")),
            mjd_tol_sec=float(rule.get("mjd_tol_sec", 0.99e-6)),
            action=str(rule.get("action", "comment")),
            comment_prefix=str(
                rule.get("comment_prefix", "C RULE_PREFER_MULTICHANNEL")
            ),
            apply=cfg.apply,
            backup=cfg.backup,
            dry_run=cfg.dry_run,
        )
        rep["name"] = str(rule.get("name", "unnamed"))
        out_rules.append(rep)

    return {
        "psr": psr,
        "rules_path": str(rules_path),
        "rules": out_rules,
    }


# -----------------------------
# par file helpers
# -----------------------------


def ensure_parfile_defaults(
    parfile: Path,
    *,
    ensure_ephem: Optional[str] = None,
    ensure_clk: Optional[str] = None,
    ensure_ne_sw: Optional[str] = None,
    force_ne_sw_overwrite: bool = False,
    apply: bool = False,
    backup: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Ensure selected default parameters exist with requested values.

    This function updates only EPHEM/CLK/NE_SW and does not touch JUMPs.
    Missing keys are inserted near the top of the parfile.
    """
    if not parfile.exists():
        raise FileNotFoundError(str(parfile))

    wanted: Dict[str, Optional[str]] = {
        "EPHEM": ensure_ephem,
        "CLK": ensure_clk,
        "NE_SW": ensure_ne_sw,
    }
    wanted = {k: v for k, v in wanted.items() if v is not None}
    if not wanted:
        return {
            "parfile": str(parfile),
            "changed": False,
            "updated": [],
            "inserted": [],
        }

    lines = parfile.read_text(encoding="utf-8", errors="ignore").splitlines()
    new_lines: List[str] = []
    present_keys: Set[str] = set()
    updated: List[str] = []
    changed = False

    for raw in lines:
        line = cleanline(raw)
        if not line.strip():
            new_lines.append(line)
            continue
        parts = line.split()
        key = parts[0]
        present_keys.add(key)
        if (
            key == "NE_SW"
            and ensure_ne_sw is not None
            and not force_ne_sw_overwrite
            and len(parts) > 1
        ):
            new_lines.append(" ".join(parts))
            continue
        if key in wanted and len(parts) > 1 and parts[1] != str(wanted[key]):
            parts[1] = str(wanted[key])
            updated.append(key)
            changed = True
        new_lines.append(" ".join(parts))

    to_insert: List[str] = []
    inserted: List[str] = []
    for key in ("EPHEM", "CLK", "NE_SW"):
        if key in wanted and key not in present_keys:
            to_insert.append(f"{key} {wanted[key]}")
            inserted.append(key)
    if to_insert:
        changed = True
        insert_at = 0
        while insert_at < len(new_lines) and (
            not new_lines[insert_at].strip()
            or new_lines[insert_at].lstrip().startswith(("C", "#"))
        ):
            insert_at += 1
        new_lines[insert_at:insert_at] = to_insert + [""]

    if dry_run or not apply:
        return {
            "parfile": str(parfile),
            "changed": bool(changed),
            "updated": sorted(set(updated)),
            "inserted": inserted,
        }

    if not changed:
        return {
            "parfile": str(parfile),
            "changed": False,
            "updated": [],
            "inserted": [],
        }

    if backup:
        _backup_file(parfile)
    parfile.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return {
        "parfile": str(parfile),
        "changed": True,
        "updated": sorted(set(updated)),
        "inserted": inserted,
    }


def update_parfile_jumps(
    parfile: Path,
    jump_flag: str,
    jump_values: Sequence[str],
    prune_stale_jumps: bool = False,
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
        prune_stale_jumps: Drop JUMP lines with values not in ``jump_values``.
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
    removed_stale: List[str] = []

    lines = parfile.read_text(encoding="utf-8", errors="ignore").splitlines()
    new_lines: List[str] = []

    changed = False
    for raw in lines:
        line = cleanline(raw)
        if not line.strip():
            new_lines.append(line)
            continue
        parts = line.split()
        key = parts[0]

        if (
            ensure_ephem is not None
            and key == "EPHEM"
            and len(parts) > 1
            and parts[1] != ensure_ephem
        ):
            parts[1] = ensure_ephem
            changed = True
        if (
            ensure_clk is not None
            and key == "CLK"
            and len(parts) > 1
            and parts[1] != ensure_clk
        ):
            parts[1] = ensure_clk
            changed = True
        if (
            ensure_ne_sw is not None
            and key == "NE_SW"
            and len(parts) > 1
            and parts[1] != ensure_ne_sw
        ):
            parts[1] = ensure_ne_sw
            changed = True

        # consume already-present JUMPs
        if key == "JUMP" and len(parts) >= 3 and parts[1] == jump_flag:
            val = parts[2]
            if prune_stale_jumps and wanted and val not in wanted:
                removed_stale.append(val)
                changed = True
                continue
            found.add(val)

        new_lines.append(" ".join(parts))

    # If ensure_* requested but the key is missing entirely, insert it (tempo2 accepts anywhere; keep near top).
    present_keys = {
        (cleanline(line_str).split()[0] if cleanline(line_str).strip() else "")
        for line_str in lines
        if cleanline(line_str).strip()
    }
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
        while insert_at < len(new_lines) and (
            not new_lines[insert_at].strip()
            or new_lines[insert_at].lstrip().startswith(("C", "#"))
        ):
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
            "removed_stale_jumps": removed_stale,
        }

    if not changed:
        return {
            "parfile": str(parfile),
            "changed": False,
            "missing_jumps": [],
            "removed_stale_jumps": [],
        }

    if backup:
        _backup_file(parfile)

    if missing:
        new_lines.append("")
        for v in missing:
            new_lines.append(f"JUMP {jump_flag} {v} 0 0")

    parfile.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return {
        "parfile": str(parfile),
        "changed": True,
        "missing_jumps": missing,
        "removed_stale_jumps": removed_stale,
    }


# -----------------------------
# PQC outlier application (optional)
# -----------------------------


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


def _collect_qc_mjds(
    df: pd.DataFrame, cfg: FixDatasetConfig
) -> Dict[str, Dict[Optional[str], list[float]]]:
    standard = np.zeros(len(df), dtype=bool)
    if cfg.qc_remove_bad:
        for col in ("bad", "bad_day"):
            if col in df.columns:
                standard |= df[col].fillna(False).astype(bool).to_numpy()
    if cfg.qc_remove_outliers:
        for col in (
            "outlier_any",
            "bad_point",
            "bad_hard",
            "robust_outlier",
            "robust_global_outlier",
            "bad_mad",
            "bad_ou",
        ):
            if col in df.columns:
                standard |= df[col].fillna(False).astype(bool).to_numpy()
    if cfg.qc_remove_transients and "transient_id" in df.columns:
        standard |= df["transient_id"].fillna(-1).astype(int).to_numpy() >= 0

    solar = np.zeros(len(df), dtype=bool)
    if cfg.qc_remove_solar and "solar_event_member" in df.columns:
        solar |= df["solar_event_member"].fillna(False).astype(bool).to_numpy()

    orbital = np.zeros(len(df), dtype=bool)
    if cfg.qc_remove_orbital_phase and "orbital_phase_bad" in df.columns:
        orbital |= df["orbital_phase_bad"].fillna(False).astype(bool).to_numpy()

    out: Dict[str, Dict[Optional[str], list[float]]] = {
        "standard": {},
        "solar": {},
        "orbital": {},
    }

    def _build(mask: np.ndarray) -> Dict[Optional[str], list[float]]:
        if not mask.any():
            return {}
        if "_timfile" in df.columns:
            mapping: Dict[Optional[str], list[float]] = {}
            for timfile, sub in df.loc[mask, ["_timfile", "mjd"]].groupby("_timfile"):
                key = str(timfile)
                try:
                    key = Path(key).name
                except Exception:
                    pass
                mapping[key] = [float(x) for x in sub["mjd"].to_numpy()]
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
    if (
        not mjd_maps.get("standard")
        and not mjd_maps.get("solar")
        and not mjd_maps.get("orbital")
    ):
        return {"pulsar": psr, "qc_csv": str(qc_csv), "matched": 0, "changed": False}

    action = str(cfg.qc_action or "comment").strip().lower()
    if action not in {"comment", "delete"}:
        return {
            "pulsar": psr,
            "qc_csv": str(qc_csv),
            "error": f"Unsupported qc_action: {cfg.qc_action}",
        }
    solar_action = str(cfg.qc_solar_action or "comment").strip().lower()
    if solar_action not in {"comment", "delete"}:
        return {
            "pulsar": psr,
            "qc_csv": str(qc_csv),
            "error": f"Unsupported qc_solar_action: {cfg.qc_solar_action}",
        }
    orbital_action = str(cfg.qc_orbital_phase_action or "comment").strip().lower()
    if orbital_action not in {"comment", "delete"}:
        return {
            "pulsar": psr,
            "qc_csv": str(qc_csv),
            "error": f"Unsupported qc_orbital_phase_action: {cfg.qc_orbital_phase_action}",
        }

    tol = float(cfg.qc_merge_tol_days or (2.0 / 86400.0))
    comment_prefix = str(cfg.qc_comment_prefix or "C QC_OUTLIER").strip()
    solar_prefix = str(cfg.qc_solar_comment_prefix or "# QC_SOLAR").strip()
    orbital_prefix = str(
        cfg.qc_orbital_phase_comment_prefix or "# QC_BINARY_ECLIPSE"
    ).strip()

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
            (key not in std_map)
            and (None not in std_map)
            and (key not in solar_map)
            and (None not in solar_map)
            and (key not in orbital_map)
            and (None not in orbital_map)
        ):
            continue
        target_mjds = np.asarray(std_map.get(key, std_map.get(None, [])), dtype=float)
        target_mjds_solar = np.asarray(
            solar_map.get(key, solar_map.get(None, [])), dtype=float
        )
        target_mjds_orbital = np.asarray(
            orbital_map.get(key, orbital_map.get(None, [])), dtype=float
        )
        if (
            target_mjds.size == 0
            and target_mjds_solar.size == 0
            and target_mjds_orbital.size == 0
        ):
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

            mjd = mjd_from_toa_line(raw, time_offset_sec=time_offset_sec)
            if mjd is None:
                new_lines.append(raw)
                continue

            is_solar = target_mjds_solar.size > 0 and np.any(
                np.abs(target_mjds_solar - mjd) <= tol
            )
            is_orbital = target_mjds_orbital.size > 0 and np.any(
                np.abs(target_mjds_orbital - mjd) <= tol
            )
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
                    new_lines.append(
                        f"{orbital_prefix} {raw}" if orbital_prefix else raw
                    )
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
                    new_lines.append(
                        f"{comment_prefix} {raw}" if comment_prefix else raw
                    )
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
        return {
            "parfile": str(parfile),
            "timfile": str(timfile),
            "par_removed": par_removed,
            "tim_removed": tim_removed,
        }

    if par_removed and backup:
        _backup_file(parfile)
    if tim_removed and backup:
        _backup_file(timfile)

    if par_removed:
        parfile.write_text("\n".join(par_lines) + "\n", encoding="utf-8")
    if tim_removed:
        timfile.write_text("\n".join(tim_lines) + "\n", encoding="utf-8")

    return {
        "parfile": str(parfile),
        "timfile": str(timfile),
        "par_removed": par_removed,
        "tim_removed": tim_removed,
    }


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
        rows = [
            line_str.rstrip("\n").split()
            for line_str in lines
            if line_str.strip().startswith(key)
        ]
        return rows[0] if rows else None

    raj = _get("RAJ")
    decj = _get("DECJ")
    pmra = _get("PMRA")
    pmdec = _get("PMDEC")

    if not (raj and decj):
        raise ValueError(f"Missing RAJ/DECJ in {parfile}")

    radec = SkyCoord(
        ra=raj[1], dec=decj[1], unit=(units.hourangle, units.deg), frame="icrs"
    )
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
            pmelong = str(
                getattr(pm_ecl, "pm_lon_coslat", 0 * units.mas / units.yr).to_value(
                    units.mas / units.yr
                )
            )
            pmelat = str(
                getattr(pm_ecl, "pm_lat", 0 * units.mas / units.yr).to_value(
                    units.mas / units.yr
                )
            )
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
        rows = [
            line_str.rstrip("\n").split()
            for line_str in lines
            if line_str.strip().startswith(key)
        ]
        return rows[0] if rows else None

    elong = _get("ELONG")
    elat = _get("ELAT")
    pmelong = _get("PMELONG")
    pmelat = _get("PMELAT")

    if not (elong and elat):
        raise ValueError(f"Missing ELONG/ELAT in {parfile}")

    ecl = SkyCoord(
        lon=float(elong[1]) * units.deg,
        lat=float(elat[1]) * units.deg,
        frame="barycentrictrueecliptic",
    )
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
            pmra = str(
                getattr(icrs_pm, "pm_ra_cosdec", 0 * units.mas / units.yr).to_value(
                    units.mas / units.yr
                )
            )
            pmdec = str(
                getattr(icrs_pm, "pm_dec", 0 * units.mas / units.yr).to_value(
                    units.mas / units.yr
                )
            )
        except Exception:
            pmra = "0"
            pmdec = "0"

    raj = radec.ra.to_string(unit=units.hourangle, sep=":", pad=True)
    decj = radec.dec.to_string(unit=units.deg, sep=":", pad=True, alwayssign=True)
    return raj, decj, pmra, pmdec, lines


def convert_par_coordinates(
    parfile: Path,
    mode: str,
    apply: bool = False,
    backup: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
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
    if cfg.apply and cfg.dedupe_toas_within_tim:
        tims = list_backend_timfiles(psr_dir)
        reps = []
        for t in tims:
            try:
                reps.append(
                    dedupe_timfile_toas(
                        t,
                        apply=cfg.apply,
                        backup=cfg.backup,
                        dry_run=cfg.dry_run,
                        mjd_tol_sec=float(cfg.dedupe_mjd_tol_sec),
                        freq_tol_mhz=cfg.dedupe_freq_tol_mhz,
                        freq_tol_auto=bool(cfg.dedupe_freq_tol_auto),
                    )
                )
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

    # Declarative relabeling for splitting/renaming backend groups after PQC.
    if cfg.relabel_rules_path:
        try:
            rep = apply_relabel_rules(psr_dir, cfg)
        except Exception as e:
            rep = {"error": str(e)}
        report["steps"].append({"apply_relabel_rules": rep})

    if cfg.generate_alltim_variants:
        try:
            rep = generate_alltim_variants(psr_dir, cfg)
        except Exception as e:
            rep = {"error": str(e)}
        report["steps"].append({"generate_alltim_variants": rep})

    if cfg.jump_reference_variants:
        try:
            rep = build_variant_reference_jump_pars(psr_dir, cfg)
        except Exception as e:
            rep = {"error": str(e)}
        report["steps"].append({"build_variant_reference_jump_pars": rep})

    # Declarative overlap handling (preferred) for system-specific duplicate policy.
    if cfg.apply and cfg.overlap_rules_path:
        try:
            rep = apply_overlap_rules(psr_dir, cfg)
        except Exception as e:
            rep = {"error": str(e)}
        report["steps"].append({"apply_overlap_rules": rep})
    # Legacy WSRT-P2 special-case switch (kept for backwards compatibility).
    elif cfg.apply and cfg.wsrt_p2_prefer_dual_channel:
        try:
            rep = wsrt_p2_prefer_dual_channel(
                psr_dir,
                mjd_tol_sec=float(cfg.wsrt_p2_mjd_tol_sec),
                action=str(cfg.wsrt_p2_action or "comment"),
                comment_prefix=str(
                    cfg.wsrt_p2_comment_prefix or "C WSRT_P2_PREFER_DUAL"
                ),
                apply=cfg.apply,
                backup=cfg.backup,
                dry_run=cfg.dry_run,
            )
        except Exception as e:
            rep = {"error": str(e)}
        report["steps"].append({"wsrt_p2_prefer_dual_channel": rep})

    # Cheap overlap removal (exact duplicates across known overlapping backend tims)
    if cfg.apply and cfg.remove_overlaps_exact:
        try:
            rep = remove_overlaps_exact(
                psr_dir,
                overlap_catalog_path=cfg.overlap_exact_catalog_path,
                apply=cfg.apply,
                backup=cfg.backup,
                dry_run=cfg.dry_run,
            )
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

    if parfile.exists() and any(
        v is not None for v in (cfg.ensure_ephem, cfg.ensure_clk, cfg.ensure_ne_sw)
    ):
        rep = ensure_parfile_defaults(
            parfile,
            ensure_ephem=cfg.ensure_ephem,
            ensure_clk=cfg.ensure_clk,
            ensure_ne_sw=cfg.ensure_ne_sw,
            force_ne_sw_overwrite=bool(cfg.force_ne_sw_overwrite),
            apply=cfg.apply,
            backup=cfg.backup,
            dry_run=cfg.dry_run,
        )
        report["steps"].append({"ensure_parfile_defaults": rep})

    if cfg.insert_missing_jumps and parfile.exists():
        # gather jump values across backend tim files
        vals: Set[str] = set()
        for t in list_backend_timfiles(psr_dir):
            vals |= extract_flag_values(t, cfg.jump_flag)

        rep = update_parfile_jumps(
            parfile,
            jump_flag=str(cfg.jump_flag),
            jump_values=sorted(vals),
            prune_stale_jumps=bool(cfg.prune_stale_jumps),
            ensure_ephem=None,
            ensure_clk=None,
            ensure_ne_sw=None,
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
                removed += int(rep.get("par_removed", 0) or 0) + int(
                    rep.get("tim_removed", 0) or 0
                )
        rows.append((psr, added_includes, missing_jumps, removed))

    summary_path = out_dir / "fix_dataset_summary.tsv"
    header = "pulsar\tadded_includes\tmissing_jumps\tremoved_lines\n"
    summary_path.write_text(
        header + "\n".join([f"{a}\t{b}\t{c}\t{d}" for a, b, c, d in rows]) + "\n",
        encoding="utf-8",
    )

    return detail_path


def _infer_freq_tol(freqs: np.ndarray, bw: Optional[float]) -> Optional[float]:
    """Infer a frequency tolerance (MHz) from observed spacing or bandwidth."""
    if freqs.size >= 2:
        uniq = np.unique(np.round(freqs.astype(float), 6))
        if uniq.size >= 2:
            diffs = np.diff(np.sort(uniq))
            diffs = diffs[diffs > 0]
            if diffs.size:
                return max(0.01, min(1.0, float(diffs.min()) * 0.25))
    if bw is not None and bw > 0:
        return max(0.01, min(1.0, float(bw) / 1024.0))
    return None


def dedupe_timfile_toas(
    timfile: Path,
    apply: bool = False,
    backup: bool = True,
    dry_run: bool = False,
    mjd_tol_sec: float = 0.0,
    freq_tol_mhz: Optional[float] = None,
    freq_tol_auto: bool = False,
) -> Dict[str, object]:
    """Remove duplicate TOA lines within a single .tim file (FORMAT 1).

    Default: exact duplicates on first 4 columns only. When tolerances are
    provided, treat TOAs as duplicates if they share the same file token and
    are within (mjd_tol_sec, freq_tol_mhz).
    """
    if not timfile.exists():
        return {"timfile": str(timfile), "changed": False, "removed": 0}

    lines = timfile.read_text(encoding="utf-8", errors="ignore").splitlines()
    removed = 0
    new_lines: List[str] = []
    changed = False

    # If no tolerance requested, keep legacy exact-key behavior.
    if mjd_tol_sec <= 0.0 and not freq_tol_auto and freq_tol_mhz is None:
        seen: Set[Tuple[str, str, str, str]] = set()
        for raw in lines:
            key = toa_key_from_line(raw)
            if key is None:
                new_lines.append(cleanline(raw))
                continue
            if key in seen:
                removed += 1
                changed = True
                continue
            seen.add(key)
            new_lines.append(cleanline(raw))
    else:
        # Tolerance-based dedupe: same file token + close MJD + close freq.
        tol_days = float(mjd_tol_sec) / 86400.0
        bw = _BACKEND_BW_TABLE.get(timfile.name)
        freqs = []
        parsed = []
        for raw in lines:
            if not is_toa_line(raw):
                parsed.append((raw, None, None, None))
                continue
            parts = raw.strip().split()
            if len(parts) < 4:
                parsed.append((raw, None, None, None))
                continue
            try:
                freq = float(parts[1])
                mjd = float(parts[2])
            except Exception:
                parsed.append((raw, None, None, None))
                continue
            freqs.append(freq)
            parsed.append((raw, parts[0], mjd, freq))

        if freq_tol_mhz is None and freq_tol_auto:
            freq_tol_mhz = _infer_freq_tol(
                np.array(freqs, dtype=float), float(bw) if bw is not None else None
            )
        if freq_tol_mhz is None:
            freq_tol_mhz = 0.0

        kept: List[Tuple[str, float, float]] = []
        for raw, ftoken, mjd, freq in parsed:
            if ftoken is None or mjd is None or freq is None:
                new_lines.append(cleanline(raw))
                continue
            is_dup = False
            for k_ftoken, k_mjd, k_freq in kept:
                if k_ftoken != ftoken:
                    continue
                if abs(mjd - k_mjd) <= tol_days and abs(freq - k_freq) <= float(
                    freq_tol_mhz
                ):
                    is_dup = True
                    break
            if is_dup:
                removed += 1
                changed = True
                continue
            kept.append((ftoken, mjd, freq))
            new_lines.append(cleanline(raw))

    if dry_run or not apply or not changed:
        return {
            "timfile": str(timfile),
            "changed": bool(changed),
            "removed": int(removed),
        }

    if backup:
        _backup_file(timfile)
    timfile.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
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
            TelescopeMissingError,
            parse_tim_toa_table,
            infer_sys_group_pta,
            apply_flags_to_timfile,
            update_mapping_table,
            load_system_flag_mapping,
            SystemInferenceConfig,
        )
    except Exception as e:
        return {
            "timfile": str(timfile),
            "error": f"system_flag_inference import failed: {e}",
        }

    override_backend = _infer_backend_override(cfg, timfile)
    # Resolve dataset root (parent of pulsar dir).
    if timfile.parent.name == "tims":
        psr_dir = timfile.parent.parent
    else:
        psr_dir = timfile.parent
    fallback_root = psr_dir.parent

    def _root_from_cfg_path(path_value: Optional[str]) -> Optional[Path]:
        if not path_value:
            return None
        p = Path(path_value)
        if p.is_absolute():
            return p.parent
        return None

    dataset_root = (
        _root_from_cfg_path(cfg.system_flag_table_path)
        or _root_from_cfg_path(cfg.system_flag_mapping_path)
        or fallback_root
    )
    override_telescope = None
    mapping = None

    def _has_legacy_sys_flag(tpath: Path) -> bool:
        try:
            lines = tpath.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return False
        for raw in lines:
            s = raw.strip()
            if not s or s.startswith(("#", "C")):
                continue
            head = s.split(maxsplit=1)[0]
            if head in TIM_DIRECTIVES:
                continue
            parts = s.split()
            if "-sys" not in parts:
                continue
            i = parts.index("-sys")
            if i + 1 >= len(parts):
                continue
            val = parts[i + 1]
            if val.startswith("NRT.NUPPI.") or val.startswith("LOFAR."):
                continue
            if val in _LEGACY_SYS_ALLOWLIST:
                return True
        return False

    # Optional mapping/allowlist file (editable by users).
    try:
        mapping_path = (
            Path(cfg.system_flag_mapping_path)
            if cfg.system_flag_mapping_path
            else (dataset_root / "system_flag_mapping.json")
        )
        if not mapping_path.is_absolute():
            mapping_path = dataset_root / mapping_path
        if mapping_path.exists():
            mapping = load_system_flag_mapping(mapping_path)
            if not override_backend:
                be_map = mapping.get("backend_by_timfile", {}) or {}
                override_backend = be_map.get(timfile.name) or be_map.get(str(timfile))
            tel_map = mapping.get("telescope_by_timfile", {}) or {}
            override_telescope = tel_map.get(timfile.name) or tel_map.get(str(timfile))
    except Exception:
        mapping = None

    cfg_override = None
    if mapping:
        base_cfg = SystemInferenceConfig()
        cfg_override = SystemInferenceConfig(
            backend_allowlist=mapping.get("backend_allowlist"),
            telescope_allowlist=(
                mapping.get("telescope_allowlist")
                if mapping.get("telescope_allowlist") is not None
                else base_cfg.telescope_allowlist
            ),
            backend_aliases=mapping.get("backend_aliases", {}),
            telescope_aliases=mapping.get("telescope_aliases", {}),
        )

    try:
        inferred = infer_sys_group_pta(
            timfile,
            cfg=cfg_override if cfg_override is not None else SystemInferenceConfig(),
            override_backend=override_backend,
            override_telescope=override_telescope,
        )
    except (BackendMissingError, TelescopeMissingError) as e:
        msg = str(e)
        if cfg.raise_on_backend_missing:
            raise
        return {
            "timfile": str(timfile),
            "error": msg,
            "sample_toa_line": getattr(e, "sample_toa_line", None),
        }

    legacy_sys = _has_legacy_sys_flag(timfile)

    try:
        tel_vals = inferred.get("tel", pd.Series([], dtype=object))
        if any(
            str(t).upper() == "LOFAR" for t in pd.Series(tel_vals).dropna().unique()
        ):
            return {"timfile": str(timfile), "skipped": True, "reason": "LOFAR"}
    except Exception:
        pass

    wsrt_p2_mismatches = None
    try:
        tel_vals = inferred.get("tel", pd.Series([], dtype=object))
        be_vals = inferred.get("backend", pd.Series([], dtype=object))
        tel_set = {str(t).upper() for t in pd.Series(tel_vals).dropna().unique()}
        be_set = {str(b).upper() for b in pd.Series(be_vals).dropna().unique()}
        if tel_set == {"WSRT"} and be_set == {"P2"}:
            df = parse_tim_toa_table(timfile)
            idx_to_sys = {
                int(r.line_idx): str(r.sys) for r in inferred.itertuples(index=False)
            }
            mismatches = 0
            for r in df.itertuples(index=False):
                flags = r.flags or {}
                sys_val = flags.get("-sys")
                want = idx_to_sys.get(int(r.line_idx))
                if want and sys_val and str(sys_val) != str(want):
                    mismatches += 1
            wsrt_p2_mismatches = mismatches
    except Exception:
        wsrt_p2_mismatches = None

    def _is_new_wsrt_p2_timfile(tpath: Path) -> bool:
        try:
            lines = tpath.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return False
        has_gof = False
        has_pta = False
        has_group = False
        for raw in lines:
            s = raw.strip()
            if not s or s.startswith(("#", "C")):
                continue
            head = s.split(maxsplit=1)[0]
            if head in TIM_DIRECTIVES:
                continue
            if "-gof" in s:
                has_gof = True
            if "-pta" in s:
                has_pta = True
            if "-group" in s:
                has_group = True
        return has_gof and not (has_pta or has_group)

    allow_overwrite = cfg.system_flag_overwrite_existing and not legacy_sys
    try:
        tel_vals = inferred.get("tel", pd.Series([], dtype=object))
        be_vals = inferred.get("backend", pd.Series([], dtype=object))
        tel_set = {str(t).upper() for t in pd.Series(tel_vals).dropna().unique()}
        be_set = {str(b).upper() for b in pd.Series(be_vals).dropna().unique()}
        if not (
            tel_set == {"WSRT"}
            and be_set == {"P2"}
            and _is_new_wsrt_p2_timfile(timfile)
        ):
            allow_overwrite = False
    except Exception:
        allow_overwrite = False
    if cfg.wsrt_p2_force_sys_by_freq:
        try:
            tel_vals = inferred.get("tel", pd.Series([], dtype=object))
            be_vals = inferred.get("backend", pd.Series([], dtype=object))
            tel_set = {str(t).upper() for t in pd.Series(tel_vals).dropna().unique()}
            be_set = {str(b).upper() for b in pd.Series(be_vals).dropna().unique()}
            if tel_set == {"WSRT"} and be_set == {"P2"}:
                allow_overwrite = True
        except Exception:
            pass

    stats = apply_flags_to_timfile(
        timfile,
        inferred,
        apply=cfg.apply,
        backup=cfg.backup,
        dry_run=cfg.dry_run,
        overwrite_existing=allow_overwrite,
    )
    if wsrt_p2_mismatches is not None:
        stats["wsrt_p2_sys_mismatches"] = int(wsrt_p2_mismatches)

    # Update a global mapping table at dataset root (keyed by timfile basename).
    try:
        # Dataset root is assumed to be parent of psr_dir; fall back to tim's grandparent.
        mapping_path = (
            Path(cfg.system_flag_table_path)
            if cfg.system_flag_table_path
            else (dataset_root / "system_flag_table.json")
        )
        if not mapping_path.is_absolute():
            mapping_path = dataset_root / mapping_path
        inferred2 = inferred.copy()
        inferred2["timfile"] = timfile.name
        update_mapping_table(mapping_path, inferred2)
        stats["mapping_table"] = str(mapping_path)
        if cfg.apply and not cfg.dry_run:
            for fname in ("system_flag_table.json", "system_flag_table.toml"):
                per_psr = psr_dir / fname
                try:
                    if per_psr.exists() and per_psr.resolve() != mapping_path.resolve():
                        per_psr.unlink()
                except Exception:
                    pass
    except Exception as e:
        stats["mapping_error"] = str(e)

    return stats


def _timfile_signature(timfile: Path) -> str:
    """Hash signature of TOA keys for duplicate-tim detection (order-independent)."""
    keys: List[str] = []
    for raw in timfile.read_text(encoding="utf-8", errors="ignore").splitlines():
        k = toa_key_from_line(raw)
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
    overlap_map: Optional[Dict[str, List[str]]] = None,
    overlap_catalog_path: Optional[str] = None,
    apply: bool = False,
    backup: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Cheap overlap remover: for known overlapping backend pairs, comment out exact duplicate TOAs in 'drop' files.

    This will NOT attempt fuzzy time/freq matching; it only comments exact duplicates
    based on first 4 columns.

    Args:
        psr_dir: Pulsar directory.
        overlap_map: Mapping of keep -> drop backend tim basenames. If None, load
            from ``overlap_catalog_path`` or default catalog path.
        overlap_catalog_path: Optional TOML path for overlap map.
        apply: If True, write changes to disk.
        backup: If True, create a backup before writing.
        dry_run: If True, do not write but return planned changes.

    Returns:
        Stats dictionary summarizing commented duplicates.
    """
    loaded_from: Optional[str] = None
    if overlap_map is None:
        cat = Path(overlap_catalog_path) if overlap_catalog_path else _default_overlap_exact_catalog_path()
        if not cat.is_absolute():
            cat = psr_dir.parent / cat
        if cat.exists():
            overlap_map = _load_overlap_exact_catalog(cat)
            loaded_from = str(cat)
        else:
            overlap_map = {}

    tims_by_name = {t.name: t for t in list_backend_timfiles(psr_dir)}
    changed_files = []
    total_commented = 0

    for retain_name, drop_list in (overlap_map or {}).items():
        retain = tims_by_name.get(retain_name)
        if retain is None:
            continue

        # Build retain key set
        retain_keys: Set[Tuple[str, str, str, str]] = set()
        for raw in retain.read_text(encoding="utf-8", errors="ignore").splitlines():
            k = toa_key_from_line(raw)
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
                k = toa_key_from_line(raw)
                if k is None:
                    new_lines.append(cleanline(raw))
                    continue
                if k in retain_keys and not raw.lstrip().startswith("C"):
                    # Non-destructive default: keep line but comment it with reason.
                    new_lines.append(
                        "C OVERLAP_DUPLICATE(retain="
                        + retain_name
                        + ") "
                        + cleanline(raw)
                    )
                    commented += 1
                    file_changed = True
                else:
                    new_lines.append(cleanline(raw))

            if file_changed:
                total_commented += commented
                changed_files.append(str(drop))
                if not dry_run and apply:
                    if backup:
                        _backup_file(drop)
                    drop.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    rep = {"changed_files": changed_files, "commented": int(total_commented)}
    if loaded_from is not None:
        rep["overlap_catalog"] = loaded_from
    return rep

def _parse_toa_records(
    timfile: Path,
) -> Tuple[List[str], List[Tuple[int, str, float, float]]]:
    """Parse TOA rows from a timfile as (line_idx, file_token, mjd, freq_mhz)."""
    lines = timfile.read_text(encoding="utf-8", errors="ignore").splitlines()
    recs: List[Tuple[int, str, float, float]] = []
    for i, raw in enumerate(lines):
        if not is_toa_line(raw):
            continue
        parts = raw.strip().split()
        if len(parts) < 4:
            continue
        try:
            freq = float(parts[1])
            mjd = float(parts[2])
        except Exception:
            continue
        recs.append((i, parts[0], mjd, freq))
    return lines, recs


def _dual_channel_epoch_map(
    recs: List[Tuple[int, str, float, float]],
    tol_days: float,
) -> Dict[str, List[float]]:
    """Return token->epoch_mjd list where at least two channels exist at same epoch."""
    by_token: Dict[str, List[Tuple[float, float]]] = {}
    for _, token, mjd, freq in recs:
        by_token.setdefault(token, []).append((mjd, freq))
    epochs: Dict[str, List[float]] = {}
    for token, vals in by_token.items():
        vals.sort(key=lambda v: v[0])
        token_epochs: List[float] = []
        cluster: List[Tuple[float, float]] = []
        for mjd, freq in vals:
            if not cluster:
                cluster = [(mjd, freq)]
                continue
            if abs(mjd - cluster[-1][0]) <= tol_days:
                cluster.append((mjd, freq))
                continue
            uniq_freqs = {round(f, 6) for _, f in cluster}
            if len(cluster) >= 2 and len(uniq_freqs) >= 2:
                token_epochs.append(float(np.mean([x[0] for x in cluster])))
            cluster = [(mjd, freq)]
        if cluster:
            uniq_freqs = {round(f, 6) for _, f in cluster}
            if len(cluster) >= 2 and len(uniq_freqs) >= 2:
                token_epochs.append(float(np.mean([x[0] for x in cluster])))
        if token_epochs:
            epochs[token] = sorted(token_epochs)
    return epochs


def prefer_multichannel_pair_rule(
    psr_dir: Path,
    *,
    primary_glob: str,
    secondary_glob: str,
    mjd_tol_sec: float = 0.99e-6,
    action: str = "comment",
    comment_prefix: str = "C RULE_PREFER_MULTICHANNEL",
    apply: bool = False,
    backup: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Generic overlap rule: prefer file with dual-channel epochs in file pairs.

    Pairing logic:
    - files matching ``primary_glob`` create canonical keys by filename.
    - files matching ``secondary_glob`` are paired to primary by removing one
      trailing ``_2`` before ``.tim`` (e.g. ``WSRT.P2.1380_2.tim`` -> ``WSRT.P2.1380.tim``).
    """
    mode = str(action or "comment").strip().lower()
    if mode not in {"comment", "delete"}:
        mode = "comment"
    tol_days = float(mjd_tol_sec) / 86400.0
    tims = list_backend_timfiles(psr_dir)

    primary_files = [t for t in tims if fnmatch(t.name, str(primary_glob))]
    secondary_files = [t for t in tims if fnmatch(t.name, str(secondary_glob))]
    primary_by_key = {p.name: p for p in primary_files}
    secondary_by_key: Dict[str, Path] = {}
    for t in secondary_files:
        key = re.sub(r"_2(?=\\.tim$)", "", t.name, flags=re.IGNORECASE)
        secondary_by_key[key] = t

    pairs: List[Tuple[Path, Path]] = []
    for key, base in sorted(primary_by_key.items()):
        alt = secondary_by_key.get(key)
        if alt is not None:
            pairs.append((base, alt))

    changed_files: List[str] = []
    details: List[Dict[str, object]] = []
    total_affected = 0

    for primary, secondary in pairs:
        p_lines, p_recs = _parse_toa_records(primary)
        s_lines, s_recs = _parse_toa_records(secondary)
        p_dual = _dual_channel_epoch_map(p_recs, tol_days)
        s_dual = _dual_channel_epoch_map(s_recs, tol_days)
        p_dual_count = sum(len(v) for v in p_dual.values())
        s_dual_count = sum(len(v) for v in s_dual.values())

        if p_dual_count == 0 and s_dual_count == 0:
            details.append(
                {"pair": [primary.name, secondary.name], "skipped": "no_dual_epochs"}
            )
            continue
        if p_dual_count > s_dual_count:
            dual_file, dual_epochs = primary, p_dual
            single_file, single_lines, single_recs = secondary, s_lines, s_recs
        elif s_dual_count > p_dual_count:
            dual_file, dual_epochs = secondary, s_dual
            single_file, single_lines, single_recs = primary, p_lines, p_recs
        else:
            details.append(
                {"pair": [primary.name, secondary.name], "skipped": "ambiguous_dual_epochs"}
            )
            continue

        to_mark: List[int] = []
        for idx, token, mjd, _ in single_recs:
            epochs = dual_epochs.get(token)
            if not epochs:
                continue
            pos = int(np.searchsorted(np.array(epochs), mjd))
            neighbors = []
            if 0 <= pos < len(epochs):
                neighbors.append(epochs[pos])
            if pos - 1 >= 0:
                neighbors.append(epochs[pos - 1])
            if any(abs(mjd - e) <= tol_days for e in neighbors):
                to_mark.append(idx)

        if not to_mark:
            details.append(
                {
                    "pair": [primary.name, secondary.name],
                    "dual_file": dual_file.name,
                    "single_file": single_file.name,
                    "affected": 0,
                }
            )
            continue

        to_mark_set = set(to_mark)
        out_lines: List[str] = []
        for i, raw in enumerate(single_lines):
            if i not in to_mark_set:
                out_lines.append(cleanline(raw))
                continue
            total_affected += 1
            if mode == "delete":
                continue
            out_lines.append(
                f"{comment_prefix}(dual={dual_file.name},tol={mjd_tol_sec:.3g}s) "
                + cleanline(raw)
            )

        changed_files.append(str(single_file))
        details.append(
            {
                "pair": [primary.name, secondary.name],
                "dual_file": dual_file.name,
                "single_file": single_file.name,
                "affected": len(to_mark),
                "action": mode,
            }
        )
        if not dry_run and apply:
            if backup:
                _backup_file(single_file)
            single_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    result_key = "commented" if mode == "comment" else "deleted"
    return {
        "primary_glob": str(primary_glob),
        "secondary_glob": str(secondary_glob),
        "changed_files": changed_files,
        result_key: int(total_affected),
        "pairs": details,
    }


def wsrt_p2_prefer_dual_channel(
    psr_dir: Path,
    *,
    mjd_tol_sec: float = 0.99e-6,
    action: str = "comment",
    comment_prefix: str = "C WSRT_P2_PREFER_DUAL",
    apply: bool = False,
    backup: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Prefer dual-channel WSRT P2 TOAs over single-channel duplicates.

    Legacy wrapper around the generic ``prefer_multichannel_pair_rule``.
    """
    return prefer_multichannel_pair_rule(
        psr_dir,
        primary_glob="WSRT.P2.*.tim",
        secondary_glob="WSRT.P2.*_2.tim",
        mjd_tol_sec=mjd_tol_sec,
        action=action,
        comment_prefix=comment_prefix,
        apply=apply,
        backup=backup,
        dry_run=dry_run,
    )
