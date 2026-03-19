"""Optional PQC integration for outlier detection.

This module wraps the external ``pqc`` package to generate per-TOA quality
control flags and summary statistics. It is designed to fail gracefully when
``pqc`` or its dependencies are not installed.

See Also:
    pleb.pipeline.run_pipeline: Pipeline stage that invokes PQC.
    pleb.dataset_fix: Optional application of PQC flags to data.
"""

from __future__ import annotations

from dataclasses import asdict
from .compat import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import json
import subprocess
import sys
import os
import shutil
from fnmatch import fnmatch
from contextlib import contextmanager
import re

import pandas as pd
try:  # Python 3.11+
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore

from .logging_utils import get_logger

logger = get_logger("pleb.qc")


@dataclass(slots=True)
class PTAQCConfig:
    """Configuration for the optional pqc outlier detection stage.

    This stage is intentionally optional: the pipeline runs without ``pqc``
    installed and will skip QC if dependencies are missing.

    Attributes:
        backend_col: Backend grouping column used by PQC.
        drop_unmatched: Drop TOAs without matching tim metadata.
        merge_tol_seconds: Merge tolerance for matching (seconds).
        tau_corr_minutes: OU correlation time for bad-measurement detection.
        fdr_q: FDR threshold for bad-measurement detection.
        mark_only_worst_per_day: Mark only the worst TOA per day.
        tau_rec_days: Recovery time for transient scan (days).
        window_mult: Window multiplier for transient scan.
        min_points: Minimum points for transient scan.
        delta_chi2_thresh: Delta-chi2 threshold for transient scan.
        suppress_overlap: Suppress overlapping transient windows.
        exp_dip_min_duration_days: Minimum exponential dip duration (days).
        step_enabled: Enable step-change detection.
        step_min_points: Minimum points for step detection.
        step_delta_chi2_thresh: Delta-chi2 threshold for step detection.
        step_scope: Scope for step detection (global/branch/both).
        dm_step_enabled: Enable DM-like step detection.
        dm_step_min_points: Minimum points for DM step detection.
        dm_step_delta_chi2_thresh: Delta-chi2 threshold for DM step detection.
        dm_step_scope: Scope for DM step detection.
        robust_enabled: Enable robust MAD-based outlier detection.
        robust_z_thresh: Z threshold for robust outliers.
        robust_scope: Scope for robust outlier detection.
        add_orbital_phase: Add orbital-phase feature.
        add_solar_elongation: Add solar-elongation feature.
        add_elevation: Add elevation feature.
        add_airmass: Add airmass feature.
        add_parallactic_angle: Add parallactic-angle feature.
        add_freq_bin: Add frequency-bin feature.
        freq_bins: Number of frequency bins.
        observatory_path: Observatory file path for alt/az features.
        structure_mode: Feature-structure mode.
        structure_detrend_features: Features to detrend against.
        structure_test_features: Features to test for structure.
        structure_nbins: Bin count for structure tests.
        structure_min_per_bin: Minimum points per bin.
        structure_p_thresh: p-value threshold for structure detection.
        structure_circular_features: Features with circular topology.
        structure_group_cols: Grouping columns for structure tests.
        outlier_gate_enabled: Enable hard sigma gate for outliers.
        outlier_gate_sigma: Sigma threshold for outlier gate.
        outlier_gate_resid_col: Residual column for outlier gate.
        outlier_gate_sigma_col: Sigma column for outlier gate.
        event_instrument: Enable per-event membership diagnostics.
        gaussian_bump_enabled: Enable Gaussian-bump event detection.
        gaussian_bump_min_duration_days: Minimum bump duration in days.
        gaussian_bump_max_duration_days: Maximum bump duration in days.
        gaussian_bump_n_durations: Number of duration grid points.
        gaussian_bump_min_points: Minimum points for bump detection.
        gaussian_bump_delta_chi2_thresh: Delta-chi2 threshold for bump detection.
        gaussian_bump_suppress_overlap: Suppress overlapping bumps.
        gaussian_bump_member_eta: Per-point membership SNR threshold.
        gaussian_bump_freq_dependence: Fit 1/f^alpha dependence.
        gaussian_bump_freq_alpha_min: Lower bound for alpha.
        gaussian_bump_freq_alpha_max: Upper bound for alpha.
        gaussian_bump_freq_alpha_tol: Optimization tolerance for alpha.
        gaussian_bump_freq_alpha_max_iter: Max iterations for alpha optimizer.
        glitch_enabled: Enable glitch event detection.
        glitch_min_points: Minimum points for glitch detection.
        glitch_delta_chi2_thresh: Delta-chi2 threshold for glitch detection.
        glitch_suppress_overlap: Suppress overlapping glitches.
        glitch_member_eta: Per-point membership SNR threshold.
        glitch_peak_tau_days: Peak exponential timescale for glitch model.
        glitch_noise_k: Noise-aware threshold multiplier.
        glitch_mean_window_days: Rolling-mean window (days) for zero-crossing.
        glitch_min_duration_days: Minimum glitch duration (days).
        backend_profiles_path: Optional TOML file with per-backend pqc overrides.
    """

    backend_col: str = "group"
    drop_unmatched: bool = False

    # Merge tolerance (seconds) when matching libstempo arrays to tim metadata
    merge_tol_seconds: float = 2.0

    # Bad measurement detection (OU innovations)
    tau_corr_minutes: float = 30.0
    fdr_q: float = 0.01
    mark_only_worst_per_day: bool = True

    # Transient scan (jump + exp recovery)
    tau_rec_days: float = 7.0
    window_mult: float = 5.0
    min_points: int = 6
    delta_chi2_thresh: float = 25.0
    suppress_overlap: bool = True
    exp_dip_min_duration_days: float = 21.0

    # Step-change detection (global + per-backend)
    step_enabled: bool = True
    step_min_points: int = 20
    step_delta_chi2_thresh: float = 25.0
    step_scope: str = "both"

    # DM-like step detection (freq-scaled)
    dm_step_enabled: bool = True
    dm_step_min_points: int = 20
    dm_step_delta_chi2_thresh: float = 25.0
    dm_step_scope: str = "both"

    # Robust (MAD-based) outliers
    robust_enabled: bool = True
    robust_z_thresh: float = 5.0
    robust_scope: str = "both"

    # Feature extraction
    add_orbital_phase: bool = True
    add_solar_elongation: bool = True
    add_elevation: bool = False
    add_airmass: bool = False
    add_parallactic_angle: bool = False
    add_freq_bin: bool = False
    freq_bins: int = 8
    observatory_path: str | None = None

    # Feature-structure diagnostics/detrending
    structure_mode: str = "none"
    structure_detrend_features: list[str] | None = None
    structure_test_features: list[str] | None = None
    structure_nbins: int = 12
    structure_min_per_bin: int = 3
    structure_p_thresh: float = 0.01
    structure_circular_features: list[str] | None = None
    structure_group_cols: list[str] | None = None

    # Outlier gate (hard sigma)
    outlier_gate_enabled: bool = False
    outlier_gate_sigma: float = 3.0
    outlier_gate_resid_col: str | None = None
    outlier_gate_sigma_col: str | None = None
    event_instrument: bool = False

    # Solar-elongation events
    solar_events_enabled: bool = False
    solar_approach_max_deg: float = 30.0
    solar_min_points_global: int = 30
    solar_min_points_year: int = 10
    solar_min_points_near_zero: int = 3
    solar_tau_min_deg: float = 2.0
    solar_tau_max_deg: float = 60.0
    solar_member_eta: float = 1.0
    solar_freq_dependence: bool = True
    solar_freq_alpha_min: float = 0.0
    solar_freq_alpha_max: float = 4.0
    solar_freq_alpha_tol: float = 1e-3
    solar_freq_alpha_max_iter: int = 64
    # Orbital-phase based flagging
    orbital_phase_cut_enabled: bool = False
    orbital_phase_cut_center: float = 0.25
    orbital_phase_cut: float | None = None
    orbital_phase_cut_sigma: float = 3.0
    orbital_phase_cut_nbins: int = 18
    orbital_phase_cut_min_points: int = 20

    # Eclipse events
    eclipse_events_enabled: bool = False
    eclipse_center_phase: float = 0.25
    eclipse_min_points: int = 30
    eclipse_width_min: float = 0.01
    eclipse_width_max: float = 0.5
    eclipse_member_eta: float = 1.0
    eclipse_freq_dependence: bool = True
    eclipse_freq_alpha_min: float = 0.0
    eclipse_freq_alpha_max: float = 4.0
    eclipse_freq_alpha_tol: float = 1e-3
    eclipse_freq_alpha_max_iter: int = 64

    # Gaussian-bump events
    gaussian_bump_enabled: bool = False
    gaussian_bump_min_duration_days: float = 60.0
    gaussian_bump_max_duration_days: float = 1500.0
    gaussian_bump_n_durations: int = 6
    gaussian_bump_min_points: int = 20
    gaussian_bump_delta_chi2_thresh: float = 25.0
    gaussian_bump_suppress_overlap: bool = True
    gaussian_bump_member_eta: float = 1.0
    gaussian_bump_freq_dependence: bool = True
    gaussian_bump_freq_alpha_min: float = 0.0
    gaussian_bump_freq_alpha_max: float = 4.0
    gaussian_bump_freq_alpha_tol: float = 1e-3
    gaussian_bump_freq_alpha_max_iter: int = 64

    # Glitch events
    glitch_enabled: bool = False
    glitch_min_points: int = 30
    glitch_delta_chi2_thresh: float = 25.0
    glitch_suppress_overlap: bool = True
    glitch_member_eta: float = 1.0
    glitch_peak_tau_days: float = 30.0
    glitch_noise_k: float = 1.0
    glitch_mean_window_days: float = 180.0
    glitch_min_duration_days: float = 1000.0
    backend_profiles_path: str | None = None


@contextmanager
def _pushd(path: Path):
    """Temporarily change the working directory."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _load_backend_profiles(path: Path) -> list[tuple[str, Dict[str, Any]]]:
    """Load per-backend PQC override profiles from TOML.

    TOML format:
        [backend_profiles]
        "LOFAR.*" = { robust_z_thresh = 6.0, fdr_q = 0.02 }
        "WSRT.P2.334" = { delta_chi2_thresh = 18.0 }
    """
    if tomllib is None:
        raise RuntimeError("tomllib unavailable; Python 3.11+ required.")
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("backend_profiles", {})
    out: list[tuple[str, Dict[str, Any]]] = []
    if not isinstance(raw, dict):
        return out
    for patt, vals in raw.items():
        p = str(patt).strip()
        if not p or not isinstance(vals, dict):
            continue
        out.append((p, dict(vals)))
    return out


def _match_backend_overrides(
    backend_value: str, profiles: list[tuple[str, Dict[str, Any]]]
) -> Dict[str, Any]:
    """Resolve profile overrides for a backend value.

    Priority:
    1) exact match
    2) fnmatch glob match (in declaration order)
    """
    out: Dict[str, Any] = {}
    for patt, vals in profiles:
        if patt == backend_value:
            out.update(vals)
    if out:
        return out
    for patt, vals in profiles:
        if fnmatch(backend_value, patt):
            out.update(vals)
    return out


def _extract_flag_value(line: str, flag: str) -> Optional[str]:
    parts = line.strip().split()
    if flag not in parts:
        return None
    i = parts.index(flag)
    if i + 1 >= len(parts):
        return None
    return str(parts[i + 1])


def _is_toa_line(raw: str) -> bool:
    s = raw.strip()
    if not s:
        return False
    if s.startswith(("C", "#")):
        return False
    head = s.split()[0]
    return head not in {
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


def _prepare_backend_filtered_par(
    parfile: Path,
    backend_col: str,
    backend_value: str,
    work_dir: Path,
) -> Optional[Path]:
    """Create a temporary par/all.tim pair filtered to one backend value."""
    psr = parfile.stem
    alltim = parfile.with_name(f"{psr}_all.tim")
    if not alltim.exists():
        return None

    flag = backend_col if backend_col.startswith("-") else f"-{backend_col}"
    work_dir.mkdir(parents=True, exist_ok=True)

    tmp_par = work_dir / parfile.name
    tmp_all = work_dir / alltim.name
    tmp_par.write_text(parfile.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

    include_lines: list[str] = []
    direct_toas: list[str] = []
    for raw in alltim.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if s.startswith("INCLUDE"):
            parts = s.split(maxsplit=1)
            if len(parts) == 2:
                include_lines.append(parts[1].strip())
        elif _is_toa_line(raw):
            v = _extract_flag_value(raw, flag)
            if v == backend_value:
                direct_toas.append(raw.rstrip("\n"))

    filtered_includes: list[str] = []
    for rel in include_lines:
        src = alltim.parent / rel
        if not src.exists():
            continue
        kept: list[str] = []
        for raw in src.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not _is_toa_line(raw):
                kept.append(raw.rstrip("\n"))
                continue
            v = _extract_flag_value(raw, flag)
            if v == backend_value:
                kept.append(raw.rstrip("\n"))
        if not any(_is_toa_line(x) for x in kept):
            continue
        dst = work_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
        filtered_includes.append(rel)

    if not filtered_includes and not direct_toas:
        return None

    out_lines = ["FORMAT 1"]
    if direct_toas:
        direct_file = work_dir / "__direct.tim"
        direct_file.write_text("\n".join(direct_toas) + "\n", encoding="utf-8")
        out_lines.append("INCLUDE __direct.tim")
    for rel in filtered_includes:
        out_lines.append(f"INCLUDE {rel}")
    tmp_all.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return tmp_par


def _row_key_series(df: pd.DataFrame, backend_col: str) -> pd.Series:
    b = df.get(backend_col, pd.Series([""] * len(df), index=df.index)).astype(str)
    mjd = pd.to_numeric(df.get("mjd", pd.Series([pd.NA] * len(df), index=df.index)), errors="coerce").round(12)
    freq = pd.to_numeric(df.get("freq", pd.Series([pd.NA] * len(df), index=df.index)), errors="coerce").round(6)
    if "_timfile" in df.columns:
        t = df["_timfile"].astype(str)
    elif "filename" in df.columns:
        t = df["filename"].astype(str)
    else:
        t = pd.Series([""] * len(df), index=df.index)
    return b + "|" + mjd.astype(str) + "|" + freq.astype(str) + "|" + t


def run_pqc_for_parfile(
    parfile: Path,
    out_csv: Path,
    cfg: PTAQCConfig,
    settings_out: Optional[Path] = None,
) -> pd.DataFrame:
    """Run pqc on a pulsar parfile and write a CSV.

    Args:
        parfile: Path to ``<PSR>.par`` (expects sibling ``<PSR>_all.tim``).
        out_csv: Output CSV path.
        cfg: pqc configuration.

    Returns:
        The QC table produced by pqc.

    Raises:
        RuntimeError: If pqc cannot be imported.

    Examples:
        Run PQC for a pulsar and write CSV output::

            df = run_pqc_for_parfile(Path("J1234+5678.par"), Path("qc.csv"), PTAQCConfig())
    """
    parfile = Path(parfile)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        from pqc.pipeline import run_pipeline as qc_run  # type: ignore
        from pqc.config import (  # type: ignore
            BadMeasConfig,
            FeatureConfig,
            MergeConfig,
            StructureConfig,
            TransientConfig,
            StepConfig,
            RobustOutlierConfig,
            OutlierGateConfig,
            SolarCutConfig,
            OrbitalPhaseCutConfig,
            EclipseConfig,
            GaussianBumpConfig,
            GlitchConfig,
            ExpDipConfig,
        )

        # Sanity check: ensure pqc config classes are importable
        _ = (
            BadMeasConfig,
            FeatureConfig,
            MergeConfig,
            StructureConfig,
            TransientConfig,
            ExpDipConfig,
            StepConfig,
            RobustOutlierConfig,
            OutlierGateConfig,
            SolarCutConfig,
            OrbitalPhaseCutConfig,
            EclipseConfig,
            GaussianBumpConfig,
            GlitchConfig,
        )
        logger.info("pqc config classes loaded: %s", ",".join([c.__name__ for c in _]))
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pqc is not installed (or failed to import). Install your outlier package first, then rerun with run_pqc=true. "
            "If you're installing from a local zip/folder: pip install <path-to-pqc>."
        ) from e

    def _run_once(par_in: Path, cfg_in: PTAQCConfig, settings_path: Optional[Path]) -> pd.DataFrame:
        merge_cfg = MergeConfig(tol_days=float(cfg_in.merge_tol_seconds) / 86400.0)
        bad_cfg = BadMeasConfig(
            tau_corr_days=float(cfg_in.tau_corr_minutes) / (60.0 * 24.0),
            fdr_q=float(cfg_in.fdr_q),
            mark_only_worst_per_day=bool(cfg_in.mark_only_worst_per_day),
        )
        tr_cfg = TransientConfig(
            tau_rec_days=float(cfg_in.tau_rec_days),
            window_mult=float(cfg_in.window_mult),
            min_points=int(cfg_in.min_points),
            delta_chi2_thresh=float(cfg_in.delta_chi2_thresh),
            suppress_overlap=bool(cfg_in.suppress_overlap),
            instrument=bool(cfg_in.event_instrument),
        )
        dip_cfg = ExpDipConfig(min_duration_days=float(cfg_in.exp_dip_min_duration_days))
        feature_cfg = FeatureConfig(
            add_orbital_phase=bool(cfg_in.add_orbital_phase),
            add_solar_elongation=bool(cfg_in.add_solar_elongation),
            add_elevation=bool(cfg_in.add_elevation),
            add_airmass=bool(cfg_in.add_airmass),
            add_parallactic_angle=bool(cfg_in.add_parallactic_angle),
            add_freq_bin=bool(cfg_in.add_freq_bin),
            freq_bins=int(cfg_in.freq_bins),
            observatory_path=(str(cfg_in.observatory_path) if cfg_in.observatory_path else None),
        )
        struct_cfg = StructureConfig(
            mode=str(cfg_in.structure_mode),
            detrend_features=(
                tuple(cfg_in.structure_detrend_features)
                if cfg_in.structure_detrend_features
                else StructureConfig().detrend_features
            ),
            structure_features=(
                tuple(cfg_in.structure_test_features)
                if cfg_in.structure_test_features
                else StructureConfig().structure_features
            ),
            nbins=int(cfg_in.structure_nbins),
            min_per_bin=int(cfg_in.structure_min_per_bin),
            p_thresh=float(cfg_in.structure_p_thresh),
            circular_features=(
                tuple(cfg_in.structure_circular_features)
                if cfg_in.structure_circular_features
                else StructureConfig().circular_features
            ),
            structure_group_cols=(
                tuple(cfg_in.structure_group_cols) if cfg_in.structure_group_cols else None
            ),
        )

        step_cfg = StepConfig(
            enabled=bool(cfg_in.step_enabled),
            min_points=int(cfg_in.step_min_points),
            delta_chi2_thresh=float(cfg_in.step_delta_chi2_thresh),
            scope=str(cfg_in.step_scope),
            instrument=bool(cfg_in.event_instrument),
        )
        dm_cfg = StepConfig(
            enabled=bool(cfg_in.dm_step_enabled),
            min_points=int(cfg_in.dm_step_min_points),
            delta_chi2_thresh=float(cfg_in.dm_step_delta_chi2_thresh),
            scope=str(cfg_in.dm_step_scope),
            instrument=bool(cfg_in.event_instrument),
        )
        robust_cfg = RobustOutlierConfig(
            enabled=bool(cfg_in.robust_enabled),
            z_thresh=float(cfg_in.robust_z_thresh),
            scope=str(cfg_in.robust_scope),
        )
        gate_cfg = OutlierGateConfig(
            enabled=bool(cfg_in.outlier_gate_enabled),
            sigma_thresh=float(cfg_in.outlier_gate_sigma),
            resid_col=(
                str(cfg_in.outlier_gate_resid_col) if cfg_in.outlier_gate_resid_col else None
            ),
            sigma_col=(
                str(cfg_in.outlier_gate_sigma_col) if cfg_in.outlier_gate_sigma_col else None
            ),
        )
        solar_cfg = SolarCutConfig(
            enabled=bool(cfg_in.solar_events_enabled),
            approach_max_deg=float(cfg_in.solar_approach_max_deg),
            min_points_global=int(cfg_in.solar_min_points_global),
            min_points_year=int(cfg_in.solar_min_points_year),
            min_points_near_zero=int(cfg_in.solar_min_points_near_zero),
            tau_min_deg=float(cfg_in.solar_tau_min_deg),
            tau_max_deg=float(cfg_in.solar_tau_max_deg),
            member_eta=float(cfg_in.solar_member_eta),
            freq_dependence=bool(cfg_in.solar_freq_dependence),
            freq_alpha_min=float(cfg_in.solar_freq_alpha_min),
            freq_alpha_max=float(cfg_in.solar_freq_alpha_max),
            freq_alpha_tol=float(cfg_in.solar_freq_alpha_tol),
            freq_alpha_max_iter=int(cfg_in.solar_freq_alpha_max_iter),
        )
        orbital_cfg = OrbitalPhaseCutConfig(
            enabled=bool(cfg_in.orbital_phase_cut_enabled),
            center_phase=float(cfg_in.orbital_phase_cut_center),
            limit_phase=(
                float(cfg_in.orbital_phase_cut) if cfg_in.orbital_phase_cut is not None else None
            ),
            sigma_thresh=float(cfg_in.orbital_phase_cut_sigma),
            nbins=int(cfg_in.orbital_phase_cut_nbins),
            min_points=int(cfg_in.orbital_phase_cut_min_points),
        )
        eclipse_cfg = EclipseConfig(
            enabled=bool(cfg_in.eclipse_events_enabled),
            center_phase=float(cfg_in.eclipse_center_phase),
            min_points=int(cfg_in.eclipse_min_points),
            width_min=float(cfg_in.eclipse_width_min),
            width_max=float(cfg_in.eclipse_width_max),
            member_eta=float(cfg_in.eclipse_member_eta),
            freq_dependence=bool(cfg_in.eclipse_freq_dependence),
            freq_alpha_min=float(cfg_in.eclipse_freq_alpha_min),
            freq_alpha_max=float(cfg_in.eclipse_freq_alpha_max),
            freq_alpha_tol=float(cfg_in.eclipse_freq_alpha_tol),
            freq_alpha_max_iter=int(cfg_in.eclipse_freq_alpha_max_iter),
        )
        bump_cfg = GaussianBumpConfig(
            enabled=bool(cfg_in.gaussian_bump_enabled),
            min_duration_days=float(cfg_in.gaussian_bump_min_duration_days),
            max_duration_days=float(cfg_in.gaussian_bump_max_duration_days),
            n_durations=int(cfg_in.gaussian_bump_n_durations),
            min_points=int(cfg_in.gaussian_bump_min_points),
            delta_chi2_thresh=float(cfg_in.gaussian_bump_delta_chi2_thresh),
            suppress_overlap=bool(cfg_in.gaussian_bump_suppress_overlap),
            member_eta=float(cfg_in.gaussian_bump_member_eta),
            freq_dependence=bool(cfg_in.gaussian_bump_freq_dependence),
            freq_alpha_min=float(cfg_in.gaussian_bump_freq_alpha_min),
            freq_alpha_max=float(cfg_in.gaussian_bump_freq_alpha_max),
            freq_alpha_tol=float(cfg_in.gaussian_bump_freq_alpha_tol),
            freq_alpha_max_iter=int(cfg_in.gaussian_bump_freq_alpha_max_iter),
        )
        glitch_cfg = GlitchConfig(
            enabled=bool(cfg_in.glitch_enabled),
            min_points=int(cfg_in.glitch_min_points),
            delta_chi2_thresh=float(cfg_in.glitch_delta_chi2_thresh),
            suppress_overlap=bool(cfg_in.glitch_suppress_overlap),
            member_eta=float(cfg_in.glitch_member_eta),
            peak_tau_days=float(cfg_in.glitch_peak_tau_days),
            noise_k=float(cfg_in.glitch_noise_k),
            mean_window_days=float(cfg_in.glitch_mean_window_days),
            min_duration_days=float(cfg_in.glitch_min_duration_days),
        )
        if settings_path is None:
            settings_path = out_csv.parent / "run_settings" / f"{par_in.stem}.pqc_settings.toml"
        settings_path = Path(settings_path)
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with _pushd(out_csv.parent):
            return qc_run(
                par_in,
                backend_col=str(cfg_in.backend_col),
                bad_cfg=bad_cfg,
                tr_cfg=tr_cfg,
                dip_cfg=dip_cfg,
                merge_cfg=merge_cfg,
                feature_cfg=feature_cfg,
                struct_cfg=struct_cfg,
                step_cfg=step_cfg,
                dm_cfg=dm_cfg,
                robust_cfg=robust_cfg,
                gate_cfg=gate_cfg,
                solar_cfg=solar_cfg,
                orbital_cfg=orbital_cfg,
                eclipse_cfg=eclipse_cfg,
                bump_cfg=bump_cfg,
                glitch_cfg=glitch_cfg,
                drop_unmatched=bool(cfg_in.drop_unmatched),
                settings_out=settings_path,
            )

    df = _run_once(parfile, cfg, settings_out)

    if cfg.backend_profiles_path:
        prof_path = Path(str(cfg.backend_profiles_path))
        profiles = _load_backend_profiles(prof_path)
        if profiles:
            if str(cfg.backend_col) not in df.columns:
                logger.warning(
                    "pqc backend profiles requested, but backend column '%s' is missing.",
                    cfg.backend_col,
                )
            else:
                key_base = _row_key_series(df, str(cfg.backend_col))
                idx_by_key = {k: i for i, k in enumerate(key_base.astype(str))}
                backends = sorted({str(x) for x in df[str(cfg.backend_col)].dropna().unique()})
                for be in backends:
                    overrides = _match_backend_overrides(be, profiles)
                    if not overrides:
                        continue
                    payload = asdict(cfg)
                    payload.update(overrides)
                    payload["backend_profiles_path"] = None
                    cfg_be = PTAQCConfig(**payload)
                    tmp = out_csv.parent / ".pqc_backend_profiles" / parfile.stem / re.sub(r"[^A-Za-z0-9._-]+", "_", be)
                    tmp_par = _prepare_backend_filtered_par(parfile, str(cfg.backend_col), be, tmp)
                    if tmp_par is None:
                        continue
                    be_settings = out_csv.parent / "run_settings" / f"{parfile.stem}.{be}.pqc_settings.toml"
                    try:
                        df_be = _run_once(tmp_par, cfg_be, be_settings)
                    except Exception as e:
                        logger.warning("pqc backend profile run failed for %s: %s", be, e)
                        continue
                    key_be = _row_key_series(df_be, str(cfg.backend_col)).astype(str)
                    shared_cols = [c for c in df_be.columns if c in df.columns]
                    matched = 0
                    for j, k in enumerate(key_be):
                        i = idx_by_key.get(k)
                        if i is None:
                            continue
                        matched += 1
                        for c in shared_cols:
                            df.iat[i, df.columns.get_loc(c)] = df_be.iat[j, df_be.columns.get_loc(c)]
                    logger.info(
                        "Applied pqc backend profile for %s: overrides=%s matched_rows=%d",
                        be,
                        ",".join(sorted(overrides.keys())),
                        matched,
                    )

    _assert_timfile_metadata(df, source=str(parfile))
    df.to_csv(out_csv, index=False)
    return df


def run_pqc_for_parfile_subprocess(
    parfile: Path,
    out_csv: Path,
    cfg: PTAQCConfig,
    timeout: Optional[float] = None,
    settings_out: Optional[Path] = None,
) -> pd.DataFrame:
    """Run pqc in a subprocess to isolate segfaults from libstempo.

    Args:
        parfile: Path to ``<PSR>.par`` (expects sibling ``<PSR>_all.tim``).
        out_csv: Output CSV path.
        cfg: pqc configuration.
        timeout: Optional timeout in seconds.

    Returns:
        QC dataframe read back from the CSV.

    Raises:
        RuntimeError: If the subprocess fails or does not write an output CSV.
    """
    parfile = Path(parfile)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "parfile": str(parfile),
        "out_csv": str(out_csv),
        "cfg": asdict(cfg),
        "settings_out": (str(settings_out) if settings_out is not None else None),
    }
    payload_path = out_csv.parent / f".pqc_{parfile.stem}.json"
    # Config may contain Path objects (e.g. observatory/profile paths).
    payload_path.write_text(json.dumps(payload, default=str), encoding="utf-8")

    code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from pleb.outlier_qc import PTAQCConfig, run_pqc_for_parfile\n"
        "payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "cfg = PTAQCConfig(**payload['cfg'])\n"
        "settings_out = payload.get('settings_out')\n"
        "run_pqc_for_parfile(Path(payload['parfile']), Path(payload['out_csv']), cfg, "
        "settings_out=(Path(settings_out) if settings_out else None))\n"
    )
    try:
        env = os.environ.copy()
        t2 = shutil.which("tempo2")
        if t2:
            t2_dir = str(Path(t2).resolve().parent)
            cur_path = env.get("PATH", "")
            if t2_dir not in cur_path.split(":"):
                env["PATH"] = f"{t2_dir}:{cur_path}" if cur_path else t2_dir
        proc = subprocess.run(
            [sys.executable, "-c", code, str(payload_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    finally:
        try:
            payload_path.unlink()
        except Exception:
            pass

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = f"pqc subprocess failed with code {proc.returncode}"
        if stderr:
            msg += f"; stderr: {stderr}"
        if stdout:
            msg += f"; stdout: {stdout}"
        raise RuntimeError(msg)

    if not out_csv.exists():
        raise RuntimeError("pqc subprocess completed but output CSV was not created.")

    string_cols = ["be", "fe", "f", "tmplt", "flag", "B", "g", "h"]
    dtype_map = {c: "string" for c in string_cols}
    df = pd.read_csv(out_csv, dtype=dtype_map)
    _assert_timfile_metadata(df, source=str(parfile))
    return df


def _assert_timfile_metadata(df: pd.DataFrame, source: str) -> None:
    """Ensure every row has timfile metadata."""
    if "_timfile" in df.columns:
        missing = int(df["_timfile"].isna().sum())
        if missing:
            sample = df.loc[df["_timfile"].isna(), ["mjd", "freq"]].head(5)
            raise RuntimeError(
                f"{source}: {missing} rows missing _timfile metadata. Sample:\n"
                f"{sample.to_string(index=False)}"
            )
        return
    if "filename" in df.columns:
        missing = int(df["filename"].isna().sum())
        if missing:
            sample = df.loc[df["filename"].isna(), ["mjd", "freq"]].head(5)
            raise RuntimeError(
                f"{source}: {missing} rows missing filename metadata. Sample:\n"
                f"{sample.to_string(index=False)}"
            )
        return
    raise RuntimeError(
        f"{source}: no timfile metadata columns found (_timfile/filename)."
    )


def summarize_pqc(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a compact summary of a pqc output dataframe.

    Args:
        df: QC output dataframe.

    Returns:
        Summary dictionary with counts of TOAs and flagged items.

    Examples:
        Summarize PQC output::

            summary = summarize_pqc(df)
    """
    out: Dict[str, Any] = {"n_toas": int(len(df))}
    if "bad" in df.columns:
        out["n_bad"] = int(df["bad"].fillna(False).sum())
    if "bad_day" in df.columns and "day" in df.columns:
        out["n_bad_days"] = int(df.loc[df["bad_day"].fillna(False), "day"].nunique())
    if "transient_id" in df.columns:
        out["n_transient_toas"] = int(
            (df["transient_id"].fillna(-1).astype(int) != -1).sum()
        )
        out["n_transients"] = int(
            df.loc[
                df["transient_id"].fillna(-1).astype(int) != -1, "transient_id"
            ].nunique()
        )
    if "solar_event_member" in df.columns:
        out["n_solar_events"] = int(df["solar_event_member"].fillna(False).sum())
    if "orbital_phase_bad" in df.columns:
        out["n_orbital_phase_bad"] = int(df["orbital_phase_bad"].fillna(False).sum())
    return out
