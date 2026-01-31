"""Optional PQC integration for outlier detection.

This module wraps the external ``pqc`` package to generate per-TOA quality
control flags and summary statistics. It is designed to fail gracefully when
``pqc`` or its dependencies are not installed.

See Also:
    pleb.pipeline.run_pipeline: Pipeline stage that invokes PQC.
    pleb.dataset_fix: Optional application of PQC flags to data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import json
import subprocess
import sys
import os
from contextlib import contextmanager

import pandas as pd

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


def run_pqc_for_parfile(parfile: Path, out_csv: Path, cfg: PTAQCConfig) -> pd.DataFrame:
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
        )

        # Sanity check: ensure pqc config classes are importable
        _ = (
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
        )
        logger.info("pqc config classes loaded: %s", ",".join([c.__name__ for c in _]))
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pqc is not installed (or failed to import). Install your outlier package first, then rerun with run_pqc=true. "
            "If you're installing from a local zip/folder: pip install <path-to-pqc>."
        ) from e

    merge_cfg = MergeConfig(tol_days=float(cfg.merge_tol_seconds) / 86400.0)
    bad_cfg = BadMeasConfig(
        tau_corr_days=float(cfg.tau_corr_minutes) / (60.0 * 24.0),
        fdr_q=float(cfg.fdr_q),
        mark_only_worst_per_day=bool(cfg.mark_only_worst_per_day),
    )
    tr_cfg = TransientConfig(
        tau_rec_days=float(cfg.tau_rec_days),
        window_mult=float(cfg.window_mult),
        min_points=int(cfg.min_points),
        delta_chi2_thresh=float(cfg.delta_chi2_thresh),
        suppress_overlap=bool(cfg.suppress_overlap),
        instrument=bool(cfg.event_instrument),
    )
    feature_cfg = FeatureConfig(
        add_orbital_phase=bool(cfg.add_orbital_phase),
        add_solar_elongation=bool(cfg.add_solar_elongation),
        add_elevation=bool(cfg.add_elevation),
        add_airmass=bool(cfg.add_airmass),
        add_parallactic_angle=bool(cfg.add_parallactic_angle),
        add_freq_bin=bool(cfg.add_freq_bin),
        freq_bins=int(cfg.freq_bins),
        observatory_path=(str(cfg.observatory_path) if cfg.observatory_path else None),
    )
    struct_cfg = StructureConfig(
        mode=str(cfg.structure_mode),
        detrend_features=(
            tuple(cfg.structure_detrend_features)
            if cfg.structure_detrend_features
            else StructureConfig().detrend_features
        ),
        structure_features=(
            tuple(cfg.structure_test_features)
            if cfg.structure_test_features
            else StructureConfig().structure_features
        ),
        nbins=int(cfg.structure_nbins),
        min_per_bin=int(cfg.structure_min_per_bin),
        p_thresh=float(cfg.structure_p_thresh),
        circular_features=(
            tuple(cfg.structure_circular_features)
            if cfg.structure_circular_features
            else StructureConfig().circular_features
        ),
        structure_group_cols=(
            tuple(cfg.structure_group_cols) if cfg.structure_group_cols else None
        ),
    )

    step_cfg = StepConfig(
        enabled=bool(cfg.step_enabled),
        min_points=int(cfg.step_min_points),
        delta_chi2_thresh=float(cfg.step_delta_chi2_thresh),
        scope=str(cfg.step_scope),
        instrument=bool(cfg.event_instrument),
    )
    dm_cfg = StepConfig(
        enabled=bool(cfg.dm_step_enabled),
        min_points=int(cfg.dm_step_min_points),
        delta_chi2_thresh=float(cfg.dm_step_delta_chi2_thresh),
        scope=str(cfg.dm_step_scope),
        instrument=bool(cfg.event_instrument),
    )
    robust_cfg = RobustOutlierConfig(
        enabled=bool(cfg.robust_enabled),
        z_thresh=float(cfg.robust_z_thresh),
        scope=str(cfg.robust_scope),
    )
    gate_cfg = OutlierGateConfig(
        enabled=bool(cfg.outlier_gate_enabled),
        sigma_thresh=float(cfg.outlier_gate_sigma),
        resid_col=(
            str(cfg.outlier_gate_resid_col) if cfg.outlier_gate_resid_col else None
        ),
        sigma_col=(
            str(cfg.outlier_gate_sigma_col) if cfg.outlier_gate_sigma_col else None
        ),
    )
    solar_cfg = SolarCutConfig(
        enabled=bool(cfg.solar_events_enabled),
        approach_max_deg=float(cfg.solar_approach_max_deg),
        min_points_global=int(cfg.solar_min_points_global),
        min_points_year=int(cfg.solar_min_points_year),
        min_points_near_zero=int(cfg.solar_min_points_near_zero),
        tau_min_deg=float(cfg.solar_tau_min_deg),
        tau_max_deg=float(cfg.solar_tau_max_deg),
        member_eta=float(cfg.solar_member_eta),
        freq_dependence=bool(cfg.solar_freq_dependence),
        freq_alpha_min=float(cfg.solar_freq_alpha_min),
        freq_alpha_max=float(cfg.solar_freq_alpha_max),
        freq_alpha_tol=float(cfg.solar_freq_alpha_tol),
        freq_alpha_max_iter=int(cfg.solar_freq_alpha_max_iter),
    )
    orbital_cfg = OrbitalPhaseCutConfig(
        enabled=bool(cfg.orbital_phase_cut_enabled),
        center_phase=float(cfg.orbital_phase_cut_center),
        limit_phase=(
            float(cfg.orbital_phase_cut) if cfg.orbital_phase_cut is not None else None
        ),
        sigma_thresh=float(cfg.orbital_phase_cut_sigma),
        nbins=int(cfg.orbital_phase_cut_nbins),
        min_points=int(cfg.orbital_phase_cut_min_points),
    )

    # libstempo/tempo2 sometimes emit scratch outputs in the CWD; isolate per pulsar.
    with _pushd(out_csv.parent):
        df = qc_run(
            parfile,
            backend_col=str(cfg.backend_col),
            bad_cfg=bad_cfg,
            tr_cfg=tr_cfg,
            merge_cfg=merge_cfg,
            feature_cfg=feature_cfg,
            struct_cfg=struct_cfg,
            step_cfg=step_cfg,
            dm_cfg=dm_cfg,
            robust_cfg=robust_cfg,
            gate_cfg=gate_cfg,
            solar_cfg=solar_cfg,
            orbital_cfg=orbital_cfg,
            drop_unmatched=bool(cfg.drop_unmatched),
        )

    _assert_timfile_metadata(df, source=str(parfile))
    df.to_csv(out_csv, index=False)
    return df


def run_pqc_for_parfile_subprocess(
    parfile: Path, out_csv: Path, cfg: PTAQCConfig, timeout: Optional[float] = None
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
    }
    payload_path = out_csv.parent / f".pqc_{parfile.stem}.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from pleb.outlier_qc import PTAQCConfig, run_pqc_for_parfile\n"
        "payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "cfg = PTAQCConfig(**payload['cfg'])\n"
        "run_pqc_for_parfile(Path(payload['parfile']), Path(payload['out_csv']), cfg)\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, str(payload_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
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
    raise RuntimeError(f"{source}: no timfile metadata columns found (_timfile/filename).")


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
