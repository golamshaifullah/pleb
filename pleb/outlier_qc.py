"""Optional pqc integration for outlier detection."""

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

    This stage is intentionally optional: the pipeline runs without pqc
    installed and will skip QC if dependencies are missing.
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
            StepConfig, RobustOutlierConfig,
        )
        # Sanity check: ensure pqc config classes are importable
        _ = (BadMeasConfig, FeatureConfig, MergeConfig, StructureConfig, TransientConfig, StepConfig, RobustOutlierConfig)
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
        detrend_features=tuple(cfg.structure_detrend_features) if cfg.structure_detrend_features else StructureConfig().detrend_features,
        structure_features=tuple(cfg.structure_test_features) if cfg.structure_test_features else StructureConfig().structure_features,
        nbins=int(cfg.structure_nbins),
        min_per_bin=int(cfg.structure_min_per_bin),
        p_thresh=float(cfg.structure_p_thresh),
        circular_features=tuple(cfg.structure_circular_features) if cfg.structure_circular_features else StructureConfig().circular_features,
        structure_group_cols=tuple(cfg.structure_group_cols) if cfg.structure_group_cols else None,
    )

    step_cfg = StepConfig(
        enabled=bool(cfg.step_enabled),
        min_points=int(cfg.step_min_points),
        delta_chi2_thresh=float(cfg.step_delta_chi2_thresh),
        scope=str(cfg.step_scope),
    )
    dm_cfg = StepConfig(
        enabled=bool(cfg.dm_step_enabled),
        min_points=int(cfg.dm_step_min_points),
        delta_chi2_thresh=float(cfg.dm_step_delta_chi2_thresh),
        scope=str(cfg.dm_step_scope),
    )
    robust_cfg = RobustOutlierConfig(
        enabled=bool(cfg.robust_enabled),
        z_thresh=float(cfg.robust_z_thresh),
        scope=str(cfg.robust_scope),
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
            drop_unmatched=bool(cfg.drop_unmatched),
        )

    df.to_csv(out_csv, index=False)
    return df


def run_pqc_for_parfile_subprocess(parfile: Path, out_csv: Path, cfg: PTAQCConfig, timeout: Optional[float] = None) -> pd.DataFrame:
    """Run pqc in a subprocess to isolate segfaults from libstempo.

    If the subprocess fails, raise RuntimeError with stderr for logging.
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
    return pd.read_csv(out_csv, dtype=dtype_map)


def summarize_pqc(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a compact summary of a pqc output dataframe.

    Args:
        df: QC output dataframe.

    Returns:
        Summary dictionary with counts of TOAs and flagged items.
    """
    out: Dict[str, Any] = {"n_toas": int(len(df))}
    if "bad" in df.columns:
        out["n_bad"] = int(df["bad"].fillna(False).sum())
    if "bad_day" in df.columns and "day" in df.columns:
        out["n_bad_days"] = int(df.loc[df["bad_day"].fillna(False), "day"].nunique())
    if "transient_id" in df.columns:
        out["n_transient_toas"] = int((df["transient_id"].fillna(-1).astype(int) != -1).sum())
        out["n_transients"] = int(df.loc[df["transient_id"].fillna(-1).astype(int) != -1, "transient_id"].nunique())
    return out
