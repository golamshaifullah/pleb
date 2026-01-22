from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import os
from contextlib import contextmanager

import pandas as pd

from .logging_utils import get_logger

logger = get_logger("data_combination_pipeline.qc")


@dataclass(slots=True)
class PTAQCConfig:
    """Configuration for the optional pta_qc outlier detection stage.

    This stage is intentionally optional: the pipeline will run without pta_qc installed.
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


@contextmanager
def _pushd(path: Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def run_pta_qc_for_parfile(parfile: Path, out_csv: Path, cfg: PTAQCConfig) -> pd.DataFrame:
    """Run pta_qc on a pulsar parfile and write a CSV.

    Parameters
    ----------
    parfile:
        Path to <PSR>.par. Expects sibling <PSR>_all.tim.
    out_csv:
        Where to write the CSV output.
    cfg:
        PTAQCConfig options.

    Returns
    -------
    pd.DataFrame
        The QC table produced by pta_qc.
    """
    parfile = Path(parfile)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        from pta_qc.pipeline import run_pipeline as qc_run  # type: ignore
        from pta_qc.config import BadMeasConfig, TransientConfig, MergeConfig  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pta_qc is not installed (or failed to import). Install your outlier package first, then rerun with run_pta_qc=true. "
            "If you're installing from a local zip/folder: pip install <path-to-pta_qc>."
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

    # libstempo/tempo2 sometimes emit scratch outputs in the CWD; isolate per pulsar.
    with _pushd(out_csv.parent):
        df = qc_run(
            parfile,
            backend_col=str(cfg.backend_col),
            bad_cfg=bad_cfg,
            tr_cfg=tr_cfg,
            merge_cfg=merge_cfg,
            drop_unmatched=bool(cfg.drop_unmatched),
        )

    df.to_csv(out_csv, index=False)
    return df


def summarize_pta_qc(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a tiny summary dict from a pta_qc output dataframe."""
    out: Dict[str, Any] = {"n_toas": int(len(df))}
    if "bad" in df.columns:
        out["n_bad"] = int(df["bad"].fillna(False).sum())
    if "bad_day" in df.columns and "day" in df.columns:
        out["n_bad_days"] = int(df.loc[df["bad_day"].fillna(False), "day"].nunique())
    if "transient_id" in df.columns:
        out["n_transient_toas"] = int((df["transient_id"].fillna(-1).astype(int) != -1).sum())
        out["n_transients"] = int(df.loc[df["transient_id"].fillna(-1).astype(int) != -1, "transient_id"].nunique())
    return out
