"""Orchestrate the data-combination pipeline end to end.

This module coordinates git branch management, tempo2 runs, report generation,
and optional quality-control steps. It stitches together the core building
blocks in :mod:`pleb.tempo2`, :mod:`pleb.reports`, and :mod:`pleb.dataset_fix`.

See Also:
    pleb.config.PipelineConfig: Primary configuration model.
    pleb.param_scan.run_param_scan: Fit-only parameter scan workflow.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, List

import os
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover

    def tqdm(x, **kwargs):
        return x


from .config import PipelineConfig
from .git_tools import checkout, require_clean_repo
from .logging_utils import get_logger, set_log_dir
from .plotting import (
    plot_covmat_heatmaps,
    plot_pulsars_per_system,
    plot_residuals,
    plot_systems_per_pulsar,
)
from .reports import (
    write_change_reports,
    write_model_comparison_summary,
    write_new_param_significance,
    write_outlier_tables,
)
from .run_report import generate_run_report
from .tempo2 import run_tempo2_for_pulsar
from .tim_utils import count_toa_lines, parse_include_lines, list_backend_timfiles
from .utils import (
    discover_pulsars,
    make_output_tree,
    which_or_raise,
    cleanup_empty_dirs,
    remove_tree_if_exists,
)

# Add-ons from FixDataset.ipynb / AnalysePulsars.ipynb
from .dataset_fix import (
    FixDatasetConfig,
    _find_qc_csv,
    apply_fixdataset_branch,
    fix_pulsar_dataset,
    write_fix_report,
)
from .outlier_qc import PTAQCConfig, run_pqc_for_parfile_subprocess, summarize_pqc
from .pulsar_analysis import analyse_binary_from_par, BinaryAnalysisConfig
from .qc_report import generate_cross_pulsar_coincidence_report, generate_qc_report
from .whitenoise_integration import (
    WhiteNoiseStageConfig,
    estimate_white_noise_for_pulsar,
    resolve_timfile_for_pulsar,
)

logger = get_logger("pleb")


def _discover_pqc_variants(psr_dir: Path, psr: str) -> List[str]:
    """Discover available variant include files for one pulsar.

    Parameters
    ----------
    psr_dir : pathlib.Path
        Pulsar directory.
    psr : str
        Pulsar name.

    Returns
    -------
    list of str
        Sorted variant names (for example ``["legacy", "new"]``). The base
        include ``<PSR>_all.tim`` is not returned.
    """
    out: List[str] = []
    seen: set[str] = set()
    for p in sorted(
        {
            *psr_dir.glob(f"{psr}_*_all.tim"),
            *psr_dir.glob(f"{psr}_all.*.tim"),
        }
    ):
        name = p.name
        pref_us = f"{psr}_"
        suff_us = "_all.tim"
        if name.startswith(pref_us) and name.endswith(suff_us):
            v = name[len(pref_us) : -len(suff_us)]
            if v and v != "all" and v not in seen:
                out.append(v)
                seen.add(v)
            continue
        pref = f"{psr}_all."
        if not name.startswith(pref) or not name.endswith(".tim"):
            continue
        v = name[len(pref) : -len(".tim")]
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _prepare_variant_pqc_workspace(
    psr_dir: Path,
    psr: str,
    variant: str,
    workspace: Path,
) -> Path:
    """Prepare a temporary per-variant workspace for PQC input discovery.

    The PQC runner expects ``<PSR>.par`` and sibling ``<PSR>_all.tim``. This
    workspace provides those names while reusing the pulsar ``tims/`` folder.
    """
    workspace.mkdir(parents=True, exist_ok=True)

    variant_par = psr_dir / f"{psr}_{variant}.par"
    if not variant_par.exists():
        variant_par = psr_dir / f"{psr}.{variant}.par"
    base_par = psr_dir / f"{psr}.par"
    src_par = variant_par if variant_par.exists() else base_par
    if not src_par.exists():
        raise FileNotFoundError(str(src_par))

    target_par = workspace / f"{psr}.par"
    shutil.copy2(src_par, target_par)

    src_all = psr_dir / f"{psr}_{variant}_all.tim"
    if not src_all.exists():
        src_all = psr_dir / f"{psr}_all.{variant}.tim"
    if not src_all.exists():
        raise FileNotFoundError(str(src_all))
    target_all = workspace / f"{psr}_all.tim"
    shutil.copy2(src_all, target_all)

    link_tims = workspace / "tims"
    if link_tims.exists() or link_tims.is_symlink():
        if link_tims.is_symlink() or link_tims.is_file():
            link_tims.unlink(missing_ok=True)
        else:
            shutil.rmtree(link_tims, ignore_errors=True)
    try:
        link_tims.symlink_to(psr_dir / "tims", target_is_directory=True)
    except Exception:
        # Fallback for filesystems without symlink support.
        shutil.copytree(psr_dir / "tims", link_tims, dirs_exist_ok=True)

    return target_par


def _variant_alltim_toa_count(alltim: Path) -> int:
    """Return the total TOA count referenced by one variant ``*_all.tim``."""
    total = 0
    base_dir = alltim.parent
    for rel in sorted(parse_include_lines(alltim)):
        try:
            inc = (base_dir / rel).resolve()
        except Exception:
            inc = base_dir / rel
        total += count_toa_lines(inc)
    return total


def _canonical_timfile_keys(df: pd.DataFrame) -> pd.Series:
    if "_timfile_base" in df.columns:
        return df["_timfile_base"].fillna("").astype(str)
    if "_timfile" in df.columns:
        return df["_timfile"].fillna("").astype(str).map(
            lambda x: Path(x).name if x else ""
        )
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def _rows_near_any(targets: np.ndarray, refs: np.ndarray, tol_days: float) -> np.ndarray:
    if targets.size == 0 or refs.size == 0:
        return np.zeros(targets.shape, dtype=bool)
    refs = np.asarray(refs, dtype=float)
    refs.sort()
    pos = np.searchsorted(refs, targets)
    out = np.zeros(targets.shape, dtype=bool)
    left = pos - 1
    left_ok = (left >= 0) & (left < len(refs))
    if left_ok.any():
        out[left_ok] |= np.abs(refs[left[left_ok]] - targets[left_ok]) <= tol_days
    right_ok = (pos >= 0) & (pos < len(refs))
    if right_ok.any():
        out[right_ok] |= np.abs(refs[pos[right_ok]] - targets[right_ok]) <= tol_days
    return out


def _homogenize_variant_outlier_flags(
    qc_rows: List[Dict[str, object]],
    *,
    outlier_cols: List[str],
    tol_seconds: float,
) -> None:
    """Union outlier-like flags across successful variant QC CSVs for each pulsar."""
    if tol_seconds <= 0:
        tol_seconds = 1e-6
    tol_days = float(tol_seconds) / 86400.0
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in qc_rows:
        if str(row.get("qc_status", "")) != "success":
            continue
        variant = str(row.get("variant", ""))
        qc_csv = str(row.get("qc_csv", ""))
        if not variant or variant == "base" or not qc_csv:
            continue
        grouped.setdefault(str(row.get("pulsar", "")), []).append(row)

    total_csvs_changed = 0
    total_cells_changed = 0
    summary_updates: Dict[tuple[str, str], Dict[str, object]] = {}
    for pulsar, rows in grouped.items():
        if len(rows) < 2:
            continue
        frames: Dict[tuple[str, str], pd.DataFrame] = {}
        csv_by_key: Dict[tuple[str, str], Path] = {}
        per_col_refs: Dict[str, Dict[str, np.ndarray]] = {}
        candidate_cols: set[str] = set()
        for row in rows:
            key = (str(row.get("pulsar", "")), str(row.get("variant", "")))
            csv_path = Path(str(row.get("qc_csv", "")))
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, low_memory=False)
            if "mjd" not in df.columns:
                continue
            df["__timkey__"] = _canonical_timfile_keys(df)
            frames[key] = df
            csv_by_key[key] = csv_path
            candidate_cols.update(c for c in outlier_cols if c in df.columns)
        if not frames or not candidate_cols:
            continue

        for col in sorted(candidate_cols):
            refs_by_tim: Dict[str, List[float]] = {}
            for df in frames.values():
                mask = df[col].fillna(False).astype(bool).to_numpy()
                if not mask.any():
                    continue
                sub = df.loc[mask, ["__timkey__", "mjd"]]
                for timkey, grp in sub.groupby("__timkey__"):
                    refs_by_tim.setdefault(str(timkey), []).extend(
                        float(x) for x in grp["mjd"].to_numpy()
                    )
            per_col_refs[col] = {
                timkey: np.unique(np.asarray(vals, dtype=float))
                for timkey, vals in refs_by_tim.items()
                if vals
            }

        for key, df in frames.items():
            changed = False
            for col, refs_by_tim in per_col_refs.items():
                if col not in df.columns:
                    continue
                current = df[col].fillna(False).astype(bool).to_numpy()
                updated = current.copy()
                for timkey, idx in df.groupby("__timkey__").indices.items():
                    refs = refs_by_tim.get(str(timkey))
                    if refs is None or len(refs) == 0:
                        continue
                    mjds = df.iloc[idx]["mjd"].to_numpy(dtype=float)
                    updated[idx] |= _rows_near_any(mjds, refs, tol_days)
                if not np.array_equal(current, updated):
                    df[col] = updated
                    total_cells_changed += int((updated != current).sum())
                    changed = True
            if "outlier_any" in df.columns:
                src_cols = [c for c in candidate_cols if c != "outlier_any" and c in df.columns]
                if src_cols:
                    current = df["outlier_any"].fillna(False).astype(bool).to_numpy()
                    union = (
                        df[src_cols].fillna(False).astype(bool).any(axis=1).to_numpy()
                    )
                    updated = current | union
                    if not np.array_equal(current, updated):
                        df["outlier_any"] = updated
                        total_cells_changed += int((updated != current).sum())
                        changed = True
            if not changed:
                continue
            csv_path = csv_by_key[key]
            df.drop(columns=["__timkey__"], errors="ignore").to_csv(csv_path, index=False)
            total_csvs_changed += 1
            summary_updates[key] = summarize_pqc(df.drop(columns=["__timkey__"], errors="ignore"))

    for row in qc_rows:
        key = (str(row.get("pulsar", "")), str(row.get("variant", "")))
        if key in summary_updates and str(row.get("qc_status", "")) == "success":
            for k in list(row.keys()):
                if k.startswith("metric.") or k in {
                    "n_toas",
                    "n_bad",
                    "bad_fraction",
                    "n_events",
                    "n_event_members",
                    "event_fraction",
                    "event_stability",
                    "residual_cleanliness",
                    "residual_whiteness",
                    "scaled_residual_cleanliness",
                    "backend_inconsistency_penalty",
                    "overfragmentation_penalty",
                    "parameter_complexity_penalty",
                    "stability",
                }:
                    row.pop(k, None)
            row.update(summary_updates[key])

    if total_csvs_changed > 0:
        logger.info(
            "Homogenized outlier flags across variants for %d QC CSV(s); updated %d cell(s).",
            total_csvs_changed,
            total_cells_changed,
        )


def _cfg_get(cfg, name: str, default=None):
    """Safely read a config value from an object or environment.

    Args:
        cfg: Config object (typically :class:`PipelineConfig`).
        name: Attribute name to read.
        default: Fallback value when missing.

    Returns:
        The resolved config value or ``default``.

    Notes:
        This keeps :class:`PipelineConfig` schema changes optional for
        notebook-driven workflows by allowing environment overrides.
    """
    try:
        return getattr(cfg, name)
    except Exception:
        pass

    env_key = {
        "fix_branch_name": "FIXDATASET_BRANCH_NAME",
        "fix_commit_message": "FIXDATASET_COMMIT_MESSAGE",
        "fix_base_branch": "FIXDATASET_BASE_BRANCH",
    }.get(name)
    if env_key:
        v = os.environ.get(env_key, "")
        if v != "":
            return v
    return default


def _cfg_get_bool(cfg, name: str, default: bool = False) -> bool:
    """Resolve a config value as a boolean.

    Args:
        cfg: Config object (typically :class:`PipelineConfig`).
        name: Attribute name to read.
        default: Fallback value when missing.

    Returns:
        Boolean interpretation of the configuration value.

    Notes:
        This avoids Python's ``bool("0")`` pitfall for environment overrides.
    """
    v = None
    try:
        v = getattr(cfg, name)
    except Exception:
        v = None

    if v is None:
        env_key = {
            "fix_apply": "FIXDATASET_APPLY",
            "run_fix_dataset": "RUN_FIX_DATASET",
        }.get(name)
        if env_key:
            v = os.environ.get(env_key)

    if v is None:
        return bool(default)

    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))

    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    # Fall back to Python truthiness for odd values
    return bool(s)


def _fix_cfg_fields() -> set[str]:
    """Return the supported :class:`FixDatasetConfig` field names.

    Returns:
        A set of field names supported by the in-repo FixDataset implementation.
    """
    try:
        from dataclasses import fields as dc_fields

        return {f.name for f in dc_fields(FixDatasetConfig)}
    except Exception:  # pragma: no cover
        # fallback for non-dataclass implementations (unlikely)
        return set(getattr(FixDatasetConfig, "__annotations__", {}).keys())


def _build_fixdataset_config(
    cfg,
    *,
    apply: bool,
    qc_results_dir: Path | None = None,
    qc_branch: str | None = None,
) -> FixDatasetConfig:
    """Create a :class:`FixDatasetConfig` from a :class:`PipelineConfig`.

    Args:
        cfg: Pipeline configuration.
        apply: Whether FixDataset should apply and commit changes.
        qc_results_dir: Optional override for QC CSV directory.
        qc_branch: Optional override for QC branch subdirectory name.

    Returns:
        A :class:`FixDatasetConfig` with only supported fields populated.

    Notes:
        The pipeline config may contain more ``fix_*`` fields than are supported
        by a given FixDataset version; unsupported fields are ignored.
    """
    # Apply-mode safety: avoid leaving backup artifacts that dirty the repo.
    dry_run = bool(_cfg_get(cfg, "fix_dry_run", False))
    backup_default = False if apply else True
    backup = bool(_cfg_get(cfg, "fix_backup", backup_default))

    if apply:
        dry_run = False

    # canonical knobs (supported by the dataset_fix.py shipped with this repo)
    kwargs = dict(
        apply=bool(apply),
        backup=bool(backup),
        dry_run=bool(dry_run),
        update_alltim_includes=bool(_cfg_get(cfg, "fix_update_alltim_includes", True)),
        min_toas_per_backend_tim=int(
            _cfg_get(cfg, "fix_min_toas_per_backend_tim", 10) or 10
        ),
        required_tim_flags=dict(_cfg_get(cfg, "fix_required_tim_flags", {}) or {}),
        infer_system_flags=bool(_cfg_get(cfg, "fix_infer_system_flags", False)),
        system_flag_table_path=_cfg_get(cfg, "fix_system_flag_table_path", None),
        system_flag_mapping_path=_cfg_get(cfg, "fix_system_flag_mapping_path", None),
        flag_sys_freq_rules_enabled=bool(
            _cfg_get(cfg, "fix_flag_sys_freq_rules_enabled", False)
        ),
        flag_sys_freq_rules_path=_cfg_get(cfg, "fix_flag_sys_freq_rules_path", None),
        generate_alltim_variants=bool(
            _cfg_get(cfg, "fix_generate_alltim_variants", False)
        ),
        backend_classifications_path=_cfg_get(
            cfg, "fix_backend_classifications_path", None
        ),
        alltim_variants_path=_cfg_get(cfg, "fix_alltim_variants_path", None),
        relabel_rules_path=_cfg_get(cfg, "fix_relabel_rules_path", None),
        overlap_rules_path=_cfg_get(cfg, "fix_overlap_rules_path", None),
        overlap_exact_catalog_path=_cfg_get(
            cfg, "fix_overlap_exact_catalog_path", None
        ),
        jump_reference_variants=bool(
            _cfg_get(cfg, "fix_jump_reference_variants", False)
        ),
        jump_reference_keep_tmp=bool(
            _cfg_get(cfg, "fix_jump_reference_keep_tmp", False)
        ),
        jump_reference_jump_flag=str(
            _cfg_get(cfg, "fix_jump_reference_jump_flag", "-sys") or "-sys"
        ),
        jump_reference_csv_dir=_cfg_get(cfg, "fix_jump_reference_csv_dir", None),
        tempo2_home_dir=str(getattr(cfg, "home_dir", "")),
        tempo2_dataset_name=str(getattr(cfg, "dataset_name", "")),
        tempo2_singularity_image=str(getattr(cfg, "singularity_image", "")),
        system_flag_overwrite_existing=bool(
            _cfg_get(cfg, "fix_system_flag_overwrite_existing", False)
        ),
        wsrt_p2_force_sys_by_freq=bool(
            _cfg_get(cfg, "fix_wsrt_p2_force_sys_by_freq", False)
        ),
        wsrt_p2_prefer_dual_channel=bool(
            _cfg_get(cfg, "fix_wsrt_p2_prefer_dual_channel", False)
        ),
        wsrt_p2_mjd_tol_sec=float(_cfg_get(cfg, "fix_wsrt_p2_mjd_tol_sec", 0.99e-6)),
        wsrt_p2_action=str(_cfg_get(cfg, "fix_wsrt_p2_action", "comment") or "comment"),
        wsrt_p2_comment_prefix=str(
            _cfg_get(cfg, "fix_wsrt_p2_comment_prefix", "C WSRT_P2_PREFER_DUAL")
            or "C WSRT_P2_PREFER_DUAL"
        ),
        backend_overrides=dict(_cfg_get(cfg, "fix_backend_overrides", {}) or {}),
        raise_on_backend_missing=bool(
            _cfg_get(cfg, "fix_raise_on_backend_missing", False)
        ),
        dedupe_toas_within_tim=bool(_cfg_get(cfg, "fix_dedupe_toas_within_tim", False)),
        dedupe_mjd_tol_sec=float(_cfg_get(cfg, "fix_dedupe_mjd_tol_sec", 0.0)),
        dedupe_freq_tol_mhz=_cfg_get(cfg, "fix_dedupe_freq_tol_mhz", None),
        dedupe_freq_tol_auto=bool(_cfg_get(cfg, "fix_dedupe_freq_tol_auto", False)),
        check_duplicate_backend_tims=bool(
            _cfg_get(cfg, "fix_check_duplicate_backend_tims", False)
        ),
        remove_overlaps_exact=bool(_cfg_get(cfg, "fix_remove_overlaps_exact", False)),
        insert_missing_jumps=bool(_cfg_get(cfg, "fix_insert_missing_jumps", True)),
        jump_flag=str(_cfg_get(cfg, "fix_jump_flag", "-sys") or "-sys"),
        prune_stale_jumps=bool(_cfg_get(cfg, "fix_prune_stale_jumps", False)),
        ensure_ephem=_cfg_get(cfg, "fix_ensure_ephem", None),
        ensure_clk=_cfg_get(cfg, "fix_ensure_clk", None),
        ensure_ne_sw=_cfg_get(cfg, "fix_ensure_ne_sw", None),
        force_ne_sw_overwrite=bool(_cfg_get(cfg, "fix_force_ne_sw_overwrite", False)),
        remove_patterns=list(
            _cfg_get(cfg, "fix_remove_patterns", ["NRT.NUPPI.", "NRT.NUXPI."]) or []
        ),
        coord_convert=_cfg_get(cfg, "fix_coord_convert", None),
    )

    # Extended knobs (present in pipelineb; only applied if FixDatasetConfig supports them)
    kwargs.update(
        dict(
            prune_missing_includes=bool(
                _cfg_get(cfg, "fix_prune_missing_includes", True)
            ),
            drop_small_backend_includes=bool(
                _cfg_get(cfg, "fix_drop_small_backend_includes", True)
            ),
            system_flag_update_table=bool(
                _cfg_get(cfg, "fix_system_flag_update_table", True)
            ),
            default_backend=_cfg_get(cfg, "fix_default_backend", None),
            group_flag=str(_cfg_get(cfg, "fix_group_flag", "-group") or "-group"),
            pta_flag=str(_cfg_get(cfg, "fix_pta_flag", "-pta") or "-pta"),
            pta_value=_cfg_get(cfg, "fix_pta_value", None),
            standardize_par_values=bool(
                _cfg_get(cfg, "fix_standardize_par_values", True)
            ),
            prune_small_system_toas=bool(
                _cfg_get(cfg, "fix_prune_small_system_toas", False)
            ),
            prune_small_system_flag=str(
                _cfg_get(cfg, "fix_prune_small_system_flag", "-sys") or "-sys"
            ),
            qc_remove_outliers=bool(_cfg_get(cfg, "fix_qc_remove_outliers", False)),
            qc_outlier_cols=_cfg_get(cfg, "fix_qc_outlier_cols", None),
            qc_action=str(_cfg_get(cfg, "fix_qc_action", "comment") or "comment"),
            qc_backend_col=str(_cfg_get(cfg, "fix_qc_backend_col", "sys") or "sys"),
            qc_comment_prefix=str(
                _cfg_get(cfg, "fix_qc_comment_prefix", "C QC_OUTLIER") or "C QC_OUTLIER"
            ),
            qc_remove_bad=bool(_cfg_get(cfg, "fix_qc_remove_bad", True)),
            qc_remove_transients=bool(_cfg_get(cfg, "fix_qc_remove_transients", False)),
            qc_remove_solar=bool(_cfg_get(cfg, "fix_qc_remove_solar", False)),
            qc_solar_action=str(
                _cfg_get(cfg, "fix_qc_solar_action", "comment") or "comment"
            ),
            qc_solar_comment_prefix=str(
                _cfg_get(cfg, "fix_qc_solar_comment_prefix", "C QC_SOLAR")
                or "C QC_SOLAR"
            ),
            qc_remove_orbital_phase=bool(
                _cfg_get(cfg, "fix_qc_remove_orbital_phase", False)
            ),
            qc_orbital_phase_action=str(
                _cfg_get(cfg, "fix_qc_orbital_phase_action", "comment") or "comment"
            ),
            qc_orbital_phase_comment_prefix=str(
                _cfg_get(
                    cfg, "fix_qc_orbital_phase_comment_prefix", "C QC_BIANRY_ECLIPSE"
                )
                or "C QC_BIANRY_ECLIPSE"
            ),
            qc_write_pqc_flag=bool(_cfg_get(cfg, "fix_qc_write_pqc_flag", False)),
            qc_pqc_flag_name=str(
                _cfg_get(cfg, "fix_qc_pqc_flag_name", "-pqc") or "-pqc"
            ),
            qc_pqc_good_value=str(
                _cfg_get(cfg, "fix_qc_pqc_good_value", "good") or "good"
            ),
            qc_pqc_bad_value=str(_cfg_get(cfg, "fix_qc_pqc_bad_value", "bad") or "bad"),
            qc_pqc_event_prefix=str(
                _cfg_get(cfg, "fix_qc_pqc_event_prefix", "event_") or "event_"
            ),
            qc_bad_tau_corr_days=float(
                _cfg_get(cfg, "fix_qc_bad_tau_corr_days", 0.02) or 0.02
            ),
            qc_bad_fdr_q=float(_cfg_get(cfg, "fix_qc_bad_fdr_q", 0.01) or 0.01),
            qc_bad_mark_only_worst_per_day=bool(
                _cfg_get(cfg, "fix_qc_bad_mark_only_worst_per_day", True)
            ),
            qc_tr_tau_rec_days=float(
                _cfg_get(cfg, "fix_qc_tr_tau_rec_days", 7.0) or 7.0
            ),
            qc_tr_window_mult=float(_cfg_get(cfg, "fix_qc_tr_window_mult", 5.0) or 5.0),
            qc_tr_min_points=int(_cfg_get(cfg, "fix_qc_tr_min_points", 6) or 6),
            qc_tr_delta_chi2_thresh=float(
                _cfg_get(cfg, "fix_qc_tr_delta_chi2_thresh", 25.0) or 25.0
            ),
            qc_tr_suppress_overlap=bool(
                _cfg_get(cfg, "fix_qc_tr_suppress_overlap", True)
            ),
            qc_merge_tol_days=float(
                _cfg_get(cfg, "fix_qc_merge_tol_days", 2.0 / 86400.0) or (2.0 / 86400.0)
            ),
            qc_results_dir=(
                qc_results_dir
                if qc_results_dir is not None
                else _cfg_get(cfg, "fix_qc_results_dir", None)
            ),
            qc_branch=(
                qc_branch
                if qc_branch is not None
                else _cfg_get(cfg, "fix_qc_branch", None)
            ),
            qc_require_csv=bool(_cfg_get(cfg, "fix_qc_require_csv", True)),
        )
    )

    supported = _fix_cfg_fields()
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return FixDatasetConfig(**filtered)


def _pqc_available() -> bool:
    """Return True if the optional ``pqc`` package is importable."""
    try:
        return importlib.util.find_spec("pqc") is not None
    except Exception:
        return False


def _raise_if_fixdataset_failed(
    reports: List[Dict[str, object]], *, stage: str, branch: str
) -> None:
    errors = []
    for rep in reports:
        err = rep.get("error")
        if not err:
            continue
        psr = str(rep.get("psr") or rep.get("pulsar") or "<unknown>")
        errors.append(f"{psr}: {err}")
    if not errors:
        return
    preview = "; ".join(errors[:5])
    if len(errors) > 5:
        preview += f"; ... ({len(errors)} pulsars failed)"
    raise RuntimeError(
        f"FixDataset {stage} failed on branch {branch}. {preview}"
    )


def _validate_fixdataset_qc_inputs(
    pulsars: List[str], cfg: FixDatasetConfig, *, branch: str
) -> None:
    if not (cfg.qc_remove_outliers or cfg.qc_write_pqc_flag):
        return
    errors = []
    for psr in pulsars:
        try:
            manifest_rows: List[Dict[str, object]] = []
            if cfg.qc_results_dir is not None:
                summary_path = Path(cfg.qc_results_dir) / "qc_summary.tsv"
                if not summary_path.exists():
                    summary_path = Path(cfg.qc_results_dir).parent / "qc_summary.tsv"
                if summary_path.exists():
                    try:
                        df = pd.read_csv(summary_path, sep="\t")
                        if "pulsar" in df.columns:
                            df = df[df["pulsar"].astype(str) == psr]
                        if "branch" in df.columns:
                            df = df[df["branch"].astype(str) == str(cfg.qc_branch or "")]
                        manifest_rows = df.to_dict(orient="records")
                    except Exception:
                        manifest_rows = []
            if manifest_rows:
                statuses = {
                    str(r.get("qc_status", "")).strip() or (
                        "pqc_failed" if str(r.get("qc_error", "")).strip() else "success"
                    )
                    for r in manifest_rows
                }
                if "success" in statuses:
                    continue
                if "pqc_failed" in statuses or "prepare_failed" in statuses:
                    errors.append(
                        f"{psr}: unresolved QC status in manifest: {sorted(statuses)}"
                    )
                    continue
                if statuses == {"empty_variant"}:
                    continue
                errors.append(f"{psr}: no successful QC status in manifest: {sorted(statuses)}")
                continue
            _find_qc_csv(psr, cfg)
        except Exception as e:
            errors.append(f"{psr}: {e}")
    if not errors:
        return
    preview = "; ".join(errors[:5])
    if len(errors) > 5:
        preview += f"; ... ({len(errors)} pulsars failed)"
    raise RuntimeError(
        f"FixDataset QC input validation failed for branch {branch}. {preview}"
    )


def _warn_backend_tim_drift(
    repo,
    cfg,
    pulsars: List[str],
    *,
    baseline_branch: str,
    compare_branch: str,
    return_branch: str,
) -> None:
    """Warn if backend tim inventory differs from a baseline branch.

    This is intentionally warning-only. It exists to surface workflow drift
    when later stages assume Step 1 has already canonicalized backend timfiles.
    """
    baseline_branch = str(baseline_branch or "").strip()
    compare_branch = str(compare_branch or "").strip()
    if not baseline_branch or not compare_branch or baseline_branch == compare_branch:
        return

    dataset_root = Path(cfg.dataset_name)
    baseline: Dict[str, set[str]] = {}
    compare: Dict[str, set[str]] = {}

    try:
        checkout(repo, baseline_branch)
        for psr in pulsars:
            psr_dir = dataset_root / psr
            if not psr_dir.exists():
                baseline[psr] = set()
                continue
            baseline[psr] = {t.name for t in list_backend_timfiles(psr_dir)}

        checkout(repo, compare_branch)
        for psr in pulsars:
            psr_dir = dataset_root / psr
            if not psr_dir.exists():
                compare[psr] = set()
                continue
            compare[psr] = {t.name for t in list_backend_timfiles(psr_dir)}
    finally:
        checkout(repo, return_branch)

    drift_msgs: List[str] = []
    for psr in pulsars:
        b = baseline.get(psr, set())
        c = compare.get(psr, set())
        if b == c:
            continue
        added = sorted(c - b)
        removed = sorted(b - c)
        parts: List[str] = []
        if added:
            parts.append(f"added={added}")
        if removed:
            parts.append(f"removed={removed}")
        drift_msgs.append(f"{psr}: " + ", ".join(parts))

    if not drift_msgs:
        return

    preview = "; ".join(drift_msgs[:10])
    if len(drift_msgs) > 10:
        preview += f"; ... ({len(drift_msgs)} pulsars differ)"
    logger.warning(
        "Backend tim inventory differs between baseline branch %s and branch %s. %s",
        baseline_branch,
        compare_branch,
        preview,
    )


def _apply_fixdataset_and_commit(
    repo,
    cfg,
    pulsars: List[str],
    out_paths: Dict[str, Path],
    *,
    base_branch: str,
    new_branch: str,
    commit_message: str,
) -> str:
    """Create a new branch, apply FixDataset, and commit changes.

    Args:
        repo: GitPython repository object rooted at ``cfg.home_dir``.
        cfg: Pipeline configuration (resolved paths).
        pulsars: Pulsar names to process.
        out_paths: Output directory mapping from :func:`make_output_tree`.
        base_branch: Base branch name to branch from.
        new_branch: Name of the new FixDataset branch to create.
        commit_message: Commit message for FixDataset changes.

    Returns:
        The name of the created branch (``new_branch``).

    Raises:
        RuntimeError: If the branch already exists.
    """
    require_clean_repo(repo)
    checkout(repo, base_branch)
    logger.info("Checked out branch %s", base_branch)

    existing = {h.name for h in getattr(repo, "heads", [])}
    if new_branch in existing:
        raise RuntimeError(
            f"Requested fix branch '{new_branch}' already exists. Choose a different name."
        )

    repo.git.checkout("-b", new_branch)
    logger.info("Checked out branch %s", new_branch)

    qc_results_dir = _cfg_get(cfg, "fix_qc_results_dir", None)
    qc_branch = _cfg_get(cfg, "fix_qc_branch", None)
    if not qc_results_dir:
        if out_paths.get("qc") is not None:
            qc_results_dir = out_paths["qc"] / new_branch
            qc_branch = None
            logger.info("No fix_qc_results_dir provided; using %s.", qc_results_dir)
    elif qc_branch is None:
        qc_branch = new_branch

    if qc_results_dir is not None and not isinstance(qc_results_dir, Path):
        qc_results_dir = Path(qc_results_dir)

    fcfg = _build_fixdataset_config(
        cfg, apply=True, qc_results_dir=qc_results_dir, qc_branch=qc_branch
    )
    _validate_fixdataset_qc_inputs(pulsars, fcfg, branch=new_branch)

    dataset_root = Path(cfg.dataset_name)
    n_jobs = max(1, int(getattr(cfg, "jobs", 1) or 1))
    if fcfg.infer_system_flags:
        reports = apply_fixdataset_branch(
            dataset_root,
            pulsars,
            fcfg,
            branch=new_branch,
            jobs=n_jobs,
        )
    else:
        reports = []
        if n_jobs == 1:
            for pulsar in tqdm(pulsars, desc=f"fix-dataset (apply on {new_branch})"):
                rep = fix_pulsar_dataset(dataset_root / pulsar, fcfg)
                rep["branch"] = new_branch
                reports.append(rep)
        else:

            def _run_fix(p: str) -> Dict[str, object]:
                try:
                    rep = fix_pulsar_dataset(dataset_root / p, fcfg)
                    rep["branch"] = new_branch
                    return rep
                except Exception as e:
                    return {"psr": p, "branch": new_branch, "error": str(e)}

            futures = []
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                for pulsar in pulsars:
                    futures.append(ex.submit(_run_fix, pulsar))
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"fix-dataset (apply on {new_branch})",
                ):
                    reports.append(fut.result())

    write_fix_report(reports, out_paths["fix_dataset"] / new_branch)
    _raise_if_fixdataset_failed(reports, stage="apply", branch=new_branch)

    dataset_prefix = ""
    try:
        dataset_prefix = dataset_root.relative_to(cfg.home_dir).as_posix().strip("/")
    except Exception:
        if not dataset_root.exists():
            logger.warning(
                "Dataset path %s does not exist; staging changes from repo root.",
                dataset_root,
            )
            dataset_prefix = ""

    changed = [p for p in repo.git.diff("--name-only").splitlines() if p.strip()]
    untracked = list(getattr(repo, "untracked_files", []) or [])
    paths = list(dict.fromkeys(changed + untracked))

    def _want(p: str) -> bool:
        pp = p.replace("\\", "/")
        if dataset_prefix and not pp.startswith(dataset_prefix + "/"):
            return False
        # Backups / scratch artifacts should not block cleanliness checks:
        if pp.endswith(".orig"):
            return False
        if pp.endswith(".par") or pp.endswith(".tim"):
            return True
        # system flag table (if created/updated)
        if pp.endswith("system_flag_table.json") or pp.endswith(
            "system_flag_table.toml"
        ):
            return True
        return False

    to_stage = [p for p in paths if _want(p)]

    if to_stage:
        repo.git.add("--", *to_stage)
        repo.index.commit(commit_message)
    else:
        repo.git.commit("--allow-empty", "-m", commit_message + " (no changes)")

    # If backups were produced anyway, they will keep the repo dirty. Delete them to preserve pipeline invariants.
    for p in list(getattr(repo, "untracked_files", []) or []):
        if p.endswith(".orig"):
            try:
                (cfg.home_dir / p).unlink()
            except Exception:
                pass

    require_clean_repo(repo)
    return new_branch


def _commit_branch_artifacts(
    repo,
    cfg,
    *,
    branch: str,
    out_paths: Dict[str, Path],
    commit_message: str,
) -> None:
    """Commit persistent run artifacts for a fix-apply branch.

    This is intentionally broader than the initial data-only commit: it stages
    report outputs, CSVs, plots, and other useful run artifacts that were
    generated after the branch was created.
    """
    repo_root = Path(cfg.home_dir).resolve()

    def _rel_if_within_repo(path_like) -> str | None:
        try:
            p = Path(path_like).resolve()
        except Exception:
            return None
        try:
            return p.relative_to(repo_root).as_posix()
        except Exception:
            return None

    prefixes: set[str] = set()
    for p in (
        out_paths.get("base"),
        out_paths.get("tag"),
        getattr(cfg, "results_dir", None),
        getattr(cfg, "qc_report_dir", None),
        getattr(cfg, "fix_jump_reference_csv_dir", None),
    ):
        rel = _rel_if_within_repo(p) if p is not None else None
        if rel:
            prefixes.add(rel.rstrip("/"))

    if not prefixes:
        return

    changed = [p for p in repo.git.diff("--name-only").splitlines() if p.strip()]
    untracked = list(getattr(repo, "untracked_files", []) or [])
    paths = list(dict.fromkeys(changed + untracked))

    allowed_exts = {
        ".par",
        ".tim",
        ".csv",
        ".tsv",
        ".pdf",
        ".png",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
        ".txt",
        ".html",
        ".svg",
    }
    allowed_names = {
        "run_report.pdf",
        "fix_dataset_report.pdf",
        "fix_dataset_report.json",
        "qc_summary.tsv",
        "whitenoise_summary.tsv",
        "binary_analysis.tsv",
    }

    def _want(path_str: str) -> bool:
        pp = path_str.replace("\\", "/")
        if not any(pp == pref or pp.startswith(pref + "/") for pref in prefixes):
            return False
        if pp.endswith(".orig"):
            return False
        parts = pp.split("/")
        if any(part.startswith(".pqc_") for part in parts):
            return False
        if any(part in {"work", "logs", "__pycache__"} for part in parts):
            return False
        name = Path(pp).name
        if name in allowed_names:
            return True
        return Path(pp).suffix.lower() in allowed_exts

    to_stage = [p for p in paths if _want(p)]
    if not to_stage:
        return

    repo.git.add("--", *to_stage)
    if repo.is_dirty(untracked_files=True):
        repo.index.commit(f"{commit_message} [artifacts]")
        logger.info(
            "Committed run artifacts on branch %s (%d paths).", branch, len(to_stage)
        )


def run_pipeline(config: PipelineConfig) -> Dict[str, Path]:
    """Run the full diagnostics pipeline.

    Concurrency model:
        * Git branch checkouts are single-threaded.
        * Within each branch, pulsars can be processed concurrently using ``cfg.jobs``.
        * Each pulsar uses its own work directory to avoid tempo2 output collisions.

    Args:
        config: Pipeline configuration.

    Returns:
        Mapping of output path labels to their filesystem paths.

    Raises:
        FileNotFoundError: If required paths (home_dir, singularity image) are missing.
        RuntimeError: For missing dependencies or invalid configuration.

    Examples:
        Run the pipeline programmatically::

            from pathlib import Path
            from pleb.config import PipelineConfig
            from pleb.pipeline import run_pipeline

            cfg = PipelineConfig(
                home_dir=Path("/data/epta"),
                singularity_image=Path("/images/tempo2.sif"),
                dataset_name="EPTA",
            )
            outputs = run_pipeline(cfg)
    """
    cfg = config.resolved()
    set_log_dir(Path(cfg.home_dir) / "logs")
    dataset_root = Path(cfg.dataset_name)

    # Convenience toggles for older CLI usage.
    if cfg.make_plots is not None and not bool(cfg.make_plots):
        cfg.make_toa_coverage_plots = False
        cfg.make_covariance_heatmaps = False
        cfg.make_residual_plots = False
    if cfg.make_reports is not None and not bool(cfg.make_reports):
        cfg.make_outlier_reports = False
        cfg.make_change_reports = False
    if cfg.make_covmat is not None:
        cfg.make_covariance_heatmaps = bool(cfg.make_covmat)

    run_fix_dataset = _cfg_get_bool(cfg, "run_fix_dataset", False)
    fix_apply = _cfg_get_bool(cfg, "fix_apply", False)
    run_pqc = bool(getattr(cfg, "run_pqc", False))
    run_whitenoise = bool(getattr(cfg, "run_whitenoise", False))

    logger.info(
        "Config flags: run_fix_dataset=%s fix_apply=%s run_pqc=%s run_whitenoise=%s",
        run_fix_dataset,
        fix_apply,
        run_pqc,
        run_whitenoise,
    )
    if fix_apply:
        logger.info(
            "FixDataset apply config: branch=%s base=%s commit_message=%s",
            str(_cfg_get(cfg, "fix_branch_name", "") or "").strip() or "<missing>",
            str(_cfg_get(cfg, "fix_base_branch", "") or "").strip() or "<auto>",
            str(_cfg_get(cfg, "fix_commit_message", "") or "").strip() or "<default>",
        )
    logger.info("pqc available: %s", _pqc_available())

    if not cfg.home_dir.exists():
        raise FileNotFoundError(f"home_dir does not exist: {cfg.home_dir}")
    if not cfg.dataset_name.exists():
        warnings.warn(
            f"Dataset {cfg.dataset_name} does not exist. Assuming the pulsar folders live in {cfg.home_dir}.",
            stacklevel=2,
        )
    if not cfg.singularity_image.exists():
        raise FileNotFoundError(
            f"singularity_image does not exist: {cfg.singularity_image}"
        )

    which_or_raise(
        "singularity",
        hint="Install Singularity/Apptainer or load it in your environment.",
    )

    try:
        from git import Repo  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "GitPython is required to run the pipeline (branch checkouts). Install GitPython."
        ) from e

    repo = Repo(str(cfg.home_dir))
    require_clean_repo(repo)
    current_branch = repo.active_branch.name
    logger.info("Current git branch: %s", current_branch)

    # Branch selection
    compare_branches: List[str] = [
        str(b).strip() for b in cfg.branches if str(b).strip()
    ]
    compare_branches = list(dict.fromkeys(compare_branches))  # preserve order
    reference_branch = str(cfg.reference_branch).strip() if cfg.reference_branch else ""

    branches_to_run = compare_branches.copy()
    change_reports_enabled = bool(cfg.make_change_reports) and bool(reference_branch)
    if getattr(cfg, "testing_mode", False):
        logger.info("Testing mode enabled: change reports will be skipped.")
        change_reports_enabled = False

    if (
        reference_branch
        and reference_branch not in branches_to_run
        and change_reports_enabled
    ):
        branches_to_run.append(reference_branch)

    fix_branch_name = ""
    base_branch = ""
    commit_message = ""
    if fix_apply:
        fix_branch_name = str(_cfg_get(cfg, "fix_branch_name", "") or "").strip()
        if not fix_branch_name:
            stamp = datetime.now().strftime("%d%m%y%H%M")
            fix_branch_name = f"branch_run_{stamp}"
            logger.info(
                "No fix_branch_name provided; using autogenerated branch '%s'.",
                fix_branch_name,
            )

        base_branch = str(_cfg_get(cfg, "fix_base_branch", "") or "").strip()
        if not base_branch:
            base_branch = str(reference_branch or current_branch)

        commit_message = (
            str(_cfg_get(cfg, "fix_commit_message", "") or "").strip()
            or "FixDataset: apply automated dataset fixes"
        )

    # Pulsar selection
    if cfg.pulsars == "ALL":
        discovery_branch = base_branch if fix_apply and base_branch else current_branch
        if discovery_branch != current_branch:
            checkout(repo, discovery_branch)
            logger.info(
                "Temporarily checked out branch %s to discover pulsars.",
                discovery_branch,
            )
        try:
            pulsars = discover_pulsars(dataset_root)
        finally:
            if discovery_branch != current_branch:
                checkout(repo, current_branch)
        if not pulsars and fix_apply and base_branch and base_branch != current_branch:
            raise RuntimeError(
                f"No pulsars found under {dataset_root} on base branch '{base_branch}'."
            )
    else:
        pulsars = list(cfg.pulsars)  # type: ignore[arg-type]
    if not pulsars:
        raise RuntimeError("No pulsars selected/found.")

    if fix_apply:
        drift_branch = str(
            _cfg_get(cfg, "fix_warn_backend_tim_drift_from_branch", "") or ""
        ).strip()
        if drift_branch and not bool(_cfg_get(cfg, "fix_update_alltim_includes", True)):
            _warn_backend_tim_drift(
                repo,
                cfg,
                pulsars,
                baseline_branch=drift_branch,
                compare_branch=base_branch,
                return_branch=current_branch,
            )

    out_paths = make_output_tree(
        cfg.results_dir, compare_branches, cfg.outdir_name, lazy=True
    )
    logger.info("Writing outputs to: %s", out_paths["tag"])

    # Ensure fix-dataset output path exists even if the output tree helper doesn't pre-create it
    if "fix_dataset" not in out_paths:
        out_paths["fix_dataset"] = out_paths["tag"] / "fix_dataset"
    out_paths["fix_dataset"].mkdir(parents=True, exist_ok=True)

    binary_rows: List[Dict[str, object]] = []
    qc_rows: List[Dict[str, object]] = []
    qc_enabled = run_pqc
    if run_pqc and not _pqc_available():
        logger.error(
            "run_pqc=true but pqc is not importable. QC stage will be skipped. "
            "Install pqc (and libstempo) to enable QC."
        )
        qc_enabled = False

    # If requested, apply FixDataset on a new branch and commit the resulting .par/.tim files.
    if fix_apply:
        logger.info(
            "Applying FixDataset on new branch '%s' (base: %s) and committing .par/.tim changes.",
            fix_branch_name,
            base_branch,
        )
        _apply_fixdataset_and_commit(
            repo,
            cfg,
            pulsars,
            out_paths,
            base_branch=base_branch,
            new_branch=fix_branch_name,
            commit_message=commit_message,
        )
        checkout(repo, current_branch)

    try:
        for branch in branches_to_run:
            logger.info("=== Branch: %s ===", branch)
            checkout(repo, branch)

            # Forced fix_dataset reporting per branch (report-only; never modifies the repo in this loop)
            fcfg = _build_fixdataset_config(
                cfg, apply=False, qc_results_dir=out_paths.get("qc"), qc_branch=branch
            )
            reports = []
            if run_fix_dataset and not fix_apply:
                n_jobs = max(1, int(getattr(cfg, "jobs", 1) or 1))
                if n_jobs == 1:
                    for pulsar in tqdm(pulsars, desc=f"fix-dataset ({branch})"):
                        rep = fix_pulsar_dataset(dataset_root / pulsar, fcfg)
                        rep["branch"] = branch
                        reports.append(rep)
                else:

                    def _run_fix(p: str) -> Dict[str, object]:
                        try:
                            rep = fix_pulsar_dataset(dataset_root / p, fcfg)
                            rep["branch"] = branch
                            return rep
                        except Exception as e:
                            return {"psr": p, "branch": branch, "error": str(e)}

                    futures = []
                    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                        for pulsar in pulsars:
                            futures.append(ex.submit(_run_fix, pulsar))
                        for fut in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=f"fix-dataset ({branch})",
                        ):
                            reports.append(fut.result())
                write_fix_report(reports, out_paths["fix_dataset"] / branch)
                _raise_if_fixdataset_failed(
                    reports, stage="report-only", branch=branch
                )
            elif not run_fix_dataset:
                logger.info(
                    "FixDataset report-only stage skipped (run_fix_dataset=false)."
                )
            else:
                logger.info("FixDataset report-only stage skipped (fix_apply=true).")

            # tempo2 runs (parallelizable across pulsars)
            if cfg.run_tempo2:
                n_jobs = max(1, int(getattr(cfg, "jobs", 1) or 1))
                # pipelineb feature: if we just applied fixes and we're only running one branch,
                # force tempo2 rerun unless user explicitly disabled it.
                force_rerun = bool(cfg.force_rerun) or (
                    _cfg_get_bool(cfg, "fix_apply", False)
                    and len(branches_to_run) == 1
                    and bool(getattr(cfg, "run_fix_dataset", True))
                )

                if n_jobs == 1:
                    for pulsar in tqdm(pulsars, desc=f"tempo2 ({branch})"):
                        run_tempo2_for_pulsar(
                            home_dir=cfg.home_dir,
                            dataset_name=cfg.dataset_name,
                            singularity_image=cfg.singularity_image,
                            out_paths=out_paths,
                            pulsar=pulsar,
                            branch=branch,
                            epoch=str(cfg.epoch),
                            force_rerun=force_rerun,
                        )
                else:
                    futures = []
                    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                        for pulsar in pulsars:
                            futures.append(
                                ex.submit(
                                    run_tempo2_for_pulsar,
                                    home_dir=cfg.home_dir,
                                    dataset_name=cfg.dataset_name,
                                    singularity_image=cfg.singularity_image,
                                    out_paths=out_paths,
                                    pulsar=pulsar,
                                    branch=branch,
                                    epoch=str(cfg.epoch),
                                    force_rerun=force_rerun,
                                )
                            )
                        for fut in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=f"tempo2 ({branch})",
                        ):
                            fut.result()

            if run_whitenoise:
                wn_cfg = WhiteNoiseStageConfig(
                    source_path=(
                        Path(cfg.whitenoise_source_path)
                        if getattr(cfg, "whitenoise_source_path", None)
                        else None
                    ),
                    epoch_tolerance_seconds=float(
                        getattr(cfg, "whitenoise_epoch_tolerance_seconds", 1.0)
                    ),
                    single_toa_mode=str(
                        getattr(cfg, "whitenoise_single_toa_mode", "combined")
                    ),
                    fit_timing_model_first=bool(
                        getattr(cfg, "whitenoise_fit_timing_model_first", True)
                    ),
                    timfile_name=getattr(cfg, "whitenoise_timfile_name", None),
                )
                wn_out_dir = out_paths["whitenoise"] / branch
                wn_out_dir.mkdir(parents=True, exist_ok=True)
                wn_rows: List[Dict[str, object]] = []
                n_jobs = max(1, int(getattr(cfg, "jobs", 1) or 1))

                def _run_whitenoise(p: str) -> Dict[str, object]:
                    psr_dir = dataset_root / p
                    parfile = psr_dir / f"{p}.par"
                    timfile = resolve_timfile_for_pulsar(
                        psr_dir, p, wn_cfg.timfile_name
                    )
                    if not parfile.exists():
                        return {
                            "pulsar": p,
                            "branch": branch,
                            "parfile": str(parfile),
                            "timfile": "",
                            "success": False,
                            "error": f"Missing parfile: {parfile}",
                        }
                    if timfile is None:
                        hint = (
                            wn_cfg.timfile_name
                            if wn_cfg.timfile_name
                            else f"{p}_all.tim or {p}.tim"
                        )
                        return {
                            "pulsar": p,
                            "branch": branch,
                            "parfile": str(parfile),
                            "timfile": "",
                            "success": False,
                            "error": f"Missing timfile candidate ({hint}) under {psr_dir}",
                        }
                    try:
                        row = estimate_white_noise_for_pulsar(parfile, timfile, wn_cfg)
                    except Exception as e:
                        return {
                            "pulsar": p,
                            "branch": branch,
                            "parfile": str(parfile),
                            "timfile": str(timfile),
                            "success": False,
                            "error": str(e),
                        }
                    row.update(
                        {
                            "pulsar": p,
                            "branch": branch,
                            "parfile": str(parfile),
                            "timfile": str(timfile),
                        }
                    )
                    return row

                if n_jobs == 1:
                    for pulsar in tqdm(pulsars, desc=f"whitenoise ({branch})"):
                        wn_rows.append(_run_whitenoise(pulsar))
                else:
                    futures = []
                    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                        for pulsar in pulsars:
                            futures.append(ex.submit(_run_whitenoise, pulsar))
                        for fut in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=f"whitenoise ({branch})",
                        ):
                            wn_rows.append(fut.result())

                if wn_rows:
                    dfw = pd.DataFrame(wn_rows)
                    dfw.to_csv(
                        wn_out_dir / "whitenoise_summary.tsv",
                        sep="\t",
                        index=False,
                    )
                    n_ok = int(dfw.get("success", pd.Series([], dtype=bool)).sum())
                    n_fail = int(len(dfw) - n_ok)
                    logger.info(
                        "whitenoise stage (%s): %d success, %d failed (summary: %s)",
                        branch,
                        n_ok,
                        n_fail,
                        wn_out_dir / "whitenoise_summary.tsv",
                    )

            if qc_enabled:
                qc_cfg = PTAQCConfig(
                    backend_col=str(getattr(cfg, "pqc_backend_col", "group")),
                    drop_unmatched=bool(getattr(cfg, "pqc_drop_unmatched", False)),
                    merge_tol_seconds=float(getattr(cfg, "pqc_merge_tol_seconds", 2.0)),
                    tau_corr_minutes=float(getattr(cfg, "pqc_tau_corr_minutes", 30.0)),
                    fdr_q=float(getattr(cfg, "pqc_fdr_q", 0.01)),
                    mark_only_worst_per_day=bool(
                        getattr(cfg, "pqc_mark_only_worst_per_day", True)
                    ),
                    tau_rec_days=float(getattr(cfg, "pqc_tau_rec_days", 7.0)),
                    window_mult=float(getattr(cfg, "pqc_window_mult", 5.0)),
                    min_points=int(getattr(cfg, "pqc_min_points", 6)),
                    delta_chi2_thresh=float(
                        getattr(cfg, "pqc_delta_chi2_thresh", 25.0)
                    ),
                    exp_dip_min_duration_days=float(
                        getattr(cfg, "pqc_exp_dip_min_duration_days", 21.0)
                    ),
                    step_enabled=bool(getattr(cfg, "pqc_step_enabled", True)),
                    step_min_points=int(getattr(cfg, "pqc_step_min_points", 20)),
                    step_delta_chi2_thresh=float(
                        getattr(cfg, "pqc_step_delta_chi2_thresh", 25.0)
                    ),
                    step_scope=str(getattr(cfg, "pqc_step_scope", "both")),
                    dm_step_enabled=bool(getattr(cfg, "pqc_dm_step_enabled", True)),
                    dm_step_min_points=int(getattr(cfg, "pqc_dm_step_min_points", 20)),
                    dm_step_delta_chi2_thresh=float(
                        getattr(cfg, "pqc_dm_step_delta_chi2_thresh", 25.0)
                    ),
                    dm_step_scope=str(getattr(cfg, "pqc_dm_step_scope", "both")),
                    robust_enabled=bool(getattr(cfg, "pqc_robust_enabled", True)),
                    robust_z_thresh=float(getattr(cfg, "pqc_robust_z_thresh", 5.0)),
                    robust_scope=str(getattr(cfg, "pqc_robust_scope", "both")),
                    add_orbital_phase=bool(getattr(cfg, "pqc_add_orbital_phase", True)),
                    add_solar_elongation=bool(
                        getattr(cfg, "pqc_add_solar_elongation", True)
                    ),
                    add_elevation=bool(getattr(cfg, "pqc_add_elevation", False)),
                    add_airmass=bool(getattr(cfg, "pqc_add_airmass", False)),
                    add_parallactic_angle=bool(
                        getattr(cfg, "pqc_add_parallactic_angle", False)
                    ),
                    add_freq_bin=bool(getattr(cfg, "pqc_add_freq_bin", False)),
                    freq_bins=int(getattr(cfg, "pqc_freq_bins", 8)),
                    observatory_path=getattr(cfg, "pqc_observatory_path", None),
                    structure_mode=str(getattr(cfg, "pqc_structure_mode", "none")),
                    structure_detrend_features=getattr(
                        cfg, "pqc_structure_detrend_features", None
                    ),
                    structure_test_features=getattr(
                        cfg, "pqc_structure_test_features", None
                    ),
                    structure_nbins=int(getattr(cfg, "pqc_structure_nbins", 12)),
                    structure_min_per_bin=int(
                        getattr(cfg, "pqc_structure_min_per_bin", 3)
                    ),
                    structure_p_thresh=float(
                        getattr(cfg, "pqc_structure_p_thresh", 0.01)
                    ),
                    structure_circular_features=getattr(
                        cfg, "pqc_structure_circular_features", None
                    ),
                    structure_group_cols=getattr(cfg, "pqc_structure_group_cols", None),
                    outlier_gate_enabled=bool(
                        getattr(cfg, "pqc_outlier_gate_enabled", False)
                    ),
                    outlier_gate_sigma=float(
                        getattr(cfg, "pqc_outlier_gate_sigma", 3.0)
                    ),
                    outlier_gate_resid_col=getattr(
                        cfg, "pqc_outlier_gate_resid_col", None
                    ),
                    outlier_gate_sigma_col=getattr(
                        cfg, "pqc_outlier_gate_sigma_col", None
                    ),
                    event_instrument=bool(getattr(cfg, "pqc_event_instrument", False)),
                    solar_events_enabled=bool(
                        getattr(cfg, "pqc_solar_events_enabled", False)
                    ),
                    solar_approach_max_deg=float(
                        getattr(cfg, "pqc_solar_approach_max_deg", 30.0)
                    ),
                    solar_min_points_global=int(
                        getattr(cfg, "pqc_solar_min_points_global", 30)
                    ),
                    solar_min_points_year=int(
                        getattr(cfg, "pqc_solar_min_points_year", 10)
                    ),
                    solar_min_points_near_zero=int(
                        getattr(cfg, "pqc_solar_min_points_near_zero", 3)
                    ),
                    solar_tau_min_deg=float(getattr(cfg, "pqc_solar_tau_min_deg", 2.0)),
                    solar_tau_max_deg=float(
                        getattr(cfg, "pqc_solar_tau_max_deg", 60.0)
                    ),
                    solar_member_eta=float(getattr(cfg, "pqc_solar_member_eta", 1.0)),
                    solar_freq_dependence=bool(
                        getattr(cfg, "pqc_solar_freq_dependence", True)
                    ),
                    solar_freq_alpha_min=float(
                        getattr(cfg, "pqc_solar_freq_alpha_min", 0.0)
                    ),
                    solar_freq_alpha_max=float(
                        getattr(cfg, "pqc_solar_freq_alpha_max", 4.0)
                    ),
                    solar_freq_alpha_tol=float(
                        getattr(cfg, "pqc_solar_freq_alpha_tol", 1e-3)
                    ),
                    solar_freq_alpha_max_iter=int(
                        getattr(cfg, "pqc_solar_freq_alpha_max_iter", 64)
                    ),
                    orbital_phase_cut_enabled=bool(
                        getattr(cfg, "pqc_orbital_phase_cut_enabled", False)
                    ),
                    orbital_phase_cut_center=float(
                        getattr(cfg, "pqc_orbital_phase_cut_center", 0.25)
                    ),
                    orbital_phase_cut=getattr(cfg, "pqc_orbital_phase_cut", None),
                    orbital_phase_cut_sigma=float(
                        getattr(cfg, "pqc_orbital_phase_cut_sigma", 3.0)
                    ),
                    orbital_phase_cut_nbins=int(
                        getattr(cfg, "pqc_orbital_phase_cut_nbins", 18)
                    ),
                    orbital_phase_cut_min_points=int(
                        getattr(cfg, "pqc_orbital_phase_cut_min_points", 20)
                    ),
                    gaussian_bump_enabled=bool(
                        getattr(cfg, "pqc_gaussian_bump_enabled", False)
                    ),
                    gaussian_bump_min_duration_days=float(
                        getattr(cfg, "pqc_gaussian_bump_min_duration_days", 60.0)
                    ),
                    gaussian_bump_max_duration_days=float(
                        getattr(cfg, "pqc_gaussian_bump_max_duration_days", 1500.0)
                    ),
                    gaussian_bump_n_durations=int(
                        getattr(cfg, "pqc_gaussian_bump_n_durations", 6)
                    ),
                    gaussian_bump_min_points=int(
                        getattr(cfg, "pqc_gaussian_bump_min_points", 20)
                    ),
                    gaussian_bump_delta_chi2_thresh=float(
                        getattr(cfg, "pqc_gaussian_bump_delta_chi2_thresh", 25.0)
                    ),
                    gaussian_bump_suppress_overlap=bool(
                        getattr(cfg, "pqc_gaussian_bump_suppress_overlap", True)
                    ),
                    gaussian_bump_member_eta=float(
                        getattr(cfg, "pqc_gaussian_bump_member_eta", 1.0)
                    ),
                    gaussian_bump_freq_dependence=bool(
                        getattr(cfg, "pqc_gaussian_bump_freq_dependence", True)
                    ),
                    gaussian_bump_freq_alpha_min=float(
                        getattr(cfg, "pqc_gaussian_bump_freq_alpha_min", 0.0)
                    ),
                    gaussian_bump_freq_alpha_max=float(
                        getattr(cfg, "pqc_gaussian_bump_freq_alpha_max", 4.0)
                    ),
                    gaussian_bump_freq_alpha_tol=float(
                        getattr(cfg, "pqc_gaussian_bump_freq_alpha_tol", 1e-3)
                    ),
                    gaussian_bump_freq_alpha_max_iter=int(
                        getattr(cfg, "pqc_gaussian_bump_freq_alpha_max_iter", 64)
                    ),
                    glitch_enabled=bool(getattr(cfg, "pqc_glitch_enabled", False)),
                    glitch_min_points=int(getattr(cfg, "pqc_glitch_min_points", 30)),
                    glitch_delta_chi2_thresh=float(
                        getattr(cfg, "pqc_glitch_delta_chi2_thresh", 25.0)
                    ),
                    glitch_suppress_overlap=bool(
                        getattr(cfg, "pqc_glitch_suppress_overlap", True)
                    ),
                    glitch_member_eta=float(getattr(cfg, "pqc_glitch_member_eta", 1.0)),
                    glitch_peak_tau_days=float(
                        getattr(cfg, "pqc_glitch_peak_tau_days", 30.0)
                    ),
                    glitch_noise_k=float(getattr(cfg, "pqc_glitch_noise_k", 1.0)),
                    glitch_mean_window_days=float(
                        getattr(cfg, "pqc_glitch_mean_window_days", 180.0)
                    ),
                    glitch_min_duration_days=float(
                        getattr(cfg, "pqc_glitch_min_duration_days", 1000.0)
                    ),
                    backend_profiles_path=getattr(
                        cfg, "pqc_backend_profiles_path", None
                    ),
                )
                qc_out_dir = out_paths["qc"] / branch
                qc_out_dir.mkdir(parents=True, exist_ok=True)
                qc_settings_dir = out_paths["tag"] / "run_settings"
                qc_settings_dir.mkdir(parents=True, exist_ok=True)
                run_variants = bool(getattr(cfg, "pqc_run_variants", False))
                keep_variant_tmp = bool(getattr(cfg, "pqc_keep_variant_tmp", False))
                variant_tmp_root = (
                    out_paths["qc"] / ".pqc_variant_inputs" / str(branch)
                    if run_variants
                    else None
                )
                if variant_tmp_root is not None:
                    variant_tmp_root.mkdir(parents=True, exist_ok=True)

                tasks: List[tuple[str, str, Path, Path, Path]] = []
                for pulsar in pulsars:
                    psr_dir = dataset_root / pulsar
                    if run_variants:
                        variants = _discover_pqc_variants(psr_dir, pulsar)
                        if variants:
                            for variant in variants:
                                ws = (
                                    variant_tmp_root / pulsar / variant
                                    if variant_tmp_root is not None
                                    else psr_dir
                                )
                                try:
                                    parfile = _prepare_variant_pqc_workspace(
                                        psr_dir, pulsar, variant, ws
                                    )
                                except Exception as e:
                                    out_csv = qc_out_dir / f"{pulsar}.{variant}_qc.csv"
                                    qc_rows.append(
                                        {
                                            "pulsar": pulsar,
                                            "variant": variant,
                                            "branch": branch,
                                            "qc_csv": str(out_csv),
                                            "qc_status": "prepare_failed",
                                            "qc_error": str(e),
                                        }
                                    )
                                    continue
                                variant_all = ws / f"{pulsar}_all.tim"
                                if _variant_alltim_toa_count(variant_all) <= 0:
                                    out_csv = qc_out_dir / f"{pulsar}.{variant}_qc.csv"
                                    qc_rows.append(
                                        {
                                            "pulsar": pulsar,
                                            "variant": variant,
                                            "branch": branch,
                                            "qc_csv": str(out_csv),
                                            "qc_status": "empty_variant",
                                            "qc_error": "",
                                        }
                                    )
                                    logger.info(
                                        "Skipping PQC for %s.%s (%s): variant has no TOAs.",
                                        pulsar,
                                        variant,
                                        branch,
                                    )
                                    continue
                                out_csv = qc_out_dir / f"{pulsar}.{variant}_qc.csv"
                                settings_out = (
                                    qc_settings_dir
                                    / f"{pulsar}.{variant}.pqc_settings.toml"
                                )
                                tasks.append(
                                    (pulsar, variant, parfile, out_csv, settings_out)
                                )
                            continue

                    parfile = psr_dir / f"{pulsar}.par"
                    out_csv = qc_out_dir / f"{pulsar}_qc.csv"
                    settings_out = qc_settings_dir / f"{pulsar}.pqc_settings.toml"
                    tasks.append((pulsar, "base", parfile, out_csv, settings_out))

                def _run_pqc(
                    task: tuple[str, str, Path, Path, Path],
                ) -> Dict[str, object]:
                    p, variant, parfile, out_csv, settings_out = task
                    try:
                        df = run_pqc_for_parfile_subprocess(
                            parfile,
                            out_csv,
                            qc_cfg,
                            settings_out=settings_out,
                        )
                    except Exception as e:
                        logger.warning(
                            "pqc failed for %s (%s); skipping QC for this pulsar: %s",
                            p,
                            branch,
                            e,
                        )
                        return {
                            "pulsar": p,
                            "variant": variant,
                            "branch": branch,
                            "qc_csv": str(out_csv),
                            "qc_status": "pqc_failed",
                            "qc_error": str(e),
                        }
                    row = {
                        "pulsar": p,
                        "variant": variant,
                        "branch": branch,
                        "qc_csv": str(out_csv),
                        "qc_status": "success",
                    }
                    row.update(summarize_pqc(df))
                    return row

                def _execute_pqc_tasks(
                    task_list: List[tuple[str, str, Path, Path, Path]],
                    *,
                    n_jobs: int,
                    desc: str,
                ) -> List[Dict[str, object]]:
                    rows: List[Dict[str, object]] = []
                    if n_jobs == 1:
                        for task in tqdm(task_list, desc=desc):
                            rows.append(_run_pqc(task))
                        return rows
                    futures = []
                    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                        for task in task_list:
                            futures.append(ex.submit(_run_pqc, task))
                        for fut in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=desc,
                        ):
                            rows.append(fut.result())
                    return rows

                n_jobs = max(1, int(getattr(cfg, "jobs", 1) or 1))
                qc_rows.extend(
                    _execute_pqc_tasks(tasks, n_jobs=n_jobs, desc=f"pqc ({branch})")
                )

                auto_retry_enabled = bool(
                    getattr(cfg, "pqc_auto_retry_failed", False)
                )
                max_retry_passes = max(
                    0, int(getattr(cfg, "pqc_auto_retry_max_passes", 1) or 0)
                )
                retry_jobs = max(
                    1,
                    int(
                        getattr(
                            cfg,
                            "pqc_auto_retry_jobs",
                            min(2, n_jobs) if n_jobs > 1 else 1,
                        )
                        or 1
                    ),
                )
                if auto_retry_enabled and max_retry_passes > 0 and tasks:
                    tasks_by_key = {
                        (task[0], task[1]): task
                        for task in tasks
                    }
                    for retry_idx in range(1, max_retry_passes + 1):
                        failed_keys = {
                            (str(r.get("pulsar", "")), str(r.get("variant", "")))
                            for r in qc_rows
                            if str(r.get("qc_status", "")) == "pqc_failed"
                        }
                        retry_tasks = [
                            tasks_by_key[key]
                            for key in sorted(failed_keys)
                            if key in tasks_by_key
                        ]
                        if not retry_tasks:
                            break
                        logger.warning(
                            "Auto-retrying %d failed PQC task(s) for %s with jobs=%d (pass %d/%d).",
                            len(retry_tasks),
                            branch,
                            retry_jobs,
                            retry_idx,
                            max_retry_passes,
                        )
                        retry_rows = _execute_pqc_tasks(
                            retry_tasks,
                            n_jobs=retry_jobs,
                            desc=f"pqc retry {retry_idx} ({branch})",
                        )
                        merged_rows: Dict[tuple[str, str], Dict[str, object]] = {
                            (str(r.get("pulsar", "")), str(r.get("variant", ""))): r
                            for r in qc_rows
                        }
                        for row in retry_rows:
                            key = (
                                str(row.get("pulsar", "")),
                                str(row.get("variant", "")),
                            )
                            row["qc_retry_pass"] = retry_idx
                            row["qc_retry_attempted"] = True
                            merged_rows[key] = row
                        qc_rows = list(merged_rows.values())
                        remaining = sum(
                            1 for r in retry_rows if str(r.get("qc_status", "")) == "pqc_failed"
                        )
                        if remaining == 0:
                            break
                if bool(
                    getattr(cfg, "pqc_homogenize_outliers_across_variants", False)
                ):
                    outlier_cols = list(
                        getattr(cfg, "pqc_homogenize_outlier_cols", None)
                        or [
                            "bad_point",
                            "bad_hard",
                            "robust_outlier",
                            "robust_global_outlier",
                            "bad_mad",
                            "bad_ou",
                            "outlier_any",
                        ]
                    )
                    tol_seconds = float(
                        getattr(cfg, "pqc_homogenize_tol_seconds", None)
                        or getattr(cfg, "pqc_merge_tol_seconds", 2.0)
                        or 2.0
                    )
                    _homogenize_variant_outlier_flags(
                        qc_rows,
                        outlier_cols=outlier_cols,
                        tol_seconds=tol_seconds,
                    )
                if (
                    run_variants
                    and not keep_variant_tmp
                    and variant_tmp_root is not None
                ):
                    remove_tree_if_exists(variant_tmp_root)

            # Branch-level plots and tables (only for compare_branches, not the optional reference-only branch)
            if branch in compare_branches:
                if cfg.make_toa_coverage_plots:
                    plot_systems_per_pulsar(
                        cfg.home_dir,
                        dataset_root,
                        out_paths,
                        pulsars,
                        branch,
                        dpi=int(cfg.dpi),
                    )
                    plot_pulsars_per_system(
                        cfg.home_dir,
                        dataset_root,
                        out_paths,
                        pulsars,
                        branch,
                        dpi=int(cfg.dpi),
                    )

                if cfg.make_outlier_reports:
                    write_outlier_tables(
                        cfg.home_dir, dataset_root, out_paths, pulsars, [branch]
                    )

            # Binary analysis per branch
            if cfg.make_binary_analysis:
                bcfg = BinaryAnalysisConfig(only_models=cfg.binary_only_models)
                for pulsar in pulsars:
                    parfile = dataset_root / pulsar / f"{pulsar}.par"
                    row = analyse_binary_from_par(parfile)
                    if bcfg.only_models and row.get("BINARY") not in set(
                        bcfg.only_models
                    ):
                        continue
                    row["pulsar"] = pulsar
                    row["branch"] = branch
                    binary_rows.append(row)

        # Cross-branch reports
        if change_reports_enabled:
            branches_for_reports = list(compare_branches)
            if reference_branch not in branches_for_reports:
                branches_for_reports.append(reference_branch)
            write_change_reports(
                out_paths, pulsars, branches_for_reports, reference_branch
            )
            write_model_comparison_summary(
                out_paths, pulsars, branches_for_reports, reference_branch
            )
            write_new_param_significance(
                out_paths, pulsars, branches_for_reports, reference_branch
            )

        if cfg.make_covariance_heatmaps:
            plot_covmat_heatmaps(
                out_paths,
                pulsars,
                compare_branches,
                dpi=int(cfg.dpi),
                max_params=cfg.max_covmat_params,
            )

        if cfg.make_residual_plots:
            plot_residuals(out_paths, pulsars, compare_branches, dpi=int(cfg.dpi))

        if cfg.make_binary_analysis and binary_rows:
            df = pd.DataFrame(binary_rows)
            out_paths["binary_analysis"].mkdir(parents=True, exist_ok=True)
            df.to_csv(
                out_paths["binary_analysis"] / "binary_analysis.tsv",
                sep="\t",
                index=False,
            )

        if getattr(cfg, "run_pqc", False) and qc_rows:
            dfq = pd.DataFrame(qc_rows)
            dfq.to_csv(out_paths["qc"] / "qc_summary.tsv", sep="\t", index=False)
            if bool(getattr(cfg, "qc_cross_pulsar_enabled", False)):
                try:
                    cross_dir = None
                    if getattr(cfg, "qc_cross_pulsar_dir", None):
                        cross_dir = Path(cfg.qc_cross_pulsar_dir)
                        if not cross_dir.is_absolute():
                            cross_dir = out_paths["tag"] / cross_dir
                    cross_dir = generate_cross_pulsar_coincidence_report(
                        run_dir=out_paths["tag"],
                        report_dir=cross_dir,
                        time_col=getattr(cfg, "qc_cross_pulsar_time_col", None),
                        window_days=float(
                            getattr(cfg, "qc_cross_pulsar_window_days", 1.0)
                        ),
                        min_pulsars=int(getattr(cfg, "qc_cross_pulsar_min_pulsars", 2)),
                        include_outliers=bool(
                            getattr(cfg, "qc_cross_pulsar_include_outliers", True)
                        ),
                        include_events=bool(
                            getattr(cfg, "qc_cross_pulsar_include_events", True)
                        ),
                        outlier_cols=getattr(cfg, "qc_cross_pulsar_outlier_cols", None),
                        event_cols=getattr(cfg, "qc_cross_pulsar_event_cols", None),
                    )
                    if cross_dir is not None:
                        logger.info(
                            "Cross-pulsar coincidence report written to: %s", cross_dir
                        )
                except Exception as e:
                    logger.warning("Cross-pulsar coincidence stage failed: %s", e)

        if getattr(cfg, "qc_report", False):
            try:
                backend_col = str(
                    getattr(cfg, "qc_report_backend_col", None)
                    or getattr(cfg, "pqc_backend_col", "group")
                    or "group"
                )
                report_dir = None
                if getattr(cfg, "qc_report_dir", None):
                    report_dir = Path(cfg.qc_report_dir)
                    if not report_dir.is_absolute():
                        report_dir = out_paths["tag"] / report_dir
                report_dir = generate_qc_report(
                    run_dir=out_paths["tag"],
                    backend_col=backend_col,
                    backend=(
                        str(cfg.qc_report_backend)
                        if getattr(cfg, "qc_report_backend", None)
                        else None
                    ),
                    report_dir=report_dir,
                    no_plots=bool(getattr(cfg, "qc_report_no_plots", False)),
                    structure_group_cols=(
                        str(cfg.qc_report_structure_group_cols)
                        if getattr(cfg, "qc_report_structure_group_cols", None)
                        else None
                    ),
                    no_feature_plots=bool(
                        getattr(cfg, "qc_report_no_feature_plots", False)
                    ),
                    compact_pdf=bool(getattr(cfg, "qc_report_compact_pdf", False)),
                    compact_pdf_name=str(
                        getattr(
                            cfg, "qc_report_compact_pdf_name", "qc_compact_report.pdf"
                        )
                    ),
                    compact_outlier_cols=getattr(
                        cfg, "qc_report_compact_outlier_cols", None
                    ),
                )
                logger.info("QC report written to: %s", report_dir)
            except Exception as e:
                logger.warning("QC report generation failed: %s", e)

        if bool(getattr(cfg, "consolidated_report", True)):
            try:
                report_path = generate_run_report(
                    out_paths["tag"],
                    title=getattr(cfg, "consolidated_report_title", None)
                    or "PLEB Pipeline Run Report",
                    output_name=getattr(cfg, "consolidated_report_name", None)
                    or "run_report.pdf",
                    include_stages=getattr(cfg, "consolidated_report_stages", None),
                )
                if report_path is not None:
                    logger.info("Run report written to: %s", report_path)
            except Exception as e:
                logger.warning("Run report generation failed: %s", e)

        if getattr(cfg, "cleanup_work_dir", False):
            remove_tree_if_exists(out_paths["work"])
        if getattr(cfg, "cleanup_output_tree", False):
            cleanup_empty_dirs(out_paths["tag"])

        if fix_apply and len(branches_to_run) == 1:
            _commit_branch_artifacts(
                repo,
                cfg,
                branch=branches_to_run[0],
                out_paths=out_paths,
                commit_message=str(commit_message),
            )

        logger.info("Pipeline complete.")
        return out_paths

    finally:
        try:
            checkout(repo, current_branch)
        except Exception:
            pass
