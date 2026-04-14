"""Define configuration models for the data-combination pipeline.

This module provides :class:`PipelineConfig`, a flattened dataclass used by the
CLI and pipeline entry points to control data ingestion, fitting, reporting,
and optional FixDataset or parameter-scan stages. The config is intentionally
flat to simplify JSON/TOML serialization and CLI overrides.

See Also:
    pleb.pipeline.run_pipeline: Main pipeline entry point.
    pleb.param_scan.run_param_scan: Parameter scan entry point.
    pleb.cli: Command-line interface that consumes :class:`PipelineConfig`.
"""

from __future__ import annotations

from dataclasses import asdict, field
from .compat import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except Exception:  # pragma: no cover
        tomllib = None  # type: ignore


PulsarSelection = Union[str, List[str]]  # "ALL" or explicit list


def _resolve_repo_root(home_dir: Path | str) -> Path:
    """Resolve and validate the repository root.

    The contract is intentionally strict: ``home_dir`` must point to the Git
    repository root, i.e. the directory that contains ``.git``.
    """
    repo_root = Path(home_dir).expanduser().resolve()
    if not repo_root.exists():
        raise ValueError(f"home_dir does not exist: {repo_root}")
    if not repo_root.is_dir():
        raise ValueError(f"home_dir is not a directory: {repo_root}")
    if not (repo_root / ".git").exists():
        raise ValueError(
            "home_dir must be the git repo root (directory containing .git): "
            f"{repo_root}"
        )
    return repo_root


def _resolve_dataset_root(repo_root: Path, dataset_name: Path | str) -> Path:
    """Resolve ``dataset_name`` as a repo-relative dataset path under ``repo_root``.

    The contract is:
    - ``home_dir`` is the Git repo root
    - ``dataset_name`` is a path relative to that repo root

    Absolute dataset paths are rejected so configuration stays unambiguous.
    Paths that escape the repository root via ``..`` are also rejected.
    """
    ds = Path(dataset_name).expanduser()
    ds_raw = str(ds).strip()
    if not ds_raw:
        raise ValueError("dataset_name must be a non-empty path relative to home_dir")
    if ds.is_absolute():
        raise ValueError(
            "dataset_name must be a path relative to home_dir, not an absolute path: "
            f"{ds}"
        )
    dataset_root = (repo_root / ds).resolve()
    try:
        dataset_root.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(
            "dataset_name must resolve inside home_dir; refusing path outside repo "
            f"root: {ds}"
        ) from exc
    return dataset_root


@dataclass(slots=True)
class IngestConfig:
    """Configuration model for ingest-only CLI mode.

    Notes
    -----
    This config intentionally decouples ingest from full pipeline execution so
    users can ingest/commit datasets without specifying Tempo2/PQC settings.
    """

    ingest_mapping_file: Optional[Path] = None
    ingest_output_dir: Optional[Path] = None
    home_dir: Optional[Path] = None
    dataset_name: Optional[str] = None
    ingest_verify: bool = False
    ingest_commit_branch_name: Optional[str] = None
    ingest_commit_base_branch: Optional[str] = None
    ingest_commit_message: Optional[str] = None
    fix_ensure_ephem: Optional[str] = None
    fix_ensure_clk: Optional[str] = None
    fix_ensure_ne_sw: Optional[str] = None

    def resolved_output_root(self) -> Path:
        """Resolve ingest output root from explicit or fallback settings."""
        repo_root = None
        dataset_root = None
        if self.home_dir is not None:
            repo_root = _resolve_repo_root(self.home_dir)
            ds = self.dataset_name if self.dataset_name not in (None, "") else "."
            dataset_root = _resolve_dataset_root(repo_root, ds)
        if self.ingest_output_dir is not None and str(self.ingest_output_dir).strip():
            explicit_root = Path(self.ingest_output_dir).expanduser().resolve()
            if dataset_root is not None and explicit_root != dataset_root:
                raise ValueError(
                    "ingest_output_dir disagrees with home_dir + dataset_name: "
                    f"{explicit_root} != {dataset_root}"
                )
            if repo_root is not None:
                try:
                    explicit_root.relative_to(repo_root)
                except ValueError as exc:
                    raise ValueError(
                        "ingest_output_dir must resolve inside home_dir: "
                        f"{explicit_root}"
                    ) from exc
            return explicit_root
        if dataset_root is not None:
            return dataset_root
        raise ValueError(
            "Ingest output root is undefined. Set ingest_output_dir (or home_dir+dataset_name)."
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "ingest_verify": bool(self.ingest_verify),
            "dataset_name": self.dataset_name,
            "ingest_commit_branch_name": self.ingest_commit_branch_name,
            "ingest_commit_base_branch": self.ingest_commit_base_branch,
            "ingest_commit_message": self.ingest_commit_message,
            "fix_ensure_ephem": self.fix_ensure_ephem,
            "fix_ensure_clk": self.fix_ensure_clk,
            "fix_ensure_ne_sw": self.fix_ensure_ne_sw,
        }
        d["ingest_mapping_file"] = (
            str(self.ingest_mapping_file)
            if self.ingest_mapping_file is not None
            else None
        )
        d["ingest_output_dir"] = (
            str(self.ingest_output_dir) if self.ingest_output_dir is not None else None
        )
        d["home_dir"] = str(self.home_dir) if self.home_dir is not None else None
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "IngestConfig":
        if "pipeline" in d and isinstance(d["pipeline"], dict):
            d = d["pipeline"]

        def p_opt(x: Any) -> Optional[Path]:
            if x in (None, ""):
                return None
            return Path(x)

        def s_opt(x: Any) -> Optional[str]:
            if x in (None, ""):
                return None
            return str(x)

        return IngestConfig(
            ingest_mapping_file=p_opt(d.get("ingest_mapping_file")),
            ingest_output_dir=p_opt(d.get("ingest_output_dir")),
            home_dir=p_opt(d.get("home_dir")),
            dataset_name=s_opt(d.get("dataset_name")),
            ingest_verify=bool(d.get("ingest_verify", False)),
            ingest_commit_branch_name=s_opt(d.get("ingest_commit_branch_name")),
            ingest_commit_base_branch=s_opt(d.get("ingest_commit_base_branch")),
            ingest_commit_message=s_opt(d.get("ingest_commit_message")),
            fix_ensure_ephem=s_opt(d.get("fix_ensure_ephem")),
            fix_ensure_clk=s_opt(d.get("fix_ensure_clk")),
            fix_ensure_ne_sw=s_opt(d.get("fix_ensure_ne_sw")),
        )

    @staticmethod
    def load(path: Path) -> "IngestConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return IngestConfig.from_dict(data)
        if path.suffix.lower() in (".toml", ".tml"):
            if tomllib is None:
                raise RuntimeError(
                    "TOML config requested but tomllib is unavailable in this Python."
                )
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            return IngestConfig.from_dict(data)
        raise ValueError(
            f"Unsupported config file type: {path.suffix}. Use .json or .toml"
        )


@dataclass(slots=True)
class ParamScanConfig:
    """Configuration model for parameter-scan mode.

    Notes
    -----
    Statistical scan thresholds in this model (for example reduced
    chi-square gates and tested derivative orders) are passed through to
    :mod:`pleb.param_scan`; this class only stores and validates values.
    """

    home_dir: Path
    singularity_image: Path
    dataset_name: Optional[str] = None
    results_dir: Path = Path(".")
    reference_branch: str = "main"
    pulsars: PulsarSelection = "ALL"
    outdir_name: Optional[str] = None
    epoch: str = "55000"
    force_rerun: bool = False
    jobs: int = 1
    cleanup_output_tree: bool = True
    cleanup_work_dir: bool = False
    param_scan_typical: bool = True
    param_scan_dm_redchisq_threshold: float = 2.0
    param_scan_dm_max_order: int = 4
    param_scan_btx_max_fb: int = 3

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["home_dir"] = str(d["home_dir"])
        d["singularity_image"] = str(d["singularity_image"])
        d["results_dir"] = str(d["results_dir"])
        return d

    def to_pipeline_config(self) -> "PipelineConfig":
        return PipelineConfig.from_dict(self.to_dict())

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ParamScanConfig":
        if "param_scan" in d and isinstance(d["param_scan"], dict):
            d = d["param_scan"]
        elif "pipeline" in d and isinstance(d["pipeline"], dict):
            d = d["pipeline"]

        def p(x: Any) -> Path:
            return Path(x) if x is not None else Path(".")

        return ParamScanConfig(
            home_dir=p(d["home_dir"]),
            singularity_image=p(d["singularity_image"]),
            dataset_name=d.get("dataset_name", "."),
            results_dir=p(d.get("results_dir", ".")),
            reference_branch=str(d.get("reference_branch", "main")),
            pulsars=d.get("pulsars", "ALL"),
            outdir_name=(
                None if d.get("outdir_name") in (None, "") else d.get("outdir_name")
            ),
            epoch=str(d.get("epoch", "55000")),
            force_rerun=bool(d.get("force_rerun", False)),
            jobs=int(d.get("jobs", 1)),
            cleanup_output_tree=bool(d.get("cleanup_output_tree", True)),
            cleanup_work_dir=bool(d.get("cleanup_work_dir", False)),
            param_scan_typical=bool(d.get("param_scan_typical", True)),
            param_scan_dm_redchisq_threshold=float(
                d.get("param_scan_dm_redchisq_threshold", 2.0)
            ),
            param_scan_dm_max_order=int(d.get("param_scan_dm_max_order", 4)),
            param_scan_btx_max_fb=int(d.get("param_scan_btx_max_fb", 3)),
        )

    @staticmethod
    def load(path: Path) -> "ParamScanConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return ParamScanConfig.from_dict(data)
        if path.suffix.lower() in (".toml", ".tml"):
            if tomllib is None:
                raise RuntimeError(
                    "TOML config requested but tomllib is unavailable in this Python."
                )
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            return ParamScanConfig.from_dict(data)
        raise ValueError(
            f"Unsupported config file type: {path.suffix}. Use .json or .toml"
        )


@dataclass(slots=True)
class QCReportConfig:
    """Configuration model for ``qc-report`` mode.

    Notes
    -----
    This model configures report rendering only. It does not run PQC; it
    consumes existing ``*_qc.csv`` outputs and generates summary artifacts
    (plots, tables, optional compact PDF).
    """

    run_dir: Path
    backend_col: str = "group"
    backend: Optional[str] = None
    report_dir: Optional[Path] = None
    no_plots: bool = False
    structure_group_cols: Optional[str] = None
    no_feature_plots: bool = False
    compact_pdf: bool = False
    compact_pdf_name: str = "qc_compact_report.pdf"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "QCReportConfig":
        if "qc_report" in d and isinstance(d["qc_report"], dict):
            d = d["qc_report"]
        return QCReportConfig(
            run_dir=Path(d["run_dir"]),
            backend_col=str(d.get("backend_col", "group")),
            backend=(None if d.get("backend") in (None, "") else str(d.get("backend"))),
            report_dir=(
                None
                if d.get("report_dir") in (None, "")
                else Path(str(d.get("report_dir")))
            ),
            no_plots=bool(d.get("no_plots", False)),
            structure_group_cols=(
                None
                if d.get("structure_group_cols") in (None, "")
                else str(d.get("structure_group_cols"))
            ),
            no_feature_plots=bool(d.get("no_feature_plots", False)),
            compact_pdf=bool(d.get("compact_pdf", False)),
            compact_pdf_name=str(d.get("compact_pdf_name", "qc_compact_report.pdf")),
        )

    @staticmethod
    def load(path: Path) -> "QCReportConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return QCReportConfig.from_dict(data)
        if path.suffix.lower() in (".toml", ".tml"):
            if tomllib is None:
                raise RuntimeError(
                    "TOML config requested but tomllib is unavailable in this Python."
                )
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            return QCReportConfig.from_dict(data)
        raise ValueError(
            f"Unsupported config file type: {path.suffix}. Use .json or .toml"
        )


@dataclass(slots=True)
class WorkflowRunConfig:
    """Configuration model for workflow-file execution mode.

    Parameters
    ----------
    workflow_file : pathlib.Path
        Path to workflow definition (TOML/JSON) executed by
        :func:`pleb.workflow.run_workflow`.
    """

    workflow_file: Path

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WorkflowRunConfig":
        if "workflow" in d and isinstance(d["workflow"], dict):
            d = d["workflow"]
        wf = d.get("workflow_file")
        if wf in (None, ""):
            raise ValueError(
                "workflow config requires 'workflow_file' (or [workflow].workflow_file)."
            )
        return WorkflowRunConfig(workflow_file=Path(str(wf)))

    @staticmethod
    def load(path: Path) -> "WorkflowRunConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return WorkflowRunConfig.from_dict(data)
        if path.suffix.lower() in (".toml", ".tml"):
            if tomllib is None:
                raise RuntimeError(
                    "TOML config requested but tomllib is unavailable in this Python."
                )
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            return WorkflowRunConfig.from_dict(data)
        raise ValueError(
            f"Unsupported config file type: {path.suffix}. Use .json or .toml"
        )


@dataclass(slots=True)
class PipelineConfig:
    """Configure the data-combination pipeline.

    The configuration is intentionally flat so that it can be serialized to
    JSON/TOML and overridden via CLI flags without nested structures. Most
    fields correspond directly to CLI options and pipeline stages.

    Notes:
        The path contract is intentionally simple:

        - ``home_dir`` is the Git repository root
        - ``dataset_name`` is a path relative to ``home_dir``

        For example, with ``home_dir = "/repo"`` and
        ``dataset_name = "EPTA-DR3/epta-dr3-data-v0"``, the resolved dataset
        root is ``/repo/EPTA-DR3/epta-dr3-data-v0``.

    Examples:
        Basic construction and JSON save::

            cfg = PipelineConfig(
                home_dir=Path("/data/epta"),
                singularity_image=Path("/images/tempo2.sif"),
                dataset_name="EPTA",
            )
            cfg.save_json(Path("pipeline.json"))

    Attributes:
        home_dir: Git repository root containing ``.git``.
        singularity_image: Singularity/Apptainer image with tempo2.
        dataset_name: Dataset path relative to ``home_dir``.
        results_dir: Output directory for pipeline reports.
        branches: Git branches to run/compare.
        reference_branch: Branch used as change-report reference.
        pulsars: "ALL" or an explicit list of pulsar names.
        outdir_name: Optional output subdirectory name.
        epoch: Tempo2 epoch used for fitting.
        force_rerun: Re-run tempo2 even if outputs exist.
        jobs: Parallel workers per branch.
        run_tempo2: Whether to run tempo2.
        make_toa_coverage_plots: Generate coverage plots.
        make_change_reports: Generate change reports.
        make_covariance_heatmaps: Generate covariance heatmaps.
        make_residual_plots: Generate residual plots.
        make_outlier_reports: Generate outlier tables.
        make_plots: Convenience toggle to disable all plotting outputs.
        make_reports: Convenience toggle to disable report outputs.
        make_covmat: Convenience toggle to control covariance heatmaps.
        testing_mode: If True, skip change reports (useful for CI).
        run_pqc: Enable optional pqc stage.
        run_whitenoise: Enable optional whitenoise EFAC/EQUAD/ECORR estimation stage.
        whitenoise_source_path: Optional path to a folder containing
            ``whitenoise_estimator.py`` when not importable from the environment.
        whitenoise_epoch_tolerance_seconds: Epoch grouping tolerance in seconds.
        whitenoise_single_toa_mode: Single-TOA identifiability mode for whitenoise
            (``combined``, ``equad0``, or ``ecorr0``).
        whitenoise_fit_timing_model_first: Run a timing-model fit in libstempo
            before estimating whitenoise parameters.
        whitenoise_timfile_name: Optional timfile name template used per pulsar for
            whitenoise (supports ``{pulsar}``).
        pqc_backend_col: Backend grouping column for pqc.
        pqc_drop_unmatched: Drop unmatched TOAs in pqc.
        pqc_merge_tol_seconds: Merge tolerance in seconds for pqc.
        pqc_tau_corr_minutes: OU correlation time for pqc.
        pqc_fdr_q: False discovery rate for pqc.
        pqc_mark_only_worst_per_day: Mark only worst TOA per day.
        pqc_tau_rec_days: Recovery time for transient scan.
        pqc_window_mult: Window multiplier for transient scan.
        pqc_min_points: Minimum points for transient scan.
        pqc_delta_chi2_thresh: Delta-chi2 threshold for transients.
        pqc_exp_dip_min_duration_days: Minimum duration (days) for exp dips.
        pqc_add_orbital_phase: Add orbital-phase feature.
        pqc_add_solar_elongation: Add solar elongation feature.
        pqc_add_elevation: Add elevation feature.
        pqc_add_airmass: Add airmass feature.
        pqc_add_parallactic_angle: Add parallactic-angle feature.
        pqc_add_freq_bin: Add frequency-bin feature.
        pqc_freq_bins: Number of frequency bins if enabled.
        pqc_observatory_path: Optional observatory file path.
        pqc_structure_mode: Feature-structure mode (none/detrend/test/both).
        pqc_structure_detrend_features: Features to detrend against.
        pqc_structure_test_features: Features to test for structure.
        pqc_structure_nbins: Bin count for structure tests.
        pqc_structure_min_per_bin: Minimum points per bin.
        pqc_structure_p_thresh: p-value threshold for structure detection.
        pqc_structure_circular_features: Circular features in [0,1).
        pqc_structure_group_cols: Grouping columns for structure tests.
        pqc_outlier_gate_enabled: Enable hard sigma gate for outlier membership.
        pqc_outlier_gate_sigma: Sigma threshold for outlier gate.
        pqc_outlier_gate_resid_col: Residual column to gate on (optional).
        pqc_outlier_gate_sigma_col: Sigma column to gate on (optional).
        pqc_event_instrument: Enable per-event membership diagnostics.
        pqc_solar_events_enabled: Enable solar event detection.
        pqc_solar_approach_max_deg: Max elongation for solar approach region.
        pqc_solar_min_points_global: Min points for global fit.
        pqc_solar_min_points_year: Min points for per-year fit.
        pqc_solar_min_points_near_zero: Min points near zero elongation.
        pqc_solar_tau_min_deg: Min elongation scale for exponential.
        pqc_solar_tau_max_deg: Max elongation scale for exponential.
        pqc_solar_member_eta: Per-point membership SNR threshold.
        pqc_solar_freq_dependence: Fit 1/f^alpha dependence.
        pqc_solar_freq_alpha_min: Lower bound for alpha.
        pqc_solar_freq_alpha_max: Upper bound for alpha.
        pqc_solar_freq_alpha_tol: Optimization tolerance for alpha.
        pqc_solar_freq_alpha_max_iter: Max iterations for alpha optimizer.
        pqc_orbital_phase_cut_enabled: Enable orbital-phase based flagging.
        pqc_orbital_phase_cut_center: Eclipse center phase (0..1).
        pqc_orbital_phase_cut: Fixed orbital-phase cutoff (0..0.5), or None for auto.
        pqc_orbital_phase_cut_sigma: Sigma threshold for automatic cutoff estimation.
        pqc_orbital_phase_cut_nbins: Number of bins for cutoff estimation.
        pqc_orbital_phase_cut_min_points: Minimum points for cutoff estimation.
        pqc_eclipse_events_enabled: Enable eclipse event detection.
        pqc_eclipse_center_phase: Eclipse center phase (0..1).
        pqc_eclipse_min_points: Min points for global fit.
        pqc_eclipse_width_min: Min eclipse width in phase.
        pqc_eclipse_width_max: Max eclipse width in phase.
        pqc_eclipse_member_eta: Per-point membership SNR threshold.
        pqc_eclipse_freq_dependence: Fit 1/f^alpha dependence.
        pqc_eclipse_freq_alpha_min: Lower bound for alpha.
        pqc_eclipse_freq_alpha_max: Upper bound for alpha.
        pqc_eclipse_freq_alpha_tol: Optimization tolerance for alpha.
        pqc_eclipse_freq_alpha_max_iter: Max iterations for alpha optimizer.
        pqc_gaussian_bump_enabled: Enable Gaussian-bump event detection.
        pqc_gaussian_bump_min_duration_days: Minimum bump duration in days.
        pqc_gaussian_bump_max_duration_days: Maximum bump duration in days.
        pqc_gaussian_bump_n_durations: Number of duration grid points.
        pqc_gaussian_bump_min_points: Minimum points for bump detection.
        pqc_gaussian_bump_delta_chi2_thresh: Delta-chi2 threshold for bump detection.
        pqc_gaussian_bump_suppress_overlap: Suppress overlapping bumps.
        pqc_gaussian_bump_member_eta: Per-point membership SNR threshold.
        pqc_gaussian_bump_freq_dependence: Fit 1/f^alpha dependence.
        pqc_gaussian_bump_freq_alpha_min: Lower bound for alpha.
        pqc_gaussian_bump_freq_alpha_max: Upper bound for alpha.
        pqc_gaussian_bump_freq_alpha_tol: Optimization tolerance for alpha.
        pqc_gaussian_bump_freq_alpha_max_iter: Max iterations for alpha optimizer.
        pqc_glitch_enabled: Enable glitch event detection.
        pqc_glitch_min_points: Minimum points for glitch detection.
        pqc_glitch_delta_chi2_thresh: Delta-chi2 threshold for glitch detection.
        pqc_glitch_suppress_overlap: Suppress overlapping glitches.
        pqc_glitch_member_eta: Per-point membership SNR threshold.
        pqc_glitch_peak_tau_days: Peak exponential timescale for glitch model.
        pqc_glitch_noise_k: Noise-aware threshold multiplier.
        pqc_glitch_mean_window_days: Rolling-mean window (days) for zero-crossing.
        pqc_glitch_min_duration_days: Minimum glitch duration (days).
        pqc_backend_profiles_path: Optional TOML with per-backend pqc overrides.
        qc_report: Generate pqc report artifacts after the run.
        qc_report_backend_col: Backend column name for reports (optional).
        qc_report_backend: Optional backend key to plot.
        qc_report_dir: Output directory for reports (optional).
        qc_report_no_plots: Skip transient plots in reports.
        qc_report_structure_group_cols: Structure grouping override for reports.
        qc_report_no_feature_plots: Skip feature plots in reports.
        qc_report_compact_pdf: Generate compact composite PDF report.
        qc_report_compact_pdf_name: Filename for compact PDF report.
        run_fix_dataset: Enable FixDataset stage.
        make_binary_analysis: Enable binary analysis table.
        param_scan_typical: Enable typical param-scan profile.
        param_scan_dm_redchisq_threshold: Threshold for DM scan.
        param_scan_dm_max_order: Max DM derivative order.
        param_scan_btx_max_fb: Max FB derivative order.
        fix_apply: Whether FixDataset applies changes and commits.
        fix_branch_name: Name of FixDataset branch. If unset and fix_apply is true,
            a name is auto-generated as ``branch_run_ddmmyyhhmm``.
        fix_base_branch: Base branch for FixDataset.
        fix_commit_message: Commit message for FixDataset.
        fix_backup: Create backup before FixDataset modifications.
        fix_dry_run: If True, FixDataset does not write changes.
        fix_update_alltim_includes: Update INCLUDE lines in .tim files.
        fix_min_toas_per_backend_tim: Minimum TOAs per backend .tim.
        fix_required_tim_flags: Required flags for .tim entries.
        fix_system_flag_mapping_path: Editable system-flag mapping JSON (optional).
        fix_system_flag_mapping_path: Editable system-flag mapping JSON (optional).
        fix_relabel_rules_path: Declarative TOA relabel rules TOML (optional).
        fix_overlap_rules_path: Declarative overlap rules TOML (optional).
        fix_overlap_exact_catalog_path: TOML keep->drop map for exact overlap removal.
        fix_jump_reference_variants: Build per-variant reference-system jump parfiles.
        fix_jump_reference_keep_tmp: Keep temporary split tim/par files.
        fix_jump_reference_jump_flag: Jump flag used in generated variant parfiles.
        fix_jump_reference_csv_dir: Output directory for jump-reference CSV files.
        fix_insert_missing_jumps: Insert missing JUMP lines.
        fix_jump_flag: Flag used for inserted jumps.
        fix_prune_stale_jumps: Drop JUMPs not present in timfile flags.
        fix_ensure_ephem: Ensure ephemeris parameter exists.
        fix_ensure_clk: Ensure clock parameter exists.
        fix_ensure_ne_sw: Ensure NE_SW parameter exists.
        fix_force_ne_sw_overwrite: Overwrite existing NE_SW values when true.
        fix_remove_patterns: Patterns to remove from .par/.tim.
        fix_coord_convert: Optional coordinate conversion.
        fix_qc_remove_outliers: Comment/delete TOAs flagged by pqc outputs.
        fix_qc_action: Action for pqc outliers (comment/delete).
        fix_qc_comment_prefix: Prefix for commented TOA lines.
        fix_qc_backend_col: Backend column for pqc matching (if needed).
        fix_qc_remove_bad: Act on bad/bad_day flags.
        fix_qc_remove_transients: Act on transient flags.
        fix_qc_remove_solar: Act on solar-elongation flags.
        fix_qc_solar_action: Action for solar-flagged TOAs (comment/delete).
        fix_qc_solar_comment_prefix: Prefix for solar-flagged TOA comments.
        fix_qc_remove_orbital_phase: Act on orbital-phase flags.
        fix_qc_orbital_phase_action: Action for orbital-phase flagged TOAs (comment/delete).
        fix_qc_orbital_phase_comment_prefix: Prefix for orbital-phase TOA comments.
        fix_qc_write_pqc_flag: Add ``-pqc`` classification flag to TOA rows.
        fix_qc_pqc_flag_name: TOA flag token used for QC class (default ``-pqc``).
        fix_qc_pqc_good_value: Value for non-outlier, non-event TOAs.
        fix_qc_pqc_bad_value: Value for outlier-only TOAs.
        fix_qc_pqc_event_prefix: Prefix for event values (for example ``event_step``).
        fix_qc_merge_tol_days: MJD tolerance when matching TOAs.
        fix_qc_results_dir: Directory containing pqc CSV outputs. If unset and
            fix_apply is true, defaults to ``<results>/qc/<fix_branch_name>``.
        fix_qc_branch: Branch subdir for pqc CSV outputs. If unset and
            fix_qc_results_dir is set, defaults to ``fix_branch_name``.
        binary_only_models: Limit binary analysis to model names.
        dpi: Plot resolution.
        max_covmat_params: Max params in covariance heatmaps.
        ingest_mapping_file: JSON mapping file for ingest mode (optional).
        ingest_output_dir: Output root directory for ingest mode (optional).
        ingest_commit_branch: Create a new branch and commit ingest outputs.
        ingest_commit_branch_name: Optional name for the ingest branch.
        ingest_commit_base_branch: Base branch for the ingest commit.
        ingest_commit_message: Commit message for ingest.
        compare_public_out_dir: Optional output directory for public-release
            comparisons.
        compare_public_providers_path: Optional providers catalog path for
            public-release comparisons.
    """

    # Root of the data repository (contains pulsar folders like Jxxxx+xxxx/)
    home_dir: Path

    # Singularity/Apptainer image containing tempo2
    singularity_image: Path

    # Name of the dataset (EPTA likes to use different combinations)
    dataset_name: Optional[str] = None

    # Where to write the report output
    results_dir: Path = Path(".")

    # Branches you want to compare/diagnose
    branches: List[str] = field(default_factory=lambda: ["main", "EPTA"])

    # Reference branch for change reports
    reference_branch: str = "main"

    # Pulsars to process: "ALL" or list
    pulsars: PulsarSelection = "ALL"

    # Output directory name: None -> timestamped
    outdir_name: Optional[str] = None

    # Remove empty output folders after a run
    cleanup_output_tree: bool = True

    # Remove work/ scratch directories after successful runs
    cleanup_work_dir: bool = False

    # tempo2 settings
    epoch: str = "55000"

    # If True, re-run tempo2 even if outputs already exist
    force_rerun: bool = False

    # Number of parallel workers to run pulsars concurrently (per branch). 1 = sequential.
    jobs: int = 1

    # Pipeline toggles
    run_tempo2: bool = True
    make_toa_coverage_plots: bool = True
    make_change_reports: bool = True
    make_covariance_heatmaps: bool = True
    make_residual_plots: bool = True
    make_outlier_reports: bool = True
    make_plots: Optional[bool] = None
    make_reports: Optional[bool] = None
    make_covmat: Optional[bool] = None
    testing_mode: bool = False

    # Optional outlier/QC stage using the external `pqc` package (libstempo-based).
    # This is separate from the existing general2-based outlier summaries.
    run_pqc: bool = False
    run_whitenoise: bool = False
    whitenoise_source_path: Optional[str] = None
    whitenoise_epoch_tolerance_seconds: float = 1.0
    whitenoise_single_toa_mode: str = "combined"
    whitenoise_fit_timing_model_first: bool = True
    whitenoise_timfile_name: Optional[str] = None
    pqc_backend_col: str = "group"
    pqc_drop_unmatched: bool = False
    pqc_merge_tol_seconds: float = 2.0
    pqc_tau_corr_minutes: float = 30.0
    pqc_fdr_q: float = 0.01
    pqc_mark_only_worst_per_day: bool = True
    pqc_tau_rec_days: float = 7.0
    pqc_window_mult: float = 5.0
    pqc_min_points: int = 6
    pqc_delta_chi2_thresh: float = 25.0
    pqc_exp_dip_min_duration_days: float = 21.0

    pqc_step_enabled: bool = True
    pqc_step_min_points: int = 20
    pqc_step_delta_chi2_thresh: float = 25.0
    pqc_step_scope: str = "both"

    pqc_dm_step_enabled: bool = True
    pqc_dm_step_min_points: int = 20
    pqc_dm_step_delta_chi2_thresh: float = 25.0
    pqc_dm_step_scope: str = "both"

    pqc_robust_enabled: bool = True
    pqc_robust_z_thresh: float = 5.0
    pqc_robust_scope: str = "both"
    pqc_add_orbital_phase: bool = True
    pqc_add_solar_elongation: bool = True
    pqc_add_elevation: bool = False
    pqc_add_airmass: bool = False
    pqc_add_parallactic_angle: bool = False
    pqc_add_freq_bin: bool = False
    pqc_freq_bins: int = 8
    pqc_observatory_path: Optional[str] = None
    pqc_structure_mode: str = "none"
    pqc_structure_detrend_features: Optional[List[str]] = field(
        default_factory=lambda: ["solar_elongation_deg", "orbital_phase"]
    )
    pqc_structure_test_features: Optional[List[str]] = field(
        default_factory=lambda: ["solar_elongation_deg", "orbital_phase"]
    )
    pqc_structure_nbins: int = 12
    pqc_structure_min_per_bin: int = 3
    pqc_structure_p_thresh: float = 0.01
    pqc_structure_circular_features: Optional[List[str]] = field(
        default_factory=lambda: ["orbital_phase"]
    )
    pqc_structure_group_cols: Optional[List[str]] = None
    pqc_outlier_gate_enabled: bool = False
    pqc_outlier_gate_sigma: float = 3.0
    pqc_outlier_gate_resid_col: Optional[str] = None
    pqc_outlier_gate_sigma_col: Optional[str] = None
    pqc_event_instrument: bool = False
    pqc_solar_events_enabled: bool = False
    pqc_solar_approach_max_deg: float = 30.0
    pqc_solar_min_points_global: int = 30
    pqc_solar_min_points_year: int = 10
    pqc_solar_min_points_near_zero: int = 3
    pqc_solar_tau_min_deg: float = 2.0
    pqc_solar_tau_max_deg: float = 60.0
    pqc_solar_member_eta: float = 1.0
    pqc_solar_freq_dependence: bool = True
    pqc_solar_freq_alpha_min: float = 0.0
    pqc_solar_freq_alpha_max: float = 4.0
    pqc_solar_freq_alpha_tol: float = 1e-3
    pqc_solar_freq_alpha_max_iter: int = 64
    pqc_orbital_phase_cut_enabled: bool = False
    pqc_orbital_phase_cut_center: float = 0.25
    pqc_orbital_phase_cut: Optional[float] = None
    pqc_orbital_phase_cut_sigma: float = 3.0
    pqc_orbital_phase_cut_nbins: int = 18
    pqc_orbital_phase_cut_min_points: int = 20

    pqc_eclipse_events_enabled: bool = False
    pqc_eclipse_center_phase: float = 0.25
    pqc_eclipse_min_points: int = 30
    pqc_eclipse_width_min: float = 0.01
    pqc_eclipse_width_max: float = 0.5
    pqc_eclipse_member_eta: float = 1.0
    pqc_eclipse_freq_dependence: bool = True
    pqc_eclipse_freq_alpha_min: float = 0.0
    pqc_eclipse_freq_alpha_max: float = 4.0
    pqc_eclipse_freq_alpha_tol: float = 1e-3
    pqc_eclipse_freq_alpha_max_iter: int = 64

    pqc_gaussian_bump_enabled: bool = False
    pqc_gaussian_bump_min_duration_days: float = 60.0
    pqc_gaussian_bump_max_duration_days: float = 1500.0
    pqc_gaussian_bump_n_durations: int = 6
    pqc_gaussian_bump_min_points: int = 20
    pqc_gaussian_bump_delta_chi2_thresh: float = 25.0
    pqc_gaussian_bump_suppress_overlap: bool = True
    pqc_gaussian_bump_member_eta: float = 1.0
    pqc_gaussian_bump_freq_dependence: bool = True
    pqc_gaussian_bump_freq_alpha_min: float = 0.0
    pqc_gaussian_bump_freq_alpha_max: float = 4.0
    pqc_gaussian_bump_freq_alpha_tol: float = 1e-3
    pqc_gaussian_bump_freq_alpha_max_iter: int = 64

    pqc_glitch_enabled: bool = False
    pqc_glitch_min_points: int = 30
    pqc_glitch_delta_chi2_thresh: float = 25.0
    pqc_glitch_suppress_overlap: bool = True
    pqc_glitch_member_eta: float = 1.0
    pqc_glitch_peak_tau_days: float = 30.0
    pqc_glitch_noise_k: float = 1.0
    pqc_glitch_mean_window_days: float = 180.0
    pqc_glitch_min_duration_days: float = 1000.0
    pqc_backend_profiles_path: Optional[str] = None
    pqc_run_variants: bool = False
    pqc_keep_variant_tmp: bool = False

    # Optional reporting for pqc outputs
    qc_report: bool = False
    qc_report_backend_col: Optional[str] = None
    qc_report_backend: Optional[str] = None
    qc_report_dir: Optional[Path] = None
    qc_report_no_plots: bool = False
    qc_report_structure_group_cols: Optional[str] = None
    qc_report_no_feature_plots: bool = False
    qc_report_compact_pdf: bool = False
    qc_report_compact_pdf_name: str = "qc_compact_report.pdf"
    qc_report_compact_outlier_cols: Optional[List[str]] = None
    qc_cross_pulsar_enabled: bool = False
    qc_cross_pulsar_time_col: Optional[str] = None
    qc_cross_pulsar_window_days: float = 1.0
    qc_cross_pulsar_min_pulsars: int = 2
    qc_cross_pulsar_include_outliers: bool = True
    qc_cross_pulsar_include_events: bool = True
    qc_cross_pulsar_outlier_cols: Optional[List[str]] = None
    qc_cross_pulsar_event_cols: Optional[List[str]] = None
    qc_cross_pulsar_dir: Optional[Path] = None

    # Add-on toggles (extras from FixDataset.ipynb + AnalysePulsars.ipynb)
    run_fix_dataset: bool = False
    make_binary_analysis: bool = False

    # ---- Param scan defaults (used by --param-scan --scan-typical) ----
    param_scan_typical: bool = False
    param_scan_dm_redchisq_threshold: float = 2.0
    param_scan_dm_max_order: int = 4
    param_scan_btx_max_fb: int = 3

    # ---- FixDataset settings (flattened) ----
    fix_apply: bool = False
    fix_branch_name: Optional[str] = None
    fix_base_branch: Optional[str] = None
    fix_commit_message: Optional[str] = None
    fix_backup: bool = True
    fix_dry_run: bool = False

    fix_update_alltim_includes: bool = True
    fix_min_toas_per_backend_tim: int = 10

    # Example: {"-pta": "EPTA", "-be": "P200"}
    fix_required_tim_flags: Dict[str, str] = field(default_factory=dict)
    fix_system_flag_mapping_path: Optional[str] = None
    fix_system_flag_table_path: Optional[str] = None
    fix_flag_sys_freq_rules_enabled: bool = False
    fix_flag_sys_freq_rules_path: Optional[str] = None
    fix_generate_alltim_variants: bool = False
    fix_backend_classifications_path: Optional[str] = None
    fix_alltim_variants_path: Optional[str] = None
    fix_relabel_rules_path: Optional[str] = None
    fix_overlap_rules_path: Optional[str] = None
    fix_overlap_exact_catalog_path: Optional[str] = None
    fix_jump_reference_variants: bool = False
    fix_jump_reference_keep_tmp: bool = False
    fix_jump_reference_jump_flag: str = "-sys"
    fix_jump_reference_csv_dir: Optional[str] = None
    fix_infer_system_flags: bool = False
    fix_system_flag_overwrite_existing: bool = False
    fix_wsrt_p2_force_sys_by_freq: bool = False
    fix_wsrt_p2_prefer_dual_channel: bool = False
    fix_wsrt_p2_mjd_tol_sec: float = 0.99e-6
    fix_wsrt_p2_action: str = "comment"
    fix_wsrt_p2_comment_prefix: str = "C WSRT_P2_PREFER_DUAL"
    fix_backend_overrides: Dict[str, str] = field(default_factory=dict)
    fix_raise_on_backend_missing: bool = False
    fix_dedupe_toas_within_tim: bool = True
    fix_dedupe_mjd_tol_sec: float = 0.0
    fix_dedupe_freq_tol_mhz: Optional[float] = None
    fix_dedupe_freq_tol_auto: bool = False
    fix_check_duplicate_backend_tims: bool = False
    fix_remove_overlaps_exact: bool = True

    fix_insert_missing_jumps: bool = True
    fix_jump_flag: str = "-sys"
    fix_prune_stale_jumps: bool = False
    fix_ensure_ephem: Optional[str] = None
    fix_ensure_clk: Optional[str] = None
    fix_ensure_ne_sw: Optional[str] = None
    fix_force_ne_sw_overwrite: bool = False
    fix_remove_patterns: List[str] = field(
        default_factory=lambda: ["NRT.NUPPI.", "NRT.NUXPI."]
    )
    # "equatorial_to_ecliptic" or "ecliptic_to_equatorial"
    fix_coord_convert: Optional[str] = None
    fix_prune_missing_includes: bool = True
    fix_drop_small_backend_includes: bool = True
    fix_system_flag_update_table: bool = True
    fix_default_backend: Optional[str] = None
    fix_group_flag: str = "-group"
    fix_pta_flag: str = "-pta"
    fix_pta_value: Optional[str] = None
    fix_standardize_par_values: bool = True
    fix_prune_small_system_toas: bool = False
    fix_prune_small_system_flag: str = "-sys"

    # ---- PQC outlier application (optional) ----
    fix_qc_remove_outliers: bool = False
    fix_qc_outlier_cols: Optional[List[str]] = None
    fix_qc_action: str = "comment"
    fix_qc_comment_prefix: str = "C QC_OUTLIER"
    fix_qc_backend_col: str = "sys"
    fix_qc_remove_bad: bool = True
    fix_qc_remove_transients: bool = False
    fix_qc_remove_solar: bool = False
    fix_qc_solar_action: str = "comment"
    fix_qc_solar_comment_prefix: str = "C QC_SOLAR"
    fix_qc_remove_orbital_phase: bool = False
    fix_qc_orbital_phase_action: str = "comment"
    fix_qc_orbital_phase_comment_prefix: str = "C QC_BIANRY_ECLIPSE"
    fix_qc_write_pqc_flag: bool = False
    fix_qc_pqc_flag_name: str = "-pqc"
    fix_qc_pqc_good_value: str = "good"
    fix_qc_pqc_bad_value: str = "bad"
    fix_qc_pqc_event_prefix: str = "event_"
    fix_qc_merge_tol_days: float = 2.0 / 86400.0
    fix_qc_results_dir: Optional[Path] = None
    fix_qc_branch: Optional[str] = None

    # ---- Binary analysis settings ----
    binary_only_models: Optional[List[str]] = None

    # Plotting controls
    dpi: int = 120
    max_covmat_params: Optional[int] = None

    # ---- Ingest mode (mapping-driven) ----
    ingest_mapping_file: Optional[Path] = None
    ingest_output_dir: Optional[Path] = None
    ingest_commit_branch: bool = True
    ingest_commit_branch_name: Optional[str] = None
    ingest_commit_base_branch: Optional[str] = None
    ingest_commit_message: Optional[str] = None
    ingest_verify: bool = False
    compare_public_out_dir: Optional[Path] = None
    compare_public_providers_path: Optional[Path] = None

    def resolved(self) -> "PipelineConfig":
        """Return a copy with paths expanded and resolved.

        Contract:

        1) ``home_dir`` must be the Git repository root
        2) ``dataset_name`` must be a path relative to ``home_dir``
        3) ``dataset_name`` may not be absolute or escape outside ``home_dir``

        Returns:
            A new :class:`PipelineConfig` with resolved paths.

        Raises:
            TypeError: If ``dataset_name`` is ``None`` (it must be a string or
                path-like value when resolving).
            ValueError: If ``home_dir`` is not a Git repo root, or
                ``dataset_name`` is not a valid repo-relative path.

        Examples:
            Resolve a dataset by name relative to ``home_dir``::

                cfg = PipelineConfig(
                    home_dir=Path("/data/epta"),
                    singularity_image=Path("/images/tempo2.sif"),
                    dataset_name="EPTA",
                )
                resolved = cfg.resolved()
                assert resolved.dataset_name == Path("/data/epta/EPTA")
        """
        c = PipelineConfig(
            **{
                **asdict(self),
                "home_dir": Path(self.home_dir),
                "dataset_name": Path(self.dataset_name),
                "singularity_image": Path(self.singularity_image),
                "results_dir": Path(self.results_dir),
            }
        )
        c.home_dir = _resolve_repo_root(c.home_dir)
        c.dataset_name = _resolve_dataset_root(c.home_dir, c.dataset_name)

        c.singularity_image = c.singularity_image.expanduser().resolve()
        # Keep results under home_dir when results_dir is relative.
        if not c.results_dir.is_absolute():
            c.results_dir = (Path(c.home_dir) / c.results_dir).expanduser().resolve()
        else:
            c.results_dir = c.results_dir.expanduser().resolve()
        if c.qc_report_dir is not None:
            c.qc_report_dir = Path(c.qc_report_dir).expanduser().resolve()
        if c.qc_cross_pulsar_dir is not None:
            c.qc_cross_pulsar_dir = Path(c.qc_cross_pulsar_dir).expanduser().resolve()
        if c.fix_qc_results_dir is not None:
            c.fix_qc_results_dir = Path(c.fix_qc_results_dir).expanduser().resolve()
        if c.pqc_backend_profiles_path is not None:
            c.pqc_backend_profiles_path = (
                Path(c.pqc_backend_profiles_path).expanduser().resolve()
            )
        if c.whitenoise_source_path is not None:
            c.whitenoise_source_path = (
                Path(c.whitenoise_source_path).expanduser().resolve()
            )
        if c.fix_system_flag_mapping_path is not None:
            c.fix_system_flag_mapping_path = (
                Path(c.fix_system_flag_mapping_path).expanduser().resolve()
            )
        if c.fix_flag_sys_freq_rules_path is not None:
            c.fix_flag_sys_freq_rules_path = (
                Path(c.fix_flag_sys_freq_rules_path).expanduser().resolve()
            )
        if c.fix_system_flag_table_path is not None:
            c.fix_system_flag_table_path = (
                Path(c.fix_system_flag_table_path).expanduser().resolve()
            )
        if c.fix_backend_classifications_path is not None:
            c.fix_backend_classifications_path = (
                Path(c.fix_backend_classifications_path).expanduser().resolve()
            )
        if c.fix_alltim_variants_path is not None:
            c.fix_alltim_variants_path = (
                Path(c.fix_alltim_variants_path).expanduser().resolve()
            )
        if c.fix_relabel_rules_path is not None:
            c.fix_relabel_rules_path = (
                Path(c.fix_relabel_rules_path).expanduser().resolve()
            )
        if c.fix_overlap_rules_path is not None:
            c.fix_overlap_rules_path = (
                Path(c.fix_overlap_rules_path).expanduser().resolve()
            )
        if c.fix_overlap_exact_catalog_path is not None:
            c.fix_overlap_exact_catalog_path = (
                Path(c.fix_overlap_exact_catalog_path).expanduser().resolve()
            )
        if c.ingest_mapping_file is not None:
            c.ingest_mapping_file = Path(c.ingest_mapping_file).expanduser().resolve()
        if c.ingest_output_dir is not None:
            c.ingest_output_dir = Path(c.ingest_output_dir).expanduser().resolve()
        if c.compare_public_out_dir is not None:
            c.compare_public_out_dir = (
                Path(c.compare_public_out_dir).expanduser().resolve()
            )
        if c.compare_public_providers_path is not None:
            c.compare_public_providers_path = (
                Path(c.compare_public_providers_path).expanduser().resolve()
            )
        return c

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a JSON-friendly dict.

        Returns:
            Dictionary representation of the config with :class:`Path` values
            converted to strings.

        Examples:
            Convert to a dict suitable for JSON serialization::

                cfg = PipelineConfig(
                    home_dir=Path("/data/epta"),
                    singularity_image=Path("/images/tempo2.sif"),
                    dataset_name="EPTA",
                )
                payload = cfg.to_dict()
        """
        d = asdict(self)
        # serialize Paths
        for k in ("home_dir", "dataset_name", "singularity_image", "results_dir"):
            d[k] = str(d[k])
        if d.get("qc_report_dir") is not None:
            d["qc_report_dir"] = str(d["qc_report_dir"])
        if d.get("qc_cross_pulsar_dir") is not None:
            d["qc_cross_pulsar_dir"] = str(d["qc_cross_pulsar_dir"])
        if d.get("fix_qc_results_dir") is not None:
            d["fix_qc_results_dir"] = str(d["fix_qc_results_dir"])
        if d.get("fix_system_flag_mapping_path") is not None:
            d["fix_system_flag_mapping_path"] = str(d["fix_system_flag_mapping_path"])
        if d.get("fix_flag_sys_freq_rules_path") is not None:
            d["fix_flag_sys_freq_rules_path"] = str(d["fix_flag_sys_freq_rules_path"])
        if d.get("fix_system_flag_table_path") is not None:
            d["fix_system_flag_table_path"] = str(d["fix_system_flag_table_path"])
        if d.get("fix_backend_classifications_path") is not None:
            d["fix_backend_classifications_path"] = str(
                d["fix_backend_classifications_path"]
            )
        if d.get("fix_alltim_variants_path") is not None:
            d["fix_alltim_variants_path"] = str(d["fix_alltim_variants_path"])
        if d.get("fix_relabel_rules_path") is not None:
            d["fix_relabel_rules_path"] = str(d["fix_relabel_rules_path"])
        if d.get("fix_overlap_rules_path") is not None:
            d["fix_overlap_rules_path"] = str(d["fix_overlap_rules_path"])
        if d.get("fix_overlap_exact_catalog_path") is not None:
            d["fix_overlap_exact_catalog_path"] = str(
                d["fix_overlap_exact_catalog_path"]
            )
        if d.get("pqc_backend_profiles_path") is not None:
            d["pqc_backend_profiles_path"] = str(d["pqc_backend_profiles_path"])
        if d.get("whitenoise_source_path") is not None:
            d["whitenoise_source_path"] = str(d["whitenoise_source_path"])
        if d.get("ingest_mapping_file") is not None:
            d["ingest_mapping_file"] = str(d["ingest_mapping_file"])
        if d.get("ingest_output_dir") is not None:
            d["ingest_output_dir"] = str(d["ingest_output_dir"])
        if d.get("compare_public_out_dir") is not None:
            d["compare_public_out_dir"] = str(d["compare_public_out_dir"])
        if d.get("compare_public_providers_path") is not None:
            d["compare_public_providers_path"] = str(d["compare_public_providers_path"])
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PipelineConfig":
        """Construct a :class:`PipelineConfig` from a dict.

        Args:
            d: Dictionary of configuration values.

        Returns:
            A new :class:`PipelineConfig` instance.

        Raises:
            KeyError: If required keys (``home_dir`` or ``singularity_image``)
                are missing.

        Examples:
            Load from a dict (e.g., parsed JSON/TOML)::

                cfg = PipelineConfig.from_dict(
                    {
                        "home_dir": "/data/epta",
                        "singularity_image": "/images/tempo2.sif",
                        "dataset_name": "EPTA",
                    }
                )
        """

        def p(x: Any) -> Path:
            return Path(x) if x is not None else Path(".")

        def opt_str(key: str) -> Optional[str]:
            v = d.get(key)
            if v in (None, ""):
                return None
            return str(v)

        def opt_list_str(key: str) -> Optional[List[str]]:
            v = d.get(key)
            if v in (None, ""):
                return None
            if isinstance(v, str):
                return [s.strip() for s in v.split(",") if s.strip()]
            return [str(x) for x in list(v)]

        def list_default(key: str, default: List[str]) -> List[str]:
            v = d.get(key)
            if v in (None, ""):
                return list(default)
            return list(v)

        return PipelineConfig(
            home_dir=p(d["home_dir"]),
            dataset_name=p(d.get("dataset_name", ".")),
            singularity_image=p(d["singularity_image"]),
            results_dir=p(d.get("results_dir", ".")),
            branches=list(d.get("branches", ["main", ""])),
            reference_branch=str(d.get("reference_branch", "main")),
            pulsars=d.get("pulsars", "ALL"),
            outdir_name=(
                None if d.get("outdir_name") in (None, "") else d.get("outdir_name")
            ),
            epoch=str(d.get("epoch", "55000")),
            force_rerun=bool(d.get("force_rerun", False)),
            jobs=int(d.get("jobs", 1)),
            run_tempo2=bool(d.get("run_tempo2", True)),
            make_toa_coverage_plots=bool(d.get("make_toa_coverage_plots", True)),
            make_change_reports=bool(d.get("make_change_reports", True)),
            make_covariance_heatmaps=bool(d.get("make_covariance_heatmaps", True)),
            make_residual_plots=bool(d.get("make_residual_plots", True)),
            make_outlier_reports=bool(d.get("make_outlier_reports", True)),
            make_plots=(d.get("make_plots") if "make_plots" in d else None),
            make_reports=(d.get("make_reports") if "make_reports" in d else None),
            make_covmat=(d.get("make_covmat") if "make_covmat" in d else None),
            testing_mode=bool(d.get("testing_mode", False)),
            run_pqc=bool(d.get("run_pqc", False)),
            run_whitenoise=bool(d.get("run_whitenoise", False)),
            whitenoise_source_path=opt_str("whitenoise_source_path"),
            whitenoise_epoch_tolerance_seconds=float(
                d.get("whitenoise_epoch_tolerance_seconds", 1.0)
            ),
            whitenoise_single_toa_mode=str(
                d.get("whitenoise_single_toa_mode", "combined")
            ),
            whitenoise_fit_timing_model_first=bool(
                d.get("whitenoise_fit_timing_model_first", True)
            ),
            whitenoise_timfile_name=opt_str("whitenoise_timfile_name"),
            pqc_backend_col=str(d.get("pqc_backend_col", "group")),
            pqc_drop_unmatched=bool(d.get("pqc_drop_unmatched", False)),
            pqc_merge_tol_seconds=float(d.get("pqc_merge_tol_seconds", 2.0)),
            pqc_tau_corr_minutes=float(d.get("pqc_tau_corr_minutes", 30.0)),
            pqc_fdr_q=float(d.get("pqc_fdr_q", 0.01)),
            pqc_mark_only_worst_per_day=bool(
                d.get("pqc_mark_only_worst_per_day", True)
            ),
            pqc_tau_rec_days=float(d.get("pqc_tau_rec_days", 7.0)),
            pqc_window_mult=float(d.get("pqc_window_mult", 5.0)),
            pqc_min_points=int(d.get("pqc_min_points", 6)),
            pqc_delta_chi2_thresh=float(d.get("pqc_delta_chi2_thresh", 25.0)),
            pqc_exp_dip_min_duration_days=float(
                d.get("pqc_exp_dip_min_duration_days", 21.0)
            ),
            pqc_step_enabled=bool(d.get("pqc_step_enabled", True)),
            pqc_step_min_points=int(d.get("pqc_step_min_points", 20)),
            pqc_step_delta_chi2_thresh=float(d.get("pqc_step_delta_chi2_thresh", 25.0)),
            pqc_step_scope=str(d.get("pqc_step_scope", "both")),
            pqc_dm_step_enabled=bool(d.get("pqc_dm_step_enabled", True)),
            pqc_dm_step_min_points=int(d.get("pqc_dm_step_min_points", 20)),
            pqc_dm_step_delta_chi2_thresh=float(
                d.get("pqc_dm_step_delta_chi2_thresh", 25.0)
            ),
            pqc_dm_step_scope=str(d.get("pqc_dm_step_scope", "both")),
            pqc_add_orbital_phase=bool(d.get("pqc_add_orbital_phase", True)),
            pqc_add_solar_elongation=bool(d.get("pqc_add_solar_elongation", True)),
            pqc_add_elevation=bool(d.get("pqc_add_elevation", False)),
            pqc_add_airmass=bool(d.get("pqc_add_airmass", False)),
            pqc_add_parallactic_angle=bool(d.get("pqc_add_parallactic_angle", False)),
            pqc_add_freq_bin=bool(d.get("pqc_add_freq_bin", False)),
            pqc_freq_bins=int(d.get("pqc_freq_bins", 8)),
            pqc_observatory_path=opt_str("pqc_observatory_path"),
            pqc_structure_mode=str(d.get("pqc_structure_mode", "none")),
            pqc_structure_detrend_features=list_default(
                "pqc_structure_detrend_features",
                ["solar_elongation_deg", "orbital_phase"],
            ),
            pqc_structure_test_features=list_default(
                "pqc_structure_test_features", ["solar_elongation_deg", "orbital_phase"]
            ),
            pqc_structure_nbins=int(d.get("pqc_structure_nbins", 12)),
            pqc_structure_min_per_bin=int(d.get("pqc_structure_min_per_bin", 3)),
            pqc_structure_p_thresh=float(d.get("pqc_structure_p_thresh", 0.01)),
            pqc_structure_circular_features=list_default(
                "pqc_structure_circular_features", ["orbital_phase"]
            ),
            pqc_structure_group_cols=opt_list_str("pqc_structure_group_cols"),
            pqc_outlier_gate_enabled=bool(d.get("pqc_outlier_gate_enabled", False)),
            pqc_outlier_gate_sigma=float(d.get("pqc_outlier_gate_sigma", 3.0)),
            pqc_outlier_gate_resid_col=opt_str("pqc_outlier_gate_resid_col"),
            pqc_outlier_gate_sigma_col=opt_str("pqc_outlier_gate_sigma_col"),
            pqc_event_instrument=bool(d.get("pqc_event_instrument", False)),
            pqc_solar_events_enabled=bool(d.get("pqc_solar_events_enabled", False)),
            pqc_solar_approach_max_deg=float(d.get("pqc_solar_approach_max_deg", 30.0)),
            pqc_solar_min_points_global=int(d.get("pqc_solar_min_points_global", 30)),
            pqc_solar_min_points_year=int(d.get("pqc_solar_min_points_year", 10)),
            pqc_solar_min_points_near_zero=int(
                d.get("pqc_solar_min_points_near_zero", 3)
            ),
            pqc_solar_tau_min_deg=float(d.get("pqc_solar_tau_min_deg", 2.0)),
            pqc_solar_tau_max_deg=float(d.get("pqc_solar_tau_max_deg", 60.0)),
            pqc_solar_member_eta=float(d.get("pqc_solar_member_eta", 1.0)),
            pqc_solar_freq_dependence=bool(d.get("pqc_solar_freq_dependence", True)),
            pqc_solar_freq_alpha_min=float(d.get("pqc_solar_freq_alpha_min", 0.0)),
            pqc_solar_freq_alpha_max=float(d.get("pqc_solar_freq_alpha_max", 4.0)),
            pqc_solar_freq_alpha_tol=float(d.get("pqc_solar_freq_alpha_tol", 1e-3)),
            pqc_solar_freq_alpha_max_iter=int(
                d.get("pqc_solar_freq_alpha_max_iter", 64)
            ),
            pqc_orbital_phase_cut_enabled=bool(
                d.get("pqc_orbital_phase_cut_enabled", False)
            ),
            pqc_orbital_phase_cut_center=float(
                d.get("pqc_orbital_phase_cut_center", 0.25)
            ),
            pqc_orbital_phase_cut=(
                None
                if d.get("pqc_orbital_phase_cut") in (None, "")
                else float(d.get("pqc_orbital_phase_cut"))
            ),
            pqc_orbital_phase_cut_sigma=float(
                d.get("pqc_orbital_phase_cut_sigma", 3.0)
            ),
            pqc_orbital_phase_cut_nbins=int(d.get("pqc_orbital_phase_cut_nbins", 18)),
            pqc_orbital_phase_cut_min_points=int(
                d.get("pqc_orbital_phase_cut_min_points", 20)
            ),
            pqc_eclipse_events_enabled=bool(d.get("pqc_eclipse_events_enabled", False)),
            pqc_eclipse_center_phase=float(d.get("pqc_eclipse_center_phase", 0.25)),
            pqc_eclipse_min_points=int(d.get("pqc_eclipse_min_points", 30)),
            pqc_eclipse_width_min=float(d.get("pqc_eclipse_width_min", 0.01)),
            pqc_eclipse_width_max=float(d.get("pqc_eclipse_width_max", 0.5)),
            pqc_eclipse_member_eta=float(d.get("pqc_eclipse_member_eta", 1.0)),
            pqc_eclipse_freq_dependence=bool(
                d.get("pqc_eclipse_freq_dependence", True)
            ),
            pqc_eclipse_freq_alpha_min=float(d.get("pqc_eclipse_freq_alpha_min", 0.0)),
            pqc_eclipse_freq_alpha_max=float(d.get("pqc_eclipse_freq_alpha_max", 4.0)),
            pqc_eclipse_freq_alpha_tol=float(d.get("pqc_eclipse_freq_alpha_tol", 1e-3)),
            pqc_eclipse_freq_alpha_max_iter=int(
                d.get("pqc_eclipse_freq_alpha_max_iter", 64)
            ),
            pqc_gaussian_bump_enabled=bool(d.get("pqc_gaussian_bump_enabled", False)),
            pqc_gaussian_bump_min_duration_days=float(
                d.get("pqc_gaussian_bump_min_duration_days", 60.0)
            ),
            pqc_gaussian_bump_max_duration_days=float(
                d.get("pqc_gaussian_bump_max_duration_days", 1500.0)
            ),
            pqc_gaussian_bump_n_durations=int(
                d.get("pqc_gaussian_bump_n_durations", 6)
            ),
            pqc_gaussian_bump_min_points=int(d.get("pqc_gaussian_bump_min_points", 20)),
            pqc_gaussian_bump_delta_chi2_thresh=float(
                d.get("pqc_gaussian_bump_delta_chi2_thresh", 25.0)
            ),
            pqc_gaussian_bump_suppress_overlap=bool(
                d.get("pqc_gaussian_bump_suppress_overlap", True)
            ),
            pqc_gaussian_bump_member_eta=float(
                d.get("pqc_gaussian_bump_member_eta", 1.0)
            ),
            pqc_gaussian_bump_freq_dependence=bool(
                d.get("pqc_gaussian_bump_freq_dependence", True)
            ),
            pqc_gaussian_bump_freq_alpha_min=float(
                d.get("pqc_gaussian_bump_freq_alpha_min", 0.0)
            ),
            pqc_gaussian_bump_freq_alpha_max=float(
                d.get("pqc_gaussian_bump_freq_alpha_max", 4.0)
            ),
            pqc_gaussian_bump_freq_alpha_tol=float(
                d.get("pqc_gaussian_bump_freq_alpha_tol", 1e-3)
            ),
            pqc_gaussian_bump_freq_alpha_max_iter=int(
                d.get("pqc_gaussian_bump_freq_alpha_max_iter", 64)
            ),
            pqc_glitch_enabled=bool(d.get("pqc_glitch_enabled", False)),
            pqc_glitch_min_points=int(d.get("pqc_glitch_min_points", 30)),
            pqc_glitch_delta_chi2_thresh=float(
                d.get("pqc_glitch_delta_chi2_thresh", 25.0)
            ),
            pqc_glitch_suppress_overlap=bool(
                d.get("pqc_glitch_suppress_overlap", True)
            ),
            pqc_glitch_member_eta=float(d.get("pqc_glitch_member_eta", 1.0)),
            pqc_glitch_peak_tau_days=float(d.get("pqc_glitch_peak_tau_days", 30.0)),
            pqc_glitch_noise_k=float(d.get("pqc_glitch_noise_k", 1.0)),
            pqc_glitch_mean_window_days=float(
                d.get("pqc_glitch_mean_window_days", 180.0)
            ),
            pqc_glitch_min_duration_days=float(
                d.get("pqc_glitch_min_duration_days", 1000.0)
            ),
            pqc_backend_profiles_path=opt_str("pqc_backend_profiles_path"),
            pqc_run_variants=bool(d.get("pqc_run_variants", False)),
            pqc_keep_variant_tmp=bool(d.get("pqc_keep_variant_tmp", False)),
            qc_report=bool(d.get("qc_report", False)),
            qc_report_backend_col=opt_str("qc_report_backend_col"),
            qc_report_backend=opt_str("qc_report_backend"),
            qc_report_dir=(
                Path(d["qc_report_dir"]) if d.get("qc_report_dir") else None
            ),
            qc_report_no_plots=bool(d.get("qc_report_no_plots", False)),
            qc_report_structure_group_cols=opt_str("qc_report_structure_group_cols"),
            qc_report_no_feature_plots=bool(d.get("qc_report_no_feature_plots", False)),
            qc_report_compact_pdf=bool(d.get("qc_report_compact_pdf", False)),
            qc_report_compact_pdf_name=str(
                d.get("qc_report_compact_pdf_name", "qc_compact_report.pdf")
            ),
            qc_report_compact_outlier_cols=opt_list_str(
                "qc_report_compact_outlier_cols"
            ),
            qc_cross_pulsar_enabled=bool(d.get("qc_cross_pulsar_enabled", False)),
            qc_cross_pulsar_time_col=opt_str("qc_cross_pulsar_time_col"),
            qc_cross_pulsar_window_days=float(
                d.get("qc_cross_pulsar_window_days", 1.0)
            ),
            qc_cross_pulsar_min_pulsars=int(d.get("qc_cross_pulsar_min_pulsars", 2)),
            qc_cross_pulsar_include_outliers=bool(
                d.get("qc_cross_pulsar_include_outliers", True)
            ),
            qc_cross_pulsar_include_events=bool(
                d.get("qc_cross_pulsar_include_events", True)
            ),
            qc_cross_pulsar_outlier_cols=opt_list_str("qc_cross_pulsar_outlier_cols"),
            qc_cross_pulsar_event_cols=opt_list_str("qc_cross_pulsar_event_cols"),
            qc_cross_pulsar_dir=(
                Path(d["qc_cross_pulsar_dir"]) if d.get("qc_cross_pulsar_dir") else None
            ),
            run_fix_dataset=bool(d.get("run_fix_dataset", False)),
            make_binary_analysis=bool(d.get("make_binary_analysis", False)),
            param_scan_typical=bool(d.get("param_scan_typical", False)),
            param_scan_dm_redchisq_threshold=float(
                d.get("param_scan_dm_redchisq_threshold", 2.0)
            ),
            param_scan_dm_max_order=int(d.get("param_scan_dm_max_order", 4)),
            param_scan_btx_max_fb=int(d.get("param_scan_btx_max_fb", 3)),
            fix_apply=bool(d.get("fix_apply", False)),
            fix_branch_name=opt_str("fix_branch_name"),
            fix_base_branch=opt_str("fix_base_branch"),
            fix_commit_message=opt_str("fix_commit_message"),
            fix_backup=bool(d.get("fix_backup", True)),
            fix_dry_run=bool(d.get("fix_dry_run", False)),
            fix_update_alltim_includes=bool(d.get("fix_update_alltim_includes", True)),
            fix_min_toas_per_backend_tim=int(d.get("fix_min_toas_per_backend_tim", 10)),
            fix_required_tim_flags=dict(d.get("fix_required_tim_flags", {})),
            fix_system_flag_mapping_path=opt_str("fix_system_flag_mapping_path"),
            fix_flag_sys_freq_rules_enabled=bool(
                d.get("fix_flag_sys_freq_rules_enabled", False)
            ),
            fix_flag_sys_freq_rules_path=opt_str("fix_flag_sys_freq_rules_path"),
            fix_system_flag_table_path=opt_str("fix_system_flag_table_path"),
            fix_generate_alltim_variants=bool(
                d.get("fix_generate_alltim_variants", False)
            ),
            fix_backend_classifications_path=opt_str(
                "fix_backend_classifications_path"
            ),
            fix_alltim_variants_path=opt_str("fix_alltim_variants_path"),
            fix_relabel_rules_path=opt_str("fix_relabel_rules_path"),
            fix_overlap_rules_path=opt_str("fix_overlap_rules_path"),
            fix_overlap_exact_catalog_path=opt_str("fix_overlap_exact_catalog_path"),
            fix_jump_reference_variants=bool(
                d.get("fix_jump_reference_variants", False)
            ),
            fix_jump_reference_keep_tmp=bool(
                d.get("fix_jump_reference_keep_tmp", False)
            ),
            fix_jump_reference_jump_flag=str(
                d.get("fix_jump_reference_jump_flag", "-sys")
            ),
            fix_jump_reference_csv_dir=opt_str("fix_jump_reference_csv_dir"),
            fix_infer_system_flags=bool(d.get("fix_infer_system_flags", False)),
            fix_system_flag_overwrite_existing=bool(
                d.get("fix_system_flag_overwrite_existing", False)
            ),
            fix_wsrt_p2_force_sys_by_freq=bool(
                d.get("fix_wsrt_p2_force_sys_by_freq", False)
            ),
            fix_wsrt_p2_prefer_dual_channel=bool(
                d.get("fix_wsrt_p2_prefer_dual_channel", False)
            ),
            fix_wsrt_p2_mjd_tol_sec=float(d.get("fix_wsrt_p2_mjd_tol_sec", 0.99e-6)),
            fix_wsrt_p2_action=str(d.get("fix_wsrt_p2_action", "comment")),
            fix_wsrt_p2_comment_prefix=str(
                d.get("fix_wsrt_p2_comment_prefix", "C WSRT_P2_PREFER_DUAL")
            ),
            fix_backend_overrides=dict(d.get("fix_backend_overrides", {})),
            fix_raise_on_backend_missing=bool(
                d.get("fix_raise_on_backend_missing", False)
            ),
            fix_dedupe_toas_within_tim=bool(d.get("fix_dedupe_toas_within_tim", True)),
            fix_dedupe_mjd_tol_sec=float(d.get("fix_dedupe_mjd_tol_sec", 0.0)),
            fix_dedupe_freq_tol_mhz=(
                None
                if d.get("fix_dedupe_freq_tol_mhz") in (None, "")
                else float(d.get("fix_dedupe_freq_tol_mhz"))
            ),
            fix_dedupe_freq_tol_auto=bool(d.get("fix_dedupe_freq_tol_auto", False)),
            fix_check_duplicate_backend_tims=bool(
                d.get("fix_check_duplicate_backend_tims", False)
            ),
            fix_remove_overlaps_exact=bool(d.get("fix_remove_overlaps_exact", True)),
            fix_insert_missing_jumps=bool(d.get("fix_insert_missing_jumps", True)),
            fix_jump_flag=str(d.get("fix_jump_flag", "-sys")),
            fix_prune_stale_jumps=bool(d.get("fix_prune_stale_jumps", False)),
            fix_ensure_ephem=opt_str("fix_ensure_ephem"),
            fix_ensure_clk=opt_str("fix_ensure_clk"),
            fix_ensure_ne_sw=opt_str("fix_ensure_ne_sw"),
            fix_force_ne_sw_overwrite=bool(d.get("fix_force_ne_sw_overwrite", False)),
            fix_remove_patterns=list(
                d.get("fix_remove_patterns", ["NRT.NUPPI.", "NRT.NUXPI."])
            ),
            fix_coord_convert=opt_str("fix_coord_convert"),
            fix_prune_missing_includes=bool(d.get("fix_prune_missing_includes", True)),
            fix_drop_small_backend_includes=bool(
                d.get("fix_drop_small_backend_includes", True)
            ),
            fix_system_flag_update_table=bool(
                d.get("fix_system_flag_update_table", True)
            ),
            fix_default_backend=opt_str("fix_default_backend"),
            fix_group_flag=str(d.get("fix_group_flag", "-group")),
            fix_pta_flag=str(d.get("fix_pta_flag", "-pta")),
            fix_pta_value=opt_str("fix_pta_value"),
            fix_standardize_par_values=bool(d.get("fix_standardize_par_values", True)),
            fix_prune_small_system_toas=bool(
                d.get("fix_prune_small_system_toas", False)
            ),
            fix_prune_small_system_flag=str(
                d.get("fix_prune_small_system_flag", "-sys")
            ),
            fix_qc_remove_outliers=bool(d.get("fix_qc_remove_outliers", False)),
            fix_qc_outlier_cols=opt_list_str("fix_qc_outlier_cols"),
            fix_qc_action=str(d.get("fix_qc_action", "comment")),
            fix_qc_comment_prefix=str(d.get("fix_qc_comment_prefix", "C QC_OUTLIER")),
            fix_qc_backend_col=str(d.get("fix_qc_backend_col", "sys")),
            fix_qc_remove_bad=bool(d.get("fix_qc_remove_bad", True)),
            fix_qc_remove_transients=bool(d.get("fix_qc_remove_transients", False)),
            fix_qc_remove_solar=bool(d.get("fix_qc_remove_solar", False)),
            fix_qc_solar_action=str(d.get("fix_qc_solar_action", "comment")),
            fix_qc_solar_comment_prefix=str(
                d.get("fix_qc_solar_comment_prefix", "# QC_SOLAR")
            ),
            fix_qc_remove_orbital_phase=bool(
                d.get("fix_qc_remove_orbital_phase", False)
            ),
            fix_qc_orbital_phase_action=str(
                d.get("fix_qc_orbital_phase_action", "comment")
            ),
            fix_qc_orbital_phase_comment_prefix=str(
                d.get("fix_qc_orbital_phase_comment_prefix", "# QC_BIANRY_ECLIPSE")
            ),
            fix_qc_merge_tol_days=float(d.get("fix_qc_merge_tol_days", 2.0 / 86400.0)),
            fix_qc_results_dir=(
                Path(d["fix_qc_results_dir"]) if d.get("fix_qc_results_dir") else None
            ),
            fix_qc_branch=opt_str("fix_qc_branch"),
            binary_only_models=opt_list_str("binary_only_models"),
            dpi=int(d.get("dpi", 120)),
            max_covmat_params=(
                None
                if d.get("max_covmat_params") in (None, "")
                else d.get("max_covmat_params")
            ),
            ingest_mapping_file=(
                Path(d["ingest_mapping_file"]) if d.get("ingest_mapping_file") else None
            ),
            ingest_output_dir=(
                Path(d["ingest_output_dir"]) if d.get("ingest_output_dir") else None
            ),
            ingest_commit_branch=bool(d.get("ingest_commit_branch", False)),
            ingest_commit_branch_name=opt_str("ingest_commit_branch_name"),
            ingest_commit_base_branch=opt_str("ingest_commit_base_branch"),
            ingest_commit_message=opt_str("ingest_commit_message"),
            ingest_verify=bool(d.get("ingest_verify", False)),
            compare_public_out_dir=(
                Path(d["compare_public_out_dir"])
                if d.get("compare_public_out_dir")
                else None
            ),
            compare_public_providers_path=(
                Path(d["compare_public_providers_path"])
                if d.get("compare_public_providers_path")
                else None
            ),
        )

    @staticmethod
    def load(path: Path) -> "PipelineConfig":
        """Load configuration from a JSON or TOML file.

        Args:
            path: Path to a ``.json`` or ``.toml`` file.

        Returns:
            A :class:`PipelineConfig` instance.

        Raises:
            FileNotFoundError: If the path does not exist.
            RuntimeError: If TOML is requested but ``tomllib`` is unavailable.
            ValueError: If the file extension is unsupported.

        Examples:
            Load from JSON::

                cfg = PipelineConfig.load(Path("pipeline.json"))

            Load from TOML (``[pipeline]`` table supported)::

                cfg = PipelineConfig.load(Path("pipeline.toml"))
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return PipelineConfig.from_dict(data)

        if path.suffix.lower() in (".toml", ".tml"):
            if tomllib is None:
                raise RuntimeError(
                    "TOML config requested but tomllib is unavailable in this Python."
                )
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            # Accept either top-level keys or [pipeline] table
            if "pipeline" in data and isinstance(data["pipeline"], dict):
                data = data["pipeline"]
            return PipelineConfig.from_dict(data)

        raise ValueError(
            f"Unsupported config file type: {path.suffix}. Use .json or .toml"
        )

    def save_json(self, path: Path) -> None:
        """Write configuration to a JSON file.

        Args:
            path: Output file path.

        Examples:
            Save to disk::

                cfg.save_json(Path("pipeline.json"))
        """
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
