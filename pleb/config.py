"""Configuration models for the data-combination pipeline.

This module defines the :class:`PipelineConfig` dataclass and helpers for
loading/saving configuration from JSON and TOML files.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore


PulsarSelection = Union[str, List[str]]  # "ALL" or explicit list


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for the data-combination pipeline.

    The configuration is intentionally flat to make it easy to serialize to
    JSON/TOML and to override via CLI flags.

    Attributes:
        home_dir: Root of the data repository containing pulsar folders.
        singularity_image: Singularity/Apptainer image with tempo2.
        dataset_name: Dataset name or path (see :meth:`resolved`).
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
        testing_mode: If True, skip change reports (useful for CI).
        run_pta_qc: Enable optional pta_qc stage.
        pta_qc_backend_col: Backend grouping column for pta_qc.
        pta_qc_drop_unmatched: Drop unmatched TOAs in pta_qc.
        pta_qc_merge_tol_seconds: Merge tolerance in seconds for pta_qc.
        pta_qc_tau_corr_minutes: OU correlation time for pta_qc.
        pta_qc_fdr_q: False discovery rate for pta_qc.
        pta_qc_mark_only_worst_per_day: Mark only worst TOA per day.
        pta_qc_tau_rec_days: Recovery time for transient scan.
        pta_qc_window_mult: Window multiplier for transient scan.
        pta_qc_min_points: Minimum points for transient scan.
        pta_qc_delta_chi2_thresh: Delta-chi2 threshold for transients.
        run_fix_dataset: Enable FixDataset stage.
        make_binary_analysis: Enable binary analysis table.
        param_scan_typical: Enable typical param-scan profile.
        param_scan_dm_redchisq_threshold: Threshold for DM scan.
        param_scan_dm_max_order: Max DM derivative order.
        param_scan_btx_max_fb: Max FB derivative order.
        fix_apply: Whether FixDataset applies changes and commits.
        fix_branch_name: Name of FixDataset branch.
        fix_base_branch: Base branch for FixDataset.
        fix_commit_message: Commit message for FixDataset.
        fix_backup: Create backup before FixDataset modifications.
        fix_dry_run: If True, FixDataset does not write changes.
        fix_update_alltim_includes: Update INCLUDE lines in .tim files.
        fix_min_toas_per_backend_tim: Minimum TOAs per backend .tim.
        fix_required_tim_flags: Required flags for .tim entries.
        fix_insert_missing_jumps: Insert missing JUMP lines.
        fix_jump_flag: Flag used for inserted jumps.
        fix_ensure_ephem: Ensure ephemeris parameter exists.
        fix_ensure_clk: Ensure clock parameter exists.
        fix_ensure_ne_sw: Ensure NE_SW parameter exists.
        fix_remove_patterns: Patterns to remove from .par/.tim.
        fix_coord_convert: Optional coordinate conversion.
        binary_only_models: Limit binary analysis to model names.
        dpi: Plot resolution.
        max_covmat_params: Max params in covariance heatmaps.
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
    testing_mode: bool = False

    # Optional outlier/QC stage using the external `pta_qc` package (libstempo-based).
    # This is separate from the existing general2-based outlier summaries.
    run_pta_qc: bool = False
    pta_qc_backend_col: str = "group"
    pta_qc_drop_unmatched: bool = False
    pta_qc_merge_tol_seconds: float = 2.0
    pta_qc_tau_corr_minutes: float = 30.0
    pta_qc_fdr_q: float = 0.01
    pta_qc_mark_only_worst_per_day: bool = True
    pta_qc_tau_rec_days: float = 7.0
    pta_qc_window_mult: float = 5.0
    pta_qc_min_points: int = 6
    pta_qc_delta_chi2_thresh: float = 25.0

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

    fix_insert_missing_jumps: bool = True
    fix_jump_flag: str = "-sys"
    fix_ensure_ephem: Optional[str] = None
    fix_ensure_clk: Optional[str] = None
    fix_ensure_ne_sw: Optional[str] = None
    fix_remove_patterns: List[str] = field(default_factory=lambda: ["NRT.NUPPI.", "NRT.NUXPI."])
    # "equatorial_to_ecliptic" or "ecliptic_to_equatorial"
    fix_coord_convert: Optional[str] = None

    # ---- Binary analysis settings ----
    binary_only_models: Optional[List[str]] = None

    # Plotting controls
    dpi: int = 120
    max_covmat_params: Optional[int] = None

    def resolved(self) -> "PipelineConfig":
        """Return a copy with paths expanded/resolved.

        The ``dataset_name`` field is interpreted as:

        1) absolute path -> use as-is
        2) looks like a path (contains "/" or starts with ".") -> resolve relative to cwd
        3) plain name -> treat as a directory inside ``home_dir``

        Returns:
            A new :class:`PipelineConfig` with resolved paths.
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
        c.home_dir = c.home_dir.expanduser().resolve()
        ds_raw = str(c.dataset_name)
        ds_path = Path(ds_raw).expanduser()
    
        # Interpret dataset_name as:
        # 1) absolute path -> use as-is
        # 2) looks like a path (contains / or starts with .) -> resolve relative to config/cwd (legacy behavior)
        # 3) plain name -> treat as a directory inside home_dir (what you expect)
        if ds_path.is_absolute():
            dataset_dir = ds_path
        elif ("/" in ds_raw) or ("\\" in ds_raw) or ds_raw.startswith("."):
            dataset_dir = ds_path.resolve()
        else:
            dataset_dir = (Path(c.home_dir).expanduser() / ds_raw).resolve()
 

        c.singularity_image = c.singularity_image.expanduser().resolve()
        c.results_dir = c.results_dir.expanduser().resolve()
        return c

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a JSON-friendly dict.

        Returns:
            Dictionary representation of the config with :class:`Path` values
            converted to strings.
        """
        d = asdict(self)
        # serialize Paths
        for k in ("home_dir", "dataset_name", "singularity_image", "results_dir"):
            d[k] = str(d[k])
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PipelineConfig":
        """Construct a :class:`PipelineConfig` from a dict.

        Args:
            d: Dictionary of configuration values.

        Returns:
            A new :class:`PipelineConfig` instance.
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
            return list(v)

        return PipelineConfig(
            home_dir=p(d["home_dir"]),
            dataset_name=p(d.get("dataset_name", ".")),
            singularity_image=p(d["singularity_image"]),
            results_dir=p(d.get("results_dir", ".")),
            branches=list(d.get("branches", ["main", ""])),
            reference_branch=str(d.get("reference_branch", "main")),
            pulsars=d.get("pulsars", "ALL"),
            outdir_name=(None if d.get("outdir_name") in (None, "") else d.get("outdir_name")),
            epoch=str(d.get("epoch", "55000")),
            force_rerun=bool(d.get("force_rerun", False)),
            jobs=int(d.get("jobs", 1)),
            run_tempo2=bool(d.get("run_tempo2", True)),
            make_toa_coverage_plots=bool(d.get("make_toa_coverage_plots", True)),
            make_change_reports=bool(d.get("make_change_reports", True)),
            make_covariance_heatmaps=bool(d.get("make_covariance_heatmaps", True)),
            make_residual_plots=bool(d.get("make_residual_plots", True)),
            make_outlier_reports=bool(d.get("make_outlier_reports", True)),
            testing_mode=bool(d.get("testing_mode", False)),

            run_pta_qc=bool(d.get("run_pta_qc", False)),
            pta_qc_backend_col=str(d.get("pta_qc_backend_col", "group")),
            pta_qc_drop_unmatched=bool(d.get("pta_qc_drop_unmatched", False)),
            pta_qc_merge_tol_seconds=float(d.get("pta_qc_merge_tol_seconds", 2.0)),
            pta_qc_tau_corr_minutes=float(d.get("pta_qc_tau_corr_minutes", 30.0)),
            pta_qc_fdr_q=float(d.get("pta_qc_fdr_q", 0.01)),
            pta_qc_mark_only_worst_per_day=bool(d.get("pta_qc_mark_only_worst_per_day", True)),
            pta_qc_tau_rec_days=float(d.get("pta_qc_tau_rec_days", 7.0)),
            pta_qc_window_mult=float(d.get("pta_qc_window_mult", 5.0)),
            pta_qc_min_points=int(d.get("pta_qc_min_points", 6)),
            pta_qc_delta_chi2_thresh=float(d.get("pta_qc_delta_chi2_thresh", 25.0)),

            run_fix_dataset=bool(d.get("run_fix_dataset", False)),
            make_binary_analysis=bool(d.get("make_binary_analysis", False)),

            param_scan_typical=bool(d.get("param_scan_typical", False)),
            param_scan_dm_redchisq_threshold=float(d.get("param_scan_dm_redchisq_threshold", 2.0)),
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
            fix_insert_missing_jumps=bool(d.get("fix_insert_missing_jumps", True)),
            fix_jump_flag=str(d.get("fix_jump_flag", "-sys")),
            fix_ensure_ephem=opt_str("fix_ensure_ephem"),
            fix_ensure_clk=opt_str("fix_ensure_clk"),
            fix_ensure_ne_sw=opt_str("fix_ensure_ne_sw"),
            fix_remove_patterns=list(d.get("fix_remove_patterns", ["NRT.NUPPI.", "NRT.NUXPI."])),
            fix_coord_convert=opt_str("fix_coord_convert"),

            binary_only_models=opt_list_str("binary_only_models"),

            dpi=int(d.get("dpi", 120)),
            max_covmat_params=(None if d.get("max_covmat_params") in (None, "") else d.get("max_covmat_params")),
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
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return PipelineConfig.from_dict(data)

        if path.suffix.lower() in (".toml", ".tml"):
            if tomllib is None:
                raise RuntimeError("TOML config requested but tomllib is unavailable in this Python.")
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            # Accept either top-level keys or [pipeline] table
            if "pipeline" in data and isinstance(data["pipeline"], dict):
                data = data["pipeline"]
            return PipelineConfig.from_dict(data)

        raise ValueError(f"Unsupported config file type: {path.suffix}. Use .json or .toml")

    def save_json(self, path: Path) -> None:
        """Write configuration to a JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
