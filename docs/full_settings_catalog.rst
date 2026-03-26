Full Settings Catalog
=====================

This chapter documents all ``PipelineConfig`` keys with practical TOML and CLI examples.
Use it as an operator handbook when tuning large configuration files.

Example conventions
-------------------

- TOML examples are single-key snippets for clarity.
- CLI examples use ``--set key=value``.
- Paths are illustrative; adapt to your environment.

Core pipeline
-------------

``home_dir``
  Type: ``Path``
  Default: ``<required>``
  Meaning: Root of the data repository containing pulsar folders.
  Example TOML::

     home_dir = "/data/epta"

  Example CLI::

     pleb --config pipeline.toml --set home_dir="/data/epta"

``singularity_image``
  Type: ``Path``
  Default: ``<required>``
  Meaning: Singularity/Apptainer image with tempo2.
  Example TOML::

     singularity_image = "/images/tempo2.sif"

  Example CLI::

     pleb --config pipeline.toml --set singularity_image="/images/tempo2.sif"

``dataset_name``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Dataset name or path (see :meth:`resolved`).
  Example TOML::

     dataset_name = "DR3full"

  Example CLI::

     pleb --config pipeline.toml --set dataset_name="DR3full"

``results_dir``
  Type: ``Path``
  Default: ``Path('.')``
  Meaning: Output directory for pipeline reports.
  Example TOML::

     results_dir = "results"

  Example CLI::

     pleb --config pipeline.toml --set results_dir="results"

``branches``
  Type: ``List[str]``
  Default: ``['main', 'EPTA']``
  Meaning: Git branches to run/compare.
  Example TOML::

     branches = ["main","raw_ingest"]

  Example CLI::

     pleb --config pipeline.toml --set branches=["main","raw_ingest"]

``reference_branch``
  Type: ``str``
  Default: ``'main'``
  Meaning: Branch used as change-report reference.
  Example TOML::

     reference_branch = "main"

  Example CLI::

     pleb --config pipeline.toml --set reference_branch="main"

``pulsars``
  Type: ``PulsarSelection``
  Default: ``'ALL'``
  Meaning: "ALL" or an explicit list of pulsar names.
  Example TOML::

     pulsars = ["J1713+0747","J1022+1001"]

  Example CLI::

     pleb --config pipeline.toml --set pulsars=["J1713+0747","J1022+1001"]

``outdir_name``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Optional output subdirectory name.
  Example TOML::

     outdir_name = "run_pqc_balanced"

  Example CLI::

     pleb --config pipeline.toml --set outdir_name="run_pqc_balanced"

``cleanup_output_tree``
  Type: ``bool``
  Default: ``True``
  Meaning: Pipeline setting.
  Example TOML::

     cleanup_output_tree = false

  Example CLI::

     pleb --config pipeline.toml --set cleanup_output_tree=false

``cleanup_work_dir``
  Type: ``bool``
  Default: ``False``
  Meaning: Pipeline setting.
  Example TOML::

     cleanup_work_dir = true

  Example CLI::

     pleb --config pipeline.toml --set cleanup_work_dir=true

``epoch``
  Type: ``str``
  Default: ``'55000'``
  Meaning: Tempo2 epoch used for fitting.
  Example TOML::

     epoch = "55000"

  Example CLI::

     pleb --config pipeline.toml --set epoch="55000"

``force_rerun``
  Type: ``bool``
  Default: ``False``
  Meaning: Re-run tempo2 even if outputs exist.
  Example TOML::

     force_rerun = true

  Example CLI::

     pleb --config pipeline.toml --set force_rerun=true

``jobs``
  Type: ``int``
  Default: ``1``
  Meaning: Parallel workers per branch.
  Example TOML::

     jobs = 8

  Example CLI::

     pleb --config pipeline.toml --set jobs=8

``run_tempo2``
  Type: ``bool``
  Default: ``True``
  Meaning: Whether to run tempo2.
  Example TOML::

     run_tempo2 = false

  Example CLI::

     pleb --config pipeline.toml --set run_tempo2=false

``make_toa_coverage_plots``
  Type: ``bool``
  Default: ``True``
  Meaning: Generate coverage plots.
  Example TOML::

     make_toa_coverage_plots = false

  Example CLI::

     pleb --config pipeline.toml --set make_toa_coverage_plots=false

``make_change_reports``
  Type: ``bool``
  Default: ``True``
  Meaning: Generate change reports.
  Example TOML::

     make_change_reports = false

  Example CLI::

     pleb --config pipeline.toml --set make_change_reports=false

``make_covariance_heatmaps``
  Type: ``bool``
  Default: ``True``
  Meaning: Generate covariance heatmaps.
  Example TOML::

     make_covariance_heatmaps = false

  Example CLI::

     pleb --config pipeline.toml --set make_covariance_heatmaps=false

``make_residual_plots``
  Type: ``bool``
  Default: ``True``
  Meaning: Generate residual plots.
  Example TOML::

     make_residual_plots = false

  Example CLI::

     pleb --config pipeline.toml --set make_residual_plots=false

``make_outlier_reports``
  Type: ``bool``
  Default: ``True``
  Meaning: Generate outlier tables.
  Example TOML::

     make_outlier_reports = false

  Example CLI::

     pleb --config pipeline.toml --set make_outlier_reports=false

``make_plots``
  Type: ``Optional[bool]``
  Default: ``None``
  Meaning: Convenience toggle to disable all plotting outputs.
  Example TOML::

     make_plots = true

  Example CLI::

     pleb --config pipeline.toml --set make_plots=true

``make_reports``
  Type: ``Optional[bool]``
  Default: ``None``
  Meaning: Convenience toggle to disable report outputs.
  Example TOML::

     make_reports = true

  Example CLI::

     pleb --config pipeline.toml --set make_reports=true

``make_covmat``
  Type: ``Optional[bool]``
  Default: ``None``
  Meaning: Convenience toggle to control covariance heatmaps.
  Example TOML::

     make_covmat = true

  Example CLI::

     pleb --config pipeline.toml --set make_covmat=true

``testing_mode``
  Type: ``bool``
  Default: ``False``
  Meaning: If True, skip change reports (useful for CI).
  Example TOML::

     testing_mode = true

  Example CLI::

     pleb --config pipeline.toml --set testing_mode=true

``run_pqc``
  Type: ``bool``
  Default: ``False``
  Meaning: Enable optional pqc stage.
  Example TOML::

     run_pqc = true

  Example CLI::

     pleb --config pipeline.toml --set run_pqc=true

``qc_report``
  Type: ``bool``
  Default: ``False``
  Meaning: Generate pqc report artifacts after the run.
  Example TOML::

     qc_report = true

  Example CLI::

     pleb --config pipeline.toml --set qc_report=true

``run_fix_dataset``
  Type: ``bool``
  Default: ``False``
  Meaning: Enable FixDataset stage.
  Example TOML::

     run_fix_dataset = true

  Example CLI::

     pleb --config pipeline.toml --set run_fix_dataset=true

``make_binary_analysis``
  Type: ``bool``
  Default: ``False``
  Meaning: Enable binary analysis table.
  Example TOML::

     make_binary_analysis = true

  Example CLI::

     pleb --config pipeline.toml --set make_binary_analysis=true

PQC detector settings
---------------------

``pqc_backend_col``
  Type: ``str``
  Default: ``'group'``
  Meaning: Backend grouping column for pqc.
  Example TOML::

     pqc_backend_col = "sys"

  Example CLI::

     pleb --config pipeline.toml --set pqc_backend_col="sys"

``pqc_drop_unmatched``
  Type: ``bool``
  Default: ``False``
  Meaning: Drop unmatched TOAs in pqc.
  Example TOML::

     pqc_drop_unmatched = true

  Example CLI::

     pleb --config pipeline.toml --set pqc_drop_unmatched=true

``pqc_merge_tol_seconds``
  Type: ``float``
  Default: ``2.0``
  Meaning: Merge tolerance in seconds for pqc.
  Example TOML::

     pqc_merge_tol_seconds = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_merge_tol_seconds=1.0

``pqc_tau_corr_minutes``
  Type: ``float``
  Default: ``30.0``
  Meaning: OU correlation time for pqc.
  Example TOML::

     pqc_tau_corr_minutes = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_tau_corr_minutes=1.0

``pqc_fdr_q``
  Type: ``float``
  Default: ``0.01``
  Meaning: False discovery rate for pqc.
  Example TOML::

     pqc_fdr_q = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_fdr_q=1.0

``pqc_mark_only_worst_per_day``
  Type: ``bool``
  Default: ``True``
  Meaning: Mark only worst TOA per day.
  Example TOML::

     pqc_mark_only_worst_per_day = false

  Example CLI::

     pleb --config pipeline.toml --set pqc_mark_only_worst_per_day=false

``pqc_tau_rec_days``
  Type: ``float``
  Default: ``7.0``
  Meaning: Recovery time for transient scan.
  Example TOML::

     pqc_tau_rec_days = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_tau_rec_days=1.0

``pqc_window_mult``
  Type: ``float``
  Default: ``5.0``
  Meaning: Window multiplier for transient scan.
  Example TOML::

     pqc_window_mult = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_window_mult=1.0

``pqc_min_points``
  Type: ``int``
  Default: ``6``
  Meaning: Minimum points for transient scan.
  Example TOML::

     pqc_min_points = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_min_points=4

``pqc_delta_chi2_thresh``
  Type: ``float``
  Default: ``25.0``
  Meaning: Delta-chi2 threshold for transients.
  Example TOML::

     pqc_delta_chi2_thresh = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_delta_chi2_thresh=1.0

``pqc_exp_dip_min_duration_days``
  Type: ``float``
  Default: ``21.0``
  Meaning: Minimum duration (days) for exp dips.
  Example TOML::

     pqc_exp_dip_min_duration_days = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_exp_dip_min_duration_days=1.0

``pqc_step_enabled``
  Type: ``bool``
  Default: ``True``
  Meaning: PQC detector/feature threshold.
  Example TOML::

     pqc_step_enabled = false

  Example CLI::

     pleb --config pipeline.toml --set pqc_step_enabled=false

``pqc_step_min_points``
  Type: ``int``
  Default: ``20``
  Meaning: PQC detector/feature threshold.
  Example TOML::

     pqc_step_min_points = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_step_min_points=4

``pqc_step_delta_chi2_thresh``
  Type: ``float``
  Default: ``25.0``
  Meaning: PQC detector/feature threshold.
  Example TOML::

     pqc_step_delta_chi2_thresh = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_step_delta_chi2_thresh=1.0

``pqc_step_scope``
  Type: ``str``
  Default: ``'both'``
  Meaning: PQC detector/feature threshold.
  Example TOML::

     pqc_step_scope = "both"

  Example CLI::

     pleb --config pipeline.toml --set pqc_step_scope="both"

``pqc_dm_step_enabled``
  Type: ``bool``
  Default: ``True``
  Meaning: PQC detector/feature threshold.
  Example TOML::

     pqc_dm_step_enabled = false

  Example CLI::

     pleb --config pipeline.toml --set pqc_dm_step_enabled=false

``pqc_dm_step_min_points``
  Type: ``int``
  Default: ``20``
  Meaning: PQC detector/feature threshold.
  Example TOML::

     pqc_dm_step_min_points = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_dm_step_min_points=4

``pqc_dm_step_delta_chi2_thresh``
  Type: ``float``
  Default: ``25.0``
  Meaning: PQC detector/feature threshold.
  Example TOML::

     pqc_dm_step_delta_chi2_thresh = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_dm_step_delta_chi2_thresh=1.0

``pqc_dm_step_scope``
  Type: ``str``
  Default: ``'both'``
  Meaning: PQC detector/feature threshold.
  Example TOML::

     pqc_dm_step_scope = "both"

  Example CLI::

     pleb --config pipeline.toml --set pqc_dm_step_scope="both"

``pqc_robust_enabled``
  Type: ``bool``
  Default: ``True``
  Meaning: PQC detector/feature threshold.
  Example TOML::

     pqc_robust_enabled = false

  Example CLI::

     pleb --config pipeline.toml --set pqc_robust_enabled=false

``pqc_robust_z_thresh``
  Type: ``float``
  Default: ``5.0``
  Meaning: PQC detector/feature threshold.
  Example TOML::

     pqc_robust_z_thresh = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_robust_z_thresh=1.0

``pqc_robust_scope``
  Type: ``str``
  Default: ``'both'``
  Meaning: PQC detector/feature threshold.
  Example TOML::

     pqc_robust_scope = "backend"

  Example CLI::

     pleb --config pipeline.toml --set pqc_robust_scope="backend"

``pqc_add_orbital_phase``
  Type: ``bool``
  Default: ``True``
  Meaning: Add orbital-phase feature.
  Example TOML::

     pqc_add_orbital_phase = false

  Example CLI::

     pleb --config pipeline.toml --set pqc_add_orbital_phase=false

``pqc_add_solar_elongation``
  Type: ``bool``
  Default: ``True``
  Meaning: Add solar elongation feature.
  Example TOML::

     pqc_add_solar_elongation = false

  Example CLI::

     pleb --config pipeline.toml --set pqc_add_solar_elongation=false

``pqc_add_elevation``
  Type: ``bool``
  Default: ``False``
  Meaning: Add elevation feature.
  Example TOML::

     pqc_add_elevation = true

  Example CLI::

     pleb --config pipeline.toml --set pqc_add_elevation=true

``pqc_add_airmass``
  Type: ``bool``
  Default: ``False``
  Meaning: Add airmass feature.
  Example TOML::

     pqc_add_airmass = true

  Example CLI::

     pleb --config pipeline.toml --set pqc_add_airmass=true

``pqc_add_parallactic_angle``
  Type: ``bool``
  Default: ``False``
  Meaning: Add parallactic-angle feature.
  Example TOML::

     pqc_add_parallactic_angle = true

  Example CLI::

     pleb --config pipeline.toml --set pqc_add_parallactic_angle=true

``pqc_add_freq_bin``
  Type: ``bool``
  Default: ``False``
  Meaning: Add frequency-bin feature.
  Example TOML::

     pqc_add_freq_bin = true

  Example CLI::

     pleb --config pipeline.toml --set pqc_add_freq_bin=true

``pqc_freq_bins``
  Type: ``int``
  Default: ``8``
  Meaning: Number of frequency bins if enabled.
  Example TOML::

     pqc_freq_bins = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_freq_bins=4

``pqc_observatory_path``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Optional observatory file path.
  Example TOML::

     pqc_observatory_path = "value"

  Example CLI::

     pleb --config pipeline.toml --set pqc_observatory_path="value"

``pqc_structure_mode``
  Type: ``str``
  Default: ``'none'``
  Meaning: Feature-structure mode (none/detrend/test/both).
  Example TOML::

     pqc_structure_mode = "both"

  Example CLI::

     pleb --config pipeline.toml --set pqc_structure_mode="both"

``pqc_structure_detrend_features``
  Type: ``Optional[List[str]]``
  Default: ``['solar_elongation_deg', 'orbital_phase']``
  Meaning: Features to detrend against.
  Example TOML::

     pqc_structure_detrend_features = ["solar_elongation_deg","orbital_phase"]

  Example CLI::

     pleb --config pipeline.toml --set pqc_structure_detrend_features=["solar_elongation_deg","orbital_phase"]

``pqc_structure_test_features``
  Type: ``Optional[List[str]]``
  Default: ``['solar_elongation_deg', 'orbital_phase']``
  Meaning: Features to test for structure.
  Example TOML::

     pqc_structure_test_features = ["solar_elongation_deg","orbital_phase"]

  Example CLI::

     pleb --config pipeline.toml --set pqc_structure_test_features=["solar_elongation_deg","orbital_phase"]

``pqc_structure_nbins``
  Type: ``int``
  Default: ``12``
  Meaning: Bin count for structure tests.
  Example TOML::

     pqc_structure_nbins = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_structure_nbins=4

``pqc_structure_min_per_bin``
  Type: ``int``
  Default: ``3``
  Meaning: Minimum points per bin.
  Example TOML::

     pqc_structure_min_per_bin = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_structure_min_per_bin=4

``pqc_structure_p_thresh``
  Type: ``float``
  Default: ``0.01``
  Meaning: p-value threshold for structure detection.
  Example TOML::

     pqc_structure_p_thresh = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_structure_p_thresh=1.0

``pqc_structure_circular_features``
  Type: ``Optional[List[str]]``
  Default: ``['orbital_phase']``
  Meaning: Circular features in [0,1).
  Example TOML::

     pqc_structure_circular_features = ["orbital_phase"]

  Example CLI::

     pleb --config pipeline.toml --set pqc_structure_circular_features=["orbital_phase"]

``pqc_structure_group_cols``
  Type: ``Optional[List[str]]``
  Default: ``None``
  Meaning: Grouping columns for structure tests.
  Example TOML::

     pqc_structure_group_cols = ["sys"]

  Example CLI::

     pleb --config pipeline.toml --set pqc_structure_group_cols=["sys"]

``pqc_outlier_gate_enabled``
  Type: ``bool``
  Default: ``False``
  Meaning: Enable hard sigma gate for outlier membership.
  Example TOML::

     pqc_outlier_gate_enabled = true

  Example CLI::

     pleb --config pipeline.toml --set pqc_outlier_gate_enabled=true

``pqc_outlier_gate_sigma``
  Type: ``float``
  Default: ``3.0``
  Meaning: Sigma threshold for outlier gate.
  Example TOML::

     pqc_outlier_gate_sigma = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_outlier_gate_sigma=1.0

``pqc_outlier_gate_resid_col``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Residual column to gate on (optional).
  Example TOML::

     pqc_outlier_gate_resid_col = "post"

  Example CLI::

     pleb --config pipeline.toml --set pqc_outlier_gate_resid_col="post"

``pqc_outlier_gate_sigma_col``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Sigma column to gate on (optional).
  Example TOML::

     pqc_outlier_gate_sigma_col = "err"

  Example CLI::

     pleb --config pipeline.toml --set pqc_outlier_gate_sigma_col="err"

``pqc_event_instrument``
  Type: ``bool``
  Default: ``False``
  Meaning: Enable per-event membership diagnostics.
  Example TOML::

     pqc_event_instrument = true

  Example CLI::

     pleb --config pipeline.toml --set pqc_event_instrument=true

``pqc_solar_events_enabled``
  Type: ``bool``
  Default: ``False``
  Meaning: Enable solar event detection.
  Example TOML::

     pqc_solar_events_enabled = true

  Example CLI::

     pleb --config pipeline.toml --set pqc_solar_events_enabled=true

``pqc_solar_approach_max_deg``
  Type: ``float``
  Default: ``30.0``
  Meaning: Max elongation for solar approach region.
  Example TOML::

     pqc_solar_approach_max_deg = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_solar_approach_max_deg=1.0

``pqc_solar_min_points_global``
  Type: ``int``
  Default: ``30``
  Meaning: Min points for global fit.
  Example TOML::

     pqc_solar_min_points_global = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_solar_min_points_global=4

``pqc_solar_min_points_year``
  Type: ``int``
  Default: ``10``
  Meaning: Min points for per-year fit.
  Example TOML::

     pqc_solar_min_points_year = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_solar_min_points_year=4

``pqc_solar_min_points_near_zero``
  Type: ``int``
  Default: ``3``
  Meaning: Min points near zero elongation.
  Example TOML::

     pqc_solar_min_points_near_zero = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_solar_min_points_near_zero=4

``pqc_solar_tau_min_deg``
  Type: ``float``
  Default: ``2.0``
  Meaning: Min elongation scale for exponential.
  Example TOML::

     pqc_solar_tau_min_deg = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_solar_tau_min_deg=1.0

``pqc_solar_tau_max_deg``
  Type: ``float``
  Default: ``60.0``
  Meaning: Max elongation scale for exponential.
  Example TOML::

     pqc_solar_tau_max_deg = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_solar_tau_max_deg=1.0

``pqc_solar_member_eta``
  Type: ``float``
  Default: ``1.0``
  Meaning: Per-point membership SNR threshold.
  Example TOML::

     pqc_solar_member_eta = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_solar_member_eta=1.0

``pqc_solar_freq_dependence``
  Type: ``bool``
  Default: ``True``
  Meaning: Fit 1/f^alpha dependence.
  Example TOML::

     pqc_solar_freq_dependence = false

  Example CLI::

     pleb --config pipeline.toml --set pqc_solar_freq_dependence=false

``pqc_solar_freq_alpha_min``
  Type: ``float``
  Default: ``0.0``
  Meaning: Lower bound for alpha.
  Example TOML::

     pqc_solar_freq_alpha_min = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_solar_freq_alpha_min=1.0

``pqc_solar_freq_alpha_max``
  Type: ``float``
  Default: ``4.0``
  Meaning: Upper bound for alpha.
  Example TOML::

     pqc_solar_freq_alpha_max = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_solar_freq_alpha_max=1.0

``pqc_solar_freq_alpha_tol``
  Type: ``float``
  Default: ``0.001``
  Meaning: Optimization tolerance for alpha.
  Example TOML::

     pqc_solar_freq_alpha_tol = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_solar_freq_alpha_tol=1.0

``pqc_solar_freq_alpha_max_iter``
  Type: ``int``
  Default: ``64``
  Meaning: Max iterations for alpha optimizer.
  Example TOML::

     pqc_solar_freq_alpha_max_iter = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_solar_freq_alpha_max_iter=4

``pqc_orbital_phase_cut_enabled``
  Type: ``bool``
  Default: ``False``
  Meaning: Enable orbital-phase based flagging.
  Example TOML::

     pqc_orbital_phase_cut_enabled = true

  Example CLI::

     pleb --config pipeline.toml --set pqc_orbital_phase_cut_enabled=true

``pqc_orbital_phase_cut_center``
  Type: ``float``
  Default: ``0.25``
  Meaning: Eclipse center phase (0..1).
  Example TOML::

     pqc_orbital_phase_cut_center = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_orbital_phase_cut_center=1.0

``pqc_orbital_phase_cut``
  Type: ``Optional[float]``
  Default: ``None``
  Meaning: Fixed orbital-phase cutoff (0..0.5), or None for auto.
  Example TOML::

     pqc_orbital_phase_cut = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_orbital_phase_cut=1.0

``pqc_orbital_phase_cut_sigma``
  Type: ``float``
  Default: ``3.0``
  Meaning: Sigma threshold for automatic cutoff estimation.
  Example TOML::

     pqc_orbital_phase_cut_sigma = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_orbital_phase_cut_sigma=1.0

``pqc_orbital_phase_cut_nbins``
  Type: ``int``
  Default: ``18``
  Meaning: Number of bins for cutoff estimation.
  Example TOML::

     pqc_orbital_phase_cut_nbins = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_orbital_phase_cut_nbins=4

``pqc_orbital_phase_cut_min_points``
  Type: ``int``
  Default: ``20``
  Meaning: Minimum points for cutoff estimation.
  Example TOML::

     pqc_orbital_phase_cut_min_points = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_orbital_phase_cut_min_points=4

``pqc_eclipse_events_enabled``
  Type: ``bool``
  Default: ``False``
  Meaning: Enable eclipse event detection.
  Example TOML::

     pqc_eclipse_events_enabled = true

  Example CLI::

     pleb --config pipeline.toml --set pqc_eclipse_events_enabled=true

``pqc_eclipse_center_phase``
  Type: ``float``
  Default: ``0.25``
  Meaning: Eclipse center phase (0..1).
  Example TOML::

     pqc_eclipse_center_phase = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_eclipse_center_phase=1.0

``pqc_eclipse_min_points``
  Type: ``int``
  Default: ``30``
  Meaning: Min points for global fit.
  Example TOML::

     pqc_eclipse_min_points = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_eclipse_min_points=4

``pqc_eclipse_width_min``
  Type: ``float``
  Default: ``0.01``
  Meaning: Min eclipse width in phase.
  Example TOML::

     pqc_eclipse_width_min = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_eclipse_width_min=1.0

``pqc_eclipse_width_max``
  Type: ``float``
  Default: ``0.5``
  Meaning: Max eclipse width in phase.
  Example TOML::

     pqc_eclipse_width_max = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_eclipse_width_max=1.0

``pqc_eclipse_member_eta``
  Type: ``float``
  Default: ``1.0``
  Meaning: Per-point membership SNR threshold.
  Example TOML::

     pqc_eclipse_member_eta = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_eclipse_member_eta=1.0

``pqc_eclipse_freq_dependence``
  Type: ``bool``
  Default: ``True``
  Meaning: Fit 1/f^alpha dependence.
  Example TOML::

     pqc_eclipse_freq_dependence = false

  Example CLI::

     pleb --config pipeline.toml --set pqc_eclipse_freq_dependence=false

``pqc_eclipse_freq_alpha_min``
  Type: ``float``
  Default: ``0.0``
  Meaning: Lower bound for alpha.
  Example TOML::

     pqc_eclipse_freq_alpha_min = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_eclipse_freq_alpha_min=1.0

``pqc_eclipse_freq_alpha_max``
  Type: ``float``
  Default: ``4.0``
  Meaning: Upper bound for alpha.
  Example TOML::

     pqc_eclipse_freq_alpha_max = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_eclipse_freq_alpha_max=1.0

``pqc_eclipse_freq_alpha_tol``
  Type: ``float``
  Default: ``0.001``
  Meaning: Optimization tolerance for alpha.
  Example TOML::

     pqc_eclipse_freq_alpha_tol = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_eclipse_freq_alpha_tol=1.0

``pqc_eclipse_freq_alpha_max_iter``
  Type: ``int``
  Default: ``64``
  Meaning: Max iterations for alpha optimizer.
  Example TOML::

     pqc_eclipse_freq_alpha_max_iter = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_eclipse_freq_alpha_max_iter=4

``pqc_gaussian_bump_enabled``
  Type: ``bool``
  Default: ``False``
  Meaning: Enable Gaussian-bump event detection.
  Example TOML::

     pqc_gaussian_bump_enabled = true

  Example CLI::

     pleb --config pipeline.toml --set pqc_gaussian_bump_enabled=true

``pqc_gaussian_bump_min_duration_days``
  Type: ``float``
  Default: ``60.0``
  Meaning: Minimum bump duration in days.
  Example TOML::

     pqc_gaussian_bump_min_duration_days = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_gaussian_bump_min_duration_days=1.0

``pqc_gaussian_bump_max_duration_days``
  Type: ``float``
  Default: ``1500.0``
  Meaning: Maximum bump duration in days.
  Example TOML::

     pqc_gaussian_bump_max_duration_days = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_gaussian_bump_max_duration_days=1.0

``pqc_gaussian_bump_n_durations``
  Type: ``int``
  Default: ``6``
  Meaning: Number of duration grid points.
  Example TOML::

     pqc_gaussian_bump_n_durations = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_gaussian_bump_n_durations=4

``pqc_gaussian_bump_min_points``
  Type: ``int``
  Default: ``20``
  Meaning: Minimum points for bump detection.
  Example TOML::

     pqc_gaussian_bump_min_points = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_gaussian_bump_min_points=4

``pqc_gaussian_bump_delta_chi2_thresh``
  Type: ``float``
  Default: ``25.0``
  Meaning: Delta-chi2 threshold for bump detection.
  Example TOML::

     pqc_gaussian_bump_delta_chi2_thresh = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_gaussian_bump_delta_chi2_thresh=1.0

``pqc_gaussian_bump_suppress_overlap``
  Type: ``bool``
  Default: ``True``
  Meaning: Suppress overlapping bumps.
  Example TOML::

     pqc_gaussian_bump_suppress_overlap = false

  Example CLI::

     pleb --config pipeline.toml --set pqc_gaussian_bump_suppress_overlap=false

``pqc_gaussian_bump_member_eta``
  Type: ``float``
  Default: ``1.0``
  Meaning: Per-point membership SNR threshold.
  Example TOML::

     pqc_gaussian_bump_member_eta = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_gaussian_bump_member_eta=1.0

``pqc_gaussian_bump_freq_dependence``
  Type: ``bool``
  Default: ``True``
  Meaning: Fit 1/f^alpha dependence.
  Example TOML::

     pqc_gaussian_bump_freq_dependence = false

  Example CLI::

     pleb --config pipeline.toml --set pqc_gaussian_bump_freq_dependence=false

``pqc_gaussian_bump_freq_alpha_min``
  Type: ``float``
  Default: ``0.0``
  Meaning: Lower bound for alpha.
  Example TOML::

     pqc_gaussian_bump_freq_alpha_min = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_gaussian_bump_freq_alpha_min=1.0

``pqc_gaussian_bump_freq_alpha_max``
  Type: ``float``
  Default: ``4.0``
  Meaning: Upper bound for alpha.
  Example TOML::

     pqc_gaussian_bump_freq_alpha_max = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_gaussian_bump_freq_alpha_max=1.0

``pqc_gaussian_bump_freq_alpha_tol``
  Type: ``float``
  Default: ``0.001``
  Meaning: Optimization tolerance for alpha.
  Example TOML::

     pqc_gaussian_bump_freq_alpha_tol = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_gaussian_bump_freq_alpha_tol=1.0

``pqc_gaussian_bump_freq_alpha_max_iter``
  Type: ``int``
  Default: ``64``
  Meaning: Max iterations for alpha optimizer.
  Example TOML::

     pqc_gaussian_bump_freq_alpha_max_iter = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_gaussian_bump_freq_alpha_max_iter=4

``pqc_glitch_enabled``
  Type: ``bool``
  Default: ``False``
  Meaning: Enable glitch event detection.
  Example TOML::

     pqc_glitch_enabled = true

  Example CLI::

     pleb --config pipeline.toml --set pqc_glitch_enabled=true

``pqc_glitch_min_points``
  Type: ``int``
  Default: ``30``
  Meaning: Minimum points for glitch detection.
  Example TOML::

     pqc_glitch_min_points = 4

  Example CLI::

     pleb --config pipeline.toml --set pqc_glitch_min_points=4

``pqc_glitch_delta_chi2_thresh``
  Type: ``float``
  Default: ``25.0``
  Meaning: Delta-chi2 threshold for glitch detection.
  Example TOML::

     pqc_glitch_delta_chi2_thresh = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_glitch_delta_chi2_thresh=1.0

``pqc_glitch_suppress_overlap``
  Type: ``bool``
  Default: ``True``
  Meaning: Suppress overlapping glitches.
  Example TOML::

     pqc_glitch_suppress_overlap = false

  Example CLI::

     pleb --config pipeline.toml --set pqc_glitch_suppress_overlap=false

``pqc_glitch_member_eta``
  Type: ``float``
  Default: ``1.0``
  Meaning: Per-point membership SNR threshold.
  Example TOML::

     pqc_glitch_member_eta = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_glitch_member_eta=1.0

``pqc_glitch_peak_tau_days``
  Type: ``float``
  Default: ``30.0``
  Meaning: Peak exponential timescale for glitch model.
  Example TOML::

     pqc_glitch_peak_tau_days = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_glitch_peak_tau_days=1.0

``pqc_glitch_noise_k``
  Type: ``float``
  Default: ``1.0``
  Meaning: Noise-aware threshold multiplier.
  Example TOML::

     pqc_glitch_noise_k = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_glitch_noise_k=1.0

``pqc_glitch_mean_window_days``
  Type: ``float``
  Default: ``180.0``
  Meaning: Rolling-mean window (days) for zero-crossing.
  Example TOML::

     pqc_glitch_mean_window_days = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_glitch_mean_window_days=1.0

``pqc_glitch_min_duration_days``
  Type: ``float``
  Default: ``1000.0``
  Meaning: Minimum glitch duration (days).
  Example TOML::

     pqc_glitch_min_duration_days = 1.0

  Example CLI::

     pleb --config pipeline.toml --set pqc_glitch_min_duration_days=1.0

``pqc_backend_profiles_path``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Optional TOML with per-backend pqc overrides.
  Example TOML::

     pqc_backend_profiles_path = "configs/rules/pqc/backend_profiles.example.toml"

  Example CLI::

     pleb --config pipeline.toml --set pqc_backend_profiles_path="configs/rules/pqc/backend_profiles.example.toml"

QC report settings
------------------

``qc_report_backend_col``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Backend column name for reports (optional).
  Example TOML::

     qc_report_backend_col = "sys"

  Example CLI::

     pleb --config pipeline.toml --set qc_report_backend_col="sys"

``qc_report_backend``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Optional backend key to plot.
  Example TOML::

     qc_report_backend = "NRT.NUPPI.1484"

  Example CLI::

     pleb --config pipeline.toml --set qc_report_backend="NRT.NUPPI.1484"

``qc_report_dir``
  Type: ``Optional[Path]``
  Default: ``None``
  Meaning: Output directory for reports (optional).
  Example TOML::

     qc_report_dir = "results/qc_report"

  Example CLI::

     pleb --config pipeline.toml --set qc_report_dir="results/qc_report"

``qc_report_no_plots``
  Type: ``bool``
  Default: ``False``
  Meaning: Skip transient plots in reports.
  Example TOML::

     qc_report_no_plots = true

  Example CLI::

     pleb --config pipeline.toml --set qc_report_no_plots=true

``qc_report_structure_group_cols``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Structure grouping override for reports.
  Example TOML::

     qc_report_structure_group_cols = "value"

  Example CLI::

     pleb --config pipeline.toml --set qc_report_structure_group_cols="value"

``qc_report_no_feature_plots``
  Type: ``bool``
  Default: ``False``
  Meaning: Skip feature plots in reports.
  Example TOML::

     qc_report_no_feature_plots = true

  Example CLI::

     pleb --config pipeline.toml --set qc_report_no_feature_plots=true

``qc_report_compact_pdf``
  Type: ``bool``
  Default: ``False``
  Meaning: Generate compact composite PDF report.
  Example TOML::

     qc_report_compact_pdf = true

  Example CLI::

     pleb --config pipeline.toml --set qc_report_compact_pdf=true

``qc_report_compact_pdf_name``
  Type: ``str``
  Default: ``'qc_compact_report.pdf'``
  Meaning: Filename for compact PDF report.
  Example TOML::

     qc_report_compact_pdf_name = "qc_compact_report.pdf"

  Example CLI::

     pleb --config pipeline.toml --set qc_report_compact_pdf_name="qc_compact_report.pdf"

``qc_report_compact_outlier_cols``
  Type: ``Optional[List[str]]``
  Default: ``None``
  Meaning: QC report generation setting.
  Example TOML::

     qc_report_compact_outlier_cols = ["bad_point","robust_outlier","robust_global_outlier","bad_mad"]

  Example CLI::

     pleb --config pipeline.toml --set qc_report_compact_outlier_cols=["bad_point","robust_outlier","robust_global_outlier","bad_mad"]

FixDataset settings
-------------------

``fix_apply``
  Type: ``bool``
  Default: ``False``
  Meaning: Whether FixDataset applies changes and commits.
  Example TOML::

     fix_apply = true

  Example CLI::

     pleb --config pipeline.toml --set fix_apply=true

``fix_branch_name``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Name of FixDataset branch. If unset and fix_apply is true,
  Example TOML::

     fix_branch_name = "fix_dataset"

  Example CLI::

     pleb --config pipeline.toml --set fix_branch_name="fix_dataset"

``fix_base_branch``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Base branch for FixDataset.
  Example TOML::

     fix_base_branch = "raw_ingest"

  Example CLI::

     pleb --config pipeline.toml --set fix_base_branch="raw_ingest"

``fix_commit_message``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Commit message for FixDataset.
  Example TOML::

     fix_commit_message = "FixDataset: normalize flags"

  Example CLI::

     pleb --config pipeline.toml --set fix_commit_message="FixDataset: normalize flags"

``fix_backup``
  Type: ``bool``
  Default: ``True``
  Meaning: Create backup before FixDataset modifications.
  Example TOML::

     fix_backup = false

  Example CLI::

     pleb --config pipeline.toml --set fix_backup=false

``fix_dry_run``
  Type: ``bool``
  Default: ``False``
  Meaning: If True, FixDataset does not write changes.
  Example TOML::

     fix_dry_run = true

  Example CLI::

     pleb --config pipeline.toml --set fix_dry_run=true

``fix_update_alltim_includes``
  Type: ``bool``
  Default: ``True``
  Meaning: Update INCLUDE lines in .tim files.
  Example TOML::

     fix_update_alltim_includes = false

  Example CLI::

     pleb --config pipeline.toml --set fix_update_alltim_includes=false

``fix_min_toas_per_backend_tim``
  Type: ``int``
  Default: ``10``
  Meaning: Minimum TOAs per backend .tim.
  Example TOML::

     fix_min_toas_per_backend_tim = 4

  Example CLI::

     pleb --config pipeline.toml --set fix_min_toas_per_backend_tim=4

``fix_required_tim_flags``
  Type: ``Dict[str, str]``
  Default: ``{}``
  Meaning: Required flags for .tim entries.
  Example TOML::

     fix_required_tim_flags = { "-pta" = "EPTA" }

  Example CLI::

     pleb --config pipeline.toml --set fix_required_tim_flags={ "-pta" = "EPTA" }

``fix_system_flag_mapping_path``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Editable system-flag mapping JSON (optional).
  Example TOML::

     fix_system_flag_mapping_path = "configs/catalogs/system_flags/system_flag_mapping.example.json"

  Example CLI::

     pleb --config pipeline.toml --set fix_system_flag_mapping_path="configs/catalogs/system_flags/system_flag_mapping.example.json"

``fix_system_flag_table_path``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_system_flag_table_path = "system_flag_table.json"

  Example CLI::

     pleb --config pipeline.toml --set fix_system_flag_table_path="system_flag_table.json"

``fix_generate_alltim_variants``
  Type: ``bool``
  Default: ``False``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_generate_alltim_variants = true

  Example CLI::

     pleb --config pipeline.toml --set fix_generate_alltim_variants=true

``fix_backend_classifications_path``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_backend_classifications_path = "configs/catalogs/variants/backend_classifications_legacy_new.toml"

  Example CLI::

     pleb --config pipeline.toml --set fix_backend_classifications_path="configs/catalogs/variants/backend_classifications_legacy_new.toml"

``fix_alltim_variants_path``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_alltim_variants_path = "configs/catalogs/variants/alltim_variants_legacy_new.toml"

  Example CLI::

     pleb --config pipeline.toml --set fix_alltim_variants_path="configs/catalogs/variants/alltim_variants_legacy_new.toml"

``fix_relabel_rules_path``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Declarative TOA relabel rules TOML (optional).
  Example TOML::

     fix_relabel_rules_path = "configs/rules/relabel/relabel_rules.example.toml"

  Example CLI::

     pleb --config pipeline.toml --set fix_relabel_rules_path="configs/rules/relabel/relabel_rules.example.toml"

``fix_overlap_rules_path``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Declarative overlap rules TOML (optional).
  Example TOML::

     fix_overlap_rules_path = "configs/rules/overlap/overlap_rules.example.toml"

  Example CLI::

     pleb --config pipeline.toml --set fix_overlap_rules_path="configs/rules/overlap/overlap_rules.example.toml"

``fix_overlap_exact_catalog_path``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: TOML keep->drop map for exact overlap removal.
  Example TOML::

     fix_overlap_exact_catalog_path = "configs/catalogs/system_tables/overlapped_timfiles.toml"

  Example CLI::

     pleb --config pipeline.toml --set fix_overlap_exact_catalog_path="configs/catalogs/system_tables/overlapped_timfiles.toml"

``fix_jump_reference_variants``
  Type: ``bool``
  Default: ``False``
  Meaning: Build per-variant reference-system jump parfiles.
  Example TOML::

     fix_jump_reference_variants = true

  Example CLI::

     pleb --config pipeline.toml --set fix_jump_reference_variants=true

``fix_jump_reference_keep_tmp``
  Type: ``bool``
  Default: ``False``
  Meaning: Keep temporary split tim/par files.
  Example TOML::

     fix_jump_reference_keep_tmp = true

  Example CLI::

     pleb --config pipeline.toml --set fix_jump_reference_keep_tmp=true

``fix_jump_reference_jump_flag``
  Type: ``str``
  Default: ``'-sys'``
  Meaning: Jump flag used in generated variant parfiles.
  Example TOML::

     fix_jump_reference_jump_flag = "-sys"

  Example CLI::

     pleb --config pipeline.toml --set fix_jump_reference_jump_flag="-sys"

``fix_jump_reference_csv_dir``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Output directory for jump-reference CSV files.
  Example TOML::

     fix_jump_reference_csv_dir = "results/jump_reference"

  Example CLI::

     pleb --config pipeline.toml --set fix_jump_reference_csv_dir="results/jump_reference"

``fix_infer_system_flags``
  Type: ``bool``
  Default: ``False``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_infer_system_flags = true

  Example CLI::

     pleb --config pipeline.toml --set fix_infer_system_flags=true

``fix_system_flag_overwrite_existing``
  Type: ``bool``
  Default: ``False``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_system_flag_overwrite_existing = true

  Example CLI::

     pleb --config pipeline.toml --set fix_system_flag_overwrite_existing=true

``fix_wsrt_p2_force_sys_by_freq``
  Type: ``bool``
  Default: ``False``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_wsrt_p2_force_sys_by_freq = true

  Example CLI::

     pleb --config pipeline.toml --set fix_wsrt_p2_force_sys_by_freq=true

``fix_wsrt_p2_prefer_dual_channel``
  Type: ``bool``
  Default: ``False``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_wsrt_p2_prefer_dual_channel = true

  Example CLI::

     pleb --config pipeline.toml --set fix_wsrt_p2_prefer_dual_channel=true

``fix_wsrt_p2_mjd_tol_sec``
  Type: ``float``
  Default: ``9.9e-07``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_wsrt_p2_mjd_tol_sec = 1.0

  Example CLI::

     pleb --config pipeline.toml --set fix_wsrt_p2_mjd_tol_sec=1.0

``fix_wsrt_p2_action``
  Type: ``str``
  Default: ``'comment'``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_wsrt_p2_action = "comment"

  Example CLI::

     pleb --config pipeline.toml --set fix_wsrt_p2_action="comment"

``fix_wsrt_p2_comment_prefix``
  Type: ``str``
  Default: ``'C WSRT_P2_PREFER_DUAL'``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_wsrt_p2_comment_prefix = "C WSRT_P2_PREFER_DUAL"

  Example CLI::

     pleb --config pipeline.toml --set fix_wsrt_p2_comment_prefix="C WSRT_P2_PREFER_DUAL"

``fix_backend_overrides``
  Type: ``Dict[str, str]``
  Default: ``{}``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_backend_overrides = { "JBO.DFB.1400.tim" = "DFB" }

  Example CLI::

     pleb --config pipeline.toml --set fix_backend_overrides={ "JBO.DFB.1400.tim" = "DFB" }

``fix_raise_on_backend_missing``
  Type: ``bool``
  Default: ``False``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_raise_on_backend_missing = true

  Example CLI::

     pleb --config pipeline.toml --set fix_raise_on_backend_missing=true

``fix_dedupe_toas_within_tim``
  Type: ``bool``
  Default: ``True``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_dedupe_toas_within_tim = false

  Example CLI::

     pleb --config pipeline.toml --set fix_dedupe_toas_within_tim=false

``fix_dedupe_mjd_tol_sec``
  Type: ``float``
  Default: ``0.0``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_dedupe_mjd_tol_sec = 1.0

  Example CLI::

     pleb --config pipeline.toml --set fix_dedupe_mjd_tol_sec=1.0

``fix_dedupe_freq_tol_mhz``
  Type: ``Optional[float]``
  Default: ``None``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_dedupe_freq_tol_mhz = 1.0

  Example CLI::

     pleb --config pipeline.toml --set fix_dedupe_freq_tol_mhz=1.0

``fix_dedupe_freq_tol_auto``
  Type: ``bool``
  Default: ``False``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_dedupe_freq_tol_auto = true

  Example CLI::

     pleb --config pipeline.toml --set fix_dedupe_freq_tol_auto=true

``fix_check_duplicate_backend_tims``
  Type: ``bool``
  Default: ``False``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_check_duplicate_backend_tims = true

  Example CLI::

     pleb --config pipeline.toml --set fix_check_duplicate_backend_tims=true

``fix_remove_overlaps_exact``
  Type: ``bool``
  Default: ``True``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_remove_overlaps_exact = false

  Example CLI::

     pleb --config pipeline.toml --set fix_remove_overlaps_exact=false

``fix_insert_missing_jumps``
  Type: ``bool``
  Default: ``True``
  Meaning: Insert missing JUMP lines.
  Example TOML::

     fix_insert_missing_jumps = false

  Example CLI::

     pleb --config pipeline.toml --set fix_insert_missing_jumps=false

``fix_jump_flag``
  Type: ``str``
  Default: ``'-sys'``
  Meaning: Flag used for inserted jumps.
  Example TOML::

     fix_jump_flag = "-sys"

  Example CLI::

     pleb --config pipeline.toml --set fix_jump_flag="-sys"

``fix_prune_stale_jumps``
  Type: ``bool``
  Default: ``False``
  Meaning: Drop JUMPs not present in timfile flags.
  Example TOML::

     fix_prune_stale_jumps = true

  Example CLI::

     pleb --config pipeline.toml --set fix_prune_stale_jumps=true

``fix_ensure_ephem``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Ensure ephemeris parameter exists.
  Example TOML::

     fix_ensure_ephem = "DE440"

  Example CLI::

     pleb --config pipeline.toml --set fix_ensure_ephem="DE440"

``fix_ensure_clk``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Ensure clock parameter exists.
  Example TOML::

     fix_ensure_clk = "TT(BIPM2024)"

  Example CLI::

     pleb --config pipeline.toml --set fix_ensure_clk="TT(BIPM2024)"

``fix_ensure_ne_sw``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Ensure NE_SW parameter exists.
  Example TOML::

     fix_ensure_ne_sw = "7.9"

  Example CLI::

     pleb --config pipeline.toml --set fix_ensure_ne_sw="7.9"

``fix_force_ne_sw_overwrite``
  Type: ``bool``
  Default: ``False``
  Meaning: Overwrite existing NE_SW values when true.
  Example TOML::

     fix_force_ne_sw_overwrite = true

  Example CLI::

     pleb --config pipeline.toml --set fix_force_ne_sw_overwrite=true

``fix_remove_patterns``
  Type: ``List[str]``
  Default: ``['NRT.NUPPI.', 'NRT.NUXPI.']``
  Meaning: Patterns to remove from .par/.tim.
  Example TOML::

     fix_remove_patterns = ["NRT.NUPPI.","NRT.NUXPI."]

  Example CLI::

     pleb --config pipeline.toml --set fix_remove_patterns=["NRT.NUPPI.","NRT.NUXPI."]

``fix_coord_convert``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Optional coordinate conversion.
  Example TOML::

     fix_coord_convert = "equatorial_to_ecliptic"

  Example CLI::

     pleb --config pipeline.toml --set fix_coord_convert="equatorial_to_ecliptic"

``fix_prune_missing_includes``
  Type: ``bool``
  Default: ``True``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_prune_missing_includes = false

  Example CLI::

     pleb --config pipeline.toml --set fix_prune_missing_includes=false

``fix_drop_small_backend_includes``
  Type: ``bool``
  Default: ``True``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_drop_small_backend_includes = false

  Example CLI::

     pleb --config pipeline.toml --set fix_drop_small_backend_includes=false

``fix_system_flag_update_table``
  Type: ``bool``
  Default: ``True``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_system_flag_update_table = false

  Example CLI::

     pleb --config pipeline.toml --set fix_system_flag_update_table=false

``fix_default_backend``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_default_backend = "NUPPI"

  Example CLI::

     pleb --config pipeline.toml --set fix_default_backend="NUPPI"

``fix_group_flag``
  Type: ``str``
  Default: ``'-group'``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_group_flag = "-group"

  Example CLI::

     pleb --config pipeline.toml --set fix_group_flag="-group"

``fix_pta_flag``
  Type: ``str``
  Default: ``'-pta'``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_pta_flag = "-pta"

  Example CLI::

     pleb --config pipeline.toml --set fix_pta_flag="-pta"

``fix_pta_value``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_pta_value = "EPTA"

  Example CLI::

     pleb --config pipeline.toml --set fix_pta_value="EPTA"

``fix_standardize_par_values``
  Type: ``bool``
  Default: ``True``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_standardize_par_values = false

  Example CLI::

     pleb --config pipeline.toml --set fix_standardize_par_values=false

``fix_prune_small_system_toas``
  Type: ``bool``
  Default: ``False``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_prune_small_system_toas = true

  Example CLI::

     pleb --config pipeline.toml --set fix_prune_small_system_toas=true

``fix_prune_small_system_flag``
  Type: ``str``
  Default: ``'-sys'``
  Meaning: FixDataset cleanup/normalization setting.
  Example TOML::

     fix_prune_small_system_flag = "-sys"

  Example CLI::

     pleb --config pipeline.toml --set fix_prune_small_system_flag="-sys"

``fix_qc_remove_outliers``
  Type: ``bool``
  Default: ``False``
  Meaning: Comment/delete TOAs flagged by pqc outputs.
  Example TOML::

     fix_qc_remove_outliers = true

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_remove_outliers=true

``fix_qc_outlier_cols``
  Type: ``Optional[List[str]]``
  Default: ``None``
  Meaning: FixDataset action policy for PQC-derived flags.
  Example TOML::

     fix_qc_outlier_cols = ["bad_point","robust_outlier","robust_global_outlier","bad_mad"]

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_outlier_cols=["bad_point","robust_outlier","robust_global_outlier","bad_mad"]

``fix_qc_action``
  Type: ``str``
  Default: ``'comment'``
  Meaning: Action for pqc outliers (comment/delete).
  Example TOML::

     fix_qc_action = "comment"

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_action="comment"

``fix_qc_comment_prefix``
  Type: ``str``
  Default: ``'C QC_OUTLIER'``
  Meaning: Prefix for commented TOA lines.
  Example TOML::

     fix_qc_comment_prefix = "C QC_OUTLIER"

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_comment_prefix="C QC_OUTLIER"

``fix_qc_backend_col``
  Type: ``str``
  Default: ``'sys'``
  Meaning: Backend column for pqc matching (if needed).
  Example TOML::

     fix_qc_backend_col = "sys"

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_backend_col="sys"

``fix_qc_remove_bad``
  Type: ``bool``
  Default: ``True``
  Meaning: Act on bad/bad_day flags.
  Example TOML::

     fix_qc_remove_bad = false

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_remove_bad=false

``fix_qc_remove_transients``
  Type: ``bool``
  Default: ``False``
  Meaning: Act on transient flags.
  Example TOML::

     fix_qc_remove_transients = true

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_remove_transients=true

``fix_qc_remove_solar``
  Type: ``bool``
  Default: ``False``
  Meaning: Act on solar-elongation flags.
  Example TOML::

     fix_qc_remove_solar = true

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_remove_solar=true

``fix_qc_solar_action``
  Type: ``str``
  Default: ``'comment'``
  Meaning: Action for solar-flagged TOAs (comment/delete).
  Example TOML::

     fix_qc_solar_action = "comment"

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_solar_action="comment"

``fix_qc_solar_comment_prefix``
  Type: ``str``
  Default: ``'# QC_SOLAR'``
  Meaning: Prefix for solar-flagged TOA comments.
  Example TOML::

     fix_qc_solar_comment_prefix = "# QC_SOLAR"

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_solar_comment_prefix="# QC_SOLAR"

``fix_qc_remove_orbital_phase``
  Type: ``bool``
  Default: ``False``
  Meaning: Act on orbital-phase flags.
  Example TOML::

     fix_qc_remove_orbital_phase = true

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_remove_orbital_phase=true

``fix_qc_orbital_phase_action``
  Type: ``str``
  Default: ``'comment'``
  Meaning: Action for orbital-phase flagged TOAs (comment/delete).
  Example TOML::

     fix_qc_orbital_phase_action = "comment"

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_orbital_phase_action="comment"

``fix_qc_orbital_phase_comment_prefix``
  Type: ``str``
  Default: ``'# QC_BIANRY_ECLIPSE'``
  Meaning: Prefix for orbital-phase TOA comments.
  Example TOML::

     fix_qc_orbital_phase_comment_prefix = "# QC_BINARY_ECLIPSE"

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_orbital_phase_comment_prefix="# QC_BINARY_ECLIPSE"

``fix_qc_merge_tol_days``
  Type: ``float``
  Default: ``2.0 / 86400.0``
  Meaning: MJD tolerance when matching TOAs.
  Example TOML::

     fix_qc_merge_tol_days = 1.0

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_merge_tol_days=1.0

``fix_qc_results_dir``
  Type: ``Optional[Path]``
  Default: ``None``
  Meaning: Directory containing pqc CSV outputs. If unset and
  Example TOML::

     fix_qc_results_dir = "results/EPTA_combination_report_20260319T1200"

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_results_dir="results/EPTA_combination_report_20260319T1200"

``fix_qc_branch``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Branch subdir for pqc CSV outputs. If unset and
  Example TOML::

     fix_qc_branch = "main"

  Example CLI::

     pleb --config pipeline.toml --set fix_qc_branch="main"

Param-scan settings
-------------------

``param_scan_typical``
  Type: ``bool``
  Default: ``False``
  Meaning: Enable typical param-scan profile.
  Example TOML::

     param_scan_typical = true

  Example CLI::

     pleb --config pipeline.toml --set param_scan_typical=true

``param_scan_dm_redchisq_threshold``
  Type: ``float``
  Default: ``2.0``
  Meaning: Threshold for DM scan.
  Example TOML::

     param_scan_dm_redchisq_threshold = 1.0

  Example CLI::

     pleb --config pipeline.toml --set param_scan_dm_redchisq_threshold=1.0

``param_scan_dm_max_order``
  Type: ``int``
  Default: ``4``
  Meaning: Max DM derivative order.
  Example TOML::

     param_scan_dm_max_order = 4

  Example CLI::

     pleb --config pipeline.toml --set param_scan_dm_max_order=4

``param_scan_btx_max_fb``
  Type: ``int``
  Default: ``3``
  Meaning: Max FB derivative order.
  Example TOML::

     param_scan_btx_max_fb = 4

  Example CLI::

     pleb --config pipeline.toml --set param_scan_btx_max_fb=4

Binary analysis settings
------------------------

``binary_only_models``
  Type: ``Optional[List[str]]``
  Default: ``None``
  Meaning: Limit binary analysis to model names.
  Example TOML::

     binary_only_models = ["ELL1","BTX"]

  Example CLI::

     pleb --config pipeline.toml --set binary_only_models=["ELL1","BTX"]

Ingest settings
---------------

``ingest_mapping_file``
  Type: ``Optional[Path]``
  Default: ``None``
  Meaning: JSON mapping file for ingest mode (optional).
  Example TOML::

     ingest_mapping_file = "configs/catalogs/ingest/epta_data.json"

  Example CLI::

     pleb --config pipeline.toml --set ingest_mapping_file="configs/catalogs/ingest/epta_data.json"

``ingest_output_dir``
  Type: ``Optional[Path]``
  Default: ``None``
  Meaning: Output root directory for ingest mode (optional).
  Example TOML::

     ingest_output_dir = "/data/epta-dr3"

  Example CLI::

     pleb --config pipeline.toml --set ingest_output_dir="/data/epta-dr3"

``ingest_commit_branch``
  Type: ``bool``
  Default: ``True``
  Meaning: Create a new branch and commit ingest outputs.
  Example TOML::

     ingest_commit_branch = false

  Example CLI::

     pleb --config pipeline.toml --set ingest_commit_branch=false

``ingest_commit_branch_name``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Optional name for the ingest branch.
  Example TOML::

     ingest_commit_branch_name = "raw_ingest"

  Example CLI::

     pleb --config pipeline.toml --set ingest_commit_branch_name="raw_ingest"

``ingest_commit_base_branch``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Base branch for the ingest commit.
  Example TOML::

     ingest_commit_base_branch = "main"

  Example CLI::

     pleb --config pipeline.toml --set ingest_commit_base_branch="main"

``ingest_commit_message``
  Type: ``Optional[str]``
  Default: ``None``
  Meaning: Commit message for ingest.
  Example TOML::

     ingest_commit_message = "Ingest: collect tim files"

  Example CLI::

     pleb --config pipeline.toml --set ingest_commit_message="Ingest: collect tim files"

``ingest_verify``
  Type: ``bool``
  Default: ``False``
  Meaning: Ingest setting.
  Example TOML::

     ingest_verify = true

  Example CLI::

     pleb --config pipeline.toml --set ingest_verify=true

Rendering settings
------------------

``dpi``
  Type: ``int``
  Default: ``120``
  Meaning: Plot resolution.
  Example TOML::

     dpi = 150

  Example CLI::

     pleb --config pipeline.toml --set dpi=150

``max_covmat_params``
  Type: ``Optional[int]``
  Default: ``None``
  Meaning: Max params in covariance heatmaps.
  Example TOML::

     max_covmat_params = 80

  Example CLI::

     pleb --config pipeline.toml --set max_covmat_params=80

