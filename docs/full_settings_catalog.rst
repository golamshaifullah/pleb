Full Settings Catalog
=====================

This page is auto-generated from ``pleb.ux.key_catalog``.

.. note::
   Regenerate with ``python scripts/generate_settings_catalog.py``.

policy.report
-------------

``backend``
  Type: ``Optional[str]``
  Default: None
  Modes: ``qc-report``
  Level: ``balanced``

``backend_col``
  Type: ``str``
  Default: ``group``
  Modes: ``qc-report``
  Level: ``balanced``

pipeline
--------

``binary_only_models``
  Type: ``Optional[List[str]]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``minimal``

data
----

``branches``
  Type: ``List[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``minimal``

pipeline
--------

``cleanup_output_tree``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

``cleanup_work_dir``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

policy.report
-------------

``compact_pdf``
  Type: ``bool``
  Default: ``False``
  Modes: ``qc-report``
  Level: ``balanced``

``compact_pdf_name``
  Type: ``str``
  Default: ``qc_compact_report.pdf``
  Modes: ``qc-report``
  Level: ``balanced``

policy.compare_public
---------------------

``compare_public_out_dir``
  Type: ``Optional[Path]``
  Default: None
  Modes: ``workflow``
  Level: ``full``

``compare_public_providers_path``
  Type: ``Optional[Path]``
  Default: None
  Modes: ``workflow``
  Level: ``full``

paths
-----

``dataset_name``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``minimal``

pipeline
--------

``dpi``
  Type: ``int``
  Default: ``120``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

``epoch``
  Type: ``str``
  Default: ``55000``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

policy.fix
----------

``fix_alltim_variants_path``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_apply``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_backend_classifications_path``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_backend_overrides``
  Type: ``Dict[str, str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_backup``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_base_branch``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_branch_name``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_check_duplicate_backend_tims``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_commit_message``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_coord_convert``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``
  Choices: ``equatorial_to_ecliptic, ecliptic_to_equatorial``

``fix_dedupe_freq_tol_auto``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_dedupe_freq_tol_mhz``
  Type: ``Optional[float]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_dedupe_mjd_tol_sec``
  Type: ``float``
  Default: ``0.0``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_dedupe_toas_within_tim``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_default_backend``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_drop_small_backend_includes``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_dry_run``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_ensure_clk``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_ensure_ephem``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_ensure_ne_sw``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_flag_sys_freq_rules_enabled``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_flag_sys_freq_rules_path``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_force_ne_sw_overwrite``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_generate_alltim_variants``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_group_flag``
  Type: ``str``
  Default: ``-group``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_infer_system_flags``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_insert_missing_jumps``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_jump_flag``
  Type: ``str``
  Default: ``-sys``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_jump_reference_csv_dir``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_jump_reference_jump_flag``
  Type: ``str``
  Default: ``-sys``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_jump_reference_keep_tmp``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_jump_reference_variants``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_min_toas_per_backend_tim``
  Type: ``int``
  Default: ``10``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_overlap_exact_catalog_path``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_overlap_rules_path``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_prune_missing_includes``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_prune_small_system_flag``
  Type: ``str``
  Default: ``-sys``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_prune_small_system_toas``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_prune_stale_jumps``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_pta_flag``
  Type: ``str``
  Default: ``-pta``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_pta_value``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_qc_action``
  Type: ``str``
  Default: ``comment``
  Modes: ``pipeline, workflow``
  Level: ``full``
  Choices: ``comment, delete``

``fix_qc_backend_col``
  Type: ``str``
  Default: ``sys``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_branch``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_comment_prefix``
  Type: ``str``
  Default: ``C QC_OUTLIER``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_merge_tol_days``
  Type: ``float``
  Default: ``2.3148148148148147e-05``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_orbital_phase_action``
  Type: ``str``
  Default: ``comment``
  Modes: ``pipeline, workflow``
  Level: ``full``
  Choices: ``comment, delete``

``fix_qc_orbital_phase_comment_prefix``
  Type: ``str``
  Default: ``C QC_BIANRY_ECLIPSE``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_outlier_cols``
  Type: ``Optional[List[str]]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_pqc_bad_value``
  Type: ``str``
  Default: ``bad``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_pqc_event_prefix``
  Type: ``str``
  Default: ``event_``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_pqc_flag_name``
  Type: ``str``
  Default: ``-pqc``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_pqc_good_value``
  Type: ``str``
  Default: ``good``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_remove_bad``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_remove_orbital_phase``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_remove_outliers``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_remove_solar``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_remove_transients``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_results_dir``
  Type: ``Optional[Path]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_solar_action``
  Type: ``str``
  Default: ``comment``
  Modes: ``pipeline, workflow``
  Level: ``full``
  Choices: ``comment, delete``

``fix_qc_solar_comment_prefix``
  Type: ``str``
  Default: ``C QC_SOLAR``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_qc_write_pqc_flag``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_raise_on_backend_missing``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_relabel_rules_path``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_remove_overlaps_exact``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_remove_patterns``
  Type: ``List[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_required_tim_flags``
  Type: ``Dict[str, str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_standardize_par_values``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_system_flag_mapping_path``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_system_flag_overwrite_existing``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_system_flag_table_path``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_system_flag_update_table``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_update_alltim_includes``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``balanced``

``fix_wsrt_p2_action``
  Type: ``str``
  Default: ``comment``
  Modes: ``pipeline, workflow``
  Level: ``full``
  Choices: ``comment, delete``

``fix_wsrt_p2_comment_prefix``
  Type: ``str``
  Default: ``C WSRT_P2_PREFER_DUAL``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_wsrt_p2_force_sys_by_freq``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_wsrt_p2_mjd_tol_sec``
  Type: ``float``
  Default: ``9.9e-07``
  Modes: ``pipeline, workflow``
  Level: ``full``

``fix_wsrt_p2_prefer_dual_channel``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

pipeline
--------

``force_rerun``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

paths
-----

``home_dir``
  Type: ``Path``
  Default: ``/path/to/repo``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

policy.ingest
-------------

``ingest_commit_base_branch``
  Type: ``Optional[str]``
  Default: None
  Modes: ``ingest``
  Level: ``balanced``

``ingest_commit_branch``
  Type: ``bool``
  Default: ``True``
  Modes: ``ingest, pipeline, workflow``
  Level: ``minimal``

``ingest_commit_branch_name``
  Type: ``Optional[str]``
  Default: None
  Modes: ``ingest``
  Level: ``balanced``

``ingest_commit_message``
  Type: ``Optional[str]``
  Default: None
  Modes: ``ingest``
  Level: ``balanced``

``ingest_mapping_file``
  Type: ``Optional[Path]``
  Default: None
  Modes: ``ingest``
  Level: ``balanced``

``ingest_output_dir``
  Type: ``Optional[Path]``
  Default: None
  Modes: ``ingest``
  Level: ``balanced``

``ingest_verify``
  Type: ``bool``
  Default: ``False``
  Modes: ``ingest``
  Level: ``balanced``

data
----

``jobs``
  Type: ``int``
  Default: ``1``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

pipeline
--------

``make_binary_analysis``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

``make_change_reports``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

``make_covariance_heatmaps``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

run
---

``make_covmat``
  Type: ``Optional[bool]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``minimal``

pipeline
--------

``make_outlier_reports``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

run
---

``make_plots``
  Type: ``Optional[bool]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``minimal``

``make_reports``
  Type: ``Optional[bool]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``minimal``

pipeline
--------

``make_residual_plots``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

``make_toa_coverage_plots``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

``max_covmat_params``
  Type: ``Optional[int]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``minimal``

policy.report
-------------

``no_feature_plots``
  Type: ``bool``
  Default: ``False``
  Modes: ``qc-report``
  Level: ``balanced``

``no_plots``
  Type: ``bool``
  Default: ``False``
  Modes: ``qc-report``
  Level: ``balanced``

pipeline
--------

``outdir_name``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``minimal``

``param_scan_btx_max_fb``
  Type: ``int``
  Default: ``3``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

``param_scan_dm_max_order``
  Type: ``int``
  Default: ``4``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

``param_scan_dm_redchisq_threshold``
  Type: ``float``
  Default: ``2.0``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

``param_scan_typical``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

policy.pqc
----------

``pqc_add_airmass``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_add_elevation``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_add_freq_bin``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_add_orbital_phase``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_add_parallactic_angle``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_add_solar_elongation``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_backend_col``
  Type: ``str``
  Default: ``group``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_backend_profiles_path``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_delta_chi2_thresh``
  Type: ``float``
  Default: ``25.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_dm_step_delta_chi2_thresh``
  Type: ``float``
  Default: ``25.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_dm_step_enabled``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_dm_step_min_points``
  Type: ``int``
  Default: ``20``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_dm_step_scope``
  Type: ``str``
  Default: ``both``
  Modes: ``pipeline, workflow``
  Level: ``full``
  Choices: ``global, backend, both``

``pqc_drop_unmatched``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_eclipse_center_phase``
  Type: ``float``
  Default: ``0.25``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_eclipse_events_enabled``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_eclipse_freq_alpha_max``
  Type: ``float``
  Default: ``4.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_eclipse_freq_alpha_max_iter``
  Type: ``int``
  Default: ``64``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_eclipse_freq_alpha_min``
  Type: ``float``
  Default: ``0.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_eclipse_freq_alpha_tol``
  Type: ``float``
  Default: ``0.001``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_eclipse_freq_dependence``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_eclipse_member_eta``
  Type: ``float``
  Default: ``1.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_eclipse_min_points``
  Type: ``int``
  Default: ``30``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_eclipse_width_max``
  Type: ``float``
  Default: ``0.5``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_eclipse_width_min``
  Type: ``float``
  Default: ``0.01``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_event_instrument``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_exp_dip_min_duration_days``
  Type: ``float``
  Default: ``21.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_fdr_q``
  Type: ``float``
  Default: ``0.01``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_freq_bins``
  Type: ``int``
  Default: ``8``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_gaussian_bump_delta_chi2_thresh``
  Type: ``float``
  Default: ``25.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_gaussian_bump_enabled``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_gaussian_bump_freq_alpha_max``
  Type: ``float``
  Default: ``4.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_gaussian_bump_freq_alpha_max_iter``
  Type: ``int``
  Default: ``64``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_gaussian_bump_freq_alpha_min``
  Type: ``float``
  Default: ``0.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_gaussian_bump_freq_alpha_tol``
  Type: ``float``
  Default: ``0.001``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_gaussian_bump_freq_dependence``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_gaussian_bump_max_duration_days``
  Type: ``float``
  Default: ``1500.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_gaussian_bump_member_eta``
  Type: ``float``
  Default: ``1.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_gaussian_bump_min_duration_days``
  Type: ``float``
  Default: ``60.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_gaussian_bump_min_points``
  Type: ``int``
  Default: ``20``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_gaussian_bump_n_durations``
  Type: ``int``
  Default: ``6``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_gaussian_bump_suppress_overlap``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_glitch_delta_chi2_thresh``
  Type: ``float``
  Default: ``25.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_glitch_enabled``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_glitch_mean_window_days``
  Type: ``float``
  Default: ``180.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_glitch_member_eta``
  Type: ``float``
  Default: ``1.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_glitch_min_duration_days``
  Type: ``float``
  Default: ``1000.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_glitch_min_points``
  Type: ``int``
  Default: ``30``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_glitch_noise_k``
  Type: ``float``
  Default: ``1.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_glitch_peak_tau_days``
  Type: ``float``
  Default: ``30.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_glitch_suppress_overlap``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_keep_variant_tmp``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_mark_only_worst_per_day``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_merge_tol_seconds``
  Type: ``float``
  Default: ``2.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_min_points``
  Type: ``int``
  Default: ``6``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_observatory_path``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_orbital_phase_cut``
  Type: ``Optional[float]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_orbital_phase_cut_center``
  Type: ``float``
  Default: ``0.25``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_orbital_phase_cut_enabled``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_orbital_phase_cut_min_points``
  Type: ``int``
  Default: ``20``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_orbital_phase_cut_nbins``
  Type: ``int``
  Default: ``18``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_orbital_phase_cut_sigma``
  Type: ``float``
  Default: ``3.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_outlier_gate_enabled``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_outlier_gate_resid_col``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_outlier_gate_sigma``
  Type: ``float``
  Default: ``3.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_outlier_gate_sigma_col``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_robust_enabled``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_robust_scope``
  Type: ``str``
  Default: ``both``
  Modes: ``pipeline, workflow``
  Level: ``full``
  Choices: ``global, backend, both``

``pqc_robust_z_thresh``
  Type: ``float``
  Default: ``5.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_run_variants``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_solar_approach_max_deg``
  Type: ``float``
  Default: ``30.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_solar_events_enabled``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_solar_freq_alpha_max``
  Type: ``float``
  Default: ``4.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_solar_freq_alpha_max_iter``
  Type: ``int``
  Default: ``64``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_solar_freq_alpha_min``
  Type: ``float``
  Default: ``0.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_solar_freq_alpha_tol``
  Type: ``float``
  Default: ``0.001``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_solar_freq_dependence``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_solar_member_eta``
  Type: ``float``
  Default: ``1.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_solar_min_points_global``
  Type: ``int``
  Default: ``30``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_solar_min_points_near_zero``
  Type: ``int``
  Default: ``3``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_solar_min_points_year``
  Type: ``int``
  Default: ``10``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_solar_tau_max_deg``
  Type: ``float``
  Default: ``60.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_solar_tau_min_deg``
  Type: ``float``
  Default: ``2.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_step_delta_chi2_thresh``
  Type: ``float``
  Default: ``25.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_step_enabled``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_step_min_points``
  Type: ``int``
  Default: ``20``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_step_scope``
  Type: ``str``
  Default: ``both``
  Modes: ``pipeline, workflow``
  Level: ``full``
  Choices: ``global, backend, both``

``pqc_structure_circular_features``
  Type: ``Optional[List[str]]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_structure_detrend_features``
  Type: ``Optional[List[str]]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_structure_group_cols``
  Type: ``Optional[List[str]]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_structure_min_per_bin``
  Type: ``int``
  Default: ``3``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_structure_mode``
  Type: ``str``
  Default: ``none``
  Modes: ``pipeline, workflow``
  Level: ``full``
  Choices: ``none, detrend, test, both``

``pqc_structure_nbins``
  Type: ``int``
  Default: ``12``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_structure_p_thresh``
  Type: ``float``
  Default: ``0.01``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_structure_test_features``
  Type: ``Optional[List[str]]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_tau_corr_minutes``
  Type: ``float``
  Default: ``30.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_tau_rec_days``
  Type: ``float``
  Default: ``7.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``pqc_window_mult``
  Type: ``float``
  Default: ``5.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

data
----

``pulsars``
  Type: ``PulsarSelection``
  Default: ``ALL``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

policy.pqc
----------

``qc_cross_pulsar_dir``
  Type: ``Optional[Path]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``qc_cross_pulsar_enabled``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``full``

``qc_cross_pulsar_event_cols``
  Type: ``Optional[List[str]]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``qc_cross_pulsar_include_events``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``qc_cross_pulsar_include_outliers``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``qc_cross_pulsar_min_pulsars``
  Type: ``int``
  Default: ``2``
  Modes: ``pipeline, workflow``
  Level: ``full``

``qc_cross_pulsar_outlier_cols``
  Type: ``Optional[List[str]]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``qc_cross_pulsar_time_col``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``qc_cross_pulsar_window_days``
  Type: ``float``
  Default: ``1.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

run
---

``qc_report``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

policy.report
-------------

``qc_report_backend``
  Type: ``Optional[str]``
  Default: None
  Modes: ``qc-report, pipeline, workflow``
  Level: ``minimal``

``qc_report_backend_col``
  Type: ``Optional[str]``
  Default: None
  Modes: ``qc-report, pipeline, workflow``
  Level: ``minimal``

``qc_report_compact_outlier_cols``
  Type: ``Optional[List[str]]``
  Default: None
  Modes: ``qc-report, pipeline, workflow``
  Level: ``minimal``

``qc_report_compact_pdf``
  Type: ``bool``
  Default: ``False``
  Modes: ``qc-report, pipeline, workflow``
  Level: ``minimal``

``qc_report_compact_pdf_name``
  Type: ``str``
  Default: ``qc_compact_report.pdf``
  Modes: ``qc-report, pipeline, workflow``
  Level: ``minimal``

``qc_report_dir``
  Type: ``Optional[Path]``
  Default: None
  Modes: ``qc-report, pipeline, workflow``
  Level: ``minimal``

``qc_report_no_feature_plots``
  Type: ``bool``
  Default: ``False``
  Modes: ``qc-report, pipeline, workflow``
  Level: ``minimal``

``qc_report_no_plots``
  Type: ``bool``
  Default: ``False``
  Modes: ``qc-report, pipeline, workflow``
  Level: ``minimal``

``qc_report_structure_group_cols``
  Type: ``Optional[str]``
  Default: None
  Modes: ``qc-report, pipeline, workflow``
  Level: ``minimal``

data
----

``reference_branch``
  Type: ``str``
  Default: ``main``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

policy.report
-------------

``report_dir``
  Type: ``Optional[Path]``
  Default: None
  Modes: ``qc-report``
  Level: ``balanced``

paths
-----

``results_dir``
  Type: ``Path``
  Default: ``.``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

policy.report
-------------

``run_dir``
  Type: ``Path``
  Default: None
  Modes: ``qc-report``
  Level: ``minimal``

run
---

``run_fix_dataset``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

``run_pqc``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

``run_tempo2``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

``run_whitenoise``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

paths
-----

``singularity_image``
  Type: ``Path``
  Default: ``/path/to/tempo2.sif``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

policy.report
-------------

``structure_group_cols``
  Type: ``Optional[str]``
  Default: None
  Modes: ``qc-report``
  Level: ``balanced``

pipeline
--------

``testing_mode``
  Type: ``bool``
  Default: ``False``
  Modes: ``pipeline, workflow``
  Level: ``minimal``

policy.whitenoise
-----------------

``whitenoise_epoch_tolerance_seconds``
  Type: ``float``
  Default: ``1.0``
  Modes: ``pipeline, workflow``
  Level: ``full``

``whitenoise_fit_timing_model_first``
  Type: ``bool``
  Default: ``True``
  Modes: ``pipeline, workflow``
  Level: ``full``

``whitenoise_single_toa_mode``
  Type: ``str``
  Default: ``combined``
  Modes: ``pipeline, workflow``
  Level: ``full``
  Choices: ``combined, equad0, ecorr0``

``whitenoise_source_path``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

``whitenoise_timfile_name``
  Type: ``Optional[str]``
  Default: None
  Modes: ``pipeline, workflow``
  Level: ``full``

workflow
--------

``workflow_file``
  Type: ``Path``
  Default: ``""``
  Modes: ``workflow``
  Level: ``minimal``

