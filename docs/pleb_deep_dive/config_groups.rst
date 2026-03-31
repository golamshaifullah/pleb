Operational Config Groups
=========================

This chapter groups PLEB settings by operational concern.

Repository and dataset targeting
--------------------------------

Primary keys:

- ``home_dir``
- ``results_dir``
- ``dataset_name``
- ``branches``
- ``reference_branch``
- ``pulsars``

Stage toggles
-------------

Primary keys:

- ``run_fix_dataset``
- ``run_tempo2``
- ``run_pqc``
- ``run_whitenoise``
- ``qc_report``

FixDataset mutation controls
----------------------------

Primary keys:

- ``fix_apply``
- ``fix_base_branch`` / ``fix_branch_name`` / ``fix_commit_message``
- ``fix_infer_system_flags`` and overwrite controls
- ``fix_insert_missing_jumps`` / ``fix_prune_stale_jumps``
- ``fix_overlap_*`` / ``fix_relabel_*``
- ``fix_ensure_ephem`` / ``fix_ensure_clk`` / ``fix_ensure_ne_sw``

Variant generation controls
---------------------------

Primary keys:

- ``fix_generate_alltim_variants``
- ``fix_backend_classifications_path``
- ``fix_alltim_variants_path``
- jump-reference variant par options (if enabled in your profile)

System/group rule-table controls
--------------------------------

Primary keys:

- ``fix_system_flag_table_path`` (table used for ``-sys``/``-group`` inference)
- ``fix_flag_sys_freq_rules_enabled``
- ``fix_flag_sys_freq_rules_path`` (YAML rules file)

PQC forwarded controls
----------------------

Primary keys:

- ``pqc_*`` keys (detector config forwarded values)
- ``pqc_backend_profiles_path``

QC action policy controls
-------------------------

Primary keys:

- ``fix_qc_action``
- ``fix_qc_remove_outliers``
- ``fix_qc_outlier_cols``
- ``fix_qc_write_pqc_flag`` / ``fix_qc_pqc_flag_name``

Reporting controls
------------------

Primary keys:

- ``qc_report_*``
- ``qc_cross_pulsar_*``

Whitenoise and public-compare controls
--------------------------------------

Primary keys:

- ``run_whitenoise``
- ``whitenoise_*``
- ``compare_public_out_dir``
- ``compare_public_providers_path``

See also
--------

- Full per-key catalog: :doc:`../full_settings_catalog`
- Config layout map: :doc:`../config_layout`
