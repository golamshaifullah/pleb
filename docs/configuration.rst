Configuration Guide
===================

This guide helps you navigate configuration without reading all source code.

Start here
----------

1. Use :doc:`quickstart` to get a first successful run.
2. Use :doc:`configuration_reference` to tune behavior by topic.
3. Use :doc:`full_settings_catalog` when you need exact key-by-key detail.

Configuration layers
--------------------

Most users work with three layers:

- **Pipeline behavior**: what stages run (`run_tempo2`, `run_pqc`, `run_fix_dataset`, `qc_report`).
- **Detection behavior**: PQC detector thresholds and feature toggles (`pqc_*`).
- **Action behavior**: what to do with flagged TOAs (`fix_qc_*`, compact report settings).

Detection strategy checklist
----------------------------

Use this when you want controlled, reproducible outlier handling:

1. Set PQC thresholds and enabled detectors (`pqc_*`).
2. If needed, set per-backend profile overrides (`pqc_backend_profiles_path`).
3. Define explicit action columns (`fix_qc_outlier_cols`).
4. Keep actions non-destructive first (`fix_qc_action = "comment"`).
5. Use compact report columns to match your action policy (`qc_report_compact_outlier_cols`).

Minimal practical template
--------------------------

.. code-block:: toml

   run_tempo2 = true
   run_pqc = true
   run_fix_dataset = true
   fix_apply = true
   qc_report = true
   qc_report_compact_pdf = true

   pqc_backend_col = "sys"
   pqc_backend_profiles_path = "configs/rules/pqc/backend_profiles.example.toml"

   fix_qc_remove_outliers = true
   fix_qc_action = "comment"
   fix_qc_outlier_cols = ["bad_point", "robust_outlier", "robust_global_outlier", "bad_mad"]

   qc_report_compact_outlier_cols = ["bad_point", "robust_outlier", "robust_global_outlier", "bad_mad"]

Finding any key quickly
-----------------------

- Human-friendly lookup: :doc:`full_settings_catalog`
- API-level definitions and defaults: :doc:`api` (``pleb.config.PipelineConfig``)

.. toctree::
   :maxdepth: 1

   configuration_reference
   full_settings_catalog
