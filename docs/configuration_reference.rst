Configuration Reference
=======================

This page explains how to use the TOML configuration effectively, especially
for PQC detection strategy control.

If you only read one page before tuning detection behavior, read this one.

Configuration model and precedence
----------------------------------

``pleb`` uses a flat config model (``PipelineConfig``) with mode-specific
subcommands for ingest, workflow, and QC report.

Precedence is:

1. config file values (TOML/JSON),
2. CLI ``--set key=value`` overrides,
3. direct CLI flags (where available).

When both TOML and CLI are supplied, CLI wins.

Where to find the full key list
-------------------------------

There are many handles, so use these discovery paths:

- Human summary: this page + :doc:`quickstart`
- Full exhaustive key-by-key chapter (with examples): :doc:`full_settings_catalog`
- Full field list and defaults: :doc:`api` (``pleb.config.PipelineConfig``)
- Programmatic dump of all keys/defaults:

.. code-block:: bash

   python - <<'PY'
   from dataclasses import fields
   from pleb.config import PipelineConfig
   cfg = PipelineConfig(
       home_dir='.',
       singularity_image='tempo2.sif',
       dataset_name='dataset'
   )
   for f in fields(PipelineConfig):
       print(f"{f.name} = {getattr(cfg, f.name)!r}")
   PY

Detection strategy: mental model
--------------------------------

There are three independent strategy layers:

1. **PQC detection layer** (`pqc_*`): what detectors run and with which thresholds.
2. **FixDataset action layer** (`fix_qc_*`): which QC columns should trigger
   comment/delete actions in tim files.
3. **QC report compact layer** (`qc_report_compact_*`): which columns define
   compact PDF triage decisions.

Most confusion comes from mixing these layers. Keep them explicit.

TOML strategy controls (most important keys)
--------------------------------------------

PQC detector configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``run_pqc``: enable PQC stage.
- ``pqc_backend_col``: backend grouping key (`sys`, `group`, etc.).
- ``pqc_backend_profiles_path``: per-backend threshold overrides (TOML map).
- ``pqc_fdr_q`` / ``pqc_tau_corr_minutes``: bad-measurement detector behavior.
- ``pqc_robust_enabled`` / ``pqc_robust_z_thresh`` / ``pqc_robust_scope``:
  MAD-style robust outlier behavior.
- ``pqc_step_*`` / ``pqc_dm_step_*``: step and DM-step event sensitivity.
- ``pqc_solar_*`` / ``pqc_orbital_phase_*`` / ``pqc_eclipse_*``:
  event-domain detectors.
- ``pqc_gaussian_bump_*`` / ``pqc_glitch_*``: transient feature detectors.

FixDataset outlier action strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``run_fix_dataset`` + ``fix_apply``: enable/apply fix stage.
- ``fix_qc_remove_outliers``: turn on QC-driven action.
- ``fix_qc_outlier_cols``: explicit outlier columns to act on.
- ``fix_qc_action``: ``comment`` or ``delete`` (default is non-destructive comment).
- ``fix_qc_remove_bad`` / ``fix_qc_remove_transients`` /
  ``fix_qc_remove_solar`` / ``fix_qc_remove_orbital_phase``:
  selective action toggles by detector family.

Compact report strategy
~~~~~~~~~~~~~~~~~~~~~~~

- ``qc_report``: generate report stage.
- ``qc_report_compact_pdf``: write compact PDF.
- ``qc_report_compact_outlier_cols``: explicit compact outlier strategy.

Per-backend PQC profile TOML
----------------------------

``pqc_backend_profiles_path`` points to a TOML file with this shape:

.. code-block:: toml

   [backend_profiles]
   "LOFAR.*" = { robust_z_thresh = 6.0, fdr_q = 0.02 }
   "WSRT.P2.334" = { delta_chi2_thresh = 18.0, robust_z_thresh = 5.5 }

Match priority is:

1. exact backend match,
2. glob match (``fnmatch``).

Practical strategy profiles
---------------------------

Conservative (few false positives)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   run_pqc = true
   pqc_fdr_q = 0.005
   pqc_robust_z_thresh = 6.5
   pqc_delta_chi2_thresh = 25.0
   fix_qc_remove_outliers = true
   fix_qc_outlier_cols = ["bad_point", "robust_outlier", "robust_global_outlier", "bad_mad"]
   fix_qc_action = "comment"

Balanced (recommended default tuning workflow)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   run_pqc = true
   pqc_fdr_q = 0.01
   pqc_robust_z_thresh = 5.5
   pqc_step_enabled = true
   pqc_dm_step_enabled = true
   pqc_solar_events_enabled = true
   pqc_eclipse_events_enabled = true
   pqc_gaussian_bump_enabled = true
   pqc_glitch_enabled = true

   run_fix_dataset = true
   fix_apply = true
   fix_qc_remove_outliers = true
   fix_qc_outlier_cols = ["bad_point", "robust_outlier", "robust_global_outlier", "bad_mad"]
   fix_qc_action = "comment"

Aggressive (maximum sensitivity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   run_pqc = true
   pqc_fdr_q = 0.02
   pqc_robust_z_thresh = 4.5
   pqc_delta_chi2_thresh = 12.0
   pqc_step_delta_chi2_thresh = 12.0
   pqc_dm_step_delta_chi2_thresh = 12.0
   fix_qc_remove_outliers = true
   fix_qc_outlier_cols = ["bad_point", "robust_outlier", "robust_global_outlier", "bad_mad", "bad_hard"]
   fix_qc_action = "comment"

Common pitfalls
---------------

- Setting ``run_pqc=true`` alone does not modify tim files. You also need
  FixDataset action keys.
- Using ``outlier_any`` for action policy can be too broad because it may
  include event-related flags depending on context. Prefer explicit columns.
- Forgetting ``fix_qc_results_dir``/``fix_qc_branch`` can make FixDataset look
  like it ignored QC (it may be reading from the wrong run location).
- Extremely low thresholds + aggressive action can over-comment structured
  groups. Tune with report-only first.

Recommended workflow for users
------------------------------

1. Start with balanced profile and ``fix_qc_action="comment"``.
2. Inspect compact PDF + per-backend action CSVs.
3. Add per-backend overrides for known problematic systems.
4. Lock strategy in TOML and rerun reproducibly.
