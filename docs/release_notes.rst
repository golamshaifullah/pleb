Release Notes
=============

Scope Freeze Notice (Current Release Line)
------------------------------------------

The current release line is scope-frozen for user-facing surface area.

Allowed in this line:

- bug fixes,
- stability/reproducibility improvements,
- tests/CI hardening,
- documentation improvements.

Planned next-major initiative (not in current scope):

- N-pass compiler DSL.

See also:

- ``ROADMAP.md``
- ``CONTRIBUTING.md``

Since ``release/v0.1.0``
------------------------

This section summarizes notable user-facing improvements after ``v0.1.0``.

TOML-based detection strategy configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Outlier/event decision behavior can now be configured in TOML instead of being
effectively hardcoded:

- ``pqc_backend_profiles_path``: per-backend PQC threshold/profile overrides.
- ``fix_qc_outlier_cols``: choose exactly which QC columns drive FixDataset
  outlier actions.
- ``qc_report_compact_outlier_cols``: choose exactly which QC columns define
  compact PDF/report outlier decisions.

Example:

.. code-block:: toml

   run_pqc = true
   run_fix_dataset = true
   fix_qc_remove_outliers = true

   # Per-backend PQC profile overrides
   pqc_backend_profiles_path = "configs/rules/pqc/backend_profiles.example.toml"

   # FixDataset outlier action strategy
   fix_qc_outlier_cols = [
     "bad_point",
     "robust_outlier",
     "robust_global_outlier",
     "bad_mad",
   ]

   # Compact report strategy
   qc_report = true
   qc_report_compact_pdf = true
   qc_report_compact_outlier_cols = [
     "bad_point",
     "robust_outlier",
     "robust_global_outlier",
     "bad_mad",
   ]

Workflow orchestration upgrades
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Workflow files support serial/parallel execution control.
- You can run groups in parallel and enforce barriers between groups.

Ingest reproducibility and source control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Ingest lockfiles are generated and can be validated in strict mode
  (fail-fast when source trees change).
- Source-priority mapping is supported to resolve clashes across DR1/DR2/DR3
  style roots deterministically.

Declarative data-fix rules
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Declarative relabel rules (TOML) for targeted ``-sys/-group/-pta`` rewrites.
- Declarative overlap rules (TOML) for duplicate handling policies.
- Exact-overlap keep/drop catalog moved into editable TOML tables.

Mode and config usability
~~~~~~~~~~~~~~~~~~~~~~~~~

- CLI and TOML settings are 1:1 compatible; CLI overrides TOML when both are
  supplied.
- Mode-specific config models are available (pipeline, ingest, workflow,
  qc-report, param-scan).
