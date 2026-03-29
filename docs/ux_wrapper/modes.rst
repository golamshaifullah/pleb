Mode Dispatch
=============

UX ``run.mode`` controls which existing PLEB mode is invoked.

Supported values
----------------

- ``pipeline``
- ``ingest``
- ``workflow``
- ``qc-report`` (also accepts ``qc`` / ``qc_report`` aliases)

Dispatch table
--------------

- ``pipeline`` -> legacy call ``pleb --config <tmp_legacy.toml>``
- ``ingest`` -> legacy call ``pleb ingest --config <tmp_legacy.toml>``
- ``workflow`` -> legacy call ``pleb workflow --config <tmp_legacy.toml>``
- ``qc-report`` -> legacy call ``pleb qc-report --config <tmp_legacy.toml>``

Notes
-----

- Wrapper compiles ``pleb.toml`` into a temporary legacy TOML file.
- Existing runtime semantics are preserved because execution remains in current
  mode code paths.

Examples
--------

Pipeline
~~~~~~~~

.. code-block:: toml

   [run]
   mode = "pipeline"
   run_tempo2 = true
   run_pqc = true

Ingest
~~~~~~

.. code-block:: toml

   [run]
   mode = "ingest"

   [policy.ingest]
   mapping_file = "configs/catalogs/ingest/ingest_mapping_epta_data.json"

Workflow
~~~~~~~~

.. code-block:: toml

   [run]
   mode = "workflow"

   [workflow]
   file = "configs/workflows/branch_chained_fix_pqc_variants.toml"

QC report
~~~~~~~~~

.. code-block:: toml

   [run]
   mode = "qc-report"

   [policy.report]
   run_dir = "results/wf_step2_pqc_balanced_detect"

Invalid mode behavior
---------------------

Unsupported mode values fail fast with a clear error listing accepted modes.
