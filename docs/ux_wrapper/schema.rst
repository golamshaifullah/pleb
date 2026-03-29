UX Config Schema
================

The UX wrapper reads a structured TOML file with these top-level sections:

- ``[paths]``
- ``[data]``
- ``[run]``
- ``[policy]``
- ``[workflow]``
- ``[pipeline]`` (optional direct pass-through)

Section: ``[paths]``
--------------------

Typical keys:

- ``home_dir``
- ``dataset_name``
- ``results_dir``
- ``singularity_image``

These control where PLEB reads/writes and how tempo2 is executed.

Section: ``[data]``
-------------------

Typical keys:

- ``branches``
- ``reference_branch``
- ``pulsars``
- ``jobs``

These control scope and parallelism.

Section: ``[run]``
------------------

Typical keys:

- ``mode``
- ``run_tempo2``
- ``run_fix_dataset``
- ``run_pqc``
- ``qc_report``

``mode`` selects dispatch target (see :doc:`modes`).

Section: ``[policy]``
---------------------

Supports nested groups and scalar keys.

Common groups:

- ``[policy.fix]``
- ``[policy.pqc]``
- ``[policy.report]``
- ``[policy.ingest]``

Nested keys are mapped to legacy prefixes (see :doc:`mapping`).

Section: ``[workflow]``
-----------------------

Workflow-specific options. Example:

- ``file`` -> mapped to legacy ``workflow_file``.

Section: ``[pipeline]``
-----------------------

Optional direct pass-through for advanced users. Keys here are copied directly
into the legacy flat config dictionary.

Use this when a needed legacy key does not yet have a UX-friendly alias.

Complete example
----------------

.. code-block:: toml

   [paths]
   home_dir = "/work/git_projects/epta-dr3-in2p3"
   dataset_name = "EPTA-DR3/epta-dr3-data"
   results_dir = "results"
   singularity_image = "/work/git_projects/PSR_Singularity/psrpta.sif"

   [data]
   branches = ["main"]
   reference_branch = "main"
   pulsars = "ALL"
   jobs = 4

   [run]
   mode = "pipeline"
   run_tempo2 = true
   run_fix_dataset = true
   run_pqc = true
   qc_report = true

   [policy.fix]
   apply = true
   base_branch = "raw_ingest"
   branch_name = "fix_dataset_ux"
   commit_message = "FixDataset: UX run"
   qc_action = "comment"

   [policy.pqc]
   backend_col = "sys"
   merge_tol_seconds = 10.0

   [pipeline]
   fix_qc_outlier_cols = ["bad_point", "robust_outlier", "robust_global_outlier", "bad_mad"]

Validation behavior
-------------------

The wrapper itself performs lightweight validation.

For full behavior validation, always run:

.. code-block:: bash

   pleb doctor --config pleb.toml

and then a pilot run on small scope.
