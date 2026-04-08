Quick Reference
===============

This page collects the main commands and tracked example config files used by
the single-pulsar manual.


Tracked Example Files
---------------------

The repository now includes example files that mirror the playbook:

- ``configs/catalogs/ingest/single_pulsar_mapping.example.json``
- ``configs/runs/ingest/single_pulsar_ingest.example.toml``
- ``configs/runs/fixdataset/single_pulsar_step1_fix.example.toml``
- ``configs/runs/pqc/single_pulsar_pqc_detect.example.toml``
- ``configs/runs/fixdataset/single_pulsar_pqc_apply.example.toml``
- ``configs/rules/pqc/single_pulsar_backend_profiles.example.toml``
- ``configs/workflows/single_pulsar_3pass.example.toml``

These files use placeholder paths and names. Copy them to project-specific
files and replace the placeholders before running.


Main Commands
-------------

Ingest:

.. code-block:: bash

   pleb ingest --config configs/runs/ingest/single_pulsar_ingest.example.toml

Step 1 FixDataset:

.. code-block:: bash

   pleb --config configs/runs/fixdataset/single_pulsar_step1_fix.example.toml

Step 2 PQC detect:

.. code-block:: bash

   pleb --config configs/runs/pqc/single_pulsar_pqc_detect.example.toml

Step 3 QC apply:

.. code-block:: bash

   pleb --config configs/runs/fixdataset/single_pulsar_pqc_apply.example.toml

Three-pass workflow:

.. code-block:: bash

   pleb workflow --file configs/workflows/single_pulsar_3pass.example.toml


Useful Inspection Commands
--------------------------

Inspect the current git state:

.. code-block:: bash

   git status

Inspect branch differences after Step 1:

.. code-block:: bash

   git diff raw_ingest..step1_fix_flags_variants -- J1909-3744

Inspect branch differences after Step 3:

.. code-block:: bash

   git diff step2_pqc_balanced_detect..step3_apply_qc_comments -- J1909-3744

Find a pulsar directory inside the canonical dataset:

.. code-block:: bash

   rg --files /data/canonical/EPTA-DR3/epta-dr3-data | rg 'J1909-3744'

Inspect the latest resolved config inside a run directory:

.. code-block:: bash

   sed -n '1,240p' results/j1909_pqc_detect/run_settings/pipeline_config.resolved.toml


Checklist By Stage
------------------

Before ingest:

- confirm the raw source roots,
- confirm the canonical backend names,
- confirm pulsar aliases,
- confirm the intended ingest branch names.

Before Step 1:

- confirm ``fix_base_branch`` points to the ingest branch,
- confirm the pulsar exists in the canonical dataset,
- confirm the variant catalogs point to readable files,
- confirm the jump grouping flag is the intended one.

Before Step 2:

- confirm Step 1 completed and wrote the expected branch,
- confirm tempo2/container paths are valid,
- confirm ``pqc_backend_col`` is the intended grouping column,
- confirm QC is enabled and action is still disabled.

Before Step 3:

- confirm the Step-2 QC outputs exist,
- confirm ``fix_qc_results_dir`` points to the Step-2 run directory,
- confirm ``fix_qc_branch`` matches the Step-2 branch,
- confirm the action policy is still comment-first unless there is a clear
  reason otherwise.


Related Documentation
---------------------

- workflow details: :doc:`workflow`
- troubleshooting: :doc:`troubleshooting`
- full key catalog: :doc:`../full_settings_catalog`
