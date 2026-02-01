Pleb Modes and Entry Points
===========================

This chapter lists the supported modes and how to invoke them. These are the
entry points that exist today in the CLI and pipeline code. For each mode, we
also describe the outputs it produces so users know where to look for results.

Summary: modes and permutations
-------------------------------

Supported modes / entry points:

1. Pipeline run (default)
2. QC report (post-processing only)
3. Param scan (param_check)
4. Ingest
5. Workflow (multi-step orchestration)

Allowed permutations:

- **Single-shot pipeline**: any combination of pipeline toggles in one run
  (FixDataset report-only, fix_apply, PQC, QC report, binary analysis, plots).
- **Separate modes**: ``--param-scan``, ``qc-report``, and ``ingest`` are
  standalone and do not run the full pipeline in the same invocation.
- **Workflows**: any sequence or loop over ``ingest``, ``pipeline``,
  ``fix_apply``, ``param_scan``, ``qc_report``.

1. Pipeline run (default)
-------------------------

Runs tempo2 fits, plots, reports, optional FixDataset, optional PQC, and an
optional QC report (when requested).

Typical outputs (under ``<results>/<outdir>/``):

- ``tempo2/`` (tempo2 outputs and residuals)
- ``plots/`` (residual plots and diagnostic figures)
- ``reports/`` (summary tables and change reports)
- ``fix_dataset/`` (FixDataset summaries, if enabled)
- ``qc/`` (PQC CSVs and QC tables, if enabled)
- ``qc_report/`` (QC report plots and HTML, if enabled)

Trigger:

.. code-block:: bash

   pleb --config configs/settings/test_all_steps.toml

Key config flags:

- ``run_tempo2`` (enable/disable tempo2 fits)
- ``make_change_reports`` (change reports)
- ``make_binary_analysis`` (binary/orbital analysis table)
- ``run_fix_dataset`` (FixDataset report stage)
- ``run_pqc`` (PQC outlier detection)
- ``qc_report`` (generate QC report after PQC)
- ``fix_apply`` (apply FixDataset changes on a new branch)


2. QC report mode
-----------------

Generates QC report artifacts from existing QC CSVs (no re-fit).
Outputs land under ``<run_dir>/qc_report``.

Typical outputs:

- ``<run_dir>/qc_report/`` (summary plots, per-backend plots, feature plots)

Trigger (CLI subcommand):

.. code-block:: bash

   pleb qc-report --run-dir results/run_2024-01-01

Notes:

- This does not re-run PQC or tempo2; it only reads existing QC outputs.


3. Param scan mode
------------------

Fit-only scan of parameter candidates (e.g., DM/BTX/etc).
Outputs under ``<results>/<outdir>_param_scan/``.

Typical outputs:

- ``<results>/<outdir>_param_scan/`` (scan outputs, tables, and logs)

Trigger:

.. code-block:: bash

   pleb --config configs/settings/test_all_steps.toml --param-scan --scan-typical

Notes:

- Param-scan is a separate execution path and does not run the full pipeline
  in the same invocation.

Interpretation notes:

- Param scan compares fit-only candidates; it does not modify the dataset.
- Prefer candidates that improve fit quality without destabilizing other terms.


4. Ingest mode
--------------

Mapping-driven ingest of ``.par``/``.tim``/``.tmplts`` into a canonical layout.

Typical outputs:

- ``<output-dir>/<Jpulsar>/`` (canonical pulsar folders)
- ``<output-dir>/<Jpulsar>/tims/`` (renamed tim files)
- ``<output-dir>/<Jpulsar>/tmplts/`` (templates retained with original names)

Trigger:

.. code-block:: bash

   pleb ingest --mapping configs/settings/system_flag_mapping.example.json \
     --output-dir /data/pulsars

Notes:

- Ingest is standalone and does not run the pipeline.


5. FixDataset apply mode
------------------------

Not a separate CLI mode, but when ``fix_apply=true`` it creates a new branch and
writes changes.

Typical outputs:

- Git branch with the applied fixes
- Updated ``.par``/``.tim`` files in the dataset tree

Trigger:

.. code-block:: bash

   pleb --config configs/settings/test_all_steps.toml --set fix_apply=true

Workflow permutations
---------------------

Because workflows can chain steps, any of the following sequences are valid:

- ``pipeline → qc_report``
- ``pipeline → fix_apply → pipeline → qc_report``
- ``param_scan → qc_report``
- ``ingest → pipeline → qc_report``
- ``loop(pipeline → qc_report → fix_apply)`` with stop conditions

Use workflows when you need iterative schemes or repeated QC/fix cycles.
