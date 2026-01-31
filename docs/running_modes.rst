Running Pleb
============

This chapter explains the three supported ways to run Pleb:

1. Directly via the command-line interface (CLI).
2. Via a settings file (TOML/JSON config).
3. Via a workflow file (TOML/JSON) that orchestrates multiple steps.

It also lists additional entry points (Python API) and how the modes
interoperate.

Overview of modes
-----------------

CLI (no workflow) is the simplest path: you pass a settings file and optionally
override values inline. Workflows build on settings by adding a sequence of
steps and optional loops.

Supported entry points today:

- ``pleb`` (main CLI, runs the pipeline by default)
- ``pleb qc-report`` (generate QC plots from an existing run directory)
- ``pleb ingest`` (mapping-driven ingest mode)
- ``pleb workflow`` (multi-step sequences and loops)
- Python API (``pleb.pipeline.run_pipeline``, ``pleb.param_scan.run_param_scan``,
  ``pleb.qc_report.generate_qc_report``, ``pleb.ingest.ingest_dataset``)

If you want a new entry point (for example, a minimal QC-only CLI), add a
workflow step or wrap the Python API in your own script.


Mode 1: Direct CLI (no settings file)
-------------------------------------

You can run the CLI without a settings file by supplying the core parameters
via flags. This is best for quick, one-off runs.

Minimal run (CLI-only):

.. code-block:: bash

   pleb \
     --results-dir /work/git_projects/data-eptadr3/results \
     --outdir-name run_cli_only \
     --set home_dir="/work/git_projects/data-eptadr3" \
     --set dataset_name="DR3full" \
     --set pulsars='["J1713+0747","J1909-3744"]' \
     --set branches='["main"]'

Enable QC and change options:

.. code-block:: bash

   pleb \
     --results-dir /work/git_projects/data-eptadr3/results \
     --outdir-name run_cli_qc \
     --set home_dir="/work/git_projects/data-eptadr3" \
     --set dataset_name="DR3full" \
     --set pulsars='["J1713+0747"]' \
     --set branches='["main"]' \
     --qc \
     --qc-structure-mode both \
     --qc-outlier-gate

Parameter scan (CLI-only):

.. code-block:: bash

   pleb \
     --results-dir /work/git_projects/data-eptadr3/results \
     --outdir-name run_cli_scan \
     --set home_dir="/work/git_projects/data-eptadr3" \
     --set dataset_name="DR3full" \
     --set pulsars='["J1713+0747"]' \
     --set branches='["main"]' \
     --param-scan --scan-typical

Generate a QC report (CLI-only):

.. code-block:: bash

   pleb qc-report --run-dir results/run_2024-01-01

Run mapping-driven ingest (CLI-only):

.. code-block:: bash

   pleb ingest --mapping configs/settings/system_flag_mapping.example.json \
     --output-dir /data/pulsars

Compatibility notes (CLI-only):

- ``--param-scan`` runs only the parameter scan; it does not run the full
  pipeline in the same invocation.
- ``pleb qc-report`` does not run QC; it only reads existing QC CSVs and
  generates plots.
- ``pleb ingest`` is a standalone mode and does not run the pipeline.


Mode 2: CLI with a settings file
--------------------------------

Settings files (TOML or JSON) are the primary way to configure runs. They map
directly to :class:`pleb.config.PipelineConfig` and can be overridden with
``--set`` on the CLI or by workflow steps.

Typical settings file (minimal):

.. code-block:: toml

   # configs/settings/minimal.toml
   dataset_name = "DR3full"
   home_dir = "/work/git_projects/data-eptadr3"
   results_dir = "/work/git_projects/data-eptadr3/results"
   outdir_name = "run_minimal"

   pulsars = ["J1713+0747", "J1909-3744"]
   branches = ["main"]

   run_fix_dataset = true
   run_tempo2 = true
   run_pqc = false

Settings with QC and reporting:

.. code-block:: toml

   # configs/settings/qc.toml
   dataset_name = "DR3full"
   home_dir = "/work/git_projects/data-eptadr3"
   results_dir = "/work/git_projects/data-eptadr3/results"
   outdir_name = "run_qc"

   pulsars = ["J1713+0747"]
   branches = ["main"]

   run_fix_dataset = true
   run_tempo2 = true
   run_pqc = true
   pqc_backend_col = "group"
   pqc_outlier_gate_enabled = true
   pqc_outlier_gate_sigma = 6.0

Use with the CLI:

.. code-block:: bash

   pleb --config configs/settings/qc.toml

Override in place:

.. code-block:: bash

   pleb --config configs/settings/qc.toml \
     --set outdir_name="run_qc_debug" \
     --set pqc_outlier_gate_sigma=4.5

You can also take any CLI-only example from Mode 1 and move those values into
the settings file, then keep only the high-level CLI switches (for example
``--qc`` or ``--param-scan``).


Mode 3: Workflow files
----------------------

Workflows coordinate multiple steps and optional loops. This enables
iterative schemes (for example: run pipeline, apply fixes, re-run QC,
repeat until stable).

Workflow file structure:

.. code-block:: toml

   # configs/workflows/example_iterative.toml
   config = "configs/settings/test_all_steps.toml"

   [[loops]]
   name = "get_jumps"
   max_iters = 5
   steps = ["pipeline", "qc_report", "fix_apply"]
   stop_if = [{ no_changes = true }, { qc_ok = true }]

   [[loops]]
   name = "check_params"
   max_iters = 1
   steps = ["param_scan", "qc_report", "fix_apply"]
   stop_if = [{ no_changes = true }, { qc_ok = true }]

Run it:

.. code-block:: bash

   pleb workflow --file configs/workflows/example_iterative.toml

Equivalent JSON workflow:

.. code-block:: json

   {
     "config": "configs/settings/test_all_steps.toml",
     "loops": [
       {
         "name": "get_jumps",
         "max_iters": 3,
         "steps": ["pipeline", "qc_report", "fix_apply"],
         "stop_if": [{"no_changes": true}, {"qc_ok": true}]
       }
     ],
     "steps": ["qc_report"]
   }

Top-level steps (no loop):

.. code-block:: toml

   config = "configs/settings/test_all_steps.toml"

   steps = ["pipeline", "qc_report"]

Per-step overrides:

.. code-block:: toml

   config = "configs/settings/test_all_steps.toml"

   [[steps]]
   name = "pipeline"
   set = ["outdir_name=\"run_a\""]

   [[steps]]
   name = "qc_report"
   overrides = { qc_report_backend = "EFF" }

Supported workflow step names:

- ``ingest`` (mapping-driven ingest)
- ``pipeline`` (full pipeline run)
- ``fix_apply`` (pipeline run with fix-apply enabled)
- ``param_scan`` (parameter scan)
- ``qc_report`` (generate QC plots from latest pipeline output or ``run_dir``)

Stop conditions:

- ``no_changes``: stop when FixDataset reports zero changes. This uses the
  latest ``fix_dataset_summary.tsv`` from the run.
- ``qc_ok``: stop when QC summary has zero flagged counts. This uses the
  latest ``qc_summary.tsv``.

Stop conditions can be declared as a list of strings:

.. code-block:: toml

   stop_if = ["no_changes", "qc_ok"]

Or as a list of dictionaries:

.. code-block:: toml

   stop_if = [{ no_changes = true }, { qc_ok = true }]

Compatibility notes (workflows):

- ``qc_report`` requires a prior ``pipeline`` or explicit ``run_dir`` in the
  step definition.
- ``param_scan`` is independent; it does not run the full pipeline.
- ``fix_apply`` runs the pipeline with fix-apply enabled and can be placed
  anywhere in a loop.


Other ways to run Pleb (Python API)
----------------------------------

For scripting or integration into larger systems, use the Python API:

.. code-block:: python

   from pleb.config import PipelineConfig
   from pleb.pipeline import run_pipeline
   from pleb.param_scan import run_param_scan
   from pleb.qc_report import generate_qc_report
   from pleb.ingest import ingest_dataset

   cfg = PipelineConfig.load("configs/settings/test_all_steps.toml")
   out = run_pipeline(cfg)

   run_param_scan(cfg, scan_typical=True)
   generate_qc_report(run_dir=out["tag"])
   ingest_dataset(mapping_file, output_root)

This is the main additional way to run Pleb beyond the CLI, settings files,
and workflows.
