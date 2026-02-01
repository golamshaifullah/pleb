Command-Line Interface
======================

The package installs a ``pleb`` command that wraps the main pipeline. It
accepts a configuration file plus optional overrides. For a list of all
options, run:

.. code-block:: bash

   pleb --help

Core usage
----------

Run the pipeline from a config file:

.. code-block:: bash

   pleb --config pipeline.toml

Override config values in place:

.. code-block:: bash

   pleb --config pipeline.toml --set results_dir=\"results\" --set jobs=8

Common options
--------------

The most-used flags map directly to :class:`pleb.config.PipelineConfig`:

- ``--results-dir`` and ``--outdir-name`` for output organization.
- ``--force-rerun`` and ``--no-tempo2`` to control tempo2 execution.
- ``--no-change-reports`` and ``--testing`` to skip report generation.
- ``--jobs`` for parallel processing.

Quality-control (QC) flags
--------------------------

If you have optional QC dependencies installed, you can enable them with:

.. code-block:: bash

   pleb --config pipeline.toml --qc

QC options allow you to control feature extraction and structure tests, for
example:

.. code-block:: bash

   pleb --config pipeline.toml --qc \\
     --qc-backend-col group \\
     --qc-add-orbital-phase \\
     --qc-structure-mode both

Parameter scan
--------------

To compare candidate timing-model parameters, run the param-scan mode:

.. code-block:: bash

   pleb --config pipeline.toml --param-scan --scan F2 --scan \"JUMP -sys P200 0 1\"

The ``--scan-typical`` option applies a built-in profile that explores parallax,
binary derivatives, or DM derivatives based on the timing model and reduced
chi-square thresholds. [Edwards2006]_ [Hobbs2006]_

Param-check mode (fit-only scans)
---------------------------------

The ``--param-scan`` mode is the "param_check" workflow: it performs fit-only
scans of candidate parameters without running the full pipeline. It runs tempo2
on temporary ``.par`` variants and records how the fit statistics change.

Prerequisites:

- tempo2 available in your container/environment
- a clean Git repo (the scan checks out branches)
- ``home_dir``, ``dataset_name``, and ``results_dir`` set (via config or ``--set``)

Outputs:

- A standalone run directory under ``<results>/<outdir>_param_scan/``
- Per-pulsar scan tables in ``param_scan/``
- PLK logs under ``plk/``
- Scratch workspace under ``work/`` (removed if ``cleanup_work_dir=true``)

Interpreting outputs:

- The per-pulsar tables list each candidate and the change in fit metrics
  (e.g., reduced chi-square or log-likelihood proxies derived from PLK logs).
- Prefer candidates that improve fit quality without inflating uncertainties.
- If multiple candidates improve the fit similarly, prioritize those with
  physical motivation (e.g., DM derivatives for dispersive trends, BTX
  derivatives for binary timing) and check residual structure.

QC report subcommand
--------------------

If you already ran QC, you can generate report plots from a run directory:

.. code-block:: bash

   pleb qc-report --run-dir results/run_2024-01-01

Ingest subcommand
-----------------

Run mapping-driven ingest:

.. code-block:: bash

   pleb ingest --mapping configs/settings/system_flag_mapping.example.json \
     --output-dir /data/pulsars

Workflow subcommand
-------------------

Run a workflow file (TOML or JSON) with steps and loops:

.. code-block:: bash

   pleb workflow --file configs/workflows/example_iterative.toml

See :doc:`running_modes` for detailed compatibility notes and workflow
examples, and :doc:`examples` for end-to-end runs.
