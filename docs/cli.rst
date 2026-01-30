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

QC report subcommand
--------------------

If you already ran QC, you can generate report plots from a run directory:

.. code-block:: bash

   pleb qc-report --run-dir results/run_2024-01-01

See :doc:`examples` for complete workflows.
