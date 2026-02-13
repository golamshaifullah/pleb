Quickstart
==========

This page shows a minimal end-to-end run that produces residual plots and
diagnostic tables. It assumes you already have a tempo2-compatible data set
and a working tempo2 installation or container image.

Minimal configuration
---------------------

Create a ``pipeline.toml`` file:

.. code-block:: toml

   home_dir = "/data/epta"
   singularity_image = "/images/tempo2.sif"
   dataset_name = "EPTA"
   branches = ["main", "EPTA"]
   reference_branch = "main"
   results_dir = "results"
   jobs = 4

Run the pipeline
----------------

.. code-block:: bash

   pleb --config pipeline.toml

CLI overrides
-------------

Any key from the TOML config can be supplied directly as a CLI flag and will
override the file value. These are equivalent:

.. code-block:: bash

   pleb --config pipeline.toml --jobs 8 --run_fix_dataset
   pleb --config pipeline.toml --set jobs=8 --set run_fix_dataset=true

Boolean keys accept ``--key`` (true) and ``--no-key`` (false).

The pipeline prints a run tag pointing to the output directory. Inside that
directory you will find:

- Residual plots grouped by backend and pulsar.
- Change reports comparing the configured branches.
- Summary tables and QC artifacts (if enabled).

Optional QC stage
-----------------

If you have ``pqc`` and ``libstempo`` installed, you can enable outlier and
transient detection:

.. code-block:: bash

   pleb --config pipeline.toml --qc

The QC stage is designed to highlight residual outliers, transient events,
and structured behavior in residuals versus time, frequency, and orbital
phase. [Coles2011]_ [Keith2013]_

Next steps
----------

- For details on the CLI, see :doc:`cli`.
- For statistical context (chi-square, red noise, DM variations), see
  :doc:`concepts`.
