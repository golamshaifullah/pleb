Examples
========

Compare two branches of a data set
----------------------------------

This example compares timing residuals between a reference branch and a
candidate branch after updating a timing model.

.. code-block:: toml

   home_dir = "/data/epta"
   singularity_image = "/images/tempo2.sif"
   dataset_name = "EPTA"
   branches = ["main", "candidate-update"]
   reference_branch = "main"
   results_dir = "results"
   jobs = 4

Run:

.. code-block:: bash

   pleb --config pipeline.toml

Interpretation:

- Residual plots and change reports highlight pulsars where the update shifts
  residual structure or changes reduced chi-square.
- Look for frequency-dependent residuals that might indicate unmodeled DM
  variations or scattering. [Keith2013]_ [Cordes2016]_

Param-scan for DM derivatives
-----------------------------

If a pulsar lacks a binary model and shows elevated reduced chi-square, use
the built-in scan profile to test DM derivatives:

.. code-block:: bash

   pleb --config pipeline.toml --param-scan --scan-typical --scan-dm-threshold 2.0

The scan evaluates candidates by :math:`\\Delta\\chi^2` and highlights models
that reduce residual structure. [Edwards2006]_ [Hobbs2006]_

QC-assisted outlier review
--------------------------

Enable QC to detect outliers and transient residual behavior:

.. code-block:: bash

   pleb --config pipeline.toml --qc

After the run, inspect QC plots grouped by backend. Backend-specific clusters
of outliers can indicate instrumental offsets or calibration issues.
[Manchester2005]_ [Coles2011]_

Generate a QC-only report from an existing run directory:

.. code-block:: bash

   pleb qc-report --run-dir results/run_2024-01-01 --backend-col group
