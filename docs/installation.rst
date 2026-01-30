Installation
============

Python requirements
-------------------

The core package supports Python 3.10+ and depends on NumPy, Pandas, and SciPy.
Install from the repository root:

.. code-block:: bash

   pip install -e .

Optional extras
---------------

Several features depend on optional packages:

- ``libstempo`` for reading tempo2 residuals from Python.
- ``pqc`` for outlier and transient QC.
- ``matplotlib`` for plotting.

Install optional extras as needed:

.. code-block:: bash

   pip install -e .[qc,plot]

tempo2 and timing files
-----------------------

The pipeline expects standard tempo2-compatible ``.par`` and ``.tim`` files,
often stored within a pulsar directory tree. Many workflows execute tempo2
via a container image; the pipeline exposes this as ``singularity_image`` in
the configuration. See :doc:`quickstart` for a minimal configuration file and
run example. [Edwards2006]_ [Hobbs2006]_

If you are unfamiliar with pulsar timing packages and terminology, the
Handbook of Pulsar Astronomy is a practical starting point. [Lorimer2005]_
