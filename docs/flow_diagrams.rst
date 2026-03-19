Flow Diagrams
=============

This page gives human-readable flow diagrams for each major ``pleb`` mode and
for each way PQC is used.

Pipeline Mode
-------------

.. code-block:: text

   [Load config + --set overrides]
                |
                v
      [Resolve paths + branches + pulsar list]
                |
                v
      [Optional FixDataset stage]
                |
                v
      [Optional tempo2 stage]
                |
                v
      [Optional PQC stage]
                |
                v
      [Optional QC report stage]
                |
                v
          [Write run outputs]


Ingest Mode
-----------

.. code-block:: text

   [Read ingest mapping]
           |
           v
   [Discover source par/tim/tmplts]
           |
           v
   [Copy into canonical pulsar layout]
           |
           v
   [Optional verify + lockfile checks]
           |
           v
   [Optional commit to ingest branch]


Param-Scan Mode
---------------

.. code-block:: text

   [Load config + target pulsars]
           |
           v
   [Run baseline fit metrics]
           |
           v
   [Try candidate parameter/model variants]
           |
           v
   [Compare Δchi2 / z-style ranking]
           |
           v
   [Write scan report + tables]


QC-Report Mode
--------------

.. code-block:: text

   [Find existing *_qc.csv files]
           |
           v
   [Diagnostics + summary plots + feature plots]
           |
           v
   [Optional compact PDF (decision/action pages)]
           |
           v
   [Write qc_report directory]


Workflow Mode
-------------

.. code-block:: text

   [Load base config + workflow file]
           |
           v
   [Apply global set/overrides]
           |
           v
   [Run top-level steps/groups]
           |
           v
   [Run loops (optional): serial or parallel per loop/group]
           |
           v
   [Evaluate stop_if conditions between iterations]
           |
           v
   [Return final workflow context]


PQC Usage 1: Detection Only
---------------------------

.. code-block:: text

   [Pipeline: run_pqc=true]
           |
           v
   [Generate *_qc.csv + qc_summary.tsv]
           |
           v
   [No edits to tim/par from PQC flags]


PQC Usage 2: Detect Then Apply QC Comments/Deletes
--------------------------------------------------

.. code-block:: text

   Stage A:
     [Pipeline run with run_pqc=true] -> [Write *_qc.csv]

   Stage B:
     [FixDataset apply step]
          |
          v
     [Load QC CSV for each pulsar]
          |
          v
     [Match TOAs by MJD (+ backend/timfile where available)]
          |
          v
     [Comment/Delete matching TOAs per fix_qc_* settings]
          |
          v
     [Commit changes on fix branch]


PQC Usage 3: Standalone QC Report on Existing QC CSVs
-----------------------------------------------------

.. code-block:: text

   [pleb qc-report --run-dir ...]
           |
           v
   [Read existing *_qc.csv only]
           |
           v
   [Generate diagnostics + plots + optional compact PDF]


PQC Core Detection Sequence
---------------------------

The sequence below reflects how PQC is executed from ``pleb``.

.. code-block:: text

   [Parse tim metadata (INCLUDE recursion)]
                   |
                   v
   [Load libstempo timing arrays]
                   |
                   v
   [Merge timing arrays + tim metadata]
                   |
                   v
   [Ensure backend keys: sys/group]
                   |
                   v
   [Feature extraction]
     - orbital phase
     - solar elongation
     - optional elevation / airmass / parallactic angle
     - optional freq bins
                   |
                   v
   [Optional structure detrending/tests]
                   |
                   v
   [Outlier detectors]
     - bad measurement (OU/FDR)
     - robust MAD outliers (backend/global)
     - optional hard sigma gate
                   |
                   v
   [Event detectors]
     - transient windows
     - exp dips
     - step + DM-step
     - solar events
     - eclipse events
     - gaussian bumps
     - glitches
                   |
                   v
   [Event-aware reconciliation]
     - TOAs assigned to events are removed from bad_point
     - event_member is computed
     - outlier_any compatibility field is set
                   |
                   v
   [Write per-TOA QC flags to *_qc.csv]

