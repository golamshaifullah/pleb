Overview and Ownership
======================

This chapter defines what PLEB is responsible for and how it integrates with
PQC.

PLEB owns
---------

- data ingest/discovery/copy/verification,
- branch-aware dataset update control,
- fix policies for tim/par edits,
- stage ordering and run layout (serial/parallel/workflow grouping),
- output file placement and reproducibility metadata.

PQC owns
--------

- detector-level feature and outlier decisions,
- row-level QC labels,
- detector diagnostics and model-specific statistics.

Operational model
-----------------

Think of PLEB as:

1. a stage coordinator,
2. a policy applier,
3. an output/reproducibility manager.

The practical boundary is:

- PLEB decides **what to run, where to read/write, and how to apply outputs**.
- PQC decides **which points/events are flagged by detector logic**.

Cross-reference
---------------

- PLEB config and layout: :doc:`../configuration`, :doc:`../config_layout`
- PQC detector internals: https://golamshaifullah.github.io/pqc/index.html
