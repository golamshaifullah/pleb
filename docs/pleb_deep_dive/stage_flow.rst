Stage-by-Stage Flow
===================

This chapter describes the execution sequence and the expected inputs/outputs of each stage.

Stage 0: Input resolution
-------------------------

PLEB resolves:

- dataset root and result root,
- pulsar scope,
- branch/base branch controls,
- runtime image/binary controls.

Common failures here are path and branch mismatches.

Stage 1: Ingest (optional)
--------------------------

Ingest discovers and copies source files according to mapping rules, then
verifies expected copy sets.

Key outputs:

- pulsar tree under ingest output root,
- ingest reports,
- optional ingest commit branch.

Stage 2: FixDataset (optional)
------------------------------

FixDataset applies policy-driven tim/par edits.

Typical actions:

- infer/overwrite ``-sys`` and ``-group``,
- insert/prune JUMPs,
- apply overlap/relabel rules,
- enforce par defaults,
- generate variants,
- optional TOA QC labeling via ``-pqc`` flag,
- comment-line normalization (comments written as ``C ...`` with leading
  whitespace removed).

Stage 3: tempo2 products (optional)
------------------------------------

When enabled, tempo2 products are generated and stored under run results.

Stage 4: PQC detection (optional)
---------------------------------

PLEB runs PQC over selected pulsars/branches with configured forwarded
settings and stores QC outputs in run results.

Stage 5: Post-QC apply (optional)
---------------------------------

PLEB can apply comments/deletions based on selected QC columns and policy
controls.

Stage 6: Reporting and coincidence (optional)
---------------------------------------------

PLEB generates QC summaries, optional compact PDF, and optional cross-pulsar
coincidence report.

Stage 7: Whitenoise (optional)
------------------------------

PLEB can run optional whitenoise noise estimation on cleaned products. This is
typically done after QC and apply stages.

Stage 8: Public-release comparison (optional)
---------------------------------------------

PLEB can compare local par values with public releases (EPTA/NANOGrav/IPTA
provider catalog).

Execution layout
------------------

PLEB supports:

- serial runs,
- parallel runs (per stage where supported),
- grouped serial/parallel workflow stages.
- serial/parallel groups with barriers between groups.

Use workflows when you need stage barriers (e.g. detect first, apply later).

Workflow contract note
----------------------

Workflow files are versioned. Current supported contract is
``workflow_version = 1``.
