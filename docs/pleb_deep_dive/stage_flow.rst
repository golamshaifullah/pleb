Stage-by-Stage Flow
===================

This chapter describes the execution sequence and stage contracts.

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
- generate variants.

Stage 3: tempo2 products (optional)
------------------------------------

When enabled, tempo2 products are generated and stored under run results.

Stage 4: PQC detection (optional)
---------------------------------

PLEB runs PQC over selected pulsars/branches with configured pass-through
settings and stores QC outputs in run results.

Stage 5: Post-QC apply (optional)
---------------------------------

PLEB can apply comments/deletions based on selected QC columns and policy
controls.

Stage 6: Reporting (optional)
-----------------------------

PLEB generates QC summaries, optional compact PDF, and optional cross-pulsar
coincidence report.

Execution topology
------------------

PLEB supports:

- serial runs,
- parallel runs (per stage where supported),
- grouped serial/parallel workflow stages.

Use workflows when you need stage barriers (e.g. detect first, apply later).
