Ingest Deep Dive
================

This chapter focuses on ingest behavior and reproducibility controls.

How ingest resolves sources
---------------------------

Ingest uses mapping catalogs to determine:

- source roots,
- backend glob selection,
- ignore suffix handling,
- alias and priority behavior.

Verification behavior
---------------------

After copy, ingest verifies expected tim presence and reports missing files.

Interpret missing-file warnings as one of:

- mapping mismatch,
- source-tree drift,
- renamed files,
- expected optional sources not present.

Lockfiles
---------

Lockfiles capture resolved source match sets for reproducibility.

Strict mode behavior:

- aborts ingest when source-tree content differs from lock snapshot.

Recommended lifecycle:

1. exploratory ingest with relaxed strictness,
2. stabilize mapping,
3. freeze lock snapshot,
4. enforce strict lock validation for production reruns.

Branch behavior
---------------

Ingest commit target is controlled by ingest base/branch settings. Ensure the
intended git repo root is used (not a nested output path with independent git
state).

Troubleshooting checklist
-------------------------

- confirm mapping paths exist,
- confirm output root is the intended repo tree,
- confirm base branch exists in that repo,
- inspect lock validation report on abort.

See also
--------

- Ingest mapping schema: ``configs/schemas/ingest_mapping.schema.json``
- Config layout: :doc:`../config_layout`
