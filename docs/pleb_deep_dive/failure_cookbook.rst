Failure Cookbook and Recipes
============================

This chapter provides fast diagnosis patterns for common operational failures.

Common failure signatures
-------------------------

``pathspec '<branch>' did not match any file(s) known to git``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Likely cause:

- base branch does not exist in the repo where stage is mutating.

Checks:

- verify actual git root used by stage,
- list local branches,
- update base branch setting.

``InvalidGitRepositoryError: <path>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Likely cause:

- stage points to non-repo path as mutation root.

Checks:

- run from intended repo root,
- confirm ``home_dir`` / output roots,
- avoid nested output paths with independent repository assumptions.

``Ingest lockfile validation failed: source tree changed ...``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Likely cause:

- source data changed after lock snapshot.

Checks:

- inspect validation diff,
- decide whether to accept and refresh lock,
- disable strict lock only for exploratory runs.

``Refusing to fit for bad JUMP ... had no data points in range``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Likely cause:

- JUMP systems present in par but missing in active tim include set.

Checks:

- confirm expected tim files are ingested,
- verify ``-sys`` assignment and include composition,
- prune stale jumps where policy allows.

``pqc failed for <pulsar>; skipping QC for this pulsar``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Likely cause:

- runtime dependency issue,
- bad path handed to pqc,
- serialization/config mismatch.

Checks:

- verify pqc importability in active environment,
- verify per-pulsar par/tim paths,
- inspect stderr payload in run log.

``No *_qc.csv files found under <run_dir>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Likely cause:

- report stage pointed at wrong run dir,
- pqc produced no outputs due to prior failure.

Checks:

- verify run tag path,
- confirm pqc outputs exist before report stage.

Operational recipes
-------------------

Recipe 1: ingest-only validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Run ingest with verify enabled.
- Confirm expected tim/par population.
- Freeze/update lockfile if mapping stabilized.

Recipe 2: harmonize flags/jumps without apply deletion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Enable fixdataset and normalization controls.
- Keep ``fix_qc_action = "comment"``.
- Review outputs before any destructive policy.

Recipe 3: detect first, apply second
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Workflow Step A: detect on branch ``X``.
- Workflow Step B: apply comments from Step A results onto branch ``Y``.

Recipe 4: cross-pulsar anomaly sweep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Enable cross-pulsar report on completed QC outputs.
- Use result as triage prioritization, then inspect backend-level evidence.
