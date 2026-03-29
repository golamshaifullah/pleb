Troubleshooting
===============

This page covers common wrapper issues and fixes.

``Refusing to overwrite existing file: pleb.toml``
--------------------------------------------------

Cause:

- ``pleb init`` was run where ``pleb.toml`` already exists.

Fix:

- use ``--force`` if overwrite is intended.

``missing_required=home_dir,singularity_image`` from ``doctor``
----------------------------------------------------------------

Cause:

- required path keys are absent in resolved config.

Fix:

- set under ``[paths]``:
  - ``home_dir``
  - ``singularity_image``

``Unsupported run.mode=...``
----------------------------

Cause:

- unsupported ``run.mode`` value.

Fix:

- use one of: ``pipeline``, ``ingest``, ``workflow``, ``qc-report``.

Run executes but behavior is unexpected
--------------------------------------

Likely causes:

- key mapped differently than expected,
- same key set in multiple sections,
- advanced key missing in UX aliases.

Debug steps:

1. run ``pleb explain --config pleb.toml``
2. move critical advanced keys into ``[pipeline]`` explicitly
3. rerun ``pleb doctor``
4. pilot run on small pulsar subset

Workflow mode fails to find file
--------------------------------

Cause:

- ``[workflow].file`` missing or wrong path.

Fix:

- set valid path, e.g.
  ``configs/workflows/branch_chained_fix_pqc_variants.toml``.

Preset merge gives unexpected values
------------------------------------

Cause:

- deep-merge override replaced scalar/list at same path.

Fix:

- inspect ``pleb.toml`` after applying preset,
- reapply intended local edits,
- keep a project baseline preset for your team.

When to fall back to legacy config files
----------------------------------------

Use legacy run/workflow TOML directly when:

- you need advanced keys not yet modeled in UX sections,
- you are validating behavior parity against historical runs,
- you are debugging low-level mode-specific behavior.

The wrapper is additive, not mandatory.
