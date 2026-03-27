FixDataset Deep Dive
====================

This chapter covers mutation policy and branch-safe application patterns.

FixDataset as layered transformation
------------------------------------

A practical order for understanding behavior:

1. flag normalization (``-sys``, ``-group``, ``-pta``),
2. overlap/relabel policy application,
3. jump insertion/pruning,
4. par default enforcement,
5. variant generation,
6. optional QC-based apply actions.

Policy boundaries
-----------------

Use catalogs for static mappings/tables and rules for strategy decisions:

- catalogs: ``configs/catalogs/*``
- rules: ``configs/rules/*``

Non-destructive first
---------------------

Recommended default apply policy is comment-first:

- ``fix_qc_action = "comment"``

Only enable deletion after reviewing QC summaries and action lists.

Branch safety pattern
---------------------

Use branch chaining:

- base branch = previously validated stage,
- new branch per mutation pass,
- commit message documents the policy pass applied.

Variant generation
------------------

FixDataset can generate:

- multiple include files: ``J*_all.<variant>.tim``
- variant par products (when configured)

Ensure classification + variant catalogs are both set and compatible.

Debug order for missing expected outputs
----------------------------------------

1. verify stage toggles,
2. verify branch/apply controls,
3. verify catalog/rule file paths,
4. verify output placement in pulsar and run result trees.

See also
--------

- Config groups: :doc:`config_groups`
- Full settings catalog: :doc:`../full_settings_catalog`
