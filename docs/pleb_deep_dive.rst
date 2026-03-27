PLEB Deep Dive
==============

This landing page is a quick operational overview.

Use it to navigate the deep-dive subchapters by concern, instead of scanning
one long monolithic page.

Quick overview
--------------

PLEB is the orchestration layer for:

- ingesting and structuring timing data,
- applying deterministic dataset-fix policies,
- running tempo2 and PQC stages,
- applying post-QC actions,
- generating run and review artifacts with reproducibility metadata.

A practical production pattern is:

1. ingest into a controlled branch,
2. run fix normalization and variant generation,
3. run detection,
4. apply comments/actions in a separate pass,
5. publish compact reports for reviewer triage.

How to read the deep dive
-------------------------

- If you are new: start with **Overview and Ownership** + **Stage-by-Stage Flow**.
- If you are configuring runs: read **Operational Config Groups** and
  :doc:`configuration`.
- If you are debugging production failures: go directly to **Failure Cookbook
  and Recipes**.

Deep-dive chapters
------------------

.. toctree::
   :maxdepth: 1

   pleb_deep_dive/overview
   pleb_deep_dive/stage_flow
   pleb_deep_dive/config_groups
   pleb_deep_dive/ingest
   pleb_deep_dive/fixdataset
   pleb_deep_dive/tempo2_qc_reporting
   pleb_deep_dive/failure_cookbook

Related docs
------------

- Config layout map: :doc:`config_layout`
- Configuration system guide: :doc:`configuration`
- Full key-by-key catalog: :doc:`full_settings_catalog`
- CLI and run-mode usage: :doc:`cli`, :doc:`running_modes`, :doc:`modes`
- PQC detector internals: https://golamshaifullah.github.io/pqc/index.html
