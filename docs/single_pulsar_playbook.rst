Single-Pulsar Playbook
======================

This guide is a step-by-step manual for running the full single-pulsar
workflow.

The goal is practical understanding. After working through these pages, a user
should be able to:

- identify the files involved in each stage,
- place configuration keys in the correct file type,
- understand what each stage is intended to produce,
- build a usable single-pulsar branch with consistent system flags and jumps,
- run PQC in a controlled and reproducible way,
- explain the rationale behind a PQC configuration,
- keep detection and mutation conceptually separate.

The pages below are ordered in the recommended progression.

.. toctree::
   :maxdepth: 2

   single_pulsar_playbook/overview
   single_pulsar_playbook/files_and_configs
   single_pulsar_playbook/ingest
   single_pulsar_playbook/fixdataset_first_pass
   single_pulsar_playbook/pqc_detection
   single_pulsar_playbook/backend_profiles
   single_pulsar_playbook/workflow
   single_pulsar_playbook/troubleshooting
   single_pulsar_playbook/quick_reference


How To Use This Manual
----------------------

Do not skip directly to PQC.

The order matters:

1. start with the filesystem layout and config roles,
2. then ingest and the canonical data tree,
3. then the first FixDataset pass that creates a usable branch with
   consistent flags and jumps,
4. then pure detection with PQC,
5. then non-destructive QC application,
6. only then discuss tuning, backend-specific overrides, and workflows.

Core principle:

``pleb`` coordinates stages and file movement. ``pqc`` is the detector layer
used within that process. In practice, it is easier to understand the data
tree and branch structure first, then QC.

Further reading:

- configuration system details: :doc:`configuration`
- layout of ``configs/``: :doc:`config_layout`
- CLI and mode entry points: :doc:`cli`, :doc:`running_modes`
- stage-level operational notes: :doc:`pleb_deep_dive`
