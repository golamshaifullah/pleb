Single-Pulsar Playbook
======================

This guide is a step-by-step manual for learning the full workflow.

The intended outcome is not just "they can run a command." The intended
outcome is:

- they know what files exist,
- they know which config keys belong in which file,
- they know what each stage is trying to achieve,
- they can build a basic single-pulsar branch with system flags and jumps,
- they can run PQC in a controlled way,
- they can explain why a given PQC configuration was chosen,
- they can separate detection from data mutation.

The pages below are ordered in the recommended learning sequence.

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

1. learn the filesystem layout and config roles,
2. learn ingest and the canonical data tree,
3. learn the first FixDataset pass that creates a usable branch with
   consistent flags and jumps,
4. learn pure detection with PQC,
5. learn non-destructive QC application,
6. only then discuss tuning, backend-specific overrides, and workflows.

Core principle:

``pleb`` coordinates stages and file movement. ``pqc`` is a detector layer
inside that process. It is usually easier to understand the data tree and
branch structure first, then QC.

Further reading:

- configuration system details: :doc:`configuration`
- layout of ``configs/``: :doc:`config_layout`
- CLI and mode entry points: :doc:`cli`, :doc:`running_modes`
- stage-level operational notes: :doc:`pleb_deep_dive`
