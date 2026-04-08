Single-Pulsar Three-Pass Workflow
=================================

Once the separate stages are understood, the next step is to chain them into
one reproducible workflow file.

Workflow mode should not be the first introduction to the pipeline. It is most
useful after the individual stages are already understood.


The Three Passes
----------------

For one pulsar, a clean three-pass workflow is:

Pass 1
  Build the first coherent branch with system flags, jumps, and variants.

Pass 2
  Run tempo2 and PQC on top of Pass 1 without applying QC edits.

Pass 3
  Read the QC outputs from Pass 2 and comment flagged TOAs on a new branch.


The Three Run Profiles
----------------------

You should already have these files:

- ``configs/runs/fixdataset/single_pulsar_step1_fix.toml``
- ``configs/runs/pqc/single_pulsar_pqc_detect.toml``
- ``configs/runs/fixdataset/single_pulsar_pqc_apply.toml``

The third one is new and is shown below.

In practice, the workflow file works best when each referenced run profile can
also be executed directly on its own. That makes stage-level debugging much
simpler.


The QC-Apply Profile
--------------------

Example:
``configs/runs/fixdataset/single_pulsar_pqc_apply.toml``

Tracked repository example:
``configs/runs/fixdataset/single_pulsar_pqc_apply.example.toml``

.. code-block:: toml

   home_dir = "/data/canonical"
   dataset_name = "EPTA-DR3/epta-dr3-data"
   results_dir = "results"
   singularity_image = "/work/containers/psrpta.sif"

   branches = ["step3_apply_qc_comments"]
   reference_branch = "step3_apply_qc_comments"
   pulsars = ["J1909-3744"]
   jobs = 1
   outdir_name = "j1909_pqc_apply"

   run_tempo2 = false
   run_pqc = false
   qc_report = false
   make_plots = false
   make_reports = false
   make_covmat = false

   run_fix_dataset = true
   fix_apply = true
   fix_base_branch = "step2_pqc_balanced_detect"
   fix_branch_name = "step3_apply_qc_comments"
   fix_commit_message = "Step3: apply PQC comments for J1909-3744"

   fix_qc_results_dir = "results/j1909_pqc_detect/qc"
   fix_qc_branch = "step2_pqc_balanced_detect"

   fix_qc_remove_outliers = true
   fix_qc_action = "comment"
   fix_qc_outlier_cols = ["bad_point", "robust_outlier", "robust_global_outlier", "bad_mad"]
   fix_qc_remove_bad = true
   fix_qc_remove_transients = false
   fix_qc_remove_solar = false
   fix_qc_remove_orbital_phase = false

   fix_generate_alltim_variants = true
   fix_backend_classifications_path = "configs/catalogs/variants/backend_classifications_legacy_new.toml"
   fix_alltim_variants_path = "configs/catalogs/variants/alltim_variants_legacy_new.toml"
   fix_jump_reference_variants = true
   fix_jump_reference_jump_flag = "-sys"

This is the single-pulsar version of
``configs/workflows/steps/step3_apply_qc_comments_variants.toml``.


How To Explain The QC-Apply Keys
--------------------------------

``fix_qc_results_dir``
  Directory containing the QC outputs produced by the detect run.

``fix_qc_branch``
  Branch name that the QC outputs correspond to.

``fix_qc_remove_outliers``
  Enable QC-driven action.

``fix_qc_action = "comment"``
  Comment flagged TOAs rather than delete them. This is the recommended first
  policy.

``fix_qc_outlier_cols``
  Explicit QC columns that should count as actionable outlier evidence.

``fix_qc_remove_transients = false``
  Do not automatically comment transient or event families until their meaning
  has been reviewed explicitly in the QC outputs.


Why The Apply Stage Uses Explicit Outlier Columns
-------------------------------------------------

Action policy is distinct from detection policy.

For a first apply pass, a narrow explicit list like:

.. code-block:: toml

   fix_qc_outlier_cols = ["bad_point", "robust_outlier", "robust_global_outlier", "bad_mad"]

is better than a vague "anything suspicious" rule.

This keeps the first apply pass auditable.


How ``fix_qc_results_dir`` And ``fix_qc_branch`` Work Together
---------------------------------------------------------------

These two keys are easy to misunderstand.

``fix_qc_results_dir``
  Points to the run-directory location where the QC outputs were written.

``fix_qc_branch``
  Tells the apply stage which branch-specific QC subdirectory or context those
  outputs correspond to.

Together they define the hand-off from Step 2 to Step 3:

- Step 2 generates QC outputs under its run directory,
- Step 3 reads those outputs back in and applies the chosen action policy to a
  new branch.

If these paths are wrong, the apply stage can appear to ignore QC results even
though the real problem is that it is reading the wrong run location.


The Workflow File
-----------------

Once the three run profiles exist, make a workflow file under
``configs/workflows/``.

Example:
``configs/workflows/single_pulsar_3pass.toml``

Tracked repository example:
``configs/workflows/single_pulsar_3pass.example.toml``

.. code-block:: toml

   config = "configs/runs/fixdataset/single_pulsar_step1_fix.toml"
   mode = "serial"

   [[groups]]
   name = "step1_fix_flags_and_jumps"
   mode = "serial"

   [[groups.steps]]
   name = "pipeline"
   config = "configs/runs/fixdataset/single_pulsar_step1_fix.toml"

   [[groups]]
   name = "step2_detect"
   mode = "serial"

   [[groups.steps]]
   name = "pipeline"
   config = "configs/runs/pqc/single_pulsar_pqc_detect.toml"

   [[groups]]
   name = "step3_apply"
   mode = "serial"

   [[groups.steps]]
   name = "pipeline"
   config = "configs/runs/fixdataset/single_pulsar_pqc_apply.toml"

This is the stripped-down single-pulsar form of the repository's branch-chained
workflow pattern in ``configs/workflows/branch_chained_fix_pqc_variants.toml``.


How Run Directories And Branches Relate In The Workflow
-------------------------------------------------------

The workflow coordinates two parallel pieces of state:

- dataset branches,
- run directories under ``results_dir``.

These are related, but they are not the same thing.

Example sequence:

- Pass 1 writes branch ``step1_fix_flags_variants`` and run directory
  ``results/j1909_step1_fix``,
- Pass 2 writes branch ``step2_pqc_balanced_detect`` and run directory
  ``results/j1909_pqc_detect``,
- Pass 3 writes branch ``step3_apply_qc_comments`` and run directory
  ``results/j1909_pqc_apply``.

The branch names define the mutation history of the dataset. The run
directories define where logs, summaries, plots, and QC products are stored.


How To Run The Workflow
-----------------------

Run:

.. code-block:: bash

   pleb workflow --file configs/workflows/single_pulsar_3pass.toml

This is most useful after the stages have already been run manually at least
once.


When To Prefer Manual Runs Over The Workflow File
-------------------------------------------------

Use the workflow file when:

- the stage order is stable,
- the branch hand-off is already understood,
- the goal is repeatability.

Run stages manually when:

- a config is still being tuned,
- a branch name or output path is changing,
- the detect/apply hand-off is being debugged,
- one stage is failing and needs isolated inspection.


Why Branch Chaining Matters
---------------------------

Each pass starts from the previous branch and writes a new branch:

- ``raw_ingest`` -> ``step1_fix_flags_variants``
- ``step1_fix_flags_variants`` -> ``step2_pqc_balanced_detect``
- ``step2_pqc_balanced_detect`` -> ``step3_apply_qc_comments``

This branch pattern matters because it preserves the logic of each stage:

- Step 1 changes metadata and jump structure,
- Step 2 generates diagnostics,
- Step 3 applies selected QC actions.


Debugging Workflow Mode
-----------------------

If the full workflow fails, do not debug the whole workflow at once.

Instead:

1. identify which pass failed,
2. run that pass directly with ``pleb --config ...``,
3. inspect the resolved config in ``run_settings/``,
4. fix the stage-specific issue,
5. rerun the workflow.

This avoids treating workflow mode like a black box.


Final Rule
----------

A workflow file is a convenience layer, not a substitute for understanding the
individual run profiles.

If each of the three run profiles cannot be explained independently, the
workflow file is still too opaque for routine use.


Related Documentation
---------------------

- workflow mode overview: :doc:`../running_modes`
- branch-chained workflow examples: :doc:`../configuration`
- repository example workflow:
  ``configs/workflows/branch_chained_fix_pqc_variants.toml``
