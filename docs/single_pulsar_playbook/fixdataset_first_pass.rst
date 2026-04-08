First FixDataset Pass: Build A Usable Single-Pulsar Branch
==========================================================

This is the stage that produces the first branch that is usually practical to
analyze and maintain.

The aim is not QC yet. The aim is:

- consistent system flags,
- consistent jump structure,
- a clean branch boundary,
- optional variant products for later use.


Why This Stage Comes Before PQC
-------------------------------

PQC grouping and interpretation depend on the data layout and metadata being
reasonably coherent.

If PQC is run before system flags and jumps are in order, the output may be
technically correct but operationally hard to interpret.

So the first serious branch-building pass should usually be:

- infer or normalize system flags,
- insert missing jumps,
- prune stale jumps,
- optionally generate variants,
- commit that state to a new branch.


The Step-1 Profile
------------------

Create a run profile under ``configs/runs/fixdataset/``.

Example:
``configs/runs/fixdataset/single_pulsar_step1_fix.toml``

Tracked repository example:
``configs/runs/fixdataset/single_pulsar_step1_fix.example.toml``

.. code-block:: toml

   home_dir = "/data/canonical"
   dataset_name = "EPTA-DR3/epta-dr3-data"
   results_dir = "results"
   singularity_image = "/work/containers/psrpta.sif"

   branches = ["step1_fix_flags_variants"]
   reference_branch = "step1_fix_flags_variants"
   pulsars = ["J1909-3744"]
   jobs = 1
   outdir_name = "j1909_step1_fix"

   run_tempo2 = false
   run_pqc = false
   qc_report = false
   make_plots = false
   make_reports = false
   make_covmat = false

   run_fix_dataset = true
   fix_apply = true
   fix_base_branch = "raw_ingest"
   fix_branch_name = "step1_fix_flags_variants"
   fix_commit_message = "Step1: normalize flags and jumps for J1909-3744"

   fix_infer_system_flags = true
   fix_system_flag_overwrite_existing = true
   fix_insert_missing_jumps = true
   fix_prune_stale_jumps = true
   fix_jump_flag = "-sys"

   fix_generate_alltim_variants = true
   fix_backend_classifications_path = "configs/catalogs/variants/backend_classifications_legacy_new.toml"
   fix_alltim_variants_path = "configs/catalogs/variants/alltim_variants_legacy_new.toml"

   fix_jump_reference_variants = true
   fix_jump_reference_jump_flag = "-sys"
   fix_jump_reference_keep_tmp = false
   fix_jump_reference_csv_dir = "results/jump_reference"

   fix_dedupe_toas_within_tim = true
   fix_remove_overlaps_exact = true

   fix_ensure_ephem = "DE440"
   fix_ensure_clk = "TT(BIPM2024)"

This is the single-pulsar version of the repository's Step 1 pattern
from ``configs/workflows/steps/step1_fix_flags_variants.toml``.


What Each Key Is For
--------------------

Core routing keys:

``fix_base_branch``
  Existing branch to mutate from. For a newly ingested dataset, this is often
  ``raw_ingest``.

``fix_branch_name``
  New branch that receives the edits.

``branches`` and ``reference_branch``
  Keep these aligned with the branch this stage is meant to operate on.

Mutation keys:

``fix_infer_system_flags``
  Infer or normalize system labels used later by jump logic and PQC grouping.

``fix_system_flag_overwrite_existing``
  Overwrite existing inconsistent values. Use carefully, but for an initial
  harmonization pass it is often the right choice.

``fix_insert_missing_jumps``
  Insert jumps that should exist based on backend/system structure.

``fix_prune_stale_jumps``
  Remove jumps that no longer map to real data structure.

``fix_jump_flag``
  Flag used as the jump grouping reference. Commonly ``-sys``.
  This choice should align with the grouping logic that later stages use.

Variant keys:

``fix_generate_alltim_variants``
  Generate variant include products for downstream analysis or review.

``fix_backend_classifications_path``
  Classification catalog used to decide which backends belong to which variant
  families.

``fix_alltim_variants_path``
  Variant-definition catalog.

``fix_jump_reference_variants``
  Build reference-system jump variants.
  This is useful when the workflow needs a reproducible set of variant products
  for later comparison or review.

Consistency keys:

``fix_ensure_ephem``
  Ensure a specific ephemeris in parfiles.

``fix_ensure_clk``
  Ensure a specific clock string in parfiles.


Why ``run_tempo2`` Is Usually Disabled Here
-------------------------------------------

The point of this pass is to establish a coherent branch structure and a
coherent data-model boundary. Running tempo2 at the same time can blur that
boundary because it adds fit products and diagnostics to a run whose main
purpose is mutation.

It is therefore usually cleaner to:

1. finish the metadata and jump pass,
2. inspect the resulting branch,
3. run tempo2 and PQC in the next pass.


What "Basic Par File With All The Jumps" Means
----------------------------------------------

This phrase needs a precise operational meaning.

The goal is not "every possible jump anyone could imagine." The goal is:

- the parfile reflects the current backend/system structure,
- missing expected jumps are inserted,
- stale or obsolete jumps are removed,
- the branch becomes a sensible baseline for timing and QC.

In other words, this stage creates the first operationally coherent timing
model branch.


How To Run The First Pass
-------------------------

Run:

.. code-block:: bash

   pleb --config configs/runs/fixdataset/single_pulsar_step1_fix.toml

Because ``fix_apply = true``, this is a branch-mutating run.


What To Inspect After The First Pass
------------------------------------

Inspect:

1. the new git branch,
2. the pulsar parfile,
3. the pulsar ``_all.tim`` and variant tim products,
4. the FixDataset summary outputs under the run directory,
5. the jump reference CSV outputs if enabled.

Check:

- whether ``-sys`` values now look consistent,
- whether expected jump lines exist in the parfile,
- whether obviously obsolete jump lines were removed,
- whether variant files were created where expected.


How To Read The Outputs
-----------------------

For a single pulsar, the most useful direct inspection points are usually:

- the pulsar parfile in the dataset branch,
- the pulsar ``_all.tim`` file,
- the backend tim files under ``tims/``,
- any variant products generated by the run,
- the run summary files under the run directory.

The parfile answers:

- which jumps exist,
- whether ephemeris and clock defaults were enforced,
- whether the model now reflects the intended backend partitioning.

The tim files answer:

- whether ``-sys`` and related flags were normalized consistently,
- whether include structure still looks sane after mutation.


What Jumps Mean In This Workflow
--------------------------------

A jump is not merely a nuisance parameter. In this workflow, jumps express the
partitioning implied by backend/system differences.

The following points are operationally important:

- jumps are tied to how the data are grouped,
- wrong grouping gives wrong jump structure,
- missing jumps can bias the fit,
- stale jumps can clutter the model and confuse interpretation,
- jump maintenance is a data-structure task before it is a statistical task.


Common First-Pass Errors
------------------------

- using the wrong ``fix_base_branch``,
- forgetting to limit to one pulsar,
- running with ``run_tempo2=true`` when the goal is only mutation,
- enabling too many unrelated fix actions at once,
- generating variants without understanding the catalogs used.


What This Stage Usually Writes
------------------------------

Depending on the exact settings and input data, this pass may write:

- updated tim files with normalized flags,
- a revised pulsar parfile with jump maintenance applied,
- regenerated ``_all.tim`` include files,
- optional variant tim products,
- jump-reference CSV outputs under ``fix_jump_reference_csv_dir``,
- run summaries in the run directory under ``results_dir``.

Because this pass mutates the dataset branch, it is useful to compare the
``raw_ingest`` branch and the Step-1 branch directly with git after the run.


How This Stage Connects To PQC
------------------------------

The output of this stage is not just a cleaned branch. It is also the branch
that defines the grouping vocabulary for later QC:

- backend/system flags are more coherent,
- expected jumps are present,
- stale jumps are removed,
- variant products, if enabled, are available on the branch that the detect
  stage will read.

This is why Step 2 commonly sets:

.. code-block:: toml

   fix_base_branch = "step1_fix_flags_variants"

and then uses ``pqc_backend_col = "sys"`` or another grouping key that now has
cleaner semantics than it had before this pass.


Related Documentation
---------------------

- FixDataset stage overview: :doc:`../pleb_deep_dive/fixdataset`
- operational grouping of fix keys: :doc:`../pleb_deep_dive/config_groups`
- broader configuration behavior: :doc:`../configuration`
- full key catalog for ``fix_*`` settings: :doc:`../full_settings_catalog`
