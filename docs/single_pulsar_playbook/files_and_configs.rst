Files, Directories, And Config Roles
====================================

Read this page before running any commands.

If files and keys are placed inconsistently, the workflow still may run, but
it becomes difficult to review, reuse, or debug.


Three Different Trees
---------------------

This workflow uses three different directory trees:

1. the raw source-data tree,
2. the canonical ingested dataset tree,
3. the ``pleb`` config tree.

They serve different purposes and should be kept conceptually separate.


Raw Source-Data Tree
--------------------

This is whatever external layout the source data currently uses. It may be
messy, split by telescope, backend, year, or release bundle.

Typical contents:

- scattered ``.tim`` files,
- one or more ``.par`` roots,
- optional template files,
- inconsistent backend naming.

This tree should usually not be edited during normal training runs.


Canonical Ingested Dataset Tree
-------------------------------

After ingest, ``pleb`` expects a regular pulsar-oriented layout. For each
pulsar, the canonical form is:

.. code-block:: text

   <dataset_root>/
     J1909-3744/
       J1909-3744.par
       J1909-3744_all.tim
       tims/
         EFF.P200.1360.tim
         NRT.NUPPI.1480.tim
       tmplts/
         ...

This is the tree later stages operate on.

Key facts about this tree:

- the pulsar directory is the main unit of work,
- the pulsar parfile sits at the pulsar root,
- backend tim files sit under ``tims/``,
- ``Jxxxx_all.tim`` is the include file that gathers backend tims.


The ``pleb`` Config Tree
------------------------

Inside this repository, config files are separated by role.

Use this as a hard rule:

- ``configs/runs/*`` are executable run profiles,
- ``configs/catalogs/*`` are shared data assets,
- ``configs/rules/*`` are policy files,
- ``configs/workflows/*`` are multi-step orchestration files,
- ``configs/state/*`` is generated runtime state.

This split is described in :doc:`../config_layout` and
``configs/README.md``. Preserving this split makes later maintenance much
easier.


Which Keys Go In Which File
---------------------------

This distinction is the main source of avoidable config drift.

Put these in a run profile under ``configs/runs/...``:

- environment-specific paths such as ``home_dir``, ``results_dir``,
  ``singularity_image``,
- run scope such as ``pulsars``, ``branches``, ``reference_branch``,
- stage toggles such as ``run_tempo2``, ``run_pqc``, ``run_fix_dataset``,
- run-local policy choices that are specific to one analysis pass.

Put these in a catalog file under ``configs/catalogs/...``:

- ingest mapping JSON,
- backend classification tables,
- variant definitions,
- stable system lookup tables.

Put these in a rule file under ``configs/rules/...``:

- per-backend PQC overrides,
- overlap action policies,
- relabel policies.

Put these in a workflow file under ``configs/workflows/...``:

- stage order,
- branch hand-off between stages,
- per-step overrides,
- one command that runs a known sequence.


Recommended Single-Pulsar File Set
----------------------------------

For a single-pulsar setup, create a small dedicated set of files.

.. code-block:: text

   configs/
     catalogs/
       ingest/
         single_pulsar_mapping.json
     runs/
       ingest/
         single_pulsar_ingest.toml
       fixdataset/
         single_pulsar_step1_fix.toml
         single_pulsar_pqc_apply.toml
       pqc/
         single_pulsar_pqc_detect.toml
     rules/
       pqc/
         single_pulsar_backend_profiles.toml
     workflows/
       single_pulsar_3pass.toml

This is not required by the code, but it is a clean layout for learning and
maintenance.


How Paths Resolve Across These Files
------------------------------------

The most important path relationship in a single-pulsar setup is:

1. ingest writes a canonical dataset tree to ``ingest_output_dir``,
2. later run profiles refer to that tree through ``home_dir`` and
   ``dataset_name``,
3. run outputs are written separately under ``results_dir``.

Example:

- ingest writes to
  ``/data/canonical/EPTA-DR3/epta-dr3-data``
- later run profiles use
  ``home_dir = "/data/canonical"``
- and
  ``dataset_name = "EPTA-DR3/epta-dr3-data"``
- while writing results under
  ``results_dir = "results"``

This means the dataset itself and the run outputs are separate concerns:

- the dataset tree is the input tree being analyzed or mutated,
- the results tree is where logs, summaries, QC products, plots, and run
  metadata are written.


How Branches Fit Into The Picture
---------------------------------

The branch keys refer to the git repository that contains the canonical dataset
tree, not to the ``results_dir``.

For mutating stages, the important branch keys are:

``fix_base_branch``
  Existing branch used as the starting point for a mutation pass.

``fix_branch_name``
  New branch written by that mutation pass.

``branches``
  The branch or branches that the run processes.

``reference_branch``
  The comparison anchor for reports and, in practice, the branch that the run
  is organized around.

In a branch-chained workflow, these values should form a simple sequence rather
than a branching tangle.


The Most Important Run Keys
---------------------------

These are the first keys to understand.

``home_dir``
  Root that contains the dataset tree.

``dataset_name``
  Dataset directory or dataset identifier resolved under ``home_dir``.
  In most single-pulsar setups this is a relative path under ``home_dir``, not an
  independent root.

``results_dir``
  Where output run directories are written.
  This is separate from the dataset tree and should remain separate.

``branches``
  Which data-repo git branches to process.
  For a single-pulsar training run, this should usually be a one-element list.

``reference_branch``
  The branch used as reference for comparisons and often the branch the run is
  conceptually anchored to.

``pulsars``
  Either ``"ALL"`` or an explicit list such as ``["J1909-3744"]``.
  For a single-pulsar setup, use a one-element list.

``run_tempo2``
  Whether the fit stage runs.

``run_fix_dataset``
  Whether FixDataset logic runs.

``fix_apply``
  Whether FixDataset actually writes mutations to the dataset branch.
  When ``fix_apply = true``, branch names and commit messages are part of the
  operational state of the run and should be chosen deliberately.

``run_pqc``
  Whether PQC detectors run.


What Not To Do
--------------

Avoid these patterns:

- store experimental one-off absolute paths in shared catalogs,
- modify a shared example config directly without copying it,
- use ``pulsars = "ALL"`` in an initial single-pulsar setup,
- mix the detect stage and apply stage without explaining the difference,
- apply QC deletion before they understand the QC columns.


Minimal Naming Convention
-------------------------

Use names that expose stage and purpose:

- ``single_pulsar_ingest.toml``
- ``single_pulsar_step1_fix.toml``
- ``single_pulsar_pqc_detect.toml``
- ``single_pulsar_pqc_apply.toml``
- ``single_pulsar_backend_profiles.toml``
- ``single_pulsar_3pass.toml``

That makes later debugging much easier than ambiguous names like
``test.toml`` or ``config2.toml``.


How The Config Layers Work Together
-----------------------------------

For a single-pulsar workflow, the interaction between files usually looks like
this:

1. the ingest run profile points to an ingest mapping catalog,
2. the first FixDataset run profile points to variant catalogs and optional
   system/group rule tables,
3. the PQC detect run profile points to global ``pqc_*`` settings and
   optionally to ``pqc_backend_profiles_path``,
4. the QC-apply run profile points back to the QC outputs from the detect run,
5. the workflow file, if used, sequences those run profiles and passes branch
   names from one stage to the next.

This is the operational reason the repo separates runs, catalogs, rules, and
workflows.


A Minimal Mental Model
----------------------

When deciding where to add a new setting or file, ask three questions:

1. Is this about one specific run invocation?
   Put it in a run profile.
2. Is this a reusable mapping or lookup table?
   Put it in a catalog.
3. Is this a reusable behavior choice?
   Put it in a rule file.

If the answer is instead "this describes the order in which several run
profiles should execute," the right place is a workflow file.


Related Documentation
---------------------

- full layout guide: :doc:`../config_layout`
- config authoring and precedence: :doc:`../configuration`
- quick index of config-tree purpose: ``configs/README.md``
- full key catalog: :doc:`../full_settings_catalog`
