Configuration Guide
===================

This chapter explains how to structure, author, validate, and operate PLEB
configuration at scale.

For the standard filesystem map of ``configs/`` (directory-by-directory),
start with :doc:`config_layout`.

It is intentionally focused on configuration system behavior (layout,
composition, precedence, lifecycle, and reproducibility). Detector statistics
and PQC internals are deliberately out of scope and covered in PQC docs.

Use this guide when you need to:

- understand which file type belongs where,
- split policy from run-time inputs,
- chain complex branch-based workflows,
- keep runs reproducible under evolving source trees,
- maintain a large, multi-user config repo without drift.


Scope and Positioning
---------------------

PLEB configuration is organized around five concerns:

1. run intent,
2. reusable domain catalogs,
3. policy rules,
4. coordination of multi-step execution,
5. generated state for reproducibility.

The configuration model is intentionally rule-based. In practice, this means:

- Python code defines **execution semantics**.
- Config files define **dataset-specific behavior**.

When deciding where to put a change:

- if behavior should vary by dataset/release, prefer config;
- if behavior should never vary and is a core invariant, keep it in code.


How To Read This Chapter
------------------------

If you are onboarding quickly:

1. read :ref:`config-mental-model`;
2. read :ref:`config-layout`;
3. use :ref:`config-minimal-patterns` for your first runnable profile;
4. use :doc:`full_settings_catalog` for per-key lookup.

The settings catalog page is generated from the UX key registry:

- source of truth: ``pleb/ux/key_catalog.py``
- generator: ``scripts/generate_settings_catalog.py``

If you are maintaining production pipelines:

1. read :ref:`config-composition`;
2. read :ref:`config-repro`;
3. read :ref:`config-workflow-branching`;
4. read :ref:`config-troubleshooting`.


.. _config-mental-model:

Mental Model
------------

Think of PLEB config as a layered system:

Layer A: Run profiles
~~~~~~~~~~~~~~~~~~~~~

Run profiles tell PLEB what to execute *for one run invocation*.

Examples:

- ``configs/runs/ingest/ingest_epta_data.toml``
- ``configs/runs/pqc/pqc_balanced.toml``
- ``configs/runs/fixdataset/fixdataset_discover_jumps_variants.toml``

Layer B: Catalogs and rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Catalogs and rules are reusable across runs.

- Catalogs = static lookup/mapping data.
- Rules = decision policy (relabel/overlap/profile overrides).

Examples:

- ``configs/catalogs/system_tables/overlapped_timfiles.toml``
- ``configs/catalogs/system_flags/system_flag_mapping.ingest_epta_data.json``
- ``configs/rules/pqc/backend_profiles.example.toml``

Layer C: Workflows
~~~~~~~~~~~~~~~~~~

Workflow files describe **ordering and grouping** of runs.

They coordinate serial/parallel blocks and pass branch hand-off via overrides.

Examples:

- ``configs/workflows/branch_chained_fix_pqc_variants.toml``
- ``configs/workflows/j1713_j1022_stress_parallel_serial.toml``

Layer D: Generated state
~~~~~~~~~~~~~~~~~~~~~~~~

Generated state captures runtime snapshots (for reproducibility and validation).

Examples:

- ``configs/state/lockfiles/ingest_mapping_epta_data.lock.json``


.. _config-layout:

Directory Layout (Current Canonical)
------------------------------------

The standard layout is:

.. code-block:: text

   configs/
     runs/
       ingest/
       fixdataset/
       pqc/
       pipeline/
     workflows/
       steps/
     catalogs/
       ingest/
       public_releases/
       system_flags/
       system_tables/
       variants/
     rules/
       overlap/
       relabel/
       pqc/
     schemas/
     state/
       lockfiles/
     settings/  # legacy

Interpretation by intent
~~~~~~~~~~~~~~~~~~~~~~~~

- ``runs/``: files you pass to ``--config``.
- ``workflows/``: files you pass to ``pleb workflow --file``.
- ``catalogs/``: reusable data assets.
- ``rules/``: reusable behavior policy.
- ``schemas/``: validation/tooling files.
- ``state/``: generated snapshots.
- ``settings/``: legacy area; avoid for new config.


Run Profiles
------------

Run profiles are mode-specific TOML files that define one execution context.

General rules:

- Keep run profiles executable without editing Python code.
- Keep paths explicit and auditable.
- Keep branch semantics explicit for mutating stages.
- Prefer comments that explain *why* values are chosen.

Subfolders by mode
~~~~~~~~~~~~~~~~~~

``configs/runs/ingest``
^^^^^^^^^^^^^^^^^^^^^^^

Use for ingest-only or ingest-focused entry points.

Typical keys include:

- ``mode = "ingest"`` (when relevant),
- ``ingest_mapping_file``,
- ``ingest_output_dir``,
- ``ingest_base_branch`` / ``ingest_branch_name``,
- lockfile strictness choices.

``configs/runs/fixdataset``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use for fix dataset operations, variants, jump handling, and optionally PQC
integration when applying actions.

Typical keys include:

- ``run_fix_dataset = true``,
- ``fix_apply`` and branch controls,
- system/group inference controls,
- jump insertion/pruning controls,
- overlap/relabel rule paths,
- variant generation controls.

``configs/runs/pqc``
^^^^^^^^^^^^^^^^^^^^

Use for PQC-focused runs (balanced/feature hunt/apply variants).

Typical keys include:

- ``run_tempo2`` and ``run_pqc``,
- ``pqc_*`` detector toggles/profiles,
- ``fix_qc_*`` action controls,
- report toggles.

``configs/runs/pipeline``
^^^^^^^^^^^^^^^^^^^^^^^^^

Use for broad, mixed stage profiles that combine ingest/fix/tempo2/pqc/reporting
in one config.

These profiles are useful for quick end-to-end runs, but large production
operations usually migrate to workflows for clearer stage boundaries.


Workflows and Step Configs
--------------------------

Workflow files are run plans.

A workflow usually:

1. points to one or more step configs,
2. defines serial/parallel grouping,
3. injects per-step overrides (especially branch hand-off),
4. writes stage outputs under distinct run directories.

``configs/workflows/steps/`` holds reusable per-step configs.

This avoids copy-pasting full configs into every workflow file.

.. _config-workflow-branching:

Branch-chained stage pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common production pattern is:

1. Step A (Fix): base = ``raw_ingest`` -> writes ``step1_fix_*`` branch.
2. Step B (Detect): base = Step A branch -> writes ``step2_pqc_*`` branch.
3. Step C (Apply): base = Step B branch -> writes ``step3_apply_*`` branch.

This gives a fully auditable mutation chain with clear rollback points.


Catalog Files
-------------

Catalogs encode reusable data.

``configs/catalogs/ingest``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ingest source mappings:

- source roots,
- backend-to-glob associations,
- ignore suffixes,
- alias handling,
- optional source priority mappings.

``configs/catalogs/system_flags``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

System/telescope/backend inference assets:

- JSON mapping/allowlist files,
- YAML sys-frequency rule files.

``configs/catalogs/system_tables``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

System-level tables loaded by fix/legacy logic:

- jumps-per-system,
- backend bandwidth lookup,
- overlaps tables,
- PTA system labels,
- related lookup tables.

``configs/catalogs/variants``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Variant generation assets:

- backend classifications,
- include file variant definitions.

``configs/catalogs/public_releases``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provider catalog for public release comparison mode.


Rule Files
----------

Rules describe policy decisions that are meant to stay editable without code
changes.

``configs/rules/relabel``
~~~~~~~~~~~~~~~~~~~~~~~~~

Declarative relabel logic for backend/system remapping.

``configs/rules/overlap``
~~~~~~~~~~~~~~~~~~~~~~~~~

Declarative overlap policy (for example, prefer multi-channel over single
channel in matched conditions).

``configs/rules/pqc``
~~~~~~~~~~~~~~~~~~~~~

Per-backend PQC profile overrides for threshold/toggle customization.


Schemas
-------

``configs/schemas`` stores validation and UI schema assets.

Use these for:

- editor/GUI generated forms,
- basic pre-flight validation,
- tooling that needs machine-readable key metadata.


Generated State
---------------

``configs/state`` stores generated files tied to run state, not authored
source configuration.

Current standard usage:

- ``configs/state/lockfiles`` for ingest lock snapshots and validation records.

Keep these separate from authored catalogs/rules so generated data cannot be
mistaken for policy.


.. _config-composition:

Composition and Precedence
--------------------------

PLEB configuration composes from:

1. mode defaults,
2. TOML file values,
3. CLI overrides.

Precedence is reproducible:

- CLI overrides take priority over TOML.
- TOML values take priority over defaults.

Practical implications:

- put stable baseline behavior in TOML;
- use CLI overrides for temporary experimental changes;
- if a run is meant to be reproducible, record override usage in run logs.

PLEB already writes executed command/config snapshots into run outputs; keep
those files with results.


Path Strategy
-------------

Use relative paths for repository-local assets when possible.

Examples:

- ``configs/catalogs/ingest/ingest_mapping_epta_data.json``
- ``configs/rules/pqc/backend_profiles.example.toml``

Use absolute paths only for environment-specific data roots that are not part
of the repo.

When mixing absolute and relative values:

- keep catalogs/rules repo-relative,
- keep source data roots absolute,
- keep comments in TOML documenting environment assumptions.


Naming and Versioning Conventions
---------------------------------

Recommended naming:

- run profile: ``<mode>_<intent>.toml``
- workflow: ``<scope>_<intent>.toml``
- rules/catalogs: ``<domain>_<scope>.<ext>``

For iterative experiments:

- avoid ambiguous names like ``testconfig2.toml`` in long-lived branches,
- prefer date/scope suffixes,
- keep branch names and config names semantically aligned.


Minimal Patterns
----------------

.. _config-minimal-patterns:

Minimal ingest profile
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   mode = "ingest"
   ingest_mapping_file = "configs/catalogs/ingest/ingest_mapping_epta_data.json"
   ingest_output_dir = "/data/epta-dr3"
   ingest_verify = true

Minimal balanced PQC profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   run_tempo2 = true
   run_pqc = true
   run_fix_dataset = false
   qc_report = true

   pqc_backend_col = "sys"
   pqc_backend_profiles_path = "configs/rules/pqc/backend_profiles.example.toml"

   qc_report_compact_pdf = true

Minimal fix+apply profile
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   run_fix_dataset = true
   fix_apply = true
   fix_base_branch = "raw_ingest"
   fix_branch_name = "fix_dataset_apply"
   fix_commit_message = "FixDataset: apply policy updates"

   fix_infer_system_flags = true
   fix_insert_missing_jumps = true
   fix_qc_action = "comment"


Workflow pattern (detect then apply)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   config = "configs/workflows/steps/step1_fix_flags_variants.toml"

   [[groups]]
   name = "detect"
   mode = "serial"

   [[groups.steps]]
   name = "pipeline"
   config = "configs/workflows/steps/step2_pqc_balanced_detect.toml"

   [groups.steps.overrides]
   fix_base_branch = "step1_fix_flags_variants"
   fix_branch_name = "step2_pqc_balanced_detect"

   [[groups]]
   name = "apply"
   mode = "serial"

   [[groups.steps]]
   name = "pipeline"
   config = "configs/workflows/steps/step3_apply_qc_comments_variants.toml"

   [groups.steps.overrides]
   fix_base_branch = "step2_pqc_balanced_detect"
   fix_branch_name = "step3_apply_qc_comments_variants"


.. _config-repro:

Reproducibility and Lockfiles
-----------------------------

Ingest lockfiles allow strict source-tree drift detection.

Recommended operating model:

Exploration phase
~~~~~~~~~~~~~~~~~

- run ingest with non-strict lock behavior,
- validate source mappings and expected copy set,
- iterate until mapping is stable.

Freeze phase
~~~~~~~~~~~~

- generate/update lock snapshot,
- commit lock files under ``configs/state/lockfiles``,
- enable strict validation for production reruns.

CI/production rerun phase
~~~~~~~~~~~~~~~~~~~~~~~~~

- fail fast when lock validation detects source drift,
- review diff, then either:
  - accept/update lock, or
  - reject unexpected source changes.


Migration Notes (Old -> Current Layout)
---------------------------------------

If you still reference legacy paths, migrate as follows:

- ``configs/system_tables/...``
  -> ``configs/catalogs/system_tables/...``

- ``configs/runs/workflow_steps/...``
  -> ``configs/workflows/steps/...``

- ``configs/settings/*.lock*.json``
  -> ``configs/state/lockfiles/...``

Also update any custom scripts, CI invocations, and docs snippets.


Common Failure Modes
--------------------

.. _config-troubleshooting:

1. Wrong repo root / wrong branch scope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptoms:

- stage writes to unexpected directory,
- branch fallback warnings,
- outputs missing where expected.

Checks:

- verify ``home_dir`` and stage-specific output roots,
- verify current working directory when invoking CLI,
- verify branch names exist in target repo.

2. Ingest strict lock aborts unexpectedly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptoms:

- ingest aborts with "source tree changed since lockfile was generated".

Checks:

- inspect lock validation report under ``configs/state/lockfiles``,
- inspect newly added or missing source files,
- decide whether to update lock or reject drift.

3. Workflow override mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptoms:

- step runs from wrong base branch,
- apply stage points to wrong QC results directory.

Checks:

- inspect ``[groups.steps.overrides]`` in workflow file,
- ensure step output branch names match next step base branch,
- ensure ``fix_qc_results_dir`` points to previous detect run output.

4. Policy scattered in too many run profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptoms:

- small edits require changing many run files,
- inconsistent behavior across similar runs.

Fix:

- move shared policy into ``configs/catalogs`` / ``configs/rules``,
- keep run files focused on invocation scope and branch/runtime controls.

5. Legacy paths still embedded in docs/scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptoms:

- runtime errors on missing files under old directories.

Fix:

- grep for old path prefixes,
- migrate references,
- keep a short migration checklist in PR description.


Operational Best Practices
--------------------------

Repository hygiene
~~~~~~~~~~~~~~~~~~

- keep configs in git;
- review config changes like code;
- avoid storing transient editor checkpoint files.

Change management
~~~~~~~~~~~~~~~~~

- when changing catalogs/rules, note expected behavioral impact in PR;
- when changing workflows, describe branch hand-off graph;
- when changing run profiles, document intended invocation command.

Separation of concerns
~~~~~~~~~~~~~~~~~~~~~~

- authored configs under ``runs/catalogs/rules/workflows``;
- generated state under ``state``;
- avoid writing generated files into authored folders.

Documentation hygiene
~~~~~~~~~~~~~~~~~~~~~

Whenever config layout changes:

1. update ``configs/README.md``;
2. update this chapter;
3. update quickstart and examples that contain explicit paths;
4. update any tests that assert file locations.


Config Authoring Checklist
--------------------------

Before committing a new or updated profile:

- Does the file live in the correct folder for its role?
- Are branch names explicit for mutating stages?
- Are path references current (non-legacy)?
- Are reusable policies referenced via catalogs/rules rather than duplicated?
- Is the intended invocation command documented nearby?
- If ingest strict lock is used, are lockfiles present and current?


Cross References
----------------

- Quick start with practical commands: :doc:`quickstart`
- Run mode behavior and CLI patterns: :doc:`running_modes`, :doc:`modes`, :doc:`cli`
- Pipeline architecture and stage behavior: :doc:`pleb_deep_dive`
- Exact key-by-key configuration reference: :doc:`full_settings_catalog`
- Structured reference by topic: :doc:`configuration_reference`
- PQC detector internals and statistics:
  https://golamshaifullah.github.io/pqc/index.html


Configuration by Lifecycle Phase
--------------------------------

This section maps configuration concerns to the lifecycle of a typical
data-combination campaign.

Phase 1: Source onboarding
~~~~~~~~~~~~~~~~~~~~~~~~~~

Primary assets:

- ingest run profiles in ``configs/runs/ingest/``,
- ingest mapping catalogs in ``configs/catalogs/ingest/``,
- optional strict lock behavior under ``configs/state/lockfiles/``.

Primary risks:

- incorrect source root selection,
- wildcard patterns that over-match,
- provider naming drift not reflected in mappings.

Configuration controls that matter most:

- explicit source roots and mapping globs,
- ignore-suffix filters,
- source priority and alias handling,
- ingest verify toggles,
- lock strict mode decisions.

Phase 2: Dataset normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Primary assets:

- fixdataset run profiles in ``configs/runs/fixdataset/``,
- system flag mapping and rules under ``configs/catalogs/system_flags/``,
- overlap/relabel policy under ``configs/rules/``.

Primary risks:

- inconsistent ``-sys`` / ``-group`` assignment,
- stale or missing JUMPs,
- duplicate TOAs from overlapping files,
- accidental destructive action while iterating.

Configuration controls that matter most:

- ``fix_infer_system_flags`` and related overwrite controls,
- ``fix_insert_missing_jumps`` / ``fix_prune_stale_jumps``,
- overlap/relabel rule file paths,
- ``fix_qc_action`` and delete/comment policy,
- branch controls for change isolation.

Phase 3: Timing + QC detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Primary assets:

- pqc/pipeline run profiles,
- PQC backend profile rules in ``configs/rules/pqc/``,
- workflow files for serial/parallel staging.

Primary risks:

- over-flagging from overly aggressive thresholds,
- under-flagging from weak detector configuration,
- stage order ambiguity (detect/apply mixed in one run).

Configuration controls that matter most:

- ``pqc_*`` detector forwarded values,
- backend profile path and backend key selection,
- explicit outlier column strategy in apply stage,
- workflow group ordering and branch hand-off.

Phase 4: Reporting and review
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Primary assets:

- report toggles in run profile,
- report-only mode config,
- compact report settings and cross-pulsar coincidence settings.

Primary risks:

- reports generated from wrong run directory,
- action policy not aligned with report columns,
- reviewer confusion due to missing provenance metadata.

Configuration controls that matter most:

- run directory selection for report mode,
- compact report outlier/event column selection,
- cross-pulsar enable/window/minimum thresholds,
- reproducibility metadata capture.

Phase 5: Production reruns
~~~~~~~~~~~~~~~~~~~~~~~~~~

Primary assets:

- stable workflows,
- frozen catalogs/rules,
- lockfile snapshots and validation files.

Primary risks:

- silent drift in source tree or mappings,
- accidental use of legacy paths,
- ad hoc overrides not captured in PR or run notes.

Configuration controls that matter most:

- strict lock validation where appropriate,
- pinned workflow and step config references,
- explicit branch naming conventions,
- config review checklist in code review process.


Detailed Layout Inventory
-------------------------

This inventory explains the intended role of each current subtree and when to
add new files there.

``configs/runs/ingest/``
~~~~~~~~~~~~~~~~~~~~~~~~

Put files here when the primary objective is source ingestion and dataset tree
creation.

Good additions:

- a new ingest profile for a new public release structure,
- a profile for a specific institution mirror layout,
- profiles that differ only in lock strictness policy.

Do not add:

- workflow files,
- reusable mapping catalogs (put those under ``catalogs/ingest``),
- generated lock validation outputs.

``configs/runs/fixdataset/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Put files here when the main stage is dataset mutation policy.

Good additions:

- profile for branch-chained jump discovery,
- profile for par-default-only enforcement,
- profile for variant generation experiments.

Do not add:

- generic catalogs (variants/system tables belong under ``catalogs``),
- ad hoc one-off output snapshots.

``configs/runs/pqc/``
~~~~~~~~~~~~~~~~~~~~~

Put files here when the main stage is QC detection/action coordination.

Good additions:

- balanced detector profile,
- feature-hunt detector profile,
- apply-from-qc profile with non-destructive comments.

Do not add:

- detector internals documentation,
- mixed stage workflows better represented in ``configs/workflows``.

``configs/runs/pipeline/``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this folder for broad, mode-complete profiles that are intentionally
single-file.

Good additions:

- small reproducible examples for docs,
- compatibility profiles for release tags,
- CI smoke profiles.

Do not add:

- branch-chained operational workflows (prefer ``configs/workflows``),
- low-level rules/catalog entries.

``configs/workflows/``
~~~~~~~~~~~~~~~~~~~~~~

Use this folder for top-level run plans.

Good additions:

- branch-chained detect/apply plans,
- serial/parallel staging plans,
- campaign-specific named workflows.

Do not add:

- step profile internals directly inside the workflow file when reusable.

``configs/workflows/steps/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this folder for reusable per-step run profiles that workflows reference.

Good additions:

- standard detect step,
- standard apply-comments step,
- standard fix-and-variants step.

Do not add:

- free-standing run profiles not used by workflows.

``configs/catalogs/ingest/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this folder for ingest mappings and mapping templates.

Good additions:

- new source-tree mapping JSON for a new data release,
- sanitized mapping example files.

Do not add:

- runtime lock snapshots,
- mode run profiles.

``configs/catalogs/system_flags/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this folder for manual/curated mapping and rule-based system-frequency
rules.

Good additions:

- mapping allowlists,
- alias maps,
- YAML rules for file-specific system/group synthesis.

Do not add:

- execution outputs,
- ad hoc notes not consumed by code.

``configs/catalogs/system_tables/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this folder for stable lookup tables consumed by fix/legacy loaders.

Good additions:

- updated backend bandwidth table,
- jumps-per-system table updates,
- overlap table updates with review context.

Do not add:

- rules that imply conditional behavior (those belong in ``configs/rules``).

``configs/catalogs/variants/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this folder for backend-classification and include-variant catalogs.

Good additions:

- variant classification expansions (for example adding LOFAR/NENUFAR groups),
- additional variant definitions with clear naming.

Do not add:

- run profile toggles or branch settings.

``configs/rules/relabel/`` and ``configs/rules/overlap/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use these folders for rule-based mutation policy.

Good additions:

- relabel map for known backend renaming,
- overlap preference rules with action policy.

Do not add:

- static lookup catalogs (move to ``catalogs``),
- run-specific branch context.

``configs/rules/pqc/``
~~~~~~~~~~~~~~~~~~~~~~

Use this folder for per-backend detector-profile overrides.

Good additions:

- backend-specific thresholds for stable operational use,
- explicit profile sets for conservative/aggressive modes.

Do not add:

- generic run toggles unrelated to backend profile behavior.

``configs/schemas/``
~~~~~~~~~~~~~~~~~~~~

Use this folder for schema files used by tooling and UI.

``configs/state/lockfiles/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this folder for generated ingest lock snapshots and validation files.

Treat content as generated state:

- commit when needed for strict reproducibility workflows,
- avoid hand-editing unless troubleshooting requires forensic inspection.


Configuration Design Patterns
-----------------------------

Pattern: Baseline + override
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use one baseline TOML profile and apply temporary CLI ``--set`` overrides only
for short-lived experiments.

Benefits:

- reproducible default behavior,
- low diff footprint in git,
- easier run-to-run comparison.

Pattern: Static policy extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When multiple run profiles duplicate the same policy values, extract those into
catalog/rule files and reference paths from run profiles.

Benefits:

- single source of truth for policy,
- lower risk of drift across profiles,
- easier reviewer reasoning.

Pattern: Stage-specific branch isolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use different branches for:

- raw ingest outputs,
- fix normalization outputs,
- QC action outputs.

Benefits:

- clean audit trail,
- easy rollback between stages,
- safer experimentation.

Pattern: Workflow-driven hand-off
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use workflow overrides to propagate branch/result paths between stages.

Benefits:

- explicit stage input/output expectations,
- less manual command editing,
- fewer branch/path mismatches.


Configuration Anti-Patterns and Corrections
-------------------------------------------

Anti-pattern: Policy embedded in many run files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Problem:

- one policy update requires many edits,
- divergence appears across profiles.

Correction:

- move policy into ``configs/catalogs`` or ``configs/rules``,
- keep run files as thin run wrappers.

Anti-pattern: Generated state mixed with authored config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Problem:

- unclear what is source-of-truth policy vs runtime output.

Correction:

- keep generated lock and validation files under ``configs/state`` only.

Anti-pattern: Legacy path usage in new profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Problem:

- run failure when old directories no longer exist,
- confusion for new users reading examples.

Correction:

- update to standard paths documented in :ref:`config-layout`.

Anti-pattern: Single profile doing detect and apply without clear barriers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Problem:

- harder forensic review of what was detected versus what was applied.

Correction:

- split into detect/apply steps and chain via workflow branches.


Policy Ownership and Review Model
---------------------------------

For multi-contributor teams, define ownership boundaries:

- ingest mappings: data-integration owner,
- system tables and variants: timing harmonization owner,
- PQC backend profiles: QC methodology owner,
- workflows: operations owner.

Review expectations:

- mapping changes include before/after file match summaries,
- overlap/relabel rule changes include expected TOA impact notes,
- workflow changes include stage graph and branch hand-off explanation,
- lockfile updates include source drift rationale.


Auditing and Traceability
-------------------------

A configuration system is production-ready only if runs are traceable.

PLEB supports traceability by storing run settings and executed commands in run
outputs. To make this effective:

1. keep config files version controlled;
2. include config path and branch name in analysis notes;
3. avoid manual local edits without committing;
4. use distinct ``outdir_name`` values for major stage transitions.


Scenario Playbooks
------------------

Scenario A: First-time ingest for a new release mirror
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. copy a template from ``configs/runs/ingest``;
2. point to a mapping in ``configs/catalogs/ingest``;
3. run with lock strictness disabled;
4. inspect ingest reports for missing/extra files;
5. adjust mapping and rerun;
6. freeze lockfiles under ``configs/state/lockfiles``.

Scenario B: Add new system/group assignment policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. update mapping/rules under ``configs/catalogs/system_flags``;
2. run fixdataset profile with ``fix_infer_system_flags=true``;
3. inspect resulting tim flags on a pilot pulsar subset;
4. promote to full dataset only after pilot validation.

Scenario C: Tighten PQC while preserving TOA retention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. duplicate balanced profile under ``configs/runs/pqc``;
2. adjust per-backend policy in ``configs/rules/pqc``;
3. run detect-only step in workflow;
4. inspect compact report and per-backend action summary;
5. only then enable apply-comments/deletion policy.

Scenario D: Build two-pass branch-chained campaign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Step 1 workflow group: fix flags/jumps + generate variants;
2. Step 2 group: tempo2 + pqc detect;
3. Step 3 group: apply comments from Step 2 outputs;
4. verify each step starts from previous step branch, not main.


Path Migration Cookbook
-----------------------

If you need to modernize historical configs quickly:

1. replace old prefixes:

   - ``configs/system_tables`` -> ``configs/catalogs/system_tables``
   - ``configs/runs/workflow_steps`` -> ``configs/workflows/steps``
   - ``configs/settings/*.lock*.json`` -> ``configs/state/lockfiles/*.json``

2. verify Python defaults that embed paths (system tables, overlap catalog).
3. verify docs snippets and quickstart commands.
4. verify integration tests reference new paths.


Validation Strategy for Config Changes
--------------------------------------

For non-trivial config refactors, use a layered validation strategy:

Layer 1: static checks
~~~~~~~~~~~~~~~~~~~~~~

- ensure referenced files exist,
- ensure no legacy path prefixes remain,
- ensure workflow step refs resolve.

Layer 2: pilot execution
~~~~~~~~~~~~~~~~~~~~~~~~

- run on one or two pulsars only,
- verify branch creation and output placement,
- verify expected variant/par/tim outputs.

Layer 3: campaign execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- run full pulsar set through workflow,
- inspect stage outputs and cross-stage hand-off,
- inspect compact report action summaries.

Layer 4: reproducibility rerun
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- rerun same profiles with strict lock validation where applicable,
- verify reproducible behavior and stable output file set.


What To Put In PR Descriptions For Config Changes
-------------------------------------------------

A good config-change PR should include:

1. objective (what behavior changes),
2. files changed by role (`runs`, `catalogs`, `rules`, `workflows`, `state`),
3. expected run impact (which stages, which branches),
4. pilot validation summary,
5. migration notes if paths changed,
6. rollback strategy.

This is especially important for workflow and lockfile updates where side
effects are broad.


Reference Appendices
--------------------

The following chapters provide detailed lookup material.

.. toctree::
   :maxdepth: 1

   config_layout
   configuration_reference
   full_settings_catalog
