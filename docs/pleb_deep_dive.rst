PLEB Deep Dive
==============

This chapter is a practical deep dive into how ``pleb`` works as an
orchestration and dataset-management tool.

It is intentionally focused on **PLEB behavior**:

- configuration and mode boundaries,
- stage orchestration (ingest, fix, tempo2, QC, reporting),
- variant generation,
- output layouts,
- reproducibility controls,
- CLI and TOML parity,
- troubleshooting and operational patterns.

It does **not** duplicate detector internals from ``pqc``. Where detector
statistics are relevant, this chapter links out to PQC documentation.


Scope and Mental Model
----------------------

At a high level, ``pleb`` is a coordination layer that:

1. Ingests and normalizes timing inputs from heterogeneous source trees.
2. Applies deterministic dataset fixes and policy-driven edits.
3. Runs ``tempo2`` products.
4. Optionally runs ``pqc`` over generated par/tim combinations.
5. Generates reports and action artifacts.
6. Writes reproducibility metadata (settings, command, summaries).

Think of PLEB as two things at once:

- a **state machine** for data products,
- a **policy engine** driven by TOML/CLI controls.

The state machine perspective is useful because most confusion in production
runs comes from mixing stages that read from different roots/branches/results
locations. The policy engine perspective is useful because PLEB is designed so
you can swap strategy with config changes rather than code edits.


What PLEB Owns vs What PQC Owns
-------------------------------

PLEB owns:

- input discovery and copy/verify semantics,
- branch and commit orchestration for ingest/fix stages,
- tim/par editing policy (flags, jumps, comments, deletions),
- run directory and report artifact lifecycle,
- execution topology (serial vs parallel by stage/group),
- compatibility between CLI and TOML settings.

PQC owns:

- detector execution and detector-level labels,
- event/outlier statistical decisions at row level,
- feature extraction and detector diagnostics.

In short: PLEB tells the system **what to run, where to read/write, and how to
apply outputs**; PQC computes detector-level QC labels.


Run Modes and Entry Points
--------------------------

PLEB has multiple run surfaces. The most common are:

- ``pleb --config <pipeline.toml>`` (pipeline mode)
- ``pleb ingest --config <ingest.toml>`` (ingest-only mode)
- ``pleb workflow --file <workflow.toml>`` (multi-step orchestration mode)
- ``pleb qc-report --run-dir <run_dir>`` (report-only mode)

Each mode has a narrow responsibility:

Pipeline mode
~~~~~~~~~~~~~

Runs stage toggles from one configuration object (tempo2, fix dataset, pqc,
reports, etc.) and writes a timestamped run directory under ``results_dir`` by
default.

Ingest mode
~~~~~~~~~~~

Copies source par/tim assets into an output dataset root using mapping rules.
Can optionally commit ingest changes to a branch.

Workflow mode
~~~~~~~~~~~~~

Runs grouped steps in serial/parallel orchestration. Useful for two-pass
operations (for example: detect first, apply actions second).

QC report mode
~~~~~~~~~~~~~~

Consumes existing ``*_qc.csv`` files and writes diagnostics/plots/PDF without
rerunning PQC.


Configuration Architecture
--------------------------

Configuration is intentionally flat at the ``PipelineConfig`` layer so one
config file can express a complete run when needed. Under the hood, settings
cluster into operational groups.

Core path and identity group
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These keys determine where data is read/written:

- ``home_dir``
- ``dataset_name``
- ``results_dir``
- ``singularity_image``
- ``branches``
- ``reference_branch``
- ``pulsars``

These fields are the first place to check when runs appear to operate on an
unexpected tree.

Pipeline toggle group
~~~~~~~~~~~~~~~~~~~~~

These keys determine which stages run:

- ``run_tempo2``
- ``run_fix_dataset``
- ``run_pqc``
- ``qc_report``
- ``make_*`` report/plot toggles

FixDataset policy group
~~~~~~~~~~~~~~~~~~~~~~~

These keys govern all tim/par mutation behavior:

- apply/branch controls: ``fix_apply``, ``fix_branch_name``, ``fix_base_branch``
- system flag behavior: ``fix_infer_system_flags``, overwrite policies,
  mapping/table paths
- overlap and dedupe controls
- jump insertion/pruning controls
- par default controls (EPHEM/CLK/NE_SW)
- QC action controls (comment/delete, selected columns)
- variant generation controls for ``_all`` tim variants and jump-reference par
  variants

PQC pass-through group
~~~~~~~~~~~~~~~~~~~~~~

Keys starting with ``pqc_`` are passed through to PQC execution configuration.
PLEB treats these as detector configuration payload, not as internal detector
logic.

See:

- PLEB configuration reference: :doc:`configuration_reference`
- PQC docs: https://golamshaifullah.github.io/pqc/index.html

Reporting group
~~~~~~~~~~~~~~~

PLEB has two reporting layers:

- ``qc_report_*`` for per-run report generation from QC CSVs,
- cross-pulsar post-QC coincidence report keys
  (``qc_cross_pulsar_*``) for optional multi-pulsar clustering.

CLI ↔ TOML parity model
~~~~~~~~~~~~~~~~~~~~~~~

PLEB is designed so all relevant run controls are settable from TOML and can be
overridden by CLI ``--set``.

Precedence model:

1. config file defaults/values,
2. ``--set key=value`` overrides,
3. explicit mode CLI options where applicable.

Operationally, treat TOML as your baseline profile and ``--set`` as a temporary
override mechanism for experimental runs.


Stage-by-Stage Execution Flow
-----------------------------

This section describes what each stage does in PLEB terms.

Stage 0: Input resolution and guards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before heavy work starts, PLEB resolves:

- absolute paths for key roots,
- target pulsar list,
- branch scope,
- run output directory.

It also applies guardrails:

- repository cleanliness warnings where relevant,
- branch existence checks and fallback behavior,
- optional lockfile validation in ingest.

Common failure signatures at this stage are path/branch mismatches.

Stage 1: Ingest (optional, mode-specific or workflow step)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ingest is responsible for creating a working dataset tree from source roots.

Core ingest responsibilities:

- discover matching tim/par inputs from mapping rules,
- copy into output dataset root,
- verify expected tim copies,
- emit ingest reports,
- optionally commit ingest changes.

Ingest lockfiles
^^^^^^^^^^^^^^^^

When enabled, lockfile validation compares current source-tree match sets to a
previous lock snapshot. This is useful in stable production ingestion where
unexpected source drift should fail fast.

Two common operational patterns:

- exploratory ingestion: lock strictness disabled,
- production rerun/CI ingestion: lock strictness enabled and lock tracked.

If lock strict mode is enabled and new files appear in sources, ingest aborts by
design to protect reproducibility.

Stage 2: FixDataset (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FixDataset is where PLEB applies deterministic policy edits to tim/par products.
This is usually the highest-leverage stage for harmonization.

Typical FixDataset actions include:

- ensuring/normalizing flags (`-sys`, `-group`, `-pta`, etc.),
- deduplicating overlapping TOAs according to configured rules,
- inserting missing JUMPs from discovered systems,
- pruning stale JUMPs if requested,
- updating par defaults (EPHEM/CLK/NE_SW) with force/no-force controls,
- generating include variants (``*_all.<variant>.tim``),
- optionally generating jump-reference variant par files.

Non-destructive default patterns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When possible, PLEB defaults toward comment-based operations and explicit
opt-in for destructive edits. This is intentional for review-first workflows.

Branch behavior in FixDataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``fix_apply=true`` and branch controls are set:

- PLEB checks out base branch,
- creates/updates target branch,
- writes edits,
- commits changes.

If branch controls are omitted, behavior follows local working context and may
operate without branch creation.

Stage 3: tempo2 products (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tempo2 stage runs pulsar fits/products for selected pulsars and branches.
PLEB manages process orchestration, output routing, and artifact placement.

PLEB controls:

- which pulsars and branches run,
- worker parallelism via ``jobs``,
- singularity prefix usage and command invocation,
- product output placement.

PLEB does **not** reinterpret tempo2 fit physics. It orchestrates execution and
collects products.

Stage 4: PQC execution (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``run_pqc=true``, PLEB executes PQC per pulsar and writes per-pulsar QC CSVs
into the run’s QC directory.

PLEB handles:

- process spawning and concurrency,
- per-pulsar settings TOML snapshots under run settings,
- error capture per pulsar (skip-fail behavior),
- summary TSV aggregation.

Detector internals are not in scope here; see:
https://golamshaifullah.github.io/pqc/index.html

Stage 5: Post-QC application in FixDataset (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This stage is logically part of FixDataset policy, but often run as a second
pass after PQC outputs exist.

PLEB can read QC outputs and apply policy actions:

- comment/delete TOAs based on selected QC columns,
- separate handling for outliers vs event-tagged subsets,
- per-action prefixes for traceability.

Recommended workflow is comment-first, then manual review, then stricter action.

Stage 6: Reporting (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PLEB report outputs can include:

- QC diagnostics and plots,
- compact PDF triage report,
- per-backend action CSVs,
- structure summary tables.

New optional post-QC stage can also generate cross-pulsar coincidence artifacts
(``qc_cross_pulsar_*``) without modifying source tim/par files.


Execution Topology: Serial, Parallel, Grouped Workflows
--------------------------------------------------------

PLEB supports multiple parallelism layers:

- within pipeline runs via ``jobs`` (per branch/pulsar task fan-out),
- within workflow mode via group-level ``serial``/``parallel`` blocks.

This enables patterns like:

1. parallel PQC detection across pulsars,
2. barrier synchronization,
3. parallel fix application across pulsars based on completed QC outputs.

Workflow orchestration is especially useful when you need explicit stage
barriers between detection and mutation.


Variant Generation in PLEB
--------------------------

PLEB supports generation of multiple include variants and (optionally) variant
par products, depending on enabled settings.

Include variants
~~~~~~~~~~~~~~~~

Using backend classification and variant maps, PLEB can produce:

- ``JXXXX_all.tim`` (combined),
- ``JXXXX_all.<variant>.tim`` (for example ``legacy``, ``new``, ``combined``).

Variants are data-selection products: they define which backend subsets are fed
to downstream stages.

Jump-reference variant par products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When enabled, PLEB can build variant-specific par files with consistent JUMP
blocks and selected reference-system behavior.

This feature is orchestration-heavy and depends on correct include and system
classification state. If expected variant par files are missing, verify:

- variant generation toggles are enabled,
- include variant files were generated,
- jump-reference variant stage is enabled,
- mode/branch/path values point to the same dataset root.


Output Layout and Artifact Contracts
------------------------------------

PLEB run outputs are organized by run tag under ``results_dir`` unless an
explicit output name is provided.

Common run directories:

- ``<run_tag>/run_settings/``:
  - ``command.txt`` (exact CLI used),
  - effective config snapshots.
- ``<run_tag>/qc/``:
  - per-pulsar ``*_qc.csv``,
  - ``qc_summary.tsv``.
- ``<run_tag>/qc_report/``:
  - diagnostics, plots, compact report artifacts.
- ``<run_tag>/qc_cross_pulsar/`` (optional):
  - cross-pulsar coincidence tables.

In dataset roots (not report roots), FixDataset may create:

- updated ``.tim`` and ``.par`` products,
- ``*_all*.tim`` variant includes,
- variant par files when enabled.

Artifact placement principle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operationally, keep **analysis/report artifacts** under results directories and
keep **canonical data products** under dataset roots.

If you see temporary analysis artifacts in pulsar data folders, that usually
indicates a stage writing to a fallback location due to a missing output-dir
setting.


Reproducibility Patterns
------------------------

PLEB provides several reproducibility levers:

1. command capture in ``run_settings/command.txt``,
2. config snapshot capture in run settings,
3. optional ingest lockfile validation,
4. branch-based mutation workflows,
5. deterministic mapping and policy files (classification/relabel/overlap).

Recommended reproducibility baseline:

- pin and track core settings TOML,
- track classification/relabel/overlap catalogs,
- use explicit branch names per workflow pass,
- keep ingest lock strict mode for production reruns,
- keep run settings artifacts for every generated report tag.

Branch strategy for reproducible mutation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A robust pattern for two-pass pipelines:

1. start from immutable base branch,
2. run detection branch (no edits or comment-only),
3. run apply branch from detection branch outputs,
4. compare branch diffs and retain run artifacts.

This preserves provenance from detection policy to final file edits.


PQC Integration (PLEB Pass-Through Only)
----------------------------------------

PLEB does not implement detector internals. It passes configuration to PQC and
consumes resulting CSV outputs.

In practical terms, PLEB provides:

- per-run and per-backend pass-through settings via ``pqc_*`` keys,
- per-backend profile file support,
- controlled application of resulting QC labels in FixDataset,
- report generation over QC CSV outputs.

For detector semantics, statistical assumptions, and event/outlier criteria,
use PQC documentation:

- https://golamshaifullah.github.io/pqc/index.html


Minimal End-to-End Example
--------------------------

This example shows one practical production-friendly pattern:

1. ingest,
2. fix + tempo2 + pqc,
3. report and optional cross-pulsar coincidence,
4. comment-only QC actions.

Example TOML (single-run baseline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   # Identity / paths
   home_dir = "/work/git_projects/epta_dr3alpha"
   dataset_name = "."
   singularity_image = "/work/git_projects/PSR_Singularity/psrpta.sif"
   results_dir = "results"

   # Scope
   branches = ["main"]
   reference_branch = "main"
   pulsars = "ALL"
   jobs = 10

   # Stage toggles
   run_fix_dataset = true
   fix_apply = true
   run_tempo2 = true
   run_pqc = true
   qc_report = true
   qc_report_compact_pdf = true

   # Fix branch controls
   fix_base_branch = "raw_ingest"
   fix_branch_name = "fixdataset_with_qc_comments"
   fix_commit_message = "FixDataset: harmonize flags, jumps, and QC comments"

   # Fix dataset policy (non-destructive first)
   fix_infer_system_flags = true
   fix_insert_missing_jumps = true
   fix_prune_stale_jumps = false
   fix_qc_remove_outliers = true
   fix_qc_action = "comment"
   fix_qc_outlier_cols = ["bad_point", "robust_outlier", "robust_global_outlier", "bad_mad"]

   # Variant generation
   fix_generate_alltim_variants = true
   fix_backend_classifications_path = "configs/settings/backend_classifications.toml"
   fix_alltim_variants_path = "configs/settings/alltim_variants.toml"

   # Par defaults
   fix_ensure_ephem = "DE440"
   fix_ensure_clk = "TT(BIPM2024)"
   fix_ensure_ne_sw = "7.9"
   fix_force_ne_sw_overwrite = false

   # PQC pass-through (detector details in PQC docs)
   pqc_backend_col = "group"
   pqc_backend_profiles_path = "configs/settings/pqc_backend_profiles.toml"

   # Optional post-QC cross-pulsar coincidence report
   qc_cross_pulsar_enabled = true
   qc_cross_pulsar_window_days = 1.0
   qc_cross_pulsar_min_pulsars = 2
   qc_cross_pulsar_include_outliers = true
   qc_cross_pulsar_include_events = true

Execution
~~~~~~~~~

.. code-block:: bash

   pleb --config /work/git_projects/pleb/configs/settings/pipeline.toml

What to validate after run
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. ``<run_tag>/run_settings/command.txt`` exists and matches intended command.
2. ``<run_tag>/qc/qc_summary.tsv`` has entries for targeted pulsars.
3. ``<run_tag>/qc_report/`` exists with expected diagnostics and compact PDF.
4. ``<run_tag>/qc_cross_pulsar/`` exists only when enabled.
5. target fix branch contains expected tim/par mutations.
6. variant include files exist in pulsar directories.


Workflow Example: Two-Pass Detect then Apply
---------------------------------------------

Use workflow mode when you want strict barriers:

- pass 1: detect and generate QC outputs in parallel,
- pass 2: apply comment actions from completed outputs.

Sketch:

.. code-block:: toml

   [[groups]]
   name = "detect"
   mode = "parallel"
   parallel_workers = 2

     [[groups.steps]]
     name = "pipeline"
     config = "configs/runs/pqc_balanced.toml"

   [[groups]]
   name = "apply"
   mode = "parallel"
   parallel_workers = 2

     [[groups.steps]]
     name = "pipeline"
     config = "configs/runs/fix_apply_from_qc.toml"

The key design point is barrier semantics between groups.


Common Failure Modes and Fast Diagnosis
---------------------------------------

This section lists frequent operational failures and how to diagnose them
quickly.

1) Wrong repository root for mutation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom:

- commits land in a nested data directory repo (or fail branch checks) instead
  of your intended parent repo.

Cause:

- ``home_dir``/dataset root points to a nested path that itself is a git repo,
  or branch controls target a different repository than expected.

Checks:

- run ``git rev-parse --show-toplevel`` in the exact directory PLEB uses,
- verify ``home_dir`` and ``dataset_name`` resolve to intended repo root.

Fix:

- point run config to the intended repository root,
- avoid ambiguous nested repo structures for canonical pipelines.

2) Ingest lockfile aborts with source-tree changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom:

- ingest aborts with added/missing source paths compared to lockfile.

Cause:

- source files changed after lock generation; strict validation is active.

Checks:

- inspect lock validation report and listed path deltas.

Fix:

- exploratory run: disable strict lock validation,
- production run: regenerate lockfile intentionally and commit it.

3) Missing JUMPs despite expected systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom:

- systems appear in tim flags but corresponding JUMPs are absent in par files.

Cause:

- jump insertion disabled, wrong jump flag column, or QC/fix source mismatch.

Checks:

- verify ``fix_insert_missing_jumps=true``,
- verify ``fix_jump_flag`` matches system label convention,
- verify ``fix_qc_results_dir`` and ``fix_qc_branch`` point to intended QC run.

Fix:

- align jump and QC source config; rerun FixDataset apply stage.

4) System/group inference unexpected for specific backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom:

- ``-sys`` or ``-group`` labels not matching expected center semantics.

Cause:

- backend-specific snapping rules, filename-derived center interpretation,
  overwrite policy, or relabel post-rules.

Checks:

- inspect effective inference settings and mapping/relabel files,
- verify overwrite controls and backend-specific toggles.

Fix:

- apply explicit mappings or relabel rules; rerun FixDataset.

5) PQC stage silently skipped or mostly failed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom:

- QC summary sparse; warnings about per-pulsar PQC failures.

Cause:

- missing PQC dependency/runtime, wrong par/tim path, serialization/config
  mismatch in subprocess payload.

Checks:

- confirm PQC importability in runtime env,
- inspect per-pulsar warnings and settings snapshots in run settings.

Fix:

- fix environment and path assumptions; rerun PQC-enabled stage.

Detector details are in PQC docs, not here.

6) Variant files missing even with variant keys configured
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom:

- ``*_all.<variant>.tim`` or variant par products not generated.

Cause:

- generation toggles disabled, missing classification paths, running mode not
  invoking fix stage, or writing to different dataset root than expected.

Checks:

- verify ``fix_generate_alltim_variants=true``,
- verify classification/variant TOML paths resolve correctly,
- verify run is actually in FixDataset apply path.

Fix:

- enable required toggles and rerun from intended branch/root.

7) Parallelism not observed despite high ``jobs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom:

- only one core appears active.

Cause:

- stage does not fan out at that point, or workload currently in serial segment
  (for example staged barriers in workflow).

Checks:

- verify ``jobs`` value in effective config snapshot,
- verify stage supports fan-out (tempo2/PQC segments do),
- verify workflow group mode for current step.

Fix:

- use parallel-capable stage with sufficient pulsar fan-out; configure workflow
  groups accordingly.


Operational Checklists
----------------------

Pre-run checklist
~~~~~~~~~~~~~~~~~

1. Confirm ``home_dir`` and ``dataset_name`` resolve to intended tree.
2. Confirm branch controls are set for mutation stages.
3. Confirm ingestion mapping and source availability.
4. Confirm container/runtime path to ``tempo2``.
5. Confirm PQC dependency if ``run_pqc=true``.
6. Confirm classification/relabel/overlap catalogs are in place.

Post-run checklist
~~~~~~~~~~~~~~~~~~

1. Verify run tag printed by CLI exists.
2. Check ``run_settings/command.txt``.
3. Check per-stage expected artifacts.
4. Check branch diffs match intended mutation policy.
5. Check QC compact report/action lists before destructive actions.

Promotion checklist (toward production profile)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. lock and track ingestion inputs,
2. freeze strategy TOMLs and variant catalogs,
3. enforce branch naming and commit message conventions,
4. run workflow with explicit serial/parallel group barriers,
5. archive run settings and report artifacts with branch references.


Practical Patterns for Stable Teams
-----------------------------------

Pattern A: Config-first governance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep all behavior changes in versioned TOML and catalog files:

- run profiles,
- backend classifications,
- overlap/relabel rules,
- optional backend profile overrides for PQC pass-through.

Avoid ad hoc CLI-only changes in production jobs except for explicit temporary
experiments.

Pattern B: Two-phase review loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. first pass with comment-only action policy,
2. human review of compact report and per-backend action CSVs,
3. second pass for stricter action only if justified.

Pattern C: Branch-per-pass provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use explicit branch lineage:

- ``raw_ingest`` -> ``fix_detect`` -> ``fix_apply``

This gives traceable provenance for every mutation.

Pattern D: Split data products from analytics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep mutation outputs in dataset branches and analytics outputs in results
directories. This reduces accidental pollution of canonical data trees.


Cross-References
----------------

For PLEB usage and settings:

- :doc:`quickstart`
- :doc:`configuration`
- :doc:`configuration_reference`
- :doc:`full_settings_catalog`
- :doc:`running_modes`
- :doc:`cli`
- :doc:`flow_diagrams`

For PQC detector details (statistical internals, detector semantics, event
models, and threshold interpretation):

- https://golamshaifullah.github.io/pqc/index.html


Detailed PLEB Config Groups (Operational View)
----------------------------------------------

This section expands the core config groups with an operations-first framing.
It intentionally stays at PLEB orchestration level.

Repository and dataset targeting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These keys decide *what tree* PLEB mutates and analyses:

- ``home_dir``: root context for pipeline mode operations.
- ``dataset_name``: dataset path/name under ``home_dir``.
- ``branches``: branch list used by comparison/report loops.
- ``reference_branch``: baseline branch for comparison reports.

Operational implications:

- A wrong ``home_dir`` can produce valid-looking outputs in the wrong repo.
- A wrong ``dataset_name`` can silently point to an empty/alternate tree.
- Branch defaults may differ across cloned repos; set explicitly in shared
  team configs.

Input scope and performance controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``pulsars``: ``"ALL"`` or explicit list.
- ``jobs``: worker fan-out for parallelizable segments.
- ``force_rerun``: rerun expensive steps even when products exist.

Operational implications:

- ``pulsars="ALL"`` is convenient but high-cost for iterative debugging.
- ``jobs`` affects throughput only in stages that fan out by pulsar.
- ``force_rerun`` should be explicit in benchmarking and regression tests.

Stage toggle controls
~~~~~~~~~~~~~~~~~~~~~

Core toggles:

- ``run_tempo2``
- ``run_fix_dataset``
- ``run_pqc``
- ``qc_report``
- plotting/report ``make_*`` toggles

Operational implications:

- Stage toggles should be treated as profile-level intent.
- Use separate profiles for "detect-only", "apply-only", and "full pipeline".
- Avoid one giant mutable config for all purposes; use composable profiles.

FixDataset branch/mutation controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mutation targeting:

- ``fix_apply``
- ``fix_base_branch``
- ``fix_branch_name``
- ``fix_commit_message``
- ``fix_backup``, ``fix_dry_run``

Operational implications:

- If branch controls are omitted, local working state determines behavior.
- ``fix_dry_run=true`` is useful for validating strategy before committing.
- Set explicit branch names in CI to prevent accidental branch drift.

Flag and jump harmonization controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Flag controls:

- ``fix_infer_system_flags``
- ``fix_system_flag_overwrite_existing``
- ``fix_system_flag_mapping_path``
- ``fix_system_flag_table_path``

Jump controls:

- ``fix_insert_missing_jumps``
- ``fix_jump_flag``
- ``fix_prune_stale_jumps``

Operational implications:

- Flag inference, jump insertion, and overlap rules are coupled in practice.
- If jumps are missing, confirm both flag availability and jump flag column.
- Keep mapping/table files versioned; these are part of reproducibility.

Dedupe and overlap controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``fix_dedupe_toas_within_tim``
- ``fix_dedupe_mjd_tol_sec``
- ``fix_dedupe_freq_tol_mhz``
- ``fix_dedupe_freq_tol_auto``
- ``fix_remove_overlaps_exact``
- overlap/relabel catalog paths

Operational implications:

- Aggressive dedupe can remove scientifically important rows if tolerance is too
  broad.
- Exact-overlap catalogs should be reviewed as explicit policy, not treated as
  hidden defaults.
- Keep "comment first" posture where ambiguity exists.

Par-default enforcement controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``fix_ensure_ephem``
- ``fix_ensure_clk``
- ``fix_ensure_ne_sw``
- ``fix_force_ne_sw_overwrite``

Operational implications:

- Non-force update mode is safer for preserving curated per-pulsar values.
- Force mode should be used only when standardization policy is explicit.

Variant generation controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``fix_generate_alltim_variants``
- ``fix_backend_classifications_path``
- ``fix_alltim_variants_path``
- ``fix_jump_reference_variants``
- ``fix_jump_reference_*``

Operational implications:

- Variant generation depends on both classification and include rules.
- Missing variant outputs are usually config wiring issues, not runtime errors.
- Keep variant catalogs human-editable and reviewed in PRs.

QC application controls (PLEB action policy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``fix_qc_remove_outliers``
- ``fix_qc_outlier_cols``
- ``fix_qc_action``
- selective action toggles for solar/transient/orbital subsets

Operational implications:

- Explicit ``fix_qc_outlier_cols`` is safer than broad compatibility fields.
- For team workflows, default to comment-based application before deletion.

PQC pass-through controls
~~~~~~~~~~~~~~~~~~~~~~~~~

All ``pqc_*`` keys are treated as detector config payload and passed through.
PLEB does not reinterpret detector equations or statistical internals.

For semantics and tuning rationale:
https://golamshaifullah.github.io/pqc/index.html

QC reporting and post-QC synthesis controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``qc_report_*`` for report rendering behavior.
- ``qc_cross_pulsar_*`` for optional post-QC coincidence artifacts.

Operational implications:

- These controls affect output interpretation and triage artifacts.
- They do not directly mutate canonical tim/par files unless coupled with
  FixDataset action settings.


Ingest Deep Dive
----------------

Ingest is often the highest-risk stage for silent data drift. This section
focuses on operational details and guardrails.

How ingest resolves sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ingest mapping rules define:

- source root,
- tim/par match patterns,
- ignore suffixes/exclusions,
- destination naming behavior.

Key operational idea: ingest is deterministic relative to mapping + source tree
state at run time.

Why lockfiles exist
~~~~~~~~~~~~~~~~~~~

Lockfiles capture the discovered source set. They are not required for first
exploration but are recommended for stable reruns.

Recommended practice:

- run without strict lock during mapping development,
- then generate/freeze lockfiles when ingest behavior stabilizes,
- enable strict lock validation in production reruns.

Interpreting ingest verification output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Typical classes of ingest verification messages:

- all expected tim files copied,
- missing matched files from source group,
- multiple parfiles discovered for one pulsar,
- lockfile mismatch abort.

Each class indicates a distinct remediation path:

- missing matched files: mapping mismatch or source naming drift,
- multiple parfiles: needs source-priority policy or explicit selection,
- lockfile mismatch: source drift since lock generation.

Source-priority design
~~~~~~~~~~~~~~~~~~~~~~

When multiple sources contain overlapping filenames/TOAs, configure source
priority explicitly (for example newest release wins). This should be policy,
not ad hoc runtime behavior.

Operationally, write priority rules in mapping/config and review them as code.

Ingest branch behavior
~~~~~~~~~~~~~~~~~~~~~~

If ingest commit mode is enabled:

- PLEB applies branch targeting controls for ingest output repo.
- Base branch fallback behavior applies if requested branch is absent.

If your intent is to commit to a parent repo, ensure ingest output root points
to that exact repository and not a nested path.


FixDataset Deep Dive
--------------------

FixDataset is where most production logic lives. This section focuses on
composition of rules and expected side effects.

FixDataset as layered transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A practical layered model:

1. structural cleanup and include updates,
2. flag harmonization,
3. dedupe/overlap policy,
4. jump harmonization,
5. par default normalization,
6. variant generation,
7. optional QC-driven action application.

This layered model is useful because each layer has a different failure mode.

Layer coupling to watch
~~~~~~~~~~~~~~~~~~~~~~~

``-sys``/``-group`` inference and jumps:

- missing or inconsistent system labels can cascade into missing jump insertion.

Overlap policy and dedupe:

- overlapping file policy may hide or expose within-file duplicates depending on
  ordering and tolerance settings.

Variant generation and QC application:

- if variants are generated after QC assumptions were made on non-variant
  includes, policy mismatch can occur.

Recommended debug order for FixDataset issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. confirm source tim files exist where mapping expects them,
2. inspect inferred system/group values on a small pulsar subset,
3. inspect missing/stale jump warnings,
4. verify include variants generated as expected,
5. only then apply QC-driven mutation.

Par-default update policy
~~~~~~~~~~~~~~~~~~~~~~~~~

For NE_SW specifically, force/non-force behavior matters:

- non-force mode: preserve existing explicit values,
- force mode: standardize value regardless of existing entries.

For production reproducibility, keep this policy explicit in run profiles.


tempo2 Orchestration Deep Dive
------------------------------

PLEB wraps tempo2 execution as an orchestrated stage, not as a scientific
modeling layer.

Container/runtime expectations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PLEB can call tempo2 through configured singularity/apptainer context. Typical
issues arise from:

- wrong image path,
- image missing expected tempo2 binary,
- bind path mismatch to dataset root.

When tempo2 commands fail across many pulsars, validate environment first before
changing data policy.

Parallelism and pacing
~~~~~~~~~~~~~~~~~~~~~~

Throughput depends on:

- number of pulsars,
- enabled stages after tempo2,
- ``jobs`` setting,
- workflow barrier layout.

A single visibly active core may be normal in non-fan-out segments. Verify by
stage, not by total runtime alone.


Reporting and Review Deep Dive
------------------------------

PLEB report outputs are triage tools. Treat them as operational artifacts for
review loops, not as replacements for scientific judgment.

Compact report usage pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use compact report outputs to:

- identify highest-impact backend/system clusters,
- audit decision labels and reasons,
- generate reviewer action queues.

Then apply reviewed policy in FixDataset.

Per-backend action CSV workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each pulsar/backend action CSV:

1. verify decision reasons align with policy,
2. verify event-marked rows are treated per team convention,
3. mark acceptable edits,
4. rerun apply pass in controlled branch.

Cross-pulsar coincidence report usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optional cross-pulsar coincidence artifacts help find epochs where multiple
pulsars have flagged rows in the same time window. This is useful for:

- observatory-wide anomaly triage,
- calibration-epoch review,
- broad sanity checks before aggressive action.

This stage is intentionally reporting-only.


Failure Message Cookbook
------------------------

This subsection maps representative errors to actionable checks.

``pathspec '<branch>' did not match any file(s) known to git``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meaning:

- requested base branch does not exist in target repository.

Check:

- current target repo and local branch list.

Action:

- set correct base branch, or create/sync branch first.

``InvalidGitRepositoryError: <path>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meaning:

- stage expects a git repo at configured path but path is not a repo.

Check:

- resolved ``home_dir`` and ``dataset_name`` combination.

Action:

- point configuration to repository root used for mutation.

``Ingest lockfile validation failed: source tree changed ...``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meaning:

- strict lock validation detected drift since lock generation.

Check:

- lock validation delta summary.

Action:

- exploratory run: disable strict lock,
- production: regenerate lock intentionally and commit.

``Refusing to fit for bad JUMP ... had no data points in range``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meaning:

- par contains JUMP key for system not present in current tim selection.

Check:

- include files, overlap policy, and current tim inventory.

Action:

- prune stale jumps if policy allows, or restore missing data sources.

``pqc failed for <pulsar>; skipping QC for this pulsar``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meaning:

- per-pulsar QC subprocess failed; run continues.

Check:

- per-pulsar warning details and settings snapshot.

Action:

- fix environment/path/config issue and rerun stage.

``No *_qc.csv files found under <run_dir>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meaning:

- report stage could not find QC outputs.

Check:

- whether PQC stage was enabled and succeeded for same run dir.

Action:

- point report to correct run dir, or rerun PQC stage first.


Release/Upgrade Guidance
------------------------

When upgrading profiles or moving between release branches:

1. Diff your TOML profiles against updated defaults.
2. Verify renamed/deprecated keys in ``full_settings_catalog``.
3. Run a one-pulsar canary profile before full batch.
4. Validate run_settings snapshots and artifact paths.
5. Promote to full run only after canary parity checks.

Canary profile recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A useful canary profile should:

- include at least one pulsar with rich multi-backend data,
- enable all intended major stages,
- keep mutation action as comment-only,
- produce compact QC report and action CSVs.

This catches most wiring regressions with minimal cost.


Operational Recipes
-------------------

Recipe 1: Ingest-only validation run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal: verify mapping completeness and branch targeting.

1. Run ingest with strict lock disabled.
2. Inspect ingest verify messages and missing file list.
3. Fix mapping/catalogs.
4. Regenerate lock and rerun in strict mode.

Recipe 2: Harmonize flags/jumps without QC action
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal: establish clean base dataset before statistical QC application.

1. Enable FixDataset apply with system/jump harmonization.
2. Keep QC action disabled.
3. Validate include variants and par updates.
4. Commit to dedicated branch.

Recipe 3: Detection-first QC pass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal: generate QC labels/reports with no destructive changes.

1. Enable ``run_pqc`` and reporting.
2. Keep FixDataset QC action comment-only or disabled.
3. Review compact report and per-backend action CSVs.

Recipe 4: Controlled apply pass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal: apply reviewed QC policy to data files.

1. Start from detection branch.
2. Enable FixDataset QC action on explicit outlier columns.
3. Keep branch output isolated.
4. Validate diffs before merge.

Recipe 5: Cross-pulsar anomaly sweep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal: identify suspicious epochs shared across pulsars.

1. Enable ``qc_cross_pulsar_enabled=true``.
2. Set window and min pulsars threshold.
3. Review coincidence clusters before broad policy changes.


Summary
-------

PLEB is most effective when treated as a configuration-driven orchestration
platform:

- ingest and mutation behavior are explicit policy,
- detector behavior is delegated to PQC through pass-through config,
- report artifacts and run settings provide traceable evidence,
- branch and workflow controls enforce safe review-first operations.

For detector-level meaning, threshold interpretation, and event/outlier
statistics, use the PQC deep-dive documentation:
https://golamshaifullah.github.io/pqc/index.html
