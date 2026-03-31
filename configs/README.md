# Config Layout and Usage

This directory contains all declarative configuration used by `pleb`.

The layout is intentionally split by role so users can separate:

- runnable profiles,
- reusable catalogs,
- rule files,
- workflow orchestration,
- generated run state.

The top-level principle is:

- `configs/runs/*` are things you execute,
- `configs/catalogs/*` are shared data assets,
- `configs/rules/*` are behavior policies,
- `configs/workflows/*` are multi-step execution plans,
- `configs/state/*` is generated runtime state.

## Directory map

### `configs/runs/`

Runnable TOML profiles passed to `pleb --config ...`.

Subfolders are mode-oriented:

- `ingest/`: ingest-mode and ingest-like profiles.
- `fixdataset/`: fix-focused profiles, including jump discovery and variants.
- `pqc/`: PQC-driven profiles (detection and apply variants).
- `pipeline/`: general end-to-end or legacy mixed profiles.

Use these as templates and copy into your project-specific profile files.

UX can generate these with increasing completeness:

- ``pleb init --mode pipeline --level minimal``
- ``pleb init --mode pipeline --level balanced``
- ``pleb init --mode pipeline --level full``

And can scaffold a chained three-pass workflow:

- ``pleb init --workflow-template 3pass-clean --outdir configs``

### `configs/workflows/`

Workflow mode files passed to `pleb workflow --file ...`.

- Top-level workflow files define stage groups, ordering, and step overrides.
- `workflows/steps/` stores reusable per-step config fragments.

This is where you encode serial/parallel orchestration without hardcoding in Python.

### `configs/catalogs/`

Reusable static/declarative data consumed by runs.

- `catalogs/ingest/`: ingest source mappings.
- `catalogs/public_releases/`: public release provider definitions for compare mode.
- `catalogs/system_flags/`: editable mappings/rules for `-sys` / telescope/backend inference.
- `catalogs/system_tables/`: system-level lookup tables used by fix logic.
- `catalogs/variants/`: variant definitions and backend classification tables.

These files should be version-controlled and reviewed like code.

### `configs/rules/`

Behavior rules (policy), not source catalogs.

- `rules/relabel/`: declarative relabel actions.
- `rules/overlap/`: overlap and TOA preference rules.
- `rules/pqc/`: per-backend PQC profile overrides.

Rules are expected to evolve as data idiosyncrasies are discovered.

### `configs/schemas/`

Machine-readable schema assets.

- GUI settings schema.
- Ingest mapping JSON schema.

These are for validation and tooling, not execution by themselves.

### `configs/state/`

Generated state artifacts.

- `state/lockfiles/`: ingest lock and lock-validation snapshots.

Treat this as runtime/reproducibility state, not as primary authoring inputs.

### `configs/settings/`

Legacy location from older layout revisions.

Current code prefers `configs/runs`, `configs/catalogs`, `configs/rules`, and `configs/state`.

Do not add new config assets to `configs/settings`.

## Naming conventions

Use names that communicate mode and intent.

Recommended pattern for run profiles:

- `<mode>_<purpose>.toml`
- examples: `pqc_balanced.toml`, `fixdataset_par_defaults.toml`

Recommended pattern for workflow files:

- `<dataset_or_scope>_<intent>.toml`
- examples: `j1713_stress_feature_hunt.toml`

Recommended pattern for catalogs:

- `<domain>_<scope>.<ext>`
- examples: `backend_classifications_legacy_new.toml`, `providers.toml`

## Authoring boundaries

Use this rule to keep configs maintainable:

- Put environment-independent policy in catalogs/rules.
- Put run-specific paths/branches in run profiles.
- Put multi-step branching/ordering in workflows.
- Keep generated lockfiles under `configs/state/lockfiles`.

Avoid embedding one-off absolute paths into shared catalogs unless they are intentionally local-only.

## How files are consumed

### Pipeline mode

`pleb --config configs/runs/pipeline/<file>.toml`

Consumes one run profile and optional catalog/rule paths referenced by that profile.

Optional white-noise estimation stage (EFAC/EQUAD/ECORR):

- `run_whitenoise = true`
- `whitenoise_source_path = "/work/git_projects/whitenoise/src"` (optional fallback)
- `whitenoise_epoch_tolerance_seconds = 1.0`
- `whitenoise_single_toa_mode = "combined"` (`combined`, `equad0`, `ecorr0`)
- `whitenoise_fit_timing_model_first = true`
- `whitenoise_timfile_name = "{pulsar}_all.tim"` (optional template override)

Outputs are written to:

- `<results>/<outdir>/<tag>/whitenoise/<branch>/whitenoise_summary.tsv`

### Ingest mode

`pleb ingest --config configs/runs/ingest/<file>.toml`

Consumes an ingest run profile and a mapping file from `configs/catalogs/ingest/`.

### Workflow mode

`pleb workflow --file configs/workflows/<file>.toml`

Consumes workflow file + referenced step configs from `configs/workflows/steps/` or `configs/runs/...`.

### QC report mode

`pleb qc-report --config <qc_report_profile.toml>`

Consumes report config and existing run outputs.

## CLI/TOML override behavior

`pleb` supports TOML + CLI override coexistence.

- TOML provides baseline settings.
- CLI `--set key=value` overrides specific keys.
- Workflow step overrides can chain branch names between stages.

This enables reproducible base profiles with explicit experimental deltas.

## Reproducibility guidance

For stable production runs:

1. Keep mapping catalogs and rule files in git.
2. Freeze ingest lockfiles after baseline validation.
3. Use workflow files to encode stage order and branch hand-off.
4. Store run settings alongside run outputs (pleb does this automatically).

## Suggested project workflow

1. Start from an ingest profile in `configs/runs/ingest/`.
2. Use rule files in `configs/rules/` for overlap/relabel policy.
3. Use variant/catalog files in `configs/catalogs/variants/`.
4. Run via a workflow in `configs/workflows/` for branch-chained staging.
5. Preserve generated lockfiles under `configs/state/lockfiles/`.

## Current key files (quick index)

### Ingest mappings

- `configs/catalogs/ingest/ingest_mapping_epta_data.json`
- `configs/catalogs/ingest/ingest_demo_mapping.json`
- `configs/catalogs/ingest/ingest_eptadr3.json`

### System/group inference

- `configs/catalogs/system_flags/system_flag_mapping.ingest_epta_data.json`
- `configs/catalogs/system_flags/flag_sys_freq_rules.yaml`

### System tables

- `configs/catalogs/system_tables/jumps_per_system.json`
- `configs/catalogs/system_tables/backend_bw.json`
- `configs/catalogs/system_tables/overlapped_timfiles.toml`
- `configs/catalogs/system_tables/pta_systems.json`

### Variant catalogs

- `configs/catalogs/variants/backend_classifications_legacy_new.toml`
- `configs/catalogs/variants/alltim_variants_legacy_new.toml`

### Rules

- `configs/rules/pqc/backend_profiles.example.toml`
- `configs/rules/relabel/relabel_rules.example.toml`
- `configs/rules/overlap/overlap_rules.example.toml`

### Workflows

- `configs/workflows/branch_chained_fix_pqc_variants.toml`
- `configs/workflows/j1713_j1022_stress_parallel_serial.toml`

## Maintenance checklist

When adding a new config file:

1. Place it in the role-appropriate directory.
2. Use consistent naming.
3. Add comments at top with expected invocation command.
4. Ensure paths referenced from docs are updated.
5. Keep sample/example files clearly marked as examples.

When changing layout:

1. Update `configs/README.md` (this file).
2. Update docs (`docs/configuration.rst`, `docs/quickstart.rst`, etc.).
3. Update Python defaults if path-backed (`pleb/system_tables.py`, etc.).
4. Update tests that reference old paths.

## Known anti-patterns

Avoid:

- mixing generated files with canonical config files,
- adding new run profiles under `configs/settings/`,
- duplicating identical policy data across multiple run files,
- hardcoding branch names in shared templates when they should be overridden,
- storing transient notebook checkpoint files in config trees.

## Migration note

If you have old references to:

- `configs/system_tables/...` use `configs/catalogs/system_tables/...`
- `configs/runs/workflow_steps/...` use `configs/workflows/steps/...`
- `configs/settings/*.lock*.json` use `configs/state/lockfiles/...`
