# pleb - The EPTA Data Combination Pipeline

[![CI](https://github.com/golamshaifullah/pleb/actions/workflows/ci.yml/badge.svg)](https://github.com/golamshaifullah/pleb/actions/workflows/ci.yml)

`pleb` is a command-line pipeline for ingesting, normalizing, fitting, and
reporting PTA pulsar timing datasets.

Docs: https://golamshaifullah.github.io/pleb/
Config guide (TOML strategies): https://golamshaifullah.github.io/pleb/configuration_reference.html

## Core capabilities

- Mapping-driven ingest of raw `.par/.tim/.tmplts` into a canonical tree.
- Pipeline runs across branches and pulsars with tempo2 fitting and reports.
- FixDataset stage for dataset hygiene:
  - flag normalization (`-sys`, `-group`, `-pta`)
  - include/jump maintenance
  - overlap handling and deduplication (configurable)
- Optional QC integration (via `pqc`) for outlier/event tagging.
- Workflow mode to chain ingest/pipeline/qc-report steps in one run.

## Since `release/v0.1.0`

- TOML-based detection strategy selection:
  - `fix_qc_outlier_cols` for FixDataset action policy
  - `qc_report_compact_outlier_cols` for compact report policy
  - `pqc_backend_profiles_path` for per-backend PQC overrides
- Workflow serial/parallel grouping with explicit barriers.
- Ingest lockfile validation and source-priority controls for deterministic ingest.
- Declarative relabel/overlap rule catalogs in TOML.

See `docs/release_notes.rst` for details.

## Modes

- `pipeline`: `pleb --config <pipeline.toml>`
- `ingest`: `pleb ingest --config <ingest.toml>`
- `qc-report`: `pleb qc-report --config <qc_report.toml>`
- `workflow`: `pleb workflow --file <workflow.toml>`

## Quickstart

Install from GitHub:
```bash
pip install "git+https://github.com/golamshaifullah/pleb.git"
```

Minimal pipeline run:

```bash
pleb --config configs/runs/pipeline/epta-dr3-v0.toml
```

Mapping-driven ingest:

```bash
pleb ingest --config configs/runs/ingest/ingest_eptadr3.toml
```

Run a workflow:

```bash
pleb workflow --file configs/workflows/ingest_eptadr3.toml
```

All TOML keys are also available as CLI overrides; CLI values always win:

```bash
pleb --config configs/runs/pipeline/epta-dr3-v0.toml --set jobs=8 --set run_fix_dataset=true
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest -q
```
