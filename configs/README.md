# Config Layout

`pleb` configs are now organized by purpose:

- `configs/runs/`
  - Runnable TOML files passed to `--config` (grouped by mode).
- `configs/catalogs/`
  - Reusable data tables and mappings shared across runs.
- `configs/rules/`
  - Declarative rule files (relabel/overlap behavior).
- `configs/schemas/`
  - JSON schemas and GUI schema.
- `configs/workflows/`
  - Workflow mode files.
- `configs/system_tables/`
  - System-level tables used by fix/ingest logic.

`configs/settings/` is now legacy/generated space (for older paths and lock artifacts).
