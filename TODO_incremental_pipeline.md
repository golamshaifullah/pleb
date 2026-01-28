# Incremental Pipeline Runs — TODO Outline

Goal
- Support incremental pipeline runs so minimal work happens per run, unless a fresh/clean run is explicitly requested.

Principles
- Deterministic outputs (same inputs -> same outputs directory layout).
- Traceable provenance (inputs/config/tool versions recorded).
- Safe invalidation (never reuse stale outputs).
- Opt-in clean runs (easy to force recompute).

1) Define Incremental Semantics
- Decide what “incremental” means per stage:
  - tempo2 outputs (PLK, covmat, general2)
  - plots (residuals, coverage, covmat heatmaps)
  - reports (change reports, outlier tables, model comparisons)
  - QC (pqc outputs + summary + qc report)
  - FixDataset report-only stage
  - Binary analysis table
- Decide granularity:
  - per-pulsar artifacts
  - per-branch artifacts
  - cross-branch artifacts
- Decide what constitutes a “clean run”:
  - CLI flag (e.g., `--clean` or `--force-rerun` extended to all stages)
  - config flag in `PipelineConfig`

2) Output Directory Strategy
- Replace timestamp-only directory naming with:
  - stable outdir (config-specified) + run-id inside
  - or cache directory keyed by hash of config/inputs
- Ensure consistent structure for re-use across runs.
- Allow explicit override for “fresh” run directories.

3) Input Fingerprinting
- Define fingerprints for each stage:
  - Input files: .par/.tim content (hash), per pulsar
  - Config parameters relevant to the stage (hash of subset)
  - Tool versions (tempo2, pqc, pipeline version/commit)
  - Git branch name and/or commit hash of data repo
- Store fingerprints alongside outputs (e.g., `metadata.json`).

4) Cache Index / Metadata
- Add a lightweight cache index format:
  - `cache.json` at run root
  - per-stage metadata files in artifact directories
- Track:
  - inputs hash
  - tool versions
  - creation time
  - status/success
  - output paths

5) Skip/Reuse Logic by Stage
- tempo2: already skips if outputs exist; extend to check metadata + hash.
- plots:
  - re-run only if inputs for plots changed (general2, plk, covmat).
- reports:
  - re-run only if referenced inputs changed.
- QC:
  - re-run per pulsar if .par/.tim or config changed.
- FixDataset report-only:
  - re-run only if .par/.tim or fix config changed.
- Binary analysis:
  - re-run only if .par changed.

6) Clean Run / Invalidation
- Provide explicit clean mode:
  - CLI flag to wipe or ignore caches.
  - config flag to force recompute per stage.
- Optionally, selective invalidation:
  - `--clean-tempo2`, `--clean-qc`, etc.

7) Concurrency + Safety
- Ensure cache writes are atomic (write temp + rename).
- Avoid collisions for concurrent pulsar jobs (unique per-pulsar metadata).
- Validate cache correctness on read (fail safe to recompute).

8) Integration Points
- `pleb/pipeline.py`:
  - central orchestrator for stage skipping and metadata checks.
- `pleb/tempo2.py`:
  - enhance skip condition to use metadata.
- `pleb/utils.py`:
  - output tree / cache path helpers.
- `pleb/config.py`:
  - new flags for clean/incremental controls.
- `pleb/cli.py`:
  - CLI flags to enable/disable incremental and clean runs.

9) Logging + Observability
- Log decisions for reuse vs recompute:
  - “SKIP (cache hit): <artifact>”
  - “REBUILD (cache miss): reason=…”.
- Optional summary at end:
  - cache hits/misses by stage.

10) Tests
- Unit tests for fingerprinting consistency.
- Integration tests for:
  - cache hits when inputs unchanged
  - cache misses when inputs change
  - clean run overrides
- Ensure tests don’t depend on external tempo2/pqc availability.

Notes / Open Questions
- Should cache be per-run or global across runs?
- How to handle branch switching: use commit hash instead of branch name?
- Where to store tool versions (tempo2, pqc) if not easily queryable?
- Backwards compatibility for existing run directories.
