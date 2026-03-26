# Integration Test Scaffold

This directory contains starter integration tests for current PLEB modes:

- `pipeline`
- `ingest`
- `workflow`
- `qc-report`

By default, only safe smoke checks run (CLI help + workflow file loadability).
Runtime-heavy tests are opt-in and skipped unless corresponding environment
flags are set:

- `PLEB_INTEGRATION_INGEST=1`
- `PLEB_INTEGRATION_PIPELINE=1`
- `PLEB_INTEGRATION_QC_REPORT=1`

Run only integration scaffold tests:

```bash
python -m pytest -q -m integration tests/integration
```

