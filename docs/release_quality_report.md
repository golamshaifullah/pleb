# Release quality report

`pleb release-report` builds a reader-facing quality report from existing PQC/QC CSV outputs. It does not rerun Tempo2 or PQC. It reads `*_qc.csv` files from a run directory, applies the same compact decision policy used by the QC report, and writes a PDF plus machine-readable TSV/JSON artifacts.

Typical use:

```bash
pleb release-report --run-dir results/run_2026-07-01 \
  --title "EPTA DR3 final data product quality report"
```

Pipeline use:

```toml
release_quality_report = true
release_quality_report_name = "release_quality_report.pdf"
release_quality_report_title = "EPTA DR3 final data product quality report"
release_quality_report_backend_col = "group"
release_quality_report_include_per_pulsar_pages = true
```

Outputs are written to `<run-dir>/release_quality_report/` unless overridden:

- `release_quality_report.pdf` — scorecard PDF for quick human assessment.
- `release_quality_scorecard.tsv` — one row per pulsar/variant QC table.
- `release_quality_backend_risks.tsv` — backend-level risk ranking.
- `release_quality_flagged_toas.tsv` — compact table of flagged TOAs.
- `release_quality_summary.json` — summary metadata and artifact paths.

Grades are assigned per pulsar/variant using the configured bad/review/event fractions. `BAD_TOA` and `REVIEW_EVENT` are intentionally separated: a review event is not automatically a bad TOA, but it is still a release risk that should be visible on the scorecard.
