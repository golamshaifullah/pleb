# Release-course validation — 16 November 2026

All eleven notebooks were executed in order on a fresh synthetic workspace using
the current checkout's PLEB source and the images below. Notebook 09's generated
export command ran from the host, launching both containers. Its completion cells
and all of notebook 10 then ran in the PLEB container.

| Image | SHA-256 |
|---|---|
| `pleb_dr3_upstream.sif` | `8bb65a9067eb88dd6c285385851e74a609ff528feb9ab97aa8c55770726c563b` |
| `discovery_dr3_upstream.sif` | `fbcc4ad096e9796a6fde5b1fd45580d3bd0f61f52985e4d5b79627f2593571ef` |

## Checks performed

| Lesson | Executed checks | Result |
|---|---|---|
| 00 | Synthetic inputs, software/source provenance, isolated dataset Git | Passed |
| 01 | Native ingestion, mapping and physical-row completeness | Passed |
| 02 | Flags, exact duplicate comments, variants and dual-band/revision exercise | Passed |
| 03 | Timing/PQC detection on both pulsars; selected input branch unchanged | Passed |
| 04 | Reviewed decisions, append-only history and preserved raw QC inputs | Passed with instructor synthetic decisions |
| 05 | Apply only approved rejections; audit unchanged measurements and comment states | Passed |
| 06 | Convergence, saved-PAR reload and final model approval | Passed with instructor model approval |
| 07 | Exported PAR/TIM pairs, provenance and extracted archive hashes | Passed |
| 08 | Import source decisions; two identical raw-to-release builds; tampered catalog rejected | Passed |
| 09 | Export the branch built in 08; fresh fits, clock overlay and observatory integrity | Passed |
| 09 | Independent-container residual arrays and byte-identical fresh-process Feather exports | Passed |
| 10 | Read actual Feather arrays, reproduce WRMS/plots and verify physical TIM row hashes | Passed |
| Review UI | Group variants, switch pulsars repeatedly, use Next/Previous, save bad/keep decisions | Passed with Streamlit AppTest |

The raw inputs contain 580 physical rows. Four duplicate occurrences and four
injected anomalies are commented, leaving 572 active TOAs (286 per pulsar).
Notebook 08 reproduces the selection without fitting. Notebook 09 starts from
that rebuilt branch's raw-seeded PARs and fits them explicitly; it does not copy
the previously fitted canonical models from lesson 06.

| Synthetic pulsar | Exported TOAs | Feather WRMS (µs) | CLI/libstempo maximum difference (s) |
|---|---:|---:|---:|
| J1234+5678 | 286 | 0.69408821 | 0.0 |
| J1235+5678 | 286 | 0.69389320 | 0.0 |

Both fits converged in three passes; neither reused a saved model. The exporter
reported no issues or model warnings. Feather bytes also matched across two
complete fresh course builds. The supplied `train2utc.clk` was recorded
as an overlay, no additional zero-clock stations were introduced, and the
recorded artificial observatory remained unchanged.

The final review UI test used the corrected selector code on a separate copy
of lesson 10's review inputs. It checked repeated pulsar selection and
Next/Previous navigation, keeping the displayed table and save target aligned.
It preserved both decisions in history and wrote reviewed QC without changing
original QC tables. Export-review IDs were checked against their frozen TIM
paths, line numbers and row hashes; these are not the raw source IDs used by
native release catalogs.

The 56 relevant release, residual-export and review tests passed. The full
software suite passed (304 tests, 4 skipped). Regression checks cover grouping
`PSR_combined_qc.csv`, legacy and new variants under the coordinate pulsar name,
and real GUI navigation followed by saving decisions to the selected table.
Ruff and notebook syntax checks passed. Source notebooks contain no executed
outputs.

## Scope and reproduction

Launch a new learning workspace using the command in [the course guide](README.md)
and follow 00–10 in order. Notebook 09 prints the host command to run before
continuing. Existing workspace copies are preserved by the launcher; use a new
output directory to obtain these corrected lessons.

For this noninteractive test, the validation harness supplied labelled instructor
decisions for the known injected anomalies and approved the converged artificial
models. The distributed notebooks retain empty decision lists and false approval
flags: learners must inspect the evidence and make their own decisions.

All generated data, Git branches, plots, exports and logs stayed under
`/tmp/pleb-release-course-20261116-final`, outside the software repository. That run
contains `validation-prepare.json`, `validation-inspect.json`, notebook logs,
`09_host_export.log`, `review-ui-validation.json` and `native-export/index.html`.
No observed PTA data or production release files were used or modified.

This validates the recorded images, source and synthetic example. AppTest checks
the UI's rendering and save actions, not a manual browser session. Agreement
between containers and reproducible storage does not establish residual whiteness,
model adequacy or the scientific validity of exclusions in a real PTA dataset.
