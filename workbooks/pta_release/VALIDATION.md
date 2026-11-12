# Container validation — 12 November 2026

All eleven release-training notebooks were exercised on two artificial pulsars.
Code cells were executed in order in the PLEB SIF; the native exporter ran from
host Python and launched both SIFs, as instructed in notebook 09. Notebook
sources remain unexecuted and retain their manual checkpoints.

| Image | SHA-256 |
|---|---|
| `pleb_dr3_upstream.sif` | `8bb65a9067eb88dd6c285385851e74a609ff528feb9ab97aa8c55770726c563b` |
| `discovery_dr3_upstream.sif` | `fbcc4ad096e9796a6fde5b1fd45580d3bd0f61f52985e4d5b79627f2593571ef` |

The course ran the checkout's PLEB package at `/pleb-source/pleb`, rather than
the older PLEB installed in the image. Host orchestration used the existing
`pta_pipeline` Python 3.11 environment. Scientific calculations used the image
libraries. No production dataset, release branch or container was modified.
Generated inputs, figures, logs and archives stayed under
`/tmp/pleb-release-course-20261112`, outside the source repository.

| Exercise | Result |
|---|---|
| 00: input generation, software provenance, dataset Git | Passed |
| 01: native PLEB ingest, mapping, completeness | Passed |
| 02: FixDataset, exact duplicate comments, variants, revision/dual-band example | Passed |
| 03: PLEB timing and PQC detection on both pulsars | Passed |
| 04: PLEB review IDs, append-only overrides, unchanged unreviewed actions | Passed with instructor synthetic decisions |
| 05: only explicit `mark_bad` actions applied, exact TOA-state audit | Passed |
| 06: repeated fits, parameter-step/WRMS convergence, saved-PAR reload | Passed with instructor model approval |
| 07: exported-pair check, original QC input preservation, extracted archive hashes | Passed |
| 08: two native dataset-build replays, identical files, checkout preserved | Passed |
| 08: changed decision catalog rejected before output creation | Passed |
| 09: native export, dataset clock overlay, synthetic observatory integrity | Passed |
| 09: independent Tempo2 arrays from both SIFs and fresh-process Feather re-export | Passed |
| 10: actual Feather arrays, WRMS/plots and manual-review source-row hashes | Passed |
| Streamlit AppTest on the generated residual-review inputs | Passed without app errors |

The test retained all 580 physical TOA rows. PLEB commented four duplicate
occurrences and four deliberately displaced observations, leaving 572 active
TOAs: 286 per pulsar. Both final model fits converged in three passes and
their saved PAR files passed a no-fit residual reload check. No active exact
within-file duplicates remained.

| Synthetic pulsar | Exported TOAs | Feather WRMS (µs) | CLI/libstempo residual difference (s) |
|---|---:|---:|---:|
| J1234+5678 | 286 | 0.69445961 | 0.0 |
| J1235+5678 | 286 | 0.69420694 | 0.0 |

Both exports passed exact independent-container array comparison and
byte-identical Feather reproduction. Neither reused a saved model. The
dataset-supplied `train2utc.clk` was recorded as an override; no extra UTC
station corrections were introduced. The artificial observatory was prepared
explicitly in the private synthetic export runtime and its hashes checked.

The validation harness supplied instructor decisions for the known synthetic
anomalies, labelled as training validation. It did not auto-approve real data
or change the notebook sources' empty decision lists and false approval
flags. The final archive lesson was also re-executed under a fresh archive
name after adding original QC input preservation.

The optional workflow example in 08 is generated, not executed over existing
branches. The review UI was checked with Streamlit's AppTest, not a manual
browser session. Existing interactive Jupyter setup was validated on 7 November;
this update checks the revised notebook cells, launcher shell syntax and
host-path argument handling. Older SIFs may still need the launcher's separate
Jupyter installation on first use. These results apply to the tested images
and settings, not arbitrary rebuilt images, changed policies or host setups.

Numerical agreement and storage reproducibility do not prove residual
whiteness, model adequacy or that a real excluded observation is bad.
