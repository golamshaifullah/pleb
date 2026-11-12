# Build a PTA data release with PLEB

See [container validation and its limits](VALIDATION.md).

This is the practical PLEB course. It teaches the CLI, configs, dataset branches,
stage hand-offs, statistical QC, review decisions and release provenance by
building a release from two synthetic pulsars. The separate `timing_models/`
course provides the background timing and statistics exercises.

From the PLEB checkout:

```sh
# Optional if the local image is already in singularity/pleb_singularity/.
export PLEB_LEARNING_IMAGE=/path/to/pleb_dr3_upstream.sif
# Also needed for the export exercise in 09; both default to the local SIF folder.
export PLEB_DISCOVERY_IMAGE=/path/to/discovery_dr3_upstream.sif
./learn.sh release /path/outside/repo/release-learning-v2 --port 8889
```

If the image lacks Jupyter, the first launch installs it beside the output; use the printed localhost
URL. Container acquisition is described in [the workbook guide](../README.md).
Choose the **PLEB release training (container)** kernel, open notebook 00 and
run its cells individually with Shift+Enter. Save your changes and continue
in numerical order. Each notebook explains the inputs, command, expected
products, checks and a task for the learner.

**Use a new output directory for these updated lessons.** Relaunching an old
directory preserves your older notebook copies and edits.

| Notebook | Operation the learner performs |
|---|---|
| 00 | Create a workspace, inventory synthetic provider inputs, record software and initialise dataset Git |
| 01 | Write an ingest mapping and TOML, run `pleb ingest`, verify completeness |
| 02 | Run FixDataset, comment exact duplicates, distinguish revised/dual-band measurements, inspect flags and jumps |
| 03 | Run PLEB timing/PQC detection, inspect statistical evidence and branch-specific outputs |
| 04 | Use PLEB review IDs, record append-only decisions with reasons/reviewer/time, preserve raw QC hashes |
| 05 | Apply only explicit rejections as comments, audit every changed row, distinguish deliberate keep restoration |
| 06 | Require converged fits, reload saved PARs, inspect residuals and approve timing coefficients |
| 07 | Validate exported PAR/TIM pairs, preserve provenance, package and verify a synthetic release |
| 08 | Execute `pleb dataset-build`, verify two identical checkpoint replays, reject a tampered decision catalog |
| 09 | Prepare and run the native two-container Feather export, inspect effective clocks and reproducibility checks |
| 10 | Read actual Feather residuals, reproduce WRMS/plots, prepare manual review with physical TIM row hashes |

Notebooks 04 and 06 require explicit learner review before subsequent stages
can proceed. A detector flag does not automatically become a release decision.
They retain decisions as sidecar CSVs and comment TOAs, rather than deleting
rows or adding large numbers of provenance flags.

Lesson 09 prints a command to run in a second **host terminal**, then waits for
its results. That host needs Apptainer and a Python environment importing this
checkout with its dependencies (`python -m pip install -e .`). Timing, fitting
and Discovery serialization execute in the pinned SIFs. This avoids nested
containers from a notebook kernel. The PLEB SIF has PyArrow, so notebook 10
can inspect the resulting Feathers in the original kernel. Its optional host
review app additionally needs Streamlit (`python -m pip install streamlit`).

The new lessons separate three operations: creating a release from raw inputs,
replaying a frozen selection, and validating/exporting residuals. Dataset-build
replays a checkpoint and locked catalogs; it does not repeat human review or
fit timing models, and currently does not directly import arbitrary UI bad
overrides. Apply those through FixDataset before freezing a new checkpoint.
The synthetic replay starts with empty transformation catalogs because its
checkpoint already contains the reviewed decisions.

The course runs **your checkout's PLEB source**, mounted read-only at
`/pleb-source/pleb`, with timing libraries from the container. This includes
the current reviewed-keep and comment-only duplicate changes. Notebook 00
records source hashes; notebook 07 archives that source. For a reproducible
production run, freeze a reviewed source revision and record the image hash.
The older timing course still uses the image's installed PLEB.

This is a training recipe, not a historical EPTA replay. It uses known starting
models and an artificial observatory with a defined zero clock correction.
In notebook 07, variant products move to `provenance/variant_examples/`; only the complete canonical PAR/TIM
pair is fitted and published under `data/`. A real release must fit and validate every
published variant independently and adopt its own clock, noise, event and
backend policies.

Notebook 08 creates combined aliases of that approved canonical pair for the
native replay/export exercises. Notebook 09 preloads the recorded artificial
observatory in a private runtime; this is a simulation fixture, not an extra
production station or a change to the containers. Dataset clocks overlay stock
clocks, and no additional zero-clock stations are assumed.

Two-container agreement and byte-identical serialization establish numerical
and storage reproducibility. They do not establish scientifically white or
featureless residuals. The course also requires inspection of full-range
plots, frequency/backend structure, error bars and selected row identities.

Notebook reruns that would overwrite branches/exports deliberately stop. Use
a new workspace for another full run; reopening the same Jupyter output keeps
your notebook edits. Start with the failing cell and inspect its saved log
instead of rerunning all previous notebooks.

All raw files, dataset Git history, QC tables, review decisions, fitted models
and archives are generated outside the PLEB repo. No observed PTA data are
included in these notebook sources. Authoring source:
`scripts/build_release_training_workbooks.py`; helpers supply simulation and
file/log operations, while the PLEB commands and configs remain visible in cells.
