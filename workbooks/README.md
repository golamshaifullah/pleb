# PLEB workbooks

For step-by-step training in **using PLEB to build a PTA data release**, start
with [the practical release course](pta_release/README.md):

```sh
./learn.sh release /path/outside/repo/release-learning --port 8889
```

It covers actual ingest, FixDataset, PQC, reviewed comment application, final
fits, native selection replay, audited Feather export and release packaging with synthetic inputs. The timing-model course
below provides supporting statistical exercises.

These are source notebooks and teaching helpers. No EPTA DR3 observations,
release datasets, containers or executed notebook outputs belong in this
directory. The synthetic lessons generate their PAR/TIM files when run.

## Obtain the container

Use x86-64 Linux with Apptainer or Singularity, plus Git. The ordinary notebook
launcher needs no host Python or Jupyter. Release lesson 09 additionally needs a
host Python environment importing an editable PLEB checkout to orchestrate the
two containers; lesson 10's optional host review UI also needs Streamlit.
The container recipes are maintained at
[epta-dr3-singularity](https://github.com/golamshaifullah/epta-dr3-singularity).
At the time these instructions were written, that repository provides recipes
but has no release image downloads. Obtain an existing `pleb_dr3_upstream.sif`
from your container provider or build it from the recipe below. No DR3 data
is needed for this build or for lessons 00–06.

Run these commands outside your PLEB checkout; `$HOME/pleb-learning` is an example
location you may change:

```sh
mkdir -p "$HOME/pleb-learning/containers"
git clone https://github.com/golamshaifullah/epta-dr3-singularity.git \
  "$HOME/pleb-learning/epta-dr3-singularity"
git -C "$HOME/pleb-learning/epta-dr3-singularity" checkout \
  4d339b79677b678f78f4253feb362ca4144669c1
apptainer build --fakeroot \
  "$HOME/pleb-learning/containers/pleb_dr3_upstream.sif" \
  "$HOME/pleb-learning/epta-dr3-singularity/src/pleb_dr3_upstream.def"
sha256sum "$HOME/pleb-learning/containers/pleb_dr3_upstream.sif" \
  > "$HOME/pleb-learning/containers/pleb_dr3_upstream.sif.sha256"
```

The build needs network access and compilation space. `--fakeroot` requires
the host's Apptainer user namespace configuration; if unavailable, have your
administrator build the same recipe with `sudo apptainer build` instead.
Substitute `singularity` for `apptainer` on hosts using Singularity.

The PLEB image contains Tempo2, libstempo and PLEB. The separate Discovery image
is required for release lesson 09; set `PLEB_DISCOVERY_IMAGE` to its path if it
is outside `singularity/pleb_singularity/`. Obtain it from your container provider
or build `src/discovery_dr3_upstream.def` from the same recipe checkout.
The pinned recipe installs PLEB commit
`633771099c4595ec3d409d785d3dde315dbfaf1f`. Candidate scans use that installed
version; editing your checkout does not change the installed PLEB in the image.
An image rebuilt from a recipe may have a different hash and dependency
versions from the original validated SIF. Retain its checksum and the lesson's
software-version report when comparing results.

## Open interactive notebooks

From the checkout root:

```sh
export PLEB_LEARNING_IMAGE=/path/to/pleb_dr3_upstream.sif
./learn.sh notebook "$HOME/pleb-learning/output"
```

Rebuilt DR3 recipes include the notebook runtime, so the launcher can use those
packages offline. With older SIFs, the first launch installs the pinned
[notebook add-ons](timing_models/notebook-requirements.txt) into a separate
environment under the output directory; this needs network access. The kernel
imports the container's timing libraries. Later launches reuse the installation.
Open the localhost URL printed by Jupyter in your browser. If working on a remote
machine, forward port 8888 with SSH (`ssh -L 8888:127.0.0.1:8888 user@host`) and
open that URL locally. Use `--port 8889` to select another port.

Open `00_runtime_and_known_truth.ipynb`, select **PLEB timing and statistics
(container)**, then run cells individually with **Shift+Enter**. Predict what
will happen, inspect the output, and complete the editable experiment at the
end. Use **Esc**, then **M** to turn a cell into Markdown for your interpretation.
Save with **Ctrl+S**. Continue through notebooks 01–06 in order.

Writable notebooks are copied to `OUTPUT/notebooks/`; relaunching preserves
your edits. For a fresh course copy, choose a new output directory. The source
notebooks in the checkout remain clean. Your saved notebook includes its
outputs, and you can reopen it later from the same Jupyter session.

For lesson 07, launch with `--run /path/to/private-run --pulsar J1909-3744`.
Keep that learning output private because it will contain copied real data.

## Optional batch execution

From the PLEB checkout root, set the image path once and reuse one output
directory for the sequence:

```sh
export PLEB_LEARNING_IMAGE="$HOME/pleb-learning/containers/pleb_dr3_upstream.sif"
./learn.sh 00 "$HOME/pleb-learning/output"
./learn.sh 01 "$HOME/pleb-learning/output"
# Continue with 02 through 06, in order.
```

The launcher also recognises images already located at
`containers/pleb_dr3_upstream.sif` (portable bundle) or
`singularity/pleb_singularity/pleb_dr3_upstream.sif` (existing local checkout).
Images are ignored by Git. Logs and caches go under the learning output.

| Lesson | Topic | What to learn |
|---|---|---|
| 00 | Runtime and known truth | Generate a reproducible synthetic pulsar and record its inputs |
| 01 | Residuals, units and weights | Interpret RMS, WRMS, chi-square and plots |
| 02 | Continuous parameter fitting | Set fit flags and recover an injected spin derivative |
| 03 | Dispersion, jumps and degeneracy | Distinguish frequency effects from backend offsets |
| 04 | PLEB candidate scans | Compare explicit timing models and check parameter counts |
| 05 | Uncertainties and covariance | Understand conditional errors and correlated parameters |
| 06 | Holdout and multiple trials | Recognise overfitting and selection effects |
| 07 | Optional real-pulsar scan | Apply an explicit candidate scan to a private copy |

Each lesson saves an executed notebook, standalone HTML and figures in the
output directory. Open `OUTPUT/workbooks/index.html` in a browser. Read the
source notebooks in `timing_models/` to inspect the steps before execution.
`./learn.sh all OUTPUT` runs the synthetic sequence in one invocation.

Tempo2 fits numerical timing parameters. PLEB's parameter scan compares
candidate models; `pleb optimize` tunes QC settings. Lesson 04 preserves the raw
PLEB table and checks the fitted-parameter count against Tempo2 logs because
PLEB's parsed `k_fit` can disagree with Tempo2, affecting derived significance.

## Use private DR3 data

To run lesson 07, supply your completed DR3 run from outside this checkout:

```sh
./learn.sh 07 "$HOME/pleb-learning/output" \
  --run /path/to/private-completed-dr3-run --pulsar J1909-3744
```

The launcher mounts that run read-only. The lesson copies the selected model,
TIM inputs and clocks to the learning output before running the candidate scan.
That output now contains private data: keep it outside the repository and do
not distribute it with the notebook sources.

For raw-to-release reconstruction and recorded manual decisions, use
[the native release guide](../docs/release_workflow.rst) and release lesson 08.

The eight timing-model notebooks are authored by
`scripts/build_timing_model_workbooks.py`; edit that source and regenerate them
when updating lessons. Runtime helpers live alongside the notebooks.
