# Container-backed Feather export with PLEB

Use `pleb-residual-export --config PROFILE --output-dir NEW_RUN`, or equivalently
`python -m pleb.residual_export`. The driver launches the pinned Discovery and
PLEB SIFs; no external analysis helper script or LLM is required. Use an editable
PLEB install from the matching source checkout, with Apptainer available in the
host environment. All timing calculations
and Feather serialization run inside the images.

The optional `[fit]` table accepts `max_iterations` (default 8, minimum 3).
Increasing this budget allows additional nonlinear fit passes without relaxing
the RMS or parameter-step convergence checks. Converged fits stop early. The
configured budget is recorded in the execution manifest and per-pulsar completion
state.

The data repository's own CI may generate Feathers instead. This optional tool
uses an export profile you supply; it is not part of core release reconstruction
and does not require Feather configs or products in the dataset commit.

```sh
pleb-residual-export --config /path/to/export.toml --output-dir /path/to/new-report
```

The profile declares `[input]` (`repository`, `commit`, `dataset`,
`expected_active`), `[containers.images]` (`pleb`, `discovery`), matching
`[containers.sha256]` pins, and `[clocks]` (`policy`, `sha256`).
`workbooks/pta_release/09_export_feathers_with_audited_clocks.ipynb` demonstrates
creating and using this profile with synthetic data.

The reference RMS table path is relative to the current directory. It is
optional and affects comparison columns only. The source repository and clock
policy paths resolve relative to the profile. Container paths resolve relative
to the source repository; image hashes are verified before use.

The driver freezes the specified Git dataset, verifies its effective selection
count, and overlays dataset-local clocks on the image clocks. The explicit
zero station-to-UTC files preserve the owner's confirmed timestamp convention;
normal UTC-to-TAI-to-TT corrections remain active. Clock hashes, opened files
and actual correction chains are recorded.

Existing fitted parameter flags and reference epochs are retained. Fits run at
least three passes, up to the configured budget, requiring RMS stability and
small ordinary parameter steps. No model search, new FDJUMPDM selection, noise fitting, fresh QC, TOA
selection change or uncertainty inflation is performed. Explicitly configured
extreme-JUMP repairs are recorded, and happen only in fitted working copies.

Each pulsar must pass all of these checks:

- Loaded TOA counts and filename/system/frequency/time order agree with TIMs.
- CLI and libstempo residuals agree up to one global phase zero.
- Saved model RMS agrees with the in-memory fit for weighted fits.
- Feather arrays survive an exact round trip.
- A fresh process reloads the saved fitted model and produces byte-identical
  Feather output, without refitting.
- The PLEB SIF independently reproduces the Discovery SIF's Tempo2 arrays.

Plots are read from reloaded Feather files. They show the full range, a labelled
zoom and frequency dependence, without removing outliers or recentering each
backend. Passing export checks does not establish that residuals are white or
free of astrophysical/instrumental features.

Open `OUTPUT/index.html`. Feathers are in `OUTPUT/feathers`, plots in
`OUTPUT/plots`, fitted model copies and convergence histories in `OUTPUT/fits`,
and validation/RMS/clock tables in `OUTPUT/reports`. The input dataset is never
rewritten. Rerunning the same command resumes missing/failed pulsars; successful
exports with matching checksums are reused. Changed runtime/input/code hashes
require a new output directory. `--pulsars ...` limits an explicit diagnostic
run; omit it for the full dataset.

Saved-model reuse is optional and must be declared explicitly in the export
profile with hash-checked PAR, fit-log and input-signature assets. It is reported
as reuse, not as a newly converged fit. The training profile declares no saved
models: both synthetic pulsars must converge from the branch built in lesson 08.

`--accept-code-update` is an operator-only migration option: it records the old
execution manifest and per-pulsar worker versions before accepting new driver
or worker code. Source data, images and clock policy must remain unchanged.
Normal reproduction in a new directory does not need this option.
