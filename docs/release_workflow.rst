Build a release from raw inputs
===============================

.. code-block:: bash

   pleb run --config /path/to/release.toml


``[run] mode="release"`` selects the native raw-to-release workflow. It ingests
provider files, assembles clocks, normalises systems and duplicates, applies QC
and reviewed decisions, creates PAR/TIM variants, optionally fits, and verifies
the result. Raw inputs are read-only. No historical processed checkpoint,
replacement-TOA table, saved-model fallback or external replay script is used.

Start with `the example config <../configs/examples/release.toml>`_. Backend roots
are relative to ``source.raw``. Other paths are relative to the TOML. The config
inlines the ingest mapping and processing policy; no second ingest config is
required. Use ``pleb run --config release.toml --plan`` to inspect its stages and
resolved inventory without building output.

Inputs and decisions
--------------------

The reproducibility inputs are:

- Raw TIMs, seed PARs, templates and provider clock files.
- One TOML, including backend priorities, duplicate/overlap rules, variants,
  clock conventions, QC policy and timing settings.
- A decision catalog for preserved human or inherited exclusions, plus an
  independently approved selection reference. These contain identities and
  decisions/hashes, not replacement measurements.
- A PLEB installation and the scientific runtime identified by its SHA-256.

Optional system-rule/overlap tables are declared inputs, hashed in the run state.
PLEB records source identities in sidecars to avoid Tempo2's TOA flag limit.
Each identity uses a relative raw path, input checksum and physical line number;
relocating the raw tree does not change it. Different duplicate occurrences have
different identities. Changing a source file requires a newly reviewed catalog.

``processing.dedupe_action="comment"`` retains duplicate rows. Native release mode
also requires comment actions for other exclusion operations and rejects row
removal patterns. An unmatched or repeated review ID is an error.

Declare ``processing.backend_bandwidths`` (TIM basename to MHz) for bandwidth
inference, and ``processing.overlap_exact_map`` (retained TIM basename to excluded
TIM basenames) for cross-backend overlaps. Alternatively declare an explicit
``overlap_exact_catalog_path``. Native releases never read these tables implicitly
from a software checkout; omitted bandwidths mean unknown bandwidth, not an
ambient default. Enabling exact overlaps without a declared map/catalog is an error.

Review a new release
--------------------

Use ``qc.mode="detect"`` with explicit ``[qc.settings]`` (PTAQCConfig detector
settings) and ``[qc.apply]`` (FixDataset ``qc_*`` application settings).
``review.mode="interactive"`` pauses the run after detection/application:

.. code-block:: bash

   pleb review --run-dir /path/to/run/review
   pleb run --config /path/to/release.toml --resume


The review app requires ``pip install 'pleb[gui]'``. It records source IDs, actions,
reasons, reviewer and timestamps. Detection is evidence, not human approval.
The resumed run saves ``reports/review_decisions.json``. Preserve that catalog
and its hash for later reconstructions; do not repeat human review implicitly.
A run without QC has no residual evidence to display: use detector output or the
separate residual-export/review tools when inspecting residuals.

To reconstruct an approved selection, set ``qc.mode="replay"``,
``review.mode="catalog"``, and the catalog path/SHA-256. This skips fresh PQC.
A ``keep`` decision can explicitly restore a provider/automatic comment; a
``comment`` decision excludes the identified occurrence. Decisions are not TOA edits.

To migrate an existing release **once**:

.. code-block:: bash

   pleb release-decisions --config release.toml --reference /approved/dataset \
     --output decisions.json --expected expected.json --workspace /new/import-run


The importer verifies one-to-one raw matches, print tolerance and timing context.
It reports unmatched measurements instead of copying them from the reference.
It labels unknown historical decisions ``inherited_selection``, without inventing
QC or reviewer provenance. Put the returned input inventory/catalog/reference
hashes into the config. Production runs then need no access to the old release.

Clocks and timing models
------------------------

``clocks.include_container=true`` starts from the pinned runtime's Tempo2 clocks,
then overlays raw provider clocks. Stock clocks stay in ``runtime/``; only supplied
and generated station corrections enter the dataset. ``clocks.utc_stations``
adds explicit zero station-to-UTC corrections **only for confirmed UTC inputs**.
It preserves the UTC-to-TAI-to-TT chain. Record the valid MJD range explicitly.
``reports/clocks.json`` lists supplied and effective file hashes.

Set ``timing.clock``, ``ephem``, ``weighted``, ``fit``, ``validate``, ``iterations`` and
``nobs`` explicitly. ``fit=false, validate=false`` reproduces the selection and builds
models from raw seeds; it does **not** reproduce or validate previous fitted PARs
or residuals. ``fit=true`` saves the requested Tempo2 fit into each variant PAR;
inspect the timing logs and parameter changes. No automatic repeated-convergence
loop or saved-model substitution is performed.

Optional ``[[timing.parameters]]`` entries select pulsar/variant/name/value.
Optional ``[[timing.search]]`` entries select a pulsar/variant, ``candidates`` in the
existing PLEB parameter-scan syntax and ``min_delta_bic`` (default 10). Search uses
weighted fits to the same observations and records all candidate BIC comparisons.
It does not select correlated-noise models. A lower BIC is not a substitute for
residual and physical-model review.

Outputs and verification
------------------------

The run directory contains ``dataset/``, ``reports/``, ``runtime/`` and, when requested,
``qc/``, ``review/`` and ``timing/``. Only ``dataset/`` is publishable data. Runtime files,
logs and reports stay outside it. The independent reference verifies active source
IDs, per-pulsar counts, measurement-context hashes and clock/timing policy.
An exact active duplicate fails verification.

Use a new directory for an independent build:

.. code-block:: bash

   pleb run --config release.toml --set 'output.directory="/new/run"'


``--resume`` requires unchanged config, inputs, code and generated dataset. A failed
mutating stage requires a fresh directory. Never regenerate expected hashes just
to make an unexplained failure pass. Matching TOA selection does not demonstrate
residual whiteness or justify every inherited exclusion.

Optional ``output.repository``, ``branch``, ``base`` and ``dataset_path`` publish the
finished dataset on a new **local** Git branch, using a private index. The checkout,
other dataset paths and untracked files remain untouched. Existing output branches
must already contain the identical result. No push is performed.

Feather export remains separate and optional; see
`the native residual exporter <residual_export_native.md>`_. A data repository may
instead generate those products in its own CI.
