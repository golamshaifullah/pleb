PQC Detection: Run QC Without Editing The Data
==============================================

This page covers the detect stage.

The goal of this stage is:

- run tempo2 on a coherent branch,
- run PQC detectors,
- generate review products,
- do not yet apply edits to tim files.

In this workflow, the detect stage is the first stage where residual behavior
is examined systematically rather than only structurally.


Why Run PQC At All
------------------

For a single pulsar, PQC is useful because it surfaces structure that is easy
to miss in a basic residual plot.

Examples:

- one or two isolated bad measurements,
- a problematic backend with heavy-tailed residuals,
- a step in time,
- a DM-like step,
- solar-angle contamination,
- orbital-phase dependence in binary systems,
- broader transient structure such as bumps or glitches.

Without PQC, a run may only show that "the fit looks messy." With PQC, the
output provides a set of specific hypotheses and tables to inspect.


The Detect Profile
------------------

Create a profile under ``configs/runs/pqc/``.

Example:
``configs/runs/pqc/single_pulsar_pqc_detect.toml``

Tracked repository example:
``configs/runs/pqc/single_pulsar_pqc_detect.example.toml``

.. code-block:: toml

   home_dir = "/data/canonical"
   dataset_name = "EPTA-DR3/epta-dr3-data"
   results_dir = "results"
   singularity_image = "/work/containers/psrpta.sif"

   branches = ["step2_pqc_balanced_detect"]
   reference_branch = "step2_pqc_balanced_detect"
   pulsars = ["J1909-3744"]
   jobs = 1
   outdir_name = "j1909_pqc_detect"

   run_tempo2 = true
   make_plots = true
   make_reports = true
   make_covmat = true

   run_fix_dataset = true
   fix_apply = true
   fix_base_branch = "step1_fix_flags_variants"
   fix_branch_name = "step2_pqc_balanced_detect"
   fix_commit_message = "Step2: PQC detection branch for J1909-3744"

   fix_qc_remove_outliers = false

   fix_generate_alltim_variants = true
   fix_backend_classifications_path = "configs/catalogs/variants/backend_classifications_legacy_new.toml"
   fix_alltim_variants_path = "configs/catalogs/variants/alltim_variants_legacy_new.toml"
   fix_jump_reference_variants = true
   fix_jump_reference_jump_flag = "-sys"

   run_pqc = true
   qc_report = true
   qc_report_backend_col = "sys"
   qc_report_compact_pdf = true
   qc_report_compact_outlier_cols = ["bad_point", "robust_outlier", "robust_global_outlier", "bad_mad"]

   pqc_backend_col = "sys"
   pqc_drop_unmatched = false
   pqc_merge_tol_seconds = 10.0

   pqc_add_orbital_phase = true
   pqc_add_solar_elongation = true
   pqc_add_elevation = true
   pqc_add_airmass = true
   pqc_add_parallactic_angle = true
   pqc_add_freq_bin = true
   pqc_freq_bins = 20

   pqc_structure_mode = "both"
   pqc_structure_detrend_features = ["solar_elongation_deg", "orbital_phase", "freq_bin"]
   pqc_structure_test_features = ["solar_elongation_deg", "orbital_phase", "freq_bin"]
   pqc_structure_group_cols = "sys"
   pqc_structure_nbins = 12
   pqc_structure_min_per_bin = 3
   pqc_structure_p_thresh = 0.01

   pqc_robust_enabled = true
   pqc_robust_scope = "backend"
   pqc_robust_z_thresh = 6.0
   pqc_outlier_gate_enabled = false
   pqc_event_instrument = true

   pqc_tau_corr_minutes = 30.0
   pqc_fdr_q = 0.01
   pqc_mark_only_worst_per_day = true
   pqc_tau_rec_days = 7.0
   pqc_window_mult = 5.0
   pqc_min_points = 6
   pqc_delta_chi2_thresh = 25.0

   pqc_step_enabled = true
   pqc_step_min_points = 20
   pqc_step_delta_chi2_thresh = 25.0
   pqc_step_scope = "both"

   pqc_dm_step_enabled = true
   pqc_dm_step_min_points = 20
   pqc_dm_step_delta_chi2_thresh = 25.0
   pqc_dm_step_scope = "both"

   pqc_solar_events_enabled = true
   pqc_orbital_phase_cut_enabled = true
   pqc_eclipse_events_enabled = true
   pqc_gaussian_bump_enabled = true
   pqc_glitch_enabled = true

This is the single-pulsar version of the repository's balanced detect
pattern from ``configs/workflows/steps/step2_pqc_balanced_detect.toml``.


Why This Profile Still Has FixDataset Enabled
---------------------------------------------

One point that often needs clarification is why ``run_fix_dataset = true``
appears in a detect profile.

In this pattern, Step 2 still creates a branch boundary and can still
regenerate variant products, but the QC-apply action itself is disabled.

The important switch is:

``fix_qc_remove_outliers = false``

That means the run is detection-oriented, not action-oriented.

Operationally, this profile serves three purposes at once:

- it establishes a distinct Step-2 branch,
- it keeps variant products available on that branch if they are needed,
- it runs tempo2 and PQC without using QC flags as mutation instructions.


How To Explain The Main PQC Keys
--------------------------------

Grouping keys:

``pqc_backend_col``
  The column used to define backend groups for many detectors. For this
  workflow, ``"sys"`` is a common choice after Step 1 has harmonized
  ``-sys`` values.
  This key is one of the most consequential choices in the run because many
  QC summaries and thresholds are interpreted within this grouping.

Matching keys:

``pqc_merge_tol_seconds``
  Tolerance used when matching TOAs and tim metadata.
  If this is too small, valid matches may fail. If it is too large, unrelated
  rows may be merged incorrectly.

Feature-engineering keys:

``pqc_add_orbital_phase``
  Compute orbital phase when binary parameters support it.

``pqc_add_solar_elongation``
  Add solar-angle information used by solar-structure checks.

``pqc_add_freq_bin`` and ``pqc_freq_bins``
  Add a coarse frequency-bin feature for structure tests and diagnostics.

Structure keys:

``pqc_structure_mode``
  Whether to detrend against features, test for structure, or both.
  ``"both"`` is often the most informative starting point because it keeps the
  output diagnostically rich.

``pqc_structure_group_cols``
  Grouping used in the structure stage.

Robust outlier keys:

``pqc_robust_enabled``
  Enable MAD-style robust outlier detection.

``pqc_robust_scope``
  Compute robust statistics globally, by backend, or both.

``pqc_robust_z_thresh``
  Threshold controlling sensitivity.
  Lower values increase sensitivity and false positives; higher values suppress
  marginal outliers.

Transient and event keys:

``pqc_step_*``, ``pqc_dm_step_*``
  Step and DM-step sensitivity.

``pqc_solar_*``, ``pqc_orbital_phase_*``, ``pqc_eclipse_*``
  Domain-specific structure or event detectors.

``pqc_gaussian_bump_*``, ``pqc_glitch_*``
  Broader transient family detectors.


How To Pick A First PQC Strategy
--------------------------------

For an initial run, start with a balanced profile:

- moderate false-positive control,
- robust outliers enabled,
- common event detectors enabled,
- comment-only downstream action,
- compact report enabled.

Avoid two extremes in the initial pass:

- too conservative, where almost nothing is flagged,
- too aggressive, where the output is dominated by low-value alerts.


How To Decide Whether A Detector Family Belongs In The First Run
----------------------------------------------------------------

The broad detector families in the balanced example do not all answer the same
question.

Use them selectively:

- robust outlier detection:
  useful in almost every initial pass because isolated outliers are common and
  easy to interpret,
- structure testing:
  useful early because it helps reveal residual dependence on frequency, solar
  angle, or orbital phase,
- step and DM-step detectors:
  useful when backend changes or dispersive state changes are plausible,
- solar and orbital-phase detectors:
  most useful when the pulsar, cadence, and observing geometry make those
  structures physically plausible,
- bump and glitch detectors:
  useful when there is already reason to suspect broader transient structure,
  but not always necessary in the very first exploratory run.


How To Run The Detect Stage
---------------------------

Run:

.. code-block:: bash

   pleb --config configs/runs/pqc/single_pulsar_pqc_detect.toml


What To Inspect After PQC
-------------------------

Inspect the run directory in this order:

1. ``run_settings/``
   Confirm the exact command and resolved config.
2. tempo2 output products
   Confirm the branch and pulsar were processed correctly.
3. ``qc/`` outputs
   Inspect QC CSVs and summaries.
4. compact report products
   Review the summary before opening every raw table.

Core point:

Do not start from individual flagged TOAs. Start from the summaries, then move
to backend-specific evidence.


What The QC Outputs Are For
---------------------------

The QC outputs serve at least four different purposes:

- triage:
  determine whether the pulsar has a small number of isolated issues or a
  larger structural problem,
- localization:
  determine whether the issue is tied to one backend, one time range, one
  frequency range, or one orbital/solar regime,
- comparison:
  compare the effect of threshold changes or backend overrides across reruns,
- hand-off:
  provide explicit inputs for the later QC-apply stage through
  ``fix_qc_results_dir`` and ``fix_qc_branch``.


What Not To Conclude Too Early
------------------------------

Keep these points in mind:

- a flag is not proof of astrophysical pathology,
- a flag is not automatically a deletion command,
- backend grouping choices affect what is flagged,
- threshold changes must be justified and documented,
- report outputs and action policy are separate layers.


The Apply Stage Comes Later
---------------------------

Once the detect outputs can be read confidently, the next stage is a separate
FixDataset pass that points at the QC results directory from this run.

That separation is critical. Detection should be inspectable on its own.


How To Choose A Small First Detector Set
----------------------------------------

The balanced example above enables a broad set of detectors because it mirrors
the repository's existing pattern. For a very first run on a new pulsar, it is
also reasonable to begin with a narrower subset and then expand.

A conservative first subset is:

- ``pqc_robust_enabled = true``
- ``pqc_structure_mode = "both"``
- ``pqc_step_enabled = true``
- ``pqc_dm_step_enabled = true``
- ``pqc_solar_events_enabled = true`` only if solar-angle structure is a real
  concern for the pulsar and cadence

Then add:

- orbital-phase diagnostics for binary systems,
- eclipse diagnostics when physically relevant,
- bump and glitch detectors when the residual history suggests transient
  behavior worth testing explicitly.

The reason to expand gradually is interpretability. When many detector
families fire in the first run, it is harder to tell which family is driving
the output.


How This Relates To Action Policy
---------------------------------

The detect profile defines what evidence is produced. It does not define what
will later be commented or deleted. That later decision is made by ``fix_qc_*``
keys during the apply stage.

In practical terms:

- ``pqc_*`` determines what the QC stage measures and flags,
- ``qc_report_*`` determines how those QC outputs are summarized for review,
- ``fix_qc_*`` determines which QC columns later trigger comments or deletions.

Keeping those three layers conceptually separate makes the workflow much easier
to tune and audit.

For the separation between detector strategy, action strategy, and report
strategy, see :doc:`../configuration_reference`.


Related Documentation
---------------------

- detector/action/report strategy split: :doc:`../configuration_reference`
- tempo2 and QC runtime notes: :doc:`../pleb_deep_dive/tempo2_qc_reporting`
- high-level tuning summary: :doc:`../scientist_tuning`
- PQC detector documentation: https://golamshaifullah.github.io/pqc/index.html
