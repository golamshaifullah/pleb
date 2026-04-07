Optimize Mode
=============

Purpose
-------

``pleb optimize`` is the model-selection layer for PQC and workflow settings.
It treats tuning as a repeatable search problem instead of manual threshold
adjustment.

The optimizer does not replace the existing pipeline. It sits on top of it:

1. sample a candidate settings vector,
2. run PLEB with those settings,
3. read the QC artifacts already produced by PLEB,
4. compute a weighted score,
5. keep the best configuration and a full trial table.

This keeps the execution path familiar and reduces the amount of new code that
can affect production processing.

What optimize mode uses
-----------------------

The optimizer reuses existing PLEB capabilities:

- pipeline execution,
- workflow execution,
- PQC CSV outputs,
- QC summary tables,
- branch-aware run directories,
- ordinary TOML settings overrides.

It does not introduce a second QC implementation.

Main files
----------

- ``pleb/optimize/search_space.py``:
  search-space loading and parameter sampling.
- ``pleb/optimize/trial_runner.py``:
  adapter from optimization trials to existing pipeline or workflow runs.
- ``pleb/optimize/scorers.py``:
  metric extraction from ``*_qc.csv`` outputs.
- ``pleb/optimize/optimizer.py``:
  study driver, best-trial selection, result writing.

Configuration files
-------------------

The optimizer uses three TOML file types.

Run config
~~~~~~~~~~

This defines where the base run comes from, how many trials to execute, and
where results should be written.

Example::

   [optimize]
   base_config_path = "configs/runs/pipeline/epta-dr3-v0.toml"
   execution_mode = "pipeline"
   search_space_path = "configs/optimize/search_spaces/pqc_balanced_v1.toml"
   objective_path = "configs/optimize/objectives/balanced_qc.toml"
   folds_path = "configs/optimize/folds/time_blocks.toml"
   out_dir = "results/optimize/example_pipeline"
   study_name = "example_pipeline_pqc"
   n_trials = 20
   sampler = "random"
   seed = 12345
   jobs = 1

   [optimize.fixed_overrides]
   pulsars = ["J1713+0747"]
   run_pqc = true
   run_tempo2 = true
   run_fix_dataset = false

Search-space config
~~~~~~~~~~~~~~~~~~~

This defines which settings are allowed to move during optimization.

Example::

   [parameters.pqc_fdr_q]
   type = "float"
   low = 0.001
   high = 0.05
   log = true

   [parameters.pqc_step_enabled]
   type = "bool"

   [parameters.pqc_step_delta_chi2_thresh]
   type = "float"
   low = 10.0
   high = 60.0
   depends_on = "pqc_step_enabled"
   enabled_values = [true]

Supported parameter types are:

- ``float``
- ``int``
- ``bool``
- ``categorical``
- ``fixed``

Objective config
~~~~~~~~~~~~~~~~

This defines the weighted score.

Example::

   maximize = true

   [weights]
   residual_cleanliness = 2.0
   residual_whiteness = 1.0
   event_coherence = 0.75
   stability = 0.75
   bad_fraction = -1.5
   overfragmentation_penalty = -1.0

Fold config
~~~~~~~~~~~

This defines how repeated held-out reruns are built.

Example::

   [folds]
   mode = "time_blocks"
   n_splits = 4
   time_col = "mjd"
   backend_col = "sys"
   rerun_mode = "held_in"

Current fold modes:

- ``none``:
  no fold reruns, only the full trial run is scored.
- ``time_blocks``:
  divide TOAs by MJD blocks and rerun with one block held out each time.
- ``backend_holdout``:
  rerun with one backend held out at a time.

True held-out reruns
--------------------

This mode now performs actual reruns on reduced temporary datasets.

For each trial:

1. PLEB runs once on the full dataset.
2. If folds are enabled, PLEB builds temporary dataset trees under
   ``<optimize out dir>/_fold_datasets/``.
3. For each fold, backend tim files are rewritten so held-out TOAs are removed.
4. ``*_all.tim`` include files are updated so empty backend tim files are not
   kept in the include list.
5. PLEB reruns on each held-in fold dataset.
6. The optimizer averages fold metrics and computes stability from the spread
   across fold reruns.

This is closer to a real robustness test than simply slicing one QC CSV after a
single run.

Important limitation:

- The fold reruns are held-in reruns. PLEB does not yet have a separate
  train/apply model that fits on one subset and then predicts labels for a
  separate hold-out subset.
- In practice this means optimize mode measures stability under data removal,
  which is still useful for unsupervised QC and event detection.

Metrics
-------

The optimizer currently scores trials from the QC CSV outputs.

Available metrics include:

- ``bad_fraction``
- ``event_fraction``
- ``event_coherence``
- ``residual_cleanliness``
- ``residual_whiteness``
- ``overfragmentation_penalty``
- ``backend_inconsistency_penalty``
- ``parameter_complexity_penalty``
- ``stability``
- ``event_stability``

There are also raw counts such as:

- ``n_toas``
- ``n_bad``
- ``n_events``
- ``n_event_members``

Output files
------------

Each optimization study writes:

- ``trials.csv``:
  one row per trial with score, parameters, and metrics.
- ``summary.json``:
  compact study summary.
- ``best_trial.json``:
  full record of the best trial.
- ``best_overrides.toml``:
  flat TOML snippet of the winning parameter values.
- ``report.md``:
  compact human-readable report.

Running optimize mode
---------------------

Example::

   python -m pleb.cli optimize \
       --config configs/optimize/runs/example_pipeline.toml

If ``sampler = "random"``, PLEB uses built-in random sampling.

If ``sampler = "optuna_tpe"``, Optuna must be installed in the active Python
environment.

Current limits
--------------

- optimization-level ``jobs`` must currently be ``1``;
- per-trial internal pipeline parallelism still works through ordinary PLEB
  settings such as ``jobs`` in the base pipeline config;
- workflow optimization applies sampled settings through top-level workflow
  ``set`` overrides;
- the optimizer is designed for settings selection, not for timing-model
  parameter scans. Use ``param_scan`` for timing-model experiments.

Relationship to PQC
-------------------

Optimize mode does not re-explain the detector mathematics.

For detector and statistical details, use the PQC documentation:

- https://golamshaifullah.github.io/pqc/index.html

The optimizer only answers a different question:

"Which combination of PLEB/PQC settings gives the best overall behavior under
the objective I defined?"
