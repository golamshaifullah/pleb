Scientist Tuning Guide
======================

Use profiles first, then tune specific knobs.

Profiles
--------

- ``safe``: conservative detection.
- ``sensitive``: stronger event/outlier sensitivity.
- ``expert``: wide-open advanced behavior.

Run-time profile stacking:

.. code-block:: bash

   pleb run --config configs/runs/pipeline/pleb.pipeline.toml --profile sensitive

High-impact PQC knobs
---------------------

- ``pqc_merge_tol_seconds``
- ``pqc_step_delta_chi2_thresh``
- ``pqc_dm_step_delta_chi2_thresh``
- ``pqc_gaussian_bump_delta_chi2_thresh``
- ``pqc_glitch_delta_chi2_thresh``
- ``pqc_glitch_noise_k``

Use per-backend overrides via:

- ``pqc_backend_profiles_path``
