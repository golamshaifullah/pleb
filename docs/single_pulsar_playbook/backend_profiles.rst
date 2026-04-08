Backend-Specific PQC Profiles And Tuning
========================================

This page explains how to tune PQC carefully and how to use backend-specific
overrides.

Read this after at least one balanced detect run has been completed and the
outputs have been inspected.


Why Backend-Specific Tuning Exists
----------------------------------

Not all backends behave the same way.

Different systems can differ in:

- cadence,
- TOA uncertainty scale,
- frequency coverage,
- known instrumental systematics,
- susceptibility to solar-angle or orbital-phase structure,
- tendency to generate false positives under generic thresholds.

So the workflow needs a way to say:

Use one global default strategy, but override selected thresholds for specific
backend groups.

This is usually preferable to forcing the global thresholds to fit the most
problematic backend.


Where The Override File Goes
----------------------------

Put backend-specific PQC overrides under ``configs/rules/pqc/``.

Example:
``configs/rules/pqc/single_pulsar_backend_profiles.toml``

Tracked repository example:
``configs/rules/pqc/single_pulsar_backend_profiles.example.toml``

.. code-block:: toml

   [backend_profiles]
   "EFF.P200.1360" = { robust_z_thresh = 6.5, fdr_q = 0.005 }
   "NRT.NUPPI.*" = { robust_z_thresh = 5.5, step_delta_chi2_thresh = 18.0 }

This follows the same file shape as
``configs/rules/pqc/backend_profiles.example.toml``.


How Matching Works
------------------

The repository documentation defines this precedence:

1. exact backend key match,
2. glob-pattern match,
3. global defaults from the main run profile.

Start by checking what values actually appear in
the chosen backend column, usually ``sys``.


The Run-Profile Key That Activates Overrides
--------------------------------------------

In the PQC detect run profile, add:

.. code-block:: toml

   pqc_backend_profiles_path = "configs/rules/pqc/single_pulsar_backend_profiles.toml"

This tells ``pleb`` to forward per-backend override settings to PQC.

The override file does not replace the main PQC profile. It only overrides the
specified keys for matched backends; everything else continues to come from the
main run profile.


How To Decide What The Backend Key Should Be
--------------------------------------------

The most important conceptual question is:

What should count as a backend group for this analysis?

Common choices:

``pqc_backend_col = "sys"``
  Good default when Step 1 has already normalized ``-sys`` values and you want
  QC grouping tied to system identity.

``pqc_backend_col = "group"``
  Useful when your dataset has a meaningful higher-level grouping already in
  place and you want broader aggregation.

This should not be chosen arbitrarily.

Before choosing it, answer:

- What values exist in the chosen column?
- Are those values stable and meaningful for jump/QC interpretation?
- Does this grouping align with how the parfile jumps are being managed?


A Simple Tuning Procedure
-------------------------

Use an evidence-driven loop:

1. run a balanced global profile,
2. inspect which backend groups dominate the flagged outputs,
3. decide whether the issue is true bad behavior or threshold mismatch,
4. add one override for one backend family,
5. rerun and compare.

Do not change many thresholds at once.


What To Tune First
------------------

The first tunable keys should usually be:

``robust_z_thresh``
  Controls MAD-style outlier sensitivity. This is often the safest first knob.

``fdr_q``
  Controls false discovery behavior for bad-measurement detection.

``step_delta_chi2_thresh`` and ``dm_step_delta_chi2_thresh``
  Useful when a backend is producing too many or too few step detections.

Only after those are understood should the student start tuning broader event
detectors such as bump or glitch thresholds.


What Usually Signals The Need For An Override
---------------------------------------------

An override is usually warranted when one backend repeatedly behaves
qualitatively differently from the rest of the dataset under otherwise
reasonable global settings.

Typical signals:

- one backend dominates robust outlier counts across reruns,
- one backend produces many marginal step detections while others do not,
- one backend has a known instrumental history that justifies a different
  threshold,
- one backend has markedly different cadence or frequency coverage.

An override is usually not warranted when the issue is:

- caused by bad ingest naming,
- caused by inconsistent ``-sys`` values,
- caused by a globally inappropriate threshold that affects many backends.


How To Explain Common Patterns
------------------------------

Pattern: one backend is noisy but not obviously pathological
  Raise ``robust_z_thresh`` slightly for that backend before changing global
  defaults.

Pattern: a backend has known structured behavior around instrument changes
  Consider a lower step threshold for that backend only.

Pattern: one backend dominates bad-measurement flags because uncertainties are poorly calibrated
  Review that backend's error model and only then consider changing ``fdr_q``
  or related sensitivity settings.


What Not To Use Backend Profiles For
------------------------------------

Backend overrides are not a substitute for:

- fixing bad ingest naming,
- repairing broken ``-sys`` flags,
- understanding whether the grouping column is correct,
- distinguishing detector evidence from operator action policy.

If backend profiles are being used to patch over broken metadata, return to the
ingest or Step-1 FixDataset stage instead.


A Minimal Tuning Example
------------------------

Start from a balanced detect profile and add:

.. code-block:: toml

   pqc_backend_col = "sys"
   pqc_backend_profiles_path = "configs/rules/pqc/single_pulsar_backend_profiles.toml"

Then create:

.. code-block:: toml

   [backend_profiles]
   "JBO.ROACH.1520" = { robust_z_thresh = 6.5 }
   "EFF.P200.*" = { step_delta_chi2_thresh = 20.0 }

Document why each override exists.


Documentation Standard For Tuning
---------------------------------

For every override, it should be possible to answer four questions:

1. which backend values does it match,
2. which detector parameter does it change,
3. what output symptom motivated the change,
4. what changed after rerunning.

If they cannot answer those questions, the override is not yet justified.


Related Documentation
---------------------

- strategy and per-backend override notes: :doc:`../configuration_reference`
- high-impact PQC knobs: :doc:`../scientist_tuning`
- operational config groups: :doc:`../pleb_deep_dive/config_groups`
- PQC detector documentation: https://golamshaifullah.github.io/pqc/index.html
