Troubleshooting
===============

This page lists common failure modes in the single-pulsar workflow and the
most direct way to isolate them.

General rule:

When a workflow fails, debug the stage that failed rather than the whole
workflow file. Run that stage directly with its ``--config`` profile and inspect
the resolved settings under ``run_settings/``.


Ingest Failures
---------------

Symptom: the pulsar directory is missing after ingest
  Check that the pulsar name in the source files can be resolved through
  ``pulsar_aliases`` and that the mapping roots point at real source files.

Symptom: backend tim files exist, but names are wrong
  Check the mapping ``backends`` keys. Ingest does not infer canonical backend
  names from the filesystem. The mapping key itself defines the downstream
  backend identity.

Symptom: multiple parfiles appear to match one pulsar
  Review ``par_roots`` and remove ambiguous or duplicate parfile locations.

Symptom: aggregate ``_all.tim`` files were re-ingested as backend tims
  Add ``ignore_suffixes`` or narrow ``tim_glob`` in the mapping entry.


Step-1 FixDataset Failures
--------------------------

Symptom: no branch changes appear after Step 1
  Confirm ``run_fix_dataset = true`` and ``fix_apply = true``. Then check that
  ``fix_base_branch`` points to the intended ingest branch and that the pulsar
  selection actually matches a real pulsar in that branch.

Symptom: jumps are still missing after Step 1
  Confirm ``fix_insert_missing_jumps = true`` and verify that the grouping flag
  used by ``fix_jump_flag`` matches the normalized metadata now present in the
  tim files.

Symptom: too many jumps were inserted
  Check whether ``-sys`` normalization is too granular. A fragmented grouping
  vocabulary can produce an over-fragmented jump structure.

Symptom: variant products were not generated
  Check ``fix_generate_alltim_variants = true`` and verify that both
  ``fix_backend_classifications_path`` and ``fix_alltim_variants_path`` point
  to readable files.


PQC Detect Failures
-------------------

Symptom: the run finishes but no QC outputs appear
  Confirm ``run_pqc = true`` and verify that the environment has the optional
  PQC dependencies installed. Also confirm that the run profile did not disable
  reporting unintentionally.

Symptom: QC outputs exist but look empty or nearly empty
  Check whether the pulsar selection is too narrow in combination with an
  aggressive grouping scheme, or whether the thresholds are too conservative.
  Also confirm that tempo2 actually ran and produced the expected inputs.

Symptom: QC outputs are dominated by one backend
  Determine whether this reflects real backend behavior, bad ``-sys``
  normalization, or an overly sensitive global threshold. Do not jump directly
  to backend overrides without checking the metadata first.

Symptom: orbital-phase or solar-angle outputs do not look meaningful
  Confirm that the relevant feature-engineering keys are enabled and that the
  pulsar and metadata actually support those derived features.


Apply-Stage Failures
--------------------

Symptom: Step 3 appears to ignore QC results
  Check ``fix_qc_results_dir`` and ``fix_qc_branch`` first. Most apparent
  "ignore" failures are actually path or branch mismatches in the QC hand-off.

Symptom: too many TOAs are commented in Step 3
  Review ``fix_qc_outlier_cols`` and the family toggles such as
  ``fix_qc_remove_transients``, ``fix_qc_remove_solar``, and
  ``fix_qc_remove_orbital_phase``. A broad action policy can be correct
  syntactically but too aggressive operationally.

Symptom: no TOAs are commented in Step 3
  Confirm that the QC columns listed in ``fix_qc_outlier_cols`` are actually
  present in the detect outputs and that the chosen action-family toggles are
  not excluding them.


Workflow Failures
-----------------

Symptom: the workflow fails in Step 2 or Step 3
  Run the referenced stage config directly with ``pleb --config ...`` and
  inspect the run directory. Workflow mode can hide which exact setting caused
  the failure if everything is debugged at once.

Symptom: branches and run directories are out of sync conceptually
  Check that the Step-2 detect profile writes QC outputs under the run
  directory that the Step-3 apply profile later reads through
  ``fix_qc_results_dir``.

Symptom: the workflow runs, but the branch hand-off is wrong
  Inspect ``fix_base_branch``, ``fix_branch_name``, ``branches``, and
  ``reference_branch`` in each referenced run profile. The branch chain should
  be linear and explicit.


Minimal Debug Order
-------------------

When the source of the problem is unclear, inspect in this order:

1. the referenced config file,
2. the resolved config under ``run_settings/``,
3. the branch names used by the stage,
4. the pulsar selection,
5. the stage-specific outputs in the run directory.

This order usually isolates path and branch mistakes before threshold tuning is
attempted.


Related Documentation
---------------------

- failure-oriented deep-dive notes: :doc:`../pleb_deep_dive/failure_cookbook`
- FixDataset stage details: :doc:`../pleb_deep_dive/fixdataset`
- tempo2 and QC runtime notes: :doc:`../pleb_deep_dive/tempo2_qc_reporting`
- configuration behavior and precedence: :doc:`../configuration`
