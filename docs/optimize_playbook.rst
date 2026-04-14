Playbook
=============

This assumes:

- repo: ``/work/git_projects/pleb``
- dataset repo: ``/work/git_projects/epta-dr3-in2p3``
- pulsar: ``J1713+0747``
- branch to test: ``tim_cleanup_delete_final``
- env already has pleb, pqc, `tempo2, singularity

1. Clean the scratch area

``rm -rf /tmp/pleb_opt_j1713``
``mkdir -p /tmp/pleb_opt_j1713``
``mkdir -p /tmp/pleb_opt_j1713/results``
``mkdir -p /tmp/pleb_opt_j1713/pipeline_results``

2. Copy the optimize templates

``cp /work/git_projects/pleb/configs/optimize/search_spaces/pqc_balanced_v1.toml /tmp/pleb_opt_j1713/``
``cp /work/git_projects/pleb/configs/optimize/objectives/balanced_qc.toml /tmp/pleb_opt_j1713/``

3. Create the fold config
Create ``/tmp/pleb_opt_j1713/time_blocks.toml``:

[folds]
mode = ``time_blocks``
n_splits = 2
time_col = ``mjd``
backend_col = ``sys``
rerun_mode = ``held_in``

4. Create a plain pipeline smoke-test config
Create /tmp/pleb_opt_j1713/j1713_pipeline_test.toml:

home_dir = "/work/git_projects/epta-dr3-in2p3"
dataset_name = "EPTA-DR3/epta-dr3-data"
results_dir = "/tmp/pleb_opt_j1713/pipeline_results"
singularity_image = "/work/git_projects/PSR_Singularity/psrpta.sif"

branches = ["tim_cleanup_delete_final"]
reference_branch = "tim_cleanup_delete_final"
pulsars = ["J1713+0747"]
jobs = 1
epoch = "58000"
force_rerun = true

run_tempo2 = true
run_pqc = true
run_fix_dataset = false
fix_apply = false
run_whitenoise = false
qc_report = false

make_plots = false
make_reports = false
make_covmat = false
make_toa_coverage_plots = false
make_change_reports = false
make_covariance_heatmaps = false
make_residual_plots = false
make_outlier_reports = false

pqc_backend_col = "sys"
pqc_drop_unmatched = false
pqc_event_instrument = true
pqc_structure_mode = "both"
pqc_structure_group_cols = ["sys"]

5. Run the plain pipeline smoke test

cd /work/git_projects/pleb
python -m pleb.cli --config /tmp/pleb_opt_j1713/j1713_pipeline_test.toml 2>&1 | tee /tmp/pleb_opt_j1713/j1713_pipeline_test.log

6. Verify the smoke test before touching optimize

find /tmp/pleb_opt_j1713/pipeline_results -name '*_qc.csv'

Expected:

- at least one file like:
    - /tmp/pleb_opt_j1713/pipeline_results/.../tim_cleanup_delete_final/qc/tim_cleanup_delete_final/J1713+0747_qc.csv

If none exist, stop. Optimize will not help.

7. Inspect the QC file quickly

python - <<'PY'
import pandas as pd
from pathlib import Path
paths = sorted(Path('/tmp/pleb_opt_j1713/pipeline_results').glob('**/J1713+0747_qc.csv'))
print('qc files:', [str(p) for p in paths])
df = pd.read_csv(paths[-1])
print('rows:', len(df))
for c in ['bad_point','robust_outlier','robust_global_outlier','bad_mad','event_type']:
    if c in df.columns:
        print(c, df[c].value_counts(dropna=False).head(10).to_dict())
PY

You want:

- non-empty file
- sensible flag/event columns
- not obvious nonsense

8. Create the optimize config
Create /tmp/pleb_opt_j1713/j1713_optimize.toml:

[optimize]
base_config_path = "/work/git_projects/pleb/configs/runs/pipeline/epta-dr3-v0.toml"
execution_mode = "pipeline"
search_space_path = "/tmp/pleb_opt_j1713/pqc_balanced_v1.toml"
objective_path = "/tmp/pleb_opt_j1713/balanced_qc.toml"
folds_path = "/tmp/pleb_opt_j1713/time_blocks.toml"
out_dir = "/tmp/pleb_opt_j1713/results"
study_name = "j1713_optimize_test"
n_trials = 4
sampler = "random"
seed = 12345
jobs = 1
keep_trial_runs = true
fail_fast = false
write_best_config = true

[optimize.fixed_overrides]
home_dir = "/work/git_projects/epta-dr3-in2p3"
dataset_name = "EPTA-DR3/epta-dr3-data"
results_dir = "/tmp/pleb_opt_j1713/pipeline_results"
singularity_image = "/work/git_projects/PSR_Singularity/psrpta.sif"

pulsars = ["J1713+0747"]
branches = ["tim_cleanup_delete_final"]
reference_branch = "tim_cleanup_delete_final"

run_tempo2 = true
run_pqc = true
run_fix_dataset = false
fix_apply = false
run_whitenoise = false
qc_report = false

make_plots = false
make_reports = false
make_covmat = false
make_toa_coverage_plots = false
make_change_reports = false
make_covariance_heatmaps = false
make_residual_plots = false
make_outlier_reports = false

pqc_backend_col = "sys"
pqc_drop_unmatched = false
pqc_event_instrument = true
pqc_structure_mode = "both"
pqc_structure_group_cols = ["sys"]

9. Run optimize

cd /work/git_projects/pleb
python -m pleb.cli optimize --config /tmp/pleb_opt_j1713/j1713_optimize.toml 2>&1 | tee /tmp/pleb_opt_j1713/j1713_optimize.log

10. Check whether optimize succeeded

python - <<'PY'
import pandas as pd
df = pd.read_csv('/tmp/pleb_opt_j1713/results/trials.csv')
print(df[['trial_id','status','score','run_dir','error']].to_string(index=False))
PY

What you want:

- at least some status = ok
- non-empty score
- run_dir populated
- error empty for successful trials

11. Confirm per-trial QC files exist

find /tmp/pleb_opt_j1713/pipeline_results -name '*_qc.csv'

You want trial files like:

- /tmp/pleb_opt_j1713/pipeline_results/j1713_optimize_test_trial_0001/tim_cleanup_delete_final/qc/tim_cleanup_delete_final/J1713+0747_qc.csv

12. Compare trial behavior

python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path('/tmp/pleb_opt_j1713/pipeline_results')
rows = []
for p in sorted(root.glob('j1713_optimize_test_trial_*/tim_cleanup_delete_final/qc/tim_cleanup_delete_final/J1713+0747_qc.csv')):
    df = pd.read_csv(p)
    rec = {'trial': p.parts[-5]}
    for c in ['bad_point','robust_outlier','robust_global_outlier','bad_mad']:
        if c in df.columns:
            rec[c] = int(pd.to_numeric(df[c], errors='coerce').fillna(0).sum())
    if 'event_type' in df.columns:
        rec['n_event_rows'] = int(df['event_type'].notna().sum())
    rows.append(rec)
print(pd.DataFrame(rows).to_string(index=False))
PY

You want:

- trials differ
- not all zero
- not all catastrophic

13. Inspect the winning config

less /tmp/pleb_opt_j1713/results/best_overrides.toml
less /tmp/pleb_opt_j1713/results/report.md

14. Minimal acceptance test
Call the run meaningful only if:

- the plain pipeline smoke test produced J1713+0747_qc.csv
- optimize produced at least 2 successful trials
- trial scores differ
- trial QC behavior differs
- best config is plausible
- manual inspection of best-trial QC is better than baseline

15. If something fails
Use these triage commands:

Plain pipeline failure:

rg -n "ERROR|Warning|Traceback|failed|FileNotFoundError|RuntimeError" /tmp/pleb_opt_j1713/j1713_pipeline_test.log

Optimize failure:

python - <<'PY'
import pandas as pd
df = pd.read_csv('/tmp/pleb_opt_j1713/results/trials.csv')
print(df[['trial_id','status','error','run_dir']].to_string(index=False))
PY
