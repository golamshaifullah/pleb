# data-combination-pipeline

This is the notebook refactored into a Python module + CLI.

It also includes two optional add-ons (ported from the notebooks you supplied):
- **Dataset fixing/reporting** (from `FixDataset.ipynb`)
- **Binary/orbital parameter analysis** (from `AnalysePulsars.ipynb`)

## Install (editable)

```bash
cd data_combination_pipeline_module
python -m pip install -e .
```

## Run

Create a config file (JSON or TOML). Example:

```toml
[pipeline]
home_dir = "/path/to/data-repo"
singularity_image = "/path/to/EPTAsolotiming.sif"
results_dir = "."
branches = ["master", "EPTA+InPTA"]
reference_branch = "master"
pulsars = "ALL"
epoch = "55000"
```

Then run:

```bash
data-combination-pipeline --config config.toml
```

Or:

```bash
python -m data_combination_pipeline --config config.toml
```

The command prints the output folder (the report tag directory).

## Extras

### Dataset fix/report stage

This is **report-only by default** (it will not edit files) and writes outputs to `fix_dataset/` in the report tree.

Enable it via:

```bash
data-combination-pipeline --config config.toml --fix-dataset
```

Key config knobs (all optional):

```toml
[pipeline]
run_fix_dataset = true
fix_update_alltim_includes = true
fix_min_toas_per_backend_tim = 10
fix_required_tim_flags = { "-pta" = "EPTA" }
fix_insert_missing_jumps = true
fix_jump_flag = "-sys"
fix_ensure_ephem = "DE440"
fix_ensure_clk = "TT(BIPM2021)"
fix_remove_patterns = ["NRT.NUPPI.", "NRT.NUXPI."]
# fix_coord_convert = "equatorial_to_ecliptic"  # requires astropy
```

⚠️ `fix_apply=true` is intentionally **not** supported inside `run_pipeline` yet, because it would dirty the repo and break branch switching. If you want an apply+commit workflow, the building blocks are in `data_combination_pipeline.dataset_fix`.

### Binary/orbital analysis

Writes `binary_analysis/binary_analysis.tsv` in the report tree.

```bash
data-combination-pipeline --config config.toml --binary-analysis
```

Optional config filter:

```toml
[pipeline]
binary_only_models = ["ELL1", "BT", "BTX"]
```

## Optional dependencies

Some coordinate conversion and orbital utilities need extra packages:
- `astropy` (coordinate transforms)
- `scipy` (some solvers)

You can add them via your environment manager, or extend `pyproject.toml`'s optional extras.
