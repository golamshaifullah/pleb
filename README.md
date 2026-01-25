# pta_qc

PTA residual quality-control toolkit.

## What it does

1. Parses tempo2 timfiles (supports `_all.tim` with `INCLUDE` and `TIME` blocks).
2. Loads residuals / TOA errors / frequencies from `libstempo`.
3. Matches TOAs to tim metadata by nearest MJD (tunable tolerance).
4. Ensures `sys` and `group` exist everywhere using timfile naming conventions.
5. Detects:
   - **bad measurements** (day-level FDR using OU innovations),
   - **transients** (jump + exponential recovery scan).

## Quickstart

Run QC:

```bash
python scripts/run_qc.py --par DR3full/J1909-3744/J1909-3744.par --out qc.csv
```

Summarize results:

```bash
python scripts/diagnose_qc.py --csv qc.csv --backend-col group
```

Plot detected transients:

```bash
python scripts/plot_transients.py --csv qc.csv --backend-col group --outdir plots
```

## Notes

- `sys` and `group` are forced to exist even if the original timfile omitted them.
- Default match tolerance is 2 seconds. If you see many unmatched rows, raise `--tol-seconds`.


## Tests

Run:

```bash
pytest -q
```
