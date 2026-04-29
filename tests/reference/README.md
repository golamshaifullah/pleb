# Reference Dataset Scaffold

This folder is reserved for a tiny deterministic end-to-end dataset and expected outputs.

Current status:
- scaffold only (no bundled timing data)

Recommended contents when enabling full reference tests:
- `input/`: minimal pulsar directory with par/tim files
- `expected/`: golden outputs (`qc_summary.tsv`, selected fixed par/tim, report checksums)
- `manifest.json`: file list + expected hashes

Use this scaffold to keep release-to-release behavior stable.

Helper scripts:
- `tests/reference/scripts/create_reference_input.py`
  - Copies selected pulsar directories from a larger dataset into `tests/reference/input/`.
- `tests/reference/scripts/capture_reference_expected.py`
  - Copies selected artifacts from a completed run into `tests/reference/expected/`.
- `tests/reference/scripts/write_reference_manifest.py`
  - Regenerates `tests/reference/manifest.json` from the current `input/` and `expected/` trees.

Typical flow:

```bash
python tests/reference/scripts/create_reference_input.py \
  --source-dataset /path/to/EPTA-DR3/epta-dr3-data-v1_5 \
  --pulsar J1909-3744 \
  --clean

python tests/reference/scripts/capture_reference_expected.py \
  --source-root /path/to/completed/run \
  --path qc/qc_summary.tsv \
  --path fixed/J1909-3744/J1909-3744.par \
  --path fixed/J1909-3744/J1909-3744_all.tim \
  --clean
```
