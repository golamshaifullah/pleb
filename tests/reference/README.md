# Reference Dataset Scaffold

This folder is reserved for a tiny deterministic end-to-end dataset and expected outputs.

Current status:
- scaffold only (no bundled timing data)

Recommended contents when enabling full reference tests:
- `input/`: minimal pulsar directory with par/tim files
- `expected/`: golden outputs (`qc_summary.tsv`, selected fixed par/tim, report checksums)
- `manifest.json`: file list + expected hashes

Use this scaffold to keep release-to-release behavior stable.
