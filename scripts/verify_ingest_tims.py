#!/usr/bin/env python3
"""Verify ingest copied expected tim files from mapping sources."""

from __future__ import annotations

import argparse
from pathlib import Path

from pleb.ingest import _load_mapping, verify_ingest_tims


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", required=True, help="Ingest mapping JSON file.")
    ap.add_argument("--ingest-root", required=True, help="Ingest output root.")
    args = ap.parse_args()

    mapping = _load_mapping(Path(args.mapping))
    verify_ingest_tims(Path(args.ingest_root), mapping)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
