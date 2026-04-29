#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pleb.reference_dataset_tools import stage_reference_input, write_reference_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage a minimal reference input dataset from a larger dataset tree."
    )
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--reference-root", type=Path, default=Path("tests/reference"))
    parser.add_argument("--pulsar", action="append", required=True, help="Pulsar directory name to copy.")
    parser.add_argument(
        "--extra-path",
        action="append",
        default=[],
        help="Additional path relative to the source dataset to copy into input/.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove existing input/ before copying.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dest_root = args.reference_root / "input"
    if args.clean and dest_root.exists():
        shutil.rmtree(dest_root)
    copied = stage_reference_input(
        args.source_dataset,
        dest_root,
        pulsars=list(args.pulsar),
        extra_paths=list(args.extra_path),
    )
    manifest = write_reference_manifest(
        args.reference_root,
        metadata={
            "source_dataset": str(args.source_dataset),
            "pulsars": list(args.pulsar),
            "extra_paths": list(args.extra_path),
        },
    )
    print(f"Copied {len(copied)} file(s) into {dest_root}")
    print(f"Wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
