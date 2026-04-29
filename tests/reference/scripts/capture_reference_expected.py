#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pleb.reference_dataset_tools import stage_expected_outputs, write_reference_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture golden/reference outputs from a completed run into tests/reference/expected."
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--reference-root", type=Path, default=Path("tests/reference"))
    parser.add_argument(
        "--path",
        action="append",
        required=True,
        help="File or directory path relative to source-root to capture.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove existing expected/ before copying.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dest_root = args.reference_root / "expected"
    if args.clean and dest_root.exists():
        shutil.rmtree(dest_root)
    copied = stage_expected_outputs(
        args.source_root,
        dest_root,
        relative_paths=list(args.path),
    )
    manifest = write_reference_manifest(
        args.reference_root,
        metadata={
            "source_root": str(args.source_root),
            "captured_paths": list(args.path),
        },
    )
    print(f"Copied {len(copied)} file(s) into {dest_root}")
    print(f"Wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
