#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pleb.reference_dataset_tools import write_reference_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate tests/reference/manifest.json from current input/ and expected/ trees."
    )
    parser.add_argument("--reference-root", type=Path, default=Path("tests/reference"))
    args = parser.parse_args()
    manifest = write_reference_manifest(args.reference_root)
    print(f"Wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
