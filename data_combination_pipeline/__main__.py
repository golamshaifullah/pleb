"""CLI entry point for the data-combination pipeline package.

This module enables running the package with ``python -m data_combination_pipeline``.
It delegates to :func:`data_combination_pipeline.cli.main`.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
