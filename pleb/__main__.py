"""CLI entry point for the data-combination pipeline package.

This module enables running the package with ``python -m pleb``.
It delegates to :func:`pleb.cli.main`.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
