"""Import smoke tests for public modules."""

from __future__ import annotations


def test_import_pipeline_and_cli() -> None:
    # Ensures we don't ship syntax errors (e.g. indentation mistakes).
    import pleb.pipeline  # noqa: F401
    import pleb.cli  # noqa: F401
