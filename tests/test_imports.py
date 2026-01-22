from __future__ import annotations


def test_import_pipeline_and_cli() -> None:
    # Ensures we don't ship syntax errors (e.g. indentation mistakes).
    import data_combination_pipeline.pipeline  # noqa: F401
    import data_combination_pipeline.cli  # noqa: F401
