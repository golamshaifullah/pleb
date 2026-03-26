"""CLI mode parser smoke tests."""

from __future__ import annotations

from pleb.cli import (
    build_compare_public_parser,
    build_ingest_parser,
    build_qc_report_parser,
    build_workflow_parser,
)


def test_ingest_parser_accepts_config_argument() -> None:
    p = build_ingest_parser()
    ns = p.parse_args(["--config", "configs/runs/ingest/ingest_demo.toml"])
    assert ns.config.endswith("ingest_demo.toml")


def test_workflow_parser_accepts_file_argument() -> None:
    p = build_workflow_parser()
    ns = p.parse_args(["--file", "configs/workflows/example_iterative.toml"])
    assert ns.workflow_file.endswith("example_iterative.toml")


def test_qc_report_parser_accepts_run_dir_argument() -> None:
    p = build_qc_report_parser()
    ns = p.parse_args(["--run-dir", "results/run_2026"])
    assert str(ns.run_dir).endswith("run_2026")


def test_compare_public_parser_accepts_required_arguments() -> None:
    p = build_compare_public_parser()
    ns = p.parse_args(["--out-dir", "results/public_compare"])
    assert str(ns.out_dir).endswith("public_compare")
