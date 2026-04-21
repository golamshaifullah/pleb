"""Tests for early FixDataset QC validation in the pipeline."""

from __future__ import annotations

import re

import pytest

from pleb.dataset_fix import FixDatasetConfig
from pleb.pipeline import _validate_fixdataset_qc_inputs


def test_validate_fixdataset_qc_inputs_fails_before_apply_when_csv_missing(
    tmp_path,
) -> None:
    qc_root = tmp_path / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    cfg = FixDatasetConfig(
        apply=True,
        qc_results_dir=qc_root,
        qc_branch="main",
        qc_remove_outliers=True,
        qc_require_csv=True,
    )

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "FixDataset QC input validation failed for branch step3_apply"
        ),
    ):
        _validate_fixdataset_qc_inputs(
            ["J0636+5128", "J1713+0747"], cfg, branch="step3_apply"
        )


def test_validate_fixdataset_qc_inputs_allows_missing_csv_when_disabled(
    tmp_path,
) -> None:
    qc_root = tmp_path / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    cfg = FixDatasetConfig(
        apply=True,
        qc_results_dir=qc_root,
        qc_branch="main",
        qc_remove_outliers=True,
        qc_require_csv=False,
    )

    _validate_fixdataset_qc_inputs(["J0636+5128"], cfg, branch="step3_apply")


def test_validate_fixdataset_qc_inputs_allows_empty_variant_manifest(
    tmp_path,
) -> None:
    qc_root = tmp_path / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    (qc_root / "qc_summary.tsv").write_text(
        "pulsar\tvariant\tbranch\tqc_status\tqc_csv\tqc_error\n"
        "J0636+5128\tlegacy\tmain\tempty_variant\t/path/J0636+5128.legacy_qc.csv\t\n",
        encoding="utf-8",
    )
    cfg = FixDatasetConfig(
        apply=True,
        qc_results_dir=qc_root,
        qc_branch="main",
        qc_remove_outliers=True,
        qc_require_csv=True,
    )

    _validate_fixdataset_qc_inputs(["J0636+5128"], cfg, branch="step3_apply")


def test_validate_fixdataset_qc_inputs_fails_on_manifest_pqc_failure(
    tmp_path,
) -> None:
    qc_root = tmp_path / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    (qc_root / "qc_summary.tsv").write_text(
        "pulsar\tvariant\tbranch\tqc_status\tqc_csv\tqc_error\n"
        "J1713+0747\tcombined\tmain\tpqc_failed\t/path/J1713+0747.combined_qc.csv\tsegfault\n",
        encoding="utf-8",
    )
    cfg = FixDatasetConfig(
        apply=True,
        qc_results_dir=qc_root,
        qc_branch="main",
        qc_remove_outliers=True,
        qc_require_csv=True,
    )

    with pytest.raises(RuntimeError, match="unresolved QC status"):
        _validate_fixdataset_qc_inputs(["J1713+0747"], cfg, branch="step3_apply")


def test_validate_fixdataset_qc_inputs_allows_success_even_if_combined_failed(
    tmp_path,
) -> None:
    psr = "J0000+0001"
    qc_root = tmp_path / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    (qc_root / "qc_summary.tsv").write_text(
        "pulsar\tvariant\tbranch\tqc_status\tqc_csv\tqc_error\n"
        f"{psr}\tcombined\tmain\tpqc_failed\t/path/{psr}.combined_qc.csv\tsegfault\n"
        f"{psr}\tlegacy\tmain\tsuccess\t/path/{psr}.legacy_qc.csv\t\n"
        f"{psr}\tnew\tmain\tsuccess\t/path/{psr}.new_qc.csv\t\n",
        encoding="utf-8",
    )
    cfg = FixDatasetConfig(
        apply=True,
        qc_results_dir=qc_root,
        qc_branch="main",
        qc_remove_outliers=True,
        qc_require_csv=True,
    )

    _validate_fixdataset_qc_inputs([psr], cfg, branch="step3_apply")
