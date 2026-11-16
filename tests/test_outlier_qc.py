from __future__ import annotations

import subprocess

import pandas as pd
import pytest

import pleb.outlier_qc as oqc
from pleb.outlier_qc import (
    PTAQCConfig,
    _normalize_step_memberships,
    run_pqc_for_parfile_subprocess,
)


def test_normalize_local_step_membership_fills_post_step_interval() -> None:
    df = pd.DataFrame(
        {
            "mjd": [1.0, 2.0, 3.0, 4.0, 5.0],
            "group": ["A", "A", "A", "A", "A"],
            "step_id": [-1, 0, -1, -1, 0],
            "step_t0": ["", 2.0, "", "", 2.0],
            "step_amp": ["", 1.25, "", "", 1.25],
            "step_delta_chi2": ["", 40.0, "", "", 40.0],
            "step_applicable": [True, True, True, True, True],
            "event_member": [False, True, False, False, True],
            "bad_point": [False, False, False, True, False],
            "outlier_any": [False, True, False, True, True],
        }
    )

    fixed, info = _normalize_step_memberships(df)

    assert info["changed_rows"] == 4
    assert list(fixed["step_id"]) == [-1, 0, 0, 0, 0]
    assert list(fixed["event_member"]) == [False, True, True, True, True]
    assert list(fixed["bad_point"]) == [False, False, False, False, False]
    assert list(fixed["outlier_any"]) == [False, True, True, True, True]
    assert list(fixed["step_t0"]) == ["", 2.0, 2.0, 2.0, 2.0]


def test_normalize_local_step_membership_respects_scope_and_next_step() -> None:
    df = pd.DataFrame(
        {
            "mjd": [1.0, 2.0, 3.0, 4.0, 5.0, 3.0],
            "group": ["A", "A", "A", "A", "A", "B"],
            "step_id": [-1, 0, -1, 1, -1, -1],
            "step_t0": ["", 2.0, "", 4.0, "", ""],
            "step_amp": ["", 10.0, "", 20.0, "", ""],
            "event_member": [False, True, False, True, False, False],
            "bad_point": [False, False, False, False, False, True],
        }
    )

    fixed, _ = _normalize_step_memberships(df)

    assert list(fixed["step_id"]) == [-1, 0, 0, 1, 1, -1]
    assert list(fixed["step_amp"]) == ["", 10.0, 10.0, 20.0, 20.0, ""]
    assert list(fixed["event_member"]) == [False, True, True, True, True, False]
    assert list(fixed["bad_point"]) == [False, False, False, False, False, True]


def test_normalize_global_dm_step_membership_spans_all_scopes() -> None:
    df = pd.DataFrame(
        {
            "mjd": [1.0, 2.0, 3.0, 4.0],
            "group": ["A", "B", "A", "B"],
            "dm_step_global_id": [-1, "dm0", -1, -1],
            "dm_step_global_t0": ["", 2.0, "", ""],
            "dm_step_global_amp": ["", 3.5, "", ""],
            "event_member": [False, True, False, False],
            "bad_point": [False, False, False, False],
        }
    )

    fixed, _ = _normalize_step_memberships(df)

    assert list(fixed["dm_step_global_id"]) == [-1, "dm0", "dm0", "dm0"]
    assert list(fixed["event_member"]) == [False, True, True, True]
    assert list(fixed["outlier_any"]) == [False, True, True, True]


def test_pqc_subprocess_retries_safe_libstempo_after_native_crash(
    tmp_path, monkeypatch
) -> None:
    parfile = tmp_path / "J0636+5128.par"
    parfile.write_text("PSRJ J0636+5128\n", encoding="utf-8")
    out_csv = tmp_path / "J0636+5128_qc.csv"
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            return subprocess.CompletedProcess(args, -11, stdout="", stderr="")
        out_csv.write_text("_timfile,mjd,freq\nx.tim,1.0,1400.0\n", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(oqc, "_validate_pqc_runtime_environment", lambda: None)
    monkeypatch.setattr(oqc.subprocess, "run", fake_run)

    df = run_pqc_for_parfile_subprocess(parfile, out_csv, PTAQCConfig())

    assert len(calls) == 2
    assert "run_pqc_for_parfile_safe_libstempo" in calls[1][2]
    assert df["_timfile"].tolist() == ["x.tim"]


def test_pqc_subprocess_does_not_retry_normal_python_failure(
    tmp_path, monkeypatch
) -> None:
    parfile = tmp_path / "J0000+0000.par"
    parfile.write_text("PSRJ J0000+0000\n", encoding="utf-8")
    out_csv = tmp_path / "J0000+0000_qc.csv"
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            args, 1, stdout="", stderr="ordinary pqc error"
        )

    monkeypatch.setattr(oqc, "_validate_pqc_runtime_environment", lambda: None)
    monkeypatch.setattr(oqc.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="ordinary pqc error"):
        run_pqc_for_parfile_subprocess(parfile, out_csv, PTAQCConfig())

    assert len(calls) == 1
