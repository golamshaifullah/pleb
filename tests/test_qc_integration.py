"""Integration tests for the QC pipeline."""

import importlib.util
import pytest

from pleb.outlier_qc import PTAQCConfig, run_pqc_for_parfile


def test_pqc_missing_dependency_errors(tmp_path):
    # If pqc is installed in this environment, skip (we can't run it without libstempo + real par/tim).
    if importlib.util.find_spec("pqc") is not None:
        pytest.skip("pqc present; skipping missing-dependency test.")
    par = tmp_path / "J0000+0000.par"
    par.write_text("DUMMY 1\n")
    out_csv = tmp_path / "qc.csv"
    with pytest.raises(RuntimeError) as e:
        run_pqc_for_parfile(par, out_csv, PTAQCConfig())
    assert "pqc is not installed" in str(e.value)
