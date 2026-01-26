"""Integration tests for the QC pipeline."""

import importlib.util
import pytest

from pleb.outlier_qc import PTAQCConfig, run_pta_qc_for_parfile

def test_pta_qc_missing_dependency_errors(tmp_path):
    # If pta_qc is installed in this environment, skip (we can't run it without libstempo + real par/tim).
    if importlib.util.find_spec("pta_qc") is not None:
        pytest.skip("pta_qc present; skipping missing-dependency test.")
    par = tmp_path / "J0000+0000.par"
    par.write_text("DUMMY 1\n")
    out_csv = tmp_path / "qc.csv"
    with pytest.raises(RuntimeError) as e:
        run_pta_qc_for_parfile(par, out_csv, PTAQCConfig())
    assert "pta_qc is not installed" in str(e.value)
