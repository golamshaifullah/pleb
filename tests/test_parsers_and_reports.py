from __future__ import annotations

from pathlib import Path
import math

import numpy as np

from data_combination_pipeline.parsers import read_plklog, read_general2
from data_combination_pipeline.reports import compare_plk, summarize_run


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_read_plklog_parses_table(tmp_path: Path) -> None:
    plk = tmp_path / "x.plk.log"
    _write(
        plk,
        """
Some tempo2 header text
Param Prefit Postfit Uncertainty Difference Fit
F0  100  101  0.5  1.0  1
RAJ 12:00:00.000 12:00:00.100 0.001 0.0 1

Finishing up
""",
    )
    df = read_plklog(plk)
    assert list(df.columns) == ["Param", "Prefit", "Postfit", "Uncertainty", "Difference", "Fit"]
    assert set(df["Param"]) == {"F0", "RAJ"}


def test_compare_plk_vectorized_diff_and_statuses(tmp_path: Path) -> None:
    out_paths = {"plk": tmp_path / "plk"}

    ref = out_paths["plk"] / "J0000+0000_master_plk.log"
    br = out_paths["plk"] / "J0000+0000_feature_plk.log"

    _write(
        ref,
        """
chisq = 10
reduced chisq = 1.25
number of points in fit = 2
Param Prefit Postfit Uncertainty Difference Fit
F0  100  100.0  0.5  0.0  1
RAJ 12:00:00.000 12:00:00.000 0.001 0.0 1
OLDP 0  1.0  1.0  0.0  1
""",
    )

    _write(
        br,
        """
chisq = 9
reduced chisq = 1.12
number of points in fit = 2
Param Prefit Postfit Uncertainty Difference Fit
F0  100  101.0  0.5  1.0  1
RAJ 12:00:00.000 12:00:00.100 0.001 0.0 1
NEWP 0  2.0  1.0  0.0  1
""",
    )

    df = compare_plk("feature", "master", "J0000+0000", out_paths)

    # status buckets exist
    assert set(df["status"]) == {"both", "missing", "new"}

    # numeric diff + sigma for F0
    f0 = df[df["Param"] == "F0"].iloc[0]
    assert f0["status"] == "both"
    assert float(f0["ref_postfit"]) == 100.0
    assert float(f0["branch_postfit"]) == 101.0
    # sigma = 1 / sqrt(0.5^2 + 0.5^2)
    assert math.isclose(float(f0["sigma"]), 1.0 / math.sqrt(0.5**2 + 0.5**2), rel_tol=1e-9)

    # RAJ is treated as HMS diff string
    raj = df[df["Param"] == "RAJ"].iloc[0]
    assert raj["diff"].startswith("+")
    assert ":" in str(raj["diff"])

    # new / missing params
    assert df[df["Param"] == "NEWP"].iloc[0]["status"] == "new"
    assert df[df["Param"] == "OLDP"].iloc[0]["status"] == "missing"


def test_read_general2_and_summarize_run(tmp_path: Path) -> None:
    out_paths = {
        "plk": tmp_path / "plk",
        "general2": tmp_path / "general2",
    }

    plk = out_paths["plk"] / "J0000+0000_master_plk.log"
    gen = out_paths["general2"] / "J0000+0000_master.general2"

    _write(
        plk,
        """
chisq = 10
reduced chisq = 1.25
number of points in fit = 2
Param Prefit Postfit Uncertainty Difference Fit
F0  100  100.0  0.5  0.0  1
""",
    )

    # general2 is parsed only between these markers
    _write(
        gen,
        """
Some tempo2 preamble that should be ignored
Starting general2 plugin
sat post err
55000 1.0 2.0
55001 2.0 2.0
Finished general2 plugin
Some trailer that should also be ignored
""",
    )

    df = read_general2(gen)
    assert len(df) == 2
    assert set(df.columns) >= {"sat", "post", "err"}

    summary = summarize_run(out_paths, "J0000+0000", "master")
    assert summary["chisq"] == 10.0
    assert summary["redchisq"] == 1.25
    assert summary["n_toas"] in (2.0, 2)  # parsed as float sometimes
    assert summary["wrms_post"] is not None
    assert np.isfinite(summary["wrms_post"])
