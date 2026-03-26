"""Tests for cross-pulsar post-QC coincidence reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pleb.qc_report import generate_cross_pulsar_coincidence_report


def test_cross_pulsar_coincidence_report_generation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    qc_dir = run_dir / "qc" / "main"
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Two pulsars with coincident flagged points around MJD 55000.
    pd.DataFrame(
        {
            "mjd": [55000.000, 55010.0],
            "bad_point": [1, 0],
            "transient_member": [0, 0],
        }
    ).to_csv(qc_dir / "J0001+0001_qc.csv", index=False)
    pd.DataFrame(
        {
            "mjd": [55000.004, 55020.0],
            "bad_point": [1, 0],
            "transient_member": [0, 0],
        }
    ).to_csv(qc_dir / "J0002+0002_qc.csv", index=False)

    out_dir = generate_cross_pulsar_coincidence_report(
        run_dir=run_dir,
        window_days=0.01,
        min_pulsars=2,
        include_outliers=True,
        include_events=False,
    )
    assert out_dir is not None
    assert (out_dir / "selected_row_counts.tsv").exists()
    assert (out_dir / "coincident_points.tsv").exists()
    assert (out_dir / "coincidence_clusters.tsv").exists()

    clusters = pd.read_csv(out_dir / "coincidence_clusters.tsv", sep="\t")
    assert len(clusters) >= 1
    assert int(clusters["n_pulsars"].max()) >= 2
