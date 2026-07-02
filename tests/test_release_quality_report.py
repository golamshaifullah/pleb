from pathlib import Path

import pandas as pd

from pleb.release_quality_report import (
    ReleaseQualityThresholds,
    generate_release_quality_report,
)


def test_generate_release_quality_report_from_qc_csv(tmp_path: Path):
    run_dir = tmp_path / "run"
    qc_dir = run_dir / "qc" / "main"
    qc_dir.mkdir(parents=True)
    qc_csv = qc_dir / "J0000+0000_qc.csv"
    pd.DataFrame(
        {
            "mjd": [59000.0, 59001.0, 59002.0, 59003.0],
            "residual_us": [0.1, -0.2, 5.0, 1.2],
            "group": ["A", "A", "B", "B"],
            "outlier_any": [False, False, True, True],
            "transient_member": [False, False, False, True],
        }
    ).to_csv(qc_csv, index=False)
    pd.DataFrame(
        [
            {
                "pulsar": "J0000+0000",
                "variant": "base",
                "branch": "main",
                "qc_csv": str(qc_csv),
                "qc_status": "success",
            }
        ]
    ).to_csv(run_dir / "qc" / "qc_summary.tsv", sep="\t", index=False)

    result = generate_release_quality_report(
        run_dir,
        thresholds=ReleaseQualityThresholds(
            yellow_bad_fraction=0.2,
            red_bad_fraction=0.9,
            yellow_review_fraction=0.2,
            red_review_fraction=0.9,
        ),
        include_per_pulsar_pages=True,
        top_n=10,
    )

    assert result is not None
    assert result.pdf_path.exists()
    scorecard = pd.read_csv(result.scorecard_path, sep="\t")
    assert scorecard.loc[0, "pulsar"] == "J0000+0000"
    assert int(scorecard.loc[0, "bad_toa"]) == 1
    assert int(scorecard.loc[0, "review_event"]) == 1
    risks = pd.read_csv(result.backend_risks_path, sep="\t")
    assert set(risks["backend"]) == {"A", "B"}
