from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_system_flags.py"


def test_audit_system_flags_reports_mismatches(tmp_path: Path) -> None:
    dataset_root = tmp_path / "EPTA-DR3" / "epta-dr3-data"
    tim_dir = dataset_root / "J1909-3744" / "tims"
    tim_dir.mkdir(parents=True)
    (tim_dir / "EFF.P200.1380.tim").write_text(
        "\n".join(
            [
                "FORMAT 1",
                "toa1 1380.0 58000.0 1.0 ao -sys WRONG -group WRONG -pta WRONG",
                "toa2 1381.0 58001.0 1.0 ao -sys WRONG -group WRONG -pta WRONG",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "audit"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(dataset_root),
            "--pulsars",
            "J1909-3744",
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    summary = pd.read_csv(out_dir / "system_flag_audit_summary.csv")
    mismatch = pd.read_csv(out_dir / "system_flag_audit_mismatches.csv")

    assert len(summary) == 1
    assert summary.loc[0, "timfile"] == "EFF.P200.1380.tim"
    assert int(summary.loc[0, "sys_mismatch_count"]) == 2
    assert int(summary.loc[0, "group_mismatch_count"]) == 2
    assert int(summary.loc[0, "pta_mismatch_count"]) == 2
    assert len(mismatch) == 2
    assert set(mismatch["inferred_sys"]) == {"EFF.P200.1380", "EFF.P200.1381"}
