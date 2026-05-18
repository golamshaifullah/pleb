from __future__ import annotations

from pathlib import Path

import pandas as pd

from pleb.qc_review import (
    apply_overrides,
    append_overrides,
    empty_overrides,
    load_qc_frames,
    make_override_rows,
    write_overrides,
)


def _write_qc(path: Path) -> None:
    pd.DataFrame(
        {
            "mjd": [58000.0, 58000.0, 58001.0],
            "resid_us": [0.1, 9.5, -1.0],
            "sigma_us": [1.0, 1.0, 1.0],
            "freq": [1400.0, 1400.0, 800.0],
            "backend": ["A", "A", "B"],
            "_timfile": ["tims/a.tim", "tims/a.tim", "tims/b.tim"],
            "bad_point": [False, False, True],
        }
    ).to_csv(path, index=False)


def _write_general2(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "Starting general2 plugin",
                "sat freq pre post err",
                "58000.0 1400.0 1.0e-06 5.0e-07 1.0e-07",
                "58000.0 1400.0 2.0e-06 6.0e-07 1.1e-07",
                "58001.0 800.0 3.0e-06 -7.0e-07 2.0e-07",
                "Finished general2 plugin",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_manual_override_targets_one_duplicate_mjd_row(tmp_path: Path) -> None:
    qc_path = tmp_path / "J0000+0000_qc.csv"
    _write_qc(qc_path)
    qc = load_qc_frames(tmp_path)

    # Two rows have identical MJD. Override only the second row by review_id.
    target = qc.iloc[[1]]
    overrides = make_override_rows(
        target,
        action="mark_bad",
        reason="obvious isolated residual",
        reviewer="tester",
        timestamp="2026-04-28T00:00:00+00:00",
    )
    reviewed = apply_overrides(qc, overrides)

    assert reviewed.loc[0, "reviewed_decision"] == "KEEP"
    assert reviewed.loc[1, "reviewed_decision"] == "BAD_TOA"
    assert reviewed.loc[2, "reviewed_decision"] == "BAD_TOA"
    assert reviewed.loc[1, "manual_reason"] == "obvious isolated residual"


def test_keep_override_clears_automatic_bad(tmp_path: Path) -> None:
    qc_path = tmp_path / "J0000+0000_qc.csv"
    _write_qc(qc_path)
    qc = load_qc_frames(tmp_path)

    target = qc.iloc[[2]]
    overrides = make_override_rows(target, action="keep", reason="expert veto")
    reviewed = apply_overrides(qc, overrides)

    assert reviewed.loc[2, "auto_decision"] == "BAD_TOA"
    assert reviewed.loc[2, "reviewed_decision"] == "KEEP"
    assert reviewed.loc[2, "reviewed_bad_point"] is False or not bool(
        reviewed.loc[2, "reviewed_bad_point"]
    )


def test_auto_decision_includes_new_feature_flags(tmp_path: Path) -> None:
    qc_path = tmp_path / "J0000+0000_qc.csv"
    pd.DataFrame(
        {
            "mjd": [58000.0, 58001.0, 58002.0, 58003.0],
            "resid_us": [0.1, 0.2, 0.3, 0.4],
            "sigma_us": [1.0, 1.0, 1.0, 1.0],
            "freq": [1400.0, 1400.0, 1400.0, 1400.0],
            "backend": ["A", "A", "A", "A"],
            "_timfile": ["tims/a.tim"] * 4,
            "bad_ou": [True, False, False, False],
            "bad_hard": [False, True, False, False],
            "step_id": [-1, -1, 0, -1],
            "dm_step_id": [-1, -1, -1, "dm-step"],
        }
    ).to_csv(qc_path, index=False)

    qc = load_qc_frames(tmp_path)

    assert list(qc["auto_decision"]) == ["BAD_TOA", "BAD_TOA", "EVENT", "EVENT"]


def test_override_csv_round_trip(tmp_path: Path) -> None:
    qc_path = tmp_path / "J0000+0000_qc.csv"
    _write_qc(qc_path)
    qc = load_qc_frames(tmp_path)
    overrides = append_overrides(
        empty_overrides(), make_override_rows(qc.iloc[[0]], action="mark_event")
    )

    path = tmp_path / "manual_qc_overrides.csv"
    write_overrides(overrides, path)
    loaded = pd.read_csv(path)

    assert list(loaded.columns)[0] == "override_id"
    assert loaded.loc[0, "manual_action"] == "mark_event"


def test_load_qc_frames_skips_review_exports_in_qc_review_dir(tmp_path: Path) -> None:
    raw = tmp_path / "J0000+0000_qc.csv"
    _write_qc(raw)
    review_dir = tmp_path / "qc_review"
    review_dir.mkdir()
    _write_qc(review_dir / "reviewed_qc.csv")

    qc = load_qc_frames(tmp_path)

    assert len(qc) == 3
    assert set(qc["qc_csv"].astype(str)) == {raw.resolve().name}


def test_load_qc_frames_rejects_explicit_reviewed_qc_csv(tmp_path: Path) -> None:
    reviewed = tmp_path / "qc_review" / "reviewed_qc.csv"
    reviewed.parent.mkdir()
    _write_qc(reviewed)

    try:
        load_qc_frames(csvs=[reviewed])
    except ValueError as exc:
        assert "Refusing to load reviewed QC artifact" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected reviewed_qc.csv to be rejected")


def test_load_qc_frames_attaches_tempo2_general2_postfit_columns(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    qc_path = run_dir / "qc" / "step2_detect" / "J0000+0000_qc.csv"
    qc_path.parent.mkdir(parents=True)
    _write_qc(qc_path)
    _write_general2(run_dir / "general2" / "J0000+0000_step2_detect.general2")

    qc = load_qc_frames(run_dir)

    assert "tempo2_post" in qc.columns
    assert "tempo2_post_us" in qc.columns
    assert "tempo2_pre" in qc.columns
    assert list(qc["tempo2_post"].round(12)) == [5.0e-07, 6.0e-07, -7.0e-07]
    assert list(qc["tempo2_pre"].round(12)) == [1.0e-06, 2.0e-06, 3.0e-06]
    assert list(qc["tempo2_post_us"].round(3)) == [0.5, 0.6, -0.7]
