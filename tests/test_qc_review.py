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
