"""Duplicate exclusions must preserve TOAs and their decision provenance."""

from pathlib import Path
import hashlib

import pytest

from pleb.config import PipelineConfig
from pleb.dataset_fix import dedupe_timfile_toas, fix_pulsar_dataset
from pleb.pipeline import _build_fixdataset_config
from pleb.tim_utils import is_toa_line

TOA = "obs.ar 1456.000000 58000.000000000000 1.0 jbroach -bw 200 -pqc good"


@pytest.mark.parametrize("matching", ["exact", "tolerance", "same_obs"])
def test_duplicate_comments_preserve_rows_flags_and_retained_identity(
    tmp_path, matching
):
    tim = tmp_path / "JBO.ROACH.1520.tim"
    second = TOA
    kwargs = {}
    reason = "exact_first_four_columns"
    if matching == "tolerance":
        second = TOA.replace("58000.000000000000", "58000.000000000001").replace(
            "1.0 jbroach", "1.1 jbroach"
        )
        kwargs = {"mjd_tol_sec": 0.000001, "freq_tol_mhz": 0.00001}
        reason = "time_frequency_tolerance"
    elif matching == "same_obs":
        second = TOA.replace("1456.000000", "1456.001000").replace(
            "1.0 jbroach", "1.1 jbroach"
        )
        kwargs = {"same_obs_same_bw_globs": ["JBO.ROACH*.tim"]}
        reason = "same_observation_same_bandwidth"
    original_comment = "C QC_OUTLIER " + TOA
    tim.write_text("\n".join(["FORMAT 1", original_comment, TOA, second]) + "\n")
    report = dedupe_timfile_toas(tim, apply=True, backup=False, **kwargs)
    lines = tim.read_text().splitlines()
    assert len(lines) == 4
    assert lines[1] == original_comment  # A previous exclusion is not a survivor.
    assert lines[2] == TOA
    assert lines[3] == "C DUPLICATE_TOA " + second
    assert sum(is_toa_line(line) for line in lines) == 1
    assert report["removed"] == 0
    assert report["commented"] == report["duplicates"] == 1
    decision = report["decisions"][0]
    assert decision["duplicate_line"] == 4
    assert decision["retained_line"] == 3
    assert decision["reason"] == reason
    assert decision["retained_row_sha256"] == hashlib.sha256(TOA.encode()).hexdigest()
    before = tim.read_bytes()
    again = dedupe_timfile_toas(tim, apply=True, backup=False, **kwargs)
    assert not again["changed"]
    assert again["duplicates"] == 0
    assert tim.read_bytes() == before


@pytest.mark.parametrize("kwargs", [{"dry_run": True, "apply": True}, {"apply": False}])
def test_duplicate_preview_never_writes_or_creates_backup(tmp_path, kwargs):
    tim = tmp_path / "NRT.NUPPI.1484.tim"
    original = "FORMAT 1\n" + TOA + "\n" + TOA + "\n"
    tim.write_text(original)
    report = dedupe_timfile_toas(tim, **kwargs)
    assert report["commented"] == 1
    assert report["removed"] == 0
    assert tim.read_text() == original
    assert not tim.with_suffix(".tim.orig").exists()


def test_deletion_requires_explicit_opt_in(tmp_path):
    tim = tmp_path / "backend.tim"
    tim.write_text("FORMAT 1\n" + TOA + "\n" + TOA + "\n")
    report = dedupe_timfile_toas(tim, apply=True, backup=False, action="delete")
    assert report["removed"] == 1
    assert report["commented"] == 0
    assert tim.read_text().splitlines() == ["FORMAT 1", TOA]


def test_bad_action_fails_before_writing(tmp_path):
    tim = tmp_path / "backend.tim"
    tim.write_text(TOA + "\n")
    with pytest.raises(ValueError, match="Unsupported dedupe action"):
        dedupe_timfile_toas(tim, apply=True, action="invalid")
    assert tim.read_text() == TOA + "\n"


def test_config_roundtrip_and_pipeline_propagate_comment_settings(tmp_path):
    cfg = PipelineConfig.from_dict(
        {
            "home_dir": str(tmp_path),
            "dataset_name": "dataset",
            "singularity_image": str(tmp_path / "tempo2.sif"),
            "fix_dedupe_action": "comment",
            "fix_dedupe_comment_prefix": "# DUPLICATE",
            "fix_dedupe_same_obs_same_bw_globs": ["WSRT.*.tim", "JBO.ROACH*.tim"],
            "fix_update_alltim_includes": False,
            "fix_insert_missing_jumps": False,
            "fix_remove_overlaps_exact": False,
            "fix_remove_patterns": [],
        }
    )
    cfg = PipelineConfig.from_dict(cfg.to_dict())
    fix = _build_fixdataset_config(cfg, apply=True)
    assert fix.dedupe_action == "comment"
    assert fix.dedupe_comment_prefix == "# DUPLICATE"
    assert fix.dedupe_same_obs_same_bw_globs == ["WSRT.*.tim", "JBO.ROACH*.tim"]
    psr = tmp_path / "dataset/J0000+0000"
    tim = psr / "tims/JBO.ROACH.1520.tim"
    tim.parent.mkdir(parents=True)
    tim.write_text("FORMAT 1\n" + TOA + "\n" + TOA + "\n")
    report = fix_pulsar_dataset(psr, fix)
    step = next(
        s["dedupe_toas_within_tim"]
        for s in report["steps"]
        if "dedupe_toas_within_tim" in s
    )
    assert step[0]["commented"] == 1
    assert tim.read_text().splitlines()[-1] == "C DUPLICATE " + TOA


def test_cleanup_profile_requests_comment_only_selection_changes():
    path = (
        Path(__file__).parents[1]
        / "configs/runs/fixdataset/epta_dr3_duplicate_comments.toml"
    )
    cfg = PipelineConfig.load(path)
    fix = _build_fixdataset_config(cfg, apply=True)
    assert fix.dedupe_action == "comment"
    assert not fix.backup
    assert not fix.update_alltim_includes
    assert not fix.insert_missing_jumps
    assert not fix.infer_system_flags
    assert fix.remove_patterns == []
    assert "JBO.ROACH*.tim" in fix.dedupe_same_obs_same_bw_globs
