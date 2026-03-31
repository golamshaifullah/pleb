"""Tests for dataset fix utilities."""

from __future__ import annotations

from pathlib import Path
import warnings

from pleb.tim_utils import is_toa_line
from pleb.dataset_fix import (
    FixDatasetConfig,
    apply_pqc_outliers,
    count_toa_lines,
    extract_flag_values,
    ensure_timfile_flags,
    parse_include_lines,
    remove_patterns_from_par_tim,
    update_alltim_includes,
    update_parfile_jumps,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_count_toa_lines_ignores_directives_and_comments(tmp_path: Path) -> None:
    t = tmp_path / "a.tim"
    _write(
        t,
        """
C comment
FORMAT 1
INCLUDE tims/x.tim
file1 1400 55000 1.0 1.0 -sys SYS1
# another comment
file2 1400 55001 1.0 1.0
JUMP -sys SYS2 0 0
file3 1400 55002 1.0 1.0
""",
    )
    assert count_toa_lines(t) == 3


def test_is_toa_line_allows_filename_starting_with_c_or_C() -> None:
    assert is_toa_line("Cfile 1400 55000 1 1")
    assert is_toa_line("cfile 1400 55000 1 1")
    assert not is_toa_line("C this is comment")


def test_is_toa_line_warns_for_lowercase_c_comment_marker() -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert not is_toa_line("c this is comment")
    assert any("lowercase 'c' comment marker" in str(w.message) for w in rec)


def test_update_alltim_includes_dry_run_and_apply(tmp_path: Path) -> None:
    psr = "J0000+0000"
    psr_dir = tmp_path / psr
    tims_dir = psr_dir / "tims"
    tims_dir.mkdir(parents=True)

    # backend tim with enough TOAs
    back = tims_dir / "backend.tim"
    _write(back, "\n".join([f"f 1400 {55000+i} 1 1" for i in range(12)]) + "\n")

    alltim = psr_dir / f"{psr}_all.tim"
    _write(alltim, "FORMAT 1\n")

    rep = update_alltim_includes(psr_dir, min_toas=10, apply=False, dry_run=True)
    assert rep["to_add"] == ["tims/backend.tim"]

    # apply
    rep2 = update_alltim_includes(
        psr_dir, min_toas=10, apply=True, dry_run=False, backup=False
    )
    assert rep2["added"] == 1
    assert "tims/backend.tim" in parse_include_lines(alltim)


def test_ensure_timfile_flags_only_on_toa_lines(tmp_path: Path) -> None:
    t = tmp_path / "x.tim"
    _write(
        t,
        """
FORMAT 1
file1 1400 55000 1 1
file2 1400 55001 1 1 -pta EPTA
INCLUDE tims/other.tim
""",
    )
    rep = ensure_timfile_flags(
        t, {"-pta": "EPTA", "-sys": "SYS1"}, apply=True, backup=False
    )
    assert rep["changed"] is True

    txt = t.read_text(encoding="utf-8")
    # both TOA lines get -sys; only first TOA line gets -pta
    assert "file1" in txt and "-pta EPTA" in txt
    assert txt.count("-sys SYS1") == 2
    # directive line stays as directive
    assert "INCLUDE tims/other.tim" in txt


def test_extract_flag_values_and_update_parfile_jumps(tmp_path: Path) -> None:
    tim = tmp_path / "b.tim"
    _write(
        tim,
        """
file1 1400 55000 1 1 -sys SYS1
file2 1400 55001 1 1 -sys SYS2
""",
    )
    vals = extract_flag_values(tim, "-sys")
    assert vals == {"SYS1", "SYS2"}

    par = tmp_path / "J0000+0000.par"
    _write(
        par,
        """
EPHEM DE421
CLK TT
JUMP -sys SYS1 0 0
""",
    )

    rep = update_parfile_jumps(
        par,
        jump_flag="-sys",
        jump_values=sorted(vals),
        ensure_ephem="DE440",
        apply=True,
        backup=False,
    )

    assert rep["changed"] is True
    text = par.read_text(encoding="utf-8")
    assert "EPHEM DE440" in text
    assert "JUMP -sys SYS2 0 0" in text


def test_remove_patterns_from_par_tim(tmp_path: Path) -> None:
    par = tmp_path / "x.par"
    tim = tmp_path / "x.tim"

    _write(par, "A 1\nBAD NRT.NUPPI. foo\nB 2\n")
    _write(tim, "file 1400 55000 1 1\nNRT.NUPPI. garbage\n")

    rep = remove_patterns_from_par_tim(
        par, tim, patterns=["NRT.NUPPI."], apply=True, backup=False
    )
    assert rep["par_removed"] == 1
    assert rep["tim_removed"] == 1

    assert "NRT.NUPPI." not in par.read_text(encoding="utf-8")
    assert "NRT.NUPPI." not in tim.read_text(encoding="utf-8")


def test_apply_pqc_outliers_can_write_pqc_flag_labels(tmp_path: Path) -> None:
    psr = "J0000+0000"
    psr_dir = tmp_path / psr
    tim = psr_dir / "tims" / "BACKEND.tim"
    _write(
        tim,
        ("FORMAT 1\n" "f 1400 55000 1 1\n" "f 1400 55001 1 1\n" "f 1400 55002 1 1\n"),
    )

    qc_root = tmp_path / "qc"
    qc_csv = qc_root / "main" / f"{psr}_qc.csv"
    _write(
        qc_csv,
        (
            "_timfile,mjd,bad_point,step_member,transient_id\n"
            "BACKEND.tim,55000,False,False,-1\n"
            "BACKEND.tim,55001,True,False,-1\n"
            "BACKEND.tim,55002,False,True,-1\n"
        ),
    )

    cfg = FixDatasetConfig(
        apply=True,
        backup=False,
        qc_results_dir=qc_root,
        qc_branch="main",
        qc_remove_outliers=False,
        qc_write_pqc_flag=True,
    )
    rep = apply_pqc_outliers(psr_dir, cfg)

    assert rep["changed_files"] == 1
    assert rep["pqc_flagged"] == 3
    text = tim.read_text(encoding="utf-8")
    assert "-pqc good" in text
    assert "-pqc bad" in text
    assert "-pqc event_step" in text


def test_apply_pqc_outliers_comments_use_c_space_and_strip_leading_ws(
    tmp_path: Path,
) -> None:
    psr = "J0000+0001"
    psr_dir = tmp_path / psr
    tim = psr_dir / "tims" / "BACKEND.tim"
    _write(
        tim,
        ("FORMAT 1\n" "\tf 1400 56000 1 1\n"),
    )

    qc_root = tmp_path / "qc"
    qc_csv = qc_root / "main" / f"{psr}_qc.csv"
    _write(
        qc_csv,
        ("_timfile,mjd,bad_point\n" "BACKEND.tim,56000,True\n"),
    )

    cfg = FixDatasetConfig(
        apply=True,
        backup=False,
        qc_results_dir=qc_root,
        qc_branch="main",
        qc_remove_outliers=True,
        qc_action="comment",
        qc_comment_prefix="CQC_OUTLIER",
    )
    rep = apply_pqc_outliers(psr_dir, cfg)

    assert rep["changed_files"] == 1
    lines = tim.read_text(encoding="utf-8").splitlines()
    toa_comments = [ln for ln in lines if ln.startswith("C ")]
    assert len(toa_comments) == 1
    assert toa_comments[0].startswith("C QC_OUTLIER ")
    assert "\tf 1400" not in toa_comments[0]
