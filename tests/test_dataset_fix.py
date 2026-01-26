"""Tests for dataset fix utilities."""

from __future__ import annotations

from pathlib import Path

from pleb.dataset_fix import (
    count_toa_lines,
    parse_include_lines,
    update_alltim_includes,
    ensure_timfile_flags,
    extract_flag_values,
    update_parfile_jumps,
    remove_patterns_from_par_tim,
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
    rep2 = update_alltim_includes(psr_dir, min_toas=10, apply=True, dry_run=False, backup=False)
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
    rep = ensure_timfile_flags(t, {"-pta": "EPTA", "-sys": "SYS1"}, apply=True, backup=False)
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

    rep = remove_patterns_from_par_tim(par, tim, patterns=["NRT.NUPPI."], apply=True, backup=False)
    assert rep["par_removed"] == 1
    assert rep["tim_removed"] == 1

    assert "NRT.NUPPI." not in par.read_text(encoding="utf-8")
    assert "NRT.NUPPI." not in tim.read_text(encoding="utf-8")
