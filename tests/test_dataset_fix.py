"""Tests for dataset fix utilities."""

from __future__ import annotations

from pathlib import Path
import re
import warnings
import pytest

from pleb.tim_utils import is_toa_line
from pleb.system_flag_inference import parse_tim_toa_table
from pleb.dataset_fix import (
    FixDatasetConfig,
    _find_qc_csvs,
    _variant_name_from_alltim,
    apply_pqc_outliers,
    build_variant_reference_jump_pars,
    count_toa_lines,
    extract_flag_values,
    ensure_timfile_flags,
    generate_alltim_variants,
    parse_include_lines,
    remove_patterns_from_par_tim,
    update_alltim_includes,
    update_parfile_jumps,
)
from pleb.tim_utils import extract_flag_value_from_line, parse_tim_flags_from_line


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


def test_parse_tim_flags_preserves_valueless_flags() -> None:
    line = (
        "toa 1400 55000 1 1 "
        "-xyz -aaa -bcd -pta EPTA -padd -0.193655 -gis"
    )

    flags = parse_tim_flags_from_line(line)

    assert flags["-xyz"] == ""
    assert flags["-aaa"] == ""
    assert flags["-bcd"] == ""
    assert flags["-pta"] == "EPTA"
    assert flags["-padd"] == "-0.193655"
    assert flags["-gis"] == ""
    assert extract_flag_value_from_line(line, "-pta") == "EPTA"
    assert extract_flag_value_from_line(line, "-xyz") == ""


def test_parse_tim_toa_table_does_not_swallow_next_flag_after_valueless_flag(
    tmp_path: Path,
) -> None:
    tim = tmp_path / "EFF.EBPP.2639.tim"
    _write(
        tim,
        "\n".join(
            [
                "FORMAT 1",
                (
                    "Cc062251.align.pazr.30min 2625.499 56486.8600580664295 29.730 g "
                    "-sys EFF.EBPP.2639 -padd 0.112837 -addsat +1 "
                    "-gis -pta EPTA -group EFF.EBPP.2639"
                ),
            ]
        )
        + "\n",
    )

    df = parse_tim_toa_table(tim)
    flags = df.iloc[0]["flags"]

    assert flags["-gis"] == ""
    assert flags["-pta"] == "EPTA"
    assert flags["-group"] == "EFF.EBPP.2639"


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


def test_apply_pqc_outliers_raises_when_qc_csv_missing_by_default(
    tmp_path: Path,
) -> None:
    psr = "J0000+0002"
    psr_dir = tmp_path / psr
    tim = psr_dir / "tims" / "BACKEND.tim"
    _write(tim, "FORMAT 1\nf 1400 56000 1 1\n")

    cfg = FixDatasetConfig(
        apply=True,
        backup=False,
        qc_results_dir=tmp_path / "qc",
        qc_branch="main",
        qc_remove_outliers=True,
    )
    cfg.qc_results_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match=re.escape(f"{psr}_qc.csv")):
        apply_pqc_outliers(psr_dir, cfg)


def test_apply_pqc_outliers_can_opt_out_of_missing_qc_csv_failure(
    tmp_path: Path,
) -> None:
    psr = "J0000+0003"
    psr_dir = tmp_path / psr
    tim = psr_dir / "tims" / "BACKEND.tim"
    _write(tim, "FORMAT 1\nf 1400 56000 1 1\n")

    cfg = FixDatasetConfig(
        apply=True,
        backup=False,
        qc_results_dir=tmp_path / "qc",
        qc_branch="main",
        qc_remove_outliers=True,
        qc_require_csv=False,
    )

    rep = apply_pqc_outliers(psr_dir, cfg)
    assert rep["qc_csv"] is None
    assert rep["matched"] == 0
    assert rep["changed"] is False


def test_find_qc_csvs_discovers_variant_specific_files(tmp_path: Path) -> None:
    psr = "J0000+0004"
    qc_root = tmp_path / "qc"
    _write(qc_root / "main" / f"{psr}.legacy_qc.csv", "_timfile,mjd,bad_point\n")
    _write(qc_root / "main" / f"{psr}.new_qc.csv", "_timfile,mjd,bad_point\n")

    cfg = FixDatasetConfig(
        qc_results_dir=qc_root,
        qc_branch="main",
        qc_remove_outliers=True,
    )

    found = _find_qc_csvs(psr, cfg)
    assert [p.name for p in found] == [f"{psr}.legacy_qc.csv", f"{psr}.new_qc.csv"]


def test_apply_pqc_outliers_can_merge_variant_specific_qc_csvs(
    tmp_path: Path,
) -> None:
    psr = "J0000+0005"
    psr_dir = tmp_path / psr
    tims = psr_dir / "tims"
    legacy_tim = tims / "LEGACY.tim"
    new_tim = tims / "NEW.tim"
    _write(
        legacy_tim,
        "FORMAT 1\nf 1400 56000 1 1\n",
    )
    _write(
        new_tim,
        "FORMAT 1\nf 1400 56001 1 1\n",
    )

    qc_root = tmp_path / "qc"
    _write(
        qc_root / "main" / f"{psr}.legacy_qc.csv",
        "_timfile,mjd,bad_point\nLEGACY.tim,56000,True\n",
    )
    _write(
        qc_root / "main" / f"{psr}.new_qc.csv",
        "_timfile,mjd,bad_point\nNEW.tim,56001,True\n",
    )

    cfg = FixDatasetConfig(
        apply=True,
        backup=False,
        qc_results_dir=qc_root,
        qc_branch="main",
        qc_remove_outliers=True,
        qc_action="comment",
    )
    rep = apply_pqc_outliers(psr_dir, cfg)

    assert rep["changed_files"] == 2
    assert sorted(Path(p).name for p in rep["qc_csvs"]) == [
        f"{psr}.legacy_qc.csv",
        f"{psr}.new_qc.csv",
    ]
    assert legacy_tim.read_text(encoding="utf-8").splitlines()[1].startswith(
        "C QC_OUTLIER "
    )
    assert new_tim.read_text(encoding="utf-8").splitlines()[1].startswith(
        "C QC_OUTLIER "
    )


def test_apply_pqc_outliers_treats_empty_variant_manifest_as_skip(
    tmp_path: Path,
) -> None:
    psr = "J0000+0006"
    psr_dir = tmp_path / psr
    tim = psr_dir / "tims" / "BACKEND.tim"
    _write(tim, "FORMAT 1\nf 1400 56000 1 1\n")

    qc_root = tmp_path / "qc"
    _write(
        qc_root / "qc_summary.tsv",
        (
            "pulsar\tvariant\tbranch\tqc_status\tqc_csv\tqc_error\n"
            f"{psr}\tlegacy\tmain\tempty_variant\t/path/{psr}.legacy_qc.csv\t\n"
        ),
    )

    cfg = FixDatasetConfig(
        apply=True,
        backup=False,
        qc_results_dir=qc_root,
        qc_branch="main",
        qc_remove_outliers=True,
        qc_require_csv=True,
    )

    rep = apply_pqc_outliers(psr_dir, cfg)
    assert rep["qc_csv"] is None
    assert rep["qc_statuses"] == ["empty_variant"]
    assert rep["changed"] is False


def test_variant_name_parser_accepts_underscore_and_legacy_dot_forms() -> None:
    psr = "J0000+0000"
    assert _variant_name_from_alltim(psr, Path(f"{psr}_legacy_all.tim")) == "legacy"
    assert _variant_name_from_alltim(psr, Path(f"{psr}_all.legacy.tim")) == "legacy"
    assert _variant_name_from_alltim(psr, Path(f"{psr}_all.tim")) == "base"


def test_generate_alltim_variants_uses_underscore_naming(tmp_path: Path) -> None:
    psr = "J0000+0000"
    psr_dir = tmp_path / psr
    tims_dir = psr_dir / "tims"
    tims_dir.mkdir(parents=True)

    _write(
        psr_dir / f"{psr}_all.tim",
        "FORMAT 1\nINCLUDE tims/LEGACY.tim\nINCLUDE tims/NEW.tim\n",
    )
    _write(tims_dir / "LEGACY.tim", "f 1400 55000 1 1 -sys LEGACY\n")
    _write(tims_dir / "NEW.tim", "f 1400 55001 1 1 -sys NEW\n")

    cls = tmp_path / "backend_classifications.toml"
    cls.write_text(
        """
[classes.legacy]
systems = ["LEGACY"]

[classes.new]
systems = ["NEW"]
""",
        encoding="utf-8",
    )
    variants = tmp_path / "alltim_variants.toml"
    variants.write_text(
        """
[variants.legacy]
include_classes = ["legacy"]

[variants.new]
include_classes = ["new"]
""",
        encoding="utf-8",
    )

    cfg = FixDatasetConfig(
        apply=True,
        dry_run=False,
        backup=False,
        backend_classifications_path=str(cls),
        alltim_variants_path=str(variants),
    )
    rep = generate_alltim_variants(psr_dir, cfg)

    assert (psr_dir / f"{psr}_legacy_all.tim").exists()
    assert (psr_dir / f"{psr}_new_all.tim").exists()
    assert "legacy" in rep["variants"]
    assert "new" in rep["variants"]


def test_build_variant_reference_jump_pars_uses_underscore_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    psr = "J0000+0000"
    psr_dir = tmp_path / psr
    psr_dir.mkdir(parents=True)
    _write(psr_dir / f"{psr}.par", "PSRJ J0000+0000\n")
    _write(
        psr_dir / f"{psr}_legacy_all.tim",
        "FORMAT 1\nINCLUDE tims/BACKEND.tim\n",
    )
    _write(
        psr_dir / "tims" / "BACKEND.tim",
        "toa 1400 55000 1 1 -sys SYSA\n"
        "toa 1400 55001 2 1 -sys SYSB\n",
    )

    monkeypatch.setattr("pleb.dataset_fix.build_singularity_prefix", lambda *a, **k: [])
    monkeypatch.setattr("pleb.dataset_fix.run_subprocess", lambda *a, **k: 0)
    monkeypatch.setattr("pleb.dataset_fix._parse_tempo2_redchisq", lambda *a, **k: 1.0)

    cfg = FixDatasetConfig(
        apply=True,
        dry_run=False,
        backup=False,
        tempo2_home_dir=tmp_path,
        tempo2_dataset_name=".",
        tempo2_singularity_image=tmp_path / "tempo2.sif",
    )
    (tmp_path / "tempo2.sif").write_text("", encoding="utf-8")

    rep = build_variant_reference_jump_pars(psr_dir, cfg)
    variant = rep["variants"]["legacy"]
    par_out = psr_dir / f"{psr}_legacy.par"

    assert par_out.exists()
    assert variant["par_out"] == str(par_out)
    assert Path(str(variant["csv"])).name == f"{psr}_jump_reference_legacy.csv"
