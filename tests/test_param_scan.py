"""Tests for parameter scan utilities."""

from __future__ import annotations

from pleb.param_scan import (
    parse_candidate_specs,
    apply_candidate_to_par_text,
    build_typical_candidates,
)


def test_parse_candidate_specs_group_and_values() -> None:
    cands = parse_candidate_specs(["F2", "F3=0", "F2+F3", "raw:JUMP -sys P200 0 1"])
    assert len(cands) == 4
    assert cands[0].params[0][0] == "F2"
    assert cands[1].params[0] == ("F3", "0")
    assert {p[0] for p in cands[2].params} == {"F2", "F3"}
    assert cands[3].raw_lines[0].startswith("JUMP")


def test_apply_candidate_sets_fit_flag_and_inserts_missing_param() -> None:
    base = """
C example par
F0 1.0 0
F1 -1e-15
RAJ 12:00:00.0 0
""".strip() + "\n"

    cand = parse_candidate_specs(["F0+F2=0"])[0]
    out = apply_candidate_to_par_text(base, cand)
    # F0 fit flag should become 1
    assert "F0 1.0 1" in out
    # F2 should be appended as a simple scalar fitted param
    assert "F2 0 1" in out


def test_apply_candidate_preserves_inline_comment() -> None:
    base = "F0 1.0 0  # fixed\n"
    cand = parse_candidate_specs(["F0"])[0]
    out = apply_candidate_to_par_text(base, cand)
    assert "# fixed" in out
    assert "F0 1.0 1" in out


def test_build_typical_candidates_includes_px_and_ell1_derivs() -> None:
    par = """
PSRJ J1234+5678
RAJ 12:00:00.0 0
DECJ 01:00:00.0 0
BINARY ELL1
PB 10.0 1
A1 1.0 1
EPS1 0 1
EPS2 0 1
TASC 55000 1
""".strip() + "\n"
    cands = build_typical_candidates(par, {"redchisq": 1.0}, dm_redchisq_threshold=2.0)
    labels = {c.label for c in cands}
    assert "PX" in labels
    # ELL1 profile includes these
    assert "PBDOT" in labels
    assert "XDOT" in labels
    assert "EPS1DOT+EPS2DOT" in labels


def test_build_typical_candidates_dm_derivs_trigger_only_when_no_binary_and_high_chisq() -> None:
    par = """
PSRJ J1234+5678
F0 1.0 1
DM 10.0 1
""".strip() + "\n"
    cands_low = build_typical_candidates(par, {"redchisq": 1.1}, dm_redchisq_threshold=2.0, dm_max_order=3)
    labels_low = {c.label for c in cands_low}
    # Still includes PX by default (missing)
    assert "PX" in labels_low
    assert all(not l.startswith("DM") for l in labels_low)

    cands_high = build_typical_candidates(par, {"redchisq": 3.0}, dm_redchisq_threshold=2.0, dm_max_order=3)
    labels_high = {c.label for c in cands_high}
    assert "DM1" in labels_high
    assert "DM1+DM2" in labels_high
    assert "DM1+DM2+DM3" in labels_high
