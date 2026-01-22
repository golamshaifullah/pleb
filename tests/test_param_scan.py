from __future__ import annotations

from data_combination_pipeline.param_scan import parse_candidate_specs, apply_candidate_to_par_text


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
