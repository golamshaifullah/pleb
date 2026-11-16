"""Exercise real review widgets; optional when the GUI extra is unavailable."""

import hashlib
from pathlib import Path
import sys

import pandas as pd
import pytest


def test_pulsar_switching_and_review_saves_stay_on_the_selected_table(
    tmp_path, monkeypatch
):
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("plotly")
    from pleb.qc_review import load_overrides

    pulsars = ["J1234+5678", "J1235+5678"]
    for pulsar in pulsars:
        pd.DataFrame(
            dict(
                pulsar=[pulsar] * 2,
                variant=["combined"] * 2,
                mjd=[58000, 58001],
                freq=[1400, 1400],
                backend=["A", "A"],
                tempo2_post_us=[0.1, -0.1],
                tempo2_err_us=[1, 1],
                timfile=["A.tim", "A.tim"],
                bad_point=[False, False],
                event_member=[False, False],
            )
        ).to_csv(tmp_path / f"{pulsar}_combined_qc.csv", index=False)
    raw = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in tmp_path.glob("*_qc.csv")
    }
    monkeypatch.setattr(sys, "argv", ["review", "--run-dir", str(tmp_path)])
    script = Path(__file__).resolve().parents[1] / "pleb/review_app.py"
    app = testing.AppTest.from_file(str(script), default_timeout=30).run()

    def widget(kind, label):
        return next(w for w in getattr(app, kind) if w.label == label)

    def check(pulsar):
        assert not app.exception, [e.message for e in app.exception]
        assert not app.error, [e.value for e in app.error]
        assert widget("selectbox", "Pulsar").value == pulsar
        assert app.session_state["qc_df"].pulsar.unique().tolist() == [pulsar]

    check(pulsars[0])
    assert widget("selectbox", "Pulsar").options == pulsars
    assert widget("selectbox", "Variant").options == ["combined [post-fit]"]
    for pulsar in [pulsars[1], pulsars[0], pulsars[1], pulsars[0]]:
        widget("selectbox", "Pulsar").select(pulsar).run()
        check(pulsar)
    widget("button", "Next →").click().run()
    check(pulsars[1])
    widget("button", "← Prev").click().run()
    check(pulsars[0])

    identity = app.session_state["qc_df"].iloc[0].review_id
    widget("text_input", "Reviewer").input("Synthetic UI test")
    widget("text_input", "Reason").input("Private synthetic example")
    widget("text_area", "Selected review IDs").input(identity).run()
    widget("button", "Apply to selected").click().run()
    check(pulsars[0])
    history_path = tmp_path / "qc_review/manual_qc_overrides.csv"
    assert load_overrides(history_path).manual_action.tolist() == ["mark_bad"]

    widget("selectbox", "Manual action").select("keep")
    widget("text_area", "Selected review IDs").input(identity).run()
    widget("button", "Apply to selected").click().run()
    check(pulsars[0])
    history = load_overrides(history_path)
    assert history.manual_action.tolist() == ["mark_bad", "keep"]
    assert history.review_id.tolist() == [identity, identity]
    assert history.pulsar.tolist() == [pulsars[0], pulsars[0]]
    widget("button", "Write reviewed QC CSV").click().run()
    check(pulsars[0])
    assert (tmp_path / f"qc_review/{pulsars[0]}_combined_qc.csv").is_file()
    assert raw == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in tmp_path.glob("*_qc.csv")
    }
