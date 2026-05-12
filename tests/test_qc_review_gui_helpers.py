from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


def _load_qc_review_script(monkeypatch):
    fake_streamlit = types.SimpleNamespace(session_state={})
    fake_plotly = types.ModuleType("plotly")
    fake_graph_objects = types.ModuleType("plotly.graph_objects")
    fake_plotly.graph_objects = fake_graph_objects

    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_graph_objects)

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "pleb_qc_review.py"
    spec = importlib.util.spec_from_file_location("pleb_qc_review_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fake_streamlit


def test_extract_selected_review_ids_uses_customdata(monkeypatch) -> None:
    module, _ = _load_qc_review_script(monkeypatch)

    event = {
        "selection": {
            "points": [
                {"customdata": ["rid-1", "A"]},
                {"customdata": "rid-2"},
                {"customdata": ["rid-1", "B"]},
            ]
        }
    }

    assert module._extract_selected_review_ids(event) == ["rid-1", "rid-2"]


def test_sync_plot_selection_copies_plot_widget_state(monkeypatch) -> None:
    module, fake_streamlit = _load_qc_review_script(monkeypatch)
    fake_streamlit.session_state[module.PLOT_KEY] = {
        "selection": {"points": [{"customdata": ["rid-42"]}]}
    }

    module._sync_plot_selection()

    assert fake_streamlit.session_state["selected_review_ids"] == ["rid-42"]


def test_numeric_axis_options_prioritises_mjd_uncertainty_and_phase(monkeypatch) -> None:
    module, _ = _load_qc_review_script(monkeypatch)
    df = pd.DataFrame(
        {
            "mjd": [58000.0, 58001.0],
            "uncertainty": [0.5, 0.6],
            "orbital_phase": [0.2, 0.4],
            "freq": [1400.0, 800.0],
            "bad_point": [False, True],
            "reviewed_bad_point": [False, True],
            "_plot_residual": [1.0, 2.0],
        }
    )

    options = module._numeric_axis_options(df)

    assert options[:4] == ["mjd", "uncertainty", "freq", "orbital_phase"]
    assert "bad_point" not in options
    assert "reviewed_bad_point" not in options
    assert "_plot_residual" not in options


def test_default_reviewed_path_uses_selected_qc_basename(monkeypatch) -> None:
    module, _ = _load_qc_review_script(monkeypatch)

    out = module._default_reviewed_path(
        "/tmp/run",
        "/tmp/run/qc/step4/J1909-3744.combined_qc.csv",
    )

    assert out == Path("/tmp/run/qc_review/J1909-3744.combined_qc.csv")


def test_sync_reviewed_output_default_tracks_auto_path_but_preserves_manual(monkeypatch) -> None:
    module, fake_streamlit = _load_qc_review_script(monkeypatch)

    first = module._sync_reviewed_output_default(
        "/tmp/run",
        "/tmp/run/qc/step4/J1909-3744.legacy_qc.csv",
    )
    assert first == Path("/tmp/run/qc_review/J1909-3744.legacy_qc.csv")
    assert fake_streamlit.session_state["reviewed_out_path"] == str(first)

    second = module._sync_reviewed_output_default(
        "/tmp/run",
        "/tmp/run/qc/step4/J1909-3744.combined_qc.csv",
    )
    assert second == Path("/tmp/run/qc_review/J1909-3744.combined_qc.csv")
    assert fake_streamlit.session_state["reviewed_out_path"] == str(second)

    fake_streamlit.session_state["reviewed_out_path"] = "/tmp/custom/review.csv"
    third = module._sync_reviewed_output_default(
        "/tmp/run",
        "/tmp/run/qc/step4/J1909-3744.new_qc.csv",
    )
    assert third == Path("/tmp/custom/review.csv")
    assert fake_streamlit.session_state["reviewed_out_path"] == str(third)
