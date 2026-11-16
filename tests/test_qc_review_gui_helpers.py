from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


def _load_qc_review_script(monkeypatch):
    fake_streamlit = types.SimpleNamespace(session_state={})
    fake_plotly = types.ModuleType("plotly")
    fake_graph_objects = types.ModuleType("plotly.graph_objects")
    fake_plotly.graph_objects = fake_graph_objects

    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_graph_objects)

    script_path = Path(__file__).resolve().parents[1] / "pleb" / "review_app.py"
    spec = importlib.util.spec_from_file_location("pleb_qc_review_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fake_streamlit


@pytest.mark.parametrize(
    "residual_col,error_col,errors,expected",
    [
        ("tempo2_post_us", "tempo2_err_us", [3, 4], [3, 4]),
        ("tempo2_post_us", "tempo2_err", [3e-6, 4e-6], [3, 4]),
        ("tempo2_post", "tempo2_err_us", [3, 4], [3e-6, 4e-6]),
        ("tempo2_post", "tempo2_err", [3e-6, 4e-6], [3e-6, 4e-6]),
        ("residual", "uncertainty", [3, 4], [3, 4]),
    ],
)
def test_plot_error_bars_match_residual_units(
    monkeypatch, residual_col, error_col, errors, expected
) -> None:
    module, _ = _load_qc_review_script(monkeypatch)
    frame = pd.DataFrame(
        {"mjd": [59000, 59001], residual_col: [1, 2], error_col: errors}
    )
    result = module._attach_plot_columns(frame, residual_col, "mjd")
    assert result["_plot_error"].tolist() == pytest.approx(expected)
    assert result["_plot_residual"].tolist() == [1, 2]
    assert "_plot_error" not in frame


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


@pytest.mark.parametrize(
    "filename,pulsar,variant",
    [
        ("J1234+5678_combined_qc.csv", "J1234+5678", "combined"),
        ("J1909-3744_legacy_qc.csv", "J1909-3744", "legacy"),
        ("J1909-3744_new_qc.csv", "J1909-3744", "new"),
        ("B1855+09_combined_qc.csv", "B1855+09", "combined"),
        ("J1909-3744__trial_qc.csv", "J1909-3744", "trial"),
        ("J1909-3744_qc.csv", "J1909-3744", "base"),
    ],
)
def test_native_export_variants_group_under_the_coordinate_name(
    monkeypatch, filename, pulsar, variant
) -> None:
    module, _ = _load_qc_review_script(monkeypatch)
    path = Path(filename)
    actual = module._guess_pulsar_from_qc_path(path)
    assert actual == pulsar
    assert module._guess_variant_from_qc_path(path, actual) == variant


def test_sync_plot_selection_copies_plot_widget_state(monkeypatch) -> None:
    module, fake_streamlit = _load_qc_review_script(monkeypatch)
    fake_streamlit.session_state[module.PLOT_KEY] = {
        "selection": {"points": [{"customdata": ["rid-42"]}]}
    }

    module._sync_plot_selection()

    assert fake_streamlit.session_state["selected_review_ids"] == ["rid-42"]


def test_plot_selection_defaults_to_box_and_keeps_lasso_available(monkeypatch) -> None:
    module, _ = _load_qc_review_script(monkeypatch)

    assert module.PLOT_SELECTION_MODES == ("box", "lasso")
    assert module.PLOT_DEFAULT_DRAGMODE == "select"
    assert "select2d" in module.PLOT_MODEBAR_BUTTONS_TO_ADD
    assert "lasso2d" in module.PLOT_MODEBAR_BUTTONS_TO_ADD


def test_numeric_axis_options_prioritises_mjd_uncertainty_and_phase(
    monkeypatch,
) -> None:
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


def test_feature_columns_and_masks_include_new_pqc_features(monkeypatch) -> None:
    module, _ = _load_qc_review_script(monkeypatch)
    df = pd.DataFrame(
        {
            "bad_ou": [True, False, False],
            "bad_hard": [False, True, False],
            "step_id": [-1, -1, 2],
            "dm_step_id": ["", "dm-1", "-1"],
            "orbital_phase_bad": [False, False, True],
            "reviewed_bad_point": [False, False, True],
        }
    )

    cols = module._feature_columns(df)

    assert "bad_ou" in cols
    assert "bad_hard" in cols
    assert "step_id" in cols
    assert "dm_step_id" in cols
    assert "orbital_phase_bad" in cols
    assert "reviewed_bad_point" not in cols
    assert list(module._feature_mask(df, "step_id")) == [False, False, True]
    assert list(module._feature_mask(df, "dm_step_id")) == [False, True, False]


def test_default_reviewed_path_uses_selected_qc_basename(monkeypatch) -> None:
    module, _ = _load_qc_review_script(monkeypatch)

    out = module._default_reviewed_path(
        "/tmp/run",
        "/tmp/run/qc/step4/J1909-3744.combined_qc.csv",
    )

    assert out == Path("/tmp/run/qc_review/J1909-3744.combined_qc.csv")


def test_sync_reviewed_output_default_tracks_auto_path_but_preserves_manual(
    monkeypatch,
) -> None:
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
