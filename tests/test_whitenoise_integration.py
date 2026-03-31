"""Tests for optional whitenoise integration helpers."""

from __future__ import annotations

from pathlib import Path

from pleb.whitenoise_integration import (
    WhiteNoiseStageConfig,
    estimate_white_noise_for_pulsar,
    resolve_timfile_for_pulsar,
)


def test_resolve_timfile_for_pulsar_prefers_template(tmp_path: Path) -> None:
    psr_dir = tmp_path / "J1713+0747"
    psr_dir.mkdir()
    explicit = psr_dir / "J1713+0747_all.new.tim"
    explicit.write_text("", encoding="utf-8")
    (psr_dir / "J1713+0747_all.tim").write_text("", encoding="utf-8")

    got = resolve_timfile_for_pulsar(psr_dir, "J1713+0747", "{pulsar}_all.new.tim")
    assert got == explicit


def test_resolve_timfile_for_pulsar_fallbacks(tmp_path: Path) -> None:
    psr_dir = tmp_path / "J1022+1001"
    psr_dir.mkdir()
    default_all = psr_dir / "J1022+1001_all.tim"
    default_all.write_text("", encoding="utf-8")
    assert resolve_timfile_for_pulsar(psr_dir, "J1022+1001", None) == default_all


def test_estimate_white_noise_for_pulsar_maps_result(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeResult:
        n_toas = 10
        n_epochs = 6
        has_multi_toa_epochs = True
        efac = 1.2
        efac_err = 0.1
        equad = 1e-6
        equad_err = 2e-7
        ecorr = 5e-7
        ecorr_err = 1e-7
        extra_variance_floor = 0.0
        extra_variance_floor_err = 0.0
        single_toa_mode = "combined"
        warning = ""
        success = True
        message = "ok"
        fun = 1.0

    def _fake_estimator(**kwargs):  # noqa: ARG001
        return _FakeResult()

    monkeypatch.setattr(
        "pleb.whitenoise_integration._resolve_estimator",
        lambda source_path=None: _fake_estimator,
    )

    par = tmp_path / "J1713+0747.par"
    tim = tmp_path / "J1713+0747_all.tim"
    par.write_text("", encoding="utf-8")
    tim.write_text("", encoding="utf-8")
    cfg = WhiteNoiseStageConfig()

    row = estimate_white_noise_for_pulsar(par, tim, cfg)
    assert row["n_toas"] == 10
    assert row["n_epochs"] == 6
    assert row["success"] is True
    assert row["efac"] == 1.2
    assert row["ecorr"] == 5e-7
