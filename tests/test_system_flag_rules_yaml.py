"""Tests for optional YAML-based system/group frequency rules."""

from __future__ import annotations

from pathlib import Path

import pytest

import pleb.system_flag_inference as sfi


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_yaml_rules_loader_and_application(tmp_path: Path) -> None:
    if sfi.yaml is None:
        pytest.skip("PyYAML unavailable in this test environment.")

    tim = tmp_path / "EFF.P200.1380.tim"
    _write(
        tim,
        "\n".join(
            [
                "FORMAT 1",
                "a 1366 55000 1.0 wsrt",
                "b 1424 55001 1.0 wsrt",
            ]
        )
        + "\n",
    )
    rules_path = tmp_path / "flag_sys_freq_rules.yaml"
    _write(
        rules_path,
        """
EFF.P200.1380:
  file: EFF.P200.1380.tim
  central_frequency: 1380
  pulsars:
    default: EFF.P200.1380
    JTEST: EFF.P200.1365 1370 EFF.P200.1425
""",
    )

    rules = sfi.load_flag_sys_freq_rules(rules_path)
    inferred = sfi.infer_sys_group_pta(
        tim,
        pulsar="JTEST",
        flag_sys_freq_rules=rules,
    )

    assert set(inferred["sys"].astype(str).tolist()) == {
        "EFF.P200.1365",
        "EFF.P200.1425",
    }
    assert set(inferred["group"].astype(str).tolist()) == {"EFF.P200.1380"}


def test_yaml_channel_modes_assign_single_and_dual_toas(tmp_path: Path) -> None:
    if sfi.yaml is None:
        pytest.skip("PyYAML unavailable in this test environment.")

    rules_path = tmp_path / "flag_sys_freq_rules.yaml"
    _write(
        rules_path,
        """
WSRT.P2.1380:
  file: WSRT.P2.1380.tim
  central_frequency: 1380
  pulsars:
    default: WSRT.P2.1380
  channel_modes:
    single:
      default: WSRT.P2.1380
    dual:
      default: WSRT.P2.1346 WSRT.P2.1410
""",
    )
    rules = sfi.load_flag_sys_freq_rules(rules_path)

    single_tim = tmp_path / "WSRT.P2.1380.tim"
    _write(
        single_tim,
        "\n".join(
            [
                "FORMAT 1",
                "arch_a 1366 55000.000000000000 1.0 wsrt",
                "arch_b 1424 55001.000000000000 1.0 wsrt",
            ]
        )
        + "\n",
    )
    single = sfi.infer_sys_group_pta(single_tim, flag_sys_freq_rules=rules)
    assert set(single["channel_file_mode"].astype(str)) == {"single"}
    assert set(single["sys"].astype(str)) == {"WSRT.P2.1380"}

    mixed_tim = tmp_path / "WSRT.P2.1380.tim"
    _write(
        mixed_tim,
        "\n".join(
            [
                "FORMAT 1",
                "single 1380 55000.000000000000 1.0 wsrt",
                "dual 1366 55001.000000000000 1.0 wsrt",
                "dual 1424 55001.000000000005 1.0 wsrt",
            ]
        )
        + "\n",
    )
    mixed = sfi.infer_sys_group_pta(mixed_tim, flag_sys_freq_rules=rules)
    assert set(mixed["channel_file_mode"].astype(str)) == {"mixed"}
    assert mixed["sys"].astype(str).tolist() == [
        "WSRT.P2.1380",
        "WSRT.P2.1346",
        "WSRT.P2.1410",
    ]
    assert set(mixed["group"].astype(str)) == {"WSRT.P2.1380"}
