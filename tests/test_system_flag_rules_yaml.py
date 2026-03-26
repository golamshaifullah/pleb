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
