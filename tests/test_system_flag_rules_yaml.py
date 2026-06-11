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


def test_yaml_rules_cover_plain_wsrt_p1_1380_without_duplicate_centre() -> None:
    if sfi.yaml is None:
        pytest.skip("PyYAML unavailable in this test environment.")

    rules = sfi.load_flag_sys_freq_rules(
        Path("configs/catalogs/system_flags/flag_sys_freq_rules.yaml")
    )
    rule = rules["WSRT.P1.1380.tim"]

    assert rule["central_frequency"] == 1380
    assert rule["pulsars"]["default"] == ["WSRT.P1.1380"]


def test_yaml_nuppi_rules_use_four_subband_system_labels() -> None:
    if sfi.yaml is None:
        pytest.skip("PyYAML unavailable in this test environment.")

    rules = sfi.load_flag_sys_freq_rules(
        Path("configs/catalogs/system_flags/flag_sys_freq_rules.yaml")
    )
    expected = {
        "NRT.NUPPI.1484.tim": [
            "NRT.NUPPI.1292",
            "NRT.NUPPI.1420",
            "NRT.NUPPI.1548",
            "NRT.NUPPI.1676",
        ],
        "NRT.NUPPI.1854.tim": [
            "NRT.NUPPI.1662",
            "NRT.NUPPI.1790",
            "NRT.NUPPI.1918",
            "NRT.NUPPI.2046",
        ],
        "NRT.NUPPI.2154.tim": [
            "NRT.NUPPI.1962",
            "NRT.NUPPI.2090",
            "NRT.NUPPI.2218",
            "NRT.NUPPI.2346",
        ],
        "NRT.NUPPI.2354.tim": [
            "NRT.NUPPI.2162",
            "NRT.NUPPI.2290",
            "NRT.NUPPI.2418",
            "NRT.NUPPI.2546",
        ],
        "NRT.NUPPI.2539.tim": [
            "NRT.NUPPI.2347",
            "NRT.NUPPI.2475",
            "NRT.NUPPI.2603",
            "NRT.NUPPI.2731",
        ],
    }

    for timfile, labels in expected.items():
        assert rules[timfile]["pulsars"]["default"] == labels
        assert len(rules[timfile]["pulsars"]["default"]) == 4


def test_yaml_lofar_lump_150_has_single_merged_entry() -> None:
    if sfi.yaml is None:
        pytest.skip("PyYAML unavailable in this test environment.")

    rules = sfi.load_flag_sys_freq_rules(
        Path("configs/catalogs/system_flags/flag_sys_freq_rules.yaml")
    )

    assert "LOFAR.LuMP.150.tim" in rules
    assert rules["LOFAR.LuMP.150.tim"]["pulsars"]["default"] == [
        "LOFAR.150",
        "DE601.150",
        "DE602.150",
        "DE603.150",
        "DE604.150",
        "DE605.150",
        "DE609.150",
    ]


def test_yaml_has_no_duplicate_file_targets() -> None:
    if sfi.yaml is None:
        pytest.skip("PyYAML unavailable in this test environment.")

    text = Path("configs/catalogs/system_flags/flag_sys_freq_rules.yaml").read_text(
        encoding="utf-8"
    )
    files = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("file:"):
            files.append(stripped.split(":", 1)[1].strip())

    duplicates = sorted({name for name in files if files.count(name) > 1})
    assert duplicates == []


def test_yaml_eff_s110_does_not_accept_ps110_alias() -> None:
    if sfi.yaml is None:
        pytest.skip("PyYAML unavailable in this test environment.")

    rules = sfi.load_flag_sys_freq_rules(
        Path("configs/catalogs/system_flags/flag_sys_freq_rules.yaml")
    )

    assert rules["EFF.S110.2487.tim"]["pulsars"]["default"] == [
        "EFF.S110.2487",
    ]
    all_labels = [
        label
        for spec in rules.values()
        for labels in (spec.get("pulsars", {}) or {}).values()
        for label in labels
    ]
    assert "EFF.PS110.2487" not in all_labels
