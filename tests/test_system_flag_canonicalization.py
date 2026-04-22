from __future__ import annotations

import json
from pathlib import Path

from pleb.dataset_fix import FixDatasetConfig, canonicalize_system_flags_dataset


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_canonicalize_system_flags_dataset_snaps_numeric_sys_across_pulsars(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    p1_tim = dataset_root / "J0001+0001" / "tims" / "EFF.P200.1380.tim"
    p2_tim = dataset_root / "J0002+0002" / "tims" / "EFF.P200.1380.tim"

    _write(
        p1_tim,
        "\n".join(
            [
                "FORMAT 1",
                "a 1365 55000 1.0 wsrt -sys EFF.P200.1365 -group EFF.P200.1380 -pta EPTA",
            ]
        )
        + "\n",
    )
    _write(
        p2_tim,
        "\n".join(
            [
                "FORMAT 1",
                "b 1366 55001 1.0 wsrt -sys EFF.P200.1366 -group EFF.P200.1380 -pta EPTA",
            ]
        )
        + "\n",
    )

    mapping_path = dataset_root / "system_flag_table.json"
    cfg = FixDatasetConfig(
        apply=True,
        backup=False,
        dry_run=False,
        system_flag_table_path=str(mapping_path),
    )

    reports = canonicalize_system_flags_dataset(
        dataset_root,
        ["J0001+0001", "J0002+0002"],
        cfg,
    )

    assert reports["J0001+0001"]["n_sys_rows_changed"] == 1
    assert reports["J0002+0002"]["n_sys_rows_changed"] == 0

    p1_text = p1_tim.read_text(encoding="utf-8")
    p2_text = p2_tim.read_text(encoding="utf-8")
    assert "-sys EFF.P200.1366" in p1_text
    assert "-sys EFF.P200.1366" in p2_text
    assert "-group EFF.P200.1380" in p1_text
    assert "-group EFF.P200.1380" in p2_text

    table = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert table == {"EFF.P200.1380.tim": ["EFF.P200.1366"]}


def test_canonicalize_system_flags_dataset_preserves_non_numeric_sys_labels(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    timfile = dataset_root / "J2317+1439" / "tims" / "WSRT.P1.1380.C.tim"
    _write(
        timfile,
        "\n".join(
            [
                "FORMAT 1",
                "a 1380 55000 1.0 wsrt -sys WSRT.P1.1380.C -group WSRT.P1.1380.C -pta EPTA",
            ]
        )
        + "\n",
    )

    cfg = FixDatasetConfig(apply=True, backup=False, dry_run=False)
    reports = canonicalize_system_flags_dataset(dataset_root, ["J2317+1439"], cfg)

    assert reports["J2317+1439"]["n_sys_rows_changed"] == 0
    text = timfile.read_text(encoding="utf-8")
    assert "-sys WSRT.P1.1380.C" in text
    assert "-group WSRT.P1.1380.C" in text
