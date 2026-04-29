from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pleb.dataset_fix import FixDatasetConfig, infer_and_apply_system_flags
from pleb.system_flag_inference import BackendMissingError, infer_backend


def test_infer_backend_rejects_mixed_be_values() -> None:
    df = pd.DataFrame(
        {
            "be": ["BON", "NUPPI"],
            "line": [
                "toa_a 1400 55000 1.0 ao -be BON",
                "toa_b 1400 55001 1.0 ao -be NUPPI",
            ],
        }
    )

    with pytest.raises(BackendMissingError, match="Multiple backend values found"):
        infer_backend(Path("NRT.mixed.tim"), df)


def test_infer_and_apply_system_flags_overwrites_existing_nonlegacy_flags(
    tmp_path: Path,
) -> None:
    timfile = tmp_path / "NRT.BON.1400.tim"
    timfile.write_text(
        "\n".join(
            [
                "FORMAT 1",
                "f 1400 55000 1.0 ncyobs -sys WRONG.SYS.1 -group WRONG.GROUP.1 -pta BAD",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = FixDatasetConfig(
        apply=True,
        backup=False,
        dry_run=False,
        system_flag_overwrite_existing=True,
        system_flag_table_path=str(tmp_path / "system_flag_table.json"),
    )

    rep = infer_and_apply_system_flags(timfile, cfg)

    assert rep["changed"] is True
    assert rep["overwritten"] == 3
    text = timfile.read_text(encoding="utf-8")
    assert "-sys NRT.BON.1400" in text
    assert "-group NRT.BON.1400" in text
    assert "-pta EPTA" in text
