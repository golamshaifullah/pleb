from __future__ import annotations

from pathlib import Path

from pleb.config import PipelineConfig
from pleb.pipeline import _build_fixdataset_config


def test_build_fixdataset_config_preserves_cleanup_defaults(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        home_dir=tmp_path,
        singularity_image=tmp_path / "tempo2.sif",
        dataset_name=tmp_path / "dataset",
        pulsars=["J0000+0000"],
    )

    fcfg = _build_fixdataset_config(cfg, apply=True)

    assert fcfg.dedupe_toas_within_tim is True
    assert fcfg.remove_overlaps_exact is True
