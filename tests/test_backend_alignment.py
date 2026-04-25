from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pleb.backend_alignment import (
    estimate_pairwise_backend_offsets,
    select_backend_alignment_reference,
    write_backend_alignment_outputs,
)


def _build_alignment_records(
    offsets: dict[str, float],
    toas_by_backend: dict[str, list[float]],
    *,
    err_by_backend: dict[str, float] | None = None,
    template_by_backend: dict[str, str | None] | None = None,
    seed: int = 123,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    err_by_backend = err_by_backend or {}
    template_by_backend = template_by_backend or {}
    for backend, offset in offsets.items():
        err = float(err_by_backend.get(backend, 0.25))
        template = template_by_backend.get(backend)
        for toa in toas_by_backend[backend]:
            trend = 0.15 * np.sin(float(toa) / 11.0)
            noise = rng.normal(scale=err * 0.35)
            rows.append(
                {
                    "backend": backend,
                    "toa": float(toa),
                    "residual": float(trend + offset + noise),
                    "toa_error": err,
                    "template": template,
                }
            )
    return pd.DataFrame(rows)


def test_estimate_pairwise_backend_offsets_is_robust_to_outliers() -> None:
    offsets = {"A": 0.0, "B": 1.25}
    toas = {"A": list(range(12)), "B": list(range(12))}
    df = _build_alignment_records(
        offsets,
        toas,
        err_by_backend={"A": 0.2, "B": 0.2},
        template_by_backend={"A": "A.std", "B": "B.std"},
    )
    mask = (df["backend"] == "B") & (df["toa"] == 6.0)
    df.loc[mask, "residual"] = df.loc[mask, "residual"] + 7.5

    backends, edges, templates = estimate_pairwise_backend_offsets(
        df,
        max_toa_separation=0.01,
        min_overlap_pairs=5,
    )

    assert backends == ["A", "B"]
    assert len(edges) == 1
    edge = edges[0]
    assert edge.left_backend == "A"
    assert edge.right_backend == "B"
    assert edge.delta == pytest.approx(1.25, abs=0.35)
    assert edge.inflation_factor > 1.0
    assert edge.n_pairs == 12
    assert templates["A"].score == pytest.approx(1.0)
    assert templates["B"].score == pytest.approx(1.0)


def test_select_backend_alignment_reference_prefers_well_connected_precise_backend() -> None:
    offsets = {"A": 0.0, "B": 1.0, "C": -0.7, "D": 1.8}
    toas = {
        "A": list(range(0, 40)),
        "B": list(range(0, 30)),
        "C": list(range(10, 40)),
        "D": list(range(15, 26)),
    }
    df = _build_alignment_records(
        offsets,
        toas,
        err_by_backend={"A": 0.12, "B": 0.18, "C": 0.22, "D": 0.35},
        template_by_backend={"A": "A.std", "B": "B.std", "C": None, "D": None},
    )

    result = select_backend_alignment_reference(
        df,
        max_toa_separation=0.01,
        min_overlap_pairs=5,
    )

    assert result.chosen_reference == "A"
    ranked = [row.backend for row in result.ranked_references]
    assert ranked[0] == "A"

    offset_table = result.offset_table().set_index("backend")
    for backend, true_offset in offsets.items():
        truth = true_offset - offsets["A"]
        assert offset_table.loc[backend, "offset_relative_to_reference"] == pytest.approx(
            truth, abs=0.35
        )
    assert offset_table.loc["A", "uncertainty_relative_to_reference"] == pytest.approx(0.0)
    assert result.reference_table().iloc[0]["weighted_connectivity"] >= result.reference_table().iloc[-1][
        "weighted_connectivity"
    ]


def test_select_backend_alignment_reference_rejects_disconnected_graph() -> None:
    offsets = {"A": 0.0, "B": 0.8, "C": -0.4}
    toas = {"A": [0, 1, 2, 3], "B": [0, 1, 2, 3], "C": [100, 101, 102, 103]}
    df = _build_alignment_records(offsets, toas, seed=7)

    with pytest.raises(ValueError, match="No pairwise backend offsets|disconnected"):
        select_backend_alignment_reference(
            df,
            max_toa_separation=0.01,
            min_overlap_pairs=3,
        )


def test_write_backend_alignment_outputs_smoke(tmp_path: Path) -> None:
    offsets = {"A": 0.0, "B": 0.9, "C": -0.5}
    toas = {"A": list(range(8)), "B": list(range(8)), "C": list(range(3, 11))}
    df = _build_alignment_records(
        offsets,
        toas,
        err_by_backend={"A": 0.15, "B": 0.2, "C": 0.25},
        template_by_backend={"A": "A.std", "B": "B.std", "C": "C.std"},
        seed=42,
    )
    result = select_backend_alignment_reference(
        df,
        max_toa_separation=0.01,
        min_overlap_pairs=3,
    )

    files = write_backend_alignment_outputs(result, tmp_path)

    expected = {
        "edges",
        "references",
        "offsets",
        "covariance",
        "templates",
        "summary",
        "ranking_plot",
        "graph_plot",
        "covariance_plot",
    }
    assert set(files) == expected
    for path in files.values():
        assert path.exists()
        assert path.stat().st_size > 0
