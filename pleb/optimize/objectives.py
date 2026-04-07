"""Objective definitions for optimization trials."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from .models import ObjectiveConfig


def load_objective_config(path: Path) -> ObjectiveConfig:
    """Load objective weights from TOML."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("weights", {})
    if not isinstance(raw, dict) or not raw:
        raise ValueError("objective TOML requires a non-empty [weights] table.")
    weights = {str(key): float(val) for key, val in raw.items()}
    return ObjectiveConfig(
        weights=weights,
        maximize=bool(data.get("maximize", True)),
        score_offset=float(data.get("score_offset", 0.0)),
    )


def compute_score(metrics: Dict[str, float], objective: ObjectiveConfig) -> float:
    """Compute a scalar weighted score."""
    score = float(objective.score_offset)
    for key, weight in objective.weights.items():
        score += float(weight) * float(metrics.get(key, 0.0))
    return score if objective.maximize else -score
