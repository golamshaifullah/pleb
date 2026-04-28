"""Objective definitions for optimization trials."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from .models import ObjectiveConfig


def load_objective_config(path: Path) -> ObjectiveConfig:
    """Load objective weights and optional hard constraints from TOML."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("weights", {})
    if not isinstance(raw, dict) or not raw:
        raise ValueError("objective TOML requires a non-empty [weights] table.")
    weights = {str(key): float(val) for key, val in raw.items()}
    constraints_raw = data.get("constraints", {}) or {}
    if not isinstance(constraints_raw, dict):
        raise ValueError("objective [constraints] must be a table when provided.")
    constraints = {str(key): float(val) for key, val in constraints_raw.items()}
    return ObjectiveConfig(
        weights=weights,
        maximize=bool(data.get("maximize", True)),
        score_offset=float(data.get("score_offset", 0.0)),
        constraints=constraints,
        constraint_penalty=float(data.get("constraint_penalty", 1.0e12)),
    )


def violated_constraints(
    metrics: Dict[str, float], objective: ObjectiveConfig
) -> Tuple[str, ...]:
    """Return hard objective constraints violated by the metrics.

    Supported forms:
    - ``max_<metric>``: metric must be <= threshold.
    - ``min_<metric>``: metric must be >= threshold.
    - ``<metric>``: metric must be <= threshold.
    """
    violations: list[str] = []
    for key, threshold in (objective.constraints or {}).items():
        if key.startswith("max_"):
            metric_name = key[4:]
            value = metrics.get(metric_name)
            if value is not None and float(value) > float(threshold):
                violations.append(key)
        elif key.startswith("min_"):
            metric_name = key[4:]
            value = metrics.get(metric_name)
            if value is not None and float(value) < float(threshold):
                violations.append(key)
        else:
            value = metrics.get(key)
            if value is not None and float(value) > float(threshold):
                violations.append(key)
    return tuple(violations)


def compute_score(metrics: Dict[str, float], objective: ObjectiveConfig) -> float:
    """Compute a scalar weighted score with hard-constraint penalties."""
    raw = float(objective.score_offset)
    for key, weight in objective.weights.items():
        raw += float(weight) * float(metrics.get(key, 0.0))
    score = raw if objective.maximize else -raw
    violations = violated_constraints(metrics, objective)
    if violations:
        score -= abs(float(objective.constraint_penalty)) * float(len(violations))
    return score
