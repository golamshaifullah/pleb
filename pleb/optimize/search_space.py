"""Load and sample optimization search spaces."""

from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any, Dict, List
import math

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from .models import ParameterSpec, SearchSpace


_VALID_KINDS = {"float", "int", "bool", "categorical", "fixed"}


def load_search_space(path: Path) -> SearchSpace:
    """Load and validate a parameter search space from TOML."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    raw_params = data.get("parameters", {})
    if not isinstance(raw_params, dict) or not raw_params:
        raise ValueError("search-space TOML requires a non-empty [parameters] table.")
    specs: List[ParameterSpec] = []
    for name, raw in raw_params.items():
        if not isinstance(raw, dict):
            raise ValueError(f"parameter {name!r} must be a TOML table.")
        kind = str(raw.get("type", raw.get("kind", ""))).strip().lower()
        if kind not in _VALID_KINDS:
            raise ValueError(f"parameter {name!r} has unsupported type {kind!r}.")
        spec = ParameterSpec(
            name=str(name),
            kind=kind,
            low=raw.get("low"),
            high=raw.get("high"),
            step=raw.get("step"),
            log=bool(raw.get("log", False)),
            choices=list(raw.get("choices", [])) if "choices" in raw else None,
            fixed=raw.get("value"),
            depends_on=(
                None
                if raw.get("depends_on") in (None, "")
                else str(raw.get("depends_on"))
            ),
            enabled_values=(
                list(raw.get("enabled_values", []))
                if "enabled_values" in raw
                else None
            ),
        )
        _validate_parameter(spec)
        specs.append(spec)
    return SearchSpace(parameters=specs)


def _validate_parameter(spec: ParameterSpec) -> None:
    if spec.kind in {"float", "int"}:
        if spec.low is None or spec.high is None:
            raise ValueError(f"{spec.name}: numeric parameters require low/high.")
        if float(spec.high) < float(spec.low):
            raise ValueError(f"{spec.name}: high must be >= low.")
        if spec.log and float(spec.low) <= 0:
            raise ValueError(f"{spec.name}: log-scaled parameters require low > 0.")
    if spec.kind == "categorical" and not spec.choices:
        raise ValueError(f"{spec.name}: categorical parameters require choices.")
    if spec.kind == "fixed" and spec.fixed is None:
        raise ValueError(f"{spec.name}: fixed parameters require value.")


def sample_parameters(space: SearchSpace, rng: Random) -> Dict[str, Any]:
    """Sample one parameter vector from a search space."""
    out: Dict[str, Any] = {}
    for spec in space.parameters:
        if not is_parameter_active(spec, out):
            continue
        out[spec.name] = _sample_one(spec, rng)
    return out


def is_parameter_active(spec: ParameterSpec, chosen: Dict[str, Any]) -> bool:
    """Return whether a parameter should be sampled under current conditions."""
    if not spec.depends_on:
        return True
    if spec.depends_on not in chosen:
        return False
    if spec.enabled_values is None:
        return bool(chosen.get(spec.depends_on))
    return chosen.get(spec.depends_on) in set(spec.enabled_values)


def active_parameter_count(space: SearchSpace, params: Dict[str, Any]) -> int:
    """Count active parameters for one sampled parameter vector."""
    count = 0
    chosen: Dict[str, Any] = {}
    for spec in space.parameters:
        if not is_parameter_active(spec, chosen):
            continue
        if spec.name in params:
            count += 1
            chosen[spec.name] = params[spec.name]
    return count


def _sample_one(spec: ParameterSpec, rng: Random) -> Any:
    if spec.kind == "fixed":
        return spec.fixed
    if spec.kind == "bool":
        return bool(rng.choice([False, True]))
    if spec.kind == "categorical":
        return rng.choice(list(spec.choices or []))
    if spec.kind == "float":
        return _sample_float(spec, rng)
    if spec.kind == "int":
        return _sample_int(spec, rng)
    raise ValueError(f"Unsupported parameter kind: {spec.kind!r}")


def _sample_float(spec: ParameterSpec, rng: Random) -> float:
    low = float(spec.low)
    high = float(spec.high)
    if spec.log:
        value = math.exp(rng.uniform(math.log(low), math.log(high)))
    else:
        value = rng.uniform(low, high)
    if spec.step:
        step = float(spec.step)
        value = low + round((value - low) / step) * step
        value = min(max(value, low), high)
    return float(value)


def _sample_int(spec: ParameterSpec, rng: Random) -> int:
    low = int(spec.low)
    high = int(spec.high)
    if spec.step:
        step = int(spec.step)
        choices = list(range(low, high + 1, step))
        return int(rng.choice(choices))
    return int(rng.randint(low, high))


def parameters_to_set_overrides(params: Dict[str, Any]) -> List[str]:
    """Convert sampled parameters to workflow-style KEY=VALUE overrides."""
    return [f"{key}={_to_toml_literal(value)}" for key, value in params.items()]


def _to_toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if value is None:
        return "null"
    if isinstance(value, list):
        inner = ", ".join(_to_toml_literal(v) for v in value)
        return f"[{inner}]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
