"""Helpers for optional white-noise parameter estimation.

This module provides a small adapter layer between PLEB and the external
``whitenoise`` estimator so pipeline stages can estimate EFAC/EQUAD/ECORR
without hard-coding a specific installation layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import importlib.util
import sys

EstimatorFn = Callable[..., Any]


@dataclass
class WhiteNoiseStageConfig:
    """Configuration for the optional white-noise stage."""

    source_path: Optional[Path] = None
    epoch_tolerance_seconds: float = 1.0
    single_toa_mode: str = "combined"
    fit_timing_model_first: bool = True
    timfile_name: Optional[str] = None


def _resolve_estimator(source_path: Optional[Path] = None) -> EstimatorFn:
    """Resolve ``estimate_white_noise`` from installed package or source tree."""
    try:
        from whitenoise_estimator import estimate_white_noise  # type: ignore

        return estimate_white_noise
    except Exception:
        pass

    if source_path is not None:
        src = Path(source_path).expanduser().resolve()
        if src.is_dir():
            candidate = src / "whitenoise_estimator.py"
            if candidate.exists():
                spec = importlib.util.spec_from_file_location(
                    "whitenoise_estimator", str(candidate)
                )
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["whitenoise_estimator"] = module
                    spec.loader.exec_module(module)
                    fn = getattr(module, "estimate_white_noise", None)
                    if fn is not None:
                        return fn
    raise ImportError(
        "Could not import whitenoise_estimator. Install it in the environment, "
        "or set whitenoise_source_path to a folder containing whitenoise_estimator.py."
    )


def resolve_timfile_for_pulsar(
    pulsar_dir: Path, pulsar: str, timfile_name: Optional[str]
) -> Optional[Path]:
    """Resolve the timfile used for white-noise estimation.

    Preference order:
    1) ``timfile_name`` if provided
    2) ``<pulsar>_all.tim``
    3) ``<pulsar>.tim``
    """
    if timfile_name:
        name = str(timfile_name).format(pulsar=pulsar)
        p = pulsar_dir / name
        return p if p.exists() else None

    for name in (f"{pulsar}_all.tim", f"{pulsar}.tim"):
        p = pulsar_dir / name
        if p.exists():
            return p
    return None


def estimate_white_noise_for_pulsar(
    parfile: Path,
    timfile: Path,
    cfg: WhiteNoiseStageConfig,
) -> Dict[str, object]:
    """Estimate EFAC/EQUAD/ECORR for a pulsar par/tim pair."""
    estimate = _resolve_estimator(cfg.source_path)
    res = estimate(
        parfile=str(parfile),
        timfile=str(timfile),
        epoch_tolerance_seconds=float(cfg.epoch_tolerance_seconds),
        single_toa_mode=str(cfg.single_toa_mode),
        fit_timing_model_first=bool(cfg.fit_timing_model_first),
    )

    # Support dataclass-like objects while staying robust to API evolution.
    def get(k: str, default: Any = None) -> Any:
        return getattr(res, k, default)

    return {
        "n_toas": get("n_toas"),
        "n_epochs": get("n_epochs"),
        "has_multi_toa_epochs": get("has_multi_toa_epochs"),
        "efac": get("efac"),
        "efac_err": get("efac_err"),
        "equad": get("equad"),
        "equad_err": get("equad_err"),
        "ecorr": get("ecorr"),
        "ecorr_err": get("ecorr_err"),
        "extra_variance_floor": get("extra_variance_floor"),
        "extra_variance_floor_err": get("extra_variance_floor_err"),
        "single_toa_mode": get("single_toa_mode"),
        "warning": get("warning"),
        "success": bool(get("success", True)),
        "message": get("message"),
        "fun": get("fun"),
    }
