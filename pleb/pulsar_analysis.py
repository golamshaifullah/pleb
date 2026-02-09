"""Binary/orbital analysis helpers for pulsar `.par` files.

This module provides lightweight parsing and derived-parameter calculations
intended for summary reports, not full timing-model validation.

See Also:
    pleb.kepler_orbits: Orbital mechanics helpers used in derived quantities.
    pleb.config.PipelineConfig: Enables binary analysis in the pipeline.
"""

from __future__ import annotations

from .compat import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import math
import re

import pandas as pd

from .logging_utils import get_logger
from .kepler_orbits import btx_parameters

logger = get_logger("pleb.pulsar_analysis")


@dataclass(slots=True)
class BinaryAnalysisConfig:
    """Configuration for binary/orbital diagnostics derived from .par files.

    Attributes:
        only_models: If set, only report pulsars whose ``BINARY`` parameter
            matches one of these model names.

    Examples:
        Limit output to BTX binaries::

            cfg = BinaryAnalysisConfig(only_models=["BTX"])
    """

    # If set, only write rows for pulsars with these BINARY models
    only_models: Optional[List[str]] = None


def read_parfile(parfile: Path) -> Dict[str, str]:
    """Very lightweight tempo2 .par reader.

    Args:
        parfile: Path to a `.par` file.

    Returns:
        Dict of ``KEY -> VALUE`` (as strings). Comments and blank lines are
        ignored; if a key appears multiple times, the last one wins.
    """
    params: Dict[str, str] = {}
    if not parfile.exists():
        return params

    for raw in parfile.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(("C", "#")):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            continue
        key = parts[0].strip()
        val = parts[1].strip()
        params[key] = val
    return params


def _to_float(x: Optional[str]) -> Optional[float]:
    """Convert a string to float, returning None on failure."""
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def analyse_binary_from_par(parfile: Path) -> Dict[str, object]:
    """Extract binary parameters and compute a few derived quantities.

    This is intentionally conservative: it will only compute ELL1->BTX conversion
    if EPS1/EPS2/TASC are present.

    Args:
        parfile: Path to a `.par` file.

    Returns:
        Mapping of extracted/derived parameters, including ``BINARY`` when present.

    Notes:
        Derived ELL1 quantities are reported as ``ELL1_*`` keys.
    """
    p = read_parfile(parfile)
    out: Dict[str, object] = {"parfile": str(parfile)}

    model = p.get("BINARY")
    out["BINARY"] = model

    # Common orbital params
    for k in [
        "PB",
        "A1",
        "T0",
        "OM",
        "ECC",
        "TASC",
        "EPS1",
        "EPS2",
        "PBDOT",
        "XDOT",
        "OMDOT",
        "ECCDOT",
    ]:
        if k in p:
            out[k] = _to_float(p[k]) if k not in ("BINARY",) else p[k]

    # ELL1 -> (e, om, t0)
    a1 = _to_float(p.get("A1"))
    pb = _to_float(p.get("PB"))
    eps1 = _to_float(p.get("EPS1"))
    eps2 = _to_float(p.get("EPS2"))
    tasc = _to_float(p.get("TASC"))

    if (
        a1 is not None
        and pb is not None
        and eps1 is not None
        and eps2 is not None
        and tasc is not None
    ):
        try:
            asini, pb_out, e, om, t0 = btx_parameters(a1, pb, eps1, eps2, tasc)
            out["ELL1_asini"] = asini
            out["ELL1_pb"] = pb_out
            out["ELL1_e"] = e
            out["ELL1_om_rad"] = om
            out["ELL1_om_deg"] = float(om * 180.0 / math.pi)
            out["ELL1_t0"] = t0
        except Exception as e:
            logger.warning("ELL1->BTX conversion failed for %s: %s", parfile, e)

    return out


def write_binary_analysis(
    home_dir: Path,
    out_dir: Path,
    pulsars: List[str],
    branches: List[str],
    config: Optional[BinaryAnalysisConfig] = None,
) -> Path:
    """Write a per-branch, per-pulsar binary analysis TSV.

    Looks for <home_dir>/<pulsar>/<pulsar>.par on each branch.

    Args:
        home_dir: Root data repository.
        out_dir: Output directory for the TSV.
        pulsars: Pulsar names to include.
        branches: Branch names (used for labeling).
        config: Optional binary analysis configuration.

    Returns:
        Path to the written TSV file.

    Examples:
        Write a binary analysis table for two branches::

            out_path = write_binary_analysis(
                home_dir=Path("/data/epta/EPTA"),
                out_dir=Path("results/binary"),
                pulsars=["J1234+5678"],
                branches=["main", "EPTA"],
            )
    """
    cfg = config or BinaryAnalysisConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for branch in branches:
        for psr in pulsars:
            parfile = home_dir / psr / f"{psr}.par"
            d = analyse_binary_from_par(parfile)
            d["pulsar"] = psr
            d["branch"] = branch

            if cfg.only_models:
                if d.get("BINARY") not in set(cfg.only_models):
                    continue

            rows.append(d)

    df = pd.DataFrame(rows)
    out_path = out_dir / "binary_analysis.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    return out_path
