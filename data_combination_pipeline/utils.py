from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


def which_or_raise(cmd: str, hint: str = "") -> str:
    path = shutil.which(cmd)
    if not path:
        raise RuntimeError(f"Required executable not found on PATH: {cmd}. {hint}".strip())
    return path


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def discover_pulsars(home_dir: Path) -> List[str]:
    pulsars: List[str] = []
    for entry in home_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("J") and not entry.name.startswith("."):
            pulsars.append(entry.name)
    return sorted(pulsars)


def make_output_tree(results_dir: Path, branches: List[str], outdir_name: Optional[str]) -> Dict[str, Path]:
    if outdir_name is None:
        outdir_name = "EPTA_combination_report_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    base = results_dir / outdir_name
    tag = "-".join(branches)

    paths = {
        "base": base,
        "tag": base / tag,
        "plk": base / tag / "plk",
        "covmat": base / tag / "covmat",
        "general2": base / tag / "general2",
        "change_report": base / tag / "change_report",
        "outliers": base / tag / "OutlierSummary",
        "png": base / tag / "png",
        "logs": base / tag / "logs",
        "work": base / tag / "work",
        # new outputs from FixDataset.ipynb / AnalysePulsars.ipynb integrations
        "fix_dataset": base / tag / "fix_dataset",
        "binary_analysis": base / tag / "binary_analysis",
        "param_scan": base / tag / "param_scan",
        "qc": base / tag / "qc",
    }
    for p in paths.values():
        safe_mkdir(p)
    return paths
