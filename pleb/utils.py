"""General utility helpers for the pipeline.

These helpers provide small filesystem utilities and shared path conventions
used across pipeline modules.
"""

from __future__ import annotations

import shutil
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


def which_or_raise(cmd: str, hint: str = "") -> str:
    """Resolve an executable on PATH or raise.

    Args:
        cmd: Executable name to find.
        hint: Optional hint included in the error message.

    Returns:
        Absolute path to the executable.

    Raises:
        RuntimeError: If the executable is not found on PATH.
    """
    path = shutil.which(cmd)
    if not path:
        raise RuntimeError(
            f"Required executable not found on PATH: {cmd}. {hint}".strip()
        )
    return path


def safe_mkdir(p: Path) -> None:
    """Create a directory tree if it does not exist.

    Args:
        p: Directory path to create.
    """
    p.mkdir(parents=True, exist_ok=True)


def discover_pulsars(home_dir: Path) -> List[str]:
    """Discover pulsar directories under a dataset root.

    Args:
        home_dir: Dataset root directory containing pulsar subdirectories.

    Returns:
        Sorted list of pulsar names (directory names starting with "J").

    Examples:
        List pulsars under a dataset root::

            pulsars = discover_pulsars(Path("/data/epta/EPTA"))
    """
    pulsars: List[str] = []
    for entry in home_dir.iterdir():
        if (
            entry.is_dir()
            and entry.name.startswith("J")
            and not entry.name.startswith(".")
        ):
            pulsars.append(entry.name)
    return sorted(pulsars)


def make_output_tree(
    results_dir: Path,
    branches: List[str],
    outdir_name: Optional[str],
    *,
    lazy: bool = True,
) -> Dict[str, Path]:
    """Create the output directory tree for a pipeline run.

    Args:
        results_dir: Base results directory.
        branches: Branch names used for the tag subdirectory.
        outdir_name: Optional output directory name (timestamped if None).

    Returns:
        Mapping from output labels to created paths.

    Examples:
        Create a results tree for two branches::

            paths = make_output_tree(Path("results"), ["main", "EPTA"], None)
    """
    if outdir_name is None:
        outdir_name = "EPTA_combination_report_" + datetime.now().strftime(
            "%Y%m%dT%H%M"
        )
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
    safe_mkdir(base)
    safe_mkdir(base / tag)
    safe_mkdir(paths["logs"])
    if not lazy:
        for p in paths.values():
            safe_mkdir(p)
    return paths


def cleanup_empty_dirs(root: Path) -> None:
    """Remove empty directories under a root (bottom-up)."""
    root = Path(root)
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if dirnames or filenames:
            continue
        try:
            Path(dirpath).rmdir()
        except OSError:
            pass


def remove_tree_if_exists(path: Path) -> None:
    """Remove a directory tree if it exists."""
    path = Path(path)
    if not path.exists():
        return
    shutil.rmtree(path, ignore_errors=True)
