"""Helpers for invoking the reviewer-facing synthesis package generator."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional
import sys


@dataclass(frozen=True)
class ReviewSynthesisResult:
    """Return value for the review synthesis runner."""

    out_dir: Path


@lru_cache(maxsize=1)
def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "make_review_synthesis.py"
    spec = importlib_util.spec_from_file_location(
        "pleb._review_synthesis_script", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load review synthesis script: {script_path}")
    module = importlib_util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _append_repeated(argv: list[str], flag: str, values: Optional[Iterable[str]]) -> None:
    for value in values or []:
        text = str(value).strip()
        if text:
            argv.extend([flag, text])


def run_review_synthesis(
    *,
    psr: str,
    slug: str,
    workflow_config: str | Path | None,
    repo_root: Path,
    dataset_root: Path,
    results_root: Path,
    final_branch: str,
    out: Path,
    overrides: str | Path | None = None,
    max_keep_points: int = 4000,
    top_n_rows: int = 50,
    stage_branch: Optional[Iterable[str]] = None,
    stage_run: Optional[Iterable[str]] = None,
) -> ReviewSynthesisResult:
    """Run the synthesis script in-process and return the written output path."""

    module = _load_script_module()
    argv = [
        "--psr",
        str(psr),
        "--slug",
        str(slug),
        "--repo-root",
        str(Path(repo_root).expanduser().resolve()),
        "--dataset-root",
        str(Path(dataset_root).expanduser().resolve()),
        "--results-root",
        str(Path(results_root).expanduser().resolve()),
        "--final-branch",
        str(final_branch),
        "--out",
        str(Path(out).expanduser().resolve()),
        "--max-keep-points",
        str(int(max_keep_points)),
        "--top-n-rows",
        str(int(top_n_rows)),
    ]
    if workflow_config not in (None, ""):
        argv.extend(["--workflow-config", str(Path(workflow_config).expanduser().resolve())])
    if overrides not in (None, ""):
        argv.extend(["--overrides", str(Path(overrides).expanduser().resolve())])
    _append_repeated(argv, "--stage-branch", stage_branch)
    _append_repeated(argv, "--stage-run", stage_run)

    exit_code = int(module.main(argv) or 0)
    if exit_code != 0:
        raise RuntimeError(
            f"Review synthesis script exited with code {exit_code} for output {out}"
        )
    return ReviewSynthesisResult(out_dir=Path(out).expanduser().resolve())
