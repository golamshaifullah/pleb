"""Build fold-specific temporary datasets for held-out reruns."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
import shutil

import pandas as pd

from ..config import PipelineConfig
from ..tim_utils import (
    count_toa_lines,
    extract_flag_value_from_line,
    filter_timfile,
    list_backend_timfiles,
    mjd_from_toa_line,
    parse_include_lines,
)


def build_fold_dataset(
    cfg: PipelineConfig,
    *,
    fold_cfg,
    fold_index: int,
    out_root: Path,
) -> tuple[Path, str]:
    """Create one temporary dataset with a held-out fold removed."""
    dataset_root = Path(cfg.dataset_name)
    pulsars = _selected_pulsars(cfg, dataset_root)
    manifest = _collect_toa_manifest(
        dataset_root,
        pulsars,
        backend_col=fold_cfg.backend_col,
    )
    held_out_key, membership = _compute_fold_membership(
        manifest,
        mode=fold_cfg.mode,
        n_splits=fold_cfg.n_splits,
        fold_index=fold_index,
        time_col=fold_cfg.time_col,
        backend_col=fold_cfg.backend_col,
    )
    tmp_home = out_root / f"fold_{fold_index:02d}"
    tmp_dataset = tmp_home / dataset_root.name
    tmp_home.mkdir(parents=True, exist_ok=True)
    for psr in pulsars:
        src_psr = dataset_root / psr
        dst_psr = tmp_dataset / psr
        shutil.copytree(src_psr, dst_psr, dirs_exist_ok=True)
        _filter_pulsar_timfiles(dst_psr, src_psr, membership)
        _rewrite_alltim_includes(dst_psr)
    return tmp_home, str(held_out_key)


def _selected_pulsars(cfg: PipelineConfig, dataset_root: Path) -> List[str]:
    if cfg.pulsars == "ALL":
        return sorted(
            p.name
            for p in dataset_root.iterdir()
            if p.is_dir() and p.name.startswith("J")
        )
    return [str(p) for p in cfg.pulsars]


def _collect_toa_manifest(
    dataset_root: Path, pulsars: Iterable[str], *, backend_col: str
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    flag = backend_col if backend_col.startswith("-") else f"-{backend_col}"
    for psr in pulsars:
        psr_dir = dataset_root / psr
        for tim in list_backend_timfiles(psr_dir):
            toa_index = 0
            for raw in tim.read_text(encoding="utf-8", errors="ignore").splitlines():
                mjd = mjd_from_toa_line(raw)
                if mjd is None:
                    continue
                rows.append(
                    {
                        "pulsar": psr,
                        "timfile": str(tim.relative_to(dataset_root)),
                        "toa_index": toa_index,
                        "mjd": float(mjd),
                        "backend": extract_flag_value_from_line(raw, flag),
                    }
                )
                toa_index += 1
    return pd.DataFrame(rows)


def _compute_fold_membership(
    manifest: pd.DataFrame,
    *,
    mode: str,
    n_splits: int,
    fold_index: int,
    time_col: str,
    backend_col: str,
) -> tuple[object, Dict[tuple[str, int], bool]]:
    if manifest.empty:
        return "empty", {}
    work = manifest.copy()
    if mode == "time_blocks":
        work["_fold_id"] = pd.qcut(
            pd.to_numeric(work["mjd"], errors="coerce"),
            q=min(int(n_splits), len(work)),
            labels=False,
            duplicates="drop",
        )
        available = sorted(
            int(x) for x in work["_fold_id"].dropna().astype(int).unique().tolist()
        )
        if not available:
            return "all", {
                (row["timfile"], int(row["toa_index"])): True
                for _, row in work.iterrows()
            }
        held_out = int(available[int(fold_index) % len(available)])
    elif mode == "backend_holdout":
        backends = sorted(str(x) for x in work["backend"].fillna("").unique() if str(x))
        if not backends:
            return "all", {
                (row["timfile"], int(row["toa_index"])): True
                for _, row in work.iterrows()
            }
        held_out = backends[int(fold_index) % len(backends)]
        work["_is_held_out"] = (
            work["backend"].astype(str).map(lambda val: val == held_out)
        )
    else:
        raise ValueError(f"Unsupported fold mode for reruns: {mode!r}")
    membership: Dict[tuple[str, int], bool] = {}
    for _, row in work.iterrows():
        key = (str(row["timfile"]), int(row["toa_index"]))
        if mode == "backend_holdout":
            membership[key] = not bool(row["_is_held_out"])
        else:
            membership[key] = int(row["_fold_id"]) != int(held_out)
    return held_out, membership


def _filter_pulsar_timfiles(
    dst_psr: Path, src_psr: Path, membership: Dict[tuple[str, int], bool]
) -> None:
    for src_tim in list_backend_timfiles(src_psr):
        rel = str(src_tim.relative_to(src_psr.parent))
        dst_tim = dst_psr.parent / rel

        def keep_line(_line: str, toa_index: int, *, _rel: str = rel) -> bool:
            return membership.get((_rel, toa_index), True)

        filter_timfile(src_tim, dst_tim, keep_line)


def _rewrite_alltim_includes(psr_dir: Path) -> None:
    for alltim in sorted(psr_dir.glob("*_all*.tim")):
        lines = alltim.read_text(encoding="utf-8", errors="ignore").splitlines()
        include_set = parse_include_lines(alltim)
        keep_includes = set()
        for include in include_set:
            target = (alltim.parent / include).resolve()
            if target.exists() and count_toa_lines(target) > 0:
                keep_includes.add(include)
        out_lines: List[str] = []
        for raw in lines:
            stripped = raw.strip()
            if stripped.startswith("INCLUDE"):
                parts = stripped.split(maxsplit=1)
                if len(parts) == 2 and parts[1].strip() in keep_includes:
                    out_lines.append(raw)
                continue
            out_lines.append(raw)
        alltim.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
