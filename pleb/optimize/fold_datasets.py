"""Build fold-specific temporary datasets for held-out reruns."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Set
import shutil

import pandas as pd

from ..config import PipelineConfig
from ..tim_utils import (
    count_toa_lines,
    extract_flag_value_from_line,
    filter_timfile,
    is_toa_line,
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
    variant_label: str | None = None,
) -> tuple[Path, str]:
    """Create one temporary dataset with a held-out fold removed.

    Fold membership is built from the same all-tim include files that PQC will
    use.  In particular, when ``pqc_run_variants`` is enabled and variant
    include files exist, the TOA universe is the union of the referenced
    ``<PSR>_<variant>_all.tim`` / ``<PSR>_all.<variant>.tim`` inputs, not every
    stale or alternate ``tims/*.tim`` file present in the pulsar directory.
    """
    dataset_root = Path(cfg.dataset_name).resolve()
    pulsars = _selected_pulsars(cfg, dataset_root)
    manifest = _collect_toa_manifest(
        cfg,
        dataset_root,
        pulsars,
        backend_col=fold_cfg.backend_col,
        variant_label=variant_label,
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
        _filter_pulsar_timfiles(
            cfg,
            dst_psr,
            src_psr,
            psr,
            membership,
            variant_label=variant_label,
        )
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
    cfg: PipelineConfig,
    dataset_root: Path,
    pulsars: Iterable[str],
    *,
    backend_col: str,
    variant_label: str | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    flag = backend_col if backend_col.startswith("-") else f"-{backend_col}"
    for psr in pulsars:
        psr_dir = dataset_root / psr
        for tim in _active_backend_timfiles(cfg, psr_dir, psr, variant_label=variant_label):
            toa_index = 0
            for raw in tim.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not is_toa_line(raw):
                    continue
                mjd = mjd_from_toa_line(raw)
                if mjd is not None:
                    rows.append(
                        {
                            "pulsar": psr,
                            "timfile": str(tim.relative_to(dataset_root)),
                            "toa_index": toa_index,
                            "mjd": float(mjd),
                            "backend": extract_flag_value_from_line(raw, flag),
                        }
                    )
                # Keep this in lockstep with tim_utils.filter_timfile(), which
                # passes a monotonically increasing index for every TOA-like row,
                # even if a malformed row does not yield a parseable MJD.
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
    del time_col, backend_col
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
        work["_is_held_out"] = work["backend"].astype(str).map(lambda val: val == held_out)
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
    cfg: PipelineConfig,
    dst_psr: Path,
    src_psr: Path,
    psr: str,
    membership: Dict[tuple[str, int], bool],
    *,
    variant_label: str | None = None,
) -> None:
    for src_tim in _active_backend_timfiles(cfg, src_psr, psr, variant_label=variant_label):
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


def _active_backend_timfiles(
    cfg: PipelineConfig,
    psr_dir: Path,
    psr: str,
    *,
    variant_label: str | None = None,
) -> List[Path]:
    """Return backend timfiles referenced by the active PQC all-tim files."""
    alltim_files = _active_pqc_alltim_files(cfg, psr_dir, psr, variant_label=variant_label)
    tims: List[Path] = []
    seen: Set[Path] = set()
    for alltim in alltim_files:
        for rel in sorted(parse_include_lines(alltim)):
            tim = (alltim.parent / rel).resolve()
            if tim in seen or not tim.exists() or tim.name.startswith("."):
                continue
            seen.add(tim)
            tims.append(tim)
    if tims:
        return sorted(tims)

    if variant_label not in (None, ""):
        return []

    # Backward-compatible fallback for old datasets without a usable all-tim file.
    return list_backend_timfiles(psr_dir)


def _active_pqc_alltim_files(
    cfg: PipelineConfig,
    psr_dir: Path,
    psr: str,
    *,
    variant_label: str | None = None,
) -> List[Path]:
    """Return the all-tim files PQC will use for this pulsar.

    This mirrors the pipeline's variant behavior: when ``pqc_run_variants`` is
    true and variants exist, run the variants and skip the base alltim; otherwise
    use ``<PSR>_all.tim``.  Fold reruns can pass ``variant_label`` so the TOA
    universe is the selected candidate mask rather than the union of every
    variant in the directory.
    """
    base = psr_dir / f"{psr}_all.tim"
    if variant_label in (None, ""):
        if bool(getattr(cfg, "pqc_run_variants", False)):
            variants = _discover_variant_alltim_files(psr_dir, psr)
            if variants:
                return variants
        return [base] if base.exists() else []

    if str(variant_label) == "base":
        return [base] if base.exists() else []

    variants = [
        path
        for path in _discover_variant_alltim_files(psr_dir, psr)
        if _variant_label_for_alltim_file(path, psr) == str(variant_label)
    ]
    if variants:
        return variants
    return []


def _variant_label_for_alltim_file(path: Path, psr: str) -> str:
    name = path.name
    pref_us = f"{psr}_"
    suff_us = "_all.tim"
    if name.startswith(pref_us) and name.endswith(suff_us):
        return name[len(pref_us) : -len(suff_us)]
    pref_dot = f"{psr}_all."
    if name.startswith(pref_dot) and name.endswith(".tim"):
        return name[len(pref_dot) : -len(".tim")]
    return "base"


def _discover_variant_alltim_files(psr_dir: Path, psr: str) -> List[Path]:
    """Discover variant include files accepted by the pipeline PQC runner."""
    out: List[Path] = []
    seen: Set[Path] = set()
    candidates = sorted(
        {
            *psr_dir.glob(f"{psr}_*_all.tim"),
            *psr_dir.glob(f"{psr}_all.*.tim"),
        }
    )
    for path in candidates:
        name = path.name
        pref_us = f"{psr}_"
        suff_us = "_all.tim"
        if name.startswith(pref_us) and name.endswith(suff_us):
            variant = name[len(pref_us) : -len(suff_us)]
            if not variant or variant == "all":
                continue
        else:
            pref_dot = f"{psr}_all."
            if not name.startswith(pref_dot) or not name.endswith(".tim"):
                continue
            variant = name[len(pref_dot) : -len(".tim")]
            if not variant:
                continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(path)
    return out
