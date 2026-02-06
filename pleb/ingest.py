"""Ingest pulsar timing files from arbitrary folders into a canonical layout.

This module implements a mapping-driven ingestion flow that scans user-provided
folders for .par and .tim files, resolves pulsar names (B/J aliases), and
writes a canonical dataset layout:

    Jxxxx+xxxx/Jxxxx+xxxx.par
    Jxxxx+xxxx/Jxxxx+xxxx_all.tim
    Jxxxx+xxxx/tims/TEL.BACKEND.CENFREQ.tim
    Jxxxx+xxxx/tmplts/<original_template_name>

Backend parsing is mapping-only (no inference); missing mappings raise errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from collections import Counter
from datetime import datetime
import os
import json
import warnings
import re
import shutil

try:
    import matplotlib
    import numpy as np
    import pandas as pd
except Exception:  # pragma: no cover
    matplotlib = None  # type: ignore
    np = None  # type: ignore
    pd = None  # type: ignore
plt = None  # type: ignore
LineCollection = None  # type: ignore

from .logging_utils import get_logger
from .git_tools import checkout, require_clean_repo

# Accept ASCII +/-; case-insensitive for J/B. Unicode minus variants are normalized first.
_PULSAR_RE = re.compile(r"([BJ]\d{4}[+\-]\d{2,4})", re.IGNORECASE)

logger = get_logger("pleb.ingest")


@dataclass(frozen=True)
class BackendSpec:
    """Describe a backend source root and scan behavior."""

    name: str
    root: Path
    ignore: bool = False
    tim_glob: str = "*.tim"
    ignore_suffixes: Tuple[str, ...] = ("_all.tim",)


@dataclass(frozen=True)
class IngestMapping:
    """Parsed ingest mapping configuration."""

    sources: Tuple[Path, ...]
    par_roots: Tuple[Path, ...]
    template_roots: Tuple[Path, ...]
    backends: Tuple[BackendSpec, ...]
    ignore_backends: Tuple[str, ...]
    pulsar_aliases: Dict[str, str]


class IngestError(RuntimeError):
    """Raised when ingestion fails due to mapping/structure problems."""


def _norm_backend_key(key: str) -> str:
    key = key.strip()
    if key.endswith(".tim"):
        key = key[:-4]
    return key


def _load_mapping(path: Path) -> IngestMapping:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = tuple(Path(p).expanduser().resolve() for p in data.get("sources", []))
    par_roots = tuple(
        Path(p).expanduser().resolve() for p in data.get("par_roots", sources)
    )
    template_roots = tuple(
        Path(p).expanduser().resolve() for p in data.get("template_roots", [])
    )

    backends: List[BackendSpec] = []
    raw_backends = data.get("backends", {})
    for raw_key, raw_cfg in raw_backends.items():
        key = _norm_backend_key(str(raw_key))
        cfg = raw_cfg or {}
        root_raw = cfg.get("root")
        if not root_raw:
            raise IngestError(
                f"Backend '{key}' is missing required 'root' in mapping file."
            )
        root = Path(root_raw).expanduser().resolve()
        backends.append(
            BackendSpec(
                name=key,
                root=root,
                ignore=bool(cfg.get("ignore", False)),
                tim_glob=str(cfg.get("tim_glob", "*.tim")),
                ignore_suffixes=tuple(cfg.get("ignore_suffixes", ["_all.tim"])),
            )
        )

    ignore_backends = tuple(
        _norm_backend_key(k) for k in data.get("ignore_backends", [])
    )
    pulsar_aliases = {
        str(k): str(v) for k, v in (data.get("pulsar_aliases") or {}).items()
    }
    return IngestMapping(
        sources=sources,
        par_roots=par_roots,
        template_roots=template_roots,
        backends=tuple(backends),
        ignore_backends=ignore_backends,
        pulsar_aliases=pulsar_aliases,
    )


def _extract_pulsar_name(path: Path) -> Optional[str]:
    candidates: List[str] = []
    for part in [path.name, *path.parts]:
        # Normalize common Unicode minus variants before matching.
        norm = (
            part.replace("\u2212", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
        )
        m = _PULSAR_RE.search(norm)
        if m:
            cand = m.group(1)
            cand = cand[0].upper() + cand[1:]
            candidates.append(cand)
    if not candidates:
        return None
    # Prefer J if present in any candidate
    for cand in candidates:
        if cand.startswith("J"):
            return cand
    return candidates[0]


def _is_psrfits(path: Path) -> bool:
    """Heuristic check for PSRFITS/FITS header in the first block."""
    try:
        with path.open("rb") as fh:
            block = fh.read(2880)
    except Exception:
        return False
    if len(block) < 80:
        return False
    return b"SIMPLE  =" in block and b"BITPIX" in block and b"NAXIS" in block


def _template_allowed(path: Path) -> bool:
    if path.suffix.lower() in {".par", ".tim", ".clk"}:
        return False
    if _is_psrfits(path):
        return True
    folder_tokens = {p.lower() for p in path.parts}
    if folder_tokens & {"tmplt", "tmplts", "template", "templates"}:
        return True
    allowed_exts = {
        ".ar",
        ".psrchive",
        ".psrfits",
        ".std",
        ".tpl",
        ".tmpl",
        ".template",
        ".txt",
        ".dat",
    }
    if path.name.lower().endswith(".ar.gz"):
        return True
    return path.suffix.lower() in allowed_exts


def _canonical_pulsar(name: str, aliases: Dict[str, str]) -> str:
    if name in aliases:
        return aliases[name]
    if name.startswith("J"):
        return name
    # Require explicit mapping for B-names to avoid silent mislabeling
    raise IngestError(
        f"Encountered B-name '{name}' without an explicit mapping in pulsar_aliases. "
        "Provide a B->J mapping."
    )


def _find_parfiles(
    par_roots: Iterable[Path], aliases: Dict[str, str]
) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    found: Dict[str, List[Path]] = {}
    for root in par_roots:
        if not root.exists():
            continue
        for par in root.rglob("*.par"):
            # Parfile matching should come from the filename itself.
            psr_raw = _extract_pulsar_name(Path(par.name))
            if not psr_raw:
                continue
            psr = _canonical_pulsar(psr_raw, aliases)
            found.setdefault(psr, []).append(par)
    for psr, paths in found.items():
        if len(paths) == 1:
            out[psr] = paths[0]
            continue
        # Prefer exact filename match to the pulsar name, then pick deterministically.
        exact = [p for p in paths if p.stem == psr]
        pick_from = exact or paths
        pick = sorted({p.resolve() for p in pick_from})[0]
        ignored = sorted({str(p.resolve()) for p in paths if p.resolve() != pick})
        warnings.warn(
            f"Multiple parfiles found for {psr}; using {pick} and ignoring {ignored}."
        )
        out[psr] = pick
    return out


def _find_template_files(
    template_roots: Iterable[Path], aliases: Dict[str, str]
) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for root in template_roots:
        if not root.exists():
            continue
        for tpl in root.rglob("*"):
            if tpl.is_dir():
                continue
            if not _template_allowed(tpl):
                continue
            psr_raw = _extract_pulsar_name(tpl)
            if not psr_raw:
                continue
            psr = _canonical_pulsar(psr_raw, aliases)
            out.setdefault(psr, []).append(tpl)
    return out


def _find_clockfiles(roots: Iterable[Path]) -> Dict[str, Path]:
    """Find tempo2 clock files under the provided roots."""
    best: Dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for clk in root.rglob("*.clk"):
            if clk.is_dir():
                continue
            name = clk.name
            if name not in best:
                best[name] = clk
                continue
            cur = best[name]
            try:
                cur_stat = cur.stat()
                new_stat = clk.stat()
            except Exception:
                continue
            # Prefer larger file; if equal, prefer newer mtime.
            if new_stat.st_size > cur_stat.st_size:
                best[name] = clk
            elif new_stat.st_size == cur_stat.st_size and new_stat.st_mtime > cur_stat.st_mtime:
                best[name] = clk
    return best


def _find_timfiles(
    backends: Iterable[BackendSpec],
    aliases: Dict[str, str],
    ignore_backends: Iterable[str],
) -> Dict[str, List[Tuple[str, Path]]]:
    out: Dict[str, List[Tuple[str, Path]]] = {}
    ignore_set = {b for b in ignore_backends}
    for backend in backends:
        if backend.ignore or backend.name in ignore_set:
            continue
        if not backend.root.exists():
            raise IngestError(f"Backend root does not exist: {backend.root}")
        for tim in backend.root.rglob(backend.tim_glob):
            if tim.is_dir():
                continue
            if any(tim.name.endswith(suf) for suf in backend.ignore_suffixes):
                continue
            psr_raw = _extract_pulsar_name(tim)
            if not psr_raw:
                raise IngestError(
                    f"Unable to determine pulsar for tim file: {tim} (backend {backend.name})"
                )
            psr = _canonical_pulsar(psr_raw, aliases)
            out.setdefault(psr, []).append((backend.name, tim))
    return out


def _write_all_tim(pulsar_dir: Path, tim_entries: List[Tuple[str, Path]]) -> None:
    all_tim = pulsar_dir / f"{pulsar_dir.name}_all.tim"
    include_lines = []
    for backend_name, _ in tim_entries:
        include_lines.append(f"INCLUDE tims/{backend_name}.tim")
    all_tim.write_text("\n".join(sorted(set(include_lines))) + "\n", encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _to_rgb(color):
    if isinstance(color, str) and color.startswith("#"):
        c = color.lstrip("#")
        return tuple(int(c[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return color


def _lerp(c1, c2, t):
    r1, g1, b1 = _to_rgb(c1)
    r2, g2, b2 = _to_rgb(c2)
    return (r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t)


def _draw_grad_vline(ax, x0, y1, y2, c1, c2, steps=24, lw=2.0, z=2):
    if y2 < y1:
        y1, y2, c1, c2 = y2, y1, c2, c1
    ys = np.linspace(y1, y2, steps + 1)
    segs, cols = [], []
    for i in range(steps):
        y_start, y_end = ys[i], ys[i + 1]
        t = i / steps
        segs.append([(x0, y_start), (x0, y_end)])
        cols.append(_lerp(c1, c2, t))
    lc = LineCollection(segs, colors=cols, linewidths=lw, zorder=z, capstyle="round")
    ax.add_collection(lc)


def _blend_colors(names, color_map):
    if not names:
        return (0.6, 0.6, 0.6)
    rgb = np.array([_to_rgb(color_map[n]) for n in names], dtype=float)
    return tuple(rgb.mean(axis=0).tolist())


def _plot_upset(df: pd.DataFrame, out_prefix: Path, title: str) -> None:
    # PTA-style columns: all except Pulsar
    labels = [c for c in df.columns if c != "Pulsar"]
    if not labels:
        return

    presence = df[labels].applymap(lambda x: isinstance(x, str) and x.strip() != "")
    row_order = labels

    # Colors (tab10 fallback)
    tab10 = plt.get_cmap("tab10")
    color_map = {name: matplotlib.colors.to_hex(tab10(i % 10)) for i, name in enumerate(row_order)}

    # Build subset counts
    def row_to_subset(row_bool):
        return frozenset(l for l, ok in zip(labels, row_bool) if ok)

    subset_counts = Counter()
    for _, r in presence.iterrows():
        sub = row_to_subset([bool(v) for v in r.values])
        if sub:
            subset_counts[sub] += 1

    regions = [(subset, n) for subset, n in subset_counts.items() if n > 0]
    col_data = []
    for subset, n in regions:
        mask = tuple(lbl in subset for lbl in row_order)
        col_data.append(
            {
                "mask": mask,
                "size": int(n),
                "members": [lbl for lbl in row_order if lbl in subset],
            }
        )
    col_data.sort(key=lambda d: (d["size"], sum(d["mask"]), d["mask"]), reverse=True)

    nrows = len(row_order)
    ncols = len(col_data)
    if ncols == 0:
        return

    x = np.arange(ncols)
    y = np.arange(nrows)

    fig_h = max(4.0, 1.2 + 0.45 * nrows + 0.1 * max(8, ncols))
    fig_w = max(8.0, 1.2 + 0.25 * ncols)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.2, 1.6], hspace=0.12)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_mat = fig.add_subplot(gs[1, 0], sharex=ax_bar)

    heights = [d["size"] for d in col_data]
    bar_colors = [_blend_colors(d["members"], color_map) for d in col_data]
    bars = ax_bar.bar(x, heights, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax_bar.set_ylabel("count")
    ax_bar.set_title(title)
    ax_bar.set_xticks([])
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)
    if heights:
        pad = max(0.5, 0.02 * max(heights))
        for rect, val in zip(bars, heights):
            if val >= 1:
                ax_bar.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + pad,
                    str(val),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    dot_size = 50
    off_alpha = 0.12
    for xi in x:
        ax_mat.scatter(
            np.full(nrows, xi),
            y,
            s=dot_size * 0.6,
            facecolors=(0, 0, 0, off_alpha),
            edgecolors="none",
            zorder=1,
        )
    for xi, d in enumerate(col_data):
        mask = d["mask"]
        for yi, present in enumerate(mask):
            if present:
                name = row_order[yi]
                ax_mat.scatter(
                    xi,
                    yi,
                    s=dot_size,
                    facecolors=color_map[name],
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=3,
                )
    for xi, d in enumerate(col_data):
        present_rows = [i for i, v in enumerate(d["mask"]) if v]
        if len(present_rows) <= 1:
            continue
        for a, b in zip(present_rows[:-1], present_rows[1:]):
            name_a = row_order[a]
            name_b = row_order[b]
            _draw_grad_vline(
                ax_mat,
                x0=xi,
                y1=a,
                y2=b,
                c1=color_map[name_a],
                c2=color_map[name_b],
                steps=24,
                lw=2.0,
                z=2,
            )

    ax_mat.set_xlim(-0.5, max(0, ncols - 0.5))
    ax_mat.set_ylim(-0.5, nrows - 0.5)
    ax_mat.set_xticks([])
    ax_mat.set_yticks(np.arange(nrows))
    ax_mat.set_yticklabels(row_order)
    for tick, name in zip(ax_mat.yaxis.get_ticklabels(), row_order):
        tick.set_color(color_map[name])
    ax_mat.invert_yaxis()
    for xi in x:
        ax_mat.axvline(xi, color=(0, 0, 0, 0.05), linewidth=0.5, zorder=0)
    ax_mat.set_title("Membership matrix", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_ingest_upset_plots(output_root: Path) -> None:
    global plt, LineCollection
    if matplotlib is None or np is None or pd is None:
        logger.warning("matplotlib/numpy/pandas unavailable; skipping ingest upset plots.")
        return
    if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND") is None:
        matplotlib.use("Agg")
    if plt is None or LineCollection is None:
        import matplotlib.pyplot as plt_mod
        from matplotlib.collections import LineCollection as LC
        plt = plt_mod
        LineCollection = LC

    out_dir = output_root / "ingest_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    telescopes: Dict[str, set[str]] = {}
    backends: Dict[str, set[str]] = {}

    for psr_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        tims_dir = psr_dir / "tims"
        if not tims_dir.exists():
            continue
        psr = psr_dir.name
        for tim in tims_dir.glob("*.tim"):
            if tim.name.endswith("_all.tim"):
                continue
            stem = tim.stem  # TEL.BE.CENFREQ
            parts = stem.split(".")
            if not parts:
                continue
            tel = parts[0]
            telescopes.setdefault(tel, set()).add(psr)
            backends.setdefault(stem, set()).add(psr)

    def _build_df(mapping: Dict[str, set[str]]) -> pd.DataFrame:
        pulsars = sorted({p for s in mapping.values() for p in s})
        data = {"Pulsar": pulsars}
        for label in sorted(mapping.keys()):
            present = mapping[label]
            data[label] = [label if p in present else "" for p in pulsars]
        return pd.DataFrame(data)

    if telescopes:
        df_tel = _build_df(telescopes)
        _plot_upset(df_tel, out_dir / "ingest_upset_telescopes", "Ingest: telescope membership")
    if backends:
        df_be = _build_df(backends)
        _plot_upset(df_be, out_dir / "ingest_upset_backends", "Ingest: backend membership")


def ingest_dataset(mapping_file: Path, output_root: Path) -> Dict[str, object]:
    """Ingest pulsar data into a canonical layout using a mapping file."""
    mapping = _load_mapping(mapping_file)
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    parfiles = _find_parfiles(mapping.par_roots, mapping.pulsar_aliases)
    timfiles = _find_timfiles(
        mapping.backends, mapping.pulsar_aliases, mapping.ignore_backends
    )
    templates = _find_template_files(mapping.template_roots, mapping.pulsar_aliases)
    clock_roots = list(mapping.sources) + list(mapping.par_roots) + list(mapping.template_roots)
    clock_roots.extend([b.root for b in mapping.backends])
    clockfiles = _find_clockfiles(clock_roots)

    pulsars = sorted(set(parfiles) | set(timfiles) | set(templates))
    if not pulsars:
        raise IngestError("No pulsars found from mapping sources.")

    report = {
        "output_root": str(output_root),
        "pulsars": [],
        "missing_parfiles": [],
        "missing_timfiles": [],
        "missing_templates": [],
        "clockfiles": [],
    }

    for psr in pulsars:
        psr_dir = output_root / psr
        psr_dir.mkdir(parents=True, exist_ok=True)

        if psr in parfiles:
            _copy_file(parfiles[psr], psr_dir / f"{psr}.par")
        else:
            report["missing_parfiles"].append(psr)

        tim_entries = timfiles.get(psr, [])
        if tim_entries:
            for backend_name, tim_path in tim_entries:
                backend_key = _norm_backend_key(backend_name)
                _copy_file(tim_path, psr_dir / "tims" / f"{backend_key}.tim")
            _write_all_tim(psr_dir, tim_entries)
        else:
            report["missing_timfiles"].append(psr)

        tpl_entries = templates.get(psr, [])
        if tpl_entries:
            for tpl in tpl_entries:
                _copy_file(tpl, psr_dir / "tmplts" / tpl.name)
        else:
            report["missing_templates"].append(psr)

        report["pulsars"].append(psr)

    # Sanitize tim files: replace literal "\\n" with newline characters.
    for psr in pulsars:
        psr_dir = output_root / psr
        tims_dir = psr_dir / "tims"
        candidates = [psr_dir / f"{psr}_all.tim"]
        if tims_dir.exists():
            candidates.extend(sorted(tims_dir.glob("*.tim")))
        for tim in candidates:
            if not tim.exists():
                continue
            text = tim.read_text(encoding="utf-8", errors="ignore")
            if "\\n" in text:
                tim.write_text(text.replace("\\n", "\n"), encoding="utf-8")

    if report["missing_parfiles"]:
        missing = ", ".join(sorted(report["missing_parfiles"]))
        logger.warning("Missing parfiles after ingest: %s", missing)

    if clockfiles:
        clock_dir = output_root / "clockfiles"
        for name, src in sorted(clockfiles.items()):
            _copy_file(src, clock_dir / name)
            report["clockfiles"].append(str(clock_dir / name))

    try:
        _write_ingest_upset_plots(output_root)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to generate ingest upset plots: %s", e)

    return report


def commit_ingest_changes(
    output_root: Path,
    *,
    branch_name: Optional[str] = None,
    base_branch: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> str:
    """Create a new branch at output_root and commit ingest changes."""
    try:
        from git import Repo, InvalidGitRepositoryError  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("GitPython is required for ingest commit branches.") from e

    output_root = Path(output_root).resolve()
    try:
        repo = Repo(str(output_root), search_parent_directories=False)
    except InvalidGitRepositoryError:
        repo = Repo.init(str(output_root))

    require_clean_repo(repo)
    current = repo.active_branch.name if repo.head.is_valid() else ""
    base = (base_branch or current).strip() or current
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    # Default ingest branch is "main" unless explicitly overridden.
    new_branch = (branch_name or "main").strip() or "main"

    existing = {h.name for h in getattr(repo, "heads", [])}
    if not repo.head.is_valid():
        # First commit in a new repo: create branch directly.
        repo.git.checkout("-b", new_branch)
    else:
        if new_branch in existing:
            checkout(repo, new_branch)
        else:
            checkout(repo, base or current)
            repo.git.checkout("-b", new_branch)

    repo.git.add("-A")
    msg = (commit_message or "Ingest: collected files").strip()
    if repo.is_dirty(untracked_files=True):
        repo.index.commit(msg)
    else:
        repo.git.commit("--allow-empty", "-m", msg + " (no changes)")

    # Create a read-only snapshot branch for the ingest baseline.
    # We intentionally do not check it out or modify it.
    raw_branch = "raw"
    if raw_branch not in existing:
        repo.git.branch(raw_branch, new_branch)

    if current:
        checkout(repo, current)
    return new_branch
