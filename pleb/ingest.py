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
import json
import re
import shutil

_PULSAR_RE = re.compile(r"([BJ]\\d{4}[+-]\\d{2,4})")


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
        m = _PULSAR_RE.search(part)
        if m:
            candidates.append(m.group(1))
    if not candidates:
        return None
    # Prefer J if present in any candidate
    for cand in candidates:
        if cand.startswith("J"):
            return cand
    return candidates[0]


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
    collisions: Dict[str, List[Path]] = {}
    for root in par_roots:
        if not root.exists():
            continue
        for par in root.rglob("*.par"):
            psr_raw = _extract_pulsar_name(par)
            if not psr_raw:
                continue
            psr = _canonical_pulsar(psr_raw, aliases)
            if psr in out:
                collisions.setdefault(psr, []).extend([out[psr], par])
            else:
                out[psr] = par
    if collisions:
        detail = "; ".join(
            f"{k}: {sorted({str(p) for p in v})}" for k, v in collisions.items()
        )
        raise IngestError(f"Multiple parfiles found for pulsar(s): {detail}")
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
            psr_raw = _extract_pulsar_name(tpl)
            if not psr_raw:
                continue
            psr = _canonical_pulsar(psr_raw, aliases)
            out.setdefault(psr, []).append(tpl)
    return out


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
    all_tim.write_text("\\n".join(sorted(set(include_lines))) + "\\n", encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


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

    pulsars = sorted(set(parfiles) | set(timfiles) | set(templates))
    if not pulsars:
        raise IngestError("No pulsars found from mapping sources.")

    report = {
        "output_root": str(output_root),
        "pulsars": [],
        "missing_parfiles": [],
        "missing_timfiles": [],
        "missing_templates": [],
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

    return report
