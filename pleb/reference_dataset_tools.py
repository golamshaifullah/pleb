from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_path(src: Path, dest: Path) -> list[Path]:
    copied: list[Path] = []
    if src.is_dir():
        for child in sorted(src.rglob("*")):
            if child.is_dir():
                continue
            rel = child.relative_to(src)
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, out)
            copied.append(out)
        return copied
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    copied.append(dest)
    return copied


def stage_reference_input(
    source_dataset: Path,
    dest_root: Path,
    pulsars: list[str],
    extra_paths: list[str] | None = None,
) -> list[Path]:
    copied: list[Path] = []
    extra_paths = extra_paths or []
    for pulsar in pulsars:
        src = source_dataset / pulsar
        if not src.exists():
            raise FileNotFoundError(f"Missing pulsar directory: {src}")
        copied.extend(_copy_path(src, dest_root / pulsar))
    for rel in extra_paths:
        src = source_dataset / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing extra path: {src}")
        copied.extend(_copy_path(src, dest_root / rel))
    return copied


def stage_expected_outputs(
    source_root: Path,
    dest_root: Path,
    relative_paths: list[str],
) -> list[Path]:
    copied: list[Path] = []
    for rel in relative_paths:
        src = source_root / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing expected artifact: {src}")
        copied.extend(_copy_path(src, dest_root / rel))
    return copied


def collect_manifest_entries(reference_root: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    for section in ("input", "expected"):
        base = reference_root / section
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_dir():
                continue
            entries.append(
                ManifestEntry(
                    path=str(path.relative_to(reference_root)),
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                )
            )
    return entries


def write_reference_manifest(
    reference_root: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    manifest_path = reference_root / "manifest.json"
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "path": entry.path,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
            }
            for entry in collect_manifest_entries(reference_root)
        ],
    }
    if metadata:
        payload["metadata"] = metadata
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
