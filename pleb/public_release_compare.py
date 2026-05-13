"""Download and compare pulsar parameter values from public PTA releases.

This module focuses on orchestration and normalization:

- discover latest release artifacts for configured providers,
- download/extract release assets,
- parse ``.par`` files into parameter tables,
- normalize astrometric coordinates into a common ICRS representation,
- write comparison tables for downstream review.
"""

from __future__ import annotations

from .compat import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import re
import shutil
import subprocess
import tarfile
import urllib.parse
import urllib.request
import zipfile

try:  # py3.11+
    import tomllib  # type: ignore
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord

from .ingest import _canonical_pulsar as _ingest_canonical_pulsar
from .ingest import _load_mapping as _load_ingest_mapping
from .logging_utils import get_logger

logger = get_logger("pleb.public_release_compare")


_PAR_IGNORE_PREFIXES = ("C", "#")
_PULSAR_NAME_RX = re.compile(r"([BJ]\d{4}[+-]\d{4}[A-Z]?)", re.IGNORECASE)
_COMPARE_PARAM_MAP = {
    "RAJ": "RA_ICRS_DEG",
    "ELONG": "RA_ICRS_DEG",
    "DECJ": "DEC_ICRS_DEG",
    "ELAT": "DEC_ICRS_DEG",
}
_ASTROMETRY_KEYS = {
    "RAJ",
    "DECJ",
    "ELONG",
    "ELAT",
    "PMRA",
    "PMDEC",
    "PMELONG",
    "PMELAT",
    "PX",
}
_SUPPORTED_ARCHIVE_SUFFIXES = (
    ".tar.gz",
    ".tgz",
    ".tar",
    ".zip",
)


@dataclass(slots=True)
class ProviderSpec:
    """Provider configuration for public-release discovery.

    Attributes
    ----------
    name : str
        Human-readable provider identifier used in outputs.
    type : str, optional
        Resolver backend type (currently ``"zenodo_search"``).
    query : str, optional
        Provider-specific search query used to discover release records.
    url : str, optional
        Exact provider URL (for example a Zenodo record URL or Git repo URL).
    record_id : str, optional
        Exact Zenodo record id or DOI-like token when ``type="zenodo_record"``.
    ref : str, optional
        Optional Git branch/tag/commit-ish when ``type="git_clone"``.
    max_records : int, optional
        Maximum number of candidate records inspected per provider.
    max_assets : int, optional
        Maximum number of downloadable assets selected from a record.
    asset_regex : str, optional
        Regular expression used to filter downloadable asset names.
    par_glob : str, optional
        Relative glob pattern used to locate provider-specific ``.par`` files
        inside the extracted archive or cloned repository.
    """

    name: str
    type: str = "zenodo_search"
    query: str = ""
    url: str = ""
    record_id: str = ""
    ref: str = ""
    max_records: int = 10
    max_assets: int = 1
    asset_regex: str = r".*\.(tar\.gz|tgz|tar|zip|par)$"
    par_glob: str = "**/*.par"


@dataclass(slots=True)
class ReleaseAsset:
    """Resolved downloadable release artifact.

    Attributes
    ----------
    provider : str
        Provider name that yielded this asset.
    release_id : str
        Provider-native release identifier.
    title : str
        Human-readable release title.
    url : str
        Direct download URL for the asset.
    filename : str
        Original asset filename.
    asset_type : str, optional
        Fetch mode. ``"download"`` downloads one asset URL; ``"git_clone"``
        clones a repository snapshot into the cache.
    ref : str, optional
        Optional repository ref used when ``asset_type="git_clone"``.
    par_glob : str, optional
        Relative glob pattern used to locate ``.par`` files for this asset.
    """

    provider: str
    release_id: str
    title: str
    url: str
    filename: str
    asset_type: str = "download"
    ref: str = ""
    par_glob: str = "**/*.par"


def load_provider_specs(path: Path) -> List[ProviderSpec]:
    """Load provider definitions from a TOML catalog.

    Parameters
    ----------
    path : pathlib.Path
        TOML file path. The expected shape is a top-level ``[providers]``
        table keyed by provider name.

    Returns
    -------
    list of ProviderSpec
        Parsed provider specifications. Invalid entries are skipped.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    block = data.get("providers", {})
    out: List[ProviderSpec] = []
    if not isinstance(block, dict):
        return out
    for name, spec in block.items():
        if not isinstance(spec, dict):
            continue
        out.append(
            ProviderSpec(
                name=str(name),
                type=str(spec.get("type", "zenodo_search")),
                query=str(spec.get("query", "")),
                url=str(spec.get("url", "")),
                record_id=str(spec.get("record_id", "")),
                ref=str(spec.get("ref", "")),
                max_records=int(spec.get("max_records", 10)),
                max_assets=int(spec.get("max_assets", 1)),
                asset_regex=str(
                    spec.get("asset_regex", r".*\.(tar\.gz|tgz|tar|zip|par)$")
                ),
                par_glob=str(spec.get("par_glob", "**/*.par")),
            )
        )
    return out


def _http_get_json(url: str) -> Dict[str, object]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "pleb-public-release-compare/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "pleb-public-release-compare/1.0"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        dst.write_bytes(resp.read())
    return dst


def _parse_zenodo_record_id(value: str) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return raw
    for patt in (r"/records/(\d+)", r"zenodo\.(\d+)"):
        m = re.search(patt, raw, flags=re.IGNORECASE)
        if m:
            return str(m.group(1))
    return None


def _filename_from_url(url: str) -> str:
    path = urllib.parse.urlparse(str(url or "")).path
    name = Path(path).name.strip()
    return name or "asset.bin"


def _resolve_zenodo_assets(spec: ProviderSpec) -> List[ReleaseAsset]:
    if not spec.query.strip():
        return []
    q = urllib.parse.quote_plus(spec.query)
    url = (
        "https://zenodo.org/api/records"
        f"?q={q}&sort=mostrecent&page=1&size={max(1, spec.max_records)}"
    )
    payload = _http_get_json(url)
    hits = payload.get("hits", {})
    records = hits.get("hits", []) if isinstance(hits, dict) else []
    if not isinstance(records, list):
        return []
    rx = re.compile(spec.asset_regex, re.IGNORECASE)
    assets: List[ReleaseAsset] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        rec_id = str(rec.get("id", ""))
        metadata = rec.get("metadata", {})
        title = (
            str(metadata.get("title", "")) if isinstance(metadata, dict) else ""
        ) or rec_id
        files = rec.get("files", [])
        if not isinstance(files, list):
            continue
        matched = 0
        for f in files:
            if not isinstance(f, dict):
                continue
            key = str(f.get("key", ""))
            links = f.get("links", {})
            durl = ""
            if isinstance(links, dict):
                durl = str(links.get("self", "")) or str(links.get("download", ""))
            if not key or not durl:
                continue
            if not rx.match(key):
                continue
            assets.append(
                ReleaseAsset(
                    provider=spec.name,
                    release_id=rec_id,
                    title=title,
                    url=durl,
                    filename=key,
                    par_glob=str(spec.par_glob or "**/*.par"),
                )
            )
            matched += 1
            if matched >= max(1, spec.max_assets):
                break
        if matched > 0:
            break
    return assets


def _resolve_zenodo_record_assets(spec: ProviderSpec) -> List[ReleaseAsset]:
    rec_id = _parse_zenodo_record_id(spec.record_id or spec.url or spec.query)
    if not rec_id:
        return []
    payload = _http_get_json(f"https://zenodo.org/api/records/{rec_id}")
    if not isinstance(payload, dict):
        return []
    metadata = payload.get("metadata", {})
    title = str(metadata.get("title", "")) if isinstance(metadata, dict) else ""
    title = title or rec_id
    files = payload.get("files", [])
    if not isinstance(files, list):
        return []
    rx = re.compile(spec.asset_regex, re.IGNORECASE)
    assets: List[ReleaseAsset] = []
    for f in files:
        if not isinstance(f, dict):
            continue
        key = str(f.get("key", ""))
        links = f.get("links", {})
        durl = ""
        if isinstance(links, dict):
            durl = str(links.get("self", "")) or str(links.get("download", ""))
        if not key or not durl:
            continue
        if not rx.match(key):
            continue
        assets.append(
            ReleaseAsset(
                provider=spec.name,
                release_id=rec_id,
                title=title,
                url=durl,
                filename=key,
                par_glob=str(spec.par_glob or "**/*.par"),
            )
        )
        if len(assets) >= max(1, spec.max_assets):
            break
    return assets


def _resolve_direct_url_assets(spec: ProviderSpec) -> List[ReleaseAsset]:
    url = str(spec.url or "").strip()
    if not url:
        return []
    fname = _filename_from_url(url)
    rx = re.compile(spec.asset_regex, re.IGNORECASE)
    if not rx.match(fname):
        return []
    release_id = _parse_zenodo_record_id(url) or fname
    return [
        ReleaseAsset(
            provider=spec.name,
            release_id=str(release_id),
            title=str(spec.name),
            url=url,
            filename=fname,
            par_glob=str(spec.par_glob or "**/*.par"),
        )
    ]


def _resolve_git_clone_assets(spec: ProviderSpec) -> List[ReleaseAsset]:
    url = str(spec.url or "").strip()
    if not url:
        return []
    repo_name = _filename_from_url(url)
    if repo_name.lower().endswith(".git"):
        repo_name = repo_name[:-4]
    release_id = str(spec.ref or repo_name or "repo")
    return [
        ReleaseAsset(
            provider=spec.name,
            release_id=release_id,
            title=repo_name or spec.name,
            url=url,
            filename=repo_name or "repo",
            asset_type="git_clone",
            ref=str(spec.ref or ""),
            par_glob=str(spec.par_glob or "**/*.par"),
        )
    ]


def resolve_latest_assets(spec: ProviderSpec) -> List[ReleaseAsset]:
    """Resolve latest downloadable assets for a provider.

    Parameters
    ----------
    spec : ProviderSpec
        Provider lookup specification.

    Returns
    -------
    list of ReleaseAsset
        Downloadable assets selected using provider-specific logic.

    Raises
    ------
    ValueError
        If ``spec.type`` is unsupported.
    """
    typ = spec.type.strip().lower()
    if typ == "zenodo_search":
        return _resolve_zenodo_assets(spec)
    if typ == "zenodo_record":
        return _resolve_zenodo_record_assets(spec)
    if typ == "direct_url":
        return _resolve_direct_url_assets(spec)
    if typ == "git_clone":
        return _resolve_git_clone_assets(spec)
    raise ValueError(f"Unsupported provider type: {spec.type}")


def _extract_archive(asset_file: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name_l = asset_file.name.lower()
    if any(name_l.endswith(sfx) for sfx in (".zip",)):
        with zipfile.ZipFile(asset_file, "r") as zf:
            zf.extractall(out_dir)
        return out_dir
    if any(name_l.endswith(sfx) for sfx in (".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if name_l.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(asset_file, mode) as tf:
            tf.extractall(out_dir)
        return out_dir
    if name_l.endswith(".par"):
        target = out_dir / asset_file.name
        target.write_bytes(asset_file.read_bytes())
        return out_dir
    raise ValueError(f"Unsupported asset type: {asset_file}")


def _find_par_files(root: Path, pattern: str = "**/*.par") -> List[Path]:
    pat = str(pattern or "**/*.par").strip() or "**/*.par"
    return sorted([p for p in root.glob(pat) if p.is_file()], key=lambda p: str(p))


def _clone_git_repo(url: str, dst: Path, *, ref: str | None = None) -> Path:
    git = shutil.which("git")
    if not git:
        raise RuntimeError(
            "git executable not found on PATH; cannot fetch git_clone provider."
        )
    if dst.exists():
        if (dst / ".git").exists():
            return dst
        if dst.is_dir():
            shutil.rmtree(dst, ignore_errors=True)
        else:
            dst.unlink(missing_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [git, "clone", "--depth", "1"]
    if ref:
        cmd.extend(["--branch", str(ref), "--single-branch"])
    cmd.extend([url, str(dst)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = f"git clone failed for {url}"
        if stderr:
            msg += f"; stderr: {stderr}"
        if stdout:
            msg += f"; stdout: {stdout}"
        raise RuntimeError(msg)
    return dst


def _safe_float(token: str) -> Optional[float]:
    try:
        return float(token)
    except Exception:
        return None


def _load_alias_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    try:
        mapping = _load_ingest_mapping(Path(path).expanduser())
    except Exception as e:
        logger.warning("Failed to load ingest alias mapping from %s: %s", path, e)
        return {}
    return dict(mapping.pulsar_aliases)


def _canonicalize_pulsar_token(token: str, aliases: Dict[str, str]) -> str:
    raw = str(token or "").strip().upper()
    if not raw:
        return raw
    try:
        return str(_ingest_canonical_pulsar(raw, aliases)).upper()
    except Exception:
        return raw


def _extract_par_declared_pulsar_name(text: str) -> str:
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith(_PAR_IGNORE_PREFIXES):
            continue
        toks = re.split(r"\s+", line)
        if len(toks) < 2:
            continue
        key = str(toks[0]).strip().upper()
        if key in {"PSRJ", "PSRB", "PSR"}:
            return str(toks[1]).strip().upper()
    return ""


def _canonical_pulsar_name(path: Path, aliases: Optional[Dict[str, str]] = None) -> str:
    """Resolve a canonical pulsar name from a parfile path or contents.

    ## GMS: Check if this is used somewhere.
    """
    alias_map = aliases or {}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = ""
    declared = _extract_par_declared_pulsar_name(text)
    if declared:
        return _canonicalize_pulsar_token(declared, alias_map)
    m = _PULSAR_NAME_RX.search(path.name)
    if m:
        return _canonicalize_pulsar_token(str(m.group(1)).upper(), alias_map)
    return path.stem


def _git_repo_root_for_path(path: Path) -> Optional[Path]:
    probe = Path(path).expanduser().resolve()
    res = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(probe if probe.is_dir() else probe.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return None
    root = str(res.stdout or "").strip()
    return Path(root).resolve() if root else None


def _path_in_repo_required(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception as exc:
        raise RuntimeError(f"Path {path} is not under repo root {repo_root}.") from exc


def _git_ls_files_at_ref(repo_root: Path, ref: str, prefix: str) -> List[str]:
    res = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "--", prefix],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return []
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def _git_show_file(repo_root: Path, ref: str, path_in_repo: str) -> bytes | None:
    res = subprocess.run(
        ["git", "show", f"{ref}:{path_in_repo}"],
        cwd=str(repo_root),
        capture_output=True,
        text=False,
        check=False,
    )
    if res.returncode != 0:
        return None
    return res.stdout


def _comparison_param(key: str) -> str:
    return _COMPARE_PARAM_MAP.get(str(key).upper(), str(key).upper())


def _wrapped_signed_angle_delta_deg(value: float, reference: float) -> float:
    return float(((float(value) - float(reference) + 180.0) % 360.0) - 180.0)


def _comparison_value_series(values: pd.Series, compare_param: str) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    if str(compare_param).upper() != "RA_ICRS_DEG":
        return out
    arr = out.to_numpy(dtype=float, na_value=np.nan)
    finite = np.flatnonzero(np.isfinite(arr))
    if finite.size == 0:
        return out
    ref = float(arr[finite[0]])
    arr_cmp = arr.copy()
    for idx in finite:
        arr_cmp[idx] = ref + _wrapped_signed_angle_delta_deg(float(arr[idx]), ref)
    return pd.Series(arr_cmp, index=out.index, dtype=float)


def _ecliptic_icrs(elong_deg: float, elat_deg: float) -> SkyCoord:
    return SkyCoord(
        lon=float(elong_deg) * u.deg,
        lat=float(elat_deg) * u.deg,
        frame=BarycentricTrueEcliptic(equinox="J2000"),
    ).icrs


def _normalized_coordinate_error(
    key: str, raw_value: str, raw_error: object, context: Dict[str, str]
) -> Optional[float]:
    err = _safe_float(str(raw_error))
    if err is None or not np.isfinite(err):
        return None
    err = abs(float(err))
    key_u = str(key).upper()
    if key_u == "RAJ":
        return err / 240.0
    if key_u == "DECJ":
        return err / 3600.0
    if key_u == "ELONG":
        elong = _safe_float(str(raw_value))
        elat = _safe_float(context.get("ELAT", ""))
        if elong is None or elat is None:
            return None
        if err == 0.0:
            return 0.0
        c_lo = _ecliptic_icrs(float(elong) - err, float(elat))
        c_hi = _ecliptic_icrs(float(elong) + err, float(elat))
        return 0.5 * abs(
            _wrapped_signed_angle_delta_deg(float(c_hi.ra.deg), float(c_lo.ra.deg))
        )
    if key_u == "ELAT":
        elong = _safe_float(context.get("ELONG", ""))
        elat = _safe_float(str(raw_value))
        if elong is None or elat is None:
            return None
        if err == 0.0:
            return 0.0
        c_lo = _ecliptic_icrs(float(elong), float(elat) - err)
        c_hi = _ecliptic_icrs(float(elong), float(elat) + err)
        return 0.5 * abs(float(c_hi.dec.deg) - float(c_lo.dec.deg))
    return err


def _parse_parfile(
    path: Path,
    *,
    aliases: Optional[Dict[str, str]] = None,
    pulsar_override: Optional[str] = None,
) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _parse_par_text(
        text,
        parfile_label=str(path),
        aliases=aliases,
        pulsar_override=pulsar_override,
    )


def _parse_par_text(
    text: str,
    *,
    parfile_label: str,
    aliases: Optional[Dict[str, str]] = None,
    pulsar_override: Optional[str] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    alias_map = aliases or {}
    declared = _extract_par_declared_pulsar_name(text)
    if pulsar_override:
        pulsar = str(pulsar_override).strip()
    elif declared:
        pulsar = _canonicalize_pulsar_token(declared, alias_map)
    else:
        m = _PULSAR_NAME_RX.search(Path(parfile_label).name)
        if m:
            pulsar = _canonicalize_pulsar_token(str(m.group(1)).upper(), alias_map)
        else:
            pulsar = Path(parfile_label).stem
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(_PAR_IGNORE_PREFIXES):
            continue
        toks = re.split(r"\s+", line)
        if len(toks) < 2:
            continue
        key = toks[0].strip().upper()
        if key in ("JUMP", "EFAC", "EQUAD", "T2EFAC", "T2EQUAD"):
            continue
        val = toks[1].strip()
        fit = None
        err = None
        if len(toks) >= 3:
            fit = _safe_float(toks[2])
        if len(toks) >= 4:
            err = _safe_float(toks[3])
        rows.append(
            {
                "pulsar": pulsar,
                "param": key,
                "value_raw": val,
                "fit_flag": fit,
                "error_raw": err,
                "parfile": str(parfile_label),
            }
        )
    return pd.DataFrame(rows)


def _normalize_local_pulsars(pulsars: object) -> List[str]:
    if pulsars in (None, "", "ALL"):
        return []
    if isinstance(pulsars, str):
        items = [pulsars]
    else:
        try:
            items = list(pulsars)  # type: ignore[arg-type]
        except Exception:
            items = [str(pulsars)]
    return [
        str(x).strip()
        for x in items
        if str(x).strip() and str(x).strip().upper() != "ALL"
    ]


def _discover_local_variant_parfiles(
    dataset_root: Path,
    *,
    pulsars: Sequence[str],
    aliases: Dict[str, str],
) -> List[Tuple[str, str, Path]]:
    out: List[Tuple[str, str, Path]] = []
    wanted = list(pulsars)
    if not wanted:
        wanted = [p.name for p in sorted(dataset_root.iterdir()) if p.is_dir()]
    for raw_psr in wanted:
        psr = _canonicalize_pulsar_token(raw_psr, aliases)
        psr_dir = dataset_root / psr
        if not psr_dir.is_dir():
            logger.warning(
                "Local compare_public pulsar directory not found: %s", psr_dir
            )
            continue
        variant_paths: Dict[str, Path] = {}
        base = psr_dir / f"{psr}.par"
        if base.exists():
            variant_paths["base"] = base
        for path in sorted(psr_dir.glob(f"{psr}_*.par")):
            suffix = path.stem[len(psr) + 1 :]
            if not suffix:
                continue
            variant_paths[str(suffix)] = path
        for variant, path in sorted(variant_paths.items()):
            out.append((psr, variant, path))
    return out


def _discover_local_variant_parfiles_at_ref(
    repo_root: Path,
    dataset_root: Path,
    *,
    ref: str,
    pulsars: Sequence[str],
    aliases: Dict[str, str],
) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    dataset_rel = _path_in_repo_required(repo_root, dataset_root)
    wanted = list(pulsars)
    if not wanted:
        files = _git_ls_files_at_ref(repo_root, ref, dataset_rel)
        seen: set[str] = set()
        for rel_path in files:
            try:
                psr = Path(rel_path).relative_to(dataset_rel).parts[0]
            except Exception:
                continue
            if psr and psr not in seen:
                wanted.append(psr)
                seen.add(psr)
    for raw_psr in wanted:
        psr = _canonicalize_pulsar_token(raw_psr, aliases)
        prefix = f"{dataset_rel}/{psr}"
        files = _git_ls_files_at_ref(repo_root, ref, prefix)
        if not files:
            logger.warning(
                "Local compare_public pulsar directory not found on branch %s: %s",
                ref,
                dataset_root / psr,
            )
            continue
        variant_paths: Dict[str, str] = {}
        base_rel = f"{dataset_rel}/{psr}/{psr}.par"
        if base_rel in files:
            variant_paths["base"] = base_rel
        for rel_path in sorted(files):
            path = Path(rel_path)
            if path.suffix.lower() != ".par":
                continue
            if path.parent.as_posix() != f"{dataset_rel}/{psr}":
                continue
            name = path.stem
            prefix_name = f"{psr}_"
            if name.startswith(prefix_name):
                suffix = name[len(prefix_name) :]
                if suffix:
                    variant_paths[str(suffix)] = rel_path
        for variant, rel_path in sorted(variant_paths.items()):
            out.append((psr, variant, rel_path))
    return out


def _extract_astrometry_context(df: pd.DataFrame) -> Dict[str, str]:
    ctx: Dict[str, str] = {}
    for key in ("RAJ", "DECJ", "ELONG", "ELAT", "PMRA", "PMDEC", "PMELONG", "PMELAT"):
        sub = df[df["param"] == key]
        if len(sub):
            ctx[key] = str(sub.iloc[-1]["value_raw"])
    return ctx


def _normalize_astrometry_row(
    row: pd.Series, context: Dict[str, str]
) -> Tuple[str, Optional[float], Optional[float], Optional[str], Optional[str]]:
    key = str(row["param"]).upper()
    raw = str(row["value_raw"])
    compare_param = _comparison_param(key)
    err_num = _normalized_coordinate_error(key, raw, row.get("error_raw"), context)
    if key == "RAJ":
        c = SkyCoord(
            ra=raw,
            dec=context.get("DECJ", "0d"),
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )
        return compare_param, float(c.ra.deg), err_num, "deg", "icrs"
    if key == "DECJ":
        c = SkyCoord(
            ra=context.get("RAJ", "0h"),
            dec=raw,
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )
        return compare_param, float(c.dec.deg), err_num, "deg", "icrs"
    if key == "ELONG":
        elat = context.get("ELAT")
        if not elat:
            return compare_param, None, err_num, None, None
        c = _ecliptic_icrs(float(raw), float(elat))
        return compare_param, float(c.ra.deg), err_num, "deg", "icrs"
    if key == "ELAT":
        elong = context.get("ELONG")
        if not elong:
            return compare_param, None, err_num, None, None
        c = _ecliptic_icrs(float(elong), float(raw))
        return compare_param, float(c.dec.deg), err_num, "deg", "icrs"
    if key in ("PMRA", "PMDEC", "PMELONG", "PMELAT", "PX"):
        v = _safe_float(raw)
        if v is None:
            return compare_param, None, None, None, None
        return (
            compare_param,
            float(v),
            _safe_float(str(row.get("error_raw"))),
            "mas/yr" if key.startswith("PM") else "mas",
            "icrs",
        )
    v = _safe_float(raw)
    if v is None:
        return compare_param, None, None, None, None
    return compare_param, float(v), _safe_float(str(row.get("error_raw"))), None, None


def normalize_astrometry(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize astrometric parameters into common numeric conventions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input parameter table containing at least ``provider``, ``pulsar``,
        ``parfile``, ``param``, and ``value_raw`` columns.

    Returns
    -------
    pandas.DataFrame
        Copy of the input rows augmented with:
        ``value_num`` (normalized numeric value when available),
        ``unit_norm``, ``frame_norm``, and ``is_astrometry``.

    Notes
    -----
    The function converts sky coordinates to an ICRS/degree basis where
    possible to support cross-provider comparisons on a common frame.
    """
    if df.empty:
        return df.copy()
    out_rows: List[Dict[str, object]] = []
    for (provider, pulsar, parfile), sub in df.groupby(
        ["provider", "pulsar", "parfile"], dropna=False
    ):
        ctx = _extract_astrometry_context(sub)
        for row in sub.itertuples(index=False):
            s = pd.Series(row._asdict())
            compare_param, val_num, err_num, unit, frame = _normalize_astrometry_row(
                s, ctx
            )
            out_rows.append(
                {
                    **row._asdict(),
                    "compare_param": compare_param,
                    "value_num": val_num,
                    "error_num": err_num,
                    "unit_norm": unit,
                    "frame_norm": frame,
                    "is_astrometry": str(getattr(row, "param")).upper()
                    in _ASTROMETRY_KEYS,
                }
            )
    return pd.DataFrame(out_rows)


def _download_and_extract_assets(
    assets: Iterable[ReleaseAsset], cache_root: Path
) -> Dict[str, List[Path]]:
    cache_root = Path(cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    provider_to_pars: Dict[str, List[Path]] = {}
    for a in assets:
        pdir = cache_root / a.provider / a.release_id
        pdir.mkdir(parents=True, exist_ok=True)
        if str(a.asset_type).strip().lower() == "git_clone":
            repo_root = _clone_git_repo(
                a.url, pdir / "repo", ref=(str(a.ref).strip() or None)
            )
            par_files = _find_par_files(repo_root, a.par_glob)
        else:
            fname = Path(a.filename).name or _filename_from_url(a.url)
            downloaded = pdir / fname
            if not downloaded.exists() or downloaded.stat().st_size <= 0:
                downloaded = _http_download(a.url, downloaded)
            extracted = pdir / "extracted"
            if not extracted.exists() or not any(extracted.iterdir()):
                extracted = _extract_archive(downloaded, extracted)
            par_files = _find_par_files(extracted, a.par_glob)
            if not par_files:
                extracted = _extract_archive(downloaded, extracted)
                par_files = _find_par_files(extracted, a.par_glob)
        provider_to_pars.setdefault(a.provider, []).extend(par_files)
    return provider_to_pars


def _default_public_release_cache_dir(out_dir: Path) -> Path:
    probe = Path(__file__).resolve()
    for parent in (probe.parent, *probe.parents):
        if (parent / "data_source").is_dir():
            return (parent / "public_release_cache").resolve()
    return (Path(out_dir).expanduser().resolve() / "downloads").resolve()


def _build_comparison(df: pd.DataFrame) -> pd.DataFrame:
    source_col = "source_label" if "source_label" in df.columns else "provider"
    group_param_col = "compare_param" if "compare_param" in df.columns else "param"
    error_col = "error_num" if "error_num" in df.columns else "error_raw"
    if df.empty:
        return pd.DataFrame(
            columns=[
                "pulsar",
                "param",
                "source_params",
                "n_providers",
                "providers",
                "n_numeric_values",
                "n_error_values",
                "min_value",
                "max_value",
                "span",
                "mean_error",
                "worst_provider_pair",
                "sigma_tension_max",
                "sigma_tension_mean",
                "reduced_chi2",
                "agreement_class",
            ]
        )
    grp = df.groupby(["pulsar", group_param_col], dropna=False)
    rows: List[Dict[str, object]] = []
    for (pulsar, param), sub in grp:
        vals = _comparison_value_series(sub["value_num"], str(param))
        errs = pd.to_numeric(sub[error_col], errors="coerce")
        providers = sorted(set(sub[source_col].astype(str)))
        source_params = sorted(set(sub["param"].astype(str)))
        finite_vals = vals[np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan))]
        if len(finite_vals):
            vmin = float(finite_vals.min())
            vmax = float(finite_vals.max())
            span = float(vmax - vmin)
        else:
            vmin = np.nan
            vmax = np.nan
            span = np.nan
        sigma_tension_max = np.nan
        sigma_tension_mean = np.nan
        reduced_chi2 = np.nan
        worst_provider_pair = ""
        agreement_class = "no_error_metric"
        sigma_rows = sub.copy()
        sigma_rows["value_cmp"] = vals
        sigma_rows["error_cmp"] = errs
        sigma_rows = sigma_rows[
            np.isfinite(sigma_rows["value_cmp"].to_numpy(dtype=float, na_value=np.nan))
            & np.isfinite(
                sigma_rows["error_cmp"].to_numpy(dtype=float, na_value=np.nan)
            )
            & (sigma_rows["error_cmp"].astype(float) > 0.0)
        ]
        if len(sigma_rows) >= 2:
            pair_scores: List[float] = []
            pair_labels: List[str] = []
            for i in range(len(sigma_rows) - 1):
                left = sigma_rows.iloc[i]
                for j in range(i + 1, len(sigma_rows)):
                    right = sigma_rows.iloc[j]
                    denom = float(
                        np.hypot(
                            float(left["error_cmp"]),
                            float(right["error_cmp"]),
                        )
                    )
                    if denom <= 0.0 or not np.isfinite(denom):
                        continue
                    score = (
                        abs(float(left["value_cmp"]) - float(right["value_cmp"]))
                        / denom
                    )
                    pair_scores.append(float(score))
                    pair_labels.append(f"{left[source_col]} vs {right[source_col]}")
            if pair_scores:
                max_idx = int(np.argmax(pair_scores))
                sigma_tension_max = float(pair_scores[max_idx])
                sigma_tension_mean = float(np.mean(pair_scores))
                worst_provider_pair = pair_labels[max_idx]
                agreement_class = _agreement_class(sigma_tension_max)
            sigma_vals = sigma_rows["value_cmp"].to_numpy(dtype=float)
            sigma_errs = sigma_rows["error_cmp"].to_numpy(dtype=float)
            if (
                len(sigma_vals) >= 2
                and np.all(np.isfinite(sigma_errs))
                and np.all(sigma_errs > 0.0)
            ):
                weights = 1.0 / np.square(sigma_errs)
                wmean = float(np.average(sigma_vals, weights=weights))
                chi2 = float(np.sum(np.square((sigma_vals - wmean) / sigma_errs)))
                reduced_chi2 = chi2 / float(max(len(sigma_vals) - 1, 1))
        rows.append(
            {
                "pulsar": pulsar,
                "param": param,
                "source_params": ",".join(source_params),
                "n_providers": len(providers),
                "providers": ",".join(providers),
                "n_numeric_values": int(len(finite_vals)),
                "n_error_values": int(
                    np.isfinite(errs.to_numpy(dtype=float, na_value=np.nan)).sum()
                ),
                "min_value": vmin,
                "max_value": vmax,
                "span": span,
                "mean_error": float(errs.mean()) if errs.notna().any() else np.nan,
                "worst_provider_pair": worst_provider_pair,
                "sigma_tension_max": sigma_tension_max,
                "sigma_tension_mean": sigma_tension_mean,
                "reduced_chi2": reduced_chi2,
                "agreement_class": agreement_class,
            }
        )
    return pd.DataFrame(rows)


def _build_pairwise_comparison(df: pd.DataFrame) -> pd.DataFrame:
    source_col = "source_label" if "source_label" in df.columns else "provider"
    group_param_col = "compare_param" if "compare_param" in df.columns else "param"
    error_col = "error_num" if "error_num" in df.columns else "error_raw"
    columns = [
        "pulsar",
        "param",
        "source_params",
        "left_source",
        "right_source",
        "left_provider",
        "right_provider",
        "left_variant",
        "right_variant",
        "left_kind",
        "right_kind",
        "left_value",
        "right_value",
        "left_error",
        "right_error",
        "abs_delta",
        "sigma_tension",
        "agreement_class",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, object]] = []
    for (pulsar, param), sub in df.groupby(["pulsar", group_param_col], dropna=False):
        vals = _comparison_value_series(sub["value_num"], str(param))
        errs = pd.to_numeric(sub[error_col], errors="coerce")
        source_params = ",".join(sorted(set(sub["param"].astype(str))))
        work = sub.copy()
        work["_value_cmp"] = vals
        work["_error_cmp"] = errs
        for i in range(len(work) - 1):
            left = work.iloc[i]
            for j in range(i + 1, len(work)):
                right = work.iloc[j]
                lv = _safe_float(str(left["_value_cmp"]))
                rv = _safe_float(str(right["_value_cmp"]))
                if (
                    lv is None
                    or rv is None
                    or not np.isfinite(lv)
                    or not np.isfinite(rv)
                ):
                    continue
                le = _safe_float(str(left["_error_cmp"]))
                re_ = _safe_float(str(right["_error_cmp"]))
                sigma = np.nan
                if (
                    le is not None
                    and re_ is not None
                    and np.isfinite(le)
                    and np.isfinite(re_)
                    and float(le) > 0.0
                    and float(re_) > 0.0
                ):
                    denom = float(np.hypot(float(le), float(re_)))
                    if denom > 0.0 and np.isfinite(denom):
                        sigma = abs(float(lv) - float(rv)) / denom
                rows.append(
                    {
                        "pulsar": pulsar,
                        "param": param,
                        "source_params": source_params,
                        "left_source": str(left[source_col]),
                        "right_source": str(right[source_col]),
                        "left_provider": str(left.get("provider", "")),
                        "right_provider": str(right.get("provider", "")),
                        "left_variant": str(left.get("variant", "")),
                        "right_variant": str(right.get("variant", "")),
                        "left_kind": str(left.get("source_kind", "public")),
                        "right_kind": str(right.get("source_kind", "public")),
                        "left_value": float(lv),
                        "right_value": float(rv),
                        "left_error": (
                            float(le) if le is not None and np.isfinite(le) else np.nan
                        ),
                        "right_error": (
                            float(re_)
                            if re_ is not None and np.isfinite(re_)
                            else np.nan
                        ),
                        "abs_delta": abs(float(lv) - float(rv)),
                        "sigma_tension": float(sigma) if np.isfinite(sigma) else np.nan,
                        "agreement_class": (
                            _agreement_class(float(sigma))
                            if np.isfinite(sigma)
                            else "no_error_metric"
                        ),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _build_local_vs_public_table(pair_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "pulsar",
        "param",
        "source_params",
        "local_source",
        "local_variant",
        "public_source",
        "public_provider",
        "local_value",
        "public_value",
        "local_error",
        "public_error",
        "abs_delta",
        "sigma_tension",
        "agreement_class",
    ]
    if pair_df.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, object]] = []
    for row in pair_df.itertuples(index=False):
        lk = str(getattr(row, "left_kind"))
        rk = str(getattr(row, "right_kind"))
        if {lk, rk} != {"local", "public"}:
            continue
        if lk == "local":
            local_side = "left"
            public_side = "right"
        else:
            local_side = "right"
            public_side = "left"
        rows.append(
            {
                "pulsar": getattr(row, "pulsar"),
                "param": getattr(row, "param"),
                "source_params": getattr(row, "source_params"),
                "local_source": getattr(row, f"{local_side}_source"),
                "local_variant": getattr(row, f"{local_side}_variant"),
                "public_source": getattr(row, f"{public_side}_source"),
                "public_provider": getattr(row, f"{public_side}_provider"),
                "local_value": getattr(row, f"{local_side}_value"),
                "public_value": getattr(row, f"{public_side}_value"),
                "local_error": getattr(row, f"{local_side}_error"),
                "public_error": getattr(row, f"{public_side}_error"),
                "abs_delta": getattr(row, "abs_delta"),
                "sigma_tension": getattr(row, "sigma_tension"),
                "agreement_class": getattr(row, "agreement_class"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_local_vs_public_summary(local_pair_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "pulsar",
        "local_variant",
        "public_provider",
        "n_params_compared",
        "n_params_with_sigma_metric",
        "worst_param",
        "worst_sigma_tension",
        "median_sigma_tension",
        "n_strong_tension",
    ]
    if local_pair_df.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, object]] = []
    for (pulsar, local_variant, public_provider), sub in local_pair_df.groupby(
        ["pulsar", "local_variant", "public_provider"], dropna=False
    ):
        sigma = pd.to_numeric(sub["sigma_tension"], errors="coerce")
        sigma_sub = sub[
            np.isfinite(sigma.to_numpy(dtype=float, na_value=np.nan))
        ].copy()
        if len(sigma_sub):
            sigma_sub = sigma_sub.assign(_sigma=sigma.loc[sigma_sub.index]).sort_values(
                "_sigma", ascending=False, kind="stable"
            )
            worst = sigma_sub.iloc[0]
            worst_param = str(worst["param"])
            worst_sigma = float(worst["_sigma"])
            median_sigma = float(
                pd.to_numeric(sigma_sub["_sigma"], errors="coerce").median()
            )
        else:
            worst_param = ""
            worst_sigma = np.nan
            median_sigma = np.nan
        rows.append(
            {
                "pulsar": pulsar,
                "local_variant": local_variant,
                "public_provider": public_provider,
                "n_params_compared": int(len(sub)),
                "n_params_with_sigma_metric": int(len(sigma_sub)),
                "worst_param": worst_param,
                "worst_sigma_tension": worst_sigma,
                "median_sigma_tension": median_sigma,
                "n_strong_tension": int(
                    (sub["agreement_class"] == "strong_tension").sum()
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _agreement_class(sigma_tension_max: float) -> str:
    if not np.isfinite(sigma_tension_max):
        return "no_error_metric"
    if sigma_tension_max <= 1.0:
        return "consistent"
    if sigma_tension_max <= 3.0:
        return "mild_tension"
    if sigma_tension_max <= 5.0:
        return "moderate_tension"
    return "strong_tension"


def _build_pulsar_summary(cmp_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "pulsar",
        "n_params_compared",
        "n_params_with_sigma_metric",
        "worst_param",
        "worst_sigma_tension",
        "median_sigma_tension",
        "mean_reduced_chi2",
        "n_consistent",
        "n_mild_tension",
        "n_moderate_tension",
        "n_strong_tension",
    ]
    if cmp_df.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, object]] = []
    for pulsar, sub in cmp_df.groupby("pulsar", dropna=False):
        sigma_sub = sub[
            np.isfinite(
                pd.to_numeric(sub["sigma_tension_max"], errors="coerce").to_numpy(
                    dtype=float, na_value=np.nan
                )
            )
        ].copy()
        if len(sigma_sub):
            sigma_sub = sigma_sub.sort_values(
                by="sigma_tension_max", ascending=False, kind="stable"
            )
            worst = sigma_sub.iloc[0]
            worst_param = str(worst["param"])
            worst_sigma = float(worst["sigma_tension_max"])
            median_sigma = float(
                pd.to_numeric(sigma_sub["sigma_tension_max"], errors="coerce").median()
            )
            mean_reduced_chi2 = float(
                pd.to_numeric(sigma_sub["reduced_chi2"], errors="coerce").mean()
            )
        else:
            worst_param = ""
            worst_sigma = np.nan
            median_sigma = np.nan
            mean_reduced_chi2 = np.nan
        rows.append(
            {
                "pulsar": pulsar,
                "n_params_compared": int(len(sub)),
                "n_params_with_sigma_metric": int(len(sigma_sub)),
                "worst_param": worst_param,
                "worst_sigma_tension": worst_sigma,
                "median_sigma_tension": median_sigma,
                "mean_reduced_chi2": mean_reduced_chi2,
                "n_consistent": int((sub["agreement_class"] == "consistent").sum()),
                "n_mild_tension": int((sub["agreement_class"] == "mild_tension").sum()),
                "n_moderate_tension": int(
                    (sub["agreement_class"] == "moderate_tension").sum()
                ),
                "n_strong_tension": int(
                    (sub["agreement_class"] == "strong_tension").sum()
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _write_comparison_report(
    out_path: Path,
    cmp_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    local_pair_df: Optional[pd.DataFrame] = None,
    local_summary_df: Optional[pd.DataFrame] = None,
) -> Path:
    lines = [
        "# Public Release Parameter Agreement",
        "",
        "Figure of merit: `sigma_tension_max = max(|x_i - x_j| / sqrt(sigma_i^2 + sigma_j^2))` across provider pairs.",
        "Sky-position comparisons are normalized onto shared ICRS coordinates: `RAJ/ELONG -> RA_ICRS_DEG`, `DECJ/ELAT -> DEC_ICRS_DEG`.",
        "",
        "Interpretation:",
        "- `consistent`: <= 1 sigma",
        "- `mild_tension`: > 1 and <= 3 sigma",
        "- `moderate_tension`: > 3 and <= 5 sigma",
        "- `strong_tension`: > 5 sigma",
        "- `no_error_metric`: quoted uncertainties are missing or not safely comparable",
        "",
    ]
    sigma_df = cmp_df[
        np.isfinite(
            pd.to_numeric(cmp_df.get("sigma_tension_max"), errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
        )
    ].copy()
    if len(sigma_df):
        sigma_df = sigma_df.sort_values(
            by="sigma_tension_max", ascending=False, kind="stable"
        )
        lines.extend(
            [
                "## Worst Disagreements Overall",
                "",
                "| pulsar | param | source_params | sigma_tension_max | reduced_chi2 | worst_provider_pair | agreement_class |",
                "| --- | --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        for row in sigma_df.head(25).itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(getattr(row, "pulsar")),
                        str(getattr(row, "param")),
                        str(getattr(row, "source_params") or ""),
                        f'{float(getattr(row, "sigma_tension_max")):.3f}',
                        (
                            f'{float(getattr(row, "reduced_chi2")):.3f}'
                            if np.isfinite(float(getattr(row, "reduced_chi2")))
                            else "nan"
                        ),
                        str(getattr(row, "worst_provider_pair") or ""),
                        str(getattr(row, "agreement_class")),
                    ]
                )
                + " |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "## Worst Disagreements Overall",
                "",
                "No uncertainty-aware comparisons were available.",
                "",
            ]
        )
    if local_pair_df is not None and len(local_pair_df):
        local_sigma = local_pair_df[
            np.isfinite(
                pd.to_numeric(local_pair_df["sigma_tension"], errors="coerce").to_numpy(
                    dtype=float, na_value=np.nan
                )
            )
        ].copy()
        if len(local_sigma):
            local_sigma = local_sigma.sort_values(
                by="sigma_tension", ascending=False, kind="stable"
            )
            lines.extend(
                [
                    "## Local Variants vs Public Releases",
                    "",
                    "| pulsar | local_variant | public_provider | param | sigma_tension | abs_delta | agreement_class |",
                    "| --- | --- | --- | --- | ---: | ---: | --- |",
                ]
            )
            for row in local_sigma.head(40).itertuples(index=False):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(getattr(row, "pulsar")),
                            str(getattr(row, "local_variant") or ""),
                            str(getattr(row, "public_provider") or ""),
                            str(getattr(row, "param") or ""),
                            (
                                f'{float(getattr(row, "sigma_tension")):.3f}'
                                if np.isfinite(float(getattr(row, "sigma_tension")))
                                else "nan"
                            ),
                            f'{float(getattr(row, "abs_delta")):.6g}',
                            str(getattr(row, "agreement_class") or ""),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    if local_summary_df is not None and len(local_summary_df):
        lines.extend(
            [
                "## Local Variant Summary",
                "",
                "| pulsar | local_variant | public_provider | n_params_compared | n_params_with_sigma_metric | worst_param | worst_sigma_tension | n_strong_tension |",
                "| --- | --- | --- | ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for row in local_summary_df.sort_values(
            by="worst_sigma_tension", ascending=False, kind="stable"
        ).itertuples(index=False):
            worst_sigma = getattr(row, "worst_sigma_tension")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(getattr(row, "pulsar")),
                        str(getattr(row, "local_variant")),
                        str(getattr(row, "public_provider")),
                        str(int(getattr(row, "n_params_compared"))),
                        str(int(getattr(row, "n_params_with_sigma_metric"))),
                        str(getattr(row, "worst_param") or ""),
                        (
                            f"{float(worst_sigma):.3f}"
                            if np.isfinite(float(worst_sigma))
                            else "nan"
                        ),
                        str(int(getattr(row, "n_strong_tension"))),
                    ]
                )
                + " |"
            )
        lines.append("")
    if len(summary_df):
        lines.extend(
            [
                "## Per-Pulsar Summary",
                "",
                "| pulsar | n_params_compared | n_params_with_sigma_metric | worst_param | worst_sigma_tension | median_sigma_tension | n_strong_tension |",
                "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in summary_df.sort_values(
            by="worst_sigma_tension", ascending=False, kind="stable"
        ).itertuples(index=False):
            worst_sigma = getattr(row, "worst_sigma_tension")
            median_sigma = getattr(row, "median_sigma_tension")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(getattr(row, "pulsar")),
                        str(int(getattr(row, "n_params_compared"))),
                        str(int(getattr(row, "n_params_with_sigma_metric"))),
                        str(getattr(row, "worst_param") or ""),
                        (
                            f"{float(worst_sigma):.3f}"
                            if np.isfinite(float(worst_sigma))
                            else "nan"
                        ),
                        (
                            f"{float(median_sigma):.3f}"
                            if np.isfinite(float(median_sigma))
                            else "nan"
                        ),
                        str(int(getattr(row, "n_strong_tension"))),
                    ]
                )
                + " |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def compare_public_releases(
    out_dir: Path,
    providers_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    local_dataset_root: Optional[Path] = None,
    local_branch: Optional[str] = None,
    local_pulsars: Optional[Sequence[str] | str] = None,
    alias_mapping_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Download latest public releases and compare pulsar parameter values.

    Parameters
    ----------
    out_dir : pathlib.Path
        Output directory for generated TSV reports.
    providers_path : pathlib.Path, optional
        Provider catalog TOML path. If omitted, the default catalog under
        ``configs/catalogs/public_releases/providers.toml`` is used.
    cache_dir : pathlib.Path, optional
        Shared cache directory for downloaded and extracted public releases.
        When omitted, an editable ``pleb`` checkout with ``data_source`` uses
        ``<repo_root>/public_release_cache``; otherwise the cache falls back to
        ``<out_dir>/downloads``.
    local_dataset_root : pathlib.Path, optional
        Local dataset root containing per-pulsar variant ``.par`` files. When
        provided, local variants are added as synthetic comparison sources.
    local_branch : str, optional
        Git branch or commit-ish used to read local variant ``.par`` files from
        ``local_dataset_root`` without relying on the live worktree. When not
        provided, local variants are read from the filesystem.
    local_pulsars : sequence of str or str, optional
        Pulsar selection for local variant discovery. When omitted or ``"ALL"``,
        every pulsar directory under ``local_dataset_root`` is considered.
    alias_mapping_path : pathlib.Path, optional
        Ingest mapping file used to reuse PLEB's B->J canonicalization logic.

    Returns
    -------
    dict of str to pathlib.Path
        Paths to key artifacts:
        ``out_dir``, ``raw``, ``normalized``, ``comparison``, ``summary``,
        ``report``, ``assets``.

    Raises
    ------
    RuntimeError
        If no providers are configured, no assets resolve, or no parsable
        ``.par`` files are found.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    aliases = _load_alias_map(alias_mapping_path)
    if providers_path is None:
        providers_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "catalogs"
            / "public_releases"
            / "providers.toml"
        )
    providers = load_provider_specs(Path(providers_path))
    if not providers:
        raise RuntimeError(f"No providers configured in {providers_path}")

    all_assets: List[ReleaseAsset] = []
    for spec in providers:
        try:
            assets = resolve_latest_assets(spec)
        except Exception as e:
            logger.warning("Failed to resolve assets for %s: %s", spec.name, e)
            assets = []
        if not assets:
            logger.warning("No assets resolved for provider %s", spec.name)
        all_assets.extend(assets)
    if not all_assets:
        raise RuntimeError("No downloadable release assets resolved from providers.")

    cache_root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else _default_public_release_cache_dir(out_dir)
    )
    logger.info("Using public-release cache directory: %s", cache_root)
    provider_pars = _download_and_extract_assets(all_assets, cache_root)

    frames: List[pd.DataFrame] = []
    for provider, par_files in provider_pars.items():
        for p in par_files:
            try:
                d = _parse_parfile(p, aliases=aliases)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", p, e)
                continue
            if d.empty:
                continue
            d["provider"] = provider
            d["variant"] = ""
            d["source_kind"] = "public"
            d["source_label"] = provider
            frames.append(d)
    if local_dataset_root is not None:
        ds_root = Path(local_dataset_root).expanduser().resolve()
        normalized_pulsars = _normalize_local_pulsars(local_pulsars)
        branch_ref = str(local_branch or "").strip()
        if branch_ref:
            repo_root = _git_repo_root_for_path(ds_root)
            if repo_root is None:
                raise RuntimeError(
                    f"local_branch was provided but local_dataset_root is not inside a git repo: {ds_root}"
                )
            local_specs_at_ref = _discover_local_variant_parfiles_at_ref(
                repo_root,
                ds_root,
                ref=branch_ref,
                pulsars=normalized_pulsars,
                aliases=aliases,
            )
            if not local_specs_at_ref:
                logger.warning(
                    "No local variant par files were discovered under %s on branch %s",
                    ds_root,
                    branch_ref,
                )
            for pulsar, variant, rel_path in local_specs_at_ref:
                data = _git_show_file(repo_root, branch_ref, rel_path)
                if data is None:
                    logger.warning(
                        "Failed to read local variant %s from branch %s",
                        rel_path,
                        branch_ref,
                    )
                    continue
                try:
                    d = _parse_par_text(
                        data.decode("utf-8", errors="ignore"),
                        parfile_label=f"{branch_ref}:{rel_path}",
                        aliases=aliases,
                        pulsar_override=pulsar,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to parse local variant %s from branch %s: %s",
                        rel_path,
                        branch_ref,
                        e,
                    )
                    continue
                if d.empty:
                    continue
                d["provider"] = "PLEB_LOCAL"
                d["variant"] = variant
                d["source_kind"] = "local"
                d["source_label"] = f"PLEB_LOCAL[{variant}]"
                frames.append(d)
        else:
            local_specs = _discover_local_variant_parfiles(
                ds_root,
                pulsars=normalized_pulsars,
                aliases=aliases,
            )
            if not local_specs:
                logger.warning(
                    "No local variant par files were discovered under %s", ds_root
                )
            for pulsar, variant, path in local_specs:
                try:
                    d = _parse_parfile(path, aliases=aliases, pulsar_override=pulsar)
                except Exception as e:
                    logger.warning("Failed to parse local variant %s: %s", path, e)
                    continue
                if d.empty:
                    continue
                d["provider"] = "PLEB_LOCAL"
                d["variant"] = variant
                d["source_kind"] = "local"
                d["source_label"] = f"PLEB_LOCAL[{variant}]"
                frames.append(d)
    if not frames:
        raise RuntimeError("No parsable .par files found in downloaded releases.")

    raw_df = pd.concat(frames, ignore_index=True)
    norm_df = normalize_astrometry(raw_df)
    cmp_df = _build_comparison(norm_df)
    summary_df = _build_pulsar_summary(cmp_df)
    pair_df = _build_pairwise_comparison(norm_df)
    local_pair_df = _build_local_vs_public_table(pair_df)
    local_summary_df = _build_local_vs_public_summary(local_pair_df)

    raw_out = out_dir / "public_release_parameters.raw.tsv"
    norm_out = out_dir / "public_release_parameters.normalized.tsv"
    cmp_out = out_dir / "public_release_parameters.comparison.tsv"
    pair_out = out_dir / "public_release_parameters.pairwise.tsv"
    local_pair_out = out_dir / "public_release_parameters.local_vs_public.tsv"
    local_summary_out = (
        out_dir / "public_release_parameters.local_vs_public.summary.tsv"
    )
    summary_out = out_dir / "public_release_parameters.summary.tsv"
    report_out = out_dir / "public_release_parameters.report.md"
    raw_df.to_csv(raw_out, sep="\t", index=False)
    norm_df.to_csv(norm_out, sep="\t", index=False)
    cmp_df.to_csv(cmp_out, sep="\t", index=False)
    pair_df.to_csv(pair_out, sep="\t", index=False)
    local_pair_df.to_csv(local_pair_out, sep="\t", index=False)
    local_summary_df.to_csv(local_summary_out, sep="\t", index=False)
    summary_df.to_csv(summary_out, sep="\t", index=False)
    _write_comparison_report(
        report_out,
        cmp_df,
        summary_df,
        local_pair_df=local_pair_df,
        local_summary_df=local_summary_df,
    )

    assets_out = out_dir / "resolved_assets.tsv"
    pd.DataFrame([asdict(a) for a in all_assets]).to_csv(
        assets_out, sep="\t", index=False
    )
    return {
        "out_dir": out_dir,
        "cache_dir": cache_root,
        "raw": raw_out,
        "normalized": norm_out,
        "comparison": cmp_out,
        "pairwise": pair_out,
        "local_vs_public": local_pair_out,
        "local_vs_public_summary": local_summary_out,
        "summary": summary_out,
        "report": report_out,
        "assets": assets_out,
    }
