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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import re
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

from .logging_utils import get_logger

logger = get_logger("pleb.public_release_compare")


_PAR_IGNORE_PREFIXES = ("C", "#")
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
    max_records : int, optional
        Maximum number of candidate records inspected per provider.
    max_assets : int, optional
        Maximum number of downloadable assets selected from a record.
    asset_regex : str, optional
        Regular expression used to filter downloadable asset names.
    """

    name: str
    type: str = "zenodo_search"
    query: str = ""
    max_records: int = 10
    max_assets: int = 1
    asset_regex: str = r".*\.(tar\.gz|tgz|tar|zip|par)$"


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
    """

    provider: str
    release_id: str
    title: str
    url: str
    filename: str


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
                max_records=int(spec.get("max_records", 10)),
                max_assets=int(spec.get("max_assets", 1)),
                asset_regex=str(
                    spec.get("asset_regex", r".*\.(tar\.gz|tgz|tar|zip|par)$")
                ),
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
                )
            )
            matched += 1
            if matched >= max(1, spec.max_assets):
                break
        if matched > 0:
            break
    return assets


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


def _find_par_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.par") if p.is_file()], key=lambda p: str(p))


def _safe_float(token: str) -> Optional[float]:
    try:
        return float(token)
    except Exception:
        return None


def _parse_parfile(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    pulsar = path.stem
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
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
                "parfile": str(path),
            }
        )
    return pd.DataFrame(rows)


def _extract_astrometry_context(df: pd.DataFrame) -> Dict[str, str]:
    ctx: Dict[str, str] = {}
    for key in ("RAJ", "DECJ", "ELONG", "ELAT", "PMRA", "PMDEC", "PMELONG", "PMELAT"):
        sub = df[df["param"] == key]
        if len(sub):
            ctx[key] = str(sub.iloc[-1]["value_raw"])
    return ctx


def _normalize_astrometry_row(
    row: pd.Series, context: Dict[str, str]
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    key = str(row["param"]).upper()
    raw = str(row["value_raw"])
    if key == "RAJ":
        c = SkyCoord(
            ra=raw,
            dec=context.get("DECJ", "0d"),
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )
        return float(c.ra.deg), "deg", "icrs"
    if key == "DECJ":
        c = SkyCoord(
            ra=context.get("RAJ", "0h"),
            dec=raw,
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )
        return float(c.dec.deg), "deg", "icrs"
    if key == "ELONG":
        elat = context.get("ELAT")
        if not elat:
            return None, None, None
        c = SkyCoord(
            lon=float(raw) * u.deg,
            lat=float(elat) * u.deg,
            frame=BarycentricTrueEcliptic(equinox="J2000"),
        ).icrs
        return float(c.ra.deg), "deg", "icrs"
    if key == "ELAT":
        elong = context.get("ELONG")
        if not elong:
            return None, None, None
        c = SkyCoord(
            lon=float(elong) * u.deg,
            lat=float(raw) * u.deg,
            frame=BarycentricTrueEcliptic(equinox="J2000"),
        ).icrs
        return float(c.dec.deg), "deg", "icrs"
    if key in ("PMRA", "PMDEC", "PMELONG", "PMELAT", "PX"):
        v = _safe_float(raw)
        if v is None:
            return None, None, None
        return float(v), "mas/yr" if key.startswith("PM") else "mas", "icrs"
    v = _safe_float(raw)
    if v is None:
        return None, None, None
    return float(v), None, None


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
            val_num, unit, frame = _normalize_astrometry_row(s, ctx)
            out_rows.append(
                {
                    **row._asdict(),
                    "value_num": val_num,
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
    provider_to_pars: Dict[str, List[Path]] = {}
    for a in assets:
        pdir = cache_root / a.provider / a.release_id
        pdir.mkdir(parents=True, exist_ok=True)
        fname = Path(a.filename).name or "asset.bin"
        downloaded = _http_download(a.url, pdir / fname)
        extracted = _extract_archive(downloaded, pdir / "extracted")
        par_files = _find_par_files(extracted)
        provider_to_pars.setdefault(a.provider, []).extend(par_files)
    return provider_to_pars


def _build_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "pulsar",
                "param",
                "n_providers",
                "providers",
                "min_value",
                "max_value",
                "span",
                "mean_error",
            ]
        )
    grp = df.groupby(["pulsar", "param"], dropna=False)
    rows: List[Dict[str, object]] = []
    for (pulsar, param), sub in grp:
        vals = pd.to_numeric(sub["value_num"], errors="coerce").dropna()
        errs = pd.to_numeric(sub["error_raw"], errors="coerce").dropna()
        providers = sorted(set(sub["provider"].astype(str)))
        if len(vals):
            vmin = float(vals.min())
            vmax = float(vals.max())
            span = float(vmax - vmin)
        else:
            vmin = np.nan
            vmax = np.nan
            span = np.nan
        rows.append(
            {
                "pulsar": pulsar,
                "param": param,
                "n_providers": len(providers),
                "providers": ",".join(providers),
                "min_value": vmin,
                "max_value": vmax,
                "span": span,
                "mean_error": float(errs.mean()) if len(errs) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compare_public_releases(
    out_dir: Path,
    providers_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Download latest public releases and compare pulsar parameter values.

    Parameters
    ----------
    out_dir : pathlib.Path
        Output directory for downloads and generated TSV reports.
    providers_path : pathlib.Path, optional
        Provider catalog TOML path. If omitted, the default catalog under
        ``configs/catalogs/public_releases/providers.toml`` is used.

    Returns
    -------
    dict of str to pathlib.Path
        Paths to key artifacts:
        ``out_dir``, ``raw``, ``normalized``, ``comparison``, ``assets``.

    Raises
    ------
    RuntimeError
        If no providers are configured, no assets resolve, or no parsable
        ``.par`` files are found.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
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

    cache_root = out_dir / "downloads"
    provider_pars = _download_and_extract_assets(all_assets, cache_root)

    frames: List[pd.DataFrame] = []
    for provider, par_files in provider_pars.items():
        for p in par_files:
            try:
                d = _parse_parfile(p)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", p, e)
                continue
            if d.empty:
                continue
            d["provider"] = provider
            frames.append(d)
    if not frames:
        raise RuntimeError("No parsable .par files found in downloaded releases.")

    raw_df = pd.concat(frames, ignore_index=True)
    norm_df = normalize_astrometry(raw_df)
    cmp_df = _build_comparison(norm_df)

    raw_out = out_dir / "public_release_parameters.raw.tsv"
    norm_out = out_dir / "public_release_parameters.normalized.tsv"
    cmp_out = out_dir / "public_release_parameters.comparison.tsv"
    raw_df.to_csv(raw_out, sep="\t", index=False)
    norm_df.to_csv(norm_out, sep="\t", index=False)
    cmp_df.to_csv(cmp_out, sep="\t", index=False)

    assets_out = out_dir / "resolved_assets.tsv"
    pd.DataFrame([a.__dict__ for a in all_assets]).to_csv(
        assets_out, sep="\t", index=False
    )
    return {
        "out_dir": out_dir,
        "raw": raw_out,
        "normalized": norm_out,
        "comparison": cmp_out,
        "assets": assets_out,
    }
