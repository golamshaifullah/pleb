"""Shared config parsing helpers for CLI and workflow runner."""

from __future__ import annotations

from pathlib import Path
import json
import sys

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from tomlkit import document, table, dumps as toml_dumps


def _parse_value_as_toml_literal(raw: str):
    """Parse a TOML literal from a CLI override."""
    raw = raw.strip()
    if raw == "":
        return ""
    try:
        doc = tomllib.loads(f"v = {raw}")
        return doc["v"]
    except Exception:
        return raw


def _set_dotted_key(d: dict, key: str, value):
    """Set a nested value in a dict using dotted-key notation."""
    parts = [p for p in key.split(".") if p]
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _load_config_dict(config_arg: str | None) -> dict:
    """Load a raw config dictionary from a file or stdin."""
    if not config_arg:
        return {}
    if config_arg == "-":
        text = sys.stdin.read().strip()
        if not text:
            return {}
        if text.lstrip().startswith("{") or text.lstrip().startswith("["):
            return json.loads(text)
        return tomllib.loads(text)

    path = Path(config_arg).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))
    suf = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suf == ".json":
        return json.loads(text)
    if suf == ".toml":
        return tomllib.loads(text)
    raise ValueError("Config must be .toml or .json")


def _dump_toml_no_nulls(data: dict) -> str:
    """Serialize a dict to TOML, omitting None values."""

    def to_tomlkit(obj):
        if isinstance(obj, dict):
            t = table()
            for k, v in obj.items():
                if v is None:
                    continue
                t[k] = to_tomlkit(v)
            return t
        if isinstance(obj, list):
            return [to_tomlkit(x) for x in obj]
        return obj

    doc = document()
    for k in sorted(data.keys()):
        v = data[k]
        if v is None:
            continue
        doc[k] = to_tomlkit(v)
    return toml_dumps(doc)
