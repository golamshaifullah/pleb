"""Generate docs/full_settings_catalog.rst from UX key catalog."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from pleb.ux.key_catalog import all_key_specs, default_for_spec
except Exception as exc:  # pragma: no cover
    print(f"Skipping settings catalog generation: {exc}")
    raise SystemExit(0)


def _fmt_default(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, str):
        return f"``{v}``"
    return f"``{v}``"


def main() -> int:
    out = Path("docs/full_settings_catalog.rst")
    specs = all_key_specs()
    lines = [
        "Full Settings Catalog",
        "=====================",
        "",
        "This page is auto-generated from ``pleb.ux.key_catalog``.",
        "",
        ".. note::",
        "   Regenerate with ``python scripts/generate_settings_catalog.py``.",
        "",
    ]
    current_group = None
    for s in specs:
        if s.group != current_group:
            current_group = s.group
            title = s.group
            lines.extend([title, "-" * len(title), ""])
        lines.append(f"``{s.name}``")
        lines.append(f"  Type: ``{s.type_name}``")
        lines.append(f"  Default: {_fmt_default(default_for_spec(s))}")
        lines.append(f"  Modes: ``{', '.join(s.modes)}``")
        lines.append(f"  Level: ``{s.level}``")
        if s.choices:
            lines.append(f"  Choices: ``{', '.join(s.choices)}``")
        lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
