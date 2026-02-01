#!/usr/bin/env python3
"""Minimal Streamlit GUI for Pleb configs and runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple
import subprocess

import streamlit as st

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None  # type: ignore

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from tomlkit import document, table, dumps as toml_dumps

APP_TITLE = "Pleb Minimal GUI"

MAPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "sources": {"type": "array", "items": {"type": "string"}},
        "par_roots": {"type": "array", "items": {"type": "string"}},
        "template_roots": {"type": "array", "items": {"type": "string"}},
        "ignore_backends": {"type": "array", "items": {"type": "string"}},
        "pulsar_aliases": {"type": "object", "additionalProperties": {"type": "string"}},
        "backends": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "root": {"type": "string"},
                    "ignore": {"type": "boolean"},
                    "tim_glob": {"type": "string"},
                    "ignore_suffixes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["root"],
            },
        },
    },
}

WORKFLOW_SCHEMA = {
    "type": "object",
    "properties": {
        "config": {"type": "string"},
        "steps": {"type": "array"},
        "loops": {"type": "array"},
        "set": {"type": "array", "items": {"type": "string"}},
        "overrides": {"type": "object"},
    },
    "required": ["config"],
}


def _dump_toml(data: Dict[str, Any]) -> str:
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


def _detect_format(path: str | None, text: str) -> str:
    if path:
        ext = Path(path).suffix.lower()
        if ext == ".json":
            return "json"
        if ext in (".toml", ".tml"):
            return "toml"
    s = (text or "").lstrip()
    if s.startswith("{") or s.startswith("["):
        return "json"
    return "toml"


def _parse_text(text: str, fmt: str) -> Dict[str, Any]:
    if fmt == "json":
        return json.loads(text)
    return tomllib.loads(text)


def _serialize_data(data: Dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(data, indent=2)
    return _dump_toml(data)


def _load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _save_text(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _file_browser(prefix: str, label: str, start_dir: str) -> str:
    st.write(label)
    cur = st.text_input(f"{prefix}_dir", value=start_dir, key=f"{prefix}_dir")
    sel = ""
    try:
        entries = sorted(Path(cur).iterdir())
        dirs = [p for p in entries if p.is_dir()]
        files = [p for p in entries if p.is_file()]
        options = [".. (parent)"] + [f"[D] {p.name}" for p in dirs] + [p.name for p in files]
        choice = st.selectbox(f"{prefix}_pick", options, key=f"{prefix}_pick")
        if choice == ".. (parent)":
            sel = str(Path(cur).parent)
        elif choice.startswith("[D] "):
            sel = str(Path(cur) / choice[4:])
        else:
            sel = str(Path(cur) / choice)
    except Exception:
        st.info("Enter a valid directory to browse.")
    return sel


def _file_tree(prefix: str, root: Path, max_depth: int = 4) -> str:
    """Simple expandable file tree with selectable paths."""
    sel_key = f"{prefix}_selected"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = ""

    def _walk(cur: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(cur.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception:
            return
        for p in entries:
            if p.is_dir():
                with st.expander(f"{p.name}/", expanded=False):
                    if st.button(f"Select {p}", key=f"{prefix}_dir_{p}"):
                        st.session_state[sel_key] = str(p)
                    _walk(p, depth + 1)
            else:
                if st.button(p.name, key=f"{prefix}_file_{p}"):
                    st.session_state[sel_key] = str(p)

    st.write(f"Root: {root}")
    _walk(root, 0)
    return st.session_state[sel_key]


def _validate_jsonschema(data: Dict[str, Any], schema: Dict[str, Any]) -> list[str]:
    if jsonschema is None:
        return ["jsonschema not installed; schema validation skipped."]
    try:
        jsonschema.validate(instance=data, schema=schema)
        return []
    except Exception as e:
        return [str(e)]


def _validate_settings(data: Dict[str, Any]) -> list[str]:
    try:
        from pleb.config import PipelineConfig  # type: ignore

        PipelineConfig.from_dict(data)
        return []
    except Exception as e:
        return [str(e)]


def _validate_workflow(data: Dict[str, Any]) -> list[str]:
    issues = []
    if "config" not in data:
        issues.append("Missing required key: config")
    if "steps" not in data and "loops" not in data:
        issues.append("Workflow should include steps or loops.")
    return issues


def _load_gui_schema(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _render_settings_form(schema: Dict[str, Any]) -> Dict[str, Any] | None:
    fields = schema.get("fields") or []
    if not fields:
        st.info("No schema fields loaded.")
        return None
    with st.form("settings_form"):
        form_vals: Dict[str, Any] = {}
        for field in fields:
            name = field.get("name")
            ftype = field.get("type")
            default = field.get("default")
            choices = field.get("choices")
            if not name:
                continue
            label = name
            if choices:
                if ftype == "list":
                    sel = st.multiselect(label, options=choices, default=default or [])
                else:
                    idx = choices.index(default) if default in choices else 0
                    sel = st.selectbox(label, options=choices, index=idx)
                form_vals[name] = sel
                continue
            if ftype == "bool":
                form_vals[name] = st.checkbox(label, value=bool(default))
            elif ftype == "int":
                form_vals[name] = int(
                    st.number_input(
                        label, value=int(default) if default is not None else 0, step=1
                    )
                )
            elif ftype == "float":
                form_vals[name] = float(
                    st.number_input(
                        label,
                        value=float(default) if default is not None else 0.0,
                    )
                )
            elif ftype == "list":
                txt = st.text_input(label, value=json.dumps(default or []))
                try:
                    form_vals[name] = json.loads(txt)
                except Exception:
                    form_vals[name] = default or []
            else:
                form_vals[name] = st.text_input(
                    label, value="" if default is None else str(default)
                )

        out_fmt = st.selectbox("Output format", ["toml", "json"])
        submitted = st.form_submit_button("Apply to Settings Editor")
        if submitted:
            st.session_state["settings_text"] = _serialize_data(form_vals, out_fmt)
            return form_vals
    return None


def _editor_block(key_prefix: str, title: str) -> None:
    st.subheader(title)
    col1, col2 = st.columns(2)
    state_key = f"{key_prefix}_text"
    with col1:
        load_path = st.text_input(f"{key_prefix}_load_path", value="", placeholder="Path to .json/.toml")
        uploaded = st.file_uploader(
            f"{key_prefix}_upload", type=["json", "toml"], accept_multiple_files=False
        )
        if st.button(f"Load {title}", key=f"{key_prefix}_load_btn"):
            try:
                if uploaded is not None:
                    text = uploaded.read().decode("utf-8")
                    st.session_state[state_key] = text
                elif load_path:
                    st.session_state[state_key] = _load_text(load_path)
                else:
                    st.warning("Provide a path or upload a file.")
                st.rerun()
            except Exception as e:
                st.error(f"Load failed: {e}")
    with col2:
        save_path = st.text_input(f"{key_prefix}_save_path", value="", placeholder="Path to save")
        save_fmt = st.selectbox(
            f"{key_prefix}_save_fmt", ["json", "toml"], key=f"{key_prefix}_save_fmt"
        )
        if st.button(f"Save {title}", key=f"{key_prefix}_save_btn"):
            try:
                text = st.session_state.get(state_key, "")
                data = _parse_text(text, _detect_format(save_path, text))
                out = _serialize_data(data, save_fmt)
                if not save_path:
                    st.warning("Provide a save path.")
                else:
                    _save_text(save_path, out)
                    st.success(f"Saved to {save_path}")
            except Exception as e:
                st.error(f"Save failed: {e}")

    if state_key not in st.session_state:
        st.text_area(
            f"{key_prefix}_text",
            value="",
            height=360,
            key=state_key,
        )
    else:
        st.text_area(
            f"{key_prefix}_text",
            height=360,
            key=state_key,
        )

    if st.button(f"Validate {title}", key=f"{key_prefix}_validate_btn"):
        try:
            text = st.session_state.get(state_key, "")
            fmt = _detect_format(load_path, text)
            data = _parse_text(text, fmt)
            issues: list[str] = []
            if key_prefix == "ingest":
                issues = _validate_jsonschema(data, MAPPING_SCHEMA)
            elif key_prefix == "workflow":
                issues = _validate_jsonschema(data, WORKFLOW_SCHEMA)
                issues += _validate_workflow(data)
            elif key_prefix == "settings":
                issues = _validate_settings(data)
            if issues:
                st.error("Validation failed:")
                for msg in issues:
                    st.write(f"- {msg}")
            else:
                st.success("Validation passed.")
        except Exception as e:
            st.error(f"Validation failed: {e}")


def _path_picker(prefix: str, label: str, start_dir: str, target_key: str) -> None:
    st.markdown(f"**{label}**")
    base = st.text_input(f"{prefix}_base", value=start_dir, key=f"{prefix}_base")
    chosen = _file_tree(f"{prefix}_tree", Path(base))
    if not chosen:
        chosen = _file_browser(f"{prefix}_browser", "Browse filesystem", base)
    if chosen:
        st.caption(f"Selected: {chosen}")
        cur = st.session_state.get(target_key, "")
        pending_key = f"{target_key}__pending"
        if chosen != cur and chosen != st.session_state.get(pending_key, ""):
            st.session_state[pending_key] = chosen
            sel_key = f"{prefix}_selected"
            st.session_state[sel_key] = ""
            st.rerun()
    if st.button("Use selected path", key=f"{prefix}_use"):
        if chosen:
            st.session_state[f"{target_key}__pending"] = chosen
            st.rerun()


def _ingest_form() -> Dict[str, Any]:
    st.markdown("### Ingest Mapping Builder")
    data: Dict[str, Any] = {}

    def list_paths(section_key: str, label: str, hint: str) -> list[str]:
        st.markdown(f"**{label}** — {hint}")
        items_key = f"ingest_{section_key}_items"
        if items_key not in st.session_state:
            st.session_state[items_key] = []
        rows = st.session_state[items_key]

        for i, _ in enumerate(list(rows)):
            left, right = st.columns([2, 6])
            with left:
                st.write(f"{label}[{i}]")
                if st.button("Remove", key=f"{items_key}_{i}_rm"):
                    rows.pop(i)
                    st.rerun()
            with right:
                path_key = f"{items_key}_{i}_path"
                pending_key = f"{path_key}__pending"
                if pending_key in st.session_state:
                    st.session_state[path_key] = st.session_state.pop(pending_key)
                if path_key not in st.session_state:
                    st.session_state[path_key] = rows[i]
                st.text_input("Path", key=path_key)
                _path_picker(
                    f"{items_key}_{i}",
                    "Pick path",
                    os.getcwd(),
                    path_key,
                )
                rows[i] = st.session_state.get(path_key, rows[i])

        if st.button(f"Add {label}", key=f"{items_key}_add"):
            rows.append("")
            st.rerun()
        return [r for r in rows if str(r).strip()]

    data["sources"] = list_paths("sources", "sources", "Root folders to scan")
    data["par_roots"] = list_paths("par_roots", "par_roots", "PAR file roots (optional)")
    data["template_roots"] = list_paths("template_roots", "template_roots", "Template roots (optional)")

    st.markdown("**ignore_backends** — backend names to ignore")
    ignore_key = "ingest_ignore_backends"
    if ignore_key not in st.session_state:
        st.session_state[ignore_key] = ""
    st.text_input("Comma-separated backend names", key=ignore_key)
    data["ignore_backends"] = [
        s.strip()
        for s in str(st.session_state.get(ignore_key, "")).split(",")
        if s.strip()
    ]

    st.markdown("**pulsar_aliases** — mapping like B1234+56 -> J1234+5678")
    alias_key = "ingest_aliases"
    if alias_key not in st.session_state:
        st.session_state[alias_key] = ""
    st.text_area("One mapping per line: Bxxxx+yy=Jxxxx+yyyy", key=alias_key, height=120)
    aliases: Dict[str, str] = {}
    for line in str(st.session_state.get(alias_key, "")).splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            aliases[k] = v
    data["pulsar_aliases"] = aliases

    st.markdown("**backends** — backend name (TEL.BACKEND.CENFREQ) + path to tim files")
    back_key = "ingest_backends_rows"
    if back_key not in st.session_state:
        st.session_state[back_key] = []
    back_rows = st.session_state[back_key]
    for i, _ in enumerate(list(back_rows)):
        left, right = st.columns([3, 7])
        with left:
            name_key = f"{back_key}_{i}_name"
            if name_key not in st.session_state:
                st.session_state[name_key] = back_rows[i].get("name", "")
            st.text_input("backend name", key=name_key)
            if st.button("Remove backend", key=f"{back_key}_{i}_rm"):
                back_rows.pop(i)
                st.rerun()
        with right:
            root_key = f"{back_key}_{i}_root"
            pending_key = f"{root_key}__pending"
            if pending_key in st.session_state:
                st.session_state[root_key] = st.session_state.pop(pending_key)
            if root_key not in st.session_state:
                st.session_state[root_key] = back_rows[i].get("root", "")
            st.text_input("root path", key=root_key)
            source_options = [
                s for s in (st.session_state.get("ingest_sources_items", []) or []) if str(s).strip()
            ]
            if source_options:
                base_choice = st.selectbox(
                    "Select source base",
                    options=source_options,
                    key=f"{back_key}_{i}_source_select",
                )
            else:
                base_choice = os.getcwd()
            _path_picker(
                f"{back_key}_{i}",
                "Pick tim root",
                str(base_choice),
                root_key,
            )
            ignore_key = f"{back_key}_{i}_ignore"
            if ignore_key not in st.session_state:
                st.session_state[ignore_key] = bool(back_rows[i].get("ignore", False))
            st.checkbox("ignore backend", key=ignore_key)
            glob_key = f"{back_key}_{i}_glob"
            if glob_key not in st.session_state:
                st.session_state[glob_key] = back_rows[i].get("tim_glob", "*.tim")
            st.text_input("tim_glob", key=glob_key)
            suf_key = f"{back_key}_{i}_suf"
            if suf_key not in st.session_state:
                st.session_state[suf_key] = ",".join(back_rows[i].get("ignore_suffixes", ["_all.tim"]))
            st.text_input("ignore_suffixes (comma)", key=suf_key)
            back_rows[i] = {
                "name": st.session_state[name_key],
                "root": st.session_state[root_key],
                "ignore": bool(st.session_state[ignore_key]),
                "tim_glob": st.session_state[glob_key],
                "ignore_suffixes": [s.strip() for s in st.session_state[suf_key].split(",") if s.strip()],
            }
    if st.button("Add backend mapping", key="backend_add"):
        back_rows.append(
            {"name": "", "root": "", "ignore": False, "tim_glob": "*.tim", "ignore_suffixes": ["_all.tim"]}
        )
        st.rerun()

    backends: Dict[str, Any] = {}
    for b in back_rows:
        name = (b.get("name") or "").strip()
        root = (b.get("root") or "").strip()
        if not name or not root:
            continue
        backends[name] = {
            "root": root,
            "ignore": bool(b.get("ignore", False)),
            "tim_glob": b.get("tim_glob") or "*.tim",
            "ignore_suffixes": b.get("ignore_suffixes") or ["_all.tim"],
        }
    data["backends"] = backends
    return data


def _ensure_list_state(key: str) -> None:
    if key not in st.session_state:
        st.session_state[key] = []


def _list_editor(prefix: str, label: str) -> None:
    _ensure_list_state(prefix)
    st.markdown(f"**{label}**")
    col_add, col_btn = st.columns([3, 1])
    with col_add:
        new_val = st.text_input(f"{prefix}_new", value="", placeholder="Path or value")
    with col_btn:
        if st.button("Add", key=f"{prefix}_add"):
            if new_val.strip():
                st.session_state[prefix].append(new_val.strip())
                st.rerun()
    if st.session_state[prefix]:
        for i, val in enumerate(list(st.session_state[prefix])):
            c1, c2 = st.columns([8, 1])
            c1.text_input(f"{prefix}_{i}", value=val, key=f"{prefix}_{i}_val")
            if c2.button("Remove", key=f"{prefix}_{i}_rm"):
                st.session_state[prefix].pop(i)
                st.rerun()


def _backend_editor() -> None:
    if "ingest_backends" not in st.session_state:
        st.session_state["ingest_backends"] = []

    st.markdown("**Backends (TEL.BACKEND.CENFREQ)**")
    if st.button("Add backend", key="ingest_backend_add"):
        st.session_state["ingest_backends"].append(
            {
                "name": "",
                "root": "",
                "ignore": False,
                "tim_glob": "*.tim",
                "ignore_suffixes": ["_all.tim"],
            }
        )
        st.rerun()

    for i, b in enumerate(list(st.session_state["ingest_backends"])):
        with st.expander(f"Backend {i + 1}"):
            b["name"] = st.text_input(f"backend_{i}_name", value=b.get("name", ""))
            b["root"] = st.text_input(f"backend_{i}_root", value=b.get("root", ""))
            b["ignore"] = st.checkbox(f"backend_{i}_ignore", value=bool(b.get("ignore", False)))
            b["tim_glob"] = st.text_input(
                f"backend_{i}_glob", value=b.get("tim_glob", "*.tim")
            )
            suffix_txt = st.text_input(
                f"backend_{i}_ignore_suffixes",
                value=",".join(b.get("ignore_suffixes", ["_all.tim"])),
                help="Comma-separated suffixes to ignore",
            )
            b["ignore_suffixes"] = [s.strip() for s in suffix_txt.split(",") if s.strip()]
            if st.button("Remove backend", key=f"backend_{i}_rm"):
                st.session_state["ingest_backends"].pop(i)
                st.rerun()


def _apply_ingest_builder_to_text() -> None:
    data = {}
    data["sources"] = st.session_state.get("ingest_sources", [])
    data["par_roots"] = st.session_state.get("ingest_par_roots", [])
    data["template_roots"] = st.session_state.get("ingest_template_roots", [])
    backends = {}
    for b in st.session_state.get("ingest_backends", []):
        name = (b.get("name") or "").strip()
        root = (b.get("root") or "").strip()
        if not name or not root:
            continue
        backends[name] = {
            "root": root,
            "ignore": bool(b.get("ignore", False)),
            "tim_glob": b.get("tim_glob") or "*.tim",
            "ignore_suffixes": b.get("ignore_suffixes") or ["_all.tim"],
        }
    data["backends"] = backends
    st.session_state["ingest_text"] = _serialize_data(data, "json")


def _run_cmd(cmd: list[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Minimal GUI for ingest/settings/workflow editing and runs.")

    tabs = st.tabs(["Ingest Mapping", "Settings Config", "Workflow Config", "Runner"])

    with tabs[0]:
        _editor_block("ingest", "Ingest Mapping")
        ingest_data = _ingest_form()
        if st.button("Update ingest text view from fields"):
            st.session_state["ingest_text"] = _serialize_data(ingest_data, "json")
            st.rerun()

    with tabs[1]:
        st.markdown("### Settings Form (schema-driven)")
        schema_path = st.text_input(
            "Settings schema path",
            value="configs/settings/gui_schema.json",
        )
        schema = _load_gui_schema(schema_path)
        _render_settings_form(schema)
        _editor_block("settings", "Settings Config")

    with tabs[2]:
        _editor_block("workflow", "Workflow Config")

    with tabs[3]:
        st.subheader("Run configs")
        mode = st.selectbox(
            "Mode",
            ["pipeline", "param_scan", "qc_report", "ingest", "workflow"],
        )
        cmd = ["pleb"]
        if mode == "pipeline":
            config = st.text_input("Pipeline config path", "")
            extras = st.text_area("Extra CLI args", "--qc\n", height=120)
            if st.button("Run pipeline"):
                if not config:
                    st.warning("Provide a config path.")
                else:
                    cmd = ["pleb", "--config", config]
                    if extras.strip():
                        cmd += extras.split()
                    code, out, err = _run_cmd(cmd)
                    st.code(" ".join(cmd))
                    st.text_area("stdout", out, height=200)
                    st.text_area("stderr", err, height=200)
                    st.write(f"Exit code: {code}")
        elif mode == "param_scan":
            config = st.text_input("Param-scan config path", "")
            extras = st.text_area("Extra CLI args", "--param-scan --scan-typical\n", height=120)
            if st.button("Run param-scan"):
                if not config:
                    st.warning("Provide a config path.")
                else:
                    cmd = ["pleb", "--config", config]
                    if extras.strip():
                        cmd += extras.split()
                    code, out, err = _run_cmd(cmd)
                    st.code(" ".join(cmd))
                    st.text_area("stdout", out, height=200)
                    st.text_area("stderr", err, height=200)
                    st.write(f"Exit code: {code}")
        elif mode == "qc_report":
            run_dir = st.text_input("Run directory", "")
            if st.button("Run qc-report"):
                if not run_dir:
                    st.warning("Provide a run directory.")
                else:
                    cmd = ["pleb", "qc-report", "--run-dir", run_dir]
                    code, out, err = _run_cmd(cmd)
                    st.code(" ".join(cmd))
                    st.text_area("stdout", out, height=200)
                    st.text_area("stderr", err, height=200)
                    st.write(f"Exit code: {code}")
        elif mode == "ingest":
            mapping = st.text_input("Ingest mapping path", "")
            outdir = st.text_input("Output directory", "")
            if st.button("Run ingest"):
                if not mapping or not outdir:
                    st.warning("Provide mapping and output directory.")
                else:
                    cmd = ["pleb", "ingest", "--mapping", mapping, "--output-dir", outdir]
                    code, out, err = _run_cmd(cmd)
                    st.code(" ".join(cmd))
                    st.text_area("stdout", out, height=200)
                    st.text_area("stderr", err, height=200)
                    st.write(f"Exit code: {code}")
        elif mode == "workflow":
            wf = st.text_input("Workflow file path", "")
            if st.button("Run workflow"):
                if not wf:
                    st.warning("Provide a workflow file.")
                else:
                    cmd = ["pleb", "workflow", "--file", wf]
                    code, out, err = _run_cmd(cmd)
                    st.code(" ".join(cmd))
                    st.text_area("stdout", out, height=200)
                    st.text_area("stderr", err, height=200)
                    st.write(f"Exit code: {code}")


if __name__ == "__main__":
    main()
