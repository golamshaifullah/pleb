from __future__ import annotations

import argparse
from pathlib import Path

import os
import sys
import json
import tempfile
from pathlib import Path

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from tomlkit import document, table, dumps as toml_dumps

from .config import PipelineConfig
from .pipeline import run_pipeline
from .param_scan import run_param_scan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Data combination diagnostics pipeline (tempo2 + plots + reports).")
    p.add_argument("--config",default=None,help="Path to config file (.json or .toml). Use '-' to read from stdin. Optional if using --set.",)
    p.add_argument("--set",dest="overrides",action="append",default=[],metavar="KEY=VALUE",help="Override/add a config key. Repeatable. Dotted keys allowed (e.g. fix.required_tim_flags.-pta=\"EPTA\").",)
    p.add_argument("--results-dir", type=Path, default=None, help="Override results_dir from config.")
    p.add_argument("--outdir-name", type=str, default=None, help="Override outdir_name from config.")
    p.add_argument("--force-rerun", action="store_true", help="Re-run tempo2 even if outputs exist.")
    p.add_argument("--no-tempo2", action="store_true", help="Skip tempo2 run; use existing outputs if present.")
    p.add_argument("--jobs", type=int, default=None, help="Number of parallel workers to run pulsars concurrently (per branch).")
    p.add_argument("--qc", action="store_true", help="Run optional pta_qc outlier detection (requires pta_qc + libstempo).")
    p.add_argument("--qc-backend-col", default=None, help="Backend grouping column for pta_qc (default from config: group).")
    p.add_argument("--qc-drop-unmatched", action="store_true", help="Drop TOAs unmatched to tim metadata in pta_qc.")

    # Param scan (fit-only): run baseline + candidate .par variants and compare via Δχ² / Wald z.
    p.add_argument("--param-scan", action="store_true", help="Run a parameter scan (fit-only) instead of the full pipeline.")
    p.add_argument("--scan-branch", type=str, default=None, help="Git branch to scan (default: config.reference_branch).")
    p.add_argument(
        "--scan",
        dest="scan_specs",
        action="append",
        default=None,
        help=(
            "Candidate specification. Repeatable. Examples: 'F2', 'F2=0', 'F2+F3', 'raw:JUMP -sys P200 0 1'. "
            "You can also pass a file with --scan-file."
        ),
    )
    p.add_argument("--scan-file", type=Path, default=None, help="Text file with one candidate spec per line.")
    p.add_argument(
        "--scan-typical",
        action="store_true",
        help=(
            "Use the built-in 'typical' scan profile: Parallax (PX); then binary derivatives based on BINARY model; "
            "if no BINARY and reduced chi-square is high, scan DM derivatives."
        ),
    )
    p.add_argument(
        "--scan-dm-threshold",
        type=float,
        default=None,
        help="Reduced chi-square threshold to trigger DM-derivative scan when no BINARY is present (default from config).",
    )
    p.add_argument(
        "--scan-dm-max-order",
        type=int,
        default=None,
        help="Max DM derivative order to test in the typical profile (DM1..DMN). Default from config.",
    )
    p.add_argument(
        "--scan-btx-max-fb",
        type=int,
        default=None,
        help="Max FB derivative order to test for BTX in the typical profile (FB1..FBN). Default from config.",
    )
    p.add_argument(
        "--scan-pulsar",
        dest="scan_pulsars",
        action="append",
        default=None,
        help="Limit param scan to one or more pulsars. Repeatable.",
    )
    p.add_argument("--scan-outdir", type=str, default=None, help="Override output directory name for the param scan run.")

    # Extras
    p.add_argument(
        "--fix-dataset",
        action="store_true",
        help="Run dataset fix/report stage (report-only unless fix_apply is enabled in config).",
    )
    p.add_argument(
        "--binary-analysis",
        action="store_true",
        help="Write a binary/orbital analysis table derived from .par files.",
    )
    return p

def _parse_value_as_toml_literal(raw: str):
    """
    Parse VALUE using TOML literal rules when possible (true/false, numbers, arrays, inline tables, quoted strings).
    Fallback: plain string.
    """
    raw = raw.strip()
    if raw == "":
        return ""
    try:
        # tomllib only parses full TOML docs; wrap as a key/value
        doc = tomllib.loads(f"v = {raw}")
        return doc["v"]
    except Exception:
        # fallback string (strip surrounding quotes if user provided them badly)
        return raw

def _set_dotted_key(d: dict, key: str, value):
    parts = [p for p in key.split(".") if p]
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def _load_config_dict(config_arg: str | None) -> dict:
    if not config_arg:
        return {}
    if config_arg == "-":
        text = sys.stdin.read()
        text = text.strip()
        if not text:
            return {}
        # Assume TOML if it doesn't look like JSON
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
    """
    TOML has no null; omit None values.
    """
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


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cfg = PipelineConfig.load(args.config)
    # Build config dict from file/stdin/empty, then apply --set overrides,
    # then load via PipelineConfig.load() using a temp TOML.
    cfg_dict = _load_config_dict(args.config)
    for item in args.overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got: {item!r}")
        k, vraw = item.split("=", 1)
        k = k.strip()
        v = _parse_value_as_toml_literal(vraw)
        _set_dotted_key(cfg_dict, k, v)

    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        tmp_path = tmp.name
        tmp.write(_dump_toml_no_nulls(cfg_dict))
        tmp.close()
        cfg = PipelineConfig.load(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
    if args.results_dir is not None:
        cfg.results_dir = args.results_dir
    if args.outdir_name is not None:
        cfg.outdir_name = args.outdir_name
    if args.force_rerun:
        cfg.force_rerun = True
    if args.no_tempo2:
        cfg.run_tempo2 = False

    if args.jobs is not None:
        cfg.jobs = int(args.jobs)
    if getattr(args, 'qc', False):
        cfg.run_pta_qc = True
    if getattr(args, 'qc_backend_col', None):
        cfg.pta_qc_backend_col = str(args.qc_backend_col)
    if getattr(args, 'qc_drop_unmatched', False):
        cfg.pta_qc_drop_unmatched = True

    if args.fix_dataset:
        cfg.run_fix_dataset = True

    if args.binary_analysis:
        cfg.make_binary_analysis = True

    if args.param_scan:
        specs: list[str] = []
        if args.scan_specs:
            specs.extend([str(s) for s in args.scan_specs if str(s).strip()])
        if args.scan_file is not None:
            if not args.scan_file.exists():
                raise FileNotFoundError(str(args.scan_file))
            for raw in args.scan_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith(("#", "C ", "c ")):
                    continue
                specs.append(line)

        if not specs and not args.scan_typical:
            raise SystemExit("--param-scan requires at least one --scan spec (or --scan-file), unless --scan-typical is used.")

        out_paths = run_param_scan(
            cfg,
            branch=args.scan_branch,
            pulsars=args.scan_pulsars,
            candidate_specs=specs,
            scan_typical=bool(args.scan_typical),
            dm_redchisq_threshold=args.scan_dm_threshold,
            dm_max_order=args.scan_dm_max_order,
            btx_max_fb=args.scan_btx_max_fb,
            outdir_name=args.scan_outdir,
        )
        print(str(out_paths["tag"]))
        return 0

    out_paths = run_pipeline(cfg)
    print(str(out_paths["tag"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
