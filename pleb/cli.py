"""Command-line interface for the data-combination pipeline.

This module wires config loading/overrides to :func:`run_pipeline` and
:func:`run_param_scan`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import os
import sys
import json
import tempfile

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from tomlkit import document, table, dumps as toml_dumps

from .config import PipelineConfig
from .pipeline import run_pipeline
from .param_scan import run_param_scan
from .qc_report import generate_qc_report


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        An :class:`argparse.ArgumentParser` configured with pipeline options.
    """
    p = argparse.ArgumentParser(description="Data combination diagnostics pipeline (tempo2 + plots + reports).")
    p.add_argument("--config",default=None,help="Path to config file (.json or .toml). Use '-' to read from stdin. Optional if using --set.",)
    p.add_argument("--set",dest="overrides",action="append",default=[],metavar="KEY=VALUE",help="Override/add a config key. Repeatable. Dotted keys allowed (e.g. fix.required_tim_flags.-pta=\"EPTA\").",)
    p.add_argument("--results-dir", type=Path, default=None, help="Override results_dir from config.")
    p.add_argument("--outdir-name", type=str, default=None, help="Override outdir_name from config.")
    p.add_argument("--force-rerun", action="store_true", help="Re-run tempo2 even if outputs exist.")
    p.add_argument("--no-tempo2", action="store_true", help="Skip tempo2 run; use existing outputs if present.")
    p.add_argument("--no-change-reports", action="store_true", help="Skip change report generation.")
    p.add_argument("--testing", action="store_true", help="Enable testing mode (skips change reports).")
    p.add_argument("--jobs", type=int, default=None, help="Number of parallel workers to run pulsars concurrently (per branch).")
    p.add_argument("--qc", action="store_true", help="Run optional pqc outlier detection (requires pqc + libstempo).")
    p.add_argument("--qc-backend-col", default=None, help="Backend grouping column for pqc (default from config: group).")
    p.add_argument("--qc-drop-unmatched", action="store_true", help="Drop TOAs unmatched to tim metadata in pqc.")
    p.add_argument("--qc-add-orbital-phase", action="store_true", help="Enable pqc orbital-phase feature extraction.")
    p.add_argument("--qc-no-orbital-phase", action="store_true", help="Disable pqc orbital-phase feature extraction.")
    p.add_argument("--qc-add-solar-elongation", action="store_true", help="Enable pqc solar-elongation feature extraction.")
    p.add_argument("--qc-no-solar-elongation", action="store_true", help="Disable pqc solar-elongation feature extraction.")
    p.add_argument("--qc-add-elevation", action="store_true", help="Enable pqc elevation feature extraction.")
    p.add_argument("--qc-add-airmass", action="store_true", help="Enable pqc airmass feature extraction.")
    p.add_argument("--qc-add-parallactic-angle", action="store_true", help="Enable pqc parallactic-angle feature extraction.")
    p.add_argument("--qc-add-freq-bin", action="store_true", help="Enable pqc frequency-bin feature extraction.")
    p.add_argument("--qc-freq-bins", type=int, default=None, help="Number of pqc frequency bins if enabled.")
    p.add_argument("--qc-observatory-path", type=Path, default=None, help="Observatory file path for pqc alt/az features.")
    p.add_argument("--qc-structure-mode", default=None, help="pqc structure mode: none/detrend/test/both.")
    p.add_argument("--qc-structure-detrend-features", default=None, help="Comma-separated feature columns to detrend against.")
    p.add_argument("--qc-structure-test-features", default=None, help="Comma-separated feature columns to test for structure.")
    p.add_argument("--qc-structure-circular-features", default=None, help="Comma-separated circular feature columns.")
    p.add_argument("--qc-structure-nbins", type=int, default=None, help="Number of bins for pqc structure tests.")
    p.add_argument("--qc-structure-min-per-bin", type=int, default=None, help="Minimum points per bin for structure tests.")
    p.add_argument("--qc-structure-p-thresh", type=float, default=None, help="p-value threshold for structure detection.")
    p.add_argument("--qc-structure-group-cols", default=None, help="Comma-separated grouping columns for structure tests.")
    p.add_argument("--qc-outlier-gate", action="store_true", help="Enable hard sigma gate for pqc outliers.")
    p.add_argument("--qc-outlier-gate-sigma", type=float, default=None, help="Sigma threshold for pqc outlier gate.")
    p.add_argument("--qc-outlier-gate-resid-col", default=None, help="Residual column for pqc outlier gate.")
    p.add_argument("--qc-outlier-gate-sigma-col", default=None, help="Sigma column for pqc outlier gate.")
    p.add_argument("--qc-event-instrument", action="store_true", help="Enable per-event membership diagnostics for pqc.")

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


def build_qc_report_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for qc-report."""
    p = argparse.ArgumentParser(description="Generate PQC report summaries and plots from a run directory.")
    p.add_argument("qc_report", nargs="?", help=argparse.SUPPRESS)
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory containing qc outputs.")
    p.add_argument("--backend-col", default="group", help="Backend column name (default: group).")
    p.add_argument("--backend", default=None, help="Optional: filter plots to a single backend key.")
    p.add_argument("--structure-group-cols", default=None, help="Comma-separated group columns; use ; for multiple groupings.")
    p.add_argument("--no-feature-plots", action="store_true", help="Skip feature (e.g., orbital phase/solar) plots.")
    p.add_argument("--report-dir", type=Path, default=None, help="Output directory for report artifacts (default: <run-dir>/qc_report).")
    p.add_argument("--no-plots", action="store_true", help="Skip transient plot generation.")
    return p


def run_qc_report(argv: list[str] | None) -> int:
    """Generate QC diagnostics and plots for all pqc CSVs in a run directory."""
    args = build_qc_report_parser().parse_args(argv)
    report_dir = generate_qc_report(
        run_dir=Path(args.run_dir),
        backend_col=str(args.backend_col),
        backend=(str(args.backend) if args.backend is not None else None),
        report_dir=(Path(args.report_dir) if args.report_dir else None),
        no_plots=bool(args.no_plots),
        structure_group_cols=(str(args.structure_group_cols) if args.structure_group_cols else None),
        no_feature_plots=bool(args.no_feature_plots),
    )
    print(str(report_dir))
    return 0

def _parse_value_as_toml_literal(raw: str):
    """Parse a TOML literal from a CLI override.

    Args:
        raw: Raw string value from ``--set KEY=VALUE``.

    Returns:
        Parsed TOML value when possible; otherwise the raw string.
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


def _parse_csv_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    items = [p.strip() for p in str(raw).split(",")]
    items = [p for p in items if p]
    return items or None

def _set_dotted_key(d: dict, key: str, value):
    """Set a nested value in a dict using dotted-key notation.

    Args:
        d: Dictionary to update in-place.
        key: Dotted key path (e.g., ``"fix.required_tim_flags.-pta"``).
        value: Value to set.
    """
    parts = [p for p in key.split(".") if p]
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def _load_config_dict(config_arg: str | None) -> dict:
    """Load a raw config dictionary from a file or stdin.

    Args:
        config_arg: Path to a config file or "-" to read from stdin.

    Returns:
        The raw configuration dictionary (top-level keys).

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        ValueError: If the file extension is unsupported.
    """
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
    """Serialize a dict to TOML, omitting ``None`` values.

    Args:
        data: Data to serialize.

    Returns:
        TOML string with ``None`` values omitted.
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
    """Run the CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "qc-report":
        return run_qc_report(argv)
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
    if args.no_change_reports:
        cfg.make_change_reports = False
    if args.testing:
        cfg.testing_mode = True

    if args.jobs is not None:
        cfg.jobs = int(args.jobs)
    if getattr(args, 'qc', False):
        cfg.run_pqc = True
    if getattr(args, 'qc_backend_col', None):
        cfg.pqc_backend_col = str(args.qc_backend_col)
    if getattr(args, 'qc_drop_unmatched', False):
        cfg.pqc_drop_unmatched = True
    if getattr(args, 'qc_add_orbital_phase', False):
        cfg.pqc_add_orbital_phase = True
    if getattr(args, 'qc_no_orbital_phase', False):
        cfg.pqc_add_orbital_phase = False
    if getattr(args, 'qc_add_solar_elongation', False):
        cfg.pqc_add_solar_elongation = True
    if getattr(args, 'qc_no_solar_elongation', False):
        cfg.pqc_add_solar_elongation = False
    if getattr(args, 'qc_add_elevation', False):
        cfg.pqc_add_elevation = True
    if getattr(args, 'qc_add_airmass', False):
        cfg.pqc_add_airmass = True
    if getattr(args, 'qc_add_parallactic_angle', False):
        cfg.pqc_add_parallactic_angle = True
    if getattr(args, 'qc_add_freq_bin', False):
        cfg.pqc_add_freq_bin = True
    if getattr(args, 'qc_freq_bins', None) is not None:
        cfg.pqc_freq_bins = int(args.qc_freq_bins)
    if getattr(args, 'qc_observatory_path', None) is not None:
        cfg.pqc_observatory_path = str(args.qc_observatory_path)
    if getattr(args, 'qc_structure_mode', None):
        cfg.pqc_structure_mode = str(args.qc_structure_mode)
    if getattr(args, 'qc_structure_detrend_features', None):
        cfg.pqc_structure_detrend_features = _parse_csv_list(args.qc_structure_detrend_features)
    if getattr(args, 'qc_structure_test_features', None):
        cfg.pqc_structure_test_features = _parse_csv_list(args.qc_structure_test_features)
    if getattr(args, 'qc_structure_circular_features', None):
        cfg.pqc_structure_circular_features = _parse_csv_list(args.qc_structure_circular_features)
    if getattr(args, 'qc_structure_nbins', None) is not None:
        cfg.pqc_structure_nbins = int(args.qc_structure_nbins)
    if getattr(args, 'qc_structure_min_per_bin', None) is not None:
        cfg.pqc_structure_min_per_bin = int(args.qc_structure_min_per_bin)
    if getattr(args, 'qc_structure_p_thresh', None) is not None:
        cfg.pqc_structure_p_thresh = float(args.qc_structure_p_thresh)
    if getattr(args, 'qc_structure_group_cols', None):
        cfg.pqc_structure_group_cols = _parse_csv_list(args.qc_structure_group_cols)
    if getattr(args, 'qc_outlier_gate', False):
        cfg.pqc_outlier_gate_enabled = True
    if getattr(args, 'qc_outlier_gate_sigma', None) is not None:
        cfg.pqc_outlier_gate_sigma = float(args.qc_outlier_gate_sigma)
    if getattr(args, 'qc_outlier_gate_resid_col', None):
        cfg.pqc_outlier_gate_resid_col = str(args.qc_outlier_gate_resid_col)
    if getattr(args, 'qc_outlier_gate_sigma_col', None):
        cfg.pqc_outlier_gate_sigma_col = str(args.qc_outlier_gate_sigma_col)
    if getattr(args, 'qc_event_instrument', False):
        cfg.pqc_event_instrument = True

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
