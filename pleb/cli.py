"""Provide the command-line interface for the data-combination pipeline.

This module wires config loading/overrides to :func:`pleb.pipeline.run_pipeline`
and :func:`pleb.param_scan.run_param_scan`, including convenience flags for
parameter scans and PQC reporting.

Examples:
    Run the full pipeline from a JSON config::

        python -m pleb.cli --config pipeline.json

    Run a parameter scan with a typical profile::

        python -m pleb.cli --config pipeline.toml --param-scan --scan-typical

    Generate a PQC report from a run directory::

        python -m pleb.cli qc-report --run-dir results/run_2024-01-01

See Also:
    pleb.config.PipelineConfig: Configuration model.
    pleb.pipeline.run_pipeline: Pipeline execution entry point.
    pleb.param_scan.run_param_scan: Parameter scan entry point.
    pleb.qc_report.generate_qc_report: QC report generator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import os
import sys
import tempfile
from dataclasses import fields

from .config import PipelineConfig
from .ingest import ingest_dataset, IngestError
from .logging_utils import set_log_dir
from .pipeline import run_pipeline
from .param_scan import run_param_scan
from .qc_report import generate_qc_report
from .config_io import (
    _dump_toml_no_nulls,
    _load_config_dict,
    _parse_value_as_toml_literal,
    _set_dotted_key,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        An :class:`argparse.ArgumentParser` configured with pipeline options.

    Notes:
        This parser is used by :func:`main` and includes both pipeline and
        param-scan options. The ``qc-report`` subcommand uses a separate parser.
    """
    p = argparse.ArgumentParser(
        description="Data combination diagnostics pipeline (tempo2 + plots + reports)."
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to config file (.json or .toml). Use '-' to read from stdin. Optional if using --set.",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help='Override/add a config key. Repeatable. Dotted keys allowed (e.g. fix.required_tim_flags.-pta="EPTA").',
    )

    # Param scan (fit-only): run baseline + candidate .par variants and compare via Δχ² / Wald z.
    p.add_argument(
        "--param-scan",
        action="store_true",
        help="Run a parameter scan (fit-only) instead of the full pipeline.",
    )
    p.add_argument(
        "--scan-branch",
        type=str,
        default=None,
        help="Git branch to scan (default: config.reference_branch).",
    )
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
    p.add_argument(
        "--scan-file",
        type=Path,
        default=None,
        help="Text file with one candidate spec per line.",
    )
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
    p.add_argument(
        "--scan-outdir",
        type=str,
        default=None,
        help="Override output directory name for the param scan run.",
    )

    return p


def build_qc_report_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the ``qc-report`` subcommand.

    Returns:
        An :class:`argparse.ArgumentParser` configured for PQC report options.
    """
    p = argparse.ArgumentParser(
        description="Generate PQC report summaries and plots from a run directory."
    )
    p.add_argument("qc_report", nargs="?", help=argparse.SUPPRESS)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing qc outputs.",
    )
    p.add_argument(
        "--backend-col", default="group", help="Backend column name (default: group)."
    )
    p.add_argument(
        "--backend",
        default=None,
        help="Optional: filter plots to a single backend key.",
    )
    p.add_argument(
        "--structure-group-cols",
        default=None,
        help="Comma-separated group columns; use ; for multiple groupings.",
    )
    p.add_argument(
        "--no-feature-plots",
        action="store_true",
        help="Skip feature (e.g., orbital phase/solar) plots.",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Output directory for report artifacts (default: <run-dir>/qc_report).",
    )
    p.add_argument(
        "--no-plots", action="store_true", help="Skip transient plot generation."
    )
    return p


def build_ingest_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the ingest subcommand."""
    p = argparse.ArgumentParser(
        description="Ingest pulsar timing files into canonical layout."
    )
    p.add_argument("ingest", nargs="?", help=argparse.SUPPRESS)
    p.add_argument(
        "--mapping",
        dest="ingest_mapping_file",
        required=False,
        help="JSON mapping file (required unless provided in config).",
    )
    p.add_argument(
        "--output-dir",
        dest="ingest_output_dir",
        default=None,
        help="Output root directory (required if no config).",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Optional config file to supply output root defaults.",
    )
    p.add_argument(
        "--ingest-commit-branch",
        dest="ingest_commit_branch",
        action="store_true",
        help="Create a new branch and commit ingest outputs.",
    )
    p.add_argument(
        "--ingest-commit-branch-name",
        dest="ingest_commit_branch_name",
        help="Explicit branch name for ingest commit (optional).",
    )
    p.add_argument(
        "--ingest-commit-base-branch",
        dest="ingest_commit_base_branch",
        help="Base branch for ingest commit (optional).",
    )
    p.add_argument(
        "--ingest-commit-message",
        dest="ingest_commit_message",
        help="Commit message for ingest (optional).",
    )
    p.add_argument(
        "--ingest-verify",
        dest="ingest_verify",
        action="store_true",
        help="Warn if expected tim files are missing after ingest.",
    )
    p.add_argument(
        "--pulsars",
        dest="ingest_pulsars",
        help="Comma-separated list of pulsars to ingest (optional).",
    )
    return p


def build_workflow_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the workflow subcommand."""
    p = argparse.ArgumentParser(description="Run a multi-step workflow file.")
    p.add_argument("workflow", nargs="?", help=argparse.SUPPRESS)
    p.add_argument(
        "--file",
        dest="workflow_file",
        required=True,
        help="Workflow file (.toml or .json).",
    )
    return p


def run_qc_report(argv: list[str] | None) -> int:
    """Generate QC diagnostics and plots for PQC CSVs in a run directory.

    Args:
        argv: Optional CLI argument list (defaults to ``sys.argv`` parsing).

    Returns:
        Exit code (0 on success).
    """
    args = build_qc_report_parser().parse_args(argv)
    report_dir = generate_qc_report(
        run_dir=Path(args.run_dir),
        backend_col=str(args.backend_col),
        backend=(str(args.backend) if args.backend is not None else None),
        report_dir=(Path(args.report_dir) if args.report_dir else None),
        no_plots=bool(args.no_plots),
        structure_group_cols=(
            str(args.structure_group_cols) if args.structure_group_cols else None
        ),
        no_feature_plots=bool(args.no_feature_plots),
    )
    print(str(report_dir))
    return 0


def run_workflow(argv: list[str] | None) -> int:
    """Run a workflow file and print the last run directory."""
    args = build_workflow_parser().parse_args(argv)
    from .workflow import run_workflow as _run

    ctx = _run(Path(args.workflow_file))
    if ctx.last_run_dir:
        print(str(ctx.last_run_dir))
    return 0


def run_ingest(argv: list[str] | None) -> int:
    """Run mapping-driven ingest mode."""
    args = build_ingest_parser().parse_args(argv)
    cfg = None
    if args.config:
        cfg = PipelineConfig.load(args.config)
    mapping_file = args.ingest_mapping_file or (
        cfg.ingest_mapping_file if cfg else None
    )
    if not mapping_file:
        raise SystemExit(
            "Ingest mode requires --mapping or ingest_mapping_file in config."
        )
    output_root = args.ingest_output_dir or (cfg.ingest_output_dir if cfg else None)
    if output_root is None and cfg is not None:
        output_root = Path(cfg.home_dir) / Path(cfg.dataset_name)
    if output_root is None or str(output_root).strip() == "":
        raise SystemExit(
            "Ingest mode requires --output-dir (or ingest_output_dir/home_dir+dataset_name in config)."
        )
    set_log_dir(Path(output_root) / "logs")
    try:
        verify = bool(
            args.ingest_verify
            or (getattr(cfg, "ingest_verify", False) if cfg else False)
        )
        pulsars = _parse_csv_list(args.ingest_pulsars) if args.ingest_pulsars else None
        report = ingest_dataset(
            Path(mapping_file), Path(output_root), verify=verify, pulsars=pulsars
        )
    except IngestError as e:
        raise SystemExit(str(e)) from e
    ingest_commit_branch = True
    if ingest_commit_branch:
        branch_name = args.ingest_commit_branch_name or (
            getattr(cfg, "ingest_commit_branch_name", None) if cfg else None
        )
        base_branch = args.ingest_commit_base_branch or (
            getattr(cfg, "ingest_commit_base_branch", None) if cfg else None
        )
        commit_message = args.ingest_commit_message or (
            getattr(cfg, "ingest_commit_message", None) if cfg else None
        )
        from .ingest import commit_ingest_changes

        new_branch = commit_ingest_changes(
            Path(output_root),
            branch_name=branch_name,
            base_branch=base_branch,
            commit_message=commit_message,
        )
        print(f"ingest commit branch: {new_branch}")
    print(str(report["output_root"]))
    return 0


def _parse_csv_list(raw: str | None) -> list[str] | None:
    """Parse a comma-separated list string.

    Args:
        raw: Comma-separated list string, or ``None``.

    Returns:
        A list of stripped items, or ``None`` if the input is empty.
    """
    if raw is None:
        return None
    items = [p.strip() for p in str(raw).split(",")]
    items = [p for p in items if p]
    return items or None


def _cli_key_to_cfg_key(key: str) -> str:
    return key.replace("-", "_")


def _unknown_args_to_overrides(unknown: list[str], cfg_keys: set[str]) -> list[str]:
    """Convert unknown CLI args into config overrides (KEY=VALUE).

    Supports:
      --foo=bar
      --foo bar
      --foo        (sets true)
      --no-foo     (sets false)
    """
    overrides: list[str] = []
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            i += 1
            continue
        raw = tok[2:]
        if "=" in raw:
            k, v = raw.split("=", 1)
            k = _cli_key_to_cfg_key(k)
            if k.startswith("no_"):
                k = k[3:]
                v = "false"
            if k not in cfg_keys:
                raise ValueError(f"Unknown config key for CLI override: {k}")
            overrides.append(f"{k}={v}")
            i += 1
            continue
        k = _cli_key_to_cfg_key(raw)
        if k.startswith("no_"):
            k = k[3:]
            if k not in cfg_keys:
                raise ValueError(f"Unknown config key for CLI override: {k}")
            overrides.append(f"{k}=false")
            i += 1
            continue
        # try to consume next token as value
        if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            v = unknown[i + 1]
            if k not in cfg_keys:
                raise ValueError(f"Unknown config key for CLI override: {k}")
            overrides.append(f"{k}={v}")
            i += 2
            continue
        # bare flag => true
        if k not in cfg_keys:
            raise ValueError(f"Unknown config key for CLI override: {k}")
        overrides.append(f"{k}=true")
        i += 1
    return overrides


def main(argv=None) -> int:
    """Run the CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code.

    Raises:
        FileNotFoundError: If a specified config or scan file cannot be found.
        SystemExit: If required scan specifications are missing.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "qc-report":
        return run_qc_report(argv)
    if argv and argv[0] == "workflow":
        return run_workflow(argv)
    if argv and argv[0] == "ingest":
        return run_ingest(argv)
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    cfg_keys = {f.name for f in fields(PipelineConfig)}
    extra_overrides = _unknown_args_to_overrides(unknown, cfg_keys)
    if extra_overrides:
        args.overrides = (args.overrides or []) + extra_overrides

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

    if getattr(cfg, "ingest_mapping_file", None):
        mapping_arg = ["--mapping", str(cfg.ingest_mapping_file)]
        output_arg = []
        if getattr(cfg, "ingest_output_dir", None):
            output_arg = ["--output-dir", str(cfg.ingest_output_dir)]
        config_arg = ["--config", str(args.config)] if args.config else []
        return run_ingest(mapping_arg + output_arg + config_arg)

    if args.param_scan:
        specs: list[str] = []
        if args.scan_specs:
            specs.extend([str(s) for s in args.scan_specs if str(s).strip()])
        if args.scan_file is not None:
            if not args.scan_file.exists():
                raise FileNotFoundError(str(args.scan_file))
            for raw in args.scan_file.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                line = raw.strip()
                if not line or line.startswith(("#", "C ", "c ")):
                    continue
                specs.append(line)

        if not specs and not args.scan_typical:
            raise SystemExit(
                "--param-scan requires at least one --scan spec (or --scan-file), unless --scan-typical is used."
            )

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
