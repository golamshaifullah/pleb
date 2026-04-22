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

import shlex
import sys
from dataclasses import fields

from .config import (
    IngestConfig,
    ParamScanConfig,
    PipelineConfig,
    QCReportConfig,
    WorkflowRunConfig,
)
from .ingest import ingest_dataset, IngestError
from .logging_utils import set_log_dir
from .pipeline import run_pipeline
from .param_scan import run_param_scan
from .qc_report import generate_qc_report
from .run_report import generate_run_report
from .public_release_compare import compare_public_releases
from .optimize.cli import load_optimization_config
from .optimize.optimizer import run_optimization
from .config_io import (
    _dump_toml_no_nulls,
    _load_config_dict,
    _parse_value_as_toml_literal,
    _set_dotted_key,
)


def _format_command(argv: list[str] | None) -> str:
    parts = ["pleb"] + [str(x) for x in (argv or [])]
    return " ".join(shlex.quote(p) for p in parts)


def _write_run_settings(
    run_dir: Path,
    argv: list[str] | None,
    cfg: PipelineConfig | None = None,
    cfg_data: dict | None = None,
) -> None:
    settings_dir = Path(run_dir) / "run_settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "command.txt").write_text(
        _format_command(argv) + "\n", encoding="utf-8"
    )
    if cfg is not None:
        resolved = cfg.resolved()
        toml = _dump_toml_no_nulls(resolved.to_dict())
        (settings_dir / "pipeline_config.resolved.toml").write_text(
            toml, encoding="utf-8"
        )
    elif cfg_data is not None:
        toml = _dump_toml_no_nulls(cfg_data)
        (settings_dir / "pipeline_config.resolved.toml").write_text(
            toml, encoding="utf-8"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with top-level pipeline and param-scan options.

    Notes
    -----
    The ``qc-report``, ``ingest``, and ``workflow`` commands use dedicated
    subcommand parsers.
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

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for QC-report options.
    """
    p = argparse.ArgumentParser(
        description="Generate PQC report summaries and plots from a run directory."
    )
    p.add_argument("qc_report", nargs="?", help=argparse.SUPPRESS)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=False,
        help="Run directory containing qc outputs.",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Optional config file (.toml/.json) for qc-report mode.",
    )
    p.add_argument(
        "--backend-col", default=None, help="Backend column name (default: group)."
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
        "--compact-pdf",
        action="store_true",
        help="Generate a compact composite PDF report.",
    )
    p.add_argument(
        "--compact-pdf-name",
        default=None,
        help="Filename for compact PDF (default: qc_compact_report.pdf).",
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
    """Build the CLI parser for ingest mode.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for ingest-only options.
    """
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
    """Build the CLI parser for workflow mode.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for workflow-file execution options.
    """
    p = argparse.ArgumentParser(description="Run a multi-step workflow file.")
    p.add_argument("workflow", nargs="?", help=argparse.SUPPRESS)
    p.add_argument(
        "--file",
        dest="workflow_file",
        required=False,
        help="Workflow file (.toml or .json).",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Optional workflow mode config file (.toml/.json).",
    )
    return p


def build_report_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for consolidated report generation."""
    p = argparse.ArgumentParser(
        description="Generate a consolidated PDF report from an existing run directory."
    )
    p.add_argument("report", nargs="?", help=argparse.SUPPRESS)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing stage outputs.",
    )
    p.add_argument(
        "--stages",
        default=None,
        help=(
            "Comma-separated consolidated report stages. "
            "Valid: summary,config,ingest,fix,qc,workflow,artifacts."
        ),
    )
    p.add_argument(
        "--output-name",
        default="run_report.pdf",
        help="Output PDF filename inside --run-dir.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Optional report title override.",
    )
    return p


def build_compare_public_parser() -> argparse.ArgumentParser:
    """Build the parser for the ``compare-public`` CLI mode.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured to resolve provider catalogs, download release
        assets, and write normalized parameter-comparison tables.
    """
    p = argparse.ArgumentParser(
        description=(
            "Download latest public NANOGrav/EPTA/IPTA releases and compare "
            "parfile parameter values."
        )
    )
    p.add_argument("compare_public", nargs="?", help=argparse.SUPPRESS)
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for downloaded assets and comparison tables.",
    )
    p.add_argument(
        "--providers",
        type=Path,
        default=None,
        help=(
            "Provider TOML catalog (default: "
            "configs/catalogs/public_releases/providers.toml)."
        ),
    )
    return p


def build_optimize_parser() -> argparse.ArgumentParser:
    """Build the parser for optimization mode."""
    p = argparse.ArgumentParser(
        description="Optimize PQC and workflow settings against QC-derived metrics."
    )
    p.add_argument("optimize", nargs="?", help=argparse.SUPPRESS)
    p.add_argument(
        "--config",
        required=True,
        help="Optimization config file (.toml or .json).",
    )
    return p


def run_qc_report(argv: list[str] | None) -> int:
    """Run the ``qc-report`` subcommand.

    Parameters
    ----------
    argv : list of str, optional
        Optional CLI argument list.

    Returns
    -------
    int
        Exit code (``0`` on success).
    """
    args = build_qc_report_parser().parse_args(argv)
    qcfg = None
    if args.config:
        qcfg = QCReportConfig.load(Path(args.config))
    run_dir = Path(args.run_dir) if args.run_dir else (qcfg.run_dir if qcfg else None)
    if run_dir is None:
        raise SystemExit("qc-report requires --run-dir (or run_dir in --config).")
    report_dir = generate_qc_report(
        run_dir=run_dir,
        backend_col=str(
            args.backend_col
            if args.backend_col is not None
            else (qcfg.backend_col if qcfg else "group")
        ),
        backend=(
            str(args.backend)
            if args.backend is not None
            else (qcfg.backend if qcfg else None)
        ),
        report_dir=(
            Path(args.report_dir)
            if args.report_dir
            else (Path(qcfg.report_dir) if qcfg and qcfg.report_dir else None)
        ),
        no_plots=bool(args.no_plots or (qcfg.no_plots if qcfg else False)),
        structure_group_cols=(
            str(args.structure_group_cols)
            if args.structure_group_cols
            else (qcfg.structure_group_cols if qcfg else None)
        ),
        no_feature_plots=bool(
            args.no_feature_plots or (qcfg.no_feature_plots if qcfg else False)
        ),
        compact_pdf=bool(args.compact_pdf or (qcfg.compact_pdf if qcfg else False)),
        compact_pdf_name=str(
            args.compact_pdf_name
            if args.compact_pdf_name is not None
            else (qcfg.compact_pdf_name if qcfg else "qc_compact_report.pdf")
        ),
    )
    _write_run_settings(Path(report_dir), argv, None)
    print(str(report_dir))
    return 0


def run_workflow(argv: list[str] | None) -> int:
    """Run the ``workflow`` subcommand.

    Parameters
    ----------
    argv : list of str, optional
        Optional CLI argument list.

    Returns
    -------
    int
        Process exit code.
    """
    args = build_workflow_parser().parse_args(argv)
    from .workflow import run_workflow as _run

    wcfg = None
    if args.config:
        wcfg = WorkflowRunConfig.load(Path(args.config))
    workflow_file = (
        Path(args.workflow_file)
        if args.workflow_file
        else (wcfg.workflow_file if wcfg else None)
    )
    if workflow_file is None:
        raise SystemExit("workflow requires --file (or workflow_file in --config).")
    ctx = _run(Path(workflow_file))
    if ctx.last_run_dir:
        _write_run_settings(Path(ctx.last_run_dir), argv, None)
        print(str(ctx.last_run_dir))
    return 0


def run_report(argv: list[str] | None) -> int:
    """Run the ``report`` subcommand."""
    args = build_report_parser().parse_args(argv)
    stages = None
    if args.stages:
        stages = [s.strip() for s in str(args.stages).split(",") if s.strip()]
    pdf_path = generate_run_report(
        Path(args.run_dir),
        title=(str(args.title) if args.title not in (None, "") else None),
        output_name=str(args.output_name),
        include_stages=stages,
    )
    if pdf_path is None:
        raise SystemExit("consolidated report generation requires matplotlib.")
    _write_run_settings(Path(args.run_dir), argv, None)
    print(str(pdf_path))
    return 0


def run_compare_public(argv: list[str] | None) -> int:
    """Execute public-release download and parameter comparison workflow.

    Parameters
    ----------
    argv : list of str, optional
        Optional CLI argument list. When ``None``, arguments are read from the
        active process command line.

    Returns
    -------
    int
        Process exit code, ``0`` on success.
    """
    args = build_compare_public_parser().parse_args(argv)
    out = compare_public_releases(
        out_dir=Path(args.out_dir),
        providers_path=Path(args.providers) if args.providers else None,
    )
    _write_run_settings(Path(out["out_dir"]), argv, None)
    print(str(out["out_dir"]))
    return 0


def run_optimize(argv: list[str] | None) -> int:
    """Execute optimization mode."""
    args = build_optimize_parser().parse_args(argv)
    cfg = load_optimization_config(Path(args.config))
    result = run_optimization(cfg)
    _write_run_settings(
        Path(result.out_dir),
        argv,
        None,
        cfg_data={
            "optimize": {"config": str(Path(args.config).expanduser().resolve())}
        },
    )
    print(str(result.out_dir))
    return 0


def run_ingest(argv: list[str] | None) -> int:
    """Run the ``ingest`` subcommand.

    Parameters
    ----------
    argv : list of str, optional
        Optional CLI argument list.

    Returns
    -------
    int
        Process exit code.
    """
    args = build_ingest_parser().parse_args(argv)
    cfg = IngestConfig()
    if args.config:
        cfg = IngestConfig.load(Path(args.config))
    mapping_file = args.ingest_mapping_file or cfg.ingest_mapping_file
    if not mapping_file:
        raise SystemExit(
            "Ingest mode requires --mapping or ingest_mapping_file in config."
        )
    try:
        output_root = (
            Path(args.ingest_output_dir).expanduser().resolve()
            if args.ingest_output_dir
            else cfg.resolved_output_root()
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    set_log_dir(output_root / "logs")
    try:
        verify = bool(args.ingest_verify or cfg.ingest_verify)
        pulsars = _parse_csv_list(args.ingest_pulsars) if args.ingest_pulsars else None
        report = ingest_dataset(
            Path(mapping_file),
            Path(output_root),
            verify=verify,
            pulsars=pulsars,
            report_metadata={
                "fix_ensure_ephem": cfg.fix_ensure_ephem,
                "fix_ensure_clk": cfg.fix_ensure_clk,
                "fix_ensure_ne_sw": cfg.fix_ensure_ne_sw,
                "ingest_commit_branch_name": args.ingest_commit_branch_name
                or cfg.ingest_commit_branch_name,
                "ingest_commit_base_branch": args.ingest_commit_base_branch
                or cfg.ingest_commit_base_branch,
            },
        )
    except IngestError as e:
        raise SystemExit(str(e)) from e
    ingest_commit_branch = True
    if ingest_commit_branch:
        branch_name = args.ingest_commit_branch_name or (cfg.ingest_commit_branch_name)
        base_branch = args.ingest_commit_base_branch or (cfg.ingest_commit_base_branch)
        commit_message = args.ingest_commit_message or (cfg.ingest_commit_message)
        from .ingest import commit_ingest_changes

        new_branch = commit_ingest_changes(
            Path(output_root),
            branch_name=branch_name,
            base_branch=base_branch,
            commit_message=commit_message,
        )
        print(f"ingest commit branch: {new_branch}")
    _write_run_settings(
        Path(output_root) / "ingest_reports",
        argv,
        None,
        cfg_data=cfg.to_dict(),
    )
    print(str(report["output_root"]))
    return 0


def _parse_csv_list(raw: str | None) -> list[str] | None:
    """Parse a comma-separated list string.

    Parameters
    ----------
    raw : str or None
        Comma-separated list string.

    Returns
    -------
    list of str or None
        Stripped tokens, or ``None`` for empty input.
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

    Parameters
    ----------
    argv : list of str, optional
        Optional argument list (defaults to ``sys.argv``).

    Returns
    -------
    int
        Process exit code.

    Raises
    ------
    FileNotFoundError
        If a specified config or scan file cannot be found.
    SystemExit
        If required scan specifications are missing.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in {"init", "run", "profile", "doctor", "explain"}:
        from .ux import run_ux_cli

        return run_ux_cli(argv)
    if argv and argv[0] == "qc-report":
        return run_qc_report(argv)
    if argv and argv[0] == "workflow":
        return run_workflow(argv)
    if argv and argv[0] == "report":
        return run_report(argv)
    if argv and argv[0] == "ingest":
        return run_ingest(argv)
    if argv and argv[0] == "compare-public":
        return run_compare_public(argv)
    if argv and argv[0] == "optimize":
        return run_optimize(argv)
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    cfg_keys = {f.name for f in fields(PipelineConfig)}
    extra_overrides = _unknown_args_to_overrides(unknown, cfg_keys)
    if extra_overrides:
        args.overrides = (args.overrides or []) + extra_overrides

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

    config_base_dir = None
    if args.config and args.config != "-":
        config_base_dir = Path(args.config).expanduser().resolve().parent

    # Ingest-only configs should not need full PipelineConfig required fields.
    ingest_mapping = cfg_dict.get("ingest_mapping_file")
    if ingest_mapping is None and isinstance(cfg_dict.get("pipeline"), dict):
        ingest_mapping = cfg_dict["pipeline"].get("ingest_mapping_file")
    if ingest_mapping and not args.param_scan:
        cfg_ingest = IngestConfig.from_dict(cfg_dict, base_dir=config_base_dir)
        mapping_arg = ["--mapping", str(cfg_ingest.ingest_mapping_file)]
        output_arg = []
        if cfg_ingest.ingest_output_dir:
            output_arg = ["--output-dir", str(cfg_ingest.ingest_output_dir)]
        return run_ingest(mapping_arg + output_arg + ["--config", args.config or "-"])

    if args.param_scan:
        pscfg = ParamScanConfig.from_dict(cfg_dict, base_dir=config_base_dir)
        cfg = pscfg.to_pipeline_config()
    else:
        cfg = PipelineConfig.from_dict(cfg_dict, base_dir=config_base_dir)

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
        _write_run_settings(Path(out_paths["tag"]), argv, cfg)
        print(str(out_paths["tag"]))
        return 0

    out_paths = run_pipeline(cfg)
    _write_run_settings(Path(out_paths["tag"]), argv, cfg)
    print(str(out_paths["tag"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
