from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline
from .param_scan import run_param_scan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Data combination diagnostics pipeline (tempo2 + plots + reports).")
    p.add_argument("--config", type=Path, required=True, help="Path to config file (.json or .toml).")
    p.add_argument("--results-dir", type=Path, default=None, help="Override results_dir from config.")
    p.add_argument("--outdir-name", type=str, default=None, help="Override outdir_name from config.")
    p.add_argument("--force-rerun", action="store_true", help="Re-run tempo2 even if outputs exist.")
    p.add_argument("--no-tempo2", action="store_true", help="Skip tempo2 run; use existing outputs if present.")
    p.add_argument("--jobs", type=int, default=None, help="Number of parallel workers to run pulsars concurrently (per branch).")

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


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cfg = PipelineConfig.load(args.config)
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

        if not specs:
            raise SystemExit("--param-scan requires at least one --scan spec (or --scan-file).")

        out_paths = run_param_scan(
            cfg,
            branch=args.scan_branch,
            pulsars=args.scan_pulsars,
            candidate_specs=specs,
            outdir_name=args.scan_outdir,
        )
        print(str(out_paths["tag"]))
        return 0

    out_paths = run_pipeline(cfg)
    print(str(out_paths["tag"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
