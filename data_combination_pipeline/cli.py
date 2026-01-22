from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Data combination diagnostics pipeline (tempo2 + plots + reports).")
    p.add_argument("--config", type=Path, required=True, help="Path to config file (.json or .toml).")
    p.add_argument("--results-dir", type=Path, default=None, help="Override results_dir from config.")
    p.add_argument("--outdir-name", type=str, default=None, help="Override outdir_name from config.")
    p.add_argument("--force-rerun", action="store_true", help="Re-run tempo2 even if outputs exist.")
    p.add_argument("--no-tempo2", action="store_true", help="Skip tempo2 run; use existing outputs if present.")
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

    out_paths = run_pipeline(cfg)
    # Print the tag directory so it's easy to find
    print(str(out_paths["tag"]))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
