#!/usr/bin/env python3
"""Generate a release-quality report from existing PLEB QC outputs.

This is a checkout-local convenience wrapper.  The reusable implementation lives
in :mod:`pleb.release_quality_report`; installed users can use the equivalent
``pleb release-report`` subcommand.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

# Allow ``python scripts/generate_release_quality_report.py`` from a source
# checkout without requiring an editable install first.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pleb.config import ReleaseQualityReportConfig  # noqa: E402
from pleb.release_quality_report import (  # noqa: E402
    ReleaseQualityThresholds,
    generate_release_quality_report,
)


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value in (None, ""):
        return None
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _choose(cli_value, config_value, default=None):
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a reader-facing release-quality report from existing PLEB QC CSVs."
    )
    p.add_argument("--run-dir", type=Path, default=None, help="Run directory containing qc outputs.")
    p.add_argument("--config", type=Path, default=None, help="Optional TOML/JSON release-quality-report config.")
    p.add_argument("--report-dir", type=Path, default=None, help="Output directory; default is <run-dir>/release_quality_report.")
    p.add_argument("--output-name", default=None, help="PDF filename; default is release_quality_report.pdf.")
    p.add_argument("--title", default=None, help="Report title override.")
    p.add_argument("--backend-col", default=None, help="Preferred backend column for risk ranking.")
    p.add_argument("--outlier-cols", default=None, help="Comma-separated compact-decision outlier columns.")
    p.add_argument("--no-per-pulsar-pages", action="store_true", help="Skip residual-vs-MJD pages.")
    p.add_argument("--per-pulsar-page-limit", type=int, default=None, help="Maximum number of per-pulsar pages.")
    p.add_argument("--top-n", type=int, default=None, help="Number of flagged TOA rows in the compact table.")
    p.add_argument("--yellow-bad-fraction", type=float, default=None)
    p.add_argument("--red-bad-fraction", type=float, default=None)
    p.add_argument("--yellow-review-fraction", type=float, default=None)
    p.add_argument("--red-review-fraction", type=float, default=None)
    p.add_argument("--yellow-event-fraction", type=float, default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ReleaseQualityReportConfig.load(args.config) if args.config else None
    run_dir = args.run_dir if args.run_dir is not None else (cfg.run_dir if cfg else None)
    if run_dir is None:
        raise SystemExit("Provide --run-dir or set run_dir in --config.")

    thresholds = ReleaseQualityThresholds(
        yellow_bad_fraction=float(_choose(args.yellow_bad_fraction, cfg.yellow_bad_fraction if cfg else None, 0.01)),
        red_bad_fraction=float(_choose(args.red_bad_fraction, cfg.red_bad_fraction if cfg else None, 0.05)),
        yellow_review_fraction=float(_choose(args.yellow_review_fraction, cfg.yellow_review_fraction if cfg else None, 0.005)),
        red_review_fraction=float(_choose(args.red_review_fraction, cfg.red_review_fraction if cfg else None, 0.02)),
        yellow_event_fraction=float(_choose(args.yellow_event_fraction, cfg.yellow_event_fraction if cfg else None, 0.10)),
    )
    include_per_pulsar_pages = False if args.no_per_pulsar_pages else (cfg.include_per_pulsar_pages if cfg else True)

    result = generate_release_quality_report(
        run_dir=Path(run_dir),
        report_dir=args.report_dir if args.report_dir is not None else (cfg.report_dir if cfg else None),
        output_name=str(_choose(args.output_name, cfg.output_name if cfg else None, "release_quality_report.pdf")),
        title=_choose(args.title, cfg.title if cfg else None, None),
        backend_col=_choose(args.backend_col, cfg.backend_col if cfg else None, None),
        outlier_cols=_parse_csv_list(args.outlier_cols) if args.outlier_cols else (cfg.outlier_cols if cfg else None),
        thresholds=thresholds,
        include_per_pulsar_pages=bool(include_per_pulsar_pages),
        per_pulsar_page_limit=int(_choose(args.per_pulsar_page_limit, cfg.per_pulsar_page_limit if cfg else None, 30)),
        top_n=int(_choose(args.top_n, cfg.top_n if cfg else None, 50)),
    )
    if result is None:
        raise SystemExit("release quality report generation requires matplotlib.")

    print(result.pdf_path)
    print(result.scorecard_path)
    print(result.backend_risks_path)
    print(result.flagged_toas_path)
    print(result.summary_json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
