#!/usr/bin/env python3
"""scripts/diagnose_qc.py

Read a CSV produced by run_qc.py and print diagnostics.

Usage:
  python scripts/diagnose_qc.py --csv out.csv --backend-col group

Outputs:
- dataset summary
- backend composition
- bad-measurement counts
- transient event table
"""

from __future__ import annotations
import argparse
import pandas as pd
from pqc.utils.diagnostics import summarize_dataset, summarize_results, export_structure_table
from pqc.utils.logging import info

def main() -> None:
    """Read a QC CSV and print diagnostic summaries."""
    ap = argparse.ArgumentParser(description="Summarize QC CSV outputs")
    ap.add_argument("--csv", required=True, help="Path to CSV produced by run_qc.py")
    ap.add_argument("--backend-col", default="group", help="Backend column name (default: group)")
    ap.add_argument(
        "--structure-group-cols",
        default=None,
        help='Comma-separated group columns; use ";" to run multiple groupings (default: backend-col)',
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    summarize_dataset(df, backend_col=args.backend_col)
    summarize_results(df, backend_col=args.backend_col)

    # Optional structure diagnostics (orbital phase, solar elongation, etc.)
    if args.structure_group_cols:
        groupings = []
        for raw in str(args.structure_group_cols).split(';'):
            cols = tuple([c.strip() for c in raw.split(',') if c.strip()])
            if cols:
                groupings.append(cols)
    else:
        groupings = [(args.backend_col,)]

    for cols in groupings:
        label = ",".join(cols) if cols else "<none>"
        struct = export_structure_table(df, group_cols=cols)
        if struct.empty:
            info(f"No structure diagnostics found for grouping: {label}")
            continue
        present = struct[struct["present"] == True]
        if not present.empty:
            info(f"Structure diagnostics for grouping [{label}] (present=True, first 30):")
            info(present.head(30).to_string(index=False))
        else:
            info(f"Structure diagnostics for grouping [{label}] (no present=True; showing first 30):")
            info(struct.head(30).to_string(index=False))

if __name__ == "__main__":
    main()
