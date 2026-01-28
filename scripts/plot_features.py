#!/usr/bin/env python3
"""Plot detected feature structure (e.g., orbital phase, solar elongation).

This script expects pqc CSV output with feature columns (e.g., orbital_phase,
solar_elongation_deg) and structure diagnostics (structure_*_present).

Usage:
  python scripts/plot_features.py --csv out.csv --backend-col group --outdir plots
  python scripts/plot_features.py --csv out.csv --structure-group-cols group;sys
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pqc.utils.diagnostics import export_structure_table

CIRCULAR_FEATURES = {"orbital_phase"}
KNOWN_FEATURES = [
    "orbital_phase",
    "solar_elongation_deg",
    "elevation_deg",
    "airmass",
    "parallactic_angle_deg",
    "freq_bin",
]


def _bin_stats(x: np.ndarray, y: np.ndarray, nbins: int = 20):
    if len(x) < 2:
        return None
    edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(nbins, np.nan)
    for i in range(nbins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if mask.any():
            med[i] = np.nanmedian(y[mask])
    return centers, med


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot feature-structure diagnostics from pqc CSV")
    ap.add_argument("--csv", required=True, help="QC output CSV")
    ap.add_argument("--outdir", default="feature_plots", help="Directory for PNGs")
    ap.add_argument("--backend-col", default="group", help="Backend column name (default: group)")
    ap.add_argument(
        "--structure-group-cols",
        default=None,
        help='Comma-separated group columns; use ";" to run multiple groupings (default: backend-col)',
    )
    ap.add_argument("--features", default=None, help="Comma-separated features to plot (default: auto)")
    ap.add_argument("--only-present", action="store_true", help="Plot only features marked present (default).")
    ap.add_argument("--include-all", action="store_true", help="Plot all features even if not present.")
    ap.add_argument("--bins", type=int, default=20, help="Number of bins for median overlay.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("CSV is empty")

    resid_col = "resid_detrended" if "resid_detrended" in df.columns else "resid"
    if resid_col not in df.columns:
        raise SystemExit("CSV missing residual column")

    if args.features:
        features = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        features = [f for f in KNOWN_FEATURES if f in df.columns]

    if not features:
        raise SystemExit("No feature columns found to plot")

    if args.structure_group_cols:
        groupings = []
        for raw in str(args.structure_group_cols).split(";"):
            cols = tuple([c.strip() for c in raw.split(",") if c.strip()])
            if cols:
                groupings.append(cols)
    else:
        groupings = [(args.backend_col,)]

    only_present = True if not args.include_all else False
    if args.only_present:
        only_present = True

    for cols in groupings:
        struct = export_structure_table(df, group_cols=cols)
        if struct.empty:
            continue

        for feat in features:
            present_col = f"structure_{feat}_present"
            for _, row in struct[struct["feature"] == feat].iterrows():
                if only_present and not bool(row.get("present", False)):
                    continue

                # filter df for this group
                sub = df.copy()
                for col in cols:
                    sub = sub[sub[col] == row.get(col)]
                if sub.empty or feat not in sub.columns:
                    continue

                x = sub[feat].to_numpy(dtype=float)
                y = sub[resid_col].to_numpy(dtype=float)
                s = sub["sigma"].to_numpy(dtype=float) if "sigma" in sub.columns else None

                # handle circular feature
                if feat in CIRCULAR_FEATURES:
                    order = np.argsort(x)
                    x, y = x[order], y[order]
                    if s is not None:
                        s = s[order]

                plt.figure(figsize=(6, 4))
                if s is not None:
                    plt.errorbar(x, y, yerr=s, fmt=".", alpha=0.4, capsize=0)
                else:
                    plt.plot(x, y, ".", alpha=0.5)

                stats = _bin_stats(x, y, nbins=int(args.bins))
                if stats is not None:
                    centers, med = stats
                    plt.plot(centers, med, "-", color="black", lw=1)

                plt.xlabel(feat)
                plt.ylabel(resid_col)
                label = ",".join([f"{c}={row.get(c)}" for c in cols]) if cols else "all"
                plt.title(f"{feat} ({label})")

                fname = f"{feat}_{label.replace('=', '_').replace(',', '_').replace(' ', '')}.png"
                plt.tight_layout()
                plt.savefig(outdir / fname, dpi=150)
                plt.close()

    print(f"Wrote feature plots to {outdir}")


if __name__ == "__main__":
    main()
