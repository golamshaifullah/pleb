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
    """Compute median y-values in evenly spaced x-bins.

    Args:
        x: X values.
        y: Y values aligned with ``x``.
        nbins: Number of bins.

    Returns:
        Tuple of (bin_centers, median_values) or ``None`` if insufficient data.
    """
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
    """Generate feature-structure plots from a PQC CSV file."""
    ap = argparse.ArgumentParser(
        description="Plot feature-structure diagnostics from pqc CSV"
    )
    ap.add_argument("--csv", required=True, help="QC output CSV")
    ap.add_argument("--outdir", default="feature_plots", help="Directory for PNGs")
    ap.add_argument(
        "--backend-col", default="group", help="Backend column name (default: group)"
    )
    ap.add_argument(
        "--structure-group-cols",
        default=None,
        help='Comma-separated group columns; use ";" to run multiple groupings (default: backend-col)',
    )
    ap.add_argument(
        "--features",
        default=None,
        help="Comma-separated features to plot (default: auto)",
    )
    ap.add_argument(
        "--only-present",
        action="store_true",
        help="Plot only features marked present (default).",
    )
    ap.add_argument(
        "--include-all",
        action="store_true",
        help="Plot all features even if not present.",
    )
    ap.add_argument(
        "--bins", type=int, default=20, help="Number of bins for median overlay."
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("CSV is empty")

    resid_col = "resid_detrended" if "resid_detrended" in df.columns else "resid"
    if resid_col not in df.columns:
        raise SystemExit("CSV missing residual column")

    if "bad_point" in df.columns:
        bad_point = df["bad_point"].fillna(False).astype(bool).to_numpy()
    else:
        bad_point = np.zeros(len(df), dtype=bool)
        if "bad_ou" in df.columns:
            bad_point |= df["bad_ou"].fillna(False).astype(bool).to_numpy()
        if "bad_mad" in df.columns:
            bad_point |= df["bad_mad"].fillna(False).astype(bool).to_numpy()
        if "robust_outlier" in df.columns:
            bad_point |= df["robust_outlier"].fillna(False).astype(bool).to_numpy()

    if "event_member" in df.columns:
        event_member = df["event_member"].fillna(False).astype(bool).to_numpy()
    else:
        event_member = np.zeros(len(df), dtype=bool)
        if "transient_id" in df.columns:
            event_member |= (
                pd.to_numeric(df["transient_id"], errors="coerce").fillna(-1).to_numpy()
                >= 0
            )
        if "step_id" in df.columns:
            event_member |= (
                pd.to_numeric(df["step_id"], errors="coerce").fillna(-1).to_numpy() >= 0
            )
        if "dm_step_id" in df.columns:
            event_member |= (
                pd.to_numeric(df["dm_step_id"], errors="coerce").fillna(-1).to_numpy()
                >= 0
            )

    if "solar_bad" in df.columns:
        solar_bad = df["solar_bad"].fillna(False).astype(bool).to_numpy()
    else:
        solar_bad = np.zeros(len(df), dtype=bool)
    if "orbital_phase_bad" in df.columns:
        orbital_bad = df["orbital_phase_bad"].fillna(False).astype(bool).to_numpy()
    else:
        orbital_bad = np.zeros(len(df), dtype=bool)

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
                s = (
                    sub["sigma"].to_numpy(dtype=float)
                    if "sigma" in sub.columns
                    else None
                )
                bad_sub = bad_point[sub.index.to_numpy()]
                event_sub = event_member[sub.index.to_numpy()]

                if feat == "solar_elongation_deg":
                    # Wrap to [-180, 180] to avoid cutoff and center at 0.
                    x = ((x + 180.0) % 360.0) - 180.0

                # handle circular feature
                if feat in CIRCULAR_FEATURES:
                    order = np.argsort(x)
                    x, y = x[order], y[order]
                    if s is not None:
                        s = s[order]

                plt.figure(figsize=(6, 4))
                good = np.isfinite(x) & np.isfinite(y)
                normal = good & (~bad_sub) & (~event_sub)
                bad_only = good & bad_sub & (~event_sub)
                event_only = good & event_sub & (~bad_sub)
                both = good & bad_sub & event_sub
                idx = sub.index.to_numpy()
                solar_only = good & solar_bad[idx]
                orbital_only = good & orbital_bad[idx]

                if s is not None:
                    plt.errorbar(
                        x[normal],
                        y[normal],
                        yerr=s[normal],
                        fmt=".",
                        alpha=0.4,
                        capsize=0,
                    )
                else:
                    plt.plot(x[normal], y[normal], ".", alpha=0.5)

                if bad_only.any():
                    plt.plot(
                        x[bad_only],
                        y[bad_only],
                        "x",
                        color="grey",
                        alpha=0.9,
                        label="bad_point",
                    )
                if event_only.any():
                    plt.scatter(
                        x[event_only],
                        y[event_only],
                        s=30,
                        marker="o",
                        facecolors="none",
                        edgecolors="red",
                        alpha=0.9,
                        label="event_member",
                    )
                if both.any():
                    plt.plot(
                        x[both],
                        y[both],
                        "x",
                        color="grey",
                        alpha=0.9,
                        label="bad_point+event_member",
                    )
                    plt.scatter(
                        x[both],
                        y[both],
                        s=30,
                        marker="o",
                        facecolors="none",
                        edgecolors="red",
                        alpha=0.9,
                        label=None,
                    )
                if solar_only.any():
                    plt.scatter(
                        x[solar_only],
                        y[solar_only],
                        s=30,
                        marker="^",
                        facecolors="none",
                        edgecolors="orange",
                        alpha=0.9,
                        label="solar_bad",
                    )
                if orbital_only.any():
                    plt.scatter(
                        x[orbital_only],
                        y[orbital_only],
                        s=30,
                        marker="s",
                        facecolors="none",
                        edgecolors="blue",
                        alpha=0.9,
                        label="orbital_phase_bad",
                    )

                stats = _bin_stats(x, y, nbins=int(args.bins))
                if stats is not None:
                    centers, med = stats
                    plt.plot(centers, med, "-", color="black", lw=1)

                plt.xlabel(feat)
                plt.ylabel(resid_col)
                if feat == "solar_elongation_deg":
                    # Center the x-axis at zero elongation (sun direction).
                    xmax = np.nanmax(np.abs(x)) if np.isfinite(x).any() else None
                    if xmax is not None and np.isfinite(xmax) and xmax > 0:
                        plt.xlim(-xmax, xmax)
                    plt.axvline(0.0, color="black", lw=0.8, alpha=0.6)
                label = ",".join([f"{c}={row.get(c)}" for c in cols]) if cols else "all"
                plt.title(
                    f"{feat} ({label}) | bad_points={int(bad_sub.sum())} | "
                    f"event_members={int(event_sub.sum())} | solar_bad={int(solar_bad[idx].sum())} | "
                    f"orbital_phase_bad={int(orbital_bad[idx].sum())}"
                )

                fname = f"{feat}_{label.replace('=', '_').replace(',', '_').replace(' ', '')}.png"
                plt.tight_layout()
                plt.savefig(outdir / fname, dpi=150)
                plt.close()

    print(f"Wrote feature plots to {outdir}")


if __name__ == "__main__":
    main()
