#!/usr/bin/env python3
"""scripts/plot_transients.py

Plot detected transient windows for quick human review (matplotlib only).

This script expects the CSV produced by run_qc.py and uses:
- columns: mjd, resid, sigma, group (or backend col), transient_id, transient_t0, transient_amp
- optionally: bad

Usage:
  python scripts/plot_transients.py --csv out.csv --backend-col group --outdir plots
  python scripts/plot_transients.py --csv out.csv --backend-col group --backend NRT.NUPPI.1484

Notes:
    - One PNG per (backend, transient_id).
    - Plots residuals vs time with +/-1 sigma errorbars.
    - Overlays fitted exponential recovery curve using the stored amplitude and t0 plus --tau-rec-days.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    """Plot transient detections from a QC CSV into per-event PNGs."""
    ap = argparse.ArgumentParser(description="Plot transient detections from QC CSV")
    ap.add_argument("--csv", required=True, help="QC output CSV")
    ap.add_argument("--outdir", default="plots", help="Directory for PNGs")
    ap.add_argument(
        "--backend-col", default="group", help="Backend column name (default: group)"
    )
    ap.add_argument(
        "--backend", default=None, help="Optional: plot only this backend key"
    )
    ap.add_argument(
        "--tau-rec-days", type=float, default=7.0, help="Tau used for overlay curve"
    )
    ap.add_argument(
        "--window-mult",
        type=float,
        default=5.0,
        help="Window length = window_mult*tau for plotting",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "transient_id" not in df.columns:
        raise SystemExit("CSV has no transient_id column (did you run detection?)")

    df = df[df["transient_id"] >= 0].copy()
    if df.empty:
        raise SystemExit("No transients found in CSV.")

    if args.backend is not None:
        df = df[df[args.backend_col].astype(str) == args.backend].copy()
        if df.empty:
            raise SystemExit(f"No transients for backend {args.backend}")

    # one plot per (backend, transient_id)
    gb = df.groupby([args.backend_col, "transient_id"], sort=False)

    for (backend, tid), sub in gb:
        sub = sub.sort_values("mjd")
        t = sub["mjd"].to_numpy()
        y = sub["resid"].to_numpy()
        s = sub["sigma"].to_numpy()

        t0 = (
            float(sub["transient_t0"].iloc[0])
            if "transient_t0" in sub.columns
            else float(t.min())
        )
        A = (
            float(sub["transient_amp"].iloc[0])
            if "transient_amp" in sub.columns
            else 0.0
        )

        # plotting window
        w_end = args.window_mult * args.tau_rec_days
        mask = (t >= t0) & (t <= t0 + w_end)
        t = t[mask]
        y = y[mask]
        s = s[mask]
        if len(t) < 2:
            continue

        plt.figure()
        plt.errorbar(t, y, yerr=s, fmt="o", capsize=2, label="event_member")

        if "bad_point" in sub.columns:
            bad_point = sub["bad_point"].fillna(False).astype(bool).to_numpy()
        else:
            bad_point = np.zeros(len(sub), dtype=bool)
            if "bad_ou" in sub.columns:
                bad_point |= sub["bad_ou"].fillna(False).astype(bool).to_numpy()
            if "bad_mad" in sub.columns:
                bad_point |= sub["bad_mad"].fillna(False).astype(bool).to_numpy()
            if "robust_outlier" in sub.columns:
                bad_point |= sub["robust_outlier"].fillna(False).astype(bool).to_numpy()

        if np.any(bad_point):
            plt.plot(
                t[bad_point],
                y[bad_point],
                "x",
                color="grey",
                alpha=0.9,
                label="bad_point",
            )

        # overlay exponential curve
        tt = np.linspace(t0, t0 + w_end, 200)
        curve = A * np.exp(-(tt - t0) / args.tau_rec_days)
        plt.plot(tt, curve)

        plt.xlabel("MJD")
        plt.ylabel("Residual")
        plt.title(
            f"{backend} transient {tid} (t0={t0:.6f}, A={A:.3g}) | bad_points={int(bad_point.sum())}"
        )
        plt.legend(fontsize=8, frameon=False)

        fname = outdir / f"transient_{backend.replace('.', '_')}_id{tid}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    main()
