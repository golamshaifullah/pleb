#!/usr/bin/env python3
"""Plot a per-pulsar QC summary: residuals vs MJD with bad points and event members.

Usage:
  python scripts/plot_qc_summary.py --csv out.csv --out summary.png
  python scripts/plot_qc_summary.py --csv out.csv --out summary.png --feature-plots --feature-outdir summary_feats
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.interpolate import UnivariateSpline
import colorsys

FEATURE_COLUMNS = [
    "orbital_phase",
    "solar_elongation_deg",
    "elevation_deg",
    "airmass",
    "parallactic_angle_deg",
    "freq_bin",
]

OUTLIER_FLAGS = [
    "bad",
    "bad_day",
    "transient_id",
    "step_id",
    "dm_step_id",
    "step_global_id",
    "dm_step_global_id",
]

MARKERS = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "8", "*"]


def _shade(color, factor):
    """Lighten or darken an RGB color by a factor."""
    r, g, b = to_rgb(color)
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)
    lightness = min(1.0, max(0.1, lightness * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, lightness, s)
    return (r2, g2, b2)


def _feature_spline(x, y, s=None):
    """Fit a smoothing spline for a feature trendline."""
    # x must be sorted
    if len(x) < 6:
        return None
    try:
        spline = UnivariateSpline(x, y, s=s)
        return spline
    except Exception:
        return None


def _rolling_std(y, window):
    """Compute a rolling standard deviation for confidence bands."""
    if len(y) < window:
        return np.full_like(y, np.nan)
    s = (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=max(3, window // 3))
        .std()
    )
    return s.to_numpy()


def main() -> None:
    """Generate a QC residual summary plot from a PQC CSV file."""
    ap = argparse.ArgumentParser(description="QC summary residual plot")
    ap.add_argument("--csv", required=True, help="QC output CSV")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument(
        "--backend-col", default="group", help="Group column (default: group)"
    )
    ap.add_argument("--system-col", default="sys", help="System column (default: sys)")
    ap.add_argument("--alpha", type=float, default=0.3, help="Point alpha")
    ap.add_argument(
        "--feature-ci",
        action="store_true",
        help="Plot confidence bands around feature splines",
    )
    ap.add_argument(
        "--ci-window", type=int, default=25, help="Rolling window for CI bands"
    )
    ap.add_argument(
        "--feature-plots",
        action="store_true",
        help="Write residual-vs-feature summary plots",
    )
    ap.add_argument(
        "--feature-outdir",
        default=None,
        help="Output directory for feature summary plots",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("CSV is empty")

    # Residual column
    resid_col = "resid_detrended" if "resid_detrended" in df.columns else "resid"
    if resid_col not in df.columns:
        raise SystemExit("CSV missing residual column")

    # Bad points + event members (explicit, type-safe)
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

    # Color map per group
    groups = (
        df[args.backend_col].astype(str)
        if args.backend_col in df.columns
        else pd.Series(["all"] * len(df))
    )
    systems = (
        df[args.system_col].astype(str)
        if args.system_col in df.columns
        else pd.Series(["all"] * len(df))
    )

    uniq_groups = list(dict.fromkeys(groups))
    cmap = plt.get_cmap("tab20")
    group_color = {g: cmap(i % cmap.N) for i, g in enumerate(uniq_groups)}

    # System marker + shade within group
    sys_map = {}
    for g in uniq_groups:
        sys_in_g = list(dict.fromkeys(systems[groups == g]))
        for i, s in enumerate(sys_in_g):
            base = group_color[g]
            factor = 0.7 + 0.3 * (i / max(1, len(sys_in_g) - 1))
            color = _shade(base, factor)
            marker = MARKERS[i % len(MARKERS)]
            sys_map[s] = (marker, color)

    # Plot
    plt.figure(figsize=(10, 6))

    # Regular points by system
    for s in dict.fromkeys(systems):
        mask = (systems == s) & (~bad_point) & (~event_member)
        if not mask.any():
            continue
        marker, color = sys_map.get(s, ("o", "C0"))
        plt.scatter(
            df.loc[mask, "mjd"],
            df.loc[mask, resid_col],
            s=14,
            marker=marker,
            color=color,
            alpha=args.alpha,
            label=str(s),
        )

    both = bad_point & event_member
    bad_only = bad_point & (~event_member)
    event_only = event_member & (~bad_point)

    # Bad points: grey X
    if bad_only.any():
        plt.scatter(
            df.loc[bad_only, "mjd"],
            df.loc[bad_only, resid_col],
            s=28,
            marker="x",
            color="grey",
            alpha=0.9,
            label="bad_point",
        )

    # Event members: open red circles
    if event_only.any():
        plt.scatter(
            df.loc[event_only, "mjd"],
            df.loc[event_only, resid_col],
            s=36,
            marker="o",
            facecolors="none",
            edgecolors="red",
            alpha=0.9,
            label="event_member",
        )

    # Both: layered marker
    if both.any():
        plt.scatter(
            df.loc[both, "mjd"],
            df.loc[both, resid_col],
            s=28,
            marker="x",
            color="grey",
            alpha=0.9,
            label="bad_point+event_member",
        )
        plt.scatter(
            df.loc[both, "mjd"],
            df.loc[both, resid_col],
            s=36,
            marker="o",
            facecolors="none",
            edgecolors="red",
            alpha=0.9,
            label=None,
        )

    # Solar-flagged: open orange triangles
    if solar_bad.any():
        plt.scatter(
            df.loc[solar_bad, "mjd"],
            df.loc[solar_bad, resid_col],
            s=36,
            marker="^",
            facecolors="none",
            edgecolors="orange",
            alpha=0.9,
            label="solar_bad",
        )
    if orbital_bad.any():
        plt.scatter(
            df.loc[orbital_bad, "mjd"],
            df.loc[orbital_bad, resid_col],
            s=36,
            marker="s",
            facecolors="none",
            edgecolors="blue",
            alpha=0.9,
            label="orbital_phase_bad",
        )

    # Feature splines
    for feat in FEATURE_COLUMNS:
        if feat not in df.columns:
            continue
        # Only plot if structure indicates presence or if no structure cols exist
        present_col = f"structure_{feat}_present"
        if present_col in df.columns:
            if not df[present_col].fillna(False).any():
                continue
            sub = df[df[present_col].fillna(False)].copy()
        else:
            sub = df.copy()
        if sub.empty:
            continue
        sub = sub.sort_values("mjd")
        x = sub["mjd"].to_numpy(dtype=float)
        y = sub[resid_col].to_numpy(dtype=float)
        spline = _feature_spline(x, y, s=None)
        if spline is None:
            continue
        xs = np.linspace(x.min(), x.max(), 300)
        ys = spline(xs)
        plt.plot(xs, ys, lw=2, label=f"{feat} spline")

        if args.feature_ci:
            # rolling std around spline on data points
            res = y - spline(x)
            std = _rolling_std(res, window=max(7, int(args.ci_window)))
            # interpolate std to spline grid
            if np.isfinite(std).any():
                std_i = np.interp(xs, x, np.nan_to_num(std, nan=np.nanmedian(std)))
                plt.fill_between(xs, ys - std_i, ys + std_i, alpha=0.15)

    plt.xlabel("MJD")
    plt.ylabel(resid_col)
    bad_count = int(bad_point.sum())
    event_count = int(event_member.sum())
    solar_count = int(solar_bad.sum())
    orbital_count = int(orbital_bad.sum())
    plt.title(
        f"{Path(args.csv).name} | bad_points={bad_count} | event_members={event_count} | "
        f"solar_bad={solar_count} | orbital_phase_bad={orbital_count}"
    )
    plt.legend(fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()

    if args.feature_plots:
        outdir = (
            Path(args.feature_outdir) if args.feature_outdir else Path(args.out).parent
        )
        outdir.mkdir(parents=True, exist_ok=True)
        feature_list = [
            "orbital_phase",
            "solar_elongation_deg",
            "elevation_deg",
            "parallactic_angle_deg",
        ]
        for feat in feature_list:
            if feat not in df.columns:
                continue
            x = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(df[resid_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid):
                continue
            plt.figure(figsize=(7, 5))
            for s in dict.fromkeys(systems):
                mask = (systems == s) & (~bad_point) & (~event_member) & valid
                if not mask.any():
                    continue
                marker, color = sys_map.get(s, ("o", "C0"))
                plt.scatter(
                    x[mask],
                    y[mask],
                    s=14,
                    marker=marker,
                    color=color,
                    alpha=args.alpha,
                    label=str(s),
                )
            bad_only = bad_point & (~event_member) & valid
            event_only = event_member & (~bad_point) & valid
            both = bad_point & event_member & valid
            if bad_only.any():
                plt.scatter(
                    x[bad_only],
                    y[bad_only],
                    s=28,
                    marker="x",
                    color="grey",
                    alpha=0.9,
                    label="bad_point",
                )
            if event_only.any():
                plt.scatter(
                    x[event_only],
                    y[event_only],
                    s=36,
                    marker="o",
                    facecolors="none",
                    edgecolors="red",
                    alpha=0.9,
                    label="event_member",
                )
            if both.any():
                plt.scatter(
                    x[both],
                    y[both],
                    s=28,
                    marker="x",
                    color="grey",
                    alpha=0.9,
                    label="bad_point+event_member",
                )
                plt.scatter(
                    x[both],
                    y[both],
                    s=36,
                    marker="o",
                    facecolors="none",
                    edgecolors="red",
                    alpha=0.9,
                    label=None,
                )
            plt.xlabel(feat)
            plt.ylabel(resid_col)
            plt.title(f"{Path(args.csv).name} | {feat}")
            plt.legend(fontsize=7, ncol=2, frameon=False)
            plt.tight_layout()
            out_path = outdir / f"{Path(args.out).stem}_{feat}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()

    print(f"Wrote summary plot to {args.out}")


if __name__ == "__main__":
    main()
