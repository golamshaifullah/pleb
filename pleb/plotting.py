"""Plotting helpers for pipeline outputs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter

from .parsers import read_covmat, read_general2, read_tim_file
from .logging_utils import get_logger

logger = get_logger("pleb.plotting")

try:
    import seaborn as sns  # type: ignore
    HAVE_SEABORN = True
except Exception:
    HAVE_SEABORN = False

def freedman_diaconis_bins(x: np.ndarray, max_bins: int = 200) -> int:
    """Compute histogram bin count using the Freedman–Diaconis rule.

    Args:
        x: Input data array.
        max_bins: Maximum number of bins to return.

    Returns:
        Suggested number of bins.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 10
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    if iqr == 0:
        return min(max_bins, 30)
    bin_width = 2 * iqr * (x.size ** (-1/3))
    if bin_width <= 0:
        return min(max_bins, 30)
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return int(np.clip(bins, 5, max_bins))

class MathTextSciFormatter(ScalarFormatter):
    def __init__(self, fmt: str = "%1.1e"):
        """Create a scalar formatter using MathText scientific notation."""
        super().__init__(useMathText=True)
        self.fmt = fmt

    def _set_format(self):
        self.format = self.fmt

def savefig(fig: plt.Figure, path: Path, dpi: int) -> None:
    """Save a Matplotlib figure to disk and close it.

    Args:
        fig: Matplotlib figure.
        path: Output file path.
        dpi: Resolution in dots-per-inch.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_systems_per_pulsar(home_dir: Path, dataset_name: Path, out_paths: Dict[str, Path], pulsars: List[str], branch: str, dpi: int) -> None:
    """Plot per-pulsar system timelines and write summary tables.

    Args:
        home_dir: Root data repository.
        dataset_name: Dataset name or path.
        out_paths: Output directory mapping from :func:`make_output_tree`.
        pulsars: Pulsar names to include.
        branch: Branch name for labeling.
        dpi: Output resolution.
    """
    fig, axes = plt.subplots(nrows=len(pulsars), ncols=1, sharex=True, figsize=(12, max(2, 3*len(pulsars))))
    if len(pulsars) == 1:
        axes = [axes]

    summary_path = out_paths["tag"] / f"{branch}_summary.tsv"
    all_summary_path = out_paths["tag"] / f"{branch}_all_summary.tsv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("branch\tpulsar\tsystem\tmjd_min\tmjd_max\tmean_cadence\tmedian_cadence\tntoas\tepochs\n")
    with open(all_summary_path, "w", encoding="utf-8") as f:
        f.write("branch\tpulsar\tsystem\tmjd_min\tmjd_max\tmean_cadence\tmedian_cadence\tntoas\tepochs\n")

    for ax, pulsar in zip(axes, pulsars):
        tim_dir = home_dir / dataset_name / pulsar / "tims"
        if not tim_dir.exists():
            ax.set_title(f"{pulsar}: no tims/ directory")
            ax.axis("off")
            continue

        all_toas: List[float] = []
        tim_files = sorted(tim_dir.glob("*.tim"))
        for y, timfile in enumerate(tim_files):
            dmf = read_tim_file(timfile)
            if dmf.empty or 2 not in dmf.columns:
                continue
            mjd = pd.to_numeric(dmf[2], errors="coerce").dropna().to_numpy()
            if mjd.size == 0:
                continue

            system = timfile.stem
            ax.plot(mjd, np.full_like(mjd, y, dtype=float), linestyle="none", marker=".", markersize=2)

            epochs = int(pd.Series(mjd.astype(int)).nunique())
            ntoas = int(mjd.size)
            cadence = np.diff(np.sort(mjd))
            mean_cad = float(np.nanmean(cadence)) if cadence.size else np.nan
            med_cad = float(np.nanmedian(cadence)) if cadence.size else np.nan

            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(f"{branch}\t{pulsar}\t{system}\t{mjd.min():.6f}\t{mjd.max():.6f}\t{mean_cad:.3f}\t{med_cad:.3f}\t{ntoas}\t{epochs}\n")

            all_toas.extend(mjd.tolist())

        if all_toas:
            all_toas_np = np.array(sorted(all_toas), dtype=float)
            cadence = np.diff(all_toas_np)
            mean_cad = float(np.nanmean(cadence)) if cadence.size else np.nan
            med_cad = float(np.nanmedian(cadence)) if cadence.size else np.nan
            with open(all_summary_path, "a", encoding="utf-8") as f:
                f.write(f"{branch}\t{pulsar}\tALL\t{all_toas_np.min():.6f}\t{all_toas_np.max():.6f}\t{mean_cad:.3f}\t{med_cad:.3f}\t{all_toas_np.size}\t{int(pd.Series(all_toas_np.astype(int)).nunique())}\n")

        ax.set_title(f"{pulsar} on {branch}")
        ax.set_ylabel("system index")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("MJD")
    savefig(fig, out_paths["png"] / f"SystemsPerPulsar_{branch}.png", dpi=dpi)

def plot_pulsars_per_system(home_dir: Path, dataset_name: Path, out_paths: Dict[str, Path], pulsars: List[str], branch: str, dpi: int) -> None:
    """Plot per-system timelines across pulsars.

    Args:
        home_dir: Root data repository.
        dataset_name: Dataset name or path.
        out_paths: Output directory mapping from :func:`make_output_tree`.
        pulsars: Pulsar names to include.
        branch: Branch name for labeling.
        dpi: Output resolution.
    """
    system_to_data = {}  # system -> list of (pulsar_index, mjd_array)
    for p_idx, pulsar in enumerate(pulsars):
        tim_dir = home_dir / dataset_name / pulsar / "tims"
        if not tim_dir.exists():
            continue
        for timfile in sorted(tim_dir.glob("*.tim")):
            dmf = read_tim_file(timfile)
            if dmf.empty or 2 not in dmf.columns:
                continue
            mjd = pd.to_numeric(dmf[2], errors="coerce").dropna().to_numpy()
            if mjd.size == 0:
                continue
            system = timfile.stem
            system_to_data.setdefault(system, []).append((p_idx, mjd))

    systems = sorted(system_to_data.keys())
    if not systems:
        logger.warning("No systems found for TOA systemwise plot on %s", branch)
        return

    n = len(systems)
    ncols = min(6, max(1, int(math.ceil(math.sqrt(n)))))
    nrows = int(math.ceil(n / ncols))

    fig = plt.figure(figsize=(4*ncols, 3*nrows))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.3, hspace=0.6)

    for i, system in enumerate(systems):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        for p_idx, mjd in system_to_data[system]:
            ax.plot(mjd, np.full_like(mjd, p_idx, dtype=float), linestyle="none", marker=".", markersize=2)
        ax.set_title(system, fontsize=10)
        ax.set_ylim(-0.5, len(pulsars) - 0.5)
        ax.grid(alpha=0.2)

    fig.suptitle(f"Pulsars per system on {branch}", y=0.995)
    savefig(fig, out_paths["png"] / f"PulsarsPerSystem_{branch}.png", dpi=dpi)

def plot_covmat_heatmaps(out_paths: Dict[str, Path], pulsars: List[str], branches: List[str], dpi: int, max_params: Optional[int] = None) -> None:
    """Plot covariance matrix heatmaps per pulsar/branch.

    Args:
        out_paths: Output directory mapping from :func:`make_output_tree`.
        pulsars: Pulsar names to include.
        branches: Branches to include.
        dpi: Output resolution.
        max_params: Optional maximum number of parameters to display.
    """
    for pulsar in pulsars:
        for branch in branches:
            cov_file = out_paths["covmat"] / f"{pulsar}_{branch}.covmat"
            if not cov_file.exists():
                continue
            try:
                df = read_covmat(cov_file)
            except Exception as e:
                logger.warning("Failed to read covmat for %s on %s: %s", pulsar, branch, e)
                continue

            if max_params is not None and df.shape[0] > max_params:
                df = df.iloc[:max_params, :max_params]

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)

            if HAVE_SEABORN:
                cmap = sns.diverging_palette(250, 30, l=65, center="light", as_cmap=True)  # type: ignore
                sns.heatmap(df, ax=ax, cmap=cmap, xticklabels=True, yticklabels=True)  # type: ignore
            else:
                im = ax.imshow(df.values, aspect="auto")
                fig.colorbar(im, ax=ax)
                ax.set_xticks(range(df.shape[1]))
                ax.set_yticks(range(df.shape[0]))
                ax.set_xticklabels(df.columns, rotation=90, fontsize=6)
                ax.set_yticklabels(df.index, fontsize=6)

            ax.set_title(f"Covariance matrix: {pulsar} on {branch}")
            savefig(fig, out_paths["png"] / f"CovMat_{pulsar}_{branch}.png", dpi=dpi)

def plot_residuals(out_paths: Dict[str, Path], pulsars: List[str], branches: List[str], dpi: int) -> None:
    """Plot timing residuals per pulsar/branch.

    Args:
        out_paths: Output directory mapping from :func:`make_output_tree`.
        pulsars: Pulsar names to include.
        branches: Branches to include.
        dpi: Output resolution.
    """
    summary_path = out_paths["tag"] / "residual_summary.tsv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("branch\tpulsar\twrms_post\tq32_post\tq68_post\tn\n")

    for pulsar in pulsars:
        for branch in branches:
            gen_file = out_paths["general2"] / f"{pulsar}_{branch}.general2"
            if not gen_file.exists():
                continue

            try:
                df = read_general2(gen_file)
            except Exception as e:
                logger.warning("Failed to read general2 for %s on %s: %s", pulsar, branch, e)
                continue

            if not {"sat","pre","post","err"}.issubset(set(df.columns)):
                logger.warning("general2 missing required columns for %s on %s", pulsar, branch)
                continue

            sat = pd.to_numeric(df["sat"], errors="coerce")
            pre = pd.to_numeric(df["pre"], errors="coerce")
            post = pd.to_numeric(df["post"], errors="coerce")
            err = pd.to_numeric(df["err"], errors="coerce") * 1e-6  # μs -> s (heuristic)

            good = sat.notna() & pre.notna() & post.notna() & err.notna()
            if good.sum() < 2:
                continue

            sat = sat[good].to_numpy()
            pre = pre[good].to_numpy()
            post = post[good].to_numpy()
            err = err[good].to_numpy()

            w = np.where(err > 0, 1.0 / (err**2), np.nan)
            w = np.where(np.isfinite(w), w, 0.0)
            if w.sum() > 0:
                mu = np.sum(w * post) / np.sum(w)
                wrms = np.sqrt(np.sum(w * (post - mu) ** 2) / np.sum(w))
            else:
                wrms = float(np.std(post))

            q32, q68 = np.quantile(post, [0.32, 0.68])

            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(f"{branch}\t{pulsar}\t{wrms:.6e}\t{q32:.6e}\t{q68:.6e}\t{post.size}\n")

            fig1 = plt.figure(figsize=(8, 5))
            ax1 = fig1.add_subplot(111)
            ax1.hist(pre, bins=freedman_diaconis_bins(pre), density=True, alpha=0.6, label="pre-fit")
            ax1.hist(post, bins=freedman_diaconis_bins(post), density=True, alpha=0.6, label="post-fit")
            ax1.set_title(f"Residual distribution: {pulsar} on {branch}")
            ax1.set_xlabel("residual")
            ax1.set_ylabel("density")
            ax1.legend()
            savefig(fig1, out_paths["png"] / f"ResidualHist_{pulsar}_{branch}.png", dpi=dpi)

            fig2 = plt.figure(figsize=(10, 5))
            ax2 = fig2.add_subplot(111)
            ax2.errorbar(sat, post, yerr=err, fmt="none", linewidth=0.5, alpha=0.8)
            ax2.scatter(sat, pre, s=6, alpha=0.35, label="pre-fit")
            outliers = np.abs(post) >= (3.0 * np.std(post))
            if outliers.any():
                ax2.scatter(sat[outliers], post[outliers], s=20, facecolors="none", edgecolors="red", linewidths=0.8, label="3σ outliers")
            ax2.set_title(f"Residuals vs time: {pulsar} on {branch}")
            ax2.set_xlabel("MJD (sat)")
            ax2.yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
            ax2.grid(alpha=0.25)
            ax2.legend()
            savefig(fig2, out_paths["png"] / f"ResidualsVsTime_{pulsar}_{branch}.png", dpi=dpi)
