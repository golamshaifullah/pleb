"""Human-readable optimization summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .models import OptimizationResult


def write_markdown_report(result: OptimizationResult) -> Path:
    """Write a compact Markdown report for an optimization study."""
    out_path = Path(result.out_dir) / "report.md"
    lines = [
        f"# Optimization Report: {result.config.study_name}",
        "",
        f"- Trials: {len(result.trials)}",
        f"- Sampler: `{result.config.sampler}`",
        f"- Execution mode: `{result.config.execution_mode}`",
    ]
    best = result.best_trial
    if best is None:
        lines.extend(["", "No successful trial was produced."])
    else:
        lines.extend(
            [
                "",
                "## Best Trial",
                "",
                f"- Trial ID: `{best.trial_id}`",
                f"- Score: `{best.score}`",
                f"- Run dir: `{best.run_dir}`",
                "",
                "## Parameters",
                "",
            ]
        )
        for key, value in sorted(best.params.items()):
            lines.append(f"- `{key}` = `{value}`")
        lines.extend(["", "## Metrics", ""])
        for key, value in sorted(best.metrics.items()):
            lines.append(f"- `{key}` = `{value}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_pdf_report(result: OptimizationResult) -> Optional[Path]:
    """Write a PDF optimization report when plotting dependencies are available."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        return None

    trials = [trial for trial in result.trials if trial.run_dir is not None]
    if not trials:
        return None

    baseline_df, baseline_label = _load_baseline(result)
    pdf_path = Path(result.out_dir) / "report.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title(f"Optimization Report: {result.config.study_name}", fontsize=16)
        summary_rows = []
        if baseline_df is not None:
            summary_rows.append(_summary_row("baseline", baseline_df, score=None))
        for trial in trials:
            trial_df = _load_trial_frame(trial.run_dir)
            if trial_df is None:
                continue
            summary_rows.append(
                _summary_row(f"trial_{trial.trial_id:04d}", trial_df, score=trial.score)
            )
        if summary_rows:
            sdf = pd.DataFrame(summary_rows)
            cols = [
                "label",
                "score",
                "n_toas",
                "n_bad",
                "n_clean",
                "clean_rms",
                "n_events",
            ]
            show = sdf[cols].copy()
            table = ax.table(
                cellText=show.values.tolist(),
                colLabels=cols,
                loc="upper left",
                cellLoc="left",
                colLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.2)
        else:
            ax.text(0.03, 0.92, "No trial QC CSV files were available.", va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for trial in trials:
            trial_df = _load_trial_frame(trial.run_dir)
            if trial_df is None:
                continue
            trial_label = f"trial_{trial.trial_id:04d}"
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle(
                (
                    f"{trial_label} vs baseline"
                    if baseline_df is not None
                    else trial_label
                ),
                fontsize=14,
            )
            left_df = baseline_df if baseline_df is not None else trial_df
            left_label = baseline_label if baseline_df is not None else trial_label
            _plot_clean_residuals_mjd(axes[0, 0], left_df, left_label)
            _plot_clean_residuals_mjd(axes[0, 1], trial_df, trial_label)
            _plot_clean_residuals_sigma(axes[1, 0], left_df, left_label)
            _plot_clean_residuals_sigma(axes[1, 1], trial_df, trial_label)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.axis("off")
            rows = []
            if baseline_df is not None:
                rows.append(_summary_row(left_label, left_df, score=None))
            rows.append(_summary_row(trial_label, trial_df, score=trial.score))
            tdf = pd.DataFrame(rows)
            cols = [
                "label",
                "score",
                "n_toas",
                "n_bad",
                "n_clean",
                "clean_rms",
                "n_events",
            ]
            table = ax.table(
                cellText=tdf[cols].values.tolist(),
                colLabels=cols,
                loc="upper left",
                cellLoc="left",
                colLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.4)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return pdf_path


def _load_baseline(result: OptimizationResult) -> tuple[Optional[pd.DataFrame], str]:
    baseline_dir = result.config.baseline_run_dir
    if baseline_dir is None:
        baseline_dir = _discover_baseline_run_dir(result)
    if baseline_dir is None:
        return None, "baseline"
    frame = _load_trial_frame(baseline_dir)
    return frame, Path(baseline_dir).name


def _discover_baseline_run_dir(result: OptimizationResult) -> Optional[Path]:
    root = None
    fixed = result.config.fixed_overrides or {}
    results_dir = fixed.get("results_dir")
    if results_dir not in (None, ""):
        root = Path(str(results_dir))
    if root is None or not root.exists():
        return None
    trial_dirs = {
        Path(trial.run_dir).resolve()
        for trial in result.trials
        if trial.run_dir is not None
    }
    candidates = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            resolved = child.resolve()
        except Exception:
            resolved = child
        if resolved in trial_dirs:
            continue
        if list(child.glob("**/*_qc.csv")):
            candidates.append(child)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_trial_frame(run_dir: Path | str) -> Optional[pd.DataFrame]:
    paths = sorted(Path(run_dir).glob("**/*_qc.csv"))
    if not paths:
        return None
    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame["_source_csv"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) != 0.0
    return s.fillna(False).astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})


def _bad_mask(df: pd.DataFrame) -> pd.Series:
    cols = [
        "bad_point",
        "robust_outlier",
        "robust_global_outlier",
        "bad_mad",
        "bad_ou",
        "bad_hard",
        "bad",
    ]
    mask = pd.Series([False] * len(df), index=df.index)
    for col in cols:
        mask |= _bool_series(df, col)
    return mask


def _event_count(df: pd.DataFrame) -> int:
    if "event_type" in df.columns:
        return int(df["event_type"].notna().sum())
    for col in (
        "event_member",
        "transient_member",
        "gaussian_bump_member",
        "glitch_member",
        "eclipse_member",
        "solar_event_member",
    ):
        if col in df.columns:
            return int(_bool_series(df, col).sum())
    return 0


def _summary_row(
    label: str, df: pd.DataFrame, score: Optional[float]
) -> dict[str, object]:
    bad = _bad_mask(df)
    resid = pd.to_numeric(
        df.get("resid_us", df.get("resid", pd.Series(dtype=float))), errors="coerce"
    )
    clean = resid.loc[~bad & resid.notna()].astype(float)
    clean_rms = float((clean.pow(2).mean()) ** 0.5) if not clean.empty else float("nan")
    return {
        "label": label,
        "score": None if score is None else round(float(score), 6),
        "n_toas": int(len(df)),
        "n_bad": int(bad.sum()),
        "n_clean": int((~bad).sum()),
        "clean_rms": round(clean_rms, 6) if clean.notna().any() else None,
        "n_events": _event_count(df),
    }


def _plot_clean_residuals_mjd(ax, df: pd.DataFrame, label: str) -> None:
    clean = _clean_xy(df, x_col="mjd")
    if clean.empty:
        ax.set_title(f"{label}: no clean residuals")
        return
    ax.scatter(clean["x"], clean["resid"], s=8, alpha=0.75)
    ax.set_title(f"{label}: clean residual vs MJD")
    ax.set_xlabel("MJD")
    ax.set_ylabel("residual")


def _plot_clean_residuals_sigma(ax, df: pd.DataFrame, label: str) -> None:
    sigma_col = "sigma_us" if "sigma_us" in df.columns else "sigma"
    clean = _clean_xy(df, x_col=sigma_col)
    if clean.empty:
        ax.set_title(f"{label}: no clean residuals")
        return
    ax.scatter(clean["x"], clean["resid"], s=8, alpha=0.75)
    ax.set_title(f"{label}: clean residual vs uncertainty")
    ax.set_xlabel(sigma_col)
    ax.set_ylabel("residual")


def _clean_xy(df: pd.DataFrame, *, x_col: str) -> pd.DataFrame:
    x = pd.to_numeric(df.get(x_col, pd.Series(dtype=float)), errors="coerce")
    resid = pd.to_numeric(
        df.get("resid_us", df.get("resid", pd.Series(dtype=float))), errors="coerce"
    )
    bad = _bad_mask(df)
    valid = x.notna() & resid.notna() & (~bad)
    return pd.DataFrame(
        {"x": x.loc[valid].astype(float), "resid": resid.loc[valid].astype(float)}
    )
