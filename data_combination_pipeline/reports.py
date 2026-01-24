from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from functools import lru_cache
import numpy as np
import pandas as pd

from .parsers import read_plklog, read_general2, read_tim_file
from .logging_utils import get_logger

logger = get_logger("data_combination_pipeline.reports")

def _chi2_sf(x: float, df: float) -> float:
    """Survival function for chi-square distribution.

    Uses SciPy if available, otherwise falls back to mpmath.
    """
    if x is None or df is None:
        return float("nan")
    try:
        from scipy.stats import chi2  # type: ignore
        return float(chi2.sf(x, df))
    except Exception:
        try:
            import mpmath as mp  # type: ignore
            # sf = Q(k/2, x/2)
            return float(mp.gammainc(df / 2.0, x / 2.0, mp.inf) / mp.gamma(df / 2.0))
        except Exception:
            return float("nan")




@lru_cache(maxsize=512)
def _read_plk_cached(path_str: str) -> pd.DataFrame:
    """Read a .plk log once per process.

    The change report loops can otherwise re-read the same reference file many times.
    """
    return read_plklog(Path(path_str))


def _maybe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _mjd_day_int(series: pd.Series) -> pd.Series:
    """Return integer MJD day as pandas nullable Int64.

    Accepts fractional MJDs and non-numeric tokens; non-numeric become <NA>.
    """
    s = pd.to_numeric(series, errors="coerce")
    return np.floor(s).astype("Int64")



def _hms_to_seconds(hms: str) -> Optional[float]:
    """Parse (+/-)HH:MM:SS(.sss) or (+/-)DD:MM:SS(.sss) into seconds."""
    if not isinstance(hms, str) or ":" not in hms:
        return None
    parts = hms.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        h0 = parts[0]
        sign = -1.0 if h0.startswith("-") else 1.0
        hh = abs(float(h0))
        mm = float(parts[1])
        ss = float(parts[2])
        return sign * (hh * 3600.0 + mm * 60.0 + ss)
    except Exception:
        return None


def _format_seconds_as_hms(seconds: float) -> str:
    sign = "+" if seconds >= 0 else "-"
    s = abs(float(seconds))
    hh = int(s // 3600)
    s -= hh * 3600
    mm = int(s // 60)
    s -= mm * 60
    return f"{sign}{hh:d}:{mm:02d}:{s:012.9f}"


def _parse_plk_stats(plk_path: Path) -> Dict[str, Optional[float]]:
    """Extract fit statistics from tempo2 stdout captured in the plk log.

    tempo2 output formats vary, so we use tolerant regexes and return None if absent.
    """
    text = plk_path.read_text(encoding="utf-8", errors="ignore") if plk_path.exists() else ""
    if not text:
        return {"chisq": None, "redchisq": None, "n_toas": None}

    def _rx(patterns: List[str]) -> Optional[float]:
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    continue
        return None

    chisq = _rx([
        r"chisq\s*=\s*([0-9.+\-eE]+)",
        r"chi\s*\^?2\s*=\s*([0-9.+\-eE]+)",
    ])
    redchisq = _rx([
        r"reduced\s*chisq\s*=\s*([0-9.+\-eE]+)",
        r"red\s*chisq\s*=\s*([0-9.+\-eE]+)",
    ])
    n_toas = _rx([
        r"number\s+of\s+\w*points\s+in\s+fit\s*=\s*([0-9.+\-eE]+)",
        r"ntoa\s*=\s*([0-9.+\-eE]+)",
        r"number\s+of\s+toas\s*=\s*([0-9.+\-eE]+)",
    ])
    return {"chisq": chisq, "redchisq": redchisq, "n_toas": n_toas}


def compare_plk(branch: str, reference_branch: str, pulsar: str, out_paths: Dict[str, Path]) -> pd.DataFrame:
    """Compare post-fit parameters between a branch and a reference branch.

    Optimizations vs the original notebook:
      * cache file reads (huge when reference is compared against many branches)
      * vectorized merge instead of Python loops
      * add uncertainty-normalized "sigma" deltas when possible
    """
    ref_path = out_paths["plk"] / f"{pulsar}_{reference_branch}_plk.log"
    br_path = out_paths["plk"] / f"{pulsar}_{branch}_plk.log"

    df_ref = _read_plk_cached(str(ref_path))
    df_br = _read_plk_cached(str(br_path))

    keep = [c for c in ["Param", "Postfit", "Uncertainty", "Fit"] if c in df_ref.columns]
    df_ref = df_ref[keep].copy()
    df_br = df_br[[c for c in keep if c in df_br.columns]].copy()

    m = df_ref.merge(df_br, on="Param", how="outer", suffixes=("_ref", "_branch"), indicator=True)
    m["status"] = m["_merge"].map({"both": "both", "left_only": "missing", "right_only": "new"})
    m = m.drop(columns=["_merge"])

    # numeric diffs
    ref_num = _maybe_float_series(m.get("Postfit_ref", pd.Series(dtype=float)))
    br_num = _maybe_float_series(m.get("Postfit_branch", pd.Series(dtype=float)))
    diff_num = br_num - ref_num

    # hms diffs (RA/DEC style)
    ref_hms = m.get("Postfit_ref", pd.Series(dtype=object)).astype("string")
    br_hms = m.get("Postfit_branch", pd.Series(dtype=object)).astype("string")
    ref_sec = ref_hms.apply(lambda x: _hms_to_seconds(str(x)) if x is not pd.NA else None)
    br_sec = br_hms.apply(lambda x: _hms_to_seconds(str(x)) if x is not pd.NA else None)
    diff_sec = (pd.to_numeric(br_sec, errors="coerce") - pd.to_numeric(ref_sec, errors="coerce"))
    diff_hms_str = diff_sec.apply(lambda x: _format_seconds_as_hms(float(x)) if pd.notna(x) else pd.NA)

    # uncertainty-normalized significance (numeric only)
    uref = _maybe_float_series(m.get("Uncertainty_ref", pd.Series(dtype=float)))
    ubr = _maybe_float_series(m.get("Uncertainty_branch", pd.Series(dtype=float)))
    denom = np.sqrt(uref**2 + ubr**2)
    sigma = np.where((np.isfinite(diff_num)) & (denom > 0), np.abs(diff_num) / denom, np.nan)

    # Choose diff representation: HMS if applicable, else numeric, else string
    diff_display = diff_num.astype("float64")
    diff_display = diff_display.where(pd.notna(diff_display), pd.NA)
    diff_display = diff_display.astype("object")
    use_hms = diff_hms_str.notna()
    diff_display = diff_display.where(~use_hms, diff_hms_str.astype("object"))

    # for truly non-numeric values, fall back to "branch | ref" for readability
    needs_fallback = pd.isna(diff_display) & m["Postfit_branch"].notna() & m["Postfit_ref"].notna()
    diff_display = diff_display.where(~needs_fallback, (m["Postfit_branch"].astype(str) + " | " + m["Postfit_ref"].astype(str)))

    out = pd.DataFrame({
        "Param": m["Param"],
        "status": m["status"],
        "ref_postfit": m.get("Postfit_ref"),
        "branch_postfit": m.get("Postfit_branch"),
        "ref_uncertainty": m.get("Uncertainty_ref"),
        "branch_uncertainty": m.get("Uncertainty_branch"),
        "diff": diff_display,
        "sigma": sigma,
        "ref_fit": m.get("Fit_ref"),
        "branch_fit": m.get("Fit_branch"),
    })

    # stable order
    return out.sort_values(["status", "Param"], kind="mergesort", ignore_index=True)

def write_change_reports(out_paths: Dict[str, Path], pulsars: List[str], branches: List[str], reference_branch: str) -> None:
    combined = []
    for branch in branches:
        if branch == reference_branch:
            continue
        for pulsar in pulsars:
            try:
                df = compare_plk(branch, reference_branch, pulsar, out_paths)
            except FileNotFoundError:
                logger.warning("Missing plk logs for change report: %s (%s vs %s)", pulsar, branch, reference_branch)
                continue
            out_file = out_paths["change_report"] / f"{pulsar}_change_{reference_branch}_to_{branch}.tsv"
            df.to_csv(out_file, sep="\t", index=False)
            df2 = df.copy()
            df2.insert(0, "pulsar", pulsar)
            df2.insert(1, "branch", branch)
            combined.append(df2)

    if combined:
        combined_df = pd.concat(combined, ignore_index=True)
        combined_df.to_csv(out_paths["change_report"] / f"ALL_change_{reference_branch}_summary.tsv", sep="\t", index=False)


def _fit_params_count(plk_df: pd.DataFrame) -> Optional[int]:
    if "Fit" not in plk_df.columns:
        return None
    s = plk_df["Fit"].astype(str).str.strip().str.lower()
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        return int((num.fillna(0) > 0).sum())
    # fallback: treat common truthy markers as fitted
    return int(s.isin({"t", "true", "y", "yes", "fit", "fitted", "1"}).sum())


def summarize_run(out_paths: Dict[str, Path], pulsar: str, branch: str) -> Dict[str, Optional[float]]:
    """Summarize a tempo2 run to support model comparison.

    Returns chisq/redchisq if parseable, number of TOAs, number of fitted params,
    and WRMS of post-fit residuals (heuristic units depending on your tempo2 build/plugin).
    """
    plk_path = out_paths["plk"] / f"{pulsar}_{branch}_plk.log"
    gen_path = out_paths["general2"] / f"{pulsar}_{branch}.general2"

    stats = _parse_plk_stats(plk_path)

    # Table-derived info
    try:
        plk_df = _read_plk_cached(str(plk_path))
    except Exception:
        plk_df = pd.DataFrame()
    k = _fit_params_count(plk_df)

    # n from general2 if missing
    n = stats.get("n_toas")
    wrms = None
    if gen_path.exists():
        try:
            df = read_general2(gen_path)
            if n is None:
                n = float(len(df))
            if {"post", "err"}.issubset(df.columns):
                post = pd.to_numeric(df["post"], errors="coerce")
                err = pd.to_numeric(df["err"], errors="coerce")
                good = post.notna() & err.notna() & (err > 0)
                if good.sum() > 1:
                    y = post[good].to_numpy(dtype=float)
                    # general2 often reports err in microseconds; keep the same heuristic as plotting.py
                    e = (err[good].to_numpy(dtype=float) * 1e-6)
                    w = 1.0 / (e**2)
                    mu = np.sum(w * y) / np.sum(w)
                    wrms = float(np.sqrt(np.sum(w * (y - mu) ** 2) / np.sum(w)))
        except Exception:
            pass

    out = {
        "chisq": stats.get("chisq"),
        "redchisq": stats.get("redchisq"),
        "n_toas": n,
        "k_fit": float(k) if k is not None else None,
        "wrms_post": wrms,
    }

    # model selection heuristics (Gaussian residual assumption)
    if out["chisq"] is not None and out["k_fit"] is not None:
        out["aic"] = float(out["chisq"] + 2.0 * out["k_fit"])  # type: ignore
        if out["n_toas"] is not None and out["n_toas"] > 0:
            out["bic"] = float(out["chisq"] + out["k_fit"] * np.log(out["n_toas"]))  # type: ignore
        else:
            out["bic"] = None
    else:
        out["aic"] = None
        out["bic"] = None

    return out


def write_model_comparison_summary(out_paths: Dict[str, Path], pulsars: List[str], branches: List[str], reference_branch: str) -> None:
    """Write a per-pulsar model comparison summary table vs a reference branch."""
    rows = []
    for pulsar in pulsars:
        ref_stats = summarize_run(out_paths, pulsar, reference_branch)
        for branch in branches:
            if branch == reference_branch:
                continue
            st = summarize_run(out_paths, pulsar, branch)
            row = {
                "pulsar": pulsar,
                "branch": branch,
                "reference": reference_branch,
                **{f"ref_{k}": v for k, v in ref_stats.items()},
                **{f"br_{k}": v for k, v in st.items()},
            }

            # deltas
            for k in ["chisq", "redchisq", "wrms_post", "aic", "bic"]:
                rv, bv = ref_stats.get(k), st.get(k)
                row[f"delta_{k}"] = (bv - rv) if (rv is not None and bv is not None) else None

            # Rapid nested-model test heuristic:
            # If the branch model is an extension of the reference, then
            # Δχ² = χ²_ref - χ²_branch ~ χ²(df=Δk) under standard assumptions.
            rk = ref_stats.get("k_fit")
            bk = st.get("k_fit")
            rc = ref_stats.get("chisq")
            bc = st.get("chisq")

            delta_k = (bk - rk) if (rk is not None and bk is not None) else None
            delta_chisq_improve = (rc - bc) if (rc is not None and bc is not None) else None

            row["delta_k_fit"] = delta_k
            row["lrt_delta_chisq"] = delta_chisq_improve
            if (delta_k is not None) and (delta_chisq_improve is not None) and (delta_k > 0) and (delta_chisq_improve > 0):
                row["lrt_p_value"] = _chi2_sf(float(delta_chisq_improve), float(delta_k))
            else:
                row["lrt_p_value"] = None

            rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    out_file = out_paths["change_report"] / f"MODEL_COMPARISON_{reference_branch}.tsv"
    df.to_csv(out_file, sep="	", index=False)

def write_new_param_significance(out_paths: Dict[str, Path], pulsars: List[str], branches: List[str], reference_branch: str, z_threshold: float = 3.0) -> None:
    """Summarize 'new' parameters in each branch vs reference and their Wald z = |x|/σ.

    This is a rapid way to screen whether newly-added fitted parameters look significant,
    without running extra tempo2 fits.
    """
    rows = []
    for pulsar in pulsars:
        ref_path = out_paths["plk"] / f"{pulsar}_{reference_branch}_plk.log"
        if not ref_path.exists():
            continue
        for branch in branches:
            if branch == reference_branch:
                continue
            br_path = out_paths["plk"] / f"{pulsar}_{branch}_plk.log"
            if not br_path.exists():
                continue

            df = compare_plk(branch, reference_branch, pulsar, out_paths)
            if df.empty:
                continue

            # Consider only parameters that are new in the branch and appear to be fitted
            # (Fit column can be missing or non-numeric depending on tempo2 build).
            new_df = df[df.get("status") == "new"].copy()

            # Pull branch postfit/unc
            post = pd.to_numeric(new_df.get("Postfit_branch"), errors="coerce")
            unc = pd.to_numeric(new_df.get("Uncertainty_branch"), errors="coerce")
            z = (post.abs() / unc).where((unc > 0) & post.notna(), np.nan)

            n_new = int(new_df.shape[0])
            n_z = int(np.isfinite(z).sum())
            n_sig = int((z >= float(z_threshold)).sum()) if n_z else 0

            max_z = float(np.nanmax(z.to_numpy())) if n_z else None
            if max_z is not None and np.isfinite(max_z):
                max_param = str(new_df.loc[z.idxmax(), "Param"]) if z.idxmax() in new_df.index else None
            else:
                max_param = None

            rows.append({
                "pulsar": pulsar,
                "branch": branch,
                "reference": reference_branch,
                "n_new_params": n_new,
                "n_new_with_numeric_sigma": n_z,
                f"n_new_sig_z>={float(z_threshold):g}": n_sig,
                "max_new_param_z": max_z,
                "max_new_param": max_param,
            })

    if not rows:
        return

    out = pd.DataFrame(rows)
    out_file = out_paths["change_report"] / f"NEW_PARAM_SIGNIFICANCE_{reference_branch}.tsv"
    out.to_csv(out_file, sep="\t", index=False)


def write_outlier_tables(home_dir: Path, dataset_name: Path, out_paths: Dict[str, Path], pulsars: List[str], branches: List[str]) -> None:
    for pulsar in pulsars:
        tim_dir = home_dir / pulsar / "tims"
        tim_lookup = []
        if tim_dir.exists():
            for timfile in sorted(tim_dir.glob("*.tim")):
                dmf = read_tim_file(timfile)
                if dmf.empty or 2 not in dmf.columns:
                    continue
                mjd = pd.to_numeric(dmf[2], errors="coerce").dropna()
                if mjd.empty:
                    continue
                tmp = pd.DataFrame({
                    "mjd_int": _mjd_day_int(mjd),
                    "system": timfile.stem,
                    "timfile": str(timfile),
                })
                tim_lookup.append(tmp)
        tim_lookup_df = pd.concat(tim_lookup, ignore_index=True) if tim_lookup else pd.DataFrame({"mjd_int": pd.Series(dtype="Int64"), "system": pd.Series(dtype="string"), "timfile": pd.Series(dtype="string")})

        for branch in branches:
            gen_file = out_paths["general2"] / f"{pulsar}_{branch}.general2"
            if not gen_file.exists():
                continue
            try:
                df = read_general2(gen_file)
            except Exception as e:
                logger.warning("Failed to read general2 for outliers: %s on %s (%s)", pulsar, branch, e)
                continue

            if "sat" not in df.columns or "post" not in df.columns:
                continue

            df["mjd_int"] = _mjd_day_int(df["sat"])
            df["post_num"] = pd.to_numeric(df["post"], errors="coerce")

            good = df["mjd_int"].notna() & df["post_num"].notna()
            keep_cols = ["mjd_int", "post_num"] + [c for c in df.columns if c in ("err", "freq", "solarangle", "pre")]
            df = df.loc[good, keep_cols]

            if df.empty:
                continue

            sigma = float(df["post_num"].std())
            if not np.isfinite(sigma) or sigma == 0:
                continue

            outliers = df[np.abs(df["post_num"]) >= 3.0 * sigma].copy()
            if outliers.empty:
                continue

            if not tim_lookup_df.empty:
                outliers = outliers.merge(tim_lookup_df, how="left", on="mjd_int")

            out_file = out_paths["outliers"] / f"PSR{pulsar}_{branch}_Outliers.tsv"
            outliers.to_csv(out_file, sep="\t", index=False)
