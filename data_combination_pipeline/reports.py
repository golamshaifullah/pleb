from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .parsers import read_plklog, read_general2, read_tim_file
from .logging_utils import get_logger

logger = get_logger("data_combination_pipeline.reports")

def _as_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def get_hms_diff(hms_f: str, hms_m: str) -> str:
    def _split(s: str):
        hh, mm, ss = s.split(":")
        return float(hh), float(mm), float(ss)
    hf, mf, sf = _split(hms_f)
    hm, mm, sm = _split(hms_m)
    dh, dm, ds = (hf - hm), (mf - mm), (sf - sm)
    sign = -1 if ds < 0 else 1
    dh, dm, ds = sign*abs(dh), sign*abs(dm), sign*abs(ds)
    return f"{dh:+.0f}:{abs(dm):02.0f}:{abs(ds):012.9f}"

def compare_plk(branch: str, reference_branch: str, pulsar: str, out_paths: Dict[str, Path]) -> pd.DataFrame:
    df_ref = read_plklog(out_paths["plk"] / f"{pulsar}_{reference_branch}_plk.log")
    df_b = read_plklog(out_paths["plk"] / f"{pulsar}_{branch}_plk.log")

    df_ref = df_ref.set_index("Param")
    df_b = df_b.set_index("Param")

    params = sorted(set(df_ref.index) | set(df_b.index))
    rows = []
    for p in params:
        in_ref = p in df_ref.index
        in_b = p in df_b.index
        if in_ref and in_b:
            pf = df_b.loc[p, "Postfit"]
            pr = df_ref.loc[p, "Postfit"]
            if isinstance(pf, str) and ":" in str(pf) and isinstance(pr, str) and ":" in str(pr):
                diff = get_hms_diff(str(pf), str(pr))
            else:
                vf = _as_float(pf)
                vr = _as_float(pr)
                diff = (vf - vr) if (vf is not None and vr is not None) else f"{pf} | {pr}"
            rows.append((p, "both", pr, pf, diff))
        elif in_b and not in_ref:
            rows.append((p, "new", None, df_b.loc[p, "Postfit"], df_b.loc[p, "Postfit"]))
        else:
            rows.append((p, "missing", df_ref.loc[p, "Postfit"], None, None))

    return pd.DataFrame(rows, columns=["Param", "status", "ref_postfit", "branch_postfit", "diff"])

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

def write_outlier_tables(home_dir: Path, out_paths: Dict[str, Path], pulsars: List[str], branches: List[str]) -> None:
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
                    "mjd_int": mjd.astype(int).astype(float),
                    "system": timfile.stem,
                    "timfile": str(timfile),
                })
                tim_lookup.append(tmp)
        tim_lookup_df = pd.concat(tim_lookup, ignore_index=True) if tim_lookup else pd.DataFrame(columns=["mjd_int","system","timfile"])

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

            df["mjd_int"] = pd.to_numeric(df["sat"], errors="coerce").astype("Int64")
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
