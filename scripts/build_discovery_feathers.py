#!/usr/bin/env python3
"""Validate EPTA-DR3 residuals and optionally write Discovery Feather files.

The default mode is report-only. Feather files are written only when
``--write-feathers`` is supplied and all residual quality gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

from pleb.tim_utils import is_toa_line, parse_tim_flags_from_line


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path("/work/git_projects/epta-dr3-in2p3")
    )
    parser.add_argument("--ref", default="release/v1.0.0")
    parser.add_argument("--release-path", default="EPTA-DR3/epta-dr3-data-pleb")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Use an existing release tree instead of git archive",
    )
    parser.add_argument("--pulsars", nargs="+", help="Only process these pulsar names")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--feather-dir", type=Path)
    parser.add_argument("--write-feathers", action="store_true")
    parser.add_argument("--allow-failed-gates", action="store_true")
    parser.add_argument(
        "--tempo2-container",
        type=Path,
        default=Path("/work/Containers/pleb_tempo2.sif"),
        help="Use this PLEB image as the authoritative residual calculator",
    )
    parser.add_argument("--bin-days", type=float, default=90.0)
    parser.add_argument("--min-bin-toas", type=int, default=8)
    parser.add_argument("--max-robust-z", type=float, default=10.0)
    parser.add_argument(
        "--enforce-isolated-outliers",
        action="store_true",
        help="Fail on isolated large residuals instead of reporting them only",
    )
    parser.add_argument("--max-bin-median-z", type=float, default=5.0)
    parser.add_argument("--max-system-median-z", type=float, default=3.0)
    parser.add_argument("--max-trend-span-z", type=float, default=3.0)
    parser.add_argument("--trend-sigma", type=float, default=5.0)
    return parser.parse_args()


def export_release(repo: Path, ref: str, release_path: str, destination: Path) -> Path:
    archive = destination / "release.tar"
    with archive.open("wb") as stream:
        subprocess.run(
            ["git", "-C", str(repo), "archive", ref, release_path],
            check=True,
            stdout=stream,
        )
    with tarfile.open(archive) as tar:
        tar.extractall(destination)
    return destination / release_path


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))
    return center, scale


def container_residuals(
    par: Path, tim: Path, image: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return site MJD, post-fit residual [s], and TOA error [s]."""
    template = "PLEBROW\\t{sat}\\t{post}\\t{err}\\n"
    command = [
        "apptainer",
        "exec",
        "--env",
        "TEMPO2=/opt/tempo2/share/tempo2",
        str(image),
        "/opt/tempo2/bin/tempo2",
        "-output",
        "general2",
        "-f",
        str(par),
        str(tim),
        "-nobs",
        "120000",
        "-epoch",
        "centre",
        "-s",
        template,
    ]
    clock_dir = par.parent.parent / "clockfiles"
    if clock_dir.is_dir():
        command.extend(["-clkdir", str(clock_dir)])
    completed = subprocess.run(
        command,
        cwd=par.parent,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = []
    for line in completed.stdout.splitlines():
        if not line.startswith("PLEBROW\t"):
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            continue
        # general2 {post} is seconds, whereas {err} is microseconds.
        rows.append((float(fields[1]), float(fields[2]), float(fields[3]) * 1e-6))
    if not rows:
        raise RuntimeError(f"{par.name}: PLEB Tempo2 returned no residual rows")
    values = np.asarray(rows, dtype=float)
    return values[:, 0], values[:, 1], values[:, 2]


def expanded_system_flags(tim: Path) -> np.ndarray:
    """Read active TOAs in Tempo2 INCLUDE order and return their -sys values."""
    systems: list[str] = []

    def walk(path: Path, stack: tuple[Path, ...] = ()) -> None:
        path = path.resolve()
        if path in stack:
            raise RuntimeError(f"recursive TIM INCLUDE involving {path}")
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = raw.strip()
            parts = stripped.split(maxsplit=1)
            if parts and parts[0].upper() == "INCLUDE":
                if len(parts) != 2:
                    raise RuntimeError(f"invalid INCLUDE in {path}: {raw}")
                walk(path.parent / parts[1], stack + (path,))
            elif is_toa_line(raw):
                flags = parse_tim_flags_from_line(raw)
                system = flags.get("-sys")
                if not system:
                    raise RuntimeError(f"active TOA without -sys in {path}: {raw}")
                systems.append(system)

    walk(tim)
    return np.asarray(systems, dtype=str)


def binned_medians(
    toas: np.ndarray, z: np.ndarray, width: float, minimum: int
) -> pd.DataFrame:
    edges = np.arange(np.floor(toas.min() / width) * width, toas.max() + width, width)
    if len(edges) < 2:
        edges = np.array([toas.min(), toas.max() + 1.0])
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (toas >= left) & (toas < right)
        if int(selected.sum()) >= minimum:
            rows.append(
                {
                    "mjd": float(np.median(toas[selected])),
                    "median_z": float(np.median(z[selected])),
                    "n_toas": int(selected.sum()),
                }
            )
    return pd.DataFrame(rows, columns=["mjd", "median_z", "n_toas"])


def diagnose(
    psr, args: argparse.Namespace, *, authoritative=None
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if authoritative is None:
        toas = np.asarray(psr.stoas, dtype=float)
        residuals = np.asarray(psr.residuals, dtype=float)
        errors = np.asarray(psr.toaerrs, dtype=float)
    else:
        toas, residuals, errors = authoritative
        if len(toas) != len(psr.flags["sys"]):
            raise RuntimeError(
                f"{psr.name}: PLEB rows ({len(toas)}) do not match TIM flags "
                f"({len(psr.flags['sys'])})"
            )
    # Enterprise stores barycentric TOAs as seconds from MJD 0. Convert to
    # days for binning and for human-readable diagnostic tables.
    if np.nanmedian(toas) > 1.0e6:
        toas = toas / 86400.0
    # Use the timing-model system flag.  ``backend_flags`` is an Enterprise
    # convenience grouping and is not guaranteed to be the value used by
    # Tempo2 JUMPs or by ``plk -colour -sys``.
    if "sys" not in psr.flags:
        raise RuntimeError(f"{psr.name}: TIM data have no -sys flag")
    systems = np.asarray(psr.flags["sys"]).astype(str)
    finite = np.isfinite(toas) & np.isfinite(residuals) & np.isfinite(errors)
    if not finite.all():
        toas, residuals, errors, systems = (
            value[finite] for value in (toas, residuals, errors, systems)
        )
    if len(toas) < 3:
        raise RuntimeError(f"{psr.name}: fewer than three finite TOAs")
    order = np.argsort(toas)
    toas, residuals, errors, systems = (
        value[order] for value in (toas, residuals, errors, systems)
    )
    center, scale = robust_location_scale(residuals)
    if scale <= 0:
        raise RuntimeError(f"{psr.name}: residual scale is zero")
    z = (residuals - center) / scale
    all_weights = 1.0 / np.square(errors)
    weighted_mean = float(np.sum(all_weights * residuals) / np.sum(all_weights))
    weighted_rms = float(
        np.sqrt(np.sum(all_weights * np.square(residuals)) / np.sum(all_weights))
    )
    weighted_centered_rms = float(
        np.sqrt(
            np.sum(all_weights * np.square(residuals - weighted_mean))
            / np.sum(all_weights)
        )
    )
    bins = binned_medians(toas, z, args.bin_days, args.min_bin_toas)
    system_rows = []
    for system in sorted(set(systems)):
        selected = systems == system
        group_residuals = residuals[selected]
        group_errors = errors[selected]
        weights = 1.0 / np.square(group_errors)
        weighted_mean = float(np.sum(weights * group_residuals) / np.sum(weights))
        system_rows.append(
            {
                "pulsar": psr.name,
                "system": system,
                "n_toas": int(selected.sum()),
                "weighted_mean_us": weighted_mean * 1e6,
                "median_residual_us": float(np.median(group_residuals) * 1e6),
                "median_robust_z": float(np.median(z[selected])),
            }
        )
    system_table = pd.DataFrame(system_rows)
    regression = stats.linregress(toas, residuals)
    span_z = abs(float(regression.slope) * float(np.ptp(toas)) / scale)
    slope_sigma = (
        abs(float(regression.slope / regression.stderr))
        if regression.stderr and np.isfinite(regression.stderr)
        else 0.0
    )
    max_bin = float(bins["median_z"].abs().max()) if not bins.empty else 0.0
    max_system_median_z = (
        float(system_table["median_robust_z"].abs().max())
        if not system_table.empty
        else 0.0
    )
    gates = {
        "finite_values": bool(finite.all()),
        "isolated_outliers": bool(
            not args.enforce_isolated_outliers or np.max(np.abs(z)) <= args.max_robust_z
        ),
        "time_binned_structure": bool(max_bin <= args.max_bin_median_z),
        "system_offset": bool(max_system_median_z <= args.max_system_median_z),
        "long_term_trend": bool(
            span_z <= args.max_trend_span_z or slope_sigma <= args.trend_sigma
        ),
    }
    result = {
        "pulsar": psr.name,
        "n_toas": int(len(toas)),
        "n_nonfinite_removed": int((~finite).sum()),
        "residual_median_us": center * 1e6,
        "residual_mad_sigma_us": scale * 1e6,
        "residual_rms_us": float(np.sqrt(np.mean(residuals**2)) * 1e6),
        "weighted_mean_us": weighted_mean * 1e6,
        "weighted_rms_us": weighted_rms * 1e6,
        "weighted_centered_rms_us": weighted_centered_rms * 1e6,
        "max_abs_robust_z": float(np.max(np.abs(z))),
        "max_abs_bin_median_z": max_bin,
        "max_abs_system_median_z": max_system_median_z,
        "trend_span_z": span_z,
        "trend_slope_sigma": slope_sigma,
        "passed": bool(all(gates.values())),
        **{f"gate_{name}": value for name, value in gates.items()},
    }
    points = pd.DataFrame(
        {
            "pulsar": psr.name,
            "mjd": toas,
            "residual_us": residuals * 1e6,
            "toa_error_us": errors * 1e6,
            "robust_z": z,
            "system": systems,
        }
    )
    bins.insert(0, "pulsar", psr.name)
    return result, points, bins, system_table


def plot_diagnostics(points: pd.DataFrame, bins: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for system, group in points.groupby("system", sort=True):
        axes[0].errorbar(
            group["mjd"],
            group["residual_us"],
            yerr=group["toa_error_us"],
            fmt=".",
            ms=2,
            alpha=0.55,
            lw=0.3,
            label=system,
        )
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_ylabel("Residual (us)")
    axes[0].set_title(str(points["pulsar"].iloc[0]))
    if points["system"].nunique() <= 15:
        axes[0].legend(fontsize=6, ncol=3)
    axes[1].plot(points["mjd"], points["robust_z"], ".", ms=2, alpha=0.4)
    if not bins.empty:
        axes[1].plot(bins["mjd"], bins["median_z"], "o-", color="tab:red", ms=3)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_ylabel("Robust z")
    axes[1].set_xlabel("MJD")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = arguments()
    if args.write_feathers and args.feather_dir is None:
        raise SystemExit("--write-feathers requires --feather-dir")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.report_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.report_dir / "mplconfig"))

    try:
        from discovery.pulsar import Pulsar as DiscoveryPulsar
        from enterprise.pulsar import Pulsar as EnterprisePulsar
    except ImportError as error:
        raise SystemExit(
            "Run with the Discovery environment, for example "
            "/work/mambaforge/envs/discovery-gpu-env/bin/python"
        ) from error

    temporary = None
    if args.data_dir:
        data_dir = args.data_dir.resolve()
    else:
        temporary = tempfile.TemporaryDirectory(prefix="epta-dr3-v1-feather-")
        data_dir = export_release(
            args.repo.resolve(), args.ref, args.release_path, Path(temporary.name)
        )

    pairs = []
    for par in sorted(data_dir.glob("J*/J*_combined.par")):
        if args.pulsars and par.parent.name not in args.pulsars:
            continue
        tim = par.with_name(par.stem.replace("_combined", "_combined_all") + ".tim")
        if not tim.exists():
            raise RuntimeError(f"missing TIM for {par}: {tim}")
        pairs.append((par, tim))
    if not pairs:
        raise RuntimeError(f"no combined par/TIM pairs found under {data_dir}")

    results, all_points, all_bins, all_systems, pulsars = [], [], [], [], []
    load_failures = []
    for par, tim in pairs:
        name = par.parent.name
        print(f"loading {name}", flush=True)
        try:
            authoritative = container_residuals(par, tim, args.tempo2_container)
            systems_from_tim = expanded_system_flags(tim)
            enterprise_psr = None
            try:
                enterprise_psr = EnterprisePulsar(str(par), str(tim), ephem="DE440")
                enterprise_psr._residuals = authoritative[1].copy()
                enterprise_psr._toaerrs = authoritative[2].copy()
                enterprise_psr._flags["sys"] = systems_from_tim
                psr = enterprise_psr
            except Exception as enterprise_error:
                if args.write_feathers:
                    raise RuntimeError(
                        f"{name}: Enterprise load required for Feather writing failed"
                    ) from enterprise_error
                print(
                    f"warning: {name}: host Enterprise load failed; "
                    "using PLEB residuals plus TIM flags for reporting",
                    file=sys.stderr,
                )
                psr = SimpleNamespace(name=name, flags={"sys": systems_from_tim})
            result, points, bins, systems = diagnose(
                psr, args, authoritative=authoritative
            )
            plot_diagnostics(points, bins, plot_dir / f"{name}.png")
            results.append(result)
            all_points.append(points)
            all_bins.append(bins)
            all_systems.append(systems)
            if enterprise_psr is not None:
                pulsars.append(enterprise_psr)
        except Exception as error:  # keep a complete release-level failure report
            load_failures.append({"pulsar": name, "error": repr(error)})

    summary = pd.DataFrame(results).sort_values("pulsar") if results else pd.DataFrame()
    summary.to_csv(args.report_dir / "residual_quality_summary.csv", index=False)
    if all_points:
        pd.concat(all_points, ignore_index=True).to_csv(
            args.report_dir / "residual_points.csv.gz", index=False
        )
        pd.concat(all_bins, ignore_index=True).to_csv(
            args.report_dir / "time_bin_medians.csv", index=False
        )
        pd.concat(all_systems, ignore_index=True).to_csv(
            args.report_dir / "system_offsets.csv", index=False
        )
    overall_pass = (
        not load_failures and not summary.empty and bool(summary["passed"].all())
    )
    report = {
        "repo": str(args.repo),
        "ref": args.ref,
        "release_path": args.release_path,
        "pairs_found": len(pairs),
        "pairs_checked": len(results),
        "failed_quality_gates": (
            summary.loc[~summary["passed"], "pulsar"].tolist()
            if not summary.empty
            else []
        ),
        "load_failures": load_failures,
        "overall_pass": overall_pass,
        "thresholds": {
            "bin_days": args.bin_days,
            "min_bin_toas": args.min_bin_toas,
            "max_robust_z": args.max_robust_z,
            "enforce_isolated_outliers": args.enforce_isolated_outliers,
            "max_bin_median_z": args.max_bin_median_z,
            "max_system_median_z": args.max_system_median_z,
            "max_trend_span_z": args.max_trend_span_z,
            "trend_sigma": args.trend_sigma,
        },
    }
    (args.report_dir / "residual_quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    if args.write_feathers:
        if not overall_pass and not args.allow_failed_gates:
            print("Quality gates failed; no Feather files written.", file=sys.stderr)
            return 2
        args.feather_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for psr in pulsars:
            output = args.feather_dir / psr.name
            DiscoveryPulsar.save_feather(psr, str(output))
            manifest.append(
                {"pulsar": psr.name, "file": output.name, "sha256": sha256(output)}
            )
        pd.DataFrame(manifest).to_csv(
            args.feather_dir / "manifest.tsv", sep="\t", index=False
        )

    if temporary is not None:
        temporary.cleanup()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
