#!/usr/bin/env python3
"""Clock-audited Tempo2 fits and row-validated Discovery Feather exports.

Run inside the DR3 Discovery SIF; input and output roots are explicit.
"""

from pathlib import Path
from collections import Counter
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
import re
import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_runtime(data, root, utc_config=None):
    stock = Path("/opt/tempo2/share/tempo2")
    runtime = root / "runtime/tempo2"
    if not runtime.exists():
        runtime.mkdir(parents=True)
        for p in stock.iterdir():
            if p.name in {"clock", "observatory"}:
                shutil.copytree(p, runtime / p.name)
            else:
                (runtime / p.name).symlink_to(p, target_is_directory=p.is_dir())
    overrides = {}
    for p in sorted((data / "clockfiles").iterdir()):
        if p.is_file():
            shutil.copy2(p, runtime / "clock" / p.name)
            overrides[p.name] = dict(source=str(p), sha256=sha(p))
    explicit_utc = {}
    if utc_config:
        policy = json.loads(utc_config.read_text())
        lo, hi = policy["mjd_range"]
        for station in policy["station_clocks"]:
            assert re.fullmatch(r"[A-Za-z0-9_]+", station), station
            name = f"{station}2utc_explicit.clk"
            path = runtime / "clock" / name
            content = f'# UTC({station}) UTC\n# {policy["description"]}\n{lo:.1f} 0.0\n{hi:.1f} 0.0\n'
            assert (
                name not in overrides or path.read_text() == content
            ), "Cannot replace a different dataset-supplied clock"
            path.write_text(content)
            explicit_utc[name] = dict(
                station_clock=f"UTC({station})",
                target="UTC",
                correction_seconds=0.0,
                mjd_range=[lo, hi],
                sha256=sha(path),
            )
    manifest = dict(
        stock_runtime=str(stock),
        dataset_overrides=overrides,
        explicit_utc_policy=json.loads(utc_config.read_text()) if utc_config else None,
        explicit_utc_files=explicit_utc,
        resolved_clock_files={
            p.name: sha(p) for p in (runtime / "clock").iterdir() if p.is_file()
        },
    )
    (root / "reports/clock_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return runtime


def configure(root):
    for name in ["logs", "cache", "config", "matplotlib"]:
        (root / "runtime" / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(
        TEMPO2=str(root / "runtime/tempo2"),
        PLEB_LOG_DIR=str(root / "runtime/logs"),
        XDG_CONFIG_HOME=str(root / "runtime/config"),
        XDG_CACHE_HOME=str(root / "runtime/cache"),
        MPLCONFIGDIR=str(root / "runtime/matplotlib"),
        MPLBACKEND="Agg",
    )


def stats(text):
    k = re.findall(r"Number of fit parameters:\s*(\d+)", text)
    n = re.findall(r"Number of points in fit\s*=\s*(\d+)", text)
    v = re.findall(
        r"Chisqr/nfree\s*=\s*([\d.eE+\-]+)\s*/\s*(\d+)\s*=\s*([\d.eE+\-]+)", text
    )
    if not (k and n):
        raise ValueError("Tempo2 did not report valid fit statistics")
    result = dict(n=int(n[-1]), k=int(k[-1]), weighted_fit=bool(v))
    if v:
        chi, dof, red = v[-1]
        result.update(chisq=float(chi), dof=int(dof), red_chi2=float(red))
        assert result["n"] - result["k"] == result["dof"] and np.isfinite(
            result["chisq"]
        ), result
    else:
        result.update(chisq=None, dof=result["n"] - result["k"], red_chi2=None)
    return result


def tempo(par, tim, folder, root, fit=True):
    folder.mkdir(parents=True, exist_ok=True)
    command = [
        "tempo2",
        "-f",
        str(par),
        str(tim),
        "-nobs",
        "120000",
        "-clkdir",
        str(root / "runtime/tempo2/clock") + "/",
        "-showchisq",
    ]
    command += ["-newpar"] if fit else ["-nofit"]
    env = dict(
        os.environ,
        CLOCK_TRACE=str(folder / "clock_opens.tsv"),
        LD_PRELOAD=str(root / "runtime/clocktrace.so"),
    )
    out = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True)
    text = out.stdout + "\n" + out.stderr
    (folder / "tempo2.log").write_text(text)
    (folder / "command.json").write_text(json.dumps(command, indent=2))
    if out.returncode or re.search(r"ERROR:", text, re.I):
        raise RuntimeError("Tempo2 failed: " + str(folder / "tempo2.log"))
    return folder / "new.par" if fit else par, stats(text) if fit else {}


def export_cli(par, tim, folder, root):
    cmd = [
        "tempo2",
        "-f",
        str(par),
        str(tim),
        "-nobs",
        "120000",
        "-nofit",
        "-clkdir",
        str(root / "runtime/tempo2/clock") + "/",
        "-output",
        "general2",
        "-s",
        r"DR3 {sat} {post} {err} {freq} {bat} |{clkchain}|{clk_corrA}\n",
    ]
    out = subprocess.run(
        cmd,
        cwd=folder,
        env=dict(
            os.environ,
            CLOCK_TRACE=str(folder / "clock_opens.tsv"),
            LD_PRELOAD=str(root / "runtime/clocktrace.so"),
        ),
        capture_output=True,
        text=True,
    )
    text = out.stdout + "\n" + out.stderr
    (folder / "general2.log").write_text(text)
    if out.returncode or "ERROR:" in text:
        raise RuntimeError(str(folder / "general2.log"))
    lines = [line for line in text.splitlines() if line.startswith("DR3 ")]
    rows = [list(map(float, line.split()[1:6])) for line in lines]
    chains = Counter()
    explicit = json.loads((root / "reports/clock_manifest.json").read_text())[
        "explicit_utc_files"
    ]
    sites = {item["station_clock"][4:-1].lower() for item in explicit.values()}
    for line in lines:
        _, chain, corrections = line.split("|")
        chains[chain] += 1
        if chain.split("->")[0].lower() in sites:
            assert chain.split("->")[1] == "UTC", chain
            assert float(corrections.split(",")[0]) == 0.0, (chain, corrections)
    (folder / "clock_chains.json").write_text(json.dumps(dict(chains), indent=2) + "\n")
    a = np.array(rows)
    assert len(a) and a.shape[1] == 5 and np.isfinite(a).all()
    return a


def metric(r, sigma):
    w = 1 / sigma**2
    mean = np.sum(w * r) / np.sum(w)
    return dict(
        n=len(r),
        weighted_mean_s=float(mean),
        wrms_us=float(np.sqrt(np.average((r - mean) ** 2, weights=w)) * 1e6),
        rms_us=float(np.std(r) * 1e6),
    )


def tim_metadata(tim, stack=()):
    from pleb.tim_utils import is_toa_line, parse_tim_flags_from_line

    tim = tim.resolve()
    if tim in stack:
        raise ValueError("Recursive INCLUDE")
    result = []
    skip = False
    time_offset = 0.0
    for raw in tim.read_text().splitlines():
        parts = raw.split()
        if not parts or raw.lstrip().startswith(("C", "#")):
            continue
        directive = parts[0].upper()
        if directive == "END":
            break
        if directive == "SKIP":
            skip = True
            continue
        if directive == "NOSKIP":
            skip = False
            continue
        if directive == "TIME":
            time_offset += float(parts[1])
            continue
        if skip:
            continue
        if directive == "INCLUDE":
            result.extend(tim_metadata(tim.parent / parts[1], stack + (tim,)))
            continue
        if is_toa_line(raw):
            try:
                frequency = float(parts[1])
                float(parts[2])
                float(parts[3])
            except (ValueError, IndexError):
                continue
            flags = parse_tim_flags_from_line(raw)
            result.append(
                dict(
                    filename=parts[0],
                    frequency=frequency,
                    mjd=float(
                        np.longdouble(parts[2])
                        + (
                            np.longdouble(time_offset)
                            + np.longdouble(flags.get("-addsat", "0"))
                        )
                        / 86400
                    ),
                    telescope=parts[4],
                    system=flags.get("-sys", ""),
                )
            )
    return result


def legacy_metadata(call):
    """Translate the removed byte-string dtype alias only during a metadata call."""
    original = np.zeros

    def zeros(shape, dtype=float, *args, **kwargs):
        if isinstance(dtype, str) and re.fullmatch(r"a\d+", dtype):
            dtype = "S" + dtype[1:]
        return original(shape, dtype, *args, **kwargs)

    with patch.object(np, "zeros", zeros):
        return call()


def plot_feather(path, output):
    from discovery.pulsar import Pulsar
    import matplotlib.pyplot as plt

    psr = Pulsar.read_feather(str(path))
    sigma = np.asarray(psr.toaerrs)
    r = np.asarray(psr.residuals)
    m = metric(r, sigma)
    centred = (r - m["weighted_mean_s"]) * 1e6
    frame = pd.DataFrame(
        dict(
            mjd=np.asarray(psr.stoas) / 86400,
            residual_us=centred,
            sigma_us=sigma * 1e6,
            system=psr.flags["sys"],
            frequency_MHz=psr.freqs,
        )
    )
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), layout="constrained")
    for system, g in frame.groupby("system"):
        axes[0].errorbar(
            g.mjd,
            g.residual_us,
            yerr=g.sigma_us,
            fmt=".",
            ms=2,
            lw=0.35,
            alpha=0.5,
            label=system,
        )
        axes[1].scatter(g.mjd, g.residual_us, s=4, alpha=0.5)
        axes[2].scatter(g.frequency_MHz, g.residual_us, s=4, alpha=0.5)
    # Full range first. The second panel is an explicitly labelled diagnostic zoom.
    limit = max(5 * m["wrms_us"], 0.1)
    outside = int(np.sum(np.abs(centred) > limit))
    axes[1].set_ylim(-limit, limit)
    axes[1].set_title(
        f"Zoom ±{limit:.3g} µs; {outside} point centres outside; all points shown above"
    )
    axes[0].set_title(
        f'{psr.name} — read from Feather; WRMS {m["wrms_us"]:.4g} µs; N={len(r)}'
    )
    for ax in axes:
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("Residual (µs)")
    axes[0].set_xlabel("Site MJD")
    axes[1].set_xlabel("Site MJD")
    axes[2].set_xlabel("SSB frequency (MHz)")
    axes[0].legend(fontsize=6, ncol=4)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return frame, m


NUMBER = r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+\-]?\d+)?"


def dec(value):
    return Decimal(value.replace("D", "E").replace("d", "e"))


def epochs(path):
    out = {}
    for line in path.read_text().splitlines():
        a = line.split()
        if a and a[0] in ["PEPOCH", "POSEPOCH", "DMEPOCH"]:
            out[a[0]] = str(dec(a[1]))
    return out


def parameters(path, log=""):
    result = {}
    for line in path.read_text().splitlines():
        a = line.split()
        if not a or a[0] in ["C", "#"] or line.lstrip().startswith("#"):
            continue
        name = a[0].upper()
        if name in ["JUMP", "FDJUMPDM"]:
            if len(a) < 5 or a[-1] not in ["1", "2"]:
                continue
            key = name + "[" + " ".join(a[1:-2]) + "]"
            value = a[-2]
            unc = None
            unit = "s" if name == "JUMP" else "pc cm^-3"
        else:
            if len(a) < 3 or a[2] not in ["1", "2"]:
                continue
            key = name
            value = a[1]
            unc = a[3] if len(a) > 3 else None
            unit = "Tempo2 PAR units"
        if ":" in value:
            # Prefer the full-precision radian values from Tempo2 below.
            parts = value.split(":")
            sign = -1 if parts[0].startswith("-") else 1
            value = str(
                sign * (abs(dec(parts[0])) * 3600 + dec(parts[1]) * 60 + dec(parts[2]))
            )
            unit = "seconds of RA time" if name == "RAJ" else "arcseconds"
        try:
            v = dec(value)
            e = dec(unc) if unc else None
            if not v.is_finite():
                continue
        except InvalidOperation:
            continue
        result[key] = dict(
            value=str(v),
            uncertainty=str(e) if e is not None and e.is_finite() else None,
            unit=unit,
            kind=(
                "jump"
                if name == "JUMP"
                else "fdjumpdm" if name == "FDJUMPDM" else "timing"
            ),
        )
    for line in log.splitlines():
        m = re.match(
            r"^\s*(RAJ|DECJ)\s+\(rad\)\s+("
            + NUMBER
            + r")\s+("
            + NUMBER
            + r")\s+("
            + NUMBER
            + r")\s+("
            + NUMBER
            + r")\s+Y\b",
            line,
        )
        if m and m[1] in result:
            result[m[1]].update(
                value=str(dec(m[3])), uncertainty=str(dec(m[4])), unit="radians"
            )
        m = re.match(
            r"^\s*(?:Jump\s+\d+|FDJUMPDM)\s*\((.*?)\):\s*("
            + NUMBER
            + r")\s+("
            + NUMBER
            + r")\s+Y\b",
            line,
            re.I,
        )
        if m:
            prefix = (
                "FDJUMPDM" if line.lstrip().upper().startswith("FDJUMPDM") else "JUMP"
            )
            mask = m[1].split()
            key = prefix + "[" + " ".join(mask[:-2]) + "]"
            if key in result:
                result[key]["uncertainty"] = str(dec(m[3]))
    return result


def maximum_step(previous, current):
    values = []
    for key, v in current.items():
        if key not in previous or v["kind"] == "jump":
            continue
        p = previous[key]
        if p["unit"] != v["unit"] or not v["uncertainty"]:
            continue
        sigma = abs(dec(v["uncertainty"]))
        if sigma > 0:
            values.append(float(abs(dec(v["value"]) - dec(p["value"])) / sigma))
    return max(values, default=0.0)


def converged_fit(par, tim, folder, root, max_iterations=8):
    if max_iterations < 3:
        raise ValueError("At least three fit iterations must be allowed")
    previous = None
    last = None
    history = []
    chosen = par
    for iteration in range(1, max_iterations + 1):
        chosen, stat = tempo(chosen, tim, folder / f"pass{iteration:02}", root)
        log = (chosen.parent / "tempo2.log").read_text()
        current = parameters(chosen, log)
        rms = float(
            re.findall(r"RMS post-fit residual\s*=\s*(" + NUMBER + ")", log)[-1]
        )
        step = maximum_step(previous, current) if previous is not None else None
        relative = abs(rms - last) / max(abs(last), 1e-9) if last is not None else None
        if history:
            assert (
                stat["n"] == history[0]["n"] and stat["k"] == history[0]["k"]
            ), "Fit count changed"
            assert stat["weighted_fit"] == history[0]["weighted_fit"] and set(
                current
            ) == set(previous)
        history.append(
            dict(
                iteration=iteration,
                **stat,
                postfit_RMS_us=rms,
                max_parameter_step_sigma=step,
                relative_rms_change=relative,
            )
        )
        if iteration >= 3 and relative < 1e-5 and step < 0.02:
            (folder / "convergence.json").write_text(
                json.dumps(history, indent=2) + "\n"
            )
            return chosen, stat, history
        previous = current
        last = rms
    (folder / "convergence.json").write_text(json.dumps(history, indent=2) + "\n")
    raise RuntimeError("Timing fit did not converge: " + str(folder))


def process(
    data,
    root,
    psr,
    variant="combined",
    fd=False,
    repair_jumps=False,
    reuse_fit=False,
    model_file=None,
    max_fit_iterations=8,
):
    import libstempo
    from enterprise.pulsar import Tempo2Pulsar
    from discovery.pulsar import Pulsar
    from pleb.param_scan import parse_candidate_specs, apply_candidate_to_par_text
    from scipy.stats import chi2, f as f_distribution

    base = root / "fits" / psr / variant
    base.mkdir(parents=True, exist_ok=True)
    par = data / psr / f"{psr}_{variant}.par"
    tim = data / psr / f"{psr}_{variant}_all.tim"
    assert par.is_file() and tim.is_file(), (par, tim)
    if reuse_fit:
        previous = json.loads((root / "reports" / f"{psr}_{variant}.json").read_text())
        model_reused = previous.get("model_reused", False)
        chosen = Path(previous["fitted_par"])
        s0 = previous["baseline_fit"]
        comparison = previous["fd_comparison"]
        repairs = previous["jump_repairs"]
        fit_history = previous["fit_history"]
    elif model_file is not None:
        chosen = Path(model_file)
        s0 = stats((chosen.parent / "tempo2.log").read_text())
        comparison = None
        repairs = []
        fit_history = []
        model_reused = True
    else:
        model_reused = False
        source_model = par.read_text()
        without_fd = (
            "\n".join(
                line
                for line in source_model.splitlines()
                if not re.match(r"^\s*FDJUMPDM\s+-sys\s+NFR\.LUPPI\.75\s", line)
            )
            + "\n"
        )
        baseline_input = par
        if fd and without_fd.splitlines() != source_model.splitlines():
            base.mkdir(parents=True, exist_ok=True)
            baseline_input = base / "without_fdjumpdm.par"
            baseline_input.write_text(without_fd)
        repairs = []
        if repair_jumps:
            lines = baseline_input.read_text().splitlines()
            for i, line in enumerate(lines):
                fields = line.split()
                if (
                    len(fields) == 5
                    and fields[0] == "JUMP"
                    and abs(float(fields[3])) > 1e6
                ):
                    repairs.append(dict(original_line=line, initial_value_seconds=0.0))
                    fields[3] = "0"
                    lines[i] = " ".join(fields)
            if repairs:
                base.mkdir(parents=True, exist_ok=True)
                baseline_input = base / "jump_repair_initial.par"
                baseline_input.write_text("\n".join(lines) + "\n")
        baseline, s0, fit_history = converged_fit(
            baseline_input,
            tim,
            base / "baseline",
            root,
            max_iterations=max_fit_iterations,
        )
        chosen = baseline
        comparison = None
        if fd:
            line = "FDJUMPDM -sys NFR.LUPPI.75 0 1"
            candidate = parse_candidate_specs(["raw:" + line])[0]
            initial = base / "fd_initial.par"
            initial.write_text(apply_candidate_to_par_text(without_fd, candidate))
            fitted, s1 = tempo(initial, tim, base / "fdjumpdm", root)
            assert s0["n"] == s1["n"] and s1["k"] == s0["k"] + 1, (s0, s1)
            assert (
                s0["weighted_fit"] and s1["weighted_fit"]
            ), "FDJUMPDM test requires weighted fits"
            gain = s0["chisq"] - s1["chisq"]
            comparison = dict(
                baseline=s0,
                candidate=s1,
                delta_chisq=gain,
                delta_BIC=float(
                    s1["chisq"]
                    + s1["k"] * np.log(s1["n"])
                    - s0["chisq"]
                    - s0["k"] * np.log(s0["n"])
                ),
                nested_white_noise_p=float(chi2.sf(max(gain, 0), 1)),
                mask="-sys NFR.LUPPI.75",
            )
            comparison["F_statistic"] = float(max(gain, 0) / (s1["chisq"] / s1["dof"]))
            comparison["F_test_p"] = float(
                f_distribution.sf(comparison["F_statistic"], 1, s1["dof"])
            )
            comparison["delta_BIC_unknown_common_scale"] = float(
                s1["n"] * np.log(s1["chisq"] / s0["chisq"]) + np.log(s1["n"])
            )
            comparison["interpretation"] = (
                "Nested white-noise tests conditional on this timing model; not a red-noise robustness test."
            )
            match = re.search(
                r"FDJUMPDM\s*\([^\n]+\):\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+Y",
                (base / "fdjumpdm/tempo2.log").read_text(),
            )
            assert match, "Missing fitted FDJUMPDM uncertainty"
            comparison["coefficient"] = dict(
                value=float(match[1]), uncertainty=float(match[2]), units="pc cm^-3"
            )
            comparison["selected"] = bool(
                gain > 0
                and comparison["delta_BIC"] < -6
                and comparison["delta_BIC_unknown_common_scale"] < -6
            )
            if comparison["selected"]:
                chosen = fitted
    os.environ["CLOCK_TRACE"] = str(base / "libstempo_clock_opens.tsv")
    t2 = libstempo.tempopulsar(
        parfile=str(chosen), timfile=str(tim), maxobs=120000, dofit=False
    )
    arrays = export_cli(chosen, tim, base, root)
    np.savez(base / "tempo2_arrays.npz", arrays=arrays)
    sat = np.asarray(t2.stoas, dtype=float)
    res = np.asarray(t2.residuals(), dtype=float)
    sigma = np.asarray(t2.toaerrs, dtype=float) * 1e-6
    freq = np.asarray(t2.freqs, dtype=float)
    assert len(arrays) == len(sat), (len(arrays), len(sat))
    # Verify order before using any arrays. Never graft an unverified residual vector.
    assert np.max(np.abs(sat - arrays[:, 0])) < 2e-10, "Site-TOA ordering mismatch"
    assert np.max(np.abs(freq - arrays[:, 3])) < 2e-5, "Frequency ordering mismatch"
    assert np.allclose(
        sigma, arrays[:, 2] * 1e-6, rtol=1e-8, atol=1e-14
    ), "TOA-error mismatch"
    phase = float(np.mean(res - arrays[:, 1]))
    difference = float(np.max(np.abs(res - arrays[:, 1] - phase)))
    assert difference < 1e-8, ("CLI/libstempo residual mismatch", difference)
    reload_metrics = metric(res, sigma)
    fitlog = (chosen.parent / "tempo2.log").read_text()
    fitted_rms = float(
        re.findall(r"RMS post-fit residual\s*=\s*([\d.eE+\-]+)", fitlog)[-1]
    )
    selected_stats = stats(fitlog)
    if selected_stats["weighted_fit"]:
        assert abs(reload_metrics["wrms_us"] - fitted_rms) < max(
            0.005, 0.01 * fitted_rms
        ), (
            "Saved PAR failed residual RMS check",
            reload_metrics["wrms_us"],
            fitted_rms,
        )
    raw_flags = {str(k): np.asarray(t2.flagvals(k)).astype(str) for k in t2.flags()}
    assert "sys" in raw_flags and np.all(raw_flags["sys"] != "")
    fdpars = {
        str(p): dict(value=float(t2[p].val), uncertainty=float(t2[p].err))
        for p in t2.pars(which="set")
        if str(p).startswith("FDJUMPDM")
    }
    if comparison and comparison["selected"]:
        for value in fdpars.values():
            value.update(
                comparison["coefficient"], uncertainty_source="Tempo2 fit output"
            )
    metadata = tim_metadata(tim)
    names = np.char.decode(legacy_metadata(t2.filename), "utf-8")
    if len(metadata) != len(sat):
        loaded = Counter(
            (Path(n).name, str(s)) for n, s in zip(names, raw_flags["sys"])
        )
        missing = []
        for row in metadata:
            key = (Path(row["filename"]).name, row["system"])
            if loaded[key]:
                loaded[key] -= 1
            else:
                missing.append(row)
        pd.DataFrame(missing).to_csv(base / "tim_parser_unloaded_rows.csv", index=False)
    assert len(metadata) == len(sat), (
        "TIM/Tempo2 row selection mismatch",
        len(metadata),
        len(sat),
    )
    assert [Path(row["filename"]).name for row in metadata] == [
        Path(n).name for n in names
    ], "TIM filename order"
    assert np.array_equal(
        [row["system"] for row in metadata], raw_flags["sys"]
    ), "TIM system order"
    assert (
        np.max(np.abs(np.array([row["frequency"] for row in metadata]) - freq)) < 2e-5
    ), "TIM frequency order"
    delta = np.array([row["mjd"] for row in metadata]) - sat
    if np.max(np.abs(delta)) >= 2e-10:
        debug = pd.DataFrame(metadata)
        debug["loaded_mjd"] = sat
        debug["delta_seconds"] = delta * 86400
        debug.loc[np.abs(delta) >= 2e-10].to_csv(
            base / "tim_time_offsets.csv", index=False
        )
    assert np.max(np.abs(delta)) < 2e-10, "TIM site-time order"

    class MetadataAdapter:
        # Only the dtype spelling is adapted, during the original telescope call.
        # Timing methods and arrays are delegated to the same libstempo object.
        def __getattr__(self, key):
            return getattr(t2, key)

        def __getitem__(self, key):
            return t2[key]

        def __setitem__(self, key, value):
            t2[key] = value

        def telescope(self):
            return legacy_metadata(t2.telescope)

    e = Tempo2Pulsar(MetadataAdapter(), sort=False, drop_t2pulsar=False)
    design_columns = [str(name) for name in e.fitpars]
    assert len(design_columns) == e.Mmat.shape[1], (design_columns, e.Mmat.shape)
    d = Pulsar()
    order = np.argsort(e.toas, kind="stable")
    for field in [
        "toas",
        "stoas",
        "toaerrs",
        "residuals",
        "freqs",
        "Mmat",
        "sunssb",
        "pos_t",
        "planetssb",
    ]:
        setattr(d, field, np.asarray(getattr(e, field))[order].copy())
    d.flags = {k: v[order].copy() for k, v in raw_flags.items()}
    d.backend_flags = d.flags["sys"].copy()
    for field in ["name", "dm", "dmx", "pdist", "pos", "phi", "theta"]:
        if hasattr(e, field):
            setattr(d, field, getattr(e, field))
    out = root / "feathers" / f"{psr}_{variant}.feather"
    Pulsar.save_feather(d, str(out))
    reloaded = Pulsar.read_feather(str(out))
    for field in [
        "toas",
        "stoas",
        "toaerrs",
        "residuals",
        "Mmat",
        "sunssb",
        "pos_t",
        "planetssb",
    ]:
        assert np.array_equal(
            np.asarray(getattr(d, field)), np.asarray(getattr(reloaded, field))
        ), field
    assert np.array_equal(d.flags["sys"], reloaded.flags["sys"])
    # A second independent serialization must be byte-identical.
    repeat = out.with_suffix(".repeat.feather")
    Pulsar.save_feather(d, str(repeat))
    assert sha(out) == sha(repeat), "Non-reproducible Feather serialization"
    repeat.unlink()
    points, m = plot_feather(out, root / "plots" / f"{psr}_{variant}.png")
    points.to_csv(base / "feather_points.csv.gz", index=False)
    summary = dict(
        pulsar=psr,
        variant=variant,
        **m,
        fit_converged=not model_reused,
        model_reused=model_reused,
        worker_sha256=sha(Path(__file__)),
        fit_history=fit_history,
        baseline_fit=s0,
        fd_comparison=comparison,
        fd_parameters=fdpars,
        jump_repairs=repairs,
        source_par=str(par),
        source_par_sha256=sha(par),
        fitted_par=str(chosen),
        fitted_par_sha256=sha(chosen),
        source_tim_sha256=sha(tim),
        feather=str(out),
        feather_sha256=sha(out),
        cli_libstempo_max_difference_s=difference,
        global_phase_difference_s=phase,
        exact_feather_roundtrip=True,
        repeated_serialization_identical=True,
        design_columns=design_columns,
        tim_text_parser_rows=len(metadata),
        tim_text_extra_rows=0,
        telescope_metadata_adapter="original libstempo accessor with aNN -> SNN dtype spelling",
    )
    original_tres = re.search(r"^TRES\s+(\S+)", par.read_text(), re.M)
    summary["source_TRES_us"] = float(original_tres[1]) if original_tres else None
    summary["tempo2_in_memory_postfit_RMS_us"] = fitted_rms
    summary["selected_fit"] = selected_stats
    summary["clock_chains"] = json.loads((base / "clock_chains.json").read_text())
    (root / "reports" / f"{psr}_{variant}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "pulsar",
                    "variant",
                    "n",
                    "wrms_us",
                    "cli_libstempo_max_difference_s",
                    "fd_comparison",
                ]
            }
        ),
        flush=True,
    )
    return summary


def verify(data, root, psr, variant):
    """Independent Tempo2 execution, including from the second pinned SIF."""
    summary = json.loads((root / "reports" / f"{psr}_{variant}.json").read_text())
    base = root / "fits" / psr / variant
    folder = base / "independent_verification"
    folder.mkdir(exist_ok=True)
    a = export_cli(
        Path(summary["fitted_par"]),
        data / psr / f"{psr}_{variant}_all.tim",
        folder,
        root,
    )
    reference = np.load(base / "tempo2_arrays.npz")["arrays"]
    assert a.shape == reference.shape
    assert np.array_equal(
        a, reference
    ), "Independent Tempo2 execution changed exported values"
    result = dict(
        pulsar=psr,
        variant=variant,
        n=len(a),
        independent_tempo2_arrays_identical=True,
        maximum_residual_difference_seconds=float(
            np.max(np.abs(a[:, 1] - reference[:, 1]))
        ),
        clock_chains=json.loads((folder / "clock_chains.json").read_text()),
    )
    (root / "reports" / f"{psr}_{variant}_verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


def noise(root, psr, variant, flag="group"):
    from discovery.pulsar import Pulsar
    from pleb.white_noise_selection import screen_backend

    path = root / "feathers" / f"{psr}_{variant}.feather"
    p = Pulsar.read_feather(str(path))
    support = []
    candidates = []
    labels = np.asarray(p.flags[flag])
    assert np.all(labels != ""), "Missing backend labels"
    for backend in sorted(set(labels)):
        selected = labels == backend
        try:
            context, models, decision = screen_backend(
                np.asarray(p.residuals)[selected],
                np.asarray(p.toaerrs)[selected],
                np.asarray(p.stoas)[selected],
                np.asarray(p.Mmat)[selected],
                epoch_seconds=1.0,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            context = dict(n_toas=int(sum(selected)))
            models = []
            decision = dict(status="failed", error=str(error))
        support.append(
            dict(
                pulsar=psr,
                variant=variant,
                backend_flag=flag,
                backend=backend,
                **context,
                **decision,
            )
        )
        candidates.extend(
            dict(pulsar=psr, backend_flag=flag, backend=backend, **row)
            for row in models
        )
        print(psr, backend, context["n_toas"], decision, flush=True)
    pd.DataFrame(support).to_csv(
        root / "noise" / f"{psr}_{variant}_support.csv", index=False
    )
    pd.DataFrame(candidates).to_csv(
        root / "noise" / f"{psr}_{variant}_models.csv", index=False
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "phase", choices=["runtime", "process", "repeat", "plot", "noise", "verify"]
    )
    p.add_argument("--data", type=Path, default=Path("/run/input/data"))
    p.add_argument("--root", type=Path, default=Path("/run"))
    p.add_argument("--pulsar")
    p.add_argument("--variant", default="combined")
    p.add_argument("--fd", action="store_true")
    p.add_argument("--utc-config", type=Path)
    p.add_argument("--repair-large-jumps", action="store_true")
    p.add_argument("--max-fit-iterations", type=int, default=8)
    p.add_argument(
        "--fitted-model", type=Path, help="Previously validated model; no refit"
    )
    p.add_argument(
        "--noise-flag",
        default="group",
        help="Use hardware/band group so subchannels share ECORR epochs",
    )
    a = p.parse_args()
    if a.phase == "runtime":
        prepare_runtime(a.data, a.root, a.utc_config)
        return
    configure(a.root)
    if a.phase in {"process", "repeat"}:
        process(
            a.data,
            a.root,
            a.pulsar,
            a.variant,
            a.fd,
            a.repair_large_jumps,
            reuse_fit=a.phase == "repeat",
            model_file=a.fitted_model,
            max_fit_iterations=a.max_fit_iterations,
        )
    elif a.phase == "noise":
        noise(a.root, a.pulsar, a.variant, a.noise_flag)
    elif a.phase == "verify":
        verify(a.data, a.root, a.pulsar, a.variant)
    else:
        plot_feather(
            a.root / "feathers" / f"{a.pulsar}_{a.variant}.feather",
            a.root / "plots" / f"{a.pulsar}_{a.variant}.png",
        )


if __name__ == "__main__":
    main()
