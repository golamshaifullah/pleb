"""Container-backed, clock-audited Feather export and residual report.

Run with python -m pleb.residual_export --config PROFILE --output-dir NEW_RUN.
All timing calculations and Discovery exports execute in the pinned SIFs.
"""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import html
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from .tim_utils import active_count
from .release import file_digest
from .utils import safe_path


def report(root, psrs, profile):
    previous = {}
    old = profile.get("previous_report")
    if old and Path(old).exists():
        with Path(old).open() as f:
            previous = {r["pulsar"]: r for r in csv.DictReader(f)}
    rows = []
    issues = []
    table = []
    for psr in psrs:
        state_path = root / "reports" / f"run_{psr}.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        path = root / "reports" / f"{psr}_combined.json"
        if not path.exists() or not state.get("passed"):
            issues.append(f"{psr}: " + state.get("error", "export incomplete"))
            continue
        r = json.loads(path.read_text())
        old = previous.get(psr, {})
        row = dict(
            pulsar=psr,
            n_toas=r["n"],
            wrms_us=r["wrms_us"],
            rms_us=r["rms_us"],
            fit_converged=r["fit_converged"],
            model_reused=r.get("model_reused", False),
            weighted_fit=r["selected_fit"]["weighted_fit"],
            previous_n_toas=old.get("n_toas"),
            previous_wrms_us=old.get("wrms_us"),
            independent_containers_identical=state["verified"],
            independently_reproduced_feather=state["repeated_feather_identical"],
            cli_libstempo_max_difference_s=r["cli_libstempo_max_difference_s"],
            feather_sha256=r["feather_sha256"],
        )
        row["wrms_change_percent"] = (
            100 * (r["wrms_us"] / float(old["wrms_us"]) - 1)
            if old.get("wrms_us") and float(old["wrms_us"])
            else None
        )
        rows.append(row)
        p = html.escape(psr)
        key = p + "_combined"
        delta = (
            f"{row['wrms_change_percent']:+.2f}%"
            if row["wrms_change_percent"] is not None
            else "—"
        )
        table.append(
            f'<tr><td><a href="plots/{key}.png">{p}</a></td><td>{r["n"]}</td><td>{r["wrms_us"]:.5g}</td><td>{r["rms_us"]:.5g}</td><td>{delta}</td><td><a href="feathers/{key}.feather">Feather</a> · <a href="reports/{key}.json">Checks</a> · <a href="fits/{p}/combined/baseline/convergence.json">Fit convergence</a></td></tr>'
        )
    with (root / "reports/residual_quality.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["pulsar"])
        w.writeheader()
        w.writerows(rows)
    warnings = [
        r["pulsar"]
        + ": previous validated model reused; nonlinear refit did not converge"
        for r in rows
        if r["model_reused"]
    ]
    summary = dict(
        pulsars=len(rows),
        expected_pulsars=len(psrs),
        toas=sum(r["n_toas"] for r in rows),
        feather_files=len(rows),
        independent_container_checks=sum(
            r["independent_containers_identical"] for r in rows
        ),
        independently_reproduced_feathers=sum(
            r["independently_reproduced_feather"] for r in rows
        ),
        issues=issues,
        model_warnings=warnings,
        model_search=False,
        fresh_qc=False,
        white_noise_screen=False,
        comparison="Previous report uses different TOA selection; RMS changes are descriptive, not a matched-data test.",
    )
    (root / "reports/final_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Audited Feather residuals</title><style>body{font:16px sans-serif;max-width:1200px;margin:2em auto;padding:0 1em}td,th{padding:.4em .7em;text-align:left}tr:nth-child(even){background:#eee}a{color:#125da9}</style><h1>Residuals read from Discovery Feather files</h1><p>All selected TOAs and original error bars are retained. Plots show the full range, a labelled zoom, and residuals versus frequency. Only a single global phase zero is removed. Existing timing terms are refitted to convergence, except explicitly declared saved models listed below; no model search, new QC cuts or white-noise selection is run.</p><p>WRMS changes compare the previous report's different selection; they are descriptive, not a matched-data test. Correct serialization does not establish featureless or statistically white residuals.</p><p><a href="reports/residual_quality.csv">Counts, RMS and verification CSV</a> · <a href="reports/clock_manifest.json">Clock provenance</a> · <a href="reports/final_summary.json">Completion checks</a></p>"""
    page += f'<p>{len(rows)}/{len(psrs)} pulsars complete; {summary["toas"]:,} exported TOAs. {len(issues)} incomplete/error entries.</p>'
    if warnings:
        page += (
            "<p><strong>Model review:</strong> "
            + html.escape("; ".join(warnings))
            + ". Export validation passed; a newly converged fit is not claimed for these pulsars.</p>"
        )
    if issues:
        page += "<pre>" + html.escape("\n".join(issues)) + "</pre>"
    page += (
        "<table><tr><th>Pulsar / plot</th><th>TOAs</th><th>WRMS (µs)</th><th>RMS (µs)</th><th>WRMS vs previous</th><th>Files</th></tr>"
        + "".join(table)
        + "</table></html>\n"
    )
    (root / "index.html").write_text(page)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--pulsars", nargs="+")
    p.add_argument("--jobs", type=int)
    p.add_argument(
        "--accept-code-update",
        action="store_true",
        help="Record changed driver/worker versions while retaining previously verified exports",
    )
    a = p.parse_args(argv)
    profile = tomllib.loads(a.config.read_text())
    max_fit_iterations = int(profile.get("fit", {}).get("max_iterations", 8))
    if max_fit_iterations < 3:
        raise ValueError("At least three fit iterations must be allowed")
    root = a.output_dir.resolve()
    source = profile["input"]
    repo = (a.config.resolve().parent / source["repository"]).resolve()
    code = Path(__file__).resolve().parents[1]
    for name, expected_hash in profile.get("code", {}).items():
        assert file_digest(safe_path(code / "pleb", name)) == expected_hash, (
            "Export code differs from profile: " + name
        )
    for name in [
        "input",
        "runtime",
        "fits",
        "reports",
        "logs",
        "feathers",
        "plots",
        "tools",
    ]:
        (root / name).mkdir(parents=True, exist_ok=True)
    frozen = root / "input/data"
    if not frozen.exists():
        frozen.mkdir()
        import tarfile

        with tempfile.TemporaryFile() as archive:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "archive",
                    source["commit"] + ":" + source["dataset"],
                ],
                stdout=archive,
                check=True,
            )
            archive.seek(0)
            with tarfile.open(fileobj=archive) as tar:
                for member in tar:
                    path = safe_path(frozen, member.name)
                    if member.isdir():
                        path.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(tar.extractfile(member).read())
                    else:
                        raise RuntimeError("Unsupported source member")
        if source.get("canonical_names", False):
            for folder in frozen.glob("J*"):
                for ext in [".par", "_all.tim"]:
                    target = folder / (folder.name + "_combined" + ext)
                    shutil.copyfile(folder / (folder.name + ext), target)
    manifest = {
        str(path.relative_to(frozen)): file_digest(path)
        for path in sorted(frozen.rglob("*"))
        if path.is_file()
    }
    saved = root / "reports/frozen_inputs.json"
    if saved.exists():
        assert json.loads(saved.read_text()) == manifest, "Frozen inputs changed"
    else:
        saved.write_text(json.dumps(manifest, indent=2) + "\n")
    expected = sum(active_count(path) for path in frozen.glob("J*/*_combined_all.tim"))
    assert expected == source["expected_active"], (expected, source["expected_active"])
    images = {
        k: (repo / v).resolve() for k, v in profile["containers"]["images"].items()
    }
    assert {k: file_digest(v) for k, v in images.items()} == profile["containers"][
        "sha256"
    ], "Container hashes changed"
    policy = (a.config.resolve().parent / profile["clocks"]["policy"]).resolve()
    assert file_digest(policy) == profile["clocks"]["sha256"]
    shutil.copyfile(policy, root / "runtime/utc_clocks.json")
    locked = dict(
        commit=source["commit"],
        dataset=source["dataset"],
        images=profile["containers"]["sha256"],
        policy_sha256=file_digest(policy),
        worker_sha256=file_digest(code / "pleb/residual_export_worker.py"),
        driver_sha256=file_digest(Path(__file__)),
    )
    locked["saved_models"] = {
        item["pulsar"]: {
            field: item[field + "_sha256"] for field in ["par", "log", "signature"]
        }
        for item in profile.get("saved_models", [])
    }
    locked["max_fit_iterations"] = max_fit_iterations
    execution = root / "reports/execution.json"
    if execution.exists():
        previous = json.loads(execution.read_text())
        if previous != locked:
            assert (
                a.accept_code_update
            ), "Runtime code/input changed: use a new output directory"
            assert all(
                previous[k] == locked[k]
                for k in ["commit", "dataset", "images", "policy_sha256"]
            ), "Scientific input/runtime changed"
            history = root / "reports/execution_history"
            history.mkdir(exist_ok=True)
            (history / (previous["worker_sha256"] + ".json")).write_text(
                json.dumps(previous, indent=2) + "\n"
            )
            for statefile in (root / "reports").glob("run_*.json"):
                state = json.loads(statefile.read_text())
                state.setdefault("worker_sha256", previous["worker_sha256"])
                state.setdefault(
                    "max_fit_iterations", previous.get("max_fit_iterations", 8)
                )
                if previous.get("saved_models", {}).get(state["pulsar"]) != locked[
                    "saved_models"
                ].get(state["pulsar"]):
                    state["passed"] = False
                statefile.write_text(json.dumps(state, indent=2) + "\n")
    execution.write_text(json.dumps(locked, indent=2) + "\n")

    def prefix(image):
        return [
            "apptainer",
            "exec",
            "--cleanenv",
            "--no-home",
            "--bind",
            f"{root}:/analysis:rw",
            "--bind",
            f"{code}:/pleb-code:ro",
            "--pwd",
            "/analysis",
            str(images[image]),
            "env",
            "PYTHONPATH=/pleb-code",
            "PYTHONDONTWRITEBYTECODE=1",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "PLEB_LOG_DIR=/analysis/runtime/logs",
            "XDG_CONFIG_HOME=/analysis/runtime/config",
            "XDG_CACHE_HOME=/analysis/runtime/cache",
            "MPLCONFIGDIR=/analysis/runtime/matplotlib",
            "MPLBACKEND=Agg",
        ]

    subprocess.run(
        prefix("discovery")
        + [
            "gcc",
            "-shared",
            "-fPIC",
            "-O2",
            "/pleb-code/pleb/resources/clock_file_trace.c",
            "-ldl",
            "-o",
            "/analysis/runtime/clocktrace.so",
        ],
        check=True,
    )

    def execute(phase, psr=None):
        command = prefix("pleb" if phase == "verify" else "discovery") + [
            "python",
            "-m",
            "pleb.residual_export_worker",
            phase,
            "--root",
            "/analysis",
            "--data",
            "/analysis/input/data",
        ]
        if psr:
            command += ["--pulsar", psr]
        if phase == "runtime":
            command += ["--utc-config", "/analysis/runtime/utc_clocks.json"]
        if phase == "process":
            command += ["--max-fit-iterations", str(max_fit_iterations)]
        if phase == "process" and psr in profile.get("repair_large_jumps", []):
            command += ["--repair-large-jumps"]
        if phase == "process" and psr in saved_models:
            command += ["--fitted-model", saved_models[psr]]
        with (root / "logs" / f'{psr or "all"}_{phase}.log').open("w") as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)

    saved_models = {}
    execute("runtime")
    resolved_clocks = json.loads((root / "reports/clock_manifest.json").read_text())[
        "resolved_clock_files"
    ]
    for item in profile.get("saved_models", []):
        assets = {}
        for field in ["par", "log", "signature"]:
            path = (a.config.resolve().parent / item[field]).resolve()
            assert (
                file_digest(path) == item[field + "_sha256"]
            ), "Saved model asset changed"
            assets[field] = path
        signature = json.loads(assets["signature"].read_text())
        assert all(
            file_digest(safe_path(frozen, name)) == expected
            for name, expected in signature["tim_files"].items()
        ), "Saved model TIM inputs differ"
        ignored = set(signature.get("ignored_clock_metadata", []))
        assert ignored <= {
            "EXPLICIT_UTC.md"
        }, "Only non-clock documentation may be ignored"
        assert {
            k: v for k, v in resolved_clocks.items() if k not in ignored
        } == signature["clock_files"], "Saved model clock corrections differ"
        destination = root / "runtime/saved_models" / item["pulsar"]
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(assets["par"], destination / "new.par")
        shutil.copyfile(assets["log"], destination / "tempo2.log")
        saved_models[item["pulsar"]] = (
            "/analysis/runtime/saved_models/" + item["pulsar"] + "/new.par"
        )
    psrs = a.pulsars or sorted(
        path.parent.name for path in frozen.glob("J*/*_combined.par")
    )

    def run(psr):
        assert Path(psr).name == psr
        statefile = root / "reports" / f"run_{psr}.json"
        state = (
            json.loads(statefile.read_text())
            if statefile.exists()
            else dict(pulsar=psr)
        )
        feather = root / "feathers" / f"{psr}_combined.feather"
        if (
            state.get("passed")
            and feather.is_file()
            and state["feather_sha256"] == file_digest(feather)
        ):
            return True
        try:
            execute("process", psr)
            first = file_digest(feather)
            execute("repeat", psr)
            second = file_digest(feather)
            assert first == second, "Independent Feather reload changed bytes"
            execute("verify", psr)
            result = json.loads((root / "reports" / f"{psr}_combined.json").read_text())
            assert result["n"] == active_count(frozen / psr / f"{psr}_combined_all.tim")
            state.update(
                passed=True,
                verified=True,
                repeated_feather_identical=True,
                feather_sha256=second,
                n=result["n"],
                worker_sha256=locked["worker_sha256"],
                max_fit_iterations=max_fit_iterations,
                model_reused=result.get("model_reused", False),
            )
            state.pop("error", None)
        except Exception as exc:
            state.update(passed=False, error=str(exc))
        statefile.write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(state), flush=True)
        return state["passed"]

    with ThreadPoolExecutor(max_workers=a.jobs or profile.get("jobs", 3)) as pool:
        success = list(pool.map(run, psrs))
    assert manifest == {
        str(path.relative_to(frozen)): file_digest(path)
        for path in sorted(frozen.rglob("*"))
        if path.is_file()
    }, "Input files changed during export"
    summary = report(root, psrs, profile)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if all(success) and not summary["issues"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
