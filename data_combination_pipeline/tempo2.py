from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from .logging_utils import get_logger

logger = get_logger("data_combination_pipeline.tempo2")


def build_singularity_prefix(
    home_dir: Path,
    dataset_name: Path,
    singularity_image: Path,
    extra_binds: List[tuple[Path, str]] | None = None,
) -> List[str]:
    """Build the singularity/apptainer exec prefix.

    By default, binds `home_dir` to `/data` inside the container.
    Optionally bind additional host paths to container paths.
    """
    cmd: List[str] = ["singularity", "exec"]
    binds: List[tuple[Path, str]] = [(home_dir/dataset_name, "/data")]
    if extra_binds:
        binds.extend(extra_binds)
    for host, cont in binds:
        cmd += ["--bind", f"{Path(host).resolve()}:{cont}"]
    cmd.append(str(singularity_image))
    return cmd

def tempo2_paths_in_container(pulsar: str) -> Tuple[str, str]:
    par = f"/data/{pulsar}/{pulsar}.par"
    tim = f"/data/{pulsar}/{pulsar}_all.tim"
    return par, tim

def run_subprocess(cmd: List[str], stdout_path: Path, cwd: Path | None = None) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Running: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None)
    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(res.stdout)
        if res.stderr:
            f.write("\n\n# --- STDERR ---\n")
            f.write(res.stderr)
    if res.returncode != 0:
        logger.warning("Command failed (%s): %s", res.returncode, " ".join(cmd))
    return res.returncode

def run_tempo2_for_pulsar(
    home_dir: Path,
    dataset_name: Path,
    singularity_image: Path,
    out_paths: Dict[str, Path],
    pulsar: str,
    branch: str,
    epoch: str,
    force_rerun: bool = False,
) -> None:
    prefix = build_singularity_prefix(home_dir, dataset_name, singularity_image)
    par, tim = tempo2_paths_in_container(pulsar)

    work_dir = out_paths.get("work", out_paths["logs"]) / branch / pulsar
    work_dir.mkdir(parents=True, exist_ok=True)

    plk_out = out_paths["plk"] / f"{pulsar}_{branch}_plk.log"
    cov_out = out_paths["covmat"] / f"{pulsar}_{branch}.covmat"
    gen_out = out_paths["general2"] / f"{pulsar}_{branch}.general2"

    if (not force_rerun) and plk_out.exists() and cov_out.exists() and gen_out.exists():
        return

    plk_cmd = prefix + [
        "tempo2",
        "-f", par, tim,
        "-showchisq",
        "-colour",
        "-sys",
        "-epoch", str(epoch),
    ]
    run_subprocess(plk_cmd, plk_out, cwd=work_dir)

    matrix_cmd = prefix + [
        "tempo2",
        "-output",
        "matrix",
        "-f", par, tim,
    ]
    rc = run_subprocess(matrix_cmd, cov_out, cwd=work_dir)
    if rc != 0:
        # fallback syntax in some builds
        matrix_cmd = prefix + ["tempo2", "-output", "matrix", "-f", par, tim]
        run_subprocess(matrix_cmd, cov_out, cwd=work_dir)

    gen2_strings = "{sat} {bat} {clock0} {clock1} {clock2} {clock3} {clock4} {shapiro} {shapiroJ} {shapiroS} {shapiroV} {shapiroU} {shapiroN} {tropo} {roemer} {tt} {tt2tb} {earth_ssb} {earth_ssb1} {earth_ssb2} {earth_ssb3} {sun_earth1} {sun_earth2} {sun_earth3} {ism} {elev} {npulse} {clock} {ipm} {freq} {pre} {pre_phase} {post} {post_phase} {err} {solarangle} {binphase}\n"
    gen_cmd = prefix + [
        "tempo2",
        "-output",
        "general2",
        "-f", par, tim,
        "-s", gen2_strings,
    ]
    run_subprocess(gen_cmd, gen_out, cwd=work_dir)
