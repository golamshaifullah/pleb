from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from .logging_utils import get_logger

logger = get_logger("data_combination_pipeline.tempo2")

def build_singularity_prefix(home_dir: Path, singularity_image: Path) -> List[str]:
    return [
        "singularity",
        "exec",
        "--bind",
        f"{home_dir}:/data",
        str(singularity_image),
    ]

def tempo2_paths_in_container(pulsar: str) -> Tuple[str, str]:
    par = f"/data/{pulsar}/{pulsar}.par"
    tim = f"/data/{pulsar}/{pulsar}_all.tim"
    return par, tim

def run_subprocess(cmd: List[str], stdout_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Running: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
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
    singularity_image: Path,
    out_paths: Dict[str, Path],
    pulsar: str,
    branch: str,
    epoch: str,
    force_rerun: bool = False,
) -> None:
    prefix = build_singularity_prefix(home_dir, singularity_image)
    par, tim = tempo2_paths_in_container(pulsar)

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
    run_subprocess(plk_cmd, plk_out)

    matrix_cmd = prefix + [
        "tempo2",
        "-output",
        "-plugin", "matrix",
        "-f", par, tim,
    ]
    rc = run_subprocess(matrix_cmd, cov_out)
    if rc != 0:
        # fallback syntax in some builds
        matrix_cmd = prefix + ["tempo2", "-output", "matrix", "-f", par, tim]
        run_subprocess(matrix_cmd, cov_out)

    gen2_strings = "{sat} {bat} {clock0} {clock1} {clock2} {clock3} {clock4} {shapiro} {shapiroJ} {shapiroS} {shapiroV} {shapiroU} {shapiroN} {tropo} {roemer} {tt} {tt2tb} {earth_ssb} {earth_ssb1} {earth_ssb2} {earth_ssb3} {sun_earth1} {sun_earth2} {sun_earth3} {ism} {elev} {npulse} {clock} {ipm} {freq} {pre} {pre_phase} {post} {post_phase} {err} {solarangle} {binphase}"
    gen_cmd = prefix + [
        "tempo2",
        "-output",
        "-plugin", "general2",
        "-f", par, tim,
        "-s", gen2_strings,
    ]
    run_subprocess(gen_cmd, gen_out)
