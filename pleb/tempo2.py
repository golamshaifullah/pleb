"""tempo2 execution helpers for the pipeline.

This module wraps the tempo2 CLI invocation used by the pipeline and
parameter-scan workflows. It assumes tempo2 is available inside a Singularity
or Apptainer container.

See Also:
    pleb.pipeline.run_pipeline: Main workflow orchestration.
    pleb.param_scan.run_param_scan: Fit-only parameter scan workflow.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from .logging_utils import get_logger
from .dataset_fix import count_toa_lines, parse_include_lines
from .utils import safe_mkdir

logger = get_logger("pleb.tempo2")


def build_singularity_prefix(
    home_dir: Path,
    dataset_name: Path,
    singularity_image: Path,
    extra_binds: List[tuple[Path, str]] | None = None,
) -> List[str]:
    """Build the singularity/apptainer exec prefix.

    By default, binds ``home_dir/dataset_name`` to ``/data`` inside the
    container. Optionally bind additional host paths to container paths.

    Args:
        home_dir: Root data repository.
        dataset_name: Dataset name/path under ``home_dir``.
        singularity_image: Container image path.
        extra_binds: Extra ``(host_path, container_path)`` binds.

    Returns:
        A command prefix suitable for ``subprocess`` execution.
    """
    cmd: List[str] = ["singularity", "exec"]
    binds: List[tuple[Path, str]] = [(home_dir / dataset_name, "/data")]
    if extra_binds:
        binds.extend(extra_binds)
    for host, cont in binds:
        cmd += ["--bind", f"{Path(host).resolve()}:{cont}"]
    cmd.append(str(singularity_image))
    return cmd


def tempo2_paths_in_container(pulsar: str) -> Tuple[str, str]:
    """Return container paths for a pulsar's `.par` and `.tim` files.

    Args:
        pulsar: Pulsar name.

    Returns:
        Tuple of ``(par_path, tim_path)`` inside the container.
    """
    par = f"/data/{pulsar}/{pulsar}.par"
    tim = f"/data/{pulsar}/{pulsar}_all.tim"
    return par, tim


def run_subprocess(cmd: List[str], stdout_path: Path, cwd: Path | None = None) -> int:
    """Run a subprocess and capture stdout/stderr to a file.

    Args:
        cmd: Command list to execute.
        stdout_path: File to write stdout/stderr logs to.
        cwd: Optional working directory.

    Returns:
        Subprocess return code.

    Notes:
        Both stdout and stderr are captured into ``stdout_path`` for later
        inspection.
    """
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Running: %s", " ".join(cmd))
    res = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None
    )
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
    """Run tempo2 to generate PLK, covariance, and general2 outputs for a pulsar.

    Args:
        home_dir: Root data repository.
        dataset_name: Dataset name or path.
        singularity_image: Singularity/Apptainer image containing tempo2.
        out_paths: Output directory mapping from :func:`make_output_tree`.
        pulsar: Pulsar name.
        branch: Branch name (used for output filenames).
        epoch: Tempo2 epoch value.
        force_rerun: If True, run even when outputs already exist.

    Notes:
        Outputs are written to ``plk``, ``covmat``, and ``general2`` subtrees in
        ``out_paths``.
    """
    prefix = build_singularity_prefix(home_dir, dataset_name, singularity_image)
    par, tim = tempo2_paths_in_container(pulsar)
    # Compute total TOAs from all INCLUDE tim files for -nobs.
    psr_dir = Path(home_dir) / Path(dataset_name) / pulsar
    all_tim = psr_dir / f"{pulsar}_all.tim"
    total_toas = 0
    includes = parse_include_lines(all_tim)
    if includes:
        for rel in includes:
            total_toas += count_toa_lines(psr_dir / rel)
    else:
        total_toas = count_toa_lines(all_tim)
    nobs = max(1, int(total_toas * 1.01) + 1)

    work_dir = out_paths.get("work", out_paths["logs"]) / branch / pulsar
    safe_mkdir(work_dir)

    safe_mkdir(out_paths["plk"])
    safe_mkdir(out_paths["covmat"])
    safe_mkdir(out_paths["general2"])

    plk_out = out_paths["plk"] / f"{pulsar}_{branch}_plk.log"
    cov_out = out_paths["covmat"] / f"{pulsar}_{branch}.covmat"
    gen_out = out_paths["general2"] / f"{pulsar}_{branch}.general2"

    if (not force_rerun) and plk_out.exists() and cov_out.exists() and gen_out.exists():
        return

    plk_cmd = prefix + [
        "tempo2",
        "-f",
        par,
        tim,
        "-nobs",
        str(nobs),
        "-showchisq",
        "-colour",
        "-sys",
        "-epoch",
        str(epoch),
    ]
    run_subprocess(plk_cmd, plk_out, cwd=work_dir)

    matrix_cmd = prefix + [
        "tempo2",
        "-output",
        "matrix",
        "-f",
        par,
        tim,
        "-nobs",
        str(nobs),
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
        "-f",
        par,
        tim,
        "-nobs",
        str(nobs),
        "-s",
        gen2_strings,
    ]
    run_subprocess(gen_cmd, gen_out, cwd=work_dir)
