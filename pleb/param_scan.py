"""Parameter-scan utilities for rapid fit diagnostics.

This module implements a fit-only workflow that evaluates candidate parameter
additions or edits by running tempo2 on temporary `.par` variants. It is used
by the CLI ``--param-scan`` mode.

See Also:
    pleb.pipeline.run_pipeline: Full pipeline workflow.
    pleb.reports.write_new_param_significance: Related reporting utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re

import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import PipelineConfig
from .git_tools import checkout, require_clean_repo
from .logging_utils import get_logger
from .tempo2 import build_singularity_prefix, run_subprocess, tempo2_paths_in_container
from .utils import (
    which_or_raise,
    discover_pulsars,
    safe_mkdir,
    cleanup_empty_dirs,
    remove_tree_if_exists,
)
from .parsers import read_plklog
import numpy as np

logger = get_logger("pleb.param_scan")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A candidate model modification.

    Each candidate is expressed as a set of parameter edits/inserts applied to a
    base `.par` file, plus optional raw lines that are inserted verbatim.

    Attributes:
        label: Human- and filesystem-safe label for the candidate.
        params: Tuples of ``(PARAM, optional value)`` to add or modify.
        raw_lines: Raw `.par` lines to append verbatim.
    """

    label: str
    params: Tuple[Tuple[str, Optional[str]], ...]  # (PARAM, optional value)
    raw_lines: Tuple[str, ...] = ()


_SAFE_LABEL_RX = re.compile(r"[^A-Za-z0-9._+-]+")


def _safe_label(s: str) -> str:
    """Return a filesystem-safe label derived from a spec string."""
    s = s.strip()
    s = s.replace(" ", "_")
    s = _SAFE_LABEL_RX.sub("_", s)
    return s[:120] if len(s) > 120 else s


def parse_candidate_specs(specs: Sequence[str]) -> List[Candidate]:
    """Parse CLI candidate specs.

    Supported forms (repeatable):
      * "F2" -> fit F2 (enable fit flag); add "F2 0 1" if missing
      * "F2=0" -> set/start value and fit it
      * "F2+F3" or "F2,F3" -> group candidate (adds both)
      * "raw:..." -> insert raw par line verbatim (label derived from first token)

    Args:
        specs: Candidate specification strings.

    Returns:
        A list of parsed :class:`Candidate` objects.

    Examples:
        Parse a mix of candidates::

            specs = ["F2", "F2=0", "F2+F3", "raw:JUMP -sys P200 0 1"]
            candidates = parse_candidate_specs(specs)
    """

    out: List[Candidate] = []
    for spec in specs:
        if not spec or not str(spec).strip():
            continue
        s = str(spec).strip()
        if s.lower().startswith("raw:"):
            line = s[4:].strip()
            tok = line.split()[0] if line.split() else "RAW"
            label = _safe_label(f"raw_{tok}")
            out.append(Candidate(label=label, params=tuple(), raw_lines=(line,)))
            continue

        # split group
        parts = [p.strip() for p in re.split(r"[+,]", s) if p.strip()]
        params: List[Tuple[str, Optional[str]]] = []
        for p in parts:
            if "=" in p:
                name, val = p.split("=", 1)
                params.append((name.strip().upper(), val.strip()))
            else:
                params.append((p.strip().upper(), None))

        label = _safe_label(
            "+".join([f"{n}{'' if v is None else '=' + v}" for n, v in params])
        )
        out.append(Candidate(label=label, params=tuple(params)))
    return out


def _parse_par_params(par_text: str) -> Dict[str, List[str]]:
    """Parse a .par text into a dict of PARAM -> tokens.

    Notes:
        * Only the first occurrence of a given PARAM is retained.
        * Comment lines starting with '#', 'C ' or 'c ' are ignored.
        * Inline '#' comments are stripped.
    """
    out: Dict[str, List[str]] = {}
    for raw in par_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("#", "C ", "c ")):
            continue
        main, _ = _split_inline_comment(raw)
        parts = main.split()
        if not parts:
            continue
        key = parts[0].upper()
        if key not in out:
            out[key] = parts
    return out


def _is_fit_flag_one(tokens: Sequence[str]) -> bool:
    """Return True if the TEMPO2 fit-flag token is ``1``."""
    if not tokens:
        return False
    last = str(tokens[-1]).strip()
    return last == "1"


def build_typical_candidates(
    par_text: str,
    base_stats: Dict[str, Optional[float]],
    *,
    dm_redchisq_threshold: float = 2.0,
    dm_max_order: int = 4,
    btx_max_fb: int = 3,
) -> List[Candidate]:
    """Build a typical scan profile based on base fit statistics.

    Args:
        par_text: Base .par file text.
        base_stats: Fit statistics from :func:`summarize_plk_only`.
        dm_redchisq_threshold: Reduced chi-square threshold to trigger DM scans.
        dm_max_order: Maximum DM derivative order to include.
        btx_max_fb: Maximum FB derivative order to include.

    Returns:
        List of candidate model modifications.

    Notes:
        The typical profile only considers deterministic timing parameters and
        intentionally avoids noise-model terms.
    """
    # User-intent profile implemented:
    #   1) Always test Parallax (PX) if missing or not fitted.
    #   2) If a binary model is present, test model-appropriate binary derivatives.
    #   3) If no binary model AND reduced chi-square is high, test additional DM derivatives.
    # This intentionally does *not* do noise modelling terms.
    params = _parse_par_params(par_text)

    # 1) Parallax
    cands: List[Candidate] = []
    px_tokens = params.get("PX")
    if px_tokens is None or not _is_fit_flag_one(px_tokens):
        cands.append(Candidate(label="PX", params=(("PX", None),)))

    # Determine binary model (if present)
    bin_model = None
    bline = params.get("BINARY")
    if bline and len(bline) >= 2:
        bin_model = str(bline[1]).upper()

    # 2) Binary derivatives (model-dependent)
    if bin_model:
        if bin_model == "BTX":
            # BTX commonly uses FB0, FB1, FB2... as orbital frequency derivatives.
            # We scan FB1..FBn in a nested/stepwise way.
            max_fb = max(1, int(btx_max_fb))
            for n in range(1, max_fb + 1):
                ps = tuple((f"FB{i}", None) for i in range(1, n + 1))
                label = "+".join([p for p, _ in ps])
                cands.append(Candidate(label=label, params=ps))

        elif bin_model.startswith("ELL1"):
            # ELL1 family: scan PBDOT, XDOT, and pair EPS1DOT+EPS2DOT.
            cands.append(Candidate(label="PBDOT", params=(("PBDOT", None),)))
            cands.append(Candidate(label="XDOT", params=(("XDOT", None),)))
            cands.append(
                Candidate(
                    label="EPS1DOT+EPS2DOT",
                    params=(("EPS1DOT", None), ("EPS2DOT", None)),
                )
            )

        else:
            # Generic (BT/DD/DDK/T2/etc.): common derivatives.
            for nm in ("PBDOT", "XDOT", "OMDOT", "EDOT"):
                cands.append(Candidate(label=nm, params=((nm, None),)))

    # 3) DM derivatives if no binary model and fit quality is poor
    else:
        red = base_stats.get("redchisq")
        if red is not None and float(red) >= float(dm_redchisq_threshold):
            m = max(1, int(dm_max_order))
            for n in range(1, m + 1):
                ps = tuple((f"DM{i}", None) for i in range(1, n + 1))
                label = "+".join([p for p, _ in ps])
                cands.append(Candidate(label=label, params=ps))

    return cands


def _split_inline_comment(line: str) -> Tuple[str, str]:
    """Split a line into content and inline ``#`` comment."""
    # Keep '#' style inline comments intact.
    if "#" in line:
        main, comment = line.split("#", 1)
        return main.rstrip(), "#" + comment
    return line.rstrip(), ""


def apply_candidate_to_par_text(par_text: str, cand: Candidate) -> str:
    """Apply a candidate modification to a .par text block.

    Args:
        par_text: Original .par file contents.
        cand: Candidate modification to apply.

    Returns:
        Modified .par text.

    Notes:
        This function is conservative. If you need to insert complex lines
        (e.g., ``JUMP`` with flags), use ``raw:`` specifications so the line is
        inserted verbatim.
    """
    # Heuristics:
    #   * If a parameter exists as the first token of a non-comment line, edit its value (optional)
    #     and ensure the fit flag is present and set to 1.
    #   * If a parameter is missing, append a simple scalar line: "PARAM <value-or-0> 1".
    #   * raw_lines are appended if not already present (exact-match check, stripped).
    # This is intentionally conservative; complex multi-token params (e.g. JUMP/EFAC with flags)
    # should be supplied via raw:... lines.
    lines = par_text.splitlines()
    wanted = {name.upper(): val for name, val in cand.params}
    seen: set[str] = set()
    out_lines: List[str] = []

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        # Par comments often start with "C" or "#".
        if stripped.startswith(("#", "C ", "c ")):
            out_lines.append(line)
            continue

        main, comment = _split_inline_comment(line)
        parts = main.split()
        if not parts:
            out_lines.append(line)
            continue

        key = parts[0].upper()
        if key in wanted:
            seen.add(key)
            # Ensure at least PARAM VALUE
            if len(parts) == 1:
                parts.append(wanted[key] if wanted[key] is not None else "0")
            # Set value if provided
            if wanted[key] is not None and len(parts) >= 2:
                parts[1] = str(wanted[key])

            # Ensure fit flag at end and set to 1
            if len(parts) >= 3 and parts[-1] in {"0", "1"}:
                parts[-1] = "1"
            else:
                parts.append("1")

            out_lines.append(
                " ".join(parts)
                + (" " if comment and not comment.startswith(" ") else "")
                + comment
            )
        else:
            out_lines.append(line)

    # Append any missing simple scalar params
    for name, val in cand.params:
        nm = name.upper()
        if nm in seen:
            continue
        out_lines.append(f"{nm} {val if val is not None else '0'} 1")

    # Append raw lines if not already present
    existing = {line_str.strip() for line_str in out_lines if line_str.strip()}
    for raw_line in cand.raw_lines:
        rl = raw_line.strip()
        if rl and rl not in existing:
            out_lines.append(rl)

    return "\n".join(out_lines) + "\n"


def _run_fit_only_plk(
    *,
    cfg: PipelineConfig,
    pulsar: str,
    par_host_path: Path,
    work_dir: Path,
    plk_out: Path,
) -> None:
    """Run tempo2 in 'fit-only' mode (plk stdout capture) for a given par file."""

    par_container = f"/work/{par_host_path.name}"
    _, tim_container = tempo2_paths_in_container(pulsar)
    prefix = build_singularity_prefix(
        cfg.home_dir,
        cfg.dataset_name,
        cfg.singularity_image,
        extra_binds=[(work_dir, "/work")],
    )

    cmd = prefix + [
        "tempo2",
        "-f",
        par_container,
        tim_container,
        "-showchisq",
        "-colour",
        "-sys",
        "-epoch",
        str(cfg.epoch),
    ]
    run_subprocess(cmd, plk_out, cwd=work_dir)


def _parse_plk_stats_text(text: str) -> Dict[str, Optional[float]]:
    """Parse a PLK statistics text block into numeric metrics."""
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

    chisq = _rx(
        [
            r"chisq\s*=\s*([0-9.+\-eE]+)",
            r"chi\s*\^?2\s*=\s*([0-9.+\-eE]+)",
        ]
    )
    redchisq = _rx(
        [
            r"reduced\s*chisq\s*=\s*([0-9.+\-eE]+)",
            r"red\s*chisq\s*=\s*([0-9.+\-eE]+)",
        ]
    )
    n_toas = _rx(
        [
            r"number\s+of\s+\w*points\s+in\s+fit\s*=\s*([0-9.+\-eE]+)",
            r"ntoa\s*=\s*([0-9.+\-eE]+)",
            r"number\s+of\s+toas\s*=\s*([0-9.+\-eE]+)",
        ]
    )
    return {"chisq": chisq, "redchisq": redchisq, "n_toas": n_toas}


def _fit_params_count(plk_df: pd.DataFrame) -> Optional[int]:
    """Return the number of fitted parameters inferred from a PLK log table."""
    if plk_df is None or plk_df.empty or "Fit" not in plk_df.columns:
        return None
    s = plk_df["Fit"].astype(str).str.strip().str.lower()
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        return int((num.fillna(0) > 0).sum())
    return int(s.isin({"t", "true", "y", "yes", "fit", "fitted", "1"}).sum())


def summarize_plk_only(plk_path: Path) -> Dict[str, Optional[float]]:
    """Summarize fit statistics from a PLK log.

    Args:
        plk_path: Path to a tempo2 ``*_plk.log`` file.

    Returns:
        Dict with ``chisq``, ``redchisq``, ``n_toas``, and ``k_fit`` where available.

    Examples:
        Summarize a plk log to drive candidate selection::

            stats = summarize_plk_only(Path("J1234+5678_plk.log"))
    """
    # Summarize a tempo2 fit from just the captured stdout (plk log).
    text = (
        plk_path.read_text(encoding="utf-8", errors="ignore")
        if plk_path.exists()
        else ""
    )
    st = _parse_plk_stats_text(text)
    try:
        df = read_plklog(plk_path)
    except Exception:
        df = pd.DataFrame()
    k = _fit_params_count(df)
    out: Dict[str, Optional[float]] = {
        "chisq": st.get("chisq"),
        "redchisq": st.get("redchisq"),
        "n_toas": st.get("n_toas"),
        "k_fit": float(k) if k is not None else None,
    }
    if out["chisq"] is not None and out["k_fit"] is not None:
        out["aic"] = float(out["chisq"] + 2.0 * out["k_fit"])  # type: ignore[index]
        if out["n_toas"] is not None and out["n_toas"] > 0:
            out["bic"] = float(out["chisq"] + out["k_fit"] * np.log(out["n_toas"]))  # type: ignore[index]
        else:
            out["bic"] = None
    else:
        out["aic"] = None
        out["bic"] = None
    return out


def _extract_param_z(
    plk_df: pd.DataFrame, names: Iterable[str]
) -> Tuple[Optional[float], str]:
    """Return (max_z, 'name=z;...') for the given parameter names."""
    if plk_df is None or plk_df.empty:
        return None, ""
    zs: List[Tuple[str, float]] = []
    for nm in names:
        if "Param" not in plk_df.columns:
            continue
        sub = plk_df[plk_df["Param"].astype(str).str.upper() == str(nm).upper()].copy()
        if sub.empty:
            continue
        post = pd.to_numeric(sub.get("Postfit"), errors="coerce")
        unc = pd.to_numeric(sub.get("Uncertainty"), errors="coerce")
        z = (post.abs() / unc).where((unc > 0) & post.notna(), pd.NA)
        if z.notna().any():
            zv = float(z.max())
            zs.append((str(nm).upper(), zv))
    if not zs:
        return None, ""
    max_z = max(z for _, z in zs)
    details = ";".join(
        [f"{n}={z:.3g}" for n, z in sorted(zs, key=lambda x: x[1], reverse=True)]
    )
    return max_z, details


def run_param_scan(
    cfg: PipelineConfig,
    *,
    branch: Optional[str] = None,
    pulsars: Optional[Sequence[str]] = None,
    candidate_specs: Sequence[str] = (),
    scan_typical: bool = False,
    dm_redchisq_threshold: Optional[float] = None,
    dm_max_order: Optional[int] = None,
    btx_max_fb: Optional[int] = None,
    outdir_name: Optional[str] = None,
) -> Dict[str, Path]:
    """Run a per-pulsar parameter scan.

    This creates temporary .par variants (in a per-job work directory) and runs tempo2
    to estimate whether adding/enabling parameters meaningfully improves the fit.

    Output:
      * param_scan/param_scan_<branch>.tsv (combined)
      * param_scan/<pulsar>/param_scan_<pulsar>.tsv (per pulsar)
      * plk/ scan plk logs for baseline and each candidate

    Args:
        cfg: Pipeline configuration.
        branch: Branch to scan (defaults to ``cfg.reference_branch``).
        pulsars: Optional list of pulsars to scan.
        candidate_specs: Candidate specifications (see :func:`parse_candidate_specs`).
        scan_typical: If True, include the typical scan profile.
        dm_redchisq_threshold: Override for DM scan trigger threshold.
        dm_max_order: Override for maximum DM derivative order.
        btx_max_fb: Override for maximum FB derivative order.
        outdir_name: Optional output directory name.

    Returns:
        Mapping of output path labels to their filesystem paths.

    Raises:
        FileNotFoundError: If required paths are missing.
        RuntimeError: If no pulsars are selected or dependencies are missing.
        ValueError: If no candidates are specified.

    Examples:
        Run a typical parameter scan on the reference branch::

            from pleb.config import PipelineConfig
            from pleb.param_scan import run_param_scan

            cfg = PipelineConfig(
                home_dir=Path("/data/epta"),
                singularity_image=Path("/images/tempo2.sif"),
                dataset_name="EPTA",
            )
            outputs = run_param_scan(cfg, scan_typical=True)
    """
    cfg = cfg.resolved()
    if not cfg.home_dir.exists():
        raise FileNotFoundError(f"home_dir does not exist: {cfg.home_dir}")
    if not cfg.singularity_image.exists():
        raise FileNotFoundError(
            f"singularity_image does not exist: {cfg.singularity_image}"
        )
    which_or_raise(
        "singularity",
        hint="Install Singularity/Apptainer or load it in your environment.",
    )

    # Allow config to set a default "typical" scan profile.
    scan_typical = bool(scan_typical or getattr(cfg, "param_scan_typical", False))

    scan_branch = branch or cfg.reference_branch
    if not scan_branch:
        raise ValueError(
            "No branch provided for param scan (and config.reference_branch is empty)."
        )

    # Pulsar selection
    if pulsars is None:
        if cfg.pulsars == "ALL":
            pulsar_list = discover_pulsars(cfg.home_dir / cfg.dataset_name)
        else:
            pulsar_list = list(cfg.pulsars)  # type: ignore[arg-type]
    else:
        pulsar_list = list(pulsars)
    if not pulsar_list:
        raise RuntimeError("No pulsars selected/found.")

    manual_candidates = (
        parse_candidate_specs(candidate_specs) if candidate_specs else []
    )
    if not scan_typical and not manual_candidates:
        raise ValueError(
            "No candidates specified (provide --scan, --scan-file, or use --scan-typical)."
        )

    # Defaults for the typical profile (can be overridden by CLI args, then config)
    if dm_redchisq_threshold is None:
        dm_redchisq_threshold = float(
            getattr(cfg, "param_scan_dm_redchisq_threshold", 2.0) or 2.0
        )
    if dm_max_order is None:
        dm_max_order = int(getattr(cfg, "param_scan_dm_max_order", 4) or 4)
    if btx_max_fb is None:
        btx_max_fb = int(getattr(cfg, "param_scan_btx_max_fb", 3) or 3)

    # Output tree (standalone, to avoid tag collisions with the main pipeline)
    if outdir_name is None:
        if cfg.outdir_name:
            outdir_name = f"{cfg.outdir_name}_param_scan"
        else:
            from datetime import datetime

            outdir_name = "PARAM_SCAN_" + datetime.now().strftime("%Y%m%dT%H%M%S")

    base = cfg.results_dir.expanduser().resolve() / outdir_name
    tag = base / _safe_label(scan_branch)
    out_paths = {
        "base": base,
        "tag": tag,
        "plk": tag / "plk",
        "param_scan": tag / "param_scan",
        "work": tag / "work",
        "logs": tag / "logs",
    }
    safe_mkdir(out_paths["base"])
    safe_mkdir(out_paths["tag"])
    safe_mkdir(out_paths["logs"])

    try:
        from git import Repo  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "GitPython is required to run --param-scan (branch checkouts). Install GitPython."
        ) from e

    repo = Repo(str(cfg.home_dir))
    require_clean_repo(repo)
    current_branch = repo.active_branch.name

    def _scan_one(pulsar: str) -> List[Dict[str, object]]:
        # One worker per pulsar: runs baseline + all candidates sequentially.
        parfile = cfg.home_dir / cfg.dataset_name / pulsar / f"{pulsar}.par"
        if not parfile.exists():
            logger.warning("Missing par file: %s", parfile)
            return []

        base_text = parfile.read_text(encoding="utf-8", errors="ignore")
        p_work = out_paths["work"] / scan_branch / pulsar
        safe_mkdir(p_work)

        # Baseline
        base_par = p_work / f"{pulsar}_BASE.par"
        base_par.write_text(base_text, encoding="utf-8")
        base_plk = out_paths["plk"] / f"{pulsar}_{scan_branch}_BASE_plk.log"
        safe_mkdir(out_paths["plk"])

        if cfg.force_rerun or not base_plk.exists():
            _run_fit_only_plk(
                cfg=cfg,
                pulsar=pulsar,
                par_host_path=base_par,
                work_dir=p_work,
                plk_out=base_plk,
            )

        base_stats = summarize_plk_only(base_plk)

        rows: List[Dict[str, object]] = []
        rows.append(
            {
                "pulsar": pulsar,
                "branch": scan_branch,
                "candidate": "BASE",
                "chisq": base_stats.get("chisq"),
                "redchisq": base_stats.get("redchisq"),
                "n_toas": base_stats.get("n_toas"),
                "k_fit": base_stats.get("k_fit"),
                "delta_k_fit": 0,
                "lrt_delta_chisq": 0.0,
                "lrt_p_value": None,
                "max_param_z": None,
                "param_zs": "",
            }
        )

        # Candidate selection (per pulsar)
        cands_all: List[Candidate] = []
        if scan_typical:
            cands_all.extend(
                build_typical_candidates(
                    base_text,
                    base_stats,
                    dm_redchisq_threshold=float(dm_redchisq_threshold),
                    dm_max_order=int(dm_max_order),
                    btx_max_fb=int(btx_max_fb),
                )
            )
        cands_all.extend(manual_candidates)

        # De-dup + skip no-op edits
        uniq: Dict[
            Tuple[str, Tuple[Tuple[str, Optional[str]], ...], Tuple[str, ...]],
            Candidate,
        ] = {}
        for c in cands_all:
            key = (c.label, c.params, c.raw_lines)
            if key not in uniq:
                # filter out no-ops
                try:
                    if apply_candidate_to_par_text(base_text, c) == base_text:
                        continue
                except Exception:
                    pass
                uniq[key] = c
        cands_final = list(uniq.values())

        # Candidate runs
        for cand in cands_final:
            mod_text = apply_candidate_to_par_text(base_text, cand)
            mod_par = p_work / f"{pulsar}_{cand.label}.par"
            mod_par.write_text(mod_text, encoding="utf-8")

            mod_plk = out_paths["plk"] / f"{pulsar}_{scan_branch}_{cand.label}_plk.log"
            if cfg.force_rerun or not mod_plk.exists():
                _run_fit_only_plk(
                    cfg=cfg,
                    pulsar=pulsar,
                    par_host_path=mod_par,
                    work_dir=p_work,
                    plk_out=mod_plk,
                )

            cand_stats = summarize_plk_only(mod_plk)

            # Nested-model LRT
            delta_k = None
            if (
                base_stats.get("k_fit") is not None
                and cand_stats.get("k_fit") is not None
            ):
                delta_k = float(cand_stats["k_fit"]) - float(base_stats["k_fit"])  # type: ignore[index]
            delta_chisq = None
            if (
                base_stats.get("chisq") is not None
                and cand_stats.get("chisq") is not None
            ):
                delta_chisq = float(base_stats["chisq"]) - float(cand_stats["chisq"])  # type: ignore[index]

            # p-value (use the same chi2 tail utility as reports.py)
            pval = None
            if (
                delta_k is not None
                and delta_chisq is not None
                and delta_k > 0
                and delta_chisq > 0
            ):
                # Use SciPy if available, else fall back to mpmath.
                try:
                    from scipy.stats import chi2  # type: ignore

                    pval = float(chi2.sf(float(delta_chisq), float(delta_k)))
                except Exception:
                    try:
                        import mpmath as mp  # type: ignore

                        pval = float(
                            mp.gammainc(delta_k / 2.0, delta_chisq / 2.0, mp.inf)
                            / mp.gamma(delta_k / 2.0)
                        )
                    except Exception:
                        pval = None

            # z-scores for the scanned params (where possible)
            try:
                cand_df = read_plklog(mod_plk)
            except Exception:
                cand_df = pd.DataFrame()

            max_z, details = _extract_param_z(cand_df, [n for n, _ in cand.params])

            rows.append(
                {
                    "pulsar": pulsar,
                    "branch": scan_branch,
                    "candidate": cand.label,
                    "chisq": cand_stats.get("chisq"),
                    "redchisq": cand_stats.get("redchisq"),
                    "n_toas": cand_stats.get("n_toas"),
                    "k_fit": cand_stats.get("k_fit"),
                    "delta_k_fit": delta_k,
                    "lrt_delta_chisq": delta_chisq,
                    "lrt_p_value": pval,
                    "max_param_z": max_z,
                    "param_zs": details,
                }
            )

        # Per-pulsar file
        dfp = pd.DataFrame(rows)
        safe_mkdir(out_paths["param_scan"])
        pdir = out_paths["param_scan"] / pulsar
        safe_mkdir(pdir)
        dfp.to_csv(pdir / f"param_scan_{pulsar}.tsv", sep="\t", index=False)
        return rows

    try:
        logger.info("=== Param scan on branch: %s ===", scan_branch)
        checkout(repo, scan_branch)

        n_jobs = max(1, int(getattr(cfg, "jobs", 1) or 1))
        all_rows: List[Dict[str, object]] = []
        if n_jobs == 1:
            for psr in pulsar_list:
                all_rows.extend(_scan_one(psr))
        else:
            futures = []
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                for psr in pulsar_list:
                    futures.append(ex.submit(_scan_one, psr))
                for fut in as_completed(futures):
                    all_rows.extend(fut.result())

        if all_rows:
            df = pd.DataFrame(all_rows)

            # rank within each pulsar by p-value then delta chisq
            def _rank_key(x):
                try:
                    return float(x)
                except Exception:
                    return float("inf")

            df["p_value_for_sort"] = df["lrt_p_value"].apply(_rank_key)
            df = df.sort_values(
                ["pulsar", "p_value_for_sort", "lrt_delta_chisq"],
                ascending=[True, True, False],
            )
            df = df.drop(columns=["p_value_for_sort"])
            out_file = (
                out_paths["param_scan"] / f"param_scan_{_safe_label(scan_branch)}.tsv"
            )
            df.to_csv(out_file, sep="\t", index=False)

        if getattr(cfg, "cleanup_work_dir", False):
            remove_tree_if_exists(out_paths["work"])
        if getattr(cfg, "cleanup_output_tree", False):
            cleanup_empty_dirs(out_paths["tag"])
        return {k: Path(v) for k, v in out_paths.items()}

    finally:
        try:
            checkout(repo, current_branch)
        except Exception:
            pass
