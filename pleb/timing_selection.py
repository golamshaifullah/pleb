"""Timing-model comparison using PLEB's existing parameter specifications."""

import math
from pathlib import Path
import re

from .param_scan import apply_candidate_to_par_text, parse_candidate_specs


def fit_statistics(text):
    n = re.findall(r"Number of points in fit\s*=\s*(\d+)", text)
    k = re.findall(r"Number of fit parameters:\s*(\d+)", text)
    chi = re.findall(
        r"Chisqr/nfree\s*=\s*([\d.eE+\-]+)\s*/\s*(-?\d+)\s*=\s*([\d.eE+\-]+)", text
    )
    if not (n and k and chi):
        raise ValueError("Model selection requires weighted Tempo2 fit statistics")
    value, dof, reduced = chi[-1]
    result = dict(
        n=int(n[-1]),
        k=int(k[-1]),
        chisq=float(value),
        dof=int(dof),
        reduced_chisq=float(reduced),
    )
    if (
        result["n"] - result["k"] != result["dof"]
        or result["dof"] <= 0
        or not math.isfinite(result["chisq"])
        or result["chisq"] <= 0
    ):
        raise ValueError("Invalid Tempo2 fit statistics")
    return result


def select_timing_model(
    par,
    baseline_statistics,
    candidates,
    folder,
    execute,
    *,
    min_delta_bic=10.0,
    common_noise_scale=False,
):
    """Compare specified models; always fit each trial from the same baseline.

    These BIC scores are conditional white-noise comparisons, not evidence
    calibrated against correlated noise. ``execute`` is the configured Tempo2
    runner and returns the fitted PAR and its stdout/stderr text.
    """

    def score(stat):
        likelihood = (
            stat["n"] * math.log(stat["chisq"] / stat["n"])
            if common_noise_scale
            else stat["chisq"]
        )
        return likelihood + stat["k"] * math.log(stat["n"])

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    original = Path(par).read_text()
    base_score = score(baseline_statistics)
    best, best_score = Path(par), base_score
    trials = []
    for number, candidate in enumerate(parse_candidate_specs(candidates), 1):
        trial = folder / f"candidate{number:02}"
        trial.mkdir()
        source = trial / "candidate.par"
        source.write_text(apply_candidate_to_par_text(original, candidate))
        fitted, text = execute(source, trial)
        statistics = fit_statistics(text)
        if statistics["n"] != baseline_statistics["n"]:
            raise ValueError("Timing-model comparison changed the selected TOAs")
        value = score(statistics)
        trials.append(
            dict(
                candidate=candidate.label,
                statistics=statistics,
                delta_bic=base_score - value,
                conditional_white_noise=True,
            )
        )
        if base_score - value >= min_delta_bic and value < best_score:
            best, best_score = fitted, value
    return best, dict(
        baseline=baseline_statistics,
        min_delta_bic=min_delta_bic,
        common_noise_scale=common_noise_scale,
        selected=str(best),
        trials=trials,
    )
