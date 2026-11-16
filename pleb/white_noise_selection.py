"""Per-backend model screening with the whitenoise estimator's covariance convention.

Uses its epoch grouper and evaluates the equivalent restricted likelihood with
diagonal/epoch blocks and a small timing-design Gram matrix, avoiding dense N²
storage. BIC differences are screens under a white-noise model, not calibrated
discovery probabilities or protection against unmodelled red noise.
"""

from itertools import combinations
import numpy as np
from scipy import linalg, optimize


def design_basis(matrix):
    matrix = np.asarray(matrix, dtype=float)
    lengths = np.linalg.norm(matrix, axis=0)
    matrix = matrix[:, lengths > 0] / lengths[lengths > 0]
    u, s, _ = linalg.svd(matrix, full_matrices=False, check_finite=False)
    rank = (
        int(np.sum(s > s[0] * max(matrix.shape) * np.finfo(float).eps)) if len(s) else 0
    )
    return u[:, :rank]


def restricted_nll_gradient(log_params, active, r, sigma, basis, epochs):
    values = dict(F=1.0, Q=0.0, J=0.0)
    values.update({k: float(np.exp(x)) for k, x in zip(active, log_params)})
    f, q, j = values["F"], values["Q"] * 1e-6, values["J"] * 1e-6
    d = f * f * (sigma * sigma + q * q)
    inverse = 1 / d
    n_epochs = int(epochs.max()) + 1
    sums = np.bincount(epochs, weights=inverse, minlength=n_epochs)
    denominator = 1 + j * j * sums
    v = np.column_stack([r, basis])
    w = inverse[:, None] * v
    grouped = np.zeros((n_epochs, v.shape[1]))
    np.add.at(grouped, epochs, w)
    cinv = w - inverse[:, None] * (j * j / denominator)[epochs, None] * grouped[epochs]
    cr, cm = cinv[:, 0], cinv[:, 1:]
    if basis.shape[1]:
        gram = basis.T @ cm
        cho = linalg.cho_factor(gram, check_finite=False)
        b = basis.T @ cr
        beta = linalg.cho_solve(cho, b, check_finite=False)
        pr = cr - cm @ beta
        gram_inverse = linalg.cho_solve(cho, np.eye(len(gram)), check_finite=False)
        correction = np.sum((cm @ gram_inverse) * cm, axis=1)
        logdet_gram = 2 * np.log(np.diag(cho[0])).sum()
    else:
        pr = cr
        correction = np.zeros(len(r))
        logdet_gram = 0.0
        gram_inverse = np.zeros((0, 0))
    logdet = np.log(d).sum() + np.log(denominator).sum() + logdet_gram
    nll = 0.5 * (logdet + r @ pr)
    diagonal_p = inverse - j * j * inverse**2 / denominator[epochs] - correction
    difference = diagonal_p - pr * pr
    derivatives = {
        "F": float(d @ difference),
        "Q": float(f * f * q * q * np.sum(difference)),
    }
    if "J" in active:
        group_cm = np.zeros((n_epochs, basis.shape[1]))
        np.add.at(group_cm, epochs, cm)
        trace = float(
            np.sum(sums / denominator) - np.sum((group_cm @ gram_inverse) * group_cm)
        )
        group_pr = np.bincount(epochs, weights=pr, minlength=n_epochs)
        derivatives["J"] = float(j * j * (trace - group_pr @ group_pr))
    return float(nll), np.array([derivatives[k] for k in active])


def screen_backend(residuals, sigma, toa_seconds, design_matrix, epoch_seconds=1.0):
    import whitenoise_estimator as wn

    prepared = wn._prepare_data(residuals, sigma, toa_seconds, design_matrix)
    r, sigma, t, m = (
        prepared.residuals,
        prepared.sigma,
        prepared.toa_seconds,
        prepared.design_matrix,
    )
    epochs = wn._group_epochs(t, epoch_seconds)
    basis = design_basis(m)
    n_eff = len(r) - basis.shape[1]
    spread = float(np.std(sigma**2) / np.mean(sigma**2))
    context = dict(
        n_toas=len(r),
        n_epochs=len(epochs.multiplicities),
        design_rank=basis.shape[1],
        multi_toa_epochs=sum(x > 1 for x in epochs.multiplicities),
        epoch_tolerance_seconds=epoch_seconds,
        sigma_squared_relative_spread=spread,
        ecorr_structurally_identifiable=epochs.has_multi_toa_epochs,
        efac_equad_structurally_identifiable=spread > 1e-3,
    )
    if n_eff < 15 or len(r) < 25:
        return context, [], dict(status="insufficient_data")
    scale = max(float(np.std(r) * 1e6), float(np.median(sigma) * 1e6), 1e-3)
    candidates = []
    for size in range(4):
        for active in combinations("FQJ", size):
            if "J" in active and not epochs.has_multi_toa_epochs:
                continue
            if "F" in active and "Q" in active and spread <= 1e-3:
                continue
            name = "+".join(active) or "nominal"
            if not active:
                value, _ = restricted_nll_gradient(
                    [], active, r, sigma, basis, epochs.epoch_ids
                )
                candidates.append(
                    dict(
                        model=name,
                        k=0,
                        nll=value,
                        success=True,
                        efac=1.0,
                        equad_us=0.0,
                        ecorr_us=0.0,
                    )
                )
                continue
            bounds = [
                (
                    (np.log(0.03), np.log(100))
                    if k == "F"
                    else (np.log(1e-5), np.log(max(scale * 100, 1.0)))
                )
                for k in active
            ]
            starts = []
            for fraction in (0.1, 1.0):
                starts.append(
                    np.array(
                        [
                            0.0 if k == "F" else np.log(max(scale * fraction, 1e-4))
                            for k in active
                        ]
                    )
                )
            fits = [
                optimize.minimize(
                    restricted_nll_gradient,
                    x,
                    args=(active, r, sigma, basis, epochs.epoch_ids),
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 150, "ftol": 1e-10},
                )
                for x in starts
            ]
            fit = min(fits, key=lambda z: z.fun)
            values = dict(F=1.0, Q=0.0, J=0.0)
            values.update({k: float(np.exp(v)) for k, v in zip(active, fit.x)})
            candidates.append(
                dict(
                    model=name,
                    k=len(active),
                    nll=float(fit.fun),
                    success=bool(fit.success),
                    efac=values["F"],
                    equad_us=values["Q"],
                    ecorr_us=values["J"],
                    message=str(fit.message),
                )
            )
    for row in candidates:
        row["BIC"] = 2 * row["nll"] + row["k"] * np.log(n_eff)
        row["AIC"] = 2 * row["nll"] + 2 * row["k"]
    valid = [x for x in candidates if x["success"] and np.isfinite(x["BIC"])]
    if not valid:
        return context, candidates, dict(status="optimisation_failed")
    best = min(valid, key=lambda x: x["BIC"])
    support = {}
    for term, label in [("F", "efac"), ("Q", "equad"), ("J", "ecorr")]:
        with_term = [x for x in valid if term in x["model"].split("+")]
        without = [x for x in valid if term not in x["model"].split("+")]
        delta = (
            min(x["BIC"] for x in without) - min(x["BIC"] for x in with_term)
            if with_term and without
            else float("nan")
        )
        support[label + "_delta_BIC"] = float(delta)
        support[label + "_supported_BIC6"] = bool(np.isfinite(delta) and delta > 6)
    if not epochs.has_multi_toa_epochs:
        support["equad_interpretation"] = (
            "extra variance floor; EQUAD and ECORR cannot be separated"
        )
        support["additive_floor_supported_BIC6"] = support["equad_supported_BIC6"]
        support["equad_supported_BIC6"] = None
        support["ecorr_supported_BIC6"] = None
    return (
        context,
        candidates,
        dict(
            status="screened",
            best_model=best["model"],
            best_efac=best["efac"],
            best_equad_us=best["equad_us"],
            best_ecorr_us=best["ecorr_us"],
            **support,
        ),
    )
