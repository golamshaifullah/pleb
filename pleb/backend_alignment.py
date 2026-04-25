"""Robust reference-set selection for backend alignment.

This module estimates pairwise backend offsets from overlapping TOAs, solves a
robust global offset graph with the gauge constraint ``sum(b)=0``, and ranks
candidate reference backends using graph stability and diagnostic metrics.

The input is a table-like collection with one row per TOA and at least these
fields:

- ``backend``
- ``toa``
- ``residual``
- ``toa_error``
- optional ``template``

Residual and uncertainty units are carried through transparently; output offset
units are the same as the input residual units.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BackendAlignmentObservation:
    """Single TOA measurement used for backend alignment."""

    backend: str
    toa: float
    residual: float
    toa_error: float
    template: Optional[str] = None


@dataclass(frozen=True)
class TemplateDiagnostic:
    """Per-backend template summary used in scoring and edge inflation."""

    backend: str
    template_names: tuple[str, ...]
    n_template_rows: int
    support_fraction: float
    consistent: bool
    score: float


@dataclass(frozen=True)
class PairwiseOffsetEstimate:
    """Robust pairwise additive offset estimate between two backends."""

    left_backend: str
    right_backend: str
    delta: float
    sigma: float
    weight: float
    n_pairs: int
    median_toa_separation: float
    residual_rms: float
    chi2_red: float
    inflation_factor: float
    template_confidence: float


@dataclass(frozen=True)
class ReferenceSelectionWeights:
    """Weights for ranking candidate alignment references."""

    connectivity: float = 0.30
    uncertainty: float = 0.30
    edge_rms: float = 0.20
    stability: float = 0.15
    template: float = 0.05

    def normalized(self) -> "ReferenceSelectionWeights":
        total = (
            float(self.connectivity)
            + float(self.uncertainty)
            + float(self.edge_rms)
            + float(self.stability)
            + float(self.template)
        )
        if total <= 0:
            raise ValueError("ReferenceSelectionWeights must sum to a positive value")
        return ReferenceSelectionWeights(
            connectivity=float(self.connectivity) / total,
            uncertainty=float(self.uncertainty) / total,
            edge_rms=float(self.edge_rms) / total,
            stability=float(self.stability) / total,
            template=float(self.template) / total,
        )


@dataclass(frozen=True)
class ReferenceCandidate:
    """Scored candidate backend for use as the alignment reference."""

    backend: str
    score: float
    weighted_connectivity: float
    mean_relative_uncertainty: float
    incident_edge_rms: float
    leave_one_out_stability: float
    template_score: float
    connectivity_component: float
    uncertainty_component: float
    edge_rms_component: float
    stability_component: float
    template_component: float


@dataclass
class BackendAlignmentResult:
    """Full backend-alignment inference result."""

    backends: list[str]
    edges: list[PairwiseOffsetEstimate]
    offsets: dict[str, float]
    covariance: np.ndarray
    template_diagnostics: dict[str, TemplateDiagnostic]
    ranked_references: list[ReferenceCandidate]
    chosen_reference: str
    propagated_uncertainty_relative_to_reference: dict[str, float]
    warnings: list[str]

    def edge_table(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(edge) for edge in self.edges)

    def reference_table(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(ref) for ref in self.ranked_references)

    def offset_table(self) -> pd.DataFrame:
        ref = self.chosen_reference
        rows = []
        for backend in self.backends:
            rows.append(
                {
                    "backend": backend,
                    "offset": float(self.offsets[backend]),
                    "offset_relative_to_reference": float(
                        self.offsets[backend] - self.offsets[ref]
                    ),
                    "uncertainty_relative_to_reference": float(
                        self.propagated_uncertainty_relative_to_reference[backend]
                    ),
                    "is_reference": backend == ref,
                }
            )
        return pd.DataFrame(rows)

    def covariance_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.covariance, index=self.backends, columns=self.backends)

    def template_table(self) -> pd.DataFrame:
        rows = []
        for backend in self.backends:
            diag = self.template_diagnostics[backend]
            rows.append(
                {
                    "backend": backend,
                    "template_names": ",".join(diag.template_names),
                    "n_template_rows": diag.n_template_rows,
                    "support_fraction": diag.support_fraction,
                    "consistent": diag.consistent,
                    "template_score": diag.score,
                }
            )
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class _LinearSolveResult:
    offsets: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray


def _coerce_alignment_dataframe(
    records: pd.DataFrame | Sequence[BackendAlignmentObservation | Mapping[str, object]],
) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        rows: list[dict[str, object]] = []
        for rec in records:
            if isinstance(rec, BackendAlignmentObservation):
                rows.append(asdict(rec))
            else:
                rows.append(dict(rec))
        df = pd.DataFrame(rows)

    required = ["backend", "toa", "residual", "toa_error"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing backend-alignment columns: {missing}")

    out = df.copy()
    out["backend"] = out["backend"].astype(str)
    out["toa"] = pd.to_numeric(out["toa"], errors="coerce")
    out["residual"] = pd.to_numeric(out["residual"], errors="coerce")
    out["toa_error"] = pd.to_numeric(out["toa_error"], errors="coerce")
    if "template" not in out.columns:
        out["template"] = None
    out["template"] = out["template"].where(out["template"].notna(), None)

    bad = (
        out["backend"].astype(str).str.strip().eq("")
        | ~np.isfinite(out["toa"].to_numpy(dtype=float))
        | ~np.isfinite(out["residual"].to_numpy(dtype=float))
        | ~np.isfinite(out["toa_error"].to_numpy(dtype=float))
        | (out["toa_error"].to_numpy(dtype=float) <= 0)
    )
    if bool(np.any(bad)):
        raise ValueError("Backend-alignment input contains invalid TOA rows")

    return out.sort_values(["backend", "toa"]).reset_index(drop=True)


def _normalize_metric(values: Sequence[float], *, higher_is_better: bool) -> list[float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [0.0 for _ in values]
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi == lo:
        return [1.0 if np.isfinite(v) else 0.0 for v in arr]
    scores: list[float] = []
    for value in arr:
        if not np.isfinite(value):
            scores.append(0.0)
            continue
        frac = float((value - lo) / (hi - lo))
        scores.append(frac if higher_is_better else (1.0 - frac))
    return scores


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return float(np.sum(weights * values) / total)


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    return float(np.sqrt(np.sum(weights * values**2) / total))


def _robust_location(
    values: np.ndarray,
    sigmas: np.ndarray,
    *,
    huber_k: float = 1.5,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[float, np.ndarray]:
    base_w = 1.0 / np.maximum(sigmas, 1e-12) ** 2
    mu = _weighted_mean(values, base_w)
    robust_w = np.ones_like(base_w)
    for _ in range(max_iter):
        z = (values - mu) / np.maximum(sigmas, 1e-12)
        robust_w = np.ones_like(z)
        mask = np.abs(z) > huber_k
        robust_w[mask] = huber_k / np.abs(z[mask])
        w = base_w * robust_w
        mu_new = _weighted_mean(values, w)
        if abs(mu_new - mu) <= tol * max(1.0, abs(mu), abs(mu_new)):
            mu = mu_new
            break
        mu = mu_new
    return mu, robust_w


def _compute_template_diagnostics(df: pd.DataFrame) -> dict[str, TemplateDiagnostic]:
    diagnostics: dict[str, TemplateDiagnostic] = {}
    for backend, grp in df.groupby("backend", sort=True):
        names = []
        for value in grp["template"].tolist():
            if value in (None, ""):
                continue
            text = str(value).strip()
            if text:
                names.append(text)
        unique = tuple(sorted(set(names)))
        n_rows = int(len(names))
        support_fraction = float(n_rows / max(len(grp), 1))
        consistent = bool(unique) and len(unique) == 1
        if len(unique) == 0:
            score = 0.0
        elif consistent:
            score = 1.0
        else:
            score = 0.5
        diagnostics[str(backend)] = TemplateDiagnostic(
            backend=str(backend),
            template_names=unique,
            n_template_rows=n_rows,
            support_fraction=support_fraction,
            consistent=consistent,
            score=float(score),
        )
    return diagnostics


def _match_overlapping_rows(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    max_toa_separation: float,
) -> pd.DataFrame:
    left_sorted = left.sort_values("toa").reset_index(drop=True)
    right_sorted = right.sort_values("toa").reset_index(drop=True)
    right_toas = right_sorted["toa"].to_numpy(dtype=float)
    right_toas_list = right_toas.tolist()
    used = np.zeros(len(right_sorted), dtype=bool)
    rows: list[dict[str, float]] = []

    for lrow in left_sorted.itertuples(index=False):
        toa = float(lrow.toa)
        pos = bisect_left(right_toas_list, toa)
        best_idx: Optional[int] = None
        best_sep: Optional[float] = None
        for cand in (pos - 1, pos, pos + 1):
            if cand < 0 or cand >= len(right_toas) or used[cand]:
                continue
            sep = abs(float(right_toas[cand]) - toa)
            if sep > max_toa_separation:
                continue
            if best_sep is None or sep < best_sep:
                best_idx = cand
                best_sep = sep
        if best_idx is None or best_sep is None:
            continue
        used[best_idx] = True
        rrow = right_sorted.iloc[int(best_idx)]
        rows.append(
            {
                "left_toa": float(lrow.toa),
                "right_toa": float(rrow["toa"]),
                "delta": float(rrow["residual"]) - float(lrow.residual),
                "sigma": float(
                    np.hypot(float(lrow.toa_error), float(rrow["toa_error"]))
                ),
                "toa_separation": float(best_sep),
            }
        )

    return pd.DataFrame(rows)


def _estimate_pairwise_offset(
    left_backend: str,
    right_backend: str,
    matches: pd.DataFrame,
    *,
    left_template: TemplateDiagnostic,
    right_template: TemplateDiagnostic,
    huber_k: float,
    template_penalty_scale: float,
    error_floor: float,
) -> PairwiseOffsetEstimate:
    deltas = matches["delta"].to_numpy(dtype=float)
    sigmas = np.maximum(matches["sigma"].to_numpy(dtype=float), error_floor)
    loc, robust_w = _robust_location(deltas, sigmas, huber_k=huber_k)
    base_w = 1.0 / sigmas**2
    residuals = deltas - loc
    red_denom = max(len(deltas) - 1, 1)
    chi2_red = float(np.sum((residuals / sigmas) ** 2) / red_denom)
    residual_rms = _weighted_rms(residuals, base_w)
    mad = float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))
    eff_w = base_w * robust_w
    eff_sum = float(np.sum(eff_w))
    nominal_sigma = float(np.sqrt(1.0 / max(eff_sum, 1e-12)))
    eff_n = float(eff_sum**2 / max(np.sum(eff_w**2), 1e-12))
    robust_sigma = mad / np.sqrt(max(eff_n, 1.0)) if np.isfinite(mad) else 0.0
    inflation = max(1.0, np.sqrt(max(chi2_red, 0.0)))
    if nominal_sigma > 0 and robust_sigma > 0:
        inflation = max(inflation, robust_sigma / nominal_sigma)
    template_confidence = 0.5 * (left_template.score + right_template.score)
    template_penalty = 1.0 + template_penalty_scale * (1.0 - template_confidence)
    sigma_edge = max(nominal_sigma, error_floor) * inflation * template_penalty

    return PairwiseOffsetEstimate(
        left_backend=left_backend,
        right_backend=right_backend,
        delta=float(loc),
        sigma=float(sigma_edge),
        weight=float(1.0 / max(sigma_edge, error_floor) ** 2),
        n_pairs=int(len(matches)),
        median_toa_separation=float(np.median(matches["toa_separation"].to_numpy(dtype=float))),
        residual_rms=float(residual_rms),
        chi2_red=float(chi2_red),
        inflation_factor=float(inflation * template_penalty),
        template_confidence=float(template_confidence),
    )


def estimate_pairwise_backend_offsets(
    records: pd.DataFrame | Sequence[BackendAlignmentObservation | Mapping[str, object]],
    *,
    max_toa_separation: float = 7.0,
    min_overlap_pairs: int = 3,
    huber_k: float = 1.5,
    template_penalty_scale: float = 0.5,
    error_floor: float = 1e-6,
) -> tuple[list[str], list[PairwiseOffsetEstimate], dict[str, TemplateDiagnostic]]:
    """Estimate robust pairwise additive offsets between backends.

    Pairwise offsets are inferred from overlapping TOAs using a robust Huber
    location estimate of residual differences. Missing or inconsistent template
    support inflates edge uncertainty and is also reported separately.
    """
    df = _coerce_alignment_dataframe(records)
    template_diags = _compute_template_diagnostics(df)
    backends = sorted(df["backend"].unique().tolist())
    per_backend = {b: g.copy() for b, g in df.groupby("backend", sort=True)}

    edges: list[PairwiseOffsetEstimate] = []
    for idx, left_backend in enumerate(backends):
        for right_backend in backends[idx + 1 :]:
            matches = _match_overlapping_rows(
                per_backend[left_backend],
                per_backend[right_backend],
                max_toa_separation=max_toa_separation,
            )
            if len(matches) < int(min_overlap_pairs):
                continue
            edges.append(
                _estimate_pairwise_offset(
                    left_backend,
                    right_backend,
                    matches,
                    left_template=template_diags[left_backend],
                    right_template=template_diags[right_backend],
                    huber_k=huber_k,
                    template_penalty_scale=template_penalty_scale,
                    error_floor=error_floor,
                )
            )

    return backends, edges, template_diags


def _connected_backends(backends: Sequence[str], edges: Sequence[PairwiseOffsetEstimate]) -> list[str]:
    if not backends:
        return []
    adjacency = {backend: set() for backend in backends}
    for edge in edges:
        adjacency[edge.left_backend].add(edge.right_backend)
        adjacency[edge.right_backend].add(edge.left_backend)
    seen = set()
    queue = [str(backends[0])]
    while queue:
        node = queue.pop()
        if node in seen:
            continue
        seen.add(node)
        queue.extend(sorted(adjacency[node] - seen))
    return sorted(seen)


def _solve_constrained_wls(
    backends: Sequence[str],
    edges: Sequence[PairwiseOffsetEstimate],
    *,
    huber_k: float = 1.5,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> _LinearSolveResult:
    n_backends = len(backends)
    if n_backends == 1:
        return _LinearSolveResult(
            offsets=np.zeros(1, dtype=float),
            covariance=np.zeros((1, 1), dtype=float),
            residuals=np.zeros(0, dtype=float),
        )
    if not edges:
        raise ValueError("Cannot solve backend offsets without pairwise edges")
    connected = _connected_backends(backends, edges)
    if len(connected) != n_backends:
        missing = sorted(set(backends) - set(connected))
        raise ValueError(
            "Backend offset graph is disconnected; missing connectivity for "
            + ", ".join(missing)
        )

    index = {backend: i for i, backend in enumerate(backends)}
    a = np.zeros((len(edges), n_backends), dtype=float)
    y = np.zeros(len(edges), dtype=float)
    sigma = np.zeros(len(edges), dtype=float)
    for row, edge in enumerate(edges):
        a[row, index[edge.left_backend]] = -1.0
        a[row, index[edge.right_backend]] = 1.0
        y[row] = float(edge.delta)
        sigma[row] = float(edge.sigma)

    base_w = 1.0 / np.maximum(sigma, 1e-12) ** 2
    robust_w = np.ones_like(base_w)
    b = np.zeros(n_backends, dtype=float)
    cov = np.zeros((n_backends, n_backends), dtype=float)

    for _ in range(max_iter):
        w = base_w * robust_w
        lhs = a.T @ (w[:, None] * a)
        rhs = a.T @ (w * y)
        constraint = np.ones((n_backends, 1), dtype=float)
        kkt = np.block(
            [[lhs, constraint], [constraint.T, np.zeros((1, 1), dtype=float)]]
        )
        kkt_rhs = np.concatenate([rhs, np.zeros(1, dtype=float)])
        sol = np.linalg.solve(kkt, kkt_rhs)
        b_new = sol[:n_backends]
        resid = y - a @ b_new
        z = resid / np.maximum(sigma, 1e-12)
        robust_new = np.ones_like(z)
        mask = np.abs(z) > huber_k
        robust_new[mask] = huber_k / np.abs(z[mask])
        if np.linalg.norm(b_new - b) <= tol * max(1.0, np.linalg.norm(b), np.linalg.norm(b_new)):
            b = b_new
            robust_w = robust_new
            kkt_inv = np.linalg.pinv(kkt)
            cov = kkt_inv[:n_backends, :n_backends]
            break
        b = b_new
        robust_w = robust_new
        kkt_inv = np.linalg.pinv(kkt)
        cov = kkt_inv[:n_backends, :n_backends]

    residuals = y - a @ b
    return _LinearSolveResult(offsets=b, covariance=cov, residuals=residuals)


def _propagated_relative_uncertainty(
    covariance: np.ndarray,
    backends: Sequence[str],
    reference_backend: str,
) -> dict[str, float]:
    index = {backend: i for i, backend in enumerate(backends)}
    ref_idx = index[reference_backend]
    out: dict[str, float] = {}
    for backend in backends:
        idx = index[backend]
        variance = (
            covariance[idx, idx]
            + covariance[ref_idx, ref_idx]
            - 2.0 * covariance[idx, ref_idx]
        )
        out[backend] = float(np.sqrt(max(variance, 0.0)))
    return out


def _leave_one_out_stability(
    candidate: str,
    *,
    backends: Sequence[str],
    edges: Sequence[PairwiseOffsetEstimate],
    full_offsets: Mapping[str, float],
    huber_k: float,
) -> float:
    incident = [edge for edge in edges if candidate in (edge.left_backend, edge.right_backend)]
    if not incident:
        return float("inf")
    full_rel = np.asarray(
        [float(full_offsets[b] - full_offsets[candidate]) for b in backends], dtype=float
    )
    shifts: list[float] = []
    weights: list[float] = []
    for edge in incident:
        reduced = [other for other in edges if other is not edge]
        try:
            solved = _solve_constrained_wls(backends, reduced, huber_k=huber_k)
        except ValueError:
            shifts.append(float("inf"))
            weights.append(float(edge.weight))
            continue
        sub_offsets = {
            backend: float(value) for backend, value in zip(backends, solved.offsets)
        }
        sub_rel = np.asarray(
            [float(sub_offsets[b] - sub_offsets[candidate]) for b in backends],
            dtype=float,
        )
        diff = sub_rel - full_rel
        shifts.append(float(np.sqrt(np.mean(diff**2))))
        weights.append(float(edge.weight))
    if not shifts:
        return float("inf")
    if not np.isfinite(np.asarray(shifts, dtype=float)).all():
        return float("inf")
    return float(np.average(np.asarray(shifts, dtype=float), weights=np.asarray(weights, dtype=float)))


def _rank_reference_candidates(
    backends: Sequence[str],
    edges: Sequence[PairwiseOffsetEstimate],
    offsets: Mapping[str, float],
    covariance: np.ndarray,
    template_diags: Mapping[str, TemplateDiagnostic],
    *,
    score_weights: ReferenceSelectionWeights,
    huber_k: float,
) -> list[ReferenceCandidate]:
    weight_by_backend = {backend: 0.0 for backend in backends}
    edge_rms_by_backend: dict[str, list[tuple[float, float]]] = {backend: [] for backend in backends}
    for edge in edges:
        for backend in (edge.left_backend, edge.right_backend):
            weight_by_backend[backend] += float(edge.weight)
            edge_rms_by_backend[backend].append((float(edge.residual_rms), float(edge.weight)))

    uncertainty_raw: list[float] = []
    connectivity_raw: list[float] = []
    edge_rms_raw: list[float] = []
    stability_raw: list[float] = []
    template_raw: list[float] = []
    rel_unc_cache: dict[str, float] = {}
    stability_cache: dict[str, float] = {}
    incident_rms_cache: dict[str, float] = {}

    for backend in backends:
        rel_unc = _propagated_relative_uncertainty(covariance, backends, backend)
        rel_unc_vals = [rel_unc[b] for b in backends if b != backend]
        mean_rel_unc = float(np.mean(rel_unc_vals)) if rel_unc_vals else 0.0
        rel_unc_cache[backend] = mean_rel_unc
        stability = _leave_one_out_stability(
            backend,
            backends=backends,
            edges=edges,
            full_offsets=offsets,
            huber_k=huber_k,
        )
        stability_cache[backend] = stability
        incident = edge_rms_by_backend[backend]
        if incident:
            vals = np.asarray([x[0] for x in incident], dtype=float)
            wts = np.asarray([x[1] for x in incident], dtype=float)
            incident_rms = float(np.average(vals, weights=wts))
        else:
            incident_rms = float("inf")
        incident_rms_cache[backend] = incident_rms
        connectivity_raw.append(float(weight_by_backend[backend]))
        uncertainty_raw.append(mean_rel_unc)
        edge_rms_raw.append(incident_rms)
        stability_raw.append(stability)
        template_raw.append(float(template_diags[backend].score))

    conn_component = _normalize_metric(connectivity_raw, higher_is_better=True)
    unc_component = _normalize_metric(uncertainty_raw, higher_is_better=False)
    rms_component = _normalize_metric(edge_rms_raw, higher_is_better=False)
    stability_component = _normalize_metric(stability_raw, higher_is_better=False)
    template_component = _normalize_metric(template_raw, higher_is_better=True)
    weights = score_weights.normalized()

    ranked: list[ReferenceCandidate] = []
    for idx, backend in enumerate(backends):
        score = (
            weights.connectivity * conn_component[idx]
            + weights.uncertainty * unc_component[idx]
            + weights.edge_rms * rms_component[idx]
            + weights.stability * stability_component[idx]
            + weights.template * template_component[idx]
        )
        ranked.append(
            ReferenceCandidate(
                backend=str(backend),
                score=float(score),
                weighted_connectivity=float(connectivity_raw[idx]),
                mean_relative_uncertainty=float(rel_unc_cache[backend]),
                incident_edge_rms=float(incident_rms_cache[backend]),
                leave_one_out_stability=float(stability_cache[backend]),
                template_score=float(template_diags[backend].score),
                connectivity_component=float(conn_component[idx]),
                uncertainty_component=float(unc_component[idx]),
                edge_rms_component=float(rms_component[idx]),
                stability_component=float(stability_component[idx]),
                template_component=float(template_component[idx]),
            )
        )

    ranked.sort(
        key=lambda row: (
            -float(row.score),
            float(row.mean_relative_uncertainty),
            -float(row.weighted_connectivity),
            float(row.incident_edge_rms),
            float(row.leave_one_out_stability),
            str(row.backend),
        )
    )
    return ranked


def select_backend_alignment_reference(
    records: pd.DataFrame | Sequence[BackendAlignmentObservation | Mapping[str, object]],
    *,
    max_toa_separation: float = 7.0,
    min_overlap_pairs: int = 3,
    huber_k: float = 1.5,
    template_penalty_scale: float = 0.5,
    error_floor: float = 1e-6,
    score_weights: ReferenceSelectionWeights | None = None,
) -> BackendAlignmentResult:
    """Infer global backend offsets and choose a robust alignment reference."""
    backends, edges, template_diags = estimate_pairwise_backend_offsets(
        records,
        max_toa_separation=max_toa_separation,
        min_overlap_pairs=min_overlap_pairs,
        huber_k=huber_k,
        template_penalty_scale=template_penalty_scale,
        error_floor=error_floor,
    )
    if not backends:
        raise ValueError("No backends were provided for backend alignment")
    if len(backends) == 1:
        backend = backends[0]
        ranked = [
            ReferenceCandidate(
                backend=backend,
                score=1.0,
                weighted_connectivity=0.0,
                mean_relative_uncertainty=0.0,
                incident_edge_rms=0.0,
                leave_one_out_stability=0.0,
                template_score=float(template_diags[backend].score),
                connectivity_component=1.0,
                uncertainty_component=1.0,
                edge_rms_component=1.0,
                stability_component=1.0,
                template_component=float(template_diags[backend].score),
            )
        ]
        return BackendAlignmentResult(
            backends=backends,
            edges=[],
            offsets={backend: 0.0},
            covariance=np.zeros((1, 1), dtype=float),
            template_diagnostics=template_diags,
            ranked_references=ranked,
            chosen_reference=backend,
            propagated_uncertainty_relative_to_reference={backend: 0.0},
            warnings=[],
        )
    if not edges:
        raise ValueError(
            "No pairwise backend offsets could be estimated; check overlap coverage or relax min_overlap_pairs"
        )

    solve = _solve_constrained_wls(backends, edges, huber_k=huber_k)
    offsets = {backend: float(value) for backend, value in zip(backends, solve.offsets)}
    ranked = _rank_reference_candidates(
        backends,
        edges,
        offsets,
        solve.covariance,
        template_diags,
        score_weights=score_weights or ReferenceSelectionWeights(),
        huber_k=huber_k,
    )
    chosen = ranked[0].backend
    propagated = _propagated_relative_uncertainty(solve.covariance, backends, chosen)
    warnings: list[str] = []
    for backend, diag in template_diags.items():
        if diag.score <= 0:
            warnings.append(f"No template diagnostic available for backend {backend}")
    return BackendAlignmentResult(
        backends=list(backends),
        edges=list(edges),
        offsets=offsets,
        covariance=solve.covariance,
        template_diagnostics=dict(template_diags),
        ranked_references=ranked,
        chosen_reference=chosen,
        propagated_uncertainty_relative_to_reference=propagated,
        warnings=warnings,
    )


def _plot_reference_ranking(result: BackendAlignmentResult, path: Path) -> None:
    import matplotlib.pyplot as plt

    table = result.reference_table()
    fig, ax = plt.subplots(figsize=(max(6, 1.25 * len(table)), 4))
    ax.bar(table["backend"], table["score"], color="#3b82f6")
    ax.set_ylabel("reference score")
    ax.set_title("Backend-alignment reference ranking")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_offset_graph(result: BackendAlignmentResult, path: Path) -> None:
    import matplotlib.pyplot as plt

    offsets = result.offset_table().set_index("backend")
    x = np.arange(len(result.backends), dtype=float)
    y = np.asarray(
        [offsets.loc[backend, "offset_relative_to_reference"] for backend in result.backends],
        dtype=float,
    )
    yerr = np.asarray(
        [
            offsets.loc[backend, "uncertainty_relative_to_reference"]
            for backend in result.backends
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(result.backends)), 4.5))
    max_w = max((edge.weight for edge in result.edges), default=1.0)
    index = {backend: i for i, backend in enumerate(result.backends)}
    for edge in result.edges:
        xi = x[index[edge.left_backend]]
        xj = x[index[edge.right_backend]]
        yi = y[index[edge.left_backend]]
        yj = y[index[edge.right_backend]]
        alpha = 0.2 + 0.6 * float(edge.weight / max_w)
        ax.plot([xi, xj], [yi, yj], color="#94a3b8", linewidth=1.2, alpha=alpha)
    ax.errorbar(x, y, yerr=yerr, fmt="o", color="#0f172a", capsize=3)
    for xi, yi, backend in zip(x, y, result.backends):
        label = backend
        if backend == result.chosen_reference:
            label = f"{backend} (ref)"
        ax.text(xi, yi, label, fontsize=9, ha="center", va="bottom")
    ax.set_xticks([])
    ax.set_ylabel("offset relative to chosen reference")
    ax.set_title("Backend offset graph solution")
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_covariance_heatmap(result: BackendAlignmentResult, path: Path) -> None:
    import matplotlib.pyplot as plt

    cov = result.covariance_table().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(result.backends)), max(4, 0.9 * len(result.backends))))
    im = ax.imshow(cov, cmap="viridis")
    ax.set_xticks(np.arange(len(result.backends)))
    ax.set_xticklabels(result.backends, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(result.backends)))
    ax.set_yticklabels(result.backends)
    ax.set_title("Backend-offset covariance")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_backend_alignment_outputs(
    result: BackendAlignmentResult,
    out_dir: Path | str,
    *,
    prefix: str = "backend_alignment",
) -> dict[str, Path]:
    """Write alignment tables, plots, and a JSON summary to disk."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    files = {
        "edges": out_path / f"{prefix}_edges.csv",
        "references": out_path / f"{prefix}_references.csv",
        "offsets": out_path / f"{prefix}_offsets.csv",
        "covariance": out_path / f"{prefix}_covariance.csv",
        "templates": out_path / f"{prefix}_templates.csv",
        "summary": out_path / f"{prefix}_summary.json",
        "ranking_plot": out_path / f"{prefix}_ranking.png",
        "graph_plot": out_path / f"{prefix}_graph.png",
        "covariance_plot": out_path / f"{prefix}_covariance.png",
    }
    result.edge_table().to_csv(files["edges"], index=False)
    result.reference_table().to_csv(files["references"], index=False)
    result.offset_table().to_csv(files["offsets"], index=False)
    result.covariance_table().to_csv(files["covariance"], index=True)
    result.template_table().to_csv(files["templates"], index=False)
    summary = {
        "chosen_reference": result.chosen_reference,
        "backends": result.backends,
        "warnings": result.warnings,
        "n_edges": len(result.edges),
    }
    files["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_reference_ranking(result, files["ranking_plot"])
    _plot_offset_graph(result, files["graph_plot"])
    _plot_covariance_heatmap(result, files["covariance_plot"])
    return files


__all__ = [
    "BackendAlignmentObservation",
    "BackendAlignmentResult",
    "PairwiseOffsetEstimate",
    "ReferenceCandidate",
    "ReferenceSelectionWeights",
    "TemplateDiagnostic",
    "estimate_pairwise_backend_offsets",
    "select_backend_alignment_reference",
    "write_backend_alignment_outputs",
]
