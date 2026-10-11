"""Conditional AR estimand boundaries, independently checked with dense Gaussians."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import linalg


def log_normal(y, mean, covariance):
    r = y - mean
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0
    return -.5 * (r @ np.linalg.solve(covariance, r) + logdet)


def conditional_operator(n, phi):
    p = len(phi)
    W = np.zeros((n - p, n))
    for row, t in enumerate(range(p, n)):
        W[row, t] = 1.
        for lag, coefficient in enumerate(phi, start=1):
            W[row, t - lag] = -coefficient
    return W


def integrated_flat_nuisance_log(y, mean, covariance, nuisance):
    precision = np.linalg.inv(covariance)
    normal = nuisance.T @ precision @ nuisance
    P = precision - precision @ nuisance @ np.linalg.solve(normal, nuisance.T @ precision)
    r = y - mean
    return -.5 * (r @ P @ r + np.linalg.slogdet(covariance)[1] + np.linalg.slogdet(normal)[1])


def run():
    rng = np.random.default_rng(43251)
    n, p, phi = 14, 1, np.array([.8])
    W = conditional_operator(n, phi)
    covariance = linalg.toeplitz(phi[0] ** np.arange(n)) / (1 - phi[0]**2)
    means = rng.normal(size=(2, n))
    means[:, :p] = 0.
    observed = rng.normal(size=n)
    full_bf = (log_normal(observed, means[0], covariance)
               - log_normal(observed, means[1], covariance))
    conditional_bf = (log_normal(W @ observed, W @ means[0], np.eye(n - p))
                      - log_normal(W @ observed, W @ means[1], np.eye(n - p)))
    assert abs(full_bf - conditional_bf) < 1e-11
    assert np.max(np.abs(W @ covariance @ W.T - np.eye(n - p))) < 1e-12

    # Uncertain mean kernel alone retains equality if it cannot affect prefix.
    kernel_variance = .4
    designs = rng.normal(size=(2, n))
    designs[:, :p] = 0.
    kernel_mean = .7
    integrated_full = [log_normal(observed, d * kernel_mean,
                                 covariance + kernel_variance * np.outer(d, d)) for d in designs]
    integrated_conditional = [log_normal(W @ observed, W @ d * kernel_mean,
                                        np.eye(n - p) + kernel_variance * np.outer(W @ d, W @ d))
                              for d in designs]
    uncertain_bf_error = abs((integrated_full[0] - integrated_full[1])
                             - (integrated_conditional[0] - integrated_conditional[1]))
    assert uncertain_bf_error < 1e-11

    # Prefix has no ERP candidate term, but identifies unknown constant nuisance.
    # Profiling/integrating a flat nuisance using only conditional rows loses it.
    n = 3
    W = conditional_operator(n, [.8])
    R = .25 * linalg.toeplitz(.8 ** np.arange(n)) / (1 - .8**2)
    y = np.array([4., 5., 5.])
    plus, minus = np.array([0., 1., 1.]), np.array([0., -1., -1.])
    nuisance = np.ones((n, 1))
    full = [integrated_flat_nuisance_log(y, d, R, nuisance) for d in [plus, minus]]
    conditional = [integrated_flat_nuisance_log(W @ y, W @ d, W @ R @ W.T, W @ nuisance)
                   for d in [plus, minus]]
    assert abs((full[0] - full[1]) - (conditional[0] - conditional[1])) > 1.
    flip_y = np.array([4., 1., -2.])
    flip_full = [integrated_flat_nuisance_log(flip_y, d, R, nuisance) for d in [plus, minus]]
    flip_cond = [integrated_flat_nuisance_log(W @ flip_y, W @ d, W @ R @ W.T, W @ nuisance)
                 for d in [plus, minus]]
    assert (flip_full[0] - flip_full[1]) * (flip_cond[0] - flip_cond[1]) < 0

    # A proper prior makes the exact Bayesian conditioning identity explicit.
    prior_variance = 100.
    total_covariance = R + prior_variance * nuisance @ nuisance.T
    prefix_covariance = total_covariance[:p, :p]
    cross = total_covariance[p:, :p]
    conditional_covariance = (total_covariance[p:, p:]
                              - cross @ np.linalg.solve(prefix_covariance, cross.T))
    correct = []
    proper_full = []
    naive = []
    for d in [plus, minus]:
        proper_full.append(log_normal(y, d, total_covariance))
        conditioned_mean = d[p:] + cross @ np.linalg.solve(prefix_covariance, y[:p] - d[:p])
        correct.append(log_normal(y[p:], conditioned_mean, conditional_covariance))
        naive_cov = W @ R @ W.T + prior_variance * (W @ nuisance) @ (W @ nuisance).T
        naive.append(log_normal(W @ y, W @ d, naive_cov))
    exact_conditional_error = abs((proper_full[0] - proper_full[1]) - (correct[0] - correct[1]))
    assert exact_conditional_error < 1e-11
    assert abs((naive[0] - naive[1]) - (correct[0] - correct[1])) > 1.

    # If upstream processing causes candidate-dependent pre-response, even known
    # parameters no longer permit the startup density to cancel.
    known_prefix_means = [plus.copy(), minus.copy()]
    known_prefix_means[0][0] = 2.
    known_prefix_means[1][0] = -2.
    known_full = [log_normal(y, d, R) for d in known_prefix_means]
    known_cond = [log_normal(W @ y, W @ d, W @ R @ W.T) for d in known_prefix_means]
    known_prefix = [log_normal(y[:p], d[:p], R[:p, :p]) for d in known_prefix_means]
    observed_difference = (known_full[0] - known_full[1]) - (known_cond[0] - known_cond[1])
    prefix_difference = known_prefix[0] - known_prefix[1]
    assert abs(observed_difference - prefix_difference) < 1e-10
    assert abs(prefix_difference) > 1.
    return {
        "evidence_scope": "software_correctness", "seed": 43251,
        "fixed_parameters_candidate_independent_prefix": {
            "full_BF": full_bf, "conditional_BF": conditional_bf,
            "absolute_error": abs(full_bf - conditional_bf)},
        "uncertain_kernel_no_prefix_effect_known_nuisance": {"BF_error": uncertain_bf_error},
        "flat_nuisance_counterexample": {
            "y": y.tolist(), "candidate_means": [plus.tolist(), minus.tolist()],
            "full_BF": full[0] - full[1], "conditional_rows_only_BF": conditional[0] - conditional[1],
            "prefix_has_no_candidate_mean": True,
            "ranking_reversal": {"y": flip_y.tolist(), "full_BF": flip_full[0] - flip_full[1],
                                 "conditional_rows_only_BF": flip_cond[0] - flip_cond[1]}},
        "proper_nuisance_prior_conditioning": {
            "full_BF": proper_full[0] - proper_full[1], "properly_conditioned_BF": correct[0] - correct[1],
            "naive_unupdated_nuisance_prior_BF": naive[0] - naive[1],
            "exact_conditioning_error": exact_conditional_error},
        "candidate_dependent_prefix": {
            "full_minus_conditional_BF": observed_difference, "prefix_BF": prefix_difference},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    result = run()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
