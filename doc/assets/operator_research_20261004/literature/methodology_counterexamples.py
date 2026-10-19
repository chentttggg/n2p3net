"""Independent small mathematical probes, not EEG performance evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import integrate, linalg, signal, special, stats


def log_density(y, mean, covariance):
    residual = y - mean
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0
    return -0.5 * (residual @ np.linalg.solve(covariance, residual) + logdet)


def run():
    rng = np.random.default_rng(61004)
    output = {"evidence_scope": "software_correctness", "seed": 61004}
    n = 12
    factor = rng.normal(size=(n, n))
    C = factor @ factor.T + np.eye(n)
    F = np.eye(n) + rng.normal(size=(n, n)) / 6
    y, m1, m2 = rng.normal(size=(3, n))
    original = log_density(y, m1, C) - log_density(y, m2, C)
    transformed = (log_density(F @ y, F @ m1, F @ C @ F.T)
                   - log_density(F @ y, F @ m2, F @ C @ F.T))
    assert abs(original - transformed) < 1e-10
    output["invertible_joint_transform"] = {
        "log_bayes_factor": original,
        "transformed_log_bayes_factor": transformed,
        "absolute_error": abs(original - transformed),
    }

    # Lossy linear maps cannot add population Gaussian discrimination.
    delta = m1 - m2
    P = rng.normal(size=(5, n))
    original_energy = delta @ np.linalg.solve(C, delta)
    projected_energy = (P @ delta) @ np.linalg.solve(P @ C @ P.T, P @ delta)
    assert projected_energy <= original_energy + 1e-12
    output["rank_reduction"] = {
        "raw_mahalanobis_energy": original_energy,
        "projected_mahalanobis_energy": projected_energy,
    }

    # Signal-only transform can reverse a correct candidate under known covariance.
    F = np.diag([4., .2])
    C = np.eye(2)
    witness = None
    for _ in range(1000):
        y, m1, m2 = rng.normal(size=(3, 2))
        original = log_density(y, m1, C) - log_density(y, m2, C)
        wrong = (log_density(F @ y, m1, F @ C @ F.T)
                 - log_density(F @ y, m2, F @ C @ F.T))
        if original * wrong < 0:
            witness = {"sample": y.tolist(), "means": [m1.tolist(), m2.tolist()],
                       "correct_log_bayes_factor": original,
                       "signal_only_wrong_log_bayes_factor": wrong}
            break
    assert witness is not None
    output["filter_signal_without_mean_reversal"] = witness

    # Finite filtfilt plus compact causal design is not convolution with one
    # compact causal filtered kernel, even with an interior impulse to estimate it.
    n, length, onset = 360, 80, 250
    sfreq = 100.
    sos = signal.butter(2, [.5, 15.], fs=sfreq, btype="bandpass", output="sos")
    h = np.sin(np.pi * (np.arange(length) + .5) / length) ** 2
    raw = np.zeros(n)
    raw[onset:onset + length] = h
    true_mean = signal.sosfiltfilt(sos, raw)
    interior_onset = 140
    interior = np.zeros(n)
    interior[interior_onset:interior_onset + length] = h
    filtered_kernel = signal.sosfiltfilt(sos, interior)[interior_onset:interior_onset + length]
    compact_design_mean = np.zeros(n)
    compact_design_mean[onset:onset + length] = filtered_kernel
    relative_error = np.linalg.norm(true_mean - compact_design_mean) / np.linalg.norm(true_mean)
    pre_onset_fraction = np.linalg.norm(true_mean[:onset]) ** 2 / np.linalg.norm(true_mean) ** 2
    assert relative_error > .1 and pre_onset_fraction > .01
    # The fixed finite matrix itself remains exactly linear (up to rounding).
    F = signal.sosfiltfilt(sos, np.eye(n), axis=0)
    linear_error = np.max(np.abs(F @ raw - true_mean))
    assert linear_error < 1e-10
    output["finite_filter_compact_causal_mismatch"] = {
        "relative_mean_error": relative_error,
        "pre_onset_energy_fraction": pre_onset_fraction,
        "exact_finite_matrix_max_error": linear_error,
        "settings": {"sfreq": sfreq, "band_hz": [.5, 15.], "order": 2,
                     "n_samples": n, "response_samples": length, "onset": onset},
    }

    # A state driven by the same residual is correlated, not independent noise.
    C_joint = np.array([[1., -.8], [-.8, 1.]])
    A = np.array([[1., 1.]])
    true_variance = float((A @ C_joint @ A.T).item())
    independent_variance = float(np.trace(C_joint))
    assert true_variance == 0.3999999999999999
    output["discarding_raw_state_cross_terms"] = {
        "correct_variance": true_variance, "wrong_independent_sum": independent_variance,
        "wrong_over_correct": independent_variance / true_variance,
    }
    # If the observed residual and random initial state are jointly Gaussian,
    # conditioning the initial state on that observation reduces its covariance.
    conditional_gamma = C_joint[1, 1] - C_joint[1, 0] ** 2 / C_joint[0, 0]
    output["conditional_state_covariance"] = {
        "prior_gamma": C_joint[1, 1], "gamma_given_observed_residual": conditional_gamma,
        "state_mean_per_observed_unit": C_joint[1, 0] / C_joint[0, 0],
    }
    assert abs(conditional_gamma - .36) < 1e-12

    # Integrating one GLOBAL inverse-gamma scale produces a multivariate t.
    # It does not equal a product of per-coordinate t densities.
    alpha, beta = 3., 2.
    mean = np.zeros(2)
    shape = np.array([[1., .3], [.3, 2.]])
    observation = np.array([2., -1.])
    root = np.linalg.cholesky(shape)
    quadratic = np.sum(linalg.solve_triangular(root, observation, lower=True) ** 2)
    logdet = 2 * np.log(np.diag(root)).sum()
    d = len(mean)
    log_closed = (special.gammaln(alpha + d / 2) - special.gammaln(alpha)
                  + alpha * np.log(beta) - .5 * logdet - d / 2 * np.log(2 * np.pi)
                  - (alpha + d / 2) * np.log(beta + quadratic / 2))
    numeric = integrate.quad(
        lambda variance: stats.multivariate_normal.pdf(observation, mean, variance * shape)
        * stats.invgamma.pdf(variance, a=alpha, scale=beta), 0, np.inf,
        epsabs=1e-12, epsrel=1e-10)[0]
    scipy_log = stats.multivariate_t.logpdf(observation, mean, beta / alpha * shape, 2 * alpha)
    assert abs(log_closed - np.log(numeric)) < 1e-9
    assert abs(log_closed - scipy_log) < 1e-10
    output["inverse_gamma_global_scale_marginal"] = {
        "closed_log_density": float(log_closed), "quadrature_log_density": float(np.log(numeric)),
        "scipy_multivariate_t_log_density": float(scipy_log),
        "scope": "one shared scalar variance, NOT inverse-Wishart matrix-t regression",
    }
    logits = np.array([8., 7., -2.])
    output["temperature_preserves_argmax"] = [int(np.argmax(logits / t)) for t in [.1, 1., 10., 100.]]
    assert len(set(output["temperature_preserves_argmax"])) == 1
    output["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return output


if __name__ == "__main__":
    result = run()
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
