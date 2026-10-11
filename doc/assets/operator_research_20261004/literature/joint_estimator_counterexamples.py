"""Independent estimator/gain probes; no real data or production modification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import linalg, optimize, signal
from sklearn.covariance import LedoitWolf


def burg1(x, covariance=None):
    if covariance is not None:
        x = linalg.solve_triangular(np.linalg.cholesky(covariance), x.T, lower=True).T
    f, b = x[1:], x[:-1]
    return 2 * float(np.sum(f * b)) / float(np.sum(f * f) + np.sum(b * b))


def correlation_shrinkage(x):
    scales = np.sqrt(np.mean(x**2, axis=0))
    normalized = x / scales
    covariance = LedoitWolf(assume_centered=True).fit(normalized).covariance_
    return covariance * scales[:, None] * scales[None, :]


def acf1(x):
    return np.sum(x[1:] * x[:-1], axis=0) / np.sum(x**2, axis=0)


def stationary_transform(x, phi):
    result = x.copy()
    result[0] *= np.sqrt(1 - phi**2)
    result[1:] -= phi * x[:-1]
    return result


def gaussian_objective(records, designs, beta, phi, covariance):
    root = np.linalg.cholesky(covariance)
    dimension = root.shape[0]
    n = sum(len(r) for r in records)
    temporal_logdet = -len(records) * np.log(1 - phi**2)
    quadratic = 0.
    for y, design in zip(records, designs, strict=True):
        e = stationary_transform(y - design @ beta, phi)
        white = linalg.solve_triangular(root, e.T, lower=True)
        quadratic += float(np.sum(white**2))
    return (n * 2 * np.log(np.diag(root)).sum()
            + dimension * temporal_logdet + quadratic)


def fit_joint(records, designs):
    # Explicit exact stationary AR(1) Gaussian ML, no ridge/shrinkage.
    # This probe does not claim the current research implementation is ML.
    phi = 0.
    beta = np.linalg.lstsq(np.vstack(designs), np.vstack(records), rcond=None)[0]
    covariance = np.eye(records[0].shape[1])
    history = []
    for _ in range(80):
        transformed_y = np.vstack([stationary_transform(y, phi) for y in records])
        transformed_d = np.vstack([stationary_transform(d, phi) for d in designs])
        beta = np.linalg.lstsq(transformed_d, transformed_y, rcond=None)[0]
        e = transformed_y - transformed_d @ beta
        covariance = e.T @ e / len(e)
        objective_before_ar = gaussian_objective(records, designs, beta, phi, covariance)
        result = optimize.minimize_scalar(
            lambda value, beta=beta, covariance=covariance:
                gaussian_objective(records, designs, beta, value, covariance),
            bounds=(-.995, .995), method="bounded", options={"xatol": 1e-12})
        assert result.success
        phi = float(result.x)
        objective = gaussian_objective(records, designs, beta, phi, covariance)
        assert objective <= objective_before_ar + 1e-7
        if history:
            assert objective <= history[-1]["objective"] + 1e-7
        history.append({"phi": phi, "objective": float(objective)})
        if len(history) > 1 and abs(history[-1]["objective"] - history[-2]["objective"]) < 1e-9:
            break
    # End by solving mean and covariance at the actually saved temporal state.
    transformed_y = np.vstack([stationary_transform(y, phi) for y in records])
    transformed_d = np.vstack([stationary_transform(d, phi) for d in designs])
    beta = np.linalg.lstsq(transformed_d, transformed_y, rcond=None)[0]
    e = transformed_y - transformed_d @ beta
    covariance = e.T @ e / len(e)
    return phi, beta, covariance, history


def run():
    rng = np.random.default_rng(1042026)
    n, c = 18000, 8
    # Deliberately nonseparable process: one different spectral shape.
    phis = np.array([.05] + [.92] * (c - 1))
    x = np.column_stack([signal.lfilter([np.sqrt(1 - phi**2)], [1., -phi],
                                       rng.normal(size=n + 500))[500:] for phi in phis])
    x -= x.mean(axis=0)
    gain = np.diag([100.] + [1.] * (c - 1))
    gained = x @ gain
    empirical = x.T @ x / len(x)
    gained_covariance = gain.T @ empirical @ gain
    unweighted = burg1(x)
    unweighted_gain = burg1(gained)
    weighted = burg1(x, empirical)
    weighted_gain = burg1(gained, gained_covariance)
    assert abs(unweighted - unweighted_gain) > .5
    assert abs(weighted - weighted_gain) < 1e-12
    assert abs(burg1(x * 1e6) - unweighted) < 1e-12
    residual_unweighted_gain = x[1:] - unweighted_gain * x[:-1]
    residual_weighted_gain = x[1:] - weighted_gain * x[:-1]

    raw_lw = LedoitWolf(assume_centered=True).fit(x).covariance_
    gain_lw = LedoitWolf(assume_centered=True).fit(gained).covariance_
    restored_raw_lw = np.linalg.solve(gain.T, np.linalg.solve(gain, gain_lw.T).T)
    relative_lw_error = np.linalg.norm(restored_raw_lw - raw_lw) / np.linalg.norm(raw_lw)
    correlation_lw = correlation_shrinkage(x)
    gain_correlation_lw = correlation_shrinkage(gained)
    restored_correlation = np.linalg.solve(gain.T, np.linalg.solve(gain, gain_correlation_lw.T).T)
    relative_correlation_error = np.linalg.norm(restored_correlation - correlation_lw) / np.linalg.norm(correlation_lw)
    assert relative_lw_error > .05
    assert relative_correlation_error < 1e-12

    # Exact shared-AR data: matrix-normal ML should obey channel congruence.
    innovation_root = np.array([[1., 0., 0.], [.5, .8, 0.], [.2, -.1, 1.1]])
    beta_true = rng.normal(size=(3, 3))
    designs, records = [], []
    for _ in range(3):
        n_record = 170
        t = np.linspace(-1, 1, n_record)
        design = np.column_stack([np.ones(n_record), t, np.exp(-((t - .1) / .3)**2)])
        errors = signal.lfilter([1.], [1., -.86],
                               rng.normal(size=(n_record + 600, 3)) @ innovation_root.T, axis=0)[600:]
        designs.append(design)
        records.append(design @ beta_true + errors)
    phi, beta, covariance, history = fit_joint(records, designs)
    gain3 = np.diag([30., .2, 4.])
    phi_gain, beta_gain, covariance_gain, history_gain = fit_joint([r @ gain3 for r in records], designs)
    assert abs(phi - phi_gain) < 2e-7
    beta_error = np.linalg.norm(beta_gain - beta @ gain3) / np.linalg.norm(beta @ gain3)
    cov_error = np.linalg.norm(covariance_gain - gain3.T @ covariance @ gain3) / np.linalg.norm(gain3.T @ covariance @ gain3)
    assert beta_error < 1e-6 and cov_error < 1e-6
    objective_shift = history_gain[-1]["objective"] - history[-1]["objective"]
    expected_shift = 2 * sum(len(r) for r in records) * np.linalg.slogdet(gain3)[1]
    assert abs(objective_shift - expected_shift) < 1e-6

    # R and S have one reciprocal scale freedom, fixed by unit innovations.
    R = linalg.toeplitz(.4**np.arange(5)) / (1 - .4**2)
    S = covariance
    gauge_error = float(np.max(np.abs(np.kron(S, R) - np.kron(S / 7., R * 7.))))
    assert gauge_error < 1e-12

    # Two independent formulations of the ridge + diffuse-nuisance marginal.
    # The dense covariance method does not use the whiten/project formula.
    marginal_errors = []
    n_small, width, nuisance_rank = 23, 4, 2
    nuisance = np.column_stack([np.ones(n_small), np.linspace(-1, 1, n_small)])
    design = rng.normal(size=(n_small, width))
    observed = rng.normal(size=(n_small, 3))
    ridge = .7
    for phi_small in [0., .5, .9]:
        R_small = linalg.toeplitz(phi_small**np.arange(n_small)) / (1 - phi_small**2)
        root = np.linalg.cholesky(R_small)
        white_n = linalg.solve_triangular(root, nuisance, lower=True)
        complement = np.linalg.qr(white_n, mode="complete")[0][:, nuisance_rank:]
        d_bar = complement.T @ linalg.solve_triangular(root, design, lower=True)
        y_bar = complement.T @ linalg.solve_triangular(root, observed, lower=True)
        gram = ridge * np.eye(width) + d_bar.T @ d_bar
        scatter = y_bar.T @ y_bar - y_bar.T @ d_bar @ np.linalg.solve(gram, d_bar.T @ y_bar)
        logdet = (np.linalg.slogdet(R_small)[1]
                  + np.linalg.slogdet(white_n.T @ white_n)[1]
                  + np.linalg.slogdet(gram)[1] - width * np.log(ridge))
        K = R_small + design @ design.T / ridge
        inverse_K = np.linalg.inv(K)
        inverse_Kn = inverse_K @ nuisance
        inverse_n_gram = np.linalg.inv(nuisance.T @ inverse_Kn)
        projection = inverse_K - inverse_Kn @ inverse_n_gram @ inverse_Kn.T
        dense_scatter = observed.T @ projection @ observed
        dense_logdet = np.linalg.slogdet(K)[1] + np.linalg.slogdet(nuisance.T @ inverse_Kn)[1]
        error = {"phi": phi_small, "scatter_max_error": float(np.max(np.abs(scatter - dense_scatter))),
                 "logdet_error": float(abs(logdet - dense_logdet))}
        assert error["scatter_max_error"] < 1e-10 and error["logdet_error"] < 1e-10
        marginal_errors.append(error)
    return {
        "evidence_scope": "software_correctness", "seed": 1042026,
        "nonseparable_gain_counterexample": {
            "true_channel_phis": phis.tolist(), "pooled_burg_before_gain": unweighted,
            "pooled_burg_after_first_channel_gain100": unweighted_gain,
            "covariance_weighted_burg": weighted, "weighted_after_gain": weighted_gain,
            "global_unit_scale_error": abs(burg1(x * 1e6) - unweighted),
            "residual_channel_acf1_unweighted_gain": acf1(residual_unweighted_gain).tolist(),
            "residual_channel_acf1_weighted": acf1(residual_weighted_gain).tolist(),
            "interpretation": "gain equivariance fixed; shared temporal shape still misspecified",
        },
        "shrinkage_gain_equivariance": {
            "raw_ledoit_wolf_relative_error_after_inverse_gain": relative_lw_error,
            "correlation_shrinkage_relative_error": relative_correlation_error,
            "scope": "nonzero diagonal gain; not arbitrary spatial mixing",
        },
        "exact_stationary_alternating_ML": {
            "true_phi": .86, "fitted_phi": phi, "after_gain_phi": phi_gain,
            "history": history, "history_after_gain": history_gain,
            "beta_gain_relative_error": beta_error, "covariance_gain_relative_error": cov_error,
            "objective_jacobian_shift_error": abs(objective_shift - expected_shift),
            "scope": "three independent AR1 records, fixed mean+nuisance design, no ridge or shrinkage",
        },
        "reciprocal_covariance_scale_error": gauge_error,
        "ridge_diffuse_nuisance_marginal_dense_parity": marginal_errors,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    result = run()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
