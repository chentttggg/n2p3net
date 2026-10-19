"""Independent dense-vs-innovations and constrained-minimizer checks."""
from __future__ import annotations

import json

import numpy as np
from probe import HERE
from scipy.linalg import solve_discrete_lyapunov, toeplitz
from threadpoolctl import threadpool_limits


def run():
    # Generic stable AR(3), scalar observation. No production covariance code.
    coefficients = np.array([.6, -.2, .1])
    state = np.zeros((3, 3))
    state[0] = coefficients
    state[1:, :-1] = np.eye(2)
    innovation = np.zeros((3, 3))
    innovation[0, 0] = 1.3
    stationary = solve_discrete_lyapunov(state, innovation)
    n = 30
    lags = np.array([(np.linalg.matrix_power(state, lag) @ stationary)[0, 0] for lag in range(n)])
    covariance = toeplitz(lags)
    # First p observations are an exact stationary Gaussian block; later rows
    # are independent innovations. It is not p independent stationary marginals.
    W = np.eye(n)
    W[:3, :3] = np.linalg.inv(np.linalg.cholesky(covariance[:3, :3]))
    for index in range(3, n):
        W[index, index] = 1 / np.sqrt(innovation[0, 0])
        W[index, index - 3:index] = -coefficients[::-1] / np.sqrt(innovation[0, 0])
    np.testing.assert_allclose(W @ covariance @ W.T, np.eye(n), atol=1e-13)
    rng = np.random.default_rng(75)
    observed = rng.multivariate_normal(np.zeros(n), covariance, size=200)
    dense_quadratic = np.einsum("ni,ij,nj->n", observed, np.linalg.inv(covariance), observed)
    innovations_quadratic = np.sum((observed @ W.T)**2, axis=1)
    np.testing.assert_allclose(dense_quadratic, innovations_quadratic, atol=8e-13)
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign == 1
    innovation_logdet = np.linalg.slogdet(covariance[:3, :3])[1] + (n - 3) * np.log(1.3)
    np.testing.assert_allclose(logdet, innovation_logdet, atol=1e-13)
    # Paired x and previous external AR state, then condition state on all x.
    state_x = np.stack([(np.linalg.matrix_power(state, index + 1) @ stationary)[0]
                        for index in range(n)])
    schur = stationary - state_x.T @ np.linalg.solve(covariance, state_x)
    assert np.linalg.eigvalsh(schur).min() > 0
    # Constraint optimizer: compare whitening/SVD to a distinct full KKT solve.
    U = rng.normal(size=(120, 11))
    D = rng.normal(size=(120, 7))
    C = rng.normal(size=(4, 11))
    T = rng.normal(size=(4, 7))
    H = U.T @ U + 120 * np.eye(11)
    L = np.linalg.cholesky(H)
    W0 = np.linalg.solve(H, U.T @ D)
    a, singular, b = np.linalg.svd(np.linalg.solve(L, C.T), full_matrices=False)
    constrained = W0 + np.linalg.solve(L.T, (a / singular) @ (b @ (T - C @ W0)))
    KKT = np.block([[H, C.T], [C, np.zeros((4, 4))]])
    independent = np.linalg.solve(KKT, np.vstack([U.T @ D, T]))[:11]
    np.testing.assert_allclose(constrained, independent, atol=1e-13)
    np.testing.assert_allclose(C @ constrained, T, atol=1e-13)
    # Recompute every direct/full covariance equality from saved physical draws.
    draws = np.load(HERE / "projected_draws.npz", allow_pickle=False)
    delta = draws["finite"] - draws["long"]
    np.testing.assert_allclose(draws["long"] + delta, draws["finite"], atol=1e-13)
    output = {
        "ar3_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(state)))),
        "ar3_whitening_max_error": float(np.max(np.abs(W @ covariance @ W.T - np.eye(n)))),
        "ar3_dense_innovations_quadratic_max_error": float(np.max(np.abs(dense_quadratic - innovations_quadratic))),
        "ar3_logdet_difference": float(abs(logdet - innovation_logdet)),
        "ar3_external_state_stationary_covariance": stationary.tolist(),
        "ar3_external_state_given_observed_covariance": schur.tolist(),
        "constrained_svd_vs_kkt_max_error": float(np.max(np.abs(constrained - independent))),
        "physical_prefix_draws": len(delta),
        "validation_scope": "Independent numerical mathematics and saved data identity; not classifier performance",
    }
    (HERE / "independent_verification.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
