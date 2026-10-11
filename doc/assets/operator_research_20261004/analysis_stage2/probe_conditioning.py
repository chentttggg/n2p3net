"""Independent conditioning and covariance-scale gauge counterexamples."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular, toeplitz
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits


def density(x, mean, covariance):
    residual = x-mean
    return -.5*(residual@np.linalg.solve(covariance, residual)+np.linalg.slogdet(covariance)[1])


def condition(x, mean, covariance, p):
    regression = np.linalg.solve(covariance[:p, :p], covariance[:p, p:]).T
    mu = mean[p:]+regression@(x[:p]-mean[:p])
    cov = covariance[p:, p:]-regression@covariance[:p, p:]
    return density(x[p:], mu, cov)


def conditioning():
    rng = np.random.default_rng(108541)
    n, p = 40, 2
    coefficients = np.array([1.5, -.7])
    impulse = np.zeros(20000)
    impulse[0] = 1.
    response = lfilter([1.], np.r_[1., -coefficients], impulse)
    R = toeplitz([response[:-lag or None]@response[lag:] for lag in range(n)])
    x = rng.normal(size=n)
    means = rng.normal(size=(2, n))*.4
    means[:, :p] = 0.
    # Stationary start factors cancel because the entire prefix is candidate-free.
    full = [density(x, m, R) for m in means]
    conditional = [condition(x, m, R, p) for m in means]
    W = np.eye(n)
    for lag, coefficient in enumerate(coefficients, 1):
        W[p:, p-lag:n-lag] -= coefficient*np.eye(n-p)
    W = W[p:]
    innovation = [density(W@x, W@m, np.eye(n-p)) for m in means]
    error = abs((full[1]-full[0])-(conditional[1]-conditional[0]))
    assert error < 1e-11
    np.testing.assert_allclose(np.diff(conditional), np.diff(innovation), atol=1e-11)
    bad_means = means.copy()
    bad_means[1, :p] = [2., -1.]
    bad_full = [density(x, m, R) for m in bad_means]
    bad_conditional = [condition(x, m, R, p) for m in bad_means]
    informative_gap = float(np.diff(bad_full)[0]-np.diff(bad_conditional)[0])
    assert abs(informative_gap) > .1
    # A shared unknown offset is not known merely because prefix D is zero.
    # Full integration conditions its prior on the prefix. Naively reusing the
    # unchanged offset prior after AR conditioning is a different density.
    Z = np.ones(n)
    nuisance_cov = R+3*np.outer(Z, Z)
    full_nuisance = [density(x, m, nuisance_cov) for m in means]
    correct_conditional = [condition(x, m, nuisance_cov, p) for m in means]
    naive_cov = np.eye(n-p)+3*np.outer(W@Z, W@Z)
    naive = [density(W@x, W@m, naive_cov) for m in means]
    np.testing.assert_allclose(np.diff(full_nuisance), np.diff(correct_conditional), atol=1e-11)
    nuisance_gap = float(np.diff(full_nuisance)[0]-np.diff(naive)[0])
    assert abs(nuisance_gap) > 1e-4
    return {"known_nuisance_candidate_free_prefix_BF_error": error,
            "informative_prefix_dropping_changes_log_BF": informative_gap,
            "naive_unknown_nuisance_conditioning_changes_log_BF": nuisance_gap,
            "scope": "conditional_likelihood_is_equal_only_under_explicit_conditioning_assumptions"}


def prior_scale_gauge():
    rng = np.random.default_rng(548734)
    train, test, rank = 60, 40, 4
    Ktrain, Ktest = toeplitz(.7**np.arange(train)), toeplitz(.7**np.arange(test))
    sigma, ridge, factor = .4, 1., 200.
    D = rng.normal(size=(train, rank))
    truth = rng.normal(size=rank)
    y = D@truth+np.sqrt(sigma)*np.linalg.cholesky(Ktrain)@rng.normal(size=train)
    candidates = rng.normal(size=(3, test, rank))
    x = candidates[1]@truth+np.sqrt(sigma)*np.linalg.cholesky(Ktest)@rng.normal(size=test)

    def fit_score(scale, penalty):
        WD = solve_triangular(np.linalg.cholesky(scale*Ktrain), D, lower=True)
        Wy = solve_triangular(np.linalg.cholesky(scale*Ktrain), y, lower=True)
        row_cov = np.linalg.inv(WD.T@WD+penalty*np.eye(rank))
        mean = row_cov@WD.T@Wy
        return np.array([density(x, d@mean, (scale*Ktest+d@row_cov@d.T)*(sigma/scale)) for d in candidates])

    base = fit_score(1., ridge)
    incorrectly_fixed = fit_score(factor, ridge)
    correctly_scaled = fit_score(factor, ridge/factor)
    exact_error = float(np.max(np.abs((base-base[0])-(correctly_scaled-correctly_scaled[0]))))
    wrong_error = float(np.max(np.abs((base-base[0])-(incorrectly_fixed-incorrectly_fixed[0]))))
    assert exact_error < 1e-10 and wrong_error > 1.
    return {"covariance_factor": factor, "unchanged_ridge_max_log_BF_change": wrong_error,
            "ridge_divided_by_factor_BF_error": exact_error,
            "same_likelihood_noise_different_prior_if_ridge_not_scaled": True,
            "canonical_marginal_scale_prior": "unit-innovation representation uses effective ridge = declared ridge / stationary marginal variance per unit innovation"}


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        result = {"schema": "n2p3_conditioning_prior_gauge_counterexamples/1", "conditioning": conditioning(),
                  "prior_scale_gauge": prior_scale_gauge(), "no_real_data": True}
    Path(__file__).with_name("conditioning_probe.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
    print(json.dumps(result, indent=2))
