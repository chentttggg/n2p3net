"""Levinson startup innovations versus independent high-precision Yule-Walker."""
from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits


def reflection_stepdown(coefficients):
    current = [-value for value in coefficients]
    output = [mp.mpf(0)]*len(current)
    for size in range(len(current), 0, -1):
        k = current[-1]
        output[size-1] = k
        current = [(current[i]-k*current[size-2-i])/(1-k*k) for i in range(size-1)]
    return output


def coefficients_from_parcor(values):
    current = []
    for k in values:
        current = [current[i]+k*current[-1-i] for i in range(len(current))]+[k]
    return [-value for value in current]


def mp_reference(reflections):
    coefficients = coefficients_from_parcor([mp.mpf(float(k)) for k in reflections])
    order = len(coefficients)
    equations = mp.eye(order+1)
    for j, value in enumerate(coefficients, 1):
        equations[0, j] -= value
    for k in range(1, order+1):
        for j, value in enumerate(coefficients, 1):
            equations[k, abs(k-j)] -= value
    drive = mp.zeros(order+1, 1)
    drive[0] = 1
    gamma = mp.lu_solve(equations, drive)
    covariance = mp.matrix([[gamma[abs(i-j)] for j in range(order)] for i in range(order)])
    return covariance, mp.cholesky(covariance)**-1


def factor(reflections):
    order = len(reflections)
    prefix = np.zeros((order, order))
    log_variance = -np.cumsum(np.log1p(-reflections[::-1]**2))[::-1]
    polynomial = np.empty(0)
    for t in range(order):
        scale = np.exp(-.5*log_variance[t])
        prefix[t, t] = scale
        prefix[t, :t] = polynomial[::-1]*scale
        polynomial = np.r_[polynomial+reflections[t]*polynomial[::-1], reflections[t]]
    return prefix, -polynomial, log_variance


def main():
    mp.mp.dps = 100
    rng = np.random.default_rng(109493)
    results = []
    for order, pole in [(1, .7), (2, .8), (4, .99), (8, .99)]:
        r = mp.mpf(str(pole))
        analytic_coeff = [(-1)**(j+1)*mp.binomial(order, j)*r**j for j in range(1, order+1)]
        parcor = np.array([float(value) for value in reflection_stepdown(analytic_coeff)])
        covariance, inverse_root = mp_reference(parcor)
        W, coeff, log_variance = factor(parcor)
        reference = np.array(inverse_root.tolist(), dtype=float)
        relative_factor_error = np.linalg.norm(W-reference)/np.linalg.norm(reference)
        assert relative_factor_error < 2e-14
        eps = mp.matrix(rng.normal(size=order).tolist())
        exact_sample = mp.cholesky(covariance)*eps
        float_sample = np.array(list(exact_sample), dtype=float)
        best_given_float = np.array(list(inverse_root*mp.matrix(float_sample.tolist())), dtype=float)
        direct = W@float_sample
        quantization_loss = np.linalg.norm(best_given_float-np.array(list(eps), dtype=float))
        double_algorithm_loss = np.linalg.norm(direct-best_given_float)
        rounded_cov = np.array(covariance.tolist(), dtype=float)
        try:
            rounded_inverse_root = np.linalg.inv(np.linalg.cholesky(rounded_cov))
            cholesky_relative = float(np.linalg.norm(rounded_inverse_root-reference)/np.linalg.norm(reference))
        except np.linalg.LinAlgError:
            cholesky_relative = None
        results.append({"order": order, "repeated_pole": pole, "parcor": parcor.tolist(),
            "float_polynomial_max_pole_radius": float(np.max(np.abs(np.roots(np.r_[1., -coeff])))),
            "startup_factor_condition": float(np.linalg.cond(W)),
            "factor_relative_error_vs_100digit_covariance": relative_factor_error,
            "rounded_covariance_Cholesky_factor_relative_error": cholesky_relative,
            "finite_observation_quantization_innovation_error": quantization_loss,
            "double_factor_application_error_given_float_observation": double_algorithm_loss,
            "marginal_variance_per_unit_innovation": float(np.exp(log_variance[0]))})
    result = {"schema": "n2p3_stationary_innovations_derivation/1", "precision_digits": 100,
              "seed": 109493, "records": results,
              "interpretation": "factorization repairs covariance construction, not information lost to finite observation precision"}
    Path(__file__).with_name("stationary_innovations_probe.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
