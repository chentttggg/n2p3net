"""Exact stored-float checks for local conditional and projection bounds."""

from fractions import Fraction

import numpy as np
import pytest

from research import channel_noise_mechanism as owner


def exact(value):
    return Fraction.from_float(float(value))


def exact_product(left, right):
    return [[sum(exact(a) * exact(b) for a, b in zip(row, column, strict=True))
             for column in right.T] for row in left]


def test_positive_dot_and_scaled_norm_bounds_cover_exact_stored_operands():
    left = np.array([[.1, .3, .7], [1.0000000000000002, 3., .01]])
    right = np.array([[.3, 2.], [.7, .01], [1.1, 7.]])
    bound = owner._projection_upper_matmul(left, right)
    expected = exact_product(left, right)
    for i, j in np.ndindex(bound.shape):
        assert exact(bound[i, j]) >= expected[i][j]
    columns = np.array([[1e200, 1e-200, 0.], [3e200, 2e-200, 0.], [2e200, 3e-200, -0.]])
    norms = owner._projection_norm_upper(columns)
    for j in range(columns.shape[1]):
        assert exact(norms[j]) ** 2 >= sum(exact(value) ** 2 for value in columns[:, j])
    assert norms[-1] == 0.


@pytest.mark.parametrize("mode", ["orthogonal", "nonorthogonal", "flat", "extreme_scales"])
def test_projection_column_errors_cover_exact_stored_projection(mode):
    rng = np.random.default_rng(20260918)
    q, _ = np.linalg.qr(rng.normal(size=(11, 3)), mode="reduced")
    values = rng.normal(size=(11, 3))
    if mode == "nonorthogonal":
        q[:, 1] = .75 * q[:, 0] + .25 * q[:, 1]
    elif mode == "flat":
        values = q @ np.array([[1e14, 0., 3.], [0., 3e13, 2.], [2e14, 1e13, 1.]])
    elif mode == "extreme_scales":
        values *= np.array([1e200, 1e-200, 1.])
    dot = np.array(exact_product(q.T, values), dtype=object)
    image = [[sum(exact(a) * b for a, b in zip(row, column, strict=True))
              for column in dot.T] for row in q]
    actual = values - q @ (q.T @ values)
    bounds = owner._projection_arithmetic_columns(values, q)
    for j in range(values.shape[1]):
        squared_error = sum((exact(actual[i, j]) - exact(values[i, j]) + image[i][j]) ** 2
                            for i in range(len(values)))
        assert exact(bounds[j]) ** 2 >= squared_error


@pytest.mark.parametrize("kind", ["normal_boundary", "lost_dot", "normalization", "overflow"])
def test_projection_arithmetic_cannot_credit_unresolved_operands(kind):
    with pytest.raises(owner.DiffuseProjectionResolutionError):
        if kind == "normal_boundary":
            owner._projection_require_normal_products(np.array([[np.nextafter(1., 0.)]]),
                                                       np.array([[np.finfo(float).tiny]]))
        elif kind == "lost_dot":
            owner._projection_upper_matmul(np.ones((1, 2)) * 1e-200, np.ones((2, 1)) * 1e-200)
        elif kind == "normalization":
            owner._projection_scaled_columns(np.array([[1e308], [1e-308]]))
        else:
            owner._projection_upper_matmul(np.ones((1, 2)) * 1e308, np.ones((2, 1)) * 1e308)


@pytest.mark.parametrize("fraction,scale", [(.01, 1.), (.01, 1e100), (.01, 1e-100), (.99, 1.)])
def test_minimizer_bound_keeps_residual_coupling_and_valid_global_cap(fraction, scale):
    # A=diag(1,sigma), with an extra orthogonal row; b lies in that row.
    # E couples its residual to the weak column. Omitting E.T r gives zero.
    sigma, response_norm = 1e-6 * scale, 1e8 * scale
    perturbation = fraction * sigma
    bound, terms = owner._conditional_minimizer_norm(np.zeros(2),
        np.nextafter(2 * scale, np.inf), response_norm, sigma, perturbation, 0., 0.)
    target = exact(perturbation) * exact(response_norm) / (exact(sigma) ** 2 + exact(perturbation) ** 2)
    assert exact(bound) >= target > 0
    assert exact(terms["local"]) >= target
    assert exact(terms["global"]) >= target
    if fraction == .99:
        assert terms["global"] < terms["local"]


@pytest.mark.parametrize("response_norm", [1e-200, 1e-160, np.finfo(float).tiny])
def test_positive_minimizer_quotient_underflow_is_typed(response_norm):
    with pytest.raises(owner.DiffuseProjectionResolutionError, match="underflow|nonnormal"):
        owner._conditional_minimizer_norm(np.zeros(1), 1e150, response_norm, 1e150,
                                         0., response_norm, response_norm)


def test_true_structural_zero_keeps_zero_bound_and_no_manufactured_terms():
    bound, terms = owner._conditional_minimizer_norm(np.zeros(1), 1., 0., 1., 0., 0., 0.)
    assert bound == 0.
    assert all(value == 0. for value in terms.values())
    np.testing.assert_array_equal(owner._projection_arithmetic_columns(np.zeros((4, 3)),
                                                                   np.eye(4)[:, :2]), np.zeros(3))
