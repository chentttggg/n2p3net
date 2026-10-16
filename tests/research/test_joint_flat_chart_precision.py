"""Independent exact stored-coordinate and disjoint-row counterexamples."""
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from research.channel_noise_mechanism import DiffuseProjectionResolutionError
from research.joint_flat_chart_precision import (
    _stored_product_residual,
    saved_source_chart_certificate,
)
from research.joint_flat_factor_history import FactoredQRTree, _draw_resolved_gaussian


def _multiply(left, right):
    return [[sum((a*b for a, b in zip(row, column, strict=True)), Fraction())
             for column in zip(*right, strict=True)] for row in left]


def _transpose(matrix):
    return list(map(list, zip(*matrix, strict=True)))


def _inverse(matrix):
    count = len(matrix)
    joined = [list(row)+[Fraction(i == j) for j in range(count)] for i, row in enumerate(matrix)]
    for column in range(count):
        pivot = next(i for i in range(column, count) if joined[i][column])
        joined[column], joined[pivot] = joined[pivot], joined[column]
        scale = joined[column][column]
        joined[column] = [value/scale for value in joined[column]]
        for row in range(count):
            if row != column:
                multiplier = joined[row][column]
                joined[row] = [a-multiplier*b for a, b in zip(joined[row], joined[column], strict=True)]
    return [row[count:] for row in joined]


def test_small_saved_chart_certificate_covers_exact_fraction_covariance_and_rejects_another_prior():
    temporal = np.array([[.7, .2], [0., 1.1]])
    spatial = np.array([[1.3e-6, 0.], [.3e-6, .8e-6]])
    factor = np.kron(temporal, spatial)
    _, upper = np.linalg.qr(factor.T)
    old = upper.T*np.where(np.diag(upper) < 0, -1., 1.)[None, :]
    certificate = saved_source_chart_certificate(temporal, spatial, old)
    # Exact real Kronecker products of the original binary64 T and S, not
    # the already rounded NumPy Kronecker matrix used by the source QR.
    exact = [[Fraction.from_float(temporal[i, j])*Fraction.from_float(spatial[c, d])
        for j in range(2) for d in range(2)] for i in range(2) for c in range(2)]
    original = [[Fraction.from_float(value) for value in row] for row in old]
    relative = _multiply(_inverse(exact), original)
    covariance = _multiply(relative, _transpose(relative))
    error_squared = sum(((value-Fraction(i == j))**2 for i, row in enumerate(covariance)
                         for j, value in enumerate(row)), Fraction())
    assert Fraction.from_float(certificate['covariance_relation_error'])**2 >= error_squared
    assert certificate['covariance_relation_error'] < 1e-10
    with pytest.raises(DiffuseProjectionResolutionError, match='original saved source prior'):
        saved_source_chart_certificate(temporal, spatial, 2*old)


def test_compensated_witness_retains_cancelled_unit_and_rejects_nonfinite_operands():
    residual, error = _stored_product_residual(np.array([[1e16, 1., -1e16]]), np.ones((3, 1)), np.zeros((1, 1)))
    assert residual[0, 0] == -1. and error[0, 0] < 1e-14
    with pytest.raises(ValueError, match='finite invertible'):
        saved_source_chart_certificate(np.array([[np.nan]]), np.eye(1), np.eye(1))


def test_disjoint_row_error_geometry_avoids_false_mean_rejection():
    # The same bounded errors may be perfectly correlated across children;
    # disjoint row embeddings still combine their norms in quadrature.
    tree = FactoredQRTree.__new__(FactoredQRTree)
    tree.nodes = [None, None, np.array([[1., 0.]]), np.array([[1., 0.]])]
    one_error = .6*np.sqrt(np.finfo(float).eps)
    tree.errors = [None, None, np.array([0., one_error]), np.array([0., one_error])]
    tree._merge(1)
    rng = SimpleNamespace(normal=lambda size: np.zeros(size))
    result = _draw_resolved_gaussian(tree.root[:, :1], tree.root[:, 1:], tree.errors[1], rng)
    np.testing.assert_array_equal(result, [[0.]])


def test_same_row_image_errors_still_add_in_the_original_leaf():
    image = SimpleNamespace(onsets=(1,), pulse=np.zeros((1, 1)), stop=2,
        block=lambda _lo, _hi: (np.zeros((1, 2)), np.array([[4e-8, 0.]])))
    axis = SimpleNamespace(prefix_images=(image,), prior_mass=np.ones(1))
    record = SimpleNamespace(tail_rows=np.zeros((1, 3)), tail_errors=np.array([[0., 3e-8, 0.]]),
        startup_native_rows=1, native_rows=2, drift_width=1, response_width=1)
    tree = FactoredQRTree(record, (axis,), leaf_rows=1, cache_bytes=1024)
    tree.set_target(0, 0)
    assert tree.errors[1][1] >= 7e-8 > 6e-8


def test_prior_uncertainty_stays_on_prior_rows_and_keeps_its_mean_action():
    design = np.vstack((np.eye(2), np.zeros((1, 2))))
    response = np.array([[0.], [0.], [1e6]])
    rng = SimpleNamespace(normal=lambda size: np.zeros(size))
    result = _draw_resolved_gaussian(design, response, np.zeros(3), rng, proper_prior_covariance_error=1e-10)
    np.testing.assert_array_equal(result, [[0.], [0.]])
    # A genuine off-diagonal prior precision perturbation can move a large
    # mean into a weak direction. Its action cannot be discarded as unrelated.
    with pytest.raises(DiffuseProjectionResolutionError, match='conditional mean'):
        _draw_resolved_gaussian(np.eye(2), np.array([[0.], [1e6]]), np.zeros(3), rng,
                                proper_prior_covariance_error=1e-10)


def test_likelihood_error_contraction_keeps_the_actual_mean_direction():
    rng = SimpleNamespace(normal=lambda size: np.zeros(size))
    response = np.array([[1e5], [0.]])
    # The uncertain second design column has zero coefficient; with no
    # residual, its uncertainty changes covariance but cannot move this mean.
    value = _draw_resolved_gaussian(np.eye(2), response, np.array([0., 1e-10, 0.]), rng)
    np.testing.assert_array_equal(value, response)
    # Moving the same error into the active column can move the other
    # coefficient by 1e-5, which must fail the original posterior metric gate.
    with pytest.raises(DiffuseProjectionResolutionError, match='conditional mean'):
        _draw_resolved_gaussian(np.eye(2), response, np.array([1e-10, 0., 0.]), rng)


def test_balanced_gram_covers_exact_stored_products_and_preserves_original_qr():
    from research.joint_flat_chart_precision import balanced_gram_certificate
    from research.joint_flat_factor_history import _augmented_qr
    from research.joint_response_scale import _augmented_qr as original_qr

    values = np.random.default_rng(918).normal(size=(37, 5))
    gram, error = balanced_gram_certificate(values)
    for i in range(5):
        for j in range(5):
            exact = sum((Fraction.from_float(a)*Fraction.from_float(b)
                for a, b in zip(values[:, i], values[:, j], strict=True)), Fraction())
            assert abs(Fraction.from_float(gram[i, j])-exact) <= Fraction.from_float(error[i, j])
    old, _ = original_qr(values, 11, with_error=True)
    current, _ = _augmented_qr(values, 11, with_error=True)
    np.testing.assert_array_equal(current, old)
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(DiffuseProjectionResolutionError, match='not representable'):
            balanced_gram_certificate(np.array([[1e308]]))
