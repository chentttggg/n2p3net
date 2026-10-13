"""Analytic measures and finite-filter direction, with no public data or fits."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal

from data.finite_filter_operator import FiniteFilterOperator
from transfer.exact_covariance import ExactPropagatedCovariance, ar_toeplitz_covariance
from transfer.temporal_noise import TemporalNoise


def _noise(coefficients=(), variance=2.0):
    return TemporalNoise.from_parameters(
        coefficients, innovation_variance=variance, reference_id="exact-support-analytic"
    )


def _operator(gain=1.0):
    return FiniteFilterOperator(((gain, 0, 0, 1, 0, 0),), 0)


def _difference():
    return FiniteFilterOperator(((1, -1, 0, 1, 0, 0),), 0)


@pytest.mark.parametrize("rho", [0.0, 0.7, -0.4])
def test_identity_ar_density_has_exact_unridged_covariance(rho):
    noise = _noise([] if rho == 0 else [rho])
    result = ExactPropagatedCovariance.compute(_operator(), noise, 4)
    expected = toeplitz(2 * rho ** np.arange(4) / (1 - rho**2))
    np.testing.assert_allclose(result.K_ar, expected, atol=1e-14)
    np.testing.assert_allclose(result.K_propagated, expected, atol=1e-14)
    np.testing.assert_allclose(result.cholesky_propagated @ result.cholesky_propagated.T,
                               expected, atol=1e-14)
    r = np.array([[1.0, 2], [0.2, -0.3], [0.1, 0.5], [-0.4, 0.6]])
    expected_density = sum(multivariate_normal.logpdf(column, cov=expected) for column in r.T)
    assert result.log_density(r) == pytest.approx(expected_density, abs=2e-14)
    assert result.log_likelihood_correction(r) == pytest.approx(0, abs=1e-14)
    assert result.provenance["regularization_ridge"] == 0
    assert result.provenance["measure"] == "ambient_spd"


def test_spd_correction_and_covariance_gain_not_square_root_chart_error():
    result = ExactPropagatedCovariance.compute(_operator(0.5), _noise(), 4)
    # Bidirectional gain is .25; variance gain is .0625.
    r = np.array([[1.0], [-0.4], [0.6], [0.3]])
    expected = multivariate_normal.logpdf(r[:, 0], cov=0.125 * np.eye(4))
    approximate = multivariate_normal.logpdf(r[:, 0], cov=2 * np.eye(4))
    assert result.log_density(r) == pytest.approx(expected, abs=2e-14)
    assert result.log_likelihood_correction(r) == pytest.approx(expected - approximate, abs=2e-14)
    gain = result.whitening_gain()
    np.testing.assert_allclose(gain["covariance_after_ar_whitening"], 0.0625 * np.eye(4))
    assert gain["frobenius_deviation_from_identity"] == pytest.approx(1.875)
    assert gain["singular_values_min"] == pytest.approx(0.25)


def test_weak_positive_direction_is_not_replaced_by_ridge_or_rank_floor():
    # This finite SOS operator is invertible, even though its covariance has
    # eigenvalue ratio ~2.5e-17. The former 1e-10 floor mislabeled it singular.
    operator = FiniteFilterOperator(((1, -0.9999, 0, 1, 0, 0),), 0)
    F = operator.apply(np.eye(2))
    result = ExactPropagatedCovariance.compute(operator, _noise(variance=1), 2)
    assert result.numerical_rank == 2
    assert 0 < result.eigenvalues[0] < 1e-10 * result.eigenvalues[-1]
    r = F[:, 1:2]
    # Independent 80-digit determinant of the actual executed F. Its second
    # column corresponds exactly to unit pre-filter residual e_2, energy=1.
    with localcontext() as context:
        context.prec = 80
        determinant = (Decimal.from_float(F[0, 0]) * Decimal.from_float(F[1, 1])
                       - Decimal.from_float(F[0, 1]) * Decimal.from_float(F[1, 0]))
        log_volume = float(abs(determinant).ln())
    expected = -0.5 - log_volume - np.log(2 * np.pi)
    assert result.log_density(r) == pytest.approx(expected, abs=3e-8)
    assert np.sum(result.whiten(r)**2) == pytest.approx(1, abs=3e-8)


@pytest.mark.parametrize("variance", [1e-200, 1e200])
def test_covariance_scale_does_not_underflow_or_overflow_comparison(variance):
    result = ExactPropagatedCovariance.compute(_operator(0.5), _noise(variance=variance), 3)
    assert result.relative_error_trace == pytest.approx(0.9375)
    assert np.isfinite(result.log_density(np.zeros((3, 1))))


def test_non_symmetric_filter_direction_and_declared_support_density():
    F = np.array([[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, 0, 0]])
    operator = _difference()
    np.testing.assert_array_equal(operator.apply(np.eye(4)), F)
    K = ar_toeplitz_covariance(_noise([0.3]), 4)
    Q = np.eye(4)[:, :3]
    result = ExactPropagatedCovariance.compute(operator, _noise([0.3]), 4, support_basis=Q)
    expected = F @ K @ F.T
    np.testing.assert_allclose(result.K_propagated, expected, atol=1e-14)
    assert np.linalg.norm(expected - F.T @ K @ F) > 1
    assert result.cholesky_propagated is None
    assert result.support_dimension == 3
    r = F @ np.array([[0.2, 0.5], [0.3, 0.7], [0.4, -0.2], [0.1, -0.5]])
    expected_density = sum(multivariate_normal.logpdf(v[:3], cov=expected[:3, :3]) for v in r.T)
    assert result.log_density_on_support(r) == pytest.approx(expected_density, abs=1e-13)
    np.testing.assert_allclose(Q @ result.cholesky_support @ result.whiten(r), r, atol=1e-14)
    for method in (result.log_density, result.log_likelihood_correction):
        with pytest.raises(ValueError, match="SPD"):
            method(r)
    gain = result.whitening_gain()
    assert gain["singular_values_min"] == 0
    assert np.isinf(gain["condition_number"])


def test_support_chart_rotation_keeps_volume_and_quadratic():
    Q = np.eye(4)[:, :3]
    rotation = np.array([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]])
    original = ExactPropagatedCovariance.compute(_difference(), _noise(), 4, support_basis=Q)
    rotated = ExactPropagatedCovariance.compute(_difference(), _noise(), 4, support_basis=Q @ rotation)
    r = np.array([[0.3, 0.1], [-0.4, 0.9], [0.2, 0.5], [0, 0]])
    assert original.log_density_on_support(r) == pytest.approx(rotated.log_density_on_support(r), abs=1e-13)
    assert np.sum(original.whiten(r)**2) == pytest.approx(np.sum(rotated.whiten(r)**2), abs=1e-13)


@pytest.mark.parametrize("scale", [1, 1e-200, 1e200])
def test_off_support_rejection_uses_each_channels_scale(scale):
    result = ExactPropagatedCovariance.compute(_difference(), _noise(), 4,
                                               support_basis=np.eye(4)[:, :3])
    r = np.zeros((4, 2))
    r[0, 0] = 1e100
    r[3, 1] = scale
    with pytest.raises(ValueError, match="outside"):
        result.whiten(r)


def test_point_mass_is_valid_zero_rank_measure_and_rejects_every_nonzero():
    result = ExactPropagatedCovariance.compute(_operator(0), _noise(), 4,
                                               support_basis=np.empty((4, 0)))
    np.testing.assert_array_equal(result.K_propagated, np.zeros((4, 4)))
    assert result.numerical_rank == 0
    assert result.whiten(np.zeros((4, 2))).shape == (0, 2)
    assert result.log_density_on_support(np.zeros((4, 2))) == 0
    with pytest.raises(ValueError, match="outside"):
        result.whiten(np.full((4, 1), 1e-200))
    with pytest.raises(ValueError, match="SPD"):
        result.log_density(np.zeros((4, 1)))


def test_numerical_rank_does_not_automatically_define_a_density():
    result = ExactPropagatedCovariance.compute(_difference(), _noise(), 4)
    assert result.numerical_rank == 3
    assert result.support_basis is None
    with pytest.raises(ValueError, match="declare support_basis"):
        result.whiten(np.zeros((4, 1)))


def test_full_rank_operator_with_unresolved_covariance_retains_raw_diagnostic():
    # A genuine stable, invertible finite lowpass has three resolved operator
    # singular values; forming its covariance loses the tiny positive modes.
    operator = FiniteFilterOperator(((1e-5, 0, 0, 1, -0.99999, 0),), 0)
    F = operator.apply(np.eye(3))
    singular_values = np.linalg.svd(F, compute_uv=False)
    assert singular_values[-1] > 3 * np.finfo(float).eps * singular_values[0]
    raw = F @ F.T
    raw = (raw + raw.T) / 2
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(raw)
    result = ExactPropagatedCovariance.compute(operator, _noise(variance=1), 3)
    np.testing.assert_array_equal(result.K_propagated, raw)
    assert result.numerical_rank == 3
    assert result.cholesky_propagated is None
    assert result.support_basis is None
    assert result.provenance["measure"] == "raw_covariance_support_unresolved"
    assert result.provenance["ambient_factorization_status"] == "raw_covariance_cholesky_unresolved"
    for method in (result.whiten, result.log_density, result.log_likelihood_correction):
        with pytest.raises(ValueError):
            method(np.zeros((3, 1)))
    # Raw diagnostics continue to work, with no new law claimed or ridge added.
    assert np.isfinite(result.relative_error_trace)
    result.whitening_gain()
    result.spectral_comparison(64)
    result.assert_integrity()


@pytest.mark.parametrize("basis", [np.eye(3), np.eye(4)[:, :2], 2 * np.eye(4)[:, :3], np.eye(4)])
def test_wrong_or_superfluous_support_is_rejected(basis):
    with pytest.raises(ValueError, match="support"):
        ExactPropagatedCovariance.compute(_difference(), _noise(), 4, support_basis=basis)


def test_regularized_diagnostic_uses_explicit_changed_law_and_preserves_state():
    result = ExactPropagatedCovariance.compute(_operator(0), _noise(), 3)
    binding = result._integrity_sha256
    r = np.array([[1.0], [2.0], [-1.0]])
    first = result.regularized_diagnostic(r, ridge=0.25)
    second = result.regularized_diagnostic(r, ridge=0.5)
    assert first["measure"] == "regularized_ambient_diagnostic"
    np.testing.assert_allclose(first["cholesky"] @ first["whitened"], r)
    assert first["log_density"] == pytest.approx(multivariate_normal.logpdf(r[:, 0], cov=0.25 * np.eye(3)))
    assert first["covariance_sha256"] != second["covariance_sha256"]
    assert result._integrity_sha256 == binding
    result.assert_integrity()
    with pytest.raises(ValueError, match="positive"):
        result.regularized_diagnostic(r, ridge=0)


@pytest.mark.parametrize("field", ["K_propagated", "K_ar", "cholesky_propagated",
                                  "support_basis", "cholesky_support", "eigenvalues",
                                  "numerical_rank", "operator_digest", "noise_digest",
                                  "n_samples", "relative_error_trace", "provenance"])
def test_all_used_state_is_integrity_bound(field):
    result = ExactPropagatedCovariance.compute(_operator(), _noise(), 3)
    value = getattr(result, field)
    if isinstance(value, np.ndarray):
        with pytest.raises(ValueError):
            value.setflags(write=True)
        value = value.copy()
        value.flat[0] += 0.1
    elif isinstance(value, dict):
        value = {**value, "measure": "forged"}
    elif isinstance(value, str):
        value += "forged"
    else:
        value += 1
    object.__setattr__(result, field, value)
    with pytest.raises(ValueError, match="integrity"):
        result.whiten(np.zeros((3, 1)))


@pytest.mark.parametrize("values", [np.zeros((2, 1)), np.zeros((3, 0)), np.zeros(3),
                                   np.full((3, 1), np.nan), np.ones((3, 1), dtype=complex)])
def test_residual_shape_and_finite_real_contract(values):
    result = ExactPropagatedCovariance.compute(_operator(), _noise(), 3)
    with pytest.raises(ValueError):
        result.whiten(values)


def test_conditional_prefix_cannot_be_called_stationary_covariance():
    noise = TemporalNoise.from_parameters([0.3], innovation_variance=1,
                                          reference_id="conditional", startup_mode="conditional_prefix")
    with pytest.raises(ValueError, match="conditional-prefix"):
        ExactPropagatedCovariance.compute(_operator(), noise, 4)
