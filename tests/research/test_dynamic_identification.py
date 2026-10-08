"""Independent density oracles and B11 hierarchy/conditioning counterexamples."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import block_diag, solve_discrete_lyapunov
from scipy.stats import gamma, multivariate_normal, multivariate_t

from research.dynamic_identification import (
    ScaleMixtureDensity,
    ScaleMixturePrecisionError,
    noise_scale_mixture_conditional_logpdf,
    noise_scale_mixture_logpdf,
)


def _density(y, mean, noise, gaussian, nu):
    return noise_scale_mixture_logpdf(y, mean, noise, gaussian, degrees_of_freedom=nu,
                                    relative_tolerance=1e-10)


def _dense_lambda_density(y, mean, noise, gaussian, nu):
    """Integrate in lambda, using independently formed dense covariance."""
    def integrand(value):
        covariance = noise @ noise.T / value + gaussian @ gaussian.T
        return np.exp(multivariate_normal.logpdf(y, mean=mean, cov=covariance)
                      + gamma.logpdf(value, nu / 2, scale=2 / nu))

    result, error = quad(integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-10, limit=300)
    assert error < result * 1e-8
    return result


@pytest.mark.parametrize("nu", [.5, 1., 2., 4., 30.])
def test_no_erp_uncertainty_matches_normalized_multivariate_t(nu):
    noise = np.array([[2., 0.], [.3, .7]])
    y, mean = np.array([3., -2.]), np.array([.2, -.1])
    result = _density(y, mean, noise, np.empty((2, 0)), nu)
    expected = multivariate_t.logpdf(y, loc=mean, shape=noise @ noise.T, df=nu)
    assert result.log_density == pytest.approx(expected, abs=1e-9)
    assert result.relative_error_estimate <= 1e-8
    assert result.dimensions == 2
    assert result.degrees_of_freedom == nu


def test_actual_prefix_updates_noise_scale_and_degrees_of_freedom():
    nu, x, y = 4., 5., .3
    result = noise_scale_mixture_conditional_logpdf([x, y], [0., 0.], np.eye(2),
        np.empty((2, 0)), conditioning_rows=1, degrees_of_freedom=nu)
    # A zero covariance cross-block is not independence under a shared t scale.
    expected = multivariate_t.logpdf([y], loc=[0.], shape=[[(nu + x * x) / (nu + 1)]],
                                     df=nu + 1)
    assert result.log_density == pytest.approx(expected, abs=1e-9)
    wrong_unchanged_prior = multivariate_t.logpdf([y], loc=[0.], shape=[[1.]], df=nu)
    assert abs(result.log_density - wrong_unchanged_prior) > .5
    assert result.relative_error_estimate < 1e-8


def test_independent_erp_uncertainty_uses_noise_only_scale_not_total_covariance():
    noise = np.array([[1., 0.], [.3, .8]])
    gaussian = np.array([[2.], [-.4]])
    y, mean, nu = np.array([4., -.3]), np.array([.2, -.1]), 4.
    result = _density(y, mean, noise, gaussian, nu)
    direct = _dense_lambda_density(y, mean, noise, gaussian, nu)
    assert result.log_density == pytest.approx(np.log(direct), abs=1e-9)
    wrong_common_scale = multivariate_t.logpdf(y, loc=mean,
        shape=noise @ noise.T + gaussian @ gaussian.T, df=nu)
    assert abs(result.log_density - wrong_common_scale) > .1
    # Units: noise-only finite covariance is nu/(nu-2)*Kn + Kg.
    noise_only_covariance = nu / (nu - 2) * noise @ noise.T + gaussian @ gaussian.T
    common_scale_covariance = nu / (nu - 2) * (noise @ noise.T + gaussian @ gaussian.T)
    assert np.linalg.norm(common_scale_covariance - noise_only_covariance) > 4.


def test_conditional_is_ratio_of_integrals_not_prior_average_of_conditionals():
    noise = np.array([[1., 0.], [.2, .9]])
    gaussian = np.array([[1.], [.6]])
    y, mean, nu = np.array([8., .5]), np.zeros(2), 3.
    result = noise_scale_mixture_conditional_logpdf(y, mean, noise, gaussian,
        conditioning_rows=1, degrees_of_freedom=nu)
    joint = _dense_lambda_density(y, mean, noise, gaussian, nu)
    prefix = _dense_lambda_density(y[:1], mean[:1], noise[:1, :1], gaussian[:1], nu)
    assert result.log_density == pytest.approx(np.log(joint / prefix), abs=1e-9)

    def wrong_integrand(value):
        covariance = noise @ noise.T / value + gaussian @ gaussian.T
        gain = covariance[1, 0] / covariance[0, 0]
        conditional_variance = covariance[1, 1] - gain * covariance[0, 1]
        return np.exp(multivariate_normal.logpdf(y[1:], mean=[gain * y[0]],
                      cov=[[conditional_variance]]) + gamma.logpdf(value, nu / 2, scale=2 / nu))

    wrong, _ = quad(wrong_integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-10)
    assert abs(result.log_density - np.log(wrong)) > .2


def test_conditional_density_normalizes_with_independent_gaussian_erp():
    noise, gaussian, nu = np.eye(2), np.array([[1.], [.6]]), 5.
    prefix = _density([2.], [0.], noise[:1, :1], gaussian[:1], nu)

    def density(tail):
        joint = _density([2., tail], [0., 0.], noise, gaussian, nu)
        return np.exp(joint.log_density - prefix.log_density)

    integral, error = quad(density, -np.inf, np.inf, epsabs=2e-7, epsrel=2e-7, limit=100)
    assert integral == pytest.approx(1., abs=3e-7)
    assert error < 3e-7


def test_voltage_units_and_candidate_density_ratios_are_preserved():
    noise = np.array([[1., 0., 0.], [.3, .8, 0.], [.2, -.1, 1.1]])
    gaussian = np.array([[.4], [.7], [-.3]])
    y, mean = np.array([2., 1., -1.]), np.array([.1, .2, .3])
    base = noise_scale_mixture_conditional_logpdf(y, mean, noise, gaussian,
        conditioning_rows=1, degrees_of_freedom=4.)
    scaled = noise_scale_mixture_conditional_logpdf(y * 1e6, mean * 1e6,
        noise * 1e6, gaussian * 1e6, conditioning_rows=1, degrees_of_freedom=4.)
    assert scaled.log_density == pytest.approx(base.log_density - 2 * np.log(1e6), abs=1e-8)


def test_one_decision_scale_is_not_independent_scale_for_each_flash():
    # The full joint has a single latent Gamma draw. Splitting it changes the law.
    y = np.array([5., .2])
    whole = _density(y, np.zeros(2), block_diag([[1.]], [[1.]]), np.empty((2, 0)), 3.)
    separate = sum(_density([v], [0.], np.eye(1), np.empty((1, 0)), 3.).log_density for v in y)
    assert abs(whole.log_density - separate) > .5


def test_gaussian_limit_preserves_independent_erp_covariance():
    y, mean = np.array([2., .5]), np.array([.1, .2])
    noise, gaussian = np.eye(2), np.array([[1.], [-.2]])
    result = _density(y, mean, noise, gaussian, 1e5)
    expected = multivariate_normal.logpdf(y, mean=mean, cov=noise @ noise.T + gaussian @ gaussian.T)
    assert result.log_density == pytest.approx(expected, abs=5e-5)


def test_narrow_gamma_peak_cannot_silently_lose_half_normalization_mass():
    # A single optimum knot let adaptive quadrature miss half the peak at nu=1e6
    # while claiming 2e-12 relative error. The oracle is the known t density.
    result = _density([0.], [0.], [[1.]], np.empty((1, 0)), 1e6)
    expected = multivariate_t.logpdf([0.], loc=[0.], shape=[[1.]], df=1e6)
    assert result.log_density == pytest.approx(expected, abs=2e-9)
    assert result.gamma_normalizer_log_error_bound == pytest.approx(1 / (1680 * (5e5) ** 7))


def test_gaussian_factor_rotation_preserves_the_declared_distribution():
    noise = np.array([[1., 0.], [.2, .7]])
    factor = np.array([[1., .3], [.1, -.2]])
    rotation = np.array([[0., -1.], [1., 0.]])
    original = _density([2., -.3], [0., .1], noise, factor, 4.)
    rotated = _density([2., -.3], [0., .1], noise, factor @ rotation, 4.)
    assert rotated.log_density == pytest.approx(original.log_density, abs=1e-9)


def test_conditional_minimum_requested_precision_uses_actual_combined_budget():
    result = noise_scale_mixture_conditional_logpdf([.2, .3], [0., 0.], np.eye(2),
        np.array([[.1], [.3]]), conditioning_rows=1, degrees_of_freedom=4.,
        relative_tolerance=1e-12)
    assert result.relative_error_estimate <= 1e-12


def test_joint_success_cannot_silently_exceed_conditional_precision_request(monkeypatch):
    density = ScaleMixtureDensity(0., 8e-13, 0., 0., (-10., 5.), 2, 4.)
    monkeypatch.setattr("research.dynamic_identification.noise_scale_mixture_logpdf",
                        lambda *args, **kwargs: density)
    with pytest.raises(ScaleMixturePrecisionError, match="combined"):
        noise_scale_mixture_conditional_logpdf([.2, .3], [0., 0.], np.eye(2),
            np.empty((2, 0)), conditioning_rows=1, degrees_of_freedom=4., relative_tolerance=1e-12)


@pytest.mark.parametrize("bad_noise", [np.zeros((2, 2)), [[1., .2], [0., 1.]],
                                         [[1., 0.], [0., 1j]]])
def test_unidentified_or_complex_noise_has_no_jitter_or_pseudoinverse_fallback(bad_noise):
    with pytest.raises(ValueError):
        _density([0., 0.], [0., 0.], bad_noise, np.empty((2, 0)), 4.)


@pytest.mark.parametrize("nu", [0., -1., np.inf, True])
def test_invalid_degrees_are_rejected(nu):
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        _density([0.], [0.], np.eye(1), np.empty((1, 0)), nu)


def test_unrepresentable_quadratic_is_explicit_precision_failure():
    with pytest.raises(ScaleMixturePrecisionError, match="energy"):
        _density([1e300], [0.], np.eye(1), np.empty((1, 0)), 4.)


def test_failed_interior_error_estimate_cannot_return_a_score(monkeypatch):
    monkeypatch.setattr("research.dynamic_identification.quad", lambda *args, **kwargs: (1., .1))
    with pytest.raises(ScaleMixturePrecisionError, match="interior error"):
        _density([0.], [0.], np.eye(1), np.empty((1, 0)), 4.)


@pytest.mark.parametrize("nu", [.1, 2., 100.])
def test_extreme_tail_keeps_omitted_gamma_mass_bounded(nu):
    noise = np.eye(1)
    result = _density([1e4], [0.], noise, np.empty((1, 0)), nu)
    expected = multivariate_t.logpdf([1e4], loc=[0.], shape=[[1.]], df=nu)
    assert result.log_density == pytest.approx(expected, abs=2e-8)
    assert result.analytic_omitted_tail_relative_bound < 1e-10
    assert result.integration_log_gamma_bounds[0] < -10


def test_noise_and_gaussian_observable_covariance_can_have_a_variance_alias():
    # Perfect observability of h does not identify ERP variance versus an
    # unrestricted event-locked Gaussian noise variance on that same h.
    h = np.array([[1.], [.8], [.4]])
    covariance_one = .2 * h @ h.T + .8 * h @ h.T + np.eye(3)
    covariance_two = .8 * h @ h.T + .2 * h @ h.T + np.eye(3)
    assert np.array_equal(covariance_one, covariance_two)


def test_latent_state_similarity_changes_parameters_without_changing_observations():
    # An observable model's state basis is still a gauge, not a physiological ID.
    transition = np.diag([.8, .4])
    observation = np.array([[1., 1.]])
    stationary = solve_discrete_lyapunov(transition, np.diag([.2, .3]))
    change = np.array([[2., .5], [0., .3]])
    inverse = np.linalg.inv(change)
    transformed_transition = change @ transition @ inverse
    transformed_observation = observation @ inverse
    transformed_stationary = change @ stationary @ change.T
    assert np.linalg.matrix_rank(np.vstack((observation, observation @ transition))) == 2
    for lag in range(12):
        original = observation @ np.linalg.matrix_power(transition, lag) @ stationary @ observation.T
        transformed = transformed_observation @ np.linalg.matrix_power(
            transformed_transition, lag) @ transformed_stationary @ transformed_observation.T
        assert transformed == pytest.approx(original, abs=1e-14)
