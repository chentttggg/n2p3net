"""Proper finite-scale joint models and independently computed migration seams."""

from itertools import product

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, multivariate_t, t

from research.joint_response_scale import (
    DecisionBlock,
    DiscreteGammaRule,
    FixedHyperparameters,
    GlobalGaussianPrior,
    NumericalResolutionError,
    conditional_gaussian_components,
    conditional_tail,
    gamma_prior_rule,
    joint_evidence,
)


def _fixtures():
    prior = GlobalGaussianPrior([.2], [[.7]], "source_only")
    calibration = [DecisionBlock("CA", [2.2], [[1.]], [[.6]], "target_calibration"),
                   DecisionBlock("CB", [-.1], [[.8]], [[.9]], "target_calibration")]
    test = DecisionBlock("TT", [.1, .8], [[1.], [.5]], [[.7, 0], [.2, .9]], "read_only_inference")
    return prior, calibration, test, FixedHyperparameters(6., 1.3, "externally_fixed")


def _prefix(test):
    return DecisionBlock(test.decision_id, test.samples[:1], test.design[:1], test.noise_root[:1, :1], test.role)


def _dense_component(prior, blocks, hyper, scales):
    """Full observation covariance; no implementation precision-chart helper."""
    blocks = tuple(blocks)
    D = np.vstack([v.design for v in blocks])
    y = np.concatenate([v.samples for v in blocks])
    V0 = prior.root @ prior.root.T
    covariance = D @ V0 @ D.T
    start = 0
    for block, scale in zip(blocks, scales, strict=True):
        stop = start + len(block.samples)
        covariance[start:stop, start:stop] += hyper.relative_noise_shape / scale * block.noise_root @ block.noise_root.T
        start = stop
    residual = y - D @ prior.mean
    gain = V0 @ D.T @ np.linalg.inv(covariance)
    sign, determinant = np.linalg.slogdet(covariance)
    assert sign == 1
    log_density = -.5 * (len(y) * np.log(2 * np.pi) + determinant
                         + residual @ np.linalg.solve(covariance, residual))
    return (float(log_density),
            prior.mean + gain @ residual, V0 - gain @ D @ V0)


def _dense_finite(prior, blocks, hyper, rule):
    scales = np.array(list(product(rule.nodes, repeat=len(blocks))))
    logs = np.array([sum(np.log(rule.weights[i]) for i in indices)
                     for indices in product(range(len(rule.nodes)), repeat=len(blocks))])
    components = [_dense_component(prior, blocks, hyper, row) for row in scales]
    terms = logs + np.array([v[0] for v in components])
    mass = float(logsumexp(terms))
    weights = np.exp(terms - mass)
    means = np.array([v[1] for v in components])
    covariances = np.array([v[2] for v in components])
    mean = weights @ means
    covariance = np.einsum("b,bij->ij", weights, covariances) + ((means - mean).T * weights) @ (means - mean)
    scale_mean = weights @ scales
    scale_covariance = ((scales - scale_mean).T * weights) @ (scales - scale_mean)
    return mass, mean, covariance, scale_mean, scale_covariance


def _actual_tail_mass(prior, calibration, test, hyper, rule):
    past = joint_evidence(prior, tuple(calibration) + (_prefix(test),), hyper, scale_rule=rule)

    def density(tail):
        changed = DecisionBlock(test.decision_id, [test.samples[0], tail], test.design, test.noise_root, test.role)
        full = joint_evidence(prior, tuple(calibration) + (changed,), hyper, scale_rule=rule)
        return np.exp(full.log_evidence - past.log_evidence)

    return quad(density, -np.inf, np.inf, epsabs=1e-10, epsrel=1e-10)[0]


def _continuous_beta_first(prior, blocks, hyper):
    """Reverse order: continuous Gamma -> t per decision, one shared beta last."""
    def integrand(beta):
        logarithm = multivariate_normal.logpdf([beta], prior.mean, prior.root @ prior.root.T)
        for block in blocks:
            logarithm += multivariate_t.logpdf(block.samples - block.design[:, 0] * beta,
                shape=hyper.relative_noise_shape * block.noise_root @ block.noise_root.T,
                df=hyper.degrees_of_freedom)
        return np.exp(logarithm)

    return float(np.log(quad(integrand, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)[0]))


def test_migration_coarse_common_rule_normalizes_locked_d0_counterexample():
    # Same physical fixture that exposed 0.9801555 mass at old tilted order6.
    # The normalized finite model may be coarse; properness does not certify
    # continuous-Gamma accuracy, Gamma mean precision one, or a t tail law.
    prior = GlobalGaussianPrior([.3], [[1.4]], "external")
    calibration = [DecisionBlock("C", [.8], [[0.]], [[1.1]], "target_calibration")]
    test = DecisionBlock("T", [.4, 1.2], [[0.], [0.]], [[.7, 0], [.2, .9]], "read_only_inference")
    hyper = FixedHyperparameters(2.5, 1.7, "externally_fixed")
    rule = gamma_prior_rule(2.5, 6)
    value = conditional_tail(prior, calibration, test, 1, hyper, scale_rule=rule)
    assert value.full.scale_rule is value.prefix.scale_rule is rule
    assert _actual_tail_mass(prior, calibration, test, hyper, rule) == pytest.approx(1., abs=1e-9)
    assert rule.weights @ rule.nodes == pytest.approx(.17874942498277163, abs=1e-13)
    K = test.noise_root @ test.noise_root.T
    location = K[1, 0] / K[0, 0] * .4
    scale = np.sqrt(1.7 * (2.5 + .4**2 / (1.7 * K[0, 0])) / 3.5
                    * (K[1, 1] - K[1, 0]**2 / K[0, 0]))
    exact = t.logpdf(1.2, 3.5, location, scale)
    assert abs(value.conditional_log_density - exact) > .3
    dense_full = _dense_finite(prior, calibration + [test], hyper, rule)
    dense_prefix = _dense_finite(prior, calibration + [_prefix(test)], hyper, rule)
    assert value.conditional_log_density == pytest.approx(dense_full[0] - dense_prefix[0], abs=1e-12)
    assert value.full.numerical_program == "proper_discrete_scale_joint_approximation"


def test_migration_shared_beta_two_c_actual_tail_and_independent_finite_moments():
    prior, calibration, test, hyper = _fixtures()
    rule = gamma_prior_rule(6., 8)
    value = conditional_tail(prior, calibration, test, 1, hyper, scale_rule=rule)
    dense_full = _dense_finite(prior, calibration + [test], hyper, rule)
    dense_prefix = _dense_finite(prior, calibration + [_prefix(test)], hyper, rule)
    for actual, oracle in ((value.full, dense_full), (value.prefix, dense_prefix)):
        assert actual.log_evidence == pytest.approx(oracle[0], abs=1e-12)
        for result, expected in zip((actual.beta_mean, actual.beta_covariance,
                                    actual.scale_mean, actual.scale_covariance), oracle[1:], strict=True):
            np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)
    assert _actual_tail_mass(prior, calibration, test, hyper, rule) == pytest.approx(1., abs=1e-9)
    C = joint_evidence(prior, calibration, hyper, scale_rule=rule)
    assert np.linalg.norm(C.beta_mean - value.prefix.beta_mean) > .01
    assert np.linalg.norm(C.scale_mean - value.prefix.scale_mean[:2]) > .01
    assert abs(value.prefix.scale_covariance[0, -1]) > 1e-5
    assert value.full.decision_ids == ("CA", "CB", "TT")
    # The same public component evaluator can be reused by another inference
    # engine without redefining the prior/decision classes or the physical law.
    scales = np.array([[.7, 1.2, .9], [1.1, .5, 2.]])
    likelihoods, means, covariances = conditional_gaussian_components(prior, calibration + [test], hyper, scales)
    for i, row in enumerate(scales):
        expected = _dense_component(prior, calibration + [test], hyper, row)
        assert likelihoods[i] == pytest.approx(expected[0], abs=1e-12)
        np.testing.assert_allclose(means[i], expected[1], atol=1e-12)
        np.testing.assert_allclose(covariances[i], expected[2], atol=1e-12)


def test_migration_continuous_reverse_order_oracle_and_no_c_reduction():
    prior, calibration, test, hyper = _fixtures()
    rule = gamma_prior_rule(6., 64)
    for C in (calibration, []):
        actual = conditional_tail(prior, C, test, 1, hyper, scale_rule=rule)
        expected = (_continuous_beta_first(prior, C + [test], hyper)
                    - _continuous_beta_first(prior, C + [_prefix(test)], hyper))
        assert actual.conditional_log_density == pytest.approx(expected, abs=1e-9)
    empty = joint_evidence(prior, [], hyper, scale_rule=rule)
    assert empty.log_evidence == 0 and empty.evaluated_nodes == 0
    np.testing.assert_array_equal(empty.beta_mean, prior.mean)
    np.testing.assert_array_equal(empty.beta_covariance, prior.root @ prior.root.T)


def test_migration_gaussian_infinity_uses_cross_record_beta_covariance():
    prior, calibration, test, _ = _fixtures()
    hyper = FixedHyperparameters(np.inf, 1.3, "externally_fixed")
    rule = gamma_prior_rule(np.inf, 1)
    value = conditional_tail(prior, calibration, test, 1, hyper, scale_rule=rule)
    full = _dense_component(prior, calibration + [test], hyper, [1., 1., 1.])
    past = _dense_component(prior, calibration + [_prefix(test)], hyper, [1., 1., 1.])
    assert value.conditional_log_density == pytest.approx(full[0] - past[0], abs=1e-12)
    np.testing.assert_allclose(value.full.beta_mean, full[1], atol=1e-12)
    np.testing.assert_allclose(value.full.beta_covariance, full[2], atol=1e-12)
    np.testing.assert_array_equal(value.full.scale_mean, [1., 1., 1.])
    np.testing.assert_array_equal(value.full.scale_covariance, np.zeros((3, 3)))
    assert rule.quadrature_order == 0 and value.full.numerical_program == "Gaussian_limit"


def test_migration_actual_inputs_outputs_and_rule_are_bytes_immutable():
    mean, root, y, design, noise = (np.array([.2]), np.array([[.7]]), np.array([.1, .8]),
                                  np.array([[1.], [.5]]), np.array([[.7, 0], [.2, .9]]))
    prior = GlobalGaussianPrior(mean, root, "C_design_only")
    test = DecisionBlock("T", y, design, noise, "read_only_inference")
    hyper = FixedHyperparameters(6., 1.3, "frozen_C_plugin")
    rule = gamma_prior_rule(6., 12)
    original = conditional_tail(prior, [], test, 1, hyper, scale_rule=rule)
    for array in (mean, root, y, design, noise):
        array[:] = 99.
    unchanged = conditional_tail(prior, [], test, 1, hyper, scale_rule=rule)
    assert original.conditional_log_density == unchanged.conditional_log_density
    for array in (prior.mean, prior.root, test.samples, test.design, test.noise_root,
                  rule.nodes, rule.weights, original.full.beta_mean, original.full.beta_covariance,
                  original.full.scale_mean, original.full.scale_covariance):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    replay = DiscreteGammaRule(rule.nodes, rule.weights, rule.degrees_of_freedom, rule.quadrature_order,
                              rule.log_precision_bounds, rule.excluded_gamma_prior_probability, rule.method)
    assert conditional_tail(prior, [], test, 1, hyper, scale_rule=replay).conditional_log_density == pytest.approx(
        original.conditional_log_density, abs=1e-12)


def test_migration_c_roles_programs_and_scale_identity_are_explicit():
    prior, calibration, test, hyper = _fixtures()
    rule = gamma_prior_rule(6., 6)
    with pytest.raises(ValueError, match="independent prior"):
        GlobalGaussianPrior([0.], [[1.]], "old_gaussian_C_response_posterior")
    with pytest.raises(ValueError, match="repeat"):
        conditional_tail(prior, calibration + [calibration[0]], test, 1, hyper, scale_rule=rule)
    with pytest.raises(ValueError, match="declared C"):
        conditional_tail(prior, [test], test, 1, hyper, scale_rule=rule)
    with pytest.raises(ValueError, match="prior nu"):
        conditional_tail(prior, calibration, test, 1, hyper, scale_rule=gamma_prior_rule(4., 6))
    with pytest.raises(ValueError):
        FixedHyperparameters(True, 1., "externally_fixed")
    with pytest.raises(ValueError):
        GlobalGaussianPrior([1j], [[1.]], "external")


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_migration_unresolved_beta_moments_fail_not_nan_or_false_zero(scale):
    prior = GlobalGaussianPrior([0.], [[scale]], "external")
    hyper = FixedHyperparameters(6., 1., "externally_fixed")
    rule = gamma_prior_rule(6., 6)
    with pytest.raises(NumericalResolutionError):
        joint_evidence(prior, [], hyper, scale_rule=rule)
    block = DecisionBlock("C", [0.], [[0.]], [[1.]], "target_calibration")
    with pytest.raises(NumericalResolutionError):
        joint_evidence(prior, [block], hyper, scale_rule=rule)


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_migration_unresolved_scale_moments_fail_not_nan_or_false_point_prior(scale):
    prior = GlobalGaussianPrior([0.], [[1.]], "external")
    hyper = FixedHyperparameters(6., 1., "externally_fixed")
    rule = DiscreteGammaRule([scale, 2 * scale], [.5, .5], 6., 2)
    block = DecisionBlock("C", [0.], [[0.]], [[1.]], "target_calibration")
    with pytest.raises(NumericalResolutionError):
        joint_evidence(prior, [block], hyper, scale_rule=rule)
