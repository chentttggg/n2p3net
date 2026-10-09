"""Independent small seams for continuous-Gamma joint-response sampling."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, t

from research.joint_response_sampling import (
    PredictiveSamplingError,
    conditional_beta_chart,
    conditional_scale_parameters,
    conditional_tail_components,
    posterior_diagnostics,
    prefix_importance_predict,
    prepare_model,
    prepare_prediction,
    run_gibbs,
    targeted_prefix_predict,
)
from research.joint_response_scale import DecisionBlock, FixedHyperparameters, GlobalGaussianPrior


def _fixture(nu=4.):
    prior = GlobalGaussianPrior([0., .7], [[.5, 0], [.15, .8]], "external")
    c = DecisionBlock("C1", [.2, 4.], [[1., 0.], [.5, 1.]], [[.6, 0], [.15, .7]], "target_calibration")
    test = DecisionBlock("T1", [.4, .2], [[1., 1.], [0., 1.]], [[.7, 0], [.2, .9]], "read_only_inference")
    hyper = FixedHyperparameters(nu, 1.7, "externally_fixed")
    return prior, c, test, hyper


def _draws(model, draws=400, burnin=100, seeds=(20260918, 20260919, 20260920, 20260921)):
    starts = np.stack([np.full(len(model.prior.mean), (-1 if i % 2 else 1) * 3.) for i in range(len(seeds))])
    return run_gibbs(model, burnin=burnin, draws=draws, seeds=seeds, initial_charts=starts)


def _policy():
    # Explicit synthetic caller policy, not a module default/accuracy certificate.
    return {"batch_size": 20, "max_rhat": 1.2, "min_bulk_ess": 20, "min_tail_ess": 20}


def _importance(samples, prediction, **changes):
    options = _policy() | {"min_weight_ess_fraction": .001, "max_weight_fraction": .2} | changes
    return prefix_importance_predict(samples, prediction, **options)


def test_migration_shared_cross_prior_and_conditional_scales_match_dense_gaussian():
    prior, c, test, hyper = _fixture()
    past = DecisionBlock(test.decision_id, test.samples[:1], test.design[:1], test.noise_root[:1, :1], test.role)
    model = prepare_model(prior, [c, past], hyper)
    lam = np.array([.35, 1.2])
    mean, root = conditional_beta_chart(model, lam)
    D, y = np.vstack((c.design, past.design)), np.r_[c.samples, past.samples]
    noise = np.zeros((3, 3))
    noise[:2, :2] = hyper.relative_noise_shape * c.noise_root @ c.noise_root.T / lam[0]
    noise[2:, 2:] = hyper.relative_noise_shape * past.noise_root @ past.noise_root.T / lam[1]
    V = prior.root @ prior.root.T
    gain = V @ D.T @ np.linalg.inv(noise + D @ V @ D.T)
    np.testing.assert_allclose(prior.mean + prior.root @ mean, prior.mean + gain @ (y - D @ prior.mean), atol=1e-13)
    np.testing.assert_allclose(prior.root @ cho_solve((root, True), np.eye(2)) @ prior.root.T,
                               V - gain @ D @ V, atol=1e-13)
    chart = np.array([.2, -.6])
    shapes, rates = conditional_scale_parameters(model, chart)
    beta = prior.mean + prior.root @ chart
    energy = [np.sum(solve_triangular(v.noise_root, v.samples - v.design @ beta, lower=True) ** 2)
              / hyper.relative_noise_shape for v in (c, past)]
    np.testing.assert_allclose(shapes, [3., 2.5], atol=0)
    np.testing.assert_allclose(rates, (4 + np.asarray(energy)) / 2, atol=1e-13)
    assert model.hyperparameters is hyper and model.prior is prior
    with pytest.raises(ValueError):
        model.blocks[0].augmented_root.flags.writeable = True


def test_migration_no_c_gaussian_sampling_has_the_physical_prior_and_zero_extra_lambdas():
    prior, _, _, hyper = _fixture(np.inf)
    model = prepare_model(prior, (), hyper)
    mean, root = conditional_beta_chart(model, [])
    np.testing.assert_array_equal(mean, [0., 0.])
    np.testing.assert_array_equal(root, np.eye(2))
    seeds = (101, 102)
    samples = run_gibbs(model, burnin=0, draws=40, seeds=seeds, initial_charts=np.zeros((2, 2)))
    expected = np.stack([np.random.default_rng(seed).normal(size=(40, 2)) for seed in seeds])
    np.testing.assert_array_equal(samples.chart, expected)
    np.testing.assert_allclose(samples.beta, prior.mean + expected @ prior.root.T, atol=0)
    assert samples.scales.shape == (2, 40, 0)
    assert samples.gaussian_limit
    with pytest.raises(ValueError):
        samples.chart.flags.writeable = True


@pytest.mark.parametrize("nu", [4., np.inf])
def test_dense_conditionals_match_same_prefix_t_and_gaussian_components(nu):
    prior, _, test, hyper = _fixture(nu)
    model = prepare_model(prior, (), hyper)
    prediction = prepare_prediction(model, test, conditioning_rows=1)
    charts = np.array([[-.3, .2], [.8, -.5], [0., 1.]])
    full, prefix, conditional = conditional_tail_components(prediction, charts)
    K = hyper.relative_noise_shape * test.noise_root @ test.noise_root.T
    means = (prior.mean + charts @ prior.root.T) @ test.design.T
    gain = K[1, 0] / K[0, 0]
    schur = K[1, 1] - gain * K[0, 1]
    for i, mean in enumerate(means):
        location = mean[1] + gain * (test.samples[0] - mean[0])
        if np.isfinite(nu):
            delta = (test.samples[0] - mean[0]) ** 2 / K[0, 0]
            expected = t.logpdf(test.samples[1], nu + 1, loc=location,
                                scale=np.sqrt((nu + delta) / (nu + 1) * schur))
        else:
            expected = multivariate_normal.logpdf([test.samples[1]], mean=[location], cov=[[schur]])
        assert conditional[i] == pytest.approx(expected, abs=1e-12)
        assert conditional[i] == full[i] - prefix[i]


def test_migration_empirical_c_weights_are_fixed_and_normalized_with_cached_physical_test(monkeypatch):
    prior, c, test, hyper = _fixture()
    model = prepare_model(prior, [c], hyper)
    samples = _draws(model)
    prediction = prepare_prediction(model, test, conditioning_rows=1)
    result = _importance(samples, prediction)
    _, prefix, _ = conditional_tail_components(prediction, samples.chart)
    weights = np.exp(prefix - logsumexp(prefix))
    np.testing.assert_array_equal(result["prefix_weights"], weights)
    assert result["finite_predictive_mixture_proper"]
    assert not result["continuous_Gamma_target_accuracy_certified"]
    assert result["diagnostics"]["scope"] == "finite_MCMC_coordinate_diagnostics"
    # Independently integrate the dense conditional-t mixture using the SAME
    # finite weights; no reweighting by tail and no Gamma quadrature dependency.
    beta = samples.beta.reshape(-1, 2)
    means = beta @ test.design.T
    K = hyper.relative_noise_shape * test.noise_root @ test.noise_root.T
    gain = K[1, 0] / K[0, 0]
    locations = means[:, 1] + gain * (test.samples[0] - means[:, 0])
    scale = np.sqrt((4 + (test.samples[0] - means[:, 0]) ** 2 / K[0, 0]) / 5
                    * (K[1, 1] - gain * K[0, 1]))
    def density(value):
        return float(weights @ t.pdf(value, 5, loc=locations, scale=scale))
    mass, error = quad(density, -np.inf, np.inf, epsabs=1e-9, epsrel=1e-9)
    assert mass == pytest.approx(1., abs=2e-9)
    assert error < 2e-9
    expected = np.log(density(test.samples[1]))
    assert result["log_density"] == pytest.approx(expected, abs=1e-12)
    def forbid_rewhitening(*args, **kwargs):
        raise AssertionError("reusing prepared test must not whiten the physical model again")
    monkeypatch.setattr("research.joint_response_sampling.solve_triangular", forbid_rewhitening)
    reused = _importance(samples, prediction)
    assert reused["log_density"] == result["log_density"]


def test_migration_targeted_sampling_uses_same_prefix_once_and_same_global_model():
    prior, c, test, hyper = _fixture()
    prefix = DecisionBlock(test.decision_id, test.samples[:1], test.design[:1], test.noise_root[:1, :1], test.role)
    model = prepare_model(prior, [c, prefix], hyper)
    samples = _draws(model, seeds=(201, 202, 203, 204))
    prediction = prepare_prediction(model, test, conditioning_rows=1)
    result = targeted_prefix_predict(samples, prediction, **_policy())
    _, _, component = conditional_tail_components(prediction, samples.chart)
    assert result["log_density"] == pytest.approx(logsumexp(component) - np.log(len(component)), abs=1e-14)
    assert samples.scales.shape == (4, 400, 2)  # One C and the SAME test prefix.
    assert result["uncertainty_scope"] == "conditional_mean_delta_method_batch_MCSE_estimate"
    assert not result["continuous_Gamma_target_accuracy_certified"]
    wrong_test = DecisionBlock(test.decision_id, test.samples + [.1, 0.], test.design, test.noise_root, test.role)
    with pytest.raises(ValueError, match="exactly once"):
        targeted_prefix_predict(samples, prepare_prediction(model, wrong_test, conditioning_rows=1), **_policy())


@pytest.mark.parametrize("attack", ["duplicate", "role", "test_id", "hyperparameters", "prior", "lambda_count"])
def test_complete_decision_prior_noise_and_role_identity_attacks_are_rejected(attack):
    prior, c, test, hyper = _fixture()
    model = prepare_model(prior, [c], hyper)
    if attack == "duplicate":
        with pytest.raises(ValueError):
            prepare_model(prior, [c, c], hyper)
    elif attack == "role":
        with pytest.raises(ValueError):
            prepare_model(prior, [test, c], hyper)
    elif attack == "test_id":
        with pytest.raises(ValueError):
            prepare_prediction(model, replace(test, decision_id=c.decision_id), conditioning_rows=1)
    else:
        samples = _draws(model, draws=40, burnin=0)
        if attack == "lambda_count":
            with pytest.raises(ValueError, match="every complete"):
                replace(samples, scales=np.empty((4, 40, 0)))
        else:
            other_prior = replace(prior, mean=prior.mean + [.1, 0.]) if attack == "prior" else prior
            other_hyper = replace(hyper, relative_noise_shape=2.) if attack == "hyperparameters" else hyper
            other = prepare_model(other_prior, [c], other_hyper)
            with pytest.raises(ValueError, match="same physical"):
                _importance(samples, prepare_prediction(other, test, conditioning_rows=1))


def test_caller_declared_importance_collapse_is_typed_failure_without_implicit_algorithm_switch():
    prior, c, test, hyper = _fixture()
    model = prepare_model(prior, [c], hyper)
    samples = _draws(model)
    prediction = prepare_prediction(model, test, conditioning_rows=1)
    with pytest.raises(PredictiveSamplingError, match="weights collapsed"):
        _importance(samples, prediction, min_weight_ess_fraction=.999999)
    assert samples.model is model and len(samples.model.blocks) == 1
    with pytest.raises(ValueError, match="thresholds"):
        _importance(samples, prediction, max_rhat=np.nan)


def test_opposing_modes_can_give_false_green_when_every_chain_starts_in_one_mode():
    prior = GlobalGaussianPrior([0.], [[3.]], "external")
    hyper = FixedHyperparameters(4., 1., "externally_fixed")
    blocks = [DecisionBlock(f"C{i}", [value], [[1.]], [[.1]], "target_calibration")
              for i, value in enumerate([-4., 4.])]
    model = prepare_model(prior, blocks, hyper)
    missed = run_gibbs(model, burnin=100, draws=1000, seeds=(201, 202, 203, 204),
                       initial_charts=np.full((4, 1), 4 / 3))
    diagnostics = posterior_diagnostics(missed)
    assert diagnostics["max_rank_folded_split_rhat"] < 1.03
    assert diagnostics["min_bulk_ess"] > 500
    # Independent prior/likelihood symmetry is the exact mean oracle, not a
    # second sampler or a numerical quadrature convergence assumption.
    assert abs(missed.beta.mean()) > 2.  # Exact population posterior mean is 0.
    assert not diagnostics["continuous_Gamma_target_accuracy_certified"]
    assert not diagnostics["unvisited_modes_detected"]


def test_constant_observed_functional_is_not_given_infinite_or_certified_ess():
    prior, _, test, hyper = _fixture(np.inf)
    test = replace(test, design=np.zeros_like(test.design))
    model = prepare_model(prior, (), hyper)
    samples = _draws(model, draws=160, burnin=0)
    result = _importance(samples, prepare_prediction(model, test, conditioning_rows=1))
    diagnostics = result["log_density_influence_diagnostics"]
    assert diagnostics["constant_at_retained_draws"]
    assert diagnostics["bulk_ess"] is None and diagnostics["rank_folded_split_rhat"] is None
    assert result["conditional_C_scale_mean"].shape == (0,)
    assert result["log_density_batch_mcse"] == 0.
    assert not result["continuous_Gamma_target_accuracy_certified"]


def test_physical_voltage_units_leave_chart_conditionals_and_tail_jacobian_consistent():
    prior, c, test, hyper = _fixture()
    model = prepare_model(prior, [c], hyper)
    prediction = prepare_prediction(model, test, conditioning_rows=1)
    chart = np.array([[-.3, .2], [.8, -.5]])
    _, _, original = conditional_tail_components(prediction, chart)
    gain = 1e6
    scaled_prior = GlobalGaussianPrior(prior.mean * gain, prior.root * gain, prior.program)
    scaled_c = replace(c, samples=c.samples * gain, noise_root=c.noise_root * gain)
    scaled_test = replace(test, samples=test.samples * gain, noise_root=test.noise_root * gain)
    scaled_model = prepare_model(scaled_prior, [scaled_c], hyper)
    scaled_prediction = prepare_prediction(scaled_model, scaled_test, conditioning_rows=1)
    _, _, scaled = conditional_tail_components(scaled_prediction, chart)
    mean, root = conditional_beta_chart(model, np.array([.7]))
    scaled_mean, scaled_root = conditional_beta_chart(scaled_model, np.array([.7]))
    np.testing.assert_allclose(scaled_mean, mean, atol=1e-13)
    np.testing.assert_allclose(scaled_root @ scaled_root.T, root @ root.T, atol=1e-13)
    np.testing.assert_allclose(scaled, original - np.log(gain), atol=1e-12)
