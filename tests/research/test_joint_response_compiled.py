"""Small physical oracles for canonical augmented-row compilation and prediction."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from research.channel_noise_mechanism import FittedChannelNoise, NoiseGeometry
from research.joint_response_sampling import (
    JointResponseSamples,
    conditional_beta_tail_components,
    conditional_scale_parameters,
    conditional_tail_components,
    prepare_model,
    prepare_prediction,
    targeted_conditional_beta_predict,
)
from research.joint_response_scale import (
    CompiledDecision,
    DecisionBlock,
    DiscreteGammaRule,
    FixedHyperparameters,
    GlobalGaussianPrior,
    compile_decision,
    compile_owned_decision,
    conditional_gaussian_components,
    joint_evidence,
)


def fixture(nu=4.):
    rng = np.random.default_rng(41)
    prior = GlobalGaussianPrior([.2, -.1], [[.7, 0], [.3, .8]], "external")
    hyper = FixedHyperparameters(nu, 1.3, "externally_fixed")
    c = DecisionBlock("C", rng.normal(size=7), rng.normal(size=(7, 2)),
                      np.eye(7), "target_calibration")
    t = DecisionBlock("T", rng.normal(size=9), rng.normal(size=(9, 2)),
                      np.eye(9), "read_only_inference")
    return prior, hyper, c, t


def dense_log(prior, hyper, blocks, scales):
    design = np.vstack([v.design for v in blocks])
    samples = np.concatenate([v.samples for v in blocks])
    covariance = block_diag(*[hyper.relative_noise_shape * v.noise_root @ v.noise_root.T / lam
                              for v, lam in zip(blocks, scales, strict=True)])
    covariance += design @ prior.root @ prior.root.T @ design.T
    return multivariate_normal.logpdf(samples, mean=design @ prior.mean, cov=covariance)


def test_compilation_retains_raw_dimensions_and_zero_coefficient_residual():
    prior, hyper, c, _ = fixture()
    c = replace(c, design=np.zeros_like(c.design))
    record = compile_decision(prior, hyper, c, qr_chunk_rows=2)
    assert record.observed_rows == 7 and record.augmented_root.shape == (3, 3)
    model = prepare_model(prior, [record], hyper)
    shapes, rates = conditional_scale_parameters(model, np.array([1., -.5]))
    np.testing.assert_array_equal(shapes, [(4 + 7) / 2])
    assert rates[0] == pytest.approx((4 + c.samples @ c.samples / 1.3) / 2)
    with pytest.raises(ValueError):
        record.augmented_root.flags.writeable = True
    with pytest.raises(TypeError, match="factory"):
        CompiledDecision()
    with pytest.raises(TypeError, match="proper"):
        prepare_model(prior, [{"flat_nuisance": True}], hyper)


def test_augmented_root_preserves_energy_where_gram_subtraction_cancels():
    prior = GlobalGaussianPrior([0.], [[1.]], "external")
    hyper = FixedHyperparameters(4., 1., "externally_fixed")
    design, samples = np.zeros((100, 1)), np.zeros(100)
    design[0, 0], samples[:2] = 1e8, [1e8, 1.]
    record = compile_decision(prior, hyper,
                              DecisionBlock("C", samples, design, np.eye(100), "target_calibration"))
    shapes, rates = conditional_scale_parameters(prepare_model(prior, [record], hyper), [1.])
    assert shapes[0] == 52 and rates[0] == pytest.approx(2.5, abs=1e-14)


@pytest.mark.parametrize("family", ["shared", "private"])
def test_owned_full_map_matches_dense_ar1_and_uses_public_jacobian(family, monkeypatch):
    prior, hyper, _, _ = fixture()
    geometry = NoiseGeometry(("Cz", "Pz"), 128., "V", "native", "software", 2)
    phi = np.array([.4, .4] if family == "shared" else [.55, -.2])
    sigma = np.array([[1., .2], [.2, 1.4]])
    noise = FittedChannelNoise.from_parameters(phi[:, None], sigma, geometry=geometry,
               conditioning_samples=4, reference_id="software-known", family=family)
    n = 10
    rng = np.random.default_rng(64)
    y, design = rng.normal(size=(n, 2)), rng.normal(size=(n, 2, 2))
    covariance = np.empty((2*n, 2*n))
    stationary = sigma / (1 - phi[:, None] * phi[None, :])
    for i in range(n):
        for j in range(n):
            block = np.diag(phi**abs(i-j)) @ stationary
            covariance[2*i:2*i+2, 2*j:2*j+2] = block if i >= j else block.T
    dense = DecisionBlock("T", y.ravel(), design.reshape(-1, 2),
                          np.linalg.cholesky(covariance), "read_only_inference")
    calls = []
    actual = FittedChannelNoise.whiten
    def counted(owner, values):
        calls.append(values.shape)
        return actual(owner, values)
    monkeypatch.setattr(FittedChannelNoise, "whiten", counted)
    record = compile_owned_decision(prior, hyper, noise, decision_id="T", role="read_only_inference",
        native_samples=n, raw_batches=[(0, y[:3], design[:3]), (3, y[3:], design[3:])],
        conditioning_samples=2, qr_chunk_rows=3)
    assert calls == [(n, 2, 3)]
    assert (record.observed_rows, record.prefix.observed_rows, record.tail.observed_rows) == (20, 4, 16)
    expected = compile_decision(prior, hyper, dense, conditioning_rows=4, qr_chunk_rows=5)
    u = rng.normal(size=(5, 2))
    model = prepare_model(prior, (), hyper)
    got = conditional_tail_components(prepare_prediction(model, record, conditioning_rows=4), u)
    want = conditional_tail_components(prepare_prediction(model, expected, conditioning_rows=4), u)
    np.testing.assert_allclose(got, want, atol=2e-13)
    for left, right in [(record, expected), (record.prefix, expected.prefix), (record.tail, expected.tail)]:
        assert left.noise_logdet == pytest.approx(right.noise_logdet, abs=2e-14)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="contiguous"):
        compile_owned_decision(prior, hyper, noise, decision_id="bad", role="target_calibration",
            native_samples=n, raw_batches=[(1, y, design)])


@pytest.mark.parametrize("nu", [4., np.inf])
def test_beta_collapsed_states_match_independent_dense_joint_conditionals(nu):
    prior, hyper, c, t = fixture(nu)
    record = compile_decision(prior, hyper, t, conditioning_rows=3, qr_chunk_rows=2)
    model = prepare_model(prior, [c, record.prefix], hyper)
    prediction = prepare_prediction(model, record, conditioning_rows=3, draw_batch_size=2)
    scales = np.array([[.35, 1.2], [2., .4], [1., 1.]]) if np.isfinite(nu) else np.ones((3, 2))
    got = conditional_beta_tail_components(model, prediction, scales)
    past = DecisionBlock("T", t.samples[:3], t.design[:3], t.noise_root[:3, :3], t.role)
    want = [dense_log(prior, hyper, [c, t], state) - dense_log(prior, hyper, [c, past], state)
            for state in scales]
    np.testing.assert_allclose(got, want, atol=2e-13)
    # Legacy dense prefix construction remains bound by its original input operands.
    dense_model = prepare_model(prior, [c, past], hyper)
    np.testing.assert_allclose(conditional_beta_tail_components(dense_model, prediction, scales), got, atol=2e-13)
    component = conditional_gaussian_components(prior, [c, record], hyper, scales)[0]
    np.testing.assert_allclose(component, [dense_log(prior, hyper, [c, t], s) for s in scales], atol=2e-13)
    wrong = replace(t, samples=t.samples + np.r_[.1, np.zeros(8)])
    with pytest.raises(ValueError, match="exactly once"):
        conditional_beta_tail_components(model, prepare_prediction(model, wrong, conditioning_rows=3), scales)


def test_explicit_beta_predict_keeps_every_draw_and_admits_binding_once(monkeypatch):
    prior, hyper, c, t = fixture(np.inf)
    record = compile_decision(prior, hyper, t, conditioning_rows=3)
    model = prepare_model(prior, [c, record.prefix], hyper)
    prediction = prepare_prediction(model, record, conditioning_rows=3, draw_batch_size=3)
    rng = np.random.default_rng(65)
    samples = JointResponseSamples(model, rng.normal(size=(2, 16, 2)), np.ones((2, 16, 2)),
                                   0, 16, (1, 2), 0.)
    import research.joint_response_sampling as sampling
    calls = []
    binding = sampling._targeted_prediction_binding
    def counted(*args):
        calls.append(1)
        return binding(*args)
    monkeypatch.setattr(sampling, "_targeted_prediction_binding", counted)
    result = targeted_conditional_beta_predict(samples, prediction, batch_size=4,
                                               max_rhat=10., min_bulk_ess=0., min_tail_ess=0.)
    assert len(calls) == 1 and result["actual_retained_joint_scale_draws"] == 32
    assert result["component_tail_logs"].shape == (32,)
    assert result["log_density_batch_mcse"] == 0.
    assert result["universal_variance_improvement_over_lambda_collapsed_t_not_claimed"]
    assert not result["continuous_Gamma_target_accuracy_certified"]


def test_compiled_finite_evidence_and_physical_moments_match_dense_mixture():
    prior, hyper, c, _ = fixture()
    rule = DiscreteGammaRule([.4, 1.7], [.3, .7], 4., 2)
    record = compile_decision(prior, hyper, c, qr_chunk_rows=3)
    result = joint_evidence(prior, [record], hyper, scale_rule=rule, batch_size=1)
    logs = np.array([dense_log(prior, hyper, [c], [state]) for state in rule.nodes])
    logs += np.log(rule.weights)
    mass = logsumexp(logs)
    weights = np.exp(logs-mass)
    covariances, means = [], []
    prior_covariance = prior.root @ prior.root.T
    for state in rule.nodes:
        covariance = hyper.relative_noise_shape * c.noise_root @ c.noise_root.T / state
        gain = np.linalg.solve(covariance+c.design @ prior_covariance @ c.design.T,
                               c.design @ prior_covariance).T
        means.append(prior.mean+gain @ (c.samples-c.design @ prior.mean))
        covariances.append(prior_covariance-gain @ c.design @ prior_covariance)
    means, covariances = np.asarray(means), np.asarray(covariances)
    mean = weights @ means
    covariance = np.einsum("b,bij->ij", weights, covariances)+((means-mean).T*weights) @ (means-mean)
    assert result.log_evidence == pytest.approx(mass, abs=2e-13)
    np.testing.assert_allclose(result.beta_mean, mean, atol=2e-13)
    np.testing.assert_allclose(result.beta_covariance, covariance, atol=2e-13)


def test_tail_carrier_and_pasted_prediction_are_not_admitted_as_independent_sources():
    prior, hyper, _, t = fixture()
    full = compile_decision(prior, hyper, t, conditioning_rows=3)
    other = compile_decision(prior, hyper, replace(t, samples=t.samples+1), conditioning_rows=3)
    model = prepare_model(prior, [full.prefix], hyper)
    prediction = prepare_prediction(model, full, conditioning_rows=3)
    with pytest.raises(ValueError, match="same factory"):
        replace(prediction, tail=other.tail)
    with pytest.raises(ValueError, match="independent Gamma"):
        prepare_model(prior, [full.tail], hyper)
    with pytest.raises(ValueError, match="same physical"):
        prepare_model(replace(prior, mean=prior.mean+.1), [full.prefix], hyper)
    with pytest.raises(TypeError, match="factory"):
        replace(full, source_identity=other.source_identity)
    # No-C is exactly the global proper prior plus the actual prefix scale.
    scales = np.array([[.5], [1.7]])
    got = conditional_beta_tail_components(model, prediction, scales)
    past = DecisionBlock("T", t.samples[:3], t.design[:3], t.noise_root[:3, :3], t.role)
    np.testing.assert_allclose(got, [dense_log(prior, hyper, [t], s)-dense_log(prior, hyper, [past], s)
                                   for s in scales], atol=2e-13)


def test_owned_raw_identity_precedes_chart_rounding_and_is_reader_partition_invariant():
    # Independent reviewer counterexample: transformed D[0,1e16] and D[1,1e16]
    # are equal in binary64 under this proper prior root; original D is not.
    prior = GlobalGaussianPrior([0., 0.], [[1., 0.], [1., 1.]], "external")
    hyper = FixedHyperparameters(4., 1., "externally_fixed")
    geometry = NoiseGeometry(("Cz",), 128., "V", "native", "software", 1)
    noise = FittedChannelNoise.from_parameters([[0.]], [[1.]], geometry=geometry,
        conditioning_samples=2, reference_id="software-known", family="shared")
    y, design = np.zeros((4, 1)), np.zeros((4, 1, 2))
    design[:, :, 1] = 1e16
    wrong = design.copy()
    wrong[0, 0, 0] = 1.
    np.testing.assert_array_equal(design @ prior.root, wrong @ prior.root)
    def compile_row(d, batches):
        return compile_owned_decision(prior, hyper, noise, decision_id="T", role="read_only_inference",
            native_samples=4, raw_batches=[(start, y[start:stop], d[start:stop]) for start, stop in batches],
            conditioning_samples=3, qr_chunk_rows=2)
    actual = compile_row(design, [(0, 4)])
    repartitioned = compile_row(design, [(0, 2), (2, 4)])
    different = compile_row(wrong, [(0, 4)])
    assert actual.source_identity == repartitioned.source_identity
    assert actual.prefix.raw_input_identity == repartitioned.prefix.raw_input_identity
    assert actual.source_identity != different.source_identity
    assert actual.prefix.raw_input_identity != different.prefix.raw_input_identity
    model = prepare_model(prior, [different.prefix], hyper)
    prediction = prepare_prediction(model, actual, conditioning_rows=3)
    with pytest.raises(ValueError, match="exactly once"):
        conditional_beta_tail_components(model, prediction, [[1.]])
