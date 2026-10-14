"""Independent dense Gaussian oracle and original eight-channel vertical path."""
import hashlib
import io
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from scipy.linalg import block_diag, solve_triangular
from scipy.special import logsumexp

from data.contract import DEFAULT_P300_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from research.channel_noise_mechanism import (
    NoiseGeometry,
    PhysicalCoefficientPrior,
    StructuredGaussianMarginalPair,
    StructuredGaussianMarginalProgram,
    _StoredTemporalNoiseAdapter,
    _structured_log_jacobian_error,
)
from research.contracts import semantic_sha256
from research.history_artifact import (
    finite_c_candidate_evidence,
    predict_finite_c_history,
    replay_finite_c_history_capsule,
    save_finite_c_history_capsule,
)
from research.latent_response_reference import ResponseState
from research.multichannel_c_latent import FiniteCHistoryModel, _design, _nuisance, _program
from research.response_history_identification import finite_conditional_mixture
from tests.support.bnci import bnci_mat_fields
from transfer.ensemble_views import (
    CalibrationPrefixCompiler,
    _bnci_prediction_view,
    calibration_prefix_identity,
)
from transfer.noise_model import array_digest
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec
from transfer.temporal_noise import TemporalNoise


def _dense_marginal(y, design, nuisance, temporal, covariance, prior_mean, prior_factor):
    """Independent raw covariance + flat Lebesgue integral, for small matrices."""
    channels = y.shape[1]
    whitening = temporal.whiten(np.eye(len(y)))
    time_root = np.linalg.solve(whitening, np.eye(len(y)))
    covariance_noise = np.kron(time_root@time_root.T, covariance)
    mapping = np.kron(design, np.eye(channels))
    v = covariance_noise+mapping@prior_factor@prior_factor.T@mapping.T
    z, residual = nuisance.reshape(y.size, -1), y.ravel()-mapping@prior_mean.ravel()
    vi = np.linalg.inv(v)
    zz = z.T@vi@z
    projected = vi-vi@z@np.linalg.solve(zz, z.T@vi)
    result = -.5*(residual@projected@residual+np.linalg.slogdet(v)[1]
                  +np.linalg.slogdet(zz)[1]+(y.size-z.shape[1])*np.log(2*np.pi))
    joint = np.column_stack((mapping, z))
    precision_prior = np.zeros((joint.shape[1],)*2)
    precision_prior[:prior_factor.shape[0], :prior_factor.shape[0]] = np.linalg.inv(prior_factor@prior_factor.T)
    rhs_prior = precision_prior@np.r_[prior_mean.ravel(), np.zeros(z.shape[1])]
    precision = precision_prior+joint.T@np.linalg.solve(covariance_noise, joint)
    joint_covariance = np.linalg.inv(precision)
    joint_mean = joint_covariance@(rhs_prior+joint.T@np.linalg.solve(covariance_noise, y.ravel()))
    return float(result), joint_mean, joint_covariance


def _small_program():
    rng = np.random.default_rng(131401)
    noise = TemporalNoise.from_reflections([-.3], innovation_variance=1., reference_id="fixed-small-law")
    covariance = np.array([[1., .25], [.25, 1.8]])
    y = rng.normal(size=(12, 2))
    design = rng.normal(size=(12, 2))
    mean = np.array([[.2, -.1], [.35, .15]])
    factor = np.array([[1., .2, .1, 0.], [0., .8, .1, .1], [0., 0., 1.1, .2], [0., 0., 0., .7]])
    z = _nuisance(12, 2, 2)
    return noise, covariance, y, design, mean, factor, z


def test_dense_multichannel_joint_c_posterior_and_full_prefix_continuation():
    noise, covariance, y, design, mean, factor, z = _small_program()
    cut, conditioning = 6, 9
    c = StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise, covariance, 2),
        y[:cut], mean, factor, z[:cut], intercept_chart=True)
    state = c.posterior_state(design[:cut])
    cm, expected_mean, expected_covariance = _dense_marginal(y[:cut], design[:cut], z[:cut],
        noise, covariance, mean, factor)
    chart = block_diag(factor, state.flat_transform)
    translation = np.r_[mean.ravel(), y[0], np.zeros(2)]
    physical_mean = translation+chart@state.mean
    physical_factor = chart@solve_triangular(state.precision_root, np.eye(len(state.mean)))
    np.testing.assert_allclose(state.log_marginal, cm, rtol=0, atol=5e-12)
    np.testing.assert_allclose(physical_mean, expected_mean, rtol=0, atol=5e-12)
    np.testing.assert_allclose(physical_factor@physical_factor.T, expected_covariance, rtol=0, atol=5e-12)
    full = StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise, covariance, 2),
        y, mean, factor, z, intercept_chart=True, flat_coordinate_transform=state.flat_transform)
    pair = full.continue_posterior(design, state, conditioning_samples=conditioning)
    fm, _, _ = _dense_marginal(y, design, z, noise, covariance, mean, factor)
    pm, _, _ = _dense_marginal(y[:conditioning], design[:conditioning], z[:conditioning],
                             noise, covariance, mean, factor)
    np.testing.assert_allclose([pair.joint_log_marginal, pair.prefix_log_marginal, pair.conditional_log_density],
                              [fm-cm, pm-cm, fm-pm], rtol=0, atol=1e-11)
    assert pair.resolution["C_rows_reused_in_update"] == pair.resolution["new_flat_dimensions"] == 0
    # The physical shared beta remains correlated across distinct event rows.
    beta_cov = expected_covariance[:4, :4]
    left, right = np.kron(design[6:7], np.eye(2)), np.kron(design[10:11], np.eye(2))
    assert np.linalg.norm(left@beta_cov@right.T) > .01


def test_physical_basis_channel_unit_and_flat_coordinate_relations():
    noise, covariance, y, design, mean, factor, z = _small_program()
    baseline = _dense_marginal(y, design, z, noise, covariance, mean, factor)[0]
    basis = np.array([[1.2, .25], [-.15, .9]])
    chart = np.kron(basis, np.eye(2))
    changed_mean = np.linalg.solve(basis, mean)
    changed_factor = np.linalg.solve(chart, factor)
    changed = StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise, covariance, 2),
        y, changed_mean, changed_factor, z, intercept_chart=True).posterior_state(design@basis)
    np.testing.assert_allclose(changed.log_marginal, baseline, rtol=0, atol=6e-12)
    permutation = [1, 0]
    pchart = np.kron(np.eye(2), np.eye(2)[permutation])
    reordered = StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise,
        covariance[np.ix_(permutation, permutation)], 2), y[:, permutation], mean[:, permutation],
        pchart@factor, _nuisance(12, 2, 2), intercept_chart=True).posterior_state(design)
    np.testing.assert_allclose(reordered.log_marginal, baseline, rtol=0, atol=6e-12)
    scale = 7.
    scaled = StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise, covariance*scale**2, 2),
        y*scale, mean*scale, factor*scale, z, intercept_chart=True).posterior_state(design)
    np.testing.assert_allclose(scaled.log_marginal, baseline-(y.size-z.shape[2])*np.log(scale), rtol=0, atol=7e-12)
    flat_change = np.diag([2., 3., 4., 5.])
    alternate = StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise, covariance, 2),
        y, mean, factor, z@flat_change, intercept_chart=True).posterior_state(design)
    np.testing.assert_allclose(alternate.log_marginal, baseline-np.log(np.linalg.det(flat_change)), rtol=0, atol=6e-12)


def test_amplitude_global_beta_gauge_requires_the_same_physical_prior_transform():
    noise, covariance, y, design, mean, factor, z = _small_program()
    baseline = StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise, covariance, 2),
        y, mean, factor, z, intercept_chart=True).posterior_state(design)
    # a -> 2a and beta -> beta/2 has identical signal law only when the full
    # physical beta prior changes with the chart, including its uncertainty.
    same_law = StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise, covariance, 2),
        y, mean/2, factor/2, z, intercept_chart=True).posterior_state(design*2)
    np.testing.assert_allclose(same_law.log_marginal, baseline.log_marginal, rtol=0, atol=3e-12)
    different_prior = StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise, covariance, 2),
        y, mean, factor, z, intercept_chart=True).posterior_state(design*2)
    assert abs(different_prior.log_marginal-baseline.log_marginal) > .1


def test_sparse_logmass_keeps_tiny_states_and_prefix_updates_before_conditional():
    # Exp(-1000) is binary64 zero, but its likelihood offsets make this state
    # carry visible posterior mass. It must be evaluated and retained in logs.
    pairs = [StructuredGaussianMarginalPair(p+c, p, c, None, np.empty(0), np.empty(0)) for p, c in
             [(0., -4.), (1000., -.1), (0., -2.), (1000., -3.)]]
    result = finite_conditional_mixture(("a", "b"), ((0,), (1,), (2,), (3,)), [0, 0, 1, 1],
        [-np.log(2), -1000-np.log(2), -np.log(2), -1000-np.log(2)], pairs, tail_scalar_rows=3)
    assert len(result.state_indices) == 4
    assert np.all(np.isfinite(result.state_prefix_log_weights))
    assert result.current_candidate_indices.dtype == np.dtype(np.int64)
    joint_log_weights = (result.state_full_log_weights
                        +result.log_whole_observed_current_probability[result.current_candidate_indices])
    np.testing.assert_allclose(logsumexp(joint_log_weights), 0., rtol=0, atol=3e-15)
    with pytest.raises(ValueError):
        result.current_candidate_indices.setflags(write=True)
    np.testing.assert_allclose(np.exp(result.state_prefix_log_weights), .5, rtol=0, atol=4e-14)
    expected = np.array([logsumexp([np.log(.5)-4, np.log(.5)-.1]),
                         logsumexp([np.log(.5)-2, np.log(.5)-3])])
    np.testing.assert_allclose(result.conditional_log_likelihood, expected, rtol=0, atol=2e-13)
    wrong = np.array([-4., -2.])
    assert np.max(np.abs(wrong-result.conditional_log_likelihood)) > 2.
    permuted = finite_conditional_mixture(("b", "a"), ((0,), (1,), (2,), (3,)), [1, 1, 0, 0],
        [-np.log(2), -1000-np.log(2), -np.log(2), -1000-np.log(2)], pairs, tail_scalar_rows=3)
    np.testing.assert_array_equal(permuted.log_candidate_probability, result.log_candidate_probability[::-1])


def test_c_logmass_error_is_propagated_to_complete_mixture_resolution():
    from research.response_history_identification import HistoryMixtureResolutionError

    certificate = {"full": {"absolute_log_density_error_estimate": 0.},
        "prefix": {"absolute_log_density_error_estimate": 0.},
        "conditional": {"absolute_log_density_error_estimate": 0., "actual_tail_scalar_rows": 3},
        "source_sha256": "a"*64}
    pairs = [StructuredGaussianMarginalPair(0., 0., 0., certificate, np.empty(0), np.empty(0)) for _ in range(4)]
    arguments = (("a", "b"), ((0,), (1,), (2,), (3,)), [0, 0, 1, 1], [-np.log(4)]*4, pairs)
    accepted = finite_conditional_mixture(*arguments, tail_scalar_rows=3, c_logmass_errors=[0.]*4)
    assert accepted.mixture_resolution["normalized_log_probability_error_estimate"] < 1e-12
    with pytest.raises(HistoryMixtureResolutionError, match="mixture error estimate"):
        finite_conditional_mixture(*arguments, tail_scalar_rows=3, c_logmass_errors=[1e-5, 0., 1e-5, 0.])


def test_two_declared_q_laws_match_dense_full_ratio_and_no_current_prefix_degeneracy():
    noise, covariance, y, design, mean, factor, z = _small_program()
    cut, conditioning = 6, 9
    c_program = StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise, covariance, 2),
        y[:cut], mean, factor, z[:cut], intercept_chart=True)
    state = c_program.posterior_state(design[:cut])
    program = StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise, covariance, 2),
        y, mean, factor, z, intercept_chart=True, flat_coordinate_transform=state.flat_transform)
    pairs, marginals = [], []
    for q in range(2):
        actual = design.copy()
        actual[cut:, 1] *= 1+q
        pairs.append(program.continue_posterior(actual, state, conditioning_samples=conditioning))
        full = _dense_marginal(y, actual, z, noise, covariance, mean, factor)[0]
        prefix = _dense_marginal(y[:conditioning], actual[:conditioning], z[:conditioning], noise,
                                  covariance, mean, factor)[0]
        marginals.append((full, prefix))
    arguments = (("a", "b"), ((0,), (1,)), [0, 1], [-np.log(2)]*2, pairs)
    full = finite_conditional_mixture(*arguments, tail_scalar_rows=6, scoring_law="whole_observed_current_q")
    conditional = finite_conditional_mixture(*arguments, tail_scalar_rows=6)
    full_reference = np.asarray([m[0] for m in marginals])
    full_reference -= logsumexp(full_reference)
    conditional_reference = np.asarray([m[0]-m[1] for m in marginals])
    conditional_reference -= logsumexp(conditional_reference)
    np.testing.assert_allclose(full.log_candidate_probability, full_reference, rtol=0, atol=1e-11)
    np.testing.assert_allclose(conditional.log_candidate_probability, conditional_reference, rtol=0, atol=1e-11)
    assert np.max(np.abs(full.log_candidate_probability-conditional.log_candidate_probability)) > .01
    # When half-open prefix precedes the first current ERP row, its q evidence
    # is identical and the complete-current and conditional-tail laws agree.
    no_prefix = [StructuredGaussianMarginalPair(c, 0., c, None, np.empty(0), np.empty(0)) for c in (-1., -3.)]
    f = finite_conditional_mixture(("a", "b"), ((0,), (1,)), [0, 1], [-np.log(2)]*2, no_prefix,
        tail_scalar_rows=6, scoring_law="whole_observed_current_q")
    c = finite_conditional_mixture(("a", "b"), ((0,), (1,)), [0, 1], [-np.log(2)]*2, no_prefix, tail_scalar_rows=6)
    np.testing.assert_array_equal(f.log_candidate_probability, c.log_candidate_probability)


def test_same_c_chart_is_required_and_prefix_must_identify_flat_measure():
    noise, covariance, y, design, mean, factor, z = _small_program()
    state = StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise, covariance, 2),
        y[:6], mean, factor, z[:6], intercept_chart=True).posterior_state(design[:6])
    program = StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise, covariance, 2),
        y, mean, factor, z, intercept_chart=True, flat_coordinate_transform=state.flat_transform)
    changed = design.copy()
    changed[1, 0] += .01
    with pytest.raises(ValueError, match="actual C operands"):
        program.continue_posterior(changed, state, conditioning_samples=9)
    with pytest.raises(ValueError, match="leave an observed"):
        program.continue_posterior(design, state, conditioning_samples=12)
    changed_noise = StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise, 2*covariance, 2),
        y, mean, factor, z, intercept_chart=True, flat_coordinate_transform=state.flat_transform)
    with pytest.raises(ValueError, match="actual C operands"):
        changed_noise.continue_posterior(design, state, conditioning_samples=9)
    nonupper = state.flat_transform.copy()
    nonupper[1, 0] = .01
    with pytest.raises(ValueError, match="saved upper flat chart"):
        StructuredGaussianMarginalProgram(_StoredTemporalNoiseAdapter(noise, covariance, 2),
            y, mean, factor, z, intercept_chart=True, flat_coordinate_transform=nonupper)


def test_long_normalizer_uses_actual_small_reductions_and_linear_sample_magnitude():
    from decimal import Decimal, localcontext

    noise, covariance, _, _, _, _, _ = _small_program()
    adapter = _StoredTemporalNoiseAdapter(noise, covariance, 2)
    errors = [_structured_log_jacobian_error(adapter, n) for n in (10000, 20000)]
    assert errors[1] < 2.001*errors[0]
    with localcontext() as context:
        context.prec = 80
        startup = sum(Decimal.from_float(float(s)).ln() for s in noise._startup_scales)
        spatial = sum(Decimal.from_float(float(s)).ln() for s in np.diag(adapter._spatial_root))
        for n, error in zip((10000, 20000), errors, strict=True):
            expected = 2*startup-n*spatial
            actual = Decimal.from_float(adapter.whitening_logabsdet(n))
            assert float(abs(actual-expected)) <= error


@pytest.fixture(scope="module")
def actual_native_c(tmp_path_factory):
    reused = os.environ.get("C_LATENT_REUSE_RECEIPT")
    if reused:
        receipt = json.loads(Path(reused).read_text(encoding="utf-8"))
        payload = Path(receipt["source_path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == receipt["source_sha256"]
        model_path = Path(receipt["model_path"])
        model = FiniteCHistoryModel.load(model_path, expected_sha256=receipt["model_sha256"])
        compiler = CalibrationPrefixCompiler(payload, source_sha256=receipt["source_sha256"],
            source_relative_path="A01.mat", budgets=(119,), profiles=(),
            authorization_preprocessing=preprocessing_spec_from_contract(DEFAULT_P300_DATA_CONTRACT),
            response_seconds=.4, pre_roll_seconds=.2)
        prefix = compiler.compile(119)
        rows = np.flatnonzero(compiler._record.selection_index == 1)
        start = int(compiler._record.flash_sample[rows[0]])-51
        stop = int(compiler._record.flash_sample[rows[-1]])+102
        view = _bnci_prediction_view(compiler._record, rows, start, stop, available_sample=stop)
        return compiler, prefix, view, model, model_path, receipt["model_sha256"], json.loads(model.protocol.plan_bytes)
    fields = bnci_mat_fields()
    # Preserve all strict original dimensions, 35 schedules, 64 sample clocks
    # and 32 sample flashes. Populate only the needed actual observed prefix.
    fields["X"][:21000] = np.random.default_rng(141310).normal(0, 2, (21000, 8)).astype(np.float32)
    stream = io.BytesIO()
    savemat(stream, {"data": fields}, do_compression=True, long_field_names=True)
    payload = stream.getvalue()
    compiler = CalibrationPrefixCompiler(payload, source_sha256=hashlib.sha256(payload).hexdigest(),
        source_relative_path="A01.mat", budgets=(119,), profiles=(),
        authorization_preprocessing=preprocessing_spec_from_contract(DEFAULT_P300_DATA_CONTRACT),
        response_seconds=.4, pre_roll_seconds=.2)
    prefix = compiler.compile(119)
    dictionary = ResponseDictionary.compile(ResponseDictionarySpec("gaussian", .4, 1), sample_rate_hz=256.)
    target = prefix.target_contract
    geometry = NoiseGeometry(tuple(target["channel_names"]), 256., "V", target["reference"],
                             "native_unfiltered_volts/1", 8)
    precision = (np.eye(16)+.12*np.diag(np.ones(15), 1)+.05*np.diag(np.ones(15), -1))/2e-6
    prior = PhysicalCoefficientPrior(np.full((2, 8), .4e-6), precision, dictionary.matrix,
        geometry, "predeclared-physical-prior", {"role": "analytic_reference", "source_sha256": "a"*64})
    noise = TemporalNoise.from_reflections([-.15], innovation_variance=1., reference_id="predeclared-native-shape")
    covariance = 4e-12*(.8*np.eye(8)+.2*np.ones((8, 8)))
    states = (ResponseState(1., .6, 0., 1., .5), ResponseState(1., 1.4, 0., 1., .5))
    plan = {"schema": "n2p3_finite_c_latent_protocol/1", "freeze_role": "predeclared_protocol",
        "parameter_role": "fixed_external_numeric_parameters", "calibration_prefix_digest": calibration_prefix_identity(prefix),
        "dictionary_digest": dictionary.digest(), "physical_prior_digest": semantic_sha256(prior.record()),
        "temporal_noise_digest": noise.identity_digest, "innovation_covariance_sha256": array_digest(covariance),
        "conditioning_samples": 64, "response_states": [s.__dict__ for s in states],
        "scoring_laws": ["fixed_current_prior_conditional_tail", "whole_observed_current_q"],
        "default_scoring_law": "fixed_current_prior_conditional_tail",
        "maximum_calibration_components": 10000, "maximum_prediction_states": 10000}
    model = FiniteCHistoryModel.fit(compiler.c_latent_fit_scope(prefix, dictionary=dictionary, prior=prior,
        noise=noise, plan=json.dumps(plan).encode(), innovation_covariance=covariance))
    path = tmp_path_factory.mktemp("actual-finite-C")/"model.npz"
    identity = model.save(path)
    model = FiniteCHistoryModel.load(path, expected_sha256=identity)
    rows = np.flatnonzero(compiler._record.selection_index == 1)
    start = int(compiler._record.flash_sample[rows[0]])-51
    stop = int(compiler._record.flash_sample[rows[-1]])+102
    view = _bnci_prediction_view(compiler._record, rows, start, stop, available_sample=stop)
    return compiler, prefix, view, model, path, identity, plan


def test_actual_partial_one_label_preserves_six_legal_targets_without_fit(actual_native_c):
    compiler, _, _, model, _, _, plan = actual_native_c
    early = compiler.compile(1)
    plan = {**plan, "calibration_prefix_digest": calibration_prefix_identity(early)}
    scope = compiler.c_latent_fit_scope(early, dictionary=model.dictionary, prior=model.prior,
        noise=model.noise, plan=json.dumps(plan).encode(), innovation_covariance=model.innovation_covariance)
    assert len(scope.allowed_candidates[scope.selections[0].selection_id]) == 6
    assert scope.binding["reachable_positive_components"] == 12
    assert scope.binding["fit_label_event_indices"] == (0, 1)


def test_actual_native_partial_c_producer_scopes_and_saved_component_weights(actual_native_c):
    compiler, prefix, view, model, _, _, _ = actual_native_c
    assert prefix.budget == 119 and prefix.raw.right_censored
    assert len(model.prior.geometry.channels) == 8 and len(model.components) == 2
    assert model.binding["fit_authority"]["fit_label_event_indices"] == (0, 119)
    assert model.binding["unique_Gaussian_posteriors"] == 2
    terms = np.array([c["log_prior"]+model.posteriors[c["posterior_index"]].log_marginal for c in model.components])
    np.testing.assert_array_equal(terms-logsumexp(terms), [c["log_posterior_weight"] for c in model.components])
    capability = compiler.c_latent_history_scope(view, prefix=prefix, model=model)
    assert capability.binding["fit_read_support"] == (0, model.calibration_samples)
    assert capability.binding["conditioning_read_support"] == (model.calibration_samples, view.raw_scope[0]+64)
    assert capability.binding["score_support"] == (view.raw_scope[0]+64, view.raw_scope[1])
    assert capability.binding["current_ERP_can_affect_prefix"] is True
    assert capability.conditioning_samples == int(capability.selections[-1].onsets[0])+13
    assert len(capability.selections[0].onsets) == 120  # C119 selection continuation uses the same state.
    assert capability.binding["future_label_access"] is False


def test_actual_single_component_precision_breakdown_and_saved_prior_image(actual_native_c, monkeypatch):
    compiler, prefix, view, model, _, _, _ = actual_native_c
    capability = compiler.c_latent_history_scope(view, prefix=prefix, model=model)
    def forbidden(*args, **kwargs):
        raise AssertionError("continuation recomputed the actual producer prior chart")
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    primitive = _program(model.prior, model.dictionary, model.noise, model.innovation_covariance,
        capability.samples, model.protocol.record["conditioning_samples"],
        transform=model.posteriors[0].flat_transform, factor=model.prior_factor,
        factor_source_error=model.prior_factor_source_error)
    c = model.components[0]
    c_selection, current = capability.selections
    design = _design(model.dictionary, (c_selection, current), (*c["target_indices"], 0),
        (model.protocol.states[c["response_state_indices"][0]], model.protocol.states[0]),
        observed_samples=len(capability.samples))
    pair = primitive.continue_posterior(design, model.posteriors[c["posterior_index"]],
        conditioning_samples=capability.conditioning_samples)
    numerical = {"C": model.posteriors[c["posterior_index"]].resolution["C"], **pair.resolution}
    print(json.dumps(numerical, sort_keys=True))
    c_error = numerical["C"]["absolute_log_density_error_estimate"]
    prefix_error = numerical["prefix"]["absolute_log_density_error_estimate"]
    conditional_error = numerical["conditional"]["absolute_log_density_error_estimate"]
    rows = (len(capability.samples)-capability.conditioning_samples)*8
    assert 2*(2*c_error+2*prefix_error+conditional_error) < rows*np.sqrt(np.finfo(float).eps)


@pytest.mark.parametrize("mutation", ["reference", "sample_rate_hz", "source_sha256", "available_sample", "samples"])
def test_actual_authority_rejects_wrong_native_reference_clock_source_cut_and_signal(actual_native_c, mutation):
    compiler, prefix, view, model, _, _, _ = actual_native_c
    changes = {"reference": "left earlobe", "sample_rate_hz": 255., "source_sha256": "f"*64,
               "available_sample": view.available_sample+1, "samples": view.samples+1e-9}
    with pytest.raises(ValueError):
        compiler.c_latent_history_scope(replace(view, **{mutation: changes[mutation]}), prefix=prefix, model=model)


def test_actual_capsule_model_only_replay_and_held_truth_isolation(actual_native_c, tmp_path, monkeypatch):
    compiler, prefix, view, model, path, identity, _ = actual_native_c
    before = path.read_bytes()
    original = predict_finite_c_history(model, compiler, prefix, view)
    assert original["status"] == "complete", original.get("failure")
    assert original["prediction"]["evaluated_positive_states"] == 144
    assert original["prediction"]["AR_restarts_at_selection_boundaries"] == 0
    assert original["prediction"]["C_refits"] == original["prediction"]["new_prior_applications"] == 0
    assert np.ptp(original["prediction"]["log_prefix_evidence_by_current"]) > 1e-5
    expected_full = np.asarray(original["prediction"]["log_evidence"])+np.asarray(
        original["prediction"]["log_prefix_evidence_by_current"])
    expected_full -= logsumexp(expected_full)
    np.testing.assert_allclose(original["prediction"]["log_whole_observed_current_probability"],
                              expected_full, rtol=0, atol=2e-10)
    record = compiler._record
    object.__setattr__(record, "target_row", np.zeros_like(record.target_row))
    object.__setattr__(record, "target_col", np.zeros_like(record.target_col))
    # None of these decoder held-truth mirrors belongs to fit/predict inputs.
    object.__setattr__(record, "raw_is_target", np.zeros_like(record.raw_is_target))
    attacked = predict_finite_c_history(model, compiler, prefix, view)
    assert semantic_sha256(original) == semantic_sha256(attacked)
    def forbidden(*args, **kwargs):
        raise AssertionError("model-only replay refitted a C posterior or noise")
    monkeypatch.setattr(FiniteCHistoryModel, "fit", forbidden)
    monkeypatch.setattr(StructuredGaussianMarginalProgram, "posterior_state", forbidden)
    monkeypatch.setattr(TemporalNoise, "fit", forbidden)
    capsule = save_finite_c_history_capsule(tmp_path/"capsule", model_path=path, model_sha256=identity,
        compiler=compiler, prefix=prefix, view=view, scoring_law="whole_observed_current_q")
    replay = replay_finite_c_history_capsule(tmp_path/"capsule", **capsule)
    assert replay["status"] == "complete"
    assert replay["prediction"]["scoring_law"] == "whole_observed_current_q"
    np.testing.assert_array_equal(replay["prediction"]["log_probability"],
                                  replay["prediction"]["log_whole_observed_current_probability"])
    assert np.max(np.abs(np.asarray(original["prediction"]["log_probability"])
                         -np.asarray(replay["prediction"]["log_probability"]))) > 1e-5
    assert path.read_bytes() == before == (tmp_path/"capsule/finite_c_model.npz").read_bytes()
    evidence = finite_c_candidate_evidence(model, compiler, prefix, view)
    assert evidence.probability_available and len(evidence.probability) == 36


def test_actual_complete_schedule_right_censored_tail_and_saved_replay(actual_native_c, tmp_path, monkeypatch):
    compiler, prefix, view, model, path, identity, _ = actual_native_c
    stop = int(view.onsets[-1])+view.raw_scope[0]+32
    rows = np.flatnonzero(compiler._record.selection_index == 1)
    censored = _bnci_prediction_view(compiler._record, rows, view.raw_scope[0], stop, available_sample=stop)
    scope = compiler.c_latent_history_scope(censored, prefix=prefix, model=model)
    assert scope.binding["current_schedule_complete"] is True
    assert scope.binding["current_response_right_censored"] is True
    assert scope.binding["full_actual_event_ledger"][-1]["retained_read_support"][1] == stop
    def forbidden(*args, **kwargs):
        raise AssertionError("right-censored replay fitted C again")
    monkeypatch.setattr(FiniteCHistoryModel, "fit", forbidden)
    monkeypatch.setattr(StructuredGaussianMarginalProgram, "posterior_state", forbidden)
    capsule = save_finite_c_history_capsule(tmp_path/"censored", model_path=path, model_sha256=identity,
        compiler=compiler, prefix=prefix, view=censored)
    actual = replay_finite_c_history_capsule(tmp_path/"censored", **capsule)
    assert actual["status"] == "complete", actual.get("failure")
    assert actual["prediction"]["current_response_right_censored"] is True
    assert actual["prediction"]["evaluated_positive_states"] == 144


def test_nominal_single_state_uses_same_complete_context_program(actual_native_c):
    compiler, prefix, view, model, _, _, plan = actual_native_c
    plan = {**plan, "response_states": [ResponseState(1., 1., 0., 1., 1.).__dict__]}
    nominal = FiniteCHistoryModel.fit(compiler.c_latent_fit_scope(prefix, dictionary=model.dictionary, prior=model.prior,
        noise=model.noise, plan=json.dumps(plan).encode(), innovation_covariance=model.innovation_covariance))
    scope = compiler.c_latent_history_scope(view, prefix=prefix, model=nominal)
    assert scope.binding["full_read_support"] == (0, view.available_sample)
    assert len(nominal.components) == 1
    prediction = predict_finite_c_history(nominal, compiler, prefix, view)
    assert prediction["status"] == "complete", prediction.get("failure")
    assert prediction["prediction"]["evaluated_positive_states"] == 36
    assert prediction["prediction"]["new_prior_applications"] == 0
