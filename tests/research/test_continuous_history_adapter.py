"""Actual saved producer seams and independent dense precision/quotient oracles."""
from itertools import product

import numpy as np
import pytest
from scipy.linalg import null_space
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from research.continuous_history_adapter import (
    ContinuousHistoryUnsupportedError,
    _stored_temporal_root,
    adapt_continuous_history,
)
from research.response_history_identification import PublicResponseHistory, PublicSelection
from tests.transfer.test_continuous_dictionary import inputs as canonical_calibration_records
from transfer.continuous_decision import ContinuousDecisionModel
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec
from transfer.temporal_noise import TemporalNoise


@pytest.fixture(scope="module")
def actual_model_package(tmp_path_factory):
    """One explicit synthetic C fit; every adapter consumer reads its saved state."""
    dictionary = ResponseDictionary.compile(ResponseDictionarySpec("gaussian", .5, 3), sample_rate_hz=64.)
    model = ContinuousDecisionModel.fit(canonical_calibration_records(), dictionary=dictionary, sfreq=64.,
        channels=("a", "b"), candidate_ids=("one", "two"), operator_identity="native-fixture",
        noise_order=1, conditioning_samples=32)
    path = tmp_path_factory.mktemp("actual-continuous-history")/"model.npz"
    model.save(path)
    return path


def loaded_fixture(package):
    return ContinuousDecisionModel.load(package)


def inputs(model):
    previous = PublicSelection("previous", ("p0", "p1"), np.array([90, 94]), np.eye(2))
    current = PublicSelection("current", model.candidate_ids, np.array([132, 140]), np.eye(2))
    history = PublicResponseHistory((previous, current), "current", np.array([[.4, .1], [.2, .3]]))
    n, origin = 72, 100
    design = manual_design(model.basis, history, (1, 0), origin, n)
    z = np.column_stack((np.ones(n), (2*np.arange(n)-31)/31))
    sigma_root = np.linalg.cholesky(model.innovation_covariance)
    samples = (design@model.kernel_mean + z@np.array([[.1, -.2], [.01, .02]])
               + .1*np.random.default_rng(8111).normal(size=(n, 2))@sigma_root.T)
    kwargs = dict(origin_sample=origin, available_sample=origin+n, sfreq=model.sfreq,
                  channels=model.channels, operator_identity=model.operator_identity, record_id="held-history-software")
    return history, samples, kwargs


def manual_design(basis, history, state, origin, n):
    base = max(0, origin-len(basis)+1)
    impulses = np.zeros((origin-base+n, 2))
    for selection, q in zip(history.selections, state, strict=True):
        for j, onset in enumerate(selection.onsets):
            i = int(onset)-base
            if 0 <= i < len(impulses):
                impulses[i, 0] += 1
                impulses[i, 1] += selection.membership[j, q]
    return np.column_stack([np.convolve(impulse, column)[origin-base:origin-base+n]
                            for impulse in impulses.T for column in basis.T])


def dense_oracle(model, history, samples, origin):
    """Invert full precision and use orthogonal quotient, never producer factor sums."""
    n, c = samples.shape
    f = model.noise.whiten(np.eye(n))
    noise = np.linalg.inv(f.T@f)
    r = model.posterior_state["precision_root"]
    h = np.linalg.inv(r.T@r)
    sigma = model.innovation_covariance
    u = model.conditioning_samples
    z = np.column_stack((np.ones(n), (2*np.arange(n)-(u-1))/(u-1)))
    states = list(product(*(range(len(s.candidate_ids)) for s in history.selections)))
    full, prefix = [], []
    for state in states:
        d = manual_design(model.basis, history, state, origin, n)
        covariance = noise+d@h@d.T
        residual = samples-d@model.kernel_mean
        for stop, output in ((n, full), (u, prefix)):
            complement = null_space(z[:stop].T)
            quotient = complement.T@residual[:stop]
            v = complement.T@covariance[:stop, :stop]@complement
            volume = -.5*c*np.linalg.slogdet(z[:stop].T@z[:stop])[1]
            output.append(multivariate_normal.logpdf(quotient.flatten(order="F"), cov=np.kron(sigma, v))+volume)
    logpi = np.log(history.joint_label_prior.reshape(-1))
    evidence = np.array([logsumexp(logpi[[q, q+2]]+np.asarray(full)[[q, q+2]])
                         -logsumexp(logpi[[q, q+2]]+np.asarray(prefix)[[q, q+2]]) for q in range(2)])
    probability = evidence+np.log(history.joint_label_prior.sum(axis=0))
    probability -= logsumexp(probability)
    return np.asarray(full), np.asarray(prefix), evidence, probability


def test_saved_actual_load_adapter_history_matches_dense_precision_quotient_without_refit(monkeypatch, actual_model_package):
    import transfer.continuous_decision as response_owner
    import transfer.temporal_noise as noise_owner

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only adapter rebuilt/refitted a frozen producer")

    monkeypatch.setattr(response_owner, "_kernel_posterior_factors", forbidden)
    monkeypatch.setattr(noise_owner, "_canonical_stationary", forbidden)
    monkeypatch.setattr(ContinuousDecisionModel, "fit", forbidden)
    monkeypatch.setattr(TemporalNoise, "fit", forbidden)
    model = loaded_fixture(actual_model_package)
    history, samples, kwargs = inputs(model)
    identity = (model.digest(), model.noise.identity_digest)
    adapted = adapt_continuous_history(model, history, samples, **kwargs)
    actual = adapted.evidence()
    full, prefix, evidence, probabilities = dense_oracle(model, history, samples, kwargs["origin_sample"])
    np.testing.assert_allclose(actual.state_joint_log_marginal, full, rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(actual.state_prefix_log_marginal, prefix, rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(actual.conditional_log_likelihood, evidence, rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(actual.log_candidate_probability, probabilities, rtol=3e-9, atol=3e-9)
    np.testing.assert_array_equal(adapted.posterior.basis, model.basis)
    np.testing.assert_array_equal(adapted.posterior.mean, model.kernel_mean)
    np.testing.assert_array_equal(adapted.posterior.precision_root, model.posterior_state["precision_root"])
    assert identity == (model.digest(), model.noise.identity_digest)
    assert adapted.binding["model_digest"] == identity[0]


def test_actual_stored_whitening_image_preserves_prefix_and_does_not_double_innovation_scale(actual_model_package):
    model = loaded_fixture(actual_model_package)
    history, samples, kwargs = inputs(model)
    adapted = adapt_continuous_history(model, history, samples, **kwargs)
    temporal_root = adapted.observation.temporal_root
    f = model.noise.whiten(np.eye(len(samples)))
    np.testing.assert_allclose(f@temporal_root, np.eye(len(samples)), rtol=0, atol=2e-14)
    np.testing.assert_allclose(f@samples, model.noise.whiten(samples), rtol=3e-14, atol=3e-14)
    u = model.conditioning_samples
    prefix_root = _stored_temporal_root(model.noise, u)
    np.testing.assert_allclose(temporal_root[:u, :u], prefix_root, rtol=0, atol=2e-14)
    np.testing.assert_allclose(adapted.posterior.spatial_root@adapted.posterior.spatial_root.T,
                               model.innovation_covariance, rtol=3e-14, atol=3e-14)
    assert abs(model.noise.innovation_variance-1) > 1e-3
    expected = np.linalg.inv(f.T@f)
    np.testing.assert_allclose(temporal_root@temporal_root.T, expected, rtol=3e-13, atol=3e-13)
    assert np.max(np.abs(temporal_root@temporal_root.T-model.noise.innovation_variance*expected)) > 1e-3


@pytest.mark.parametrize("mismatch", ["sfreq", "channels", "operator_identity", "candidate_order"])
def test_actual_physical_program_and_current_vocabulary_mismatch_is_rejected(mismatch, actual_model_package):
    model = loaded_fixture(actual_model_package)
    history, samples, kwargs = inputs(model)
    if mismatch == "candidate_order":
        old = history.selections[1]
        changed = PublicSelection(old.selection_id, old.candidate_ids[::-1], old.onsets, old.membership[:, ::-1])
        history = PublicResponseHistory((history.selections[0], changed), "current", history.joint_label_prior[:, ::-1])
    else:
        kwargs[mismatch] = {"sfreq": model.sfreq+1, "channels": model.channels[::-1],
                            "operator_identity": model.operator_identity+"-different-reference"}[mismatch]
    with pytest.raises(ValueError, match="differs"):
        adapt_continuous_history(model, history, samples, **kwargs)


def test_available_observation_and_actual_future_clock_are_distinct_rejections(actual_model_package):
    model = loaded_fixture(actual_model_package)
    history, samples, kwargs = inputs(model)
    with pytest.raises(ValueError, match="availability"):
        adapt_continuous_history(model, history, samples, **{**kwargs, "available_sample": 171})
    future = PublicSelection("future", ("f0", "f1"), np.array([172, 180]), np.eye(2))
    prior = np.repeat(history.joint_label_prior[:, :, None]/2, 2, axis=2)
    unavailable = PublicResponseHistory((*history.selections, future), "current", prior)
    with pytest.raises(ContinuousHistoryUnsupportedError) as caught:
        adapt_continuous_history(model, unavailable, samples, **kwargs)
    assert caught.value.record()["reason"] == "future_actual_public_event_clock_is_not_available"


def test_calibration_record_id_cannot_be_reused_for_history_observations(actual_model_package):
    model = loaded_fixture(actual_model_package)
    history, samples, kwargs = inputs(model)
    kwargs["record_id"] = model.provenance["records"][0]["record_id"]
    with pytest.raises(ValueError, match="distinct held"):
        adapt_continuous_history(model, history, samples, **kwargs)


def test_actual_conditional_prefix_noise_is_typed_unsupported_instead_of_restarting_noise():
    noise = TemporalNoise.from_parameters([.2], innovation_variance=1., reference_id="rectangular-software",
                                          startup_mode="conditional_prefix")
    assert noise.whiten(np.eye(8)).shape == (7, 8)
    with pytest.raises(ContinuousHistoryUnsupportedError) as caught:
        _stored_temporal_root(noise, 8)
    assert caught.value.reason == "rectangular_conditional_prefix_has_no_joint_stationary_covariance"


def test_no_neighbour_adapter_matches_actual_existing_conditional_score_program(actual_model_package):
    model = loaded_fixture(actual_model_package)
    base, _, kwargs = inputs(model)
    current = base.selections[1]
    history = PublicResponseHistory((current,), "current", np.array([.5, .5]))
    n = 72
    design = manual_design(model.basis, history, (0,), kwargs["origin_sample"], n)
    nuisance = np.column_stack((np.ones(n), (2*np.arange(n)-31)/31))
    samples = (design@model.kernel_mean+nuisance@np.array([[.2, -.1], [.3, -.2]])
               + .1*np.random.default_rng(8222).normal(size=(n, 2))@np.linalg.cholesky(model.innovation_covariance).T)
    adapted = adapt_continuous_history(model, history, samples, **kwargs)
    actual = adapted.evidence()
    original = model.score(samples, current.onsets-kwargs["origin_sample"], current.membership,
        sfreq=model.sfreq, channels=model.channels, operator_identity=model.operator_identity,
        candidate_ids=model.candidate_ids, record_id=kwargs["record_id"])
    np.testing.assert_allclose(actual.conditional_log_likelihood, original.log_evidence, rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(actual.log_candidate_probability, original.log_probability, rtol=3e-9, atol=3e-9)
