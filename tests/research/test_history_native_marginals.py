"""Independent dense oracles for stored-noise augmented-QR history marginals."""
from dataclasses import replace

import numpy as np
import pytest

from research.history_likelihood import NativeHistoryMarginals
from research.response_history_identification import (
    NeighborEventPrefix,
    PublicResponseHistory,
    PublicSelection,
    ResponseObservation,
    ResponsePosterior,
    StructuredResponseObservation,
    _log_marginal,
    conditional_history_evidence,
    response_history_design,
)
from transfer.continuous_decision import CalibrationRecord, ContinuousDecisionModel
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec


@pytest.fixture(scope="module")
def actual_model(tmp_path_factory):
    rng = np.random.default_rng(76510)
    records = tuple(CalibrationRecord(f"synthetic-C-{i}", rng.normal(size=(80, 2)),
        np.array([10, 22, 38, 51]), np.array([False, True, False, True])) for i in range(2))
    dictionary = ResponseDictionary.compile(ResponseDictionarySpec("gaussian", .2, 2), sample_rate_hz=64.)
    model = ContinuousDecisionModel.fit(records, sfreq=64., channels=("a", "b"),
        candidate_ids=("q0", "q1"), operator_identity="native-fixture", dictionary=dictionary,
        noise_order=1, conditioning_samples=16)
    path = tmp_path_factory.mktemp("history-actual-producer")/"model.npz"
    model.save(path)
    return ContinuousDecisionModel.load(path)


def operands(model):
    current = PublicSelection("current", model.candidate_ids, np.array([50, 55]), np.eye(2))
    neighbour = NeighborEventPrefix("previous", ("p0", "p1", "p2"), np.array([30]), [[1, 0, 0]])
    history = PublicResponseHistory((neighbour, current), "current", np.array([[.1, .2], [.2, .1], [.3, .1]]))
    n, u = 30, model.conditioning_samples
    z = np.column_stack((np.ones(n), (2*np.arange(n)-(u-1))/(u-1)))
    y = np.random.default_rng(77610).normal(size=(n, 2))
    posterior = ResponsePosterior(model.basis, model._predictive_kernel_mean, model._posterior_precision_root,
        np.linalg.cholesky(model.innovation_covariance), model.sfreq)
    observed = StructuredResponseObservation(y, 35, z, u, True)
    return posterior, history, observed


def test_native_full_prefix_marginals_match_dense_factor_oracle_and_keep_flat_chart(actual_model):
    model = actual_model
    posterior, history, observation = operands(model)
    f = model.noise.whiten(np.eye(len(observation.samples)))
    root = np.linalg.inv(f)
    dense = ResponseObservation(observation.samples, observation.origin_sample, root, observation.nuisance,
        observation.conditioning_samples, True)
    primitive = NativeHistoryMarginals(model, observation)
    state = (1, 0)
    design = response_history_design(posterior, history, observation, state)
    full, prefix = primitive(design)
    np.testing.assert_allclose(full, _log_marginal(dense.samples, design, root, dense.nuisance, posterior), atol=1e-11)
    u = dense.conditioning_samples
    np.testing.assert_allclose(prefix, _log_marginal(dense.samples[:u], design[:u], root[:u, :u],
        dense.nuisance[:u], posterior), atol=1e-11)
    expected = conditional_history_evidence(posterior, history, dense)
    actual = conditional_history_evidence(posterior, history, observation, marginal_evaluator=primitive)
    np.testing.assert_allclose(actual.log_candidate_probability, expected.log_candidate_probability, atol=1e-11)
    change = np.array([[2., -.1], [.7, .5]])
    changed = replace(observation, nuisance=observation.nuisance@change)
    result = conditional_history_evidence(posterior, history, changed,
        marginal_evaluator=NativeHistoryMarginals(model, changed))
    np.testing.assert_allclose(result.conditional_log_likelihood, actual.conditional_log_likelihood, atol=1e-11)


def test_native_path_never_whitens_n_by_n_or_recompiles_noise(actual_model, monkeypatch):
    model = actual_model
    posterior, history, observation = operands(model)
    original = type(model.noise).whiten

    def guarded(self, values):
        assert values.shape != (len(observation.samples), len(observation.samples))
        return original(self, values)

    monkeypatch.setattr(type(model.noise), "whiten", guarded)
    actual = conditional_history_evidence(posterior, history, observation,
        marginal_evaluator=NativeHistoryMarginals(model, observation))
    assert len(actual.state_indices) == 6
    assert actual.probabilities.shape == (2,)


def test_neighbour_zero_and_identical_columns_keep_full_target_mass():
    neighbour = NeighborEventPrefix("next", ("a", "b", "c"), np.array([15]), [[1, 0, 0]])
    np.testing.assert_array_equal(neighbour.membership[:, 1], neighbour.membership[:, 2])
    assert neighbour.candidate_ids == ("a", "b", "c")
    with pytest.raises(ValueError, match="current selection"):
        PublicResponseHistory((neighbour,), "next", np.ones(3)/3)
    with pytest.raises(ValueError, match="full proper public"):
        PublicSelection("current", ("a", "b", "c"), np.array([15]), [[1, 0, 0]])
