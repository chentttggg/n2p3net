"""Exact finite shifts on actual integer clocks, not fractional event claims."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from research.latent_response_reference import ResponseState, analytic_basis, selection_design
from transfer.continuous_decision import _event_design
from transfer.dictionary_artifact import DictionaryArtifactError
from transfer.noise_model import array_digest
from transfer.response_dictionary import (
    ResponseDictionary,
    ResponseDictionaryEvaluation,
    ResponseDictionarySpec,
)

LEGAL = [(family, endpoint) for family in ("gaussian", "cosine", "bspline", "raised_cosine", "multiscale_bspline")
         for endpoint in ("free", "hard_zero") if (family, endpoint) != ("gaussian", "hard_zero")]


def dictionary(family, endpoint, *, rate=20.1, seconds=.5):
    return ResponseDictionary.compile(ResponseDictionarySpec(family, seconds,
        9 if family == "multiscale_bspline" else 5, endpoint), sample_rate_hz=rate)


@pytest.mark.parametrize("family,endpoint", LEGAL)
def test_zero_state_uses_saved_rows_while_nonzero_state_uses_exact_query_artifacts(family, endpoint):
    d = dictionary(family, endpoint)
    onsets, labels = np.array([2, 10]), np.array([0, 1])
    zero, native_ledger = selection_design(d, onsets, labels, origin_sample=0, observed_samples=30,
        state=ResponseState(1., 1., 0., 1., 1.))
    assert zero.tobytes() == _event_design(onsets, labels, 30, d.matrix).tobytes()
    assert native_ledger["events"][0]["response_evaluation"]["program"] == "original_saved_native_matrix_slice/1"
    state = ResponseState(.75, 1.25, .025, 1.1, 1.)
    design, ledger = selection_design(d, onsets, labels, origin_sample=0, observed_samples=20, state=state)
    assert ledger["events"][1]["right_censored"]
    assert ledger["events"][0]["onset_sample"] == 2
    assert ledger["dictionary_identity_digest"] == d.identity_digest
    assert ledger["dictionary_producer"] == d.metadata()["implementation_sha256"]
    n, k = len(design), d.matrix.shape[1]
    replay = np.zeros((n, 2*k))
    for row in ledger["events"]:
        artifact = ResponseDictionaryEvaluation.from_record(row["response_evaluation"], dictionary=d)
        first, stop = row["response_sample_centers_half_open"]
        stop = min(stop, n)
        assert len(artifact.matrix) == stop-first
        replay[first:stop, :k] += state.common_amplitude*artifact.matrix
        replay[first:stop, k:] += row["target_indicator"]*state.target_amplitude*artifact.matrix
    assert replay.tobytes() == design.tobytes()
    assert ledger["design_sha256"] == array_digest(replay)


@pytest.mark.parametrize("family,endpoint", LEGAL)
def test_large_absolute_clock_translation_keeps_same_shapes_and_exact_sites(family, endpoint):
    d = dictionary(family, endpoint, rate=32.)
    onsets, labels = np.array([2, 10]), np.array([0, 1])
    state = ResponseState(1., 1., .013, 1.125, 1.)
    design, _ = selection_design(d, onsets, labels, origin_sample=0, observed_samples=30, state=state)
    translated, ledger = selection_design(d, onsets+2**60, labels, origin_sample=2**60,
        observed_samples=30, state=state)
    assert translated.tobytes() == design.tobytes()
    assert ledger["events"][0]["onset_sample"] == 2**60+2


@pytest.mark.parametrize("family,endpoint", LEGAL)
def test_time_units_query_and_latency_pull_back_together(family, endpoint):
    d = dictionary(family, endpoint, rate=16., seconds=1.)
    # Eight-fold time units retain L, and delta*fs is exactly identical.
    scaled = dictionary(family, endpoint, rate=2., seconds=8.)
    old, old_record = selection_design(d, np.array([3]), np.array([1]), origin_sample=2,
        observed_samples=20, state=ResponseState(1., 1., .0625, 1.25, 1.))
    new, new_record = selection_design(scaled, np.array([3]), np.array([1]), origin_sample=2,
        observed_samples=20, state=ResponseState(1., 1., .5, 1.25, 1.))
    assert old.tobytes() == new.tobytes()
    assert old_record["events"][0]["response_sample_centers_half_open"] == new_record["events"][0]["response_sample_centers_half_open"]


def test_exact_interior_cell_must_not_round_to_endpoint_before_expression():
    d = dictionary("bspline", "free", rate=256., seconds=1.2)
    value = Fraction(307)-Fraction(1, 2**80)
    assert float(value) == 307.
    query = d.evaluate([value], coordinate="sample", support="half_open")
    at_endpoint = d.evaluate([307], coordinate="sample", support="half_open")
    assert query.support_mask[0] and not at_endpoint.support_mask[0]
    assert query.matrix[0, -1] == 1. and query.matrix[0, -2] > 0.
    assert np.array_equal(at_endpoint.matrix, np.zeros_like(at_endpoint.matrix))


def test_causal_shift_does_not_round_onset_and_is_distinct_from_fixed_mask():
    d = dictionary("gaussian", "free", rate=16.)
    onsets, labels = np.array([4]), np.array([1])
    positive = ResponseState(1., 1., 2**-20, 1., 1.)
    shifted, ledger = selection_design(d, onsets, labels, origin_sample=4, observed_samples=10, state=positive)
    assert shifted[0, 0] == 0. and ledger["events"][0]["onset_sample"] == 4
    fixed, _ = selection_design(d, onsets, labels, origin_sample=4, observed_samples=10,
        state=positive, support="fixed_mask")
    assert fixed[0, 0] > 0. and fixed.tobytes() != shifted.tobytes()
    with pytest.raises(ValueError, match="actual public sample clock"):
        selection_design(d, np.array([4.49]), labels, origin_sample=4, observed_samples=10, state=positive)


def test_operator_filtering_is_not_implied_by_the_exact_response_evaluator():
    d = dictionary("cosine", "free", rate=16.)
    response = d.evaluate([0, -1], coordinate="sample", support="half_open")
    assert response.matrix[0, 0] == 1. and response.matrix[1, 0] == 0.
    # A toy acausal observation operator may move future response into an
    # earlier output. Dictionary half-open support does not authorize its read.
    raw = np.array([0., 0., 0., 0., response.matrix[0, 0]])
    operator = np.eye(5)+np.diag(np.full(4, .25), 1)
    assert (operator@raw)[3] == .25
    assert response.metadata()["program"]["derivative"] == 0
    with pytest.raises(TypeError):
        d.evaluate([.1], derivative=1)
    with pytest.raises(DictionaryArtifactError):
        d.evaluate([.1], support="filtered_actual_operator")
    np.testing.assert_array_equal(analytic_basis(d, [Fraction(1, 16)]), d.evaluate([1], coordinate="sample").matrix)
