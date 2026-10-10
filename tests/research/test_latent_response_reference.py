"""Finite B07 state law: physical cells and independent whole-density oracles."""
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.special import logsumexp

from research import latent_response_reference as owner
from research.joint_response_scale import GlobalGaussianPrior, NumericalResolutionError
from research.latent_response_reference import (
    ResponseState,
    analytic_basis,
    finite_selection_evidence,
    hierarchical_prior,
    selection_design,
)
from transfer.continuous_decision import _event_design
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec


def _dictionary(family="cosine", endpoint="free", *, rate=20., seconds=.5, columns=1):
    return ResponseDictionary.compile(ResponseDictionarySpec(family, seconds, columns, endpoint), sample_rate_hz=rate)


def _fixture():
    dictionary = _dictionary()
    states = (ResponseState(.55, .55, 0., .9, .5), ResponseState(1.45, 1.45, .04, 1.1, .5))
    prior = hierarchical_prior([.15, .5], [[.9, .16], [.16, .6]], np.diag([.07, .12]), 3)
    calibration = []
    for index, labels in enumerate(([1, 0], [0, 1])):
        calibration.append({"samples": np.array([.2, -.1, .8, .3, -.4, .7, .5, .1, -.2, .3])+index*.09,
            "onsets": np.array([1, 4]), "labels": np.array(labels), "origin_sample": 0,
            "available_sample": 10, "noise_root": .65*np.eye(10), "record_id": f"C-{index}"})
    observations = {"samples": np.array([0., .1, 1.8, 2.3, 1.7, 1.2, .9, .8, .5, -.2, .4,
                                        .2, -.1, .1, .3, -.3, .2, -.1, .4, .2, .1, -.1]),
        "onsets": np.array([2, 8]), "origin_sample": 0, "available_sample": 22,
        "noise_root": .65*np.eye(22)+np.diag(np.full(21, .11), -1), "record_id": "read-only"}
    return dictionary, states, prior, calibration, observations, np.array([[1, 0], [0, 1]]), 7


def _manual_constant_design(dictionary, item, labels, state, index, selections):
    """Direct per-cell rational test; no author's support/compile/design helper."""
    assert dictionary.matrix.shape[1] == 1 and dictionary.spec.family == "cosine"
    matrix = np.zeros((len(item["samples"]), 2*(1+selections)))
    rate = Fraction.from_float(dictionary.sample_rate_hz)
    duration = Fraction(len(dictionary.matrix), 1)/rate
    latency, width = Fraction.from_float(state.latency_seconds), Fraction.from_float(state.width_multiplier)
    for row in range(len(matrix)):
        for event, target in zip(item["onsets"], labels, strict=True):
            coordinate = (Fraction(item["origin_sample"]+row-int(event), 1)/rate-latency)/width
            if 0 <= coordinate < duration:
                matrix[row, 0] += state.common_amplitude
                matrix[row, 1] += target*state.target_amplitude
                matrix[row, 2*(1+index)] += 1.
                matrix[row, 2*(1+index)+1] += target
    return matrix


def _dense_component(prior, observations, designs):
    """Independent dense observation covariance; one shared prior, once."""
    matrix = np.vstack(designs)
    y = np.concatenate([item["samples"] for item in observations])
    covariance_prior = prior.root@prior.root.T
    noise = block_diag(*[item["noise_root"]@item["noise_root"].T for item in observations])
    covariance = noise+matrix@covariance_prior@matrix.T
    residual = y-matrix@prior.mean
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign == 1
    density = -.5*(len(y)*np.log(2*np.pi)+logdet+residual@np.linalg.solve(covariance, residual))
    gain = np.linalg.solve(covariance, matrix@covariance_prior).T
    return float(density), prior.mean+gain@residual, covariance_prior-gain@matrix@covariance_prior


def test_real_research_entry_integrates_shared_prior_once_and_updates_prefix_weights():
    dictionary, states, prior, calibration, observation, membership, cut = _fixture()
    actual = finite_selection_evidence(prior, dictionary, calibration, observation, membership, states, prefix=cut)
    assert Path(owner.__file__).resolve() == Path(__file__).resolve().parents[2]/"src/research/latent_response_reference.py"
    c_nodes = list(product(range(2), repeat=2))
    c_designs, c_logs = [], []
    for index, nodes in enumerate(c_nodes):
        designs = [_manual_constant_design(dictionary, item, item["labels"], states[node], axis, 3)
                   for axis, (item, node) in enumerate(zip(calibration, nodes, strict=True))]
        c_designs.append(designs)
        log, mean, covariance = _dense_component(prior, calibration, designs)
        c_logs.append(log)
        assert actual["c_log_evidence"][index] == pytest.approx(log, abs=2e-12)
        np.testing.assert_allclose(actual["c_coefficient_means"][index], mean, atol=2e-12)
        np.testing.assert_allclose(actual["c_coefficient_covariances"][index], covariance, atol=2e-12)
        split_prior_log = sum(_dense_component(prior, [item], [matrix])[0]
                              for item, matrix in zip(calibration, designs, strict=True))
        assert abs(split_prior_log-log) > .01
    dense_full, dense_prefix = [], []
    for candidate in range(2):
        full, before = [], []
        for state in states:
            design = _manual_constant_design(dictionary, observation, membership[:, candidate], state, 2, 3)
            prefix = {**observation, "samples": observation["samples"][:cut],
                      "noise_root": observation["noise_root"][:cut, :cut]}
            for matrices in c_designs:
                full.append(_dense_component(prior, [*calibration, observation], [*matrices, design])[0]-3*np.log(2))
                before.append(_dense_component(prior, [*calibration, prefix], [*matrices, design[:cut]])[0]-3*np.log(2))
        dense_full.append(full)
        dense_prefix.append(before)
    dense_full, dense_prefix = np.asarray(dense_full), np.asarray(dense_prefix)
    np.testing.assert_allclose(actual["full_joint_terms"], dense_full, atol=2e-12)
    np.testing.assert_allclose(actual["prefix_joint_terms"], dense_prefix, atol=2e-12)
    expected = logsumexp(dense_full, axis=1)-logsumexp(dense_prefix, axis=1)
    np.testing.assert_allclose(actual["conditional_log_density"], expected, atol=2e-12)
    mass_C = np.exp(np.array(c_logs)-logsumexp(c_logs))
    np.testing.assert_allclose(actual["c_posterior_mass"], mass_C, atol=2e-12)
    prefix_mass = np.exp(dense_prefix-logsumexp(dense_prefix, axis=1)[:, None])
    np.testing.assert_allclose(actual["state_posterior_given_C_prefix_q"], prefix_mass, atol=2e-12)
    frozen_mass = np.tile(mass_C/2, 2)
    wrong = logsumexp(np.log(frozen_mass)[None, :]+dense_full-dense_prefix, axis=1)
    assert np.max(abs(wrong-expected)) > .01
    assert actual["finite_component_evaluations"] == 36
    assert not actual["normalized_candidate_posterior_claimed"]
    assert "c_designs" not in actual


def test_candidate_and_calibration_order_permutations_keep_same_conditional_program():
    dictionary, states, prior, calibration, observation, membership, cut = _fixture()
    original = finite_selection_evidence(prior, dictionary, calibration, observation, membership, states, prefix=cut)
    candidate = finite_selection_evidence(prior, dictionary, calibration, observation, membership[:, ::-1], states, prefix=cut)
    reordered = finite_selection_evidence(prior, dictionary, calibration[::-1], observation, membership, states, prefix=cut)
    np.testing.assert_allclose(candidate["conditional_log_density"][::-1], original["conditional_log_density"], atol=1e-12)
    np.testing.assert_allclose(reordered["conditional_log_density"], original["conditional_log_density"], atol=1e-12)


def test_inference_entry_never_reads_calibration_labels_or_target_truth_from_observation():
    class UnlabeledObservation(dict):
        def __getitem__(self, key):
            if key in {"labels", "target_candidate", "target_truth"}:
                raise AssertionError("inference cannot consume target truth")
            return super().__getitem__(key)

    dictionary, states, prior, calibration, observation, membership, cut = _fixture()
    result = finite_selection_evidence(prior, dictionary, calibration, UnlabeledObservation(observation),
                                        membership, states, prefix=cut)
    assert np.isfinite(result["conditional_log_density"]).all()


@pytest.mark.parametrize("stop,retained,censored", [(1308, 306, True), (1309, 307, False)])
def test_exact_next_up_latency_keeps_legal_last_cell_and_censor(stop, retained, censored):
    dictionary = _dictionary(rate=256., seconds=1.2)
    delta = np.nextafter(1/256, np.inf)
    design, ledger = selection_design(dictionary, np.array([1000]), np.array([1]), origin_sample=998,
                                      observed_samples=stop-998, state=ResponseState(1., 1., delta, 1., 1.))
    record = ledger["events"][0]
    assert record["response_sample_centers_half_open"] == [1002, 1309]
    assert record["retained_native_rows"] == retained and record["right_censored"] is censored
    assert np.count_nonzero(design[:, 0]) == retained
    assert record["response_seconds_display_only"]


@pytest.mark.parametrize("latency,width,origin,stop", [
    (np.nextafter(0., 1.), 1., 998, 1310), (0., np.nextafter(1., np.inf), 998, 1310),
    (.035, 1.1, 1003, 1100), (.035, 1.1, 500, 800), (.035, 1.1, 1500, 1600)])
def test_half_open_exact_cells_and_intersections_share_one_ledger(latency, width, origin, stop):
    dictionary = _dictionary(rate=256., seconds=1.2)
    design, ledger = selection_design(dictionary, np.array([1000]), np.array([1]), origin_sample=origin,
        observed_samples=stop-origin, state=ResponseState(1., 1., latency, width, 1.))
    rate, delta, scale = Fraction(256), Fraction.from_float(latency), Fraction.from_float(width)
    cells = [sample for sample in range(origin, stop) if 0 <= (Fraction(sample-1000)/rate-delta)/scale < Fraction(307)/rate]
    assert (np.flatnonzero(design[:, 0])+origin).tolist() == cells
    assert ledger["events"][0]["retained_native_rows"] == len(cells)


@pytest.mark.parametrize("family,endpoint", [("gaussian", "free"), ("cosine", "free"), ("cosine", "hard_zero")])
def test_zero_state_consumes_saved_native_matrix_bitwise(family, endpoint):
    dictionary = _dictionary(family, endpoint, rate=20.1, seconds=1.2, columns=5)
    design, ledger = selection_design(dictionary, np.array([2, 9]), np.array([0, 1]), origin_sample=0,
        observed_samples=40, state=ResponseState(1., 1., 0., 1., 1.))
    np.testing.assert_array_equal(design, _event_design(np.array([2, 9]), np.array([0, 1]), 40, dictionary.matrix))
    assert ledger["physical_duration_definition"] == "exact_L_over_stored_sample_rate"


def test_large_clock_and_latency_preserve_nonconstant_waveform_without_float_subtraction():
    dictionary = _dictionary(rate=256., seconds=1.2, columns=3)
    frame = 2**60
    shift = 2**53
    original, _ = selection_design(dictionary, np.array([1000]), np.array([1]), origin_sample=998,
        observed_samples=311, state=ResponseState(1., 1., 0., 1., 1.))
    absolute, _ = selection_design(dictionary, np.array([frame+1000]), np.array([1]), origin_sample=frame+998,
        observed_samples=311, state=ResponseState(1., 1., 0., 1., 1.))
    delayed, _ = selection_design(dictionary, np.array([1000]), np.array([1]), origin_sample=shift+998,
        observed_samples=311, state=ResponseState(1., 1., shift/256, 1., 1.))
    np.testing.assert_array_equal(absolute, original)
    np.testing.assert_allclose(delayed, original, atol=2e-15, rtol=0)


def test_physical_shift_is_distinct_from_old_fixed_mask_and_extreme_display_is_typed():
    dictionary = _dictionary(rate=256., seconds=1.2)
    state = ResponseState(1., 1., .035, 1.1, 1.)
    physical, ledger = selection_design(dictionary, np.array([1000]), np.array([1]), origin_sample=998,
        observed_samples=400, state=state)
    fixed, fixed_ledger = selection_design(dictionary, np.array([1000]), np.array([1]), origin_sample=998,
        observed_samples=400, state=state, support="fixed_mask")
    assert not np.array_equal(physical, fixed)
    assert ledger["events"][0]["response_sample_centers_half_open"] != fixed_ledger["events"][0]["response_sample_centers_half_open"]
    with pytest.raises(NumericalResolutionError, match="display_is_not_representable"):
        selection_design(dictionary, np.array([1000]), np.array([1]), origin_sample=998,
            observed_samples=20, state=ResponseState(1., 1., 0., np.finfo(float).max, 1.))


@pytest.mark.parametrize("arm", ["free", "soft", "hard"])
def test_proper_endpoint_priors_use_declared_dictionary_and_one_prior_density(arm):
    free = _dictionary(columns=3, rate=16.)
    hard = _dictionary(endpoint="hard_zero", columns=3, rate=16.)
    embedding = block_diag(hard.constraint_map, hard.constraint_map)
    endpoints = block_diag(free.endpoint_rows, free.endpoint_rows)
    covariance = np.diag(np.linspace(.2, .8, 6))
    mean = np.linspace(-.4, .7, 6)
    local = .15*np.eye(6)
    precision = np.linalg.inv(covariance)
    actual = free
    if arm == "soft":
        covariance = np.linalg.inv(precision+endpoints.T@endpoints/.3**2)
        mean = covariance@precision@mean
        local = np.linalg.inv(np.linalg.inv(local)+endpoints.T@endpoints/.3**2)
    elif arm == "hard":
        original_mean, original_covariance = mean, covariance
        covariance = np.linalg.inv(embedding.T@precision@embedding)
        mean = covariance@embedding.T@precision@mean
        local = np.linalg.inv(embedding.T@np.linalg.inv(local)@embedding)
        conditional_mean = original_mean-original_covariance@endpoints.T@np.linalg.solve(
            endpoints@original_covariance@endpoints.T, endpoints@original_mean)
        np.testing.assert_allclose(embedding@mean, conditional_mean, atol=1e-15)
        assert np.max(abs(mean-embedding.T@original_mean)) > .01
        actual = hard
    covariance, local = (covariance+covariance.T)/2, (local+local.T)/2
    prior = hierarchical_prior(mean, covariance, local, 1)
    observation = {"samples": np.linspace(-.2, .7, 15), "onsets": np.array([1, 5]),
                   "origin_sample": 0, "available_sample": 15, "record_id": "endpoint-read-only", "noise_root": .7*np.eye(15)}
    state, member, cut = ResponseState(1., 1., 0., 1., 1.), np.array([[1, 0], [0, 1]]), 4
    result = finite_selection_evidence(prior, actual, [], observation, member, [state], prefix=cut)
    for q in range(2):
        # An independent scalar event insertion of the actual saved functions.
        k = actual.matrix.shape[1]
        global_design = np.zeros((15, 2*k))
        for event, target in zip(observation["onsets"], member[:, q], strict=True):
            length = min(len(actual.matrix), 15-int(event))
            global_design[event:event+length, :k] += actual.matrix[:length]
            global_design[event:event+length, k:] += target*actual.matrix[:length]
        matrix = np.column_stack((global_design, global_design))
        full = _dense_component(prior, [observation], [matrix])[0]
        prefix = {**observation, "samples": observation["samples"][:cut], "noise_root": observation["noise_root"][:cut, :cut]}
        before = _dense_component(prior, [prefix], [matrix[:cut]])[0]
        assert result["conditional_log_density"][q] == pytest.approx(full-before, abs=2e-12)
    assert result["c_log_evidence"][0] == 0.
    np.testing.assert_allclose(result["c_coefficient_means"][0], prior.mean, atol=1e-15)
    np.testing.assert_allclose(result["c_coefficient_covariances"][0], prior.root@prior.root.T, atol=1e-15)


def test_derivative_dictionary_keeps_physical_units_and_free_cosine_redundancy():
    dictionary = _dictionary(rate=64., seconds=1.2, columns=4)
    omega = np.asarray(dictionary.metadata()["generation"]["angular_frequencies_per_second"])
    times = dictionary.times
    np.testing.assert_allclose(analytic_basis(dictionary, times, 2), -omega**2*np.cos(times[:, None]*omega), atol=1e-14)
    assert np.linalg.matrix_rank(np.column_stack((dictionary.matrix, analytic_basis(dictionary, times, 2)))) == 4
    delta = .001
    beta = np.array([.1, .4, -.3, .2])
    translated = analytic_basis(dictionary, times-delta)@beta
    first = dictionary.matrix@beta-delta*analytic_basis(dictionary, times, 1)@beta
    second = first+.5*delta**2*analytic_basis(dictionary, times, 2)@beta
    assert np.linalg.norm(second-translated) < np.linalg.norm(first-translated)/30


@pytest.mark.parametrize("violation", ["negative_latency", "unanchored_amplitude", "availability", "local_mean", "cross_selection", "duplicate_record"])
def test_research_law_prior_and_available_clock_contracts_reject_false_semantics(violation):
    dictionary, states, prior, calibration, observation, membership, cut = _fixture()
    if violation == "negative_latency":
        states = [ResponseState(1., 1., -.01, 1., 1.)]
    elif violation == "unanchored_amplitude":
        states = [ResponseState(2., 2., 0., 1., 1.)]
    elif violation == "availability":
        observation = {**observation, "available_sample": 21}
    elif violation == "local_mean":
        mean = prior.mean.copy()
        mean[2] = 1.
        prior = GlobalGaussianPrior(mean, prior.root, "external")
    elif violation == "cross_selection":
        root = prior.root.copy()
        root[2, 0] = .1
        prior = GlobalGaussianPrior(prior.mean, root, "external")
    else:
        observation = {**observation, "record_id": calibration[0]["record_id"]}
    with pytest.raises(ValueError):
        finite_selection_evidence(prior, dictionary, calibration, observation, membership, states, prefix=cut)
