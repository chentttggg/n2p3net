"""Independent convolution/dense-Gaussian oracles for the finite-history reference."""
from itertools import product

import numpy as np
import pytest
from scipy.linalg import block_diag, null_space, toeplitz
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from research.response_history_identification import (
    HistoryNuisanceResolutionError,
    PublicResponseHistory,
    PublicSelection,
    ResponseObservation,
    ResponsePosterior,
    conditional_history_evidence,
    response_history_design,
)


def fixture(*, nuisance=False, prior=None, outside=False, channels=1):
    basis = np.array([[1.], [2.], [1.]])
    mean = np.array([[.2], [2.]]) if channels == 1 else np.array([[.2, -.1], [2., 1.]])
    spatial = np.ones((1, 1)) if channels == 1 else np.array([[1., 0.], [.3, .7]])
    posterior = ResponsePosterior(basis, mean, np.diag([10., 5.]), spatial, 10.)
    previous = PublicSelection("previous", ("p0", "p1"), np.array([4, 6] if outside else [5, 9]), np.eye(2))
    current = PublicSelection("current", ("q0", "q1"), np.array([12, 14]), np.eye(2))
    prior = np.full((2, 2), .25) if prior is None else prior
    history = PublicResponseHistory((previous, current), "current", prior)
    n = 7
    temporal = np.linalg.cholesky(toeplitz(.3**np.arange(n))/(1-.3**2))*.2
    z = np.column_stack((np.ones(n), np.arange(n)-1.)) if nuisance else np.empty((n, 0))
    observation = ResponseObservation(np.zeros((n, channels)), 10, temporal, z, 3 if nuisance else 2)
    design = manual_design(posterior, history, observation, (1, 0))
    samples = design@mean+np.array([.1, -.05, 0., .05, -.1, .02, 0.])[:, None]
    if nuisance:
        samples += z@np.array([[.7]*channels, [-.3]*channels])
    return posterior, history, ResponseObservation(samples, 10, temporal, z, observation.conditioning_samples)


def manual_design(posterior, history, observation, state):
    """Independent impulse convolution in an extended physical sample window."""
    origin, n, length = observation.origin_sample, len(observation.samples), len(posterior.basis)
    base = max(0, origin-length+1)
    common, target = np.zeros(origin-base+n), np.zeros(origin-base+n)
    for selection, q in zip(history.selections, state, strict=True):
        for i, sample in enumerate(selection.onsets):
            index = int(sample)-base
            if 0 <= index < len(common):
                common[index] += 1.
                target[index] += selection.membership[i, q]
    columns = []
    for impulse in (common, target):
        for basis in posterior.basis.T:
            columns.append(np.convolve(impulse, basis)[origin-base:origin-base+n])
    return np.column_stack(columns)


def oracle(posterior, history, observation):
    """Full vector Gaussian integrals, independent of producer QR factor sums."""
    covariance_beta = np.linalg.inv(posterior.precision_root.T@posterior.precision_root)
    sigma = posterior.spatial_root@posterior.spatial_root.T
    noise = observation.temporal_root@observation.temporal_root.T
    states = list(product(*(range(len(s.candidate_ids)) for s in history.selections)))
    current_axis = [s.selection_id for s in history.selections].index(history.current_selection_id)
    current_indices = np.asarray([state[current_axis] for state in states])
    full, prefix = [], []
    u = observation.conditioning_samples
    for state in states:
        design = manual_design(posterior, history, observation, state)
        covariance = noise+design@covariance_beta@design.T
        error = observation.samples-design@posterior.mean
        def marginal(residual, temporal, nuisance):
            if nuisance.shape[1]:
                complement = null_space(nuisance.T)
                residual = complement.T@residual
                temporal = complement.T@temporal@complement
                # Lebesgue dU in the original common nuisance coordinates,
                # not density only on an arbitrarily chosen orthogonal quotient.
                volume = -.5*posterior.mean.shape[1]*np.linalg.slogdet(nuisance.T@nuisance)[1]
            else:
                volume = 0.
            return multivariate_normal.logpdf(residual.flatten(order="F"), cov=np.kron(sigma, temporal))+volume

        full.append(marginal(error, covariance, observation.nuisance))
        prefix.append(marginal(error[:u], covariance[:u, :u], observation.nuisance[:u]))
    logpi = np.log(history.joint_label_prior.reshape(-1))
    candidates = len(history.selections[current_axis].candidate_ids)
    evidence = [logsumexp(logpi[current_indices == q]+np.asarray(full)[current_indices == q])
                -logsumexp(logpi[current_indices == q]+np.asarray(prefix)[current_indices == q])
                for q in range(candidates)]
    other_axes = tuple(i for i in range(len(history.selections)) if i != current_axis)
    priors = history.joint_label_prior.sum(axis=other_axes) if other_axes else history.joint_label_prior
    probabilities = np.asarray(evidence)+np.log(priors)
    probabilities -= logsumexp(probabilities)
    return np.asarray(evidence), probabilities, np.asarray(full), np.asarray(prefix)


def test_public_past_onsets_contribute_their_actual_visible_response_tail():
    posterior, history, observation = fixture()
    design = response_history_design(posterior, history, observation, (1, 0))
    expected = np.column_stack(([2., 1., 1., 2., 2., 2., 1.], [2., 1., 1., 2., 1., 0., 0.]))
    np.testing.assert_array_equal(design, expected)
    np.testing.assert_array_equal(design, manual_design(posterior, history, observation, (1, 0)))


@pytest.mark.parametrize("channels", [1, 2])
def test_exact_finite_history_matches_independent_full_vector_gaussian(channels):
    posterior, history, observation = fixture(prior=np.array([[.45, .05], [.1, .4]]), channels=channels)
    actual = conditional_history_evidence(posterior, history, observation)
    expected, probability, joint, prefix = oracle(posterior, history, observation)
    np.testing.assert_allclose(actual.conditional_log_likelihood, expected, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(actual.log_candidate_probability, probability, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(actual.state_joint_log_marginal, joint, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(actual.state_prefix_log_marginal, prefix, rtol=3e-11, atol=3e-11)
    assert actual.measure_kind == "proper_Gaussian_density_ratio"
    assert actual.program == "finite_history_conditional_tail_fixed_current_prior"


def test_prefix_updates_whole_neighbour_state_weights_instead_of_frozen_prior_averaging():
    posterior, history, observation = fixture()
    result = conditional_history_evidence(posterior, history, observation)
    assert result.state_prefix_probability_given_current[0, 2] > .99
    assert result.state_prefix_probability_given_current[1, 3] > .99
    wrong = []
    for q in range(2):
        selected = [q, q+2]
        components = result.state_joint_log_marginal[selected]-result.state_prefix_log_marginal[selected]
        wrong.append(logsumexp(np.log([.5, .5])+components))
    assert np.max(np.abs(np.asarray(wrong)-result.conditional_log_likelihood)) > .1


def test_shared_response_posterior_has_cross_covariance_not_independent_extra_gamma():
    posterior, history, observation = fixture()
    state = (1, 0)
    design = manual_design(posterior, history, observation, state)
    whole = design[:, [1]]
    neighbour = np.array([2., 1., 0., 0., 0., 0., 0.])[:, None]
    current = whole-neighbour
    variance = 1/25
    correct = variance*whole@whole.T
    wrong = variance*(current@current.T+neighbour@neighbour.T)
    assert correct[0, 2] == variance*2.
    assert wrong[0, 2] == 0.
    np.testing.assert_allclose(correct-wrong, variance*(current@neighbour.T+neighbour@current.T), rtol=0, atol=1e-16)


def test_whole_selection_mask_prior_is_correlated_and_not_independent_flash_labels():
    _, history, _ = fixture()
    member = history.selections[0].membership
    masks = member.T
    np.testing.assert_array_equal(masks.sum(axis=1), np.ones(2))
    centered = masks-masks.mean(axis=0)
    covariance = centered.T@centered/2
    np.testing.assert_array_equal(covariance, [[.25, -.25], [-.25, .25]])
    assert not np.array_equal(covariance, .25*np.eye(2))


def test_joint_history_prior_may_depend_on_current_protocol_target_without_uniform_fallback():
    posterior, history, observation = fixture(prior=np.diag([.5, .5]))
    result = conditional_history_evidence(posterior, history, observation)
    assert result.evaluated_states == 2
    np.testing.assert_array_equal(result.state_prefix_probability_given_current, [[1., 0., 0., 0.], [0., 0., 0., 1.]])
    np.testing.assert_allclose(result.conditional_log_likelihood,
        [result.state_joint_log_marginal[i]-result.state_prefix_log_marginal[i] for i in (0, 3)], rtol=0, atol=1e-14)


def test_nonoverlapping_unknown_history_cannot_fabricate_information_and_reduces_to_current_only():
    posterior, history, observation = fixture(outside=True, prior=np.array([[.4, .1], [.2, .3]]))
    result = conditional_history_evidence(posterior, history, observation)
    current = PublicResponseHistory((history.selections[1],), "current", history.joint_label_prior.sum(axis=0))
    alone = conditional_history_evidence(posterior, current, observation)
    np.testing.assert_allclose(result.conditional_log_likelihood, alone.conditional_log_likelihood, rtol=3e-12, atol=3e-12)
    assert result.identical_design_state_groups == ((0, 2), (1, 3))
    np.testing.assert_allclose(result.state_prefix_probability_given_current[0, [0, 2]], [2/3, 1/3], rtol=3e-12)


def test_diffuse_nuisance_uses_common_measure_and_coordinate_volume_in_both_marginals():
    posterior, history, observation = fixture(nuisance=True)
    reference = conditional_history_evidence(posterior, history, observation)
    assert reference.measure_kind == "common_sigma_finite_nuisance_marginal_ratio"
    change = np.array([[2., .2], [0., .4]])
    transformed = ResponseObservation(observation.samples, observation.origin_sample, observation.temporal_root,
                                      observation.nuisance@change, observation.conditioning_samples)
    actual = conditional_history_evidence(posterior, history, transformed)
    np.testing.assert_allclose(actual.conditional_log_likelihood, reference.conditional_log_likelihood, rtol=3e-11, atol=3e-11)
    volume = -posterior.mean.shape[1]*np.linalg.slogdet(change)[1]
    np.testing.assert_allclose(actual.state_joint_log_marginal-reference.state_joint_log_marginal, volume, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(actual.state_prefix_log_marginal-reference.state_prefix_log_marginal, volume, rtol=3e-11, atol=3e-11)
    shifted = ResponseObservation(observation.samples+observation.nuisance@np.array([[17.], [-3.]]),
        observation.origin_sample, observation.temporal_root, observation.nuisance, observation.conditioning_samples)
    np.testing.assert_allclose(conditional_history_evidence(posterior, history, shifted).conditional_log_likelihood,
                               reference.conditional_log_likelihood, rtol=3e-10, atol=3e-10)


def test_common_diffuse_marginal_matches_independent_orthogonal_quotient_with_volume():
    posterior, history, observation = fixture(nuisance=True, prior=np.array([[.45, .05], [.1, .4]]), channels=2)
    actual = conditional_history_evidence(posterior, history, observation)
    expected, probabilities, full, prefix = oracle(posterior, history, observation)
    np.testing.assert_allclose(actual.state_joint_log_marginal, full, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(actual.state_prefix_log_marginal, prefix, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(actual.conditional_log_likelihood, expected, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(actual.log_candidate_probability, probabilities, rtol=3e-11, atol=3e-11)


def test_prefix_must_identify_the_same_diffuse_columns_instead_of_dropping_them():
    posterior, history, observation = fixture(nuisance=True)
    short = ResponseObservation(observation.samples, 10, observation.temporal_root, observation.nuisance, 1)
    with pytest.raises(HistoryNuisanceResolutionError):
        conditional_history_evidence(posterior, history, short)


def test_voltage_unit_change_preserves_fixed_prior_candidate_scores_and_updates_tail_density_units():
    posterior, history, observation = fixture(nuisance=True)
    reference = conditional_history_evidence(posterior, history, observation)
    scale = 1e6
    p = ResponsePosterior(posterior.basis, scale*posterior.mean, posterior.precision_root,
                          scale*posterior.spatial_root, posterior.sample_rate_hz)
    x = ResponseObservation(scale*observation.samples, observation.origin_sample, observation.temporal_root,
                            observation.nuisance, observation.conditioning_samples)
    result = conditional_history_evidence(p, history, x)
    shift = -(len(x.samples)-x.conditioning_samples)*x.samples.shape[1]*np.log(scale)
    np.testing.assert_allclose(result.conditional_log_likelihood, reference.conditional_log_likelihood+shift, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(result.log_candidate_probability, reference.log_candidate_probability, rtol=3e-11, atol=3e-11)


def test_same_response_chart_change_and_candidate_permutation_preserve_physical_program():
    posterior, history, observation = fixture()
    reference = conditional_history_evidence(posterior, history, observation)
    scale = 2.
    chart = block_diag([[scale]], [[scale]])
    p = ResponsePosterior(scale*posterior.basis, np.linalg.solve(chart, posterior.mean),
                          posterior.precision_root@chart, posterior.spatial_root, posterior.sample_rate_hz)
    np.testing.assert_allclose(conditional_history_evidence(p, history, observation).conditional_log_likelihood,
                               reference.conditional_log_likelihood, rtol=3e-12, atol=3e-12)
    current = history.selections[1]
    permuted = PublicSelection(current.selection_id, tuple(reversed(current.candidate_ids)), current.onsets,
                               current.membership[:, ::-1])
    h = PublicResponseHistory((history.selections[0], permuted), "current", history.joint_label_prior[:, ::-1])
    result = conditional_history_evidence(posterior, h, observation)
    assert result.candidate_ids == tuple(reversed(reference.candidate_ids))
    np.testing.assert_allclose(result.log_candidate_probability, reference.log_candidate_probability[::-1], rtol=3e-12, atol=3e-12)


def test_general_nonorthogonal_basis_pullback_preserves_shared_kernel_mixture():
    original, history, observation = fixture(nuisance=True)
    basis = np.column_stack((original.basis[:, 0], [0., 1., -.5]))
    posterior = ResponsePosterior(basis, np.array([[.2], [.1], [2.], [-.3]]), np.diag([10., 8., 5., 3.]),
                                  original.spatial_root, original.sample_rate_hz)
    reference = conditional_history_evidence(posterior, history, observation)
    change = np.array([[2., .2], [0., .4]])
    chart = block_diag(change, change)
    transformed = ResponsePosterior(basis@change, np.linalg.solve(chart, posterior.mean),
        posterior.precision_root@chart, posterior.spatial_root, posterior.sample_rate_hz)
    actual = conditional_history_evidence(transformed, history, observation)
    np.testing.assert_allclose(actual.conditional_log_likelihood, reference.conditional_log_likelihood, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(actual.state_prefix_probability_given_current,
                               reference.state_prefix_probability_given_current, rtol=3e-11, atol=3e-11)


def test_incomplete_state_prior_or_requested_current_tail_is_not_silently_truncated():
    posterior, history, observation = fixture()
    with pytest.raises(ValueError, match="Cartesian"):
        PublicResponseHistory(history.selections, "current", np.array([.5, .5]))
    with pytest.raises(ValueError, match="positive prior"):
        PublicResponseHistory(history.selections, "current", np.array([[.5, 0.], [.5, 0.]]))
    cropped = ResponseObservation(observation.samples[:-1], 10, observation.temporal_root[:-1, :-1],
                                  observation.nuisance[:-1], 2)
    with pytest.raises(ValueError, match="complete observed"):
        conditional_history_evidence(posterior, history, cropped)


def test_correlated_past_and_future_whole_selections_update_future_before_its_first_flash():
    posterior, base, observation = fixture(channels=2)
    future = PublicSelection("future", ("f0", "f1", "f2"), np.array([15, 18, 20]), np.eye(3))
    prior = np.array([[[.01, .18, .01], [.01, .16, .03]],
                      [[.22, .01, .02], [.27, .05, .03]]])
    history = PublicResponseHistory((*base.selections, future), "current", prior)
    old_design = manual_design(posterior, base, observation, (1, 0))
    new_design = manual_design(posterior, history, observation, (1, 0, 0))
    samples = observation.samples+(new_design-old_design)@posterior.mean
    observed = ResponseObservation(samples, observation.origin_sample, observation.temporal_root,
                                   observation.nuisance, observation.conditioning_samples)
    actual = conditional_history_evidence(posterior, history, observed)
    evidence, probabilities, full, prefix = oracle(posterior, history, observed)
    np.testing.assert_allclose(actual.conditional_log_likelihood, evidence, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(actual.log_candidate_probability, probabilities, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(actual.state_joint_log_marginal, full, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(actual.state_prefix_log_marginal, prefix, rtol=3e-11, atol=3e-11)
    assert actual.evaluated_states == 12
    assert int(future.onsets[0]) >= observed.origin_sample+observed.conditioning_samples
    states = np.asarray(actual.state_indices)
    for q in range(2):
        old_future_mass = prior[:, q, 0].sum()/prior[:, q, :].sum()
        new_future_mass = actual.state_prefix_probability_given_current[q, states[:, 2] == 0].sum()
        assert new_future_mass-old_future_mass > .15


def test_fixed_current_prior_conditional_tail_program_is_not_full_data_bayes_posterior():
    posterior, history, observation = fixture(prior=np.array([[.45, .05], [.1, .4]]))
    actual = conditional_history_evidence(posterior, history, observation)
    _, expected, full, _ = oracle(posterior, history, observation)
    logprior = np.log(history.joint_label_prior.reshape(-1))
    full_candidate = np.array([logsumexp(logprior[[q, q+2]]+full[[q, q+2]]) for q in range(2)])
    full_candidate -= logsumexp(full_candidate)
    np.testing.assert_allclose(actual.log_candidate_probability, expected, rtol=3e-11, atol=3e-11)
    fixed_log_odds = np.diff(actual.log_candidate_probability)[0]
    full_log_odds = np.diff(full_candidate)[0]
    assert abs(fixed_log_odds-full_log_odds) > 1.
