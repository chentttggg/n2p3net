"""Closed-form public integration contracts for the continuous reference."""

import numpy as np
from scipy.special import logsumexp

from research.collapsed_continuous_inference import (
    GaussianProposal,
    replay_saved_cloud,
    run_smc,
    save_cloud,
)
from research.continuous_integration_pilot import pilot_from_positive_cloud


class CommonGaussianTarget:
    """Known integrals across a 1600-nat P/F gap, without any analytic API."""

    dimension = 2
    endpoints = ("P", "F")
    q_values = ("left", "middle", "right")
    refusals = {}
    identity = "common-gaussian-public-contract"
    log_prior = np.log([.2, .3, .5])
    log_integrals = np.array([[900., 890., 860.], [-700., -701., -708.]])

    def __init__(self, reference):
        self.reference = reference

    def evaluate_many(self, points):
        return self.reference.log_density(points)[:, None, None]+self.log_integrals[None, :, :]


def test_common_cloud_restores_all_candidate_scales_and_saved_integral(tmp_path):
    proposal = GaussianProposal(np.array([.4, -.2]), np.array([[.8, 0.], [.1, 1.3]]))
    target = CommonGaussianTarget(proposal)
    scales = target.log_integrals+np.array([[1., -2., .5], [-1., .7, 3.]])
    result, arrays = run_smc(target, proposal, scales, inference_seed=90817,
        particles=8, replicates=3, temperatures=(0., .4, 1.), moves=1, pcn_scale=.3)
    expected_whole = target.log_integrals[1]-logsumexp(target.log_integrals[1])
    conditional = target.log_prior+target.log_integrals[1]-target.log_integrals[0]
    expected_conditional = conditional-logsumexp(conditional)
    np.testing.assert_allclose(result["probabilities"]["whole_log_probability"],
        expected_whole, rtol=0., atol=5e-13)
    np.testing.assert_allclose(result["probabilities"]["conditional_log_probability"],
        expected_conditional, rtol=0., atol=5e-13)
    np.testing.assert_allclose(np.array(result["pooled_scaled_log_components"])+scales,
        target.log_integrals, rtol=0., atol=5e-13)
    assert result["quality_kind"] == "replicated_mc_estimate"
    assert result["finite_sample_certificate"] is False
    save_cloud(tmp_path/"cloud.npz", result, arrays)
    replay = replay_saved_cloud(result)
    for key, value in replay.items():
        np.testing.assert_array_equal(value, result["probabilities"][key])


def test_separate_pilot_uses_density_correction_for_every_endpoint_candidate():
    reference = GaussianProposal(np.array([.4, -.2]), np.array([[.8, 0.], [.1, 1.3]]))
    target = CommonGaussianTarget(reference)
    pilot = pilot_from_positive_cloud(target, reference, pilot_seed=90716,
        particles=12, covariance_reference_fraction=.2)
    np.testing.assert_allclose(pilot.component_log_scales, target.log_integrals,
        rtol=0., atol=2e-13)
    assert pilot.receipt["candidate_support"] == list(target.q_values)
    assert pilot.receipt["truth_read"] is False
    assert len(pilot.proposal.components) == len(target.endpoints)*len(target.q_values)
    assert np.isfinite(pilot.proposal.log_density(pilot.arrays["pilot_points"])).all()
