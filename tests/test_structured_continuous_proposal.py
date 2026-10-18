"""Independent dense Gaussian oracles for record-conditional proposal algebra."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_t

from research.collapsed_continuous_inference import GaussianProposal, run_smc
from research.structured_continuous_proposal import (
    ConditionalDriftBlock,
    RecordConditionalGaussian,
    RecordConditionalStudent,
    pilot_from_structured_cloud,
    proposal_from_record,
    with_heavy_reference,
)


def _example():
    return RecordConditionalGaussian(np.array([.2, -.1]), np.array([[1.1, 0.], [.15, .8]]), (
        ConditionalDriftBlock("record-A", np.array([[2., .3], [0., -1.4]]),
            np.array([.3, -.4]), np.array([[.5, -.2], [.1, .3]]),
            np.array([[.8, 0.], [-.1, .9]])),
        ConditionalDriftBlock("record-B", np.array([[.7]]), np.array([.1]),
            np.array([[-.2, .4]]), np.array([[.6]])),
    ))


def _dense(proposal):
    """Construct the full linear map solely for an independent small oracle."""
    factor = np.zeros((proposal.dimension, proposal.dimension))
    factor[:proposal.rank, :proposal.rank] = proposal.head_root
    means, offset = [proposal.head_mean], proposal.rank
    for block in proposal.blocks:
        stop = offset+len(block.mean)
        means.append(np.linalg.solve(block.chart_root, block.mean))
        factor[offset:stop, :proposal.rank] = np.linalg.solve(
            block.chart_root, block.regression@proposal.head_root)
        factor[offset:stop, offset:stop] = np.linalg.solve(block.chart_root, block.root)
        offset = stop
    mean = np.concatenate(means)
    return GaussianProposal(mean, np.linalg.cholesky(factor@factor.T)), factor


def _noise(rng, proposal, count):
    return np.concatenate([rng.normal(size=(count, proposal.rank)),
        *(rng.normal(size=(count, len(block.mean))) for block in proposal.blocks)], axis=1)


def test_physical_density_draw_and_pcn_equal_independent_joint_gaussian():
    proposal = _example()
    dense, factor = _dense(proposal)
    points = np.random.default_rng(71).normal(size=(13, proposal.dimension))
    np.testing.assert_allclose(proposal.log_density(points), dense.log_density(points),
        rtol=0., atol=3e-14)
    noise = _noise(np.random.default_rng(72), proposal, 13)
    np.testing.assert_allclose(proposal.draw(np.random.default_rng(72), 13),
        dense.mean+noise@factor.T, rtol=0., atol=2e-15)
    scale = .4
    noise = _noise(np.random.default_rng(73), proposal, 13)
    expected = dense.mean+np.sqrt(1-scale**2)*(points-dense.mean)+scale*noise@factor.T
    np.testing.assert_allclose(proposal.reversible_move(np.random.default_rng(73), points, scale),
        expected, rtol=0., atol=2e-15)
    restored = proposal_from_record(proposal.record())
    np.testing.assert_array_equal(restored.log_density(points), proposal.log_density(points))


def test_structured_pilot_and_smc_retain_absolute_components_and_all_candidates():
    reference = _example()

    class Target:
        dimension = reference.dimension
        endpoints = ("P", "F")
        q_values = ("a", "b", "c")
        log_prior = np.log([.2, .3, .5])
        identity = "structured-proposal-closed-form-contract"
        refusals = {}
        integrals = np.array([[600., 590., 580.], [-800., -805., -804.]])

        def evaluate_many(self, points):
            return reference.log_density(points)[:, None, None]+self.integrals

    target = Target()
    pilot = pilot_from_structured_cloud(target, reference, pilot_seed=74,
        particles=24, covariance_reference_fraction=.15)
    np.testing.assert_allclose(pilot.component_log_scales, target.integrals, rtol=0., atol=3e-13)
    assert pilot.receipt["candidate_support"] == list(target.q_values)
    assert len(pilot.proposal.components) == 6
    assert all(isinstance(c, RecordConditionalGaussian) for c in pilot.proposal.components)
    restored = proposal_from_record(pilot.proposal.record())
    points = pilot.arrays["pilot_points"]
    np.testing.assert_array_equal(restored.log_density(points), pilot.proposal.log_density(points))
    # An exactly matched reference gives a zero-variance component estimator.
    scales = target.integrals+np.array([[1., -2., 3.], [-3., 2., .7]])
    result, _ = run_smc(target, reference, scales, inference_seed=75,
        particles=8, replicates=3, temperatures=(0., .5, 1.), moves=1, pcn_scale=.4)
    np.testing.assert_allclose(np.array(result["pooled_scaled_log_components"])+scales,
        target.integrals, rtol=0., atol=5e-13)
    expected = target.integrals[1]-logsumexp(target.integrals[1])
    np.testing.assert_allclose(result["probabilities"]["whole_log_probability"], expected,
        rtol=0., atol=5e-13)


def test_zero_source_rank_retains_a_proper_record_drift_law():
    proposal = RecordConditionalGaussian(np.empty(0), np.empty((0, 0)), (
        ConditionalDriftBlock("record", np.array([[2.]]), np.array([.4]),
            np.empty((1, 0)), np.array([[.7]])),))
    dense = GaussianProposal(np.array([.2]), np.array([[.35]]))
    points = np.array([[-1.], [.2], [.7]])
    np.testing.assert_allclose(proposal.log_density(points), dense.log_density(points),
        rtol=0., atol=2e-15)


def test_shared_gamma_t_reference_matches_independent_multivariate_t_and_codec():
    base = _example()
    dense, factor = _dense(base)
    reference = RecordConditionalStudent(base, 4.)
    points = np.random.default_rng(81).normal(size=(17, base.dimension))
    expected = multivariate_t.logpdf(points, loc=dense.mean, shape=factor@factor.T, df=4.)
    np.testing.assert_allclose(reference.log_density(points), expected, rtol=0., atol=3e-14)
    rng = np.random.default_rng(82)
    noise = _noise(rng, base, len(points))
    precision = rng.gamma(2., .5, size=len(points))
    expected_draws = dense.mean+(noise@factor.T)/np.sqrt(precision[:, None])
    np.testing.assert_allclose(reference.draw(np.random.default_rng(82), len(points)),
        expected_draws, rtol=0., atol=3e-15)
    np.testing.assert_array_equal(reference.reversible_move(np.random.default_rng(82), points, .2),
        reference.draw(np.random.default_rng(82), len(points)))
    mixture = with_heavy_reference(base, base, weight=.1, degrees_of_freedom=4.)
    restored = proposal_from_record(mixture.record())
    np.testing.assert_array_equal(restored.log_density(points), mixture.log_density(points))
    # A Gaussian/t density ratio decays in all directions, including a local
    # drift direction coupled to u. The positive t mass preserves the tail.
    far = dense.mean+np.array([[0., 0., 0., 0., 0.], [10., -7., 9., 4., -8.],
                             [100., -70., 90., 40., -80.]])
    ratio = base.log_density(far)-mixture.log_density(far)
    assert np.isfinite(ratio).all() and ratio[2] < ratio[1] < ratio[0]
