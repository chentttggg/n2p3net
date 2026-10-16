"""Raw-coordinate posterior oracles for joint history, separate from its QR."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import norm

from research.channel_noise_mechanism import FittedChannelNoise, NoiseGeometry
from research.joint_flat_chain import (
    AugmentedQRTree,
    DiscreteTargetAxis,
    SparseResponseImage,
    _axis_logits,
    _draw_coefficients,
    rao_blackwell_tail_components,
    run_joint_history_gibbs,
    summarize_joint_tail,
)
from research.joint_flat_sparse import compile_sparse_response_image, whiten_record_prefix


@pytest.mark.parametrize("start", [1, 6, 14])
def test_sparse_difference_is_the_original_full_w_not_a_record_reset(start):
    geometry = NoiseGeometry(("Cz", "Pz"), 16., "V", "native", "analytic", 2)
    noise = FittedChannelNoise.from_parameters([[.4], [-.2]], [[1.3, .3], [.3, .8]],
        geometry=geometry, conditioning_samples=6, reference_id="raw-coordinate-independent-oracle")
    rng = np.random.default_rng(498)
    pulse = rng.normal(size=(5, 2, 3))
    whole = np.zeros((24, 2, 3))
    whole[start:start+len(pulse)] = pulse
    actual = compile_sparse_response_image(noise, pulse, start_native=start, total_native=24)
    expected = noise.whiten(whole).reshape(48, 3)
    joined = np.zeros_like(expected)
    joined[actual.image.start:actual.image.start+len(actual.image.values)] = actual.image.values
    np.testing.assert_array_equal(joined, expected)
    assert actual.binding["new_recordings"] == actual.binding["new_observed_padding_rows"] == 0
    # Original correlated AR covariance, independent of the whitening helper.
    phi, sigma = np.array([.4, -.2]), np.array([[1.3, .3], [.3, .8]])
    stationary = sigma/(1-phi[:, None]*phi[None, :])
    covariance = np.empty((48, 48))
    for left in range(24):
        for right in range(24):
            block = np.diag(phi**abs(left-right))@stationary
            covariance[2*left:2*left+2, 2*right:2*right+2] = block if left >= right else block.T
    raw = whole.reshape(48, 3)
    np.testing.assert_allclose(joined.T@joined, raw.T@np.linalg.solve(covariance, raw), atol=5e-13)


def test_short_prefix_uses_original_saved_leading_map():
    geometry = NoiseGeometry(("Cz",), 16., "V", "native", "analytic", 1)
    noise = FittedChannelNoise.from_parameters([[.4]], [[.7]], geometry=geometry,
        conditioning_samples=6, reference_id="leading-original-map")
    values = np.array([[[1., 2.]], [[3., 1.]]])
    white, _ = whiten_record_prefix(noise, values)
    expected = np.linalg.solve(noise.compiled_prefix_root[:2, :2], values.reshape(2, 2))
    np.testing.assert_allclose(white.reshape(2, 2), expected, atol=1e-15)


def test_qr_leaves_return_to_bit_exact_original_state_without_update_history():
    rng = np.random.default_rng(64)
    base = rng.normal(size=(33, 4))
    a = SparseResponseImage(2, rng.normal(size=(13, 2)))
    b = SparseResponseImage(4, rng.normal(size=(12, 2)))
    second = SparseResponseImage(7, rng.normal(size=(13, 2)))
    tree = AugmentedQRTree(base, leaf_rows=8)
    tree.set_image(0, a, response_width=1)
    tree.set_image(1, second, response_width=1)
    original, root = tree.rows.copy(), tree.root.copy()
    for _ in range(30):
        tree.set_image(0, b, response_width=1)
        tree.set_image(0, a, response_width=1)
    np.testing.assert_array_equal(tree.rows, original)
    np.testing.assert_array_equal(tree.root, root)
    vector = np.array([.4, -.2, .8, -1.])
    assert np.linalg.norm(root@vector)**2 == pytest.approx(np.linalg.norm(original@vector)**2, abs=1e-12)


def test_original_operand_cancellation_cannot_silently_change_the_posterior():
    from fractions import Fraction

    from research.channel_noise_mechanism import DiffuseProjectionResolutionError

    # Exact real sums of the supplied binary64 operands give x=[1,0,0].
    # Flat a integration leaves x' M x=2/3, with one proper beta prior.
    exact = Fraction.from_float(1.e16)+Fraction(1)-Fraction.from_float(1.e16)
    xmx = exact*exact*Fraction(2, 3)
    assert Fraction(1)/(1+xmx) == Fraction(3, 5)
    assert exact*Fraction(2, 3)/(1+xmx) == Fraction(2, 5)
    base = np.array([[1.e16, 1., 1.], [0., 1., 0.], [0., 1., 0.]])
    tree = AugmentedQRTree(base, leaf_rows=2)
    tree.set_image(0, SparseResponseImage(0, [[1., 0.]]), response_width=1)
    tree.set_image(1, SparseResponseImage(0, [[-1.e16, 0.]]), response_width=1)
    assert tree.rows[0, 0] == 0. and tree.source_errors[0] >= 1.
    # The rounded zero column would instead yield beta mean=0, variance=1.
    with pytest.raises(DiffuseProjectionResolutionError, match="posterior metric"):
        _draw_coefficients((tree,), [1.], 1, np.random.default_rng(19))


def test_original_q_conditional_excludes_axis_identity_when_images_are_shared():
    base = np.column_stack((np.zeros(4), np.ones(4), [1., -.5, .4, .3]))
    image = SparseResponseImage(0, np.column_stack((np.ones(4)*.2, np.ones(4)*.3)))
    zero = SparseResponseImage(0, np.zeros((4, 2)))
    axis = DiscreteTargetAxis("current", ("a", "b"), [.5, .5], (zero, image), (zero, image))
    tree = AugmentedQRTree(base, leaf_rows=2)
    tree.set_image(0, image, response_width=1)
    tree.set_image(1, image, response_width=1)
    theta, precision = np.array([.8, -.2]), 1.3
    actual = _axis_logits(tree, axis, 0, theta, precision, 1)
    prior_residual = base[:, -1]-base[:, :-1]@theta-image.values@np.r_[theta[:1], 1.]
    reference = [np.log(.5)-.5*precision*np.linalg.norm(prior_residual-c.values@np.r_[theta[:1], 1.])**2
                 for c in axis.prefix_images]
    np.testing.assert_allclose(actual, reference, atol=2e-15)


def problem():
    n = 12
    time = np.arange(n)
    y = .25*np.cos(time*.7)+.11*time/n
    common = .3*np.sin(time*.6)
    # Source beta=.7+u, a flat intercept in noise units.
    base = np.column_stack((common, np.ones(n)/np.sqrt(n), y-.7*common))
    f1, f2 = np.zeros(n), np.zeros(n)
    f1[1:6], f2[6:11] = [0., .3, .7, .5, .1], [.1, .6, .5, .3, 0.]
    axes = []
    for index, waveform in enumerate((f1, f2)):
        images = tuple(SparseResponseImage(0, np.column_stack((sign*waveform, .7*sign*waveform)))
                       for sign in (-1., 1.))
        empty = tuple(SparseResponseImage(0, np.empty((0, 2))) for _ in range(2))
        axes.append(DiscreteTargetAxis(str(index), ("a", "b"), [.4, .6], images, empty))
    return base, axes, y, common, (f1, f2)


def exact_collapsed(nu, y, common, waveforms):
    """Integrate physical flat intercept and Gamma, then raw Gaussian u."""
    values, umean, lambda_mean = [], [], []
    dimension = len(y)-1
    for q0 in range(2):
        for q1 in range(2):
            design = common+(2*q0-1)*waveforms[0]+(2*q1-1)*waveforms[1]
            prior = [.4, .6][q0]*[.4, .6][q1]

            def weight(u, design=design):
                residual = y-design*(.7+u)
                projected = residual-residual.mean()
                energy = projected@projected
                if np.isposinf(nu):
                    likelihood = np.exp(-.5*energy)
                else:
                    # Common da-volume/normalizing constants cancel among q.
                    likelihood = (1+energy/nu)**(-(nu+dimension)/2)
                return norm.pdf(u)*likelihood

            mass = quad(weight, -9, 9, epsabs=1e-11)[0]
            values.append(prior*mass)
            umean.append(prior*quad(lambda u: u*weight(u), -9, 9, epsabs=1e-11)[0])
            if np.isposinf(nu):
                lambda_mean.append(prior*mass)
            else:
                def lambda_integral(u, design=design, weight=weight):
                    residual = y-design*(.7+u)
                    projected = residual-residual.mean()
                    return weight(u)*(nu+dimension)/(nu+projected@projected)

                lambda_mean.append(prior*quad(lambda_integral, -9, 9, epsabs=1e-11)[0])
    total = sum(values)
    return np.asarray(values).reshape(2, 2)/total, sum(umean)/total, sum(lambda_mean)/total


@pytest.mark.parametrize("nu", [4., np.inf])
def test_joint_gibbs_matches_raw_collapsed_two_axis_posterior_and_precision(nu):
    base, axes, y, common, waveforms = problem()
    samples = run_joint_history_gibbs(base, axes, response_width=1, degrees_of_freedom=nu,
        initial_coefficients=[[0., 0.], [2., -.7]], initial_precisions=[[1.], [1.]],
        initial_targets=[[0, 0], [1, 1]], seeds=(31, 79), burnin=400, draws=6000,
        leaf_rows=8)
    expected_q, expected_u, expected_precision = exact_collapsed(nu, y, common, waveforms)
    observed = np.zeros((2, 2))
    for state in samples.targets.reshape(-1, 2):
        observed[tuple(state)] += 1/samples.targets.shape[0]/samples.targets.shape[1]
    np.testing.assert_allclose(observed, expected_q, atol=.025)
    assert samples.coefficients[:, :, 0].mean() == pytest.approx(expected_u, abs=.035)
    assert samples.precisions.mean() == pytest.approx(expected_precision, abs=.07)
    assert samples.record_scalar_rows == (12,) and samples.raw_prefix_rows == 12
    if np.isposinf(nu):
        np.testing.assert_array_equal(samples.precisions, np.ones_like(samples.precisions))


def test_rb_current_q_and_fixed_tail_are_proper_without_indicator_counts():
    base, axes, _, _, _ = problem()
    samples = run_joint_history_gibbs(base, axes, response_width=1, degrees_of_freedom=4.,
        initial_coefficients=[[0., 0.], [1., 1.]], initial_precisions=[[1.], [1.]],
        initial_targets=[[0, 0], [1, 1]], seeds=(49, 93), burnin=20, draws=32,
        leaf_rows=8)
    # Tail-only axis also retains its complete prior in a consistent chain.
    tail = np.array([[.15, .4, .1], [.25, .4, -.3]])
    tail_images = tuple(SparseResponseImage(0, np.column_stack((sign*np.array([.3, .2]),
                                                             .7*sign*np.array([.3, .2]))))
                        for sign in (-1., 1.))
    current = DiscreteTargetAxis("1", ("a", "b"), [.4, .6], axes[1].prefix_images, tail_images)
    lp, lt = rao_blackwell_tail_components(base, tail, (axes[0], current), samples,
        current_axis=1, response_width=1, tail_noise_logdet=0.)
    np.testing.assert_allclose(logsumexp(lp, axis=2), 0., atol=1e-14)
    whole = summarize_joint_tail(lp, lt, current_prior=[.4, .6],
        scoring_law="whole_observed_current_q", batch_size=4)
    fixed = summarize_joint_tail(lp, lt, current_prior=[.4, .6],
        scoring_law="fixed_current_prior_conditional_tail", batch_size=4)
    assert whole["probability"].sum() == pytest.approx(1.)
    assert fixed["probability"].sum() == pytest.approx(1.)
    assert np.all(whole["prefix_current_q_probability"] > 0)
    assert np.max(abs(whole["probability"]-fixed["probability"])) > 1e-4
    assert whole["prefix_weights_frozen_before_tail"]


def test_two_actual_records_share_one_beta_prior_and_keep_two_scales():
    record0 = np.column_stack((np.ones(7), np.ones(7), np.zeros(7), np.arange(7)*.1))
    record1 = np.column_stack((np.ones(9), np.zeros(9), np.ones(9), np.arange(9)*.2))
    samples = run_joint_history_gibbs(np.vstack((record0, record1)), (), response_width=1,
        degrees_of_freedom=4., record_spans=((0, 7), (7, 16)),
        initial_coefficients=np.zeros((2, 3)), initial_precisions=np.ones((2, 2)),
        initial_targets=np.empty((2, 0), dtype=int), seeds=(451, 830), burnin=100, draws=1200)
    # Both records' intercepts absorb beta completely; its prior remains N(0,1).
    assert samples.coefficients[:, :, 0].mean() == pytest.approx(0., abs=.07)
    assert samples.coefficients[:, :, 0].var() == pytest.approx(1., abs=.1)
    assert samples.precisions.shape[-1] == 2 and samples.record_scalar_rows == (7, 9)
    # Collapsing each true record's intercept leaves 6 and 8 effective rows.
    energies = [np.sum((np.arange(7)*.1-.3)**2), np.sum((np.arange(9)*.2-.8)**2)]
    means = [(4+dimension)/(4+energy) for dimension, energy in zip((6, 8), energies, strict=True)]
    np.testing.assert_allclose(samples.precisions.mean(axis=(0, 1)), means, atol=.17)
