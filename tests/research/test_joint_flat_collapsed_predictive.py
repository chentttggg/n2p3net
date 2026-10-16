"""Raw covariance oracle for beta/all-drift Rao-Blackwell prediction."""
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp

from research.channel_noise_mechanism import FittedChannelNoise, NoiseGeometry
from research.joint_flat_chain import JointHistorySamples
from research.joint_flat_collapsed_predictive import collapsed_tail_components
from research.joint_flat_factor_history import FactoredTargetAxis
from research.joint_flat_history_builder import (
    prepare_shared_record,
    pulse_target_image,
    shared_history_kernel,
)


def test_collapsed_current_and_genuine_tail_only_axis_match_raw_covariance_shape9():
    c, prefix, p_stop, full_stop = 2, 4, 28, 36
    sigma = np.array([[1.1, .27], [.27, .7]])
    geometry = NoiseGeometry(('Cz', 'Pz'), 10., 'V', 'native', 'analytic-original', 2)
    noise = FittedChannelNoise.from_parameters([[.25], [.25]], sigma, geometry=geometry,
        conditioning_samples=prefix, reference_id='raw collapsed predictive shape9', family='shared')
    basis, mean = np.array([[.2], [.65], [.5], [.1]]), np.array([[.2, -.1], [.4, .3]])
    temporal_factor = np.array([[.45, .05], [0., .35]])
    params = SimpleNamespace(noise=noise, basis=basis, prior=SimpleNamespace(mean=mean.ravel()),
        hyperparameters=SimpleNamespace(relative_noise_shape=9.), digest='3'*64)
    source = SimpleNamespace(parameters=params, spatial_covariance_factor=noise._spatial_root,
        temporal_covariance_factor=temporal_factor, temporal_factor_errors=np.zeros_like(temporal_factor),
        canonical_factor_binding={'saved_prior_chart_relation_error': 0.}, digest='4'*64)
    kernel = shared_history_kernel(source)
    onsets = (6, 10, 25, 27, 30, 32)
    samples = np.column_stack((.2*np.sin(np.arange(full_stop)*.7), .15*np.cos(np.arange(full_stop)*.4)))
    full, _ = prepare_shared_record(kernel, samples, onsets, np.zeros(6, dtype=bool), record_id='native-original')
    past, _ = prepare_shared_record(kernel, samples[:p_stop], onsets, np.zeros(6, dtype=bool), record_id='native-original')
    axes = []
    for ordinal, candidates in enumerate(((25, 27), (30, 32))):
        pi = [.4, .6] if ordinal == 0 else [.55, .45]
        images_p = tuple(pulse_target_image(kernel, (onset,), stop=p_stop, empty_start=p_stop) for onset in candidates)
        images_t = tuple(pulse_target_image(kernel, (onset,), stop=full_stop, empty_start=full_stop) for onset in candidates)
        axes.append(FactoredTargetAxis(str(ordinal), ('q0', 'q1'), pi, images_p, images_t, 0))
    assert all(image.stop == image.start for image in axes[1].prefix_images)
    assert any(image.stop > p_stop for image in axes[1].tail_images)
    precisions = np.tile(np.array([.7, 1.3]*4)[None, :, None], (2, 1, 1))
    targets = np.empty((2, 8, 2), dtype=np.int64)
    targets[:, :, 0], targets[:, :, 1] = 0, np.array([0, 1]*4)
    retained = JointHistorySamples(np.zeros((2, 8, 8)), precisions, targets, (21, 37), 0, 1, 4.,
                                  p_stop*c, 0., 0, (p_stop*c,))
    offset = p_stop-prefix
    noise_logdet = 2*(full_stop-p_stop)*np.log(np.diag(noise._spatial_root)).sum()
    noise_logdet += (full_stop-p_stop)*c*np.log(9.)
    actual_w, actual_f = collapsed_tail_components((past,), full.tail_rows[offset:], full.tail_errors[offset:],
        axes, retained, current_axis=0, tail_native_start=p_stop, tail_noise_logdet=noise_logdet,
        tail_noise_logdet_error=1e-13, leaf_rows=8, cache_bytes=65536)
    raw_cov = np.kron(.25**np.abs(np.arange(full_stop)[:, None]-np.arange(full_stop)[None, :])/(1-.25**2), sigma)*9.
    factor = np.kron(temporal_factor, np.linalg.cholesky(sigma))
    drift = np.kron(np.column_stack((np.ones(full_stop), np.arange(full_stop)/10.)), 3*np.linalg.cholesky(sigma))
    def marginal(count, q, other, lam):
        d = np.zeros((full_stop, 2))
        for onset in onsets:
            d[onset:onset+len(basis), 0] += basis[:, 0]
        for onset in (25 if q == 0 else 27, 30 if other == 0 else 32):
            d[onset:onset+len(basis), 1] += basis[:, 0]
        root = np.linalg.cholesky(raw_cov[:count*c, :count*c])
        design = np.kron(d[:count], np.eye(c))
        a = np.linalg.solve(root, np.column_stack((design@factor, drift[:count*c])))
        y = np.linalg.solve(root, (samples[:count]-d[:count]@mean).ravel())
        rows = np.vstack((np.sqrt(lam)*a, np.column_stack((np.eye(4), np.zeros((4, 4))))))
        rhs = np.r_[np.sqrt(lam)*y, np.zeros(4)]
        augmented = np.linalg.qr(np.column_stack((rows, rhs)), mode='r')
        return -.5*augmented[8, 8]**2-np.log(np.abs(np.diag(augmented[:8, :8]))).sum()
    reference_w, reference_f = np.empty_like(actual_w), np.empty_like(actual_f)
    for chain in range(2):
        for draw in range(8):
            lam, other = precisions[chain, draw, 0], targets[chain, draw, 1]
            values_p = np.array([marginal(p_stop, q, other, lam) for q in range(2)])
            values_f = np.array([marginal(full_stop, q, other, lam) for q in range(2)])
            logits = np.log([.4, .6])+values_p
            reference_w[chain, draw] = logits-logsumexp(logits)
            reference_f[chain, draw] = values_f-values_p-.5*((full_stop-p_stop)*c*(np.log(2*np.pi)-np.log(lam))+noise_logdet)
    np.testing.assert_allclose(actual_w, reference_w, atol=3e-13)
    np.testing.assert_allclose(actual_f, reference_f, atol=3e-13)
    assert np.max(np.abs(actual_f[:, 1::2]-actual_f[:, ::2])) > .1
