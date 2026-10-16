"""Original raw-covariance oracle for the compact full-native history kernel."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from research.channel_noise_mechanism import FittedChannelNoise, NoiseGeometry
from research.joint_flat_factor_history import (
    FactoredQRTree,
    FactoredTargetAxis,
    run_factored_history_gibbs,
)
from research.joint_flat_history_builder import (
    prepare_shared_record,
    pulse_target_image,
    shared_history_kernel,
)


def compact_problem():
    n, prefix = 28, 4
    sigma = np.array([[1.1, .27], [.27, .7]])
    geometry = NoiseGeometry(('Cz', 'Pz'), 10., 'V', 'native', 'analytic-original', 2)
    noise = FittedChannelNoise.from_parameters([[.25], [.25]], sigma, geometry=geometry,
        conditioning_samples=prefix, reference_id='independent raw AR covariance', family='shared')
    basis = np.array([[.2], [.65], [.5], [.1]])
    mean = np.array([[.2, -.1], [.4, .3]])
    temporal_factor = np.array([[.45, .05], [0., .35]])
    spatial = noise._spatial_root
    params = SimpleNamespace(noise=noise, basis=basis, prior=SimpleNamespace(mean=mean.ravel()),
        hyperparameters=SimpleNamespace(relative_noise_shape=1.), digest='1'*64)
    source = SimpleNamespace(parameters=params, spatial_covariance_factor=spatial,
        temporal_covariance_factor=temporal_factor, temporal_factor_errors=np.zeros_like(temporal_factor),
        canonical_factor_binding={'saved_prior_chart_relation_error': 0.}, digest='2'*64)
    kernel = shared_history_kernel(source)
    onsets = np.array([6, 10, 17, 21])
    samples = np.column_stack((.2*np.sin(np.arange(n)*.7), .15*np.cos(np.arange(n)*.4)))
    record, _ = prepare_shared_record(kernel, samples, onsets, np.zeros(4, dtype=bool), record_id='original-record')
    p = tuple(pulse_target_image(kernel, events, stop=n, empty_start=n) for events in ((17,), (21,)))
    empty = tuple(pulse_target_image(kernel, (), stop=n, empty_start=n) for _ in range(2))
    axis = FactoredTargetAxis('actual-current', ('q0', 'q1'), [.4, .6], p, empty, 0)
    return record, axis, samples, onsets, basis, mean, temporal_factor, sigma


def test_nonzero_original_pulse_leaf_prediction_and_point_energy_agree():
    record, axis, *_ = compact_problem()
    image = axis.prefix_images[0]
    beta, drift = np.array([[.2, -.3], [.7, .1]]), np.array([[.05, -.1], [.02, .03]])
    values, _ = image.block(record.startup_native_rows, record.native_rows)
    actual, _ = image.prediction(record.startup_native_rows, record.native_rows, beta)
    assert np.linalg.norm(values) > 0
    np.testing.assert_allclose(actual, values[:, :2]@beta+values[:, 2:], atol=2e-16)
    tree = FactoredQRTree(record, (axis,), leaf_rows=5, cache_bytes=16384)
    tree.set_target(0, 0)
    root, count = tree.root.copy(), tree.qr_rebuilds
    tree.set_target(0, 1)
    tree.set_target(0, 0)
    np.testing.assert_array_equal(root, tree.root)
    assert tree.cache_hits > 0 and tree.qr_rebuilds < count+2*tree.offset
    residual, _ = tree.residual(beta, drift)
    startup = record.startup_rows@np.r_[drift.ravel(), -1.]
    raw = float(np.linalg.norm(residual)**2+startup@startup)
    assert tree.energy(beta, drift, precision=1.3) == pytest.approx(raw, abs=3e-15)


def raw_integrated_oracle(nu, samples, onsets, basis, mean, temporal_factor, sigma):
    n, c = samples.shape
    covariance = np.kron(.25**np.abs(np.arange(n)[:, None]-np.arange(n)[None, :])/(1-.25**2), sigma)
    raw_root = np.linalg.cholesky(covariance)
    factor = np.kron(temporal_factor, np.linalg.cholesky(sigma))
    physical_drift = np.kron(np.column_stack((np.ones(n), np.arange(n)/10.)), np.linalg.cholesky(sigma))
    components = []
    for q in range(2):
        d = np.zeros((n, 2))
        for onset in onsets:
            d[onset:onset+len(basis), 0] += basis[:, 0]
        onset = onsets[2+q]
        d[onset:onset+len(basis), 1] += basis[:, 0]
        design = np.kron(d, np.eye(c))
        a = np.linalg.solve(raw_root, np.column_stack((design@factor, physical_drift)))
        y = np.linalg.solve(raw_root, (samples-d@mean).ravel())
        def law(lam, a=a, y=y):
            prior = np.column_stack((np.eye(4), np.zeros((4, 4))))
            full = np.vstack((np.sqrt(lam)*a, prior))
            yy = np.r_[np.sqrt(lam)*y, np.zeros(4)]
            root = np.linalg.qr(np.column_stack((full, yy)), mode='r')
            r, rhs = root[:8, :8], root[:8, 8]
            value = np.linalg.solve(r, rhs)
            logmass = .5*n*c*np.log(lam)-np.log(np.abs(np.diag(r))).sum()-.5*root[8, 8]**2
            if not np.isposinf(nu):
                logmass += (nu/2-1)*np.log(lam)-nu*lam/2
            return logmass, value
        if np.isposinf(nu):
            logmass, value = law(1.)
            components.append((logmass+np.log([.4, .6][q]), value, 1.))
        else:
            center = law(1.)[0]
            def integrand(t, coordinate=None, center=center, law=law):
                lam = np.exp(t)
                logmass, value = law(lam)
                weight = np.exp(logmass-center+t)
                return weight*(1 if coordinate is None else lam if coordinate == 8 else value[coordinate])
            mass = quad(integrand, -12., 8., epsabs=2e-10)[0]
            value = np.array([quad(lambda t, i=i: integrand(t, i), -12., 8., epsabs=2e-10)[0]/mass for i in range(8)])
            expected_lambda = quad(lambda t: integrand(t, 8), -12., 8., epsabs=2e-10)[0]/mass
            components.append((center+np.log(mass)+np.log([.4, .6][q]), value, expected_lambda))
    weights = np.exp(np.array([item[0] for item in components])-logsumexp([item[0] for item in components]))
    return weights, sum(w*item[1] for w, item in zip(weights, components, strict=True)), sum(
        w*item[2] for w, item in zip(weights, components, strict=True))


@pytest.mark.parametrize('nu', [4., np.inf])
def test_factored_native_startup_and_explicit_gamma_match_independent_raw_covariance(nu):
    record, axis, samples, onsets, basis, mean, temporal_factor, sigma = compact_problem()
    expected_q, expected_theta, expected_lambda = raw_integrated_oracle(nu, samples, onsets, basis, mean, temporal_factor, sigma)
    retained = run_factored_history_gibbs((record,), (axis,), degrees_of_freedom=nu,
        initial_coefficients=np.array([np.zeros((2, 2)), np.ones((2, 2))]),
        initial_drifts=[np.array([np.zeros((2, 2)), np.ones((2, 2))*.1])],
        initial_precisions=[[1.], [1.]], initial_targets=[[0], [1]], seeds=(503, 961),
        burnin=300, draws=3000, sweeps_per_draw=1, leaf_rows=8, cache_bytes=65536)
    actual_q = np.bincount(retained.targets.ravel(), minlength=2)/retained.targets.size
    np.testing.assert_allclose(actual_q, expected_q, atol=.035)
    np.testing.assert_allclose(retained.coefficients.mean(axis=(0, 1)), expected_theta, atol=.045)
    assert retained.precisions.mean() == pytest.approx(expected_lambda, rel=.025)
    assert retained.raw_prefix_rows == 56 and retained.record_scalar_rows == (56,)
