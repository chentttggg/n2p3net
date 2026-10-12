"""Targeted B05 full/prefix potential checks over stored temporal state."""
from __future__ import annotations

import numpy as np
import pytest

from research.channel_noise_mechanism import (
    DiffuseProjectionResolutionError,
    native_history_marginal_program,
)
from transfer.temporal_noise import TemporalNoise


def _operands():
    rng = np.random.default_rng(20261012)
    n, channels, width, prefix = 30, 2, 4, 16
    covariance = np.array([[1.0, 0.2], [0.2, 1.0]])
    temporal = TemporalNoise.from_parameters(
        np.zeros(0), innovation_variance=1.0, reference_id="structured-test",
        startup_mode="stationary_innovations")
    mean = rng.normal(size=(width, channels))
    samples = rng.normal(size=(n, channels))
    nuisance = np.column_stack((np.ones(n), np.linspace(-1.0, 1.0, n)))
    design = rng.normal(size=(n, width))
    return temporal, covariance, mean, samples, nuisance, design, prefix


def test_absolute_full_prefix_potentials_match_independent_channel_dense_qr():
    temporal, covariance, mean, samples, nuisance, design, prefix = _operands()
    pair = native_history_marginal_program(
        mean=mean, precision_root=np.eye(len(mean)), innovation_covariance=covariance,
        temporal_noise=temporal, samples=samples, nuisance=nuisance,
        conditioning_samples=prefix)(design)

    chol = np.linalg.cholesky(covariance)
    whitening = np.kron(np.eye(len(samples)), np.linalg.inv(chol))
    factor = np.kron(np.eye(len(mean)), chol)
    response = np.einsum("nk,kcr->ncr", design, factor.reshape(len(mean), 2, -1))
    response = response.reshape(-1, response.shape[-1])
    observed = (samples - design @ mean).reshape(-1)
    flat_z = np.zeros((len(samples), 2, 4))
    for channel in range(2):
        flat_z[:, channel, 2 * channel:2 * channel + 2] = nuisance
    flat_z = flat_z.reshape(-1, flat_z.shape[-1])
    matrix = np.column_stack((whitening @ response, whitening @ flat_z))
    proper = np.zeros((factor.shape[1], matrix.shape[1]))
    proper[:, :factor.shape[1]] = np.eye(factor.shape[1])
    right = whitening @ observed

    def dense(rows, values):
        q, root = np.linalg.qr(np.vstack((rows, proper)), mode="reduced")
        values = np.r_[values, np.zeros(len(proper))]
        remainder = values - q @ (q.T @ values)
        return float(-0.5 * (
            remainder @ remainder + 2 * np.log(np.abs(np.diag(root))).sum()
            + (len(values) - len(proper) - flat_z.shape[1]) * np.log(2 * np.pi)
        ) + np.log(np.abs(np.diag(whitening)).sum() * 0 + 1.0))

    # The AR0 spatial whitening determinant is -n*log|chol(Sigma)|.
    logdet = -len(samples) * np.log(np.diag(chol)).sum()
    full = dense(matrix, right) + logdet
    rows = prefix * 2
    pref = dense(matrix[:rows], right[:rows]) + prefix * (-np.log(np.diag(chol)).sum())
    np.testing.assert_allclose(pair.joint_log_marginal, full, atol=2e-11)
    np.testing.assert_allclose(pair.prefix_log_marginal, pref, atol=2e-11)
    np.testing.assert_allclose(pair.conditional_log_density, full - pref, atol=2e-11)
    assert pair.resolution["same_flat_measure"] == "original_physical_da_full_and_prefix"


def test_stored_temporal_owner_never_receives_n_by_n_whitening(monkeypatch):
    temporal, covariance, mean, samples, nuisance, design, prefix = _operands()
    original = type(temporal).whiten

    def guarded(self, values):
        assert values.shape[0] == len(samples)
        assert values.shape[0] != values.shape[1]
        return original(self, values)

    monkeypatch.setattr(type(temporal), "whiten", guarded)
    native_history_marginal_program(
        mean=mean, precision_root=np.eye(len(mean)), innovation_covariance=covariance,
        temporal_noise=temporal, samples=samples, nuisance=nuisance,
        conditioning_samples=prefix)(design)


def test_large_raw_intercept_drift_is_typed_unresolved():
    temporal, covariance, mean, samples, nuisance, design, prefix = _operands()
    drift = nuisance @ np.diag([0.9, 1.1])
    with pytest.raises(DiffuseProjectionResolutionError, match="resolution|unresolved"):
        native_history_marginal_program(
            mean=mean, precision_root=np.eye(len(mean)), innovation_covariance=covariance,
            temporal_noise=temporal, samples=samples + 1e6 * drift, nuisance=nuisance,
            conditioning_samples=prefix)(design)
