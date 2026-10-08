"""Independent p=2, C=3 Gaussian oracles for a frozen B09 -> B05 adapter."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from research.channel_noise_mechanism import FittedChannelNoise, FrozenERPResponse, NoiseGeometry
from transfer.noise_model import array_digest
from transfer.spatial_signal_prior import (
    SpatialLayout,
    SpatialObservation,
    SpatialResponseBlock,
    SpatialSignalPosterior,
    SpatialSignalPrior,
)


def _fixture(*, binding_changes=None):
    layout = SpatialLayout(("Fz", "Cz", "Pz"), "left earlobe", "a" * 64,
                           ("EEG",) * 3, "V")
    basis = np.array([[1.], [.3], [-.2]])
    binding = {"dictionary_sha256": "b" * 64, "actual_basis_sha256": array_digest(basis),
               "vectorization": "column_major_response_by_native_EEG", "sample_rate_hz": 128.,
               "operator_identity": "native_raw_identity"}
    if binding_changes:
        binding.update(binding_changes)
    prior = SpatialSignalPrior(layout,
        np.array([[.25, -.3, .5], [.1, .4, -.2]]),
        np.array([[1.4, .3], [.3, .7]]),
        np.array([[2., .4, -.1], [.4, 1.3, .2], [-.1, .2, .9]]),
        binding, {"role": "predeclared", "authority_sha256": "c" * 64})
    design = np.array([[1., 0.], [.2, 1.], [-.3, .4], [.7, -.5], [1.1, .6]])
    y = np.array([[.2, -.3, .5], [.4, .7, -.1], [.6, -.2, .8],
                  [-.1, .4, .3], [.5, .2, -.6]])
    noise_covariance = np.array([[1.7, .25, -.1], [.25, .8, .12], [-.1, .12, 1.2]])
    authorization = {"record_id": "participant-A:C", "fit_scope": "target_calibration",
        "source_sha256": "d" * 64, "dictionary_sha256": "b" * 64,
        "temporal_program_sha256": "e" * 64, "raw_sample_support": [100, 105],
        "original_view": {"participant_key": "participant-A"}}
    block = SpatialResponseBlock(y, design, SpatialObservation.native(layout), authorization)
    posterior = SpatialSignalPosterior.fit(prior, (block,),
        native_noise_covariance=noise_covariance, noise_binding={"authority_sha256": "f" * 64})
    geometry = NoiseGeometry(layout.channels, 128., "V", layout.acquisition_reference,
                             "native_raw_identity", 3)
    ledger = [{"record_id": authorization["record_id"], "participant_key": "participant-A",
               "recording_id": "physical-recording-A", "source_sha256": "d" * 64,
               "read_support": [100, 105], "fit_scope": "target_calibration"}]
    return posterior, geometry, ledger, design, y, noise_covariance


def _adapt(posterior, geometry, ledger):
    return FrozenERPResponse.from_spatial_posterior(posterior, geometry=geometry,
        source_model_sha256="1" * 64, response_id="frozen_B09", training_records=ledger)


def _permutation(rows, channels):
    """Explicit index definition, independent of the adapter reshape operation."""
    result = np.zeros((rows * channels, rows * channels))
    for row in range(rows):
        for channel in range(channels):
            result[row * channels + channel, channel * rows + row] = 1.
    return result


def _dense_b09_posterior(posterior, design, y, noise):
    prior = posterior.prior
    h = np.kron(prior.signal_covariance, prior.temporal_covariance)
    x = np.kron(np.eye(3), design)
    n = np.kron(noise, np.eye(len(y)))
    precision = np.linalg.solve(h, np.eye(6)) + x.T @ np.linalg.solve(n, x)
    rhs = np.linalg.solve(h, prior.mean.flatten(order="F"))
    rhs += x.T @ np.linalg.solve(n, y.flatten(order="F"))
    return np.linalg.solve(precision, rhs), np.linalg.solve(precision, np.eye(6))


def _private_ar1_covariance(phi, innovations, n):
    """Closed-form diagonal-lag VAR1; no production whitening/compiler call."""
    phi = np.asarray(phi)
    stationary = innovations / (1. - phi[:, None] * phi[None, :])
    covariance = np.zeros((3 * n, 3 * n))
    for t in range(n):
        for s in range(n):
            lag = np.diag(phi ** abs(t - s)) @ stationary
            covariance[3*t:3*t+3, 3*s:3*s+3] = lag if t >= s else lag.T
    return covariance


def _conditional_density(y, design, mean, covariance, noise, prefix):
    """Dense Schur conditional with prefix-estimated flat affine drift."""
    n = len(y)
    x = np.kron(design, np.eye(3))
    total = noise + x @ covariance @ x.T
    drift = np.column_stack((np.ones(n), (2*np.arange(n) - (prefix-1)) / (prefix-1)))
    z = np.kron(drift, np.eye(3))
    cut = prefix * 3
    pp, tp, tt = total[:cut, :cut], total[cut:, :cut], total[cut:, cut:]
    residual = y.ravel() - x @ mean.ravel()
    zsolve = np.linalg.solve(pp, z[:cut])
    drift_covariance = np.linalg.solve(z[:cut].T @ zsolve, np.eye(6))
    prefix_solve = np.linalg.solve(pp, residual[:cut])
    drift_mean = drift_covariance @ z[:cut].T @ prefix_solve
    remaining_drift = z[cut:] - tp @ zsolve
    conditional_mean = tp @ prefix_solve + remaining_drift @ drift_mean
    conditional_covariance = tt - tp @ np.linalg.solve(pp, tp.T)
    conditional_covariance += remaining_drift @ drift_covariance @ remaining_drift.T
    centered = residual[cut:] - conditional_mean
    return float(-.5 * (centered @ np.linalg.solve(conditional_covariance, centered)
        + np.linalg.slogdet(conditional_covariance)[1] + (n-prefix)*3*np.log(2*np.pi)))


def _held():
    rng = np.random.default_rng(20260918)
    return rng.normal(size=(11, 3)), rng.normal(size=(3, 11, 2))


@pytest.mark.parametrize("family,phi", [("shared", [.35]*3), ("private", [.65, -.25, .12])])
def test_closed_form_2_by_3_direction_before_adapter_implementation(family, phi):
    """Existing B05 consumer against independent B09 and VAR1 dense laws."""
    posterior, geometry, ledger, design, y_c, sigma_c = _fixture()
    mean_f, covariance_f = _dense_b09_posterior(posterior, design, y_c, sigma_c)
    permutation = _permutation(2, 3)
    covariance_c = permutation @ covariance_f @ permutation.T
    # This mathematical input is built from the independent dense posterior,
    # not from the proposed precision-factor conversion or producer covariance.
    response = FrozenERPResponse(mean_f.reshape(2, 3, order="F"),
        np.linalg.cholesky(covariance_c), geometry, "1"*64, "dense_B09_posterior",
        {"fit_scope": "target_calibration", "records": ledger})
    samples, designs = _held()
    innovations = np.array([[1.1, .25, -.12], [.25, 1.6, .3], [-.12, .3, .9]])
    noise = FittedChannelNoise.from_parameters(np.asarray(phi)[:, None], innovations,
        geometry=geometry, conditioning_samples=4, reference_id="known_VAR1", family=family)
    scored = noise.score(samples, designs, response, candidate_ids=("A", "B", "C"), geometry=geometry,
        record_id="held-B", recording_id="physical-recording-A", sample_start=200)
    expected = [_conditional_density(samples, d, response.mean, covariance_c,
        _private_ar1_covariance(phi, innovations, len(samples)), 4) for d in designs]
    np.testing.assert_allclose(scored["log_density"], expected, rtol=2e-12, atol=3e-11)


def test_general_posterior_factor_and_design_permutation_preserve_same_gaussian():
    posterior, geometry, ledger, design, y, noise = _fixture()
    response = _adapt(posterior, geometry, ledger)
    mean_f, covariance_f = _dense_b09_posterior(posterior, design, y, noise)
    p = _permutation(2, 3)
    np.testing.assert_allclose(posterior.mean.flatten(order="F"), mean_f, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(posterior.covariance, covariance_f, rtol=2e-12, atol=2e-14)
    np.testing.assert_array_equal(response.mean, posterior.mean)
    np.testing.assert_allclose(response.mean.ravel(), p @ mean_f, rtol=2e-12, atol=2e-14)
    actual_covariance = response.coefficient_factor @ response.coefficient_factor.T
    np.testing.assert_allclose(actual_covariance, p @ covariance_f @ p.T, rtol=3e-12, atol=2e-14)
    samples_c_from_f = _permutation(len(y), 3)
    np.testing.assert_array_equal(np.kron(design, np.eye(3)) @ p,
                                 samples_c_from_f @ np.kron(np.eye(3), design))
    receipt = response.provenance["spatial_posterior_conversion"]
    assert receipt["posterior_artifact_sha256"] == posterior.record()["artifact_sha256"]
    assert receipt["source_order"] == "column_major_response_by_native_EEG"
    assert receipt["destination_order"] == "row_major_coefficient_then_observed_channel"
    assert receipt["noise_refitted"] is False
    assert json.loads(json.dumps(response.provenance["records"])) == ledger


@pytest.mark.parametrize("family,phi", [("shared", [.35]*3), ("private", [.65, -.25, .12])])
def test_general_frozen_spatial_response_scores_shared_and_private_noise_dense_oracle(family, phi):
    posterior, geometry, ledger, *_ = _fixture()
    response = _adapt(posterior, geometry, ledger)
    y, designs = _held()
    innovation = np.array([[1.1, .25, -.12], [.25, 1.6, .3], [-.12, .3, .9]])
    noise = FittedChannelNoise.from_parameters(np.asarray(phi)[:, None], innovation,
        geometry=geometry, conditioning_samples=4, reference_id="known_VAR1", family=family)
    frozen_before = posterior.record()
    result = noise.score(y, designs, response, candidate_ids=("A", "B", "C"), geometry=geometry,
                         record_id="held-B", recording_id="physical-recording-A", sample_start=200)
    p = _permutation(2, 3)
    dense_covariance = p @ posterior.covariance @ p.T
    temporal_spatial = _private_ar1_covariance(phi, innovation, len(y))
    expected = [_conditional_density(y, d, posterior.mean, dense_covariance, temporal_spatial, 4)
                for d in designs]
    np.testing.assert_allclose(result["log_density"], expected, rtol=2e-12, atol=3e-11)
    assert result["truth_used"] is False and result["parameters_updated"] is False
    assert posterior.record() == frozen_before
    if family == "private":
        stationary = temporal_spatial[:3, :3]
        lag_map = temporal_spatial[3:6, :3] @ np.linalg.solve(stationary, np.eye(3))
        np.testing.assert_allclose(lag_map, np.diag(phi), rtol=1e-13, atol=1e-14)
        assert np.linalg.norm(lag_map - np.trace(lag_map)/3*np.eye(3)) > .5


def test_omitting_order_conversion_or_transposing_inverse_changes_prediction():
    posterior, geometry, ledger, *_ = _fixture()
    response = _adapt(posterior, geometry, ledger)
    p = _permutation(2, 3)
    r = posterior.precision_root
    correct = p @ posterior.covariance @ p.T
    # R^T R is precision: R^-1 R^-T is covariance. R^-T used as a
    # covariance factor gives R^-T R^-1, which is a different model.
    inverse_transposed = np.linalg.solve(r.T, np.eye(6))
    wrong_orientation = p @ inverse_transposed @ inverse_transposed.T @ p.T
    assert np.linalg.norm(correct - posterior.covariance) > .1
    assert np.linalg.norm(correct - wrong_orientation) > .01
    y, designs = _held()
    noise = _private_ar1_covariance([.65, -.25, .12], np.eye(3), len(y))
    honest = np.array([_conditional_density(y, d, response.mean, correct, noise, 4) for d in designs])
    wrong_order = np.array([_conditional_density(y, d, response.mean, posterior.covariance, noise, 4)
                            for d in designs])
    wrong_root = np.array([_conditional_density(y, d, response.mean, wrong_orientation, noise, 4)
                           for d in designs])
    assert np.max(np.abs(honest - wrong_order)) > .01
    assert np.max(np.abs(honest - wrong_root)) > .001


@pytest.mark.parametrize("field,value", [
    ("source_sha256", "9"*64), ("read_support", [100, 106]),
    ("record_id", "different-C"), ("participant_key", "different-person"),
    ("fit_scope", "source_training"),
])
def test_conversion_cannot_invent_training_authority(field, value):
    posterior, geometry, ledger, *_ = _fixture()
    bad = copy.deepcopy(ledger)
    bad[0][field] = value
    with pytest.raises(ValueError, match="training|producer|support|participant|scope"):
        _adapt(posterior, geometry, bad)


@pytest.mark.parametrize("changes", [
    {"sfreq": 256.}, {"unit": "uV"}, {"reference": "right earlobe"},
    {"channels": ("Pz", "Cz", "Fz")}, {"operator_identity": "different_operator"},
])
def test_conversion_preserves_actual_physical_response_coordinates(changes):
    posterior, geometry, ledger, *_ = _fixture()
    with pytest.raises(ValueError, match="physical|geometry|sampling|operator|layout"):
        _adapt(posterior, replace(geometry, **changes), ledger)


def test_consumer_rejects_held_overlap_with_converted_response_training_support():
    posterior, geometry, ledger, *_ = _fixture()
    response = _adapt(posterior, geometry, ledger)
    noise = FittedChannelNoise.from_parameters(np.zeros((3, 1)), np.eye(3), geometry=geometry,
        conditioning_samples=4, reference_id="known_shared", family="shared")
    y, designs = _held()
    with pytest.raises(ValueError, match="overlaps the frozen ERP"):
        noise.score(y, designs, response, candidate_ids=("A", "B", "C"), geometry=geometry,
                    record_id="different-held-name", recording_id="physical-recording-A", sample_start=103)
