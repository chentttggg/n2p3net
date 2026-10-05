"""Behavioral checks against dense, unwhitened Gaussian reference models."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import null_space, toeplitz
from scipy.signal import lfilter
from scipy.special import logsumexp

from transfer.continuous_decision import (
    CalibrationRecord,
    ContinuousDecisionModel,
    ContinuousDecisionResult,
)
from transfer.noise_model import array_digest, metadata_digest
from transfer.temporal_noise import TemporalNoise


def _dense_time_covariance(coefficients, n_samples):
    # Independent impulse-response sum; neither production whitening nor its
    # PARCOR/startup factor is used to obtain the expected covariance.
    impulse = np.zeros(20000)
    impulse[0] = 1.
    response = lfilter([1.], np.r_[1., -np.asarray(coefficients)], impulse)
    return toeplitz([response[:-lag] @ response[lag:] if lag else response @ response
                     for lag in range(n_samples)])


def _convolution_design(onsets, labels, n_samples, basis):
    impulse = np.zeros(n_samples)
    impulse[onsets] = 1.
    target = np.zeros(n_samples)
    target[onsets] = labels
    return np.column_stack([np.convolve(events, column)[:n_samples]
                            for events in (impulse, target) for column in basis.T])


def _dense_evidence(samples, onsets, membership, model):
    n_samples, n_channels = samples.shape
    complement = null_space(np.column_stack([
        np.ones(n_samples), np.linspace(-1., 1., n_samples),
    ]).T)
    time = _dense_time_covariance(model.noise.coefficients, n_samples)
    evidence = []
    for candidate in membership.T:
        design = _convolution_design(onsets, candidate, n_samples, model.basis)
        residual = complement.T @ (samples - design @ model.kernel_mean)
        covariance = np.kron(
            model.innovation_covariance,
            complement.T @ (time + design @ model.kernel_row_covariance @ design.T) @ complement,
        )
        vector = residual.ravel(order="F")
        evidence.append(-.5 * (vector @ np.linalg.solve(covariance, vector)
                               + np.linalg.slogdet(covariance)[1]))
        assert covariance.shape == ((n_samples - 2) * n_channels,) * 2
    return np.asarray(evidence)


def _records(n_samples=850, dtype=np.float64):
    rng = np.random.default_rng(6017)
    basis = np.array([[0., .1], [.3, .4], [1., .5], [.8, .1], [.2, 0.]])
    onsets = np.arange(6, n_samples - len(basis), 17, dtype=np.int32)
    kernel = np.array([[.3, -.2], [.1, .2], [1.3, .7], [.2, -.3]])
    records = []
    for index in range(3):
        labels = (np.arange(len(onsets)) + index) % 3 == 0
        white = rng.normal(0, .3, (n_samples + 200, 2))
        signal = lfilter([1.], [1., -.7, .2], white, axis=0)[200:]
        signal += _convolution_design(onsets, labels, n_samples, basis) @ kernel
        signal += np.linspace(-.5 + index, .6 - index, n_samples)[:, None]
        records.append(CalibrationRecord(f"cal-{index}", signal.astype(dtype), onsets, labels))
    return records, basis


def _settings(basis):
    return dict(sfreq=50., channels=("Fz", "Pz"), candidate_ids=("a", "b", "c"),
                operator_identity="native-unfiltered-research", basis=basis, ridge=1.)


@pytest.mark.parametrize("order", [0, 1, 4, 8, 16, 32])
def test_explicit_ar_fit_uses_segmented_owner_and_exact_startup(order, tmp_path):
    records, basis = _records(dtype=np.float32)
    model = ContinuousDecisionModel.fit(records, **_settings(basis), noise_order=order,
                                        fit_scope="source_training")
    assert model.noise_order == order
    assert model.phi is None
    assert model.noise.order == order
    assert model.noise.startup_mode == "stationary_innovations"
    assert model.noise.provenance["fit_scope"] == "source_training"
    assert model.noise.provenance["available_lag_rows"] == sum(len(r.samples)-order for r in records)
    assert model.provenance["kernel_prior_scale"] == "innovation"
    assert model.provenance["compute_dtype"] == np.dtype(np.float64).str
    for record, ledger in zip(records, model.provenance["records"], strict=True):
        assert ledger["read_sample_support"] == (0, len(record.samples))
        assert ledger["input_dtypes"]["samples"] == record.samples.dtype.str
        assert ledger["raw_input_sha256"] == array_digest(record.samples, record.onsets, record.target_mask)
    probe = records[0].samples + .037
    membership = np.eye(3, dtype=bool)[np.arange(len(records[0].onsets)) % 3]
    options = dict(sfreq=50., channels=("Fz", "Pz"), candidate_ids=("a", "b", "c"),
                   operator_identity=model.operator_identity, record_id="held-out")
    initial_state = metadata_digest(model._metadata())
    result = model.score(probe, records[0].onsets, membership, **options)
    model.score(probe[::-1].copy(), records[0].onsets, membership, **{**options, "record_id": "other"})
    repeated = model.score(probe, records[0].onsets, membership, **options)
    np.testing.assert_array_equal(result.log_evidence, repeated.log_evidence)
    assert metadata_digest(model._metadata()) == initial_state
    assert result.n_unique_samples == len(probe)
    path = tmp_path / f"ar{order}.npz"
    model.save(path)
    loaded = ContinuousDecisionModel.load(path)
    np.testing.assert_array_equal(loaded.noise.reflection_coefficients, model.noise.reflection_coefficients)
    restored = loaded.score(probe, records[0].onsets, membership, **options)
    np.testing.assert_array_equal(restored.log_probability, result.log_probability)
    assert not loaded.noise.coefficients.flags.writeable
    assert records[0].samples.flags.writeable


@pytest.mark.parametrize("coefficients", [[], [.7], [.7, -.2, .08, 0., 0., .03, -.01, .02]])
def test_multichannel_candidate_density_matches_dense_gaussian_with_diffuse_nuisance(coefficients):
    basis = np.array([[.1, 0.], [1., .2], [.5, 1.], [0., .3]])
    mean = np.array([[.2, -.4], [.1, .3], [.8, .6], [.3, -.2]])
    row = np.diag([.3, .4, .2, .6]) + .04
    spatial = np.array([[.8, .2], [.2, .5]])
    noise = TemporalNoise.from_parameters(coefficients, innovation_variance=13.,
                                         reference_id="dense_convolution_reference",
                                         startup_mode="stationary_innovations")
    model = ContinuousDecisionModel(basis, mean, row, spatial, None, 50., ("Fz", "Pz"),
                                    ("a", "b", "c"), "native", {
                                        "fit_scope": "analytic_reference",
                                        "reference_id": "dense_convolution_reference", "records": [],
                                    }, noise=noise)
    onsets = np.array([2, 6, 9, 13, 17, 22])
    membership = np.eye(3, dtype=bool)[[0, 1, 2, 2, 1, 0]]
    samples = np.random.default_rng(339).normal(size=(29, 2))
    options = dict(operator_identity="native", sfreq=50., channels=("Fz", "Pz"),
                   candidate_ids=("a", "b", "c"), record_id="held-out")
    result = model.score(samples, onsets, membership, **options)
    expected = _dense_evidence(samples, onsets, membership, model)
    np.testing.assert_allclose(result.log_probability, expected-logsumexp(expected), rtol=0, atol=2e-12)
    # Shared diffuse nuisance is removed in the same *noise* geometry as D/Y.
    drifted = samples + np.column_stack([np.linspace(-100, 100, len(samples)),
                                        np.linspace(17, -4, len(samples))])
    translated = model.score(drifted, onsets, membership, **options)
    np.testing.assert_allclose(translated.log_probability, result.log_probability, rtol=0, atol=2e-11)
    order = [2, 0, 1]
    vocabulary = tuple(model.candidate_ids[index] for index in order)
    permuted = replace(model, candidate_ids=vocabulary).score(
        samples, onsets, membership[:, order], **{**options, "candidate_ids": vocabulary})
    np.testing.assert_allclose(permuted.log_probability, result.log_probability[order], rtol=0, atol=2e-12)
    assert permuted.winner == result.winner
    scaled = replace(model, kernel_mean=mean*1e6, innovation_covariance=spatial*1e12)
    in_microvolts = scaled.score(samples*1e6, onsets, membership, **options)
    np.testing.assert_allclose(in_microvolts.log_probability, result.log_probability, rtol=0, atol=3e-12)


def test_ar8_fit_posterior_matches_dense_gls_and_response_basis_gauge():
    records, basis = _records(n_samples=90)
    options = _settings(basis)
    model = ContinuousDecisionModel.fit(records, **options, noise_order=8)
    gram = np.zeros((4, 4))
    cross = np.zeros((4, 2))
    for record in records:
        design = _convolution_design(record.onsets, record.target_mask, len(record.samples), basis)
        complement = null_space(np.column_stack([
            np.ones(len(record.samples)), np.linspace(-1., 1., len(record.samples)),
        ]).T)
        covariance = complement.T @ _dense_time_covariance(model.noise.coefficients, len(record.samples)) @ complement
        D, Y = complement.T @ design, complement.T @ record.samples
        gram += D.T @ np.linalg.solve(covariance, D)
        cross += D.T @ np.linalg.solve(covariance, Y)
    posterior = np.linalg.inv(gram + np.eye(4))
    np.testing.assert_allclose(model.kernel_row_covariance, posterior, rtol=0, atol=3e-13)
    np.testing.assert_allclose(model.kernel_mean, posterior @ cross, rtol=0, atol=3e-13)
    # Same physical response/prior in a rescaled coefficient coordinate.
    multiplier = 7.
    transformed = ContinuousDecisionModel.fit(records, **{**options, "basis": basis*multiplier,
                                                          "ridge": multiplier**2}, noise_order=8)
    np.testing.assert_allclose(transformed.noise.coefficients, model.noise.coefficients, rtol=0, atol=3e-13)
    np.testing.assert_allclose(transformed.kernel_mean*multiplier, model.kernel_mean, rtol=0, atol=3e-13)
    np.testing.assert_allclose(transformed.kernel_row_covariance*multiplier**2,
                               model.kernel_row_covariance, rtol=0, atol=3e-13)
    np.testing.assert_allclose(transformed.innovation_covariance, model.innovation_covariance, rtol=0, atol=3e-13)


def test_log_probability_survives_probability_underflow():
    result = ContinuousDecisionResult(("a", "b", "c"), np.array([0., -1000., -1100.]),
                                      np.array([1., 0., 0.]), "a", "ok", 20, "0"*64)
    np.testing.assert_array_equal(result.log_probability, [0., -1000., -1100.])
    assert -result.log_probability[1] == 1000.
    assert result.record()["log_probability"] == [0., -1000., -1100.]


def test_legacy_default_keeps_original_clipped_ar1_numerics_and_saved_schema(tmp_path):
    rng = np.random.default_rng(44)
    basis = np.exp(-.5*((np.arange(20)-9)/3)**2)[:, None]
    onsets = np.arange(30, 231, 25)
    records = []
    for index in range(4):
        labels = np.arange(len(onsets)) % 3 == index % 3
        samples = rng.normal(0, .2, (280, 1))
        samples += _convolution_design(onsets, labels, len(samples), basis) @ np.array([[.5], [3.]])
        records.append(CalibrationRecord(f"cal-{index}", samples, onsets, labels))
    model = ContinuousDecisionModel.fit(records, sfreq=50., channels=("Cz",),
                                        candidate_ids=("a", "b", "c"), operator_identity="legacy", basis=basis)
    assert model.noise is None
    assert model.noise_order == 1
    assert model.phi == pytest.approx(-.058261736994349105, rel=0, abs=2e-15)
    np.testing.assert_allclose(model.kernel_mean, [[.5262718018379688], [2.9704207980705686]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(model.kernel_row_covariance,
                               [[.009342073839397665, -.006889312252949218],
                                [-.006889312252949218, .02082007415494376]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(model.innovation_covariance, [[.039808825583212616]], rtol=0, atol=1e-14)
    path = tmp_path / "legacy.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        assert json.loads(archive["metadata"].item())["schema"] == "n2p3_continuous_decision_model/2"
    loaded = ContinuousDecisionModel.load(path)
    assert loaded.phi == model.phi and loaded.noise is None


def test_ar_fit_refuses_unsupported_training_scope_order_and_insufficient_support():
    records, basis = _records(n_samples=50)
    with pytest.raises(ValueError, match="scope"):
        ContinuousDecisionModel.fit(records, **_settings(basis), noise_order=8, fit_scope="target_test")
    with pytest.raises(ValueError, match="lag rows"):
        ContinuousDecisionModel.fit(records, **_settings(basis), noise_order=32)
    with pytest.raises(ValueError, match="order"):
        ContinuousDecisionModel.fit(records, **_settings(basis), noise_order=8.5)


@pytest.mark.parametrize("mutation", ["scope", "records", "source", "dtype", "support"])
def test_ar_reload_requires_noise_scope_source_support_and_exact_dtype(mutation, tmp_path):
    records, basis = _records(n_samples=90)
    model = ContinuousDecisionModel.fit(records, **_settings(basis), noise_order=8)
    original = tmp_path / "ar8.npz"
    model.save(original)
    with np.load(original, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    metadata = json.loads(payload["metadata"].item())
    if mutation == "scope":
        metadata["provenance"]["fit_scope"] = "source_training"
    elif mutation == "records":
        metadata["provenance"]["records"][0]["record_id"] = "other"
    elif mutation == "source":
        metadata["provenance"]["implementation_sha256"] = "0" * 64
    elif mutation == "support":
        metadata["provenance"]["records"][0]["read_sample_support"][1] -= 1
    else:
        payload["basis"] = payload["basis"].astype(np.float32)
        metadata["state_sha256"] = array_digest(*[payload[key] for key in (
            "basis", "kernel_mean", "kernel_row_covariance", "innovation_covariance",
        )])
        metadata["state_dtypes"]["basis"] = np.dtype(np.float32).str
    metadata.pop("artifact_sha256")
    metadata["artifact_sha256"] = metadata_digest(metadata)
    payload["metadata"] = np.asarray(json.dumps(metadata))
    changed = tmp_path / "changed.npz"
    np.savez(changed, **payload)
    with pytest.raises(ValueError):
        ContinuousDecisionModel.load(changed)


def test_new_ar_calibration_censoring_does_not_become_score_authorization():
    records, basis = _records(n_samples=90)
    censored = [replace(record, samples=record.samples[:-3], right_censored=True) for record in records]
    model = ContinuousDecisionModel.fit(censored, **_settings(basis), noise_order=8)
    assert all(record["read_sample_support"] == (0, 87) for record in model.provenance["records"])
    onsets = np.array([6, 23, 40, 57, 74, 84])
    with pytest.raises(ValueError, match="complete response tails"):
        model.score(censored[0].samples + .01, onsets, np.eye(3)[np.arange(6) % 3],
                    operator_identity=model.operator_identity, sfreq=50., channels=("Fz", "Pz"),
                    candidate_ids=("a", "b", "c"), record_id="held")


def test_censored_ar_fit_does_not_read_unavailable_future_in_a_backing_buffer():
    records, basis = _records(n_samples=90)
    buffers = [np.vstack([record.samples, np.full((300, 2), 10000.)]) for record in records]
    prefix = [replace(record, samples=buffer[:77], right_censored=True)
              for record, buffer in zip(records, buffers, strict=True)]
    before = ContinuousDecisionModel.fit(prefix, **_settings(basis), noise_order=8)
    assert all(ledger["right_censored_events"] == 1 for ledger in before.provenance["records"])
    for buffer in buffers:
        buffer[77:] = np.random.default_rng(579).normal(0, 1e7, buffer[77:].shape)
    after = ContinuousDecisionModel.fit(prefix, **_settings(basis), noise_order=8)
    for key in ("kernel_mean", "kernel_row_covariance", "innovation_covariance"):
        np.testing.assert_array_equal(getattr(before, key), getattr(after, key))
    assert before._metadata() == after._metadata()


def test_scored_underflow_keeps_finite_dense_gaussian_nll():
    basis = np.array([[0.], [1.], [.5]])
    noise = TemporalNoise.from_parameters([.4, -.1, .03, 0., 0., 0., .02, .01],
                                         innovation_variance=7., reference_id="underflow_oracle",
                                         startup_mode="stationary_innovations")
    model = ContinuousDecisionModel(basis, np.array([[.2], [100.]]), np.eye(2)*1e-8,
                                    np.array([[.4]]), None, 10., ("Cz",), ("a", "b"), "native", {
                                        "fit_scope": "analytic_reference", "reference_id": "underflow_oracle",
                                        "records": [],
                                    }, noise=noise)
    onsets = np.array([2, 7, 11, 15, 20, 24])
    membership = np.eye(2, dtype=bool)[[0, 1, 0, 1, 0, 1]]
    samples = _convolution_design(onsets, membership[:, 0], 29, basis) @ model.kernel_mean
    result = model.score(samples, onsets, membership, operator_identity="native", sfreq=10.,
                         channels=("Cz",), candidate_ids=("a", "b"), record_id="held")
    dense = _dense_evidence(samples, onsets, membership, model)
    assert result.probabilities[1] == 0.
    assert np.isfinite(result.log_probability[1]) and result.log_probability[1] < -1000.
    np.testing.assert_allclose(result.log_probability, dense-logsumexp(dense), rtol=0, atol=2e-9)


@pytest.mark.parametrize("noise_order", [None, 8])
def test_continuous_input_does_not_silently_drop_complex_signal_components(noise_order):
    records, basis = _records(n_samples=90)
    complex_input = replace(records[0], samples=records[0].samples + .1j)
    with pytest.raises(ValueError, match="real numeric"):
        ContinuousDecisionModel.fit([complex_input, *records[1:]], **_settings(basis),
                                    noise_order=noise_order)
    model = ContinuousDecisionModel.fit(records, **_settings(basis), noise_order=noise_order)
    with pytest.raises(ValueError, match="real numeric"):
        model.score(complex_input.samples, complex_input.onsets,
                    np.eye(3)[np.arange(len(complex_input.onsets)) % 3],
                    operator_identity=model.operator_identity, sfreq=50., channels=("Fz", "Pz"),
                    candidate_ids=("a", "b", "c"), record_id="held")
