"""A valid cancelling ridge fit must publish one consistent QR posterior."""
from dataclasses import replace

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from tests.transfer.test_ensemble_fit import selections
from transfer.continuous_decision import (
    CalibrationRecord,
    ContinuousDecisionModel,
    _event_design,
    _stationary_transform,
    _whiten_and_project,
)
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec


@pytest.mark.parametrize("noise_order", [0, None])
def test_stationary_fit_owns_the_same_mean_factor_and_saved_response(tmp_path, noise_order):
    groups = selections()
    copied = replace(groups[0], view=replace(groups[0].view,
        record_id="independent-copied-waveform", decision_id="copied-selection",
        samples=groups[0].view.samples * 1.01, raw_scope=(3400, 3951), available_sample=3951))
    actual = (groups[3], copied)
    model = ContinuousDecisionModel.fit([
        CalibrationRecord(group.view.record_id, group.view.samples, group.view.onsets, group.labels)
        for group in actual], sfreq=256., channels=("Fz", "Cz", "Pz"),
        candidate_ids=("a", "b"), operator_identity="native_unfiltered_volts/1", noise_order=noise_order,
        dictionary=ResponseDictionary.compile(ResponseDictionarySpec("gaussian", 1.2, 2), sample_rate_hz=256.))
    state = model.posterior_state
    assert model.provenance["kernel_prior_fit_statistics"]["method"] == "all_C_actual_noise_per_record_diffuse_QR_response"
    path = tmp_path / "owned-posterior.npz"
    model.save(path)
    loaded = ContinuousDecisionModel.load(path)
    np.testing.assert_array_equal(loaded.kernel_mean, model.kernel_mean)
    for key in ("precision_root", "response", "orthogonal"):
        np.testing.assert_array_equal(loaded.posterior_state[key], state[key])
    values = np.random.default_rng(404).normal(0., 1e-6, (551, 3))
    settings = {"sfreq": 256., "channels": model.channels, "candidate_ids": model.candidate_ids,
                "operator_identity": model.operator_identity, "record_id": "new-software-decision"}
    before = model.score(values, groups[0].view.onsets, groups[0].view.decision.membership, **settings)
    after = loaded.score(values, groups[0].view.onsets, groups[0].view.decision.membership, **settings)
    np.testing.assert_array_equal(before.log_evidence, after.log_evidence)
    # Persistence remains tied to the real producer: a changed mean cannot be
    # credited by weakening the componentwise posterior relation.
    wrong = model.kernel_mean.copy()
    wrong[3, 0] += 1e-8
    with pytest.raises(ValueError, match="posterior mean"):
        replace(model, kernel_mean=wrong)


@pytest.mark.parametrize("noise_order", [0, None])
def test_redundant_large_basis_keeps_true_smoothing_df_and_noise_scale(noise_order):
    rng = np.random.default_rng(11801)
    wave = np.exp(-((np.arange(9) - 4) / 2.2) ** 2)
    basis = 1e8 * np.column_stack((wave, wave))
    onsets = np.arange(4, 90, 13, dtype=np.int64)
    records = [CalibrationRecord(f"df-C-{index}", rng.normal(size=(110, 2)), onsets,
        (np.arange(len(onsets)) % 3 == index % 3).astype(np.int64)) for index in range(3)]
    model = ContinuousDecisionModel.fit(records, sfreq=20., channels=("a", "b"),
        candidate_ids=("one", "two"), operator_identity="redundant-numeric-basis", basis=basis,
        ridge=1., noise_order=noise_order)
    transforms = [(_stationary_transform(model.noise, len(record.samples)) if model.noise is not None
        else lambda values: _whiten_and_project(values, model.phi)) for record in records]
    design = np.vstack([transform(_event_design(record.onsets, record.target_mask, len(record.samples), basis))
        for record, transform in zip(records, transforms, strict=True)])
    response = np.vstack([transform(record.samples) for record, transform in zip(records, transforms, strict=True)])
    # Two identical columns per group have exactly two physical signal
    # directions. The reduced two-column model is an independent df oracle;
    # its ridge is unchanged after the sqrt(2) isometric coordinate transform.
    eigenvalues = np.linalg.eigvalsh(2 * design[:, [0, 2]].T @ design[:, [0, 2]])
    expected_df = float(np.sum(eigenvalues / (eigenvalues + 1.)))
    assert model.provenance["effective_parameters_per_channel"] == pytest.approx(expected_df, rel=0, abs=2e-12)
    residual_df = len(design) - 2 * len(records) - expected_df
    assert model.provenance["residual_degrees_of_freedom"] == pytest.approx(residual_df, rel=0, abs=2e-12)
    residual = response - design @ model.kernel_mean
    expected_noise = LedoitWolf(assume_centered=True).fit(residual).covariance_ * len(response) / residual_df
    scale = max(float(np.trace(expected_noise) / 2),
        float(np.mean(response**2)) * np.finfo(float).eps, np.finfo(float).tiny)
    expected_noise += np.eye(2) * scale * 1e-8
    np.testing.assert_allclose(model.innovation_covariance, expected_noise, rtol=2e-13, atol=2e-13)
