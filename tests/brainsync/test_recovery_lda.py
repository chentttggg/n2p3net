from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from data.epochs import materialize_dataset_identity
from tests.support.brainsync import write_bids_session
from tests.support.quality import declare_continuous_quality, personal_dataset
from transfer.calibrated import (
    CalibratedPredictor,
    _input_contract,
    load_calibrated_predictor,
    save_calibrated_predictor,
)
from transfer.calibration_candidates import (
    fit_calibration_candidate,
    score_calibration_candidate,
    waveform_lda_recipe,
)


def _fit(dataset):
    candidate = fit_calibration_candidate(
        dataset, waveform_lda_recipe(), np.ones(dataset.n_epochs, dtype=bool),
    )
    metadata = {
        "model_kind": "target_only_shrinkage_lda",
        "input_contract": _input_contract(dataset),
        "participant_key": materialize_dataset_identity(dataset).authority_keys()[0],
    }
    return candidate, CalibratedPredictor(metadata, candidate.state)


def _mask_rows(dataset, rows):
    observed = np.ones(dataset.X.shape[:2], dtype=bool)
    observed[rows, 0] = False
    values = dataset.X.copy()
    values[~observed] = 0.
    return replace(dataset, X=values, trial_channel_mask=observed, qc_features=None)


def test_channel_quality_lda_uses_complete_observations_when_common_features_are_insufficient():
    dataset = declare_continuous_quality(personal_dataset())
    changed = _mask_rows(dataset, [0, 21, 40])
    candidate = fit_calibration_candidate(changed, waveform_lda_recipe(), np.ones(dataset.n_epochs, dtype=bool))
    np.testing.assert_array_equal(candidate.waveform_channels, [0, 1, 2])
    assert candidate.fit_history["fit_scope"]["training_rows"] == dataset.n_epochs - 3
    assert candidate.fit_history["fit_scope"]["requested_training_rows"] == dataset.n_epochs
    assert np.isnan(candidate.predict_logit(changed.X, changed.trial_channel_mask)[[0, 21, 40]]).all()


def test_complete_union_training_digest_excludes_unused_rows_but_request_digest_keeps_them():
    dataset = _mask_rows(declare_continuous_quality(personal_dataset()), [0, 21, 40])
    training = np.ones(dataset.n_epochs, dtype=bool)
    first = fit_calibration_candidate(dataset, waveform_lda_recipe(), training)
    values, labels, observed = dataset.X.copy(), dataset.y.copy(), dataset.trial_channel_mask.copy()
    values[[0, 21, 40], 1] *= 100
    labels[[0, 21, 40]] = 1 - labels[[0, 21, 40]]
    observed[21, 2] = False
    values[~observed] = 0.
    changed = replace(dataset, X=values, y=labels, trial_channel_mask=observed)
    second = fit_calibration_candidate(changed, waveform_lda_recipe(), training)

    assert first.fit_history["training_data_sha256"] == second.fit_history["training_data_sha256"]
    assert first.fit_history["requested_training_data_sha256"] != second.fit_history["requested_training_data_sha256"]
    for key, value in first.state.items():
        np.testing.assert_array_equal(value.numpy(), second.state[key].numpy())


@pytest.mark.parametrize("initialization", ["source", "scratch"])
def test_neural_candidate_preserves_existing_training_digest(initialization):
    import hashlib

    from models.n2p3net import N2P3Net
    from transfer.recipes import TargetRecipe

    dataset = personal_dataset()
    training = np.arange(dataset.n_epochs) < 40
    source = N2P3Net(
        n_channels=dataset.n_channels, n_times=dataset.n_times, sfreq=dataset.preprocessing.sfreq,
        tmin_s=dataset.preprocessing.tmin_ms / 1000., temporal_filters=2, mst_features_per_scale=2,
    )
    expected = hashlib.sha256()
    for array in (dataset.X[training], dataset.y[training], np.ones(dataset.X.shape[:2], dtype=bool)[training]):
        expected.update(np.ascontiguousarray(array).tobytes())
    fitted = fit_calibration_candidate(
        dataset, TargetRecipe(initialization, initialization=initialization, epochs=1),
        training, source_trunk=source,
    )
    assert fitted.fit_history["training_data_sha256"] == expected.hexdigest()
    assert fitted.fit_history["requested_training_data_sha256"] == expected.hexdigest()


def test_epoch_quality_cannot_reduce_continuous_minimum_observation_floor():
    from data.artifact import FoldLocalArtifactPolicy
    from transfer.quality import apply_epoch_quality, fit_epoch_quality

    dataset = declare_continuous_quality(personal_dataset(), minimum_channel_fraction=2/3)
    fitted = fit_epoch_quality(dataset, np.ones(dataset.n_epochs, dtype=bool), FoldLocalArtifactPolicy())
    fitted = replace(fitted, ptp_thresholds=np.full(3, 1.5), flat_std_thresholds=np.zeros(3),
                     global_scale_log_robust_std=1000., selected_bad_channel_fraction=.5)
    mask = np.ones(dataset.X.shape[:2], dtype=bool)
    mask[0, 2] = False
    values = dataset.X.copy()
    values[0, 0] *= 10
    values[~mask] = 0.
    changed = replace(dataset, X=values, trial_channel_mask=mask, qc_features=None)
    transformed = fitted.transform(changed.X, mask)
    assert transformed.trial_channel_mask[0].sum() == 1
    assert not transformed.drop_epoch_mask[0]
    output = apply_epoch_quality(changed, fitted)
    assert output.event_timeline.evidence_indices[0] == -1
    assert output.event_timeline.statuses[0] == "artifact_rejected"


@pytest.mark.parametrize("missing", [[1, 4, 20], list(range(60))], ids=["partial", "all"])
def test_continuous_only_lda_marks_missing_selected_channels_unavailable_per_row(tmp_path, missing):
    dataset = declare_continuous_quality(personal_dataset())
    candidate, predictor = _fit(dataset)
    expected = candidate.predict_logit(dataset.X)
    missing = np.asarray(missing)
    changed = _mask_rows(dataset, missing)

    assert candidate.channel_quality_availability is True
    assert candidate.epoch_quality is None and predictor.epoch_quality is None
    actual = candidate.predict_logit(changed.X, changed.trial_channel_mask)
    persisted = predictor.predict_logit(changed)
    assert actual.shape == persisted.shape == (dataset.n_epochs,)
    assert np.isnan(actual[missing]).all() and np.isnan(persisted[missing]).all()
    retained = np.ones(dataset.n_epochs, dtype=bool)
    retained[missing] = False
    np.testing.assert_allclose(actual[retained], expected[retained], atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(persisted[retained], expected[retained], atol=1e-5, rtol=1e-6)
    restored = load_calibrated_predictor(save_calibrated_predictor(tmp_path / "lda.pt", predictor))
    np.testing.assert_array_equal(restored.predict_logit(changed), persisted)
    np.testing.assert_array_equal(changed.event_timeline.event_ids, dataset.event_timeline.event_ids)


@pytest.mark.parametrize("channel_local", [None, False])
def test_lda_without_frozen_channel_quality_authority_still_rejects_missing_features(channel_local):
    dataset = personal_dataset()
    if channel_local is not None:
        dataset = declare_continuous_quality(dataset, channel_local=channel_local)
    candidate, predictor = _fit(dataset)
    changed = _mask_rows(dataset, np.asarray([2]))

    with pytest.raises(ValueError, match="selected channel"):
        candidate.predict_logit(changed.X, changed.trial_channel_mask)
    with pytest.raises(ValueError, match="selected EEG channel"):
        predictor.predict_logit(changed)


def test_recovery_missing_lda_evidence_keeps_full_r5_block_denominator(tmp_path):
    from data.brainsync import load_brainsync_sessions

    root = write_bids_session(
        tmp_path / "r5", session_id="r5", target=4,
        started_utc="2026-09-06T00:00:00Z", blocks=(tuple(range(1, 10)) * 5,),
    )
    dataset = declare_continuous_quality(load_brainsync_sessions([root]))
    candidate, _ = _fit(dataset)
    changed = _mask_rows(dataset, np.arange(dataset.n_epochs))
    logits = candidate.predict_logit(changed.X, changed.trial_channel_mask)
    score = score_calibration_candidate(
        changed, logits, np.ones(changed.n_epochs, dtype=bool), protocol="brainsync_operational_r5",
    )

    assert score["requested_validation_rows"] == 45
    assert score["unavailable_validation_rows"] == 45
    assert score["operational_hit_r5"] == 0.
    assert score["session_count"] == 1 and len(score["blocks"]) == 1
    assert score["failure_denominator"] == "all_scheduled_blocks"
