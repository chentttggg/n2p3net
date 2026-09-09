from __future__ import annotations

from dataclasses import replace

import mne
import numpy as np
import pytest

from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc
from data.epochs import materialize_dataset_identity
from research.contracts import semantic_sha256
from tests.test_personal_quality import _personal_dataset
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


def _declare_quality(dataset, *, channel_local=True, minimum_channel_fraction=1.):
    identity = materialize_dataset_identity(dataset)
    raw = mne.io.RawArray(
        np.random.default_rng(83).normal(0., 1e-6, (dataset.n_channels, 1000)),
        mne.create_info(dataset.channel_names, dataset.preprocessing.sfreq, "eeg"), verbose=False,
    )
    fitted = fit_continuous_qc(
        raw,
        spec=ContinuousQCSpec(12., 1e-7, 10, channel_local=channel_local,
                              minimum_channel_fraction=minimum_channel_fraction),
        fit_sample_mask=np.ones(raw.n_times, dtype=bool),
        fit_provenance={
            "participant_key": identity.authority_keys()[0],
            "calibration_session_ids": sorted(dataset.metadata["session"].unique()),
        },
    )
    record = fitted.record()
    return replace(dataset, provenance={**dataset.provenance, "continuous_artifact_processing": {
        "status": "applied", "model": record, "model_digest": semantic_sha256(record),
    }})


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
    dataset = _declare_quality(_personal_dataset())
    changed = _mask_rows(dataset, [0, 21, 40])
    candidate = fit_calibration_candidate(changed, waveform_lda_recipe(), np.ones(dataset.n_epochs, dtype=bool))
    np.testing.assert_array_equal(candidate.waveform_channels, [0, 1, 2])
    assert candidate.fit_history["fit_scope"]["training_rows"] == dataset.n_epochs - 3
    assert candidate.fit_history["fit_scope"]["requested_training_rows"] == dataset.n_epochs
    assert np.isnan(candidate.predict_logit(changed.X, changed.trial_channel_mask)[[0, 21, 40]]).all()


def test_complete_union_training_digest_excludes_unused_rows_but_request_digest_keeps_them():
    dataset = _mask_rows(_declare_quality(_personal_dataset()), [0, 21, 40])
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

    dataset = _personal_dataset()
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

    dataset = _declare_quality(_personal_dataset(), minimum_channel_fraction=2/3)
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


def test_continuous_only_lda_marks_missing_selected_channels_unavailable_per_row(tmp_path):
    dataset = _declare_quality(_personal_dataset())
    candidate, predictor = _fit(dataset)
    expected = candidate.predict_logit(dataset.X)
    missing = np.asarray([1, 4, 20])
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


def test_continuous_only_lda_returns_aligned_nans_when_every_row_lacks_a_selected_channel():
    dataset = _declare_quality(_personal_dataset())
    candidate, predictor = _fit(dataset)
    changed = _mask_rows(dataset, np.arange(dataset.n_epochs))

    assert np.isnan(candidate.predict_logit(changed.X, changed.trial_channel_mask)).all()
    actual = predictor.predict_logit(changed)
    assert actual.shape == (dataset.n_epochs,) and np.isnan(actual).all()
    np.testing.assert_array_equal(changed.event_timeline.event_ids, dataset.event_timeline.event_ids)


@pytest.mark.parametrize("channel_local", [None, False])
def test_lda_without_frozen_channel_quality_authority_still_rejects_missing_features(channel_local):
    dataset = _personal_dataset()
    if channel_local is not None:
        dataset = _declare_quality(dataset, channel_local=channel_local)
    candidate, predictor = _fit(dataset)
    changed = _mask_rows(dataset, np.asarray([2]))

    with pytest.raises(ValueError, match="selected channel"):
        candidate.predict_logit(changed.X, changed.trial_channel_mask)
    with pytest.raises(ValueError, match="selected EEG channel"):
        predictor.predict_logit(changed)


def test_recovery_missing_lda_evidence_keeps_full_r5_block_denominator(tmp_path):
    from data.brainsync import load_brainsync_sessions
    from tests.test_brainsync import _write_bids_session

    root = _write_bids_session(
        tmp_path / "r5", session_id="r5", target=4,
        started_utc="2026-09-06T00:00:00Z", blocks=(tuple(range(1, 10)) * 5,),
    )
    dataset = _declare_quality(load_brainsync_sessions([root]))
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
