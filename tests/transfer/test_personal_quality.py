from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from data.artifact import FoldLocalArtifactPolicy, apply_fitted_artifact_model
from data.artifact_sidecar import artifact_model_from_record
from tests.support.quality import personal_dataset, personal_training_arrays


def test_personal_cv_uses_explicit_sessions_without_replacing_subject_identity():
    values, subjects, groups = personal_training_arrays()
    fitted = FoldLocalArtifactPolicy().fit(values, subjects, fit_groups=groups)

    assert fitted.fit_groups == ("session-a", "session-b", "session-c")
    assert fitted.fit_subject_ids == ("P001",)
    assert fitted.record()["cv_evidence"]["status"] == "group_cross_validated"
    assert fitted.record()["cv_evidence"]["n_groups"] == 3
    train = np.ones(len(values), dtype=bool)
    _, _, kept, audit = apply_fitted_artifact_model(
        fitted, values, subjects, train, ~train, fit_groups=groups,
    )
    assert kept.any()
    assert audit["fit_subject_ids"] == ["P001"]
    restored = artifact_model_from_record(fitted.record())
    assert restored.record() == fitted.record()
    np.testing.assert_array_equal(
        restored.transform(values).drop_epoch_mask, fitted.transform(values).drop_epoch_mask,
    )


def test_single_group_fallback_has_no_cross_validation_evidence():
    values, subjects, _ = personal_training_arrays()
    fitted = FoldLocalArtifactPolicy().fit(values, subjects)

    assert fitted.record()["cv_evidence"]["status"] == "fallback_no_valid_cv"
    assert fitted.record()["cv_evidence"]["n_groups"] == 1
    assert fitted.record()["cv_evidence"]["fallback_reason"] == "fewer_than_two_fit_groups"


def test_fitted_policy_rejects_different_rows_even_with_same_group_set():
    values, subjects, groups = personal_training_arrays()
    fitted = FoldLocalArtifactPolicy().fit(values, subjects, fit_groups=groups)
    changed = values.copy()
    changed[0, 0, 0] += 1e-7
    train = np.ones(len(values), dtype=bool)
    with pytest.raises(ValueError, match="fit.*digest|training.*digest"):
        apply_fitted_artifact_model(
            fitted, changed, subjects, train, ~train, fit_groups=groups,
        )


def test_legacy_record_reload_marks_unavailable_fit_evidence():
    from data.artifact_sidecar import artifact_model_from_record

    values, subjects, _ = personal_training_arrays()
    record = FoldLocalArtifactPolicy().fit(values, subjects).record()
    for key in ("cv_evidence", "fit_subject_ids", "fit_group_source", "fit_data_digest", "fit_provenance"):
        del record[key]
    restored = artifact_model_from_record(record)
    assert restored.record()["cv_evidence"]["status"] == "unavailable_legacy_record"
    assert restored.fit_data_digest is None


@pytest.mark.parametrize("groups", [np.repeat("", 60), np.repeat("session", 59), np.repeat(np.nan, 60)])
def test_explicit_fit_groups_must_be_aligned_and_nonempty(groups):
    values, subjects, _ = personal_training_arrays()
    with pytest.raises(ValueError, match="fit_groups"):
        FoldLocalArtifactPolicy().fit(values, subjects, fit_groups=groups)


def test_personal_quality_fit_binds_only_training_rows_and_authoritative_identity():
    from transfer.quality import fit_epoch_quality

    dataset = personal_dataset()
    training = np.arange(dataset.n_epochs) < 40
    fitted = fit_epoch_quality(dataset, training, FoldLocalArtifactPolicy())
    changed_values = dataset.X.copy()
    changed_values[~training] *= 1e4
    changed = replace(dataset, X=changed_values, y=1 - dataset.y)
    other = fit_epoch_quality(changed, training, FoldLocalArtifactPolicy())

    assert fitted.record() == other.record()
    scope = fitted.fit_provenance
    assert scope["group_axis"] == "session"
    assert scope["participant_key"] == dataset.identity_table.record_for("P001").authority_key()
    assert scope["session_ids"] == ["session-a", "session-b"]
    assert scope["source_reference"] == "average"
    assert len(scope["training_event_ids"]) == 40
    assert len(scope["training_event_ids_digest"]) == 64


def test_single_session_quality_uses_declared_blocks_without_future_group_selection():
    from transfer.quality import fit_epoch_quality

    dataset = personal_dataset()
    training = np.arange(dataset.n_epochs) < 20
    fitted = fit_epoch_quality(dataset, training, FoldLocalArtifactPolicy())
    assert fitted.fit_provenance["group_axis"] == "block"
    assert len(fitted.fit_groups) == 2
    assert fitted.fit_subject_ids == ("P001",)
    assert fitted.cv_evidence["n_groups"] == 2


def test_quality_global_drop_keeps_plan_denominator_and_never_changes_original():
    from transfer.quality import apply_epoch_quality, fit_epoch_quality

    dataset = personal_dataset()
    fitted = fit_epoch_quality(dataset, np.arange(dataset.n_epochs) < 40, FoldLocalArtifactPolicy())
    # Isolate the whole-epoch physical-scale gate from local channel rejection.
    fitted = replace(fitted, ptp_thresholds=np.full(3, 100.), flat_std_thresholds=np.zeros(3))
    values = dataset.X.copy()
    values[-20:] *= 1e4
    dataset = replace(dataset, X=values)
    transformed = fitted.transform(values)
    assert transformed.drop_epoch_mask[-20:].all()
    assert not transformed.all_channels_bad[-20:].any()

    cleaned = apply_epoch_quality(dataset, fitted)
    cleaned.validate()
    assert cleaned is not dataset
    assert cleaned.n_epochs == 40
    assert cleaned.event_timeline.n_events == 60
    assert (cleaned.event_timeline.statuses[-20:] == "artifact_rejected").all()
    assert (cleaned.event_timeline.evidence_indices[-20:] == -1).all()
    assert np.isnan(cleaned.event_timeline.evidence_available_times_s[-20:]).all()
    np.testing.assert_array_equal(cleaned.event_timeline.event_ids, dataset.event_timeline.event_ids)
    assert (dataset.event_timeline.statuses == "available").all()
    assert dataset.trial_channel_mask is None
    assert cleaned.provenance["epoch_artifact_processing"]["n_retained_epochs"] == 40
    assert cleaned.provenance["epoch_artifact_processing"]["n_rejected_epochs"] == 20
    assert cleaned.verified_cache_attestation is None


def test_quality_preserves_missing_events_and_updates_local_qc_masks():
    from data.qc_features import compute_epoch_qc_features
    from transfer.quality import apply_epoch_quality, fit_epoch_quality

    dataset = personal_dataset()
    keep = np.arange(dataset.n_epochs) != 5
    timeline = dataset.event_timeline
    evidence = np.arange(dataset.n_epochs, dtype=np.int64)
    evidence[~keep] = -1
    evidence[6:] -= 1
    statuses = timeline.statuses.astype(object).copy()
    statuses[5] = "boundary_dropped"
    availability = timeline.evidence_available_times_s.copy()
    availability[5] = np.nan
    dataset = replace(
        dataset, X=dataset.X[keep], y=dataset.y[keep], subject_ids=dataset.subject_ids[keep],
        metadata=dataset.metadata.loc[keep].reset_index(drop=True),
        event_timeline=replace(timeline, evidence_indices=evidence, statuses=statuses.astype(str),
                               evidence_available_times_s=availability),
    )
    fitted = fit_epoch_quality(dataset, np.arange(dataset.n_epochs) < 40, FoldLocalArtifactPolicy())
    values = dataset.X.copy()
    values[-1, 0, 0] = 1e-2
    dataset = replace(dataset, X=values, qc_features=compute_epoch_qc_features(
        values, channel_mask=dataset.channel_mask,
    ))
    fitted = replace(fitted, ptp_thresholds=np.full(3, 100.), flat_std_thresholds=np.zeros(3),
                     selected_bad_channel_fraction=.5, global_scale_log_robust_std=100.)
    cleaned = apply_epoch_quality(dataset, fitted)
    assert cleaned.n_epochs == dataset.n_epochs
    assert cleaned.event_timeline.n_events == 60
    assert cleaned.event_timeline.statuses[5] == "boundary_dropped"
    assert cleaned.event_timeline.evidence_indices[5] == -1
    assert not cleaned.trial_channel_mask[-1, 0]
    assert not cleaned.qc_features.observed_mask[-1, 0]
    assert (cleaned.X[-1, 0] == 0.).all()
    cleaned.validate()


def test_all_rejected_epochs_fail_explicitly_without_returning_invalid_dataset():
    from transfer.quality import apply_epoch_quality, fit_epoch_quality

    dataset = personal_dataset()
    fitted = fit_epoch_quality(dataset, np.arange(dataset.n_epochs) < 40, FoldLocalArtifactPolicy())
    with pytest.raises(ValueError, match="no usable.*epoch"):
        apply_epoch_quality(replace(dataset, X=dataset.X * 1e10), fitted)


@pytest.mark.parametrize("field", ["ptp_thresholds", "flat_std_thresholds", "selected_quantiles"])
def test_restored_model_rejects_nonfinite_threshold_corruption(field):
    from data.artifact_sidecar import artifact_model_from_record

    values, subjects, _ = personal_training_arrays()
    record = FoldLocalArtifactPolicy().fit(values, subjects).record()
    record[field][0] = np.nan
    with pytest.raises(ValueError, match="invalid artifact model"):
        artifact_model_from_record(record)


def test_quality_reload_keeps_structured_preprocessing_contract_comparable():
    import json

    from data.artifact_sidecar import artifact_model_from_record
    from transfer.quality import apply_epoch_quality, fit_epoch_quality

    dataset = personal_dataset()
    dataset = replace(dataset, preprocessing=replace(
        dataset.preprocessing, baseline_mode="trial_reference", trial_reference_window_ms=(-200., 0.),
    ))
    fitted = fit_epoch_quality(dataset, np.arange(dataset.n_epochs) < 40, FoldLocalArtifactPolicy())
    restored = artifact_model_from_record(json.loads(json.dumps(fitted.record())))
    actual = apply_epoch_quality(dataset, restored)
    expected = apply_epoch_quality(dataset, fitted)
    np.testing.assert_array_equal(actual.X, expected.X)
    assert actual.provenance["epoch_artifact_processing"] == expected.provenance["epoch_artifact_processing"]


def test_missing_channel_quality_record_hashes_and_roundtrips_without_nonfinite_json():
    import json

    from data.artifact_sidecar import artifact_model_from_record
    from research.contracts import semantic_sha256
    from transfer.quality import apply_epoch_quality, fit_epoch_quality

    dataset = personal_dataset()
    values = dataset.X.copy()
    values[:, 1] = 0.
    dataset = replace(dataset, X=values, channel_mask=np.asarray([True, False, True]))
    quality = fit_epoch_quality(dataset, np.arange(dataset.n_epochs) < 40, FoldLocalArtifactPolicy())
    record = quality.record()
    digest = semantic_sha256(record)
    restored = artifact_model_from_record(json.loads(json.dumps(record, allow_nan=False)))

    assert record["fit_observed_channel_mask"] == [True, False, True]
    assert record["ptp_thresholds"][1] is None
    assert record["flat_std_thresholds"][1] is None
    assert np.isposinf(restored.ptp_thresholds[1])
    assert np.isneginf(restored.flat_std_thresholds[1])
    assert semantic_sha256(restored.record()) == digest
    expected = apply_epoch_quality(dataset, quality)
    actual = apply_epoch_quality(dataset, restored)
    np.testing.assert_array_equal(actual.X, expected.X)
    np.testing.assert_array_equal(actual.trial_channel_mask, expected.trial_channel_mask)


def test_legacy_missing_channel_sentinels_load_into_explicit_null_record():
    from data.artifact_sidecar import artifact_model_from_record

    values, subjects, _ = personal_training_arrays()
    observed = np.ones(values.shape[:2], dtype=bool)
    observed[:, 1] = False
    values[:, 1] = 0.
    fitted = FoldLocalArtifactPolicy().fit(values, subjects, observed)
    legacy = fitted.record()
    del legacy["fit_observed_channel_mask"]
    legacy["ptp_thresholds"][1] = float("inf")
    legacy["flat_std_thresholds"][1] = float("-inf")
    restored = artifact_model_from_record(legacy)
    assert restored.record()["fit_observed_channel_mask"] == [True, False, True]
    assert restored.record()["ptp_thresholds"][1] is None


def test_declared_missing_channel_cannot_hide_finite_thresholds():
    from data.artifact_sidecar import artifact_model_from_record

    values, subjects, _ = personal_training_arrays()
    record = FoldLocalArtifactPolicy().fit(values, subjects).record()
    record["fit_observed_channel_mask"] = [True, False, True]
    with pytest.raises(ValueError, match="invalid artifact model"):
        artifact_model_from_record(record)
