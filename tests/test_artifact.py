from __future__ import annotations

import numpy as np
import pytest

from data.artifact import (
    FoldLocalArtifactPolicy,
    apply_fitted_artifact_model,
    apply_fold_local_artifact_policy,
)
from data.qc_features import EpochQCFeatures, compute_epoch_qc_features


def _clean_epochs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    return rng.normal(0.0, 1.0, size=(24, 4, 32)).astype(np.float32), np.repeat(
        np.arange(6), 4
    ).astype(str)


def _v2_fit_reference(policy, X, subject_ids, trial_channel_mask):
    """Scalar reference for training-clean / fixed-validation-median CV."""

    observed = np.asarray(trial_channel_mask, dtype=bool)
    features = compute_epoch_qc_features(
        X,
        channel_mask=np.ones(X.shape[1], dtype=bool),
        trial_channel_mask=observed,
    )
    subject_keys = np.asarray(subject_ids).astype(str)
    subjects = np.unique(subject_keys)
    chosen = []
    for channel in range(X.shape[1]):
        if len(subjects) < 2:
            chosen.append(policy.candidate_quantiles[-1])
            continue
        errors = np.zeros(len(policy.candidate_quantiles), dtype=float)
        counts = np.zeros(len(policy.candidate_quantiles), dtype=np.int64)
        folds = np.array_split(subjects, min(policy.cv_splits, len(subjects)))
        for held_out_subjects in folds:
            validation_subjects = np.isin(subject_keys, held_out_subjects)
            training_subjects = ~validation_subjects
            training = training_subjects & observed[:, channel]
            validation = validation_subjects & observed[:, channel]
            if training.sum() < policy.min_clean_epochs or validation.sum() < policy.min_clean_epochs:
                continue
            target = np.median(X[validation, channel, :], axis=0)
            flat_threshold = np.quantile(
                features.channel_std_v[training, channel], policy.flat_quantile
            )
            thresholds = np.quantile(
                features.relative_ptp[training, channel], policy.candidate_quantiles
            )
            for candidate, threshold in enumerate(thresholds):
                clean = training & (features.relative_ptp[:, channel] <= threshold)
                clean &= features.channel_std_v[:, channel] > flat_threshold
                if clean.sum() < policy.min_clean_epochs:
                    continue
                mean = X[clean, channel, :].mean(axis=0)
                errors[candidate] += np.mean((mean - target) ** 2)
                counts[candidate] += 1
        valid = counts > 0
        if not valid.any():
            chosen.append(policy.candidate_quantiles[-1])
            continue
        mean_errors = np.full(len(errors), np.inf)
        mean_errors[valid] = errors[valid] / counts[valid]
        best = float(np.min(mean_errors))
        tolerance = best * 1e-6 + np.finfo(float).eps
        chosen.append(
            policy.candidate_quantiles[
                int(np.flatnonzero(mean_errors <= best + tolerance)[-1])
            ]
        )

    ptp_thresholds = np.empty(X.shape[1], dtype=float)
    flat_thresholds = np.empty(X.shape[1], dtype=float)
    for channel in range(X.shape[1]):
        values = features.relative_ptp[observed[:, channel], channel]
        scales = features.channel_std_v[observed[:, channel], channel]
        if len(values) == 0:
            ptp_thresholds[channel] = np.inf
            flat_thresholds[channel] = -np.inf
        else:
            ptp_thresholds[channel] = float(np.quantile(values, chosen[channel]))
            flat_thresholds[channel] = float(np.quantile(scales, policy.flat_quantile))
    return np.asarray(chosen), ptp_thresholds, flat_thresholds


def test_batched_fit_matches_v2_training_clean_reference_with_trial_channel_mask() -> None:
    rng = np.random.default_rng(18)
    X = rng.normal(0.0, 1.0, size=(36, 4, 32)).astype(np.float32)
    subjects = np.repeat(np.arange(9), 4).astype(str)
    trial_mask = np.ones((len(X), X.shape[1]), dtype=bool)
    trial_mask[::3, 1] = False
    trial_mask[1::4, 3] = False
    X[~trial_mask] = 0.0
    policy = FoldLocalArtifactPolicy(global_scale_mad_z=1e9)

    expected_quantiles, expected_ptp, expected_flat = _v2_fit_reference(
        policy, X, subjects, trial_mask
    )
    actual = policy.fit(X, subjects, trial_mask)

    np.testing.assert_array_equal(actual.selected_quantiles, expected_quantiles)
    np.testing.assert_allclose(actual.ptp_thresholds, expected_ptp)
    np.testing.assert_allclose(actual.flat_std_thresholds, expected_flat)


def test_local_ptp_masks_one_bad_channel_without_dropping_epoch() -> None:
    X, subjects = _clean_epochs()
    X[3, 2, 5] = 100.0
    train = np.arange(len(X)) < 20
    test = ~train

    transformed, mask, kept_train, audit = apply_fold_local_artifact_policy(
        FoldLocalArtifactPolicy(max_bad_channel_fraction=0.25), X, subjects, train, test
    )

    assert not mask[3, 2]
    assert np.all(transformed[3, 2] == 0.0)
    assert kept_train[3]
    assert audit["train"]["n_epochs"] == 20


def test_many_bad_channels_drop_train_but_not_test_rows() -> None:
    X, subjects = _clean_epochs()
    X[2, :2, 0] = 100.0
    X[22, :2, 0] = 100.0
    train = np.arange(len(X)) < 20
    test = ~train

    _, mask, kept_train, audit = apply_fold_local_artifact_policy(
        FoldLocalArtifactPolicy(max_bad_channel_fraction=0.25), X, subjects, train, test
    )

    assert not kept_train[2]
    assert mask[22].any()
    assert audit["test"]["n_epochs_over_bad_channel_limit"] == 1


def test_global_epoch_scale_gate_drops_training_but_keeps_test_denominator() -> None:
    X, subjects = _clean_epochs()
    train = np.arange(len(X)) < 20
    test = ~train
    X[3] *= 100.0
    X[22] *= 100.0
    features = compute_epoch_qc_features(X, channel_mask=np.ones(X.shape[1], dtype=bool))

    _, _, kept_train, audit = apply_fold_local_artifact_policy(
        FoldLocalArtifactPolicy(global_scale_mad_z=4.0),
        X,
        subjects,
        train,
        test,
        qc_features=features,
    )

    assert not kept_train[3]
    assert audit["test"]["n_epochs"] == int(test.sum())
    assert audit["test"]["n_global_epoch_scale"] >= 1


def test_kappa_selection_requires_training_coverage() -> None:
    n_subjects = 4
    repeats = 4
    n_epochs = n_subjects * repeats
    X = np.zeros((n_epochs, 3, 8), dtype=np.float32)
    subjects = np.repeat(np.arange(n_subjects), repeats).astype(str)
    relative_ptp = np.ones((n_epochs, 3), dtype=np.float32)
    relative_ptp[::repeats, 0] = 10.0
    features = EpochQCFeatures(
        relative_ptp=relative_ptp,
        channel_std_v=np.ones((n_epochs, 3), dtype=np.float32),
        epoch_scale_v=np.ones(n_epochs, dtype=np.float32),
        observed_mask=np.ones((n_epochs, 3), dtype=bool),
    )
    policy = FoldLocalArtifactPolicy(
        candidate_bad_channel_fractions=(0.0, 0.5),
        min_training_epoch_retention=0.9,
        global_scale_mad_z=1e9,
        cv_splits=2,
        min_clean_epochs=2,
    )

    selected = policy._choose_bad_channel_fraction(
        X,
        features,
        subjects,
        np.full(3, 0.5, dtype=float),
    )

    assert selected == 0.5


def test_test_extreme_does_not_change_train_fitted_thresholds() -> None:
    X, subjects = _clean_epochs()
    train = np.arange(len(X)) < 20
    test = ~train
    policy = FoldLocalArtifactPolicy()
    _, _, _, first = apply_fold_local_artifact_policy(policy, X, subjects, train, test)
    X[test, 0, 0] = 1e6
    _, _, _, second = apply_fold_local_artifact_policy(policy, X, subjects, train, test)

    assert first["ptp_thresholds"] == second["ptp_thresholds"]
    assert first["selected_quantiles"] == second["selected_quantiles"]


def test_frozen_policy_matches_direct_fold_fit_and_audit() -> None:
    X, subjects = _clean_epochs()
    X[3, 2, 5] = 100.0
    train = np.arange(len(X)) < 20
    test = ~train
    features = compute_epoch_qc_features(X, channel_mask=np.ones(X.shape[1], dtype=bool))
    policy = FoldLocalArtifactPolicy(global_scale_mad_z=1e9)

    direct = apply_fold_local_artifact_policy(
        policy,
        X,
        subjects,
        train,
        test,
        qc_features=features,
    )
    fitted = policy.fit(X[train], subjects[train], qc_features=features.subset(train))
    frozen = apply_fitted_artifact_model(
        fitted,
        X,
        subjects,
        train,
        test,
        qc_features=features,
    )

    np.testing.assert_array_equal(frozen[0], direct[0])
    np.testing.assert_array_equal(frozen[1], direct[1])
    np.testing.assert_array_equal(frozen[2], direct[2])
    assert frozen[3] == direct[3]


def test_transform_can_defer_zero_fill_without_changing_mask_decision() -> None:
    X, subjects = _clean_epochs()
    fitted = FoldLocalArtifactPolicy(global_scale_mad_z=1e9).fit(X, subjects)
    probe = X.copy()
    probe[0, 2, 5] = 100.0

    materialized = fitted.transform(probe, materialize_masked_data=True)
    deferred = fitted.transform(probe, materialize_masked_data=False)

    np.testing.assert_array_equal(deferred.trial_channel_mask, materialized.trial_channel_mask)
    assert deferred.X is probe
    assert not deferred.trial_channel_mask[0, 2]
    assert deferred.X[0, 2, 5] == 100.0
    assert materialized.X[0, 2, 5] == 0.0


def test_all_bad_test_epoch_fails_closed() -> None:
    X, subjects = _clean_epochs()
    train = np.arange(len(X)) < 20
    test = ~train
    X[22] = 0.0

    with pytest.raises(ValueError, match="every channel"):
        apply_fold_local_artifact_policy(FoldLocalArtifactPolicy(), X, subjects, train, test)


def test_binary_evaluator_preserves_test_denominator_and_records_quality() -> None:
    from baselines.evaluate import evaluate_binary

    X, subjects = _clean_epochs()
    y = np.tile(np.array([0, 1], dtype=np.int64), len(X) // 2)
    X[:, 0, :] += 2.0 * y[:, None] - 1.0
    X[22, 2, 0] = 100.0
    train = subjects != "5"
    test = ~train

    class MaskAwareScoreModel:
        fit_accepts_group_ids = True
        fit_accepts_trial_channel_mask = True
        predict_accepts_trial_channel_mask = True

        def fit(self, values, labels, group_ids=None, trial_channel_mask=None):
            assert len(values) == len(labels) == len(group_ids) == len(trial_channel_mask)
            self.calibration_logits_ = np.array([-2.0, -1.0, 1.0, 2.0])
            self.calibration_labels_ = np.array([0, 0, 1, 1])
            self.calibration_source_ = "group_disjoint_validation"
            return self

        def predict_logit(self, values, trial_channel_mask=None):
            assert np.all(values[~trial_channel_mask] == 0.0)
            return values[:, 0, :].mean(axis=1)

    summary = evaluate_binary(
        MaskAwareScoreModel(),
        X,
        y,
        subjects,
        [(train, test)],
        fold_protocol="partial_loso",
        artifact_policy=FoldLocalArtifactPolicy(),
    )

    fold = summary.per_fold[0]
    assert fold.n_test_trials == int(test.sum())
    assert fold.artifact_quality is not None
    assert fold.artifact_quality["test"]["n_epochs"] == int(test.sum())
