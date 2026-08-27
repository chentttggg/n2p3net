from __future__ import annotations

import numpy as np
import pytest

from data.artifact import (
    FoldLocalArtifactPolicy,
    _relative_peak_to_peak,
    apply_fold_local_artifact_policy,
)


def _clean_epochs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    return rng.normal(0.0, 1.0, size=(24, 4, 32)).astype(np.float32), np.repeat(
        np.arange(6), 4
    ).astype(str)


def _legacy_fit(policy, X, subject_ids, trial_channel_mask):
    """Reference implementation for checking the batched fit semantics."""

    observed = np.asarray(trial_channel_mask, dtype=bool)
    ptp = _relative_peak_to_peak(X, observed)
    std = np.std(X.astype(np.float64, copy=False), axis=2)
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
            validation = validation_subjects & observed[:, channel]
            training = ~validation_subjects & observed[:, channel]
            if training.sum() < policy.min_clean_epochs or validation.sum() < policy.min_clean_epochs:
                continue
            template = np.median(X[training, channel, :], axis=0)
            thresholds = np.quantile(ptp[training, channel], policy.candidate_quantiles)
            clean = validation[:, None] & (ptp[:, channel, None] <= thresholds)
            clean_counts = clean.sum(axis=0)
            valid = clean_counts >= policy.min_clean_epochs
            if not valid.any():
                continue
            means = clean.T.astype(np.float64) @ X[:, channel, :]
            means /= np.maximum(clean_counts[:, None], 1)
            errors[valid] += np.mean((means[valid] - template) ** 2, axis=1)
            counts[valid] += 1
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
        values = ptp[observed[:, channel], channel]
        scales = std[observed[:, channel], channel]
        if len(values) == 0:
            ptp_thresholds[channel] = np.inf
            flat_thresholds[channel] = -np.inf
        else:
            ptp_thresholds[channel] = float(np.quantile(values, chosen[channel]))
            flat_thresholds[channel] = float(np.quantile(scales, policy.flat_quantile))
    return np.asarray(chosen), ptp_thresholds, flat_thresholds


def test_batched_fit_matches_legacy_with_trial_channel_mask() -> None:
    rng = np.random.default_rng(18)
    X = rng.normal(0.0, 1.0, size=(36, 4, 32)).astype(np.float32)
    subjects = np.repeat(np.arange(9), 4).astype(str)
    trial_mask = np.ones((len(X), X.shape[1]), dtype=bool)
    trial_mask[::3, 1] = False
    trial_mask[1::4, 3] = False
    X[~trial_mask] = 0.0
    policy = FoldLocalArtifactPolicy()

    expected_quantiles, expected_ptp, expected_flat = _legacy_fit(
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
        fit_accepts_subject_ids = True
        fit_accepts_trial_channel_mask = True
        predict_accepts_trial_channel_mask = True

        def fit(self, values, labels, subject_ids=None, trial_channel_mask=None):
            assert len(values) == len(labels) == len(subject_ids) == len(trial_channel_mask)
            self.calibration_logits_ = np.array([-2.0, -1.0, 1.0, 2.0])
            self.calibration_labels_ = np.array([0, 0, 1, 1])
            self.calibration_source_ = "subject_disjoint_validation"
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
