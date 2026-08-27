from __future__ import annotations

import numpy as np
import pytest

from data.artifact import FoldLocalArtifactPolicy, apply_fold_local_artifact_policy


def _clean_epochs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    return rng.normal(0.0, 1.0, size=(24, 4, 32)).astype(np.float32), np.repeat(
        np.arange(6), 4
    ).astype(str)


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
