from __future__ import annotations

import numpy as np

from baselines.classic import WindowLogisticRegression
from baselines.evaluate import evaluate_binary, evaluate_candidate_selection, loso_folds


def _p300_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    subjects = np.repeat(np.arange(6), 12)
    y = np.tile(np.array([0, 1] * 6, dtype=np.int64), 6)
    X = rng.normal(size=(len(y), 3, 96)).astype(np.float32)
    X[y == 1, 2, 42:62] += 2.5
    return X, y, subjects


def test_binary_loso_reports_auc_and_bacc() -> None:
    X, y, subjects = _p300_data()
    summary = evaluate_binary(
        WindowLogisticRegression(sfreq=128.0, tmin=-0.2, window_ms=(125.0, 300.0)),
        X,
        y,
        subjects,
        loso_folds(subjects),
    )

    assert summary.auc_mean > 0.8
    assert summary.balanced_acc_mean > 0.7


def test_candidate_evidence_uses_calibrated_target_scores() -> None:
    X, y, physical_subjects = _p300_data()
    codes = np.tile(np.tile(np.arange(3), 4), 6)
    groups = np.repeat(np.arange(6).astype(str), 12)
    # Candidate 1 is the target in each decision group.
    y = (codes == 1).astype(np.int64)
    X[y == 1, 2, 42:62] += 2.5
    summary = evaluate_candidate_selection(
        WindowLogisticRegression(sfreq=128.0, tmin=-0.2, window_ms=(125.0, 300.0)),
        X,
        y,
        codes,
        groups,
        {str(subject): 1 for subject in range(6)},
        loso_folds(physical_subjects),
        candidate_vocab=(0, 1, 2),
        fold_subject_ids=physical_subjects,
    )

    assert summary.primary_hit_rate > 0.8
