from __future__ import annotations

import numpy as np
import pytest
import torch

from baselines.classic import WindowLogisticRegression
from baselines.evaluate import (
    _resolve_fold_execution,
    evaluate_binary,
    evaluate_candidate_selection,
    loso_folds,
)
from data.artifact import FoldLocalArtifactPolicy
from data.qc_features import compute_epoch_qc_features


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


def test_threaded_cpu_folds_match_serial_results() -> None:
    X, y, subjects = _p300_data()
    model = WindowLogisticRegression(sfreq=128.0, tmin=-0.2, window_ms=(125.0, 300.0))
    serial = evaluate_binary(model, X, y, subjects, loso_folds(subjects))
    parallel = evaluate_binary(
        model,
        X,
        y,
        subjects,
        loso_folds(subjects),
        n_jobs=2,
        parallel_backend="thread",
        cpu_threads=2,
    )

    assert parallel.execution_backend == "thread"
    assert parallel.effective_n_jobs == 2
    assert parallel.cpu_threads_per_worker == 1
    assert parallel.balanced_acc_mean == serial.balanced_acc_mean
    assert parallel.auc_mean == serial.auc_mean


def test_process_cpu_folds_share_the_input_matrix() -> None:
    X, y, subjects = _p300_data()
    summary = evaluate_binary(
        WindowLogisticRegression(sfreq=128.0, tmin=-0.2, window_ms=(125.0, 300.0)),
        X,
        y,
        subjects,
        loso_folds(subjects),
        n_jobs=2,
        parallel_backend="process",
        cpu_threads=2,
    )

    assert summary.execution_backend == "process"
    assert summary.effective_n_jobs == 2
    assert summary.cpu_threads_per_worker == 1
    assert summary.input_transport == "shared_memory"
    assert summary.auc_mean > 0.8


def test_process_folds_share_cached_qc_features() -> None:
    X, y, subjects = _p300_data()
    features = compute_epoch_qc_features(X, channel_mask=np.ones(X.shape[1], dtype=bool))
    summary = evaluate_binary(
        WindowLogisticRegression(sfreq=128.0, tmin=-0.2, window_ms=(125.0, 300.0)),
        X,
        y,
        subjects,
        loso_folds(subjects),
        qc_features=features,
        artifact_policy=FoldLocalArtifactPolicy(global_scale_mad_z=1e9),
        n_jobs=2,
        parallel_backend="process",
        cpu_threads=2,
    )

    assert summary.execution_backend == "process"
    assert summary.input_transport == "shared_memory"
    assert summary.auc_mean > 0.8


def test_gpu_execution_uses_processes_and_never_shared_threads() -> None:
    class LargeGpuRuntime:
        def recommended_concurrent_workers(self, requested: int, *, cap: int) -> int:
            return min(requested, cap)

    class GpuModel:
        device = torch.device("cuda:0")
        runtime = LargeGpuRuntime()

    backend, workers = _resolve_fold_execution(
        GpuModel(),
        n_jobs=4,
        parallel_backend="auto",
        n_folds=5,
        max_gpu_jobs=None,
    )
    assert (backend, workers) == ("process", 2)

    with pytest.warns(RuntimeWarning, match="GPU folds require isolated processes"):
        backend, workers = _resolve_fold_execution(
            GpuModel(),
            n_jobs=4,
            parallel_backend="thread",
            n_folds=5,
            max_gpu_jobs=4,
        )
    assert (backend, workers) == ("serial", 1)
