from __future__ import annotations

import numpy as np
import pytest
import torch

import baselines.evaluate as evaluate_module
from baselines.classic import WindowLogisticRegression
from baselines.evaluate import (
    _resolve_fold_execution,
    evaluate_binary,
    evaluate_candidate_selection,
    loso_folds,
    paired_permutation_test,
    precompute_fold_local_artifact_models,
    resolve_artifact_qc_workers,
    within_subject_folds,
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


def test_within_subject_folds_hold_out_complete_groups() -> None:
    subjects = np.repeat(["s1", "s2"], 12)
    groups = np.tile(np.repeat(["run-1", "run-2", "run-3"], 4), 2)

    naive_test = np.arange(len(groups)) % 2 == 0
    naive_train = ~naive_test
    assert set(groups[naive_train]) & set(groups[naive_test])

    folds = within_subject_folds(subjects, groups, fraction=1 / 3, seed=7)

    assert len(folds) == 2
    for train, test in folds:
        assert len(np.unique(subjects[train | test])) == 1
        assert set(groups[train]).isdisjoint(set(groups[test]))
        for group in np.unique(groups[test]):
            subject = np.unique(subjects[test])[0]
            expected = (subjects == subject) & (groups == group)
            assert np.all(test[expected])


def test_paired_permutation_uses_finite_sample_plus_one_correction() -> None:
    delta, p_value = paired_permutation_test(
        np.ones(8), np.zeros(8), n_perm=100, seed=4
    )

    assert delta == 1.0
    assert 0.0 < p_value <= 1.0


def test_within_subject_folds_refuse_random_epoch_fallback() -> None:
    subjects = np.repeat(["s1", "s2"], 8)
    groups = subjects.copy()

    with pytest.raises(ValueError, match="random epoch splits are forbidden"):
        within_subject_folds(subjects, groups)


def test_within_subject_evaluation_uses_groups_for_inner_validation() -> None:
    rng = np.random.default_rng(17)
    groups = np.repeat([f"run-{index}" for index in range(6)], 12)
    subjects = np.repeat("single-subject", len(groups))
    y = np.tile(np.array([0, 1] * 6, dtype=np.int64), 6)
    X = rng.normal(size=(len(y), 3, 96)).astype(np.float32)
    X[y == 1, 2, 42:62] += 2.5
    folds = within_subject_folds(subjects, groups, fraction=0.2, seed=3)
    model = WindowLogisticRegression(sfreq=128.0, tmin=-0.2, window_ms=(125.0, 300.0))

    with pytest.raises(ValueError, match="at least four available groups"):
        evaluate_binary(model, X, y, subjects, folds)

    summary = evaluate_binary(
        model,
        X,
        y,
        subjects,
        folds,
        fit_group_ids=groups,
    )

    assert summary.auc_mean > 0.8
    assert summary.per_fold[0].threshold_source == "group_disjoint_validation"


def test_binary_loso_reports_auc_and_bacc() -> None:
    X, y, subjects = _p300_data()
    fold_events = []
    summary = evaluate_binary(
        WindowLogisticRegression(sfreq=128.0, tmin=-0.2, window_ms=(125.0, 300.0)),
        X,
        y,
        subjects,
        loso_folds(subjects),
        fold_id_offset=4,
        on_fold_end=lambda fold_id, result: fold_events.append((fold_id, result)),
    )

    assert summary.auc_mean > 0.8
    assert summary.balanced_acc_mean > 0.7
    assert [fold_id for fold_id, _ in fold_events] == list(range(4, 10))
    assert [result.auc for _, result in fold_events] == [result.auc for result in summary.per_fold]


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
        fit_group_ids=physical_subjects,
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
    assert summary.artifact_qc_workers == 2
    assert summary.artifact_qc_cpu_threads_per_worker == 1
    assert summary.auc_mean > 0.8


def test_precomputed_artifact_models_are_reused_across_candidates(monkeypatch) -> None:
    X, y, subjects = _p300_data()
    folds = loso_folds(subjects)
    features = compute_epoch_qc_features(X, channel_mask=np.ones(X.shape[1], dtype=bool))
    policy = FoldLocalArtifactPolicy(global_scale_mad_z=1e9)
    fitted = precompute_fold_local_artifact_models(
        X,
        subjects,
        folds,
        trial_channel_mask=None,
        qc_features=features,
        artifact_policy=policy,
        artifact_qc_jobs=1,
        cpu_threads=1,
    )

    def fail_recompute(*_args, **_kwargs):
        raise AssertionError("frozen fold-local QC must be reused")

    monkeypatch.setattr(evaluate_module, "precompute_fold_local_artifact_models", fail_recompute)
    summary = evaluate_binary(
        WindowLogisticRegression(sfreq=128.0, tmin=-0.2, window_ms=(125.0, 300.0)),
        X,
        y,
        subjects,
        folds,
        qc_features=features,
        artifact_policy=policy,
        fitted_artifact_models=fitted,
    )

    assert summary.artifact_qc_workers == 0
    assert summary.auc_mean > 0.8


def test_precomputed_artifact_models_validate_cpu_worker_budget() -> None:
    X, _, subjects = _p300_data()
    features = compute_epoch_qc_features(X, channel_mask=np.ones(X.shape[1], dtype=bool))
    policy = FoldLocalArtifactPolicy(global_scale_mad_z=1e9)

    with pytest.raises(ValueError, match="artifact_qc_jobs"):
        precompute_fold_local_artifact_models(
            X,
            subjects,
            loso_folds(subjects),
            trial_channel_mask=None,
            qc_features=features,
            artifact_policy=policy,
            artifact_qc_jobs=0,
            cpu_threads=2,
        )


def test_artifact_qc_worker_default_scales_with_cpu_budget() -> None:
    assert (
        resolve_artifact_qc_workers(
            64,
            artifact_qc_jobs=None,
            cpu_threads=128,
            available_threads=128,
        )
        == 16
    )
    assert (
        resolve_artifact_qc_workers(
            64,
            artifact_qc_jobs=24,
            cpu_threads=128,
            available_threads=128,
        )
        == 24
    )
    assert (
        resolve_artifact_qc_workers(
            64,
            artifact_qc_jobs=None,
            cpu_threads=32,
            available_threads=128,
        )
        == 16
    )


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
