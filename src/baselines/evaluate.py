"""Grouped P300 evaluation with fold-local calibration and candidate aggregation."""

from __future__ import annotations

import copy
import multiprocessing as mp
import time
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import shared_memory

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from baselines.calibration import calibration_data_from_model, fit_logit_calibration
from baselines.validation import subject_disjoint_validation_split
from data.artifact import FoldLocalArtifactPolicy, apply_fold_local_artifact_policy
from models.decision import decide
from train.runtime import configure_spawned_worker_threads, cpu_thread_budget, resolve_cpu_threads


def loso_folds(subject_ids: Sequence[object]) -> list[tuple[np.ndarray, np.ndarray]]:
    subjects = np.asarray(subject_ids).astype(str)
    unique = np.unique(subjects)
    if len(unique) < 2:
        raise ValueError("LOSO requires at least two subjects.")
    return [(subjects != subject, subjects == subject) for subject in unique]


def within_subject_folds(
    subject_ids: Sequence[object],
    *,
    fraction: float = 0.2,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create chronological-agnostic grouped folds only when called explicitly."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0,1).")
    subjects = np.asarray(subject_ids).astype(str)
    rng = np.random.default_rng(seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for subject in np.unique(subjects):
        rows = np.flatnonzero(subjects == subject)
        if len(rows) < 2:
            continue
        n_test = max(1, int(round(len(rows) * fraction)))
        test_rows = rng.choice(rows, size=n_test, replace=False)
        test = np.zeros(len(subjects), dtype=bool)
        test[test_rows] = True
        train = np.zeros(len(subjects), dtype=bool)
        train[rows] = ~test[rows]
        folds.append((train, test))
    if not folds:
        raise ValueError("No subject has enough trials for a within-subject fold.")
    return folds


@dataclass
class BinaryFoldResult:
    balanced_acc: float
    auc: float
    n_test_trials: int
    threshold: float
    threshold_source: str
    fit_sec: float
    epochs_ran: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    artifact_quality: dict[str, object] | None = None
    device: str | None = None
    precision: str | None = None
    batch_size: int | None = None
    fit_peak_allocated_mb: float | None = None
    fit_peak_reserved_mb: float | None = None
    oom_retries: int = 0
    shared_worker_count: int = 1


@dataclass
class BinarySummary:
    balanced_acc_mean: float
    balanced_acc_std: float
    auc_mean: float
    per_fold: list[BinaryFoldResult] = field(default_factory=list)
    execution_backend: str = "serial"
    effective_n_jobs: int = 1
    input_transport: str = "direct"
    cpu_threads_per_worker: int = 1


@dataclass
class CandidateFoldResult(BinaryFoldResult):
    hit_rate: float = float("nan")
    n_decisions: int = 0
    decision_records: list[tuple[object, object, str]] = field(default_factory=list)


@dataclass
class CandidateSummary(BinarySummary):
    hit_rate_mean: float = float("nan")
    primary_hit_rate: float = float("nan")
    subject_records: list[tuple[object, object, str]] = field(default_factory=list)


@dataclass(frozen=True)
class _SharedArraySpec:
    name: str
    shape: tuple[int, ...]
    dtype: str


class _SharedFoldInputs:
    """Own read-only process-shared source arrays for independent fold workers."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        trial_channel_mask: np.ndarray | None,
    ) -> None:
        self._blocks: list[shared_memory.SharedMemory] = []
        try:
            self.X = self._share(X)
            self.y = self._share(y)
            self.subject_ids = self._share(subject_ids)
            self.trial_channel_mask = (
                None if trial_channel_mask is None else self._share(trial_channel_mask)
            )
        except Exception:
            self.close()
            self.unlink()
            raise

    def _share(self, value: np.ndarray) -> _SharedArraySpec:
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("Process-shared fold inputs cannot use object dtypes.")
        block = shared_memory.SharedMemory(create=True, size=max(array.nbytes, 1))
        target = np.ndarray(array.shape, dtype=array.dtype, buffer=block.buf)
        target[...] = array
        self._blocks.append(block)
        return _SharedArraySpec(block.name, tuple(array.shape), array.dtype.str)

    def close(self) -> None:
        for block in self._blocks:
            try:
                block.close()
            except OSError:
                pass

    def unlink(self) -> None:
        for block in self._blocks:
            try:
                block.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> _SharedFoldInputs:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
        self.unlink()


def _attach_shared_array(spec: _SharedArraySpec) -> tuple[np.ndarray, shared_memory.SharedMemory]:
    block = shared_memory.SharedMemory(name=spec.name)
    array = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype), buffer=block.buf)
    array.setflags(write=False)
    return array, block


def _validate_binary_inputs(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray) -> None:
    if X.ndim != 3 or not len(X):
        raise ValueError("X must be a non-empty (N,C,T) array.")
    if not np.issubdtype(X.dtype, np.floating) or not np.isfinite(X).all():
        raise ValueError("X must be finite floating-point EEG data.")
    if y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y must be an integer vector aligned with X.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Evaluation requires binary labels {0,1}.")
    if subject_ids.shape != (len(X),):
        raise ValueError("subject_ids must align with X.")


def _validate_folds(
    folds: Sequence[tuple[np.ndarray, np.ndarray]], n_rows: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not folds:
        raise ValueError("At least one fold is required.")
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for index, (train, test) in enumerate(folds):
        train = np.asarray(train, dtype=bool)
        test = np.asarray(test, dtype=bool)
        if train.shape != (n_rows,) or test.shape != (n_rows,):
            raise ValueError(f"Fold {index} masks must have shape ({n_rows},).")
        if not train.any() or not test.any() or (train & test).any():
            raise ValueError(f"Fold {index} must have non-overlapping non-empty train/test rows.")
        output.append((train, test))
    return output


def _predict(model: object, X: np.ndarray, trial_channel_mask: np.ndarray | None) -> np.ndarray:
    kwargs = {}
    if getattr(model, "predict_accepts_trial_channel_mask", False) and trial_channel_mask is not None:
        kwargs["trial_channel_mask"] = trial_channel_mask
    logits = np.asarray(model.predict_logit(X, **kwargs), dtype=float)
    if logits.shape != (len(X),) or not np.isfinite(logits).all():
        raise ValueError("predict_logit must return one finite value per epoch.")
    return logits


def _fit(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    train: np.ndarray,
    trial_channel_mask: np.ndarray | None,
) -> None:
    for name in ("calibration_logits_", "calibration_labels_", "calibration_source_"):
        setattr(model, name, None)
    if getattr(model, "fit_accepts_subject_ids", False):
        kwargs = {"subject_ids": subject_ids[train]}
        if getattr(model, "fit_accepts_trial_channel_mask", False) and trial_channel_mask is not None:
            kwargs["trial_channel_mask"] = trial_channel_mask[train]
        model.fit(X[train], y[train], **kwargs)
        return

    split = subject_disjoint_validation_split(
        subject_ids[train], fraction=0.1, min_subjects=2, max_subjects=12, seed=0
    )
    outer_X, outer_y = X[train], y[train]
    model.fit(outer_X[split.train_mask], outer_y[split.train_mask])
    model.calibration_logits_ = _predict(
        model,
        outer_X[split.validation_mask],
        None if trial_channel_mask is None else trial_channel_mask[train][split.validation_mask],
    )
    model.calibration_labels_ = outer_y[split.validation_mask]
    model.calibration_source_ = "subject_disjoint_validation"


def _fold_result(
    prototype: object,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    trial_channel_mask: np.ndarray | None,
    artifact_policy: FoldLocalArtifactPolicy | None,
    fold_id: int | None = None,
    shared_worker_count: int = 1,
) -> tuple[BinaryFoldResult, np.ndarray, object]:
    model = copy.deepcopy(prototype)
    configure_budget = getattr(model, "configure_runtime_worker_budget", None)
    if callable(configure_budget):
        configure_budget(shared_worker_count)
    configure_fold = getattr(model, "configure_evaluation_fold", None)
    if callable(configure_fold):
        configure_fold(fold_id)
    artifact_quality = None
    if artifact_policy is not None:
        X, trial_channel_mask, train, artifact_quality = apply_fold_local_artifact_policy(
            artifact_policy, X, subject_ids, train, test, trial_channel_mask
        )
    started = time.perf_counter()
    _fit(model, X, y, subject_ids, train, trial_channel_mask)
    fit_sec = time.perf_counter() - started
    calibration_logits, calibration_y, source = calibration_data_from_model(model, X[train], y[train])
    calibration = fit_logit_calibration(calibration_logits, calibration_y, source=source)
    logits = _predict(model, X[test], None if trial_channel_mask is None else trial_channel_mask[test])
    y_test = y[test]
    bacc = float("nan")
    auc = float("nan")
    if len(np.unique(y_test)) == 2:
        bacc = float(balanced_accuracy_score(y_test, logits >= calibration.threshold))
        auc = float(roc_auc_score(y_test, logits))
    history = getattr(model, "last_history", {}) or {}
    runtime = getattr(model, "last_runtime", {}) or {}
    memory = runtime.get("memory", {}) if isinstance(runtime, dict) else {}
    result = BinaryFoldResult(
        balanced_acc=bacc,
        auc=auc,
        n_test_trials=int(test.sum()),
        threshold=calibration.threshold,
        threshold_source=calibration.source,
        fit_sec=fit_sec,
        epochs_ran=len(history.get("train_losses", ())),
        train_losses=[float(value) for value in history.get("train_losses", ())],
        val_losses=[float(value) for value in history.get("val_losses", ())],
        best_epoch=history.get("best_epoch"),
        artifact_quality=artifact_quality,
        device=runtime.get("device") if isinstance(runtime, dict) else None,
        precision=runtime.get("precision") if isinstance(runtime, dict) else None,
        batch_size=runtime.get("batch_size") if isinstance(runtime, dict) else None,
        fit_peak_allocated_mb=(
            memory.get("peak_allocated_mb") if isinstance(memory, dict) else None
        ),
        fit_peak_reserved_mb=(
            memory.get("peak_reserved_mb") if isinstance(memory, dict) else None
        ),
        oom_retries=int(runtime.get("oom_retries", 0)) if isinstance(runtime, dict) else 0,
        shared_worker_count=(
            int(runtime.get("shared_worker_count", 1)) if isinstance(runtime, dict) else 1
        ),
    )
    return result, calibration.to_llr(logits), calibration


def _resolve_fold_execution(
    model: object,
    *,
    n_jobs: int,
    parallel_backend: str,
    n_folds: int,
    max_gpu_jobs: int | None,
) -> tuple[str, int]:
    """Choose a safe fold executor without opening competing CUDA contexts."""

    if n_jobs < 1:
        raise ValueError("n_jobs must be positive.")
    if parallel_backend not in {"auto", "process", "thread"}:
        raise ValueError("parallel_backend must be 'auto', 'process', or 'thread'.")
    if max_gpu_jobs is not None and max_gpu_jobs < 1:
        raise ValueError("max_gpu_jobs must be positive or None.")
    if n_jobs == 1 or n_folds == 1:
        return "serial", 1
    device = getattr(model, "device", None)
    if getattr(device, "type", None) in {"cuda", "xpu"}:
        if parallel_backend == "thread":
            warnings.warn(
                "GPU folds require isolated processes; thread execution was reduced to one worker "
                "to avoid shared CUDA RNG and allocator state.",
                RuntimeWarning,
                stacklevel=3,
            )
            return "serial", 1
        if max_gpu_jobs is None:
            runtime = getattr(model, "runtime", None)
            recommend = getattr(runtime, "recommended_concurrent_workers", None)
            gpu_limit = int(recommend(n_jobs, cap=2)) if callable(recommend) else 1
        else:
            gpu_limit = int(max_gpu_jobs)
        return "process", min(int(n_jobs), n_folds, gpu_limit)
    if parallel_backend == "auto":
        parallel_backend = (
            "process" if getattr(model, "runtime_requires_exclusive_lease", False) else "thread"
        )
    return parallel_backend, min(int(n_jobs), n_folds)


def _run_fold_task(
    task: tuple[
        int,
        object,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        FoldLocalArtifactPolicy | None,
        int,
        int,
    ],
) -> tuple[int, BinaryFoldResult, np.ndarray, object]:
    (
        fold_index,
        prototype,
        X,
        y,
        subject_ids,
        train,
        test,
        trial_channel_mask,
        artifact_policy,
        fold_id,
        shared_worker_count,
    ) = task
    result, llr, calibration = _fold_result(
        prototype,
        X,
        y,
        subject_ids,
        train,
        test,
        trial_channel_mask,
        artifact_policy,
        fold_id=fold_id,
        shared_worker_count=shared_worker_count,
    )
    return fold_index, result, llr, calibration


def _run_shared_fold_task(
    task: tuple[
        int,
        object,
        _SharedArraySpec,
        _SharedArraySpec,
        _SharedArraySpec,
        np.ndarray,
        np.ndarray,
        _SharedArraySpec | None,
        FoldLocalArtifactPolicy | None,
        int,
        int,
        int,
    ],
) -> tuple[int, BinaryFoldResult, np.ndarray, object]:
    """Attach shared source arrays for one process worker, then close handles."""

    (
        fold_index,
        prototype,
        X_spec,
        y_spec,
        subject_spec,
        train,
        test,
        mask_spec,
        artifact_policy,
        fold_id,
        shared_worker_count,
        cpu_threads,
    ) = task
    handles: list[shared_memory.SharedMemory] = []
    try:
        configure_spawned_worker_threads(cpu_threads)
        X, handle = _attach_shared_array(X_spec)
        handles.append(handle)
        y, handle = _attach_shared_array(y_spec)
        handles.append(handle)
        subject_ids, handle = _attach_shared_array(subject_spec)
        handles.append(handle)
        trial_channel_mask = None
        if mask_spec is not None:
            trial_channel_mask, handle = _attach_shared_array(mask_spec)
            handles.append(handle)
        with cpu_thread_budget(cpu_threads):
            return _run_fold_task(
                (
                    fold_index,
                    prototype,
                    X,
                    y,
                    subject_ids,
                    train,
                    test,
                    trial_channel_mask,
                    artifact_policy,
                    fold_id,
                    shared_worker_count,
                )
            )
    finally:
        for handle in handles:
            try:
                handle.close()
            except OSError:
                pass


def _run_fold_tasks(
    prototype: object,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    trial_channel_mask: np.ndarray | None,
    artifact_policy: FoldLocalArtifactPolicy | None,
    n_jobs: int,
    parallel_backend: str,
    fold_id_offset: int,
    max_gpu_jobs: int | None,
    cpu_threads: int | None,
) -> tuple[list[tuple[int, BinaryFoldResult, np.ndarray, object]], str, int, str, int]:
    backend, effective_n_jobs = _resolve_fold_execution(
        prototype,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        n_folds=len(folds),
        max_gpu_jobs=max_gpu_jobs,
    )
    cpu_threads_per_worker = resolve_cpu_threads(effective_n_jobs, total_threads=cpu_threads)
    tasks = [
        (
            index,
            prototype,
            X,
            y,
            subject_ids,
            train,
            test,
            trial_channel_mask,
            artifact_policy,
            index + fold_id_offset,
            effective_n_jobs,
        )
        for index, (train, test) in enumerate(folds)
    ]
    if backend == "serial":
        with cpu_thread_budget(cpu_threads_per_worker):
            results = [_run_fold_task(task) for task in tasks]
        return results, backend, effective_n_jobs, "direct", cpu_threads_per_worker
    if backend == "thread":
        with cpu_thread_budget(cpu_threads_per_worker):
            with ThreadPoolExecutor(max_workers=effective_n_jobs) as executor:
                results = list(executor.map(_run_fold_task, tasks))
        transport = "direct"
    else:
        with _SharedFoldInputs(X, y, subject_ids, trial_channel_mask) as shared_inputs:
            shared_tasks = [
                (
                    index,
                    prototype,
                    shared_inputs.X,
                    shared_inputs.y,
                    shared_inputs.subject_ids,
                    train,
                    test,
                    shared_inputs.trial_channel_mask,
                    artifact_policy,
                    index + fold_id_offset,
                    effective_n_jobs,
                    cpu_threads_per_worker,
                )
                for index, (train, test) in enumerate(folds)
            ]
            with ProcessPoolExecutor(
                max_workers=effective_n_jobs,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                results = list(executor.map(_run_shared_fold_task, shared_tasks))
        transport = "shared_memory"
    return (
        sorted(results, key=lambda value: value[0]),
        backend,
        effective_n_jobs,
        transport,
        cpu_threads_per_worker,
    )


def evaluate_binary(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    trial_channel_mask: np.ndarray | None = None,
    artifact_policy: FoldLocalArtifactPolicy | None = None,
    fold_protocol: str | None = None,
    n_jobs: int = 1,
    parallel_backend: str = "auto",
    fold_id_offset: int = 0,
    max_gpu_jobs: int | None = None,
    cpu_threads: int | None = None,
) -> BinarySummary:
    """Evaluate a target detector with held-out BACC and AUC."""

    del fold_protocol  # Runner metadata; split masks remain the executable protocol.
    X, y, subject_ids = np.asarray(X), np.asarray(y), np.asarray(subject_ids).astype(str)
    _validate_binary_inputs(X, y, subject_ids)
    if trial_channel_mask is not None:
        trial_channel_mask = np.asarray(trial_channel_mask, dtype=bool)
        if trial_channel_mask.shape != X.shape[:2] or not trial_channel_mask.any(axis=1).all():
            raise ValueError("trial_channel_mask must retain at least one channel per epoch.")
    if fold_id_offset < 0:
        raise ValueError("fold_id_offset must be non-negative.")
    fold_results, backend, effective_n_jobs, input_transport, cpu_threads_per_worker = _run_fold_tasks(
        model,
        X,
        y,
        subject_ids,
        _validate_folds(folds, len(X)),
        trial_channel_mask=trial_channel_mask,
        artifact_policy=artifact_policy,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        fold_id_offset=fold_id_offset,
        max_gpu_jobs=max_gpu_jobs,
        cpu_threads=cpu_threads,
    )
    results = [result for _, result, _, _ in fold_results]
    bacc = np.asarray([fold.balanced_acc for fold in results], dtype=float)
    auc = np.asarray([fold.auc for fold in results], dtype=float)
    return BinarySummary(
        balanced_acc_mean=float(np.nanmean(bacc)),
        balanced_acc_std=float(np.nanstd(bacc)),
        auc_mean=float(np.nanmean(auc)),
        per_fold=results,
        execution_backend=backend,
        effective_n_jobs=effective_n_jobs,
        input_transport=input_transport,
        cpu_threads_per_worker=cpu_threads_per_worker,
    )


def evaluate_candidate_selection(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    candidate_codes: np.ndarray,
    selection_group_ids: np.ndarray,
    truth_by_group: Mapping[object, object],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    candidate_vocab: Sequence[int],
    *,
    fold_subject_ids: np.ndarray | None = None,
    event_timeline: object | None = None,
    trial_channel_mask: np.ndarray | None = None,
    artifact_policy: FoldLocalArtifactPolicy | None = None,
    fold_protocol: str | None = None,
    n_jobs: int = 1,
    parallel_backend: str = "auto",
    fold_id_offset: int = 0,
    max_gpu_jobs: int | None = None,
    cpu_threads: int | None = None,
) -> CandidateSummary:
    """Evaluate calibrated 9-choice candidate evidence on held-out groups."""

    del fold_protocol, event_timeline  # Runner metadata; split masks remain the executable protocol.
    X, y = np.asarray(X), np.asarray(y)
    group_ids = np.asarray(selection_group_ids).astype(str)
    fit_subjects = (
        group_ids if fold_subject_ids is None else np.asarray(fold_subject_ids).astype(str)
    )
    codes = np.asarray(candidate_codes)
    _validate_binary_inputs(X, y, group_ids)
    if fit_subjects.shape != (len(X),):
        raise ValueError("fold_subject_ids must align with X.")
    if codes.shape != (len(X),):
        raise ValueError("candidate_codes must align with X.")
    if trial_channel_mask is not None:
        trial_channel_mask = np.asarray(trial_channel_mask, dtype=bool)
        if trial_channel_mask.shape != X.shape[:2] or not trial_channel_mask.any(axis=1).all():
            raise ValueError("trial_channel_mask must retain at least one channel per epoch.")
    if fold_id_offset < 0:
        raise ValueError("fold_id_offset must be non-negative.")
    results: list[CandidateFoldResult] = []
    records: list[tuple[object, object, str]] = []
    validated_folds = _validate_folds(folds, len(X))
    fold_results, backend, effective_n_jobs, input_transport, cpu_threads_per_worker = _run_fold_tasks(
        model,
        X,
        y,
        fit_subjects,
        validated_folds,
        trial_channel_mask=trial_channel_mask,
        artifact_policy=artifact_policy,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        fold_id_offset=fold_id_offset,
        max_gpu_jobs=max_gpu_jobs,
        cpu_threads=cpu_threads,
    )
    for fold_index, binary, llr, _ in fold_results:
        _, test = validated_folds[fold_index]
        decision = decide(llr, codes[test], group_ids[test], candidate_vocab, center_logits=False)
        fold_records = [
            (predicted, truth_by_group[str(group)], str(group))
            for predicted, group in zip(decision.predicted, decision.subject_ids, strict=True)
            if str(group) in truth_by_group
        ]
        hit_rate = (
            float(np.mean([predicted == truth for predicted, truth, _ in fold_records]))
            if fold_records
            else float("nan")
        )
        results.append(
            CandidateFoldResult(**binary.__dict__, hit_rate=hit_rate, n_decisions=len(fold_records), decision_records=fold_records)
        )
        records.extend(fold_records)
    bacc = np.asarray([fold.balanced_acc for fold in results], dtype=float)
    auc = np.asarray([fold.auc for fold in results], dtype=float)
    hit = np.asarray([fold.hit_rate for fold in results], dtype=float)
    overall = float(np.mean([predicted == truth for predicted, truth, _ in records])) if records else float("nan")
    return CandidateSummary(
        balanced_acc_mean=float(np.nanmean(bacc)),
        balanced_acc_std=float(np.nanstd(bacc)),
        auc_mean=float(np.nanmean(auc)),
        per_fold=results,
        hit_rate_mean=float(np.nanmean(hit)),
        primary_hit_rate=overall,
        subject_records=records,
        execution_backend=backend,
        effective_n_jobs=effective_n_jobs,
        input_transport=input_transport,
        cpu_threads_per_worker=cpu_threads_per_worker,
    )


def paired_permutation_test(
    scores_a: Sequence[float], scores_b: Sequence[float], *, n_perm: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """Two-sided paired sign-flip permutation test for independent outer units."""

    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("paired scores must be non-empty one-dimensional arrays of equal length.")
    delta = a - b
    observed = float(delta.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(int(n_perm), len(delta)))
    p_value = float((np.abs((signs * delta).mean(axis=1)) >= abs(observed)).mean())
    return observed, p_value
