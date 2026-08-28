"""Grouped P300 evaluation with fold-local calibration and candidate aggregation."""

from __future__ import annotations

import copy
import multiprocessing as mp
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from baselines.calibration import calibration_data_from_model, fit_logit_calibration
from baselines.validation import group_disjoint_validation_split
from data.artifact import (
    FoldLocalArtifactModel,
    FoldLocalArtifactPolicy,
    apply_fitted_artifact_model,
    apply_fold_local_artifact_policy,
)
from data.artifact_sidecar import (
    default_fold_artifact_sidecar_path,
    fold_artifact_fingerprint,
    load_fold_artifact_sidecar,
    save_fold_artifact_sidecar,
)
from data.qc_features import EpochQCFeatures
from models.decision import decide
from train.runtime import (
    available_cpu_threads,
    configure_spawned_worker_threads,
    cpu_thread_budget,
    resolve_cpu_threads,
)


def loso_folds(subject_ids: Sequence[object]) -> list[tuple[np.ndarray, np.ndarray]]:
    subjects = np.asarray(subject_ids).astype(str)
    unique = np.unique(subjects)
    if len(unique) < 2:
        raise ValueError("LOSO requires at least two subjects.")
    return [(subjects != subject, subjects == subject) for subject in unique]


def within_subject_folds(
    subject_ids: Sequence[object],
    group_ids: Sequence[object],
    *,
    fraction: float = 0.2,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create within-subject folds by holding out complete acquisition groups."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0,1).")
    subjects = np.asarray(subject_ids).astype(str)
    groups = np.asarray(group_ids).astype(str)
    if subjects.ndim != 1 or groups.shape != subjects.shape:
        raise ValueError("subject_ids and group_ids must be aligned one-dimensional arrays.")
    if np.any(np.char.strip(groups) == ""):
        raise ValueError("within-subject evaluation requires non-empty group_ids.")
    rng = np.random.default_rng(seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for subject in np.unique(subjects):
        subject_rows = subjects == subject
        subject_groups = np.unique(groups[subject_rows])
        if len(subject_groups) < 2:
            continue
        n_test = min(len(subject_groups) - 1, max(1, int(round(len(subject_groups) * fraction))))
        test_groups = rng.choice(subject_groups, size=n_test, replace=False)
        test = subject_rows & np.isin(groups, test_groups)
        train = subject_rows & ~np.isin(groups, test_groups)
        folds.append((train, test))
    if not folds:
        raise ValueError(
            "No subject has at least two acquisition groups; random epoch splits are forbidden."
        )
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
    best_task_val_loss: float | None = None
    final_task_val_auc: float | None = None
    artifact_quality: dict[str, object] | None = None
    device: str | None = None
    precision: str | None = None
    batch_size: int | None = None
    validation_batch_size: int | None = None
    preloaded: bool | None = None
    shuffle_each_epoch: bool | None = None
    fused_adam_requested: bool | None = None
    compile_mode_requested: str | None = None
    fused_adam: bool = False
    compile_mode: str | None = None
    compile_scope: str | None = None
    optimizer_fallback_reason: str | None = None
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
    artifact_qc_workers: int = 0
    artifact_qc_cpu_threads_per_worker: int = 0


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


@dataclass(frozen=True)
class _SharedQCFeatureSpecs:
    relative_ptp: _SharedArraySpec
    channel_std_v: _SharedArraySpec
    epoch_scale_v: _SharedArraySpec
    observed_mask: _SharedArraySpec


class _SharedFoldInputs:
    """Own read-only process-shared source arrays for independent fold workers."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        trial_channel_mask: np.ndarray | None,
        qc_features: EpochQCFeatures | None,
    ) -> None:
        self._blocks: list[shared_memory.SharedMemory] = []
        try:
            self.X = self._share(X)
            self.y = self._share(y)
            self.subject_ids = self._share(subject_ids)
            self.trial_channel_mask = (
                None if trial_channel_mask is None else self._share(trial_channel_mask)
            )
            self.qc_features = (
                None
                if qc_features is None
                else _SharedQCFeatureSpecs(
                    relative_ptp=self._share(qc_features.relative_ptp),
                    channel_std_v=self._share(qc_features.channel_std_v),
                    epoch_scale_v=self._share(qc_features.epoch_scale_v),
                    observed_mask=self._share(qc_features.observed_mask),
                )
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
    if getattr(model, "fit_accepts_group_ids", False):
        kwargs = {"group_ids": subject_ids[train]}
        if getattr(model, "fit_accepts_trial_channel_mask", False) and trial_channel_mask is not None:
            kwargs["trial_channel_mask"] = trial_channel_mask[train]
        model.fit(X[train], y[train], **kwargs)
        return

    split = group_disjoint_validation_split(
        subject_ids[train], fraction=0.1, min_groups=2, max_groups=12, seed=0
    )
    outer_X, outer_y = X[train], y[train]
    model.fit(outer_X[split.train_mask], outer_y[split.train_mask])
    model.calibration_logits_ = _predict(
        model,
        outer_X[split.validation_mask],
        None if trial_channel_mask is None else trial_channel_mask[train][split.validation_mask],
    )
    model.calibration_labels_ = outer_y[split.validation_mask]
    model.calibration_source_ = "group_disjoint_validation"


def _can_defer_artifact_zero_fill(model: object) -> bool:
    """Return whether the model enforces an explicit dynamic channel mask.

    This is deliberately an opt-in capability.  A false mask must otherwise
    mean a literal zero signal for legacy estimators, whose interface has no
    way to prevent rejected values from entering their feature extraction.
    """

    return bool(
        getattr(model, "fit_accepts_trial_channel_mask", False)
        and getattr(model, "predict_accepts_trial_channel_mask", False)
        and getattr(model, "accepts_unmaterialized_trial_channel_mask", False)
    )


def _fold_result(
    prototype: object,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    trial_channel_mask: np.ndarray | None,
    qc_features: EpochQCFeatures | None,
    artifact_policy: FoldLocalArtifactPolicy | None,
    fitted_artifact_model: FoldLocalArtifactModel | None = None,
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
    if fitted_artifact_model is not None:
        X, trial_channel_mask, train, artifact_quality = apply_fitted_artifact_model(
            fitted_artifact_model,
            X,
            subject_ids,
            train,
            test,
            trial_channel_mask,
            qc_features,
            materialize_masked_data=not _can_defer_artifact_zero_fill(model),
        )
    elif artifact_policy is not None:
        X, trial_channel_mask, train, artifact_quality = apply_fold_local_artifact_policy(
            artifact_policy,
            X,
            subject_ids,
            train,
            test,
            trial_channel_mask,
            qc_features,
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
        best_task_val_loss=history.get("best_task_val_loss"),
        final_task_val_auc=history.get("final_task_val_auc"),
        artifact_quality=artifact_quality,
        device=runtime.get("device") if isinstance(runtime, dict) else None,
        precision=runtime.get("precision") if isinstance(runtime, dict) else None,
        batch_size=runtime.get("batch_size") if isinstance(runtime, dict) else None,
        validation_batch_size=runtime.get("validation_batch_size")
        if isinstance(runtime, dict)
        else None,
        preloaded=runtime.get("preloaded") if isinstance(runtime, dict) else None,
        shuffle_each_epoch=runtime.get("shuffle_each_epoch")
        if isinstance(runtime, dict)
        else None,
        fused_adam_requested=(
            runtime.get("fused_adam_requested") if isinstance(runtime, dict) else None
        ),
        compile_mode_requested=(
            runtime.get("compile_mode_requested") if isinstance(runtime, dict) else None
        ),
        fused_adam=bool(runtime.get("fused_adam", False)) if isinstance(runtime, dict) else False,
        compile_mode=runtime.get("compile_mode") if isinstance(runtime, dict) else None,
        compile_scope=runtime.get("compile_scope") if isinstance(runtime, dict) else None,
        optimizer_fallback_reason=(
            runtime.get("optimizer_fallback_reason") if isinstance(runtime, dict) else None
        ),
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
            gpu_limit = int(recommend(n_jobs, cap=4)) if callable(recommend) else 1
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
        EpochQCFeatures | None,
        FoldLocalArtifactPolicy | None,
        FoldLocalArtifactModel | None,
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
        qc_features,
        artifact_policy,
        fitted_artifact_model,
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
        qc_features,
        artifact_policy,
        fitted_artifact_model=fitted_artifact_model,
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
        _SharedQCFeatureSpecs | None,
        FoldLocalArtifactPolicy | None,
        FoldLocalArtifactModel | None,
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
        qc_specs,
        artifact_policy,
        fitted_artifact_model,
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
        qc_features = None
        if qc_specs is not None:
            relative_ptp, handle = _attach_shared_array(qc_specs.relative_ptp)
            handles.append(handle)
            channel_std_v, handle = _attach_shared_array(qc_specs.channel_std_v)
            handles.append(handle)
            epoch_scale_v, handle = _attach_shared_array(qc_specs.epoch_scale_v)
            handles.append(handle)
            observed_mask, handle = _attach_shared_array(qc_specs.observed_mask)
            handles.append(handle)
            qc_features = EpochQCFeatures(
                relative_ptp=relative_ptp,
                channel_std_v=channel_std_v,
                epoch_scale_v=epoch_scale_v,
                observed_mask=observed_mask,
            )
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
                    qc_features,
                    artifact_policy,
                    fitted_artifact_model,
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


def _fit_artifact_policy_task(
    task: tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        EpochQCFeatures | None,
        FoldLocalArtifactPolicy,
    ],
) -> tuple[int, FoldLocalArtifactModel]:
    """Fit one fold's small QC policy without constructing a classifier."""

    fold_index, X, subject_ids, train, trial_channel_mask, qc_features, policy = task
    train_features = None if qc_features is None else qc_features.subset(train)
    return (
        fold_index,
        policy.fit(
            X[train],
            subject_ids[train],
            None if trial_channel_mask is None else trial_channel_mask[train],
            train_features,
        ),
    )


def _fit_shared_artifact_policy_task(
    task: tuple[
        int,
        _SharedArraySpec,
        _SharedArraySpec,
        np.ndarray,
        _SharedArraySpec | None,
        _SharedQCFeatureSpecs | None,
        FoldLocalArtifactPolicy,
        int,
    ],
) -> tuple[int, FoldLocalArtifactModel]:
    """Attach shared EEG/QC arrays before one CPU-only policy fit."""

    fold_index, X_spec, subject_spec, train, mask_spec, qc_specs, policy, cpu_threads = task
    handles: list[shared_memory.SharedMemory] = []
    try:
        configure_spawned_worker_threads(cpu_threads)
        X, handle = _attach_shared_array(X_spec)
        handles.append(handle)
        subject_ids, handle = _attach_shared_array(subject_spec)
        handles.append(handle)
        trial_channel_mask = None
        if mask_spec is not None:
            trial_channel_mask, handle = _attach_shared_array(mask_spec)
            handles.append(handle)
        qc_features = None
        if qc_specs is not None:
            relative_ptp, handle = _attach_shared_array(qc_specs.relative_ptp)
            handles.append(handle)
            channel_std_v, handle = _attach_shared_array(qc_specs.channel_std_v)
            handles.append(handle)
            epoch_scale_v, handle = _attach_shared_array(qc_specs.epoch_scale_v)
            handles.append(handle)
            observed_mask, handle = _attach_shared_array(qc_specs.observed_mask)
            handles.append(handle)
            qc_features = EpochQCFeatures(
                relative_ptp=relative_ptp,
                channel_std_v=channel_std_v,
                epoch_scale_v=epoch_scale_v,
                observed_mask=observed_mask,
            )
        with cpu_thread_budget(cpu_threads):
            return _fit_artifact_policy_task(
                (
                    fold_index,
                    X,
                    subject_ids,
                    train,
                    trial_channel_mask,
                    qc_features,
                    policy,
                )
            )
    finally:
        for handle in handles:
            try:
                handle.close()
            except OSError:
                pass


def resolve_artifact_qc_workers(
    n_folds: int,
    *,
    artifact_qc_jobs: int | None,
    cpu_threads: int | None,
    available_threads: int | None = None,
) -> int:
    """Choose a bounded QC worker count from the usable CPU budget.

    Each fit has serial masked-median work plus BLAS contractions.  One worker
    per usable CPU thread keeps the serial median scans busy. The automatic
    policy is capped at 16 workers because each fit holds a training-set slice
    in addition to the shared source array.
    """

    if n_folds < 1:
        raise ValueError("n_folds must be positive.")
    if artifact_qc_jobs is not None and artifact_qc_jobs < 1:
        raise ValueError("artifact_qc_jobs must be positive or None.")
    available = available_cpu_threads() if available_threads is None else int(available_threads)
    if available < 1:
        raise ValueError("available_threads must be positive.")
    budget = available if cpu_threads is None else min(int(cpu_threads), available)
    requested = min(16, budget) if artifact_qc_jobs is None else int(artifact_qc_jobs)
    return min(n_folds, budget, requested)


def precompute_fold_local_artifact_models(
    X: np.ndarray,
    subject_ids: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    trial_channel_mask: np.ndarray | None,
    qc_features: EpochQCFeatures | None,
    artifact_policy: FoldLocalArtifactPolicy | None,
    artifact_qc_jobs: int | None,
    cpu_threads: int | None,
) -> dict[int, FoldLocalArtifactModel]:
    """Compile all outer-fold QC policies on CPU before model fitting.

    The return payload contains only fold-local fitted thresholds.  It is small
    enough to pass to accelerator workers and prevents each worker from
    rebuilding the same CPU CV state between GPU training phases.
    """

    if artifact_policy is None:
        return {}
    workers = resolve_artifact_qc_workers(
        len(folds),
        artifact_qc_jobs=artifact_qc_jobs,
        cpu_threads=cpu_threads,
    )
    cpu_threads_per_worker = resolve_cpu_threads(workers, total_threads=cpu_threads)
    if workers == 1:
        with cpu_thread_budget(cpu_threads_per_worker):
            pairs = [
                _fit_artifact_policy_task(
                    (
                        index,
                        X,
                        subject_ids,
                        train,
                        trial_channel_mask,
                        qc_features,
                        artifact_policy,
                    )
                )
                for index, (train, _) in enumerate(folds)
            ]
    else:
        with _SharedFoldInputs(
            X,
            np.empty(0, dtype=np.int8),
            subject_ids,
            trial_channel_mask,
            qc_features,
        ) as shared_inputs:
            tasks = [
                (
                    index,
                    shared_inputs.X,
                    shared_inputs.subject_ids,
                    train,
                    shared_inputs.trial_channel_mask,
                    shared_inputs.qc_features,
                    artifact_policy,
                    cpu_threads_per_worker,
                )
                for index, (train, _) in enumerate(folds)
            ]
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                pairs = list(executor.map(_fit_shared_artifact_policy_task, tasks))
    return dict(pairs)


def resolve_fold_local_artifact_models(
    X: np.ndarray,
    subject_ids: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    cache_path: str | Path,
    cache_sha256: str,
    trial_channel_mask: np.ndarray | None,
    qc_features: EpochQCFeatures | None,
    artifact_policy: FoldLocalArtifactPolicy | None,
    artifact_qc_jobs: int | None,
    cpu_threads: int | None,
) -> tuple[dict[int, FoldLocalArtifactModel], dict[str, object]]:
    """Load an exact fold-QC sidecar or compute and persist it once."""

    if artifact_policy is None:
        return {}, {"enabled": False, "hit": False, "fit_seconds": 0.0}
    normalized_folds = [
        (np.asarray(train, dtype=bool), np.asarray(test, dtype=bool)) for train, test in folds
    ]
    fingerprint = fold_artifact_fingerprint(
        cache_sha256=cache_sha256,
        folds=normalized_folds,
        policy=artifact_policy,
    )
    sidecar_path = default_fold_artifact_sidecar_path(cache_path, fingerprint)
    started = time.perf_counter()
    loaded = load_fold_artifact_sidecar(
        sidecar_path,
        expected_fingerprint=fingerprint,
        expected_fold_count=len(normalized_folds),
    )
    if loaded is not None:
        return loaded, {
            "enabled": True,
            "hit": True,
            "fingerprint": fingerprint,
            "path": str(sidecar_path),
            "fit_seconds": 0.0,
            "load_seconds": time.perf_counter() - started,
        }
    models = precompute_fold_local_artifact_models(
        X,
        subject_ids,
        normalized_folds,
        trial_channel_mask=trial_channel_mask,
        qc_features=qc_features,
        artifact_policy=artifact_policy,
        artifact_qc_jobs=artifact_qc_jobs,
        cpu_threads=cpu_threads,
    )
    fit_seconds = time.perf_counter() - started
    save_fold_artifact_sidecar(
        sidecar_path,
        fingerprint=fingerprint,
        cache_sha256=cache_sha256,
        policy=artifact_policy,
        models=models,
    )
    return models, {
        "enabled": True,
        "hit": False,
        "fingerprint": fingerprint,
        "path": str(sidecar_path),
        "fit_seconds": fit_seconds,
        "load_seconds": 0.0,
    }


def _run_fold_tasks(
    prototype: object,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    trial_channel_mask: np.ndarray | None,
    qc_features: EpochQCFeatures | None,
    artifact_policy: FoldLocalArtifactPolicy | None,
    fitted_artifact_models: Mapping[int, FoldLocalArtifactModel] | None,
    artifact_qc_jobs: int | None,
    n_jobs: int,
    parallel_backend: str,
    fold_id_offset: int,
    max_gpu_jobs: int | None,
    cpu_threads: int | None,
    on_fold_result: Callable[[tuple[int, BinaryFoldResult, np.ndarray, object]], None] | None = None,
) -> tuple[list[tuple[int, BinaryFoldResult, np.ndarray, object]], str, int, str, int, int, int]:
    backend, effective_n_jobs = _resolve_fold_execution(
        prototype,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        n_folds=len(folds),
        max_gpu_jobs=max_gpu_jobs,
    )
    cpu_threads_per_worker = resolve_cpu_threads(effective_n_jobs, total_threads=cpu_threads)
    artifact_qc_workers = 0
    if artifact_policy is not None and fitted_artifact_models is None:
        artifact_qc_workers = resolve_artifact_qc_workers(
            len(folds), artifact_qc_jobs=artifact_qc_jobs, cpu_threads=cpu_threads
        )
    artifact_qc_cpu_threads_per_worker = (
        0
        if artifact_qc_workers == 0
        else resolve_cpu_threads(artifact_qc_workers, total_threads=cpu_threads)
    )
    if fitted_artifact_models is None:
        fitted_artifact_models = precompute_fold_local_artifact_models(
            X,
            subject_ids,
            folds,
            trial_channel_mask=trial_channel_mask,
            qc_features=qc_features,
            artifact_policy=artifact_policy,
            artifact_qc_jobs=artifact_qc_jobs,
            cpu_threads=cpu_threads,
        )
    else:
        fitted_artifact_models = dict(fitted_artifact_models)
        expected = set(range(len(folds))) if artifact_policy is not None else set()
        if set(fitted_artifact_models) != expected:
            raise ValueError("fitted_artifact_models must contain exactly one model per fold.")
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
            qc_features,
            artifact_policy,
            fitted_artifact_models.get(index),
            index + fold_id_offset,
            effective_n_jobs,
        )
        for index, (train, test) in enumerate(folds)
    ]
    if backend == "serial":
        with cpu_thread_budget(cpu_threads_per_worker):
            results = []
            for task in tasks:
                result = _run_fold_task(task)
                results.append(result)
                if on_fold_result is not None:
                    on_fold_result(result)
        return (
            results,
            backend,
            effective_n_jobs,
            "direct",
            cpu_threads_per_worker,
            artifact_qc_workers,
            artifact_qc_cpu_threads_per_worker,
        )
    if backend == "thread":
        with cpu_thread_budget(cpu_threads_per_worker):
            with ThreadPoolExecutor(max_workers=effective_n_jobs) as executor:
                futures = [executor.submit(_run_fold_task, task) for task in tasks]
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if on_fold_result is not None:
                        on_fold_result(result)
        transport = "direct"
    else:
        with _SharedFoldInputs(X, y, subject_ids, trial_channel_mask, qc_features) as shared_inputs:
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
                    shared_inputs.qc_features,
                    artifact_policy,
                    fitted_artifact_models.get(index),
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
                futures = [executor.submit(_run_shared_fold_task, task) for task in shared_tasks]
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if on_fold_result is not None:
                        on_fold_result(result)
        transport = "shared_memory"
    return (
        sorted(results, key=lambda value: value[0]),
        backend,
        effective_n_jobs,
        transport,
        cpu_threads_per_worker,
        artifact_qc_workers,
        artifact_qc_cpu_threads_per_worker,
    )


def evaluate_binary(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    fit_group_ids: np.ndarray | None = None,
    trial_channel_mask: np.ndarray | None = None,
    qc_features: EpochQCFeatures | None = None,
    artifact_policy: FoldLocalArtifactPolicy | None = None,
    fitted_artifact_models: Mapping[int, FoldLocalArtifactModel] | None = None,
    artifact_qc_jobs: int | None = None,
    fold_protocol: str | None = None,
    on_fold_end: Callable[[int, BinaryFoldResult], None] | None = None,
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
    fit_groups = (
        subject_ids
        if fit_group_ids is None
        else np.asarray(fit_group_ids).astype(str)
    )
    if fit_groups.shape != (len(X),) or np.any(np.char.strip(fit_groups) == ""):
        raise ValueError("fit_group_ids must contain one non-empty group per epoch.")
    if trial_channel_mask is not None:
        trial_channel_mask = np.asarray(trial_channel_mask, dtype=bool)
        if trial_channel_mask.shape != X.shape[:2] or not trial_channel_mask.any(axis=1).all():
            raise ValueError("trial_channel_mask must retain at least one channel per epoch.")
    if qc_features is not None:
        qc_features.validate(n_epochs=len(X), n_channels=X.shape[1])
    if fold_id_offset < 0:
        raise ValueError("fold_id_offset must be non-negative.")
    (
        fold_results,
        backend,
        effective_n_jobs,
        input_transport,
        cpu_threads_per_worker,
        artifact_qc_workers,
        artifact_qc_cpu_threads_per_worker,
    ) = _run_fold_tasks(
        model,
        X,
        y,
        fit_groups,
        _validate_folds(folds, len(X)),
        trial_channel_mask=trial_channel_mask,
        qc_features=qc_features,
        artifact_policy=artifact_policy,
        fitted_artifact_models=fitted_artifact_models,
        artifact_qc_jobs=artifact_qc_jobs,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        fold_id_offset=fold_id_offset,
        max_gpu_jobs=max_gpu_jobs,
        cpu_threads=cpu_threads,
        on_fold_result=(
            None
            if on_fold_end is None
            else lambda result: on_fold_end(result[0] + fold_id_offset, result[1])
        ),
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
        artifact_qc_workers=artifact_qc_workers,
        artifact_qc_cpu_threads_per_worker=artifact_qc_cpu_threads_per_worker,
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
    fit_group_ids: np.ndarray | None = None,
    event_timeline: object | None = None,
    trial_channel_mask: np.ndarray | None = None,
    qc_features: EpochQCFeatures | None = None,
    artifact_policy: FoldLocalArtifactPolicy | None = None,
    fitted_artifact_models: Mapping[int, FoldLocalArtifactModel] | None = None,
    artifact_qc_jobs: int | None = None,
    fold_protocol: str | None = None,
    on_fold_end: Callable[[int, CandidateFoldResult], None] | None = None,
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
    fit_groups = (
        group_ids if fit_group_ids is None else np.asarray(fit_group_ids).astype(str)
    )
    codes = np.asarray(candidate_codes)
    _validate_binary_inputs(X, y, group_ids)
    if fit_groups.shape != (len(X),) or np.any(np.char.strip(fit_groups) == ""):
        raise ValueError("fit_group_ids must contain one non-empty group per epoch.")
    if codes.shape != (len(X),):
        raise ValueError("candidate_codes must align with X.")
    if trial_channel_mask is not None:
        trial_channel_mask = np.asarray(trial_channel_mask, dtype=bool)
        if trial_channel_mask.shape != X.shape[:2] or not trial_channel_mask.any(axis=1).all():
            raise ValueError("trial_channel_mask must retain at least one channel per epoch.")
    if qc_features is not None:
        qc_features.validate(n_epochs=len(X), n_channels=X.shape[1])
    if fold_id_offset < 0:
        raise ValueError("fold_id_offset must be non-negative.")
    results: list[CandidateFoldResult] = []
    records: list[tuple[object, object, str]] = []
    validated_folds = _validate_folds(folds, len(X))
    (
        fold_results,
        backend,
        effective_n_jobs,
        input_transport,
        cpu_threads_per_worker,
        artifact_qc_workers,
        artifact_qc_cpu_threads_per_worker,
    ) = _run_fold_tasks(
        model,
        X,
        y,
        fit_groups,
        validated_folds,
        trial_channel_mask=trial_channel_mask,
        qc_features=qc_features,
        artifact_policy=artifact_policy,
        fitted_artifact_models=fitted_artifact_models,
        artifact_qc_jobs=artifact_qc_jobs,
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
        result = CandidateFoldResult(
            **binary.__dict__,
            hit_rate=hit_rate,
            n_decisions=len(fold_records),
            decision_records=fold_records,
        )
        results.append(result)
        if on_fold_end is not None:
            on_fold_end(fold_index + fold_id_offset, result)
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
        artifact_qc_workers=artifact_qc_workers,
        artifact_qc_cpu_threads_per_worker=artifact_qc_cpu_threads_per_worker,
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
    exceedances = int(np.count_nonzero(np.abs((signs * delta).mean(axis=1)) >= abs(observed)))
    # Plus-one correction prevents an impossible p=0 from a finite Monte Carlo sample.
    p_value = float((exceedances + 1) / (int(n_perm) + 1))
    return observed, p_value
