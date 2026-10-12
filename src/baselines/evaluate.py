"""Grouped P300 evaluation with fold-local calibration and candidate aggregation."""

from __future__ import annotations

import copy
import multiprocessing as mp
import time
import warnings
from collections import Counter
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
    apply_fitted_artifact_model_with_test_exclusion,
    apply_fold_local_artifact_policy,
)
from data.artifact_sidecar import (
    default_fold_artifact_sidecar_path,
    fold_artifact_fingerprint,
    load_fold_artifact_sidecar,
    save_fold_artifact_sidecar,
)
from data.qc_features import EpochQCFeatures
from models.candidate_evidence import (
    DEFAULT_EVIDENCE_AGGREGATION,
    EVIDENCE_AGGREGATIONS,
    candidate_evidence_scores,
    decide_by_group,
)
from train.runtime import (
    available_cpu_threads,
    configure_spawned_worker_threads,
    cpu_thread_budget,
    resolve_cpu_threads,
)

LOSO_METRICS_SCHEMA_V1 = "n2p3_eeg_loso_metrics/1"
LOSO_METRICS_SCHEMA_V2 = "n2p3_eeg_loso_metrics/2"


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
    calibration_mode: str | None = None
    calibration_slope: float | None = None
    calibration_temperature: float | None = None
    calibration_offset: float | None = None
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
    materialize_shuffled_epoch: bool | None = None
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
    metric_undefined_folds: list[int] = field(default_factory=list)


def _metric_undefined_folds(results: Sequence[BinaryFoldResult], offset: int) -> list[int]:
    """Folds whose effective test set lacks one class are excluded from AUC/BACC means."""

    return [index + offset for index, fold in enumerate(results) if not np.isfinite(fold.auc)]


@dataclass
class CandidateFoldResult(BinaryFoldResult):
    hit_rate: float = float("nan")
    n_decisions: int = 0
    decision_records: list[tuple[object, object, str]] = field(default_factory=list)
    hit_correct_by_repetition: dict[str, int] | None = None
    hit_total_by_repetition: dict[str, int] | None = None
    hit_scope: dict[str, object] | None = None
    raw_all_hit_rate: float = float("nan")


@dataclass
class CandidateSummary(BinarySummary):
    hit_rate_mean: float = float("nan")
    primary_hit_rate: float | None = None
    primary_hit_scope: dict[str, object] | None = None
    raw_all_hit_rate: float = float("nan")
    subject_records: list[tuple[object, object, str]] = field(default_factory=list)
    hit_at_repetition_macro: dict[str, float] | None = None
    primary_participant_macro_hit_rate: float | None = None
    participant_macro_scope: dict[str, object] | None = None
    hit_at_repetition_participant_macro: dict[str, float] | None = None


def _participant_macro(
    outcomes: Mapping[str, Sequence[bool]], participants: Sequence[str]
) -> float:
    """Average participant means, retaining unavailable participants as misses."""

    values = []
    for participant in participants:
        decisions = outcomes.get(participant, ())
        values.append(float(np.mean(decisions)) if len(decisions) else 0.0)
    return float(np.mean(values)) if values else float("nan")


def validate_loso_metric_record(record: Mapping[str, object]) -> dict[str, object]:
    """Read a metric view; legacy values do not certify a requested denominator."""

    schema = record.get("metric_schema", LOSO_METRICS_SCHEMA_V1)
    if schema == LOSO_METRICS_SCHEMA_V1:
        if any(key in record for key in (
            "primary_participant_macro_hit_rate", "hit_at_repetition_participant_macro",
            "participant_macro_scope", "primary_decision_hit_rate_decision_micro",
            "hit_at_repetition_decision_micro",
        )):
            raise ValueError("legacy LOSO metric record mixes version-2 participant-macro fields")
        return {
            "metric_schema": LOSO_METRICS_SCHEMA_V1,
            "primary_decision_hit_rate": record.get("primary_decision_hit_rate"),
            "primary_decision_hit_rate_decision_micro": None,
            "primary_participant_macro_hit_rate": None,
            "hit_at_repetition_macro": record.get("hit_at_repetition_macro"),
            "hit_at_repetition_decision_micro": None,
            "hit_at_repetition_participant_macro": None,
            "participant_macro_scope": None,
        }
    if schema != LOSO_METRICS_SCHEMA_V2:
        raise ValueError(f"unsupported LOSO metric schema: {schema!r}")
    required = (
        "primary_decision_hit_rate", "primary_decision_hit_rate_decision_micro",
        "primary_participant_macro_hit_rate", "hit_at_repetition_macro",
        "hit_at_repetition_decision_micro", "hit_at_repetition_participant_macro",
        "primary_hit_scope", "participant_macro_scope",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"LOSO metric schema v2 lacks fields: {missing}")
    if record["primary_decision_hit_rate"] != record["primary_decision_hit_rate_decision_micro"]:
        raise ValueError("LOSO decision-micro aliases disagree")
    if record["hit_at_repetition_macro"] != record["hit_at_repetition_decision_micro"]:
        raise ValueError("LOSO repetition decision-micro aliases disagree")
    scope = record["participant_macro_scope"]
    if not isinstance(scope, Mapping) or scope.get("reduction") not in {
        "participant_macro_equal_weight", "not_applicable_binary"
    }:
        raise ValueError("LOSO participant-macro scope is missing its reduction contract")
    count = scope.get("participant_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("LOSO participant-macro scope has an invalid participant count")
    if scope["reduction"] == "participant_macro_equal_weight" and count < 1:
        raise ValueError("LOSO participant-macro scope needs a positive participant count")
    if scope["reduction"] == "not_applicable_binary" and count != 0:
        raise ValueError("binary LOSO participant-macro scope must have zero participants")
    if scope["reduction"] == "not_applicable_binary":
        if scope.get("denominator") != "not_applicable" or any(
            record[key] is not None for key in required if key != "participant_macro_scope"
        ):
            raise ValueError("binary LOSO metric scope must have no candidate metrics")
    else:
        primary_scope = record["primary_hit_scope"]
        if (scope.get("denominator") != "requested_pre_qc_decision_groups"
                or scope.get("unavailable_policy") != "incorrect"
                or scope.get("participant_identity") != "explicit_authority_key"
                or not isinstance(primary_scope, Mapping)
                or primary_scope.get("denominator") != scope["denominator"]
                or primary_scope.get("kind") != "complete_repetition_terminal"):
            raise ValueError("LOSO participant-macro scope disagrees with the requested decision contract")
        for key in ("primary_decision_hit_rate_decision_micro", "primary_participant_macro_hit_rate"):
            _validate_hit_rate(record[key], key)
        micro_curve = record["hit_at_repetition_decision_micro"]
        macro_curve = record["hit_at_repetition_participant_macro"]
        if (not isinstance(micro_curve, Mapping) or not micro_curve
                or not isinstance(macro_curve, Mapping) or micro_curve.keys() != macro_curve.keys()):
            raise ValueError("LOSO repetition micro/macro scopes disagree")
        for curve in (micro_curve, macro_curve):
            for level, value in curve.items():
                if not isinstance(level, str) or not level.isdecimal() or int(level) < 1:
                    raise ValueError("LOSO repetition levels must be positive integer strings")
                _validate_hit_rate(value, f"hit@{level}")
    return {
        "metric_schema": LOSO_METRICS_SCHEMA_V2,
        "primary_decision_hit_rate_decision_micro": record["primary_decision_hit_rate_decision_micro"],
        "primary_decision_hit_rate": record["primary_decision_hit_rate_decision_micro"],
        "primary_participant_macro_hit_rate": record["primary_participant_macro_hit_rate"],
        "primary_hit_scope": record["primary_hit_scope"],
        "hit_at_repetition_macro": record["hit_at_repetition_macro"],
        "hit_at_repetition_decision_micro": record["hit_at_repetition_decision_micro"],
        "hit_at_repetition_participant_macro": record["hit_at_repetition_participant_macro"],
        "participant_macro_scope": dict(scope),
    }


def _validate_hit_rate(value: object, name: str) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or not 0.0 <= value <= 1.0):
        raise ValueError(f"LOSO {name} must be a finite hit rate in [0, 1]")


def _candidate_repetition_predictions(
    logits, codes, group_ids, repetitions, truth_by_group, candidate_vocab, *,
    requested_schedule=None, max_repetitions=None, center_logits=False,
    aggregation=DEFAULT_EVIDENCE_AGGREGATION, curve_repetitions=None,
):
    """Score complete physical prefixes against the immutable pre-QC schedule."""
    logits, codes = np.asarray(logits, dtype=float), np.asarray(codes)
    groups, repetitions = np.asarray(group_ids).astype(str), np.asarray(repetitions)
    if any(value.ndim != 1 for value in (logits, codes, groups, repetitions)) or not (
        len(logits) == len(codes) == len(groups) == len(repetitions)
    ):
        raise ValueError("logits, codes, group_ids, and repetitions must align as vectors.")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain NaN/inf; cannot aggregate candidate evidence.")
    if (not np.issubdtype(repetitions.dtype, np.integer)
            or np.issubdtype(repetitions.dtype, np.bool_) or np.any(repetitions < 0)):
        raise ValueError("repetitions must be non-negative integers.")
    vocabulary = np.asarray(candidate_vocab)
    if vocabulary.ndim != 1 or len(vocabulary) < 2 or len(np.unique(vocabulary)) != len(vocabulary):
        raise ValueError("candidate_vocab must be a unique vector with at least two codes.")
    if aggregation not in EVIDENCE_AGGREGATIONS:
        raise ValueError(f"candidate evaluation requires one of {sorted(EVIDENCE_AGGREGATIONS)}")
    if not np.isin(codes, vocabulary).all():
        raise ValueError("candidate codes outside the declared vocabulary fail closed")
    if max_repetitions is not None and (type(max_repetitions) is not int or max_repetitions < 1):
        raise ValueError("max_repetitions must be a positive integer.")
    if requested_schedule is None:
        requested_schedule = {group: (codes[groups == group], repetitions[groups == group])
                              for group in np.unique(groups)}
    if not requested_schedule:
        raise ValueError("candidate repetition scoring requires a requested schedule")
    if set(np.unique(groups)) - set(requested_schedule):
        raise ValueError("surviving evidence contains an unrequested group")
    normalized = {}
    terminals = {}
    for group, (requested_codes, requested_repetitions) in requested_schedule.items():
        requested_codes, requested_repetitions = np.asarray(requested_codes), np.asarray(requested_repetitions)
        if (requested_codes.ndim != 1 or requested_codes.shape != requested_repetitions.shape
                or not len(requested_codes) or not np.isin(requested_codes, vocabulary).all()
                or not np.issubdtype(requested_repetitions.dtype, np.integer)
                or np.issubdtype(requested_repetitions.dtype, np.bool_)
                or np.any(requested_repetitions < 0)):
            raise ValueError("requested schedule requires aligned candidate codes and integer repetitions")
        if group not in truth_by_group or truth_by_group[group] not in vocabulary:
            raise ValueError("every requested group needs truth in the frozen vocabulary")
        origin = int(requested_repetitions.min())
        terminal = int(requested_repetitions.max()) - origin + 1
        terminals[group] = terminal if max_repetitions is None else max_repetitions
        selected = groups == group
        requested_counts = Counter(zip(requested_repetitions.tolist(), requested_codes.tolist(), strict=True))
        observed_counts = Counter(zip(repetitions[selected].tolist(), codes[selected].tolist(), strict=True))
        if any(count > requested_counts.get(key, 0) for key, count in observed_counts.items()):
            raise ValueError("surviving evidence exceeds the declared physical repetition schedule")
        normalized[group] = (selected, requested_codes, requested_repetitions - origin,
                             repetitions[selected] - origin)
    limit = max(terminals.values()) if curve_repetitions is None else curve_repetitions
    predictions = {level: {} for level in range(1, limit + 1)}
    for group, (group_mask, requested_codes, requested_repetitions, observed_repetitions) in normalized.items():
        group_codes, group_values = codes[group_mask], logits[group_mask]
        requested_counts = np.zeros((limit, len(vocabulary)), dtype=np.int64)
        observed_counts = np.zeros_like(requested_counts)
        for index, candidate in enumerate(vocabulary):
            requested = requested_repetitions[(requested_codes == candidate) & (requested_repetitions < limit)]
            observed = observed_repetitions[(group_codes == candidate) & (observed_repetitions < limit)]
            requested_counts[:, index] = np.bincount(requested, minlength=limit)
            observed_counts[:, index] = np.bincount(observed, minlength=limit)
        complete_prefixes = np.logical_and.accumulate(
            np.all((requested_counts == 1) & (observed_counts == 1), axis=1)
        )
        for level in predictions:
            predicted = None
            if complete_prefixes[level - 1]:
                prefix = observed_repetitions < level
                values = group_values[prefix]
                if center_logits:
                    values = values - values.mean()
                predicted = candidate_evidence_scores(values, group_codes[prefix], vocabulary,
                                                       aggregation=aggregation).predicted
            predictions[level][group] = predicted
    return predictions, terminals


def hit_correct_total_by_repetition(
    logits: np.ndarray,
    codes: np.ndarray,
    group_ids: np.ndarray,
    truth_by_group: Mapping[object, object],
    repetitions: np.ndarray,
    candidate_vocab: Sequence[int],
    *,
    max_repetitions: int | None = None,
    center_logits: bool = False,
    aggregation: str = DEFAULT_EVIDENCE_AGGREGATION,
    requested_schedule: Mapping[str, tuple[Sequence, Sequence]] | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return operational hit@R counts, preserving every requested group.

    Every candidate must occur exactly once in each declared repetition through
    R, in both the original schedule and surviving evidence. Missing repetitions
    or candidates are misses. The original pre-QC schedule fixes the repetition
    origin and terminal budget, even when every row of a group is unavailable.
    Centering and aggregation only consume the currently scored prefix.
    """
    predictions, _ = _candidate_repetition_predictions(
        logits, codes, group_ids, repetitions, truth_by_group, candidate_vocab,
        requested_schedule=requested_schedule, max_repetitions=max_repetitions,
        center_logits=center_logits, aggregation=aggregation,
    )
    correct = {str(level): sum(predicted is not None and predicted == truth_by_group[group]
                              for group, predicted in values.items())
               for level, values in predictions.items()}
    total = {str(level): len(values) for level, values in predictions.items()}
    return correct, total


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
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    n_rows: int,
    *,
    group_ids: np.ndarray | None = None,
    domain_ids: np.ndarray | None = None,
    fold_protocol: str | None = None,
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

    grouped_protocols = {"loso", "partial_loso", "inner_sensitivity_screen"}
    if fold_protocol in grouped_protocols:
        if group_ids is None:
            raise ValueError(f"{fold_protocol} validation requires group_ids.")
        groups = np.asarray(group_ids).astype(str)
        if groups.shape != (n_rows,) or np.any(np.char.strip(groups) == ""):
            raise ValueError("fold group_ids must contain one non-empty group per row.")
        test_counts = np.zeros(n_rows, dtype=np.int64)
        held_out_groups: list[str] = []
        for index, (train, test) in enumerate(output):
            train_groups = set(np.unique(groups[train]).tolist())
            test_groups = np.unique(groups[test]).tolist()
            if len(test_groups) != 1:
                raise ValueError(
                    f"Fold {index} in {fold_protocol} must hold out exactly one complete group."
                )
            held_out = str(test_groups[0])
            if held_out in train_groups:
                raise ValueError(
                    f"Fold {index} in {fold_protocol} leaks held-out group {held_out!r} into train."
                )
            held_out_groups.append(held_out)
            test_counts += test.astype(np.int64)
        if len(set(held_out_groups)) != len(held_out_groups) or np.any(test_counts > 1):
            raise ValueError(f"{fold_protocol} test folds must not repeat held-out groups or rows.")
        if fold_protocol == "loso" and np.any(test_counts != 1):
            raise ValueError("loso test folds must cover every row exactly once.")
    if fold_protocol in {"domain_lodo", "domain_ceiling"}:
        if domain_ids is None:
            raise ValueError(f"{fold_protocol} validation requires domain_ids.")
        domains = np.asarray(domain_ids).astype(str)
        if domains.shape != (n_rows,) or np.any(np.char.strip(domains) == ""):
            raise ValueError("fold domain_ids must contain one non-empty domain per row.")
        selected_test_domains: list[str] = []
        test_counts = np.zeros(n_rows, dtype=np.int64)
        for index, (train, test) in enumerate(output):
            test_counts += test.astype(np.int64)
            train_domains = set(np.unique(domains[train]).tolist())
            test_domains = set(np.unique(domains[test]).tolist())
            if not train_domains or not test_domains:
                raise ValueError(f"Fold {index} must contain train and test domains.")
            if fold_protocol == "domain_lodo":
                if len(test_domains) != 1 or train_domains & test_domains:
                    raise ValueError(
                        f"Fold {index} in domain_lodo must hold out one complete domain."
                    )
                selected_test_domains.append(next(iter(test_domains)))
            else:
                if len(test_domains) != 1 or train_domains != test_domains:
                    raise ValueError(
                        f"Fold {index} in domain_ceiling must train and test within one domain."
                    )
                selected_test_domains.append(next(iter(test_domains)))
        if fold_protocol == "domain_lodo" and len(set(selected_test_domains)) != len(
            selected_test_domains
        ):
            raise ValueError("domain_lodo test folds must not repeat held-out domains.")
        if np.any(test_counts != 1):
            raise ValueError(
                f"{fold_protocol} test folds must cover every selected row exactly once."
            )
    return output


def _predict(model: object, X: np.ndarray, trial_channel_mask: np.ndarray | None) -> np.ndarray:
    kwargs = {}
    if (
        getattr(model, "predict_accepts_trial_channel_mask", False)
        and trial_channel_mask is not None
    ):
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
        if (
            getattr(model, "fit_accepts_trial_channel_mask", False)
            and trial_channel_mask is not None
        ):
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
    test_effective = test
    if fitted_artifact_model is not None:
        exclude_unusable = bool(
            getattr(fitted_artifact_model.policy, "exclude_unusable_test_epochs", False)
        )
        if exclude_unusable:
            X, trial_channel_mask, train, test_effective, artifact_quality = (
                apply_fitted_artifact_model_with_test_exclusion(
                    fitted_artifact_model,
                    X,
                    subject_ids,
                    train,
                    test,
                    trial_channel_mask,
                    qc_features,
                    materialize_masked_data=not _can_defer_artifact_zero_fill(model),
                )
            )
        else:
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
    calibration_logits, calibration_y, source = calibration_data_from_model(model)
    pos_weight = getattr(model, "training_pos_weight_", None)
    train_prior = getattr(model, "training_prior_", None)
    calibration = fit_logit_calibration(
        calibration_logits,
        calibration_y,
        source=source,
        pos_weight=None if pos_weight is None else float(pos_weight),
        train_prior=None if train_prior is None else float(train_prior),
    )
    logits = _predict(
        model,
        X[test_effective],
        None if trial_channel_mask is None else trial_channel_mask[test_effective],
    )
    y_test = y[test_effective]
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
        n_test_trials=int(test_effective.sum()),
        threshold=calibration.threshold,
        threshold_source=calibration.source,
        fit_sec=fit_sec,
        calibration_mode=calibration.mode,
        calibration_slope=calibration.llr_slope,
        calibration_temperature=calibration.temperature,
        calibration_offset=calibration.offset,
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
        materialize_shuffled_epoch=runtime.get("materialize_shuffled_epoch")
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
        fit_peak_reserved_mb=(memory.get("peak_reserved_mb") if isinstance(memory, dict) else None),
        oom_retries=int(runtime.get("oom_retries", 0)) if isinstance(runtime, dict) else 0,
        shared_worker_count=(
            int(runtime.get("shared_worker_count", 1)) if isinstance(runtime, dict) else 1
        ),
    )
    return result, calibration.to_llr(logits), test_effective, calibration


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
) -> tuple[int, BinaryFoldResult, np.ndarray, np.ndarray, object]:
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
    result, llr, test_effective, calibration = _fold_result(
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
    return fold_index, result, llr, test_effective, calibration


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
) -> tuple[int, BinaryFoldResult, np.ndarray, np.ndarray, object]:
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
    on_fold_result: Callable[[tuple[int, BinaryFoldResult, np.ndarray, np.ndarray, object]], None]
    | None = None,
) -> tuple[
    list[tuple[int, BinaryFoldResult, np.ndarray, np.ndarray, object]], str, int, str, int, int, int
]:
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
    domain_ids: np.ndarray | None = None,
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

    X, y, subject_ids = np.asarray(X), np.asarray(y), np.asarray(subject_ids).astype(str)
    _validate_binary_inputs(X, y, subject_ids)
    fit_groups = subject_ids if fit_group_ids is None else np.asarray(fit_group_ids).astype(str)
    if fit_groups.shape != (len(X),) or np.any(np.char.strip(fit_groups) == ""):
        raise ValueError("fit_group_ids must contain one non-empty group per epoch.")
    if domain_ids is not None:
        domain_ids = np.asarray(domain_ids).astype(str)
        if domain_ids.shape != (len(X),) or np.any(np.char.strip(domain_ids) == ""):
            raise ValueError("domain_ids must contain one non-empty domain per epoch.")
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
        _validate_folds(
            folds,
            len(X),
            group_ids=subject_ids,
            domain_ids=domain_ids,
            fold_protocol=fold_protocol,
        ),
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
    results = [result for _, result, _, _, _ in fold_results]
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
        metric_undefined_folds=_metric_undefined_folds(results, fold_id_offset),
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
    participant_ids: np.ndarray | None = None,
    domain_ids: np.ndarray | None = None,
    event_timeline: object | None = None,
    trial_channel_mask: np.ndarray | None = None,
    qc_features: EpochQCFeatures | None = None,
    artifact_policy: FoldLocalArtifactPolicy | None = None,
    fitted_artifact_models: Mapping[int, FoldLocalArtifactModel] | None = None,
    artifact_qc_jobs: int | None = None,
    fold_protocol: str | None = None,
    on_fold_end: Callable[[int, CandidateFoldResult], None] | None = None,
    repetition_indices: np.ndarray | None = None,
    center_decision_logits: bool = False,
    aggregation: str = DEFAULT_EVIDENCE_AGGREGATION,
    max_repetitions: int | None = None,
    n_jobs: int = 1,
    parallel_backend: str = "auto",
    fold_id_offset: int = 0,
    max_gpu_jobs: int | None = None,
    cpu_threads: int | None = None,
) -> CandidateSummary:
    """Evaluate held-out decisions, with macro only for explicit participant authority.

    participant_ids are canonical identity keys supplied by the caller. Neither
    fit_group_ids nor domain_ids defines the participant reduction unit.
    """

    del event_timeline  # Candidate-chain validation remains an ingress responsibility.
    X, y = np.asarray(X), np.asarray(y)
    group_ids = np.asarray(selection_group_ids).astype(str)
    fit_groups = group_ids if fit_group_ids is None else np.asarray(fit_group_ids).astype(str)
    codes = np.asarray(candidate_codes)
    _validate_binary_inputs(X, y, group_ids)
    if fit_groups.shape != (len(X),) or np.any(np.char.strip(fit_groups) == ""):
        raise ValueError("fit_group_ids must contain one non-empty group per epoch.")
    if participant_ids is not None:
        participant_ids = np.asarray(participant_ids).astype(str)
        if participant_ids.shape != (len(X),) or np.any(np.char.strip(participant_ids) == ""):
            raise ValueError("participant_ids must contain one non-empty authority key per epoch.")
    if domain_ids is not None:
        domain_ids = np.asarray(domain_ids).astype(str)
        if domain_ids.shape != (len(X),) or np.any(np.char.strip(domain_ids) == ""):
            raise ValueError("domain_ids must contain one non-empty domain per epoch.")
    if codes.shape != (len(X),):
        raise ValueError("candidate_codes must align with X.")
    if aggregation not in EVIDENCE_AGGREGATIONS:
        raise ValueError(f"candidate evaluation requires one of {sorted(EVIDENCE_AGGREGATIONS)}")
    if max_repetitions is not None and (type(max_repetitions) is not int or max_repetitions < 1):
        raise ValueError("max_repetitions must be a positive integer")
    if max_repetitions is not None and repetition_indices is None:
        raise ValueError("a repetition budget requires explicit repetition_indices")
    repetitions = None if repetition_indices is None else np.asarray(repetition_indices)
    if repetitions is not None and (
        repetitions.shape != (len(X),) or not np.issubdtype(repetitions.dtype, np.integer)
        or np.issubdtype(repetitions.dtype, np.bool_) or np.any(repetitions < 0)
    ):
        raise ValueError("repetition_indices must align with X rows as non-negative integers.")
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
    raw_records: list[tuple[object, object, str]] = []
    validated_folds = _validate_folds(
        folds,
        len(X),
        group_ids=fit_groups,
        domain_ids=domain_ids,
        fold_protocol=fold_protocol,
    )
    requested_groups = [tuple(np.unique(group_ids[test])) for _, test in validated_folds]
    if any(group not in truth_by_group or truth_by_group[group] not in candidate_vocab
           for groups in requested_groups for group in groups):
        raise ValueError("every requested group needs truth in the frozen vocabulary")
    participant_by_group: dict[str, str] = {}
    if participant_ids is not None:
        for _, test in validated_folds:
            for group, owner in zip(group_ids[test], participant_ids[test], strict=True):
                if group in participant_by_group and participant_by_group[group] != owner:
                    raise ValueError("each requested decision group must belong to exactly one participant")
                participant_by_group[group] = str(owner)
    requested_participants = tuple(dict.fromkeys(participant_by_group.values()))
    terminal_outcomes: dict[str, list[bool]] = {}
    level_outcomes: dict[str, dict[str, list[bool]]] = {}
    schedules = ([] if repetitions is None else [
        {group: (codes[test & (group_ids == group)], repetitions[test & (group_ids == group)])
         for group in groups}
        for (_, test), groups in zip(validated_folds, requested_groups, strict=True)
    ])
    curve_limit = (None if repetitions is None else max_repetitions or max(
        int(values.max() - values.min()) + 1 for schedule in schedules for _, values in schedule.values()
    ))
    hit_scope = {"kind": "complete_repetition_terminal" if repetitions is not None else "raw_all_complete_vocabulary",
                 "aggregation": aggregation, "denominator": "requested_pre_qc_decision_groups",
                 "terminal_policy": ("declared_max_repetitions" if max_repetitions is not None else
                                     "pre_qc_group_schedule" if repetitions is not None else "no_repetition_claim"),
                 "max_repetitions": max_repetitions, "curve_repetitions": curve_limit,
                 "complete_repeat_rule": "one_observation_per_candidate_per_declared_repeat" if repetitions is not None else None}
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
    for fold_index, binary, llr, test_effective, _ in fold_results:
        decision = decide_by_group(
            llr,
            codes[test_effective],
            group_ids[test_effective],
            candidate_vocab,
            center_logits=center_decision_logits,
            aggregation=aggregation,
        )
        raw_predictions = dict(zip(decision.group_ids, decision.predicted, strict=True))
        fold_raw_records = [(raw_predictions.get(group), truth_by_group[group], group)
                            for group in requested_groups[fold_index]]
        fold_records = fold_raw_records
        hit_correct: dict[str, int] | None = None
        hit_total: dict[str, int] | None = None
        terminals = None
        if repetitions is not None:
            predictions, terminals = _candidate_repetition_predictions(
                llr, codes[test_effective], group_ids[test_effective], repetitions[test_effective],
                truth_by_group, candidate_vocab, requested_schedule=schedules[fold_index],
                max_repetitions=max_repetitions, curve_repetitions=curve_limit,
                center_logits=center_decision_logits, aggregation=aggregation,
            )
            hit_correct = {str(level): sum(predicted is not None and predicted == truth_by_group[group]
                                          for group, predicted in values.items())
                           for level, values in predictions.items()}
            hit_total = {str(level): len(values) for level, values in predictions.items()}
            fold_records = [(predictions[terminals[group]][group], truth_by_group[group], group)
                            for group in requested_groups[fold_index]]
            if participant_ids is not None:
                for level, values in predictions.items():
                    outcomes = level_outcomes.setdefault(str(level), {})
                    for group, predicted in values.items():
                        outcomes.setdefault(participant_by_group[group], []).append(
                            predicted is not None and predicted == truth_by_group[group]
                        )
        if participant_ids is not None:
            for predicted, truth, group in fold_records:
                terminal_outcomes.setdefault(participant_by_group[group], []).append(predicted == truth)
        hit_rate = float(np.mean([predicted == truth for predicted, truth, _ in fold_records]))
        result = CandidateFoldResult(
            **binary.__dict__,
            hit_rate=hit_rate,
            n_decisions=len(fold_records),
            decision_records=fold_records,
            hit_correct_by_repetition=hit_correct,
            hit_total_by_repetition=hit_total,
            hit_scope={**hit_scope, "group_terminal_repetitions": terminals},
            raw_all_hit_rate=float(np.mean([predicted == truth for predicted, truth, _ in fold_raw_records])),
        )
        results.append(result)
        if on_fold_end is not None:
            on_fold_end(fold_index + fold_id_offset, result)
        records.extend(fold_records)
        raw_records.extend(fold_raw_records)
    bacc = np.asarray([fold.balanced_acc for fold in results], dtype=float)
    auc = np.asarray([fold.auc for fold in results], dtype=float)
    hit = np.asarray([fold.hit_rate for fold in results], dtype=float)
    overall = (
        float(np.mean([predicted == truth for predicted, truth, _ in records]))
        if records
        else float("nan")
    )
    macro_hits: dict[str, float] | None = None
    participant_macro_hits: dict[str, float] | None = None
    if repetition_indices is not None and results:
        sum_correct: dict[str, int] = {}
        sum_total: dict[str, int] = {}
        for fold in results:
            for key, value in (fold.hit_correct_by_repetition or {}).items():
                sum_correct[key] = sum_correct.get(key, 0) + int(value)
            for key, value in (fold.hit_total_by_repetition or {}).items():
                sum_total[key] = sum_total.get(key, 0) + int(value)
        macro_hits = {
            key: float(sum_correct[key]) / sum_total[key]
            for key in sorted(sum_total, key=lambda item: int(item))
            if sum_total[key] > 0
        }
        if participant_ids is not None:
            participant_macro_hits = {
                level: _participant_macro(outcomes, requested_participants)
                for level, outcomes in level_outcomes.items()
            }
    return CandidateSummary(
        balanced_acc_mean=float(np.nanmean(bacc)),
        balanced_acc_std=float(np.nanstd(bacc)),
        auc_mean=float(np.nanmean(auc)),
        per_fold=results,
        hit_rate_mean=float(np.nanmean(hit)),
        primary_hit_rate=overall if repetitions is not None else None,
        primary_participant_macro_hit_rate=(
            _participant_macro(terminal_outcomes, requested_participants)
            if repetitions is not None and participant_ids is not None else None
        ),
        primary_hit_scope=hit_scope,
        participant_macro_scope=(
            {"reduction": "participant_macro_equal_weight",
             "participant_count": len(requested_participants),
             "participant_identity": "explicit_authority_key",
             "denominator": "requested_pre_qc_decision_groups",
             "unavailable_policy": "incorrect"}
            if repetitions is not None and participant_ids is not None else None
        ),
        raw_all_hit_rate=float(np.mean([predicted == truth for predicted, truth, _ in raw_records])),
        subject_records=records,
        hit_at_repetition_macro=macro_hits,
        hit_at_repetition_participant_macro=participant_macro_hits,
        execution_backend=backend,
        effective_n_jobs=effective_n_jobs,
        input_transport=input_transport,
        cpu_threads_per_worker=cpu_threads_per_worker,
        artifact_qc_workers=artifact_qc_workers,
        artifact_qc_cpu_threads_per_worker=artifact_qc_cpu_threads_per_worker,
        metric_undefined_folds=_metric_undefined_folds(results, fold_id_offset),
    )
