"""Chronological within-subject prefix/suffix splitting and evaluation.

This module refuses zero-phase caches because continuous forward-backward
filtering smears future suffix samples into training prefix epochs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from data.contract import (  # noqa: E402
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    assert_p300_input_contract,
)
from data.epochs import EpochDataset  # noqa: E402


@dataclass(frozen=True)
class PrefixSuffixSplit:
    prefix_mask: np.ndarray
    suffix_mask: np.ndarray
    group_ids: np.ndarray
    candidate_codes: np.ndarray
    target_codes: np.ndarray
    truth_by_group: dict[str, int]
    repetition_indices: np.ndarray
    onset_times_s: np.ndarray
    evidence_available_times_s: np.ndarray
    suffix_repetition_indices: np.ndarray
    usable_groups: tuple[str, ...]
    candidate_vocab: tuple[int, ...]
    excluded_groups: dict[str, str]
    selected_scheduled_repetitions: dict[
        str, dict[str, dict[str, tuple[int, ...]]]
    ] = field(default_factory=dict)
    evidence_cost_by_group: dict[str, dict[str, object]] = field(default_factory=dict)
    evidence_cost_by_repetition: dict[
        str, dict[str, dict[str, float | int]]
    ] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationDecisionSplit:
    """Known complete decisions followed by later unknown target decisions."""

    calibration_mask: np.ndarray
    test_mask: np.ndarray
    group_ids: np.ndarray
    candidate_codes: np.ndarray
    target_codes: np.ndarray
    repetition_indices: np.ndarray
    test_repetition_indices: np.ndarray
    truth_by_group: dict[str, int]
    candidate_vocab: tuple[int, ...]
    usable_subjects: tuple[str, ...]
    calibration_groups_by_subject: dict[str, tuple[str, ...]]
    requested_test_groups_by_subject: dict[str, tuple[str, ...]]
    test_groups_by_subject: dict[str, tuple[str, ...]]
    failed_test_groups_by_subject: dict[str, dict[str, str]]
    excluded_subjects: dict[str, str]
    excluded_groups: dict[str, str]


@dataclass(frozen=True)
class ChronologicalValidationSplit:
    """A temporal inner split with validation strictly after optimization rows.

    ``train_mask`` and ``validation_mask`` refer to the rows supplied to the
    adapter, while the repetition tuples preserve the declared schedule.  The
    helper deliberately does not sample groups: selecting the latest complete
    repetitions makes the direction of information flow explicit.
    """

    train_mask: np.ndarray
    validation_mask: np.ndarray
    train_repetitions: tuple[int, ...]
    validation_repetitions: tuple[int, ...]
    embargo_mask: np.ndarray | None = None


def calibration_decision_split(
    dataset: EpochDataset,
    *,
    calibration_selections: int,
    test_repetitions: int,
    max_test_selections: int | None = None,
    candidate_vocabulary: Sequence[int] | None = None,
) -> CalibrationDecisionSplit:
    """Split complete early decisions from later target-changing decisions."""

    dataset.validate(require_labels=True)
    assert_p300_input_contract(
        dataset.preprocessing, SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
    )
    if calibration_selections < 1 or test_repetitions < 1:
        raise ValueError("calibration_selections and test_repetitions must be positive.")
    if max_test_selections is not None and max_test_selections < 1:
        raise ValueError("max_test_selections must be positive or None.")

    timeline = dataset.event_timeline
    evidence = np.asarray(timeline.evidence_indices, dtype=np.int64)
    available_events = np.flatnonzero(evidence >= 0)
    if len(np.unique(evidence[available_events])) != len(available_events):
        raise ValueError("candidate event evidence mapping must be one-to-one.")
    scheduled_candidates = np.asarray(timeline.candidate_ids).astype(str)
    scheduled_targets = np.asarray(timeline.target_candidate_ids).astype(str)
    scheduled_repetitions = np.asarray(timeline.repetition_indices, dtype=np.int64)
    scheduled_groups = np.asarray(timeline.group_ids).astype(str)
    scheduled_sessions = np.asarray(timeline.session_ids).astype(str)
    scheduled_onsets = np.asarray(timeline.onset_times_s, dtype=float)
    scheduled_available = np.asarray(timeline.evidence_available_times_s, dtype=float)
    try:
        candidate_values = scheduled_candidates[available_events].astype(np.int64)
        _ = scheduled_targets[available_events].astype(np.int64)
    except ValueError as exc:
        raise ValueError("candidate and target ids must encode integer choices.") from exc
    vocabulary = (
        tuple(sorted(np.unique(candidate_values).tolist()))
        if candidate_vocabulary is None
        else tuple(int(value) for value in candidate_vocabulary)
    )
    if len(vocabulary) < 2 or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("candidate_vocabulary must contain at least two unique choices.")

    n_epochs = dataset.n_epochs
    group_by_epoch = np.empty(n_epochs, dtype=object)
    candidate_by_epoch = np.full(n_epochs, -1, dtype=np.int64)
    target_by_epoch = np.full(n_epochs, -1, dtype=np.int64)
    repetition_by_epoch = np.full(n_epochs, -1, dtype=np.int64)
    onset_by_epoch = np.empty(n_epochs, dtype=float)
    available_by_epoch = np.empty(n_epochs, dtype=float)
    session_by_epoch = np.empty(n_epochs, dtype=object)
    for event in available_events:
        epoch = int(evidence[event])
        group_by_epoch[epoch] = scheduled_groups[event]
        candidate_by_epoch[epoch] = int(scheduled_candidates[event])
        target_by_epoch[epoch] = int(scheduled_targets[event])
        repetition_by_epoch[epoch] = int(scheduled_repetitions[event])
        onset_by_epoch[epoch] = float(scheduled_onsets[event])
        available_by_epoch[epoch] = float(scheduled_available[event])
        session_by_epoch[epoch] = scheduled_sessions[event]

    session_start = np.zeros(n_epochs, dtype=float)
    if "session_start_timestamp_s" in dataset.metadata:
        raw_starts = dataset.metadata["session_start_timestamp_s"].to_numpy()
        session_start = np.asarray(
            [np.nan if value is None else float(value) for value in raw_starts], dtype=float
        )
    unique_sessions = np.unique(session_by_epoch.astype(str))
    if len(unique_sessions) > 1 and not np.isfinite(session_start).all():
        raise ValueError(
            "multi-session decision splitting requires session_start_timestamp_s metadata."
        )
    session_start = np.where(np.isfinite(session_start), session_start, 0.0)
    absolute_onset = session_start + onset_by_epoch
    absolute_available = session_start + available_by_epoch

    calibration_mask = np.zeros(n_epochs, dtype=bool)
    test_mask = np.zeros(n_epochs, dtype=bool)
    test_repetitions_out = np.full(n_epochs, -1, dtype=np.int64)
    truth_by_group: dict[str, int] = {}
    usable_subjects: list[str] = []
    calibration_by_subject: dict[str, tuple[str, ...]] = {}
    requested_by_subject: dict[str, tuple[str, ...]] = {}
    test_by_subject: dict[str, tuple[str, ...]] = {}
    failed_by_subject: dict[str, dict[str, str]] = {}
    excluded_subjects: dict[str, str] = {}
    excluded_groups: dict[str, str] = {}
    epoch_start_offset_s = float(dataset.preprocessing.tmin_ms) / 1000.0

    def group_rows(group: str) -> np.ndarray:
        return np.flatnonzero(group_by_epoch == group)

    def complete(rows: np.ndarray, repetitions: int | None) -> bool:
        if len(np.unique(target_by_epoch[rows])) != 1:
            return False
        if set(candidate_by_epoch[rows].tolist()) != set(vocabulary):
            return False
        for candidate in vocabulary:
            candidate_rows = rows[candidate_by_epoch[rows] == candidate]
            candidate_repetitions = np.sort(repetition_by_epoch[candidate_rows])
            if repetitions is None:
                if len(candidate_repetitions) < 1 or not np.array_equal(
                    candidate_repetitions,
                    np.arange(len(candidate_repetitions), dtype=np.int64),
                ):
                    return False
            elif not np.array_equal(
                candidate_repetitions[candidate_repetitions < repetitions],
                np.arange(repetitions, dtype=np.int64),
            ):
                return False
        return True

    epoch_subjects = np.asarray(dataset.subject_ids).astype(str)
    for subject in np.unique(epoch_subjects):
        subject = str(subject)
        subject_rows = np.flatnonzero(epoch_subjects == subject)
        groups = np.unique(group_by_epoch[subject_rows].astype(str)).tolist()
        ordered = tuple(
            sorted(groups, key=lambda group: float(np.min(absolute_onset[group_rows(group)])))
        )
        if len(ordered) <= calibration_selections:
            excluded_subjects[subject] = "insufficient_decisions"
            continue
        calibration = ordered[:calibration_selections]
        later = ordered[calibration_selections:]
        requested = tuple(
            later if max_test_selections is None else later[:max_test_selections]
        )
        requested_by_subject[subject] = requested
        calibration_rows = [group_rows(group) for group in calibration]
        invalid_calibration = [
            group for group, rows in zip(calibration, calibration_rows, strict=True)
            if not complete(rows, None)
        ]
        if invalid_calibration:
            for group in invalid_calibration:
                excluded_groups[group] = "incomplete_calibration_decision"
            excluded_subjects[subject] = "invalid_calibration_decision"
            failed = {group: "calibration_failed" for group in requested}
            failed_by_subject[subject] = failed
            excluded_groups.update(failed)
            continue
        calibration_rows_array = np.concatenate(calibration_rows)
        boundary = float(np.max(absolute_available[calibration_rows_array]))
        eligible: list[str] = []
        failed: dict[str, str] = {}
        for group in requested:
            rows = group_rows(group)
            if float(np.min(absolute_onset[rows] + epoch_start_offset_s)) <= boundary:
                reason = "selection_overlaps_calibration_evidence"
            elif not complete(rows, test_repetitions):
                reason = "insufficient_complete_test_repetitions"
            else:
                reason = ""
            if reason:
                failed[group] = reason
                excluded_groups[group] = reason
                continue
            selected = rows[repetition_by_epoch[rows] < test_repetitions]
            test_mask[selected] = True
            test_repetitions_out[selected] = repetition_by_epoch[selected]
            truth_by_group[group] = int(np.unique(target_by_epoch[selected])[0])
            eligible.append(group)
        failed_by_subject[subject] = failed
        if not eligible:
            excluded_subjects[subject] = "no_eligible_unknown_test_decision"
            continue
        calibration_mask[calibration_rows_array] = True
        usable_subjects.append(subject)
        calibration_by_subject[subject] = calibration
        test_by_subject[subject] = tuple(eligible)

    if not usable_subjects:
        raise ValueError("No subject has valid calibration and later unknown decisions.")
    return CalibrationDecisionSplit(
        calibration_mask=calibration_mask,
        test_mask=test_mask,
        group_ids=group_by_epoch.astype(str),
        candidate_codes=candidate_by_epoch,
        target_codes=target_by_epoch,
        repetition_indices=repetition_by_epoch,
        test_repetition_indices=test_repetitions_out,
        truth_by_group=truth_by_group,
        candidate_vocab=vocabulary,
        usable_subjects=tuple(usable_subjects),
        calibration_groups_by_subject=calibration_by_subject,
        requested_test_groups_by_subject=requested_by_subject,
        test_groups_by_subject=test_by_subject,
        failed_test_groups_by_subject=failed_by_subject,
        excluded_subjects=excluded_subjects,
        excluded_groups=excluded_groups,
    )


def chronological_time_validation_split(
    onset_times_s: Sequence[float],
    evidence_available_times_s: Sequence[float],
    labels: Sequence[int],
    *,
    epoch_start_offset_s: float,
    min_positive_per_partition: int = 2,
) -> ChronologicalValidationSplit:
    """Choose a real-time train/validation boundary with an explicit embargo."""

    onset = np.asarray(onset_times_s, dtype=float)
    available_at = np.asarray(evidence_available_times_s, dtype=float)
    y = np.asarray(labels)
    if onset.ndim != 1 or available_at.shape != onset.shape or y.shape != onset.shape:
        raise ValueError("onset/evidence/labels must be aligned one-dimensional arrays.")
    if not np.isfinite(onset).all() or not np.isfinite(available_at).all():
        raise ValueError("chronological timestamps must be finite.")
    if np.any(available_at < onset):
        raise ValueError("evidence cannot be available before onset.")
    if not np.issubdtype(y.dtype, np.integer) or set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("chronological time validation requires integer labels {0,1}.")
    if min_positive_per_partition < 1:
        raise ValueError("min_positive_per_partition must be positive.")

    epoch_starts = onset + float(epoch_start_offset_s)
    target_epoch_starts = np.sort(epoch_starts[y == 1])
    if len(target_epoch_starts) < 2 * min_positive_per_partition:
        raise ValueError("not enough target trials for chronological train/validation.")
    # Start with the latest validation window containing the requested target
    # count, then move earlier only if class or embargo constraints require it.
    candidate_boundaries = target_epoch_starts[: min_positive_per_partition - 1 : -1]
    for boundary in candidate_boundaries:
        train = available_at < boundary
        validation = epoch_starts >= boundary
        embargo = ~(train | validation)
        if int(np.count_nonzero(y[train] == 1)) < min_positive_per_partition:
            continue
        if int(np.count_nonzero(y[validation] == 1)) < min_positive_per_partition:
            continue
        if set(np.unique(y[train]).tolist()) != {0, 1} or set(
            np.unique(y[validation]).tolist()
        ) != {0, 1}:
            continue
        if not float(np.max(available_at[train])) < float(np.min(epoch_starts[validation])):
            continue
        return ChronologicalValidationSplit(
            train_mask=train,
            validation_mask=validation,
            train_repetitions=(),
            validation_repetitions=(),
            embargo_mask=embargo,
        )
    raise ValueError(
        "no real-time chronological split retains both classes and the requested target counts."
    )


def chronological_validation_split(
    repetition_indices: Sequence[int],
    labels: Sequence[int],
    *,
    fraction: float | None = 0.1,
    min_repetitions: int = 2,
    max_repetitions: int = 6,
) -> ChronologicalValidationSplit:
    """Build an earlier-repetition train / later-repetition validation split.

    Repetition metadata is part of the protocol, so gaps, negative values, and
    duplicate schedule values fail closed instead of being silently sorted or
    compressed.  Both partitions must retain the two binary classes whenever
    validation is enabled; otherwise early stopping/calibration would be
    undefined and the caller must choose a larger prefix.

    ``fraction=None`` explicitly disables validation and returns all rows in the
    training mask.  This mirrors the fit-only option used by small unit tests
    while keeping the default transfer path chronological.
    """

    raw_repetitions = np.asarray(repetition_indices)
    if raw_repetitions.ndim != 1:
        raise ValueError("repetition_indices must be one-dimensional.")
    if not np.issubdtype(raw_repetitions.dtype, np.integer) or np.issubdtype(
        raw_repetitions.dtype, np.bool_
    ):
        raise ValueError("repetition_indices must use an integer dtype.")
    repetitions = raw_repetitions.astype(np.int64, copy=False)

    labels_array = np.asarray(labels)
    if labels_array.ndim != 1 or labels_array.shape != repetitions.shape:
        raise ValueError("labels and repetition_indices must be aligned one-dimensional arrays.")
    if not np.issubdtype(labels_array.dtype, np.integer) or np.issubdtype(
        labels_array.dtype, np.bool_
    ):
        raise ValueError("labels must use an integer dtype.")
    labels_array = labels_array.astype(np.int64, copy=False)
    if set(np.unique(labels_array).tolist()) != {0, 1}:
        raise ValueError("chronological validation requires both binary classes in the input.")
    if len(repetitions) == 0:
        raise ValueError("chronological validation requires at least one row.")

    unique_repetitions = np.unique(repetitions)
    if unique_repetitions[0] < 0 or not np.array_equal(
        unique_repetitions,
        np.arange(unique_repetitions[0], unique_repetitions[-1] + 1, dtype=np.int64),
    ):
        raise ValueError(
            "repetition_indices must be non-negative and contiguous; missing schedule "
            "repetitions cannot be repaired by re-numbering."
        )

    train_mask = np.ones(len(repetitions), dtype=bool)
    validation_mask = np.zeros(len(repetitions), dtype=bool)
    if fraction is None:
        return ChronologicalValidationSplit(
            train_mask=train_mask,
            validation_mask=validation_mask,
            train_repetitions=tuple(int(value) for value in unique_repetitions),
            validation_repetitions=(),
        )
    if not isinstance(fraction, (int, float, np.integer, np.floating)) or not 0.0 < float(
        fraction
    ) < 1.0:
        raise ValueError("fraction must be in (0,1) or None.")
    if (
        isinstance(min_repetitions, bool)
        or isinstance(max_repetitions, bool)
        or int(min_repetitions) < 1
        or int(max_repetitions) < int(min_repetitions)
    ):
        raise ValueError("invalid chronological validation repetition bounds.")

    n_repetitions = len(unique_repetitions)
    n_validation = int(round(float(fraction) * n_repetitions))
    n_validation = max(int(min_repetitions), min(int(max_repetitions), n_validation))
    if n_validation >= n_repetitions:
        raise ValueError(
            "chronological validation needs at least one earlier training repetition; "
            f"got {n_repetitions} total and requested {n_validation} validation repetitions."
        )
    validation_repetitions = unique_repetitions[-n_validation:]
    train_repetitions = unique_repetitions[:-n_validation]
    validation_mask = np.isin(repetitions, validation_repetitions)
    train_mask = ~validation_mask
    if set(np.unique(labels_array[train_mask]).tolist()) != {0, 1}:
        raise ValueError(
            "chronological training repetitions do not retain both binary classes; "
            "increase the calibration prefix."
        )
    if set(np.unique(labels_array[validation_mask]).tolist()) != {0, 1}:
        raise ValueError(
            "chronological validation repetitions do not retain both binary classes; "
            "increase the calibration prefix."
        )
    return ChronologicalValidationSplit(
        train_mask=train_mask,
        validation_mask=validation_mask,
        train_repetitions=tuple(int(value) for value in train_repetitions),
        validation_repetitions=tuple(int(value) for value in validation_repetitions),
    )


def causal_prefix_suffix_split(
    dataset: EpochDataset,
    *,
    prefix_repetitions: int,
    test_repetitions: int | None,
    min_candidate_repetitions: int = 2,
    contract: object | None = None,
) -> PrefixSuffixSplit:
    """Return chronological train/test masks for every complete selection group.

    The split is strict: every candidate must supply the requested prefix and
    suffix around one global raw-sample embargo, and the group must have exactly
    one target. Excluded groups and reasons are returned rather than silently
    repaired. ``contract`` selects the executable causal input contract; it
    defaults to the canonical 2 Hz / 800 ms one and cohort runners pass their
    own (e.g. the revised GTN 0.1 Hz / 1200 ms contract).
    """

    dataset.validate(require_labels=True)
    assert_p300_input_contract(
        dataset.preprocessing, contract or SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
    )
    if prefix_repetitions < 0 or (
        test_repetitions is not None and test_repetitions < 1
    ):
        raise ValueError(
            "prefix repetitions must be non-negative and test repetitions positive or None."
        )
    if min_candidate_repetitions < 1:
        raise ValueError("min_candidate_repetitions must be positive.")
    if not dataset.event_timeline.supports_full_candidate_chain:
        raise ValueError("dataset must expose a complete candidate/repetition chain.")
    timeline = dataset.event_timeline
    scheduled_groups = np.asarray(timeline.group_ids).astype(str)
    scheduled_candidates = np.asarray(timeline.candidate_ids).astype(str)
    scheduled_targets = np.asarray(timeline.target_candidate_ids).astype(str)
    scheduled_repetitions = np.asarray(timeline.repetition_indices, dtype=np.int64)
    scheduled_onsets = np.asarray(timeline.onset_times_s, dtype=np.float64)
    scheduled_available_at = np.asarray(
        timeline.evidence_available_times_s, dtype=np.float64
    )
    evidence_indices = np.asarray(timeline.evidence_indices, dtype=np.int64)
    if len(scheduled_groups) != timeline.n_events:
        raise ValueError("scheduled event fields are not aligned.")

    # Build the model-row view after retaining the original scheduled ordinal.
    # The operational policy below may wait for R *available* evidence rows,
    # but every selected row keeps its source repetition index and acquisition
    # cost so that waiting cannot be mistaken for a fixed-schedule estimand.
    scheduled_vocabulary = tuple(sorted(np.unique(scheduled_candidates).tolist()))
    code_by_candidate = {value: index for index, value in enumerate(scheduled_vocabulary)}
    candidate_vocab = tuple(range(len(scheduled_vocabulary)))
    available_events = np.flatnonzero(evidence_indices >= 0)
    available_events = available_events[
        np.argsort(evidence_indices[available_events], kind="stable")
    ]
    group_ids = scheduled_groups[available_events]
    candidates = np.asarray(
        [code_by_candidate[value] for value in scheduled_candidates[available_events]],
        dtype=np.int64,
    )
    targets = np.asarray(
        [code_by_candidate[value] for value in scheduled_targets[available_events]],
        dtype=np.int64,
    )
    repetitions = scheduled_repetitions[available_events]
    onset_times_s = scheduled_onsets[available_events]
    evidence_available_times_s = scheduled_available_at[available_events]
    if not (
        len(group_ids) == len(candidates) == len(targets) == len(repetitions) == dataset.n_epochs
    ):
        raise ValueError("scheduled-event evidence mapping is not aligned with epoch rows.")

    prefix = np.zeros(dataset.n_epochs, dtype=bool)
    suffix = np.zeros(dataset.n_epochs, dtype=bool)
    suffix_reps = np.full(dataset.n_epochs, -1, dtype=np.int64)
    usable: list[str] = []
    excluded: dict[str, str] = {}
    selected_schedule: dict[str, dict[str, dict[str, tuple[int, ...]]]] = {}
    evidence_cost: dict[str, dict[str, object]] = {}
    evidence_cost_by_repetition: dict[
        str, dict[str, dict[str, float | int]]
    ] = {}
    for group in np.unique(scheduled_groups):
        group = str(group)
        scheduled_rows = np.flatnonzero(scheduled_groups == group)
        group_targets = np.unique(scheduled_targets[scheduled_rows])
        if len(group_targets) != 1:
            excluded[group] = "mixed_target"
            continue
        group_candidates = np.unique(scheduled_candidates[scheduled_rows])
        if tuple(sorted(group_candidates.tolist())) != scheduled_vocabulary:
            excluded[group] = "incomplete_candidate_vocabulary"
            continue

        epoch_rows = np.flatnonzero(group_ids == group)
        prefix_by_code: dict[int, np.ndarray] = {}
        ordered_by_code: dict[int, np.ndarray] = {}
        insufficient = False
        for code in candidate_vocab:
            code_rows = epoch_rows[candidates[epoch_rows] == code]
            code_rows = code_rows[np.argsort(onset_times_s[code_rows], kind="stable")]
            minimum = max(prefix_repetitions, min_candidate_repetitions if prefix_repetitions else 0)
            if len(code_rows) < minimum:
                insufficient = True
                break
            ordered_by_code[int(code)] = code_rows
            prefix_by_code[int(code)] = code_rows[:prefix_repetitions]
        if insufficient:
            excluded[group] = "insufficient_available_prefix_evidence"
            continue

        if prefix_repetitions:
            pre_epoch = np.concatenate(list(prefix_by_code.values()))
            boundary_s = float(np.max(evidence_available_times_s[pre_epoch]))
        else:
            pre_epoch = np.empty(0, dtype=np.int64)
            boundary_s = float("-inf")
        epoch_start_offset_s = float(dataset.preprocessing.tmin_ms) / 1000.0
        if prefix_repetitions and not np.isfinite(boundary_s):
            excluded[group] = "invalid_prefix_evidence_time"
            continue
        suffix_by_code: dict[int, np.ndarray] = {}
        for code, code_rows in ordered_by_code.items():
            later = code_rows[onset_times_s[code_rows] + epoch_start_offset_s > boundary_s]
            if test_repetitions is None:
                if len(later) < 1:
                    insufficient = True
                    break
                suffix_by_code[code] = later
                continue
            if len(later) < test_repetitions:
                insufficient = True
                break
            suffix_by_code[code] = later[:test_repetitions]
        if insufficient:
            excluded[group] = "insufficient_available_suffix_after_time_embargo"
            continue
        post_epoch = np.concatenate(list(suffix_by_code.values()))

        prefix[pre_epoch] = True
        suffix[post_epoch] = True
        for code_rows in suffix_by_code.values():
            suffix_reps[code_rows] = np.arange(len(code_rows), dtype=np.int64)
        selected_schedule[group] = {
            "prefix": {
                str(code): tuple(int(value) for value in repetitions[rows])
                for code, rows in prefix_by_code.items()
            },
            "suffix": {
                str(code): tuple(int(value) for value in repetitions[rows])
                for code, rows in suffix_by_code.items()
            },
        }
        group_scheduled_onsets = scheduled_onsets[scheduled_rows]
        first_onset = float(np.min(group_scheduled_onsets))
        prefix_end = boundary_s if prefix_repetitions else first_onset
        suffix_end = float(np.max(evidence_available_times_s[post_epoch]))
        candidate_counts = {
            str(code): int(len(rows)) for code, rows in suffix_by_code.items()
        }
        balanced_repetitions = min(candidate_counts.values())
        evidence_cost[group] = {
            "prefix_available_trials": int(len(pre_epoch)),
            "suffix_available_trials": int(len(post_epoch)),
            "suffix_available_trials_by_candidate": candidate_counts,
            "balanced_all_repetitions": int(balanced_repetitions),
            "prefix_elapsed_seconds": float(max(0.0, prefix_end - first_onset)),
            "suffix_elapsed_seconds_after_prefix": float(max(0.0, suffix_end - prefix_end)),
            "scheduled_events_through_suffix": int(
                np.count_nonzero(group_scheduled_onsets <= np.max(onset_times_s[post_epoch]))
            ),
        }
        evidence_cost_by_repetition[group] = {}
        for repetition_count in range(1, balanced_repetitions + 1):
            balanced_rows = np.concatenate(
                [rows[:repetition_count] for rows in suffix_by_code.values()]
            )
            balanced_end = float(
                np.max(evidence_available_times_s[balanced_rows])
            )
            evidence_cost_by_repetition[group][str(repetition_count)] = {
                "available_trials": int(len(balanced_rows)),
                "scheduled_events_through_evidence": int(
                    np.count_nonzero(
                        group_scheduled_onsets
                        <= np.max(onset_times_s[balanced_rows])
                    )
                ),
                "elapsed_seconds_after_prefix": float(
                    max(0.0, balanced_end - prefix_end)
                ),
            }
        usable.append(group)

    if not usable:
        raise ValueError(
            "No selection group has enough complete candidate repetitions for "
            f"{prefix_repetitions}+{test_repetitions or 'all'} chronological "
            "within-subject folds."
        )
    truth_by_group = {
        group: int(code_by_candidate[np.unique(scheduled_targets[np.asarray(scheduled_groups) == group])[0]])
        for group in np.unique(scheduled_groups).astype(str)
    }
    return PrefixSuffixSplit(
        prefix_mask=prefix,
        suffix_mask=suffix,
        group_ids=group_ids,
        candidate_codes=candidates,
        target_codes=targets,
        truth_by_group=truth_by_group,
        repetition_indices=repetitions,
        onset_times_s=onset_times_s,
        evidence_available_times_s=evidence_available_times_s,
        suffix_repetition_indices=suffix_reps,
        usable_groups=tuple(usable),
        candidate_vocab=candidate_vocab,
        excluded_groups=excluded,
        selected_scheduled_repetitions=selected_schedule,
        evidence_cost_by_group=evidence_cost,
        evidence_cost_by_repetition=evidence_cost_by_repetition,
    )
