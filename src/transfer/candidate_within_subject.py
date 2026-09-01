"""Chronological calibration/test splits for candidate-membership datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.candidate_task import (
    CandidateMembershipMetadata,
    CandidateTaskContract,
    candidate_task_contract_from_provenance,
    validate_candidate_membership_metadata,
)
from data.contract import assert_causal_p300_input_contract
from data.epochs import EpochDataset
from transfer.outcomes import DecisionKey


@dataclass(frozen=True)
class CandidatePrefixSuffixSplit:
    prefix_mask: np.ndarray
    suffix_mask: np.ndarray
    suffix_repetition_indices: np.ndarray
    usable_decisions: tuple[DecisionKey, ...]


@dataclass(frozen=True)
class CandidateCalibrationDecisionSplit:
    """Known early decisions for calibration, later unknown decisions for test."""

    calibration_mask: np.ndarray
    test_mask: np.ndarray
    calibration_decision_indices: np.ndarray
    test_repetition_indices: np.ndarray
    usable_subjects: tuple[str, ...]
    calibration_selections_by_subject: dict[str, tuple[str, ...]]
    requested_test_selections_by_subject: dict[str, tuple[str, ...]]
    test_selections_by_subject: dict[str, tuple[str, ...]]
    failed_test_selections_by_subject: dict[str, dict[str, str]]
    excluded_subjects: dict[str, str]


def _validated_metadata(
    dataset: EpochDataset,
) -> tuple[CandidateTaskContract, CandidateMembershipMetadata]:
    dataset.validate(require_labels=True)
    assert_causal_p300_input_contract(dataset.preprocessing)
    if dataset.metadata is None or dataset.y is None:
        raise ValueError("candidate dataset requires metadata and labels.")
    contract = candidate_task_contract_from_provenance(dataset.provenance)
    metadata = validate_candidate_membership_metadata(
        dataset.metadata, contract, labels=np.asarray(dataset.y)
    )
    return contract, metadata


def _complete_repetition_block(
    rows: np.ndarray,
    metadata: CandidateMembershipMetadata,
    contract: CandidateTaskContract,
    *,
    count: int,
) -> bool:
    if len(rows) != len(contract.candidate_ids) * count:
        return False
    expected = set(contract.candidate_ids)
    for repetition in range(count):
        block = rows[metadata.repetition_indices[rows] == repetition]
        if len(block) != len(expected):
            return False
        values = metadata.candidate_ids[block].tolist()
        if len(values) != len(set(values)) or set(values) != expected:
            return False
    return True


def _epoch_times(dataset: EpochDataset) -> tuple[np.ndarray, np.ndarray]:
    evidence = np.asarray(dataset.event_timeline.evidence_indices, dtype=np.int64)
    available_events = np.flatnonzero(evidence >= 0)
    epoch_indices = evidence[available_events]
    if len(epoch_indices) != dataset.n_epochs or not np.array_equal(
        np.sort(epoch_indices), np.arange(dataset.n_epochs)
    ):
        raise ValueError(
            "candidate split requires one causal timeline event for every retained epoch."
        )
    epoch_available_at = np.empty(dataset.n_epochs, dtype=float)
    epoch_onsets = np.empty(dataset.n_epochs, dtype=float)
    epoch_available_at[epoch_indices] = np.asarray(
        dataset.event_timeline.evidence_available_times_s, dtype=float
    )[available_events]
    epoch_onsets[epoch_indices] = np.asarray(dataset.event_timeline.onset_times_s, dtype=float)[
        available_events
    ]
    return epoch_onsets, epoch_available_at


def candidate_calibration_decision_split(
    dataset: EpochDataset,
    *,
    calibration_selections: int,
    test_repetitions: int,
    max_test_selections: int | None = None,
) -> CandidateCalibrationDecisionSplit:
    """Split known early selections from later, truth-hidden test selections."""

    contract, metadata = _validated_metadata(dataset)
    if calibration_selections < 1 or test_repetitions < 1:
        raise ValueError("calibration_selections and test_repetitions must be positive.")
    if max_test_selections is not None and max_test_selections < 1:
        raise ValueError("max_test_selections must be positive or None.")
    subject_ids = np.asarray(dataset.subject_ids).astype(str)
    epoch_onsets, epoch_available_at = _epoch_times(dataset)

    calibration_mask = np.zeros(dataset.n_epochs, dtype=bool)
    test_mask = np.zeros(dataset.n_epochs, dtype=bool)
    calibration_order = np.full(dataset.n_epochs, -1, dtype=np.int64)
    test_rep = np.full(dataset.n_epochs, -1, dtype=np.int64)
    usable_subjects: list[str] = []
    calibration_by_subject: dict[str, tuple[str, ...]] = {}
    requested_test_by_subject: dict[str, tuple[str, ...]] = {}
    test_by_subject: dict[str, tuple[str, ...]] = {}
    failed_test_by_subject: dict[str, dict[str, str]] = {}
    excluded_subjects: dict[str, str] = {}
    epoch_start_offset_s = float(dataset.preprocessing.tmin_ms) / 1000.0

    for raw_subject in np.unique(subject_ids):
        subject = str(raw_subject)
        subject_rows = np.flatnonzero(subject_ids == subject)
        selections = np.unique(metadata.selection_ids[subject_rows])
        ordered = tuple(
            sorted(
                selections.tolist(),
                key=lambda selection: float(
                    np.min(
                        epoch_onsets[
                            subject_rows[metadata.selection_ids[subject_rows] == selection]
                        ]
                    )
                ),
            )
        )
        if len(ordered) <= calibration_selections:
            excluded_subjects[subject] = "insufficient_decisions"
            continue
        calibration = ordered[:calibration_selections]
        later = ordered[calibration_selections:]
        requested_test = tuple(
            str(selection)
            for selection in (later if max_test_selections is None else later[:max_test_selections])
        )
        requested_test_by_subject[subject] = requested_test
        calibration_rows: list[np.ndarray] = []
        invalid_calibration = False
        for order, selection in enumerate(calibration):
            rows = subject_rows[metadata.selection_ids[subject_rows] == selection]
            unique_repetitions = np.unique(metadata.repetition_indices[rows])
            if not np.array_equal(
                unique_repetitions, np.arange(len(unique_repetitions))
            ) or not _complete_repetition_block(
                rows,
                metadata,
                contract,
                count=len(unique_repetitions),
            ):
                invalid_calibration = True
                break
            if (
                len(np.unique(metadata.target_rows[rows])) != 1
                or len(np.unique(metadata.target_columns[rows])) != 1
            ):
                invalid_calibration = True
                break
            calibration_rows.append(rows)
            calibration_order[rows] = order
        if invalid_calibration:
            excluded_subjects[subject] = "invalid_calibration_decision"
            calibration_order[subject_rows] = -1
            failed_test = {selection: "calibration_failed" for selection in requested_test}
            failed_test_by_subject[subject] = failed_test
            continue
        calibration_rows_array = np.concatenate(calibration_rows)
        boundary_s = float(np.max(epoch_available_at[calibration_rows_array]))

        failed_test: dict[str, str] = {}
        eligible_test: list[str] = []
        for selection in requested_test:
            rows = subject_rows[metadata.selection_ids[subject_rows] == selection]
            if np.min(epoch_onsets[rows] + epoch_start_offset_s) <= boundary_s:
                reason = "selection_overlaps_calibration_evidence"
                failed_test[str(selection)] = reason
                continue
            selected = rows[metadata.repetition_indices[rows] < test_repetitions]
            if not _complete_repetition_block(
                selected,
                metadata,
                contract,
                count=test_repetitions,
            ):
                reason = "insufficient_complete_test_repetitions"
                failed_test[str(selection)] = reason
                continue
            if (
                len(np.unique(metadata.target_rows[selected])) != 1
                or len(np.unique(metadata.target_columns[selected])) != 1
            ):
                reason = "mixed_test_target"
                failed_test[str(selection)] = reason
                continue
            eligible_test.append(str(selection))
        failed_test_by_subject[subject] = failed_test
        if not eligible_test:
            excluded_subjects[subject] = "no_eligible_unknown_test_decision"
            calibration_order[subject_rows] = -1
            continue

        calibration_mask[calibration_rows_array] = True
        for selection in eligible_test:
            rows = subject_rows[
                (metadata.selection_ids[subject_rows] == selection)
                & (metadata.repetition_indices[subject_rows] < test_repetitions)
            ]
            test_mask[rows] = True
            test_rep[rows] = metadata.repetition_indices[rows]
        usable_subjects.append(subject)
        calibration_by_subject[subject] = tuple(str(value) for value in calibration)
        test_by_subject[subject] = tuple(eligible_test)

    return CandidateCalibrationDecisionSplit(
        calibration_mask=calibration_mask,
        test_mask=test_mask,
        calibration_decision_indices=calibration_order,
        test_repetition_indices=test_rep,
        usable_subjects=tuple(usable_subjects),
        calibration_selections_by_subject=calibration_by_subject,
        requested_test_selections_by_subject=requested_test_by_subject,
        test_selections_by_subject=test_by_subject,
        failed_test_selections_by_subject=failed_test_by_subject,
        excluded_subjects=excluded_subjects,
    )


def candidate_prefix_suffix_split(
    dataset: EpochDataset,
    *,
    prefix_repetitions: int,
    test_repetitions: int,
) -> CandidatePrefixSuffixSplit:
    """Split each subject/selection by chronological repetition index."""

    contract, metadata = _validated_metadata(dataset)
    if prefix_repetitions < 1 or test_repetitions < 1:
        raise ValueError("prefix/test repetitions must be positive.")
    subject_ids = np.asarray(dataset.subject_ids).astype(str)
    prefix = np.zeros(dataset.n_epochs, dtype=bool)
    suffix = np.zeros(dataset.n_epochs, dtype=bool)
    suffix_repetitions = np.full(dataset.n_epochs, -1, dtype=np.int64)
    usable: list[DecisionKey] = []
    decision_keys = tuple(
        dict.fromkeys(
            DecisionKey(subject, selection)
            for subject, selection in zip(subject_ids, metadata.selection_ids, strict=True)
        )
    )
    for key in decision_keys:
        rows = np.flatnonzero(
            (subject_ids == key.subject_id) & (metadata.selection_ids == key.decision_id)
        )
        pre = rows[metadata.repetition_indices[rows] < prefix_repetitions]
        post = rows[
            (metadata.repetition_indices[rows] >= prefix_repetitions)
            & (metadata.repetition_indices[rows] < prefix_repetitions + test_repetitions)
        ]
        if not _complete_repetition_block(pre, metadata, contract, count=prefix_repetitions):
            continue
        shifted = metadata.repetition_indices.copy()
        shifted[post] -= prefix_repetitions
        shifted_metadata = CandidateMembershipMetadata(
            candidate_ids=metadata.candidate_ids,
            row_codes=metadata.row_codes,
            column_codes=metadata.column_codes,
            target_rows=metadata.target_rows,
            target_columns=metadata.target_columns,
            selection_ids=metadata.selection_ids,
            repetition_indices=shifted,
            is_target=metadata.is_target,
        )
        if not _complete_repetition_block(post, shifted_metadata, contract, count=test_repetitions):
            continue
        prefix[pre] = True
        suffix[post] = True
        suffix_repetitions[post] = shifted[post]
        usable.append(key)
    if not usable:
        raise ValueError(
            "No candidate decision has enough complete repetitions for "
            f"{prefix_repetitions}+{test_repetitions} chronological folds."
        )
    return CandidatePrefixSuffixSplit(
        prefix_mask=prefix,
        suffix_mask=suffix,
        suffix_repetition_indices=suffix_repetitions,
        usable_decisions=tuple(usable),
    )
