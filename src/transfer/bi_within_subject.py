"""Chronological prefix/suffix splitting for the BI2014a candidate cache.

The split is defined over ``metadata.selection_id`` and
``metadata.repetition_index`` because the BI event timeline deliberately does
not claim a complete one-target-per-group candidate chain: each flash is
partial row/column evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.contract import assert_causal_p300_input_contract
from data.epochs import EpochDataset


@dataclass(frozen=True)
class BI2014APrefixSuffixSplit:
    prefix_mask: np.ndarray
    suffix_mask: np.ndarray
    suffix_repetition_indices: np.ndarray
    usable_selections: tuple[str, ...]


@dataclass(frozen=True)
class BI2014ACalibrationDecisionSplit:
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
    excluded_selections: dict[str, str]


def _complete_bi_repetition_block(
    rows: np.ndarray,
    repetitions: np.ndarray,
    row_codes: np.ndarray,
    col_codes: np.ndarray,
    *,
    count: int,
) -> bool:
    if len(rows) != 12 * count:
        return False
    for repetition in range(count):
        block = rows[repetitions[rows] == repetition]
        if len(block) != 12:
            return False
        if set(row_codes[block][row_codes[block] >= 0].tolist()) != set(range(6)):
            return False
        if set(col_codes[block][col_codes[block] >= 0].tolist()) != set(range(6)):
            return False
    return True


def bi2014a_calibration_decision_split(
    dataset: EpochDataset,
    *,
    calibration_selections: int,
    test_repetitions: int,
    max_test_selections: int | None = None,
) -> BI2014ACalibrationDecisionSplit:
    """Split each subject from known early characters to later unknown characters.

    Unlike a repetition split inside one character, this protocol never trains
    on the answer of a tested selection.  A complete selection-level embargo
    also prevents the last calibration epoch window from containing samples of
    the first tested character.
    """

    dataset.validate(require_labels=True)
    assert_causal_p300_input_contract(dataset.preprocessing)
    if calibration_selections < 3 or test_repetitions < 1:
        raise ValueError("calibration_selections>=3 and test_repetitions>=1 are required.")
    if max_test_selections is not None and max_test_selections < 1:
        raise ValueError("max_test_selections must be positive or None.")
    required = {
        "flash_code",
        "row_code",
        "col_code",
        "target_row",
        "target_col",
        "selection_id",
        "repetition_index",
    }
    missing = required - set(dataset.metadata.columns)
    if missing:
        raise ValueError(f"BI candidate metadata is missing columns {sorted(missing)}.")

    subject_ids = np.asarray(dataset.subject_ids).astype(str)
    selection_ids = dataset.metadata["selection_id"].astype(str).to_numpy()
    repetitions = dataset.metadata["repetition_index"].to_numpy(dtype=np.int64)
    row_codes = dataset.metadata["row_code"].to_numpy(dtype=np.int64)
    col_codes = dataset.metadata["col_code"].to_numpy(dtype=np.int64)
    target_rows = dataset.metadata["target_row"].to_numpy(dtype=np.int64)
    target_cols = dataset.metadata["target_col"].to_numpy(dtype=np.int64)
    evidence = np.asarray(dataset.event_timeline.evidence_indices, dtype=np.int64)
    scheduled_available = np.asarray(
        dataset.event_timeline.evidence_available_times_s, dtype=float
    )
    scheduled_onsets = np.asarray(dataset.event_timeline.onset_times_s, dtype=float)
    available_events = np.flatnonzero(evidence >= 0)
    epoch_available_at = np.empty(dataset.n_epochs, dtype=float)
    epoch_onsets = np.empty(dataset.n_epochs, dtype=float)
    epoch_available_at[evidence[available_events]] = scheduled_available[available_events]
    epoch_onsets[evidence[available_events]] = scheduled_onsets[available_events]

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
    excluded_selections: dict[str, str] = {}
    epoch_start_offset_s = float(dataset.preprocessing.tmin_ms) / 1000.0

    for subject in np.unique(subject_ids):
        subject = str(subject)
        subject_rows = np.flatnonzero(subject_ids == subject)
        selections = np.unique(selection_ids[subject_rows])
        ordered = tuple(
            sorted(
                selections.tolist(),
                key=lambda selection: float(
                    np.min(epoch_onsets[subject_rows[selection_ids[subject_rows] == selection]])
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
            for selection in (
                later
                if max_test_selections is None
                else later[:max_test_selections]
            )
        )
        requested_test_by_subject[subject] = requested_test
        calibration_rows: list[np.ndarray] = []
        invalid_calibration = False
        for order, selection in enumerate(calibration):
            rows = subject_rows[selection_ids[subject_rows] == selection]
            unique_reps = np.unique(repetitions[rows])
            if not np.array_equal(unique_reps, np.arange(len(unique_reps))) or not _complete_bi_repetition_block(
                rows,
                repetitions,
                row_codes,
                col_codes,
                count=len(unique_reps),
            ):
                excluded_selections[str(selection)] = "incomplete_calibration_selection"
                invalid_calibration = True
                break
            if len(np.unique(target_rows[rows])) != 1 or len(np.unique(target_cols[rows])) != 1:
                excluded_selections[str(selection)] = "mixed_calibration_target"
                invalid_calibration = True
                break
            calibration_rows.append(rows)
            calibration_order[rows] = order
        if invalid_calibration:
            excluded_subjects[subject] = "invalid_calibration_decision"
            calibration_order[subject_rows] = -1
            failed_test = {
                selection: "calibration_failed" for selection in requested_test
            }
            failed_test_by_subject[subject] = failed_test
            excluded_selections.update(failed_test)
            continue
        calibration_rows_array = np.concatenate(calibration_rows)
        boundary_s = float(np.max(epoch_available_at[calibration_rows_array]))

        failed_test: dict[str, str] = {}
        eligible_test: list[str] = []
        for selection in requested_test:
            rows = subject_rows[selection_ids[subject_rows] == selection]
            if np.min(epoch_onsets[rows] + epoch_start_offset_s) <= boundary_s:
                reason = "selection_overlaps_calibration_evidence"
                excluded_selections[str(selection)] = reason
                failed_test[str(selection)] = reason
                continue
            selected = rows[repetitions[rows] < test_repetitions]
            if not _complete_bi_repetition_block(
                selected,
                repetitions,
                row_codes,
                col_codes,
                count=test_repetitions,
            ):
                reason = "insufficient_complete_test_repetitions"
                excluded_selections[str(selection)] = reason
                failed_test[str(selection)] = reason
                continue
            if len(np.unique(target_rows[selected])) != 1 or len(np.unique(target_cols[selected])) != 1:
                reason = "mixed_test_target"
                excluded_selections[str(selection)] = reason
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
                (selection_ids[subject_rows] == selection)
                & (repetitions[subject_rows] < test_repetitions)
            ]
            test_mask[rows] = True
            test_rep[rows] = repetitions[rows]
        usable_subjects.append(subject)
        calibration_by_subject[subject] = tuple(str(value) for value in calibration)
        test_by_subject[subject] = tuple(eligible_test)

    if not usable_subjects:
        raise ValueError(
            "No BI subject has enough known calibration decisions and later unknown test decisions."
        )
    return BI2014ACalibrationDecisionSplit(
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
        excluded_selections=excluded_selections,
    )


def bi2014a_prefix_suffix_split(
    dataset: EpochDataset,
    *,
    prefix_repetitions: int,
    test_repetitions: int,
) -> BI2014APrefixSuffixSplit:
    """Split BI selection groups by their chronological repetition index."""

    dataset.validate(require_labels=True)
    assert_causal_p300_input_contract(dataset.preprocessing)
    required = {
        "flash_code",
        "row_code",
        "col_code",
        "target_row",
        "target_col",
        "selection_id",
        "repetition_index",
    }
    missing = required - set(dataset.metadata.columns)
    if missing:
        raise ValueError(f"BI candidate metadata is missing columns {sorted(missing)}.")
    if prefix_repetitions < 1 or test_repetitions < 1:
        raise ValueError("prefix/test repetitions must be positive.")

    selection_ids = dataset.metadata["selection_id"].astype(str).to_numpy()
    repetitions = dataset.metadata["repetition_index"].to_numpy(dtype=np.int64)
    prefix = np.zeros(dataset.n_epochs, dtype=bool)
    suffix = np.zeros(dataset.n_epochs, dtype=bool)
    suffix_reps = np.full(dataset.n_epochs, -1, dtype=np.int64)
    usable: list[str] = []
    for selection in np.unique(selection_ids):
        rows = np.flatnonzero(selection_ids == selection)
        pre = rows[repetitions[rows] < prefix_repetitions]
        post = rows[
            (repetitions[rows] >= prefix_repetitions)
            & (repetitions[rows] < prefix_repetitions + test_repetitions)
        ]
        # BI repetitions contain exactly 12 flashes; require complete rows so
        # a partial repetition can never leak into either split.
        if len(pre) != 12 * prefix_repetitions or len(post) != 12 * test_repetitions:
            continue
        prefix[pre] = True
        suffix[post] = True
        suffix_reps[post] = repetitions[post] - prefix_repetitions
        usable.append(str(selection))
    if not usable:
        raise ValueError(
            "No BI selection has enough complete repetitions for "
            f"{prefix_repetitions}+{test_repetitions} chronological folds."
        )
    return BI2014APrefixSuffixSplit(
        prefix_mask=prefix,
        suffix_mask=suffix,
        suffix_repetition_indices=suffix_reps,
        usable_selections=tuple(usable),
    )
