"""Decision evaluation for the BI2014a 6x6 row/column speller.

Unlike the 9-choice digit task, a BI flash is partial evidence: it votes for
one of six rows or one of six columns. The character is the intersection of
the winning row and winning column.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from data.bi2014a_schedule import BI2014A_FLASH_SCHEDULE
from transfer.outcomes import (
    CandidateCoverage,
    DecisionKey,
    DecisionOutcome,
    DecisionStatus,
)


def bi2014a_expected_candidate_counts(repetitions: int) -> dict[str, int]:
    repetitions = int(repetitions)
    if repetitions < 1:
        raise ValueError("repetitions must be positive.")
    return {
        f"{axis}:{candidate}": repetitions
        for axis in ("row", "column")
        for candidate in range(BI2014A_FLASH_SCHEDULE.grid_size)
    }


def _aligned_optional_times(
    values: Sequence[float] | None, *, n_events: int, name: str
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) != n_events:
        raise ValueError(f"{name} must be one-dimensional and aligned with logits.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/inf.")
    return array


def decision_outcomes_at_repetition_6x6(
    logits: Sequence[float],
    flash_codes: Sequence[int],
    target_rows: Sequence[int],
    target_cols: Sequence[int],
    selection_ids: Sequence,
    repetition_indices: Sequence[int],
    *,
    subject_ids: Sequence | None = None,
    onset_times_s: Sequence[float] | None = None,
    evidence_available_times_s: Sequence[float] | None = None,
    max_repetitions: int | None = None,
) -> tuple[DecisionOutcome, ...]:
    """Return one primary outcome for every selection and repetition prefix."""

    logits = np.asarray(logits, dtype=float)
    flash_codes = np.asarray(flash_codes, dtype=np.int64)
    target_rows = np.asarray(target_rows, dtype=np.int64)
    target_cols = np.asarray(target_cols, dtype=np.int64)
    selection_ids = np.asarray(selection_ids).astype(str)
    repetition_indices = np.asarray(repetition_indices, dtype=np.int64)
    arrays = (
        logits,
        flash_codes,
        target_rows,
        target_cols,
        selection_ids,
        repetition_indices,
    )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("decision arrays must be one-dimensional.")
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("decision arrays must be aligned.")
    if not len(logits):
        raise ValueError("decision arrays are empty.")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain NaN/inf.")
    if np.any(repetition_indices < 0):
        raise ValueError("repetition_indices must be non-negative.")
    if subject_ids is None:
        subjects = np.repeat("unspecified", len(logits))
    else:
        subjects = np.asarray(subject_ids).astype(str)
        if subjects.ndim != 1 or len(subjects) != len(logits):
            raise ValueError("subject_ids must be one-dimensional and aligned with logits.")
    onset = _aligned_optional_times(
        onset_times_s, n_events=len(logits), name="onset_times_s"
    )
    available = _aligned_optional_times(
        evidence_available_times_s,
        n_events=len(logits),
        name="evidence_available_times_s",
    )
    if onset is not None and available is not None and np.any(available < onset):
        raise ValueError("evidence is available before its stimulus onset.")

    decoded = tuple(BI2014A_FLASH_SCHEDULE.decode(code) for code in flash_codes)
    candidate_keys = np.asarray([event.candidate_key for event in decoded])
    decision_keys = tuple(
        dict.fromkeys(
            DecisionKey(subject_id=subject, decision_id=selection)
            for subject, selection in zip(subjects, selection_ids, strict=True)
        )
    )
    if max_repetitions is None:
        max_repetitions = int(repetition_indices.max()) + 1
    max_repetitions = int(max_repetitions)
    if max_repetitions < 1:
        raise ValueError("max_repetitions must be positive.")

    outcomes: list[DecisionOutcome] = []
    for key in decision_keys:
        decision_rows = (subjects == key.subject_id) & (selection_ids == key.decision_id)
        for repetition in range(1, max_repetitions + 1):
            selected = decision_rows & (repetition_indices < repetition)
            observed_counts = Counter(candidate_keys[selected].tolist())
            coverage = CandidateCoverage.from_mappings(
                bi2014a_expected_candidate_counts(repetition), observed_counts
            )
            selected_onset = None if onset is None else onset[selected]
            selected_available = None if available is None else available[selected]
            timing = {
                "onset_start_s": (
                    float(selected_onset.min())
                    if selected_onset is not None and len(selected_onset)
                    else None
                ),
                "onset_end_s": (
                    float(selected_onset.max())
                    if selected_onset is not None and len(selected_onset)
                    else None
                ),
                "evidence_available_s": (
                    float(selected_available.max())
                    if selected_available is not None and len(selected_available)
                    else None
                ),
            }

            truth_rows = np.unique(target_rows[selected])
            truth_cols = np.unique(target_cols[selected])
            truth_valid = (
                len(truth_rows) == 1
                and len(truth_cols) == 1
                and 0 <= truth_rows[0] < BI2014A_FLASH_SCHEDULE.grid_size
                and 0 <= truth_cols[0] < BI2014A_FLASH_SCHEDULE.grid_size
            )
            target = (
                f"row:{int(truth_rows[0])}|column:{int(truth_cols[0])}"
                if truth_valid
                else None
            )
            if not coverage.complete:
                outcomes.append(
                    DecisionOutcome(
                        key=key,
                        evidence_level=repetition,
                        status=DecisionStatus.INCOMPLETE,
                        coverage=coverage,
                        target_candidate=target,
                        failure_reason="incomplete_candidate_coverage",
                        **timing,
                    )
                )
                continue
            if not truth_valid:
                outcomes.append(
                    DecisionOutcome(
                        key=key,
                        evidence_level=repetition,
                        status=DecisionStatus.INCOMPLETE,
                        coverage=coverage,
                        failure_reason="missing_or_inconsistent_target_candidate",
                        **timing,
                    )
                )
                continue

            target_semantics_valid = all(
                event.is_target
                == (
                    event.candidate_index
                    == (
                        int(truth_rows[0])
                        if event.axis == "row"
                        else int(truth_cols[0])
                    )
                )
                for event_index in np.flatnonzero(selected)
                for event in (decoded[int(event_index)],)
            )
            if not target_semantics_valid:
                outcomes.append(
                    DecisionOutcome(
                        key=key,
                        evidence_level=repetition,
                        status=DecisionStatus.INCOMPLETE,
                        coverage=coverage,
                        target_candidate=target,
                        failure_reason="flash_code_target_semantics_mismatch",
                        **timing,
                    )
                )
                continue

            row_score = np.zeros(BI2014A_FLASH_SCHEDULE.grid_size, dtype=float)
            col_score = np.zeros(BI2014A_FLASH_SCHEDULE.grid_size, dtype=float)
            for event_index in np.flatnonzero(selected):
                event = decoded[int(event_index)]
                scores = row_score if event.axis == "row" else col_score
                scores[event.candidate_index] += logits[event_index]
            row_winners = np.flatnonzero(
                np.isclose(row_score, float(row_score.max()), rtol=1e-12, atol=1e-12)
            )
            col_winners = np.flatnonzero(
                np.isclose(col_score, float(col_score.max()), rtol=1e-12, atol=1e-12)
            )
            if len(row_winners) != 1 or len(col_winners) != 1:
                outcomes.append(
                    DecisionOutcome(
                        key=key,
                        evidence_level=repetition,
                        status=DecisionStatus.TIE,
                        coverage=coverage,
                        target_candidate=target,
                        failure_reason="non_unique_row_or_column_maximum",
                        **timing,
                    )
                )
                continue
            predicted = f"row:{int(row_winners[0])}|column:{int(col_winners[0])}"
            outcomes.append(
                DecisionOutcome(
                    key=key,
                    evidence_level=repetition,
                    status=(
                        DecisionStatus.CORRECT
                        if predicted == target
                        else DecisionStatus.INCORRECT
                    ),
                    coverage=coverage,
                    target_candidate=target,
                    predicted_candidate=predicted,
                    **timing,
                )
            )
    return tuple(outcomes)
