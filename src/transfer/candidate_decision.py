"""Decision evaluation for candidate-membership tasks.

Each event contributes evidence to one dataset-declared candidate set. A
row/column speller is the current concrete contract: the selected item is the
intersection of the winning row and column. Acquisition flash codes are not
part of this layer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from data.candidate_task import CandidateTaskContract
from transfer.outcomes import (
    CandidateCoverage,
    DecisionKey,
    DecisionOutcome,
    DecisionStatus,
)


def candidate_event_key(candidate_id: int) -> str:
    return f"candidate:{int(candidate_id)}"


def expected_candidate_counts(contract: CandidateTaskContract, repetitions: int) -> dict[str, int]:
    repetitions = int(repetitions)
    if repetitions < 1:
        raise ValueError("repetitions must be positive.")
    return {
        candidate_event_key(candidate_id): repetitions for candidate_id in contract.candidate_ids
    }


def row_column_target(row: int, column: int) -> str:
    return f"row:{int(row)}|column:{int(column)}"


def parse_row_column_target(
    value: object,
    contract: CandidateTaskContract,
    *,
    name: str = "candidate target",
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    decoded: dict[str, int] = {}
    for part in value.split("|"):
        axis, separator, index_text = part.partition(":")
        if not separator or axis in decoded:
            raise ValueError(f"{name} must contain one row and one column.")
        try:
            index = int(index_text)
        except ValueError as error:
            raise ValueError(f"{name} has a non-integer index.") from error
        if str(index) != index_text:
            raise ValueError(f"{name} must use canonical integer indices.")
        decoded[axis] = index
    if set(decoded) != {"row", "column"}:
        raise ValueError(f"{name} must contain one row and one column.")
    if not 0 <= decoded["row"] < contract.n_rows or not 0 <= decoded["column"] < contract.n_columns:
        raise ValueError(f"{name} is outside the declared candidate grid.")
    return row_column_target(decoded["row"], decoded["column"])


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


def decision_outcomes_at_repetition(
    logits: Sequence[float],
    candidate_ids: Sequence[int],
    target_rows: Sequence[int],
    target_columns: Sequence[int],
    selection_ids: Sequence,
    repetition_indices: Sequence[int],
    *,
    contract: CandidateTaskContract,
    subject_ids: Sequence | None = None,
    onset_times_s: Sequence[float] | None = None,
    evidence_available_times_s: Sequence[float] | None = None,
    max_repetitions: int | None = None,
) -> tuple[DecisionOutcome, ...]:
    """Return one primary outcome for every decision and evidence prefix."""

    logits = np.asarray(logits, dtype=float)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    target_rows = np.asarray(target_rows, dtype=np.int64)
    target_columns = np.asarray(target_columns, dtype=np.int64)
    selection_ids = np.asarray(selection_ids).astype(str)
    repetition_indices = np.asarray(repetition_indices, dtype=np.int64)
    arrays = (
        logits,
        candidate_ids,
        target_rows,
        target_columns,
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
    if not np.isin(candidate_ids, contract.candidate_ids).all():
        raise ValueError("candidate_ids contain values outside the task contract.")
    if subject_ids is None:
        subjects = np.repeat("unspecified", len(logits))
    else:
        subjects = np.asarray(subject_ids).astype(str)
        if subjects.ndim != 1 or len(subjects) != len(logits):
            raise ValueError("subject_ids must be one-dimensional and aligned with logits.")
    onset = _aligned_optional_times(onset_times_s, n_events=len(logits), name="onset_times_s")
    available = _aligned_optional_times(
        evidence_available_times_s,
        n_events=len(logits),
        name="evidence_available_times_s",
    )
    if onset is not None and available is not None and np.any(available < onset):
        raise ValueError("evidence is available before its stimulus onset.")

    candidate_keys = np.asarray([candidate_event_key(value) for value in candidate_ids], dtype=str)
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
                expected_candidate_counts(contract, repetition), observed_counts
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
            truth_columns = np.unique(target_columns[selected])
            truth_valid = (
                len(truth_rows) == 1
                and len(truth_columns) == 1
                and 0 <= truth_rows[0] < contract.n_rows
                and 0 <= truth_columns[0] < contract.n_columns
            )
            target = (
                row_column_target(int(truth_rows[0]), int(truth_columns[0]))
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

            row_scores = np.zeros(contract.n_rows, dtype=float)
            column_scores = np.zeros(contract.n_columns, dtype=float)
            for event_index in np.flatnonzero(selected):
                candidate_id = int(candidate_ids[event_index])
                if candidate_id < contract.n_rows:
                    row_scores[candidate_id] += logits[event_index]
                else:
                    column_scores[candidate_id - contract.n_rows] += logits[event_index]
            row_winners = np.flatnonzero(
                np.isclose(row_scores, float(row_scores.max()), rtol=1e-12, atol=1e-12)
            )
            column_winners = np.flatnonzero(
                np.isclose(
                    column_scores,
                    float(column_scores.max()),
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
            if len(row_winners) != 1 or len(column_winners) != 1:
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
            predicted = row_column_target(int(row_winners[0]), int(column_winners[0]))
            outcomes.append(
                DecisionOutcome(
                    key=key,
                    evidence_level=repetition,
                    status=(
                        DecisionStatus.CORRECT if predicted == target else DecisionStatus.INCORRECT
                    ),
                    coverage=coverage,
                    target_candidate=target,
                    predicted_candidate=predicted,
                    **timing,
                )
            )
    return tuple(outcomes)
