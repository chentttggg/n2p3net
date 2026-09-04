"""Decision evaluation for candidate-membership tasks.

Each event contributes evidence to one dataset-declared candidate set. A
row/column speller is the current concrete contract: the selected item is the
intersection of the winning row and column. Acquisition flash codes are not
part of this layer.
"""

from __future__ import annotations

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

    # Vectorized row/column scoring. Each event contributes to one (decision,
    # repetition, candidate) cell; a prefix sum over repetitions reproduces the
    # per-level cumulative sum aggregation without the per-(decision, level)
    # Python loop, and a single isclose pass reproduces tie abstention. This is
    # the hot path for every result task (single-subject decision ledger), so
    # it stays in numpy; the working set is a few hundred events, far below the
    # size where a GPU round-trip would pay for itself.
    n_rows = int(contract.n_rows)
    n_columns = int(contract.n_columns)
    n_candidates = n_rows + n_columns
    n_decisions = len(decision_keys)

    key_to_index = {key: index for index, key in enumerate(decision_keys)}
    decision_index = np.fromiter(
        (
            key_to_index[DecisionKey(subject_id=subject, decision_id=selection)]
            for subject, selection in zip(subjects, selection_ids, strict=True)
        ),
        dtype=np.int64,
        count=len(logits),
    )
    candidate_index = candidate_ids  # canonical rows-then-columns: 0..n_candidates-1
    rep_index = repetition_indices  # 0-based, < max_repetitions

    counts = np.zeros((n_decisions, max_repetitions, n_candidates), dtype=np.int64)
    accumulated = np.zeros((n_decisions, max_repetitions, n_candidates), dtype=np.float64)
    np.add.at(counts, (decision_index, rep_index, candidate_index), 1)
    np.add.at(accumulated, (decision_index, rep_index, candidate_index), logits)
    cum_counts = np.cumsum(counts, axis=1)
    cum_logits = np.cumsum(accumulated, axis=1)

    # Coverage: repetition r (1-based) is complete when every candidate was
    # observed exactly r times across the first r repetitions.
    levels = np.arange(1, max_repetitions + 1)
    complete = np.all(cum_counts == levels[None, :, None], axis=2)

    # Target row/column is a selection property, constant across its events.
    target_row_by_decision = np.full(n_decisions, -1, dtype=np.int64)
    target_col_by_decision = np.full(n_decisions, -1, dtype=np.int64)
    truth_valid = np.zeros(n_decisions, dtype=bool)
    for index in range(n_decisions):
        mask = decision_index == index
        rows = np.unique(target_rows[mask])
        cols = np.unique(target_columns[mask])
        if (
            rows.size == 1
            and cols.size == 1
            and 0 <= rows[0] < n_rows
            and 0 <= cols[0] < n_columns
        ):
            target_row_by_decision[index] = int(rows[0])
            target_col_by_decision[index] = int(cols[0])
            truth_valid[index] = True

    # Row and column are two independent candidate sets; a non-unique maximum
    # on either axis abstains (None), matching the historical rtol=atol=1e-12.
    row_scores = cum_logits[:, :, :n_rows]
    col_scores = cum_logits[:, :, n_rows:]
    row_max = row_scores.max(axis=2)
    row_predicted = np.where(
        np.sum(np.isclose(row_scores, row_max[:, :, None], rtol=1e-12, atol=1e-12), axis=2) != 1,
        -1,
        np.argmax(row_scores, axis=2),
    )
    col_max = col_scores.max(axis=2)
    col_predicted = np.where(
        np.sum(np.isclose(col_scores, col_max[:, :, None], rtol=1e-12, atol=1e-12), axis=2) != 1,
        -1,
        np.argmax(col_scores, axis=2),
    )

    # Timing: prefix min/max over repetitions; +inf/-inf mark empty prefixes.
    onset_start = onset_end = evidence_available = None
    if onset is not None:
        onset_min = np.full((n_decisions, max_repetitions, n_candidates), np.inf)
        onset_max = np.full((n_decisions, max_repetitions, n_candidates), -np.inf)
        np.minimum.at(onset_min, (decision_index, rep_index, candidate_index), onset)
        np.maximum.at(onset_max, (decision_index, rep_index, candidate_index), onset)
        onset_start = np.minimum.accumulate(onset_min.min(axis=2), axis=1)
        onset_end = np.maximum.accumulate(onset_max.max(axis=2), axis=1)
    if available is not None:
        available_max = np.full((n_decisions, max_repetitions, n_candidates), -np.inf)
        np.maximum.at(available_max, (decision_index, rep_index, candidate_index), available)
        evidence_available = np.maximum.accumulate(available_max.max(axis=2), axis=1)

    expected_by_level = {
        r: expected_candidate_counts(contract, r) for r in range(1, max_repetitions + 1)
    }

    outcomes: list[DecisionOutcome] = []
    for d, key in enumerate(decision_keys):
        target = (
            row_column_target(target_row_by_decision[d], target_col_by_decision[d])
            if truth_valid[d]
            else None
        )
        for r in range(1, max_repetitions + 1):
            k = r - 1
            observed = {
                candidate_event_key(int(c)): int(cum_counts[d, k, c])
                for c in range(n_candidates)
                if cum_counts[d, k, c] > 0
            }
            coverage = CandidateCoverage.from_mappings(expected_by_level[r], observed)
            timing = {
                "onset_start_s": (
                    float(onset_start[d, k])
                    if onset_start is not None and np.isfinite(onset_start[d, k])
                    else None
                ),
                "onset_end_s": (
                    float(onset_end[d, k])
                    if onset_end is not None and np.isfinite(onset_end[d, k])
                    else None
                ),
                "evidence_available_s": (
                    float(evidence_available[d, k])
                    if evidence_available is not None and np.isfinite(evidence_available[d, k])
                    else None
                ),
            }
            if not complete[d, k]:
                outcomes.append(
                    DecisionOutcome(
                        key=key,
                        evidence_level=r,
                        status=DecisionStatus.INCOMPLETE,
                        coverage=coverage,
                        target_candidate=target,
                        failure_reason="incomplete_candidate_coverage",
                        **timing,
                    )
                )
                continue
            if not truth_valid[d]:
                outcomes.append(
                    DecisionOutcome(
                        key=key,
                        evidence_level=r,
                        status=DecisionStatus.INCOMPLETE,
                        coverage=coverage,
                        failure_reason="missing_or_inconsistent_target_candidate",
                        **timing,
                    )
                )
                continue
            row_winner = int(row_predicted[d, k])
            col_winner = int(col_predicted[d, k])
            if row_winner < 0 or col_winner < 0:
                outcomes.append(
                    DecisionOutcome(
                        key=key,
                        evidence_level=r,
                        status=DecisionStatus.TIE,
                        coverage=coverage,
                        target_candidate=target,
                        failure_reason="non_unique_row_or_column_maximum",
                        **timing,
                    )
                )
                continue
            predicted = row_column_target(row_winner, col_winner)
            outcomes.append(
                DecisionOutcome(
                    key=key,
                    evidence_level=r,
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
