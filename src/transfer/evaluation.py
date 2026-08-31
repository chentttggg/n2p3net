"""Within-subject decision evaluation helpers.

These functions evaluate a **held-out chronological suffix** only. Repetition
indices may retain their original scheduled origin (for example, 8, 9, ...
after an eight-repetition prefix); the validated contiguous suffix is normalized
internally for the 1..R curve. Do not pass training prefix trials here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from models.decision import (
    COUNT_AGGREGATIONS,
    DEFAULT_EVIDENCE_AGGREGATION,
    DEFAULT_EVIDENCE_COUNT_POWER,
    count_tempered_evidence_scores,
)


def _aggregate_scores(
    logits: np.ndarray,
    digits: np.ndarray,
    *,
    aggregation: str,
    vocabulary: np.ndarray | None = None,
    logit_variances: np.ndarray | None = None,
    evidence_count_power: float = DEFAULT_EVIDENCE_COUNT_POWER,
) -> tuple[dict[int, float], int | None]:
    valid_aggregations = COUNT_AGGREGATIONS | {"trim0.2", "precision"}
    if aggregation not in valid_aggregations:
        raise ValueError(f"aggregation must be one of {sorted(valid_aggregations)}.")
    logits = np.asarray(logits, dtype=float)
    digits = np.asarray(digits, dtype=np.int64)
    variances = None if logit_variances is None else np.asarray(logit_variances, dtype=float)
    if variances is not None and variances.shape != logits.shape:
        raise ValueError("logit_variances must align with logits.")
    if aggregation == "precision" and variances is None:
        raise ValueError("precision aggregation requires per-trial predictive variances.")
    if vocabulary is None:
        vocabulary = np.unique(digits)
    vocabulary = np.asarray(vocabulary, dtype=np.int64)
    if vocabulary.ndim != 1 or len(vocabulary) == 0:
        raise ValueError("candidate vocabulary must be a non-empty one-dimensional array.")
    scores: dict[int, float] = {}
    counts: dict[int, int] = {}
    for digit in vocabulary:
        sel = digits == digit
        counts[int(digit)] = int(sel.sum())
        if not sel.any():
            scores[int(digit)] = -np.inf
            continue
        values = logits[sel]
        if aggregation == "sum":
            scores[int(digit)] = float(values.sum())
        elif aggregation == "mean":
            scores[int(digit)] = float(values.mean())
        elif aggregation == "tempered_evidence":
            scores[int(digit)] = float(
                count_tempered_evidence_scores(
                    np.asarray([values.sum()]),
                    np.asarray([len(values)], dtype=float),
                    np.asarray([len(values)], dtype=float),
                    count_power=evidence_count_power,
                )[0]
            )
        elif aggregation == "trim0.2":
            ordered = np.sort(values)
            trim = int(np.floor(0.2 * len(ordered)))
            kept = ordered[trim : len(ordered) - trim] if trim else ordered
            scores[int(digit)] = float(kept.sum())
        elif aggregation == "precision":
            assert variances is not None
            candidate_variances = variances[sel]
            if not np.isfinite(candidate_variances).all() or np.any(candidate_variances <= 0.0):
                raise ValueError("predictive variances must be finite and positive.")
            weights = 1.0 / candidate_variances
            scores[int(digit)] = float(np.dot(weights, values) / weights.sum())
    finite_scores = np.asarray([scores[int(digit)] for digit in vocabulary], dtype=float)
    maximum = float(np.max(finite_scores))
    tied = np.flatnonzero(np.isclose(finite_scores, maximum, rtol=1e-12, atol=1e-12))
    predicted = int(vocabulary[tied[0]]) if len(tied) == 1 else None
    return scores, predicted


def hit_at_repetition(
    logits: Sequence[float],
    digits: Sequence[int],
    group_ids: Sequence,
    truth_by_group: Mapping[object, object],
    repetition_indices: Sequence[int],
    *,
    aggregation: str = DEFAULT_EVIDENCE_AGGREGATION,
    max_repetitions: int | None = None,
    logit_variances: Sequence[float] | None = None,
    candidate_vocabulary: Sequence[int] | None = None,
    evidence_count_power: float = DEFAULT_EVIDENCE_COUNT_POWER,
) -> dict[int, float]:
    """Return 9-choice hit rate at every repetition prefix 1..R."""

    logits = np.asarray(logits, dtype=float)
    digits = np.asarray(digits, dtype=np.int64)
    group_ids = np.asarray(group_ids).astype(str)
    repetition_indices = np.asarray(repetition_indices, dtype=np.int64)
    if not (len(logits) == len(digits) == len(group_ids) == len(repetition_indices)):
        raise ValueError("logits/digits/group_ids/repetition_indices must be aligned.")
    if len(repetition_indices) == 0:
        raise ValueError("repetition_indices cannot be empty.")
    if np.any(repetition_indices < 0):
        raise ValueError("repetition_indices must be non-negative.")
    unique_repetitions = np.unique(repetition_indices)
    if not np.array_equal(
        unique_repetitions,
        np.arange(unique_repetitions[0], unique_repetitions[-1] + 1, dtype=np.int64),
    ):
        raise ValueError(
            "repetition_indices must be contiguous; missing suffix repetitions cannot be "
            "compressed."
        )
    # Keep the splitter's output auditable in the original schedule while using
    # a zero-based view for the public 1..R aggregation curve.
    repetition_indices = repetition_indices - unique_repetitions[0]
    if not np.isfinite(logits).all():
        raise ValueError("logits contain NaN/inf.")
    variances = None if logit_variances is None else np.asarray(logit_variances, dtype=float)
    if variances is not None and variances.shape != logits.shape:
        raise ValueError("logit_variances must align with logits.")
    vocabulary = (
        np.unique(digits)
        if candidate_vocabulary is None
        else np.asarray(candidate_vocabulary, dtype=np.int64)
    )
    if vocabulary.ndim != 1 or len(np.unique(vocabulary)) != len(vocabulary):
        raise ValueError("candidate_vocabulary must be one-dimensional and unique.")
    if len(vocabulary) < 2:
        raise ValueError("candidate selection requires at least two candidate codes.")
    groups = np.unique(group_ids)
    if max_repetitions is None:
        max_repetitions = int(repetition_indices.max()) + 1
    if max_repetitions < 1:
        raise ValueError("max_repetitions must be positive.")
    hits: dict[int, float] = {}
    for r in range(1, max_repetitions + 1):
        correct = 0
        total = 0
        for group in groups:
            truth = truth_by_group.get(group)
            if truth is None:
                continue
            sel = (group_ids == group) & (repetition_indices < r)
            if not sel.any():
                continue
            # A candidate with no surviving trial is an incomplete decision,
            # not a candidate that should disappear from the argmax.  Count the
            # group as a miss while keeping it in the requested denominator.
            if not all(np.any(digits[sel] == candidate) for candidate in vocabulary):
                total += 1
                continue
            scores, predicted = _aggregate_scores(
                logits[sel],
                digits[sel],
                aggregation=aggregation,
                vocabulary=vocabulary,
                logit_variances=None if variances is None else variances[sel],
                evidence_count_power=evidence_count_power,
            )
            del scores
            correct += int(predicted is not None and predicted == int(truth))
            total += 1
        hits[r] = float(correct / total) if total else float("nan")
    return hits


def candidate_evidence_endpoints(
    logits: Sequence[float],
    digits: Sequence[int],
    group_ids: Sequence,
    truth_by_group: Mapping[object, object],
    repetition_indices: Sequence[int],
    *,
    aggregation: str = DEFAULT_EVIDENCE_AGGREGATION,
    max_repetitions: int | None = None,
    candidate_vocabulary: Sequence[int] | None = None,
    evidence_count_power: float = DEFAULT_EVIDENCE_COUNT_POWER,
) -> dict[str, object]:
    """Evaluate raw-all, balanced-all, and complete balanced hit@R endpoints.

    ``raw_all`` consumes every post-boundary observation. ``balanced_all`` uses
    the first ``R_s=min_d n_{s,d}`` observations from every candidate in a
    group. The hit@R curve includes a group only while every candidate supplies
    at least R observations; callers can use the returned correct/eligible
    counts with their full requested-group denominator for operational misses.
    """

    values = np.asarray(logits, dtype=float)
    candidates = np.asarray(digits, dtype=np.int64)
    groups_array = np.asarray(group_ids).astype(str)
    repetitions = np.asarray(repetition_indices, dtype=np.int64)
    if not (
        values.ndim
        == candidates.ndim
        == groups_array.ndim
        == repetitions.ndim
        == 1
    ) or not (
        len(values) == len(candidates) == len(groups_array) == len(repetitions)
    ):
        raise ValueError("endpoint arrays must be aligned one-dimensional vectors.")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("endpoint logits must be finite and non-empty.")
    if np.any(repetitions < 0):
        raise ValueError("endpoint repetition_indices must be non-negative.")
    vocabulary = (
        np.unique(candidates)
        if candidate_vocabulary is None
        else np.asarray(candidate_vocabulary, dtype=np.int64)
    )
    if (
        vocabulary.ndim != 1
        or len(vocabulary) < 2
        or len(np.unique(vocabulary)) != len(vocabulary)
    ):
        raise ValueError(
            "candidate_vocabulary must be a unique one-dimensional vector with at least two codes."
        )

    raw_correct = 0
    balanced_correct = 0
    total = 0
    raw_predictions: dict[str, int | None] = {}
    balanced_predictions: dict[str, int | None] = {}
    counts_by_group: dict[str, dict[str, int]] = {}
    balanced_repetitions_by_group: dict[str, int] = {}
    group_rows: dict[str, np.ndarray] = {}
    for group in np.unique(groups_array):
        group = str(group)
        truth = truth_by_group.get(group)
        if truth is None:
            continue
        rows = np.flatnonzero(groups_array == group)
        group_rows[group] = rows
        counts = {
            str(int(candidate)): int(np.count_nonzero(candidates[rows] == candidate))
            for candidate in vocabulary
        }
        counts_by_group[group] = counts
        balanced_repetitions = min(counts.values())
        balanced_repetitions_by_group[group] = balanced_repetitions
        total += 1
        if balanced_repetitions == 0:
            raw_predictions[group] = None
            balanced_predictions[group] = None
            continue
        _, raw_predicted = _aggregate_scores(
            values[rows],
            candidates[rows],
            aggregation=aggregation,
            vocabulary=vocabulary,
            evidence_count_power=evidence_count_power,
        )
        balanced_rows = rows[repetitions[rows] < balanced_repetitions]
        _, balanced_predicted = _aggregate_scores(
            values[balanced_rows],
            candidates[balanced_rows],
            aggregation=aggregation,
            vocabulary=vocabulary,
            evidence_count_power=evidence_count_power,
        )
        raw_predictions[group] = raw_predicted
        balanced_predictions[group] = balanced_predicted
        raw_correct += int(raw_predicted is not None and raw_predicted == int(truth))
        balanced_correct += int(
            balanced_predicted is not None and balanced_predicted == int(truth)
        )

    if max_repetitions is None:
        max_repetitions = max(
            balanced_repetitions_by_group.values(),
            default=0,
        )
    if max_repetitions < 1:
        raise ValueError("max_repetitions must be positive when endpoint groups exist.")
    correct_by_repetition: dict[int, int] = {}
    eligible_by_repetition: dict[int, int] = {}
    hit_by_repetition: dict[int, float] = {}
    for repetition_count in range(1, max_repetitions + 1):
        correct = 0
        eligible = 0
        for group, rows in group_rows.items():
            counts = counts_by_group[group]
            if any(count < repetition_count for count in counts.values()):
                continue
            selected = rows[repetitions[rows] < repetition_count]
            _, predicted = _aggregate_scores(
                values[selected],
                candidates[selected],
                aggregation=aggregation,
                vocabulary=vocabulary,
                evidence_count_power=evidence_count_power,
            )
            eligible += 1
            correct += int(
                predicted is not None and predicted == int(truth_by_group[group])
            )
        correct_by_repetition[repetition_count] = correct
        eligible_by_repetition[repetition_count] = eligible
        hit_by_repetition[repetition_count] = (
            float(correct / eligible) if eligible else float("nan")
        )

    return {
        "raw_all_hit_rate": float(raw_correct / total) if total else float("nan"),
        "balanced_all_hit_rate": (
            float(balanced_correct / total) if total else float("nan")
        ),
        "raw_all_correct": raw_correct,
        "balanced_all_correct": balanced_correct,
        "total_groups": total,
        "raw_predictions_by_group": raw_predictions,
        "balanced_predictions_by_group": balanced_predictions,
        "candidate_counts_by_group": counts_by_group,
        "balanced_repetitions_by_group": balanced_repetitions_by_group,
        "hit_by_repetition": hit_by_repetition,
        "correct_by_repetition": correct_by_repetition,
        "eligible_by_repetition": eligible_by_repetition,
    }
