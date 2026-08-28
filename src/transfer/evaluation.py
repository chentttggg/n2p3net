"""Within-subject decision evaluation helpers.

These functions evaluate a **held-out chronological suffix** only. Repetition
indices are assumed to restart at zero inside the suffix; do not pass training
prefix trials here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _aggregate_scores(
    logits: np.ndarray,
    digits: np.ndarray,
    *,
    aggregation: str,
    vocabulary: np.ndarray | None = None,
    logit_variances: np.ndarray | None = None,
) -> tuple[dict[int, float], int]:
    if aggregation not in {"sum", "mean", "trim0.2", "precision"}:
        raise ValueError("aggregation must be sum, mean, trim0.2, or precision.")
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
        elif aggregation == "trim0.2":
            lower, upper = np.quantile(values, [0.2, 0.8])
            kept = values[(values >= lower) & (values <= upper)]
            scores[int(digit)] = float(kept.sum())
        elif aggregation == "precision":
            assert variances is not None
            candidate_variances = variances[sel]
            if not np.isfinite(candidate_variances).all() or np.any(candidate_variances <= 0.0):
                raise ValueError("predictive variances must be finite and positive.")
            weights = 1.0 / candidate_variances
            scores[int(digit)] = float(np.dot(weights, values) / weights.sum())
    best = max(scores.items(), key=lambda item: (item[1], -item[0]))
    return scores, best[0]


def hit_at_repetition(
    logits: Sequence[float],
    digits: Sequence[int],
    group_ids: Sequence,
    truth_by_group: Mapping[object, object],
    repetition_indices: Sequence[int],
    *,
    aggregation: str = "sum",
    max_repetitions: int | None = None,
    logit_variances: Sequence[float] | None = None,
    candidate_vocabulary: Sequence[int] | None = None,
) -> dict[int, float]:
    """Return 9-choice hit rate at every repetition prefix 1..R."""

    logits = np.asarray(logits, dtype=float)
    digits = np.asarray(digits, dtype=np.int64)
    group_ids = np.asarray(group_ids).astype(str)
    repetition_indices = np.asarray(repetition_indices, dtype=np.int64)
    if not (len(logits) == len(digits) == len(group_ids) == len(repetition_indices)):
        raise ValueError("logits/digits/group_ids/repetition_indices must be aligned.")
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
            scores, predicted = _aggregate_scores(
                logits[sel],
                digits[sel],
                aggregation=aggregation,
                vocabulary=vocabulary,
                logit_variances=None if variances is None else variances[sel],
            )
            del scores
            correct += int(predicted == int(truth))
            total += 1
        hits[r] = float(correct / total) if total else float("nan")
    return hits
