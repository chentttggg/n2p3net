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
) -> tuple[dict[int, float], int]:
    vocabulary = np.arange(1, 10, dtype=np.int64)
    scores: dict[int, float] = {}
    counts: dict[int, int] = {}
    for digit in vocabulary:
        sel = digits == digit
        counts[int(digit)] = int(sel.sum())
        if not sel.any():
            scores[int(digit)] = -np.inf
            continue
        values = np.asarray(logits, dtype=float)[sel]
        if aggregation == "sum":
            scores[int(digit)] = float(values.sum())
        elif aggregation == "mean":
            scores[int(digit)] = float(values.mean())
        elif aggregation == "trim0.2":
            lower, upper = np.quantile(values, [0.2, 0.8])
            kept = values[(values >= lower) & (values <= upper)]
            scores[int(digit)] = float(kept.sum())
        elif aggregation == "precision":
            variance = float(np.var(values))
            scores[int(digit)] = float(values.sum() / max(variance, 1e-12))
        else:
            raise ValueError("aggregation must be sum, mean, trim0.2, or precision.")
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
            )
            del scores
            correct += int(predicted == int(truth))
            total += 1
        hits[r] = float(correct / total) if total else float("nan")
    return hits
