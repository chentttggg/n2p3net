"""Leakage-safe P300 metrics and paired bootstrap utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .types import AuditInputError

MetricName = Literal["binary_auc", "binary_balanced_accuracy", "digit_hit"]


def _as_score(score: np.ndarray, n: int) -> np.ndarray:
    value = np.asarray(score, dtype=float).reshape(-1)
    if value.shape[0] != n:
        raise AuditInputError(f"score has {value.shape[0]} rows; expected {n}.")
    if not np.all(np.isfinite(value)):
        raise AuditInputError("score contains NaN or infinite values.")
    return value


def binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(target).reshape(-1)
    s = _as_score(score, y.shape[0])
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def binary_balanced_accuracy(
    target: np.ndarray, score: np.ndarray, threshold: float = 0.0
) -> float:
    y = np.asarray(target).reshape(-1)
    s = _as_score(score, y.shape[0])
    return float(balanced_accuracy_score(y, s >= threshold))


def digit_hit(
    score: np.ndarray,
    digits: np.ndarray,
    thought_numbers: np.ndarray,
    subjects: np.ndarray,
    *,
    aggregation: Literal["sum", "mean"] = "sum",
    expected_choices: int | None = 9,
) -> float:
    """Compute the subject-level 9-choice hit rate from trial logits.

    Scores are aggregated only within ``(subject, digit)``.  A thought number
    is required per subject; using a trial label here would silently turn the
    metric into a different task, so the function rejects inconsistent input.
    """

    scores = np.asarray(score, dtype=float).reshape(-1)
    digits = np.asarray(digits).reshape(-1)
    thoughts = np.asarray(thought_numbers).reshape(-1)
    subjects = np.asarray(subjects).reshape(-1)
    n = scores.shape[0]
    if not (digits.shape[0] == thoughts.shape[0] == subjects.shape[0] == n):
        raise AuditInputError("digit_hit inputs must have the same length.")
    if not np.all(np.isfinite(scores)):
        raise AuditInputError("score contains NaN or infinite values.")
    if aggregation not in ("sum", "mean"):
        raise AuditInputError(f"Unsupported digit aggregation {aggregation!r}.")
    if expected_choices is not None and expected_choices < 2:
        raise AuditInputError("expected_choices must be at least 2 or None.")

    hits: list[float] = []
    for subject in np.unique(subjects):
        row = subjects == subject
        subject_thoughts = np.unique(thoughts[row])
        if subject_thoughts.size != 1:
            raise AuditInputError(f"subject {subject!r} has inconsistent thought_numbers.")
        thought = subject_thoughts[0]
        present_digits = np.unique(digits[row])
        if expected_choices is not None and present_digits.size != expected_choices:
            raise AuditInputError(
                f"subject {subject!r} has {present_digits.size} choices; "
                f"expected exactly {expected_choices}."
            )
        if present_digits.size < 2:
            raise AuditInputError(f"subject {subject!r} must have at least two choices.")
        if not np.any(present_digits == thought):
            raise AuditInputError(
                f"subject {subject!r} thought number {thought!r} is absent from choices."
            )
        aggregated: dict[object, float] = {}
        for digit in present_digits:
            values = scores[row & (digits == digit)]
            aggregated[digit.item() if hasattr(digit, "item") else digit] = (
                float(values.sum()) if aggregation == "sum" else float(values.mean())
            )
        prediction = max(aggregated, key=aggregated.get)
        hits.append(float(prediction == (thought.item() if hasattr(thought, "item") else thought)))
    return float(np.mean(hits)) if hits else float("nan")


@dataclass(frozen=True)
class MetricSpec:
    """A named metric with explicit metadata requirements."""

    name: MetricName
    threshold: float = 0.0
    digit_aggregation: Literal["sum", "mean"] = "sum"
    digit_choices: int | None = 9

    def evaluate(
        self,
        target: np.ndarray,
        score: np.ndarray,
        *,
        digits: np.ndarray | None = None,
        thought_numbers: np.ndarray | None = None,
        subjects: np.ndarray | None = None,
    ) -> float:
        if self.name == "binary_auc":
            return binary_auc(target, score)
        if self.name == "binary_balanced_accuracy":
            return binary_balanced_accuracy(target, score, self.threshold)
        if digits is None or thought_numbers is None or subjects is None:
            raise AuditInputError("digit_hit requires digits, thought_numbers, and subjects.")
        return digit_hit(
            score,
            digits,
            thought_numbers,
            subjects,
            aggregation=self.digit_aggregation,
            expected_choices=self.digit_choices,
        )


@dataclass(frozen=True)
class BootstrapSummary:
    """Paired bootstrap summary for a metric difference."""

    estimate: float
    lower: float
    upper: float
    p_value: float
    valid_replicates: int


def _sample_indices(
    subjects: np.ndarray,
    rng: np.random.Generator,
    *,
    unit: Literal["row", "subject"],
) -> np.ndarray:
    if unit == "row":
        return rng.integers(0, subjects.shape[0], size=subjects.shape[0])
    unique = np.unique(subjects)
    sampled = rng.choice(unique, size=unique.shape[0], replace=True)
    return np.concatenate([np.flatnonzero(subjects == subject) for subject in sampled])


def paired_bootstrap_drop(
    metric: MetricSpec,
    target: np.ndarray,
    baseline_score: np.ndarray,
    edited_score: np.ndarray,
    *,
    subjects: np.ndarray,
    digits: np.ndarray | None = None,
    thought_numbers: np.ndarray | None = None,
    n_bootstrap: int = 128,
    seed: int = 0,
) -> BootstrapSummary:
    """Estimate ``metric(baseline) - metric(edited)`` with paired resampling."""

    y = np.asarray(target).reshape(-1)
    subjects = np.asarray(subjects).reshape(-1)
    baseline = _as_score(baseline_score, y.shape[0])
    edited = _as_score(edited_score, y.shape[0])
    if subjects.shape[0] != y.shape[0]:
        raise AuditInputError("subjects must align with target and scores.")
    base = metric.evaluate(
        y, baseline, digits=digits, thought_numbers=thought_numbers, subjects=subjects
    )
    edit = metric.evaluate(
        y, edited, digits=digits, thought_numbers=thought_numbers, subjects=subjects
    )
    estimate = base - edit
    unit: Literal["row", "subject"] = "subject" if metric.name == "digit_hit" else "row"
    rng = np.random.default_rng(seed)
    drops: list[float] = []
    for _ in range(int(n_bootstrap)):
        idx = _sample_indices(subjects, rng, unit=unit)
        try:
            b = metric.evaluate(
                y[idx],
                baseline[idx],
                digits=None if digits is None else np.asarray(digits)[idx],
                thought_numbers=None
                if thought_numbers is None
                else np.asarray(thought_numbers)[idx],
                subjects=subjects[idx],
            )
            e = metric.evaluate(
                y[idx],
                edited[idx],
                digits=None if digits is None else np.asarray(digits)[idx],
                thought_numbers=None
                if thought_numbers is None
                else np.asarray(thought_numbers)[idx],
                subjects=subjects[idx],
            )
        except AuditInputError:
            continue
        except ValueError:
            continue
        if np.isfinite(b) and np.isfinite(e):
            drops.append(float(b - e))
    if not drops:
        return BootstrapSummary(estimate, float("nan"), float("nan"), float("nan"), 0)
    values = np.asarray(drops)
    p_value = (1.0 + float(np.count_nonzero(values <= 0.0))) / (values.size + 1.0)
    return BootstrapSummary(
        estimate=estimate,
        lower=float(np.quantile(values, 0.025)),
        upper=float(np.quantile(values, 0.975)),
        p_value=p_value,
        valid_replicates=int(values.size),
    )


def paired_bootstrap_contrast(
    metric: MetricSpec,
    target: np.ndarray,
    left_score: np.ndarray,
    right_score: np.ndarray,
    *,
    subjects: np.ndarray,
    digits: np.ndarray | None = None,
    thought_numbers: np.ndarray | None = None,
    n_bootstrap: int = 128,
    seed: int = 0,
) -> BootstrapSummary:
    """Estimate ``metric(left) - metric(right)`` with shared bootstrap rows."""

    return paired_bootstrap_drop(
        metric,
        target,
        left_score,
        right_score,
        subjects=subjects,
        digits=digits,
        thought_numbers=thought_numbers,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values while preserving NaNs."""

    p = np.asarray(p_values, dtype=float).reshape(-1)
    result = np.full(p.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return result
    order = finite[np.argsort(p[finite])]
    ranked = p[order] * finite.size / np.arange(1, finite.size + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result
