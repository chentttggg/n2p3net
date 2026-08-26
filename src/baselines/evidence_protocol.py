"""Fail-closed fold and acquisition-budget semantics for P300 evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from data.events import AVAILABLE, ScheduledEventTimeline


@dataclass(frozen=True)
class EvidenceBudget:
    kind: str
    value: int | float | None

    def __post_init__(self) -> None:
        if self.kind == "all":
            if self.value is not None:
                raise ValueError("all evidence budget must use value=None.")
            return
        if self.kind in {"exact", "prefix_minK", "flash"}:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, np.integer)):
                raise ValueError(f"{self.kind} evidence budget must be a positive integer.")
            if int(self.value) < 1:
                raise ValueError(f"{self.kind} evidence budget must be positive.")
            object.__setattr__(self, "value", int(self.value))
            return
        if self.kind == "time":
            if self.value is None or not np.isfinite(float(self.value)) or float(self.value) <= 0.0:
                raise ValueError(
                    "time evidence budget must be a positive finite number of seconds."
                )
            object.__setattr__(self, "value", float(self.value))
            return
        raise ValueError(f"Unknown evidence budget kind {self.kind!r}.")

    @property
    def token(self) -> str:
        if self.kind == "all":
            return "all"
        if self.kind in {"exact", "prefix_minK", "flash"}:
            return f"{self.kind}@{int(self.value)}"
        if self.kind == "time":
            return f"time@{float(self.value):g}s"
        raise ValueError(f"Unknown evidence budget kind {self.kind!r}.")


def build_evidence_budgets(
    evidence_ks: Sequence[int | None],
    *,
    flash_budgets: Sequence[int] = (),
    time_budgets_s: Sequence[float] = (),
) -> tuple[EvidenceBudget, ...]:
    output: list[EvidenceBudget] = []
    for value in evidence_ks:
        if value is None:
            candidates = (EvidenceBudget("all", None),)
        else:
            k = int(value)
            if k < 1:
                raise ValueError("Evidence K budgets must be positive.")
            candidates = (EvidenceBudget("exact", k), EvidenceBudget("prefix_minK", k))
        for candidate in candidates:
            if candidate not in output:
                output.append(candidate)
    for value in flash_budgets:
        candidate = EvidenceBudget("flash", int(value))
        if int(value) < 1:
            raise ValueError("Flash budgets must be positive.")
        if candidate not in output:
            output.append(candidate)
    for value in time_budgets_s:
        candidate = EvidenceBudget("time", float(value))
        if float(value) <= 0.0:
            raise ValueError("Time budgets must be positive.")
        if candidate not in output:
            output.append(candidate)
    return tuple(output)


def validate_outer_folds(
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    subject_ids: np.ndarray,
    *,
    protocol: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validate row and subject isolation before any model is fitted."""

    if protocol not in {"loso", "partial_loso", "custom", "within_subject"}:
        raise ValueError("fold protocol must be loso, partial_loso, custom, or within_subject.")
    subjects = np.asarray(subject_ids).astype(str)
    n_rows = len(subjects)
    if n_rows == 0 or not folds:
        raise ValueError("Evaluation needs non-empty rows and folds.")
    normalized: list[tuple[np.ndarray, np.ndarray]] = []
    test_counts = np.zeros(n_rows, dtype=np.int64)
    tested_subjects: list[str] = []
    for fold_index, (train_raw, test_raw) in enumerate(folds):
        train_array = np.asarray(train_raw)
        test_array = np.asarray(test_raw)
        if train_array.dtype != np.bool_ or test_array.dtype != np.bool_:
            raise ValueError(f"Fold {fold_index} masks must have boolean dtype.")
        if train_array.shape != (n_rows,) or test_array.shape != (n_rows,):
            raise ValueError(
                f"Fold {fold_index} masks must be ({n_rows},), got "
                f"{train_array.shape}/{test_array.shape}."
            )
        train = train_array.copy()
        test = test_array.copy()
        if not bool(train.any()) or not bool(test.any()):
            raise ValueError(f"Fold {fold_index} has an empty train or test partition.")
        if bool((train & test).any()):
            raise ValueError(f"Fold {fold_index} has overlapping train/test rows.")
        train_subjects = set(subjects[train].tolist())
        test_subjects = set(subjects[test].tolist())
        if protocol == "within_subject":
            if len(train_subjects) != 1 or train_subjects != test_subjects:
                raise ValueError(
                    f"Within-subject fold {fold_index} must contain one shared subject."
                )
        elif train_subjects & test_subjects:
            raise ValueError(
                f"Fold {fold_index} leaks test subjects into train: "
                f"{sorted(train_subjects & test_subjects)[:5]}."
            )
        if protocol in {"loso", "partial_loso"}:
            if not np.array_equal(train, ~test):
                raise ValueError(f"LOSO fold {fold_index} train mask must complement test mask.")
            if len(test_subjects) != 1:
                raise ValueError(f"LOSO fold {fold_index} must hold out exactly one subject.")
        test_counts += test.astype(np.int64)
        tested_subjects.extend(sorted(test_subjects))
        normalized.append((train, test))
    if np.any(test_counts > 1):
        raise ValueError("An evaluation row appears in more than one outer test fold.")
    if protocol != "within_subject" and len(tested_subjects) != len(set(tested_subjects)):
        raise ValueError("An evaluation subject appears in more than one outer test fold.")
    if protocol == "loso":
        if not np.all(test_counts == 1):
            raise ValueError("Complete LOSO requires every available row in exactly one test fold.")
        if set(tested_subjects) != set(np.unique(subjects).tolist()):
            raise ValueError("Complete LOSO must test every available subject exactly once.")
    return normalized


def evidence_row_indices(
    timeline: ScheduledEventTimeline,
    group: str,
    digit_vocab: Sequence[int],
    budget: EvidenceBudget,
) -> np.ndarray | None:
    """Map one frozen acquisition budget to model-ready evidence rows."""

    timeline.validate()
    groups = np.asarray(timeline.group_ids).astype(str)
    rows = np.flatnonzero(groups == str(group))
    if len(rows) == 0:
        return None
    order = np.argsort(np.asarray(timeline.onset_times_s)[rows], kind="stable")
    rows = rows[order]
    statuses = np.asarray(timeline.statuses).astype(str)[rows]
    evidence = np.asarray(timeline.evidence_indices, dtype=np.int64)[rows]
    digits = np.asarray(timeline.stimulus_ids, dtype=np.int64)[rows]
    available = (statuses == AVAILABLE) & (evidence >= 0)
    vocab = tuple(int(value) for value in digit_vocab)

    if budget.kind == "all":
        selected = np.flatnonzero(available)
        if any(not np.any(digits[selected] == digit) for digit in vocab):
            return None
    elif budget.kind == "exact":
        selected_parts: list[np.ndarray] = []
        for digit in vocab:
            positions = np.flatnonzero(available & (digits == digit))
            if len(positions) < int(budget.value):
                return None
            selected_parts.append(positions[: int(budget.value)])
        selected = np.sort(np.concatenate(selected_parts))
    elif budget.kind == "prefix_minK":
        checkpoints: list[int] = []
        for digit in vocab:
            positions = np.flatnonzero(available & (digits == digit))
            if len(positions) < int(budget.value):
                return None
            checkpoints.append(int(positions[int(budget.value) - 1]))
        selected = np.flatnonzero(available & (np.arange(len(rows)) <= max(checkpoints)))
    elif budget.kind == "flash":
        prefix = np.arange(len(rows)) < int(budget.value)
        selected = np.flatnonzero(available & prefix)
        if any(not np.any(digits[selected] == digit) for digit in vocab):
            return None
    elif budget.kind == "time":
        if not timeline.online_causal:
            raise ValueError(
                "time@T is unavailable because the cached preprocessing is not online-causal."
            )
        origin = float(np.asarray(timeline.onset_times_s)[rows[0]])
        available_at = np.asarray(timeline.evidence_available_times_s, dtype=float)[rows]
        selected = np.flatnonzero(available & ((available_at - origin) <= float(budget.value)))
        if any(not np.any(digits[selected] == digit) for digit in vocab):
            return None
    else:
        raise ValueError(f"Unknown evidence budget kind {budget.kind!r}.")
    return evidence[selected].astype(np.int64)


def row_acquisition_indices(timeline: ScheduledEventTimeline) -> np.ndarray:
    """Return the scheduled-event ordinal for each model-ready epoch row."""

    timeline.validate()
    output = np.empty(timeline.n_available, dtype=np.int64)
    groups = np.asarray(timeline.group_ids).astype(str)
    evidence = np.asarray(timeline.evidence_indices, dtype=np.int64)
    onset = np.asarray(timeline.onset_times_s, dtype=float)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        rows = rows[np.argsort(onset[rows], kind="stable")]
        for ordinal, row in enumerate(rows):
            if evidence[row] >= 0:
                output[evidence[row]] = ordinal
    return output
