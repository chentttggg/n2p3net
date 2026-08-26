"""Stable data contracts for the P300 Probe--Erase--Closure audit.

The audit package intentionally does not import N2P3-Net, braindecode, or any
model implementation.  These contracts are the boundary between an EEG
experiment and the audit machinery.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class AuditInputError(ValueError):
    """Raised when an audit input violates an explicit contract."""


def _as_1d(value: Sequence[Any] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise AuditInputError(f"{name} must be one-dimensional, got shape {array.shape}.")
    return array


@dataclass(frozen=True)
class P300AuditData:
    """Trial-level data consumed by the audit.

    Parameters
    ----------
    X:
        EEG epochs with shape ``(n_trials, n_channels, n_times)``.  Epochs are
        stimulus locked and must use the supplied ``time_ms`` axis.
    target:
        Binary target/non-target labels.  These labels are used only by the
        task metric; the feature extractor never receives them.
    subjects:
        Subject identifier per trial.  All scientific splits should be
        subject-disjoint.
    digits / thought_numbers:
        Optional fields required for the second-level 9-choice metric.  A
        thought number is constant for all rows belonging to one subject.
    channel_names:
        Physical channel names in the same order as ``X``.  Names are used for
        P300 spatial summaries, with deterministic index fallbacks when a
        canonical name is absent.
    """

    X: np.ndarray
    target: np.ndarray
    subjects: np.ndarray
    time_ms: np.ndarray
    digits: np.ndarray | None = None
    thought_numbers: np.ndarray | None = None
    run_ids: np.ndarray | None = None
    channel_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        epochs = np.asarray(self.X)
        target = _as_1d(self.target, "target")
        subjects = _as_1d(self.subjects, "subjects")
        time_ms = _as_1d(self.time_ms, "time_ms")
        if epochs.ndim != 3:
            raise AuditInputError(f"X must have shape (N,C,T), got {epochs.shape}.")
        n_trials, n_channels, n_times = epochs.shape
        if target.shape[0] != n_trials or subjects.shape[0] != n_trials:
            raise AuditInputError("X, target, and subjects must have the same trial count.")
        if time_ms.shape[0] != n_times:
            raise AuditInputError("time_ms length must match X.shape[-1].")
        if n_trials == 0 or n_channels == 0 or n_times < 8:
            raise AuditInputError("X must contain trials, channels, and at least 8 time points.")
        if not np.issubdtype(epochs.dtype, np.number):
            raise AuditInputError("X must be numeric.")
        if not np.all(np.isfinite(epochs)):
            raise AuditInputError("X contains NaN or infinite values.")
        if not np.all(np.isfinite(time_ms.astype(float))):
            raise AuditInputError("time_ms contains NaN or infinite values.")
        if np.any(np.diff(time_ms.astype(float)) <= 0):
            raise AuditInputError("time_ms must be strictly increasing.")
        if not np.all(np.isin(target, [0, 1])):
            raise AuditInputError("target must contain only 0/1 labels.")
        if self.channel_names and len(self.channel_names) != n_channels:
            raise AuditInputError("channel_names length must match X.shape[1].")
        if self.digits is not None:
            digits = _as_1d(self.digits, "digits")
            if digits.shape[0] != n_trials:
                raise AuditInputError("digits must have one value per trial.")
            if not np.all(np.isfinite(digits.astype(float))):
                raise AuditInputError("digits contains NaN or infinite values.")
        if self.thought_numbers is not None:
            thoughts = _as_1d(self.thought_numbers, "thought_numbers")
            if thoughts.shape[0] != n_trials:
                raise AuditInputError("thought_numbers must have one value per trial.")
        if self.run_ids is not None and len(_as_1d(self.run_ids, "run_ids")) != n_trials:
            raise AuditInputError("run_ids must have one value per trial.")

        object.__setattr__(self, "X", epochs.astype(np.float64, copy=False))
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "time_ms", time_ms.astype(float, copy=False))
        if self.digits is not None:
            object.__setattr__(self, "digits", np.asarray(self.digits))
        if self.thought_numbers is not None:
            object.__setattr__(self, "thought_numbers", np.asarray(self.thought_numbers))
        if self.run_ids is not None:
            object.__setattr__(self, "run_ids", np.asarray(self.run_ids))
        if not self.channel_names:
            object.__setattr__(self, "channel_names", tuple(f"CH{i}" for i in range(n_channels)))
        else:
            object.__setattr__(self, "channel_names", tuple(str(x) for x in self.channel_names))

    @property
    def n_trials(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.X.shape[2])

    @property
    def has_digit_labels(self) -> bool:
        return self.digits is not None and self.thought_numbers is not None

    def subset(self, indices: Sequence[int] | np.ndarray) -> P300AuditData:
        """Return a validated view copy for a split or a bootstrap sample."""

        idx = np.asarray(indices, dtype=int)
        if idx.ndim != 1:
            raise AuditInputError("subset indices must be one-dimensional.")
        if np.any(idx < 0) or np.any(idx >= self.n_trials):
            raise AuditInputError("subset indices are outside the trial range.")
        return P300AuditData(
            X=self.X[idx],
            target=self.target[idx],
            subjects=self.subjects[idx],
            time_ms=self.time_ms,
            digits=None if self.digits is None else self.digits[idx],
            thought_numbers=None if self.thought_numbers is None else self.thought_numbers[idx],
            run_ids=None if self.run_ids is None else self.run_ids[idx],
            channel_names=self.channel_names,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class P300Split:
    """Explicit train/validation/test row indices.

    The audit never silently creates a random split.  This makes it possible
    to reuse the exact LOSO folds used by the model benchmark and makes
    leakage visible at the call site.
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    require_subject_disjoint: bool = True

    def __post_init__(self) -> None:
        normalized = []
        for name, value in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            idx = np.asarray(value, dtype=int)
            if idx.ndim != 1 or idx.size == 0:
                raise AuditInputError(f"{name} indices must be a non-empty one-dimensional array.")
            if np.any(idx < 0):
                raise AuditInputError(f"{name} indices must be non-negative.")
            if np.unique(idx).size != idx.size:
                raise AuditInputError(f"{name} indices contain duplicates.")
            normalized.append(idx)
        if np.intersect1d(normalized[0], normalized[1]).size:
            raise AuditInputError("train and validation indices overlap.")
        if np.intersect1d(normalized[0], normalized[2]).size:
            raise AuditInputError("train and test indices overlap.")
        if np.intersect1d(normalized[1], normalized[2]).size:
            raise AuditInputError("validation and test indices overlap.")
        object.__setattr__(self, "train", normalized[0])
        object.__setattr__(self, "validation", normalized[1])
        object.__setattr__(self, "test", normalized[2])

    def validate_against(self, data: P300AuditData) -> None:
        """Validate bounds, coverage, and subject-level separation."""

        for name, idx in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            if np.any(idx >= data.n_trials):
                raise AuditInputError(f"{name} contains an index outside n_trials={data.n_trials}.")
        if self.require_subject_disjoint:
            groups = [
                set(data.subjects[idx].tolist()) for idx in (self.train, self.validation, self.test)
            ]
            if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
                raise AuditInputError(
                    "train/validation/test subjects overlap; subject leakage is forbidden."
                )

    @property
    def all_indices(self) -> np.ndarray:
        return np.concatenate((self.train, self.validation, self.test))


@dataclass(frozen=True)
class FeatureTable:
    """Scalar P300 feature matrix with immutable registry metadata."""

    values: np.ndarray
    names: tuple[str, ...]
    families: tuple[str, ...]
    descriptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise AuditInputError(f"feature values must have shape (N,F), got {values.shape}.")
        if len(self.names) != values.shape[1] or len(self.families) != values.shape[1]:
            raise AuditInputError("feature names and families must match the feature column count.")
        if len(set(self.names)) != len(self.names):
            raise AuditInputError("feature names must be unique.")
        if self.descriptions and len(self.descriptions) != values.shape[1]:
            raise AuditInputError("feature descriptions must match the feature column count.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "families", tuple(self.families))
        if not self.descriptions:
            object.__setattr__(self, "descriptions", tuple("" for _ in self.names))

    @property
    def n_trials(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    def column(self, name: str) -> np.ndarray:
        try:
            index = self.names.index(name)
        except ValueError as exc:
            raise AuditInputError(f"Unknown feature {name!r}.") from exc
        return self.values[:, index]
