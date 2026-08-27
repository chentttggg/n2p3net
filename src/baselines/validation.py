"""Shared group-disjoint validation split for all trainable baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupValidationSplit:
    train_mask: np.ndarray
    validation_mask: np.ndarray
    validation_groups: tuple[object, ...]

    @property
    def n_validation_groups(self) -> int:
        return len(self.validation_groups)


@dataclass(frozen=True)
class SubjectAuditSplit:
    optimization_mask: np.ndarray
    audit_mask: np.ndarray
    audit_subjects: tuple[object, ...]

    @property
    def n_audit_subjects(self) -> int:
        return len(self.audit_subjects)


def group_disjoint_validation_split(
    group_ids: np.ndarray,
    *,
    fraction: float | None,
    min_groups: int = 2,
    max_groups: int = 12,
    min_train_groups: int = 2,
    seed: int = 0,
) -> GroupValidationSplit:
    """Select complete validation groups deterministically.

    ``fraction=None`` explicitly disables validation for fit-only callers.
    A requested validation split with fewer than four groups fails closed;
    no training-set resubstitution calibration is permitted.
    """

    group_ids = np.asarray(group_ids)
    if group_ids.ndim != 1:
        raise ValueError(f"group_ids must be one-dimensional, got {group_ids.shape}.")
    unique = np.unique(group_ids)
    all_train = np.ones(len(group_ids), dtype=bool)
    no_validation = np.zeros(len(group_ids), dtype=bool)
    if fraction is None:
        return GroupValidationSplit(all_train, no_validation, ())
    if len(unique) < 4:
        raise ValueError(
            "Group-disjoint validation requires at least four available groups; "
            "training-set resubstitution calibration is forbidden."
        )
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"validation fraction must be in (0,1), got {fraction}.")
    if min_groups < 1 or max_groups < min_groups:
        raise ValueError("invalid validation group bounds.")

    count = int(round(float(fraction) * len(unique)))
    count = max(min_groups, min(max_groups, count))
    count = min(count, len(unique) - min_train_groups)
    if count <= 0:
        return GroupValidationSplit(all_train, no_validation, ())

    rng = np.random.default_rng(int(seed))
    selected = tuple(rng.choice(unique, size=count, replace=False).tolist())
    validation_mask = np.isin(group_ids, selected)
    return GroupValidationSplit(~validation_mask, validation_mask, selected)


def subject_disjoint_audit_split(
    subject_ids: np.ndarray,
    *,
    eligible_mask: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    n_subjects: int = 4,
    min_optimization_subjects: int = 4,
    seed: int = 0,
) -> SubjectAuditSplit:
    """Hold out untouched audit subjects from an existing optimization pool."""

    subject_ids = np.asarray(subject_ids)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)
    if subject_ids.ndim != 1 or eligible_mask.shape != subject_ids.shape:
        raise ValueError("subject_ids and eligible_mask must be aligned one-dimensional arrays.")
    if candidate_mask is None:
        candidate_mask = np.ones_like(eligible_mask)
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
        if candidate_mask.shape != subject_ids.shape:
            raise ValueError("candidate_mask must align with subject_ids.")
    if n_subjects < 0 or min_optimization_subjects < 1:
        raise ValueError("Invalid audit subject counts.")
    candidates = np.unique(subject_ids[eligible_mask & candidate_mask])
    count = min(int(n_subjects), max(0, len(candidates) - int(min_optimization_subjects)))
    if count < 2:
        return SubjectAuditSplit(eligible_mask.copy(), np.zeros_like(eligible_mask), ())
    rng = np.random.default_rng(int(seed) + 9173)
    selected = tuple(rng.choice(candidates, size=count, replace=False).tolist())
    audit_mask = eligible_mask & np.isin(subject_ids, selected)
    optimization_mask = eligible_mask & ~audit_mask
    return SubjectAuditSplit(optimization_mask, audit_mask, selected)
