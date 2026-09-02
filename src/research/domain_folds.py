"""Leakage-safe outer folds for cross-domain EEG experiments.

The fold constructors operate on already prepared, explicitly annotated rows.
They do not infer domains from dataset names or subject identifiers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


DOMAIN_LODO_PROTOCOL = "domain_lodo"
DOMAIN_CEILING_PROTOCOL = "domain_ceiling"


@dataclass(frozen=True)
class DomainFold:
    """One auditable outer fold and its domain/subject scope."""

    fold_id: int
    protocol: str
    held_out_domain: str
    held_out_subjects: tuple[str, ...]
    train_domains: tuple[str, ...]
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    train: np.ndarray
    test: np.ndarray

    def record(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "protocol": self.protocol,
            "held_out_domain": self.held_out_domain,
            "held_out_subjects": list(self.held_out_subjects),
            "train_domains": list(self.train_domains),
            "train_subjects": list(self.train_subjects),
            "test_subjects": list(self.test_subjects),
            "n_train_rows": int(self.train.sum()),
            "n_test_rows": int(self.test.sum()),
        }


def _aligned_axes(
    subject_ids: Sequence[object],
    domain_ids: Sequence[object],
) -> tuple[np.ndarray, np.ndarray]:
    subjects = np.asarray(subject_ids).astype(str)
    domains = np.asarray(domain_ids).astype(str)
    if subjects.ndim != 1 or domains.shape != subjects.shape:
        raise ValueError("subject_ids and domain_ids must be aligned one-dimensional arrays.")
    if not len(subjects):
        raise ValueError("domain fold construction requires non-empty rows.")
    if np.any(np.char.strip(subjects) == "") or np.any(np.char.strip(domains) == ""):
        raise ValueError("subject_ids and domain_ids must contain non-empty values.")
    pairs = np.column_stack((subjects, domains))
    for subject in np.unique(subjects):
        subject_domains = np.unique(domains[subjects == subject])
        if len(subject_domains) != 1:
            raise ValueError(
                f"subject {subject!r} occurs in multiple domains; "
                "domain folds require one domain per subject."
            )
    del pairs
    return subjects, domains


def _requested_domains(domains: np.ndarray, requested: Sequence[object] | None) -> tuple[str, ...]:
    available = tuple(sorted(np.unique(domains).tolist()))
    if requested is None:
        selected = available
    else:
        selected = tuple(str(value).strip() for value in requested)
        if not selected or any(not value for value in selected):
            raise ValueError("requested domains must contain non-empty values.")
        if len(set(selected)) != len(selected):
            raise ValueError("requested domains must be unique.")
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(f"requested domains are absent from rows: {unknown}.")
    return selected


def domain_lodo_folds(
    subject_ids: Sequence[object],
    domain_ids: Sequence[object],
    *,
    domains: Sequence[object] | None = None,
) -> list[DomainFold]:
    """Hold out each complete domain while training on every other domain."""

    subjects, domains_array = _aligned_axes(subject_ids, domain_ids)
    selected = _requested_domains(domains_array, domains)
    if len(selected) < 2:
        raise ValueError("domain LODO requires at least two domains.")
    folds: list[DomainFold] = []
    for fold_id, held_out in enumerate(selected):
        test = domains_array == held_out
        train = np.isin(domains_array, [domain for domain in selected if domain != held_out])
        if not train.any() or not test.any():
            raise ValueError(f"domain LODO fold {held_out!r} has empty train or test rows.")
        train_domains = tuple(sorted(np.unique(domains_array[train]).tolist()))
        train_subjects = tuple(sorted(np.unique(subjects[train]).tolist()))
        test_subjects = tuple(sorted(np.unique(subjects[test]).tolist()))
        folds.append(
            DomainFold(
                fold_id=fold_id,
                protocol=DOMAIN_LODO_PROTOCOL,
                held_out_domain=held_out,
                held_out_subjects=test_subjects,
                train_domains=train_domains,
                train_subjects=train_subjects,
                test_subjects=test_subjects,
                train=train,
                test=test,
            )
        )
    _validate_domain_coverage(folds, subjects, domains_array, selected, protocol=DOMAIN_LODO_PROTOCOL)
    return folds


def domain_ceiling_folds(
    subject_ids: Sequence[object],
    domain_ids: Sequence[object],
    *,
    domains: Sequence[object] | None = None,
) -> list[DomainFold]:
    """Hold out each subject within its own domain for a same-domain ceiling."""

    subjects, domains_array = _aligned_axes(subject_ids, domain_ids)
    selected = _requested_domains(domains_array, domains)
    folds: list[DomainFold] = []
    fold_id = 0
    for domain in selected:
        domain_subjects = tuple(sorted(np.unique(subjects[domains_array == domain]).tolist()))
        if len(domain_subjects) < 2:
            raise ValueError(
                f"domain ceiling requires at least two subjects in domain {domain!r}."
            )
        for held_out_subject in domain_subjects:
            test = (domains_array == domain) & (subjects == held_out_subject)
            train = (domains_array == domain) & (subjects != held_out_subject)
            if not train.any() or not test.any():
                raise ValueError(
                    f"domain ceiling fold {domain!r}/{held_out_subject!r} has empty train or test rows."
                )
            train_subjects = tuple(sorted(np.unique(subjects[train]).tolist()))
            folds.append(
                DomainFold(
                    fold_id=fold_id,
                    protocol=DOMAIN_CEILING_PROTOCOL,
                    held_out_domain=domain,
                    held_out_subjects=(held_out_subject,),
                    train_domains=(domain,),
                    train_subjects=train_subjects,
                    test_subjects=(held_out_subject,),
                    train=train,
                    test=test,
                )
            )
            fold_id += 1
    _validate_domain_coverage(folds, subjects, domains_array, selected, protocol=DOMAIN_CEILING_PROTOCOL)
    return folds


def _validate_domain_coverage(
    folds: Sequence[DomainFold],
    subjects: np.ndarray,
    domains: np.ndarray,
    selected: Sequence[str],
    *,
    protocol: str,
) -> None:
    if not folds:
        raise ValueError("domain fold construction produced no folds.")
    n_rows = len(subjects)
    test_counts = np.zeros(n_rows, dtype=np.int64)
    for fold in folds:
        if fold.protocol != protocol:
            raise ValueError("domain fold protocol metadata is inconsistent.")
        if fold.train.shape != (n_rows,) or fold.test.shape != (n_rows,):
            raise ValueError("domain fold masks are misaligned with rows.")
        if not fold.train.any() or not fold.test.any() or bool((fold.train & fold.test).any()):
            raise ValueError("domain fold masks must be non-empty and disjoint.")
        test_counts += fold.test.astype(np.int64)
        if protocol == DOMAIN_LODO_PROTOCOL:
            if set(np.unique(domains[fold.train])) & set(np.unique(domains[fold.test])):
                raise ValueError("LODO fold leaks held-out domain into training rows.")
        else:
            if set(np.unique(domains[fold.train])) != set(np.unique(domains[fold.test])):
                raise ValueError("domain ceiling fold mixes domains between train and test.")
            if set(np.unique(subjects[fold.train])) & set(np.unique(subjects[fold.test])):
                raise ValueError("domain ceiling fold leaks held-out subject into training rows.")
    if np.any(test_counts != 1):
        raise ValueError(f"{protocol} test folds must cover selected rows exactly once.")
    selected_mask = np.isin(domains, selected)
    if not np.array_equal(test_counts.astype(bool), selected_mask):
        raise ValueError(f"{protocol} test folds do not match requested domain scope.")

