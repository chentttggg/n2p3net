"""Chronological within-subject prefix/suffix splitting and evaluation.

This module refuses zero-phase caches because continuous forward-backward
filtering smears future suffix samples into training prefix epochs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.contract import assert_causal_p300_input_contract
from data.epochs import EpochDataset


@dataclass(frozen=True)
class PrefixSuffixSplit:
    prefix_mask: np.ndarray
    suffix_mask: np.ndarray
    group_ids: np.ndarray
    candidate_codes: np.ndarray
    target_codes: np.ndarray
    truth_by_group: dict[str, int]
    repetition_indices: np.ndarray
    suffix_repetition_indices: np.ndarray
    usable_groups: tuple[str, ...]


def causal_prefix_suffix_split(
    dataset: EpochDataset,
    *,
    prefix_repetitions: int,
    test_repetitions: int,
    min_candidate_repetitions: int = 2,
) -> PrefixSuffixSplit:
    """Return chronological train/test masks for every complete selection group.

    The split is strict: a group is usable only when every candidate digit has
    at least ``prefix_repetitions + test_repetitions`` trials and exactly one
    target. Groups with incomplete candidate chains are skipped rather than
    silently repaired.
    """

    dataset.validate(require_labels=True)
    assert_causal_p300_input_contract(dataset.preprocessing)
    if prefix_repetitions < 1 or test_repetitions < 1:
        raise ValueError("prefix/test repetitions must be positive.")
    if min_candidate_repetitions < 1:
        raise ValueError("min_candidate_repetitions must be positive.")
    if not dataset.event_timeline.supports_full_candidate_chain:
        raise ValueError("dataset must expose a complete candidate/repetition chain.")
    encoded = dataset.event_timeline.encoded_candidate_selection(require_full_chain=True)
    group_ids = np.asarray(encoded.group_ids, dtype=str)
    candidates = np.asarray(encoded.candidate_codes, dtype=np.int64)
    targets = np.asarray(encoded.target_codes, dtype=np.int64)
    repetitions = np.asarray(encoded.repetition_indices, dtype=np.int64)
    if not (len(group_ids) == len(candidates) == len(targets) == len(repetitions) == dataset.n_epochs):
        raise ValueError("encoded candidate selection is not aligned with epoch rows.")

    prefix = np.zeros(dataset.n_epochs, dtype=bool)
    suffix = np.zeros(dataset.n_epochs, dtype=bool)
    suffix_reps = np.full(dataset.n_epochs, -1, dtype=np.int64)
    usable: list[str] = []
    for group in np.unique(group_ids):
        rows = np.flatnonzero(group_ids == group)
        group_targets = np.unique(targets[rows])
        if len(group_targets) != 1:
            continue
        counts = {int(code): int(np.count_nonzero(candidates[rows] == code)) for code in np.unique(candidates[rows])}
        if any(count < prefix_repetitions + test_repetitions for count in counts.values()):
            continue
        if any(
            count < min_candidate_repetitions
            for count in counts.values()
        ):
            continue
        pre = rows[repetitions[rows] < prefix_repetitions]
        post = rows[
            (repetitions[rows] >= prefix_repetitions)
            & (repetitions[rows] < prefix_repetitions + test_repetitions)
        ]
        # The strict inequality above assumes repetitions start at zero and are
        # contiguous for every candidate (validated by the event timeline).
        if len(pre) < prefix_repetitions * len(counts) or len(post) < test_repetitions * len(counts):
            continue
        prefix[pre] = True
        suffix[post] = True
        suffix_reps[post] = repetitions[rows][
            (repetitions[rows] >= prefix_repetitions)
            & (repetitions[rows] < prefix_repetitions + test_repetitions)
        ] - prefix_repetitions
        usable.append(str(group))

    if not usable:
        raise ValueError(
            "No selection group has enough complete candidate repetitions for "
            f"{prefix_repetitions}+{test_repetitions} chronological within-subject folds."
        )
    truth_by_group = {str(group): int(target) for group, target in encoded.truth_by_group.items()}
    return PrefixSuffixSplit(
        prefix_mask=prefix,
        suffix_mask=suffix,
        group_ids=group_ids,
        candidate_codes=candidates,
        target_codes=targets,
        truth_by_group=truth_by_group,
        repetition_indices=repetitions,
        suffix_repetition_indices=suffix_reps,
        usable_groups=tuple(usable),
    )
