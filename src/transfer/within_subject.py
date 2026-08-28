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
    candidate_vocab: tuple[int, ...]
    excluded_groups: dict[str, str]


def causal_prefix_suffix_split(
    dataset: EpochDataset,
    *,
    prefix_repetitions: int,
    test_repetitions: int,
    min_candidate_repetitions: int = 2,
) -> PrefixSuffixSplit:
    """Return chronological train/test masks for every complete selection group.

    The split is strict: every candidate must supply the requested prefix and
    suffix around one global raw-sample embargo, and the group must have exactly
    one target. Excluded groups and reasons are returned rather than silently
    repaired.
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

    timeline = dataset.event_timeline
    evidence_indices = np.asarray(timeline.evidence_indices, dtype=np.int64)
    available = evidence_indices >= 0
    onset_times_s = np.empty(dataset.n_epochs, dtype=np.float64)
    onset_times_s[evidence_indices[available]] = np.asarray(
        timeline.onset_times_s, dtype=np.float64
    )[available]
    evidence_available_times_s = np.empty(dataset.n_epochs, dtype=np.float64)
    evidence_available_times_s[evidence_indices[available]] = np.asarray(
        timeline.evidence_available_times_s, dtype=np.float64
    )[available]

    prefix = np.zeros(dataset.n_epochs, dtype=bool)
    suffix = np.zeros(dataset.n_epochs, dtype=bool)
    suffix_reps = np.full(dataset.n_epochs, -1, dtype=np.int64)
    usable: list[str] = []
    excluded: dict[str, str] = {}
    candidate_vocab = tuple(range(len(encoded.vocabulary)))
    for group in np.unique(group_ids):
        rows = np.flatnonzero(group_ids == group)
        group_targets = np.unique(targets[rows])
        if len(group_targets) != 1:
            excluded[str(group)] = "mixed_target"
            continue
        ordered_by_code: dict[int, np.ndarray] = {}
        prefix_by_code: dict[int, np.ndarray] = {}
        insufficient = False
        for code in candidate_vocab:
            code_rows = rows[candidates[rows] == code]
            order = np.argsort(onset_times_s[code_rows], kind="stable")
            code_rows = code_rows[order]
            if len(code_rows) < max(min_candidate_repetitions, prefix_repetitions):
                insufficient = True
                break
            ordered_by_code[int(code)] = code_rows
            prefix_by_code[int(code)] = code_rows[:prefix_repetitions]
        if insufficient:
            excluded[str(group)] = "insufficient_prefix_candidate_repetitions"
            continue

        pre = np.concatenate(list(prefix_by_code.values()))
        boundary_s = float(np.max(evidence_available_times_s[pre]))
        epoch_start_offset_s = float(dataset.preprocessing.tmin_ms) / 1000.0
        suffix_by_code: dict[int, np.ndarray] = {}
        for code, code_rows in ordered_by_code.items():
            later = code_rows[
                onset_times_s[code_rows] + epoch_start_offset_s > boundary_s
            ]
            if len(later) < test_repetitions:
                insufficient = True
                break
            suffix_by_code[code] = later[:test_repetitions]
        if insufficient:
            excluded[str(group)] = "insufficient_suffix_after_time_embargo"
            continue
        post = np.concatenate(list(suffix_by_code.values()))
        if not float(np.max(evidence_available_times_s[pre])) < float(
            np.min(onset_times_s[post] + epoch_start_offset_s)
        ):
            raise AssertionError("prefix/suffix construction failed to produce a global time boundary.")

        prefix[pre] = True
        suffix[post] = True
        for code_rows in suffix_by_code.values():
            suffix_reps[code_rows] = np.arange(test_repetitions, dtype=np.int64)
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
        candidate_vocab=candidate_vocab,
        excluded_groups=excluded,
    )
