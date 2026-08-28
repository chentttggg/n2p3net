"""Chronological prefix/suffix splitting for the BI2014a candidate cache.

The split is defined over ``metadata.selection_id`` and
``metadata.repetition_index`` because the BI event timeline deliberately does
not claim a complete one-target-per-group candidate chain: each flash is
partial row/column evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.contract import assert_causal_p300_input_contract
from data.epochs import EpochDataset


@dataclass(frozen=True)
class BI2014APrefixSuffixSplit:
    prefix_mask: np.ndarray
    suffix_mask: np.ndarray
    suffix_repetition_indices: np.ndarray
    usable_selections: tuple[str, ...]


def bi2014a_prefix_suffix_split(
    dataset: EpochDataset,
    *,
    prefix_repetitions: int,
    test_repetitions: int,
) -> BI2014APrefixSuffixSplit:
    """Split BI selection groups by their chronological repetition index."""

    dataset.validate(require_labels=True)
    assert_causal_p300_input_contract(dataset.preprocessing)
    required = {
        "flash_code",
        "row_code",
        "col_code",
        "target_row",
        "target_col",
        "selection_id",
        "repetition_index",
    }
    missing = required - set(dataset.metadata.columns)
    if missing:
        raise ValueError(f"BI candidate metadata is missing columns {sorted(missing)}.")
    if prefix_repetitions < 1 or test_repetitions < 1:
        raise ValueError("prefix/test repetitions must be positive.")

    selection_ids = dataset.metadata["selection_id"].astype(str).to_numpy()
    repetitions = dataset.metadata["repetition_index"].to_numpy(dtype=np.int64)
    prefix = np.zeros(dataset.n_epochs, dtype=bool)
    suffix = np.zeros(dataset.n_epochs, dtype=bool)
    suffix_reps = np.full(dataset.n_epochs, -1, dtype=np.int64)
    usable: list[str] = []
    for selection in np.unique(selection_ids):
        rows = np.flatnonzero(selection_ids == selection)
        pre = rows[repetitions[rows] < prefix_repetitions]
        post = rows[
            (repetitions[rows] >= prefix_repetitions)
            & (repetitions[rows] < prefix_repetitions + test_repetitions)
        ]
        # BI repetitions contain exactly 12 flashes; require complete rows so
        # a partial repetition can never leak into either split.
        if len(pre) != 12 * prefix_repetitions or len(post) != 12 * test_repetitions:
            continue
        prefix[pre] = True
        suffix[post] = True
        suffix_reps[post] = repetitions[post] - prefix_repetitions
        usable.append(str(selection))
    if not usable:
        raise ValueError(
            "No BI selection has enough complete repetitions for "
            f"{prefix_repetitions}+{test_repetitions} chronological folds."
        )
    return BI2014APrefixSuffixSplit(
        prefix_mask=prefix,
        suffix_mask=suffix,
        suffix_repetition_indices=suffix_reps,
        usable_selections=tuple(usable),
    )
