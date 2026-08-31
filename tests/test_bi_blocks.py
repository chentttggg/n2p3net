from __future__ import annotations

import numpy as np
import pytest

from experiments.prepare_bi2014a_blocks import balanced_subject_blocks


def test_balanced_subject_blocks_are_complete_equal_and_deterministic() -> None:
    subject_ids = np.concatenate(
        [np.repeat(f"s{index}", count) for index, count in enumerate(range(1, 9), 1)]
    )

    first = balanced_subject_blocks(subject_ids, n_blocks=4)
    second = balanced_subject_blocks(subject_ids[::-1], n_blocks=4)

    assert first == second
    assert all(len(block) == 2 for block in first)
    assert sorted(subject for block in first for subject in block) == [
        f"s{index}" for index in range(1, 9)
    ]


def test_balanced_subject_blocks_require_equal_capacity() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        balanced_subject_blocks(np.asarray(["s1", "s2", "s3"]), n_blocks=2)
