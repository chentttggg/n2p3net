from __future__ import annotations

import numpy as np
import pytest

from train.source_sampling import select_pds_rows


def test_pds_selection_is_deterministic_and_respects_label_group_quotas() -> None:
    source = np.asarray(
        [
            [-3.0, 0.0],
            [-2.5, 0.1],
            [-0.5, 0.0],
            [0.0, 0.2],
            [2.5, 0.0],
            [3.0, -0.1],
        ]
    )
    target = np.asarray([[2.8, 0.0], [2.9, 0.1]])
    labels = np.asarray([0, 0, 1, 1, 1, 1])
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])

    first = select_pds_rows(
        source,
        target,
        n_select=4,
        selection_scope="target_calibration",
        seed=17,
        labels=labels,
        groups=groups,
        min_per_label=1,
        min_per_group=1,
    )
    second = select_pds_rows(
        source,
        target,
        n_select=4,
        selection_scope="target_calibration",
        seed=17,
        labels=labels,
        groups=groups,
        min_per_label=1,
        min_per_group=1,
    )

    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)
    assert set(groups[first.selected_indices]) == {"a", "b", "c"}
    assert set(labels[first.selected_indices]) == {0, 1}
    assert first.record()["selected_count"] == 4


def test_pds_target_similarity_prefers_nearby_rows_without_duplicates() -> None:
    source = np.asarray([[-10.0], [-1.0], [0.0], [1.0], [10.0]])
    target = np.asarray([[0.1], [0.2]])

    selection = select_pds_rows(
        source,
        target,
        n_select=2,
        selection_scope="target_unlabelled_buffer",
        seed=0,
        target_temperature=1.0,
        diversity_weight=0.0,
        bandwidth=1.0,
    )

    assert set(selection.selected_indices.tolist()) == {2, 3}
    assert len(np.unique(selection.selected_indices)) == 2


def test_pds_rejects_test_scope_and_unsatisfied_quota() -> None:
    source = np.zeros((4, 2))
    target = np.zeros((2, 2))
    with pytest.raises(ValueError, match="target test scope is forbidden"):
        select_pds_rows(
            source,
            target,
            n_select=2,
            selection_scope="target_test",
        )
    with pytest.raises(ValueError, match="more rows than n_select"):
        select_pds_rows(
            source,
            target,
            n_select=1,
            selection_scope="target_calibration",
            labels=np.asarray([0, 0, 1, 1]),
            min_per_label=1,
        )
