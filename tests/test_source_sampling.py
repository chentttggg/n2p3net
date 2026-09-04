from __future__ import annotations

import json

import numpy as np
import pytest

from train.source_sampling import (
    array_digest,
    load_pds_selection_record,
    pds_selection_from_record,
    select_pds_rows,
)


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


def test_pds_incremental_repulsion_matches_full_kernel_reference() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(10, 4))
    target = rng.normal(size=(3, 4))
    bandwidth = 1.4
    seed = 5

    selection = select_pds_rows(
        source,
        target,
        n_select=4,
        selection_scope="target_calibration",
        seed=seed,
        diversity_weight=1.0,
        bandwidth=bandwidth,
    )

    log_quality = (
        -np.sum((source - target.mean(axis=0)[None, :]) ** 2, axis=1) / 2.0
    )
    squared = source[:, None, :] - source[None, :, :]
    kernel = np.exp(-np.sum(squared * squared, axis=2) / (2.0 * bandwidth**2))
    tie_break = np.random.default_rng(seed).uniform(0.0, 1e-9, len(source))
    selected: list[int] = []
    penalty = np.zeros(len(source), dtype=float)
    while len(selected) < 4:
        candidates = [index for index in range(len(source)) if index not in selected]
        score = log_quality[candidates] - penalty[candidates] + tie_break[candidates]
        chosen = candidates[int(np.argmax(score))]
        selected.append(chosen)
        penalty += kernel[chosen]
    np.testing.assert_array_equal(selection.selected_indices, np.asarray(selected))


def test_tsd_far_distance_quality_does_not_underflow_before_tie_break() -> None:
    source = np.asarray([[1e6, 0.0], [1e6 + 1e-3, 0.0]])
    target = np.asarray([[0.0, 0.0]])

    for seed in (0, 1):
        selection = select_pds_rows(
            source,
            target,
            n_select=1,
            selection_scope="target_calibration",
            seed=seed,
        )
        assert selection.selected_indices.tolist() == [0]



def test_pds_bandwidth_subsamples_large_pools_without_quadratic_kernel() -> None:
    rng = np.random.default_rng(13)
    source = rng.normal(size=(600, 3))
    target = rng.normal(size=(9, 3))

    selection = select_pds_rows(
        source,
        target,
        n_select=5,
        selection_scope="target_calibration",
        seed=0,
        diversity_weight=1.0,
    )

    assert selection.bandwidth_estimator == "subsampled_median_512"
    assert len(selection.selected_indices) == 5
    assert selection.record()["bandwidth_estimator"] == "subsampled_median_512"


def test_pds_selection_record_round_trip(tmp_path) -> None:
    source = np.arange(12, dtype=float).reshape(6, 2)
    target = np.asarray([[5.0, 0.0], [6.0, 0.0]])
    selection = select_pds_rows(
        source,
        target,
        n_select=3,
        selection_scope="target_calibration",
        seed=3,
    )
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection.record()), encoding="utf-8")

    loaded, raw = load_pds_selection_record(path)

    np.testing.assert_array_equal(loaded.selected_indices, selection.selected_indices)
    assert loaded.record() == selection.record()
    assert raw["selected_digest"] == array_digest(selection.selected_indices)


def test_pds_selection_record_rejects_edited_indices_and_source_digest(tmp_path) -> None:
    source = np.arange(12, dtype=float).reshape(6, 2)
    target = np.asarray([[5.0, 0.0], [6.0, 0.0]])
    selection = select_pds_rows(
        source,
        target,
        n_select=3,
        selection_scope="target_calibration",
        seed=3,
    )
    record = selection.record()
    replacement = next(index for index in range(len(source)) if index not in selection.selected_indices)
    record["selected_indices"][0] = replacement
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="selected_indices digest"):
        load_pds_selection_record(path)

    with pytest.raises(ValueError, match="source_digest"):
        pds_selection_from_record(
            selection.record(),
            expected_source_digest="0" * 64,
        )
