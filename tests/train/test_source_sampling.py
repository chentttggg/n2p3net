from __future__ import annotations

import json

import numpy as np
import pytest

from train.source_sampling import (
    PDSSelection,
    array_digest,
    load_pds_selection_record,
    pds_selection_from_record,
    select_dense_pds_rows,
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

    first, second = (
        select_pds_rows(
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
        for _ in range(2)
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

    log_quality = -np.sum((source - target.mean(axis=0)[None, :]) ** 2, axis=1) / 2.0
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
    record = selection.record()
    assert record["sampler_kind"] == "target_similarity_diverse"
    assert record["pds_radius"] is None
    assert record["pds_mingling_mixture"] is None
    assert record["pds_attempts"] == 0
    assert record["pds_fallback_count"] == 0

    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection.record()), encoding="utf-8")

    loaded, raw = load_pds_selection_record(path)

    np.testing.assert_array_equal(loaded.selected_indices, selection.selected_indices)
    assert loaded.record() == selection.record()
    assert raw["selected_digest"] == array_digest(selection.selected_indices)
    assert loaded.pds_radius is None
    assert loaded.pds_mingling_mixture is None
    assert loaded.pds_attempts == 0
    assert loaded.pds_fallback_count == 0

    record = selection.record()
    replacement = next(
        index for index in range(len(source)) if index not in selection.selected_indices
    )
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


def test_dense_pds_hard_disk_accepts_beyond_and_rejects_within_radius() -> None:
    source = np.asarray([[0.0], [0.4], [1.0]])
    target = np.asarray([[2.0]])
    labels = np.asarray([0, 0, 1])

    accepted = select_dense_pds_rows(
        source,
        target,
        n_select=2,
        selection_scope="target_calibration",
        seed=0,
        labels=labels,
        pds_radius=0.5,
        pds_mingling_mixture=(0.5, 0.5),
    )
    assert accepted.selected_indices.tolist() == [2, 0]
    assert accepted.pds_attempts == 2
    assert accepted.pds_fallback_count == 0
    assert float(np.linalg.norm(source[2] - source[0])) > 0.5

    with_fallback = select_dense_pds_rows(
        source,
        target,
        n_select=3,
        selection_scope="target_calibration",
        seed=0,
        labels=labels,
        pds_radius=0.5,
        pds_mingling_mixture=(0.5, 0.5),
        pds_max_attempts=5,
    )
    assert with_fallback.selected_indices.tolist() == [2, 0, 1]
    assert with_fallback.pds_attempts == 5
    assert with_fallback.pds_fallback_count == 1
    assert len(np.unique(with_fallback.selected_indices)) == 3
    assert with_fallback.sampler_kind == "dense_pds"
    assert with_fallback.bandwidth == 1.0
    assert with_fallback.bandwidth_estimator == "explicit"
    assert with_fallback.target_temperature == 1.0
    assert with_fallback.diversity_weight == 0.0

    tiny_budget = select_dense_pds_rows(
        source,
        target,
        n_select=3,
        selection_scope="target_calibration",
        seed=0,
        labels=labels,
        pds_radius=0.5,
        pds_mingling_mixture=(0.5, 0.5),
        pds_max_attempts=1,
    )
    assert tiny_budget.selected_indices.tolist() == [2, 0, 1]
    assert tiny_budget.pds_attempts == 2
    assert tiny_budget.pds_fallback_count == 2


def test_dense_pds_fallback_breaks_ties_with_rng_permutation() -> None:
    source = np.asarray([[0.0], [1.0], [1.0]])
    target = np.asarray([[0.0]])
    selection = select_dense_pds_rows(
        source,
        target,
        n_select=3,
        selection_scope="target_calibration",
        seed=0,
        labels=np.asarray([0, 1, 2]),
        pds_radius=0.5,
        pds_mingling_mixture=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        pds_max_attempts=1,
    )

    assert selection.selected_indices.tolist() == [1, 0, 2]
    assert selection.pds_fallback_count == 2
    assert selection.pds_attempts == 2


@pytest.mark.parametrize(
    "coordinates,labels,radius,mixture",
    [
        ([1e8, 1e8 + 1e-3, 0.0], [0, 0, 1], 5e-4, (1.0 - 1e-12, 1e-12)),
        ([0.0, 1e-200], [0, 1], 1e-201, (0.5, 0.5)),
    ],
    ids=["cancellation", "underflow"],
)
def test_dense_pds_extreme_coordinates_do_not_cancel_or_underflow(
    coordinates, labels, radius, mixture
):
    source = np.asarray(coordinates)[:, None]
    selection = select_dense_pds_rows(
        source,
        np.asarray([[0.0]]),
        n_select=2,
        selection_scope="target_calibration",
        seed=0,
        labels=np.asarray(labels),
        pds_radius=radius,
        pds_mingling_mixture=mixture,
    )
    assert selection.selected_indices.tolist() == [1, 0]
    assert selection.pds_attempts == 2
    assert selection.pds_fallback_count == 0
    assert abs(source[1, 0] - source[0, 0]) > radius


def test_dense_pds_large_close_coordinate_gpu_path_matches_stable_cpu(monkeypatch):
    from train import source_sampling

    source = np.concatenate((1e8 + np.arange(30) * 1e-3, [0.0]))[:, None]
    target = np.zeros((1, 1))
    options = dict(
        n_select=28,
        selection_scope="target_calibration",
        seed=0,
        labels=np.concatenate((np.zeros(30, dtype=int), [1])),
        pds_radius=5e-4,
        pds_mingling_mixture=(1.0 - 1e-12, 1e-12),
        pds_max_attempts=100,
    )
    with monkeypatch.context() as cpu:
        cpu.setattr(source_sampling, "_torch", None)
        expected = select_dense_pds_rows(source, target, **options)
    actual = select_dense_pds_rows(source, target, **options)
    np.testing.assert_array_equal(actual.selected_indices, expected.selected_indices)
    assert actual.pds_attempts == expected.pds_attempts == 28
    assert actual.pds_fallback_count == expected.pds_fallback_count == 0


def test_dense_pds_default_mixture_is_empirical_frequency() -> None:
    source = np.zeros((5, 2))
    target = np.zeros((1, 2))
    labels = np.asarray([0, 0, 0, 1, 2])

    selection = select_dense_pds_rows(
        source,
        target,
        n_select=1,
        selection_scope="target_calibration",
        seed=11,
        labels=labels,
        pds_radius=0.1,
    )

    assert selection.pds_mingling_mixture is not None
    np.testing.assert_allclose(
        selection.pds_mingling_mixture,
        np.asarray([0.6, 0.2, 0.2]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_dense_pds_custom_mixture_is_honored_across_deterministic_seeds() -> None:
    source = np.zeros((200, 2))
    target = np.zeros((1, 2))
    labels = np.asarray([0] * 50 + [1] * 150)

    selected_class_zero = 0
    for seed in range(1200):
        selection = select_dense_pds_rows(
            source,
            target,
            n_select=1,
            selection_scope="target_calibration",
            seed=seed,
            labels=labels,
            pds_radius=0.5,
            pds_mingling_mixture=(0.8, 0.2),
        )
        selected_class_zero += int(labels[selection.selected_indices[0]] == 0)

    assert 850 <= selected_class_zero <= 1050


def test_dense_pds_target_features_only_bind_provenance() -> None:
    source = np.asarray([[0.0], [0.5], [1.0], [2.0]])
    labels = np.asarray([0, 0, 1, 1])
    first, second = (
        select_dense_pds_rows(
            source,
            np.asarray([[target_location]]),
            n_select=3,
            selection_scope="target_calibration",
            seed=4,
            labels=labels,
            pds_radius=0.25,
            pds_mingling_mixture=(0.5, 0.5),
            pds_max_attempts=100,
        )
        for target_location in (10.0, -10.0)
    )
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)
    assert first.target_digest != second.target_digest


def test_dense_pds_selection_record_round_trip_and_legacy(tmp_path) -> None:
    source = np.asarray([[0.0], [0.4], [1.0]])
    target = np.asarray([[2.0]])
    selection = select_dense_pds_rows(
        source,
        target,
        n_select=3,
        selection_scope="target_calibration",
        seed=0,
        labels=np.asarray([0, 0, 1]),
        pds_radius=0.5,
        pds_mingling_mixture=(0.5, 0.5),
        pds_max_attempts=5,
    )
    record = selection.record()
    assert record["sampler_kind"] == "dense_pds"
    assert record["pds_radius"] == 0.5
    assert record["pds_mingling_mixture"] == [0.5, 0.5]
    assert record["pds_attempts"] == 5
    assert record["pds_fallback_count"] == 1

    path = tmp_path / "dense_pds_selection.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    loaded, raw = load_pds_selection_record(path)

    np.testing.assert_array_equal(loaded.selected_indices, selection.selected_indices)
    assert loaded.record() == selection.record()
    assert raw["pds_radius"] == 0.5

    legacy_record = dict(record)
    legacy_record["schema"] = "n2p3_pds_source_selection/1"
    legacy = pds_selection_from_record(legacy_record)
    assert legacy.sampler_kind == "target_similarity_diverse"
    assert legacy.pds_radius is None
    assert legacy.pds_mingling_mixture is None
    assert legacy.pds_attempts == 0
    assert legacy.pds_fallback_count == 0

    malformed = dict(record)
    malformed["pds_attempts"] = float("inf")
    with pytest.raises(ValueError, match="invalid PDS source-selection dense fields"):
        pds_selection_from_record(malformed)
    malformed["pds_attempts"] = 1.5
    with pytest.raises(ValueError, match="invalid PDS source-selection dense fields"):
        pds_selection_from_record(malformed)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"labels": None}, "labels"),
        ({"labels": np.asarray([0, 1])}, "align"),
        ({"labels": np.asarray(["a", "b", "c", "d"])}, "integer"),
        ({"labels": np.zeros(4, dtype=int)}, "at least two"),
        ({"pds_radius": 0.0}, "pds_radius"),
        ({"pds_radius": np.inf}, "pds_radius"),
        ({"pds_max_attempts": 0}, "pds_max_attempts"),
        ({"pds_mingling_mixture": (0.5, 0.25, 0.25)}, "align"),
        ({"pds_mingling_mixture": (0.0, 1.0)}, "positive"),
        ({"pds_mingling_mixture": (0.5, 0.4)}, "sum"),
    ],
)
def test_dense_pds_rejects_invalid_sampling_inputs(changes, match) -> None:
    kwargs = {"labels": np.asarray([0, 0, 1, 1]), **changes}
    with pytest.raises(ValueError, match=match):
        select_dense_pds_rows(
            np.zeros((4, 2)),
            np.zeros((2, 2)),
            n_select=2,
            selection_scope="target_calibration",
            **kwargs,
        )


@pytest.mark.parametrize(
    "changes,match",
    [
        (
            {"sampler_kind": "dense_pds", "pds_mingling_mixture": (0.5, 0.5), "pds_attempts": 1},
            "pds_radius",
        ),
        (
            {
                "sampler_kind": "dense_pds",
                "pds_radius": 1.0,
                "pds_mingling_mixture": [0.5, 0.5],
                "pds_attempts": 1,
            },
            "mixture",
        ),
        (
            {
                "sampler_kind": "dense_pds",
                "pds_radius": 1.0,
                "pds_mingling_mixture": (0.5, 0.5),
                "pds_attempts": 0,
            },
            "pds_attempts",
        ),
        (
            {
                "sampler_kind": "dense_pds",
                "pds_radius": 1.0,
                "pds_mingling_mixture": (0.5, 0.5),
                "pds_attempts": 1,
                "pds_fallback_count": 2,
            },
            "fallback",
        ),
        ({"pds_radius": 1.0}, "radius"),
        ({"pds_attempts": 1}, "attempt"),
    ],
)
def test_dense_pds_rejects_invalid_selection_records(changes, match) -> None:
    selected = np.asarray([0, 2])
    with pytest.raises(ValueError, match=match):
        PDSSelection(
            selected_indices=selected,
            source_count=4,
            target_count=2,
            selection_scope="target_calibration",
            seed=0,
            target_temperature=1.0,
            diversity_weight=1.0,
            bandwidth=1.0,
            source_digest="1" * 64,
            target_digest="2" * 64,
            selected_digest=array_digest(selected),
            **changes,
        )
