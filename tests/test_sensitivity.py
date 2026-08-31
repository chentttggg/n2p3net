from __future__ import annotations

import numpy as np
import pytest

from experiments.run_n2p3_sensitivity import (
    DELTAS,
    _anchor_subjects_for_folds,
    _subject_sort_key,
    architecture_for_sample_rate,
    build_batch_retry_candidates,
    build_boundary_candidates,
    build_candidates,
    build_kernel_fine_candidates,
    build_kernel_range_candidates,
    build_lower_boundary_candidates,
    build_secondary_candidates,
)


def test_local_sensitivity_grid_has_one_shared_baseline_and_deduplicates_integers() -> None:
    candidates = build_candidates(base_batch_size=512)

    assert candidates[0].name == "baseline"
    assert candidates[0].relative_delta == 0.0
    assert len(candidates) == 31
    assert len({candidate.name for candidate in candidates}) == len(candidates)
    assert set(DELTAS) == {-0.15, -0.05, 0.05, 0.15}


def test_local_sensitivity_grid_maps_continuous_and_discrete_axes() -> None:
    candidates = build_candidates(base_batch_size=512)
    by_name = {candidate.name: candidate for candidate in candidates}

    assert by_name["lr_m15"].deep_overrides["lr"] == pytest.approx(0.00085)
    assert by_name["lr_p15"].deep_overrides["lr"] == pytest.approx(0.00115)
    assert by_name["batch_size_m15"].batch_size == 435
    assert by_name["batch_size_p15"].batch_size == 589
    assert by_name["temporal_kernel_size_m15"].architecture_overrides == {
        "temporal_kernel_size": 29
    }
    assert by_name["temporal_kernel_size_p15"].architecture_overrides == {
        "temporal_kernel_size": 41
    }
    assert "temporal_filters_m05" not in by_name
    assert "temporal_filters_p05" not in by_name
    assert by_name["temporal_filters_m15"].architecture_overrides == {
        "temporal_filters": 7
    }
    assert by_name["temporal_filters_p15"].architecture_overrides == {
        "temporal_filters": 9
    }


def test_256_hz_architecture_preserves_physical_kernel_spans() -> None:
    architecture = architecture_for_sample_rate(256.0)

    assert architecture.temporal_kernel_size == 69
    assert architecture.mst_kernel_sizes == (9, 33)


def test_subject_subset_uses_natural_numeric_order() -> None:
    subjects = ["1", "10", "2", "20", "3"]

    assert sorted(subjects, key=_subject_sort_key) == ["1", "2", "3", "10", "20"]


def test_anchor_subjects_are_read_from_fold_masks_not_a_second_sort() -> None:
    subjects = np.asarray(["1", "10", "2", "1", "10", "2"])
    folds = [
        (subjects != held_out, subjects == held_out)
        for held_out in np.unique(subjects)
    ]

    assert _anchor_subjects_for_folds(subjects, folds) == ["1", "10", "2"]


def test_boundary_grid_extends_both_temporal_kernel_directions() -> None:
    candidates = build_boundary_candidates(base_batch_size=512)
    by_name = {candidate.name: candidate for candidate in candidates}

    assert len(candidates) == 3
    assert by_name["temporal_kernel_size_45"].architecture_overrides == {
        "temporal_kernel_size": 45
    }
    assert by_name["temporal_kernel_size_85"].architecture_overrides == {
        "temporal_kernel_size": 85
    }


def test_lower_boundary_grid_uses_only_three_wide_probes() -> None:
    candidates = build_lower_boundary_candidates(base_batch_size=512)

    assert [candidate.architecture_overrides["temporal_kernel_size"] for candidate in candidates] == [
        45,
        35,
        25,
    ]


def test_kernel_range_grid_uses_requested_35_to_85_step_10() -> None:
    candidates = build_kernel_range_candidates(base_batch_size=512)

    assert [candidate.architecture_overrides["temporal_kernel_size"] for candidate in candidates] == [
        35,
        45,
        55,
        65,
        75,
        85,
    ]


def test_kernel_fine_grid_keeps_every_kernel_odd() -> None:
    candidates = build_kernel_fine_candidates(base_batch_size=512)
    kernels = [candidate.architecture_overrides["temporal_kernel_size"] for candidate in candidates]

    assert kernels == [25, 29, 33, 35, 37, 41, 45]
    assert all(kernel % 2 == 1 for kernel in kernels)


def test_secondary_grid_fixes_the_winning_kernel_and_excludes_kernel_axis() -> None:
    candidates = build_secondary_candidates(base_batch_size=512)

    assert len(candidates) == 27
    assert all(candidate.axis != "temporal_kernel_size" for candidate in candidates)
    assert all(
        candidate.architecture_overrides["temporal_kernel_size"] == 35
        for candidate in candidates
    )


def test_batch_retry_grid_contains_only_baseline_and_real_batch_variants() -> None:
    candidates = build_batch_retry_candidates(base_batch_size=512)

    assert [candidate.batch_size for candidate in candidates] == [512, 435, 486, 538, 589]
    assert all(
        candidate.architecture_overrides == {"temporal_kernel_size": 35}
        for candidate in candidates
    )
