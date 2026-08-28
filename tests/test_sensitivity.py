from __future__ import annotations

import pytest

from experiments.run_n2p3_sensitivity import (
    DELTAS,
    _subject_sort_key,
    architecture_for_sample_rate,
    build_boundary_candidates,
    build_candidates,
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
        "temporal_kernel_size": 55
    }
    assert by_name["temporal_kernel_size_p15"].architecture_overrides == {
        "temporal_kernel_size": 75
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

    assert architecture.temporal_kernel_size == 129
    assert architecture.mst_kernel_sizes == (9, 33)


def test_subject_subset_uses_natural_numeric_order() -> None:
    subjects = ["1", "10", "2", "20", "3"]

    assert sorted(subjects, key=_subject_sort_key) == ["1", "2", "3", "10", "20"]


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
