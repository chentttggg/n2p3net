from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.run_cross_dataset_transfer import planned_splits
from transfer.binary import (
    STANDARD_CALIBRATION_TRIALS,
    binary_time_split,
    calibration_arms,
    resolve_calibration_splits,
    validate_calibration_budgets,
)


def builder_for(n):
    onset = np.arange(n, dtype=float)
    return lambda k, maximum: binary_time_split(
        onset, onset + 1.2, calibration_trials=k,
        maximum_calibration_trials=maximum, tmin_s=-0.2,
    )


def test_standard_budgets_and_zero_calibration_arm_semantics():
    assert STANDARD_CALIBRATION_TRIALS == (0, 200, 800, 2000)
    arms = ("zero_shot", "full_fine", "target_only", "shrinkage_lda")
    assert calibration_arms(arms, 0, STANDARD_CALIBRATION_TRIALS) == ("zero_shot",)
    assert calibration_arms(arms, 200, STANDARD_CALIBRATION_TRIALS) == arms[1:]


def test_insufficient_2000_uses_eighty_percent_and_shared_test_for_zero():
    splits, missing = resolve_calibration_splits(
        STANDARD_CALIBRATION_TRIALS, builder_for(2100), minimum_test_trials=200, available_trials=2100,
    )
    assert set(splits) == {0, 200, 800, 1680}
    assert 2000 in missing
    assert len(splits[0].calibration_rows) == 0
    for split in splits.values():
        np.testing.assert_array_equal(split.test_rows, splits[1680].test_rows)
        assert not np.intersect1d(split.calibration_rows, split.test_rows).size
    assert splits[1680].test_rows[0] == 1681
    assert "effective=1680" in missing[2000]


def test_full_budget_is_included_only_with_sufficient_purged_test():
    splits, missing = resolve_calibration_splits(
        STANDARD_CALIBRATION_TRIALS, builder_for(2600), minimum_test_trials=200, available_trials=2600,
    )
    assert set(splits) == set(STANDARD_CALIBRATION_TRIALS) and missing == {}
    for split in splits.values():
        np.testing.assert_array_equal(split.test_rows, splits[2000].test_rows)


def test_short_recording_uses_80_percent_without_relabeling_as_2000():
    splits, missing = resolve_calibration_splits(
        STANDARD_CALIBRATION_TRIALS, builder_for(150), minimum_test_trials=200, available_trials=150,
    )
    assert set(splits) == {0, 120} and set(missing) == {200, 800, 2000}
    assert splits[0].calibration_rows.size == 0 and len(splits[120].calibration_rows) == 120
    assert len(splits[0].test_rows) == 29
    assert 2000 not in splits


def test_zero_only_never_fabricates_positive_calibration():
    splits, missing = resolve_calibration_splits(
        (0,), builder_for(250), minimum_test_trials=200, available_trials=250,
    )
    assert set(splits) == {0} and missing == {}
    assert splits[0].calibration_rows.size == 0 and len(splits[0].test_rows) == 250


def test_single_recording_reserves_about_twenty_percent_before_purge():
    splits, adjustments = resolve_calibration_splits(
        STANDARD_CALIBRATION_TRIALS, builder_for(2400), minimum_test_trials=200, available_trials=2400,
    )
    assert max(splits) == 1920 and 2000 in adjustments
    assert len(splits[1920].test_rows) == 479


def test_eighty_percent_fallback_still_rejects_reset_clocks():
    onset = np.tile(np.arange(50, dtype=float), 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_calibration_splits(
            STANDARD_CALIBRATION_TRIALS,
            lambda k, maximum: binary_time_split(onset, onset+1, calibration_trials=k, maximum_calibration_trials=maximum, tmin_s=-.2),
            minimum_test_trials=200, available_trials=100,
        )


def test_cross_dataset_recordings_share_original_later_test_and_log_missing_budget():
    n = 1800
    timeline = SimpleNamespace(
        evidence_indices=np.arange(n), onset_times_s=np.tile(np.arange(600), 3),
        evidence_available_times_s=np.tile(np.arange(600), 3) + 1.2,
        session_ids=np.repeat(["0", "1", "2"], 600), run_ids=np.repeat("0", n),
    )
    dataset = SimpleNamespace(subject_ids=np.repeat("P1", n), n_epochs=n, event_timeline=timeline)
    missing = {}
    rows, splits = planned_splits(dataset, "P1", {
        "kind": "last_session_holdout", "session_order": ["0", "1", "2"],
        "run_order": ["0"], "test_session": "2",
    }, STANDARD_CALIBRATION_TRIALS, unavailable=missing)
    assert set(splits) == {0, 200, 800, 960} and set(missing) == {2000}
    for split in splits.values():
        np.testing.assert_array_equal(rows[split.test_rows], np.arange(1200, 1800))


@pytest.mark.parametrize("budgets", [(), (-1, 200), (0, 200, 200), (200, 0), (0, True), (0, 200.0)])
def test_invalid_budget_declarations_are_rejected(budgets):
    with pytest.raises(ValueError, match="budgets"):
        validate_calibration_budgets(budgets)
