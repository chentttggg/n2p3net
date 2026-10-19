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
        onset,
        onset + 1.2,
        calibration_trials=k,
        maximum_calibration_trials=maximum,
        tmin_s=-0.2,
    )


def test_standard_budgets_and_zero_calibration_arm_semantics():
    assert STANDARD_CALIBRATION_TRIALS == (0, 200, 800, 2000)
    arms = ("zero_shot", "full_fine", "target_only", "shrinkage_lda")
    assert calibration_arms(arms, 0, STANDARD_CALIBRATION_TRIALS) == ("zero_shot",)
    assert calibration_arms(arms, 200, STANDARD_CALIBRATION_TRIALS) == arms[1:]


@pytest.mark.parametrize(
    "n,requested,actual,test_start,adjusted",
    [
        pytest.param(
            2100,
            STANDARD_CALIBRATION_TRIALS,
            (0, 200, 800, 1680),
            1681,
            (2000,),
            id="limited-high-budget",
        ),
        pytest.param(
            2600, STANDARD_CALIBRATION_TRIALS, (0, 200, 800, 2000), 2001, (), id="full-budget"
        ),
        pytest.param(
            150, STANDARD_CALIBRATION_TRIALS, (0, 120), 121, (200, 800, 2000), id="short-recording"
        ),
        pytest.param(250, (0,), (0,), 0, (), id="zero-only"),
        pytest.param(
            2400,
            STANDARD_CALIBRATION_TRIALS,
            (0, 200, 800, 1920),
            1921,
            (2000,),
            id="reserve-test-fifth",
        ),
    ],
)
def test_budget_fallback_preserves_actual_calibration_and_common_purged_test(
    n, requested, actual, test_start, adjusted
):
    splits, missing = resolve_calibration_splits(
        requested,
        builder_for(n),
        minimum_test_trials=200,
        available_trials=n,
    )
    assert set(splits) == set(actual)
    assert set(missing) == set(adjusted)
    for budget in actual:
        np.testing.assert_array_equal(splits[budget].calibration_rows, np.arange(budget))
        np.testing.assert_array_equal(splits[budget].test_rows, np.arange(test_start, n))
        assert not np.intersect1d(splits[budget].calibration_rows, splits[budget].test_rows).size
    for budget in adjusted:
        assert f"effective={actual[-1]}" in missing[budget]


def test_eighty_percent_fallback_still_rejects_reset_clocks():
    onset = np.tile(np.arange(50, dtype=float), 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_calibration_splits(
            STANDARD_CALIBRATION_TRIALS,
            lambda k, maximum: binary_time_split(
                onset,
                onset + 1,
                calibration_trials=k,
                maximum_calibration_trials=maximum,
                tmin_s=-0.2,
            ),
            minimum_test_trials=200,
            available_trials=100,
        )


def test_limited_test_size_is_explicit_even_when_no_calibration_budget_changes():
    splits, adjustments = resolve_calibration_splits(
        (0, 200), builder_for(300), minimum_test_trials=200, available_trials=300,
    )
    assert set(splits) == {0, 200} and adjustments == {}
    record = splits[200].test_budget.record()
    assert record["requested_minimum_test_trials"] == 200
    assert record["actual_test_trials"] == 99
    assert record["minimum_satisfied"] is False
    assert record["policy"] == "limited_data_80_percent"
    assert record["relaxation_reason"]
    assert splits[0].test_budget == splits[200].test_budget


def test_strict_test_minimum_reduces_calibration_without_relaxing_test_floor():
    splits, adjustments = resolve_calibration_splits(
        (0, 200), builder_for(300), minimum_test_trials=200, available_trials=300,
        test_size_policy="strict_minimum",
    )
    assert set(splits) == {0, 99}
    assert len(splits[99].test_rows) == 200
    assert splits[99].test_budget.minimum_satisfied
    assert 200 in adjustments


def test_short_fixed_test_does_not_shrink_an_available_independent_training_pool():
    from transfer.binary import BinaryTimeSplit

    def builder(k, _maximum):
        if k > 300:
            raise ValueError("calibration pool exhausted")
        return BinaryTimeSplit(np.arange(k), np.arange(300, 400), np.empty(0, dtype=int))

    splits, adjustments = resolve_calibration_splits(
        (0, 300), builder, minimum_test_trials=200, available_trials=300,
        separate_test_recordings=True,
    )
    assert set(splits) == {0, 300} and adjustments == {}
    assert splits[300].test_budget.actual_test_trials == 100
    assert not splits[300].test_budget.minimum_satisfied
    with pytest.raises(ValueError, match="minimum"):
        resolve_calibration_splits(
            (0, 300), builder, minimum_test_trials=200, available_trials=300,
            separate_test_recordings=True, test_size_policy="strict_minimum",
        )


def test_cross_dataset_recordings_share_original_later_test_and_log_missing_budget():
    n = 1800
    timeline = SimpleNamespace(
        evidence_indices=np.arange(n),
        onset_times_s=np.tile(np.arange(600), 3),
        evidence_available_times_s=np.tile(np.arange(600), 3) + 1.2,
        session_ids=np.repeat(["0", "1", "2"], 600),
        run_ids=np.repeat("0", n),
    )
    dataset = SimpleNamespace(subject_ids=np.repeat("P1", n), n_epochs=n, event_timeline=timeline)
    missing = {}
    rows, splits = planned_splits(
        dataset,
        "P1",
        {
            "kind": "last_session_holdout",
            "session_order": ["0", "1", "2"],
            "run_order": ["0"],
            "test_session": "2",
        },
        STANDARD_CALIBRATION_TRIALS,
        unavailable=missing,
    )
    assert set(splits) == {0, 200, 800, 960} and set(missing) == {2000}
    for split in splits.values():
        np.testing.assert_array_equal(rows[split.test_rows], np.arange(1200, 1800))


@pytest.mark.parametrize("budgets", [(), (-1, 200), (0, 200, 200), (200, 0), (0, True), (0, 200.0)])
def test_invalid_budget_declarations_are_rejected(budgets):
    with pytest.raises(ValueError, match="budgets"):
        validate_calibration_budgets(budgets)
