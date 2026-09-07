from __future__ import annotations

import numpy as np
import pytest
import torch

from models.n2p3net import N2P3Net
from transfer.binary import (
    binary_recording_split,
    binary_time_split,
    scratch_like_trunk,
    target_shrinkage_scores,
)


def test_calibration_budgets_share_a_purged_suffix_without_candidate_labels():
    onsets = np.arange(30) * 0.25
    available = onsets + 1.2
    first = binary_time_split(
        onsets, available, calibration_trials=4, maximum_calibration_trials=8, tmin_s=-0.2
    )
    second = binary_time_split(
        onsets, available, calibration_trials=8, maximum_calibration_trials=8, tmin_s=-0.2
    )
    np.testing.assert_array_equal(first.test_rows, second.test_rows)
    assert first.test_rows[0] == 13
    assert np.min(onsets[first.test_rows] - 0.2) >= np.max(available[:8])
    assert len(first.calibration_rows) == 4
    assert len(first.purged_rows) == 5


def test_one_recording_cannot_silently_concatenate_reset_clocks():
    with pytest.raises(ValueError, match="strictly increasing"):
        binary_time_split(
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([1.0, 2.0, 1.0, 2.0]),
            calibration_trials=1,
            maximum_calibration_trials=2,
            tmin_s=-0.2,
        )


def test_explicit_recording_order_handles_reset_clocks_without_lexical_sorting():
    names = np.repeat(["run-10", "run-2", "run-20"], 6)
    onset = np.tile(np.arange(6, dtype=float), 3)
    kwargs = dict(
        recording_order=("run-2", "run-10", "run-20"),
        test_recordings=("run-20",),
        maximum_calibration_trials=8,
    )
    short = binary_recording_split(onset, onset + 1, names, calibration_trials=4, **kwargs)
    long = binary_recording_split(onset, onset + 1, names, calibration_trials=8, **kwargs)
    np.testing.assert_array_equal(short.calibration_rows, [6, 7, 8, 9])
    np.testing.assert_array_equal(long.calibration_rows, [6, 7, 8, 9, 10, 11, 0, 1])
    np.testing.assert_array_equal(short.test_rows, long.test_rows)
    assert not np.intersect1d(long.calibration_rows, long.test_rows).size


def test_recording_test_cannot_be_moved_before_calibration():
    names = np.repeat(["train", "test"], 6)
    onset = np.tile(np.arange(6, dtype=float), 2)
    with pytest.raises(ValueError, match="suffix"):
        binary_recording_split(
            onset,
            onset + 1,
            names,
            calibration_trials=4,
            maximum_calibration_trials=4,
            recording_order=("train", "test"),
            test_recordings=("train",),
        )


def test_availability_maximum_not_only_last_onset_controls_purge():
    onset = np.arange(10, dtype=float)
    available = onset + 1.0
    available[0] = 7.0
    split = binary_time_split(
        onset, available, calibration_trials=1, maximum_calibration_trials=2, tmin_s=0.0
    )
    assert split.test_rows[0] == 7


def test_scratch_initialization_does_not_inherit_source_weights_or_bn_stats():
    source = N2P3Net(5, n_times=179)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(42.0)
        source.st_temporal_norm.running_mean.fill_(13.0)
    first = scratch_like_trunk(source, seed=17)
    second = scratch_like_trunk(source, seed=17)
    assert first.architecture_record() == source.architecture_record()
    assert torch.equal(
        first.st_temporal_norm.running_mean, torch.zeros_like(first.st_temporal_norm.running_mean)
    )
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    assert not torch.equal(first.classifier.weight, source.classifier.weight)


def test_shrinkage_baseline_does_not_fit_test_statistics():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(40, 3, 128))
    labels = np.arange(40) % 2
    train[labels == 1] += 0.5
    test = rng.normal(size=(6, 3, 128))
    first = target_shrinkage_scores(train, labels, test, sfreq=128.0)
    changed = np.concatenate([test, np.ones((1, 3, 128)) * 1e6])
    second = target_shrinkage_scores(train, labels, changed, sfreq=128.0)
    np.testing.assert_allclose(first, second[:6], atol=1e-10)


@pytest.mark.parametrize("budget,maximum", [(0, 3), (4, 3), (3, 10)])
def test_invalid_calibration_does_not_shift_or_empty_test(budget, maximum):
    onset = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="calibration budgets"):
        binary_time_split(
            onset,
            onset + 1.0,
            calibration_trials=budget,
            maximum_calibration_trials=maximum,
            tmin_s=0.0,
        )
