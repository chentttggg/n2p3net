"""Finite and absent-channel value contracts include later EEG row blocks."""
from __future__ import annotations

import numpy as np
import pytest

from tests.support.datasets import reference_dataset


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_epoch_finite_validation_rejects_nonfinite_after_two_complete_blocks(value):
    dataset = reference_dataset(np.zeros((5033, 3, 4), np.float32), reference="nose")
    dataset.X[4097, 1, 2] = value
    with pytest.raises(ValueError, match="must be finite"):
        dataset.validate()


def test_epoch_trial_mask_validation_checks_later_blocks():
    dataset = reference_dataset(np.zeros((5033, 3, 4), np.float32), reference="nose")
    dataset.trial_channel_mask = np.ones((5033, 3), bool)
    dataset.trial_channel_mask[4097, 1] = False
    dataset.X[4097, 1, 2] = np.float32(1e-7)
    with pytest.raises(ValueError, match="Trial-masked channels"):
        dataset.validate()


def test_epoch_static_mask_validation_checks_later_blocks():
    dataset = reference_dataset(np.zeros((5033, 3, 4), np.float32), reference="nose")
    dataset.channel_mask[1] = False
    dataset.X[4097, 1, 2] = np.float32(1e-7)
    with pytest.raises(ValueError, match="Absent channels"):
        dataset.validate()


def test_epoch_nonfinite_values_remain_invalid_inside_later_masked_channels():
    dataset = reference_dataset(np.zeros((5033, 3, 4), np.float32), reference="nose")
    dataset.trial_channel_mask = np.ones((5033, 3), bool)
    dataset.trial_channel_mask[4097, 1] = False
    dataset.X[4097, 1, 2] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        dataset.validate()
