from __future__ import annotations

import pytest
import torch

from transfer.alignment import (
    rbf_mmd_loss,
)


def test_rbf_mmd_is_near_zero_for_identical_samples_and_positive_for_shift() -> None:
    source = torch.linspace(-1.0, 1.0, 12).reshape(-1, 1)
    same, same_record = rbf_mmd_loss(source, source.clone(), bandwidth=0.5)
    shifted, shifted_record = rbf_mmd_loss(source, source + 3.0, bandwidth=0.5)

    # The unbiased finite-sample estimator may be slightly negative even when
    # the two empirical samples are identical; it must still be below a clear
    # distribution shift.
    assert float(same) < float(shifted)
    assert float(shifted) > 0.1
    assert same_record["bandwidth"] == 0.5
    assert shifted_record["source_count"] == shifted_record["target_count"] == 12.0


def test_rbf_mmd_records_negative_unbiased_estimate_without_clamping() -> None:
    source = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    target = source.clone()

    loss, record = rbf_mmd_loss(source, target, bandwidth=1.0)

    assert float(loss.detach()) < 0.0
    assert record["negative_mmd"] is True
    assert record["mmd_squared"] == pytest.approx(float(loss.detach()))


def test_rbf_mmd_rejects_singleton_samples() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        rbf_mmd_loss(torch.zeros((1, 2)), torch.zeros((2, 2)))
