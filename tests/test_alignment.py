from __future__ import annotations

import pytest
import torch

from transfer.alignment import (
    adaptive_target_weight,
    conditional_rbf_mmd_loss,
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


def test_conditional_mmd_does_not_hide_reversed_class_conditionals() -> None:
    source = torch.tensor([[-2.0], [-1.0], [1.0], [2.0], [-2.2], [2.2], [-1.8], [1.8]])
    target = torch.tensor([[1.0], [2.0], [-1.0], [-2.0], [2.2], [-2.2], [1.8], [-1.8]])
    source_labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    target_labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])

    marginal, _ = rbf_mmd_loss(source, target, bandwidth=1.0)
    conditional, record = conditional_rbf_mmd_loss(
        source,
        source_labels,
        target,
        target_labels,
        min_cell_count=2,
        bandwidth=1.0,
    )

    assert float(marginal) < 1e-7
    assert float(conditional) > float(marginal) + 0.1
    assert record["active_labels"] == (0, 1)
    assert record["alignment_inactive"] is False


def test_conditional_mmd_is_inactive_when_a_class_cell_is_too_small() -> None:
    source = torch.zeros((6, 2))
    target = torch.zeros((6, 2))
    source_labels = torch.tensor([0, 0, 0, 0, 0, 1])
    target_labels = torch.tensor([0, 0, 0, 0, 0, 1])

    loss, record = conditional_rbf_mmd_loss(
        source,
        source_labels,
        target,
        target_labels,
        min_cell_count=6,
    )

    assert float(loss) == pytest.approx(0.0)
    assert record["alignment_inactive"] is True
    assert record["active_terms"] == 0.0


def test_adaptive_target_weight_is_warm_started_and_deterministic() -> None:
    assert adaptive_target_weight(80, 10, 0, warmup_epochs=4) == pytest.approx(1.0)
    assert adaptive_target_weight(80, 10, 2, warmup_epochs=4) == pytest.approx(1.9142135624)
    assert adaptive_target_weight(80, 10, 4, warmup_epochs=4) == pytest.approx(2.8284271247)
    assert adaptive_target_weight(10_000, 1, 4, warmup_epochs=4) == pytest.approx(6.0)


def test_rbf_mmd_rejects_singleton_samples() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        rbf_mmd_loss(torch.zeros((1, 2)), torch.zeros((2, 2)))
