from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from transfer.asmmd import (
    asmmd_joint_loss,
    mmd_schedule,
    rbf_mmd_logits_loss,
    target_weight,
)


def _brute_force_unbiased_mmd(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    bandwidth: float,
) -> torch.Tensor:
    """Direct Gretton et al. (2012) unbiased MMD^2, equation (7)."""

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        squared_distances = torch.cdist(left, right, p=2.0).square()
        return torch.exp(-squared_distances / (2.0 * bandwidth * bandwidth))

    if source.ndim == 1:
        source = source.unsqueeze(1)
    if target.ndim == 1:
        target = target.unsqueeze(1)

    n = source.shape[0]
    m = target.shape[0]
    k_ss = kernel(source, source)
    k_tt = kernel(target, target)
    k_st = kernel(source, target)

    source_term = (k_ss.sum() - k_ss.diagonal().sum()) / (n * (n - 1))
    target_term = (k_tt.sum() - k_tt.diagonal().sum()) / (m * (m - 1))
    cross_term = (2.0 / (n * m)) * k_st.sum()
    return source_term + target_term - cross_term


def test_target_weight_reference_values_and_clip() -> None:
    # Paper values: source/target counts 80/10 give w_t0 = sqrt(8).
    w_t0 = math.sqrt(8.0)
    assert target_weight(0, 80, 10, 40) == pytest.approx(1.0)
    assert target_weight(20, 80, 10, 40) == pytest.approx(1.0 + 0.5 * (w_t0 - 1.0))
    assert target_weight(40, 80, 10, 40) == pytest.approx(w_t0)
    assert target_weight(100, 80, 10, 40) == pytest.approx(w_t0)

    # Lower clip at 1.0 and upper clip at 6.0.
    assert target_weight(40, 1, 100, 40) == pytest.approx(1.0)
    assert target_weight(40, 10_000, 1, 40) == pytest.approx(6.0)

    # Custom finite bounds and positional minimum/maximum arguments.
    assert target_weight(40, 10, 1, 40, 2.0, 4.0) == pytest.approx(math.sqrt(10.0))


def test_target_weight_is_monotonic_in_epoch() -> None:
    weights = [target_weight(epoch, 80, 10, 40) for epoch in range(0, 101)]
    assert weights[0] == pytest.approx(1.0)
    assert all(weights[i + 1] >= weights[i] for i in range(len(weights) - 1))


def test_mmd_schedule_values() -> None:
    assert mmd_schedule(0, 10, 0.4) == pytest.approx(0.0)
    assert mmd_schedule(3, 10, 0.4) == pytest.approx(0.12)
    assert mmd_schedule(10, 10, 0.4) == pytest.approx(0.4)
    assert mmd_schedule(25, 10, 0.4) == pytest.approx(0.4)


def test_rbf_mmd_logits_loss_matches_bruteforce_unbiased_mmd() -> None:
    source = torch.tensor(
        [[-1.25, 0.40], [0.80, -0.35], [1.90, 1.10], [-0.20, 1.45]],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[0.40, 1.90], [2.20, -1.10], [-1.80, 0.95]],
        dtype=torch.float64,
    )

    loss, diagnostics = rbf_mmd_logits_loss(source, target, bandwidth=0.75)
    expected = _brute_force_unbiased_mmd(source, target, bandwidth=0.75)

    assert loss.dtype == torch.float64
    assert loss.device == source.device
    assert float(loss.detach()) == pytest.approx(float(expected), abs=1e-5)
    assert diagnostics["mmd_squared"] == pytest.approx(float(expected), abs=1e-5)
    assert diagnostics["bandwidth"] == 0.75
    assert diagnostics["source_count"] == 4.0
    assert diagnostics["target_count"] == 3.0


def test_rbf_mmd_logits_loss_supports_1d_logits_and_median_heuristic() -> None:
    source = torch.tensor([0.0, 2.0, 5.0, 7.0], dtype=torch.float64)
    target = torch.tensor([10.0, 11.0, 12.0], dtype=torch.float64)

    loss, diagnostics = rbf_mmd_logits_loss(source, target)

    combined = torch.cat((source, target))
    pairwise = torch.cdist(combined.unsqueeze(1), combined.unsqueeze(1), p=2.0)
    rows, columns = torch.triu_indices(len(combined), len(combined), offset=1)
    expected_bandwidth = float(pairwise[rows, columns].median())

    assert diagnostics["bandwidth"] == pytest.approx(expected_bandwidth)
    expected = _brute_force_unbiased_mmd(source, target, bandwidth=diagnostics["bandwidth"])
    assert float(loss.detach()) == pytest.approx(float(expected), abs=1e-5)


def test_rbf_mmd_logits_loss_reproduces_negative_mmd_and_zero_bandwidth_fallback() -> None:
    # Two well-separated duplicated samples: the unbiased estimator is
    # negative even though the empirical distributions are identical.
    source = torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float64)
    target = source.clone()
    loss, diagnostics = rbf_mmd_logits_loss(source, target, bandwidth=1.0)

    assert float(loss.detach()) < 0.0
    assert diagnostics["negative_mmd"] is True
    assert diagnostics["mmd_squared"] < 0.0

    # All-zero combined logits trigger the 1.0 bandwidth fallback and MMD^2
    # is exactly zero for identical empirical samples.
    zeros = torch.zeros((4, 2), dtype=torch.float64)
    zero_loss, zero_diagnostics = rbf_mmd_logits_loss(zeros, zeros.clone())
    assert zero_diagnostics["bandwidth"] == 1.0
    assert float(zero_loss.detach()) == pytest.approx(0.0, abs=1e-12)
    assert zero_diagnostics["negative_mmd"] is False


def test_asmmd_joint_loss_decomposes_exactly_and_gradients_flow() -> None:
    source = torch.tensor(
        [[1.40, -0.60], [-1.80, 0.30], [0.70, 1.90], [-0.40, -1.20]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor(
        [[0.60, -1.10], [1.70, 0.20], [-0.80, 0.90]],
        dtype=torch.float64,
        requires_grad=True,
    )
    source_labels = torch.tensor([0, 1, 1, 0])
    target_labels = torch.tensor([1, 0, 1])

    epoch = 12
    warmup_epochs = 24
    base_weight = 0.4
    pos_weight = 2.0
    bandwidth = 0.9

    total, diagnostics = asmmd_joint_loss(
        source,
        source_labels,
        target,
        target_labels,
        epoch=epoch,
        warmup_epochs=warmup_epochs,
        base_weight=base_weight,
        source_count=80,
        target_count=10,
        pos_weight=pos_weight,
        bandwidth=bandwidth,
    )

    target_w = target_weight(epoch, 80, 10, warmup_epochs)
    lambda_mmd = mmd_schedule(epoch, warmup_epochs, base_weight)
    class_weights = torch.tensor([1.0, pos_weight], dtype=source.dtype)
    source_ce = F.cross_entropy(source, source_labels, weight=class_weights)
    target_ce = F.cross_entropy(target, target_labels, weight=class_weights)
    mmd, mmd_diagnostics = rbf_mmd_logits_loss(source, target, bandwidth=bandwidth)

    expected = source_ce + target_w * target_ce + lambda_mmd * mmd

    assert total.dtype == torch.float64
    assert total.device == source.device
    assert float(total.detach()) == pytest.approx(float(expected.detach()), abs=1e-12)
    assert diagnostics["source_ce"] == pytest.approx(float(source_ce.detach()), abs=1e-12)
    assert diagnostics["target_ce"] == pytest.approx(float(target_ce.detach()), abs=1e-12)
    assert diagnostics["target_w"] == pytest.approx(target_w)
    assert diagnostics["lambda_mmd"] == pytest.approx(lambda_mmd)
    assert diagnostics["mmd_squared"] == pytest.approx(mmd_diagnostics["mmd_squared"])
    assert diagnostics["negative_mmd"] is mmd_diagnostics["negative_mmd"]
    assert diagnostics["bandwidth"] == bandwidth
    assert diagnostics["epoch"] == epoch
    assert diagnostics["source_count"] == 80.0
    assert diagnostics["target_count"] == 10.0
    assert diagnostics["batch_source_count"] == 4.0
    assert diagnostics["batch_target_count"] == 3.0

    total.backward()
    assert source.grad is not None
    assert target.grad is not None
    assert bool(torch.isfinite(source.grad).all())
    assert bool(torch.isfinite(target.grad).all())
    assert float(source.grad.abs().sum()) > 0.0
    assert float(target.grad.abs().sum()) > 0.0


def test_asmmd_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        target_weight(0, 0, 10, 40)
    with pytest.raises(ValueError):
        target_weight(0, 80, 10, 0)
    with pytest.raises(ValueError):
        target_weight(0, 80, 10, 40, minimum=0.0)
    with pytest.raises(ValueError):
        target_weight(0, 80, 10, 40, minimum=2.0, maximum=1.0)
    with pytest.raises(ValueError):
        mmd_schedule(0, 10, -0.1)
    with pytest.raises(ValueError):
        rbf_mmd_logits_loss(torch.zeros((1, 2)), torch.zeros((2, 2)))
    with pytest.raises(ValueError):
        rbf_mmd_logits_loss(
            torch.tensor([[0.0, float("nan")]] * 2),
            torch.zeros((2, 2)),
            bandwidth=1.0,
        )
    with pytest.raises(ValueError):
        asmmd_joint_loss(
            torch.zeros((2, 2)),
            torch.tensor([0, 2]),
            torch.zeros((2, 2)),
            torch.tensor([0, 1]),
            epoch=0,
            warmup_epochs=10,
            base_weight=0.4,
            source_count=80,
            target_count=10,
        )
