"""TemporalPoolingBypass 测试（v5.1 EEGNet 借鉴）。"""
from __future__ import annotations

import pytest
import torch

from models.bypass import TemporalPoolingBypass

D = 64
T = 256


def test_forward_shape_and_out_features():
    byp = TemporalPoolingBypass()
    Z = torch.randn(4, T, D)
    g = byp(Z)
    assert g.shape == (4, byp.f2 * byp.pool2)
    assert byp.out_features == byp.f2 * byp.pool2


def test_dynamic_t_compatible():
    """第二段 adaptive pool 固定 8 bin：T=128 也能复用同一 Linear 头维度。"""
    byp = TemporalPoolingBypass()
    g = byp(torch.randn(2, 128, D))
    assert g.shape == (2, byp.f2 * byp.pool2)


def test_not_divisible_by_pool1_raises():
    byp = TemporalPoolingBypass()
    with pytest.raises(ValueError):
        byp(torch.randn(2, 130, D))


def test_wrong_d_raises():
    byp = TemporalPoolingBypass()
    with pytest.raises(ValueError):
        byp(torch.randn(2, T, D + 1))


def test_backward_flows():
    byp = TemporalPoolingBypass()
    Z = torch.randn(2, T, D, requires_grad=True)
    byp(Z).sum().backward()
    assert Z.grad is not None
    assert Z.grad.abs().sum() > 0
