"""模块 #13 测试：数据增强。

冒烟：形状/dtype/device 保持、无 NaN、增强确实改变信号。
语义：
    - 参考抖动对常数偏移不变（参考无关性，D-ref-jitter）。
    - 时间扭曲保持端点固定（D-time-warp）。
    - 组合管线跑通。
"""

from __future__ import annotations

import numpy as np
import torch

from train.augment import (
    amplitude_jitter,
    apply_augmentations,
    channel_dropout,
    gaussian_noise,
    known_time_shift,
    reference_jitter,
    time_warp,
)


def make_x(B=4, C=8, T=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, C, T, generator=g)


# ---------------- 冒烟测试 ----------------


def test_shapes_preserved():
    X = make_x(B=3)
    for f in (time_warp, amplitude_jitter, gaussian_noise, channel_dropout, reference_jitter):
        torch.manual_seed(0)
        np.random.seed(0)
        X2 = f(X)
        assert X2.shape == (3, 8, 256), f"{f.__name__} 形状应保持"


def test_dtype_device_preserved():
    X = make_x(B=2)
    torch.manual_seed(0)
    np.random.seed(0)
    X2 = time_warp(X)  # time_warp 有 numpy 往返，最易丢 dtype/device
    assert X2.dtype == X.dtype
    assert X2.device == X.device


def test_no_nan():
    X = make_x(B=4)
    for f in (time_warp, amplitude_jitter, gaussian_noise, channel_dropout, reference_jitter):
        torch.manual_seed(0)
        np.random.seed(0)
        assert not torch.isnan(f(X)).any(), f.__name__


def test_changes_signal():
    X = make_x(B=2)
    torch.manual_seed(0)
    np.random.seed(0)
    assert not torch.allclose(time_warp(X, max_shift=16), X), "时间扭曲应改变信号"
    assert not torch.allclose(amplitude_jitter(X, scale=0.3), X), "幅值抖动应改变信号"
    assert not torch.allclose(gaussian_noise(X, sigma=0.3), X), "高斯噪声应改变信号"


# ---------------- 语义测试 ----------------


def test_reference_jitter_reference_invariance():
    """参考抖动对常数偏移不变（减凸组合参考消掉常数偏移，D-ref-jitter）。"""
    X = make_x(B=2)
    torch.manual_seed(0)
    a = reference_jitter(X, p=1.0)
    torch.manual_seed(0)  # 重置 seed，使两次凸组合权重相同
    b = reference_jitter(X + 5.0, p=1.0)
    assert torch.allclose(a, b, atol=1e-5), "参考抖动应对常数偏移不变"


def test_time_warp_endpoints_fixed():
    """时间扭曲保持端点（端点锚点固定，D-time-warp）。"""
    X = make_x(B=2)
    torch.manual_seed(0)
    np.random.seed(0)
    Xw = time_warp(X, max_shift=8)
    assert torch.allclose(Xw[:, :, 0], X[:, :, 0], atol=1e-5), "起点应保持"
    assert torch.allclose(Xw[:, :, -1], X[:, :, -1], atol=1e-5), "终点应保持"


def test_time_warp_endpoints_fixed_single_batch():
    """B=1 时锚点张量不得与 src 共享存储：否则 dst 原地修改会污染逆映射（audit 回归）。"""
    X = make_x(B=1)
    torch.manual_seed(0)
    Xw = time_warp(X, max_shift=8)
    assert torch.allclose(Xw[:, :, 0], X[:, :, 0], atol=1e-5), "B=1 起点应保持"
    assert torch.allclose(Xw[:, :, -1], X[:, :, -1], atol=1e-5), "B=1 终点应保持"


def test_channel_dropout_zeroes_some_channels():
    """通道 dropout 置 0 某些通道。"""
    X = make_x(B=8, T=64)
    torch.manual_seed(0)
    X2 = channel_dropout(X, p=0.5)
    # 至少某个 batch 的某通道被置 0
    assert (X2 == 0).any(), "通道 dropout 应置 0 某些通道"


def test_apply_augmentations_runs():
    """组合管线跑通、形状保持。"""
    X = make_x(B=4)
    X2 = apply_augmentations(X, seed=0)
    assert X2.shape == (4, 8, 256)
    assert not torch.isnan(X2).any()


def _missing_mask(C=8, present=3):
    mask = torch.zeros(C, dtype=torch.bool)
    mask[:present] = True
    return mask


def test_reference_jitter_preserves_missing_channels():
    """review v6 P0-2：参考抖动不得把缺失通道变成非零。"""
    X = make_x(B=4)
    mask = _missing_mask()
    X[:, ~mask, :] = 0.0
    X2 = reference_jitter(X, p=1.0, channel_mask=mask)
    assert (X2[:, ~mask, :] == 0.0).all(), "缺失通道经 reference_jitter 后必须恒 0"


def test_gaussian_noise_preserves_missing_channels():
    """review v6 P0-2：高斯噪声只加在存在通道。"""
    X = make_x(B=4)
    mask = _missing_mask()
    X[:, ~mask, :] = 0.0
    X2 = gaussian_noise(X, sigma=0.1, channel_mask=mask)
    assert (X2[:, ~mask, :] == 0.0).all(), "缺失通道经 gaussian_noise 后必须恒 0"


def test_apply_augmentations_preserves_missing_channels():
    """review v6 P0-2：组合增强后缺失通道仍恒 0。"""
    X = make_x(B=8)
    mask = _missing_mask()
    X[:, ~mask, :] = 0.0
    X2 = apply_augmentations(
        X,
        seed=0,
        p_time_warp=1.0,
        p_amp_jitter=1.0,
        p_noise=1.0,
        p_ch_dropout=1.0,
        p_ref_jitter=1.0,
        channel_mask=mask,
    )
    assert (X2[:, ~mask, :] == 0.0).all(), "组合增强后缺失通道必须恒 0"


def test_known_time_shift_exact_and_zero_pad():
    """L_jit 用的已知偏移：正偏移把波形右移，越界填 0。"""
    X = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
    shift = torch.tensor([1])
    X2 = known_time_shift(X, shift)
    assert torch.equal(X2, torch.tensor([[[0.0, 0.0, 1.0, 2.0]]]))

    shift = torch.tensor([-1])
    X3 = known_time_shift(X, shift)
    assert torch.equal(X3, torch.tensor([[[1.0, 2.0, 3.0, 0.0]]]))


def test_known_time_shift_batched_and_differentiable():
    X = torch.randn(4, 8, 64, requires_grad=True)
    shifts = torch.tensor([3, -2, 5, -1])
    X2 = known_time_shift(X, shifts)
    assert X2.shape == X.shape
    assert not torch.isnan(X2).any()
    X2.sum().backward()
    assert X.grad is not None and X.grad.abs().sum() > 0
