"""models/reference.py 的两级测试（冒烟 + 语义）。"""

import torch
import pytest

from models.reference import WeightedRereference


# --------------------------------------------------------------------------- #
# 冒烟测试
# --------------------------------------------------------------------------- #
def test_shape_dtype():
    ref = WeightedRereference(n_channels=8)
    X = torch.randn(4, 8, 256)
    out = ref(X)
    assert out.shape == (4, 8, 256)
    assert out.dtype == X.dtype
    assert out.requires_grad  # 参数可学习、输出可回传梯度


def test_invalid_input():
    ref = WeightedRereference(n_channels=8)
    with pytest.raises(ValueError):
        ref(torch.randn(8, 256))  # 非 (B,C,T)


# --------------------------------------------------------------------------- #
# 语义测试
# --------------------------------------------------------------------------- #
def test_default_is_car():
    """w_logits=0 → w 均匀 → 输出 = X − 通道均值 = CAR。"""
    ref = WeightedRereference(n_channels=8)
    X = torch.randn(2, 8, 16)
    out = ref(X)
    car = X - X.mean(dim=1, keepdim=True)
    assert torch.allclose(out, car, atol=1e-6)


def test_w_normalized():
    """w = softmax(logits)，Σw=1 且各分量 >= 0。"""
    ref = WeightedRereference(n_channels=8)
    w = ref.w
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)
    assert (w >= 0).all()


def test_r_matrix_correctness():
    """R = I − 1·wᵀ（外积方向 v3 修正），out == R @ X。"""
    ref = WeightedRereference(n_channels=8)
    X = torch.randn(2, 8, 16)
    w = ref.w.detach()  # (C,)
    ones = torch.ones(8, 1)
    R = torch.eye(8) - ones @ w.unsqueeze(0)  # (C,C) = I − 1·wᵀ
    expected = R @ X  # (C,C) @ (B,C,T) → (B,C,T)
    assert torch.allclose(ref(X), expected, atol=1e-6)


def test_reference_invariance_to_common_offset():
    """参考无关性：所有通道加同一常数偏移，输出严格不变（D-ref-invariance）。"""
    torch.manual_seed(0)
    ref = WeightedRereference(n_channels=8)
    X = torch.randn(4, 8, 256)
    offset = 3.7  # 模拟参考电极带来的 DC 偏移
    out1 = ref(X)
    out2 = ref(X + offset)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_gain():
    """use_gain=True 时输出 = (X − 1·wᵀX) * gain。"""
    ref = WeightedRereference(n_channels=8, use_gain=True)
    X = torch.randn(2, 8, 16)
    w = ref.w.detach()
    m = torch.einsum("c,bct->bt", w, X)
    expected = (X - m.unsqueeze(1)) * ref.gain.detach().view(1, -1, 1)
    assert torch.allclose(ref(X), expected, atol=1e-6)


def test_mask_weight_renormalized():
    """mask 重归一化：加权均值只由存在通道计算（w 置 0 后 renorm）。"""
    ref = WeightedRereference(n_channels=8)
    X = torch.randn(2, 8, 16)
    mask = torch.tensor([True, True, False, True, False, False, False, False])
    out = ref(X, channel_mask=mask)
    # 手工：存在通道上的均匀均值（w 初始均匀 → renorm 后存在通道各 1/3）
    m_manual = X[:, mask, :].mean(dim=1)  # (B, T)
    expected_present = X[:, mask, :] - m_manual.unsqueeze(1)
    assert torch.allclose(out[:, mask, :], expected_present, atol=1e-6)


def test_bf16_direct_input_keeps_dtype():
    """review v6 P1：不开 autocast、直接 bf16 输入时，参数须对齐到 X.dtype 且输出保持 bf16。"""
    ref = WeightedRereference(n_channels=8)
    X = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    out = ref(X)
    assert out.dtype == torch.bfloat16


def test_mask_no_phantom_channel():
    """幻象通道回归（v4）：缺失通道出口恒 0——若对所有通道减 m，缺失通道会变 −m(t)，
    被下游基线段标准化放大成 std≈1 的幻象（逐试次变化、坐标上冒充枕/顶区地形）。"""
    torch.manual_seed(0)
    ref = WeightedRereference(n_channels=8, use_gain=True)
    X = torch.randn(4, 8, 256)
    mask = torch.tensor([True, True, False, True, False, False, False, False])
    X[:, ~mask, :] = 0.0  # 零填充（nan_to_num 后）
    out = ref(X, channel_mask=mask)
    assert (out[:, ~mask, :] == 0.0).all(), "缺失通道必须恒 0（含 gain 之后）"
    # 存在通道确实被减了均值（非恒等）
    assert out[:, mask, :].abs().sum() > 0
