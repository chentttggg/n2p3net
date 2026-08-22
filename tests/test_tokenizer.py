"""模块 #6 测试：ERP 感知时空 token 化层。

冒烟：形状 / dtype / 无 NaN / 可选元数据 / 自定义核长。
语义（关键路径，CODING_WORKFLOW §3）：
    - 无池化保 T（E5 落地）：输出 T 恒等于输入 T，且支持任意 T（不硬编码 256）。
    - 地形先验初始化（D-spatial-prior）：短核空间权重在 PO7/PO8/Oz（索引 5/6/7）为负、
      长核在 P3/Pz/P4（索引 2/3/4）为正。
    - 坐标调制生效（D-coord-mod）：改变 E_chn 会改变空间权重与输出。
    - subject 融合生效（D-sub-bias）：改变 E_sub 会平移输出（加性 bias）。
    - 时间局部性（无池化 → 无长程扩散）：脉冲响应集中在脉冲附近。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from models.tokenizer import ERPTokenizer

C = 8
D = 64
T = 256
D_CHN = 48
D_SUB = 19


def make_tokenizer(**kw):
    torch.manual_seed(0)
    return ERPTokenizer(**kw)


def make_inputs(B=4, T=T, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, C, T, generator=g)
    E_chn = torch.randn(C, D_CHN, generator=g)
    E_sub = torch.randn(D_SUB, generator=g)
    return X, E_chn, E_sub


# ---------------- 冒烟测试 ----------------


def test_forward_shape():
    tok = make_tokenizer()
    X, E_chn, E_sub = make_inputs(B=4)
    Z = tok(X, E_chn, E_sub)
    assert Z.shape == (4, T, D)


def test_forward_dtype():
    tok = make_tokenizer()
    X, E_chn, E_sub = make_inputs(B=2)
    Z = tok(X, E_chn, E_sub)
    assert Z.dtype == torch.float32


def test_forward_no_nan():
    tok = make_tokenizer()
    X, E_chn, E_sub = make_inputs(B=4)
    Z = tok(X, E_chn, E_sub)
    assert not torch.isnan(Z).any()
    assert not torch.isinf(Z).any()


def test_forward_without_metadata():
    """E_chn/E_sub 为 None 时退化运行（不融合），不应报错。"""
    tok = make_tokenizer()
    X, _, _ = make_inputs(B=4)
    Z = tok(X)
    assert Z.shape == (4, T, D)


def test_forward_custom_kernels():
    """自定义核长（不同尺度数）也能正确跑通。"""
    tok = make_tokenizer(temporal_kernels=(33, 65))
    X, E_chn, E_sub = make_inputs(B=3)
    Z = tok(X, E_chn, E_sub)
    assert Z.shape == (3, T, D)


def test_forward_wrong_channels_raises():
    tok = make_tokenizer()
    X = torch.randn(4, 6, T)  # 通道数 6 ≠ 8
    with pytest.raises(ValueError):
        tok(X)


def test_native_3ch_forward():
    """v5.1：GTN 原生 3 导蒙太奇（不零填充到 8 导）。"""
    tok = make_tokenizer(n_channels=3, channel_names=("Fz", "Cz", "Pz"))
    X = torch.randn(4, 3, T)
    E_chn = torch.randn(3, D_CHN)
    Z = tok(X, E_chn)
    assert Z.shape == (4, T, D)


def test_3ch_requires_channel_names():
    with pytest.raises(ValueError):
        make_tokenizer(n_channels=3)


def test_spatial_prior_3ch():
    """v5.1：3 导时短核前中央负（Fz/Cz）、长核 Pz 正。"""
    tok = make_tokenizer(n_channels=3, channel_names=("Fz", "Cz", "Pz"))
    for k, prior in zip(tok.temporal_kernels, tok.spatial_priors):
        p = prior.detach()
        if k < 64:
            assert p[:, 0:2].mean() < 0, "短核应以前中央 Fz/Cz 为负"
        else:
            assert p[:, 2].mean() > 0, "长核应以 Pz 为正"


def test_max_norm_spatial():
    """v5.1：EEGNet 式 max-norm——范数超限的行被缩回 1，未超限不动。"""
    from models.tokenizer import max_norm_spatial

    W = torch.tensor([[3.0, 4.0], [0.3, 0.4]])
    Wc = max_norm_spatial(W, 1.0)
    assert torch.allclose(Wc[0].norm(), torch.tensor(1.0))
    assert torch.allclose(Wc[1], W[1])
    assert torch.equal(max_norm_spatial(W, None), W)


# ---------------- 语义测试 ----------------


def test_preserves_time_resolution():
    """无池化保 T（E5）：输出时间步恒等于输入时间步，且不硬编码 256。"""
    tok = make_tokenizer()
    for n in (128, 200, 256, 300):
        X, E_chn, E_sub = make_inputs(B=2, T=n)
        Z = tok(X, E_chn, E_sub)
        assert Z.shape[1] == n, f"T={n} 输入但输出 {Z.shape[1]}"


def test_spatial_prior_init():
    """地形先验初始化（D-spatial-prior）：短核枕区负、长核顶区正。"""
    tok = make_tokenizer()
    kernels = tok.temporal_kernels  # (13, 33, 65, 129)
    for k, prior in zip(kernels, tok.spatial_priors):
        p = prior.detach()  # (F, C)
        if k < 64:  # 短核 → N2 枕区负（索引 5/6/7）
            assert p[:, 5:8].mean() < 0, f"核长 {k} 应为 N2 枕区负"
        else:  # 长核 → P3b 顶区正（索引 2/3/4）
            assert p[:, 2:5].mean() > 0, f"核长 {k} 应为 P3b 顶区正"


def test_coord_modulation_changes_output():
    """坐标调制生效（D-coord-mod）：坐标调制权重非零时，改变 E_chn 改变空间权重与输出。

    坐标调制初始化为 0（使初始空间权重 = 纯地形先验），故初始状态下 E_chn 不影响输出；
    此处手动注入非零权重以验证「坐标 → 空间权重」机制真实存在、且会被训练激活。
    """
    tok = make_tokenizer()
    for mod in tok.coord_mods:
        nn.init.normal_(mod.weight, std=0.1)
    X, E_chn, E_sub = make_inputs(B=2)
    E_chn2 = torch.randn(C, D_CHN)

    Z1 = tok(X, E_chn, E_sub)
    Z2 = tok(X, E_chn2, E_sub)
    assert not torch.allclose(Z1, Z2), "改变 E_chn 应改变输出（坐标调制生效）"


def test_subject_fusion_is_additive_bias():
    """subject 融合为加性 bias（D-sub-bias）：E_sub 改变应等量平移每个时间步。"""
    tok = make_tokenizer()
    X, E_chn, _ = make_inputs(B=1)
    e1 = torch.zeros(D_SUB)
    e2 = torch.ones(D_SUB)

    Z1 = tok(X, E_chn, e1)  # (1, T, D)
    Z2 = tok(X, E_chn, e2)
    # 加性 bias：Z2 - Z1 在每个时间步应近似一致（同一 bias 向量）
    diff = Z2 - Z1  # (1, T, D)
    per_t = diff.squeeze(0)  # (T, D)
    # 每个时间步的差值向量应几乎相同（std 极小）
    assert per_t.std(dim=0).abs().max() < 1e-4, "E_sub 融合应为逐时间步一致的加性 bias"


def test_temporal_locality():
    """时间局部性（无池化 → 无长程扩散）：脉冲响应集中在脉冲附近。"""
    tok = make_tokenizer()
    tok.eval()
    t0 = 128
    X = torch.zeros(1, C, T)
    X[:, :, t0] = 1.0  # 单点脉冲
    E_chn = torch.zeros(C, D_CHN)
    E_sub = torch.zeros(D_SUB)
    with torch.no_grad():
        Z = tok(X, E_chn, E_sub)  # (1, T, D)
    energy = Z[0].pow(2).sum(-1)  # (T,)
    near = energy[t0 - 70 : t0 + 70].sum()
    far = energy[: t0 - 70].sum() + energy[t0 + 70 :].sum()
    assert near > far, "脉冲响应应集中在脉冲附近（无池化导致的局部性）"


def test_autocast_bf16_preserves_dtype():
    """PE dtype 对齐（D-time-pe）：AMP(bf16) 下输出保持 bf16，不 promote 到 float32。

    回归：修复前 pe 硬编码 float32，autocast 下 token 流是 bf16，Z+pe 会 promote 到
    float32（实测 bf16+float32→float32），破坏 bf16 省显存/加速语义。
    """
    tok = make_tokenizer()
    X = torch.randn(2, C, T)
    E_chn = torch.randn(C, D_CHN)
    E_sub = torch.randn(D_SUB)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        Z = tok(X, E_chn, E_sub)
    assert Z.dtype == torch.bfloat16, f"autocast 下输出应保持 bf16，得到 {Z.dtype}"
