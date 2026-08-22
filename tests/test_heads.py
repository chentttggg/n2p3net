"""模块 #9 测试：Stage 3 多任务头。

冒烟：形状 / dtype / 无 NaN / p=sigmoid(logit) / 异常。
语义：
    - 成分分离：Head-A 用 P3b+P3a、Head-B 用 N2、Head-D 用 P3b（改某成分只影响对应头）。
    - logit 输出（决策层对数似然比累积用，非概率）。
"""

from __future__ import annotations

import pytest
import torch

from models.heads import MultiTaskHeads

D = 64


def make_heads(**kw):
    torch.manual_seed(0)
    return MultiTaskHeads(**kw)


def make_h(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, 3, D, generator=g)


def make_g(B=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, D, generator=g)


# ---------------- 冒烟测试 ----------------


def test_forward_shape():
    heads = make_heads()
    out = heads(make_h(B=4), global_features=make_g(B=4))
    assert out.logit_target.shape == (4, 1)
    assert out.logit_early.shape == (4, 1)
    assert out.amplitude.shape == (4, 1)


def test_forward_dtype():
    heads = make_heads()
    out = heads(make_h(B=2), global_features=make_g(B=2))
    assert out.logit_target.dtype == torch.float32
    assert out.logit_early.dtype == torch.float32
    assert out.amplitude.dtype == torch.float32


def test_forward_no_nan():
    heads = make_heads()
    out = heads(make_h(B=4), global_features=make_g(B=4))
    assert not torch.isnan(out.logit_target).any()
    assert not torch.isnan(out.logit_early).any()
    assert not torch.isnan(out.amplitude).any()


def test_p_is_sigmoid():
    heads = make_heads()
    out = heads(make_h(B=2), global_features=make_g(B=2))
    assert torch.allclose(out.p_target, torch.sigmoid(out.logit_target))
    assert torch.allclose(out.p_early, torch.sigmoid(out.logit_early))
    # 概率在 (0,1)
    assert (out.p_target > 0).all() and (out.p_target < 1).all()


def test_global_bypass_requires_features():
    """D-global-bypass：默认开启时缺失 global_features 必须显式报错。"""
    heads = make_heads()
    with pytest.raises(ValueError):
        heads(make_h(B=4))


def test_global_bypass_disabled_keeps_old_contract():
    """use_global_bypass=False 保留旧结构，heads(H) 仍可用。"""
    heads = make_heads(use_global_bypass=False)
    out = heads(make_h(B=4))
    assert out.logit_target.shape == (4, 1)
    assert out.logit_early.shape == (4, 1)
    assert out.amplitude.shape == (4, 1)


def test_wrong_shape_raises():
    heads = make_heads()
    H = torch.randn(4, 5, D)  # 成分数 5 ≠ 3
    with pytest.raises(ValueError):
        heads(H, global_features=make_g(B=4))


# ---------------- 语义测试 ----------------


def test_head_component_separation():
    """成分分离：Head-A 用 P3b+P3a、Head-B 用 N2、Head-D 用 P3b。"""
    heads = make_heads()
    heads.eval()  # 关闭 dropout，保证成分/旁路消融的 allclose 比较确定
    H = torch.zeros(2, 3, D)
    H[:, 0] = 1.0  # N2
    H[:, 1] = 2.0  # P3a
    H[:, 2] = 3.0  # P3b
    g_global = torch.zeros(2, D)
    base = heads(H, global_features=g_global)

    # 改 N2（索引 0）→ 只影响 logit_early
    h2 = H.clone()
    h2[:, 0] = 100.0
    o2 = heads(h2, global_features=g_global)
    assert not torch.allclose(base.logit_early, o2.logit_early), "改 N2 应影响 early"
    assert torch.allclose(base.logit_target, o2.logit_target), "改 N2 不应影响 target"
    assert torch.allclose(base.amplitude, o2.amplitude), "改 N2 不应影响 amplitude"

    # 改 P3b（索引 2）→ 影响 target 和 amplitude，不影响 early
    h3 = H.clone()
    h3[:, 2] = 200.0
    o3 = heads(h3, global_features=g_global)
    assert not torch.allclose(base.logit_target, o3.logit_target), "改 P3b 应影响 target"
    assert not torch.allclose(base.amplitude, o3.amplitude), "改 P3b 应影响 amplitude"
    assert torch.allclose(base.logit_early, o3.logit_early), "改 P3b 不应影响 early"

    # 改 P3a（索引 1）→ 影响 target，不影响 early/amplitude
    h4 = H.clone()
    h4[:, 1] = 300.0
    o4 = heads(h4, global_features=g_global)
    assert not torch.allclose(base.logit_target, o4.logit_target), "改 P3a 应影响 target"
    assert torch.allclose(base.logit_early, o4.logit_early), "改 P3a 不应影响 early"
    assert torch.allclose(base.amplitude, o4.amplitude), "改 P3a 不应影响 amplitude"

    # 改 global_features → 只影响 target（旁路只进 Head-A）
    g2 = torch.ones(2, D)
    o5 = heads(H, global_features=g2)
    assert not torch.allclose(base.logit_target, o5.logit_target), "改 global 应影响 target"
    assert torch.allclose(base.logit_early, o5.logit_early), "改 global 不应影响 early"
    assert torch.allclose(base.amplitude, o5.amplitude), "改 global 不应影响 amplitude"


def test_grad_flows_to_all_heads():
    """三路输出均可反向传播（梯度回传链路完整，含 global 旁路）。"""
    heads = make_heads()
    H = make_h(B=2)
    H.requires_grad_(True)
    g_global = make_g(B=2)
    g_global.requires_grad_(True)
    out = heads(H, global_features=g_global)
    (out.logit_target.sum() + out.logit_early.sum() + out.amplitude.sum()).backward()
    assert H.grad is not None
    assert not torch.isnan(H.grad).any()
    assert g_global.grad is not None
    assert g_global.grad.abs().sum() > 0, "global 旁路应接收非零梯度"
