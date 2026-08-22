"""模块 #12 测试：总损失。

冒烟：各损失标量 / 无 NaN / total=分项加权和。
语义：
    - L_tau 在 τ=τ0 时为 0。
    - L_tau 不监督 τ0（梯度抵消，D-tau0-not-supervised）。
    - pos_weight 生效（正样本加权）。
    - RBF-MMD：同分布≈0、不同分布>0。
    - λ4=0 时 L_MMD=0。
"""

from __future__ import annotations

import torch

from models.heads import HeadsOutput
from models.n2p3net import N2P3NetOutput
from train.losses import (
    Losses,
    _bce_with_pos_weight,
    compute_losses,
    rbf_mmd2,
    tau_regularization,
)

TAU0 = torch.tensor([220.0, 300.0, 350.0])


def make_output(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return N2P3NetOutput(
        heads=HeadsOutput(
            logit_target=torch.randn(B, 1, generator=g),
            logit_early=torch.randn(B, 1, generator=g),
            amplitude=torch.randn(B, 1, generator=g),
        ),
        tau=torch.randn(B, 3, generator=g),
        sigma=torch.randn(3, 2),
        H=torch.randn(B, 3, 64),
        attention=None,
    )


# ---------------- 冒烟测试 ----------------


def test_losses_are_scalars():
    out = make_output(B=4)
    y = torch.randint(0, 2, (4, 1)).float()
    losses = compute_losses(out, TAU0, y)
    for v in (losses.total, losses.target, losses.early, losses.tau, losses.mmd):
        assert v.dim() == 0, "损失应为 0 维标量"
        assert not torch.isnan(v)


def test_total_is_weighted_sum():
    out = make_output(B=4)
    y = torch.randint(0, 2, (4, 1)).float()
    l2, l3 = 0.3, 1e-2
    losses = compute_losses(out, TAU0, y, lambda2=l2, lambda3=l3)
    expected = losses.target + l2 * losses.early + l3 * losses.tau + 0.0 * losses.mmd
    assert torch.allclose(losses.total, expected)


# ---------------- 语义测试 ----------------


def test_tau_zero_when_at_prior():
    """τ=τ0 时 L_tau=0。"""
    tau = torch.tensor([[220.0, 300.0, 350.0], [220.0, 300.0, 350.0]])
    L = tau_regularization(tau, TAU0, 50.0)
    assert L.item() == 0.0


def test_tau0_not_supervised():
    """L_tau 不监督 τ0（τ−τ0 中 τ0 梯度抵消，D-tau0-not-supervised）。"""
    tau0 = torch.tensor([220.0, 300.0, 350.0], requires_grad=True)
    dtau = torch.tensor([[0.0, -15.0, 15.0]])  # 常数偏移（不依赖 tau0）
    tau = tau0 + dtau  # τ 依赖 tau0（模拟 component_window 的 τ=τ0+Δτ）
    L = tau_regularization(tau, tau0, 50.0)
    L.backward()
    assert tau0.grad is not None
    assert tau0.grad.abs().sum().item() == 0.0, "L_tau 不应监督 τ0（只正则 Δτ）"


def test_pos_weight_effect():
    """pos_weight 生效：正样本损失被放大。"""
    logits = torch.tensor([[2.0], [-2.0]])
    y = torch.tensor([[1.0], [0.0]])
    l1 = _bce_with_pos_weight(logits, y, 1.0)
    l8 = _bce_with_pos_weight(logits, y, 8.0)
    assert l8.item() > l1.item(), "pos_weight=8 应放大正样本损失"


def test_mmd_same_vs_different():
    """RBF-MMD：不同分布 > 同分布。"""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(50, 8, generator=g)
    y_same = torch.randn(50, 8, generator=g)
    y_diff = torch.randn(50, 8, generator=g) + 3.0
    mmd_same = rbf_mmd2(x, y_same)
    mmd_diff = rbf_mmd2(x, y_diff)
    assert mmd_diff.item() > mmd_same.item(), "不同分布 MMD 应大于同分布"


def test_mmd_median_heuristic_d64_nonzero_gradient():
    """review v6 P1：D=64 下 median heuristic 的 MMD 非零且可回传梯度（固定 bw=1 会坍缩为 0）。"""
    x = torch.randn(128, 64, requires_grad=True)
    y = torch.randn(128, 64) + 1.0
    mmd = rbf_mmd2(x, y)  # bandwidth=None → median heuristic
    assert mmd.item() > 1e-3, f"median heuristic 下 MMD 不应坍缩，得到 {mmd.item()}"
    mmd.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "MMD 梯度不应为全零"


def test_mmd_accepts_3d_features_with_time_pooling():
    """N2P3NetOutput.features=(B,T,D) 时，compute_losses 须池化为 (B,D) 再算 MMD（audit P1-2）。"""
    out = make_output(B=8)
    y = torch.randint(0, 2, (8, 1)).float()
    z3 = torch.randn(8, 16, 64)
    domain_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    losses = compute_losses(
        out, TAU0, y, lambda4=1.0, z_features=z3, domain_ids=domain_ids,
        lambda2=0.0, lambda3=0.0, lambda_amp=0.0,
    )
    assert torch.isfinite(losses.mmd)
    assert losses.mmd.item() != 0.0


def test_mmd_bf16_inputs_work():
    """rbf_mmd2 内部提升为 fp32，bf16 输入不应崩（audit P1-3）。"""
    x = torch.randn(16, 64, dtype=torch.bfloat16, requires_grad=True)
    y = torch.randn(16, 64, dtype=torch.bfloat16) + 1.0
    mmd = rbf_mmd2(x, y)
    assert torch.isfinite(mmd)
    mmd.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_lambda4_disabled():
    """λ4=0 时 L_MMD=0（Phase 2 零开销）。"""
    out = make_output(B=4)
    y = torch.randint(0, 2, (4, 1)).float()
    losses = compute_losses(out, TAU0, y, lambda4=0.0)
    assert losses.mmd.item() == 0.0



def test_jitter_consistency_loss():
    """Phase 2 L_jit：tau_shift 偏离 tau+shift_ms 时给出正损失，且进入 total。"""
    out = make_output(B=4)
    y = torch.randint(0, 2, (4, 1)).float()
    shift_ms = torch.tensor([10.0, -12.0, 8.0, -6.0])
    tau_shift = out.tau.detach().clone() + shift_ms[:, None] + 3.0
    losses = compute_losses(
        out, TAU0, y, lambda_jit=0.2, tau_shift=tau_shift, shift_ms=shift_ms,
        lambda2=0.0, lambda3=0.0, lambda_amp=0.0,
    )
    assert losses.jit is not None and losses.jit.item() > 0
    expected = 0.2 * losses.jit
    assert torch.allclose(losses.total, losses.target + expected, atol=1e-6)