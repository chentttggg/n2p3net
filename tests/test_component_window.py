"""模块 #8 测试：参数化成分窗（PCW，项目核心模块）。

冒烟：形状 / dtype / 无 NaN / return_attention / 异常。
语义（关键路径，CODING_WORKFLOW §3）：
    - τ=峰值（D8/D9）：A_c 的 argmax 精确落在 τ_c 对应的时间索引。
    - 软窗归一化（softmax）：Σ_t A_c(t) ≈ 1。
    - σ 有界（D9 软参数化）：σ 严格落在 [lo,hi] 内，无 clamp。
    - Δτ 不对称界（v3 P1）：P3a 只前移、P3b 只后移、N2 双向。
    - 软对齐读取正确：H = Σ A·Z' 手动对照。
    - 不对称性（D9）：σ_down > σ_up 时右侧（下降沿）比左侧（上升沿）宽。
"""

from __future__ import annotations

import pytest
import torch

from models.component_window import (
    PCW_CANONICAL_DTAU_BOUNDS,
    PCW_CANONICAL_SIGMA_BOUNDS,
    PCW_CANONICAL_TAU0_BOUNDS,
    PCW_CANONICAL_TAU0_MS,
    ComponentWindow,
)
from train.losses import tau_regularization

D = 64
T = 256


def make_cw(**kw):
    torch.manual_seed(0)
    return ComponentWindow(**kw)


def make_input(B=4, T=T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g)


def _time_index(ms: float, cw: ComponentWindow) -> int:
    """物理时间 ms → 时间轴索引（四舍五入）。"""
    return int(round((ms - cw.tmin) / (cw.tmax - cw.tmin) * (T - 1)))


# ---------------- 冒烟测试 ----------------


def test_forward_shape():
    cw = make_cw()
    Z = make_input(B=4)
    H, tau, sigma = cw(Z)
    assert H.shape == (4, 3, D)
    assert tau.shape == (4, 3)
    assert sigma.shape == (3, 2)


def test_forward_dtype():
    cw = make_cw()
    Z = make_input(B=2)
    H, tau, sigma = cw(Z)
    assert H.dtype == torch.float32
    assert tau.dtype == torch.float32
    assert sigma.dtype == torch.float32


def test_forward_no_nan():
    cw = make_cw()
    Z = make_input(B=4)
    H, tau, sigma = cw(Z)
    assert not torch.isnan(H).any()
    assert not torch.isnan(tau).any()
    assert not torch.isnan(sigma).any()


def test_return_attention_shape():
    cw = make_cw()
    Z = make_input(B=4)
    H, tau, sigma, A = cw(Z, return_attention=True)
    assert A.shape == (4, 3, T)


def test_wrong_dim_raises():
    cw = make_cw()
    Z = torch.randn(4, 64)  # 2D，非 (B,T,D)
    with pytest.raises(ValueError):
        cw(Z)


# ---------------- 语义测试 ----------------


def test_peak_at_tau():
    """D8/D9 τ=峰值：A_c 的 argmax 精确落在 τ_c 对应的时间索引。"""
    cw = make_cw()
    cw.eval()
    Z = make_input(B=2)
    with torch.no_grad():
        _, tau, _, A = cw(Z, return_attention=True)
    for b in range(2):
        for c in range(3):
            expected = _time_index(tau[b, c].item(), cw)
            actual = int(A[b, c].argmax())
            assert abs(actual - expected) <= 1, (
                f"成分{c} A 峰值应在 τ={tau[b, c]:.1f}ms（索引{expected}），实际{actual}"
            )


def test_softmax_normalized():
    """软窗归一化：Σ_t A_c(t) ≈ 1。"""
    cw = make_cw()
    Z = make_input(B=2)
    with torch.no_grad():
        _, _, _, A = cw(Z, return_attention=True)
    sums = A.sum(dim=-1)  # (B, 3)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_sigma_bounded():
    """D9 软参数化：σ 严格落在 [lo,hi] 内（sigmoid 软映射，无 clamp）。"""
    cw = make_cw()
    Z = make_input(B=2)
    with torch.no_grad():
        _, _, sigma = cw(Z)
    for c in range(3):
        lo = cw.sigma_lo[c].item()
        hi = cw.sigma_hi[c].item()
        assert lo < sigma[c, 0].item() < hi, f"成分{c} σ_up 应在 ({lo},{hi}) 内"
        assert lo < sigma[c, 1].item() < hi, f"成分{c} σ_down 应在 ({lo},{hi}) 内"


def test_dtau_asymmetric_bounds():
    """v3 P1 不对称界（Phase 2 修订）：P3a 只前移（τ≤τ0）、N2 双向（|τ−τ0|≤30）、
    P3b 放宽为 [−50, +150]（覆盖真实 P300 300–500ms）。"""
    cw = make_cw()
    tau0 = cw.tau0.detach().clone()
    for _ in range(5):
        Z = make_input(B=4)
        with torch.no_grad():
            _, tau, _ = cw(Z)
        assert (torch.abs(tau[:, 0] - tau0[0]) <= 30.0 + 1e-3).all(), "N2 应 |τ−τ0|≤30"
        assert (tau[:, 1] <= tau0[1] + 1e-3).all(), "P3a 应只前移（τ≤τ0）"
        assert (tau[:, 2] - tau0[2] >= -50.0 - 1e-3).all(), "P3b 下界应 ≥ τ0−50"
        assert (tau[:, 2] - tau0[2] <= 150.0 + 1e-3).all(), "P3b 上界应 ≤ τ0+150"


def test_component_latency_order_is_hard_constrained():
    cw = make_cw(tau0_ms=(270.0, 260.0, 300.0))
    with torch.no_grad():
        _, tau, _ = cw(make_input(B=8))
    assert torch.all(tau[:, 0] < tau[:, 1])
    assert torch.all(tau[:, 1] < tau[:, 2])


def test_soft_alignment_manual():
    """软对齐读取正确：H_c = Σ_t A_c(t)·Z'(t) 手动对照。"""
    cw = make_cw()
    Z = make_input(B=2)
    with torch.no_grad():
        H, _, _, A = cw(Z, return_attention=True)
    H_manual = torch.einsum("bct,btd->bcd", A, Z)
    assert torch.allclose(H, H_manual, atol=1e-5)


def test_asymmetric_skew():
    """D9 不对称：σ_down > σ_up 时，窗右侧（下降沿）比左侧（上升沿）宽（衰减慢）。"""
    cw = make_cw()
    with torch.no_grad():
        # N2：σ_up → lo(20)、σ_down → hi(50)，制造不对称
        cw.sigma_raw[0, 0] = -10.0
        cw.sigma_raw[0, 1] = +10.0
    Z = make_input(B=1)
    with torch.no_grad():
        _, tau, _, A = cw(Z, return_attention=True)
    idx_tau = _time_index(tau[0, 0].item(), cw)
    delta = _time_index(30.0 + cw.tmin, cw)  # 30ms 对应的索引偏移
    delta = delta - _time_index(cw.tmin, cw)
    right = A[0, 0, idx_tau + delta].item()
    left = A[0, 0, idx_tau - delta].item()
    assert right > left, f"σ_down>σ_up 时右侧应比左侧宽（right={right}, left={left}）"


def test_tau0_is_learnable():
    """τ0 是可学习 Parameter（数据驱动初始化，D-tau-param）。"""
    cw = make_cw()
    assert isinstance(cw.tau0, torch.nn.Parameter)
    assert cw.tau0.requires_grad


def test_attention_softargmax_ltau_does_not_supervise_tau0():
    """正式 attention_softargmax 下 L_tau 不监督 tau0。"""
    cw = make_cw(dtau_readout="attention_softargmax")
    Z = make_input(B=4)
    _, tau, _ = cw(Z)
    loss = tau_regularization(tau, cw.tau0_bounded, 50.0)
    loss.backward()
    assert cw.tau0.grad is not None
    assert cw.tau0.grad.abs().sum().item() == pytest.approx(0.0, abs=1e-7)


def test_dtau_mlp_grad_flows():
    """D-mlp-init：global_pool 模式下 dtau_mlp 两层梯度都非零（回归：零初始化会截断第一层梯度）。"""
    cw = make_cw(dtau_readout="global_pool")
    Z = make_input(B=4)
    H, _, _ = cw(Z)
    H.sum().backward()
    assert cw.dtau_mlp[0].weight.grad is not None, "第一层应参与计算图"
    assert cw.dtau_mlp[0].weight.grad.abs().sum() > 0, (
        "dtau_mlp 第一层梯度不应为零（零初始化会截断）"
    )
    assert cw.dtau_mlp[2].weight.grad.abs().sum() > 0, "dtau_mlp 最后一层梯度不应为零"


def test_dtau_readout_variants_shape_and_finite():
    """正式读出与注册的研究对照均应输出有效 (B,3) tau。"""
    for readout in (
        "global_pool",
        "maxmean",
        "attention",
        "attention_softargmax",
    ):
        cw = make_cw(dtau_readout=readout)
        Z = make_input(B=4)
        with torch.no_grad():
            H, tau, sigma = cw(Z)
        assert H.shape == (4, 3, D)
        assert tau.shape == (4, 3)
        assert torch.isfinite(tau).all(), readout
        assert cw.dtau_readout == readout


def test_dtau_readout_invalid_raises():
    with pytest.raises(ValueError):
        ComponentWindow(dtau_readout="bad_readout")


def test_attention_softargmax_has_per_component_query():
    cw = make_cw(dtau_readout="attention_softargmax")
    assert cw.dtau_attn_query.shape == (3, D)
    Z = make_input(B=3)
    H, _, _ = cw(Z)
    H.sum().backward()
    assert cw.dtau_attn_query.grad is not None
    assert cw.dtau_attn_query.grad.abs().sum() > 0


def test_component_window_defaults_use_canonical_constants() -> None:
    cw = make_cw()
    assert tuple(cw.tau0.tolist()) == PCW_CANONICAL_TAU0_MS
    assert tuple(cw.tau0_bounds) == PCW_CANONICAL_TAU0_BOUNDS
    assert tuple(zip(cw.sigma_lo.tolist(), cw.sigma_hi.tolist(), strict=True)) == (
        PCW_CANONICAL_SIGMA_BOUNDS
    )
    assert tuple(zip(cw.dtau_lo.tolist(), cw.dtau_hi.tolist(), strict=True)) == (
        PCW_CANONICAL_DTAU_BOUNDS
    )
