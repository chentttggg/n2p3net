"""模块 #11 测试：N2P3-Net 完整模型组装。

冒烟：形状 / dtype / 无 NaN / 可选元数据 / 关闭再参考 / return_attention。
语义：
    - 基线段标准化正确（基线均值≈0、std≈1）。
    - 端到端反向传播（Phase C 集成联调核心：全链路梯度非零、无 NaN）。
    - 参数账（D-budget 透明）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.n2p3net import N2P3Net

C = 8
D = 64
T = 256
D_CHN = 48
D_SUB = 19


def make_model(**kw):
    torch.manual_seed(0)
    return N2P3Net(**kw)


def make_inputs(B=4, T=T, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, C, T, generator=g)
    E_chn = torch.randn(C, D_CHN, generator=g)
    E_sub = torch.randn(D_SUB, generator=g)
    return X, E_chn, E_sub


# ---------------- 冒烟测试 ----------------


def test_forward_shape():
    model = make_model()
    X, E_chn, E_sub = make_inputs(B=4)
    out = model(X, E_chn, E_sub)
    assert out.heads.logit_target.shape == (4, 1)
    assert out.heads.logit_early.shape == (4, 1)
    assert out.heads.amplitude.shape == (4, 1)
    assert out.tau.shape == (4, 3)
    assert out.sigma.shape == (3, 2)
    assert out.H.shape == (4, 3, D)
    assert out.attention is None


def test_forward_dtype():
    model = make_model()
    X, E_chn, E_sub = make_inputs(B=2)
    out = model(X, E_chn, E_sub)
    assert out.heads.logit_target.dtype == torch.float32


def test_forward_no_nan():
    model = make_model()
    X, E_chn, E_sub = make_inputs(B=4)
    out = model(X, E_chn, E_sub)
    assert not torch.isnan(out.heads.logit_target).any()
    assert not torch.isnan(out.tau).any()
    assert not torch.isnan(out.H).any()


def test_forward_without_metadata():
    model = make_model()
    X, _, _ = make_inputs(B=4)
    out = model(X)  # E_chn/E_sub 为 None
    assert out.heads.logit_target.shape == (4, 1)


def test_tau0_ms_forwarded_to_component_window():
    """方案 B：N2P3Net 可透传 τ0 先验（GTN 儿童数据用 460ms P3b）。"""
    model = make_model(tau0_ms=(220.0, 300.0, 460.0))
    assert torch.equal(model.component_window.tau0.data, torch.tensor([220.0, 300.0, 460.0]))


def test_tau0_bounds_forwarded_to_component_window():
    model = make_model(tau0_bounds=((180.0, 280.0), (250.0, 380.0), (350.0, 600.0)))
    assert model.component_window.tau0_hi[2].item() == 600.0
    assert model.component_window.tau0_lo[2].item() == 350.0


def test_bypass_modes_are_switchable():
    """v5.1：separable_pool 默认、mean_pool/none 可回退，输出结构一致。"""
    torch.manual_seed(0)
    model_on = N2P3Net()
    torch.manual_seed(0)
    model_off = N2P3Net(bypass_mode="none")
    X, E_chn, E_sub = make_inputs(B=3)
    out_on = model_on(X, E_chn, E_sub)
    out_off = model_off(X, E_chn, E_sub)
    assert not torch.allclose(out_on.heads.logit_target, out_off.heads.logit_target)
    assert out_off.heads.logit_early.shape == (3, 1)
    assert out_off.heads.amplitude.shape == (3, 1)


def test_native_3ch_forward():
    """v5.1：GTN 原生 3 导模型可前向，且参数预算仍 ≤50k。"""
    model = make_model(n_channels=3, channel_names=("Fz", "Cz", "Pz"))
    X = torch.randn(4, 3, T)
    E_chn = torch.randn(3, D_CHN)
    out = model(X, E_chn)
    assert out.heads.logit_target.shape == (4, 1)
    assert out.tau.shape == (4, 3)
    assert model.num_parameters() <= 50000


def test_old_mean_pool_mlp_revert_path():
    """接口回退：mean_pool + MLP 头 + 无 max-norm + 旧 dropout 可构造并可前向。"""
    model = make_model(
        bypass_mode="mean_pool",
        head_mlp=True,
        spatial_max_norm=None,
        encoder_dropout=0.1,
    )
    X, E_chn, E_sub = make_inputs(B=2)
    out = model(X, E_chn, E_sub)
    assert out.heads.logit_target.shape == (2, 1)
    assert model.bypass_mode == "mean_pool"
    assert model.tokenizer.spatial_max_norm is None


def test_forward_no_rereference():
    model = make_model(use_rereference=False)
    X, E_chn, E_sub = make_inputs(B=3)
    out = model(X, E_chn, E_sub)
    assert out.H.shape == (3, 3, D)


def test_return_attention():
    model = make_model()
    X, E_chn, E_sub = make_inputs(B=2)
    out = model(X, E_chn, E_sub, return_attention=True)
    assert out.attention.shape == (2, 3, T)


# ---------------- 语义测试 ----------------


def test_baseline_standardize():
    """基线段标准化（D-baseline）：前 51 点标准化后均值≈0、std≈1。"""
    model = make_model()
    g = torch.Generator().manual_seed(1)
    X = torch.randn(2, C, T, generator=g) * 2.0 + 5.0  # 均值 5、std 2
    X0 = model._baseline_standardize(X)
    b = X0[:, :, :51]
    assert torch.allclose(b.mean(dim=2), torch.zeros(2, C), atol=1e-3), "基线均值应≈0"
    assert torch.allclose(b.std(dim=2), torch.ones(2, C), atol=1e-3), "基线 std 应≈1"


def test_end_to_end_backward():
    """端到端反向传播（Phase C 核心）：全链路梯度非零、无 NaN。"""
    model = make_model()
    X, E_chn, E_sub = make_inputs(B=4)
    y = torch.randint(0, 2, (4, 1)).float()
    out = model(X, E_chn, E_sub)
    loss = F.binary_cross_entropy_with_logits(out.heads.logit_target, y)
    loss.backward()

    assert model.tokenizer.pointwise.weight.grad is not None
    assert model.component_window.tau0.grad is not None
    assert model.heads.head_a[1].weight.grad is not None
    assert not torch.isnan(model.tokenizer.pointwise.weight.grad).any()
    assert model.component_window.tau0.grad.abs().sum() > 0, "τ 应有非零梯度（D8）"


def test_baseline_n_derived_from_tmin_sfreq():
    """review v6 P1：baseline_n=None 时由 tmin/sfreq 推导，不再硬编码 51。"""
    model = make_model(tmin=-200.0, sfreq=256.0)
    assert model.baseline_n == 51
    model2 = make_model(tmin=-100.0, sfreq=250.0)
    assert model2.baseline_n == 25


def test_parameter_budget():
    """参数账（D-budget）：默认配置（TCN depth=3）应 ≤ E4 上限 50k。"""
    model = make_model()
    n = model.num_parameters()
    assert n <= 50000, f"默认参数 {n} 应 ≤ E4 上限 50k"


def test_nan_channels_handled():
    """review P0 修复：缺失通道 NaN 不毒化 logit/tau（入口 nan_to_num + mask 重归一化）。"""
    model = make_model()
    X = torch.randn(2, C, T)
    X[:, 3:, :] = float("nan")  # P3/P4/PO7/PO8/Oz 缺失（GTN 3 导场景）
    E_chn = torch.randn(C, D_CHN)
    E_sub = torch.randn(D_SUB)
    channel_mask = torch.tensor([True, True, True, False, False, False, False, False])
    out = model(X, E_chn, E_sub, channel_mask=channel_mask)
    assert not torch.isnan(out.heads.logit_target).any(), "logit 不应含 NaN"
    assert not torch.isnan(out.tau).any(), "tau 不应含 NaN"
    assert not torch.isnan(out.H).any(), "H 不应含 NaN"


def test_no_phantom_channel_after_stage0():
    """幻象通道回归（v4）：缺失通道经完整 Stage 0（reference + 基线段标准化）后必须恒 0。
    若 reference 对所有通道减 m，缺失通道的 −m(t) 会被基线标准化（÷m 的小 std）放大到
    std≈1，且逐试次变化——5 个缺失位置变成同一幻象的副本，冒充枕/顶区地形证据。"""
    model = make_model()
    X = torch.randn(4, C, T)
    channel_mask = torch.tensor([True, True, True, False, False, False, False, False])
    X[:, ~channel_mask, :] = 0.0  # 零填充（nan_to_num 后）

    X0 = model.reference(X, channel_mask)
    X1 = model._baseline_standardize(X0)
    missing = X1[:, ~channel_mask, :]
    assert (missing == 0.0).all(), (
        f"缺失通道经 Stage 0 后应恒 0，实测 std={missing.std(dim=2).mean().item():.4f}（幻象通道）"
    )


def test_missing_channel_no_phantom():
    """review 复审：缺失通道经 Stage 0 后保持 0（不被减 m 成幻象通道）。"""
    model = make_model()
    X = torch.randn(2, C, T)
    X[:, 3:, :] = 0.0  # 缺失通道填 0
    mask = torch.tensor([True, True, True, False, False, False, False, False])
    X0 = model.reference(X, mask)
    X0 = model._baseline_standardize(X0)
    # 缺失通道（索引 3-7）经 Stage 0 后应保持 0（不被 −m 放大成幻象）
    assert X0[:, 3:, :].abs().max() < 1e-5, "缺失通道经 Stage 0 后应保持 0"
