"""模块 #11 测试：N2P3-Net 完整模型组装。

冒烟：形状 / dtype / 无 NaN / 可选元数据 / 关闭再参考 / return_attention。
语义：
    - 基线段标准化正确（基线均值≈0、std≈1）。
    - 端到端反向传播（Phase C 集成联调核心：全链路梯度非零、无 NaN）。
    - 参数账（D-budget 透明）。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.n2p3net import N2P3Net
from models.repetition_v12 import AdditiveRepetitionEvidence

C = 8
D = 64
T = 256
D_CHN = 48
D_SUB = 19


def make_model(**kw):
    torch.manual_seed(0)
    # This fixture exercises the complete opt-in research surface. Production
    # and raw-constructor defaults are checked separately below.
    kw.setdefault("component_decoder", True)
    kw.setdefault("use_innovation_likelihood", True)
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


def test_forward_rejects_integer_channel_mask_and_nonbinary_likelihood_labels():
    model = make_model(use_innovation_likelihood=True)
    X, E_chn, E_sub = make_inputs(B=2)
    with pytest.raises(ValueError, match="boolean dtype"):
        model(X, E_chn, E_sub, channel_mask=torch.ones(C, dtype=torch.int64))
    with pytest.raises(ValueError, match="binary values"):
        model(X, E_chn, E_sub, likelihood_labels=torch.tensor([0.0, 1.2]))


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




def test_repetition_v12_opt_in_constructs_additive_evidence():
    model = make_model(use_repetition_evidence=True, repetition_v12=True)
    assert isinstance(model.repetition_evidence, AdditiveRepetitionEvidence)


def test_repetition_v12_model_forwards_with_additive_evidence():
    model = make_model(use_repetition_evidence=True, repetition_v12=True)
    X, E_chn, E_sub = make_inputs(B=4)
    out = model(X, E_chn, E_sub)
    assert torch.isfinite(out.heads.logit_target).all()
    assert isinstance(model.repetition_evidence, AdditiveRepetitionEvidence)


def test_measurement_window_contribution_is_detached():
    model = make_model(use_measurement_windows=True)
    X, E_chn, E_sub = make_inputs(B=4)
    window = torch.rand(4, T)
    out = model(X, E_chn, E_sub, measurement_window=window)

    assert out.measurement_features.shape == (4, D)
    out.heads.logit_target.sum().backward()
    assert window.grad is None
    assert model.measurement_head[1].weight.grad is not None


def test_zero_measurement_window_contributes_exactly_nothing():
    model = make_model(use_measurement_windows=True).eval()
    X, E_chn, E_sub = make_inputs(B=4)
    base = model(X, E_chn, E_sub).heads.logit_target
    with_window = model(
        X, E_chn, E_sub, measurement_window=torch.zeros(4, T)
    ).heads.logit_target
    assert torch.allclose(base, with_window, atol=1e-7)


def test_measurement_window_requires_opt_in():
    model = make_model()
    X, E_chn, E_sub = make_inputs(B=2)
    with pytest.raises(ValueError, match="use_measurement_windows"):
        model(X, E_chn, E_sub, measurement_window=torch.rand(2, T))
def test_raw_model_defaults_fail_closed() -> None:
    model = N2P3Net()
    assert model.component_decoder is None
    assert model.innovation_encoder is None
    assert model.innovation_decoder is None
    assert model.encoder.depth == 4
    assert model.encoder.encoder_type == "tcn"


def test_encoder_depth_is_coupled_through_the_full_model() -> None:
    model = make_model(encoder_depth=6)
    assert model.encoder.depth == 6
    assert model.encoder.tcn_dilations == (1, 4, 16, 32, 64, 128)
    assert len(model.encoder.blocks) == 6


def test_tau0_ms_forwarded_to_component_window():
    """方案 B：N2P3Net 可透传 τ0 先验（GTN 儿童数据用 460ms P3b）。"""
    model = make_model(tau0_ms=(220.0, 300.0, 460.0))
    assert torch.equal(model.component_window.tau0.data, torch.tensor([220.0, 300.0, 460.0]))


def test_tau0_bounds_forwarded_to_component_window():
    model = make_model(tau0_bounds=((180.0, 280.0), (250.0, 380.0), (350.0, 600.0)))
    assert model.component_window.tau0_hi[2].item() == 600.0
    assert model.component_window.tau0_lo[2].item() == 350.0


def test_deleted_bypass_api_is_rejected():
    with pytest.raises(TypeError, match="bypass_mode"):
        N2P3Net(bypass_mode="none")


def test_native_3ch_forward():
    """GTN 原生 3 导模型可前向，且不超过 80k 硬上限。"""
    model = make_model(n_channels=3, channel_names=("Fz", "Cz", "Pz"))
    X = torch.randn(4, 3, T)
    E_chn = torch.randn(3, D_CHN)
    out = model(X, E_chn)
    assert out.heads.logit_target.shape == (4, 1)
    assert out.tau.shape == (4, 3)
    assert model.num_parameters() <= 80000


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
    assert out.erp.reconstruction.shape == (2, C, T)
    assert out.erp.amplitude_mean.shape == (2, 3, C)
    assert out.erp.amplitude_variance.shape == (2, 3, C)
    assert out.erp.null_variance.shape == (C,)
    assert torch.all(out.erp.null_variance > 0)
    assert out.likelihood.likelihood_observation.shape == (2, C, T)
    assert out.likelihood.observation_mask.shape == (2, C)
    assert out.likelihood.causal_innovation.history_correction.shape == (2, 2, C, T)
    assert out.likelihood.causal_innovation.log_variance_scale.shape == (2, 2, T, C)


def test_sparse_morphology_dictionary_bounds_and_latency_semantics():
    model = make_model().eval()
    X, E_chn, E_sub = make_inputs(B=2)
    with torch.no_grad():
        out = model(X, E_chn, E_sub)
    erp = out.erp
    theta = erp.morphology_parameters
    assert erp.morphology_basis.shape == (2, 3, T)
    assert torch.all((1.0 <= theta[..., 3]) & (theta[..., 3] <= 4.0))
    assert torch.all((40.0 <= theta[..., 4]) & (theta[..., 4] <= 200.0))
    coefficient_bounds = torch.tensor([0.5, 0.7, 0.5])
    assert torch.all(erp.atom_coefficients.abs() <= coefficient_bounds + 1e-6)
    assert torch.all((0.0 <= erp.atom_gates) & (erp.atom_gates <= 1.0))
    assert torch.all((0.0 <= erp.expected_l0) & (erp.expected_l0 <= 1.0))
    assert torch.equal(erp.anchor_latency_ms, out.tau)
    assert torch.all(out.tau[:, 0] < out.tau[:, 1])
    assert torch.all(out.tau[:, 1] < out.tau[:, 2])
    assert erp.component_peak_latency_ms.shape == (2, 3)
    assert erp.waveform_peak_latency_ms.shape == (2, C)
    assert torch.all(erp.waveform_variance > 0.0)


def test_variance_path_is_faithfully_isolated_from_mean_trunk():
    model = make_model().eval()
    X, E_chn, E_sub = make_inputs(B=2)
    out = model(X, E_chn, E_sub)
    out.erp.waveform_variance.mean().backward()

    mean_modules = (
        model.tokenizer,
        model.encoder,
        model.component_window,
        model.component_decoder.amplitude_heads,
        model.component_decoder.morphology_heads,
    )
    assert all(
        parameter.grad is None for module in mean_modules for parameter in module.parameters()
    )
    variance_modules = (
        model.component_decoder.variance_heads,
        model.component_decoder.morphology_variance_heads,
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for module in variance_modules
        for parameter in module.parameters()
    )


def test_deployment_uncertainty_obeys_total_variance_identity():
    model = make_model().eval()
    X, E_chn, E_sub = make_inputs(B=1)
    prediction = model.predict_erp_uncertainty(X, E_chn, E_sub, mc_samples=2)
    assert prediction.mean.shape == (1, C, T)
    assert prediction.anchor_latency_ms.shape == (1, 3)
    assert prediction.waveform_peak_latency_ms.shape == (1, C)
    assert torch.all(prediction.aleatoric_variance > 0.0)
    assert torch.all(prediction.epistemic_variance >= 0.0)
    assert torch.allclose(
        prediction.total_variance,
        prediction.aleatoric_variance + prediction.epistemic_variance,
    )
    aggregate = prediction.aggregate_trials()
    assert aggregate.mean.shape == (C, T)
    assert aggregate.variance.shape == (C, T)
    assert torch.allclose(aggregate.mean, prediction.mean[0])
    assert torch.all(aggregate.effective_sample_size == 1.0)


def test_component_decoder_is_pcw_constrained():
    model = make_model()
    X, E_chn, E_sub = make_inputs(B=2)
    out = model(X, E_chn, E_sub, return_attention=True)
    loss = out.erp.reconstruction.square().mean()
    loss.backward()
    decoder_grad = sum(
        p.grad.abs().sum() for p in model.component_decoder.parameters() if p.grad is not None
    )
    assert decoder_grad > 0
    assert model.component_window.tau0.grad is not None


def test_likelihood_does_not_depend_on_optional_erp_decoder() -> None:
    model = make_model(component_decoder=False)
    X, E_chn, E_sub = make_inputs(B=2)
    output = model(X, E_chn, E_sub)
    assert output.erp is None
    assert output.likelihood is not None
    assert output.likelihood.likelihood_observation.shape == (2, C, T)


def test_innovation_gradient_is_strictly_isolated_from_pcw_path():
    model = make_model()
    X, E_chn, E_sub = make_inputs(B=4)
    out = model(X, E_chn, E_sub)
    out.likelihood.causal_innovation.factor_scale.sum().backward()
    pcw_modules = (model.tokenizer, model.encoder, model.component_window, model.component_decoder)
    assert all(
        parameter.grad is None for module in pcw_modules for parameter in module.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.innovation_encoder.parameters()
    )


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
    model = make_model(tmin_ms=-200.0, tmax_ms=800.0, sfreq=256.0)
    assert model.baseline_n == 51
    model2 = make_model(tmin_ms=-100.0, tmax_ms=924.0, sfreq=250.0)
    assert model2.baseline_n == 25


def test_trial_reference_uses_explicit_physical_window_without_scaling():
    model = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        tmin_ms=0.0,
        tmax_ms=1000.0,
        sfreq=256.0,
        n_time=256,
        baseline_mode="trial_reference",
        trial_reference_window_ms=(0.0, 50.0),
        trial_reference_center="mean",
        trial_reference_scale="none",
    )
    X = torch.arange(3 * 256, dtype=torch.float32).reshape(1, 3, 256)
    transformed = model._baseline_standardize(X)
    reference = X[:, :, :13].mean(dim=2, keepdim=True)
    assert model.trial_reference_slice == (0, 13)
    assert torch.allclose(transformed, X - reference)


def test_trial_reference_can_use_robust_median_mad():
    model = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        tmin_ms=0.0,
        tmax_ms=1000.0,
        sfreq=256.0,
        n_time=256,
        baseline_mode="trial_reference",
        trial_reference_window_ms=(0.0, 50.0),
        trial_reference_center="median",
        trial_reference_scale="mad",
    )
    X = torch.ones(1, 3, 256)
    X[:, :, 0] = 100.0
    transformed = model._baseline_standardize(X)
    assert torch.isfinite(transformed).all()
    assert transformed[:, :, 0].mean() > transformed[:, :, 1:].mean()


def test_parameter_budget():
    """完整默认模型不得超过 80k；上限不是要求用满的目标。"""
    model = make_model()
    n = model.num_parameters()
    assert n <= 80000, f"默认参数 {n} 超过 E4 的 80k 硬上限"


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


def test_trial_specific_masks_reach_canonical_and_residual_paths():
    """Dynamic dropout is missingness, not a zero-valued GP observation."""

    model = make_model(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        canonical_channel_names=("Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"),
        d_model=16,
        temporal_kernels=(13,),
        filters_per_scale=2,
        encoder_depth=1,
        innovation_d_model=8,
        innovation_dilations=(1,),
    )
    epochs = torch.randn(2, 3, T)
    channel_mask = torch.tensor([[True, True, True], [True, False, True]], dtype=torch.bool)
    epochs[1, 1] = 0.0
    output = model(epochs, channel_mask=channel_mask)

    variance = torch.diagonal(output.canonical_covariance, dim1=-2, dim2=-1)
    assert variance[1].mean() > variance[0].mean()
    assert torch.equal(
        output.likelihood.likelihood_observation[1, 1],
        torch.zeros_like(output.likelihood.likelihood_observation[1, 1]),
    )


def test_likelihood_path_is_end_to_end_strict_past_despite_offline_erp_path():
    torch.manual_seed(23)
    model = make_model().eval()
    X, E_chn, E_sub = make_inputs(B=2)
    changed = X.clone()
    changed[:, :, 128:] += 50.0 * torch.randn_like(changed[:, :, 128:])
    class_means = torch.randn(2, C, T)

    with torch.no_grad():
        original = model(
            X, E_chn, E_sub, likelihood_class_means=class_means
        ).likelihood.causal_innovation
        intervened = model(
            changed, E_chn, E_sub, likelihood_class_means=class_means
        ).likelihood.causal_innovation

    assert torch.allclose(
        original.history_correction[:, :, :, :129],
        intervened.history_correction[:, :, :, :129],
        atol=1e-6,
    )
    assert torch.allclose(
        original.log_variance_scale[:, :, :129],
        intervened.log_variance_scale[:, :, :129],
        atol=1e-6,
    )
    assert torch.allclose(
        original.factor_scale[:, :, :129], intervened.factor_scale[:, :, :129], atol=1e-6
    )


def test_production_model_has_no_z2_aux_head() -> None:
    model = N2P3Net()
    assert model.z2_aux_head is None
    assert model.z2_aux_head_mode == "off"


def test_z2_aux_add_and_replace_research_modes_keep_pcw_readouts() -> None:
    for mode in ("add", "replace"):
        model = make_model(
            n_channels=C,
            channel_names=("Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"),
            use_z2_aux_head=True,
            z2_aux_head_mode=mode,
            z2_aux_pool="attention",
        ).eval()
        X, E_chn, E_sub = make_inputs(B=3)
        out = model(X, E_chn, E_sub)
        assert out.heads.logit_aux is not None
        assert out.heads.logit_aux.shape == (3, 1)
        assert out.heads.logit_pcw.shape == (3, 1)
        assert out.tau.shape == (3, 3)
        assert out.H.shape == (3, 3, D)
        if mode == "replace":
            assert torch.allclose(out.heads.logit_target, out.heads.logit_aux)
        else:
            assert torch.allclose(
                out.heads.logit_target, out.heads.logit_pcw + out.heads.logit_aux
            )


def test_z2_aux_head_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="add.*replace"):
        make_model(use_z2_aux_head=True, z2_aux_head_mode="stack")
