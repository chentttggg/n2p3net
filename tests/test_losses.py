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
from train.contracts import SetMetadata
from train.losses import (
    _bce_with_pos_weight,
    compute_losses,
    estimate_reconstruction_profile,
    gtn_multi_k_cross_entropy,
    gtn_set_cross_entropy,
    heteroscedastic_contrast_nll,
    normalized_contrast_waveform_loss,
    normalized_waveform_reconstruction_loss,
    phase_preserving_band_reconstruction_loss,
    phase_preserving_contrast_loss,
    rbf_mmd2,
    tau_regularization,
    template_projection_loss,
)

TAU0 = torch.tensor([220.0, 300.0, 350.0])


def make_output(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return N2P3NetOutput(
        heads=HeadsOutput(
            logit_target=torch.randn(B, 1, generator=g),
            logit_pcw=torch.randn(B, 1, generator=g),
            logit_early=torch.randn(B, 1, generator=g),
            amplitude=torch.randn(B, 1, generator=g),
        ),
        tau=torch.randn(B, 3, generator=g),
        sigma=torch.randn(3, 2),
        H=torch.randn(B, 3, 64),
        attention=None,
    )


def test_multi_k_digit_loss_uses_online_prefix_and_skips_unavailable_k() -> None:
    digits = torch.tensor([1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    labels = (digits == 1).float()
    logits = torch.zeros(len(digits))
    logits[1:3] = 10.0
    repetition_ranks = torch.tensor([0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    metadata = SetMetadata(
        stimulus_digits=digits,
        group_ids=torch.zeros(len(digits), dtype=torch.long),
        repetition_ranks=repetition_ranks,
        sequence_ranks=torch.arange(len(digits)),
    )
    loss, coverage = gtn_multi_k_cross_entropy(
        logits,
        labels,
        metadata,
        evidence_ks=(1, 2),
        evidence_weights=(0.25, 0.75),
    )
    assert coverage == {1: 1, 2: 0}
    assert loss < 1e-6


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
        out,
        TAU0,
        y,
        lambda4=1.0,
        z_features=z3,
        domain_ids=domain_ids,
        lambda2=0.0,
        lambda3=0.0,
        lambda_amp=0.0,
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
        out,
        TAU0,
        y,
        lambda_jit=0.2,
        tau_shift=tau_shift,
        shift_ms=shift_ms,
        lambda2=0.0,
        lambda3=0.0,
        lambda_amp=0.0,
    )
    assert losses.jit is not None and losses.jit.item() > 0
    expected = 0.2 * losses.jit
    assert torch.allclose(losses.total, losses.target + expected, atol=1e-6)


def test_gtn_fixed_k_set_loss_rewards_correct_digit():
    k = 2
    digits = torch.arange(1, 10).repeat_interleave(k).repeat(2)
    groups = torch.arange(2).repeat_interleave(9 * k)
    targets = torch.tensor([3, 7])
    y = torch.cat([(digits[: 9 * k] == targets[0]), (digits[9 * k :] == targets[1])]).float()
    good = torch.where(y > 0, torch.tensor(2.0), torch.tensor(-2.0)).view(-1, 1)
    good.requires_grad_()
    bad = -good
    good_loss, n_good = gtn_set_cross_entropy(good, y, digits, groups, evidence_k=k)
    bad_loss, n_bad = gtn_set_cross_entropy(bad, y, digits, groups, evidence_k=k)
    assert n_good == n_bad == 2
    assert good_loss < bad_loss
    good_loss.backward()
    assert good.grad is not None and good.grad.abs().sum() > 0


def test_gtn_set_loss_excludes_incomplete_group():
    k = 2
    digits = torch.arange(1, 10).repeat_interleave(k).repeat(2)
    groups = torch.arange(2).repeat_interleave(9 * k)
    y = ((groups == 0) & (digits == 3)) | ((groups == 1) & (digits == 7))
    keep = torch.ones(len(digits), dtype=torch.bool)
    keep[-1] = False
    loss, n_groups = gtn_set_cross_entropy(
        torch.randn(len(digits) - 1, 1),
        y[keep].float(),
        digits[keep],
        groups[keep],
        evidence_k=k,
    )
    assert torch.isfinite(loss)
    assert n_groups == 1


def test_gtn_multi_k_uses_nested_repetition_prefixes():
    kmax = 5
    digits = torch.arange(1, 10).repeat_interleave(kmax)
    ranks = torch.arange(kmax).repeat(9)
    groups = torch.zeros(9 * kmax, dtype=torch.long)
    y = (digits == 4).float()
    logits = torch.where(y > 0, torch.tensor(2.0), torch.tensor(-2.0)).view(-1, 1)
    loss, coverage = gtn_multi_k_cross_entropy(
        logits,
        y,
        SetMetadata(digits, groups, ranks, sequence_ranks=torch.arange(len(digits))),
        evidence_ks=(3, 5),
        evidence_weights=(0.25, 0.75),
    )
    assert torch.isfinite(loss)
    assert coverage == {3: 1, 5: 1}


def test_multi_k_loss_skips_unavailable_ragged_checkpoint():
    digits = torch.arange(1, 10).repeat_interleave(3)
    ranks = torch.arange(3).repeat(9)
    groups = torch.zeros(27, dtype=torch.long)
    loss, coverage = gtn_multi_k_cross_entropy(
        torch.randn(27, 1),
        (digits == 2).float(),
        SetMetadata(
            digits,
            groups,
            ranks,
            sequence_ranks=torch.arange(len(digits)),
        ),
        evidence_ks=(3, 5),
        evidence_weights=(0.5, 0.5),
    )
    assert torch.isfinite(loss)
    assert coverage == {3: 1, 5: 0}


def test_phase_preserving_reconstruction_penalizes_phase_error():
    sfreq = 64.0
    t = torch.arange(64) / sfreq
    target = torch.sin(2 * torch.pi * 2.0 * t).view(1, 1, -1)
    shifted = torch.cos(2 * torch.pi * 2.0 * t).view(1, 1, -1).requires_grad_()
    bands = ((1.0, 4.0),)
    profile = estimate_reconstruction_profile(
        torch.cat([target, -target]),
        torch.tensor([1.0, 0.0]),
        sfreq=sfreq,
        tmin_ms=0.0,
        bands_hz=bands,
        prior_weights=(1.0,),
        channel_mask=torch.tensor([True]),
    )
    aligned_loss = phase_preserving_band_reconstruction_loss(
        target,
        target,
        sfreq=sfreq,
        bands_hz=bands,
        band_scales=profile.band_scales,
        band_weights=profile.band_weights,
    )
    shifted_loss = phase_preserving_band_reconstruction_loss(
        target,
        shifted,
        sfreq=sfreq,
        bands_hz=bands,
        band_scales=profile.band_scales,
        band_weights=profile.band_weights,
    )
    assert aligned_loss.item() == 0.0
    assert shifted_loss > aligned_loss
    shifted_loss.backward()
    assert shifted.grad is not None and shifted.grad.abs().sum() > 0


def test_band_profile_is_normalized_and_fold_local_shape():
    X = torch.randn(20, 3, 256)
    y = torch.tensor([0, 1] * 10)
    bands = ((0.5, 2.0), (2.0, 4.0), (4.0, 8.0))
    profile = estimate_reconstruction_profile(
        X,
        y,
        sfreq=256.0,
        tmin_ms=-200.0,
        bands_hz=bands,
        prior_weights=(0.5, 0.35, 0.15),
        channel_mask=torch.tensor([True, True, False]),
    )
    assert profile.band_scales.shape == profile.band_weights.shape == (3,)
    assert profile.evoked_snr.shape == (3,)
    assert torch.all(profile.band_scales > 0)
    assert torch.allclose(profile.band_weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.equal(profile.evoked_contrast[2], torch.zeros(256))
    assert profile.evoked_target_variance.shape == (3, 256)
    assert torch.all(profile.evoked_target_variance >= 0.0)
    assert torch.equal(profile.evoked_target_variance[2], torch.zeros(256))
    assert profile.bootstrap_samples == 64
    assert profile.split_half_repeats == 16
    assert -1.0 <= profile.split_half_correlation <= 1.0
    assert profile.split_half_nrmse >= 0.0


def test_reconstruction_bootstrap_is_deterministic_and_class_stratified():
    base = torch.linspace(-1.0, 1.0, 64)
    positive = base[None, None] + 0.2 * torch.randn(12, 2, 64)
    negative = -base[None, None] + 0.2 * torch.randn(20, 2, 64)
    X = torch.cat((positive, negative))
    y = torch.cat((torch.ones(12), torch.zeros(20)))
    kwargs = {
        "sfreq": 64.0,
        "tmin_ms": 0.0,
        "bands_hz": ((1.0, 4.0),),
        "prior_weights": (1.0,),
        "bootstrap_samples": 24,
        "split_half_repeats": 8,
        "bootstrap_seed": 7,
    }
    first = estimate_reconstruction_profile(X, y, **kwargs)
    second = estimate_reconstruction_profile(X, y, **kwargs)
    assert torch.equal(first.evoked_target_variance, second.evoked_target_variance)
    assert torch.any(first.evoked_target_variance > 0.0)
    assert first.evoked_target_variance.shape == first.evoked_contrast.shape


def test_faithful_contrast_variance_nll_stops_mean_gradient():
    target = torch.randn(3, 16)
    mean = torch.randn(3, 16, requires_grad=True)
    variance = torch.ones(3, 16, requires_grad=True)
    loss = heteroscedastic_contrast_nll(
        target,
        mean,
        variance,
        channel_mask=torch.ones(3, dtype=torch.bool),
        time_mask=torch.ones(16),
        target_variance=torch.full_like(variance, 0.25),
    )
    loss.backward()
    assert mean.grad is None
    assert variance.grad is not None and variance.grad.abs().sum() > 0


def test_contrast_losses_make_zero_prediction_exactly_one():
    sfreq = 64.0
    times = torch.arange(64) / sfreq
    target = torch.stack(
        (torch.sin(2 * torch.pi * 2.0 * times), torch.cos(2 * torch.pi * 3.0 * times))
    )
    labels = torch.tensor([0, 1] * 8)
    signal = (labels.float() - labels.float().mean())[:, None, None] * target[None]
    profile = estimate_reconstruction_profile(
        signal,
        labels,
        sfreq=sfreq,
        tmin_ms=0.0,
        bands_hz=((1.0, 4.0),),
        prior_weights=(1.0,),
        erp_interval_ms=(0.0, 1000.0),
    )
    template = profile.evoked_contrast
    zero = torch.zeros_like(template, requires_grad=True)
    waveform = normalized_contrast_waveform_loss(
        template,
        zero,
        channel_mask=profile.channel_mask,
        time_mask=profile.time_mask,
    )
    spectrum = phase_preserving_contrast_loss(template, zero, profile)
    projection = template_projection_loss(
        template,
        zero,
        channel_mask=profile.channel_mask,
        time_mask=profile.time_mask,
    )
    assert torch.allclose(waveform, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(spectrum, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(projection, torch.tensor(1.0), atol=1e-6)
    (waveform + spectrum + projection).backward()
    assert zero.grad is not None and zero.grad.abs().sum() > 0.0


def test_waveform_mean_loss_is_robust_huber():
    target = torch.ones(1, 1, 8)
    moderate = target.clone()
    moderate[..., 3] = 2.0
    outlier = target.clone()
    outlier[..., 3] = 20.0
    moderate_loss = normalized_waveform_reconstruction_loss(target, moderate)
    outlier_loss = normalized_waveform_reconstruction_loss(target, outlier)
    assert moderate_loss > 0.0
    assert outlier_loss > moderate_loss
    assert outlier_loss / moderate_loss < 50.0
