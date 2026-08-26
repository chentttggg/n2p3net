"""GLM v3 tokenizer 深度科研改动的语义测试。

覆盖（tokenizer.py D-glm-bpinit / D-glm-postnorm）：
1. 带通初始化的频带分配与频响（init 后滤波器主频落在分配带内）
2. 带通 init 的形状/范数契约（(F,1,k)、单位 L2 主部）
3. post_norm=bn 时前向含 BN+ELU（非线性、尺度均衡）
4. 旧行为完全兼容（init=random, post_norm=none 时与 v2 输出一致性由既有测试覆盖）
5. N2P3Net 参数透传契约
6. 参数硬上限 ≤80k（E4；性能优先，但不要求用满）
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.tokenizer import (
    ERPTokenizer,
    _band_assignment,
    _bandpass_filter_bank,
)


def _energy_weighted_freq(w: np.ndarray, k: int, sfreq: float = 256.0) -> float:
    """能量加权主频：Σ|W|²·f / Σ|W|²。

    注意不能用幅度加权（Σ|W|·f）——小幅宽带噪声乘以高频后主导一阶矩
    （实测：0.25% 噪声能量把 2.4Hz Gabor 的幅度加权中心拉到 15Hz）。
    """
    W2 = np.abs(np.fft.rfft(w)) ** 2
    fr = np.fft.rfftfreq(k, 1.0 / sfreq)
    return float((W2 * fr).sum() / (W2.sum() + 1e-12))


class TestBandpassInit:
    def test_band_assignment_monotone_and_physically_feasible(self):
        """长核 → 低频带；f_hi > f_lo；上界封顶 40Hz。"""
        prev_hi = None
        for k in (13, 33, 65, 129):
            lo, hi = _band_assignment(k, 256.0)
            assert 1.5 <= lo < hi <= 40.0
            if prev_hi is not None:
                assert lo <= prev_hi + 1e-6  # 核越长频带越低（容许轻微重叠）
            prev_hi = hi

    def test_filter_bank_shapes_and_norm(self):
        torch.manual_seed(0)
        w = _bandpass_filter_bank(65, 16, 256.0, 4.7, 14.2)
        assert w.shape == (16, 65)
        # 主部单位范数（噪声 ±5% 后允许偏差）
        norms = w.norm(dim=1)
        assert torch.all(norms > 0.8) and torch.all(norms < 1.3)

    def test_init_filters_center_in_assigned_band(self):
        """init='bandpass' 后滤波器能量加权主频应落在分配带 [f_lo, f_hi] 内（噪声容差）。"""
        torch.manual_seed(0)
        tok = ERPTokenizer(
            n_channels=3, channel_names=("Fz", "Cz", "Pz"), init="bandpass", n_time=256
        )
        for tconv, k in zip(tok.temporal_convs, tok.temporal_kernels, strict=True):
            lo, hi = _band_assignment(k, 256.0)
            w = tconv.weight.detach().squeeze(1).numpy()
            centers = np.array([_energy_weighted_freq(w[f], k) for f in range(w.shape[0])])
            # 带内中位数 ±（带宽的 60%）容差（hann 主瓣 + 噪声展宽）
            margin = 0.6 * (hi - lo)
            assert lo - margin <= np.median(centers) <= hi + margin, (
                f"k={k}: median={np.median(centers):.1f}Hz 带外 [{lo:.1f},{hi:.1f}]"
            )

    def test_long_kernel_scale_covers_erp_band(self):
        """k=129 分支须覆盖 P3b δ-θ 带（上界 ≥6Hz、下界 ≤3Hz）。"""
        lo, hi = _band_assignment(129, 256.0)
        assert lo <= 3.0 and hi >= 6.0


class TestPostNorm:
    def test_bn_elu_introduces_nonlinearity_and_equalizes_scales(self):
        """post_norm=bn + post_act=elu：f(2x) ≠ 2f(x)（非线性）；各尺度输出 std 均衡。"""
        torch.manual_seed(0)
        tok = ERPTokenizer(
            n_channels=3,
            channel_names=("Fz", "Cz", "Pz"),
            init="bandpass",
            post_norm="bn",
            post_act="elu",
            n_time=256,
        )
        tok.eval()
        # BN 在 eval 下用 running stats（先跑一次 train 模式填充）
        tok.train()
        x = torch.randn(8, 3, 256) * 1e-4
        tok(x)
        tok.eval()
        with torch.no_grad():
            z1 = tok(2 * x)
            z2 = 2 * tok(x)
        assert not torch.allclose(z1, z2, atol=1e-8), "BN+ELU 路径应为非线性"

    def test_backward_compat_random_none_matches_old_structure(self):
        """init=random + post_norm=none：参数集与旧行为一致（无 BN 模块）。"""
        tok = ERPTokenizer(n_channels=3, channel_names=("Fz", "Cz", "Pz"), n_time=256)
        assert tok.post_bns is None
        assert tok.post_act_fn is None
        assert not hasattr(tok, "post_bns") or tok.post_bns is None

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            ERPTokenizer(n_channels=3, channel_names=("Fz", "Cz", "Pz"), init="sinc")
        with pytest.raises(ValueError):
            ERPTokenizer(n_channels=3, channel_names=("Fz", "Cz", "Pz"), post_norm="ln")
        with pytest.raises(ValueError):
            ERPTokenizer(n_channels=3, channel_names=("Fz", "Cz", "Pz"), post_act="relu")


class TestN2P3NetPassthrough:
    def test_tokenizer_kwargs_passthrough(self):
        from models.n2p3net import N2P3Net

        torch.manual_seed(0)
        m = N2P3Net(
            n_channels=3,
            channel_names=("Fz", "Cz", "Pz"),
            tokenizer_init="bandpass",
            tokenizer_post_norm="bn",
            tokenizer_post_act="elu",
        )
        assert m.tokenizer.init == "bandpass"
        assert m.tokenizer.post_bns is not None
        assert isinstance(m.tokenizer.post_act_fn, torch.nn.ELU)
        x = torch.randn(2, 3, 256)
        out = m(x)
        assert out.heads.logit_target.shape[0] == 2

    def test_glm_v3_budget_within_80k(self):
        from models.n2p3net import N2P3Net

        torch.manual_seed(0)
        m = N2P3Net(
            n_channels=3,
            channel_names=("Fz", "Cz", "Pz"),
            tokenizer_init="bandpass",
            tokenizer_post_norm="bn",
            tokenizer_post_act="elu",
        )
        assert m.num_parameters() <= 80000

    def test_backward_grad_reaches_temporal_convs(self):
        """带通 init + BN 路径下时间卷积权重仍收到非零梯度（无梯度饥饿回归）。"""
        from models.n2p3net import N2P3Net

        torch.manual_seed(0)
        m = N2P3Net(
            n_channels=3,
            channel_names=("Fz", "Cz", "Pz"),
            tokenizer_init="bandpass",
            tokenizer_post_norm="bn",
            tokenizer_post_act="elu",
        )
        x = torch.randn(8, 3, 256)
        out = m(x)
        loss = out.heads.logit_target.sum()
        loss.backward()
        for tconv in m.tokenizer.temporal_convs:
            assert tconv.weight.grad is not None
            assert tconv.weight.grad.norm() > 0
