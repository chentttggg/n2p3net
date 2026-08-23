"""模块 #11：N2P3-Net 完整模型组装（把 Stage 0–3 串起来）。

职责：
    组装加权再参考（Stage 0.1）+ 基线段标准化（Stage 0.2/0.4，本模块内嵌实现，见 D-baseline）+
    tokenizer（Stage 1）+ encoder（Stage 2 序列编码）+ component_window（Stage 2 成分窗）+
    heads（Stage 3），提供端到端 forward：(B, C, T) + E_chn + E_sub → 全部输出。

明确「不做」：
    - 决策层（decision.py）：纯 numpy 后处理、无梯度，不在模型内；推理时单独调用
      decide(logit_target, digits, subject_ids)。
    - 参考抖动增强（Stage 0.3）：训练期数据增强，归 train/augment.py。
    - 损失构造（train/losses.py）。

三思决策记录（供后续会话追溯）：
    D-baseline      Stage 0.2（基线校正）+ 0.4（基线段归一化）**合并**为一次「基线段标准化」
                    (X_ref − μ_b) / σ_b，其中 μ_b/σ_b 取前 baseline_n 点（默认 51 = −200~0ms@256Hz）的
                    逐通道均值/std。理由：两步本质是同一「基线段 z-score」，合并少一次广播、语义更清晰。
                    按 blueprint 0.4 用基线段（非全窗 InstanceNorm）——全窗会把 target 的 P300 大幅值
                    抬高 std、压缩判别对比（v3 盲区）。全窗 InstanceNorm 留消融轴（Phase 5）。
    D-std-clamp     σ_b clamp 到 ≥1e-6 防除零（静默/坏通道基线平坦时 σ_b=0）。
    D-ref-optional  weighted rereference 默认启用（Stage 0.1），use_rereference=False 可关（消融）。
    D-output-struct 统一输出 N2P3NetOutput（含 heads 输出 + tau/sigma/H/A），供 losses 与可解释性消费；
                    A（窗分布）默认不返回（省内存），return_attention=True 时返回。
    D-budget        参数账：默认配置 = encoder **TCN depth=3**（2026-08-20 用户决策）→ 全模型 ≈38k，
                    符合 E4 50k。Conformer（depth=1 ≈58k 超预算）保留作消融对照、不入默认。决策理由：
                    TCN 膨胀卷积（dilation 1/4/16）感受野 ~168ms 恰好覆盖 N2→P3b ~150ms 关系，
                    而「容量非瓶颈、域差才是」（blueprint D6），无需自注意力的 O(T²) 参数。参见
                    blueprint D6 / constitution E4 修订记录。
    D-device        纯 nn.Module，不硬编码设备（DP1）。
    D-bypass        v5.1（EEGNet 借鉴）：bypass_mode="separable_pool" 默认——Z' 经 4× 池化 →
                    depthwise+pointwise separable conv → adaptive 8 bin 池化 → Dropout+Linear 头。
                    "mean_pool" 保留旧方案 B 旁路；"none" 关闭旁路（接口回退）。
                      E5 只约束可解释路径（PCW 仍吃全 T Z'），判别旁路允许池化。
      D-slim-head     Head-A/Head-B 默认 Dropout(0.25)+Linear；use_mlp_heads=True 回退旧 MLP。

契约（输入 → 输出）：
    X (B,C,T)、E_chn (C,d_chn_in) 可选、E_sub (d_sub_in,) 可选 → N2P3NetOutput。

依赖的决策：blueprint §2–§5、constitution P2/P3/P7、reference/tokenizer/encoder/
    component_window/heads 各模块契约。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn

from models.bypass import TemporalPoolingBypass
from models.reference import WeightedRereference
from models.tokenizer import ERPTokenizer
from models.encoder import Stage2Encoder
from models.component_window import ComponentWindow
from models.heads import MultiTaskHeads, HeadsOutput


@dataclass
class N2P3NetOutput:
    """N2P3-Net 完整前向输出。

    Attributes
    ----------
    heads : HeadsOutput
        多任务头输出（logit_target / logit_early / amplitude，及 p_target/p_early property）。
    tau : torch.Tensor
        (B,3) 成分潜伏期（ms）。
    sigma : torch.Tensor
        (3,2) 成分窗宽（ms；[:,0]=σ_up、[:,1]=σ_down）。
    H : torch.Tensor
        (B,3,D) 成分表示（可解释性）。
    attention : torch.Tensor | None
        (B,3,T) 软窗分布（return_attention=True 时；可解释性 / Head-D 精确幅值用）。
    """

    heads: Optional[HeadsOutput]
    tau: torch.Tensor
    sigma: torch.Tensor
    H: torch.Tensor
    attention: Optional[torch.Tensor]
    features: Optional[torch.Tensor] = None


class N2P3Net(nn.Module):
    """完整模型：(B,C,T) → N2P3NetOutput。

    Parameters
    ----------
    n_channels : int
      channel_names : Sequence[str] | None
          通道名（非 8 导蒙太奇必填；GTN 原生 3 导传 ("Fz","Cz","Pz")）。
      bypass_mode : str
          Head-A 判别旁路：separable_pool（默认，EEGNet 式 4×/8× 池化 + depthwise separable）、
          mean_pool（旧方案 B）、none（无旁路）。开关式接口便于回退。
      head_mlp : bool
          True 恢复旧版 Head-A/Head-B 的 D//2 隐藏层 MLP；默认 False 用 Dropout+Linear 瘦头。
      head_dropout / encoder_dropout : float
          瘦头 dropout 与 encoder dropout（默认 0.25，EEGNet 借鉴；旧默认 0.1 可显式回退）。
      spatial_max_norm : float | None
          tokenizer 空间权重 max-norm（默认 1.0；None 恢复旧行为）。
        通道数 C（默认 8）。
    d_model : int
        隐藏维 D（默认 64）。
    use_rereference : bool
        是否启用加权再参考（Stage 0.1，默认 True）。
    baseline_n : int | None
        基线段标准化用的基线点数。None 时由 tmin/sfreq 推导（默认 −200ms@256Hz → 51 点，
        review v6 P1 消除硬编码与 T/tmin 的耦合）。
    tau0_bounds : Sequence[tuple[float, float]] | None
        每成分 τ0 的生理界（ms），None 使用 ComponentWindow 默认（review v6 P1）。
      tau0_ms : Sequence[float] | None
          每成分 τ0 先验中心（ms），None 使用 ComponentWindow 默认（成人先验 220/300/350）。
          GTN（儿童）由 runner 显式传 220/300/460（Phase 2 失败诊断：真实 P3b 峰值 ~460–490ms）。
      global_bypass : bool
          兼容旧接口：True/False 等价于 bypass_mode="mean_pool"/"none"（仅当 bypass_mode 未显式给定时）。
    temporal_kernels : Sequence[int]
        tokenizer 多尺度时间卷积核长。
    filters_per_scale : int
        tokenizer 每尺度滤波器数。
    encoder_depth : int
        Stage 2 序列编码层数（消融轴，默认 3，配合 TCN dilation 1/4/16）。
    encoder_type : str
        "tcn"（默认，2026-08-20 决策）或 "conformer"（备选，消融对照）。
    d_chn_in / d_sub_in : int
        E_chn / E_sub 输入维（与 data/channel、data/metadata 对齐）。
      dtau_readout : str
          Δτ 读出路径（component_window.DTAU_READOUTS，默认 attention_direct）。
      encoder_norm : str
          TCN block 归一化（GLM 消融轴）："ln"（默认）或 "bn"（BatchNorm1d，跨被试
          P300 文献中 BN 是 CNN 泛化关键组件的假设检验入口）。
      sigma_bounds : Sequence[tuple[float, float]] | None
          PCW 每成分窗宽 σ 的 [lo, hi]（ms），None 用 ComponentWindow 默认
          （成人先验 N2 [20,50]、P3a/P3b [20,80]）。GTN 儿童的 P3b 宽达 300–650ms
          （ERP 实测 350–650ms 窗差值仍 11–14μV），runner 应传 P3b 上界 150。
    """

    def __init__(
        self,
        n_channels: int = 8,
        channel_names: Optional[Sequence[str]] = None,
        d_model: int = 64,
        use_rereference: bool = True,
        baseline_n: Optional[int] = None,
        tmin: float = -200.0,
        tmax: float = 800.0,
        sfreq: float = 256.0,
        n_time: int = 256,
        temporal_kernels: Sequence[int] = (13, 33, 65, 129),
        filters_per_scale: int = 16,
        encoder_depth: int = 3,
        encoder_type: str = "tcn",
        encoder_dropout: float = 0.25,
        encoder_causal: bool = False,
        d_chn_in: int = 48,
        d_sub_in: int = 19,
        n_domains: Optional[int] = None,
        tau0_bounds: Optional[Sequence[tuple[float, float]]] = None,
        tau0_ms: Optional[Sequence[float]] = None,
        dtau_readout: str = "attention_direct",
        dtau_bounds: Optional[Sequence[tuple[float, float]]] = None,
        sigma_bounds: Optional[Sequence[tuple[float, float]]] = None,
        encoder_norm: str = "ln",
        bypass_mode: str = "separable_pool",
        global_bypass: bool = True,
        head_mlp: bool = False,
        head_dropout: float = 0.25,
        spatial_max_norm: Optional[float] = 1.0,
    ):
        super().__init__()
        if tmin >= 0.0:
            raise ValueError(f"tmin 须为负值（刺激前基线），得到 {tmin} ms。")
        if tmax <= tmin:
            raise ValueError(f"tmax 须大于 tmin，得到 tmin={tmin}、tmax={tmax} ms。")
        if bypass_mode not in ("separable_pool", "mean_pool", "none"):
            raise ValueError(
                f"bypass_mode 须为 separable_pool/mean_pool/none，得到 {bypass_mode!r}。"
            )
        if baseline_n is None:
            # tmin 为 ms；tmin=−200ms、sfreq=256Hz → 51 点（review v6 P1）
            baseline_n = max(1, int(round(-tmin / 1000.0 * sfreq)))
        self.baseline_n = int(baseline_n)
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.sfreq = float(sfreq)
        self.n_time = int(n_time)
        # 旧接口兼容：未显式给 bypass_mode 时，global_bypass=True→mean_pool / False→none。
        self.bypass_mode = bypass_mode if bypass_mode != "separable_pool" or global_bypass else (
            "mean_pool" if global_bypass else "none"
        )

        # Stage 0.1 加权再参考（可选；GLM v2：门控参考层 + 按域条件化，见 reference.py）
        self.reference = (
            WeightedRereference(n_channels, n_domains=n_domains) if use_rereference else None
        )

        # Stage 1 时空 token 化（v5.1：原生通道名 + spatial max-norm）
        self.tokenizer = ERPTokenizer(
            n_channels=n_channels,
            channel_names=channel_names,
            spatial_max_norm=spatial_max_norm,
            d_model=d_model,
            temporal_kernels=temporal_kernels,
            filters_per_scale=filters_per_scale,
            d_chn_in=d_chn_in,
            d_sub_in=d_sub_in,
            tmin=self.tmin,
            tmax=self.tmax,
            n_time=self.n_time,
        )

        # Stage 2 序列编码（域条件仿射暴露，P1：n_domains 传给 encoder）+ 参数化成分窗
        self.encoder = Stage2Encoder(
            d_model=d_model,
            depth=encoder_depth,
            encoder_type=encoder_type,
            dropout=encoder_dropout,
            n_domains=n_domains,
            tcn_causal=encoder_causal,
            norm=encoder_norm,
        )
        cw_kwargs: dict = {
            "d_model": d_model,
            "dtau_readout": dtau_readout,
            "tmin": self.tmin,
            "tmax": self.tmax,
        }
        if tau0_bounds is not None:
            cw_kwargs["tau0_bounds"] = tau0_bounds
        if tau0_ms is not None:
            cw_kwargs["tau0_ms"] = tau0_ms
        if dtau_bounds is not None:
            cw_kwargs["dtau_bounds"] = dtau_bounds
        if sigma_bounds is not None:
            cw_kwargs["sigma_bounds"] = sigma_bounds
        self.component_window = ComponentWindow(**cw_kwargs)

        # v5.1 EEGNet 式判别旁路（separable_pool 默认；mean_pool 为旧方案 B；none 关闭）
        self.bypass: Optional[TemporalPoolingBypass] = None
        if self.bypass_mode == "separable_pool":
            self.bypass = TemporalPoolingBypass(d_model=d_model, dropout=head_dropout)
            bypass_dim = self.bypass.f2 * self.bypass.pool2
        elif self.bypass_mode == "mean_pool":
            bypass_dim = d_model
        else:
            bypass_dim = None

        # Stage 3 多任务头（v5.1：默认 Dropout+Linear 瘦头；use_mlp_heads 回退旧 MLP）
        self.heads = MultiTaskHeads(
            d_model=d_model,
            use_global_bypass=self.bypass_mode != "none",
            global_dim=bypass_dim,
            use_mlp_heads=head_mlp,
            dropout=head_dropout,
        )

    def _baseline_standardize(self, X: torch.Tensor) -> torch.Tensor:
        """基线段标准化（D-baseline）：(X − μ_b) / σ_b，μ_b/σ_b 取前 baseline_n 点逐通道。"""
        b = X[:, :, : self.baseline_n]  # (B, C, n_b)
        mu = b.mean(dim=2, keepdim=True)  # (B, C, 1)
        std = b.std(dim=2, keepdim=True).clamp(min=1e-6)  # (B, C, 1) 防除零（D-std-clamp）
        return (X - mu) / std

    def forward(
        self,
        X: torch.Tensor,
        E_chn: Optional[torch.Tensor] = None,
        E_sub: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        domain_id: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_heads: bool = True,
    ) -> N2P3NetOutput:
        """X (B,C,T) → N2P3NetOutput。

        channel_mask : (C,) bool 可选，缺失通道掩码（review v3 P0，传 reference 重归一化）。
        domain_id : (B,) long 可选，域条件仿射的域标签（Phase 3）。
        return_heads : bool
            L_jit 的 shifted forward 只需 τ，传 False 跳过 Head-A/B/D 以节省算力。
        """
        # 入口防御：NaN/±inf → 0（review v3 P0 + audit P2-5，缺失/异常通道不毒化）
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Stage 0：门控加权再参考（mask 重归一化 + 按域条件化）+ 基线段标准化
        X0 = (
            self.reference(X, channel_mask, domain_id=domain_id)
            if self.reference is not None
            else X
        )
        X0 = self._baseline_standardize(X0)

        # Stage 1：token 化
        Z = self.tokenizer(X0, E_chn, E_sub)

        # Stage 2：序列编码（域条件仿射）+ 参数化成分窗
        Z2 = self.encoder(Z, domain_id=domain_id)
        if return_attention:
            H, tau, sigma, A = self.component_window(Z2, return_attention=True)
        else:
            H, tau, sigma = self.component_window(Z2)
            A = None

        # Stage 3：多任务头（v5.1：separable_pool 判别旁路 / mean_pool 旧旁路 / none）
        global_features = None
        if return_heads and self.bypass_mode != "none":
            if self.bypass_mode == "separable_pool" and self.bypass is not None:
                global_features = self.bypass(Z2)
            else:
                global_features = Z2.mean(dim=1)
        heads_out = self.heads(H, global_features=global_features) if return_heads else None

        return N2P3NetOutput(
            heads=heads_out, tau=tau, sigma=sigma, H=H, attention=A, features=Z2
        )

    def num_parameters(self) -> int:
        """可学习参数量（参数账，D-budget）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
