"""模块 #6：ERP 感知时空 token 化层（Spatio-Temporal Tokenizer）。

职责（blueprint Stage 1）：
    把预处理后的单试次 EEG X ∈ R^{B×C×T} 转成 token 序列 Z ∈ R^{B×T×D}，同时融合
    通道坐标嵌入 E_chn（通道身份）与 subject 元数据嵌入 E_sub（被试信息）。

子模块（blueprint §3）：
    1.1 多尺度时间卷积银行：核长 {13, 33, 65, 129} @256Hz ≈ {51, 129, 254, 504} ms，
        跨通道共享（所有通道用同一组滤波器），每核长 F=16 滤波器，stride=1 + padding=same。
    1.2 空间深度卷积：按尺度分地形先验初始化（短核 → N2 枕区负、长核 → P3b 顶区正），
        叠加「坐标调制」（空间权重 = 地形先验 + 从 E_chn 线性生成的调制）。
    1.3 token 化：B×C×T → B×T×D + 时间位置编码（绝对潜伏期是时域任务关键）。
    1.4 融合 E_sub：subject 嵌入投影后作加性 bias 广播到每个时间步。

明确「不做」（留给下游模块）：
    - 序列编码 / 自注意力（Stage 2，encoder.py）
    - 参数化成分窗（Stage 2，component_window.py）
    - 多任务头（Stage 3，heads.py）

三思决策记录（供后续会话追溯）：
    D-tconv-shared   时间卷积「跨通道共享」= 权重在 C 个通道间共享。实现为把 (B,C,T) reshape
                     成 (B*C,1,T) 后 Conv1d(1,F,k)，而非 groups=C 的 depthwise（后者每通道独立
                     权重，违背 blueprint「跨通道共享」的原意）。数学上等效于「同一个 FIR 滤波器
                     组扫过所有通道」。
    D-spatial-prior  空间权重地形先验初始化：通道顺序 Fz,Cz,P3,Pz,P4,PO7,PO8,Oz（索引 0..7）。
                     短核（k<64 采样点）→ N2 地形（PO7/PO8/Oz 枕区负，索引 5/6/7）；
                     长核（k≥64）→ P3b 地形（P3/Pz/P4 顶区正，索引 2/3/4）。
                     先验向量归一化到单位范数 + 小噪声(0.1)打破 F 个滤波器的对称退化。
      D-native-ch      v5.1（EEGNet 借鉴）：支持原生 3 导蒙太奇。channel_names 决定先验索引，
                       避免 GTN 零填充 5 个恒 0 通道；旧 8 导零填充仍可用 n_channels=8 显式复现。
      D-spatial-maxn   v5.1（EEGNet 借鉴）：有效空间权重 W=prior+coord_mod 后按行做 max-norm=1
                       （Lawhern 2018 的 CSP 式正则）；spatial_max_norm=None 恢复旧行为。
    D-coord-mod      空间权重 = 地形先验（可学习） + 坐标调制（Linear(d_model→F)(E_chn_proj)），
                     使「坐标决定通道对成分滤波器的贡献」显式落地（坐标替代通道名的核心机制）。
                     坐标调制初始化为 0（nn.init.zeros_），故初始空间权重 ≈ 地形先验，训练中
                     坐标信息逐步注入。
    D-time-pe        时间位置编码基于物理时间（tmin~tmax ms 归一化到 [0,1]），用标准正弦 PE 公式
                     （频率 1/10000^... 递减、封顶于 1），**无频率爆炸**（复用 P0② 教训）；绝对
                     潜伏期是时域任务关键（blueprint 1.3），故必须用物理时间而非位置索引。
                     预计算默认 n_time 的 PE 存 buffer（persistent=False），forward 按 T 命中缓存
                     或动态生成；**加前显式对齐 dtype=Z.dtype**——AMP(bf16) 下 register_buffer 仍
                     是 float32、而 token 流是 bf16，Z+PE 会 promote 到 float32 破坏 autocast 语义
                     （实测确认 bf16+float32→float32）。此修复是 Phase C 集成联调前必须钉死的。
    D-nopool         全程 stride=1 + padding=same、无池化，输出 T 恒等于输入 T（E5 落地）；这是
                     本模块语义测试的核心不变量。
    D-sub-bias       E_sub 融合为「加性 bias 广播到每个时间步」，反映「年龄/性别全局调制 P300
                     潜伏期与幅值」（blueprint E8），而非逐时间步独立调制（过度设计）。
    D-device         纯 nn.Module，不硬编码设备（device-portability DP1）；设备由调用方 .to(DEVICE)。

契约（输入 → 输出）：
    X ∈ R^{B×C×T}；E_chn ∈ R^{C×d_chn_in}（坐标正弦特征，data/channel 输出 6·n_freqs）；
    E_sub ∈ R^{d_sub_in}（subject 正弦特征，data/metadata 输出 2·n_freqs+3）。
    → Z ∈ R^{B×T×D}（T 恒等于输入 T）。E_chn/E_sub 可传 None（退化为不融合）。

依赖的决策：blueprint §3（Stage 1）、constitution P3/E5、device-portability DP1、
    data/channel.D-freq-cap（频率封顶）。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn

from data.preprocess import STANDARD_CHANNELS

# 地形先验的通道索引（基于 STANDARD_CHANNELS 顺序 Fz,Cz,P3,Pz,P4,PO7,PO8,Oz）。
_N2_NEGATIVE: tuple[int, ...] = (5, 6, 7)  # PO7 / PO8 / Oz 枕区负（N2 地形）
_P3B_POSITIVE: tuple[int, ...] = (2, 3, 4)  # P3 / Pz / P4 顶区正（P3b 地形）

# 短/长核分界（采样点）。k < 64（≈250ms）归 N2，k ≥ 64 归 P3b（见 D-spatial-prior）。
_SHORT_KERNEL_LIMIT: int = 64

# v5.1（EEGNet 借鉴）：地形先验按通道名解析，支持原生 3 导 GTN 蒙太奇。
_N2_NEGATIVE_NAMES: tuple[str, ...] = ("PO7", "PO8", "Oz")
_P3B_POSITIVE_NAMES: tuple[str, ...] = ("P3", "Pz", "P4")
_N2_FRONTAL_FALLBACK: tuple[str, ...] = ("Fz", "Cz")


def _resolve_spatial_indices(
    n_channels: int, channel_names: Optional[Sequence[str]]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    """把通道名解析为 (短核负权索引, 长核正权索引, 实际通道名)。

    - 标准 8 导：保持 v4 语义（短核 → PO7/PO8/Oz 负、长核 → P3/Pz/P4 正）。
    - 其他蒙太奇（如 GTN Fz/Cz/Pz）：长核正 = 顶区 Pz/P3/P4（若有）；
      短核负 = 枕区 PO7/PO8/Oz（若有），否则回退到 Fz/Cz（前中央 N2 近似）。
    """
    if channel_names is None:
        if n_channels != 8:
            raise ValueError(
                f"n_channels={n_channels} 且未提供 channel_names；固定蒙太奇仅支持 8 导，"
                "其他通道数须显式传入 channel_names。"
            )
        names = STANDARD_CHANNELS
    else:
        names = tuple(str(c) for c in channel_names)
        if len(names) != n_channels:
            raise ValueError(f"channel_names 长度 {len(names)} 须等于 n_channels={n_channels}。")
        if len(set(names)) != len(names):
            raise ValueError(f"channel_names 含重复通道：{names}。")

    negative = tuple(i for i, n in enumerate(names) if n in _N2_NEGATIVE_NAMES)
    positive = tuple(i for i, n in enumerate(names) if n in _P3B_POSITIVE_NAMES)
    if not negative:
        negative = tuple(i for i, n in enumerate(names) if n in _N2_FRONTAL_FALLBACK)
    if not positive:
        # 最保守回退：最后一根电极（常见 Pz）作长核正权，保证非零先验。
        positive = (len(names) - 1,)
    return negative, positive, names


def max_norm_spatial(W: torch.Tensor, max_norm: float) -> torch.Tensor:
    """EEGNet 式空间权重 max-norm：逐行（滤波器）范数 > max_norm 时缩回 max_norm。

    W: (F, C)。max_norm <= 0 或 None 表示不约束（旧行为）。
    """
    if max_norm is None or float(max_norm) <= 0.0:
        return W
    row_norm = W.norm(dim=1, keepdim=True).clamp_min(1e-6)
    scale = torch.where(row_norm > float(max_norm), float(max_norm) / row_norm, torch.ones_like(row_norm))
    return W * scale


def _make_spatial_prior(
    n_channels: int,
    n_filters: int,
    positive: tuple[int, ...],
    negative: tuple[int, ...],
    noise: float = 0.1,
) -> torch.Tensor:
    """构造地形先验 (F, C)：正权索引 +1、负权索引 −1，单位范数 + 小噪声打破对称。"""
    v = torch.zeros(n_channels)
    for i in positive:
        v[i] = 1.0
    for i in negative:
        v[i] = -1.0
    n = v.norm()
    if n > 0:
        v = v / n
    w = v.unsqueeze(0).repeat(n_filters, 1)  # (F, C)
    w = w + noise * torch.randn(n_filters, n_channels)
    return w


def _build_time_pe(n_times: int, d_model: int, tmin: float, tmax: float) -> torch.Tensor:
    """物理时间位置编码 (T, D)，标准正弦公式、频率封顶于 1（无爆炸，见 D-time-pe）。

    时间轴用 arange 采样（间隔 (tmax-tmin)/n_times，真实采样），与 component_window 一致
    （review P2 时间轴修正）。
    """
    t = tmin + torch.arange(n_times, dtype=torch.float32) * (tmax - tmin) / n_times  # (T,)
    positions = (t - tmin) / (tmax - tmin)  # (T,) 归一化 [0,1]
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )  # (D/2,)
    pe = torch.zeros(n_times, d_model)
    pe[:, 0::2] = torch.sin(positions[:, None] * div_term[None, :])
    pe[:, 1::2] = torch.cos(positions[:, None] * div_term[None, :])
    return pe


class ERPTokenizer(nn.Module):
    """Stage 1 时空 token 化：(B, C, T) + E_chn + E_sub → Z (B, T, D)。

    Parameters
    ----------
    n_channels : int
        通道数 C（默认 8）。
      channel_names : Sequence[str] | None
          通道名（非 8 导蒙太奇必填；GTN 原生 3 导传 ("Fz","Cz","Pz")）。
      spatial_max_norm : float | None
          EEGNet 式空间权重 max-norm（默认 1.0；None 关闭，恢复旧行为）。
    d_model : int
        隐藏维 D（默认 64）。
    temporal_kernels : Sequence[int]
        多尺度时间卷积核长（采样点），默认 (13, 33, 65, 129)。
    filters_per_scale : int
        每尺度的滤波器数 F（默认 16）；拼接后通道数 = F × len(kernels)。
    d_chn_in : int
        E_chn 输入维（坐标正弦特征，data/channel 输出 6·n_freqs）。
    d_sub_in : int
        E_sub 输入维（subject 正弦特征，data/metadata 输出 2·n_freqs+3）。
    tmin / tmax : float
        时间窗物理范围（ms），用于时间位置编码（默认 -200 ~ +800）。
    n_time : int
        预计算时间位置编码的默认时间点数（默认 256，与 [-200,+800]ms@256Hz 一致）；
        forward 若 T 不匹配则动态生成。
    """

    def __init__(
        self,
        n_channels: int = 8,
        channel_names: Optional[Sequence[str]] = None,
        spatial_max_norm: Optional[float] = 1.0,
        d_model: int = 64,
        temporal_kernels: Sequence[int] = (13, 33, 65, 129),
        filters_per_scale: int = 16,
        d_chn_in: int = 48,
        d_sub_in: int = 19,
        tmin: float = -200.0,
        tmax: float = 800.0,
        n_time: int = 256,
    ):
        super().__init__()
        n_neg, n_pos, names = _resolve_spatial_indices(n_channels, channel_names)
        if d_model % 2 != 0:
            raise ValueError(f"d_model 须为偶数（正弦 PE 的 sin/cos 交替），得到 {d_model}。")
        self.n_channels = n_channels
        self.channel_names = names
        self.spatial_max_norm = None if spatial_max_norm is None else float(spatial_max_norm)
        self.d_model = d_model
        self.temporal_kernels = tuple(int(k) for k in temporal_kernels)
        if not self.temporal_kernels:
            raise ValueError("temporal_kernels 不能为空。")
        for k in self.temporal_kernels:
            if k <= 0 or k % 2 == 0:
                raise ValueError(
                    f"temporal_kernels 须为正奇数以保持 T 不变（padding=k//2），得到 {k}。"
                )
        self.filters_per_scale = int(filters_per_scale)
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        n_scales = len(self.temporal_kernels)

        # 嵌入投影（E_chn 共享投影到 D；E_sub 投影到 D）
        self.chn_proj = nn.Linear(d_chn_in, d_model)
        self.sub_proj = nn.Linear(d_sub_in, d_model)

        # 1.1 多尺度时间卷积（跨通道共享：输入 reshape (B*C,1,T)，Conv1d(1,F,k)）
        self.temporal_convs = nn.ModuleList(
            [
                nn.Conv1d(1, filters_per_scale, kernel_size=k, padding=k // 2)
                for k in self.temporal_kernels
            ]
        )

        # 1.2 空间深度卷积：地形先验（可学习）+ 坐标调制（从 E_chn 生成）
        self.spatial_priors = nn.ParameterList()
        self.coord_mods = nn.ModuleList()
        for k in self.temporal_kernels:
            if k < _SHORT_KERNEL_LIMIT:
                prior = _make_spatial_prior(
                    n_channels, filters_per_scale, positive=(), negative=n_neg
                )
            else:
                prior = _make_spatial_prior(
                    n_channels, filters_per_scale, positive=n_pos, negative=()
                )
            self.spatial_priors.append(nn.Parameter(prior))
            # 坐标调制初始化为 0，使初始空间权重 ≈ 地形先验（D-coord-mod）
            mod = nn.Linear(d_model, filters_per_scale)
            nn.init.zeros_(mod.weight)
            nn.init.zeros_(mod.bias)
            self.coord_mods.append(mod)

        # 1.3 pointwise（混合 4 尺度特征并投影到 D）
        self.pointwise = nn.Conv1d(filters_per_scale * n_scales, d_model, kernel_size=1)

        # 时间位置编码（预计算默认 T，缓存；D-time-pe）
        self.register_buffer(
            "time_pe", _build_time_pe(n_time, d_model, tmin, tmax), persistent=False
        )

    def forward(
        self,
        X: torch.Tensor,
        E_chn: Optional[torch.Tensor] = None,
        E_sub: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """X (B,C,T) → Z (B,T,D)。

        E_chn (C, d_chn_in) 与 E_sub (d_sub_in,) 可选；None 时跳过对应融合。
        """
        if X.dim() != 3:
            raise ValueError(f"X 须为 (B,C,T)，得到 {X.shape}。")
        B, C, T = X.shape
        if C != self.n_channels:
            raise ValueError(f"X 通道数 {C} 与初始化 n_channels={self.n_channels} 不一致。")

        # E_chn 投影（共享）
        if E_chn is not None:
            if E_chn.shape != (C, self.chn_proj.in_features):
                raise ValueError(f"E_chn 须为 (C, {self.chn_proj.in_features})，得到 {E_chn.shape}。")
            E_chn_proj = self.chn_proj(E_chn)  # (C, D)
        else:
            E_chn_proj = None

        # 时间卷积（跨通道共享）+ 空间卷积（地形先验 + 坐标调制）。
        # 先收集所有尺度的 (B,C,F,T) 特征与 (F,C) 空间权重，再做一次批量 einsum，
        # 减少逐尺度的小矩阵乘与逐尺度 cat（D-tokenizer-batched-mix）。
        scale_feats: list[torch.Tensor] = []
        scale_weights: list[torch.Tensor] = []
        for tconv, spat_prior, coord_mod in zip(
            self.temporal_convs, self.spatial_priors, self.coord_mods
        ):
            feat = tconv(X.reshape(B * C, 1, T))  # (B*C, F, T)
            feat = feat.reshape(B, C, self.filters_per_scale, T)  # (B, C, F, T)
            scale_feats.append(feat)

            # 空间权重 = 地形先验 + 坐标调制；可选 EEGNet 式 max-norm（v5.1）
            W = spat_prior  # (F, C)
            if E_chn_proj is not None:
                W_coord = coord_mod(E_chn_proj)  # (C, F)
                W = W + W_coord.transpose(0, 1)  # (F, C)
            W = max_norm_spatial(W, self.spatial_max_norm)
            scale_weights.append(W)

        feat_all = torch.cat(scale_feats, dim=2)  # (B, C, F*n_scales, T)
        W_all = torch.cat(scale_weights, dim=0)  # (F*n_scales, C)
        Z = torch.einsum("bcft,fc->bft", feat_all, W_all)  # (B, F*n_scales, T)
        Z = self.pointwise(Z)  # (B, D, T)
        Z = Z.transpose(1, 2)  # (B, T, D)

        # 时间位置编码（物理时间，缓存 + dtype 对齐，D-time-pe）
        if T == self.time_pe.shape[0]:
            pe = self.time_pe
        else:
            pe = _build_time_pe(T, self.d_model, self.tmin, self.tmax).to(self.time_pe.device)
        Z = Z + pe.to(dtype=Z.dtype)

        # E_sub 融合（加性 bias，D-sub-bias；支持 (d_sub_in,) 单被试 或 (B, d_sub_in) 跨被试，P1）
        if E_sub is not None:
            if E_sub.dim() == 1:
                E_sub_proj = self.sub_proj(E_sub)  # (D,)
                Z = Z + E_sub_proj.view(1, 1, -1)
            elif E_sub.dim() == 2:
                if E_sub.shape[0] != B:
                    raise ValueError(f"E_sub (B,d) 的 B 须匹配，得到 {E_sub.shape[0]} vs {B}。")
                E_sub_proj = self.sub_proj(E_sub)  # (B, D)
                Z = Z + E_sub_proj.unsqueeze(1)  # (B, 1, D)
            else:
                raise ValueError(f"E_sub 须为 (d,) 或 (B,d)，得到 {E_sub.shape}。")

        return Z
