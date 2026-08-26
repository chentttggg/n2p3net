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
      D-native-ch      支持原生 3 导蒙太奇。channel_names 决定先验索引，避免把 GTN 伪造为
                       8 导布局；所有入口均使用数据集的原生物理通道布局。
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
    D-glm-bpinit     GLM v3（2026-08-23，tokenizer 深度科研）：时间卷积带通（Gabor）初始化。
                     诊断证据：随机 kaiming 初始化的 FIR 频谱中心 ~60Hz（宽带噪声），训练后
                     cos_sim(init,trained)=0.90-0.98 几乎不动——时间滤波器从未学出 ERP 形状
                     （ERP 能量在 1-8Hz），判别信息全靠地形空间权重 + 下游 TCN 兜底。
                     文献依据：FBCNet 滤波器组多视图、Sinc-ShallowNet/Sinc-EEGNet 带通约束。
                     设计：w[t] = sin(2πf·t + φ)·hann(k)，f 按尺度分层 log 采样
                     （f_lo(k) = max(1.5, 1.2·fs/k)，f_hi = min(40, 3·f_lo)；长核占据
                     ERP δ-θ 带 [1.5,7]Hz），单位 L2 范数 + 随机相位 + 小噪声(0.05)破对称。
                     与 SincNet 硬参数化的区别：滤波器仍完全自由可学习（init 只是起点），
                     保留漂移到任意形状的能力。
    D-glm-postnorm   GLM v3：每尺度时间卷积后接 BatchNorm1d(F) + ELU（空间混合之前）。
                     诊断证据：① 尺度幅值失衡 4×（kaiming 1/√k 缩放，k=13 输出 std 0.141
                     vs k=129 的 0.035）；② Stage 1 纯 affine（无任何非线性/归一化），多尺度
                     线性混合塌缩为单组长 504ms FIR。文献依据：EEG-Inception（ERP 专用，
                     每卷积块 BN+ELU）、ATCNet（BN→ELU→pool）——BN+激活是 ERP-CNN 的
                     标准结构。ELU 而非 ReLU/GELU：保留负向电位（EEG 文献明确论点，
                     Santamaria-Vazquez 2020 / Altaheri 2022）。BN 作用于 (B*C,F,T) 的
                     F 通道（跨被试×通道×时间共享统计，滤波器跨通道共享故合理）。

契约（输入 → 输出）：
    X ∈ R^{B×C×T}；E_chn ∈ R^{C×d_chn_in}（坐标正弦特征，data/channel 输出 6·n_freqs）；
    E_sub ∈ R^{d_sub_in}（subject 正弦特征，data/metadata 输出 2·n_freqs+3）。
    → Z ∈ R^{B×T×D}（T 恒等于输入 T）。E_chn/E_sub 可传 None（退化为不融合）。

依赖的决策：blueprint §3（Stage 1）、constitution P3/E5、device-portability DP1、
    data/channel.D-freq-cap（频率封顶）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.channel import STANDARD_CHANNELS, channel_coords, sinusoidal_embedding
from models.canonical import CoordinateResidualAttention, RegisteredCoordinateGPProjector

# 地形先验的通道索引（基于 STANDARD_CHANNELS 顺序 Fz,Cz,P3,Pz,P4,PO7,PO8,Oz）。
_N2_NEGATIVE: tuple[int, ...] = (5, 6, 7)  # PO7 / PO8 / Oz 枕区负（N2 地形）
_P3B_POSITIVE: tuple[int, ...] = (2, 3, 4)  # P3 / Pz / P4 顶区正（P3b 地形）

# 短/长核分界（采样点）。k < 64（≈250ms）归 N2，k ≥ 64 归 P3b（见 D-spatial-prior）。
_SHORT_KERNEL_LIMIT: int = 64

# v5.1（EEGNet 借鉴）：地形先验按通道名解析，支持原生 3 导 GTN 蒙太奇。
_N2_NEGATIVE_NAMES: tuple[str, ...] = ("PO7", "PO8", "Oz")
_P3B_POSITIVE_NAMES: tuple[str, ...] = ("P3", "Pz", "P4")
_N2_FRONTAL_FALLBACK: tuple[str, ...] = ("Fz", "Cz")


@dataclass(frozen=True)
class TokenizerOutput:
    tokens: torch.Tensor
    canonical_covariance: torch.Tensor | None = None


def _resolve_spatial_indices(
    n_channels: int, channel_names: Sequence[str] | None
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
    scale = torch.where(
        row_norm > float(max_norm), float(max_norm) / row_norm, torch.ones_like(row_norm)
    )
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


def _bandpass_filter_bank(
    kernel_size: int,
    n_filters: int,
    sfreq: float,
    f_lo: float,
    f_hi: float,
    noise: float = 0.05,
) -> torch.Tensor:
    """带通（Gabor）初始化滤波器组 (F, k)：sin(2πf·t + φ)·hann(k)，单位 L2 + 小噪声。

    频率在 [f_lo, f_hi] log 采样；随机相位 φ 打破同频对称。见 D-glm-bpinit。
    noise 是相对每抽头幅度（~1/√k）的噪声比例——绝对 std 会以白噪能量主导
    频谱重心（实测教训：0.05 绝对 std 占总能 32%，把主频拉到 ~48Hz）。
    """
    if f_hi <= f_lo:
        raise ValueError(f"f_hi({f_hi}) 须 > f_lo({f_lo})。")
    freqs = torch.logspace(math.log10(f_lo), math.log10(f_hi), n_filters)  # (F,)
    t = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0  # (k,) 居中
    win = torch.hann_window(kernel_size, dtype=torch.float32)  # (k,)
    phase = torch.rand(n_filters) * 2.0 * math.pi  # (F,)
    w = torch.sin(2.0 * math.pi * freqs[:, None] * t[None, :] / sfreq + phase[:, None])
    w = w * win[None, :]
    w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-8)  # 每滤波器单位 L2
    if noise > 0:
        # 噪声 std 相对每抽头幅度缩放：总噪声能量 = noise²（占单位范数信号的 noise² 比例）
        w = w + (noise / math.sqrt(kernel_size)) * torch.randn_like(w)
    return w


def _band_assignment(kernel_size: int, sfreq: float) -> tuple[float, float]:
    """按核长分配频带：f_lo = max(1.5, 1.2·fs/k)（窗口内 ≥1.2 周期），f_hi = min(40, 3·f_lo)。

    长核自然占据低频（ERP δ-θ 带），短核占据高频——与核长的物理表达能力一致。
    """
    f_lo = max(1.5, 1.2 * sfreq / kernel_size)
    f_hi = min(40.0, 3.0 * f_lo)
    if f_hi <= f_lo:
        f_hi = min(40.0, f_lo * 1.5)
    return float(f_lo), float(f_hi)


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
    sfreq : float
        采样率（Hz），带通初始化的频率分配用（D-glm-bpinit）。
    init : str
        时间卷积初始化："random"（kaiming，旧行为）或 "bandpass"（Gabor 带通，D-glm-bpinit）。
    post_norm : str
        时间卷积后的逐尺度归一化："none"（旧行为）或 "bn"（BatchNorm1d，D-glm-postnorm）。
    post_act : str
        时间卷积后的激活："none"、"elu"（默认，保负电位）或 "gelu"。仅在 post_norm 生效路径
        中可选；单独启用无归一化的激活也允许。
    """

    def __init__(
        self,
        n_channels: int = 8,
        channel_names: Sequence[str] | None = None,
        spatial_max_norm: float | None = 1.0,
        d_model: int = 64,
        temporal_kernels: Sequence[int] = (13, 33, 65, 129),
        filters_per_scale: int = 16,
        d_chn_in: int = 48,
        d_sub_in: int = 19,
        tmin: float = -200.0,
        tmax: float = 800.0,
        n_time: int = 256,
        sfreq: float = 256.0,
        init: str = "random",
        post_norm: str = "none",
        post_act: str = "none",
        channel_positions_m: np.ndarray | None = None,
        canonical_channel_names: Sequence[str] | None = None,
        canonical_positions_m: np.ndarray | None = None,
        canonical_noise_variance: float | Sequence[float] = 0.05,
        canonical_length_scale: float = 0.055,
        canonical_residual_attention: bool = True,
        canonical_residual_limit: float = 0.10,
        temporal_spatial_fusion: bool = True,
    ):
        super().__init__()
        _, _, names = _resolve_spatial_indices(n_channels, channel_names)
        spatial_names = names
        if canonical_channel_names is not None:
            spatial_names = tuple(str(name) for name in canonical_channel_names)
            n_neg, n_pos, spatial_names = _resolve_spatial_indices(
                len(spatial_names), spatial_names
            )
        else:
            n_neg, n_pos, _ = _resolve_spatial_indices(n_channels, names)
        if d_model % 2 != 0:
            raise ValueError(f"d_model 须为偶数（正弦 PE 的 sin/cos 交替），得到 {d_model}。")
        if init not in ("random", "bandpass"):
            raise ValueError(f"init 须为 'random' 或 'bandpass'，得到 {init!r}。")
        if post_norm not in ("none", "bn"):
            raise ValueError(f"post_norm 须为 'none' 或 'bn'，得到 {post_norm!r}。")
        if post_act not in ("none", "elu", "gelu"):
            raise ValueError(f"post_act 须为 'none'/'elu'/'gelu'，得到 {post_act!r}。")
        self.n_channels = n_channels
        self.channel_names = names
        self.spatial_channel_names = spatial_names
        self.spatial_n_channels = len(spatial_names)
        self.spatial_max_norm = None if spatial_max_norm is None else float(spatial_max_norm)
        self.d_model = d_model
        self.sfreq = float(sfreq)
        self.init = init
        self.post_norm = post_norm
        self.post_act = post_act
        self.temporal_kernels = tuple(int(k) for k in temporal_kernels)
        if not self.temporal_kernels:
            raise ValueError("temporal_kernels 不能为空。")
        for k in self.temporal_kernels:
            if k <= 0 or k % 2 == 0:
                raise ValueError(
                    f"temporal_kernels 须为正奇数以保持 T 不变（padding=k//2），得到 {k}。"
                )
        self.filters_per_scale = int(filters_per_scale)
        self.temporal_spatial_fusion = bool(temporal_spatial_fusion)
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        n_scales = len(self.temporal_kernels)

        self.canonical_projector: RegisteredCoordinateGPProjector | None = None
        self.coordinate_residual_attention: CoordinateResidualAttention | None = None
        self.uncertainty_proj: nn.Linear | None = None
        if canonical_channel_names is not None:
            observed_coords, observed_mask = channel_coords(
                names,
                positions_m=channel_positions_m,
                montage=None if channel_positions_m is not None else "standard_1005",
                allow_missing=False,
            )
            query_coords, query_mask = channel_coords(
                spatial_names,
                positions_m=canonical_positions_m,
                montage=None if canonical_positions_m is not None else "standard_1005",
                allow_missing=False,
            )
            if not observed_mask.all() or not query_mask.all():
                raise ValueError(
                    "Canonical GP requires physical coordinates for every active sensor."
                )
            self.canonical_projector = RegisteredCoordinateGPProjector(
                observed_coords,
                query_coords,
                noise_variance=canonical_noise_variance,
                length_scale=canonical_length_scale,
            )
            if d_chn_in % 6 != 0:
                raise ValueError(
                    "Internal canonical coordinate encoding requires d_chn_in divisible by 6."
                )
            query_embedding = sinusoidal_embedding(query_coords, n_freqs=d_chn_in // 6)
            self.register_buffer(
                "canonical_channel_embedding",
                torch.from_numpy(query_embedding),
                persistent=False,
            )
            if canonical_residual_attention:
                self.coordinate_residual_attention = CoordinateResidualAttention(
                    self.canonical_projector.observed_positions,
                    self.canonical_projector.query_positions,
                    max_residual_fraction=canonical_residual_limit,
                )
            self.uncertainty_proj = nn.Linear(self.spatial_n_channels, d_model, bias=False)
            nn.init.zeros_(self.uncertainty_proj.weight)
        else:
            self.register_buffer("canonical_channel_embedding", None, persistent=False)

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
        # GLM v3：带通（Gabor）初始化（D-glm-bpinit）
        if init == "bandpass":
            with torch.no_grad():
                for tconv, k in zip(self.temporal_convs, self.temporal_kernels, strict=True):
                    f_lo, f_hi = _band_assignment(k, self.sfreq)
                    w = _bandpass_filter_bank(k, filters_per_scale, self.sfreq, f_lo, f_hi)
                    tconv.weight.copy_(w.unsqueeze(1))  # (F, 1, k)
                    nn.init.zeros_(tconv.bias)
        # GLM v3：每尺度 BN（D-glm-postnorm）
        self.post_bns = (
            nn.ModuleList([nn.BatchNorm1d(filters_per_scale) for _ in self.temporal_kernels])
            if post_norm == "bn"
            else None
        )
        self.post_act_fn: nn.Module | None = {
            "none": None,
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
        }[post_act]

        # 1.2 空间深度卷积：地形先验（可学习）+ 坐标调制（从 E_chn 生成）
        self.spatial_priors = nn.ParameterList()
        self.coord_mods = nn.ModuleList()
        for k in self.temporal_kernels:
            if k < _SHORT_KERNEL_LIMIT:
                prior = _make_spatial_prior(
                    self.spatial_n_channels, filters_per_scale, positive=(), negative=n_neg
                )
            else:
                prior = _make_spatial_prior(
                    self.spatial_n_channels, filters_per_scale, positive=n_pos, negative=()
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

    @property
    def uses_fused_temporal_spatial(self) -> bool:
        """Whether the exact temporal/spatial algebraic fusion is active."""

        return (
            self.temporal_spatial_fusion
            and self.canonical_projector is None
            and self.post_bns is None
            and self.post_act_fn is None
        )

    @staticmethod
    def _fused_temporal_spatial_conv(
        X: torch.Tensor,
        temporal_conv: nn.Conv1d,
        spatial_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Fold separable temporal and spatial operators into one exact Conv1d."""

        effective_weight = spatial_weight.unsqueeze(-1) * temporal_conv.weight
        effective_bias = (
            None
            if temporal_conv.bias is None
            else temporal_conv.bias * spatial_weight.sum(dim=1)
        )
        return F.conv1d(
            X,
            effective_weight,
            effective_bias,
            stride=temporal_conv.stride,
            padding=temporal_conv.padding,
            dilation=temporal_conv.dilation,
        )

    def forward(
        self,
        X: torch.Tensor,
        E_chn: torch.Tensor | None = None,
        E_sub: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | TokenizerOutput:
        """X (B,C,T) → Z (B,T,D)。

        E_chn (C, d_chn_in) 与 E_sub (d_sub_in,) 可选；None 时跳过对应融合。
        """
        if X.dim() != 3:
            raise ValueError(f"X 须为 (B,C,T)，得到 {X.shape}。")
        B, C, T = X.shape
        if C != self.n_channels:
            raise ValueError(f"X 通道数 {C} 与初始化 n_channels={self.n_channels} 不一致。")

        # E_chn 投影（共享）
        if self.canonical_projector is not None:
            expected_query = (self.spatial_n_channels, self.chn_proj.in_features)
            expected_observed = (C, self.chn_proj.in_features)
            if E_chn is not None and E_chn.shape not in (expected_query, expected_observed):
                raise ValueError(
                    f"Canonical E_chn must be observed {expected_observed} or query "
                    f"{expected_query}, got {tuple(E_chn.shape)}."
                )
            query_identity = (
                E_chn
                if E_chn is not None and E_chn.shape == expected_query
                else self.canonical_channel_embedding
            )
            E_chn_proj = self.chn_proj(query_identity)
        elif E_chn is not None:
            if E_chn.shape != (C, self.chn_proj.in_features):
                raise ValueError(
                    f"E_chn 须为 (C, {self.chn_proj.in_features})，得到 {E_chn.shape}。"
                )
            E_chn_proj = self.chn_proj(E_chn)  # (C, D)
        else:
            E_chn_proj = None

        # 时间卷积（跨通道共享）+ 空间卷积（地形先验 + 坐标调制）。
        scale_weights: list[torch.Tensor] = []
        for spat_prior, coord_mod in zip(
            self.spatial_priors, self.coord_mods, strict=True
        ):
            # 空间权重 = 地形先验 + 坐标调制；可选 EEGNet 式 max-norm（v5.1）
            W = spat_prior  # (F, C)
            if E_chn_proj is not None:
                W_coord = coord_mod(E_chn_proj)  # (C, F)
                W = W + W_coord.transpose(0, 1)  # (F, C)
            W = max_norm_spatial(W, self.spatial_max_norm)
            scale_weights.append(W)

        W_all = torch.cat(scale_weights, dim=0)  # (F*n_scales, C)
        canonical_covariance: torch.Tensor | None = None
        canonical_variance: torch.Tensor | None = None
        if self.uses_fused_temporal_spatial:
            # K_eff[f,c,k] = W[f,c] * K[f,k]. All T samples are retained while the
            # (B,C,F,T) activations and the following spatial einsum disappear.
            scale_outputs = [
                self._fused_temporal_spatial_conv(X, tconv, W)
                for tconv, W in zip(self.temporal_convs, scale_weights, strict=True)
            ]
            Z = torch.cat(scale_outputs, dim=1)  # (B, F*n_scales, T)
        else:
            scale_feats: list[torch.Tensor] = []
            for si, tconv in enumerate(self.temporal_convs):
                feat = tconv(X.reshape(B * C, 1, T))  # (B*C, F, T)
                if self.post_bns is not None:
                    feat = self.post_bns[si](feat)
                if self.post_act_fn is not None:
                    feat = self.post_act_fn(feat)
                feat = feat.reshape(B, C, self.filters_per_scale, T)
                if self.canonical_projector is not None:
                    observed_feat = feat
                    projection = self.canonical_projector(observed_feat, channel_mask=channel_mask)
                    feat = projection.mean
                    if self.coordinate_residual_attention is not None:
                        feat = self.coordinate_residual_attention(
                            observed=observed_feat,
                            gp_mean=feat,
                            channel_mask=channel_mask,
                        )
                    canonical_covariance = projection.covariance
                    canonical_variance = projection.variance
                scale_feats.append(feat)

            feat_all = torch.cat(scale_feats, dim=2)  # (B, C, F*n_scales, T)
            Z = torch.einsum("bcft,fc->bft", feat_all, W_all)
        if canonical_covariance is not None and self.canonical_projector is not None:
            # Exact uncertainty propagation through each learned linear spatial
            # functional: Var[w^T mu_Q] = w^T Sigma_Q w. Normalizing by the
            # prior variance of an independent canonical field makes the
            # attenuation dimensionless while retaining off-diagonal structure.
            weights = W_all.float()
            propagated_variance = torch.einsum(
                "fc,bcq,fq->bf", weights, canonical_covariance.float(), weights
            ).clamp_min(0.0)
            prior_variance = self.canonical_projector.kernel_variance * weights.square().sum(
                dim=-1
            ).clamp_min(1e-8)
            spatial_reliability = torch.rsqrt(1.0 + propagated_variance / prior_variance[None])
            Z = Z * spatial_reliability.to(dtype=Z.dtype).unsqueeze(-1)
        Z = self.pointwise(Z)  # (B, D, T)
        Z = Z.transpose(1, 2)  # (B, T, D)

        if canonical_variance is not None and self.uncertainty_proj is not None:
            uncertainty = torch.log1p(canonical_variance.float())
            Z = Z + self.uncertainty_proj(uncertainty).to(dtype=Z.dtype).unsqueeze(1)

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

        if return_details:
            return TokenizerOutput(tokens=Z, canonical_covariance=canonical_covariance)
        return Z
