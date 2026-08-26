"""模块 #7：Stage 2 序列编码（Sequence Encoder）。

职责（blueprint Stage 2 的 2.1）：
    对 Stage 1 输出 Z ∈ R^{B×T×D} 做「跨时间」序列编码，补齐 N2→P3b 约 150ms 的中程关系
    （Stage 1 多尺度卷积已覆盖 ~500ms 局部上下文，故此处 ROI 低、须轻量）。

生产 recipe 使用 `encoder_type="tcn"`、`encoder_depth=4`，canonical dilations 为
`(1, 4, 16, 32)`。注册的显式消融轴为 depth ∈ {0,1,2,3,4}；其中 depth=3
是轻量对照，depth=0 是恒等地板。底层构造器仍保留独立的 conformer/depth 接口：
    - depth=0：恒等（identity），参数化成分窗直接吃 Stage 1 输出（零开销地板）。
    - depth>0：堆叠 depth 个 block，两种类型（encoder_type）：
        · "conformer"：轻量 Pre-LN Transformer block（MHSA + FFN，FFN expansion=2，LayerNorm + 域条件仿射）；
        · "tcn"：膨胀 depthwise TCN（canonical dilation 由 depth 生成，depthwise + pointwise，
          LayerNorm/BN + 域条件仿射）。

明确「不做」（留给下游模块）：
    - 参数化成分窗（2.2，component_window.py）
    - 多任务头（Stage 3，heads.py）

三思决策记录（供后续会话追溯）：
    D-depth-axis     depth ∈ {0,1,2,...} 是消融轴（blueprint D6）。depth=0 是 identity（参数 0），
                     既是「序列编码 ROI 低」的诚实地板，也是超预算时的降容路径。
                     N2P3Net 生产组装默认用 tcn depth=4（2026-08-26 推理深度定案，见
                     recipe / constitution E4；depth=3 为轻量对照）。
    D-lite-conformer 「轻量 Conformer」= 去掉卷积模块的 Pre-LN Transformer block（MHSA + FFN）。
                     理由：Stage 1 已有多尺度时间卷积覆盖局部上下文，Stage 2 再加卷积模块是冗余
                     （D6「容量非瓶颈、域差才是」）。FFN expansion=2 是相对标准 4 的轻量化。
    D-budget        参数账（诚实记录）：D=64 下 depth=1 Conformer ≈ 33k（MHSA 16.6k + FFN 16.6k），
                     加 tokenizer ≈17k 合计 ≈50k。现行 E4 硬上限为 80k，但这是 ceiling 而非目标。
                     扩容只由预注册性能/互补性消融决定；无有效增益时保留 TCN 或 depth→0 小模型。
                     本模块不做「偷偷减容或增容」，参数账公开、由 roadmap Phase 2 验收裁决。
    D-norm           归一化是显式 GLM 消融轴：conformer 路径恒 LayerNorm；TCN 路径由 recipe 控制，
                     当前默认 BatchNorm1d（训练折统计、推理冻结），ln 为回退。跨域对齐用「域条件仿射」
                     （per-domain 可学习 scale/shift，加于 LayerNorm 后），不用 Split-BN。
    D-domain-affine  域条件仿射是 Phase 3 跨域微调的组件；Phase 2 单受试 n_domains=None 时不启用
                     （零参数零开销）。接口预留 domain_id 入参，Phase 3 无需改契约。**初始化为恒等**
                     （scale=1、shift=0），使初始状态 = 无跨域对齐，训练中逐步学到域差异——与
                     tokenizer 的坐标调制初始化为 0 同一模式（从「无」出发，而非从随机出发）。
    D-tcn-receptive TCN dilation 前四层固定为 (1,4,16,32)，保持既有 dep3/dep4 可比性；
                     后续层从 64 开始按 2 倍扩展。TCN 层数由 depth 唯一控制，dilation 自动取
                     对应前缀。3 层（≈168ms）已足以覆盖 N2→P3b ~150ms，4 层（≈418ms）是
                     现行推理默认，blueprint「~500ms」系把 Stage 1 感受野计入。
    D-final-norm     depth>0 时加 final LayerNorm（Pre-LN 惯例，利于下游成分窗读数）；depth=0 不加
                     （identity 不改变 tokenizer 输出）。
    D-device         纯 nn.Module，不硬编码设备（device-portability DP1）。

契约（输入 → 输出）：
    Z ∈ R^{B×T×D}（+ 可选 domain_id ∈ Z^{B}）→ Z' ∈ R^{B×T×D}（形状不变）。

依赖的决策：blueprint §4（Stage 2）、constitution D6/D7/E4/E5、device-portability DP1。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

_LEGACY_TCN_DILATIONS = (1, 4, 16, 32)
DEFAULT_ENCODER_DEPTH = 4


def default_tcn_dilations(depth: int) -> tuple[int, ...]:
    """Return the canonical dilation prefix selected by one encoder depth value.

    The first four values are frozen for dep3/dep4 comparability. Deeper
    ablations continue the same receptive-field expansion by doubling from 64.
    """
    depth = int(depth)
    if depth < 0:
        raise ValueError(f"depth 须 ≥0，得到 {depth}。")
    if depth <= len(_LEGACY_TCN_DILATIONS):
        return _LEGACY_TCN_DILATIONS[:depth]
    extension = tuple(
        _LEGACY_TCN_DILATIONS[-1] * (2 ** (index + 1))
        for index in range(depth - len(_LEGACY_TCN_DILATIONS))
    )
    return _LEGACY_TCN_DILATIONS + extension


def _apply_domain_affine(
    h: torch.Tensor,
    scale: torch.Tensor | None,
    shift: torch.Tensor | None,
    domain_id: torch.Tensor | None,
) -> torch.Tensor:
    """域条件仿射（D-domain-affine）：h → h * scale[domain] + shift[domain]。"""
    if scale is not None and domain_id is not None:
        s = scale[domain_id]  # (B, D)
        b = shift[domain_id]  # (B, D)
        return torch.addcmul(b.unsqueeze(1), h, s.unsqueeze(1))
    return h


def _apply_domain_affine_channel_first(
    h: torch.Tensor,
    scale: torch.Tensor | None,
    shift: torch.Tensor | None,
    domain_id: torch.Tensor | None,
) -> torch.Tensor:
    """Apply the same domain affine while retaining the Conv1d-friendly layout."""

    if scale is not None and domain_id is not None:
        s = scale[domain_id].unsqueeze(-1)
        b = shift[domain_id].unsqueeze(-1)
        return torch.addcmul(b, h, s)
    return h


class _ConformerBlock(nn.Module):
    """轻量 Conformer block（Pre-LN Transformer：MHSA + FFN，无卷积模块，见 D-lite-conformer）。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_expansion: int,
        dropout: float,
        n_domains: int | None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(d_model * ffn_expansion, d_model),
            nn.Dropout(dropout, inplace=True),
        )
        if n_domains is not None:
            self.dom_scale = nn.Parameter(torch.ones(n_domains, d_model))
            self.dom_shift = nn.Parameter(torch.zeros(n_domains, d_model))
        else:
            self.register_parameter("dom_scale", None)
            self.register_parameter("dom_shift", None)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor | None = None) -> torch.Tensor:
        h = _apply_domain_affine(self.ln1(x), self.dom_scale, self.dom_shift, domain_id)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        h = _apply_domain_affine(self.ln2(x), self.dom_scale, self.dom_shift, domain_id)
        x = x + self.ffn(h)
        return x


class _TCNBlock(nn.Module):
    """膨胀 depthwise TCN block（depthwise + pointwise + GELU，见 D-tcn-receptive）。

    GLM 消融轴 ``norm``："ln"（构造器旧默认）或 "bn"（生产 recipe 默认）。BN 在 (B,T) 上按特征维归一化，
    保留 token 间相对幅值的同时稳定特征尺度——跨被试 P300 文献反复报告 BN 是
    CNN 泛化的关键组件（Värbu 2020：ELU+dropout+BN 与最佳 CNN 性能相关）；
    LN 是逐 token 归一化，会抹平单 token 的幅值维度。N2P3Net 组装层显式传入 recipe 默认 bn。
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        n_domains: int | None,
        causal: bool = False,
        norm: str = "ln",
        pointwise_execution: str = "linear",
        bn_momentum: float = 0.1,
    ):
        super().__init__()
        if norm not in ("ln", "bn"):
            raise ValueError(f"norm 须为 'ln' 或 'bn'，得到 {norm!r}。")
        if pointwise_execution not in ("conv1d", "linear"):
            raise ValueError("pointwise_execution must be 'conv1d' or 'linear'.")
        if not 0.0 < bn_momentum <= 1.0:
            raise ValueError("bn_momentum must be in (0, 1].")
        self.norm_type = norm
        self.pointwise_execution = pointwise_execution
        if norm == "bn":
            self.ln = nn.BatchNorm1d(
                d_model,
                momentum=float(bn_momentum),
            )  # 输入 (B,D,T)：跨 batch/时间归一化
        else:
            self.ln = nn.LayerNorm(d_model)
        # depthwise（groups=d_model，每通道独立卷积）+ pointwise（1x1 混合）
        # causal=True 时只使用左侧 padding，离线保持 T 不变且每个输出时间点只看历史。
        padding = (kernel_size - 1) * dilation if causal else dilation
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size, padding=padding, dilation=dilation, groups=d_model
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout, inplace=True)
        if n_domains is not None:
            self.dom_scale = nn.Parameter(torch.ones(n_domains, d_model))
            self.dom_shift = nn.Parameter(torch.zeros(n_domains, d_model))
        else:
            self.register_parameter("dom_scale", None)
            self.register_parameter("dom_shift", None)

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        """x (B,T,D) → 归一化后 (B,T,D)。BN 在 (B,D,T) 上做，LN 在最后一维做。"""
        if self.norm_type == "bn":
            return self.ln(x.transpose(1, 2)).transpose(1, 2)
        return self.ln(x)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_type == "bn":
            # Keep the channel-first BN output in Conv1d layout. The previous
            # B,D,T -> B,T,D -> B,D,T round-trip was algebraically redundant.
            h = self.ln(x.transpose(1, 2))
            h = _apply_domain_affine_channel_first(
                h,
                self.dom_scale,
                self.dom_shift,
                domain_id,
            )
        else:
            h = _apply_domain_affine(
                self._apply_norm(x),
                self.dom_scale,
                self.dom_shift,
                domain_id,
            ).transpose(1, 2)
        h = self.depthwise(h)
        # 因果 padding 会让 Conv1d 输出比输入长，裁回原 T；非因果 padding=same 无影响。
        h = h[..., : x.shape[1]]
        h = self.act(h)
        if self.pointwise_execution == "linear":
            h = h.transpose(1, 2)  # (B, T, D)
            # A 1x1 Conv1d is exactly a Linear(D,D) over the folded B*T dimension.
            # Keep the Conv1d parameter shape so existing checkpoints remain loadable.
            h = F.linear(h, self.pointwise.weight.squeeze(-1), self.pointwise.bias)
        else:
            h = self.pointwise(h).transpose(1, 2)
        h = self.dropout(h)
        return x + h

    def forward_channel_first(
        self,
        x: torch.Tensor,
        domain_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute a Conv1d pointwise block without leaving (B,D,T) layout."""

        if self.pointwise_execution != "conv1d":
            raise RuntimeError("channel-first TCN execution requires pointwise_execution='conv1d'.")
        if self.norm_type == "bn":
            h = self.ln(x)
        else:
            h = self.ln(x.transpose(1, 2)).transpose(1, 2)
        h = _apply_domain_affine_channel_first(
            h,
            self.dom_scale,
            self.dom_shift,
            domain_id,
        )
        h = self.depthwise(h)[..., : x.shape[-1]]
        h = self.pointwise(self.act(h))
        return x + self.dropout(h)


class Stage2Encoder(nn.Module):
    """Stage 2 序列编码：Z (B,T,D) → Z' (B,T,D)，depth 消融 + conformer/tcn 双实现。

    Parameters
    ----------
    d_model : int
        隐藏维 D（默认 64）。
    depth : int
        序列编码层数（消融轴，D-depth-axis）。depth=0 为恒等。
    encoder_type : str
        "conformer"（轻量 Transformer）或 "tcn"（膨胀 depthwise TCN）；生产 recipe
        选择 tcn。
    n_heads : int
        Conformer 自注意力头数（默认 4）。
    ffn_expansion : int
        Conformer FFN 扩展比（默认 2）。
    dropout : float
        dropout 概率（默认 0.1）。
    n_domains : int | None
        域条件仿射的域数（Phase 3 跨域）；None 时不启用（Phase 2 零开销，D-domain-affine）。
    tcn_kernel : int
        TCN 卷积核长（默认 3）。
    tcn_dilations : Sequence[int]
        TCN 膨胀系数序列，取前 depth 个；None 时由 depth 自动生成 canonical prefix。
      tcn_causal : bool
          TCN 是否因果 padding（默认 False，离线整段编码；True 供未来在线推理）。
      norm : str
          TCN block 的归一化（GLM 消融轴）："ln"（默认，旧版）或 "bn"（跨 batch/时间
          的 BatchNorm1d；见 _TCNBlock 文档）。conformer 路径不受此参数影响（恒 LN）。
      tcn_pointwise_execution : str
          TCN 1x1 混合的执行 API："conv1d" 或等价的 B*T "linear" 调度。
    """

    def __init__(
        self,
        d_model: int = 64,
        depth: int = 1,
        encoder_type: str = "conformer",
        n_heads: int = 4,
        ffn_expansion: int = 2,
        dropout: float = 0.1,
        n_domains: int | None = None,
        tcn_kernel: int = 3,
        tcn_dilations: Sequence[int] | None = None,
        tcn_causal: bool = False,
        norm: str = "ln",
        tcn_pointwise_execution: str = "linear",
        bn_momentum: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.depth = int(depth)
        self.encoder_type = encoder_type
        self.tcn_dilations: tuple[int, ...] = ()
        if self.depth < 0:
            raise ValueError(f"depth 须 ≥0，得到 {self.depth}。")
        if encoder_type not in ("conformer", "tcn"):
            raise ValueError(f"encoder_type 须为 'conformer' 或 'tcn'，得到 {encoder_type!r}。")
        if norm not in ("ln", "bn"):
            raise ValueError(f"norm 须为 'ln' 或 'bn'，得到 {norm!r}。")
        if tcn_pointwise_execution not in ("conv1d", "linear"):
            raise ValueError("tcn_pointwise_execution must be 'conv1d' or 'linear'.")
        if not 0.0 < bn_momentum <= 1.0:
            raise ValueError("bn_momentum must be in (0, 1].")

        if self.depth == 0:
            self.blocks = nn.ModuleList()
            self.final_norm: nn.LayerNorm | None = None
        elif encoder_type == "conformer":
            if d_model % n_heads != 0:
                raise ValueError(f"d_model({d_model}) 须能被 n_heads({n_heads}) 整除。")
            self.blocks = nn.ModuleList(
                [
                    _ConformerBlock(d_model, n_heads, ffn_expansion, dropout, n_domains)
                    for _ in range(self.depth)
                ]
            )
            self.final_norm = nn.LayerNorm(d_model)
        elif encoder_type == "tcn":
            dilations = (
                default_tcn_dilations(self.depth)
                if tcn_dilations is None
                else tuple(int(d) for d in tcn_dilations)
            )
            if self.depth > len(dilations):
                raise ValueError(
                    f"TCN depth({self.depth}) 不能超过 tcn_dilations 长度({len(dilations)})；"
                    f"请显式提供更长的 tcn_dilations（review v6 P1：不静默截断）。"
                )
            dilations = dilations[: self.depth]
            self.tcn_dilations = dilations
            self.blocks = nn.ModuleList(
                [
                    _TCNBlock(
                        d_model,
                        tcn_kernel,
                        d,
                        dropout,
                        n_domains,
                        causal=tcn_causal,
                        norm=norm,
                        pointwise_execution=tcn_pointwise_execution,
                        bn_momentum=bn_momentum,
                    )
                    for d in dilations
                ]
            )
            self.final_norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"encoder_type 须为 'conformer' 或 'tcn'，得到 {encoder_type!r}。")

    def forward(self, Z: torch.Tensor, domain_id: torch.Tensor | None = None) -> torch.Tensor:
        """Z (B,T,D) → Z' (B,T,D)；domain_id (B,) 可选，用于域条件仿射。"""
        if Z.dim() != 3:
            raise ValueError(f"Z 须为 (B,T,D)，得到 {Z.shape}。")
        if self.encoder_type == "tcn" and self.blocks and all(
            block.pointwise_execution == "conv1d" for block in self.blocks
        ):
            channel_first = Z.transpose(1, 2)
            for block in self.blocks:
                channel_first = block.forward_channel_first(channel_first, domain_id)
            Z = channel_first.transpose(1, 2)
        else:
            for block in self.blocks:
                Z = block(Z, domain_id)
        if self.final_norm is not None:
            Z = self.final_norm(Z)
        return Z
