"""模块 #7：Stage 2 序列编码（Sequence Encoder）。

职责（blueprint Stage 2 的 2.1）：
    对 Stage 1 输出 Z ∈ R^{B×T×D} 做「跨时间」序列编码，补齐 N2→P3b 约 150ms 的中程关系
    （Stage 1 多尺度卷积已覆盖 ~500ms 局部上下文，故此处 ROI 低、须轻量）。

结构（消融轴 depth ∈ {0,1,2,3}，默认 depth=1）：
    - depth=0：恒等（identity），参数化成分窗直接吃 Stage 1 输出（零开销地板）。
    - depth>0：堆叠 depth 个 block，两种类型（encoder_type）：
        · "conformer"：轻量 Pre-LN Transformer block（MHSA + FFN，FFN expansion=2，LayerNorm + 域条件仿射）；
        · "tcn"：膨胀 depthwise TCN（dilation 1/4/16，depthwise + pointwise，LayerNorm + 域条件仿射）。

明确「不做」（留给下游模块）：
    - 参数化成分窗（2.2，component_window.py）
    - 多任务头（Stage 3，heads.py）

三思决策记录（供后续会话追溯）：
    D-depth-axis     depth ∈ {0,1,2,3} 是消融轴（blueprint D6）。depth=0 是 identity（参数 0），
                     既是「序列编码 ROI 低」的诚实地板，也是超预算时的降容路径。
                     本模块独立默认 conformer depth=1（保留双选项）；N2P3Net 组装时默认用
                     tcn depth=3（2026-08-20 决策，见 n2p3net.D-budget / constitution E4）。
    D-lite-conformer 「轻量 Conformer」= 去掉卷积模块的 Pre-LN Transformer block（MHSA + FFN）。
                     理由：Stage 1 已有多尺度时间卷积覆盖局部上下文，Stage 2 再加卷积模块是冗余
                     （D6「容量非瓶颈、域差才是」）。FFN expansion=2 是相对标准 4 的轻量化。
    D-budget        参数账（诚实记录）：D=64 下 depth=1 Conformer ≈ 33k（MHSA 16.6k + FFN 16.6k），
                     加 tokenizer ≈17k 合计 ≈50k，**已触及 E4 上限**。TCN 更轻（depth=3 ≈13k）。
                     集成联调（Phase C）若总预算超 50k，降容路径：depth→0 或 encoder_type→"tcn"。
                     本模块不做「偷偷减容」，参数账公开、由 roadmap Phase 2 验收裁决。
    D-no-batchnorm  全程 LayerNorm，无 BatchNorm（constitution D7）；跨域对齐用「域条件仿射」
                     （per-domain 可学习 scale/shift，加于 LayerNorm 后），不用 Split-BN。
    D-domain-affine  域条件仿射是 Phase 3 跨域微调的组件；Phase 2 单受试 n_domains=None 时不启用
                     （零参数零开销）。接口预留 domain_id 入参，Phase 3 无需改契约。**初始化为恒等**
                     （scale=1、shift=0），使初始状态 = 无跨域对齐，训练中逐步学到域差异——与
                     tokenizer 的坐标调制初始化为 0 同一模式（从「无」出发，而非从随机出发）。
    D-tcn-receptive 膨胀 TCN dilation (1,4,16) 感受野 ≈ 168ms（3 层时），足以覆盖 N2→P3b ~150ms；
                     blueprint「~500ms」系把 Stage 1 感受野（核长 129 ≈ 504ms）计入，Stage 2 无需
                     重复覆盖 500ms（Stage 1 已做）。TCN 层数由 depth 控制，dilation 取前 depth 个。
    D-final-norm     depth>0 时加 final LayerNorm（Pre-LN 惯例，利于下游成分窗读数）；depth=0 不加
                     （identity 不改变 tokenizer 输出）。
    D-device         纯 nn.Module，不硬编码设备（device-portability DP1）。

契约（输入 → 输出）：
    Z ∈ R^{B×T×D}（+ 可选 domain_id ∈ Z^{B}）→ Z' ∈ R^{B×T×D}（形状不变）。

依赖的决策：blueprint §4（Stage 2）、constitution D6/D7/E4/E5、device-portability DP1。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn


def _apply_domain_affine(
    h: torch.Tensor,
    scale: Optional[torch.Tensor],
    shift: Optional[torch.Tensor],
    domain_id: Optional[torch.Tensor],
) -> torch.Tensor:
    """域条件仿射（D-domain-affine）：h → h * scale[domain] + shift[domain]。"""
    if scale is not None and domain_id is not None:
        s = scale[domain_id]  # (B, D)
        b = shift[domain_id]  # (B, D)
        return h * s.unsqueeze(1) + b.unsqueeze(1)
    return h


class _ConformerBlock(nn.Module):
    """轻量 Conformer block（Pre-LN Transformer：MHSA + FFN，无卷积模块，见 D-lite-conformer）。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_expansion: int,
        dropout: float,
        n_domains: Optional[int],
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

    def forward(self, x: torch.Tensor, domain_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = _apply_domain_affine(self.ln1(x), self.dom_scale, self.dom_shift, domain_id)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        h = _apply_domain_affine(self.ln2(x), self.dom_scale, self.dom_shift, domain_id)
        x = x + self.ffn(h)
        return x


class _TCNBlock(nn.Module):
    """膨胀 depthwise TCN block（depthwise + pointwise + GELU，见 D-tcn-receptive）。

    GLM 消融轴 ``norm``："ln"（默认，旧版）或 "bn"。BN 在 (B,T) 上按特征维归一化，
    保留 token 间相对幅值的同时稳定特征尺度——跨被试 P300 文献反复报告 BN 是
    CNN 泛化的关键组件（Värbu 2020：ELU+dropout+BN 与最佳 CNN 性能相关）；
    LN 是逐 token 归一化，会抹平单 token 的幅值维度。默认仍 ln，bn 供 A/B。
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        n_domains: Optional[int],
        causal: bool = False,
        norm: str = "ln",
    ):
        super().__init__()
        if norm not in ("ln", "bn"):
            raise ValueError(f"norm 须为 'ln' 或 'bn'，得到 {norm!r}。")
        self.norm_type = norm
        if norm == "bn":
            self.ln = nn.BatchNorm1d(d_model)  # 输入 (B,D,T)：跨 batch/时间归一化
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

    def forward(self, x: torch.Tensor, domain_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = _apply_domain_affine(self._apply_norm(x), self.dom_scale, self.dom_shift, domain_id)
        h = h.transpose(1, 2)  # (B, D, T)
        h = self.depthwise(h)
        # 因果 padding 会让 Conv1d 输出比输入长，裁回原 T；非因果 padding=same 无影响。
        h = h[..., : x.shape[1]]
        h = self.act(h)
        h = self.pointwise(h)
        h = self.dropout(h)
        h = h.transpose(1, 2)  # (B, T, D)
        return x + h


class Stage2Encoder(nn.Module):
    """Stage 2 序列编码：Z (B,T,D) → Z' (B,T,D)，depth 消融 + conformer/tcn 双实现。

    Parameters
    ----------
    d_model : int
        隐藏维 D（默认 64）。
    depth : int
        序列编码层数 {0,1,2,3}（消融轴，D-depth-axis）。depth=0 为恒等。
    encoder_type : str
        "conformer"（轻量 Transformer，默认）或 "tcn"（膨胀 depthwise TCN）。
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
        TCN 膨胀系数序列，取前 depth 个（默认 (1, 4, 16)）。
      tcn_causal : bool
          TCN 是否因果 padding（默认 False，离线整段编码；True 供未来在线推理）。
      norm : str
          TCN block 的归一化（GLM 消融轴）："ln"（默认，旧版）或 "bn"（跨 batch/时间
          的 BatchNorm1d；见 _TCNBlock 文档）。conformer 路径不受此参数影响（恒 LN）。
    """

    def __init__(
        self,
        d_model: int = 64,
        depth: int = 1,
        encoder_type: str = "conformer",
        n_heads: int = 4,
        ffn_expansion: int = 2,
        dropout: float = 0.1,
        n_domains: Optional[int] = None,
        tcn_kernel: int = 3,
        tcn_dilations: Sequence[int] = (1, 4, 16),
        tcn_causal: bool = False,
        norm: str = "ln",
    ):
        super().__init__()
        self.d_model = d_model
        self.depth = int(depth)
        self.encoder_type = encoder_type
        if self.depth < 0:
            raise ValueError(f"depth 须 ≥0，得到 {self.depth}。")
        if self.depth > 3:
            raise ValueError(f"depth 消融轴仅支持 0..3，得到 {self.depth}。")
        if encoder_type not in ("conformer", "tcn"):
            raise ValueError(f"encoder_type 须为 'conformer' 或 'tcn'，得到 {encoder_type!r}。")
        if norm not in ("ln", "bn"):
            raise ValueError(f"norm 须为 'ln' 或 'bn'，得到 {norm!r}。")

        if self.depth == 0:
            self.blocks = nn.ModuleList()
            self.final_norm: Optional[nn.LayerNorm] = None
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
            dilations = tuple(tcn_dilations)
            if self.depth > len(dilations):
                raise ValueError(
                    f"TCN depth({self.depth}) 不能超过 tcn_dilations 长度({len(dilations)})；"
                    f"depth 超过提供的膨胀系数会被静默截断（review v6 P1）。"
                )
            dilations = dilations[: self.depth]
            self.blocks = nn.ModuleList(
                [
                    _TCNBlock(d_model, tcn_kernel, d, dropout, n_domains, causal=tcn_causal, norm=norm)
                    for d in dilations
                ]
            )
            self.final_norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"encoder_type 须为 'conformer' 或 'tcn'，得到 {encoder_type!r}。")

    def forward(
        self, Z: torch.Tensor, domain_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Z (B,T,D) → Z' (B,T,D)；domain_id (B,) 可选，用于域条件仿射。"""
        if Z.dim() != 3:
            raise ValueError(f"Z 须为 (B,T,D)，得到 {Z.shape}。")
        for block in self.blocks:
            Z = block(Z, domain_id)
        if self.final_norm is not None:
            Z = self.final_norm(Z)
        return Z
