"""模块 #9：Stage 3 时域多任务头（Multi-Task Heads）。

职责（blueprint Stage 3）：
    把参数化成分窗的输出 H ∈ R^{B×3×D}（成分 N2/P3a/P3b 的软对齐表示）转成时域判别输出：
        - Head-A 主分类：concat(H_P3b, H_P3a, g_global) → MLP → logit_target（target/non-target 二分类）
          g_global = mean_pool(Z') 是全局时间池化旁路（Phase 2 失败诊断方案 B：解除 PCW 单点瓶颈）
        - Head-B 早期证据：H_N2 → MLP → logit_early（同标签的早期证据集成头）
        - Head-D 幅值：H_P3b → 线性投影 → amplitude（P3b 幅值，可解释性 P4）

明确「不做」（已在其他模块 / 留给其他模块）：
    - Head-C 潜伏期（τ/σ）：已在 component_window.py 内嵌，τ/σ 由其输出、直接进损失 L_tau，
      不经过本模块（本模块 forward 只用 H）。
    - 决策层（猜数字 argmax）：decision.py（用本模块的 logit_target 做对数似然比累积）。
    - 损失构造：train/losses.py（BCE + pos_weight≈8 等）。

三思决策记录（供后续会话追溯）：
    D-logit-out      heads 输出 **logit**（非概率），因决策层（§6）需「Σ logit(p_target)」做对数
                     似然比累积（独立证据的对数似然比可加，概率不可加）。p_target/p_early 作为
                     HeadsOutput 的 property 由 sigmoid 派生，供损失展示/评估用。
    D-head-a-input   Head-A 只 concat P3b + P3a（不含 N2），因 N2 已被 Head-B 单独消费（早期证据），
                     避免同一成分被两个头重复监督（多任务头职责正交）。
      D-global-bypass  2026-08-22 失败诊断方案 B：Head-A 额外 concat 全局时间均值池化旁路
                       g_global=mean_pool(Z')。实测 PCW 参数（τ0/σ/Δτ）梯度比分类头弱 3–4 个
                       数量级，成分窗接近「装饰性」，判别信息主要靠 encoder 的分布式时间特征；
                       旁路把该信息直连 Head-A，不再强迫全部判别信息穿过 3 个固定宽度软窗。
                       use_global_bypass=True 时 forward 必须显式提供 global_features；
                       默认开启（方案 B），False 保留旧结构作消融对照。
    D-head-b-view    Head-B 是「同标签的早期证据集成头」（multi-view ensemble + 正则，blueprint 5），
                     不是独立任务：它与 Head-A 共享 target 标签，提供 t<300ms 的早期证据视图。
    D-head-d-proxy   Head-D 幅值：蓝图写 Â=Σ_t A_P3b(t)·X_Pz(t)（需原始 Pz 信号），但本模块只吃
                     抽象表示 H。故用 Â=Linear(H_P3b→1) 作「P3b 表示幅值」代理（H_P3b 已是 P3b 窗内
                     软对齐表示，其投影幅值与生理 P3b 幅值正相关）。若要精确物理 μV 幅值，Phase 2
                     扩展：额外传 A + 归一化后原始 X 到 heads。此为可解释性辅助输出，非判别核心。
    D-hidden-dim     MLP 隐藏层用 d_model//2（P1 克制，数千试次下过宽隐藏层易过拟合）。
    D-device         纯 nn.Module，不硬编码设备（DP1）。

契约（输入 → 输出）：
    H ∈ R^{B×3×D}（成分顺序 N2/P3a/P3b）→ HeadsOutput{logit_target, logit_early, amplitude}，
    均 (B,1)；p_target/p_early 由 property 派生。
      global_features ∈ R^{B×D}（use_global_bypass=True 时必填，通常为 mean_pool(Z')）。

依赖的决策：blueprint §5（Stage 3）/§6（决策层）、constitution P7（时域多任务）、
    component_window（输出 H 的成分顺序）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class HeadsOutput:
    """多任务头输出（均 (B,1)）。

    Attributes
    ----------
    logit_target : torch.Tensor
        主分类 logit（target/non-target），供决策层对数似然比累积。
    logit_early : torch.Tensor
        早期证据 logit（Head-B）。
    amplitude : torch.Tensor
        P3b 幅值（Head-D，可解释性输出）。
    """

    logit_target: torch.Tensor
    logit_early: torch.Tensor
    amplitude: torch.Tensor

    @property
    def p_target(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_target)

    @property
    def p_early(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_early)


class MultiTaskHeads(nn.Module):
    """Stage 3 多任务头：H (B,3,D) [+ global_features (B,D)] → HeadsOutput。

    Parameters
    ----------
    d_model : int
        隐藏维 D（默认 64）。
    use_global_bypass : bool
        是否给 Head-A 加全局时间池化旁路（默认 True，方案 B D-global-bypass）。
    global_dim : int | None
        旁路特征维（默认 d_model；v5.1 separable-pool 旁路传 f2·T''）。
    use_mlp_heads : bool
        False（默认，EEGNet 借鉴）：Head-A/Head-B 用 Dropout + Linear 的瘦头；
        True：恢复旧版 D//2 隐藏层 MLP（接口回退）。
    dropout : float
        瘦头分类前的 dropout（EEGNet 默认 0.25）。
    """

    def __init__(
        self,
        d_model: int = 64,
        use_global_bypass: bool = True,
        global_dim: int | None = None,
        use_mlp_heads: bool = False,
        dropout: float = 0.25,
    ):
        super().__init__()
        hidden = max(d_model // 2, 1)
        self.use_global_bypass = bool(use_global_bypass)
        self.global_dim = int(global_dim) if global_dim is not None else d_model
        self.use_mlp_heads = bool(use_mlp_heads)
        self.dropout = float(dropout)

        # Head-A 主分类：concat(H_P3b, H_P3a[, g_global]) → (Dropout +) Linear/MLP → logit
        head_a_in = 2 * d_model + (self.global_dim if self.use_global_bypass else 0)
        if self.use_mlp_heads:
            self.head_a = nn.Sequential(
                nn.Linear(head_a_in, hidden), nn.GELU(), nn.Linear(hidden, 1)
            )
            self.head_b = nn.Sequential(
                nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1)
            )
        else:
            self.head_a = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(head_a_in, 1))
            self.head_b = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(d_model, 1))

        # Head-D 幅值：H_P3b → 线性投影 → amplitude
        self.head_d = nn.Linear(d_model, 1)

    def forward(self, H: torch.Tensor, global_features: torch.Tensor | None = None) -> HeadsOutput:
        """H (B,3,D) [+ global_features (B,D)] → HeadsOutput（logit_target / logit_early / amplitude，均 (B,1)）。"""
        if H.dim() != 3 or H.shape[1] != 3:
            raise ValueError(f"H 须为 (B,3,D)，得到 {H.shape}。")

        h_n2 = H[:, 0]  # (B, D)
        h_p3a = H[:, 1]  # (B, D)
        h_p3b = H[:, 2]  # (B, D)

        # Head-A：concat(P3b, P3a[, g_global])（不含 N2，见 D-head-a-input / D-global-bypass）
        parts = [h_p3b, h_p3a]
        if self.use_global_bypass:
            if global_features is None:
                raise ValueError("use_global_bypass=True 时 global_features 必填（契约 D-global-bypass）。")
            if global_features.dim() != 2 or global_features.shape[0] != H.shape[0]:
                raise ValueError(
                    f"global_features 须为 (B, global_dim)，得到 {tuple(global_features.shape)}，"
                    f"期望 ({H.shape[0]},{self.global_dim})。"
                )
            if global_features.shape[1] != self.global_dim:
                raise ValueError(
                    f"global_features 特征维 {global_features.shape[1]} 须等于 global_dim={self.global_dim}。"
                )
            parts.append(global_features.to(dtype=H.dtype))
        h_a = torch.cat(parts, dim=-1)
        logit_target = self.head_a(h_a)  # (B, 1)

        # Head-B：N2 早期证据
        logit_early = self.head_b(h_n2)  # (B, 1)

        # Head-D：P3b 幅值
        amplitude = self.head_d(h_p3b)  # (B, 1)

        return HeadsOutput(logit_target=logit_target, logit_early=logit_early, amplitude=amplitude)
