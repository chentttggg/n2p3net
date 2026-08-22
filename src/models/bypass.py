"""模块：TemporalPoolingBypass（分类旁路，EEGNet 借鉴）。

v5.1 失败诊断方案：PCW 的 τ/σ/Δτ 梯度比分类头弱 3–4 个数量级，判别信息主要在
encoder 的分布式时间特征。此旁路把 Z' 经 EEGNet 式 4×/8× 平均池化 + depthwise
separable 卷积压缩成 (B, F2·(T//pool1//pool2)) 特征，直连 Head-A。

与 constitution E5 的关系：
    E5（时间分辨率全程保留）继续约束「可解释路径」（PCW 与 τ/σ 读数仍吃全 T 的 Z'）；
    本旁路只属于「判别路径」，与 PCW 并行，不做时间池化就不能进入 PCW。
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalPoolingBypass(nn.Module):
    """Z' (B,T,D) → g (B, out_features)；4× 池化 → depthwise+pointwise → 8× 池化。

    Parameters
    ----------
    d_model : int
        输入特征维 D（默认 64）。
    f2 : int
        pointwise 输出通道（EEGNet 默认 16）。
    depthwise_kernel : int
        depthwise 时间卷积核长（采样点；默认 16 ≈ 62.5ms@256Hz）。
    pool1 / pool2 : int
        两段平均池化核（默认 4 / 8；T 须能被两者整除）。
    dropout : float
        pointwise 后 dropout（EEGNet 默认 0.25）。
    """

    def __init__(
        self,
        d_model: int = 64,
        f2: int = 16,
        depthwise_kernel: int = 16,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.25,
    ):
        super().__init__()
        if d_model <= 0 or f2 <= 0:
            raise ValueError(f"d_model/f2 须 >0，得到 {d_model}/{f2}。")
        if pool1 <= 0 or pool2 <= 0:
            raise ValueError(f"pool1/pool2 须 >0，得到 {pool1}/{pool2}。")
        if depthwise_kernel <= 0:
            raise ValueError(f"depthwise_kernel 须 >0，得到 {depthwise_kernel}。")
        self.d_model = int(d_model)
        self.f2 = int(f2)
        self.pool1 = int(pool1)
        self.pool2 = int(pool2)
        self.out_features = int(f2)

        self.pool1_layer = nn.AvgPool1d(self.pool1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=depthwise_kernel, groups=d_model,
            padding=depthwise_kernel // 2,
        )
        self.pointwise = nn.Conv1d(d_model, f2, kernel_size=1)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        # 第二段用 adaptive pool 固定到 pool2 个时间 bin（EEGNet 最终 8 bin），
        # 使输出特征维不随 T 变化、Linear 头权重可复用（动态 T 兼容）。
        self.pool2_layer = nn.AdaptiveAvgPool1d(self.pool2)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Z (B,T,D) → g (B, f2·pool2)。"""
        if Z.dim() != 3:
            raise ValueError(f"Z 须为 (B,T,D)，得到 {Z.shape}。")
        B, T, D = Z.shape
        if D != self.d_model:
            raise ValueError(f"Z 特征维 {D} 须等于 d_model={self.d_model}。")
        if T % self.pool1 != 0:
            raise ValueError(f"T={T} 须能被 pool1={self.pool1} 整除。")
        h = Z.transpose(1, 2)  # (B,D,T)
        h = self.pool1_layer(h)  # T//pool1
        h = self.depthwise(h)
        h = self.elu(h)
        h = self.pointwise(h)
        h = self.dropout(h)
        h = self.pool2_layer(h)  # (B,F2,pool2)
        h = h.flatten(1)  # (B, f2*pool2)
        self.out_features = int(h.shape[1])
        return h
