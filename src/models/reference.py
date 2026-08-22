"""模块 #5：加权再参考层（Weighted Rereference）。

职责（blueprint Stage 0 的 0.1）：
    可学习加权再参考 R = I − 1·wᵀ，所有通道减同一个加权均值 1·(wᵀX)，替代硬编码 CAR。
    w ∈ R^C 过 softmax 归一化（Σw=1），初始化为均匀 = CAR（common average reference）。

明确「不做」（留给 Stage 0 后续 / train 层）：
    - 基线校正、幅值标准化（Stage 0 的 0.2/0.4）
    - 参考抖动增强（0.3，训练期数据增强）

三思决策记录（供后续会话追溯）：
    D-reref-direction  外积方向必须是 1·wᵀ（所有通道减同一加权均值 wᵀX），**不是** w·1ᵀ
                       （那是逐通道增益 w_c·Σ_j x_j，每通道减量不同、破坏「参考无关」语义；
                       review v1 笔误，v3 修正）。
    D-w-softmax        w 存为 logits（无约束参数），forward 时 softmax → Σw=1 恒成立、无需额外约束；
                       初始 logits=0 → 均匀 1/C = CAR。
    D-ref-invariance   加权再参考对「所有通道加同一常数偏移」严格不变（因 wᵀ1=Σw=1），
                       这是语义测试的核心不变量，也是「参考无关」的数学含义。
    D-mask-rezero      缺失通道恒 0（防幻象通道，v4 修正）：m 只由存在通道计算（mask 重归一化）
                       且**只对存在通道减 m**。若对所有通道减 m，零填充的缺失通道会变 −m(t)，
                       被下游基线段标准化（÷m 的小 std）放大成 std≈1 的「幻象通道」——逐试次
                       变化、所有缺失位置是同一幻象，坐标调制会误读为枕/顶区地形证据（review
                       复审实测：缺失通道 std≈1.06 ≥ 存在通道 1.04）。
                       定理：基线段标准化湮灭任何时间常数填充（μ_b=常数 → x−μ_b=0），故恒 0
                       与「可学习 mask token」数学等价——填 0 即完备解，无需 mask token。
    D-device-agnostic  纯 nn.Module，不硬编码任何设备（DP1）；设备由调用方 .to(DEVICE) 决定。

契约（输入 → 输出）：
    输入 X ∈ R^{B×C×T}；输出 X_ref ∈ R^{B×C×T}（每通道减同一加权均值，形状不变）。

依赖的决策：blueprint 0.1、constitution P2（再参考以可学习形式吸收进网络）、device-portability DP1。
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class WeightedRereference(nn.Module):
    """可学习加权再参考：X_ref = X − 1·(wᵀX)，w = softmax(w_logits)。

    Parameters
    ----------
    n_channels : int
        通道数 C（默认 8）。
    use_gain : bool
        是否启用每通道增益（吸收设备增益差，蓝图 0.1 可选）；默认关闭。
    """

    def __init__(self, n_channels: int = 8, use_gain: bool = False):
        super().__init__()
        # logits 初始化为 0 → softmax 后均匀 1/C = CAR
        self.w_logits = nn.Parameter(torch.zeros(n_channels))
        if use_gain:
            self.gain = nn.Parameter(torch.ones(n_channels))
        else:
            self.register_parameter("gain", None)

    @property
    def w(self) -> torch.Tensor:
        """归一化权重 w ∈ R^C，Σw=1。"""
        return torch.softmax(self.w_logits, dim=0)

    def forward(self, X: torch.Tensor, channel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """X ∈ R^{B×C×T} → X_ref ∈ R^{B×C×T}。

        channel_mask : (C,) bool 可选（True=通道存在）。提供时：①缺失通道权重置 0 后重归一化
                       （避免零通道稀释加权均值）；②**只对存在通道减 m**（缺失通道保持 0），
                       否则填 0 的缺失通道会变 −m、被基线段标准化放大成「幻象通道」（review 复审）。
        """
        if X.dim() != 3:
            raise ValueError(f"X 须为 (B,C,T)，得到 {X.shape}。")
        # 显式 dtype 对齐（review v4 P1 2.3 / review v6 P1）：bf16 直接输入且不开 autocast 时
        # 混用 float32 参数会 dtype mismatch 或 promote。
        w = self.w.to(X.dtype)  # (C,)
        mask: Optional[torch.Tensor] = None
        if channel_mask is not None:
            mask = channel_mask.to(dtype=w.dtype)
            w = w * mask
            w = w / w.sum().clamp(min=1e-8)  # 重归一化，防全零
        m = torch.einsum("c,bct->bt", w, X)  # (B,T) 加权均值 m = wᵀX
        if mask is not None:
            # 只对存在通道减 m，缺失通道保持恒 0（避免幻象通道）
            out = X - m.unsqueeze(1) * mask[None, :, None]
        else:
            out = X - m.unsqueeze(1)
        if self.gain is not None:
            out = out * self.gain.view(1, -1, 1)
        return out
