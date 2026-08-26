"""模块 #5：门控加权再参考层（Gated Weighted Rereference，GLM v2）。

职责（blueprint Stage 0 的 0.1）：
    out = X − g ⊙ (1·wᵀX)（仅对存在通道）：
        - w ∈ R^C 过 softmax 归一化（Σw=1，有效再参考变换的数学空间）；
        - g ∈ R^C 是**自由线性门**，init=0 → **精确恒等映射**（保留原始记录参考）。

设计演进（三思记录）：
    v1（旧版）：out = X − 1·wᵀX，w=softmax 均匀初始化 → 强制 CAR。
    v2（GLM 2026-08-23）：发现 v1 在鼻参考少导数据上**销毁判别信息**——GTN 实测
    3 导均匀 CAR 把 Pz 的 P3b 差值 10.8μV 砍到 2.5μV（信号损失 4.36×、单试次 SNR
    损失 1.90×、Fz 反转成 −4.0μV 伪信号），AUC 损失 1.5~3.3pt。文献同判：CAR 需要
    ≥32~64 导均匀覆盖才近似有效（Junghöfer et al. 2001；Luck 2014），<32 导时其偏差
    可能比单物理参考更糟；P300 惯例是鼻尖/乳突参考（上海 ERP 临床共识）。
    v2 修复：softmax 约束 Σw=1 使「恒等」（w=0）不可表达——这是 v1 的结构缺陷，
    加门 g 修复；g=0 起步 → 恒等，梯度 ∂L/∂g 经 m 直接回传、无饱和（对照：sigmoid
    门在 0 附近有 g(1−g) 因子，重蹈 tau0 梯度饥饿）。

三思决策记录（供后续会话追溯）：
    D-ref-space      w=softmax 保留：{X − 1·wᵀX : Σw=1} 正是从任意记录参考出发
                     可达的再参考变换全集（w=单通道 → 该电极参考；w=均匀 → CAR；
                     w=乳突对 → 联结乳突）。层的能力空间不变，只改「用多少」。
    D-glm-gate       g 自由线性门（无激活、无界），init=0：
                     ① 恒等可表达（修复 v1 结构缺陷）；
                     ② 梯度健康：∂out/∂g = −m(t)，与 w 路径解耦、无饱和区；
                     ③ 可解释：训练后 g·w 即每通道的有效参考权重（报告值）；
                     ④ 负值/超界（过减/反相参考）留给优化器——若学出 g>1 或 <0，
                        是数据证据而非约束失败，监控即可。
    D-glm-per-domain n_domains 给定时 w_logits/gate_raw 形状 (D,C)，按 batch 的
                     domain_id 逐样本取行：跨数据集（鼻参考 GTN ↔ 平均参考 ERP CORE
                     ↔ A1 耳参考自采 8 导）时每域学自己的参考变换，为 Phase 3 的
                     「参考无关」保留通路。单域（None）时共享 (C,)。
    D-mask-rezero    缺失通道恒 0（防幻象通道，v4 修正）：m 只由存在通道计算
                     （mask 重归一化），且**只对存在通道减 g·m**。
    D-ref-invariance 旧版「对共同常数偏移严格不变」在门开启时仍成立（g=1 时
                     offset·(1−g)→0）；g<1 时残留 offset·(1−g) 由下游基线段标准化
                     （μ_b 减除）兜底——语义测试相应更新。
    D-device         纯 nn.Module，不硬编码设备（DP1）；参数在 forward 内对齐 X.dtype。

契约（输入 → 输出）：
    输入 X ∈ R^{B×C×T}；输出 X_ref ∈ R^{B×C×T}（形状不变）。
    channel_mask ∈ {0,1}^C 可选；domain_id ∈ Z^B 可选（n_domains 给定时生效）。

依赖的决策：blueprint 0.1 / D-glm-gate / D-glm-per-domain、constitution P2、
    device-portability DP1。
"""

from __future__ import annotations

import torch
from torch import nn


class WeightedRereference(nn.Module):
    """门控加权再参考：out = X − g ⊙ (1·wᵀX)，w=softmax(w_logits)，g=自由门(init 0)。

    Parameters
    ----------
    n_channels : int
        通道数 C（默认 8）。
    use_gain : bool
        是否启用每通道增益（吸收设备增益差，蓝图 0.1 可选）；默认关闭。
    n_domains : int | None
        域数（Phase 3 跨数据集）；None 时 w/g 共享（单参数集）。
    gate_init : float
        门初值（默认 0.0 = 恒等；1.0 ≈ 旧版强制 CAR 方向）。
    """

    def __init__(
        self,
        n_channels: int = 8,
        use_gain: bool = False,
        n_domains: int | None = None,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.n_domains = n_domains
        shape = (n_channels,) if n_domains is None else (n_domains, n_channels)
        # logits 初始化为 0 → softmax 后均匀 1/C（w 方向的先验：CAR 方向，仅当门打开才生效）
        self.w_logits = nn.Parameter(torch.zeros(*shape))
        # GLM v2：自由线性门，init=0 → 精确恒等（保留记录参考；见 D-glm-gate）
        self.gate_raw = nn.Parameter(torch.full(shape, float(gate_init)))
        if use_gain:
            self.gain = nn.Parameter(torch.ones(n_channels))
        else:
            self.register_parameter("gain", None)

    @property
    def w(self) -> torch.Tensor:
        """归一化参考权重 w（Σw=1，末维 softmax）。"""
        return torch.softmax(self.w_logits, dim=-1)

    @property
    def gate(self) -> torch.Tensor:
        """每通道自由门（init 0 = 恒等；训练后 g·w 即有效参考权重）。"""
        return self.gate_raw

    def effective_reference(self) -> torch.Tensor:
        """可解释性读数：每通道的有效参考权重 g·w（(C,) 或 (D,C)）。"""
        return self.gate_raw * torch.softmax(self.w_logits, dim=-1)

    def _select(
        self,
        param: torch.Tensor,
        domain_id: torch.Tensor | None,
        B: int,
        C: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """按 domain_id 取参数行；(C,) 共享 → 广播 (B,C)；(D,C) → 索引 (B,C)。

        n_domains 给定但 domain_id 为 None 时回退域 0（主域 = GTN，P9：推理/评估
        只吃主域测试 fold，不含辅助域样本）。
        """
        p = param.to(dtype=dtype, device=device)
        if self.n_domains is None:
            return p.unsqueeze(0).expand(B, C)
        if domain_id is None:
            row0 = p[0]  # (C,) 主域行（GTN-only 推理路径）
            return row0.unsqueeze(0).expand(B, C)
        idx = domain_id.to(device=device, dtype=torch.long)
        if idx.shape[0] != B:
            raise ValueError(f"domain_id 长度 {idx.shape[0]} 须等于 batch {B}。")
        return p[idx]  # (B, C)

    def forward(
        self,
        X: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        domain_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """X ∈ R^{B×C×T} → X_ref ∈ R^{B×C×T}。

        channel_mask : (C,) bool 可选（True=通道存在）。提供时：①参考权重 w 在存在
                       通道上重归一化（零通道不稀释参考）；②只对存在通道减 g·m，
                       缺失通道保持 0（防幻象通道）。
        domain_id : (B,) long 可选，n_domains 给定时逐样本选 w/g（Phase 3）。
        """
        if X.dim() != 3:
            raise ValueError(f"X 须为 (B,C,T)，得到 {X.shape}。")
        B, C, T = X.shape
        dtype, device = X.dtype, X.device

        w = self._select(self.w, domain_id, B, C, dtype, device)  # (B, C)
        gate = self._select(self.gate_raw, domain_id, B, C, dtype, device)  # (B, C)

        mask: torch.Tensor | None = None
        if channel_mask is not None:
            mask = channel_mask.to(device=device, dtype=dtype)
            if mask.shape == (C,):
                mask = mask.unsqueeze(0).expand(B, -1)
            if mask.shape != (B, C):
                raise ValueError(
                    f"channel_mask must be ({C},) or ({B},{C}), got {tuple(mask.shape)}."
                )
            w = w * mask
            w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)  # 存在通道上重归一化

        m = torch.einsum("bc,bct->bt", w, X)  # (B, T) 参考信号 m = wᵀX
        subtract = gate.unsqueeze(-1) * m.unsqueeze(1)  # (B, C, 1)×(B, 1, T) → (B, C, T)
        if mask is not None:
            # 只对存在通道减除，缺失通道保持恒 0（避免幻象通道）
            out = X - subtract * mask.unsqueeze(-1)
        else:
            out = X - subtract
        if self.gain is not None:
            out = out * self.gain.view(1, -1, 1)
        return out
