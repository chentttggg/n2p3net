"""模块 #12：总损失（Total Loss）。

职责（blueprint §8）：
    L = λ1·L_target + λ2·L_early + λ3·L_tau + λ_amp·L_amp + λ4·L_MMD
        - L_target：Head-A 主分类 BCE（pos_weight≈8，target 占 1/9）
        - L_early ：Head-B 早期证据 BCE（同标签，同样 pos_weight≈8）
        - L_tau   ：潜伏期正则 Σ(τ_c−τ0_c)²（τ 不偏离先验中心太远，逐试次偏移 Δτ 的软约束）
        - L_amp   ：Head-D 幅值损失（有 A+X 时回归 P3b 窗内 Pz 物理幅值，否则 L2 自一致性正则，
                    保证 constitution P7 的幅值头被真实消费）
        - L_MMD   ：跨域特征级 RBF-MMD（Phase 3 可选，默认 λ4=0 不启用）

明确「不做」：
    - σ 正则：σ 由 sigmoid 软参数化天然有界 [20,80]ms，蓝图 §8 明确无需额外正则。
    - 参考抖动/时间扭曲等数据增强：train/augment.py。

三思决策记录（供后续会话追溯）：
    D-pos-weight     L_target 与 L_early 均用 pos_weight≈8：两者共享 target/non-target 标签、同受
                     1/9 不平衡，不加 pos_weight 会学「全判非目标」。蓝图 Head-A 明确写 pos_weight，
                     Head-B 因「同标签」继承同一 pos_weight。
    D-tau-scale      **尺度修正**：蓝图 §8 写 λ3≈1e-2 + L_tau=Σ(τ−τ0)²，但 τ 是 ms 单位、Δτ∈±30ms，
                     使 Δτ² ~225–900，λ3·L_tau ≈ 2–9 会**远超 L_target(≈0.7)**、反客为主（蓝图未细算
                     尺度，同 Conformer 参数账问题）。修正：L_tau = mean((τ−τ0)²)/tau_scale²，默认
                     tau_scale=50ms（蓝图 §5 的 tanh 半幅），使初始 λ3·L_tau≈1e-3 ≪ L_target。此修正
                     不改变 L_tau 的方向（仍是「Δτ 别太大」），只把它校准到「小正则」的量级。
    D-tau0-not-supervised  L_tau 用 τ−τ0（不 detach），τ0 的梯度经「τ 中的 +τ0」与「−τ0」精确抵消为 0，
                     ​故 L_tau 只正则 Δτ（逐试次偏移），**不监督 τ0**（τ0 仅被分类损失经 τ→A→H 监督）。
                     这是蓝图「τ0 数据驱动（被分类监督找群体平均）」的语义。
    D-mmd-phase3     L_MMD 是 Phase 3 跨域组件，默认 λ4=0；接口预留 z_features/domain_ids，Phase 2 零开销。
    D-device         pos_weight 显式对齐 logits 的 device/dtype（AMP bf16 兼容）。

契约（输入 → 输出）：
    output(N2P3NetOutput) + tau0(3,) + y(B,1) → Losses{total, target, early, tau, amp, mmd}（均标量）。

依赖的决策：blueprint §5/§8、constitution P7/D7、heads.D-logit-out、component_window.D-tau-param。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from models.n2p3net import N2P3NetOutput


@dataclass
class Losses:
    """总损失及各分项（均 0 维标量 Tensor，可 backward）。"""

    total: torch.Tensor
    target: torch.Tensor
    early: torch.Tensor
    tau: torch.Tensor
    mmd: torch.Tensor
    amp: Optional[torch.Tensor] = None
    jit: Optional[torch.Tensor] = None


def _bce_with_pos_weight(
    logits: torch.Tensor, y: torch.Tensor, pos_weight: float
) -> torch.Tensor:
    """BCE with logits + pos_weight（对齐 device/dtype，D-pos-weight/D-device）。"""
    pw = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    if y.dim() == 1:
        y = y.view(-1, 1)  # (N,) → (N,1) 防御（review P3：data 层出 (N,) 标签）
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)


def tau_regularization(tau: torch.Tensor, tau0: torch.Tensor, tau_scale_ms: float) -> torch.Tensor:
    """L_tau = mean((τ−τ0)²) / tau_scale²（尺度修正 D-tau-scale；τ0 不被监督 D-tau0-not-supervised）。"""
    return ((tau - tau0[None, :]) ** 2).mean() / (tau_scale_ms ** 2)


def rbf_mmd2(
    x: torch.Tensor, y: torch.Tensor, bandwidth: Optional[float] = None
) -> torch.Tensor:
    """无偏 RBF-MMD²（两分布样本 x/y，D-mmd-phase3，Phase 3 用）。

    bandwidth=None 时用 median heuristic（样本并集的成对距离中位数）；固定 1.0 在 D=64
    下会 exp(-d²/2)→0、梯度全零（review v6 P1，实测 rbf_mmd2(x,x+1)≈-1e-16）。
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(f"rbf_mmd2 输入须为 (N,D) 二维张量，得到 x={x.shape}、y={y.shape}。")
    # torch.pdist 不支持 bf16；MMD 显式提升到 fp32 计算，梯度回传不受影响。
    x = x.float()
    y = y.float()
    if x.shape[0] < 2 or y.shape[0] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    if bandwidth is None:
        z = torch.cat([x, y], dim=0)
        if z.shape[0] < 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        bandwidth = float(torch.pdist(z).median().clamp_min(1e-6).item())
        if bandwidth <= 0.0:
            bandwidth = 1.0

    def gaussian_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a[:, None, :] - b[None, :, :]  # (n, m, D)
        return torch.exp(-(diff ** 2).sum(-1) / (2.0 * bandwidth ** 2))

    kxx = gaussian_kernel(x, x)
    kyy = gaussian_kernel(y, y)
    kxy = gaussian_kernel(x, y)
    n, m = x.shape[0], y.shape[0]
    xx = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
    yy = (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
    xy = kxy.sum() / (n * m)
    return xx + yy - 2.0 * xy


def compute_losses(
    output: N2P3NetOutput,
    tau0: torch.Tensor,
    y: torch.Tensor,
    *,
    lambda2: float = 0.3,
    lambda3: float = 1e-2,
    lambda_amp: float = 0.0,
    lambda_jit: float = 0.0,
    tau_shift: Optional[torch.Tensor] = None,
    shift_ms: Optional[torch.Tensor] = None,
    pos_weight: float = 8.0,
    tau_scale_ms: float = 50.0,
    z_features: Optional[torch.Tensor] = None,
    domain_ids: Optional[torch.Tensor] = None,
    main_domain: int = 0,
    aux_domain: int = 1,
    lambda4: float = 0.0,
    mmd_bandwidth: Optional[float] = None,
    X: Optional[torch.Tensor] = None,
    pz_channel: int = 3,
) -> Losses:
    """总损失 L = λ1·L_target + λ2·L_early + λ3·L_tau + λ_amp·L_amp + λ4·L_MMD。

    Parameters
    ----------
    output : N2P3NetOutput
        模型前向输出（含 heads 的 logit_target/logit_early、tau、attention）。
    tau0 : torch.Tensor
        (3,) 先验中心（component_window.tau0_bounded）。
    y : torch.Tensor
        (B,1) target 标签（1=target, 0=non-target）。
    lambda2 / lambda3 : float
        Head-B / L_tau 权重（蓝图 §8：λ2∈{0.1,0.3,0.5}、λ3≈1e-2）。
    lambda_amp : float
        Head-D 幅值损失权重（constitution P7；默认 0 保持旧接口，Trainer 默认开启）。
      lambda_jit / tau_shift / shift_ms : float / Tensor / Tensor
          Phase 2 自监督 jitter 一致性：tau_shift 是已知偏移 X_shift 的 τ，shift_ms 为
          (B,) 物理毫秒偏移；L_jit = mean((tau_shift - tau - shift_ms)²)/tau_scale²。
          无标签，仅锚定 τ 的物理 ms 尺度（E3 兼容）。
    pos_weight : float
        BCE 正样本权重（默认 8，target 占 1/9）。
    tau_scale_ms : float
        L_tau 的归一化尺度（默认 50ms，D-tau-scale）。
    z_features / domain_ids : Tensor | None
        Phase 3 跨域 MMD 的特征（(B,D)）与域标签（(B,)）；None 或 λ4=0 时跳过。
    lambda4 : float
        L_MMD 权重（Phase 3 启用，默认 0）。
      main_domain / aux_domain : int
          P9 域标签：main_domain（默认 0）= GTN；aux_domain（默认 1）= 辅助 P300 域。
          domain_ids 存在时，L_target/L_early/L_amp 只对 main_domain 样本计算；
          L_MMD 只在 main_domain 与 aux_domain 之间计算。辅助域标签严禁进入主监督。
    X : torch.Tensor | None
        (B,C,T) 原始/增强后输入；与 output.attention 一起提供时，L_amp 使用
        P3b 软窗对 Pz 信号的物理幅值回归（blueprint Head-D）。None 时退化为
        Head-D 幅值的 L2 自一致性正则，保证幅值头一定被损失消费（review v6 P1）。
    pz_channel : int
        Pz 在 X 通道轴上的索引（默认 3，对应标准 8 导蒙太奇）。
    """
    # P9 硬隔离：domain_ids 存在时，主监督只对 GTN（main_domain）样本计算。
    loss_device = output.heads.logit_target.device
    if domain_ids is not None:
        domain_ids = domain_ids.to(device=loss_device)
        if domain_ids.numel() != output.heads.logit_target.shape[0]:
            raise ValueError(
                f"domain_ids 长度须等于 batch_size，得到 {domain_ids.numel()} "
                f"vs {output.heads.logit_target.shape[0]}。"
            )
        main_mask = (domain_ids == main_domain).view(-1)
    else:
        main_mask = None

    if main_mask is not None and not bool(main_mask.any()):
        # 当前 batch 全是辅助域：主监督与 τ 自监督均为零，只保留可选的 L_MMD。
        L_target = torch.zeros((), device=output.heads.logit_target.device,
                               dtype=output.heads.logit_target.dtype)
        L_early = torch.zeros_like(L_target)
        L_amp = torch.zeros_like(L_target)
    else:
        logit_target = (
            output.heads.logit_target[main_mask]
            if main_mask is not None
            else output.heads.logit_target
        )
        logit_early = (
            output.heads.logit_early[main_mask]
            if main_mask is not None
            else output.heads.logit_early
        )
        y_main = y[main_mask] if main_mask is not None else y
        L_target = _bce_with_pos_weight(logit_target, y_main, pos_weight)
        L_early = _bce_with_pos_weight(logit_early, y_main, pos_weight)

        if lambda_amp > 0.0:
            if X is not None and output.attention is not None:
                # Head-D 物理幅值（P9：只对 GTN 样本计算）。
                pz = pz_channel if pz_channel < X.shape[1] else X.shape[1] - 1
                a_p3b = output.attention[:, 2, :]
                a_p3b = a_p3b[main_mask] if main_mask is not None else a_p3b
                x_main = X[main_mask] if main_mask is not None else X
                amp_main = output.heads.amplitude[main_mask] if main_mask is not None else output.heads.amplitude
                target_amp = (a_p3b * x_main[:, pz, :]).sum(dim=1)
                L_amp = F.mse_loss(amp_main.squeeze(-1), target_amp)
            else:
                amp = output.heads.amplitude[main_mask] if main_mask is not None else output.heads.amplitude
                L_amp = amp.pow(2).mean()
        else:
            L_amp = torch.zeros_like(L_target)

    # P9：domain_ids 存在时 L_tau 只作用于主域；辅助域只进 L_MMD。
    if main_mask is not None and not main_mask.any():
        L_tau = torch.zeros((), device=loss_device, dtype=output.tau.dtype)
    else:
        tau_main = output.tau[main_mask] if main_mask is not None else output.tau
        L_tau = tau_regularization(tau_main, tau0, tau_scale_ms)

    if lambda_jit > 0.0 and tau_shift is not None and shift_ms is not None:
        shift_ms = shift_ms.to(device=loss_device, dtype=output.tau.dtype)
        tau_shift_main = tau_shift[main_mask] if main_mask is not None else tau_shift
        tau_main = output.tau[main_mask] if main_mask is not None else output.tau
        shift_main = shift_ms[main_mask] if main_mask is not None else shift_ms
        if tau_main.shape[0] == 0:
            L_jit = torch.zeros((), device=loss_device, dtype=output.tau.dtype)
        else:
            L_jit = ((tau_shift_main - tau_main - shift_main[:, None]) ** 2).mean() / (
                tau_scale_ms ** 2
            )
    else:
        L_jit = torch.zeros_like(L_tau)

    if lambda4 > 0.0 and z_features is not None and domain_ids is not None:
        # N2P3NetOutput.features 是 (B,T,D)；MMD 契约是 (B,D)。这里按时间维平均池化，
        # 等价于用整个 epoch 的编码均值做域对齐；2D 输入则直接使用。
        if z_features.dim() == 3:
            z_pooled = z_features.mean(dim=1)
        elif z_features.dim() == 2:
            z_pooled = z_features
        else:
            raise ValueError(
                f"z_features 须为 (B,D) 或 (B,T,D)，得到 {z_features.shape}。"
            )
        d0 = z_pooled[domain_ids == main_domain]
        d1 = z_pooled[domain_ids == aux_domain]
        L_mmd = rbf_mmd2(d0, d1, bandwidth=mmd_bandwidth)
    else:
        L_mmd = torch.zeros_like(L_target)

    total = (
        L_target
        + lambda2 * L_early
        + lambda3 * L_tau
        + lambda_amp * L_amp
        + lambda_jit * L_jit
        + lambda4 * L_mmd
    )
    return Losses(
        total=total,
        target=L_target,
        early=L_early,
        tau=L_tau,
        amp=L_amp,
        mmd=L_mmd,
        jit=L_jit,
    )
