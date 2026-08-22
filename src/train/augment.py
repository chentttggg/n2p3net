"""模块 #13：ERP 感知数据增强（Data Augmentation）。

职责（blueprint §7 3.3 + §2 0.3）：
    训练期对单试次 EEG X ∈ R^{B×C×T} 施加 ERP 感知增强，提升小样本下的泛化与鲁棒性：
        - reference_jitter  参考抖动（随机重参考到随机凸组合，教参考不变性，§2 0.3）
        - time_warp         时间扭曲（PMB-TW 分段线性，模拟 P300 潜伏期抖动）
        - amplitude_jitter  幅值抖动（随机缩放）
        - gaussian_noise    高斯噪声（相对信号 std）
        - channel_dropout   通道 dropout（随机置 0 模拟通道缺失）

明确「不做」：
    - 不在验证/测试集上做增强（增强仅训练期，由 trainer 控制调用）。
    - 不做生成式增强（GAN/扩散），数据量小、易过拟合（P1）。

三思决策记录（供后续会话追溯）：
    D-ref-jitter     参考抖动用「随机凸组合参考」（非单一随机通道）：凸组合比单通道更一般、更接近真实
                     平均参考的变体，且零参数（§2 0.3）。只在训练期以概率 p 施加，教网络参考不变性。
    D-time-warp      PMB-TW 用 numpy np.interp 实现（分段线性、3 内部锚点、偏移有界 ±max_shift 采样点）。
                     选 numpy 而非 torch grid_sample：1D 时间扭曲用 np.interp 更直观、少出错；代价是
                     .cpu().numpy() 往返（数据预处理层可接受）。端点固定、内部锚点 clip 保证单调。
    D-relative-noise 高斯噪声用「相对信号 std」（sigma × std(X)），对归一化/原始 μV 数据都适用，
                     避免依赖数据绝对尺度。
    D-ch-dropout      通道 dropout 置 0（非 NaN/mask）：tokenizer 输入已约定缺失通道填 0，训练期随机
                     置 0 与之语义一致（模拟通道缺失），且不引入 NaN 传播。
    D-order           各增强相互独立、可交换（无顺序依赖），apply_augmentations 按固定顺序组合。
    D-device          所有增强保持输入 device/dtype（time_warp 在出口恢复），AMP 兼容。

契约（输入 → 输出）：
    X ∈ R^{B×C×T} → X' ∈ R^{B×C×T}（形状不变）。

依赖的决策：blueprint §2 0.3 / §7 3.3、constitution P1、tokenizer.D-mask-zero（缺失通道置 0）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def reference_jitter(
    X: torch.Tensor, p: float = 0.5, channel_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """参考抖动：以概率 p 随机重参考到随机凸组合（D-ref-jitter）。

    channel_mask : (C,) bool 可选；提供时凸组合只由存在通道计算，且只对存在通道减参考，
                   缺失通道保持 0（review v6 P0-2，避免幻象通道）。
    """
    B, C, _ = X.shape
    mask = torch.rand(B, device=X.device) < p
    if not mask.any():
        return X
    w = torch.rand(B, C, device=X.device)
    if channel_mask is not None:
        w = w * channel_mask.to(device=X.device, dtype=w.dtype)[None, :]
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)  # 凸组合权重（存在通道上重归一化）
    ref = (X * w[:, :, None]).sum(dim=1)  # (B, T) 凸组合参考
    if channel_mask is not None:
        subtract = ref[:, None, :] * channel_mask.to(device=X.device, dtype=X.dtype)[None, :, None]
        X_jitter = X - subtract
    else:
        X_jitter = X - ref[:, None, :]
    return torch.where(mask[:, None, None], X_jitter, X)


def known_time_shift(X: torch.Tensor, shift_samples: torch.Tensor) -> torch.Tensor:
    """按已知整数采样点偏移平移每条 trial（L_jit 自监督用）。

    X: (B,C,T)；shift_samples: (B,) 整数。正数表示把波形整体向时间轴右侧移动：
        X_shift[:, :, t] = X[:, :, t - shift]；越界位置填 0。
    该操作使用 gather，可回传梯度，且偏移量精确已知，用于
    tau(x_shift) ≈ tau(x) + shift_ms 的尺度锚定。
    """
    B, C, T = X.shape
    dev = X.device
    shift = shift_samples.to(device=dev, dtype=torch.long)
    idx = torch.arange(T, device=dev)[None, :] - shift[:, None]  # (B,T)
    valid = ((idx >= 0) & (idx < T)).to(dtype=X.dtype)  # (B,T)
    safe = idx.clamp(min=0, max=T - 1)[:, None, :].expand(B, C, T)
    return X.gather(dim=2, index=safe) * valid[:, None, :]


def time_warp(X: torch.Tensor, max_shift: int = 8) -> torch.Tensor:
    """PMB-TW 分段线性时间扭曲：模拟 P300 潜伏期抖动（D-time-warp）。

    GPU 原生向量化实现（D-time-warp-gpu）：锚点扰动、单调性修正、逆映射与线性插值
    全部用 PyTorch 张量运算在 X 所在设备上完成，替代原来的 B×C 双重 Python 循环 +
    CPU/GPU 往返。端点锚点固定，因此输出首末采样点与输入严格一致。
    """
    B, C, T = X.shape
    device = X.device
    if T < 2:
        return X

    n_anchors = 3
    pos_dtype = torch.float32
    src = torch.linspace(0, T - 1, n_anchors + 2, device=device, dtype=pos_dtype)  # (5,)

    # 1. 逐 batch 扰动内部锚点，并强制单调（与旧 np.interp 版本的端点/单调语义一致）
    # 必须 clone：B=1 时 expand().contiguous() 会返回与 src 共享存储的视图，
    # 后续 dst[:,1:-1]+=shifts 会原地污染 src，破坏逆映射（review audit P1）。
    dst = src.unsqueeze(0).expand(B, -1).clone()  # (B, 5)
    shifts = (torch.rand(B, n_anchors, device=device, dtype=pos_dtype) * 2.0 - 1.0) * max_shift
    dst[:, 1:-1] += shifts
    dst = dst.clamp(0, T - 1)
    dst = torch.cummax(dst, dim=1).values
    dst = torch.flip(torch.cummin(torch.flip(dst, dims=[1]), dim=1).values, dims=[1])

    # 2. 每个输出时间点 t 反向查找其在旧时间轴的位置 old_pos（searchsorted + 线性插值）
    t_idx = torch.arange(T, device=device, dtype=pos_dtype).unsqueeze(0).expand(B, T).contiguous()
    right = torch.searchsorted(dst, t_idx).clamp(1, n_anchors + 1)  # (B, T)
    left = right - 1
    x0 = dst.gather(1, left)  # (B, T)
    x1 = dst.gather(1, right)  # (B, T)
    y0 = src[left]  # (B, T)
    y1 = src[right]  # (B, T)
    frac = ((t_idx - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0.0, 1.0)
    old_pos = y0 + (y1 - y0) * frac  # (B, T) ∈ [0, T-1]

    # 3. 用 gather 对全部 batch/channel 同时做一维线性插值（不再逐通道 np.interp）
    idx0 = old_pos.floor().long().clamp(0, T - 1).unsqueeze(1).expand(B, C, T)
    idx1 = (idx0 + 1).clamp(0, T - 1)
    w = (old_pos.unsqueeze(1) - idx0.to(pos_dtype)).to(X.dtype)  # (B, C, T)
    x0_val = X.gather(2, idx0)
    x1_val = X.gather(2, idx1)
    return x0_val * (1.0 - w) + x1_val * w


def amplitude_jitter(X: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
    """幅值抖动：X *= (1 + ε)，ε ~ U(−scale, scale)。"""
    eps = (torch.rand(X.shape[0], 1, 1, device=X.device) * 2 - 1) * scale
    return X * (1.0 + eps)


def gaussian_noise(
    X: torch.Tensor, sigma: float = 0.1, channel_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """高斯噪声：X += N(0, (sigma·std(X))²)（相对信号 std，D-relative-noise）。

    channel_mask : (C,) bool 可选；提供时噪声只加在存在通道上，缺失通道保持 0
                   （review v6 P0-2）。
    """
    if channel_mask is not None:
        present = channel_mask.to(device=X.device, dtype=torch.bool)
        std = X[:, present, :].std() if present.any() else 1.0
        noise = torch.randn_like(X) * (sigma * std)
        noise = noise * present.to(dtype=X.dtype)[None, :, None]
        return X + noise
    std = X.std() if X.numel() > 0 else 1.0
    noise = torch.randn_like(X) * (sigma * std)
    return X + noise


def channel_dropout(
    X: torch.Tensor, p: float = 0.2, channel_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """通道 dropout：随机置 0 通道（D-ch-dropout）。"""
    B, C, _ = X.shape
    mask = torch.rand(B, C, 1, device=X.device) >= p  # True=保留
    if channel_mask is not None:
        # 缺失通道始终不参与 dropout（保持 0）
        mask = mask | (~channel_mask.to(device=X.device, dtype=torch.bool)[None, :, None])
    return X * mask.to(dtype=X.dtype)


def apply_augmentations(
    X: torch.Tensor,
    *,
    p_time_warp: float = 0.5,
    max_shift: int = 8,
    p_amp_jitter: float = 0.5,
    amp_scale: float = 0.1,
    p_noise: float = 0.5,
    noise_sigma: float = 0.1,
    p_ch_dropout: float = 0.2,
    p_ref_jitter: float = 0.5,
    seed: Optional[int] = None,
    channel_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """组合增强管线（训练期调用），按固定顺序施加各增强（D-order）。

    channel_mask : (C,) bool 可选；提供时所有增强对缺失通道保持 0，出口再做一次
                   强制归零（review v6 P0-2，防止 reference_jitter/gaussian_noise
                   把零填充通道变成幻象）。
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 一次生成 5 个设备端随机数并一次同步回主机（D-aug-coins），
    # 避免逐增强 `torch.rand(1).item()` 造成每 batch 5 次 GPU→CPU 同步。
    coins = torch.rand(5, device=X.device).tolist()
    if coins[0] < p_time_warp:
        X = time_warp(X, max_shift=max_shift)
    if coins[1] < p_amp_jitter:
        X = amplitude_jitter(X, scale=amp_scale)
    if coins[2] < p_noise:
        X = gaussian_noise(X, sigma=noise_sigma, channel_mask=channel_mask)
    if coins[3] < p_ch_dropout:
        X = channel_dropout(X, p=p_ch_dropout, channel_mask=channel_mask)
    if coins[4] < p_ref_jitter:
        X = reference_jitter(X, p=1.0, channel_mask=channel_mask)  # 此处已决定施加，内部全量施加

    # 缺失通道出口强制归零（双保险，review v6 P0-2）
    if channel_mask is not None:
        X = X * channel_mask.to(device=X.device, dtype=X.dtype)[None, :, None]
    return X
