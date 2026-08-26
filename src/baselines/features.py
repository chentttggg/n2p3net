"""模块：基线特征提取（Baseline Features）。

职责（roadmap Phase 1）：
    为经典/地板基线提供统一的特征提取：downsample、手工时间窗特征、grand-average 模板。
    这是 SWLDA / 手工窗逻辑回归 / 模板匹配三个基线共享的地基（纯 numpy/scipy，无 torch）。

明确「不做」：
    - 不涉及任何分类器（分类在 classic.py / riemann.py / deep.py）。
    - 不做 xDAWN 空间滤波（那是 riemann.py 的职责，pyriemann 实现）。

三思决策记录（供后续会话追溯）：
    D-time-index    时间 ms → 采样点索引：idx = (ms/1000 − tmin)·sfreq（四舍五入）。tmin 统一为秒
                     （与 data/preprocess.py 一致，默认 −0.2 s = −200 ms）。
    D-downsample    downsample 用 scipy.signal.resample（FFT 重采样，沿时间轴）：支持非整数
                     decimation factor（256→20Hz 的 12.8），抗混叠优于朴素平均池化。SWLDA 经典实现
                     即 downsample 到 ~20Hz 降维。
    D-window-mean   手工窗地板用「窗内均值」作特征（比峰值稳、抗噪声），对应 P8 的免费地板定位。
    D-template      grand-average 模板 = target 试次的逐通道均值（epoch 对齐后直接平均），
                     是模板匹配地板（P8）的核心；用 Pearson 相关度量（见 classic.TemplateMatching）。

契约（输入 → 输出）：
    均以单试次 X ∈ R^{N×C×T}（float32，采样率 sfreq、起点 tmin）为输入，输出 numpy 特征/模板。

依赖的决策：roadmap Phase 1（基线复现）、constitution P8（免费地板）。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import resample

# 标准 8 导顺序（与 data.channel.STANDARD_CHANNELS 一致）：Fz,Cz,P3,Pz,P4,PO7,PO8,Oz。
STANDARD_CHANNELS: tuple[str, ...] = ("Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz")


def time_to_index(ms: float, sfreq: float, tmin: float) -> int:
    """物理时间 ms → 采样点索引（四舍五入，D-time-index）。

    tmin 单位为秒（与 data/preprocess.py 一致），例如 -200 ms → tmin=-0.2。
    """
    return int(round((ms / 1000.0 - tmin) * sfreq))


def downsample_epochs(X: np.ndarray, sfreq: float, target_hz: float = 20.0) -> np.ndarray:
    """downsample 单试次 (N,C,T) → (N,C,T_down)，目标 target_hz（D-downsample）。"""
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"X 须为 (N,C,T)，得到 {X.shape}。")
    T = X.shape[2]
    T_down = int(round(T * target_hz / sfreq))
    if T_down >= T:
        return X.astype(np.float32, copy=False)
    return resample(X, T_down, axis=2).astype(np.float32)


def extract_window(
    X: np.ndarray,
    sfreq: float,
    tmin: float,
    window_ms: tuple[float, float],
    channels: Sequence[int] | None = None,
) -> np.ndarray:
    """提取时间窗 [window_ms[0], window_ms[1]] 的信号 → (N, C_sel, T_window)。"""
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"X 须为 (N,C,T)，得到 {X.shape}。")
    i0 = time_to_index(window_ms[0], sfreq, tmin)
    i1 = time_to_index(window_ms[1], sfreq, tmin)
    i0 = max(0, min(i0, X.shape[2]))
    i1 = max(0, min(i1, X.shape[2]))
    X = X[:, :, i0:i1]
    if channels is not None:
        X = X[:, list(channels), :]
    return X


def window_mean_feature(
    X: np.ndarray,
    sfreq: float,
    tmin: float,
    window_ms: tuple[float, float],
    channels: Sequence[int] | None = None,
) -> np.ndarray:
    """窗内均值特征 → (N, C_sel)（D-window-mean）。"""
    W = extract_window(X, sfreq, tmin, window_ms, channels=channels)
    return W.mean(axis=2).astype(np.float32)


def grand_average_template(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """target 试次的 grand-average 模板 → (C, T)（D-template）。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    mask = y.astype(bool)
    if not mask.any():
        raise ValueError("y 中无 target 试次，无法计算 grand-average 模板。")
    return X[mask].mean(axis=0).astype(np.float32)


def subset_channels(X: np.ndarray, channel_mask) -> np.ndarray:
    """按 channel_mask 提取存在通道子集 → X[:, mask, :]（D-channel-strategy）。

    缺失通道策略：classic（SWLDA/WindowLR/TemplateMatching）与 riemann
    （XdawnRiemann）应在「存在通道子集」上训练/预测，而非零填充；零填充会让协方差奇异
    或把 0 通道当无信息噪声特征。deep 模型也必须按数据集原生通道数构造；此函数是
    experiment 层适配 classic/riemann 的统一入口。
    """
    X = np.asarray(X)
    mask = np.asarray(channel_mask, dtype=bool)
    if X.ndim != 3:
        raise ValueError(f"X 须为 (N,C,T)，得到 {X.shape}。")
    if mask.shape[0] != X.shape[1]:
        raise ValueError(f"channel_mask 长度 {mask.shape[0]} ≠ 通道数 {X.shape[1]}。")
    if not mask.any():
        raise ValueError("channel_mask 全 False：无任何存在通道。")
    return X[:, mask, :]
