"""经典基线共享特征：降采样、时间窗均值与target均值模板。

输入(N,C,T)及显式sfreq、tmin；tmin用秒，window_ms用毫秒。
时间索引按round转换；FFT降采样是整epoch算子，截取输出不缩短原始信号支持。
模板只能用fit范围的target试次构建，预测阶段不得重新拟合。
缺导由subset_channels选择实测通道，不将零占位当成真实特征。
本模块不拟合分类器或xDAWN空间滤波。

物理输入见doc/input_contract_math.zh.md；拟合与评估边界见doc/research_program.zh.md；
基线职责和执行配方见doc/technical_structure.zh.md及doc/roadmap.zh.md。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import resample


def time_to_index(ms: float, sfreq: float, tmin: float) -> int:
    """物理时间 ms → 采样点索引（四舍五入，D-time-index）。

    tmin 单位为秒（与 data/preprocess.py 一致），例如 -200 ms → tmin=-0.2。
    """
    return int(round((ms / 1000.0 - tmin) * sfreq))


def downsampled_time_points(n_times: int, sfreq: float, target_hz: float = 20.0) -> int:
    """Output length of the frozen FFT downsampling map, including no upsampling."""
    return min(n_times, int(round(n_times * target_hz / sfreq)))


def downsample_epochs(X: np.ndarray, sfreq: float, target_hz: float = 20.0) -> np.ndarray:
    """downsample 单试次 (N,C,T) → (N,C,T_down)，目标 target_hz（D-downsample）。"""
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"X 须为 (N,C,T)，得到 {X.shape}。")
    T = X.shape[2]
    T_down = downsampled_time_points(T, sfreq, target_hz)
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
