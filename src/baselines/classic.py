"""模块：经典 / 地板基线（Classic Baselines）。

职责（roadmap Phase 1 + constitution P8）：
    三个基于手工特征的基线，均输出逐试次 target 的「判别 logit」（供 decision 层对数似然比累积）：
        - SWLDA：downsample 特征 + 逐步特征选择 + LDA（P300 speller 经典方法）
        - WindowLogisticRegression：手工窗均值特征 + 逻辑回归（免费地板）
        - TemplateMatching：grand-average target 模板 + Pearson 相关（免费地板）

明确「不做」：
    - xDAWN + Riemannian（riemann.py，pyriemann 实现）
    - EEGNet / EEG-Inception / EEG Conformer（deep.py，braindecode 实现）
    - 三层评估协议 / 配对置换检验（evaluate.py，Phase 1 后续）

三思决策记录（供后续会话追溯）：
    D-unified-iface  统一接口 fit(X,y) / predict_logit(X)→(N,)。predict_logit 返回「判别分数」作 logit，
                     因 decision 层做 z-score 再 argmax，logit 绝对尺度不重要（z-score 单调）。与 heads
                     输出 logit 的契约一致（decision.py 直接消费）。
    D-swdla-topk     SWLDA 的特征选择用「单变量 t 检验 p 值」的 top-k 筛选（cap max_features），非经典
                     多变量 F 统计量逐步回归。理由：P300 判别信息集中在少数时间点，单变量近似已能选到；
                     且单变量实现简单、可测试、不易错。后向删除在单变量 p 独立假设下恒不触发（因
                     p_entry < p_removal），已移除死代码（review P2）。若 Phase 1 命中率低于 Vařeka
                     77.2% 再升级多变量判据。
    D-swdla-down     SWLDA 输入 downsample 到 ~20Hz 后 flatten（经典 Krusienski 2006 做法），标准化后
                     逐步选择；预测时复用训练期的 mean/std。
    D-lr-balanced    手工窗逻辑回归用 class_weight="balanced" 处理 1/9 不平衡（sklearn 无 pos_weight，
                     等价方向是加重 target）；对应 blueprint pos_weight≈8 的意图。
    D-template-cor   模板匹配用 flatten 后 Pearson 相关（中心化点积），即「试次波形与 target 模板的
                     形状相似度」；target 相关高、non-target 低。可限制 window_ms 到 P300 窗增强判别。

契约（输入 → 输出）：
    X ∈ R^{N×C×T}（float32）+ y ∈ {0,1}^N → fit 后 predict_logit(X) ∈ R^N（target 判别分数）。

依赖的决策：roadmap Phase 1、constitution P8（免费地板）、baselines/features.py。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from baselines.features import (
    downsample_epochs,
    grand_average_template,
    window_mean_feature,
)
from baselines.progress import EpochProgressCallback, make_epoch_progress_callback
from data.contract import DEFAULT_P300_DATA_CONTRACT


class Baseline:
    """经典基线统一接口（D-unified-iface）。"""

    # Optional arguments are capability declarations consumed by evaluate.py.
    # Defaults keep legacy adapters compatible with the common runner.
    fit_accepts_subject_ids = False
    fit_accepts_trial_context = False
    fit_accepts_group_ids = False
    fit_accepts_acquisition_indices = False
    fit_accepts_trial_channel_mask = False
    predict_accepts_acquisition_indices = False
    predict_accepts_trial_channel_mask = False
    auxiliary_predict_accepts_trial_channel_mask = False
    # A mask-aware model must opt in before evaluate.py may retain non-zero
    # values behind a dynamically rejected QC channel mask.
    accepts_unmaterialized_trial_channel_mask = False

    _epoch_progress_dir: Path | None = None
    _evaluation_fold_id: int | None = None

    def configure_epoch_progress(self, directory: str | Path | None) -> None:
        """Configure the shared epoch sink owned by an experiment runner."""

        self._epoch_progress_dir = Path(directory) if directory is not None else None

    def configure_evaluation_fold(self, fold_id: int | None) -> None:
        """Set the display fold id used by the shared progress sink."""

        self._evaluation_fold_id = None if fold_id is None else int(fold_id)

    def epoch_progress_callback(self) -> EpochProgressCallback | None:
        """Return this model's configured epoch callback, if any."""

        return make_epoch_progress_callback(self._epoch_progress_dir, self._evaluation_fold_id)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Baseline:
        raise NotImplementedError

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        """逐试次 target 判别分数 (N,)（作 logit 供 decision 层累积）。"""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """sigmoid(logit) → target 概率 (N,)。"""
        z = self.predict_logit(X)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


class SWLDA(Baseline):
    """Stepwise LDA（单变量 top-k 近似，D-swdla-topk / D-swdla-down）。

    Parameters
    ----------
    sfreq / tmin : float
        采样率（Hz）与起点物理时间（秒，与 data/preprocess.py 一致；-0.2 s = -200 ms）。
    target_hz : float
        downsample 目标频率（默认 20Hz，经典做法）。
    p_entry : float
        特征入选的单变量 t 检验 p 阈值。
    max_features : int
        最多选入的特征数。
    """

    def __init__(
        self,
        sfreq: float = DEFAULT_P300_DATA_CONTRACT.sample_rate_hz,
        tmin: float = DEFAULT_P300_DATA_CONTRACT.tmin_ms / 1000.0,
        target_hz: float = 20.0,
        p_entry: float = 0.1,
        max_features: int = 60,
    ):
        self.sfreq = sfreq
        self.tmin = tmin
        self.target_hz = target_hz
        self.p_entry = p_entry
        self.max_features = max_features
        self.lda = LinearDiscriminantAnalysis()

    def _features(self, X: np.ndarray) -> np.ndarray:
        Xd = downsample_epochs(X, self.sfreq, self.target_hz)
        return Xd.reshape(Xd.shape[0], -1)

    @staticmethod
    def _univariate_p_all(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """全部特征的单变量 t 检验 p 值（向量化，review v4 P2 性能项）。"""
        from scipy.stats import ttest_ind

        x0 = X[y == 0]
        x1 = X[y == 1]
        if x0.shape[0] < 2 or x1.shape[0] < 2:
            return np.ones(X.shape[1])
        return np.asarray(ttest_ind(x1, x0, axis=0, equal_var=False).pvalue)

    def _select_features(self, X: np.ndarray, y: np.ndarray) -> list[int]:
        """单变量 top-k 筛选（D-swdla-topk）：按 t 检验 p 值升序选特征，cap max_features。

        注：非严格 stepwise——单变量 p 值不随已选集合变化，p_entry < p_removal 时后向删除
        恒不触发，故移除（review P2 死代码）。因此一次算全 p 值再排序 top-k 与原实现等价。
        """
        pvals = self._univariate_p_all(X, y)
        order = np.argsort(pvals, kind="stable")
        selected = [int(i) for i in order if pvals[i] < self.p_entry][: self.max_features]
        return selected

    def fit(self, X: np.ndarray, y: np.ndarray) -> SWLDA:
        F = self._features(X)
        self._mean = F.mean(axis=0)
        self._std = F.std(axis=0) + 1e-8
        Fs = (F - self._mean) / self._std
        self.selected_ = self._select_features(Fs, y)
        if not self.selected_:
            self.selected_ = [0]
        self.lda.fit(Fs[:, self.selected_], y)
        return self

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "selected_"):
            raise RuntimeError("请先 fit 再 predict_logit。")
        F = self._features(X)
        Fs = (F - self._mean) / self._std
        return self.lda.decision_function(Fs[:, self.selected_]).astype(np.float64)


class WindowLogisticRegression(Baseline):
    """手工窗均值 + 逻辑回归（免费地板，D-lr-balanced / D-lr-scale）。

    Parameters
    ----------
    sfreq / tmin : float
        采样率（Hz）与起点物理时间（秒，与 data/preprocess.py 一致；-0.2 s = -200 ms）。
    window_ms : tuple[float, float]
        判别时间窗（默认 250–500 ms，P3b 窗）。
    channels : Sequence[int] | None
        限制通道子集（None = 全部）。
    """

    def __init__(
        self,
        sfreq: float = DEFAULT_P300_DATA_CONTRACT.sample_rate_hz,
        tmin: float = DEFAULT_P300_DATA_CONTRACT.tmin_ms / 1000.0,
        window_ms: tuple[float, float] = (250.0, 500.0),
        channels: Sequence[int] | None = None,
    ):
        self.sfreq = sfreq
        self.tmin = tmin
        self.window_ms = window_ms
        self.channels = channels
        # D-lr-scale：MNE 输出 V 单位（P300 幅值 ~1e-5 V），若直接喂 LogisticRegression，
        # 其 l2 正则（C=1.0）会对 ~1e-5 量级特征过度惩罚权重、使分类退化为随机（2026-08-21
        # 实测 GTN 上 bacc 0.50→0.564）。故加 StandardScaler（与 SWLDA 的手动 z-score 同义）。
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(max_iter=2000, class_weight="balanced")

    def _features(self, X: np.ndarray) -> np.ndarray:
        return window_mean_feature(X, self.sfreq, self.tmin, self.window_ms, channels=self.channels)

    def fit(self, X: np.ndarray, y: np.ndarray) -> WindowLogisticRegression:
        F = self._features(X)
        Fs = self.scaler.fit_transform(F)  # D-lr-scale
        self.lr.fit(Fs, y)
        return self

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.scaler, "mean_"):
            raise RuntimeError("请先 fit 再 predict_logit。")
        Fs = self.scaler.transform(self._features(X))  # D-lr-scale
        return self.lr.decision_function(Fs).astype(np.float64)


class TemplateMatching(Baseline):
    """grand-average target 模板 + Pearson 相关（免费地板，D-template-cor）。"""

    def __init__(
        self,
        sfreq: float = DEFAULT_P300_DATA_CONTRACT.sample_rate_hz,
        tmin: float = DEFAULT_P300_DATA_CONTRACT.tmin_ms / 1000.0,
        window_ms: tuple[float, float] | None = None,
        channels: Sequence[int] | None = None,
    ):
        self.sfreq = sfreq
        self.tmin = tmin
        self.window_ms = window_ms
        self.channels = channels

    def _view(self, X: np.ndarray) -> np.ndarray:
        """应用到 window/channels 的视图（与模板一致）。"""
        if self.window_ms is not None:
            from baselines.features import extract_window

            return extract_window(X, self.sfreq, self.tmin, self.window_ms, self.channels)
        if self.channels is not None:
            return X[:, list(self.channels), :]
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> TemplateMatching:
        self.template_ = grand_average_template(self._view(X), y).ravel()
        # NaN 防御（D-nan-guard）：输入含 NaN 时模板/相关会静默产出 NaN logit，
        # 进而被 decision 层当空集剔除、命中率分母悄悄变小（review P0）。故显式报错。
        if not np.isfinite(self.template_).all():
            raise ValueError(
                "模板含 NaN：输入存在缺失通道。请先用 subset_channels 提取子集或填 0。"
            )
        self.template_ = self.template_ - self.template_.mean()
        return self

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        """Pearson 相关向量化为矩阵-向量运算（D-template-cor-vec）。"""
        if not hasattr(self, "template_"):
            raise RuntimeError("请先 fit 再 predict_logit。")
        tpl = self.template_
        tpl_norm = np.linalg.norm(tpl) + 1e-8
        Xv = self._view(X).reshape(X.shape[0], -1)
        Xc = Xv - Xv.mean(axis=1, keepdims=True)
        # 中心化点积 / (行范数 × 模板范数)：等于逐试次 flatten Pearson 相关
        return (Xc @ tpl) / (np.linalg.norm(Xc, axis=1) * tpl_norm + 1e-8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """把 Pearson 相关 r∈[−1,1] 映射到 [0,1] 的相似度分数。

        注意：这不是校准概率；只保证单调且范围合法。TemplateMatching 主用途是
        给 decision 层提供 rank logit，不消费该概率（audit P2 修正语义）。
        """
        r = np.clip(self.predict_logit(X), -1.0, 1.0)
        return (r + 1.0) / 2.0
