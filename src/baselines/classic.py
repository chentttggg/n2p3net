"""经典特征基线：SWLDA近似、窗口逻辑回归与模板匹配。

统一接口为fit(X, y)及predict_logit(X)，输入(N,C,T)，输出逐trial target判别分数。
SWLDA采用降采样后单变量t检验筛选与LDA，不等价于经典多变量逐步回归。
窗口逻辑回归使用类平衡权重；模板匹配使用target均值模板与Pearson相关。
所有特征选择、标准化和模板仅在合法fit范围内拟合，预测复用保存状态。
空间几何基线见riemann.py，深度基线见deep.py，公共特征见features.py。

评估与拟合边界见doc/research_program.zh.md；执行配方见doc/roadmap.zh.md；
模块接口见doc/technical_structure.zh.md，输入单位和时间支持见doc/input_contract_math.zh.md。
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
            raise ValueError(
                f"SWLDA selected no feature below p_entry={self.p_entry}; each class needs at "
                "least two non-degenerate trials"
            )
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
