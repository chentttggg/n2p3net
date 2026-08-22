"""模块：xDAWN + 黎曼几何基线（Riemannian Baseline）。

职责（roadmap Phase 1 + constitution P8）：
    xDAWN 空间滤波 + 协方差估计 + 黎曼切线空间映射 + LDA，即 P300 的「xDAWN+RG」强基线
    （Kaggle BCI 竞赛冠军方案，techstack 明确列为必须纳入对照）。输出逐试次 target 判别
    logit（与 classic.py / heads 契约一致，供 decision 层对数似然比累积）。

明确「不做」：
    - 不做纯 MDM（最小距离均值）——它是另一条黎曼基线，可作 Phase 5 补充对照，非本模块职责。
    - 不做深基线（EEGNet/Inception/Conformer，deep.py）。

三思决策记录（供后续会话追溯）：
    D-xdawn-double  pyriemann 的估计器（协方差/tangent space）数值上要求 float64（scipy 的
                     eig 分解在 float32 下会放大舍入误差、可能产生非 SPD 的 log 映射 NaN）。
                    data 层输出 float32，故 fit/predict 入口统一 np.asarray(..., float64)。
    D-xdawn-virtual xDAWN 的输出是「nfilter × n_classes 个虚拟通道」（每类 nfilter 条增强
                     时间序列），**不是原始 C 通道**。协方差在这批虚拟通道上估计：8 导二分类、
                     nfilter=4 → 8 虚拟通道 → tangent 维度 8×9/2 = 36，远小于数千试次，对
                     小样本友好（不会像「C×C 协方差 + flatten」那样维度爆炸）。
    D-xdawn-order   标准配方顺序固定：xDAWN（先增强 target SNR）→ 协方差 → 切线空间 → LDA。
                     先做 xDAWN 再协方差，比直接对原始信号协方差判别力更强（target 增强后的
                     虚拟通道协方差携带更多判别结构）。
    D-rg-oas        协方差默认用 'oas'（收缩估计）。实测（review P0）：target 试次少的小样本 fold
                     （尤其 3 导子集）下 scm 会因样本不足产生非正定协方差而崩（LinAlgError/非 SPD），
                     oas 收缩则稳定。scm 仅在「大样本 + 通道数少」时可作备选。
      D-rg-oas-vec    review v6 性能复核：pyriemann 的 Covariances(estimator='oas') 对每个试次循环
                       调 sklearn OAS，242 fold LOSO 下是 xDAWN 基线的主要耗时（实测 1.4s/fold vs
                       向量化 0.034s/fold，42x）。默认路径改为模块内向量化 OAS，公式与 sklearn
                       OAS 完全一致（实测 max abs diff=0）；其余 estimator 仍走 pyriemann。
    D-xdawn-nfilter nfilter 默认 4（pyriemann 默认）。8 导下 tangent 36 维，与「few features
                     enough」一致；若 Phase 1 实测过拟合（训练/测试命中率落差大），可降到 2~3。

契约（输入 → 输出）：
    X ∈ R^{N×C×T}（float32，缺失通道须已填 0 或为有效值）+ y ∈ {0,1}^N → fit 后
    predict_logit(X) ∈ R^N（target 判别分数）。

依赖的决策：roadmap Phase 1、constitution P8（先基线后创新）、baselines/classic.Baseline
    （统一接口）。
"""

from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances, Xdawn
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from baselines.classic import Baseline


def _oas_covariances_vectorized(X: np.ndarray) -> np.ndarray:
    """向量化 OAS 协方差估计（review v6 P1 性能优化，D-rg-oas-vec）。

    公式与 sklearn.covariance._oas 完全一致（对每个试次沿时间轴中心化后收缩），
    但一次性计算所有试次，避免 pyriemann 对每个试次循环调 sklearn OAS。
    X: (N, C, T) float64 → (N, C, C) OAS 协方差。
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"X 须为 (N,C,T)，得到 {X.shape}。")
    n_trials, n_ch, n_times = X.shape
    Xc = X - X.mean(axis=2, keepdims=True)
    # SCM：S_n = Xc_n @ Xc_nᵀ / T。einsum(optimize=True) 对 (N,C,T) 批量小矩阵
    # 使用 BLAS 批量 GEMM，比显式 transpose + matmul 少一次大张量副本。
    S = np.einsum("nct,nut->ncu", Xc, Xc, optimize=True) / n_times
    # sklearn OAS 收缩系数（逐试次向量化）
    alpha = (S * S).mean(axis=(1, 2))
    mu = np.einsum("nii->n", S) / n_ch
    mu2 = mu * mu
    num = alpha + mu2
    den = (n_times + 1) * (alpha - mu2 / n_ch)
    den_safe = np.where(den == 0.0, 1.0, den)
    rho = np.minimum(num / den_safe, 1.0)
    rho[den == 0.0] = 1.0
    # C = (1−ρ)S + ρ·(trace(S)/C)·I
    C = (1.0 - rho)[:, None, None] * S
    idx = np.arange(n_ch)
    C[:, idx, idx] += rho[:, None] * mu[:, None]
    return C


def _oas_baseline_cov(X: np.ndarray) -> np.ndarray:
    """xDAWN baseline OAS 协方差（等价于 pyriemann 对 (C, N*T) 展平后调用 OAS）。

    不实际创建 N*T 展平副本，直接沿 (试次, 时间) 两个轴中心化并求和外积，
    避免 pyriemann/sklearn 对大矩阵的额外开销（review v6 性能项）。
    X: (N, C, T) float64 → (C, C)。
    """
    X = np.asarray(X, dtype=np.float64)
    n_trials, n_ch, n_times = X.shape
    mu = X.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    Xc = X - mu
    S = np.einsum("nct,ndt->cd", Xc, Xc) / (n_trials * n_times)
    alpha = (S * S).mean()
    mu_s = np.trace(S) / n_ch
    mu2 = mu_s * mu_s
    num = alpha + mu2
    den = (n_trials * n_times + 1) * (alpha - mu2 / n_ch)
    rho = 1.0 if den == 0.0 else min(num / den, 1.0)
    C = (1.0 - rho) * S
    C.flat[:: n_ch + 1] += rho * mu_s
    return C


class XdawnRiemann(Baseline):
    """xDAWN 空间滤波 + 黎曼切线空间 + LDA（P300 强基线）。

    Parameters
    ----------
    nfilter : int
        每类 xDAWN 增强滤波器数（默认 4）。虚拟通道数 = nfilter × n_classes。
    estimator : str
        协方差估计器（默认 'oas' 收缩；小样本更稳，见 D-rg-oas）。
    metric : str
        黎曼度量（默认 'riemann'，即仿射不变度量）。
    classes : Sequence | None
        xDAWN 的类别顺序（默认 None，从 y 自动推断）。
    """

    def __init__(
        self,
        nfilter: int = 4,
        estimator: str = "oas",
        metric: str = "riemann",
        classes=None,
    ):
        self.nfilter = nfilter
        self.estimator = estimator
        self.metric = metric
        self.classes = classes
        self._fitted = False

    @staticmethod
    def _as_double(X: np.ndarray) -> np.ndarray:
        """统一转 float64（D-xdawn-double）。"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"X 须为 (N,C,T)，得到 {X.shape}。")
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XdawnRiemann":
        X = self._as_double(X)
        y = np.asarray(y).astype(int)

        # 1. xDAWN 空间滤波（增强 target SNR，D-xdawn-order）
        # OAS 的 baseline_cov 用模块内向量化实现（与 pyriemann/sklearn 公式等价）。
        baseline_cov = _oas_baseline_cov(X) if self.estimator == "oas" else None
        self.xd_ = Xdawn(
            nfilter=self.nfilter,
            estimator=self.estimator,
            classes=self.classes,
            baseline_cov=baseline_cov,
        )
        Xd = self.xd_.fit_transform(X, y)  # (N, nfilter*n_classes, T)

        # 2. 协方差估计（在虚拟通道上，D-xdawn-virtual）
        # 默认 OAS 走模块内向量化实现（42x 加速且逐元素等价，D-rg-oas-vec）。
        self._cov_is_oas = self.estimator == "oas"
        if self._cov_is_oas:
            self.cov_ = None
            C = _oas_covariances_vectorized(Xd)
        else:
            self.cov_ = Covariances(estimator=self.estimator)
            C = self.cov_.fit_transform(Xd)  # (N, V, V)

        # 3. 黎曼切线空间映射（SPD → 欧氏，D-xdawn-order）
        self.ts_ = TangentSpace(metric=self.metric)
        T = self.ts_.fit_transform(C)  # (N, V*(V+1)/2)

        # 4. LDA 判别
        self.lda_ = LinearDiscriminantAnalysis()
        self.lda_.fit(T, y)
        self._fitted = True
        return self

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("请先 fit 再 predict_logit。")
        X = self._as_double(X)
        Xd = self.xd_.transform(X)
        if self._cov_is_oas:
            C = _oas_covariances_vectorized(Xd)
        else:
            C = self.cov_.transform(Xd)
        T = self.ts_.transform(C)
        return self.lda_.decision_function(T).astype(np.float64)
