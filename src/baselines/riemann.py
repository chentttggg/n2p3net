"""xDAWN、协方差、黎曼切空间与LDA组成的独立经典基线。

输入(N,C,T)，输出逐trial target判别分数，遵循classic.Baseline接口。
全部可学习变换仅在fit范围拟合，预测复用xDAWN、切空间原点与分类器。
几何计算使用float64；默认OAS收缩由本模块向量化实现。
虚拟空间增广训练类模板与试次；维数为2×min(nfilter,C)×类别数。
这保留ERP极性/时间交叉项，不能按原始通道数解释。
缺失通道应在调用前限制到共同实测布局，不把补零协方差当作完整观测。

科学比较见doc/research_program.zh.md；路线与基线配方见doc/roadmap.zh.md；
实现所有权见doc/technical_structure.zh.md，物理输入见doc/input_contract_math.zh.md。
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
        每类 xDAWN 增强滤波器数（默认 4）。模板增广协方差维数 = 2×min(nfilter,C)×n_classes。
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
        if isinstance(nfilter, bool) or not isinstance(nfilter, int) or nfilter < 1:
            raise ValueError("nfilter must be a positive integer")
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
        if not np.isfinite(X).all() or X.shape[1] < 1 or X.shape[2] < 2:
            raise ValueError("ERP covariance requires finite nonempty channel/time observations")
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> XdawnRiemann:
        self._fitted = False
        X = self._as_double(X)
        y = np.asarray(y)
        if y.shape != (len(X),) or set(np.unique(y)) != {0, 1} or len(y) <= 2:
            raise ValueError("ERP covariance requires aligned binary labels and more than two trials")
        self.input_shape_ = X.shape[1:]
        self.artifact_schema_ = "n2p3_xdawn_erp_covariance/2"
        observed = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
        observed = observed - observed.mean(axis=1, keepdims=True)
        self.observed_rank_ = int(np.linalg.matrix_rank(observed))
        if self.observed_rank_ == 0:
            raise ValueError("xDAWN requires nonzero observed spatial rank")
        self.effective_nfilter_ = min(self.nfilter, self.observed_rank_)

        # 1. xDAWN 空间滤波（增强 target SNR，D-xdawn-order）
        # OAS 的 baseline_cov 用模块内向量化实现（与 pyriemann/sklearn 公式等价）。
        baseline_cov = _oas_baseline_cov(X) if self.estimator == "oas" else None
        self.xd_ = Xdawn(
            nfilter=self.effective_nfilter_,
            estimator=self.estimator,
            classes=self.classes,
            baseline_cov=baseline_cov,
        )
        Xd = self.xd_.fit_transform(X, y)  # (N, nfilter*n_classes, T)
        # Class prototypes are fitted once in authorized training scope. Their
        # cross-covariances with each trial retain ERP polarity and timing.
        self.prototypes_ = self.xd_.evokeds_.copy()
        Xd = np.concatenate((np.broadcast_to(self.prototypes_, (len(Xd), *self.prototypes_.shape)), Xd), axis=1)

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
        if getattr(self, "artifact_schema_", None) != "n2p3_xdawn_erp_covariance/2":
            raise ValueError("legacy bare-covariance model cannot be loaded as ERP covariance")
        X = self._as_double(X)
        if X.shape[1:] != self.input_shape_:
            raise ValueError("ERP covariance channel/time geometry differs from fitted input")
        if not len(X):
            return np.empty(0, dtype=np.float64)
        Xd = self.xd_.transform(X)
        Xd = np.concatenate((np.broadcast_to(self.prototypes_, (len(Xd), *self.prototypes_.shape)), Xd), axis=1)
        if self._cov_is_oas:
            C = _oas_covariances_vectorized(Xd)
        else:
            C = self.cov_.transform(Xd)
        T = self.ts_.transform(C)
        return self.lda_.decision_function(T).astype(np.float64)
