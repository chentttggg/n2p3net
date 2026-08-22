"""baselines.riemann 模块测试：xDAWN + 黎曼切线空间 + LDA。

合成数据：target 试次在 Pz（索引 3）300–500ms 叠正高斯波，non-target 纯噪声。
语义：XdawnRiemann 能学到判别信息（AUC > 0.7），输出 float64 logit 且有限。
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from baselines.features import time_to_index
from baselines.riemann import XdawnRiemann

C = 8
T = 256
SFR = 256.0
TMIN = -0.2  # 秒；与 data/preprocess.py 一致


def make_p300(n_target=120, n_nontarget=480, seed=0):
    """target 在 Pz（索引 3）300–500ms 叠正高斯波；non-target 纯噪声。"""
    rng = np.random.default_rng(seed)
    n = n_target + n_nontarget
    X = rng.standard_normal((n, C, T)).astype(np.float32)
    y = np.zeros(n, dtype=int)
    y[:n_target] = 1
    i0 = time_to_index(300, SFR, TMIN)
    i1 = time_to_index(500, SFR, TMIN)
    t = np.arange(T)
    center = (i0 + i1) / 2
    width = (i1 - i0) / 6
    gauss = np.exp(-0.5 * ((t - center) / width) ** 2)
    X[:n_target, 3, :] += 5.0 * gauss
    idx = rng.permutation(n)
    return X[idx].astype(np.float32), y[idx]


def test_xdawn_riemann_learns():
    X, y = make_p300()
    Xtr, Xte, ytr, yte = X[:400], X[400:], y[:400], y[400:]
    clf = XdawnRiemann().fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_logit(Xte))
    assert auc > 0.7, f"xDAWN+RG AUC 应 >0.7，得到 {auc:.3f}"


def test_xdawn_riemann_output_double_finite():
    """输入 float32 → 输出 float64 且有限（D-xdawn-double）。"""
    X, y = make_p300(n_target=50, n_nontarget=200)
    clf = XdawnRiemann().fit(X, y)
    logits = clf.predict_logit(X)
    assert logits.dtype == np.float64
    assert np.isfinite(logits).all(), "logit 不应含 NaN/Inf"


def test_xdawn_riemann_predict_proba_range():
    X, y = make_p300(n_target=50, n_nontarget=200)
    clf = XdawnRiemann().fit(X, y)
    p = clf.predict_proba(X)
    assert (p >= 0).all() and (p <= 1).all()


def test_xdawn_riemann_unfitted_raises():
    clf = XdawnRiemann()
    X = np.random.randn(10, C, T).astype(np.float32)
    with pytest.raises(RuntimeError):
        clf.predict_logit(X)


def test_xdawn_3chan_oas_default():
    """GTN 3 导子集场景：默认 oas 收缩稳定（review P0——scm 会因 target 样本不足崩）。"""
    X, y = make_p300(n_target=15, n_nontarget=60)
    Xsub = X[:, :3, :]  # Fz/Cz/Pz 子集（GTN 原生 3 导）
    clf = XdawnRiemann(nfilter=3)  # 默认 estimator="oas"
    clf.fit(Xsub, y)
    logits = clf.predict_logit(Xsub)
    assert np.isfinite(logits).all(), "3 导子集 + oas 应产出有限 logit"
