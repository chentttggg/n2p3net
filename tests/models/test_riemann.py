"""baselines.riemann 模块测试：xDAWN + 黎曼切线空间 + LDA。

合成数据：target 试次在 Pz（索引 3）300–500ms 叠正高斯波，non-target 纯噪声。
语义：XdawnRiemann 能学到判别信息（AUC > 0.7），输出 float64 logit 且有限。
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from baselines.riemann import XdawnRiemann
from tests.support.datasets import make_p300_data

C = 8
T = 256
SFR = 256.0
TMIN = -0.2  # 秒；与 data/preprocess.py 一致


def test_xdawn_riemann_learns_and_returns_finite_double_scores_and_probabilities():
    X, y = make_p300_data(n_times=T, sfreq=SFR, tmin_s=TMIN)
    Xtr, Xte, ytr, yte = X[:400], X[400:], y[:400], y[400:]
    clf = XdawnRiemann().fit(Xtr, ytr)
    logits = clf.predict_logit(Xte)
    assert logits.dtype == np.float64
    assert np.isfinite(logits).all()
    assert roc_auc_score(yte, logits) > 0.7
    probabilities = clf.predict_proba(Xte)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_xdawn_riemann_unfitted_raises():
    clf = XdawnRiemann()
    X = np.random.randn(10, C, T).astype(np.float32)
    with pytest.raises(RuntimeError):
        clf.predict_logit(X)


def test_xdawn_3chan_oas_default():
    """GTN 3 导子集场景：默认 oas 收缩稳定（review P0——scm 会因 target 样本不足崩）。"""
    X, y = make_p300_data(n_target=15, n_nontarget=60, n_times=T, sfreq=SFR, tmin_s=TMIN)
    Xsub = X[:, :3, :]  # Fz/Cz/Pz 子集（GTN 原生 3 导）
    clf = XdawnRiemann(nfilter=3)  # 默认 estimator="oas"
    clf.fit(Xsub, y)
    logits = clf.predict_logit(Xsub)
    assert np.isfinite(logits).all(), "3 导子集 + oas 应产出有限 logit"
