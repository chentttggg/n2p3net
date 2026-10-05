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


def test_erp_geometry_preserves_polarity_and_matches_official_pipeline():
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline

    rng = np.random.default_rng(190)
    y = np.tile([0, 1], 40)
    template = np.sin(np.linspace(0, 2 * np.pi, 32))[None, :]
    X = .15 * rng.normal(size=(80, 2, 32)) + (2 * y[:, None, None] - 1) * template
    model = XdawnRiemann(nfilter=2).fit(X, y)
    # Bare cov(X) is invariant to polarity; ERP cross-covariance is not.
    pair = np.stack([X[1], -X[1]])
    assert model.predict_logit(pair)[0] > model.predict_logit(pair)[1] + 1
    official = make_pipeline(
        XdawnCovariances(nfilter=2, estimator="oas", xdawn_estimator="oas"),
        TangentSpace(metric="riemann"), LinearDiscriminantAnalysis(),
    ).fit(X, y)
    np.testing.assert_allclose(model.predict_logit(X), official.decision_function(X), atol=1e-6, rtol=1e-6)
    assert model.predict_logit(X[[1]])[0] > model.predict_logit(X[[1], :, ::-1])[0] + 1


def test_legacy_geometry_is_not_silently_relabelled():
    X, y = make_p300_data(n_target=15, n_nontarget=60, n_times=64, sfreq=64.)
    model = XdawnRiemann(nfilter=2).fit(X, y)
    del model.artifact_schema_
    with pytest.raises(ValueError, match="legacy"):
        model.predict_logit(X)


def test_failed_refit_invalidates_previous_model():
    X = np.random.default_rng(12).normal(size=(12, 3, 24))
    model = XdawnRiemann(nfilter=2).fit(X, np.arange(12) % 2)
    with pytest.raises(ValueError):
        model.fit(X, np.zeros(12))
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_logit(X)


def test_xdawn_clips_filters_to_actual_spatial_rank():
    X = np.repeat(np.random.default_rng(13).normal(size=(12, 1, 24)), 3, axis=1)
    model = XdawnRiemann(nfilter=3).fit(X, np.arange(12) % 2)
    assert model.xd_.filters_.shape == (2, 3)
    assert model.observed_rank_ == model.effective_nfilter_ == 1
