"""baselines 模块测试：features + classic 基线。

合成数据：target 试次在 Pz 通道 300–500ms 叠加正高斯波，non-target 为纯噪声。
冒烟：形状/概率范围/模板正确。
语义：三个基线（SWLDA/窗逻辑回归/模板匹配）能学到判别信息（AUC 显著 > 0.5）。
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from baselines.classic import SWLDA, TemplateMatching, WindowLogisticRegression
from baselines.features import (
    downsample_epochs,
    extract_window,
    grand_average_template,
    subset_channels,
    time_to_index,
    window_mean_feature,
)

C = 8
T = 128
SFR = 128.0
TMIN = -0.2  # 秒；与 data/preprocess.py 的单位一致


def make_p300_data(n_target=120, n_nontarget=480, seed=0):
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
    X[:n_target, 3, :] += 5.0 * gauss  # Pz 加正波
    # shuffle（保证 train/test 都含两类，否则 AUC 单类未定义）
    idx = rng.permutation(n)
    return X[idx].astype(np.float32), y[idx]


# ---------------- features 冒烟 ----------------


def test_time_to_index():
    assert time_to_index(-200, SFR, TMIN) == 0
    assert time_to_index(300, SFR, TMIN) == 64  # (300+200)/1000*128


def test_downsample_shape():
    X = np.random.randn(10, C, T).astype(np.float32)
    Xd = downsample_epochs(X, SFR, target_hz=20)
    assert Xd.shape == (10, C, 20)  # 250*20/250=20


def test_extract_window_shape():
    X = np.random.randn(10, C, T).astype(np.float32)
    W = extract_window(X, SFR, TMIN, (250, 500))
    assert W.shape == (10, C, 32)  # idx 58..90


def test_window_mean_feature_shape():
    X = np.random.randn(10, C, T).astype(np.float32)
    F = window_mean_feature(X, SFR, TMIN, (250, 500))
    assert F.shape == (10, C)


def test_grand_average_template():
    X, y = make_p300_data(n_target=50, n_nontarget=200)
    tpl = grand_average_template(X, y)
    assert tpl.shape == (C, T)
    # 模板应在 Pz（索引 3）的 300-500ms 有正波
    assert tpl[3, 60:100].max() > 1.0, "target 模板应在 Pz 有正波"


# ---------------- 基线语义 ----------------


def _split(X, y, n_train):
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def test_swlda_learns():
    X, y = make_p300_data()
    Xtr, Xte, ytr, yte = _split(X, y, 400)
    clf = SWLDA().fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_logit(Xte))
    assert auc > 0.7, f"SWLDA AUC 应 >0.7，得到 {auc:.3f}"


def test_window_lr_learns():
    X, y = make_p300_data()
    Xtr, Xte, ytr, yte = _split(X, y, 400)
    clf = WindowLogisticRegression().fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_logit(Xte))
    assert auc > 0.7, f"窗逻辑回归 AUC 应 >0.7，得到 {auc:.3f}"


def test_template_matching_learns():
    X, y = make_p300_data()
    Xtr, Xte, ytr, yte = _split(X, y, 400)
    clf = TemplateMatching().fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_logit(Xte))
    assert auc > 0.7, f"模板匹配 AUC 应 >0.7，得到 {auc:.3f}"


def test_predict_proba_range():
    X, y = make_p300_data(n_target=50, n_nontarget=200)
    for clf in (SWLDA(), WindowLogisticRegression(), TemplateMatching()):
        clf.fit(X, y)
        p = clf.predict_proba(X)
        assert (p >= 0).all() and (p <= 1).all(), type(clf).__name__


def test_subset_channels():
    """通道子集（D-channel-strategy）：classic/riemann 用子集而非零填充。"""
    X = np.random.randn(5, 8, 128).astype(np.float32)
    mask = np.array([True, True, True, False, False, False, False, False])
    Xsub = subset_channels(X, mask)
    assert Xsub.shape == (5, 3, 128)
    assert np.array_equal(Xsub, X[:, :3, :])


def test_subset_channels_all_false_raises():
    with pytest.raises(ValueError):
        subset_channels(np.random.randn(5, 8, 250), np.zeros(8, dtype=bool))


def test_template_nan_raises():
    """NaN 防御（D-nan-guard）：TemplateMatching 对 NaN 输入显式报错，不静默产出 NaN logit。"""
    X = np.random.randn(20, 8, 250).astype(np.float32)
    X[:, 3:, :] = float("nan")
    y = np.array([1, 0] * 10)
    with pytest.raises(ValueError):
        TemplateMatching().fit(X, y)
