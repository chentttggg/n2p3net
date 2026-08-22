"""baselines.deep 模块测试：EEGNet / EEG-Inception(ERP) / EEG Conformer 深度基线。

合成数据：target 试次在 Pz 300–500ms 叠正高斯波，non-target 纯噪声。
语义：EEGNet 学到判别（AUC>0.7）、logit 是 log-odds（target 更高）、三模型 fit/predict 跑通。
设备：显式 CPU，保证稳定与速度（真实 XPU/CUDA 的 AMP 留 Phase 1 实测）。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from baselines.deep import DeepBaseline, DeepConfig
from baselines.features import time_to_index

C = 8
T = 256
SFR = 256.0
TMIN = -0.2  # 秒；与 data/preprocess.py 一致


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
    X[:n_target, 3, :] += 5.0 * gauss
    idx = rng.permutation(n)
    return X[idx].astype(np.float32), y[idx]


def _cpu_device():
    return torch.device("cpu")


# ---------------- 语义 ----------------


def test_eegnet_learns():
    X, y = make_p300_data()
    Xtr, Xte = X[:400], X[400:]
    ytr, yte = y[:400], y[400:]
    clf = DeepBaseline("eegnet", config=DeepConfig(epochs=40), device=_cpu_device())
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_logit(Xte))
    assert auc > 0.7, f"EEGNet AUC 应 >0.7，得到 {auc:.3f}"


def test_logit_is_logodds():
    X, y = make_p300_data(n_target=40, n_nontarget=160)
    clf = DeepBaseline("eegnet", config=DeepConfig(epochs=40), device=_cpu_device())
    clf.fit(X, y)
    logits = clf.predict_logit(X)
    assert logits[y == 1].mean() > logits[y == 0].mean(), "target 试次 logit 应更高"


def test_predict_proba_range():
    X, y = make_p300_data(n_target=40, n_nontarget=160)
    clf = DeepBaseline("eegnet", config=DeepConfig(epochs=10), device=_cpu_device())
    clf.fit(X, y)
    p = clf.predict_proba(X)
    assert (p >= 0).all() and (p <= 1).all()


# ---------------- 冒烟（三模型） ----------------


def test_all_models_fit_predict():
    X, y = make_p300_data(n_target=40, n_nontarget=160)
    for name in ["eegnet", "inception", "conformer"]:
        clf = DeepBaseline(name, config=DeepConfig(epochs=2), device=_cpu_device())
        clf.fit(X, y)
        logits = clf.predict_logit(X)
        assert logits.shape == (200,), name
        assert np.isfinite(logits).all(), name


# ---------------- 契约 ----------------


def test_model_name_normalization():
    clf = DeepBaseline("EEGNet", device=_cpu_device())
    assert clf.model_name == "eegnet"


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        DeepBaseline("resnet")


def test_unfitted_predict_raises():
    clf = DeepBaseline("eegnet", device=_cpu_device())
    X = np.random.randn(10, C, T).astype(np.float32)
    with pytest.raises(RuntimeError):
        clf.predict_logit(X)


def test_wrong_channels_raises():
    clf = DeepBaseline("eegnet", device=_cpu_device())
    X = np.random.randn(10, 6, T).astype(np.float32)  # 6 通道 ≠ 8
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        clf.fit(X, y)
