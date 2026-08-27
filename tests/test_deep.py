"""baselines.deep 模块测试：EEGNet / EEG-Inception(ERP) / EEG Conformer 深度基线。

合成数据：target 试次在 Pz 300–500ms 叠正高斯波，non-target 纯噪声。
语义：EEGNet 学到判别（AUC>0.7）、logit 是 log-odds（target 更高）、三模型 fit/predict 跑通。
设备：显式 CPU，保证稳定与速度（真实 XPU/CUDA 的 AMP 留 Phase 1 实测）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from baselines.deep import DeepBaseline, DeepConfig
from baselines.features import time_to_index

C = 8
T = 128
SFR = 128.0
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


def test_cpu_runtime_record_uses_bounded_matrix_batches() -> None:
    X, y = make_p300_data(n_target=12, n_nontarget=36, seed=4)
    clf = DeepBaseline(
        "eegnet",
        config=DeepConfig(epochs=1, batch_size=32, max_update_batch_size=8),
        device=_cpu_device(),
    )

    clf.fit(X, y)

    assert clf.last_runtime["precision"] == "fp32"
    assert clf.last_runtime["batch_size"] == 8
    assert clf.last_runtime["preloaded"] is False
    assert clf.last_runtime["host_sync_policy"] == "epoch_boundary"
    assert clf.last_runtime["memory"]["device"] == "cpu"


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


def test_masked_channels_stay_zero_after_input_standardization():
    """Zero-filled missing channels must not become signal after z-scoring."""
    rng = np.random.default_rng(21)
    X = rng.standard_normal((6, C, T)).astype(np.float32)
    mask = np.ones((6, C), dtype=bool)
    mask[:, 2] = False
    X[:, 2, :] = 0.0
    clf = DeepBaseline(
        "eegnet",
        channel_mask=np.ones(C, dtype=bool),
        config=DeepConfig(epochs=1),
        device=_cpu_device(),
    )

    effective = clf._effective_trial_channel_mask(X, mask)
    clf._input_mean, clf._input_std = clf._masked_input_stats(X, effective)
    prepared = clf._prepare_input(X, effective)

    assert np.allclose(prepared[:, 2, :], 0.0)
    assert np.isclose(clf._input_mean[0, 2, 0], 0.0)


def test_masked_nonzero_channels_are_removed_by_deep_input_projection():
    rng = np.random.default_rng(22)
    X = rng.standard_normal((6, C, T)).astype(np.float32)
    mask = np.ones((6, C), dtype=bool)
    mask[:, 2] = False
    X[:, 2, :] = 123.0
    clf = DeepBaseline(
        "eegnet",
        channel_mask=np.ones(C, dtype=bool),
        config=DeepConfig(epochs=1),
        device=_cpu_device(),
    )

    effective = clf._effective_trial_channel_mask(X, mask)
    clf._input_mean, clf._input_std = clf._masked_input_stats(X, effective)
    prepared = clf._prepare_input(X, effective)

    assert clf.accepts_unmaterialized_trial_channel_mask is True
    assert np.allclose(prepared[:, 2, :], 0.0)
    assert np.isclose(clf._input_mean[0, 2, 0], 0.0)


def test_masked_input_statistics_count_every_observed_time_sample():
    """Counterexample: the denominator is observed trials times T, not trials alone."""
    X = np.ones((3, 2, 4), dtype=np.float32)
    mask = np.ones((3, 2), dtype=bool)
    clf = DeepBaseline(
        "eegnet",
        n_chans=2,
        n_times=4,
        config=DeepConfig(epochs=1),
        device=_cpu_device(),
    )

    mean, std = clf._masked_input_stats(X, mask)
    clf._input_mean, clf._input_std = mean, std
    prepared = clf._prepare_input(X, mask)

    assert np.allclose(mean, 1.0)
    assert np.allclose(std, 1e-6)
    assert np.allclose(prepared, 0.0)


def test_static_channel_mask_is_enforced():
    X, y = make_p300_data(n_target=20, n_nontarget=20)
    static = np.ones(C, dtype=bool)
    static[1] = False
    X[:, 1, :] = 0.0
    clf = DeepBaseline(
        "eegnet",
        channel_mask=static,
        config=DeepConfig(epochs=1, batch_size=40),
        device=_cpu_device(),
    )
    clf.fit(X, y)
    assert np.allclose(clf.predict_logit(X), clf.predict_logit(X, np.broadcast_to(static, (len(X), C))))


def test_group_disjoint_early_stopping_and_calibration(monkeypatch):
    """Deep baselines use the same grouped split and restore minimum-val weights."""
    X, y = make_p300_data(n_target=32, n_nontarget=96, seed=9)
    subjects = np.repeat(np.arange(8), 16)
    cfg = DeepConfig(
        epochs=5,
        batch_size=32,
        lr=2e-2,
        seed=7,
        val_group_frac=0.25,
        val_groups_min=2,
        val_groups_max=2,
        early_stop_patience=2,
    )
    clf = DeepBaseline("eegnet", config=cfg, device=_cpu_device())
    clf.fit(X, y, group_ids=subjects)

    assert clf.last_val_groups == 2
    assert len(clf.last_history["val_losses"]) >= 1
    assert clf.last_history["best_epoch"] is not None
    assert clf.calibration_source_ == "group_disjoint_validation"
    assert len(clf.calibration_logits_) == len(clf.calibration_labels_) == 32
    assert np.isfinite(clf.calibration_logits_).all()


def test_group_disjoint_history_records_validation_auc(tmp_path):
    rng = np.random.default_rng(12)
    X = rng.standard_normal((64, C, T)).astype(np.float32)
    y = np.tile([0, 1], 32).astype(np.int64)
    subjects = np.repeat(np.arange(8), 8)
    clf = DeepBaseline(
        "eegnet",
        config=DeepConfig(
            epochs=2,
            batch_size=16,
            val_group_frac=0.25,
            val_groups_min=2,
            val_groups_max=2,
        ),
        device=_cpu_device(),
    )
    clf.configure_epoch_progress(tmp_path)
    clf.configure_evaluation_fold(3)
    clf.fit(X, y, group_ids=subjects)

    assert len(clf.last_history["task_val_aucs"]) == len(clf.last_history["val_losses"])
    assert all(value is not None and 0.0 <= value <= 1.0 for value in clf.last_history["task_val_aucs"])
    rows = [
        json.loads(line) for line in (tmp_path / "fold_3.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(clf.last_history["val_losses"])
    assert all(row["task_val_auc"] is not None for row in rows)
