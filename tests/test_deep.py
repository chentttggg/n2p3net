"""baselines.deep 模块测试：EEGNet / EEG-Inception(ERP) / EEG Conformer 深度基线。

合成数据：target 试次在 Pz 300–500ms 叠正高斯波，non-target 纯噪声。
语义：EEGNet 学到判别（AUC>0.7）、logit 是 log-odds（target 更高）、三模型 fit/predict 跑通。
设备：显式 CPU，保证稳定与速度（真实 XPU/CUDA 的 AMP 留 Phase 1 实测）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from baselines.deep import (
    DeepBaseline,
    DeepConfig,
    _CrossEntropyObjective,
    resolve_source_risk_weights,
)
from baselines.features import time_to_index
from data.contract import DEFAULT_P300_DATA_CONTRACT

C = 8
T = DEFAULT_P300_DATA_CONTRACT.n_times
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
    assert clf.last_runtime["fused_adam_requested"] is True
    assert clf.last_runtime["compile_mode_requested"] == "reduce-overhead"
    assert clf.last_runtime["fused_adam"] is False
    assert clf.last_runtime["compile_mode"] is None
    assert clf.last_runtime["host_sync_policy"] == "epoch_boundary"
    assert clf.last_runtime["memory"]["device"] == "cpu"


def test_source_risk_natural_epoch_mass_is_exact_unweighted_counterexample() -> None:
    y = np.asarray([0, 1, 0, 1, 0, 0, 1, 0], dtype=np.int64)
    domains = np.asarray(["A"] * 4 + ["B"] * 4)
    units = np.asarray(["a1"] * 2 + ["a2"] * 2 + ["b1"] * 4)
    class_weight = np.where(y == 1, 8.0, 1.0)
    natural_mass = {
        domain: float(class_weight[domains == domain].sum() / class_weight.sum())
        for domain in ("A", "B")
    }

    weights, report = resolve_source_risk_weights(
        y,
        domains,
        units,
        domain_mass=natural_mass,
        within_domain_unit="epoch",
        pos_weight=8.0,
    )

    np.testing.assert_allclose(weights, 1.0)
    assert sum(item["achieved_class_weighted_mass"] for item in report["domains"]) == pytest.approx(1.0)


def test_source_risk_duplicate_epochs_do_not_increase_participant_mass() -> None:
    y = np.asarray([0, 1, 0, 1] + [0, 1] * 10, dtype=np.int64)
    domains = np.asarray(["TARGET"] * 4 + ["AUX"] * 20)
    units = np.asarray(["t1", "t1", "t2", "t2"] + ["a1"] * 20)

    weights, report = resolve_source_risk_weights(
        y,
        domains,
        units,
        domain_mass={"TARGET": 0.8, "AUX": 0.2},
        within_domain_unit="participant",
        pos_weight=8.0,
    )
    class_weight = np.where(y == 1, 8.0, 1.0)
    coefficient = weights * class_weight

    assert coefficient[domains == "TARGET"].sum() / coefficient.sum() == pytest.approx(0.8)
    assert coefficient[domains == "AUX"].sum() / coefficient.sum() == pytest.approx(0.2)
    target_units = [item for item in report["domains"] if item["domain_id"] == "TARGET"][0]
    assert target_units["units"] == 2


def test_row_weighted_ce_preserves_original_reduction_when_weights_are_one() -> None:
    logits = torch.tensor([[1.0, -1.0], [0.0, 2.0], [0.5, 0.25]])
    labels = torch.tensor([0, 1, 1])
    objective = _CrossEntropyObjective(8.0, torch.device("cpu"))
    expected = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 8.0]))(
        logits,
        labels,
    )

    actual = objective(logits, labels, torch.ones(3))

    assert torch.allclose(actual, expected)


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
    assert clf.parameter_count() > 0


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


def test_input_statistics_row_mask_changes_stats_without_dropping_training_rows():
    X, y = make_p300_data(n_target=12, n_nontarget=36, seed=27)
    stats_rows = np.zeros(len(X), dtype=bool)
    stats_rows[:24] = True
    X[stats_rows] += 10.0
    X[~stats_rows] -= 10.0
    clf = DeepBaseline(
        "eegnet",
        config=DeepConfig(epochs=1, batch_size=48),
        device=_cpu_device(),
    )
    expected_mean, expected_std = clf._masked_input_stats(
        X[stats_rows],
        np.ones((int(stats_rows.sum()), C), dtype=bool),
    )

    clf.fit(X, y, input_stats_row_mask=stats_rows)

    assert np.allclose(clf._input_mean, expected_mean)
    assert np.allclose(clf._input_std, expected_std)
    assert clf.last_runtime["batch_size"] == 48


def test_input_statistics_row_mask_fails_closed() -> None:
    X, y = make_p300_data(n_target=4, n_nontarget=12, seed=28)
    clf = DeepBaseline(
        "eegnet",
        config=DeepConfig(epochs=1, batch_size=16),
        device=_cpu_device(),
    )
    with pytest.raises(ValueError, match="must align"):
        clf.fit(X, y, input_stats_row_mask=np.ones(len(X) - 1, dtype=bool))
    with pytest.raises(ValueError, match="at least one row"):
        clf.fit(X, y, input_stats_row_mask=np.zeros(len(X), dtype=bool))


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


def test_multisource_selection_uses_only_declared_target_domain(monkeypatch) -> None:
    X, _ = make_p300_data(n_target=16, n_nontarget=48, seed=31)
    y = np.tile([0, 1], 32).astype(np.int64)
    groups = np.repeat(np.asarray([f"s{index}" for index in range(8)]), 8)
    domains = np.repeat(np.asarray(["TARGET"] * 6 + ["AUX"] * 2), 8)

    def split_target_groups(target_groups, **_):
        validation = target_groups == "s0"
        return SimpleNamespace(
            train_mask=~validation,
            validation_mask=validation,
            n_validation_groups=1,
        )

    monkeypatch.setattr(
        "baselines.deep.group_disjoint_validation_split",
        split_target_groups,
    )
    clf = DeepBaseline(
        "eegnet",
        config=DeepConfig(epochs=1, batch_size=16),
        device=_cpu_device(),
    )

    clf.fit(
        X,
        y,
        group_ids=groups,
        source_domain_ids=domains,
        source_domain_mass={"TARGET": 0.8, "AUX": 0.2},
        risk_unit_ids=groups,
        risk_within_domain_unit="participant",
        selection_domain="TARGET",
    )

    assert clf.last_source_risk is not None
    assert clf.last_source_risk["selection_domain"] == "TARGET"
    assert clf.last_source_risk["selection_rows"] == 8
    assert clf.last_val_groups == 1
    assert len(clf.calibration_labels_) == 8
    domain_rows = {
        item["domain_id"]: item["rows"]
        for item in clf.last_source_risk["training"]["domains"]
    }
    assert domain_rows == {"AUX": 16, "TARGET": 40}


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

def test_deep_config_validates_optional_accelerator_knobs() -> None:
    with pytest.raises(ValueError, match="compile_mode"):
        DeepConfig(compile_mode="invalid-mode")
    with pytest.raises(ValueError, match="fused_adam"):
        DeepConfig(fused_adam=1)  # type: ignore[arg-type]


def test_deep_defaults_request_cuda_optimizations_and_fall_back_on_cpu() -> None:
    config = DeepConfig()
    assert config.fused_adam is True
    assert config.compile_mode == "reduce-overhead"

    clf = DeepBaseline(
        "eegnet",
        n_chans=C,
        n_times=T,
        sfreq=SFR,
        device=torch.device("cpu"),
        config=config,
    )
    record = clf.optimizer_execution.record()
    assert record["fused_adam_requested"] is True
    assert record["compile_mode_requested"] == "reduce-overhead"
    assert record["fused_adam"] is False
    assert record["compile_mode"] is None
