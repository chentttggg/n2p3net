"""baselines.evaluate 模块测试：三层协议 + 配对置换检验。

合成多被试猜数字数据：每被试一个心选数字，target（当前数字==心选）在 Pz 叠正波。
语义：fold 生成正确、evaluate 端到端命中率 > chance、配对置换检验 p 值行为正确。
"""

from __future__ import annotations

import numpy as np

from baselines.classic import WindowLogisticRegression
from baselines.evaluate import (
    evaluate,
    loso_folds,
    paired_permutation_test,
    within_subject_folds,
)
from baselines.features import time_to_index

C = 8
T = 256
SFR = 256.0
TMIN = -0.2  # 秒；与 data/preprocess.py 一致


def make_multi_subject(n_subjects=8, trials_per_digit=5, seed=0):
    """多被试猜数字数据：target 在 Pz（索引 3）300–500ms 叠正高斯波。

    Returns (X, y, digits, subject_ids, true_digits)。
    """
    rng = np.random.default_rng(seed)
    i0 = time_to_index(300, SFR, TMIN)
    i1 = time_to_index(500, SFR, TMIN)
    t = np.arange(T)
    center = (i0 + i1) / 2
    width = (i1 - i0) / 6
    gauss = np.exp(-0.5 * ((t - center) / width) ** 2)

    X_list, y_list, d_list, s_list = [], [], [], []
    true_digits = {}
    for s in range(n_subjects):
        true_d = s % 9 + 1  # 心选数字 1..9 循环
        true_digits[s] = true_d
        for d in range(1, 10):
            for _ in range(trials_per_digit):
                x = rng.standard_normal((C, T)).astype(np.float32)
                is_target = d == true_d
                if is_target:
                    x[3, :] += 5.0 * gauss
                X_list.append(x)
                y_list.append(1 if is_target else 0)
                d_list.append(d)
                s_list.append(s)
    return (
        np.stack(X_list).astype(np.float32),
        np.array(y_list, dtype=int),
        np.array(d_list, dtype=int),
        np.array(s_list),
        true_digits,
    )


# ---------------- fold 生成 ----------------


def test_loso_folds():
    subject_ids = np.array([0, 0, 1, 1, 1, 2])
    folds = loso_folds(subject_ids)
    assert len(folds) == 3  # 3 个被试
    for train_mask, test_mask in folds:
        # 每个 fold 恰好留出一个被试
        assert test_mask.sum() >= 1
        assert (train_mask & test_mask).sum() == 0
        # train 是被试全集减去 test 的那个
        assert (train_mask | test_mask).all()


def test_within_subject_folds():
    subject_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    run_ids = np.array([0, 0, 1, 1, 0, 0, 1, 1, 2, 2])
    # 被试 0：2 runs → 2 folds；被试 1：3 runs → 3 folds
    folds = within_subject_folds(subject_ids, run_ids)
    assert len(folds) == 5
    for train_mask, test_mask in folds:
        train_subs = np.unique(subject_ids[train_mask])
        test_subs = np.unique(subject_ids[test_mask])
        assert len(train_subs) == 1 and len(test_subs) == 1, "fold 应限于单被试内"
        assert train_subs[0] == test_subs[0], "train/test 应属同一被试（单受试训练）"
        assert (train_mask & test_mask).sum() == 0


def test_within_subject_single_run_skipped():
    subject_ids = np.array([0, 0, 1, 1])
    run_ids = np.array([0, 0, 0, 0])  # 每个被试仅 1 run
    folds = within_subject_folds(subject_ids, run_ids)
    assert len(folds) == 0, "单 run 被试应被跳过（train 会为空）"


# ---------------- 端到端评估 ----------------


def test_evaluate_loso():
    X, y, digits, subject_ids, true_digits = make_multi_subject(n_subjects=8, trials_per_digit=5)
    folds = loso_folds(subject_ids)
    model = WindowLogisticRegression(window_ms=(250.0, 500.0))
    summary = evaluate(model, X, y, digits, subject_ids, true_digits, folds)

    assert len(summary.per_fold) == 8
    assert summary.hit_rate_mean > 0.5, (
        f"LOSO 命中率应明显 > chance 11.1%，得到 {summary.hit_rate_mean:.3f}"
    )
    assert summary.balanced_acc_mean > 0.7, (
        f"单试次 balanced acc 应 >0.7，得到 {summary.balanced_acc_mean:.3f}"
    )
    assert 0.0 <= summary.hit_rate_std <= 1.0


def test_evaluate_empty_train_raises():
    X, y, digits, subject_ids, true_digits = make_multi_subject(n_subjects=3, trials_per_digit=2)
    empty_train = (np.zeros(len(y), dtype=bool), np.ones(len(y), dtype=bool))
    import pytest

    with pytest.raises(ValueError):
        evaluate(WindowLogisticRegression(), X, y, digits, subject_ids, true_digits, [empty_train])


# ---------------- 配对置换检验 ----------------


def test_paired_permutation_identical():
    s = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    _, p = paired_permutation_test(s, s, n_perm=500, seed=0)
    assert p == 1.0, "完全无差异时 p 应为 1.0"


def test_paired_permutation_significant():
    # 10 单元完全分离（a 全对 / b 全错）：配对符号检验下精确 p = 2/2^10 ≈ 0.002
    a = np.ones(10)
    b = np.zeros(10)
    d_obs, p = paired_permutation_test(a, b, n_perm=5000, seed=0)
    assert d_obs == 1.0
    assert p < 0.05, f"完全分离应 p<0.05，得到 {p:.4f}"


def test_paired_permutation_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        paired_permutation_test([1.0, 2.0], [1.0, 2.0, 3.0])


def test_evaluate_nan_raises():
    """NaN 守卫（D-nan-guard）：X 含 NaN 时 evaluate 显式报错。"""
    import pytest

    X, y, digits, subject_ids, true_digits = make_multi_subject(n_subjects=3, trials_per_digit=2)
    X = X.copy()
    X[:, 3:, :] = float("nan")
    folds = loso_folds(subject_ids)
    with pytest.raises(ValueError):
        evaluate(WindowLogisticRegression(), X, y, digits, subject_ids, true_digits, folds)


def test_composite_group_key():
    """分组键支持 (subject, run)（D-group-key）：每 run 一个心选数字，用字符串复合键。"""
    rng = np.random.default_rng(0)
    i0, i1 = time_to_index(300, SFR, TMIN), time_to_index(500, SFR, TMIN)
    center = (i0 + i1) / 2
    width = (i1 - i0) / 6
    gauss = np.exp(-0.5 * ((np.arange(T) - center) / width) ** 2)

    X_list, y_list, d_list, s_list = [], [], [], []
    true_digits = {}
    for run, true_d in [(0, 5), (1, 8)]:
        key = f"s0_r{run}"
        true_digits[key] = true_d
        for d in range(1, 10):
            for _ in range(3):
                x = rng.standard_normal((C, T)).astype(np.float32)
                if d == true_d:
                    x[3, :] += 5.0 * gauss
                X_list.append(x)
                y_list.append(1 if d == true_d else 0)
                d_list.append(d)
                s_list.append(key)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    digits = np.array(d_list)
    subject_ids = np.array(s_list)
    folds = loso_folds(subject_ids)

    summary = evaluate(
        WindowLogisticRegression(window_ms=(250.0, 500.0)),
        X, y, digits, subject_ids, true_digits, folds,
    )
    # 两个 (subject, run) 单元都被正确计数
    assert len(summary.subject_records) == 2
    assert 0.0 <= summary.hit_rate_mean <= 1.0


# ---------------- 二分类评估（evaluate_binary） ----------------


def test_evaluate_binary_loso():
    """二分类评估（D-binary-vs-gtn）：LOSO 下 balanced_acc + AUC 明显优于随机。"""
    from baselines.evaluate import evaluate_binary

    X, y, digits, subject_ids, true_digits = make_multi_subject(n_subjects=6, trials_per_digit=5)
    folds = loso_folds(subject_ids)
    model = WindowLogisticRegression(window_ms=(250.0, 500.0))
    summary = evaluate_binary(model, X, y, subject_ids, folds)

    assert len(summary.per_fold) == 6
    assert summary.balanced_acc_mean > 0.7, (
        f"二分类 balanced_acc 应 >0.7，得到 {summary.balanced_acc_mean:.3f}"
    )
    assert summary.auc_mean > 0.8, f"二分类 AUC 应 >0.8，得到 {summary.auc_mean:.3f}"
    assert 0.0 <= summary.balanced_acc_std <= 1.0


def test_evaluate_binary_nan_raises():
    """二分类 NaN 守卫（D-binary-noguard）。"""
    import pytest

    from baselines.evaluate import evaluate_binary

    X, y, digits, subject_ids, true_digits = make_multi_subject(n_subjects=3, trials_per_digit=2)
    X = X.copy()
    X[:, 3:, :] = float("nan")
    folds = loso_folds(subject_ids)
    with pytest.raises(ValueError):
        evaluate_binary(WindowLogisticRegression(), X, y, subject_ids, folds)
