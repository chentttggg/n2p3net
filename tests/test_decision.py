"""模块 #10 测试：决策层（猜数字）。

冒烟：形状 / 异常。
语义（关键路径，CODING_WORKFLOW §3）：
    - argmax 正确（核心）：累积 logit 最大的数字被猜中。
    - 对数似然比累积（非平均）：试次数多的数字靠累积获胜。
    - 空集处理（v3 P2）：无试次数字 = −inf，不参与 argmax。
    - std=0 退化（v3 P2）：所有 score 相同时不加 z-score，不除零。
    - z-score 单调：不改变 argmax。
    - 多被试分组：各被试独立判定。
    - 命中率计算。
"""

from __future__ import annotations

import numpy as np
import pytest

from models.decision import decide, hit_rate

# ---------------- 冒烟测试 ----------------


def test_shapes():
    digits = np.array([1, 2, 3])
    logits = np.array([1.0, 2.0, 3.0])
    subject_ids = np.array([0, 0, 0])
    res = decide(logits, digits, subject_ids)
    assert res.raw_scores.shape == (1, 9)  # 默认 vocab 1-9
    assert res.z_scores.shape == (1, 9)
    assert res.predicted.shape == (1,)
    assert len(res.digit_vocab) == 9


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        decide([1.0, 2.0], [1, 2], [0])


# ---------------- 语义测试 ----------------


def test_argmax_correct():
    """核心：累积 logit 最大的数字被猜中。"""
    digits = np.array(
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
    )
    logits = np.zeros(27)
    logits[12:15] = 3.0  # 数字 5
    logits[0:3] = 1.0  # 数字 1
    subject_ids = np.zeros(27, dtype=int)
    res = decide(logits, digits, subject_ids)
    assert res.predicted[0] == 5


def test_logsum_not_average():
    """对数似然比累积（非平均）：试次多的数字靠累积获胜。

    该语义测试针对未中心化 logit（center_logits=False），避免中心化改变单被试内的基准。
    """
    digits = np.array([1] + [2] * 10)  # 数字 1 有 1 试次，数字 2 有 10 试次
    logits = np.array([5.0] + [1.0] * 10)
    subject_ids = np.zeros(11, dtype=int)
    res = decide(logits, digits, subject_ids, center_logits=False)
    # 累积：score(1)=5, score(2)=10 → 猜 2（若平均：score(1)=5 > score(2)=1 → 猜 1，错误）
    assert res.predicted[0] == 2


def test_center_logits_removes_constant_bias():
    """review v6 P0-1：常数校准偏置 c 会被 Σlogit 放大为 c·n_d 进入 argmax；
    逐被试中心化后必须恢复真实判别。"""
    rng = np.random.default_rng(0)
    true_digit = 3
    counts = rng.multinomial(200, np.ones(9) / 9)
    digits = np.concatenate([[d] * n for d, n in enumerate(counts, 1)])
    # 真实判别差 0.4 + 常数偏置 +5；未中心化时 c·n_d 主导排名
    logits = np.where(digits == true_digit, 0.2, -0.2) + 5.0
    subject_ids = np.zeros(len(digits), dtype=int)

    raw = decide(logits, digits, subject_ids, center_logits=False)
    centered = decide(logits, digits, subject_ids, center_logits=True)
    assert centered.predicted[0] == true_digit, "中心化后应命中真实数字"
    # 该反例中未中心化路径必须被常数偏置带偏（回归测试，防止中心化被删）
    assert raw.predicted[0] != true_digit


def test_aggregation_mean():
    """aggregation='mean'：每数字平均 logit 后 argmax（消融轴）。"""
    digits = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    logits = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0])
    subject_ids = np.zeros(9, dtype=int)
    res = decide(logits, digits, subject_ids, aggregation="mean")
    assert res.predicted[0] == 2


def test_aggregation_invalid_raises():
    with pytest.raises(ValueError):
        decide([1.0], [1], [0], aggregation="avg")


def test_empty_set():
    """空集（v3 P2）：数字 9 无试次 → −inf，不参与 argmax。"""
    digits = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8])  # 无数 9
    logits = np.array([1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1])  # 数字 5 最大
    subject_ids = np.zeros(16, dtype=int)
    res = decide(logits, digits, subject_ids)
    assert res.predicted[0] == 5
    assert np.isneginf(res.raw_scores[0, 8]), "数字 9（vocab 第 9 列）应为空集 −inf"


def test_std_zero_degenerate():
    """std=0 退化（v3 P2）：所有非空集 score 相同，不加 z-score、不除零。"""
    digits = np.array([1, 1, 2, 2, 3, 3])
    logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 每数字累积 = 2（std=0）
    subject_ids = np.zeros(6, dtype=int)
    res = decide(logits, digits, subject_ids)
    # std=0 退化，argmax 取第一个非空集数字（信息不足的任意判定）
    assert res.predicted[0] == 1
    assert not np.isnan(res.z_scores[0, 0]), "std=0 不应产生 NaN"


def test_zscore_monotone():
    """z-score 单调：不改变 argmax。"""
    digits = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    logits = np.array([1, 1, 1, 5, 5, 5, 3, 3, 3])  # 数字 2 累积最大
    subject_ids = np.zeros(9, dtype=int)
    res = decide(logits, digits, subject_ids)
    assert res.predicted[0] == 2
    z_argmax = res.digit_vocab[int(np.argmax(res.z_scores[0]))]
    assert z_argmax == 2, "z-score 后 argmax 应与 raw 一致"


def test_multi_subject():
    """多被试分组：各被试独立判定。"""
    digits = np.array(
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]  # 被试 0
        + [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]  # 被试 1
    )
    logits = np.zeros(36)
    logits[8:10] = 3.0  # 被试 0 数字 5
    logits[32:34] = 4.0  # 被试 1 数字 8
    subject_ids = np.array([0] * 18 + [1] * 18)
    res = decide(logits, digits, subject_ids)
    assert res.predicted[0] == 5
    assert res.predicted[1] == 8


def test_hit_rate():
    predicted = np.array([5, 8, 3], dtype=object)
    true = np.array([5, 7, 3])
    assert hit_rate(predicted, true) == pytest.approx(2 / 3)


def test_duplicate_digit_vocab_raises():
    """digit_vocab 含重复数字会双计同一数字，应显式拒绝（audit P1/P2 补防）。"""
    with pytest.raises(ValueError, match="digit_vocab"):
        decide([1.0, 2.0], [1, 2], [0, 0], digit_vocab=[1, 1, 2, 3, 4, 5, 6, 7, 8])


def test_decide_nan_raises():
    """NaN 守卫（D-nan-guard）：NaN logit 显式报错，不静默当空集（review P0）。"""
    logits = np.array([1.0, 0.5, float("nan"), 0.3, 0.8])
    digits = np.array([1, 2, 3, 4, 5])
    subject_ids = np.array(["s0"] * 5)
    with pytest.raises(ValueError):
        decide(logits, digits, subject_ids)
