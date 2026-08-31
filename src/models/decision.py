"""模块 #10：决策层（猜数字，Decision Layer）。

职责（blueprint §6）：
    把试次级 logit（每个试次的 target/non-target 对数似然比）聚合成被试级的「猜数字」判定：
        score(d) = weighted_mean(d) * n_eff(d)^beta
        被试内 z-score（去校准偏差）→ d̂* = argmax_d score(d)

    命中率 = P(d̂* = d*)，chance ≈ 11.1%（9 选 1）。

明确「不做」：
    - 不涉及任何可学习参数 / 梯度（纯后处理函数，无 nn.Module）。
    - 不处理原始 EEG，只消费 heads 输出的 logit_target。

三思决策记录（供后续会话追溯）：
    D-tempered       `beta=0/0.5/1` 分别对应 weighted mean、平方根温度化和完整有效计数
                     累积；单位权重时 `beta=1` 等于 sum。该模式使用全部 trial，同时减弱
                     随机刺激次数不齐带来的排名偏移。
    D-default-mean   245-subject GTN all-evidence comparison found beta=0 strongest for K35
                     and no worse for K65. The shared default is therefore candidate mean;
                     explicit sum remains available for frozen-result reproduction.
    D-center-logits  （review v6 P0-1，推翻旧版「score 后 z-score 去校准偏差」）分类器常数偏置 c
                      经 Σlogit 变成 c·n_d，n_d 逐数字不等时进入 argmax；score 后 z-score 是同一
                      仿射变换，去不掉该项。因此累加前必须先对每个 subject 的 logit 做被试内中心化。
                      实测 GTN 30 被试 LOSO：SWLDA 0.467 → 0.667（sum）/0.700（mean）。
    D-zscore-monotone 被试内 z-score 是**线性单调变换，数学上不改变 argmax**。其真实价值是：
                     ①输出标准化的 score 作「置信度」（如最高 z 仅 0.5σ 可标低置信）；②跨被试
                     score 可比（Phase 4 分析）。argmax 判定本身不受 z-score 影响，故实现中用
                     raw_scores 做 argmax（等价且更清晰），z_scores 仅作输出。忠实蓝图保留 z-score。
    D-vocab          数字集用显式 digit_vocab（默认 1–9），**不用 np.unique(digits)**——因为若某数字
                     全局未出现，np.unique 会丢失它、无法表示「空集」。显式 vocab 保证 9 列固定，
                     空集（-inf）可正确表示。
    D-empty-set      空集（伪迹剔除后某被试的某数字试次清零，v3 P2）：score=−inf，不参与 argmax。
    D-std-zero       std=0（某被试所有非空集数字 score 相同）退化为「不加 z-score」（v3 P2），
                     避免除零；此时 argmax 取第一个非空集数字（信息不足的任意判定）。
    D-all-empty      极端病态：某被试所有数字均空集 → predicted=None（数据问题，非判定逻辑问题）。
    D-nan-guard      NaN logit 会经 Σlogit 变 NaN score，被 isfinite 静默当空集，命中率分母悄悄变小
                     （review P0 的「论文级」失真）。故入口显式报错，不静默。
    D-decide-vec     聚合/中心化/argmax 全部用 (subject, digit) 二维桶 + bincount + 行广播完成，
                     不逐被试/逐数字掩码循环（N=50k、244 被试实测 ~18x）。

契约（输入 → 输出）：
    logits (N,) float、digits (N,) int（试次对应数字）、subject_ids (N,)（被试分组）→
    DecisionResult{predicted (n_subjects,), z_scores/raw_scores (n_subjects, n_digits),
                   subject_ids, digit_vocab}。

依赖的决策：blueprint §6、constitution D3（禁报原始准确率，用命中率）、heads.D-logit-out
    （heads 输出 logit 供本层累积）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_EVIDENCE_COUNT_POWER = 0.5
DEFAULT_EVIDENCE_AGGREGATION = "mean"
COUNT_AGGREGATIONS = frozenset({"sum", "mean", "tempered_evidence"})


def count_tempered_evidence_scores(
    weighted_sums: np.ndarray,
    weight_sums: np.ndarray,
    squared_weight_sums: np.ndarray,
    *,
    count_power: float = DEFAULT_EVIDENCE_COUNT_POWER,
) -> np.ndarray:
    """Return weighted means tempered by effective evidence count.

    With unit weights this is ``mean(logit) * n**count_power``. Therefore
    ``count_power=0`` is mean aggregation and ``count_power=1`` is sum.
    """

    weighted_sums = np.asarray(weighted_sums, dtype=float)
    weight_sums = np.asarray(weight_sums, dtype=float)
    squared_weight_sums = np.asarray(squared_weight_sums, dtype=float)
    if not weighted_sums.shape == weight_sums.shape == squared_weight_sums.shape:
        raise ValueError("weighted sums and weight moments must have identical shapes.")
    if not np.isfinite(weighted_sums).all():
        raise ValueError("weighted evidence sums contain NaN/inf.")
    if not np.isfinite(count_power) or not 0.0 <= count_power <= 1.0:
        raise ValueError("count_power must be finite and in [0, 1].")
    if np.any(weight_sums < 0.0) or np.any(squared_weight_sums < 0.0):
        raise ValueError("evidence weight moments must be non-negative.")

    scores = np.full(weighted_sums.shape, -np.inf, dtype=float)
    nonempty = (weight_sums > 0.0) & (squared_weight_sums > 0.0)
    effective_count = np.zeros(weighted_sums.shape, dtype=float)
    effective_count[nonempty] = (
        weight_sums[nonempty] ** 2 / squared_weight_sums[nonempty]
    )
    weighted_mean = np.zeros(weighted_sums.shape, dtype=float)
    weighted_mean[nonempty] = weighted_sums[nonempty] / weight_sums[nonempty]
    scores[nonempty] = weighted_mean[nonempty] * effective_count[nonempty] ** count_power
    return scores


@dataclass
class DecisionResult:
    """决策层输出。

    Attributes
    ----------
    predicted : np.ndarray
        (n_subjects,) 每个被试猜的数字（object dtype；全空集时为 None）。
    z_scores : np.ndarray
        (n_subjects, n_digits) 被试内 z-score 后 score（空集 −inf）。
    raw_scores : np.ndarray
        (n_subjects, n_digits) 原始累积 logit（空集 −inf）。
    subject_ids : np.ndarray
        (n_subjects,) 排序后的被试 id（与 predicted 行对应）。
    digit_vocab : np.ndarray
        (n_digits,) 数字集（列对应）。
    """

    predicted: np.ndarray
    z_scores: np.ndarray
    raw_scores: np.ndarray
    subject_ids: np.ndarray
    digit_vocab: np.ndarray


def decide(
    logits: Sequence[float],
    digits: Sequence[int],
    subject_ids: Sequence,
    digit_vocab: Sequence[int] | None = None,
    center_logits: bool = True,
    aggregation: str = DEFAULT_EVIDENCE_AGGREGATION,
    evidence_count_power: float = DEFAULT_EVIDENCE_COUNT_POWER,
    trial_weights: Sequence[float] | None = None,
) -> DecisionResult:
    """试次级 logit → 被试级猜数字判定。

    Parameters
    ----------
    logits : Sequence[float]
        (N,) 每个试次的 target/non-target logit（heads 输出）。
    digits : Sequence[int]
        (N,) 每个试次对应的刺激数字。
    subject_ids : Sequence
        (N,) 每个试次的被试 id（z-score 在被试内做）。
    digit_vocab : Sequence[int] | None
        完整数字集（默认 1–9）。显式指定以保证空集可表示（D-vocab）。
    center_logits : bool
        累加前是否对每个 subject 的 logit 做被试内中心化（默认 True）。
        分类器截距/类先验/pos_weight 会引入常数偏置 c；若直接 Σlogit，c 会变成
        c·n_d（n_d 为每数字试次数）进入排名，被试内 score z-score 无法消除该项。
        先中心化再累加/平均，可去掉这一校准偏置（review v6 P0-1）。
    aggregation : str
        ``sum``、``mean`` 或 ``tempered_evidence``。单位 trial 权重下，前两者等于
        温度化公式的 ``beta=1/0``。默认 ``mean``，避免 all-evidence 中随机
        candidate 次数直接进入排名；历史 sum 必须显式指定。
    evidence_count_power : float
        温度化证据的 ``beta``，范围 ``[0,1]``；默认 0.5。
    trial_weights : Sequence[float] | None
        可选的非负、label-free trial 可靠性权重。省略时每条 trial 权重为 1。

    Returns
    -------
    DecisionResult
        predicted 为每个被试猜的数字（argmax）。
    """
    if aggregation not in COUNT_AGGREGATIONS:
        raise ValueError(
            f"aggregation 须为 {sorted(COUNT_AGGREGATIONS)}，得到 {aggregation!r}。"
        )

    logits = np.asarray(logits, dtype=float)
    digits = np.asarray(digits)
    subject_ids = np.asarray(subject_ids)
    if not (len(logits) == len(digits) == len(subject_ids)):
        raise ValueError(
            f"logits/digits/subject_ids 长度须一致，得到 {len(logits)}/{len(digits)}/{len(subject_ids)}。"
        )
    # NaN 守卫（D-nan-guard）：NaN 会经 Σlogit 变成 NaN score，被 isfinite 静默当空集，
    # 导致命中率分母悄悄变小——这是「论文级」的指标失真。故显式报错，不静默。
    if not np.isfinite(logits).all():
        raise ValueError(
            "logits 含 NaN/inf：上游模型对缺失通道输出异常。请先处理缺失通道"
            "（classic/riemann 用存在通道子集，deep 按原生物理通道数构造）后再喂 decision 层。"
        )
    weights = (
        np.ones(len(logits), dtype=float)
        if trial_weights is None
        else np.asarray(trial_weights, dtype=float)
    )
    if weights.shape != logits.shape:
        raise ValueError("trial_weights 必须与 logits 对齐。")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("trial_weights 必须有限且非负。")
    if len(weights) and not np.any(weights > 0.0):
        raise ValueError("trial_weights 至少包含一个正权重。")

    if digit_vocab is None:
        digit_vocab = np.arange(1, 10)
    digit_vocab = np.asarray(digit_vocab)
    if digit_vocab.ndim != 1 or len(np.unique(digit_vocab)) != len(digit_vocab):
        raise ValueError(f"digit_vocab 须为一维且不含重复数字，得到 {digit_vocab.tolist()}。")

    uniq_subjects, inverse = np.unique(subject_ids, return_inverse=True)
    n_subs = len(uniq_subjects)
    n_digits = len(digit_vocab)

    # 0. 逐被试 logit 中心化（review v6 P0-1：去除 c·n_d 校准偏置）
    if center_logits:
        subject_weight_sums = np.bincount(inverse, weights=weights, minlength=n_subs)
        subject_logit_sums = np.bincount(
            inverse, weights=weights * logits, minlength=n_subs
        )
        if np.any(subject_weight_sums <= 0.0):
            raise ValueError("每个 subject 必须至少有一个正权重 trial。")
        centered = logits - subject_logit_sums[inverse] / subject_weight_sums[inverse]
    else:
        centered = logits

    # 1. score(d) = Σ logit（D-logsum）或每数字均值（消融轴）。
    # 向量化：一次性构造 (subject, digit) 二维桶索引，用 bincount 累加/计数，
    # 避免原实现 O(N × n_subs × n_digits) 的逐被试、逐数字掩码循环。
    digit_matches = digits[:, None] == digit_vocab[None, :]  # (N, n_digits) bool
    digit_cols = np.arange(n_digits, dtype=np.int64)[None, :]
    bucket_idx = inverse[:, None] * n_digits + digit_cols  # (N, n_digits)
    bucket_idx = bucket_idx[digit_matches]
    matched_logits = np.broadcast_to(centered[:, None], digit_matches.shape)[digit_matches]
    matched_weights = np.broadcast_to(weights[:, None], digit_matches.shape)[digit_matches]

    n_buckets = n_subs * n_digits
    bucket_counts = np.bincount(bucket_idx, minlength=n_buckets)
    bucket_weight_sums = np.bincount(
        bucket_idx, weights=matched_weights, minlength=n_buckets
    )
    bucket_squared_weight_sums = np.bincount(
        bucket_idx, weights=matched_weights**2, minlength=n_buckets
    )
    bucket_sums = np.bincount(
        bucket_idx, weights=matched_weights * matched_logits, minlength=n_buckets
    )
    bucket_counts = bucket_counts.reshape(n_subs, n_digits)
    bucket_weight_sums = bucket_weight_sums.reshape(n_subs, n_digits)
    bucket_squared_weight_sums = bucket_squared_weight_sums.reshape(n_subs, n_digits)
    bucket_sums = bucket_sums.reshape(n_subs, n_digits)

    raw_scores = np.full((n_subs, n_digits), -np.inf)
    nonempty = (bucket_counts > 0) & (bucket_weight_sums > 0.0)
    if aggregation == "sum":
        raw_scores[nonempty] = bucket_sums[nonempty]
    elif aggregation == "mean":
        raw_scores[nonempty] = bucket_sums[nonempty] / bucket_weight_sums[nonempty]
    else:
        raw_scores = count_tempered_evidence_scores(
            bucket_sums,
            bucket_weight_sums,
            bucket_squared_weight_sums,
            count_power=evidence_count_power,
        )

    # 2. 被试内 z-score（非空集上，D-zscore-monotone / D-empty-set / D-std-zero）
    z_scores = np.full((n_subs, n_digits), -np.inf)
    valid = np.isfinite(raw_scores)
    n_valid = valid.sum(axis=1)
    if n_subs > 0 and n_valid.max() > 0:
        finite_vals = np.where(valid, raw_scores, 0.0)
        mu = finite_vals.sum(axis=1) / np.maximum(n_valid, 1)
        # 先减均值再平方，避免 E[x²]−μ² 在大 score 下的灾难性抵消（audit P2-9）；
        # ddof=0 与旧实现的 np.std 默认口径一致。
        centered_vals = np.where(valid, raw_scores - mu[:, None], 0.0)
        var = (centered_vals * centered_vals).sum(axis=1) / np.maximum(n_valid, 1)
        std = np.sqrt(np.maximum(var, 0.0))
        has_valid = n_valid > 0
        std_safe = np.where(std > 1e-8, std, 1.0)
        z_vals = np.where(
            (std > 1e-8)[:, None],
            (raw_scores - mu[:, None]) / std_safe[:, None],
            raw_scores,
        )
        z_scores[has_valid] = z_vals[has_valid]

    # 3. argmax（raw_scores 上做，z-score 单调等价；空集 −inf 天然排除）。
    # Exact/numerical ties abstain: choosing the smallest encoded digit would
    # make accuracy depend on arbitrary candidate naming.
    predicted = np.full(n_subs, None, dtype=object)
    has_any = n_valid > 0
    for row in np.flatnonzero(has_any):
        maximum = float(np.max(raw_scores[row]))
        tied = np.flatnonzero(
            np.isclose(raw_scores[row], maximum, rtol=1e-12, atol=1e-12)
        )
        if len(tied) == 1:
            predicted[row] = digit_vocab[tied[0]]

    return DecisionResult(
        predicted=predicted,
        z_scores=z_scores,
        raw_scores=raw_scores,
        subject_ids=uniq_subjects,
        digit_vocab=digit_vocab,
    )


def hit_rate(predicted: Sequence, true_digits: Sequence) -> float:
    """命中率 = P(d̂* = d*)；无法判定的 ``None`` 按未命中计入分母。"""
    predicted = np.asarray(predicted, dtype=object)
    true_digits = np.asarray(true_digits)
    if predicted.shape != true_digits.shape:
        raise ValueError(
            f"predicted 与 true_digits 形状须一致，得到 {predicted.shape} / {true_digits.shape}。"
        )
    if not predicted.size:
        return 0.0
    return float((predicted == true_digits).mean())
