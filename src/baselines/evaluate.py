"""模块：评估协议（Evaluation Protocol）。

职责（roadmap Phase 1 + constitution D3/P8）：
    三层评估协议（within-subject / LOSO / 跨数据集）+ 指标（9 选 1 命中率 / balanced acc /
    AUC）+ 配对置换检验（防 +2pt 假阳性）。这是所有基线公平比较的地基，也是 Phase 2
    N2P3-Net 的对照坐标系。

    fold 生成与评估执行分离：`loso_folds` / `within_subject_folds` 只生成 (train_mask,
    test_mask) 索引；`evaluate` 负责 fit → predict_logit → decide → 指标。这样 fold 语义
    可独立测试、可复现，评估循环单一职责。

明确「不做」：
    - 不做跨数据集的 fold 生成（跨数据集 = 全源域 train + 全目标域 test，fold 由调用方
      显式给定，evaluate 只消费）。
    - 不做统计检验的 bootstrap 变体（Phase 4 若需再加）。

三思决策记录（供后续会话追溯）：
    D-hit-vs-bacc   命中率（9 选 1，chance 11.1%）与 balanced_acc/AUC（target/non-target 二分类）
                    是**两个不同层级**的指标：前者是决策层 argmax 后的「猜中数字」概率，后者是
                    单试次判别质量。两者互补，都报，但对外口径用命中率（constitution D3 禁报
                    原始准确率）。
    D-bacc-center   balanced_acc 的 0 阈值只在 logit 无常数偏置时成立；分类器截距/pos_weight
                    会引入偏置，故先对 fold 内 logit 中心化再取 0 阈值（与 decide 的中心化口径
                    一致，review v6 P0-1）。AUC 是 rank 指标，不受常数偏置影响，仍用原始 logit。
    D-fold-mask     fold 用**布尔掩码**而非索引数组——语义更清晰、防索引越界、易与 subject_ids
                     对齐。train/test 掩码互斥且可各自为空（空 train 由 evaluate 抛错，空 test
                     跳过）。
    D-within-run    within-subject 用「每个被试内按 run 留一」：train 只含该被试自己的数据
                    （单受试多次本体训练），杜绝跨被试泄漏。**注意**：GTN 每被试仅 1 run（猜 1 个
                     数字），无法 within-subject，只能用 LOSO；within-subject 仅适用于自有 8 导
                     （成人多 run）。单 run 被试在 within_subject_folds 中被跳过（train 会为空）。
    D-perm-paired   配对置换检验等价于 paired t-test 的非参版本：对每个单元（被试/fold）的
                     两模型分数差 d_i = a_i − b_i，置换时随机翻转 d_i 的符号（等价交换 a/b），
                     重算 mean，p = P(|mean_perm| ≥ |mean_obs|)。Monte Carlo +1 校正避免 p=0。
    D-auc-guard     测试集若仅一类（极端 LOSO/不平衡），roc_auc_score 会报错；此时 AUC 置 NaN
                     并继续（balanced_acc 与命中率仍可算）。
    D-nan-guard     X 含 NaN（缺失通道未处理）会让 classic/riemann 崩或静默失真，入口显式报错，
                     提示 subset_channels 或零填充（配合 decision 层的 NaN 报错，杜绝静默失真）。
    D-hit-by-unit   命中率按「猜测单元」（group key = 被试或 (subject, run)）聚合，而非按 fold 等权；
                     fold 等权在 within-subject 下会让多 run 被试权重虚高。
    D-group-key     subject_ids 是通用「分组键」，支持 (subject, run) 组合（用字符串如 f"{subj}_{run}"）：
                     每 run 猜一个数字的自有数据须含 run；GTN（每被试 1 数字）用纯 subject。
      D-parallel-fold  LOSO fold 完全独立，n_jobs>1 时用线程池并行 fold（**线程而非进程**：共享同一份
                       X/y 内存，避免 Windows spawn 下每个进程复制 ~120MB 数据）；每个 worker 深拷贝
                       未拟合模型，并用 threadpool_limits(1) 限制 BLAS，避免 6×24 线程 oversubscription。
                       实测 12 fold xDAWN：串行 0.80s → 4 线程 0.32s；全尺寸合成 fold 串行 5.4s/12 folds
                       → 6 线程 2.2s/12 folds。deep 模型在 run_gtn_baseline 中 n_jobs=1（避免显存争用）。

契约（输入 → 输出）：
    loso_folds(subject_ids) / within_subject_folds(subject_ids, run_ids) → list[(train_mask, test_mask)]
    evaluate(model, X, y, digits, subject_ids, true_digits, folds) → EvalSummary
    paired_permutation_test(scores_a, scores_b, n_perm) → (obs_diff, p_value)

依赖的决策：roadmap Phase 1、constitution D3/P8、models/decision.decide、baselines.classic.Baseline。
"""

from __future__ import annotations

import copy
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

try:  # threadpoolctl 是 sklearn 依赖；无它时并行路径退化为无 BLAS 线程限制
    from threadpoolctl import threadpool_limits
except Exception:  # noqa: BLE001
    threadpool_limits = None

from models.decision import decide


# ---------------- fold 生成 ----------------


def loso_folds(subject_ids: Sequence) -> list[tuple[np.ndarray, np.ndarray]]:
    """留一被试（leave-one-subject-out）：每个被试依次作 test、其余作 train。

    Returns
    -------
    list[(train_mask, test_mask)]，每元布尔掩码长度 = len(subject_ids)。
    """
    subject_ids = np.asarray(subject_ids)
    folds = []
    for subj in np.unique(subject_ids):
        test_mask = subject_ids == subj
        folds.append((~test_mask, test_mask))
    return folds


def within_subject_folds(
    subject_ids: Sequence, run_ids: Sequence
) -> list[tuple[np.ndarray, np.ndarray]]:
    """每个被试内按 run 留一（D-within-run）：train 仅含该被试自己的数据。

    对每个被试，其 run 分组后依次留一个 run 作 test、其余 run 作 train。
    单 run 被试（如 GTN）无法切分，跳过（train 会为空）。

    Returns
    -------
    list[(train_mask, test_mask)]，布尔掩码长度 = len(subject_ids)。
    """
    subject_ids = np.asarray(subject_ids)
    run_ids = np.asarray(run_ids)
    if len(subject_ids) != len(run_ids):
        raise ValueError("subject_ids 与 run_ids 长度须一致。")

    folds = []
    for subj in np.unique(subject_ids):
        subj_mask = subject_ids == subj
        runs = np.unique(run_ids[subj_mask])
        if len(runs) < 2:
            continue  # 单 run 无法切分（D-within-run）
        for r in runs:
            test_mask = subj_mask & (run_ids == r)
            train_mask = subj_mask & ~test_mask
            folds.append((train_mask, test_mask))
    return folds


# ---------------- 评估执行 ----------------


@dataclass
class FoldResult:
    """单折评估结果。"""

    hit_rate: float
    balanced_acc: float
    auc: float
    n_subjects: int
    n_test_trials: int


@dataclass
class EvalSummary:
    """多折评估汇总。"""

    hit_rate_mean: float
    hit_rate_std: float
    balanced_acc_mean: float
    auc_mean: float
    per_fold: list[FoldResult] = field(default_factory=list)
    # 全部折中每个猜测单元（group key）的 (predicted, true, group) 记录（供置换检验按单元对齐）
    subject_records: list[tuple[object, object, object]] = field(default_factory=list)


@contextmanager
def _fold_threadpool_limits() -> Iterator[None]:
    """并行 fold 时把 BLAS 限制为单线程，避免 6 个 fold × 24 BLAS 线程互相抢占。"""
    if threadpool_limits is not None:
        with threadpool_limits(limits=1):
            yield
    else:
        yield


def _fit_model_with_optional_subjects(model, X, y, subject_ids, train_mask) -> None:
    """按模型能力调用 fit（GLM：被试级验证早停协议）。

    声明 ``fit_accepts_subject_ids = True`` 的模型（如 N2P3NetBaseline）会收到
    ``subject_ids``，可在 fit 内按被试分组切验证集做早停（同被试试次不跨 train/val，
    杜绝试次级随机切分的同被试泄漏高估）；未声明的模型走旧 fit(X, y) 契约，完全向后兼容。
    """
    if getattr(model, "fit_accepts_subject_ids", False):
        model.fit(X[train_mask], y[train_mask], subject_ids=subject_ids[train_mask])
    else:
        model.fit(X[train_mask], y[train_mask])


def _evaluate_one_fold(
    model,
    X: np.ndarray,
    y: np.ndarray,
    digits: np.ndarray,
    subject_ids: np.ndarray,
    true_digits: Mapping,
    digit_vocab: Sequence[int],
    decision_center: bool,
    decision_aggregation: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[FoldResult, list[tuple[object, object, object]]]:
    """执行单个 fold，返回 (FoldResult, subject_records)。与 evaluate 串行路径完全同构。"""
    if not train_mask.any():
        raise ValueError("train_mask 为空：无法训练（within-subject 单 run 会触发此错误）。")

    _fit_model_with_optional_subjects(model, X, y, subject_ids, train_mask)
    logits = model.predict_logit(X[test_mask])

    # 非有限 logits 统一入口守卫（audit P2-1），避免 bacc/AUC 先吃 NaN 再在 decide 报错。
    if not np.isfinite(logits).all():
        raise ValueError("模型 predict_logit 输出含 NaN/inf，无法进入 bacc/AUC/decision。")

    # 试次级二分类指标（D-hit-vs-bacc）
    # review v6 P0-1：阈值 0 只在 logit 无常数偏置时是 balanced 阈值；
    # 先对 fold 内 logit 中心化再取 0 阈值，与决策层的中心化口径一致。
    y_test = y[test_mask]
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        logits_centered = logits - logits.mean()
        bacc = float(balanced_accuracy_score(y_test, (logits_centered > 0).astype(int)))
    else:
        bacc = np.nan  # 单类测试集无法定义 balanced accuracy（audit P2-2）
    auc = np.nan
    if n_classes == 2:  # D-auc-guard
        auc = float(roc_auc_score(y_test, logits))

    # 决策层：9 选 1 命中率
    result = decide(
        logits,
        digits[test_mask],
        subject_ids[test_mask],
        digit_vocab,
        center_logits=decision_center,
        aggregation=decision_aggregation,
    )
    records: list[tuple[object, object, object]] = []
    n_hit = 0
    n_counted = 0
    for i, subj in enumerate(result.subject_ids):
        true_d = true_digits.get(subj)
        if true_d is None:
            continue  # 该被试缺失真实数字，不计入
        records.append((result.predicted[i], true_d, subj))
        n_counted += 1
        if result.predicted[i] == true_d:
            n_hit += 1
    hit = (n_hit / n_counted) if n_counted > 0 else 0.0

    return (
        FoldResult(
            hit_rate=hit,
            balanced_acc=bacc,
            auc=auc,
            n_subjects=n_counted,
            n_test_trials=int(test_mask.sum()),
        ),
        records,
    )


def _run_fold_threaded(
    args: tuple[
        int,
        object,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Mapping,
        Sequence[int],
        bool,
        str,
        np.ndarray,
        np.ndarray,
    ],
) -> tuple[int, FoldResult, list[tuple[object, object, object]]]:
    """线程池 worker：每个 fold 深拷贝一个未拟合模型，避免多线程共享 fit 状态。"""
    (
        fold_idx,
        model_proto,
        X,
        y,
        digits,
        subject_ids,
        true_digits,
        digit_vocab,
        decision_center,
        decision_aggregation,
        train_mask,
        test_mask,
    ) = args
    model = copy.deepcopy(model_proto)
    with _fold_threadpool_limits():
        fold_result, records = _evaluate_one_fold(
            model,
            X,
            y,
            digits,
            subject_ids,
            true_digits,
            digit_vocab,
            decision_center,
            decision_aggregation,
            train_mask,
            test_mask,
        )
    return fold_idx, fold_result, records


def evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    digits: np.ndarray,
    subject_ids: np.ndarray,
    true_digits: Mapping,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    digit_vocab: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9),
    decision_center: bool = True,
    decision_aggregation: str = "sum",
    n_jobs: int = 1,
    on_fold_end: Optional[Callable] = None,
) -> EvalSummary:
    """按 folds 评估一个「有 fit/predict_logit 接口」的模型。

    Parameters
    ----------
    model : 有 fit(X,y) / predict_logit(X)→(N,) 接口的分类器（Baseline 或 N2P3-Net 包装）。
    X / y : (N,C,T) / (N,) 试次数据与 target/non-target 标签（y 用于训练）。
    digits : (N,) 每个试次对应的刺激数字（决策层累加用）。
    subject_ids : (N,) 每个试次所属被试（决策层 z-score + 命中率分组用）。
    true_digits : Mapping {分组键: 心选数字}，key 须与 subject_ids 的分组键一致（subject 或
        (subject, run) 组合，见 D-group-key）。
    folds : list[(train_mask, test_mask)] 布尔掩码。
    digit_vocab : 完整数字集（默认 1–9）。
    decision_center : bool
        决策层是否先逐被试中心化 logit（默认 True，review v6 P0-1）。
    decision_aggregation : str
        决策层聚合方式："sum"（默认）或 "mean"（消融轴）。
      n_jobs : int
          并行 fold 的线程数（默认 1 串行）。LOSO 各 fold 独立，线程池复用同一份 X/y 内存，
          不复制大数据；每个 worker 内把 BLAS 限制为单线程，避免 oversubscription
          （review v6 性能项）。
      on_fold_end : Callable | None
          逐 fold 回调 on_fold_end(fold_idx, fold_result, records)（GLM v3 实时进度：
          runner 用它把每 fold 指标增量写入 progress.jsonl 供仪表盘消费）。串行路径在
          每 fold 完成即调用；并行路径在全部完成后按 fold 顺序补调（实时性以串行为准）。

    Returns
    -------
    EvalSummary：命中率（均值/标准差）+ balanced_acc 均值 + AUC 均值 + 逐折 + 逐被试记录。
    """
    X = np.asarray(X)
    y = np.asarray(y)
    digits = np.asarray(digits)
    subject_ids = np.asarray(subject_ids)
    # review v6 P0-3：true_digits 中存在但 subject_ids 中没有的被试，说明被质量排除后仍残留
    # ground truth；若继续静默忽略会缩小命中率分母。显式告警，避免论文级口径失真。
    missing_truth = set(true_digits.keys()) - set(np.unique(subject_ids).tolist())
    if missing_truth:
        warnings.warn(
            "true_digits 包含未出现在 subject_ids 中的被试（可能被伪迹剔除/元数据缺失排除）："
            f"{sorted(missing_truth)[:5]}{' ...' if len(missing_truth) > 5 else ''}。"
            "请以实际可评估被试数作为命中率分母。",
            stacklevel=2,
        )
    # NaN 守卫（D-nan-guard）：缺失通道未处理会崩（WindowLR/xDAWN）或静默失真（TemplateMatching
    # 的 NaN logit 被 decision 当空集），故显式报错，不静默。
    if np.isnan(X).any():
        raise ValueError(
            "X 含 NaN：缺失通道须先处理——classic/riemann 用 subset_channels 提取子集，"
            "deep 零填充到 8 导。"
        )

    per_fold: list[FoldResult] = []
    subject_records: list[tuple[object, object, object]] = []

    # 预先把 fold 掩码转成 bool 并跳过空 test；空 train 语义与串行路径一致（抛错）。
    mask_folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_mask, test_mask in folds:
        train_mask = np.asarray(train_mask, dtype=bool)
        test_mask = np.asarray(test_mask, dtype=bool)
        if not train_mask.any():
            raise ValueError("train_mask 为空：无法训练（within-subject 单 run 会触发此错误）。")
        if not test_mask.any():
            continue
        mask_folds.append((train_mask, test_mask))

    if n_jobs > 1 and len(mask_folds) > 1:
        # review v6 性能项：LOSO fold 完全独立，线程池共享 X/y 内存（不复制大数据）。
        # 每个 worker 深拷贝未拟合模型；threadpool_limits 把每个 worker 的 BLAS 压到 1 线程。
        task_args = [
            (
                fold_idx,
                model,
                X,
                y,
                digits,
                subject_ids,
                true_digits,
                digit_vocab,
                decision_center,
                decision_aggregation,
                train_mask,
                test_mask,
            )
            for fold_idx, (train_mask, test_mask) in enumerate(mask_folds)
        ]
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(_run_fold_threaded, task_args))
        # 按 fold 原始顺序恢复
        for _, fold_result, records in sorted(results, key=lambda item: item[0]):
            per_fold.append(fold_result)
            subject_records.extend(records)
            if on_fold_end is not None:
                on_fold_end(len(per_fold) - 1, fold_result, records)
    else:
        for train_mask, test_mask in mask_folds:
            fold_result, records = _evaluate_one_fold(
                model,
                X,
                y,
                digits,
                subject_ids,
                true_digits,
                digit_vocab,
                decision_center,
                decision_aggregation,
                train_mask,
                test_mask,
            )
            per_fold.append(fold_result)
            subject_records.extend(records)
            if on_fold_end is not None:
                on_fold_end(len(per_fold) - 1, fold_result, records)

    baccs = np.array([f.balanced_acc for f in per_fold], dtype=float)
    baccs = baccs[np.isfinite(baccs)]
    aucs = np.array([f.auc for f in per_fold], dtype=float)
    aucs = aucs[np.isfinite(aucs)]

    # 按「猜测单元」（group key = 被试或 (subject, run)）聚合命中率，而非按 fold 等权
    # （D-hit-by-unit）。fold 等权在 within-subject 下会让多 run 被试权重虚高。
    unit_hits = np.array([pred == true for (pred, true, _) in subject_records], dtype=float)

    return EvalSummary(
        hit_rate_mean=float(unit_hits.mean()) if unit_hits.size else 0.0,
        hit_rate_std=float(unit_hits.std()) if unit_hits.size else 0.0,
        balanced_acc_mean=float(baccs.mean()) if baccs.size else 0.0,
        auc_mean=float(aucs.mean()) if aucs.size else float("nan"),
        per_fold=per_fold,
        subject_records=subject_records,
    )


# ---------------- 配对置换检验 ----------------


def paired_permutation_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_perm: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    """配对置换检验（D-perm-paired）：两模型逐单元分数的差异是否显著。

    Parameters
    ----------
    scores_a / scores_b : (n_units,) 每单元（被试/fold）的分数（命中 0/1 或 z-score）。
    n_perm : 置换次数（默认 10000）。
    seed : 随机种子（可复现）。

    Returns
    -------
    (obs_diff, p_value)：观察到的平均差异 a−b，与双侧置换 p 值。
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"scores_a/b 须为等长一维数组，得到 {a.shape}/{b.shape}。")
    if a.size == 0:
        raise ValueError("scores_a/b 不能为空数组。")
    if n_perm < 1:
        raise ValueError(f"n_perm 须 ≥1，得到 {n_perm}。")

    d = a - b
    d_obs = float(d.mean())
    rng = np.random.default_rng(seed)
    abs_obs = abs(d_obs)

    # 向量化置换（D-perm-vec）：一次生成 chunk 个符号矩阵再按行求均值，
    # 比逐次 rng.choice + Python 循环快 1~2 个数量级；chunk 限制峰值内存。
    count = 0
    done = 0
    chunk = max(1, min(n_perm, 1_000_000 // max(1, d.size)))
    while done < n_perm:
        take = min(chunk, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(take, d.size))
        perms = (d[None, :] * signs).mean(axis=1)
        count += int(np.count_nonzero(np.abs(perms) >= abs_obs))
        done += take

    p = (count + 1) / (n_perm + 1)  # +1 校正，避免 p=0
    return d_obs, p


# ---------------- 二分类评估（无猜数字口径） ----------------


@dataclass
class BinaryFoldResult:
    """二分类单折结果。"""

    balanced_acc: float
    auc: float
    n_test_trials: int


@dataclass
class BinarySummary:
    """二分类多折汇总（无命中率口径，D-hit-vs-bacc）。"""

    balanced_acc_mean: float
    balanced_acc_std: float
    auc_mean: float
    per_fold: list[BinaryFoldResult] = field(default_factory=list)


def _evaluate_one_binary_fold(
    model,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> BinaryFoldResult:
    """执行单个二分类 fold（与 evaluate_binary 串行路径完全同构）。"""
    model.fit(X[train_mask], y[train_mask])
    logits = model.predict_logit(X[test_mask])

    if not np.isfinite(logits).all():
        raise ValueError("模型 predict_logit 输出含 NaN/inf，无法进入 bacc/AUC。")

    y_test = y[test_mask]
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        # 与 evaluate 同一口径：中心化 logit 后再用 0 阈值（review v6 P0-1）
        logits_centered = logits - logits.mean()
        bacc = float(balanced_accuracy_score(y_test, (logits_centered > 0).astype(int)))
    else:
        bacc = np.nan  # 单类测试集无法定义 balanced accuracy（audit P2-2）
    auc = np.nan
    if n_classes == 2:  # D-auc-guard
        auc = float(roc_auc_score(y_test, logits))

    return BinaryFoldResult(
        balanced_acc=bacc,
        auc=auc,
        n_test_trials=int(test_mask.sum()),
    )


def _run_binary_fold_threaded(
    args: tuple[int, object, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[int, BinaryFoldResult]:
    """线程池 worker：深拷贝未拟合模型，BLAS 限单线程，避免并行 fold 互相抢占。"""
    fold_idx, model_proto, X, y, subject_ids, train_mask, test_mask = args
    model = copy.deepcopy(model_proto)
    with _fold_threadpool_limits():
        result = _evaluate_one_binary_fold(
            model, X, y, subject_ids, train_mask, test_mask
        )
    return fold_idx, result


def evaluate_binary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    n_jobs: int = 1,
) -> BinarySummary:
    """二分类评估（LOSO / within-subject）→ balanced_acc + AUC。

    适用于**无「猜数字」ground truth** 的数据集（BNCI2014_008 / Brain Invaders / ERP CORE）：
    它们只有 target/non-target 二分类标签，无「心选数字」可做 9 选 1 命中率。这是 P300 检测的
    通用口径（单试次判别质量），与 :func:evaluate（猜数字 + 二分类）互补——后者专供 GTN
    （有 thought_number ground truth）。

    三思决策记录：
        D-binary-vs-gtn 二分类与猜数字是**两个不同层级**的口径：二分类报「单试次判别质量」
                        （balanced_acc / AUC），猜数字报「决策层命中率」（9 选 1，chance 11.1%）。
                        BNCI/Brain Invaders/ERP CORE 只有前者；GTN 两者都有。故拆两个函数，
                        而非在 evaluate 里用 None 占位——语义更清晰、不硬凑 digits。
        D-binary-noguard 沿用 D-nan-guard：X 含 NaN 显式报错（classic/riemann 用子集、
                        deep 零填充），杜绝静默失真。
        D-binary-threads 与 evaluate 相同的线程池 fold 并行：每个 worker 深拷贝未拟合模型，
                        BLAS 限单线程；X/y 只读共享，不复制大数据。

    Parameters
    ----------
    model : 有 fit(X,y) / predict_logit(X)→(N,) 接口的分类器（Baseline）。
    X / y : (N,C,T) / (N,) 试次数据与 target/non-target 标签（y ∈ {0,1}）。
    subject_ids : (N,) 每个试次所属被试（LOSO 分组用）。
    folds : list[(train_mask, test_mask)] 布尔掩码。
    n_jobs : int
        并行 fold 的线程数（默认 1 串行）。各 fold 独立；线程池共享 X/y 内存。

    Returns
    -------
    BinarySummary：balanced_acc 均值/标准差 + AUC 均值 + 逐折。
    """
    X = np.asarray(X)
    y = np.asarray(y)
    subject_ids = np.asarray(subject_ids)
    # NaN 守卫（D-binary-noguard，同 D-nan-guard）
    if np.isnan(X).any():
        raise ValueError(
            "X 含 NaN：缺失通道须先处理——classic/riemann 用 subset_channels 提取子集，"
            "deep 零填充到 8 导。"
        )

    mask_folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_mask, test_mask in folds:
        train_mask = np.asarray(train_mask, dtype=bool)
        test_mask = np.asarray(test_mask, dtype=bool)
        if not train_mask.any():
            raise ValueError("train_mask 为空：无法训练。")
        if not test_mask.any():
            continue
        mask_folds.append((train_mask, test_mask))

    per_fold: list[BinaryFoldResult] = []
    if n_jobs > 1 and len(mask_folds) > 1:
        task_args = [
            (fold_idx, model, X, y, subject_ids, train_mask, test_mask)
            for fold_idx, (train_mask, test_mask) in enumerate(mask_folds)
        ]
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(_run_binary_fold_threaded, task_args))
        per_fold = [result for _, result in sorted(results, key=lambda item: item[0])]
    else:
        for train_mask, test_mask in mask_folds:
            per_fold.append(
                _evaluate_one_binary_fold(
                    model, X, y, subject_ids, train_mask, test_mask
                )
            )

    baccs = np.array([f.balanced_acc for f in per_fold], dtype=float)
    baccs = baccs[np.isfinite(baccs)]
    aucs = np.array([f.auc for f in per_fold], dtype=float)
    aucs = aucs[np.isfinite(aucs)]

    return BinarySummary(
        balanced_acc_mean=float(baccs.mean()) if baccs.size else 0.0,
        balanced_acc_std=float(baccs.std()) if baccs.size else 0.0,
        auc_mean=float(aucs.mean()) if aucs.size else float("nan"),
        per_fold=per_fold,
    )
