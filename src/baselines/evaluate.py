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
    D-bacc-threshold balanced_acc 的阈值只由训练侧被试学习（优先内部 subject-disjoint validation）；
                    测试 fold logit 中心化只保留为显式 transductive 对照。AUC 仍用原始 logit。
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
      D-parallel-fold  LOSO fold 完全独立。n_jobs>1 使用 spawn 多进程，避免 fork 继承 PyTorch/CUDA
                       线程锁并在进入首个 epoch 前死锁。每个 worker 建立独立 CUDA context，并用
                       fold CPU 预算和 PyTorch 线程限制避免 oversubscription。deep 模型在
                       confirmatory 模式固定 n_jobs=1（避免显存争用）。

契约（输入 → 输出）：
    loso_folds(subject_ids) / within_subject_folds(subject_ids, run_ids) → list[(train_mask, test_mask)]
    evaluate(model, X, y, digits, subject_ids, true_digits, folds) → EvalSummary
    paired_permutation_test(scores_a, scores_b, n_perm) → (obs_diff, p_value)

依赖的决策：roadmap Phase 1、constitution D3/P8、models/decision.decide、baselines.classic.Baseline。
"""

from __future__ import annotations

import copy
import ctypes
import json
import multiprocessing as mp
import os
import signal
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

try:  # threadpoolctl 是 sklearn 依赖；无它时并行路径退化为无 BLAS 线程限制
    from threadpoolctl import threadpool_limits
except Exception:  # noqa: BLE001
    threadpool_limits = None

from baselines.calibration import calibration_data_from_model, fit_logit_calibration
from baselines.evidence_protocol import (
    EvidenceBudget,
    build_evidence_budgets,
    evidence_row_indices,
    row_acquisition_indices,
    validate_outer_folds,
)
from baselines.experiment_protocol import validate_decision_metric_name
from baselines.repetition_metrics import (
    RepetitionEfficiencySummary,
    summarize_repetition_efficiency,
)
from baselines.validation import subject_disjoint_validation_split
from data.events import ScheduledEventTimeline
from models.decision import decide

# ---------------- fold 生成 ----------------


def loso_folds(subject_ids: Sequence) -> list[tuple[np.ndarray, np.ndarray]]:
    """留一被试（leave-one-subject-out）：每个被试依次作 test、其余作 train。

    Returns
    -------
    list[(train_mask, test_mask)]，每元布尔掩码长度 = len(subject_ids)。
    """
    subject_ids = np.asarray(subject_ids)
    if subject_ids.ndim != 1 or len(subject_ids) == 0:
        raise ValueError("subject_ids must be a non-empty one-dimensional sequence.")
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
    if subject_ids.ndim != 1 or run_ids.ndim != 1 or len(subject_ids) != len(run_ids):
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
    threshold: float = 0.0
    threshold_source: str = "unknown"
    transductive_balanced_acc: float = float("nan")
    audit: dict = field(default_factory=dict)
    fit_sec: float | None = None
    fit_peak_memory_mb: float | None = None
    epochs_ran: int | None = None
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_objective_losses: list[float] = field(default_factory=list)
    val_innovation_nlls: list[float] = field(default_factory=list)
    task_val_aucs: list[float | None] = field(default_factory=list)
    final_task_val_auc: float | None = None
    best_task_epoch: int | None = None
    best_density_epoch: int | None = None
    best_task_val_loss: float | None = None
    best_density_nll: float | None = None
    task_patience_exhausted: bool = False
    epoch_trajectory_audit: list[dict[str, object]] = field(default_factory=list)
    training_history: dict[str, object] = field(default_factory=dict)
    val_subjects: int | None = None
    audit_subjects: int | None = None
    erp_calibration: dict[str, object] | None = None
    component_summary: dict[str, object] = field(default_factory=dict)
    descriptive_decision_records: dict[str, list[dict[str, object]]] = field(default_factory=dict)


@dataclass
class DecisionMetric:
    """Subject-level GTN hit rate at one fixed evidence budget."""

    name: str
    evidence_budget: int | str
    aggregation: str
    hit_rate: float
    conditional_hit_rate: float
    n_covered: int
    n_total: int
    coverage: float
    budget_semantics: str
    subject_records: list[tuple[object, object, object]] = field(default_factory=list)


@dataclass
class ConfoundBaseline:
    name: str
    hit_rate: float
    n_subjects: int
    subject_records: list[tuple[object, object, object]] = field(default_factory=list)


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
    transductive_balanced_acc_mean: float = float("nan")
    decision_metrics: dict[str, DecisionMetric] = field(default_factory=dict)
    confound_baselines: dict[str, ConfoundBaseline] = field(default_factory=dict)
    primary_decision_metric: str = "exact_llr@3"
    primary_metric_gate: dict[str, object] = field(default_factory=dict)
    descriptive_decision_records: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    repetition_efficiency: RepetitionEfficiencySummary | None = None
    evaluation_units: tuple[str, ...] = ()
    cohort_sha256: str = ""
    dataset_sha256: str | None = None
    fold_protocol: str = "loso"

    @property
    def primary_hit_rate(self) -> float:
        metric = self.decision_metrics.get(self.primary_decision_metric)
        return metric.hit_rate if metric is not None else self.hit_rate_mean


def _budget_key(budget: int | None) -> str:
    return "all" if budget is None else str(int(budget))


def _metric_name(aggregation: str, budget: EvidenceBudget) -> str:
    if budget.kind == "all":
        return f"all_{aggregation}"
    if budget.kind == "time":
        return f"time_{aggregation}@{float(budget.value):g}s"
    return f"{budget.kind}_{aggregation}@{int(budget.value)}"


def _metric_metadata(name: str) -> tuple[str, int | float | str, str]:
    if name.startswith("all_"):
        return name.removeprefix("all_"), "all", "all_available"
    prefix, value = name.split("@", maxsplit=1)
    for semantics in ("prefix_minK", "exact", "flash", "time"):
        marker = f"{semantics}_"
        if prefix.startswith(marker):
            aggregation = prefix.removeprefix(marker)
            if semantics == "time":
                return aggregation, float(value.removesuffix("s")), semantics
            return aggregation, int(value), semantics
    raise ValueError(f"Unrecognized decision metric name {name!r}.")


def _complete_records(
    records: Sequence[tuple[object, object, object]],
    true_digits: Mapping[str, object],
    units: Sequence[str],
) -> list[tuple[object, object, object]]:
    frozen_units = {str(unit) for unit in units}
    predictions: dict[str, object] = {}
    for predicted, true, group in records:
        key = str(group)
        if key in predictions:
            raise ValueError(f"Duplicate decision record for evaluation unit {key!r}.")
        if key not in frozen_units:
            raise ValueError(f"Decision record contains non-frozen evaluation unit {key!r}.")
        if key not in true_digits or true_digits[key] != true:
            raise ValueError(f"Decision record truth mismatch for evaluation unit {key!r}.")
        predictions[key] = predicted
    return [(predictions.get(unit), true_digits[unit], unit) for unit in units]


def _build_primary_metric_gate(
    primary_metric: DecisionMetric,
    per_fold: Sequence[FoldResult],
    *,
    primary_min_coverage: float,
) -> dict[str, object]:
    """Build the non-blocking, audit-ready primary-metric claim gate.

    The gate never deletes the metric and never aborts the run.  For a chain
    primary it records the formal availability check and the per-fold
    repetition-readiness evidence used to decide formal coverage.  A failed
    gate downgrades the result to ``claim_eligible=false`` (descriptive only).
    """

    applicable = primary_metric.aggregation == "chain_llr"
    observed_coverage = float(primary_metric.coverage)
    minimum_coverage = float(primary_min_coverage)
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("primary_min_coverage must be in (0,1].")
    coverage_passed = observed_coverage >= minimum_coverage

    checks: dict[str, dict[str, object]] = {}
    if applicable:
        checks["minimum_availability_coverage"] = {
            "passed": coverage_passed,
            "observed": observed_coverage,
            "minimum": minimum_coverage,
            "n_covered": int(primary_metric.n_covered),
            "n_total": int(primary_metric.n_total),
        }
    passed = bool(not applicable or coverage_passed)
    failed_checks = ["minimum_availability_coverage"] if applicable and not coverage_passed else []

    attempted_folds = 0
    ready_folds = 0
    unready_folds: list[dict[str, object]] = []
    for fold_index, fold in enumerate(per_fold):
        repetition = (fold.audit or {}).get("repetition")
        if repetition is None:
            continue
        attempted_folds += 1
        if bool(repetition.get("ready", False)):
            ready_folds += 1
            continue
        reliability_gate = repetition.get("reliability_gate") or {}
        unready_folds.append(
            {
                "fold_index": int(fold_index),
                "ready": False,
                "reliability_gate_passed": bool(reliability_gate.get("passed", False)),
                "failure": reliability_gate.get("failure"),
                "failed_checks": list(reliability_gate.get("failed_checks", []) or []),
            }
        )

    return {
        "name": "primary_metric_claim_gate",
        "applicable": applicable,
        "passed": passed,
        "claim_eligible": passed,
        "effect": "descriptive_only_no_result_suppression",
        "checks": checks,
        "failed_checks": failed_checks,
        "observed_coverage": observed_coverage,
        "minimum_coverage": minimum_coverage,
        "n_covered": int(primary_metric.n_covered),
        "n_total": int(primary_metric.n_total),
        "repetition_fold_readiness": {
            "attempted_folds": attempted_folds,
            "ready_folds": ready_folds,
            "unready_folds": unready_folds,
        },
    }



def _first_k_occurrences_per_digit(
    digits: np.ndarray,
    subject_ids: np.ndarray,
    digit_vocab: Sequence[int],
    budget: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return chronological equal-evidence rows and subjects with full coverage."""

    mask = np.zeros(len(digits), dtype=bool)
    covered = []
    for subject in np.unique(subject_ids):
        subject_rows = np.flatnonzero(subject_ids == subject)
        complete = True
        checkpoints = []
        for digit in digit_vocab:
            rows = subject_rows[digits[subject_rows] == digit]
            required = 1 if budget is None else int(budget)
            if len(rows) < required:
                complete = False
                break
        if complete:
            if budget is None:
                selected = subject_rows
            else:
                for digit in digit_vocab:
                    positions = np.flatnonzero(digits[subject_rows] == digit)
                    checkpoints.append(int(positions[int(budget) - 1]))
                selected = subject_rows[: max(checkpoints) + 1]
            mask[np.asarray(selected, dtype=int)] = True
            covered.append(subject)
    return mask, np.asarray(covered, dtype=object)


def _records_from_decision(result, true_digits: Mapping) -> list[tuple[object, object, object]]:
    records = []
    for predicted, subject in zip(result.predicted, result.subject_ids, strict=True):
        true_digit = true_digits.get(subject)
        if true_digit is not None:
            records.append((predicted, true_digit, subject))
    return records


def _gtn_decision_variants(
    logits: np.ndarray,
    llr: np.ndarray,
    digits: np.ndarray,
    subject_ids: np.ndarray,
    true_digits: Mapping,
    digit_vocab: Sequence[int],
    budgets: Sequence[EvidenceBudget],
    event_timeline: ScheduledEventTimeline,
    global_test_rows: np.ndarray,
    evaluation_units: Sequence[str],
) -> tuple[dict[str, list[tuple[object, object, object]]], dict[str, int], int]:
    records_by_name: dict[str, list[tuple[object, object, object]]] = {}
    covered_by_name: dict[str, int] = {}
    units = tuple(str(unit) for unit in evaluation_units)
    n_total = len(units)
    global_to_local = {int(row): index for index, row in enumerate(global_test_rows)}
    for budget in budgets:
        selected_local: list[int] = []
        selected_groups: list[str] = []
        for unit in units:
            selected_global = evidence_row_indices(event_timeline, unit, digit_vocab, budget)
            if selected_global is None:
                continue
            if any(int(row) not in global_to_local for row in selected_global):
                raise ValueError(
                    f"Budget {budget.token} selected evidence outside unit {unit!r}'s test fold."
                )
            selected_local.extend(global_to_local[int(row)] for row in selected_global)
            selected_groups.extend([unit] * len(selected_global))
        for aggregation in ("sum", "mean"):
            name = _metric_name(aggregation, budget)
            rows: list[tuple[object, object, object]] = []
            if selected_local:
                local = np.asarray(selected_local, dtype=np.int64)
                result = decide(
                    logits[local],
                    digits[local],
                    np.asarray(selected_groups),
                    digit_vocab,
                    center_logits=True,
                    aggregation=aggregation,
                )
                rows = _records_from_decision(result, true_digits)
            records_by_name[name] = _complete_records(rows, true_digits, units)
            covered_by_name[name] = sum(
                predicted is not None for predicted, _, _ in records_by_name[name]
            )
        llr_name = _metric_name("llr", budget)
        llr_rows: list[tuple[object, object, object]] = []
        if selected_local:
            local = np.asarray(selected_local, dtype=np.int64)
            llr_result = decide(
                llr[local],
                digits[local],
                np.asarray(selected_groups),
                digit_vocab,
                center_logits=False,
                aggregation="sum",
            )
            llr_rows = _records_from_decision(llr_result, true_digits)
        records_by_name[llr_name] = _complete_records(llr_rows, true_digits, units)
        covered_by_name[llr_name] = sum(
            predicted is not None for predicted, _, _ in records_by_name[llr_name]
        )
    return records_by_name, covered_by_name, n_total


def _expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(len(labels), 1)
    ece = 0.0
    for idx in range(n_bins):
        upper = (
            probabilities <= edges[idx + 1] if idx == n_bins - 1 else probabilities < edges[idx + 1]
        )
        mask = (probabilities >= edges[idx]) & upper
        if mask.any():
            ece += mask.sum() / total * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(ece)


def _neural_ride_fold_audit(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    digits_test: np.ndarray,
    subject_ids_test: np.ndarray,
    true_digits: Mapping,
    digit_vocab: Sequence[int],
    event_timeline: ScheduledEventTimeline,
    global_test_rows: np.ndarray,
    trial_channel_mask_test: np.ndarray | None = None,
) -> dict:
    """Compute calibrated PCW, prequential and localization diagnostics."""

    predict_branches = getattr(model, "predict_branches", None)
    if not callable(predict_branches):
        return {}
    auxiliary_kwargs = {}
    if (
        trial_channel_mask_test is not None
        and getattr(model, "auxiliary_predict_accepts_trial_channel_mask", False)
    ):
        auxiliary_kwargs["trial_channel_mask"] = trial_channel_mask_test
    test_branches = predict_branches(X_test, **auxiliary_kwargs)
    calibration_branches = getattr(model, "calibration_branch_logits_", None)
    calibration_labels = getattr(model, "calibration_labels_", None)
    calibration_source = getattr(model, "calibration_source_", None)
    if calibration_branches is None or calibration_labels is None:
        raise ValueError(
            "Branch calibration requires subject-disjoint validation scores; "
            "training-set resubstitution calibration is forbidden."
        )
    calibration_labels = np.asarray(calibration_labels).astype(int)
    prior = float(np.clip(calibration_labels.mean(), 1e-6, 1.0 - 1e-6))
    prior_log_odds = float(np.log(prior / (1.0 - prior)))

    branch_metrics: dict[str, dict] = {}
    branch_names = ["final", "pcw"]
    for optional in ("prequential_llr", "prequential_contribution"):
        if (
            optional in calibration_branches
            and optional in test_branches
            and np.std(np.asarray(calibration_branches[optional], dtype=float)) > 0.0
        ):
            branch_names.append(optional)
    for name in branch_names:
        train_scores = np.asarray(calibration_branches[name], dtype=float)
        test_scores = np.asarray(test_branches[name], dtype=float)
        calibration = fit_logit_calibration(
            train_scores,
            calibration_labels,
            source=str(calibration_source or "model_validation"),
        )
        probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(calibration.to_llr(test_scores) + prior_log_odds, -40, 40))
        )
        auc = (
            float(roc_auc_score(y_test, test_scores))
            if len(np.unique(y_test)) == 2
            else float("nan")
        )
        bacc = (
            float(
                balanced_accuracy_score(y_test, (test_scores >= calibration.threshold).astype(int))
            )
            if len(np.unique(y_test)) == 2
            else float("nan")
        )
        records_by_name, _, n_total = _gtn_decision_variants(
            test_scores,
            calibration.to_llr(test_scores),
            digits_test,
            subject_ids_test,
            true_digits,
            digit_vocab,
            (EvidenceBudget("exact", 15),),
            event_timeline,
            global_test_rows,
            tuple(np.unique(subject_ids_test).astype(str).tolist()),
        )
        records = records_by_name["exact_llr@15"]
        covered_records = [row for row in records if row[0] is not None]
        hits = [predicted == true for predicted, true, _ in covered_records]
        branch_metrics[name] = {
            "auc": auc,
            "balanced_acc": bacc,
            "brier": float(np.mean((probabilities - y_test) ** 2)),
            "ece10": _expected_calibration_error(probabilities, y_test),
            "llr_at_15_hit_rate": float(np.mean(hits)) if hits else float("nan"),
            "llr_at_15_covered": len(covered_records),
            "llr_at_15_total": n_total,
            "threshold": float(calibration.threshold),
            "calibration_source": calibration.source,
        }

    localization: dict = {}
    component_summary: dict = {}
    predict_interpretability = getattr(model, "predict_interpretability", None)
    if callable(predict_interpretability):
        values = predict_interpretability(X_test, **auxiliary_kwargs)
        tau = np.asarray(values["tau"])
        bounds = np.asarray(values["tau_bounds"])
        span = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-6)
        saturation = ((tau - bounds[None, :, 0]) <= 0.02 * span[None]) | (
            (bounds[None, :, 1] - tau) <= 0.02 * span[None]
        )
        localization = {
            "tau_mean_ms": tau.mean(axis=0).tolist(),
            "tau_std_ms": tau.std(axis=0).tolist(),
            "tau_target_mean_ms": tau[y_test == 1].mean(axis=0).tolist(),
            "tau_nontarget_mean_ms": tau[y_test == 0].mean(axis=0).tolist(),
            "tau_target_std_ms": tau[y_test == 1].std(axis=0).tolist(),
            "tau_nontarget_std_ms": tau[y_test == 0].std(axis=0).tolist(),
            "tau_bound_saturation_fraction": saturation.mean(axis=0).tolist(),
            "tau_noise_median_abs_shift_ms": np.median(
                np.abs(tau - np.asarray(values["tau_perturbed"])), axis=0
            ).tolist(),
        }
        if "amplitude_variance" in values:
            localization["amplitude_variance_mean"] = (
                np.asarray(values["amplitude_variance"]).mean(axis=(0, 2)).tolist()
            )
            localization["erp_energy_ratio_mean"] = float(np.mean(values["erp_energy_ratio"]))
        fitted_model = getattr(model, "model_", None)
        component_window = getattr(fitted_model, "component_window", None)
        if component_window is not None:
            import torch

            sigma = component_window.sigma_lo[:, None] + (
                component_window.sigma_hi[:, None] - component_window.sigma_lo[:, None]
            ) * torch.sigmoid(component_window.sigma_raw)
            component_summary = {
                "tau0_bounded_ms": component_window.tau0_bounded.detach().cpu().tolist(),
                "sigma_ms": sigma.detach().cpu().tolist(),
                "n_test_trials": int(len(y_test)),
                "target_n": int((y_test == 1).sum()),
                "nontarget_n": int((y_test == 0).sum()),
                "tau_target_mean_ms": localization["tau_target_mean_ms"],
                "tau_target_std_ms": localization["tau_target_std_ms"],
                "tau_nontarget_mean_ms": localization["tau_nontarget_mean_ms"],
                "tau_nontarget_std_ms": localization["tau_nontarget_std_ms"],
                "branch_audit": {
                    "final_mean": float(np.mean(test_branches["final"])),
                    "final_std": float(np.std(test_branches["final"])),
                    "pcw_mean": float(np.mean(test_branches["pcw"])),
                    "pcw_std": float(np.std(test_branches["pcw"])),
                    "prequential_llr_mean": float(np.mean(test_branches["prequential_llr"])),
                    "prequential_llr_std": float(np.std(test_branches["prequential_llr"])),
                    "prequential_coefficient": float(
                        test_branches["prequential_coefficient"]
                    ),
                    "fusion_identity_max_abs_error": float(
                        np.max(
                            np.abs(
                                test_branches["final"]
                                - (
                                    float(test_branches.get("prequential_base_intercept", 0.0))
                                    + float(test_branches.get("prequential_base_slope", 1.0))
                                    * (
                                        test_branches["pcw"]
                                        + test_branches.get(
                                            "measurement_contribution",
                                            np.zeros_like(test_branches["pcw"]),
                                        )
                                    )
                                    + test_branches["prequential_contribution"]
                                )
                            )
                        )
                    ),
                },
            }

    gradients: dict = {}
    history = getattr(model, "last_history", None) or {}
    prequential = history.get("prequential_audit", {})
    prequential_fusion = history.get("prequential_fusion", {})
    diagnostics = history.get("pcw_gradient_diagnostics", {})
    for key, series in diagnostics.items():
        if key.endswith("_norms") and series:
            gradients[f"{key}_mean"] = float(np.mean(series))

    measurement_gate = history.get(
        "measurement_gate", getattr(model, "measurement_gate_", None)
    )
    state_gate = history.get("repetition_state_residual_gate", None)
    reliability_audit = history.get(
        "repetition_reliability_audit",
        getattr(model, "repetition_reliability_audit_", None),
    )
    stopping_replay = history.get("validation_stopping_replay", None)
    v12_gates = {
        "measurement": measurement_gate,
        "repetition_state_residual": state_gate,
        "reliability_fidelity": (
            reliability_audit.get("fidelity") if isinstance(reliability_audit, dict) else None
        ),
        "reliability_clean_probability": (
            reliability_audit.get("clean_probability")
            if isinstance(reliability_audit, dict)
            else None
        ),
        "stopping_replay": stopping_replay,
    }

    measured_latency: dict[str, object] | None = None
    posterior = getattr(model, "measurement_posterior_", None)
    predict_measurement = getattr(model, "predict_measurement", None)
    if callable(predict_measurement):
        posterior = predict_measurement(X_test, **auxiliary_kwargs) or posterior
    if posterior is not None and getattr(posterior, "mean_ms", None) is not None:
        width_ms = posterior.upper_ms - posterior.lower_ms
        measured_latency = {
            "n_trials": int(posterior.effective_n),
            "mean_ms_mean": float(np.mean(posterior.mean_ms)),
            "mean_ms_std": float(np.std(posterior.mean_ms)),
            "interval_width_ms_mean": float(np.mean(width_ms)),
            "interval_width_ms_median": float(np.median(width_ms)),
            "entropy_nats_mean": float(np.mean(posterior.entropy)),
            "gate": measurement_gate,
        }
    return {
        "branches": branch_metrics,
        "prequential_gate": prequential,
        "prequential_fusion": prequential_fusion,
        "prequential_coefficient": float(getattr(model, "prequential_coefficient_", 0.0)),
        "localization": localization,
        "pcw_routing": localization,
        "measured_latency": measured_latency,
        "v12_gates": v12_gates,
        "component_summary": component_summary,
        "gradients": gradients,
    }


def _confound_records(
    digits: np.ndarray,
    subject_ids: np.ndarray,
    true_digits: Mapping,
    digit_vocab: Sequence[int],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, list[tuple[object, object, object]]]:
    train_subjects = np.unique(subject_ids[train_mask])
    train_truth = [true_digits[s] for s in train_subjects if true_digits.get(s) is not None]
    prior_prediction = None
    if train_truth:
        prior_counts = np.array([sum(v == d for v in train_truth) for d in digit_vocab])
        prior_prediction = digit_vocab[int(np.argmax(prior_counts))]

    prior_records = []
    count_records = []
    for subject in np.unique(subject_ids[test_mask]):
        truth = true_digits.get(subject)
        if truth is None:
            continue
        if prior_prediction is not None:
            prior_records.append((prior_prediction, truth, subject))
        subject_digits = digits[test_mask][subject_ids[test_mask] == subject]
        counts = np.array([np.count_nonzero(subject_digits == d) for d in digit_vocab])
        count_prediction = digit_vocab[int(np.argmax(counts))]
        count_records.append((count_prediction, truth, subject))
    return {"digit_prior": prior_records, "count_only": count_records}


@contextmanager
def _cpu_threadpool_limits(cpu_threads: int) -> Iterator[None]:
    """Apply one CPU budget to BLAS and PyTorch intra-op threads."""

    if cpu_threads < 1:
        raise ValueError(f"cpu_threads must be positive, got {cpu_threads}")
    try:
        import torch
    except ImportError:
        if threadpool_limits is None:
            yield
        else:
            with threadpool_limits(limits=cpu_threads):
                yield
        return

    previous = torch.get_num_threads()
    torch.set_num_threads(cpu_threads)
    try:
        if threadpool_limits is None:
            yield
        else:
            with threadpool_limits(limits=cpu_threads):
                yield
    finally:
        torch.set_num_threads(previous)


@contextmanager
def _fold_threadpool_limits() -> Iterator[None]:
    """Apply the per-fold CPU budget to BLAS and PyTorch intra-op threads.

    Native thread-pool limits are process-global. Thread-based fold execution
    must therefore enter this context once around the executor, while spawned
    process workers and serial folds may enter it around one fold. PyTorch
    inter-op threads are configured once in spawned-worker initialization;
    unlike intra-op threads, they cannot be changed and restored repeatedly.
    """
    with _cpu_threadpool_limits(_fold_cpu_threads()):
        yield


def _fold_cpu_threads() -> int:
    """Return the CPU thread budget for one parallel fold worker."""

    raw = os.environ.get("FOLD_CPU_THREADS", "2")
    try:
        threads = int(raw)
    except ValueError as exc:
        raise ValueError(f"FOLD_CPU_THREADS must be a positive integer, got {raw!r}") from exc
    if threads < 1:
        raise ValueError(f"FOLD_CPU_THREADS must be positive, got {threads}")
    return threads


def _fit_model_with_optional_subjects(
    model,
    X,
    y,
    subject_ids,
    group_ids,
    train_mask,
    digits: np.ndarray | None = None,
    acquisition_indices: np.ndarray | None = None,
    trial_channel_mask: np.ndarray | None = None,
) -> None:
    """按模型能力调用 fit（GLM：被试级验证早停协议）。

    声明 ``fit_accepts_subject_ids = True`` 的模型（如 N2P3NetBaseline）会收到
    ``subject_ids``，可在 fit 内按被试分组切验证集做早停（同被试试次不跨 train/val，
    杜绝试次级随机切分的同被试泄漏高估）；未声明的模型走旧 fit(X, y) 契约，完全向后兼容。
    """
    for name in (
        "calibration_logits_",
        "calibration_labels_",
        "calibration_source_",
        "calibration_branch_logits_",
        "repetition_temperature_calibration_",
    ):
        try:
            setattr(model, name, None)
        except (AttributeError, TypeError):
            pass
    if getattr(model, "fit_accepts_trial_context", False):
        fit_kwargs = {
            "subject_ids": subject_ids[train_mask],
            "digits": digits[train_mask] if digits is not None else None,
        }
        if getattr(model, "fit_accepts_group_ids", False):
            fit_kwargs["group_ids"] = group_ids[train_mask]
        if getattr(model, "fit_accepts_acquisition_indices", False):
            if acquisition_indices is None:
                raise ValueError(
                    "Model requires scheduled acquisition indices, but none were supplied."
                )
            fit_kwargs["acquisition_indices"] = acquisition_indices[train_mask]
        if getattr(model, "fit_accepts_trial_channel_mask", False):
            if trial_channel_mask is not None:
                fit_kwargs["trial_channel_mask"] = trial_channel_mask[train_mask]
        model.fit(X[train_mask], y[train_mask], **fit_kwargs)
    elif getattr(model, "fit_accepts_subject_ids", False):
        fit_kwargs = {"subject_ids": subject_ids[train_mask]}
        if getattr(model, "fit_accepts_trial_channel_mask", False):
            if trial_channel_mask is not None:
                fit_kwargs["trial_channel_mask"] = trial_channel_mask[train_mask]
        model.fit(X[train_mask], y[train_mask], **fit_kwargs)
    else:
        outer_subjects = subject_ids[train_mask]
        split = subject_disjoint_validation_split(
            outer_subjects,
            fraction=0.08,
            min_subjects=2,
            max_subjects=12,
            seed=0,
        )
        outer_X, outer_y = X[train_mask], y[train_mask]
        outer_trial_channel_mask = (
            trial_channel_mask[train_mask] if trial_channel_mask is not None else None
        )
        fit_kwargs = {}
        if getattr(model, "fit_accepts_trial_channel_mask", False):
            if outer_trial_channel_mask is not None:
                fit_kwargs["trial_channel_mask"] = outer_trial_channel_mask[split.train_mask]
        model.fit(outer_X[split.train_mask], outer_y[split.train_mask], **fit_kwargs)
        if split.n_validation_subjects > 0:
            model.calibration_logits_ = _predict_logit_with_optional_trial_mask(
                model,
                outer_X[split.validation_mask],
                (
                    outer_trial_channel_mask[split.validation_mask]
                    if outer_trial_channel_mask is not None
                    else None
                ),
            )
            model.calibration_labels_ = outer_y[split.validation_mask].copy()
            model.calibration_source_ = "subject_disjoint_validation"


def _predict_logit_with_optional_trial_mask(
    model,
    X: np.ndarray,
    trial_channel_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Call prediction through the common capability-based adapter interface."""

    predict_kwargs = {}
    if getattr(model, "predict_accepts_trial_channel_mask", False):
        if trial_channel_mask is not None:
            predict_kwargs["trial_channel_mask"] = trial_channel_mask
    logits = np.asarray(model.predict_logit(X, **predict_kwargs))
    if logits.shape != (len(X),):
        raise ValueError(
            f"predict_logit must return shape ({len(X)},), got {logits.shape}."
        )
    return logits


def _evaluate_epoch_trajectory_audit(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    digits_test: np.ndarray,
    group_ids_test: np.ndarray,
    true_digits: Mapping,
    digit_vocab: Sequence[int],
    decision_center: bool,
    decision_aggregation: str,
    evidence_budgets: Sequence[EvidenceBudget],
    event_timeline: ScheduledEventTimeline,
    global_test_rows: np.ndarray,
    trial_channel_mask_test: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Score raw epoch states on outer test data for development diagnostics only."""

    predictor = getattr(model, "predict_epoch_trajectory_logits", None)
    if not callable(predictor):
        return []
    predictor_kwargs = {}
    if (
        trial_channel_mask_test is not None
        and getattr(model, "auxiliary_predict_accepts_trial_channel_mask", False)
    ):
        predictor_kwargs["trial_channel_mask"] = trial_channel_mask_test
    checkpoint_rows = predictor(X_test, **predictor_kwargs)
    if not checkpoint_rows:
        return []
    units = tuple(np.unique(group_ids_test).astype(str).tolist())
    y_test = np.asarray(y_test)
    trajectory: list[dict[str, object]] = []
    prefix_budget = EvidenceBudget("prefix_minK", 3)
    has_prefix_three = prefix_budget in evidence_budgets
    for checkpoint_row in checkpoint_rows:
        logits = np.asarray(checkpoint_row["logits"], dtype=float)
        if logits.shape != (len(y_test),) or not np.isfinite(logits).all():
            raise ValueError("Epoch trajectory checkpoint produced invalid outer-test logits.")
        auc: float | None = None
        if len(np.unique(y_test)) == 2:
            auc = float(roc_auc_score(y_test, logits))
        all_result = decide(
            logits,
            digits_test,
            group_ids_test,
            digit_vocab,
            center_logits=decision_center,
            aggregation=decision_aggregation,
        )
        all_records = _complete_records(
            _records_from_decision(all_result, true_digits), true_digits, units
        )
        all_hits = sum(
            predicted is not None and predicted == true for predicted, true, _ in all_records
        )
        prefix_hit: float | None = None
        prefix_conditional_hit: float | None = None
        prefix_coverage: float | None = None
        if has_prefix_three:
            variant_records, _, _ = _gtn_decision_variants(
                logits,
                logits,
                digits_test,
                group_ids_test,
                true_digits,
                digit_vocab,
                (prefix_budget,),
                event_timeline,
                global_test_rows,
                units,
            )
            prefix_rows = variant_records[_metric_name("sum", prefix_budget)]
            prefix_covered = sum(predicted is not None for predicted, _, _ in prefix_rows)
            prefix_hits = sum(
                predicted is not None and predicted == true
                for predicted, true, _ in prefix_rows
            )
            prefix_coverage = prefix_covered / max(len(prefix_rows), 1)
            prefix_hit = prefix_hits / max(len(prefix_rows), 1)
            prefix_conditional_hit = (
                prefix_hits / prefix_covered if prefix_covered > 0 else None
            )
        trajectory.append(
            {
                **{key: value for key, value in checkpoint_row.items() if key != "logits"},
                "diagnostic_only": True,
                "prohibited_use": "outer_test_metrics_must_not_select_checkpoints",
                "test_trial_auc": auc,
                "test_all_digit_hit_rate": all_hits / max(len(all_records), 1),
                "test_n_digit_units": len(all_records),
                "test_prefix_minK_sum_at_3_hit_rate": prefix_hit,
                "test_prefix_minK_sum_at_3_conditional_hit_rate": prefix_conditional_hit,
                "test_prefix_minK_sum_at_3_coverage": prefix_coverage,
            }
        )
    checkpoint_paths = [
        Path(str(row["checkpoint"]))
        for row in trajectory
        if row.get("checkpoint") not in (None, "in_memory")
    ]
    if checkpoint_paths:
        output = checkpoint_paths[0].parent / "trajectory.json"
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, output)
    return trajectory


def _evaluate_one_fold(
    model,
    X: np.ndarray,
    y: np.ndarray,
    digits: np.ndarray,
    subject_ids: np.ndarray,
    fold_subject_ids: np.ndarray,
    true_digits: Mapping,
    digit_vocab: Sequence[int],
    decision_center: bool,
    decision_aggregation: str,
    evidence_budgets: Sequence[EvidenceBudget],
    event_timeline: ScheduledEventTimeline,
    acquisition_indices: np.ndarray,
    trial_channel_mask: np.ndarray | None,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[
    FoldResult,
    list[tuple[object, object, object]],
    dict[str, list[tuple[object, object, object]]],
    dict[str, int],
    int,
    dict[str, list[tuple[object, object, object]]],
]:
    """Execute one fold with training-only calibration and GTN decision audits."""
    if not train_mask.any():
        raise ValueError("train_mask 为空：无法训练（within-subject 单 run 会触发此错误）。")

    _fit_model_with_optional_subjects(
        model,
        X,
        y,
        fold_subject_ids,
        subject_ids,
        train_mask,
        digits=digits,
        acquisition_indices=acquisition_indices,
        trial_channel_mask=trial_channel_mask,
    )
    fit_durations = getattr(model, "fit_durations", ()) or ()
    fit_sec = float(fit_durations[-1]) if fit_durations else None
    fit_peak_memory = getattr(model, "fit_peak_memory_mb", ()) or ()
    fit_peak_memory_mb = float(fit_peak_memory[-1]) if fit_peak_memory else None
    history = getattr(model, "last_history", None) or {}
    train_losses = [float(value) for value in history.get("train_losses", ())]
    val_losses = [float(value) for value in history.get("val_losses", ())]
    val_objective_losses = [
        float(value) for value in history.get("val_objective_losses", ())
    ]
    val_innovation_nlls = [
        float(value) for value in history.get("val_innovation_nlls", ())
    ]
    task_val_aucs = [
        None if value is None else float(value)
        for value in history.get("task_val_aucs", ())
    ]
    epoch_trajectory_audit = _evaluate_epoch_trajectory_audit(
        model,
        X[test_mask],
        y[test_mask],
        digits[test_mask],
        subject_ids[test_mask],
        true_digits,
        digit_vocab,
        decision_center,
        decision_aggregation,
        evidence_budgets,
        event_timeline,
        np.flatnonzero(test_mask),
        trial_channel_mask[test_mask] if trial_channel_mask is not None else None,
    )
    calibration_logits, calibration_y, calibration_source = calibration_data_from_model(
        model, X[train_mask], y[train_mask]
    )
    calibration = fit_logit_calibration(
        calibration_logits, calibration_y, source=calibration_source
    )
    logits = _predict_logit_with_optional_trial_mask(
        model,
        X[test_mask],
        trial_channel_mask[test_mask] if trial_channel_mask is not None else None,
    )

    # 非有限 logits 统一入口守卫（audit P2-1），避免 bacc/AUC 先吃 NaN 再在 decide 报错。
    if not np.isfinite(logits).all():
        raise ValueError("模型 predict_logit 输出含 NaN/inf，无法进入 bacc/AUC/decision。")

    # 试次级二分类指标（D-hit-vs-bacc）
    # review v6 P0-1：阈值 0 只在 logit 无常数偏置时是 balanced 阈值；
    # 先对 fold 内 logit 中心化再取 0 阈值，与决策层的中心化口径一致。
    y_test = y[test_mask]
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        bacc = float(balanced_accuracy_score(y_test, (logits >= calibration.threshold).astype(int)))
        logits_centered = logits - logits.mean()
        transductive_bacc = float(
            balanced_accuracy_score(y_test, (logits_centered >= 0).astype(int))
        )
    else:
        bacc = np.nan  # 单类测试集无法定义 balanced accuracy（audit P2-2）
        transductive_bacc = np.nan
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

    variant_records, variant_covered, variant_total = _gtn_decision_variants(
        logits,
        calibration.to_llr(logits),
        digits[test_mask],
        subject_ids[test_mask],
        true_digits,
        digit_vocab,
        evidence_budgets,
        event_timeline,
        np.flatnonzero(test_mask),
        tuple(np.unique(subject_ids[test_mask]).astype(str).tolist()),
    )
    predict_repetition = getattr(model, "predict_repetition_candidates", None)
    chain_diagnostics: dict[str, dict] = {}
    descriptive_variant_records: dict[str, list[dict[str, object]]] = {}
    if callable(predict_repetition):
        # Register every requested chain endpoint before querying the fitted
        # head. A fail-closed head contributes zero covered subjects, not a
        # silently reduced denominator.
        chain_ks = tuple(
            int(budget.value) for budget in evidence_budgets if budget.kind == "prefix_minK"
        )
        include_all_chain = any(budget.kind == "all" for budget in evidence_budgets)
        expected_chain_names: set[str] = set()
        for budget in (
            *[EvidenceBudget("prefix_minK", value) for value in chain_ks],
            *([EvidenceBudget("all", None)] if include_all_chain else []),
        ):
            chain_name = _metric_name("chain_llr", budget)
            expected_chain_names.add(chain_name)
            variant_records.setdefault(chain_name, [])
            variant_covered.setdefault(chain_name, 0)
        predict_kwargs = {
            "digit_vocab": digit_vocab,
            "evidence_budgets": (*chain_ks, *([None] if include_all_chain else [])),
        }
        if getattr(model, "predict_accepts_acquisition_indices", False):
            predict_kwargs["acquisition_indices"] = acquisition_indices[test_mask]
        if (
            trial_channel_mask is not None
            and getattr(model, "auxiliary_predict_accepts_trial_channel_mask", False)
        ):
            predict_kwargs["trial_channel_mask"] = trial_channel_mask[test_mask]
        chain_results = predict_repetition(
            X[test_mask],
            digits[test_mask],
            subject_ids[test_mask],
            **predict_kwargs,
        )
        for name, payload in chain_results.items():
            if name not in expected_chain_names:
                raise ValueError(
                    f"Repetition head returned undeclared metric {name!r}; expected "
                    f"{sorted(expected_chain_names)}."
                )
            required_payload = {"predicted", "subject_ids", "scores", "mean_reliability"}
            missing_payload = required_payload - set(payload)
            if missing_payload:
                raise ValueError(
                    f"Repetition metric {name!r} lacks fields {sorted(missing_payload)}."
                )
            predicted_values = np.asarray(payload["predicted"])
            predicted_subjects = np.asarray(payload["subject_ids"])
            scores = np.asarray(payload["scores"], dtype=float)
            reliability = np.asarray(payload["mean_reliability"], dtype=float)
            n_predictions = len(predicted_values)
            if (
                predicted_subjects.shape != (n_predictions,)
                or scores.shape != (n_predictions, len(digit_vocab))
                or reliability.shape != (n_predictions,)
                or not np.isfinite(scores).all()
                or not np.isfinite(reliability).all()
            ):
                raise ValueError(f"Repetition metric {name!r} has invalid or non-finite shapes.")
            chain_records = []
            for predicted, subject in zip(predicted_values, predicted_subjects, strict=True):
                true_digit = true_digits.get(subject)
                if true_digit is not None:
                    chain_records.append((predicted, true_digit, subject))
            fold_units = tuple(np.unique(subject_ids[test_mask]).astype(str).tolist())
            # Formal ITT records only accept predictions from a fold whose
            # repetition branch passed its reliability gate.  Gate-failed folds
            # still produce descriptive chain diagnostics below, but their
            # primary-metric rows stay unavailable so coverage cannot be faked.
            chain_formal_eligible = bool(
                payload.get("claim_eligible", getattr(model, "repetition_ready_", False))
            )
            descriptive_rows = _complete_records(chain_records, true_digits, fold_units)
            descriptive_by_subject = {
                str(subject): {
                    "scores": scores[index].tolist(),
                    "mean_reliability": float(reliability[index]),
                }
                for index, subject in enumerate(predicted_subjects)
            }
            descriptive_records = []
            for predicted, true, subject in descriptive_rows:
                detail = descriptive_by_subject.get(str(subject))
                available = predicted is not None and detail is not None
                descriptive_records.append(
                    {
                        "subject": str(subject),
                        "predicted": int(predicted) if predicted is not None else None,
                        "true": int(true),
                        "available": bool(available),
                        "hit": int(available and predicted == true),
                        "scores": detail["scores"] if detail is not None else None,
                        "mean_reliability": (
                            detail["mean_reliability"] if detail is not None else None
                        ),
                        "claim_eligible": bool(chain_formal_eligible),
                        "formal_available": bool(available and chain_formal_eligible),
                    }
                )
            descriptive_variant_records[name] = descriptive_records
            formal_chain_records = chain_records if chain_formal_eligible else []
            formal_rows = _complete_records(formal_chain_records, true_digits, fold_units)
            variant_records[name] = formal_rows
            variant_covered[name] = sum(
                predicted is not None for predicted, _, _ in formal_rows
            )
            selection_nll = float("nan")
            selection_brier = float("nan")
            selection_ece = float("nan")
            if len(scores):
                top_two = np.sort(np.partition(scores, -2, axis=1)[:, -2:], axis=1)
                margins = top_two[:, 1] - top_two[:, 0]
                shifted = scores - scores.max(axis=1, keepdims=True)
                probabilities = np.exp(shifted)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                true_indices = np.asarray(
                    [
                        digit_vocab.index(true_digits[subject])
                        for subject in payload["subject_ids"]
                        if true_digits.get(subject) in digit_vocab
                    ],
                    dtype=int,
                )
                if len(true_indices) == len(probabilities):
                    rows = np.arange(len(probabilities))
                    selection_nll = float(
                        -np.log(probabilities[rows, true_indices].clip(1e-12)).mean()
                    )
                    one_hot = np.zeros_like(probabilities)
                    one_hot[rows, true_indices] = 1.0
                    selection_brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
                    confidence = probabilities.max(axis=1)
                    correct = probabilities.argmax(axis=1) == true_indices
                    selection_ece = _expected_calibration_error(confidence, correct.astype(float))
            else:
                margins = np.asarray([], dtype=float)
            chain_diagnostics[name] = {
                "formal_eligible": bool(chain_formal_eligible),
                "formal_covered": int(
                    sum(predicted is not None for predicted, _, _ in formal_rows)
                ),
                "descriptive_covered": len(chain_records),
                "mean_reliability": (
                    float(reliability.mean()) if len(reliability) else float("nan")
                ),
                "mean_top1_log_score_margin": (
                    float(margins.mean()) if len(margins) else float("nan")
                ),
                "selection_nll": selection_nll,
                "selection_brier": selection_brier,
                "selection_ece": selection_ece,
                "reliability_std": (float(reliability.std()) if len(reliability) else float("nan")),
                "reliability_saturation_fraction": (
                    float(((reliability < 0.05) | (reliability > 0.95)).mean())
                    if len(reliability)
                    else float("nan")
                ),
                "n_covered": len(chain_records),
                "descriptive_subject_records": descriptive_records,
            }
    confounds = _confound_records(
        digits, subject_ids, true_digits, digit_vocab, train_mask, test_mask
    )
    audit = _neural_ride_fold_audit(
        model,
        X[train_mask],
        y[train_mask],
        X[test_mask],
        y_test,
        digits[test_mask],
        subject_ids[test_mask],
        true_digits,
        digit_vocab,
        event_timeline,
        np.flatnonzero(test_mask),
        trial_channel_mask[test_mask] if trial_channel_mask is not None else None,
    )
    if chain_diagnostics:
        temperature = getattr(model, "repetition_temperature_calibration_", None)
        audit["repetition"] = {
            "ready": bool(getattr(model, "repetition_ready_", False)),
            "reliability_gate": getattr(model, "repetition_reliability_audit_", None),
            "metrics": chain_diagnostics,
            "temperature": (
                float(temperature.temperature) if temperature is not None else float("nan")
            ),
            "temperature_source": (
                temperature.source if temperature is not None else "unavailable"
            ),
            "pos_weight": (
                float(temperature.pos_weight) if temperature is not None else float("nan")
            ),
            "train_prior": (
                float(temperature.train_prior) if temperature is not None else float("nan")
            ),
        }
    elif hasattr(model, "repetition_ready_"):
        audit["repetition"] = {
            "ready": bool(getattr(model, "repetition_ready_", False)),
            "reliability_gate": getattr(model, "repetition_reliability_audit_", None),
            "metrics": {},
            "temperature_source": "unavailable",
        }

    return (
        FoldResult(
            hit_rate=hit,
            balanced_acc=bacc,
            auc=auc,
            n_subjects=n_counted,
            n_test_trials=int(test_mask.sum()),
            threshold=calibration.threshold,
            threshold_source=calibration.source,
            transductive_balanced_acc=transductive_bacc,
            audit=audit,
            fit_sec=fit_sec,
            fit_peak_memory_mb=fit_peak_memory_mb,
            epochs_ran=len(train_losses),
            train_losses=train_losses,
            val_losses=val_losses,
            val_objective_losses=val_objective_losses,
            val_innovation_nlls=val_innovation_nlls,
            task_val_aucs=task_val_aucs,
            final_task_val_auc=history.get("final_task_val_auc"),
            best_task_epoch=history.get("best_task_epoch", history.get("best_epoch")),
            best_density_epoch=history.get("best_density_epoch"),
            best_task_val_loss=history.get("best_task_val_loss"),
            best_density_nll=history.get("best_density_nll"),
            task_patience_exhausted=bool(history.get("task_patience_exhausted", False)),
            epoch_trajectory_audit=epoch_trajectory_audit,
            training_history=history,
            val_subjects=getattr(model, "last_val_subjects", None),
            audit_subjects=getattr(model, "last_audit_subjects", None),
            erp_calibration=getattr(model, "last_erp_calibration", None),
            component_summary=dict(audit.get("component_summary", {})),
            descriptive_decision_records=descriptive_variant_records,
        ),
        records,
        variant_records,
        variant_covered,
        variant_total,
        confounds,
    )


def _configure_model_fold(model, fold_id: int) -> None:
    """Configure a model's fold context through the shared baseline interface."""

    configure = getattr(model, "configure_evaluation_fold", None)
    if callable(configure):
        configure(fold_id)
    elif hasattr(model, "_evaluation_fold_id"):
        # Compatibility for third-party baseline adapters that predate the
        # public configuration method.
        model._evaluation_fold_id = fold_id


def _run_fold_core(
    args: tuple[
        int,
        int,
        object,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Mapping,
        Sequence[int],
        bool,
        str,
        Sequence[EvidenceBudget],
        ScheduledEventTimeline,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> tuple[
    int,
    FoldResult,
    list[tuple[object, object, object]],
    dict[str, list[tuple[object, object, object]]],
    dict[str, int],
    int,
    dict[str, list[tuple[object, object, object]]],
]:
    """Execute one fold after the caller selects its process or thread backend."""
    (
        fold_idx,
        fold_id_offset,
        model_proto,
        X,
        y,
        digits,
        subject_ids,
        fold_subject_ids,
        true_digits,
        digit_vocab,
        decision_center,
        decision_aggregation,
        evidence_budgets,
        event_timeline,
        acquisition_indices,
        trial_channel_mask,
        train_mask,
        test_mask,
    ) = args
    model = copy.deepcopy(model_proto)
    _configure_model_fold(model, fold_idx + fold_id_offset)
    try:
        fold_result, records, variants, covered, total, confounds = _evaluate_one_fold(
            model,
            X,
            y,
            digits,
            subject_ids,
            fold_subject_ids,
            true_digits,
            digit_vocab,
            decision_center,
            decision_aggregation,
            evidence_budgets,
            event_timeline,
            acquisition_indices,
            trial_channel_mask,
            train_mask,
            test_mask,
        )
        return fold_idx, fold_result, records, variants, covered, total, confounds
    finally:
        _release_fold_model(model)


def _release_fold_model(model) -> None:
    """Drop fold-local accelerator state after metrics no longer need the model."""

    release = getattr(model, "_release_fold_runtime", None)
    if callable(release):
        release()


def _terminate_fold_worker_with_parent(parent_pid: int) -> None:
    """Ask Linux to terminate this worker when its executor parent dies."""

    if os.name != "posix":
        raise RuntimeError("Fold parent-death signaling requires a POSIX/Linux runtime.")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # The parent can exit between fork() and prctl(). Fail closed if that race
    # already happened instead of leaving an orphaned CUDA worker behind.
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def _initialize_fold_worker(parent_pid: int) -> None:
    """Configure lifecycle and CPU pools in a fresh spawned worker."""

    _terminate_fold_worker_with_parent(parent_pid)
    try:
        import torch

        fold_threads = _fold_cpu_threads()
        torch.set_num_threads(fold_threads)
        try:
            torch.set_num_interop_threads(fold_threads)
        except RuntimeError:
            pass
    except ImportError:
        pass


def _fold_process_executor_kwargs(n_jobs: int) -> dict[str, object]:
    return {
        "max_workers": n_jobs,
        "mp_context": mp.get_context("spawn"),
        "initializer": _initialize_fold_worker,
        "initargs": (os.getpid(),),
    }


def _run_fold_process(task_args):
    fold_idx = int(task_args[0])
    display_fold_idx = fold_idx + int(task_args[1])
    print(f"[fold {display_fold_idx}] worker started pid={os.getpid()}", flush=True)
    with _fold_threadpool_limits():
        return _run_fold_core(task_args)


def evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    digits: np.ndarray,
    subject_ids: np.ndarray,
    true_digits: Mapping,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    digit_vocab: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9),
    trial_channel_mask: np.ndarray | None = None,
    decision_center: bool = True,
    decision_aggregation: str = "sum",
    evidence_budgets: Sequence[int | None] = (1, 3, 5, 10, 15, None),
    primary_decision_metric: str = "exact_llr@3",
    fixed_error_rate: float = 0.05,
    efficiency_min_coverage: float = 0.90,
    primary_min_coverage: float = 0.90,
    repetition_duration_s: float | None = None,
    flash_budgets: Sequence[int] = (),
    time_budgets_s: Sequence[float] = (),
    event_timeline: ScheduledEventTimeline | None = None,
    evaluation_units: Sequence[object] | None = None,
    fold_protocol: str = "loso",
    require_complete_event_timeline: bool = True,
    dataset_sha256: str | None = None,
    n_jobs: int = 1,
    parallel_backend: str = "thread",
    fold_id_offset: int = 0,
    on_fold_end: Callable | None = None,
) -> EvalSummary:
    """按 folds 评估一个「有 fit/predict_logit 接口」的模型。

    Parameters
    ----------
    model : 有 fit(X,y) / predict_logit(X)→(N,) 接口的分类器（Baseline 或 N2P3-Net 包装）。
    X / y : (N,C,T) / (N,) 试次数据与 target/non-target 标签（y 用于训练）。
    digits : (N,) 每个试次对应的刺激数字（决策层累加用）。
    subject_ids : (N,) 每个试次的 selection group id（决策层聚合用）。外层被试隔离身份从
        event_timeline.subject_ids 按 evidence mapping 取得，不能用 run/group id 替代。
    true_digits : Mapping {分组键: 心选数字}，key 须与 subject_ids 的分组键一致（subject 或
        (subject, run) 组合，见 D-group-key）。
    folds : list[(train_mask, test_mask)] 布尔掩码。
    digit_vocab : 完整数字集（默认 1–9）。
    decision_center : bool
        决策层是否先逐被试中心化 logit（默认 True，review v6 P0-1）。
    decision_aggregation : str
        决策层聚合方式："sum"（默认）或 "mean"（消融轴）。
      n_jobs : int
          并行 fold 数（默认 1 串行）。Linux process backend 中每个 fold 由独立 worker 执行。
      on_fold_end : Callable | None
          逐 fold 回调 on_fold_end(fold_idx, fold_result, records)（GLM v3 实时进度：
          runner 用它把每 fold 指标增量写入 progress.jsonl 供仪表盘消费）。串行路径在
          每 fold 完成即调用；并行路径按实际完成顺序调用，fold_idx 保留原始编号。
      parallel_backend : {"auto", "process", "thread"}
          ``process`` 在 Linux 使用独立 fork worker 和独立 CUDA context；``thread`` 保留
          兼容路径。``auto`` 在 Linux 选择 ``process``，其他平台选择 ``thread``。
      fold_id_offset : int
          Display fold offset for batched runs. It is passed explicitly to workers so
          progress paths do not depend on process-global environment variables.

    Returns
    -------
    EvalSummary：命中率（均值/标准差）+ balanced_acc 均值 + AUC 均值 + 逐折 + 逐被试记录。
    """
    X = np.asarray(X)
    y = np.asarray(y)
    digits = np.asarray(digits)
    subject_array = np.asarray(subject_ids)
    if X.ndim != 3 or len(X) == 0:
        raise ValueError(f"X must be a non-empty (N,C,T) tensor, got {X.shape}.")
    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError("X must have a floating dtype.")
    if y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y must be a one-dimensional integer array aligned with X.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Evaluation requires binary labels {0,1}.")
    if digits.shape != (len(X),) or not np.issubdtype(digits.dtype, np.integer):
        raise ValueError("digits must be a one-dimensional integer array aligned with X.")
    if subject_array.shape != (len(X),):
        raise ValueError("subject_ids must be one-dimensional and aligned with X.")
    subject_ids = subject_array.astype(str)
    if not (len(X) == len(y) == len(digits) == len(subject_ids)):
        raise ValueError("X/y/digits/subject_ids must contain the same number of rows.")
    if trial_channel_mask is not None:
        trial_channel_mask = np.asarray(trial_channel_mask)
        if trial_channel_mask.dtype != np.dtype(bool):
            raise ValueError("trial_channel_mask must have boolean dtype.")
        if trial_channel_mask.shape != X.shape[:2]:
            raise ValueError("trial_channel_mask must have shape (N,C) matching X.")
        if not bool(trial_channel_mask.any(axis=1).all()):
            raise ValueError("Every trial must retain at least one observed channel.")
        if bool((X[~trial_channel_mask] != 0.0).any()):
            raise ValueError("X must be zero where trial_channel_mask is false.")
    if not np.isfinite(X).all():
        raise ValueError(
            "X contains NaN/inf; repair or explicitly reject invalid epochs before evaluation."
        )
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive.")
    if fold_id_offset < 0:
        raise ValueError("fold_id_offset must be non-negative.")
    if parallel_backend not in {"auto", "process", "thread"}:
        raise ValueError("parallel_backend must be one of 'auto', 'process', or 'thread'.")
    if parallel_backend == "auto":
        parallel_backend = "process" if os.name == "posix" else "thread"
    if parallel_backend == "process" and os.name != "posix":
        raise RuntimeError("parallel_backend='process' requires a POSIX/Linux runtime.")
    digit_vocab = tuple(int(value) for value in digit_vocab)
    if not digit_vocab or len(set(digit_vocab)) != len(digit_vocab):
        raise ValueError("digit_vocab must be non-empty and unique.")
    validate_decision_metric_name(primary_decision_metric)
    normalized_truth: dict[str, object] = {}
    for key, value in true_digits.items():
        normalized = str(key)
        if normalized in normalized_truth:
            raise ValueError(f"Duplicate normalized truth id {normalized!r}.")
        normalized_truth[normalized] = value
    true_digits = normalized_truth
    if event_timeline is None:
        raise ValueError(
            "evaluate requires a ScheduledEventTimeline; observed epoch rows are not a "
            "complete acquisition protocol."
        )
    event_timeline.validate(n_epochs=len(X))
    if require_complete_event_timeline and not event_timeline.complete:
        raise ValueError("Formal evaluation requires a complete scheduled-event timeline.")
    timeline_groups = set(event_timeline.groups)
    if timeline_groups != set(true_digits):
        raise ValueError(
            "Scheduled-event groups and frozen truth universe differ: "
            f"events_only={sorted(timeline_groups - set(true_digits))[:5]}, "
            f"truth_only={sorted(set(true_digits) - timeline_groups)[:5]}."
        )
    event_evidence = np.asarray(event_timeline.evidence_indices, dtype=np.int64)
    event_available = event_evidence >= 0
    aligned_groups = np.empty(len(X), dtype=object)
    aligned_fold_subjects = np.empty(len(X), dtype=object)
    aligned_digits = np.empty(len(X), dtype=np.int64)
    aligned_groups[event_evidence[event_available]] = np.asarray(event_timeline.group_ids).astype(
        str
    )[event_available]
    aligned_fold_subjects[event_evidence[event_available]] = np.asarray(
        event_timeline.subject_ids
    ).astype(str)[event_available]
    aligned_digits[event_evidence[event_available]] = np.asarray(
        event_timeline.stimulus_ids, dtype=np.int64
    )[event_available]
    if not np.array_equal(aligned_groups.astype(str), subject_ids):
        raise ValueError("Scheduled-event evidence mapping disagrees with subject_ids.")
    if not np.array_equal(aligned_digits, digits.astype(np.int64)):
        raise ValueError("Scheduled-event evidence mapping disagrees with stimulus digits.")
    per_fold: list[FoldResult] = []
    subject_records: list[tuple[object, object, object]] = []
    variant_records: dict[str, list[tuple[object, object, object]]] = {}
    variant_covered: dict[str, int] = {}
    variant_total: dict[str, int] = {}
    confound_records: dict[str, list[tuple[object, object, object]]] = {}

    budgets = build_evidence_budgets(
        evidence_budgets,
        flash_budgets=flash_budgets,
        time_budgets_s=time_budgets_s,
    )
    fold_subject_ids = aligned_fold_subjects.astype(str)
    mask_folds = validate_outer_folds(folds, fold_subject_ids, protocol=fold_protocol)
    tested_units = tuple(
        sorted(
            {
                str(subject)
                for _, test_mask in mask_folds
                for subject in np.unique(subject_ids[test_mask])
            }
        )
    )
    units = (
        tuple(str(unit) for unit in evaluation_units)
        if evaluation_units is not None
        else (tested_units if fold_protocol == "partial_loso" else tuple(sorted(true_digits)))
    )
    if not units or len(set(units)) != len(units):
        raise ValueError("evaluation_units must be non-empty and unique.")
    if not set(tested_units).issubset(units):
        raise ValueError("Outer test folds contain units outside the frozen evaluation universe.")
    if not set(units).issubset(true_digits) or not set(units).issubset(timeline_groups):
        raise ValueError("Every evaluation unit requires frozen truth and scheduled events.")
    if fold_protocol == "loso" and set(units) != set(true_digits):
        raise ValueError("Complete LOSO evaluation_units must equal the frozen truth universe.")
    timeline_for_evaluation = (
        event_timeline
        if set(units) == timeline_groups
        else event_timeline.subset_groups(set(units))
    )
    acquisition_indices = row_acquisition_indices(event_timeline)

    if n_jobs > 1 and len(mask_folds) > 1:
        # Spawn is required for CUDA safety: a forked child can inherit locked
        # PyTorch/OpenMP runtime state and sleep forever before its first epoch.
        task_args = [
            (
                fold_idx,
                fold_id_offset,
                model,
                X,
                y,
                digits,
                subject_ids,
                fold_subject_ids,
                true_digits,
                digit_vocab,
                decision_center,
                decision_aggregation,
                budgets,
                event_timeline,
                acquisition_indices,
                trial_channel_mask,
                train_mask,
                test_mask,
            )
            for fold_idx, (train_mask, test_mask) in enumerate(mask_folds)
        ]
        use_processes = parallel_backend == "process"
        executor_kwargs = (
            _fold_process_executor_kwargs(n_jobs)
            if use_processes
            else {"max_workers": n_jobs}
        )
        executor_type = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        worker = _run_fold_process if use_processes else _run_fold_core
        limits = nullcontext() if use_processes else _fold_threadpool_limits()
        with limits:
            with executor_type(**executor_kwargs) as executor:
                futures = [executor.submit(worker, arg) for arg in task_args]
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if on_fold_end is not None:
                        on_fold_end(result[0], result[1], result[2])
        # 按 fold 原始顺序恢复
        for _, fold_result, records, variants, covered, total, confounds in sorted(
            results, key=lambda item: item[0]
        ):
            per_fold.append(fold_result)
            subject_records.extend(records)
            for name, rows in variants.items():
                variant_records.setdefault(name, []).extend(rows)
                variant_covered[name] = variant_covered.get(name, 0) + covered[name]
                variant_total[name] = variant_total.get(name, 0) + total
            for name, rows in confounds.items():
                confound_records.setdefault(name, []).extend(rows)
    else:
        for fold_idx, (train_mask, test_mask) in enumerate(mask_folds):
            fold_model = copy.deepcopy(model)
            _configure_model_fold(fold_model, fold_idx + fold_id_offset)
            # Serial folds must honor the same FOLD_CPU_THREADS budget as the
            # parallel workers; the parallel branch enters this context inside
            # _run_fold_core, so the single-fold / n_jobs=1 path must not skip it.
            try:
                with _fold_threadpool_limits():
                    fold_result, records, variants, covered, total, confounds = _evaluate_one_fold(
                        fold_model,
                        X,
                        y,
                        digits,
                        subject_ids,
                        fold_subject_ids,
                        true_digits,
                        digit_vocab,
                        decision_center,
                        decision_aggregation,
                        budgets,
                        event_timeline,
                        acquisition_indices,
                        trial_channel_mask,
                        train_mask,
                        test_mask,
                    )
            finally:
                _release_fold_model(fold_model)
            per_fold.append(fold_result)
            subject_records.extend(records)
            for name, rows in variants.items():
                variant_records.setdefault(name, []).extend(rows)
                variant_covered[name] = variant_covered.get(name, 0) + covered[name]
                variant_total[name] = variant_total.get(name, 0) + total
            for name, rows in confounds.items():
                confound_records.setdefault(name, []).extend(rows)
            if on_fold_end is not None:
                on_fold_end(len(per_fold) - 1, fold_result, records)

    baccs = np.array([f.balanced_acc for f in per_fold], dtype=float)
    baccs = baccs[np.isfinite(baccs)]
    aucs = np.array([f.auc for f in per_fold], dtype=float)
    aucs = aucs[np.isfinite(aucs)]

    # 按「猜测单元」（group key = 被试或 (subject, run)）聚合命中率，而非按 fold 等权
    # （D-hit-by-unit）。fold 等权在 within-subject 下会让多 run 被试权重虚高。
    subject_records = _complete_records(subject_records, true_digits, units)
    unit_hits = np.array(
        [predicted is not None and predicted == true for predicted, true, _ in subject_records],
        dtype=float,
    )
    transductive_baccs = np.array([f.transductive_balanced_acc for f in per_fold], dtype=float)
    transductive_baccs = transductive_baccs[np.isfinite(transductive_baccs)]

    decision_metrics: dict[str, DecisionMetric] = {}
    for name, rows in variant_records.items():
        aggregation, evidence_budget, semantics = _metric_metadata(name)
        rows = _complete_records(rows, true_digits, units)
        covered_rows = [row for row in rows if row[0] is not None]
        hits = np.asarray(
            [predicted is not None and predicted == true for predicted, true, _ in rows],
            dtype=float,
        )
        conditional_hits = np.asarray(
            [predicted == true for predicted, true, _ in covered_rows], dtype=float
        )
        n_total = len(units)
        n_covered = len(covered_rows)
        decision_metrics[name] = DecisionMetric(
            name=name,
            evidence_budget=evidence_budget,
            aggregation=aggregation,
            hit_rate=float(hits.mean()),
            conditional_hit_rate=(
                float(conditional_hits.mean()) if len(conditional_hits) else float("nan")
            ),
            n_covered=n_covered,
            n_total=n_total,
            coverage=n_covered / n_total if n_total else 0.0,
            budget_semantics=semantics,
            subject_records=rows,
        )
    if primary_decision_metric not in decision_metrics:
        raise ValueError(
            f"Primary decision metric {primary_decision_metric!r} was not produced. "
            "A chain_llr primary requires a trained, validation-calibrated repetition head."
        )
    primary_metric = decision_metrics[primary_decision_metric]
    primary_metric_gate = _build_primary_metric_gate(
        primary_metric,
        per_fold,
        primary_min_coverage=primary_min_coverage,
    )

    descriptive_decision_records: dict[str, list[dict[str, object]]] = {}
    for name, metric in decision_metrics.items():
        if metric.aggregation != "chain_llr":
            continue
        by_subject: dict[str, dict[str, object]] = {}
        for fold in per_fold:
            for row in fold.descriptive_decision_records.get(name, []):
                subject = str(row["subject"])
                if subject in by_subject:
                    raise ValueError(
                        f"Duplicate descriptive decision record for evaluation unit {subject!r}."
                    )
                by_subject[subject] = row
        descriptive_decision_records[name] = [
            by_subject.get(
                unit,
                {
                    "subject": unit,
                    "predicted": None,
                    "true": int(true_digits[unit]),
                    "available": False,
                    "hit": 0,
                    "scores": None,
                    "mean_reliability": None,
                    "claim_eligible": False,
                    "formal_available": False,
                },
            )
            for unit in units
        ]

    confound_baselines: dict[str, ConfoundBaseline] = {}
    for name, rows in confound_records.items():
        rows = _complete_records(rows, true_digits, units)
        hits = np.asarray(
            [predicted is not None and predicted == true for predicted, true, _ in rows],
            dtype=float,
        )
        confound_baselines[name] = ConfoundBaseline(
            name=name,
            hit_rate=float(hits.mean()) if len(hits) else float("nan"),
            n_subjects=len(units),
            subject_records=rows,
        )

    repetition_efficiency = summarize_repetition_efficiency(
        decision_metrics,
        n_choices=len(digit_vocab),
        target_error_rate=fixed_error_rate,
        minimum_coverage=efficiency_min_coverage,
        repetition_duration_s=repetition_duration_s,
        aggregation=primary_metric.aggregation,
        budget_semantics=primary_metric.budget_semantics,
    )

    return EvalSummary(
        hit_rate_mean=float(unit_hits.mean()) if unit_hits.size else 0.0,
        hit_rate_std=float(unit_hits.std()) if unit_hits.size else 0.0,
        balanced_acc_mean=float(baccs.mean()) if baccs.size else 0.0,
        auc_mean=float(aucs.mean()) if aucs.size else float("nan"),
        per_fold=per_fold,
        subject_records=subject_records,
        transductive_balanced_acc_mean=(
            float(transductive_baccs.mean()) if transductive_baccs.size else float("nan")
        ),
        decision_metrics=decision_metrics,
        confound_baselines=confound_baselines,
        primary_decision_metric=primary_decision_metric,
        primary_metric_gate=primary_metric_gate,
        descriptive_decision_records=descriptive_decision_records,
        repetition_efficiency=repetition_efficiency,
        evaluation_units=units,
        cohort_sha256=timeline_for_evaluation.fingerprint(
            truth={unit: true_digits[unit] for unit in units}
        ),
        dataset_sha256=dataset_sha256,
        fold_protocol=fold_protocol,
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
    threshold: float = 0.0
    threshold_source: str = "unknown"
    transductive_balanced_acc: float = float("nan")
    # Keep the fold-local training trace with the result so threaded workers
    # can publish it without relying on mutable state in the parent prototype.
    epochs_ran: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_objective_losses: list[float] = field(default_factory=list)
    task_val_aucs: list[float | None] = field(default_factory=list)
    final_task_val_auc: float | None = None
    phases: list[str] = field(default_factory=list)
    best_epoch: int | None = None
    best_task_epoch: int | None = None
    best_task_val_loss: float | None = None
    task_patience_exhausted: bool = False
    fit_sec: float | None = None
    fit_peak_memory_mb: float | None = None


@dataclass
class BinarySummary:
    """二分类多折汇总（无命中率口径，D-hit-vs-bacc）。"""

    balanced_acc_mean: float
    balanced_acc_std: float
    auc_mean: float
    per_fold: list[BinaryFoldResult] = field(default_factory=list)
    transductive_balanced_acc_mean: float = float("nan")


def _evaluate_one_binary_fold(
    model,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    acquisition_indices: np.ndarray | None = None,
    trial_channel_mask: np.ndarray | None = None,
) -> BinaryFoldResult:
    """执行单个二分类 fold（与 evaluate_binary 串行路径完全同构）。"""
    # GLM v3：声明 fit_accepts_subject_ids 的模型收到被试分组（被试级验证早停；
    # 此前二分类路径漏传，导致 BNCI 路线早停静默失效——2026-08-23 bnci008 run 实测发现）
    fit_started = time.perf_counter()
    _fit_model_with_optional_subjects(
        model,
        X,
        y,
        subject_ids,
        subject_ids,
        train_mask,
        acquisition_indices=acquisition_indices,
        trial_channel_mask=trial_channel_mask,
    )
    fit_sec = time.perf_counter() - fit_started
    fit_durations = getattr(model, "fit_durations", ()) or ()
    if fit_durations:
        fit_sec = float(fit_durations[-1])
    fit_peak_memory = getattr(model, "fit_peak_memory_mb", ()) or ()
    fit_peak_memory_mb = float(fit_peak_memory[-1]) if fit_peak_memory else None
    history = getattr(model, "last_history", None) or {}
    train_losses = [float(value) for value in history.get("train_losses", ())]
    val_losses = [float(value) for value in history.get("val_losses", ())]
    val_objective_losses = [
        float(value) for value in history.get("val_objective_losses", ())
    ]
    task_val_aucs = [
        None if value is None else float(value)
        for value in history.get("task_val_aucs", ())
    ]
    phases = [str(value) for value in history.get("phases", ())]
    calibration_logits, calibration_y, calibration_source = calibration_data_from_model(
        model, X[train_mask], y[train_mask]
    )
    calibration = fit_logit_calibration(
        calibration_logits, calibration_y, source=calibration_source
    )
    logits = _predict_logit_with_optional_trial_mask(
        model,
        X[test_mask],
        trial_channel_mask[test_mask] if trial_channel_mask is not None else None,
    )

    if not np.isfinite(logits).all():
        raise ValueError("模型 predict_logit 输出含 NaN/inf，无法进入 bacc/AUC。")

    y_test = y[test_mask]
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        bacc = float(balanced_accuracy_score(y_test, (logits >= calibration.threshold).astype(int)))
        logits_centered = logits - logits.mean()
        transductive_bacc = float(
            balanced_accuracy_score(y_test, (logits_centered >= 0).astype(int))
        )
    else:
        bacc = np.nan  # 单类测试集无法定义 balanced accuracy（audit P2-2）
        transductive_bacc = np.nan
    auc = np.nan
    if n_classes == 2:  # D-auc-guard
        auc = float(roc_auc_score(y_test, logits))

    return BinaryFoldResult(
        balanced_acc=bacc,
        auc=auc,
        n_test_trials=int(test_mask.sum()),
        threshold=calibration.threshold,
        threshold_source=calibration.source,
        transductive_balanced_acc=transductive_bacc,
        epochs_ran=len(train_losses),
        train_losses=train_losses,
        val_losses=val_losses,
        val_objective_losses=val_objective_losses,
        task_val_aucs=task_val_aucs,
        final_task_val_auc=history.get("final_task_val_auc"),
        phases=phases,
        best_epoch=history.get("best_epoch"),
        best_task_epoch=history.get("best_task_epoch", history.get("best_epoch")),
        best_task_val_loss=history.get("best_task_val_loss"),
        task_patience_exhausted=bool(history.get("task_patience_exhausted", False)),
        fit_sec=fit_sec,
        fit_peak_memory_mb=fit_peak_memory_mb,
    )


def _run_binary_fold_threaded(
    args: tuple[
        int,
        int,
        object,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
    ],
) -> tuple[int, BinaryFoldResult]:
    """线程池 worker：深拷贝未拟合模型，BLAS 限单线程，避免并行 fold 互相抢占。"""
    (
        fold_idx,
        fold_id_offset,
        model_proto,
        X,
        y,
        subject_ids,
        train_mask,
        test_mask,
        acquisition_indices,
        trial_channel_mask,
    ) = args
    model = copy.deepcopy(model_proto)
    _configure_model_fold(model, fold_idx + fold_id_offset)
    try:
        result = _evaluate_one_binary_fold(
            model,
            X,
            y,
            subject_ids,
            train_mask,
            test_mask,
            acquisition_indices,
            trial_channel_mask,
        )
        return fold_idx, result
    finally:
        _release_fold_model(model)


def _run_binary_fold_process(task_args):
    fold_idx = int(task_args[0])
    display_fold_idx = fold_idx + int(task_args[1])
    print(f"[fold {display_fold_idx}] binary worker started pid={os.getpid()}", flush=True)
    with _fold_threadpool_limits():
        return _run_binary_fold_threaded(task_args)


def evaluate_binary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    acquisition_indices: np.ndarray | None = None,
    trial_channel_mask: np.ndarray | None = None,
    fold_protocol: str = "loso",
    n_jobs: int = 1,
    parallel_backend: str = "auto",
    fold_id_offset: int = 0,
    on_fold_end: Callable | None = None,
) -> BinarySummary:
    """二分类评估（LOSO / within-subject）→ balanced_acc + AUC。

    适用于**无「猜数字」ground truth** 的数据集（BNCI2014_008 / Brain Invaders / ERP CORE）：
    它们只有 target/non-target 二分类标签，无「心选数字」可做 9 选 1 命中率。这是 P300 检测的
    通用口径（单试次判别质量），与 :func:evaluate（猜数字 + 二分类）互补——后者专供 GTN
    （有 thought_number ground truth）。

    三思决策记录：
        D-binary-vs-gtn 二分类与猜数字是**两个不同层级**的口径：二分类报「单试次判别质量」
                        （balanced_acc / AUC），猜数字报 subject-level hit@K；均匀先验名义 chance
                        为 11.1%，实际还须报告训练折数字先验基线。
                        BNCI/Brain Invaders/ERP CORE 只有前者；GTN 两者都有。故拆两个函数，
                        而非在 evaluate 里用 None 占位——语义更清晰、不硬凑 digits。
        D-binary-noguard 沿用 D-nan-guard：X 含 NaN 显式报错（classic/riemann 用存在通道
                        子集，deep 按原生物理通道数构造），杜绝静默失真。
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
    subject_array = np.asarray(subject_ids)
    if subject_array.shape != (len(X),):
        raise ValueError("subject_ids must be one-dimensional and aligned with X.")
    subject_ids = subject_array.astype(str)
    if X.ndim != 3:
        raise ValueError(f"X must be (N,C,T), got {X.shape}.")
    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError("X must have a floating dtype.")
    if not (len(X) == len(y) == len(subject_ids)):
        raise ValueError("X/y/subject_ids must contain the same number of rows.")
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got {y.shape}.")
    labels = set(np.unique(y).tolist())
    if labels != {0, 1}:
        raise ValueError(f"Binary evaluation requires labels {{0,1}}, got {sorted(labels)}.")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Binary evaluation labels must have an integer dtype.")
    if trial_channel_mask is not None:
        trial_channel_mask = np.asarray(trial_channel_mask)
        if trial_channel_mask.dtype != np.dtype(bool):
            raise ValueError("trial_channel_mask must have boolean dtype.")
        if trial_channel_mask.shape != X.shape[:2]:
            raise ValueError("trial_channel_mask must have shape (N,C) matching X.")
        if not bool(trial_channel_mask.any(axis=1).all()):
            raise ValueError("Every evaluated trial must retain at least one channel.")
        if bool((X[~trial_channel_mask] != 0.0).any()):
            raise ValueError("X must be zero where trial_channel_mask is false.")
    if fold_id_offset < 0:
        raise ValueError("fold_id_offset must be non-negative.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive.")
    if parallel_backend not in {"auto", "process", "thread"}:
        raise ValueError("parallel_backend must be auto, process, or thread.")
    if parallel_backend == "auto":
        parallel_backend = "process" if os.name == "posix" else "thread"
    if parallel_backend == "process" and os.name != "posix":
        raise RuntimeError("parallel_backend='process' requires a POSIX/Linux runtime.")
    if acquisition_indices is not None:
        acquisition_indices = np.asarray(acquisition_indices)
        if acquisition_indices.shape != (len(X),):
            raise ValueError("acquisition_indices must contain one value per row of X.")
        if not np.issubdtype(acquisition_indices.dtype, np.integer):
            raise ValueError("acquisition_indices must contain integer ordinals.")
    # NaN 守卫（D-binary-noguard，同 D-nan-guard）
    if not np.isfinite(X).all():
        raise ValueError(
            "X contains NaN/inf: missing channels must be represented as zero plus an explicit mask."
        )

    mask_folds = validate_outer_folds(folds, subject_ids, protocol=fold_protocol)

    per_fold: list[BinaryFoldResult] = []
    if n_jobs > 1 and len(mask_folds) > 1:
        task_args = [
            (
                fold_idx,
                fold_id_offset,
                model,
                X,
                y,
                subject_ids,
                train_mask,
                test_mask,
                acquisition_indices,
                trial_channel_mask,
            )
            for fold_idx, (train_mask, test_mask) in enumerate(mask_folds)
        ]
        use_processes = parallel_backend == "process"
        executor_type = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        executor_kwargs = (
            _fold_process_executor_kwargs(n_jobs)
            if use_processes
            else {"max_workers": n_jobs}
        )
        worker = _run_binary_fold_process if use_processes else _run_binary_fold_threaded
        limits = nullcontext() if use_processes else _fold_threadpool_limits()
        results = []
        with limits:
            with executor_type(**executor_kwargs) as executor:
                futures = [executor.submit(worker, args) for args in task_args]
                for future in as_completed(futures):
                    fold_idx, result = future.result()
                    results.append((fold_idx, result))
                    if on_fold_end is not None:
                        on_fold_end(fold_idx, result)
        for _, result in sorted(results, key=lambda item: item[0]):
            per_fold.append(result)
    else:
        for fold_idx, (train_mask, test_mask) in enumerate(mask_folds):
            with _fold_threadpool_limits():
                _, fr = _run_binary_fold_threaded(
                    (
                        fold_idx,
                        fold_id_offset,
                        model,
                        X,
                        y,
                        subject_ids,
                        train_mask,
                        test_mask,
                        acquisition_indices,
                        trial_channel_mask,
                    )
                )
            per_fold.append(fr)
            if on_fold_end is not None:
                on_fold_end(fold_idx, fr)

    baccs = np.array([f.balanced_acc for f in per_fold], dtype=float)
    baccs = baccs[np.isfinite(baccs)]
    aucs = np.array([f.auc for f in per_fold], dtype=float)
    aucs = aucs[np.isfinite(aucs)]
    transductive = np.array([f.transductive_balanced_acc for f in per_fold], dtype=float)
    transductive = transductive[np.isfinite(transductive)]

    return BinarySummary(
        balanced_acc_mean=float(baccs.mean()) if baccs.size else float("nan"),
        balanced_acc_std=float(baccs.std()) if baccs.size else float("nan"),
        auc_mean=float(aucs.mean()) if aucs.size else float("nan"),
        per_fold=per_fold,
        transductive_balanced_acc_mean=(
            float(transductive.mean()) if transductive.size else float("nan")
        ),
    )
