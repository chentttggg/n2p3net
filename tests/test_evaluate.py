"""baselines.evaluate 模块测试：三层协议 + 配对置换检验。

合成多被试猜数字数据：每被试一个心选数字，target（当前数字==心选）在 Pz 叠正波。
语义：fold 生成正确、evaluate 端到端命中率 > chance、配对置换检验 p 值行为正确。
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from baselines.classic import WindowLogisticRegression
from baselines.evaluate import (
    _cpu_threadpool_limits,
    _evaluate_epoch_trajectory_audit,
    _fold_process_executor_kwargs,
    _fold_threadpool_limits,
    _initialize_fold_worker,
    _neural_ride_fold_audit,
    evaluate,
    evaluate_binary,
    loso_folds,
    paired_permutation_test,
    within_subject_folds,
)
from baselines.evidence_protocol import EvidenceBudget, evidence_row_indices, validate_outer_folds
from baselines.features import time_to_index
from data.events import ScheduledEventTimeline
from models.decision import decide

C = 8
T = 256
SFR = 256.0
TMIN = -0.2  # 秒；与 data/preprocess.py 一致


def test_fold_process_executor_registers_parent_death_signal(monkeypatch) -> None:
    from baselines import evaluate as evaluate_module

    monkeypatch.setattr(evaluate_module.mp, "get_context", lambda method: method)
    monkeypatch.setattr(evaluate_module.os, "getpid", lambda: 1234)

    kwargs = _fold_process_executor_kwargs(4)

    assert kwargs == {
        "max_workers": 4,
        "mp_context": "spawn",
        "initializer": _initialize_fold_worker,
        "initargs": (1234,),
    }


def test_fold_threadpool_limits_applies_fold_cpu_threads(monkeypatch) -> None:
    import torch

    monkeypatch.setenv("FOLD_CPU_THREADS", "2")
    previous = torch.get_num_threads()
    try:
        with _fold_threadpool_limits():
            assert torch.get_num_threads() == 2
    finally:
        assert torch.get_num_threads() == previous


def test_fold_threadpool_limits_fallback_restores_torch_threads(monkeypatch) -> None:
    import torch

    from baselines import evaluate as evaluate_module

    monkeypatch.setattr(evaluate_module, "threadpool_limits", None)
    monkeypatch.setenv("FOLD_CPU_THREADS", "2")
    previous = torch.get_num_threads()
    try:
        with _fold_threadpool_limits():
            assert torch.get_num_threads() == 2
    finally:
        assert torch.get_num_threads() == previous


def test_cpu_threadpool_limits_applies_explicit_parent_budget() -> None:
    import torch

    previous = torch.get_num_threads()
    try:
        with _cpu_threadpool_limits(3):
            assert torch.get_num_threads() == 3
    finally:
        assert torch.get_num_threads() == previous


def test_postprocess_cpu_threads_prefers_explicit_budget(monkeypatch) -> None:
    from experiments.run_n2p3net_gtn import _postprocess_cpu_threads

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("POSTPROCESS_CPU_THREADS", "6")

    assert _postprocess_cpu_threads() == 6


def test_postprocess_cpu_threads_defaults_to_openmp_budget(monkeypatch) -> None:
    from experiments.run_n2p3net_gtn import _postprocess_cpu_threads

    monkeypatch.delenv("POSTPROCESS_CPU_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")

    assert _postprocess_cpu_threads() == 8


def test_parent_cpu_scheduler_configures_eight_core_dispatch(monkeypatch) -> None:
    from experiments import run_n2p3net_gtn as runner

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(variable, "1")
    state = {"intraop": 16, "interop": 16}
    monkeypatch.setattr(runner.torch, "set_num_threads", lambda value: state.update(intraop=value))
    monkeypatch.setattr(runner.torch, "get_num_threads", lambda: state["intraop"])
    monkeypatch.setattr(
        runner.torch, "set_num_interop_threads", lambda value: state.update(interop=value)
    )
    monkeypatch.setattr(runner.torch, "get_num_interop_threads", lambda: state["interop"])

    configured = runner._configure_parent_cpu_scheduler(8)

    assert configured == {"intraop_threads": 8, "interop_threads": 8}
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_NUM_THREADS"] == "8"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"



def make_event_timeline(
    digits: np.ndarray,
    subjects: np.ndarray,
    *,
    statuses: np.ndarray | None = None,
    online_causal: bool = False,
) -> ScheduledEventTimeline:
    digits = np.asarray(digits, dtype=np.int64)
    subjects = np.asarray(subjects).astype(str)
    n_events = len(digits)
    statuses = (
        np.repeat("available", n_events) if statuses is None else np.asarray(statuses).astype(str)
    )
    available = statuses == "available"
    evidence = np.full(n_events, -1, dtype=np.int64)
    evidence[available] = np.arange(int(available.sum()))
    onset = np.arange(n_events, dtype=float)
    available_at = np.full(n_events, np.nan)
    available_at[available] = onset[available] + 1.0
    return ScheduledEventTimeline(
        event_ids=np.asarray([f"event:{index}" for index in range(n_events)]),
        group_ids=subjects,
        subject_ids=subjects,
        stimulus_ids=digits,
        onset_samples=np.arange(n_events, dtype=np.int64),
        onset_times_s=onset,
        evidence_available_times_s=available_at,
        evidence_indices=evidence,
        statuses=statuses,
        status_details=np.repeat("", n_events),
        dataset_ids=np.repeat("synthetic", n_events),
        session_ids=np.repeat("", n_events),
        run_ids=np.repeat("", n_events),
        selection_ids=subjects,
        complete=True,
        online_causal=online_causal,
        timing_source="synthetic_schedule",
    ).validate(n_epochs=int(available.sum()))


def test_epoch_trajectory_audit_records_auc_and_digit_hits_without_selection(tmp_path):
    digits = np.tile(np.arange(1, 10), 3)
    subjects = np.repeat("s0", len(digits))
    labels = (digits == 3).astype(np.int64)
    good_logits = np.where(labels == 1, 4.0, -1.0)

    class TrajectoryModel:
        def predict_epoch_trajectory_logits(self, X):
            return [
                {
                    "epoch": 1,
                    "phase": "joint",
                    "task_val_loss": 0.8,
                    "objective_val_loss": 1.2,
                    "val_innovation_nll": 0.4,
                    "checkpoint": str(tmp_path / "epoch_001.pt"),
                    "logits": good_logits,
                }
            ]

    rows = _evaluate_epoch_trajectory_audit(
        TrajectoryModel(),
        np.zeros((len(digits), 1, 1), dtype=np.float32),
        labels,
        digits,
        subjects,
        {"s0": 3},
        tuple(range(1, 10)),
        True,
        "sum",
        (EvidenceBudget("prefix_minK", 3),),
        make_event_timeline(digits, subjects),
        np.arange(len(digits)),
    )

    assert len(rows) == 1
    assert rows[0]["test_trial_auc"] == 1.0
    assert rows[0]["test_all_digit_hit_rate"] == 1.0
    assert rows[0]["test_prefix_minK_sum_at_3_hit_rate"] == 1.0
    assert rows[0]["diagnostic_only"] is True
    assert rows[0]["prohibited_use"] == "outer_test_metrics_must_not_select_checkpoints"
    persisted = json.loads((tmp_path / "trajectory.json").read_text(encoding="utf-8"))
    assert persisted[0]["test_trial_auc"] == 1.0


def test_epoch_trajectory_and_branch_audits_forward_test_channel_mask() -> None:
    mask = np.array([[True, False], [True, True], [True, False], [True, True]])
    X = np.zeros((4, 2, 1), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    digits = np.array([1, 2, 1, 2])
    groups = np.repeat("s0", 4)
    seen: dict[str, np.ndarray] = {}

    class AuxiliaryModel:
        auxiliary_predict_accepts_trial_channel_mask = True
        calibration_labels_ = y
        calibration_source_ = "subject_disjoint_validation"
        calibration_branch_logits_ = {
            "final": np.array([-2.0, 2.0, -1.0, 1.0]),
            "pcw": np.array([-2.0, 2.0, -1.0, 1.0]),
        }

        def predict_epoch_trajectory_logits(self, values, trial_channel_mask=None):
            seen["trajectory"] = np.asarray(trial_channel_mask).copy()
            return [{"epoch": 0, "checkpoint": "in_memory", "logits": 2.0 * y - 1.0}]

        def predict_branches(self, values, trial_channel_mask=None):
            seen["branches"] = np.asarray(trial_channel_mask).copy()
            scores = 2.0 * y - 1.0
            return {"final": scores, "pcw": scores}

        def predict_interpretability(self, values, trial_channel_mask=None):
            seen["interpretability"] = np.asarray(trial_channel_mask).copy()
            return {
                "tau": np.tile([200.0, 300.0, 400.0], (len(values), 1)),
                "tau_perturbed": np.tile([201.0, 301.0, 401.0], (len(values), 1)),
                "tau_bounds": np.array([[150.0, 250.0], [250.0, 350.0], [350.0, 500.0]]),
            }

    timeline = make_event_timeline(digits, groups)
    model = AuxiliaryModel()
    _evaluate_epoch_trajectory_audit(
        model,
        X,
        y,
        digits,
        groups,
        {"s0": 2},
        (1, 2),
        True,
        "sum",
        (),
        timeline,
        np.arange(4),
        mask,
    )
    _neural_ride_fold_audit(
        model,
        X,
        y,
        X,
        y,
        digits,
        groups,
        {"s0": 2},
        (1, 2),
        timeline,
        np.arange(4),
        mask,
    )

    assert np.array_equal(seen["trajectory"], mask)
    assert np.array_equal(seen["branches"], mask)
    assert np.array_equal(seen["interpretability"], mask)


def test_exact_k_and_prefix_min_k_have_distinct_semantics() -> None:
    digits = np.asarray([1, 1, 1, 2, 3])
    subjects = np.zeros(5, dtype=int)
    timeline = make_event_timeline(digits, subjects)
    exact = evidence_row_indices(timeline, "0", (1, 2, 3), EvidenceBudget("exact", 1))
    prefix = evidence_row_indices(timeline, "0", (1, 2, 3), EvidenceBudget("prefix_minK", 1))
    assert exact.tolist() == [0, 3, 4]
    assert prefix.tolist() == [0, 1, 2, 3, 4]

    logits = np.asarray([0.0, 10.0, 10.0, 5.0, 4.0])
    exact_result = decide(
        logits[exact], digits[exact], np.repeat("0", len(exact)), (1, 2, 3), center_logits=False
    )
    prefix_result = decide(
        logits[prefix],
        digits[prefix],
        np.repeat("0", len(prefix)),
        (1, 2, 3),
        center_logits=False,
    )
    assert exact_result.predicted.tolist() == [2]
    assert prefix_result.predicted.tolist() == [1]


def test_rejected_events_advance_flash_budget() -> None:
    timeline = make_event_timeline(
        np.asarray([1, 1, 2]),
        np.asarray(["s", "s", "s"]),
        statuses=np.asarray(["artifact_rejected", "available", "available"]),
    )
    assert evidence_row_indices(timeline, "s", (1, 2), EvidenceBudget("flash", 2)) is None
    selected = evidence_row_indices(timeline, "s", (1, 2), EvidenceBudget("flash", 3))
    assert selected.tolist() == [0, 1]


def test_time_budget_rejects_acausal_preprocessing() -> None:
    timeline = make_event_timeline(np.asarray([1, 2]), np.asarray(["s", "s"]))
    with pytest.raises(ValueError, match="not online-causal"):
        evidence_row_indices(timeline, "s", (1, 2), EvidenceBudget("time", 3.0))


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


def test_fold_validation_rejects_overlap_and_incomplete_loso() -> None:
    subjects = np.asarray(["a", "a", "b", "b"])
    same = subjects == "a"
    with pytest.raises(ValueError, match="overlapping"):
        validate_outer_folds([(same, same)], subjects, protocol="loso")
    with pytest.raises(ValueError, match="every available row"):
        validate_outer_folds([(~same, same)], subjects, protocol="loso")
    with pytest.raises(ValueError, match="more than one outer test fold"):
        validate_outer_folds([(~same, same), (~same, same)], subjects, protocol="partial_loso")


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
    summary = evaluate(
        model,
        X,
        y,
        digits,
        subject_ids,
        true_digits,
        folds,
        event_timeline=make_event_timeline(digits, subject_ids),
    )

    assert len(summary.per_fold) == 8
    assert summary.hit_rate_mean > 0.5, (
        f"LOSO 命中率应明显 > chance 11.1%，得到 {summary.hit_rate_mean:.3f}"
    )
    assert summary.balanced_acc_mean > 0.7, (
        f"单试次 balanced acc 应 >0.7，得到 {summary.balanced_acc_mean:.3f}"
    )
    assert 0.0 <= summary.hit_rate_std <= 1.0
    assert summary.per_fold[0].threshold_source == "subject_disjoint_validation"
    assert "exact_llr@5" in summary.decision_metrics
    assert summary.decision_metrics["exact_llr@5"].coverage == 1.0
    assert summary.decision_metrics["exact_sum@10"].coverage == 0.0
    assert set(summary.confound_baselines) == {"digit_prior", "count_only"}
    assert summary.repetition_efficiency is not None
    assert [point.repetitions for point in summary.repetition_efficiency.points] == [
        1,
        3,
        5,
        10,
        15,
    ]
    assert summary.repetition_efficiency.aggregation == "llr"


def test_itt_keeps_unavailable_unit_in_frozen_denominator() -> None:
    class PerfectScore:
        def fit(self, X_, y_):
            return self

        def predict_logit(self, X_):
            return np.asarray(X_)[:, 0, 0]

    available_subjects = np.repeat(np.arange(9).astype(str), 2)
    available_digits = np.tile(np.asarray([1, 2]), 9)
    all_subjects = np.concatenate([available_subjects, np.asarray(["9", "9"])])
    all_digits = np.concatenate([available_digits, np.asarray([1, 2])])
    statuses = np.concatenate(
        [np.repeat("available", len(available_subjects)), np.repeat("missing", 2)]
    )
    timeline = make_event_timeline(all_digits, all_subjects, statuses=statuses)
    X = np.zeros((len(available_subjects), 1, 1), dtype=np.float32)
    X[:, 0, 0] = np.where(available_digits == 1, 3.0, -3.0)
    y = (available_digits == 1).astype(np.int64)
    truth = {str(unit): 1 for unit in range(10)}

    summary = evaluate(
        PerfectScore(),
        X,
        y,
        available_digits,
        available_subjects,
        truth,
        loso_folds(available_subjects),
        digit_vocab=(1, 2),
        evidence_budgets=(1,),
        primary_decision_metric="exact_llr@1",
        event_timeline=timeline,
    )

    primary = summary.decision_metrics["exact_llr@1"]
    assert primary.n_total == 10 and primary.n_covered == 9
    assert primary.coverage == pytest.approx(0.9)
    assert primary.conditional_hit_rate == 1.0
    assert primary.hit_rate == pytest.approx(0.9)
    assert primary.subject_records[-1] == (None, 1, "9")


def test_evaluate_empty_train_raises():
    X, y, digits, subject_ids, true_digits = make_multi_subject(n_subjects=3, trials_per_digit=2)
    empty_train = (np.zeros(len(y), dtype=bool), np.ones(len(y), dtype=bool))
    import pytest

    with pytest.raises(ValueError):
        evaluate(
            WindowLogisticRegression(),
            X,
            y,
            digits,
            subject_ids,
            true_digits,
            [empty_train],
            event_timeline=make_event_timeline(digits, subject_ids),
        )


def test_evaluate_passes_gtn_trial_context_to_capable_model():
    seen = {}

    class ContextModel:
        fit_accepts_trial_context = True

        def fit(self, X_, y_, subject_ids=None, digits=None):
            seen["subjects"] = np.asarray(subject_ids).copy()
            seen["digits"] = np.asarray(digits).copy()
            self.calibration_logits_ = np.asarray(y_, dtype=float) * 2.0 - 1.0
            self.calibration_labels_ = np.asarray(y_)
            self.calibration_source_ = "outer_train"
            return self

        def predict_logit(self, X_):
            return np.asarray(X_)[:, 0, 0]

    X, y, digits, subjects, truth = make_multi_subject(n_subjects=3, trials_per_digit=2)
    # The score itself is irrelevant; this test protects the training metadata route.
    X[:, 0, 0] = y * 2.0 - 1.0
    train = subjects != 2
    test = subjects == 2
    evaluate(
        ContextModel(),
        X,
        y,
        digits,
        subjects,
        truth,
        [(train, test)],
        event_timeline=make_event_timeline(digits, subjects),
        fold_protocol="partial_loso",
    )
    assert np.array_equal(seen["subjects"], subjects[train].astype(str))
    assert np.array_equal(seen["digits"], digits[train])


def test_evaluate_aggregates_chain_llr_and_repetition_audit(tmp_path):
    class RepetitionModel:
        fit_accepts_trial_context = True

        def fit(self, X_, y_, subject_ids=None, digits=None):
            self.calibration_logits_ = np.asarray(y_, dtype=float) * 4.0 - 2.0
            self.calibration_labels_ = np.asarray(y_)
            self.calibration_source_ = "subject_disjoint_validation"
            self.repetition_temperature_calibration_ = SimpleNamespace(
                temperature=1.25,
                source="subject_disjoint_validation",
                pos_weight=8.0,
                train_prior=1.0 / 9.0,
            )
            self.repetition_ready_ = True
            return self

        def predict_logit(self, X_):
            return np.asarray(X_)[:, 0, 0]

        def predict_repetition_candidates(
            self, X_, digits_, subject_ids_, *, digit_vocab, evidence_budgets
        ):
            subject = np.unique(subject_ids_)[0]
            scores = np.arange(len(digit_vocab), dtype=float)[None]
            return {
                "prefix_minK_chain_llr@1": {
                    "predicted": np.asarray([digit_vocab[-1]], dtype=object),
                    "subject_ids": np.asarray([subject], dtype=object),
                    "scores": scores,
                    "mean_reliability": np.asarray([0.75]),
                },
                "prefix_minK_chain_llr@2": {
                    "predicted": np.asarray([], dtype=object),
                    "subject_ids": np.asarray([], dtype=object),
                    "scores": np.empty((0, len(digit_vocab))),
                    "mean_reliability": np.asarray([]),
                },
            }

    X, y, digits, subjects, truth = make_multi_subject(n_subjects=3, trials_per_digit=2)
    X[:, 0, 0] = y * 4.0 - 2.0
    truth[2] = 9
    train = subjects != 2
    test = subjects == 2
    summary = evaluate(
        RepetitionModel(),
        X,
        y,
        digits,
        subjects,
        truth,
        [(train, test)],
        evidence_budgets=(1, 2),
        primary_decision_metric="prefix_minK_chain_llr@1",
        event_timeline=make_event_timeline(digits, subjects),
        fold_protocol="partial_loso",
    )

    chain_one = summary.decision_metrics["prefix_minK_chain_llr@1"]
    assert chain_one.hit_rate == 1.0
    assert chain_one.coverage == 1.0
    assert chain_one.aggregation == "chain_llr"
    assert summary.decision_metrics["prefix_minK_chain_llr@2"].coverage == 0.0
    assert summary.primary_decision_metric == "prefix_minK_chain_llr@1"
    assert summary.primary_metric_gate["claim_eligible"] is True
    assert summary.primary_metric_gate["checks"]["minimum_availability_coverage"]["passed"] is True
    assert summary.primary_metric_gate["repetition_fold_readiness"] == {
        "attempted_folds": 1,
        "ready_folds": 1,
        "unready_folds": [],
    }
    assert summary.repetition_efficiency.aggregation == "chain_llr"
    assert summary.repetition_efficiency.budget_semantics == "prefix_minK"
    assert [point.repetitions for point in summary.repetition_efficiency.points] == [1, 2]

    audit = summary.per_fold[0].audit["repetition"]
    assert audit["temperature"] == 1.25
    assert audit["pos_weight"] == 8.0
    assert audit["train_prior"] == 1.0 / 9.0
    assert audit["metrics"]["prefix_minK_chain_llr@1"]["mean_reliability"] == 0.75
    assert audit["metrics"]["prefix_minK_chain_llr@1"]["mean_top1_log_score_margin"] == 1.0

    class UnreadyRepetitionModel(RepetitionModel):
        def predict_repetition_candidates(
            self, X_, digits_, subject_ids_, *, digit_vocab, evidence_budgets
        ):
            self.repetition_ready_ = False
            return super().predict_repetition_candidates(
                X_, digits_, subject_ids_, digit_vocab=digit_vocab, evidence_budgets=evidence_budgets
            )

    unready = evaluate(
        UnreadyRepetitionModel(),
        X,
        y,
        digits,
        subjects,
        truth,
        [(train, test)],
        evidence_budgets=(1, 2),
        primary_decision_metric="prefix_minK_chain_llr@1",
        event_timeline=make_event_timeline(digits, subjects),
        fold_protocol="partial_loso",
    )
    assert unready.decision_metrics["prefix_minK_chain_llr@1"].coverage == 0.0
    gate = unready.primary_metric_gate
    assert gate["applicable"] is True
    assert gate["passed"] is False
    assert gate["claim_eligible"] is False
    assert gate["name"] == "primary_metric_claim_gate"
    assert gate["effect"] == "descriptive_only_no_result_suppression"
    assert gate["failed_checks"] == ["minimum_availability_coverage"]
    assert gate["checks"]["minimum_availability_coverage"] == {
        "passed": False,
        "observed": 0.0,
        "minimum": 0.9,
        "n_covered": 0,
        "n_total": 1,
    }
    assert gate["repetition_fold_readiness"]["attempted_folds"] == 1
    assert gate["repetition_fold_readiness"]["ready_folds"] == 0
    assert gate["repetition_fold_readiness"]["unready_folds"][0]["ready"] is False

    descriptive = unready.per_fold[0].audit["repetition"]["metrics"]["prefix_minK_chain_llr@1"]
    assert descriptive["formal_eligible"] is False
    assert descriptive["formal_covered"] == 0
    assert descriptive["descriptive_covered"] == 1
    descriptive_records = unready.descriptive_decision_records["prefix_minK_chain_llr@1"]
    assert descriptive_records == [
        {
            "subject": "2",
            "predicted": 9,
            "true": 9,
            "available": True,
            "hit": 1,
            "scores": list(np.arange(9, dtype=float)),
            "mean_reliability": 0.75,
            "claim_eligible": False,
            "formal_available": False,
        }
    ]
    from experiments.run_gtn_baseline import save_subject_scores

    score_path = save_subject_scores(unready, "repetition", tmp_path / "scores.json")
    score_payload = json.loads(score_path.read_text(encoding="utf-8"))
    assert score_payload["primary_records"][0]["available"] is False
    assert score_payload["descriptive_primary_records"] == descriptive_records
    assert score_payload["descriptive_records_by_metric"][
        "prefix_minK_chain_llr@1"
    ] == descriptive_records


def test_primary_coverage_threshold_is_independent_from_efficiency_threshold():
    class ReadyRepetitionModel:
        fit_accepts_trial_context = True

        def fit(self, X_, y_, subject_ids=None, digits=None):
            self.calibration_logits_ = np.asarray(y_, dtype=float) * 4.0 - 2.0
            self.calibration_labels_ = np.asarray(y_)
            self.calibration_source_ = "subject_disjoint_validation"
            self.repetition_ready_ = True
            return self

        def predict_logit(self, X_):
            return np.asarray(X_)[:, 0, 0]

        def predict_repetition_candidates(
            self, X_, digits_, subject_ids_, *, digit_vocab, evidence_budgets
        ):
            subject = np.unique(subject_ids_)[0]
            return {
                "prefix_minK_chain_llr@1": {
                    "predicted": np.asarray([digit_vocab[-1]], dtype=object),
                    "subject_ids": np.asarray([subject], dtype=object),
                    "scores": np.arange(len(digit_vocab), dtype=float)[None],
                    "mean_reliability": np.asarray([0.8]),
                    "claim_eligible": True,
                }
            }

    X, y, digits, subjects, truth = make_multi_subject(n_subjects=3, trials_per_digit=1)
    X[:, 0, 0] = y * 4.0 - 2.0
    truth[2] = 9
    train = subjects != 2
    test = subjects == 2

    summary = evaluate(
        ReadyRepetitionModel(),
        X,
        y,
        digits,
        subjects,
        truth,
        [(train, test)],
        evidence_budgets=(1,),
        primary_decision_metric="prefix_minK_chain_llr@1",
        primary_min_coverage=1.0,
        efficiency_min_coverage=0.25,
        event_timeline=make_event_timeline(digits, subjects),
        fold_protocol="partial_loso",
    )

    assert summary.primary_metric_gate["minimum_coverage"] == 1.0
    assert summary.repetition_efficiency.minimum_coverage == 0.25


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
        evaluate(
            WindowLogisticRegression(),
            X,
            y,
            digits,
            subject_ids,
            true_digits,
            folds,
            event_timeline=make_event_timeline(digits, subject_ids),
        )


def test_outer_prequential_claim_gate_is_claim_only_and_fail_closed():
    from experiments.run_n2p3net_gtn import _outer_prequential_claim_gate

    def fold(delta_auc: float, delta_brier: float, coefficient: float = 0.2):
        return SimpleNamespace(
            audit={
                "branches": {
                    "pcw": {"auc": 0.7, "brier": 0.2},
                    "final": {"auc": 0.7 + delta_auc, "brier": 0.2 + delta_brier},
                },
                "prequential_gate": {"passed": True},
                "prequential_coefficient": coefficient,
            }
        )

    passed = _outer_prequential_claim_gate([fold(0.01, -0.01) for _ in range(5)])
    assert passed["passed"] is True
    failed = _outer_prequential_claim_gate(
        [fold(0.01, -0.01), fold(0.0, 0.0, 0.0), fold(-0.01, 0.01, 0.0)]
    )
    assert failed["passed"] is False
    assert failed["checks"]["at_least_five_outer_folds"] is False
    assert failed["checks"]["fusion_active_in_strict_majority"] is False


def test_composite_group_key():
    """分组键支持 (subject, run)（D-group-key）：每 run 一个心选数字，用字符串复合键。"""
    rng = np.random.default_rng(0)
    i0, i1 = time_to_index(300, SFR, TMIN), time_to_index(500, SFR, TMIN)
    center = (i0 + i1) / 2
    width = (i1 - i0) / 6
    gauss = np.exp(-0.5 * ((np.arange(T) - center) / width) ** 2)

    X_list, y_list, d_list, s_list = [], [], [], []
    true_digits = {}
    for subject in range(5):
        for run, true_d in [(0, 5), (1, 8)]:
            key = f"s{subject}_r{run}"
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
        X,
        y,
        digits,
        subject_ids,
        true_digits,
        folds,
        event_timeline=make_event_timeline(digits, subject_ids),
    )
    # 五名被试的两个 run 均作为独立选择单元计数。
    assert len(summary.subject_records) == 10
    assert 0.0 <= summary.hit_rate_mean <= 1.0


def test_loso_rejects_two_selection_groups_from_same_physical_subject() -> None:
    groups = np.repeat(np.asarray(["s0_run1", "s0_run2", "s1_run1"]), 2)
    physical_subjects = np.asarray(["s0", "s0", "s0", "s0", "s1", "s1"])
    digits = np.tile(np.asarray([1, 2]), 3)
    timeline = ScheduledEventTimeline(
        event_ids=np.asarray([f"e{index}" for index in range(6)]),
        group_ids=groups,
        subject_ids=physical_subjects,
        stimulus_ids=digits,
        onset_samples=np.arange(6),
        onset_times_s=np.arange(6, dtype=float),
        evidence_available_times_s=np.arange(6, dtype=float) + 1.0,
        evidence_indices=np.arange(6),
        statuses=np.repeat("available", 6),
        status_details=np.repeat("", 6),
        dataset_ids=np.repeat("synthetic", 6),
        session_ids=np.repeat("session", 6),
        run_ids=groups,
        selection_ids=groups,
        complete=True,
        online_causal=False,
        timing_source="synthetic",
    ).validate(n_epochs=6)
    X = np.zeros((6, 1, 4), dtype=np.float32)
    y = (digits == 1).astype(np.int64)
    truth = {group: 1 for group in np.unique(groups)}

    with pytest.raises(ValueError, match="leaks test subjects"):
        evaluate(
            WindowLogisticRegression(),
            X,
            y,
            digits,
            groups,
            truth,
            loso_folds(groups),
            digit_vocab=(1, 2),
            evidence_budgets=(1,),
            primary_decision_metric="exact_llr@1",
            event_timeline=timeline,
        )


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


def test_optional_mask_capability_does_not_make_mask_mandatory() -> None:
    class OptionalMaskModel:
        fit_accepts_subject_ids = True
        fit_accepts_trial_channel_mask = True
        predict_accepts_trial_channel_mask = True

        def fit(self, X_, y_, subject_ids=None, trial_channel_mask=None):
            assert trial_channel_mask is None
            self.calibration_logits_ = np.array([-2.0, -1.0, 1.0, 2.0])
            self.calibration_labels_ = np.array([0, 0, 1, 1])
            self.calibration_source_ = "subject_disjoint_validation"
            return self

        def predict_logit(self, X_, trial_channel_mask=None):
            assert trial_channel_mask is None
            return np.asarray(X_)[:, 0, 0]

    X = np.array([-2.0, 2.0, -1.0, 1.0, -3.0, 3.0], dtype=np.float32)[:, None, None]
    y = np.array([0, 1, 0, 1, 0, 1])
    subjects = np.repeat(np.arange(3), 2)

    result = evaluate_binary(
        OptionalMaskModel(),
        X,
        y,
        subjects,
        [(subjects != 2, subjects == 2)],
        fold_protocol="partial_loso",
    )
    assert result.per_fold[0].auc == 1.0


def test_serial_folds_clone_pristine_prototype_state() -> None:
    class StatefulModel:
        fit_accepts_subject_ids = True
        starts: list[int] = []

        def __init__(self):
            self.fit_count = 0

        def fit(self, X_, y_, subject_ids=None):
            self.starts.append(self.fit_count)
            self.fit_count += 1
            self.calibration_logits_ = np.array([-2.0, -1.0, 1.0, 2.0])
            self.calibration_labels_ = np.array([0, 0, 1, 1])
            self.calibration_source_ = "subject_disjoint_validation"
            return self

        def predict_logit(self, X_):
            return np.asarray(X_)[:, 0, 0]

    X = np.tile(np.array([-1.0, 1.0], dtype=np.float32), 3)[:, None, None]
    y = np.tile([0, 1], 3)
    subjects = np.repeat(np.arange(3), 2)
    prototype = StatefulModel()

    summary = evaluate_binary(prototype, X, y, subjects, loso_folds(subjects), n_jobs=1)

    assert len(summary.per_fold) == 3
    assert StatefulModel.starts == [0, 0, 0]
    assert prototype.fit_count == 0


def test_binary_evaluation_rejects_column_vector_logits() -> None:
    class BadShapeModel:
        fit_accepts_subject_ids = True

        def fit(self, X_, y_, subject_ids=None):
            self.calibration_logits_ = np.array([-2.0, -1.0, 1.0, 2.0])
            self.calibration_labels_ = np.array([0, 0, 1, 1])
            self.calibration_source_ = "subject_disjoint_validation"
            return self

        def predict_logit(self, X_):
            return np.asarray(X_)[:, 0, 0, None]

    X = np.tile(np.array([-1.0, 1.0], dtype=np.float32), 3)[:, None, None]
    y = np.tile([0, 1], 3)
    subjects = np.repeat(np.arange(3), 2)
    with pytest.raises(ValueError, match=r"must return shape \(2,\)"):
        evaluate_binary(
            BadShapeModel(),
            X,
            y,
            subjects,
            [(subjects != 2, subjects == 2)],
            fold_protocol="partial_loso",
        )


def test_all_single_class_test_folds_report_nan_summary() -> None:
    class CalibratedModel:
        fit_accepts_subject_ids = True

        def fit(self, X_, y_, subject_ids=None):
            self.calibration_logits_ = np.array([-2.0, -1.0, 1.0, 2.0])
            self.calibration_labels_ = np.array([0, 0, 1, 1])
            self.calibration_source_ = "subject_disjoint_validation"
            return self

        def predict_logit(self, X_):
            return np.asarray(X_)[:, 0, 0]

    subjects = np.repeat(np.arange(4), 2)
    y = np.repeat([0, 1, 0, 1], 2)
    X = (2.0 * y.astype(np.float32) - 1.0)[:, None, None]

    summary = evaluate_binary(CalibratedModel(), X, y, subjects, loso_folds(subjects))

    assert all(np.isnan(fold.balanced_acc) for fold in summary.per_fold)
    assert np.isnan(summary.balanced_acc_mean)
    assert np.isnan(summary.balanced_acc_std)
    assert np.isnan(summary.auc_mean)


def test_evaluate_binary_forwards_acquisition_indices_to_capable_models():
    from baselines.evaluate import evaluate_binary

    class AcquisitionAwareModel:
        fit_accepts_trial_context = True
        fit_accepts_group_ids = True
        fit_accepts_acquisition_indices = True
        seen = []

        def fit(self, X_, y_, subject_ids=None, group_ids=None, acquisition_indices=None, digits=None):
            self.seen.append(np.asarray(acquisition_indices).copy())
            self.calibration_logits_ = np.array([-2.0, -1.0, 1.0, 2.0])
            self.calibration_labels_ = np.array([0, 0, 1, 1])
            self.calibration_source_ = "subject_disjoint_validation"
            return self

        def predict_logit(self, X_):
            return np.asarray(X_)[:, 0, 0]

    X = np.zeros((6, 1, 1), dtype=np.float32)
    X[:, 0, 0] = np.array([-2.0, -1.0, 1.0, 2.0, -1.5, 1.5])
    y = np.array([0, 0, 1, 1, 0, 1])
    subjects = np.array([0, 0, 1, 1, 2, 2])
    acquisition = np.array([4, 5, 0, 1, 2, 3], dtype=np.int64)
    model = AcquisitionAwareModel()

    summary = evaluate_binary(
        model,
        X,
        y,
        subjects,
        [(subjects != 2, subjects == 2)],
        acquisition_indices=acquisition,
        fold_protocol="partial_loso",
    )

    assert summary.per_fold[0].auc > 0.9
    assert np.array_equal(AcquisitionAwareModel.seen[-1], acquisition[:4])
    assert not hasattr(model, "seen_acquisition_indices")


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


def test_binary_evaluation_requires_validation_calibration():
    """Small folds fail closed instead of calibrating on outer-train scores."""
    from baselines.evaluate import evaluate_binary

    class ScoreModel:
        def fit(self, X_, y_):
            return self

        def predict_logit(self, X_):
            return np.asarray(X_)[:, 0, 0]

    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    subjects = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    X = np.zeros((8, 1, 1), dtype=np.float32)
    X[:, 0, 0] = np.array([-2.0, -1.0, 1.0, 2.0, -1.5, -0.5, 0.5, 1.5])
    fold = [(subjects == 0, subjects == 1)]
    with pytest.raises(ValueError, match="Subject-disjoint validation requires"):
        evaluate_binary(ScoreModel(), X, y, subjects, fold, fold_protocol="partial_loso")


def test_model_validation_scores_take_priority_for_threshold():
    from baselines.evaluate import evaluate_binary

    class ValidationAwareModel:
        fit_accepts_subject_ids = True

        def fit(self, X_, y_, subject_ids=None):
            self.calibration_logits_ = np.array([-4.0, -3.0, 3.0, 4.0])
            self.calibration_labels_ = np.array([0, 0, 1, 1])
            self.calibration_source_ = "subject_disjoint_validation"
            return self

        def predict_logit(self, X_):
            return np.asarray(X_)[:, 0, 0]

    y = np.array([0, 1, 0, 1, 0, 1])
    subjects = np.array([0, 0, 1, 1, 2, 2])
    X = np.zeros((6, 1, 1), dtype=np.float32)
    fold = [(subjects != 2, subjects == 2)]
    result = evaluate_binary(
        ValidationAwareModel(), X, y, subjects, fold, fold_protocol="partial_loso"
    )
    assert result.per_fold[0].threshold_source == "subject_disjoint_validation"
