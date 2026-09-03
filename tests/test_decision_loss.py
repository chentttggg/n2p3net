"""Counterexamples for direct-membership P2 training and grouped batches."""

from __future__ import annotations

import json
from dataclasses import replace
from types import MethodType

import numpy as np
import pytest
import torch
import torch.nn as nn

from baselines.deep import DeepBaseline, DeepConfig
from baselines.n2p3net import N2P3NetBaseline
from data.candidate_task import CandidateTaskContract
from data.events import AVAILABLE, ScheduledEventTimeline
from models.adapters import FeatureTransferConfig
from train.runtime import GpuPerformanceScheduler, MatrixBatchSource
from transfer.decision_loss import (
    DECISION_TRAINING_LEDGER_SCHEMA,
    DecisionLevelLoss,
    DecisionLossConfig,
    DirectDecisionContractError,
    DirectDecisionMetadata,
    build_decision_batch_plan,
)


def _timeline(
    *,
    decision_count: int = 5,
    vocabulary: tuple[str, ...] = ("0", "1"),
    repetitions: int = 32,
) -> tuple[ScheduledEventTimeline, np.ndarray]:
    groups: list[str] = []
    candidates: list[str] = []
    targets: list[str] = []
    repetition_indices: list[int] = []
    subjects: list[str] = []
    selection_ids: list[str] = []
    for decision in range(decision_count):
        group = f"decision-{decision}"
        truth = vocabulary[decision % len(vocabulary)]
        for repetition in range(repetitions):
            for candidate in vocabulary:
                groups.append(group)
                candidates.append(candidate)
                targets.append(truth)
                repetition_indices.append(repetition)
                subjects.append(f"subject-{decision // 2}")
                selection_ids.append(str(decision))
    count = len(groups)
    timeline = ScheduledEventTimeline(
        event_ids=np.asarray([f"event-{index}" for index in range(count)]),
        group_ids=np.asarray(groups),
        subject_ids=np.asarray(subjects),
        stimulus_ids=np.arange(count, dtype=np.int64),
        onset_samples=np.arange(count, dtype=np.int64),
        onset_times_s=np.arange(count, dtype=np.float64),
        evidence_available_times_s=np.arange(count, dtype=np.float64) + 0.5,
        evidence_indices=np.arange(count, dtype=np.int64),
        statuses=np.repeat(AVAILABLE, count),
        status_details=np.repeat("", count),
        dataset_ids=np.repeat("synthetic-direct", count),
        session_ids=np.repeat("session", count),
        run_ids=np.repeat("run", count),
        selection_ids=np.asarray(selection_ids),
        complete=True,
        online_causal=True,
        timing_source="synthetic_test_clock",
        candidate_ids=np.asarray(candidates),
        target_candidate_ids=np.asarray(targets),
        repetition_indices=np.asarray(repetition_indices, dtype=np.int64),
    ).validate(n_epochs=count)
    labels = (np.asarray(candidates) == np.asarray(targets)).astype(np.int64)
    return timeline, labels


def _metadata(
    timeline: ScheduledEventTimeline,
    *,
    requested_row_mask: np.ndarray | None = None,
    retained_row_mask: np.ndarray | None = None,
    repetitions: int = 32,
) -> DirectDecisionMetadata:
    requested = (
        np.ones(timeline.n_available, dtype=bool)
        if requested_row_mask is None
        else requested_row_mask
    )
    retained = requested if retained_row_mask is None else retained_row_mask
    return DirectDecisionMetadata.from_timeline(
        timeline,
        requested_row_mask=requested,
        retained_row_mask=retained,
        max_repetitions=repetitions,
        source_event_contract="preserve",
    )


def _config(
    *,
    vocabulary: tuple[str, ...] = ("0", "1"),
    repetitions: int = 32,
    lambda_candidate: float = 0.1,
    prefix_weight_mode: str = "final",
) -> DecisionLossConfig:
    return DecisionLossConfig(
        candidate_vocabulary=vocabulary,
        max_repetitions=repetitions,
        lambda_candidate=lambda_candidate,
        temperature=0.7,
        prefix_weight_mode=prefix_weight_mode,
    )


def test_decision_level_forward_and_incomplete_coverage_are_typed() -> None:
    module = DecisionLevelLoss(num_candidates=3, temperature=0.8)
    evidence = torch.tensor([2.0, 4.0, 1.0, -1.0], requires_grad=True)
    membership = torch.tensor([0, 0, 1, 2])

    result = module(evidence, membership, 0, evidence_level=2)

    expected_scores = torch.tensor([3.0, 1.0, -1.0])
    assert result.status == "complete"
    assert result.loss is not None and result.posterior is not None
    assert torch.allclose(result.scores, expected_scores)
    assert torch.allclose(result.posterior, torch.softmax(expected_scores / 0.8, dim=0))
    assert result.argmax_candidate == "0"
    json.dumps(result.record(), allow_nan=False)

    incomplete = module(evidence[:3], membership[:3], 0)
    assert incomplete.status == "incomplete"
    assert incomplete.missing_candidates == ("2",)
    assert incomplete.scores[-1].item() == float("-inf")
    assert incomplete.loss is None
    assert incomplete.posterior is None
    assert incomplete.argmax_candidate is None
    assert incomplete.record()["scores"][-1] == "-inf"
    json.dumps(incomplete.record(), allow_nan=False)

    nonfinite = module(torch.tensor([1.0, float("nan"), 0.0]), torch.arange(3), 0)
    assert nonfinite.status == "invalid"
    assert nonfinite.failure_reason == "evidence_contains_nan_or_infinity"
    assert nonfinite.posterior is None and nonfinite.loss is None


def test_finite_candidate_mean_overflow_is_invalid_not_incomplete() -> None:
    module = DecisionLevelLoss(num_candidates=2, temperature=1.0)
    limit = torch.finfo(torch.float32).max
    result = module(
        torch.tensor([limit, limit, 0.0], dtype=torch.float32),
        torch.tensor([0, 0, 1], dtype=torch.long),
        0,
    )
    assert result.status == "invalid"
    assert result.failure_reason == "candidate_mean_overflowed"
    assert result.posterior is None and result.loss is None


def test_temperature_autograd_includes_inverse_tau_and_count_neutral_mean() -> None:
    tau = 0.4
    module = DecisionLevelLoss(num_candidates=3, temperature=tau)
    evidence = torch.tensor([0.3, 0.7, -0.2, 0.4, -0.1, -0.5], requires_grad=True)
    membership = torch.tensor([0, 0, 1, 1, 1, 2])
    result = module(evidence, membership, 1)
    assert result.loss is not None and result.posterior is not None
    result.loss.backward()

    counts = torch.bincount(membership, minlength=3).float()
    expected = torch.empty_like(evidence)
    for index in range(len(evidence)):
        candidate = int(membership[index])
        expected[index] = (
            result.posterior[candidate] - float(candidate == 1)
        ) / (tau * counts[candidate])
    assert torch.allclose(evidence.grad, expected, atol=1e-6, rtol=1e-6)
    for candidate in range(3):
        assert torch.allclose(
            evidence.grad[membership == candidate].sum(),
            (result.posterior[candidate] - float(candidate == 1)) / tau,
        )


def test_candidate_truth_permutation_and_constant_bias_leave_loss_invariant() -> None:
    module = DecisionLevelLoss(num_candidates=4, temperature=1.3)
    evidence = torch.tensor([0.2, 1.0, -0.5, 0.3, 0.7, 0.9])
    membership = torch.tensor([0, 0, 1, 2, 2, 3])
    truth = 2
    original = module(evidence, membership, truth)
    biased = module(evidence + 91.25, membership, truth)
    assert original.loss is not None and biased.loss is not None
    assert torch.allclose(original.loss, biased.loss, atol=1e-6)
    assert torch.allclose(original.posterior, biased.posterior, atol=1e-6)

    permutation = torch.tensor([2, 0, 3, 1])
    permuted = module(evidence, permutation[membership], int(permutation[truth]))
    assert permuted.loss is not None
    assert torch.allclose(original.loss, permuted.loss, atol=1e-7)
    assert original.posterior is not None and permuted.posterior is not None
    assert torch.allclose(permuted.posterior[permutation], original.posterior, atol=1e-7)

    tied = module(torch.zeros(4), torch.arange(4), 0)
    assert tied.status == "tie"
    assert tied.loss is not None and tied.posterior is not None
    assert tied.argmax_candidate is None


def test_two_logit_common_mode_bias_cancels_before_packed_decision_loss() -> None:
    module = DecisionLevelLoss(num_candidates=2, temperature=0.9)
    logits = torch.tensor(
        [[-1.0, 0.2], [-0.5, 0.3], [0.1, -0.4], [0.2, -0.1]]
    )
    targets = torch.tensor([0])
    original, _ = module.packed_loss(
        logits[:, 1] - logits[:, 0],
        targets,
        max_repetitions=2,
        prefix_weights=(0.0, 1.0),
    )
    common_mode = torch.tensor([[7.0], [-3.0], [11.0], [2.5]])
    shifted = logits + common_mode
    shifted_loss, _ = module.packed_loss(
        shifted[:, 1] - shifted[:, 0],
        targets,
        max_repetitions=2,
        prefix_weights=(0.0, 1.0),
    )
    assert torch.allclose(original, shifted_loss, atol=1e-7)


def test_prefix_weights_cover_final_budget_and_uniform_curve() -> None:
    evidence = torch.tensor(
        [
            0.0,
            2.0,
            4.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            2.0,
            1.0,
            3.0,
            5.0,
        ]
    )
    targets = torch.tensor([0, 1])
    module = DecisionLevelLoss(num_candidates=2, temperature=1.0)
    final = _config(repetitions=3, prefix_weight_mode="final")
    uniform = _config(repetitions=3, prefix_weight_mode="uniform")

    final_loss, by_prefix = module.packed_loss(
        evidence,
        targets,
        max_repetitions=3,
        prefix_weights=final.prefix_weights,
    )
    uniform_loss, uniform_prefix = module.packed_loss(
        evidence,
        targets,
        max_repetitions=3,
        prefix_weights=uniform.prefix_weights,
    )

    assert final.prefix_weights == (0.0, 0.0, 1.0)
    assert final_loss == pytest.approx(float(by_prefix[-1]))
    assert uniform.prefix_weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert torch.equal(by_prefix, uniform_prefix)
    assert uniform_loss == pytest.approx(float(by_prefix.mean()))
    with pytest.raises(ValueError, match="cannot be empty"):
        module.packed_loss(
            torch.empty(0),
            torch.empty(0, dtype=torch.long),
            max_repetitions=3,
            prefix_weights=uniform.prefix_weights,
        )
    with pytest.raises(ValueError, match="NaN or infinity"):
        module.packed_loss(
            torch.full((6,), float("inf")),
            torch.tensor([0]),
            max_repetitions=3,
            prefix_weights=uniform.prefix_weights,
        )


@pytest.mark.parametrize("bad_lambda", [-0.1, 0.01, 0.09, 0.51, float("nan")])
def test_decision_config_rejects_nonregistered_lambda_and_trial_weight(bad_lambda: float) -> None:
    with pytest.raises(ValueError, match="lambda_candidate"):
        DecisionLossConfig(("0", "1"), 2, lambda_candidate=bad_lambda)
    with pytest.raises(ValueError, match="trial_weight is fixed"):
        DecisionLossConfig(("0", "1"), 2, trial_weight=0.0)
    with pytest.raises(ValueError, match="not implemented"):
        DecisionLossConfig(("0", "1"), 2, evidence_count_beta=0.5)
    with pytest.raises(ValueError, match="constant temperature"):
        DecisionLossConfig(("0", "1"), 2, temperature_schedule="sqrt_repetition")


def test_config_and_metadata_records_are_strictly_json_serializable() -> None:
    timeline, _ = _timeline(decision_count=2)
    metadata = _metadata(timeline)
    config = _config()
    restored = DecisionLossConfig.from_record(config.record())
    assert restored == config
    json.dumps(config.record(), allow_nan=False)
    json.dumps(metadata.record(), allow_nan=False)
    assert metadata.truth_by_decision == {"decision-0": "0", "decision-1": "1"}


def test_grouped_sampler_rebalances_tail_and_covers_every_row_once() -> None:
    timeline, labels = _timeline(decision_count=5)
    metadata = _metadata(timeline)
    plan, ledger = build_decision_batch_plan(
        metadata,
        labels,
        _config(),
        max_batch_rows=256,
        phase="training",
    )

    assert plan.batch_decision_counts == (3, 2)
    assert plan.record()["batch_row_counts"] == [192, 128]
    batches = plan.batches(seed=7)
    assert sorted(np.concatenate([batch.row_indices for batch in batches]).tolist()) == list(
        range(len(labels))
    )
    decision_to_batch = {
        decision: batch_index
        for batch_index, batch in enumerate(batches)
        for decision in batch.decision_ids
    }
    assert len(decision_to_batch) == 5
    assert ledger.status == "valid"
    json.dumps(ledger.record(), allow_nan=False)


def test_grouped_sampler_finds_nonuniform_valid_partition() -> None:
    timeline, labels = _timeline(decision_count=7)
    metadata = _metadata(timeline)
    plan, ledger = build_decision_batch_plan(
        metadata,
        labels,
        _config(),
        max_batch_rows=192,
        phase="training",
    )

    assert plan.batch_decision_counts == (3, 2, 2)
    assert plan.record()["batch_row_counts"] == [192, 128, 128]
    assert ledger.status == "valid"


def test_frozen_vocabulary_survives_scope_candidate_removal() -> None:
    timeline, labels = _timeline(decision_count=3, vocabulary=("0", "1", "2"))
    candidates = np.asarray(timeline.candidate_ids)
    requested = candidates != "2"
    metadata = _metadata(timeline, requested_row_mask=requested)

    assert metadata.candidate_vocabulary == ("0", "1", "2")
    with pytest.raises(DirectDecisionContractError) as caught:
        build_decision_batch_plan(
            metadata,
            labels[metadata.source_evidence_indices],
            _config(vocabulary=("0", "1", "2")),
            max_batch_rows=metadata.n_rows,
            phase="training",
        )
    assert caught.value.record["valid"] is False
    assert any(
        decision["missing_candidates"] == ["2"]
        for decision in caught.value.record["decisions"]
    )


def test_grouped_sampler_rejects_too_few_decisions_or_rows() -> None:
    timeline, labels = _timeline(decision_count=2, repetitions=16)
    metadata = _metadata(timeline, repetitions=16)
    with pytest.raises(DirectDecisionContractError) as caught:
        build_decision_batch_plan(
            metadata,
            labels,
            _config(repetitions=16),
            max_batch_rows=128,
            phase="training",
        )
    assert "requires_at_least_4" in caught.value.record["reason"]

    timeline_full, labels_full = _timeline(decision_count=4)
    metadata_full = _metadata(timeline_full)
    with pytest.raises(DirectDecisionContractError) as budget_error:
        build_decision_batch_plan(
            metadata_full,
            labels_full,
            _config(),
            max_batch_rows=127,
            phase="training",
        )
    assert "cannot_hold" in budget_error.value.record["reason"]


def test_qc_row_removal_retains_requested_decision_and_fails_closed() -> None:
    timeline, labels = _timeline(decision_count=3)
    row_mask = np.asarray(timeline.group_ids) != "decision-0"
    metadata = _metadata(timeline, retained_row_mask=row_mask)
    assert "decision-0" in metadata.requested_decision_ids
    with pytest.raises(DirectDecisionContractError) as caught:
        build_decision_batch_plan(
            metadata,
            labels[row_mask],
            _config(),
            max_batch_rows=len(labels),
            phase="post_qc",
        )
    decision = next(
        item for item in caught.value.record["decisions"] if item["decision_id"] == "decision-0"
    )
    assert decision["status"] == "incomplete"
    assert decision["selected_rows"] == 0
    assert decision["missing_candidates"] == ["0", "1"]
    assert caught.value.record["valid"] is False


def test_scope_qc_and_repetition_horizon_have_distinct_accounting() -> None:
    timeline, labels = _timeline(decision_count=4, repetitions=50)
    groups = np.asarray(timeline.group_ids)
    requested = groups != "decision-3"
    retained = requested & (groups != "decision-0")
    metadata = _metadata(
        timeline,
        requested_row_mask=requested,
        retained_row_mask=retained,
        repetitions=8,
    )

    assert metadata.requested_decision_ids == (
        "decision-0",
        "decision-1",
        "decision-2",
    )
    assert "decision-3" not in metadata.truth_by_decision
    record = metadata.record(include_rows=False)
    assert record["requested_rows"] == 3 * 2 * 8
    assert record["retained_rows"] == 2 * 2 * 8
    assert record["later_discarded_rows"] == 3 * 2 * (50 - 8)
    assert np.all(metadata.repetition_indices < 8)
    retained_labels = labels[metadata.source_evidence_indices]
    with pytest.raises(DirectDecisionContractError) as caught:
        build_decision_batch_plan(
            metadata,
            retained_labels,
            _config(repetitions=8),
            max_batch_rows=max(metadata.n_rows, 128),
            phase="post_qc",
        )
    missing = next(
        item for item in caught.value.record["decisions"] if item["decision_id"] == "decision-0"
    )
    assert missing["qc_removed_rows"] == 16
    assert missing["selected_rows"] == 0

    excluded_truths = np.asarray(timeline.target_candidate_ids).copy()
    excluded_truths[groups == "decision-3"] = "0"
    permuted_excluded = replace(timeline, target_candidate_ids=excluded_truths)
    permuted_metadata = _metadata(
        permuted_excluded,
        requested_row_mask=requested,
        retained_row_mask=retained,
        repetitions=8,
    )
    assert permuted_metadata.timeline_fingerprint == metadata.timeline_fingerprint
    assert permuted_metadata.truth_by_decision == metadata.truth_by_decision


def test_direct_preflight_rejects_label_relation_and_missing_repetition() -> None:
    timeline, labels = _timeline(decision_count=3)
    metadata = _metadata(timeline)
    forged = labels.copy()
    forged[0] = 1 - forged[0]
    with pytest.raises(DirectDecisionContractError) as label_error:
        build_decision_batch_plan(
            metadata,
            forged,
            _config(),
            max_batch_rows=len(labels),
            phase="training",
        )
    assert "mislabeled" in label_error.value.record["reason"]
    assert label_error.value.record["decisions"][0]["label_relation_violation_rows"] == [0]

    row_mask = np.ones(len(labels), dtype=bool)
    row_mask[0] = False
    incomplete = _metadata(timeline, retained_row_mask=row_mask)
    with pytest.raises(DirectDecisionContractError) as missing_error:
        build_decision_batch_plan(
            incomplete,
            labels[row_mask],
            _config(),
            max_batch_rows=len(labels),
            phase="post_qc",
        )
    missing_slots = missing_error.value.record["decisions"][0]["missing_slots"]
    assert {"candidate": "0", "repetition": 0} in missing_slots


def test_row_column_binary_projection_and_missing_metadata_are_ineligible() -> None:
    timeline, _ = _timeline(decision_count=2)
    row_column = CandidateTaskContract(
        dataset_id="row-column",
        task_id="character",
        population={"label": "synthetic"},
        evidence_scope={"stage": "development", "product_confirmation": False},
        membership_kind="row_column",
        grid_shape=(2, 2),
        candidate_ids=(0, 1, 2, 3),
        row_candidate_ids=(0, 1),
        column_candidate_ids=(2, 3),
        target_representation="row_column_intersection",
    )
    with pytest.raises(DirectDecisionContractError, match="not_direct_membership"):
        DirectDecisionMetadata.from_timeline(
            timeline,
            requested_row_mask=np.ones(timeline.n_available, dtype=bool),
            retained_row_mask=np.ones(timeline.n_available, dtype=bool),
            max_repetitions=32,
            source_event_contract="preserve",
            candidate_task_contract=row_column,
        )
    with pytest.raises(DirectDecisionContractError, match="binary_evidence"):
        DirectDecisionMetadata.from_timeline(
            timeline,
            requested_row_mask=np.ones(timeline.n_available, dtype=bool),
            retained_row_mask=np.ones(timeline.n_available, dtype=bool),
            max_repetitions=32,
            source_event_contract="binary_evidence",
        )
    with pytest.raises(DirectDecisionContractError, match="unknown_contract"):
        DirectDecisionMetadata.from_timeline(
            timeline,
            requested_row_mask=np.ones(timeline.n_available, dtype=bool),
            retained_row_mask=np.ones(timeline.n_available, dtype=bool),
            max_repetitions=32,
            source_event_contract="unknown_contract",
        )
    missing = replace(timeline, candidate_ids=None, target_candidate_ids=None, repetition_indices=None)
    with pytest.raises(DirectDecisionContractError, match="lacks_candidate"):
        DirectDecisionMetadata.from_timeline(
            missing,
            requested_row_mask=np.ones(missing.n_available, dtype=bool),
            retained_row_mask=np.ones(missing.n_available, dtype=bool),
            max_repetitions=32,
            source_event_contract="preserve",
        )


def test_integer_metadata_axes_forbid_lossy_coercion() -> None:
    timeline, _ = _timeline(decision_count=2)
    metadata = _metadata(timeline)
    with pytest.raises(ValueError, match="coercion is forbidden"):
        replace(metadata, candidate_indices=metadata.candidate_indices.astype(float) + 0.5)


def test_matrix_batch_source_gathers_all_row_axes_in_lockstep() -> None:
    runtime = GpuPerformanceScheduler(torch.device("cpu"), precision="fp32")
    source = MatrixBatchSource(
        torch.arange(24, dtype=torch.float32).reshape(6, 4),
        torch.arange(6, dtype=torch.long),
        runtime,
        preload=False,
        row_weights=torch.arange(1, 7, dtype=torch.float32),
        row_domains=torch.tensor([1, 0, 1, 0, 1, 0]),
    )
    rows = torch.tensor([4, 1, 5], dtype=torch.long)
    xb, yb, weights, domains = source.gather_indices(rows)
    assert torch.equal(xb.cpu(), source.cpu_X[rows])
    assert yb is not None and torch.equal(yb.cpu(), source.cpu_y[rows])
    assert weights is not None and torch.equal(weights.cpu(), source.cpu_row_weights[rows])
    assert domains is not None and torch.equal(domains.cpu(), source.cpu_row_domains[rows])
    with pytest.raises(ValueError, match="unique"):
        source.gather_indices(torch.tensor([1, 1], dtype=torch.long))
    with pytest.raises(ValueError, match="cover every"):
        tuple(source.iter_index_batches((rows,)))


def test_validation_candidate_objective_is_chunk_partition_invariant() -> None:
    timeline, labels = _timeline(decision_count=6)
    metadata = _metadata(timeline)
    config = _config()
    plan_wide, _ = build_decision_batch_plan(
        metadata,
        labels,
        config,
        max_batch_rows=256,
        phase="validation",
    )
    plan_narrow, _ = build_decision_batch_plan(
        metadata,
        labels,
        config,
        max_batch_rows=128,
        phase="validation",
    )
    assert plan_wide.batch_decision_counts != plan_narrow.batch_decision_counts
    generator = torch.Generator().manual_seed(11)
    evidence = torch.randn(len(labels), generator=generator)
    logits = torch.stack((-evidence / 2, evidence / 2), dim=1)

    def aggregate(plan) -> tuple[float, tuple[float, ...]]:
        module = DecisionLevelLoss(candidate_vocabulary=("0", "1"), temperature=0.7)
        weighted = 0.0
        prefix = torch.zeros(32)
        count = 0
        for batch in plan.batches(seed=None):
            rows = torch.from_numpy(np.array(batch.row_indices, copy=True))
            targets = torch.from_numpy(np.array(batch.target_indices, copy=True))
            loss, by_prefix = module.packed_loss(
                logits[rows, 1] - logits[rows, 0],
                targets,
                max_repetitions=32,
                prefix_weights=config.prefix_weights,
            )
            n_decisions = len(batch.decision_ids)
            weighted += float(loss) * n_decisions
            prefix += by_prefix * n_decisions
            count += n_decisions
        return weighted / count, tuple(float(value) for value in prefix / count)

    wide_loss, wide_prefix = aggregate(plan_wide)
    narrow_loss, narrow_prefix = aggregate(plan_narrow)
    assert wide_loss == pytest.approx(narrow_loss, abs=1e-7)
    assert wide_prefix == pytest.approx(narrow_prefix, abs=1e-7)


class _TinyDecisionBaseline(DeepBaseline):
    def _make_model(self) -> nn.Module:
        return nn.Sequential(nn.Flatten(), nn.Linear(self.n_chans * self.n_times, 2))


@pytest.mark.parametrize("lambda_candidate", [0.0, 0.1])
def test_decision_loss_config_never_silently_disables_grouped_contract(
    lambda_candidate: float,
) -> None:
    baseline = _TinyDecisionBaseline(
        n_chans=1,
        n_times=2,
        sfreq=128.0,
        config=DeepConfig(epochs=1, compile_mode=None, fused_adam=False),
        device=torch.device("cpu"),
    )
    X = np.zeros((8, 1, 2), dtype=np.float32)
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), 4)

    with pytest.raises(DirectDecisionContractError) as caught:
        baseline.fit(
            X,
            labels,
            decision_loss_config=_config(lambda_candidate=lambda_candidate),
        )

    assert caught.value.record["reason"] == (
        "decision_loss_config_requires_direct_decision_metadata"
    )
    assert baseline.last_decision_loss is not None
    assert baseline.last_decision_loss["status"] == "invalid"


def test_deep_fit_records_candidate_validation_objective_and_grouped_control() -> None:
    timeline, labels = _timeline(decision_count=8)
    metadata = _metadata(timeline)
    rng = np.random.default_rng(12)
    X = rng.standard_normal((len(labels), 1, 2)).astype(np.float32)
    groups = np.asarray(timeline.group_ids)

    def fit(lambda_candidate: float) -> _TinyDecisionBaseline:
        baseline = _TinyDecisionBaseline(
            n_chans=1,
            n_times=2,
            sfreq=128.0,
            config=DeepConfig(
                epochs=1,
                batch_size=256,
                val_group_frac=0.5,
                val_groups_min=4,
                val_groups_max=4,
                compile_mode=None,
                fused_adam=False,
                seed=2,
            ),
            device=torch.device("cpu"),
        )
        baseline.fit(
            X,
            labels,
            group_ids=groups,
            decision_metadata=metadata,
            decision_loss_config=_config(lambda_candidate=lambda_candidate),
        )
        return baseline

    active = fit(0.1)
    control = fit(0.0)
    active_record = active.last_history["decision_loss"]
    control_record = control.last_history["decision_loss"]
    assert active_record["schema"] == DECISION_TRAINING_LEDGER_SCHEMA
    assert active_record["training_batch_plan"] == control_record["training_batch_plan"]
    validation = active_record["epoch_validation_components"][0]
    assert validation["full_objective_loss"] == pytest.approx(
        validation["trial_component_loss"]
        + 0.1 * validation["candidate_component_loss"]
    )
    assert active.last_history["val_objective_losses"][0] == pytest.approx(
        validation["full_objective_loss"]
    )
    assert active_record["early_stopping_objective"] == "trial_plus_lambda_candidate"


def test_p2_oom_fails_without_batch_halving() -> None:
    timeline, labels = _timeline(decision_count=2)
    metadata = _metadata(timeline)
    X = np.zeros((len(labels), 1, 2), dtype=np.float32)
    baseline = _TinyDecisionBaseline(
        n_chans=1,
        n_times=2,
        sfreq=128.0,
        config=DeepConfig(compile_mode=None, fused_adam=False),
        device=torch.device("cpu"),
    )
    calls = 0

    def fail_attempt(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        raise torch.OutOfMemoryError("synthetic out of memory")

    baseline._fit_attempt = MethodType(fail_attempt, baseline)
    with pytest.raises(DirectDecisionContractError) as caught:
        baseline.fit(
            X,
            labels,
            decision_metadata=metadata,
            decision_loss_config=_config(),
        )
    assert calls == 1
    assert caught.value.record["reason"] == "oom_cannot_change_complete_decision_batch_contract"
    assert baseline._fitted is False


def test_deep_fit_preserves_invalid_decision_ledger_after_qc_gap() -> None:
    timeline, labels = _timeline(decision_count=3)
    retained = np.ones(len(labels), dtype=bool)
    retained[0] = False
    metadata = _metadata(timeline, retained_row_mask=retained)
    X = np.zeros((metadata.n_rows, 1, 2), dtype=np.float32)
    baseline = _TinyDecisionBaseline(
        n_chans=1,
        n_times=2,
        sfreq=128.0,
        config=DeepConfig(epochs=1, compile_mode=None, fused_adam=False),
        device=torch.device("cpu"),
    )
    with pytest.raises(DirectDecisionContractError):
        baseline.fit(
            X,
            labels[retained],
            decision_metadata=metadata,
            decision_loss_config=_config(),
        )
    assert baseline._fitted is False
    assert baseline.last_decision_loss is not None
    assert baseline.last_decision_loss["status"] == "invalid"
    assert baseline.last_decision_loss["decisions"][0]["missing_slots"]


def test_p0_p1_p2_coexist_and_p1_ledger_names_trial_component() -> None:
    timeline, labels = _timeline(decision_count=4)
    metadata = _metadata(timeline)
    rng = np.random.default_rng(41)
    X = rng.standard_normal((len(labels), 3, 64)).astype(np.float32)
    domains = np.repeat(["a", "b"], len(labels) // 2)
    baseline = N2P3NetBaseline(
        3,
        64,
        128.0,
        config=DeepConfig(
            epochs=1,
            batch_size=256,
            val_group_frac=None,
            compile_mode=None,
            fused_adam=False,
            seed=4,
        ),
        device=torch.device("cpu"),
        tmin_s=-0.2,
        feature_transfer=FeatureTransferConfig(
            invariant_bottleneck_channels=2,
            private_channels=2,
            target_bottleneck_channels=2,
            target_kernel_size=5,
            private_domain_count=2,
            source_domain_count=2,
            source_drop_p=0.3,
        ),
        source_domain_vocabulary=("a", "b"),
        splitbn_domain_count=2,
        splitbn_target_buffer_source="pooled_source",
    )

    baseline.fit(
        X,
        labels,
        source_domain_ids=domains,
        decision_metadata=metadata,
        decision_loss_config=_config(),
    )

    source_ledger = baseline.last_history["source_residual"]
    decision_ledger = baseline.last_history["decision_loss"]
    assert source_ledger["counterfactual_loss_scope"] == "trial_component"
    assert source_ledger["counterfactual_excludes_candidate_component"] is True
    assert len(source_ledger["epoch_trial_loss_m0"]) == 1
    assert len(source_ledger["epoch_trial_loss_m1"]) == 1
    assert decision_ledger["status"] == "valid"
    assert decision_ledger["candidate_component_active"] is True


def test_grouped_p2_rejects_row_weighted_source_risk_until_estimand_exists() -> None:
    timeline, labels = _timeline(decision_count=2)
    metadata = _metadata(timeline)
    X = np.zeros((len(labels), 1, 2), dtype=np.float32)
    baseline = _TinyDecisionBaseline(
        n_chans=1,
        n_times=2,
        sfreq=128.0,
        config=DeepConfig(compile_mode=None, fused_adam=False),
        device=torch.device("cpu"),
    )
    with pytest.raises(DirectDecisionContractError, match="weighting_estimand"):
        baseline.fit(
            X,
            labels,
            source_domain_ids=np.repeat("a", len(labels)),
            source_domain_mass={"a": 1.0},
            risk_unit_ids=np.asarray(timeline.subject_ids),
            decision_metadata=metadata,
            decision_loss_config=_config(lambda_candidate=0.0),
        )
