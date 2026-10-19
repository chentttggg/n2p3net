"""Behavioral regressions across prediction, validation, and result reduction seams."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import baselines.evaluate as evaluation
from baselines.deep import DeepBaseline, DeepConfig, _CrossEntropyObjective
from experiments.analyze_candidate_cross_decision import (
    DEFAULT_METRIC,
    REQUESTED_METRIC,
    VALID_SUBJECT_METRIC,
    _analysis_records,
)
from models.candidate_evidence import candidate_evidence_scores
from train.runtime import MatrixBatchSource


@pytest.mark.parametrize("weighted_rows", [False, True])
def test_validation_weighted_ce_is_independent_of_batch_partition(weighted_rows):
    model = DeepBaseline("eegnet", config=DeepConfig(epochs=1), device=torch.device("cpu"))
    model.model_ = torch.nn.Identity()
    logits = torch.tensor([[0.0, -2.0]] * 4 + [[0.0, -5.0]] * 2)
    labels = torch.tensor([0, 0, 0, 0, 1, 1])
    row_weights = torch.tensor([0.5, 1.5, 0.75, 1.25, 2.0, 0.5]) if weighted_rows else None
    source = MatrixBatchSource(logits, labels, model.runtime, preload=False, row_weights=row_weights)
    objective = _CrossEntropyObjective(8.0, torch.device("cpu"))
    numerator, denominator = objective.components(logits, labels, row_weights)
    expected = float(numerator / denominator)
    for batch in [6, 4, 2, 1]:
        actual, scores, observed_labels = model._validation_pass(source, objective, batch)
        assert actual == pytest.approx(expected, rel=1e-6)
        np.testing.assert_array_equal(scores, (logits[:, 1] - logits[:, 0]).numpy())
        np.testing.assert_array_equal(observed_labels, labels.numpy())


def test_candidate_summary_records_folds_whose_auc_is_undefined(monkeypatch):
    defined = evaluation.BinaryFoldResult(1.0, 1.0, 3, 0.0, "fixed", 0.0)
    undefined = evaluation.BinaryFoldResult(float("nan"), float("nan"), 3, 0.0, "fixed", 0.0)
    groups = np.repeat(["a", "b"], 3)
    folds = [(groups != group, groups == group) for group in ("a", "b")]
    monkeypatch.setattr(evaluation, "_run_fold_tasks", lambda *args, **kwargs: (
        [(0, defined, np.zeros(3), folds[0][1], None), (1, undefined, np.zeros(3), folds[1][1], None)],
        "fixed_predictions", 1, "none", 1, 1, 1,
    ))
    result = evaluation.evaluate_candidate_selection(
        object(), np.zeros((6, 1, 1)), np.array([0, 1, 0, 0, 1, 0]), np.tile([0, 1, 2], 2), groups,
        {"a": 1, "b": 1}, folds, [0, 1, 2], fold_id_offset=4,
    )
    assert result.auc_mean == 1.0
    assert result.metric_undefined_folds == [5]


def test_domain_lodo_validation_checks_actual_global_coverage():
    domains = np.array(["A", "A", "B", "B"])
    folds = [(domains == "A", domains == "B"), (domains == "B", domains == "A")]
    evaluation._validate_folds(folds, 4, domain_ids=domains, fold_protocol="domain_lodo")
    with pytest.raises(ValueError, match="cover every selected row"):
        evaluation._validate_folds(folds[:1], 4, domain_ids=domains, fold_protocol="domain_lodo")


def test_candidate_summary_keeps_missing_and_fully_unavailable_requested_groups(monkeypatch):
    # Test originally requests complete, partial, and empty groups. The last
    # group's only row was removed by the upstream test-quality policy.
    test = np.array([False, False, True, True, True, True])
    effective = np.array([False, False, True, True, True, False])
    values = np.array([2.0, 0.0, 2.0])
    binary = evaluation.BinaryFoldResult(1.0, 1.0, 3, 0.0, "fixed", 0.0)
    monkeypatch.setattr(evaluation, "_run_fold_tasks", lambda *args, **kwargs: (
        [(0, binary, values, effective, None)], "fixed_predictions", 1, "none", 1, 1, 1,
    ))
    result = evaluation.evaluate_candidate_selection(
        object(), np.zeros((6, 1, 1)), np.array([0, 1, 1, 0, 1, 1]),
        np.array([0, 1, 0, 1, 0, 0]), np.array(["train", "train", "complete", "complete", "partial", "empty"]),
        {"complete": 0, "partial": 0, "empty": 0}, [(~test, test)], [0, 1],
        repetition_indices=np.zeros(6, dtype=int),
    )
    assert result.primary_hit_rate == pytest.approx(1 / 3)
    assert result.hit_at_repetition_macro == {"1": pytest.approx(1 / 3)}
    assert result.per_fold[0].n_decisions == 3
    assert result.per_fold[0].hit_total_by_repetition == {"1": 3}


def test_repetition_prediction_cannot_read_later_evidence_when_centered():
    options = dict(codes=np.array([0, 1, 0, 1]), group_ids=np.array(["g"] * 4),
                   truth_by_group={"g": 0}, repetitions=np.array([0, 0, 1, 2]),
                   candidate_vocab=[0, 1], center_logits=True, max_repetitions=3)
    before, before_total = evaluation.hit_correct_total_by_repetition(np.array([1.0, 0.0, 1.0, 0.0]), **options)
    after, after_total = evaluation.hit_correct_total_by_repetition(np.array([1.0, 0.0, 1.0, 100.0]), **options)
    assert before_total == after_total
    assert before["1"] == after["1"]
    assert before["2"] == after["2"]


def test_candidate_core_always_abstains_for_an_unobserved_candidate():
    assert candidate_evidence_scores([9.0, 1.0], [0, 1], [0, 1, 2]).predicted is None


def test_precision_reliability_weights_remove_zero_weight_pollution():
    result = candidate_evidence_scores([1.0, 100.0, 2.0], [0, 0, 1], [0, 1],
        aggregation="precision", logit_variances=[1.0, 1.0, 1.0], trial_weights=[1.0, 0.0, 1.0])
    assert result.predicted == 1
    assert result.scores[0] == 1.0
    weighted = candidate_evidence_scores([0.0, 2.0, 1.0], [0, 0, 1], [0, 1],
        aggregation="precision", logit_variances=[1.0, 2.0, 1.0], trial_weights=[1.0, 4.0, 1.0])
    assert weighted.scores[0] == pytest.approx(4 / 3)


def test_trim_explicitly_rejects_unsupported_external_weights():
    with pytest.raises(ValueError, match="trim0.2.*trial_weights"):
        candidate_evidence_scores([1.0, 100.0, 2.0], [0, 0, 1], [0, 1],
            aggregation="trim0.2", trial_weights=[1.0, 0.0, 1.0])


@pytest.mark.parametrize("scope", ["conditional_frozen_models", "training_procedure"])
def test_valid_contrast_uses_one_complete_paired_replicate_grid(scope):
    manifest = SimpleNamespace(
        design=SimpleNamespace(inference_scope=scope, training_replicate_keys=("r1", "r2"),
                               planned_contrasts=(("A", "B"),)),
        metric_name=VALID_SUBJECT_METRIC, participants=("p1", "p2"),
        iterations=1000, seed=7, confidence_level=0.95,
    )
    values = {("A", "r1", "p1"): 0.9, ("B", "r2", "p1"): 0.1,
              ("A", "r1", "p2"): 0.5, ("A", "r2", "p2"): 0.5,
              ("B", "r1", "p2"): 0.5, ("B", "r2", "p2"): 0.5}
    _, contrasts, _ = _analysis_records(manifest, ("A", "B"), values,
        {"A": {"r1": 0.7, "r2": 0.5}, "B": {"r1": 0.5, "r2": 0.3}})
    contrast = contrasts[0]
    assert set(contrast["paired_unit_differences"].values()) == {0.0}
    assert contrast["paired_support"]["included_participants"] == ["p2"]
    excluded = contrast["paired_support"]["excluded_participants"]
    assert excluded[0]["participant"] == "p1"
    assert excluded[0]["missing_left_replicates"] == ["r2"]
    assert excluded[0]["missing_right_replicates"] == ["r1"]


def test_requested_participant_operational_metric_is_the_default():
    assert DEFAULT_METRIC == REQUESTED_METRIC


def test_analysis_preserves_observed_support_before_common_cohort_selection():
    from experiments.analyze_candidate_cross_decision import _cell_values

    manifest = SimpleNamespace(
        design=SimpleNamespace(training_replicate_keys=("r1", "r2"), partition_keys=("one",),
                               inference_scope="conditional_frozen_models", planned_contrasts=(("A", "B"),)),
        metric_name=VALID_SUBJECT_METRIC, participants=("p1", "p2"),
        iterations=1000, seed=7, confidence_level=.95)
    runs = []
    for arm in ("A", "B"):
        for replicate in ("r1", "r2"):
            rates = {"p1": .5, "p2": .5}
            if (arm, replicate) == ("A", "r2"):
                del rates["p2"]
            runs.append(SimpleNamespace(spec=SimpleNamespace(arm=arm, replicate=replicate, partition="one"),
                                        participant_rates={"p1": .5, "p2": .5}, valid_participant_rates=rates))
    arms, values, means = _cell_values(manifest, tuple(runs))
    _, contrasts, _ = _analysis_records(manifest, arms, values, means)
    excluded, = contrasts[0]["paired_support"]["excluded_participants"]
    assert excluded["missing_left_replicates"] == ["r2"]
    assert excluded["missing_right_replicates"] == []
