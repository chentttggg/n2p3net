"""Physical membership, shared NumPy/Torch scoring and decision gradients."""

from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from data.candidate_task import MembershipDecision
from models.candidate_evidence import MEMBERSHIP_SCORE_METHOD, membership_decision_scores
from transfer.decision_loss import membership_decision_loss


def _direct(repetitions=2):
    return MembershipDecision.direct(
        np.tile(["1", "2", "3"], repetitions), ("1", "2", "3"),
        event_ids=tuple(f"event-{i}" for i in range(3 * repetitions)),
        repetition_indices=np.repeat(np.arange(repetitions), 3), repetitions=repetitions,
    )


def _row_column(repetitions=2):
    return MembershipDecision.row_column(
        np.tile(np.arange(5), repetitions), n_rows=2, n_columns=3,
        event_ids=tuple(f"event-{i}" for i in range(5 * repetitions)),
        repetition_indices=np.repeat(np.arange(repetitions), 5), repetitions=repetitions,
    )


@pytest.mark.parametrize("factory", [_direct, _row_column], ids=["direct", "row-column"])
def test_membership_bayes_numpy_torch_and_gradient_oracle(factory):
    decision = factory()
    values = np.linspace(-.8, .9, len(decision.event_ids))
    prior = np.arange(1, len(decision.candidate_vocabulary) + 1, dtype=float)
    prior /= prior.sum()
    evidence = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    actual = membership_decision_scores(evidence, decision, prior=prior)
    numpy_result = membership_decision_scores(values, decision, prior=prior)
    probability = 1 / (1 + np.exp(-values))
    likelihood = np.prod(np.where(decision.membership, probability[:, None], 1 - probability[:, None]), axis=0)
    expected = prior * likelihood
    expected /= expected.sum()
    np.testing.assert_allclose(numpy_result.posterior, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.posterior.detach(), expected, rtol=1e-12, atol=1e-12)
    assert actual.score_kind == MEMBERSHIP_SCORE_METHOD
    loss = membership_decision_loss(evidence, decision, 1, prior=prior)
    assert loss.loss is not None
    gradient = torch.autograd.grad(loss.loss, evidence)[0]
    expected_gradient = decision.membership @ (expected - np.eye(len(expected))[1])
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert loss.record()["score_kind"] == MEMBERSHIP_SCORE_METHOD


def test_row_column_joint_sum_equals_axis_sum_nll_and_gradient():
    decision = _row_column(3)
    evidence = torch.linspace(-.7, 1.1, 15, dtype=torch.float64, requires_grad=True)
    result = membership_decision_loss(evidence, decision, 4, temperature=.6)
    sums = evidence.reshape(3, 5).sum(0) / .6
    expected = F.cross_entropy(sums[:2][None], torch.tensor([1])) + F.cross_entropy(sums[2:][None], torch.tensor([1]))
    torch.testing.assert_close(result.loss, expected, rtol=1e-12, atol=1e-12)
    first = torch.autograd.grad(result.loss, evidence, retain_graph=True)[0]
    second = torch.autograd.grad(expected, evidence)[0]
    torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)


def test_sum_and_legacy_mean_have_declared_different_probability_scales():
    from transfer.decision_loss import DecisionLevelLoss

    decision = _direct(5)
    values = torch.linspace(-.2, .8, 15, dtype=torch.float64)
    summed = membership_decision_loss(values, decision, 2)
    legacy = DecisionLevelLoss(candidate_vocabulary=decision.candidate_vocabulary)(
        values, torch.tensor(decision.stimulus_indices.copy()), 2)
    equivalent = membership_decision_loss(values, decision, 2, temperature=5.)
    assert summed.argmax_candidate == legacy.argmax_candidate
    assert not torch.allclose(summed.posterior, legacy.posterior)
    torch.testing.assert_close(equivalent.posterior, legacy.posterior, rtol=1e-12, atol=1e-12)
    assert "score_kind" not in legacy.record()


def test_candidate_and_event_permutations_preserve_their_axes():
    decision = _direct()
    values = np.arange(6, dtype=float) / 3
    prior = np.array([.2, .3, .5])
    first = membership_decision_scores(values, decision, prior=prior)
    columns = np.array([2, 0, 1])
    renamed = replace(decision, candidate_vocabulary=tuple(decision.candidate_vocabulary[i] for i in columns),
                      membership=decision.membership[:, columns])
    second = membership_decision_scores(values, renamed, prior=prior[columns])
    np.testing.assert_allclose(first.posterior[columns], second.posterior)
    assert first.predicted == second.predicted
    rows = np.array([4, 1, 5, 0, 3, 2])
    reordered = replace(decision, event_ids=tuple(decision.event_ids[i] for i in rows),
                        membership=decision.membership[rows], stimulus_indices=decision.stimulus_indices[rows],
                        repetition_indices=decision.repetition_indices[rows], evidence_indices=decision.evidence_indices[rows])
    np.testing.assert_allclose(first.scores, membership_decision_scores(values[rows], reordered, prior=prior).scores)


@pytest.mark.parametrize("torch_backend", [False, True])
def test_missing_nan_is_neutral_but_strict_coverage_abstains(torch_backend):
    decision = _direct()
    available = np.ones(6, dtype=bool)
    available[1] = False
    values = np.array([1., np.nan, -.5, .7, .3, -.2])
    evidence = torch.tensor(values, requires_grad=True) if torch_backend else values
    result = membership_decision_scores(evidence, decision, available=available)
    assert not result.complete and not result.eligible and result.predicted is None
    assert result.reason == "incomplete_candidate_coverage"
    np.testing.assert_array_equal(result.counts, [2, 1, 2])
    if torch_backend:
        result.scores.sum().backward()
        assert evidence.grad[1] == 0 and torch.isfinite(evidence.grad).all()
        assert membership_decision_loss(evidence, decision, 0, available=available).loss is None
    with pytest.raises(ValueError, match="finite"):
        membership_decision_scores(evidence, decision)


def test_no_evidence_preserves_prior_without_an_operational_prediction():
    decision = _direct()
    result = membership_decision_scores(np.full(6, np.nan), decision,
                                        available=np.zeros(6, bool), prior=[.2, .3, .5])
    np.testing.assert_allclose(result.posterior, [.2, .3, .5])
    assert result.reason == "no_usable_evidence" and result.predicted is None


def test_missing_scheduled_slot_is_not_inferred_away():
    decision = MembershipDecision.direct(["1", "2", "1", "2"], ("1", "2", "3"),
        event_ids=("a", "b", "c", "d"), repetition_indices=np.array([0, 0, 1, 1]), repetitions=2)
    result = membership_decision_scores(np.ones(4), decision)
    assert not decision.schedule_complete
    assert result.reason == "incomplete_candidate_coverage" and result.predicted is None
    assert len(result.scores) == 3


def test_indistinguishable_candidate_columns_abstain_even_with_prior_preference():
    decision = replace(_direct(1), membership=np.ones((3, 3), dtype=bool))
    result = membership_decision_scores(np.ones(3), decision, prior=[.1, .2, .7])
    assert result.complete and not result.identifiable and not result.eligible
    assert result.reason == "unidentifiable_candidates" and result.predicted is None


@pytest.mark.parametrize("changed", ["events", "evidence", "slots", "nonbinary", "noninteger"])
def test_malformed_physical_membership_rejects(changed):
    decision = _direct()
    with pytest.raises(ValueError):
        if changed == "events":
            replace(decision, event_ids=("duplicate",) * 6)
        elif changed == "evidence":
            replace(decision, evidence_indices=np.zeros(6, dtype=np.int64))
        elif changed == "slots":
            replace(decision, repetition_indices=np.zeros(6, dtype=np.int64))
        elif changed == "nonbinary":
            replace(decision, membership=np.full((6, 3), .5))
        else:
            replace(decision, repetition_indices=np.full(6, .5))


def test_membership_record_roundtrip_preserves_missing_event_alignment():
    decision = replace(_direct(), evidence_indices=np.array([0, -1, 1, 2, 3, 4]))
    restored = MembershipDecision.from_record(decision.record())
    assert restored.record() == decision.record()
    assert not restored.membership.flags.writeable
    with pytest.raises(ValueError, match="unavailable"):
        membership_decision_scores(np.ones(6), restored, available=np.ones(6, bool))
    assert restored.prefix(1).record()["event_ids"] == ["event-0", "event-1", "event-2"]


def test_ties_are_trainable_but_not_operational_predictions():
    evidence = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    result = membership_decision_loss(evidence, _direct(), 0)
    assert result.status == "tie" and result.argmax_candidate is None and result.loss is not None
    result.loss.backward()
    assert torch.count_nonzero(evidence.grad) == 6


def test_timeline_adapter_preserves_original_evidence_rows_without_reading_truth():
    class TargetFreeTimeline:
        n_events = 6
        event_ids = np.array([f"physical-{i}" for i in range(6)])
        group_ids = np.repeat("session", 6)
        subject_ids = np.repeat("person", 6)
        candidate_ids = np.tile(["1", "2", "3"], 2)
        repetition_indices = np.repeat([5, 6], 3)
        evidence_indices = np.array([40, 41, -1, 42, 43, 44])
        complete = True

        @property
        def target_candidate_ids(self):
            raise AssertionError("inference must not read target fields")

    decision = MembershipDecision.from_timeline(TargetFreeTimeline(), np.ones(6, bool),
        repetitions=2, repetition_start=5, candidate_vocabulary=("1", "2", "3"))
    np.testing.assert_array_equal(decision.evidence_indices, [40, 41, -1, 42, 43, 44])
    assert decision.schedule_complete
    labels = np.zeros(45, dtype=np.int64)
    labels[[40, 42]] = 1
    decision.validate_epoch_labels(labels, 0)
    labels[43] = 1
    with pytest.raises(ValueError, match="labels disagree"):
        decision.validate_epoch_labels(labels, 0)


def test_row_column_timeline_adapter_uses_frozen_task_membership():
    from data.candidate_task import CandidateTaskContract

    contract = CandidateTaskContract(dataset_id="synthetic", task_id="grid", population={"label": "software"},
        evidence_scope={"stage": "development", "product_confirmation": False}, membership_kind="row_column",
        grid_shape=(2, 3), candidate_ids=(0, 1, 2, 3, 4), row_candidate_ids=(0, 1), column_candidate_ids=(2, 3, 4),
        target_representation="row_column_intersection")

    class Timeline:
        n_events = 10
        event_ids = np.array([f"event-{i}" for i in range(10)])
        group_ids = np.repeat("selection", 10)
        subject_ids = np.repeat("person", 10)
        candidate_ids = np.tile(["0", "1", "2", "3", "4"], 2)
        repetition_indices = np.repeat([0, 1], 5)
        evidence_indices = np.arange(10)
        complete = True

    actual = MembershipDecision.from_timeline(Timeline(), np.ones(10, bool), repetitions=2, task_contract=contract)
    assert actual.record() == _row_column().record()


@pytest.mark.parametrize("options", [{"temperature": 0}, {"temperature": float("nan")},
    {"prior": [.2, .3, .6]}, {"prior": [0., .5, .5]}, {"logit_offset": float("nan")},
    {"prior": torch.ones(3, requires_grad=True) / 3}])
def test_invalid_or_learned_probability_parameters_reject(options):
    with pytest.raises(ValueError):
        membership_decision_scores(torch.zeros(6), _direct(), **options)
