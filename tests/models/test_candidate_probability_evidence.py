"""Probability evidence preserves physical identity and exact log-domain risk."""

import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext

import numpy as np
import pytest

from data.candidate_task import MembershipDecision
from models.candidate_evidence import (
    CandidateEvidence,
    CandidateExpertIdentity,
    membership_decision_scores,
)


def _identity(**changes):
    return replace(CandidateExpertIdentity(
        expert_id="lda", numerical_input_digest="a" * 64,
        model_digest="b" * 64, source_digest=None, fit_digest="c" * 64,
        score_kind="test_llr", probability_calibrated=False,
    ), **changes)


def _arguments(vocabulary=("a", "b", "c")):
    return dict(candidate_vocabulary=vocabulary, participant_key="BNCI:p1",
                decision_id="selection:7", physical_support_digest="d" * 64,
                expert_identities=(_identity(),), evidence_level=2,
                expected_event_counts=(2,) * len(vocabulary),
                observed_event_counts=(2,) * len(vocabulary))


def _decision():
    return MembershipDecision.direct(
        ("a", "b", "c") * 2, ("a", "b", "c"),
        event_ids=tuple(f"event-{i}" for i in range(6)),
        repetition_indices=np.repeat(np.arange(2), 3), repetitions=2,
    )


def test_complex_log_evidence_and_probabilities_are_rejected_at_ingress():
    args = _arguments(("a", "b"))
    values = np.log([.8, .2]) + np.array([123j, 456j])
    with pytest.raises(ValueError, match="real"):
        CandidateEvidence(log_probability=values, **args)
    with pytest.raises(ValueError, match="real"):
        CandidateEvidence.from_log_evidence(values, **args)


def test_membership_probability_matches_independent_bernoulli_enumeration():
    decision = _decision()
    trial_odds = np.array([-.2, .6, -.3, .1, .8, -.1])
    scores = membership_decision_scores(trial_odds, decision)
    arguments = _arguments()
    for name in ("candidate_vocabulary", "evidence_level", "expected_event_counts", "observed_event_counts"):
        arguments.pop(name)
    result = CandidateEvidence.from_membership_scores(scores, decision, **arguments)
    p = 1. / (1. + np.exp(-trial_odds))
    likelihood = np.prod(np.where(decision.membership, p[:, None], 1. - p[:, None]), axis=0)
    np.testing.assert_allclose(result.probability, likelihood / likelihood.sum(), rtol=1e-13)
    assert result.predicted == "b"
    assert result.probability_available and result.operational_available
    assert result.evidence_level == 2


def test_complete_probability_tie_abstains_without_losing_probability():
    result = CandidateEvidence.from_log_evidence(np.zeros(3), **_arguments())
    assert result.tie and result.predicted is None
    assert result.probability_available and not result.operational_available
    assert result.failure_reason is None


def test_finite_log_probability_underflow_is_distinct_from_true_zero():
    result = CandidateEvidence.from_log_evidence([0., -1000., -2000.], **_arguments())
    assert result.probability[1] == 0.
    assert np.isfinite(result.log_probability[1])
    zero = CandidateEvidence(log_probability=np.array([0., -np.inf, -np.inf]), **_arguments())
    assert zero.probability[1] == 0.
    assert np.isneginf(zero.log_probability[1])
    assert CandidateEvidence.from_record(zero.record()).log_probability[1] == -np.inf


def test_large_common_log_evidence_offset_has_decimal_oracle():
    values = np.array([1e6, 1e6 - 2., 1e6 - 5.])
    result = CandidateEvidence.from_log_evidence(values, **_arguments())
    with localcontext() as context:
        context.prec = 60
        odds = [Decimal(int(value - values[0])).exp() for value in values]
        expected = np.array([float(value / sum(odds)) for value in odds])
    np.testing.assert_allclose(result.probability, expected, rtol=1e-13)


def test_probability_evidence_owns_immutable_storage_and_identity():
    original = np.log([.2, .3, .5])
    result = CandidateEvidence(log_probability=original, **_arguments())
    original[:] = 7.
    np.testing.assert_allclose(result.probability, [.2, .3, .5])
    assert result.log_probability.dtype == np.float64
    with pytest.raises(ValueError):
        result.log_probability.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        result.decision_id = "other"


@pytest.mark.parametrize("observed", [(0, 0, 0), (2, 1, 2)])
def test_incomplete_physical_evidence_cannot_publish_a_prior_only_distribution(observed):
    arguments = _arguments()
    arguments["observed_event_counts"] = observed
    result = CandidateEvidence.from_log_evidence([0., 20., 0.], **arguments)
    assert not result.probability_available and not result.operational_available
    assert result.failure_kind == "incomplete"
    assert result.failure_reason


def test_no_membership_evidence_does_not_publish_nonuniform_prior_prediction():
    decision = _decision()
    scores = membership_decision_scores(np.zeros(6), decision, available=np.zeros(6, dtype=bool),
                                        prior=[.05, .05, .9])
    arguments = _arguments()
    for name in ("candidate_vocabulary", "evidence_level", "expected_event_counts", "observed_event_counts"):
        arguments.pop(name)
    result = CandidateEvidence.from_membership_scores(scores, decision, **arguments)
    assert result.log_probability is None and result.predicted is None
    assert result.failure_reason == "no_usable_evidence"


def test_candidate_permutation_preserves_physical_winner_and_probabilities():
    scores = np.array([-.5, 1.2, .3])
    result = CandidateEvidence.from_log_evidence(scores, **_arguments())
    permutation = np.array([2, 0, 1])
    vocabulary = tuple(result.candidate_vocabulary[i] for i in permutation)
    permuted = CandidateEvidence.from_log_evidence(scores[permutation], **_arguments(vocabulary))
    assert result.predicted == permuted.predicted
    np.testing.assert_allclose(result.log_probability[permutation], permuted.log_probability)


@pytest.mark.parametrize("bad", [[.1, .2], [np.nan, 0., 0.], [np.inf, 0., 0.], [-np.inf] * 3])
def test_invalid_candidate_probability_ingress_is_rejected(bad):
    with pytest.raises(ValueError):
        CandidateEvidence.from_log_evidence(bad, **_arguments())


def test_missing_candidate_column_and_unnormalized_record_are_rejected():
    with pytest.raises(ValueError, match="vocabulary|shape|aligned"):
        CandidateEvidence(log_probability=np.log([.2, .8]), **_arguments())
    with pytest.raises(ValueError, match="normalized"):
        CandidateEvidence(log_probability=np.array([0., 0., 0.]), **_arguments())
    with pytest.raises(ValueError, match="unique"):
        CandidateEvidence.from_log_evidence([0., 0., 0.], **_arguments(("a", "a", "c")))


def test_expert_numeric_identity_can_differ_on_common_physical_support():
    arguments = _arguments()
    arguments["expert_identities"] = (_identity(), _identity(expert_id="raw", numerical_input_digest="f" * 64))
    result = CandidateEvidence.from_log_evidence([.2, .1, 0.], **arguments)
    assert len(result.expert_identities) == 2
    with pytest.raises(ValueError, match="digest"):
        _identity(numerical_input_digest="same-metadata")


def test_failure_reason_cannot_coexist_with_an_operational_probability():
    with pytest.raises(ValueError, match="failure"):
        CandidateEvidence(log_probability=np.log([.2, .3, .5]), failure_reason="fit exploded",
                          failure_kind="fit_failure", **_arguments())


def test_json_roundtrip_is_strict_and_preserves_true_zero_and_failure():
    result = CandidateEvidence(log_probability=np.array([0., -np.inf, -np.inf]), **_arguments())
    restored = CandidateEvidence.from_record(json.loads(json.dumps(result.record(), allow_nan=False)))
    np.testing.assert_array_equal(restored.log_probability, result.log_probability)
    failed = CandidateEvidence(log_probability=None, failure_kind="not_available",
                               failure_reason="candidate_calibration_not_estimable", **_arguments())
    restored_failure = CandidateEvidence.from_record(json.loads(json.dumps(failed.record(), allow_nan=False)))
    assert restored_failure.status == "not_available"
    assert restored_failure.log_probability is None


def test_character_string_is_not_a_declared_candidate_vocabulary():
    arguments = _arguments()
    arguments["candidate_vocabulary"] = "abc"
    with pytest.raises(ValueError, match="sequence"):
        CandidateEvidence.from_log_evidence([0., 1., 2.], **arguments)


def test_unidentifiable_physical_membership_cannot_publish_probability():
    decision = _decision()
    decision = replace(decision, membership=np.ones_like(decision.membership))
    scores = membership_decision_scores(np.arange(6, dtype=float), decision)
    arguments = _arguments()
    for name in ("candidate_vocabulary", "evidence_level", "expected_event_counts", "observed_event_counts"):
        arguments.pop(name)
    result = CandidateEvidence.from_membership_scores(scores, decision, **arguments)
    assert result.failure_reason == "unidentifiable_candidates"
    assert not result.probability_available


@pytest.mark.parametrize("mutation", ["missing_expert", "extra_field", "mismatched_shape"])
def test_saved_probability_ingress_rejects_tampered_contract(mutation):
    record = CandidateEvidence.from_log_evidence([0., 1., 2.], **_arguments()).record()
    if mutation == "missing_expert":
        record["expert_identities"] = []
    elif mutation == "extra_field":
        record["target_candidate"] = "c"
    else:
        record["log_probability"] = record["log_probability"][:-1]
    with pytest.raises(ValueError):
        CandidateEvidence.from_record(record)


def test_finite_membership_evidence_difference_overflow_is_not_encoded_as_true_zero():
    decision = MembershipDecision.direct(("a", "b"), ("a", "b"),
        event_ids=("e1", "e2"), repetition_indices=np.zeros(2, dtype=int), repetitions=1)
    with np.errstate(over="ignore"):
        scores = membership_decision_scores(np.array([1e308, -1e308]), decision)
    arguments = _arguments()
    for name in ("candidate_vocabulary", "evidence_level", "expected_event_counts", "observed_event_counts"):
        arguments.pop(name)
    with pytest.raises(ValueError, match="floating-point"):
        CandidateEvidence.from_membership_scores(scores, decision, **arguments)
