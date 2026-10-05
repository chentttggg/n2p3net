"""Behavior and independent probability oracles for the frozen arithmetic pool."""

import json
from dataclasses import replace

import numpy as np
import pytest

from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from transfer.ensemble import EnsembleProgram, EnsembleScorer


def evidence(name, probabilities, *, physical="a" * 64, numerical="b" * 64):
    return CandidateEvidence(candidate_vocabulary=("left", "right"),
        log_probability=np.log(probabilities), participant_key="fixture-participant",
        decision_id="selection-1", physical_support_digest=physical,
        expert_identities=(CandidateExpertIdentity(name, numerical, "c" * 64,
                                                   "d" * 64, "e" * 64, "fixture", False),),
        evidence_level=1, expected_event_counts=(2, 2), observed_event_counts=(2, 2))


def test_pool_is_arithmetic_probability_and_can_reduce_hit():
    left = evidence("left_expert", [.51, .49])
    right = evidence("right_expert", [.01, .99], numerical="f" * 64)
    scorer = EnsembleScorer(EnsembleProgram.equal_weights("pair", ("left_expert", "right_expert")))
    mixed = scorer.score({"left_expert": left, "right_expert": right})
    np.testing.assert_allclose(mixed.probability, [.26, .74], rtol=0, atol=1e-15)
    assert left.predicted == "left" and mixed.predicted == "right"
    assert mixed.physical_support_digest == left.physical_support_digest
    assert {item.numerical_input_digest for item in mixed.expert_identities} == {"b" * 64, "f" * 64}


def test_three_expert_group_weight_is_distinct_from_four_equal_experts():
    items = {name: evidence(name, probability) for name, probability in
             (("raw", [.2, .8]), ("toeplitz", [.2, .8]), ("xdawn", [.2, .8]), ("neural", [.9, .1]))}
    classic = EnsembleScorer(EnsembleProgram.equal_weights("classic", ("raw", "toeplitz", "xdawn")))
    four = EnsembleScorer(EnsembleProgram.equal_weights("four", tuple(items)))
    np.testing.assert_allclose(four.score(items).probability, [.375, .625])
    np.testing.assert_allclose(classic.score({name: items[name] for name in classic.program.expert_ids}).probability,
                               [.2, .8])


@pytest.mark.parametrize("changed", ["physical_support_digest", "decision_id", "participant_key",
                                    "candidate_vocabulary", "evidence_level"])
def test_same_shape_does_not_admit_a_different_physical_task(changed):
    a = evidence("a", [.8, .2])
    values = {"physical_support_digest": "1" * 64, "decision_id": "selection-2",
              "participant_key": "other-person", "candidate_vocabulary": ("right", "left"), "evidence_level": 2}
    b = replace(evidence("b", [.2, .8]), **{changed: values[changed]})
    with pytest.raises(ValueError, match="same physical"):
        EnsembleScorer(EnsembleProgram.equal_weights("pair", ("a", "b"))).score({"a": a, "b": b})


def test_failure_does_not_renormalize_the_declared_roster():
    a = evidence("a", [.8, .2])
    failed = replace(evidence("b", [.2, .8]), log_probability=None,
                     failure_kind="fit_failure", failure_reason="numerical_fit_failure")
    scorer = EnsembleScorer(EnsembleProgram.equal_weights("pair", ("a", "b")))
    mixed = scorer.score({"a": a, "b": failed})
    assert mixed.log_probability is None and not mixed.operational_available
    assert mixed.failure_kind == "fit_failure" and "b:" in mixed.failure_reason
    with pytest.raises(ValueError, match="exactly"):
        scorer.score({"a": a})


def test_probability_tie_expert_can_contribute_to_a_unique_mixture():
    scorer = EnsembleScorer(EnsembleProgram.equal_weights("pair", ("a", "b")))
    tied, useful = evidence("a", [.5, .5]), evidence("b", [.8, .2])
    assert tied.probability_available and not tied.operational_available
    assert scorer.score({"a": tied, "b": useful}).predicted == "left"


def test_log_pool_preserves_finite_underflow_nll_and_roundtrips(tmp_path):
    first = evidence("a", [.5, .5])
    first = replace(first, log_probability=np.array([0., -1000.]))
    second = replace(evidence("b", [.5, .5]), log_probability=np.array([0., -1100.]))
    scorer = EnsembleScorer(EnsembleProgram.equal_weights("pair", ("a", "b")))
    result = scorer.score({"a": first, "b": second})
    assert result.probability[1] == 0
    np.testing.assert_allclose(-result.log_probability[1], 1000 + np.log(2), atol=1e-12)
    path = tmp_path / "program.json"
    scorer.save(path)
    restored = EnsembleScorer.load(path)
    np.testing.assert_array_equal(restored.score({"a": first, "b": second}).log_probability,
                                  result.log_probability)
    altered = json.loads(path.read_bytes())
    altered["program"]["weights"] = [.8, .2]
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        EnsembleScorer.load(path)
