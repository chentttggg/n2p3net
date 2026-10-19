from __future__ import annotations

import numpy as np
import pytest

from transfer.evaluation import candidate_decision_outcomes
from transfer.outcomes import (
    CandidateCoverage,
    DecisionKey,
    DecisionOutcome,
    DecisionStatus,
    build_decision_outcome_accounting,
)


def test_candidate_outcomes_make_ties_and_missing_evidence_explicit() -> None:
    outcomes = candidate_decision_outcomes(
        logits=np.asarray([0.0, 0.0, 0.0, 0.0]),
        digits=np.asarray([1, 2, 1, 2]),
        group_ids=np.repeat("d1", 4),
        truth_by_group={"d1": 2},
        repetition_indices=np.asarray([0, 0, 1, 1]),
        subject_by_group={"d1": "s1"},
        aggregation="mean",
        max_repetitions=2,
        candidate_vocabulary=(1, 2),
        onset_times_s=np.asarray([1.0, 2.0, 3.0, 4.0]),
        evidence_available_times_s=np.asarray([2.0, 3.0, 4.0, 5.0]),
    )
    assert [outcome.status for outcome in outcomes] == [
        DecisionStatus.TIE,
        DecisionStatus.TIE,
    ]
    accounting = build_decision_outcome_accounting(
        outcomes,
        requested_decisions=[outcomes[0].key],
        evidence_levels=(1, 2),
    )
    assert accounting.to_record()["by_evidence_level"]["1"]["tie"] == 1
    record = accounting.to_record()["by_evidence_level"]["1"]
    assert record["valid"] == 1
    assert record["valid_hit_rate"] == 0.0


def test_candidate_outcome_does_not_drop_incomplete_candidate() -> None:
    outcomes = candidate_decision_outcomes(
        logits=np.asarray([1.0]),
        digits=np.asarray([1]),
        group_ids=np.asarray(["d1"]),
        truth_by_group={"d1": 1},
        repetition_indices=np.asarray([0]),
        subject_by_group={"d1": "s1"},
        aggregation="mean",
        max_repetitions=1,
        candidate_vocabulary=(1, 2),
    )
    assert outcomes[0].status == DecisionStatus.INCOMPLETE
    assert outcomes[0].coverage.missing_event_counts == {"2": 1}


def test_variable_r_outcome_accounting_owns_one_r_per_requested_decision():
    first, second = DecisionKey("p1", "d1"), DecisionKey("p2", "d2")
    horizons = {second: 8, first: 1}
    outcomes = [DecisionOutcome(first, 1, DecisionStatus.CORRECT,
                CandidateCoverage.from_mappings({"a": 1, "b": 1}, {"a": 1, "b": 1}),
                target_candidate="a", predicted_candidate="a"),
                DecisionOutcome(second, 8, DecisionStatus.FIT_FAILURE,
                CandidateCoverage.from_mappings({"a": 8, "b": 8}, {}),
                target_candidate="a", failure_reason="fit failed")]
    accounting = build_decision_outcome_accounting(outcomes, requested_decisions=[first, second],
        requested_evidence_levels=horizons)
    horizons[second] = 77
    record = accounting.to_record()
    assert record["schema"] == "n2p3_decision_outcome_accounting/2"
    assert record["requested"] == 2
    assert record["by_evidence_level"]["1"]["requested"] == 1
    assert record["by_evidence_level"]["8"]["requested"] == 1
    assert record["overall"]["correct"] == record["overall"]["fit_failure"] == 1
    assert record["overall"]["operational_hit_rate"] == .5


@pytest.mark.parametrize("mutation", ["wrong_R", "missing", "unexpected", "duplicate"])
def test_variable_r_accounting_rejects_nonplanned_or_missing_outcome_cells(mutation):
    key = DecisionKey("p", "d")
    rows = [DecisionOutcome(key, 1, DecisionStatus.ABSTAIN,
        CandidateCoverage.from_mappings({"a": 1, "b": 1}, {}), failure_reason="missing")]
    if mutation == "wrong_R":
        requested = {key: 8}
    else:
        requested = {key: 1}
        if mutation == "missing":
            rows = []
        elif mutation == "unexpected":
            rows = [DecisionOutcome(DecisionKey("p", "other"), 1, DecisionStatus.ABSTAIN,
                rows[0].coverage, failure_reason="missing")]
        else:
            rows *= 2
    with pytest.raises(ValueError, match="mismatch|duplicate"):
        build_decision_outcome_accounting(rows, requested_decisions=[key], requested_evidence_levels=requested)


def test_variable_r_accounting_empty_plan_has_zero_requested_cells():
    accounting = build_decision_outcome_accounting([], requested_decisions=[], requested_evidence_levels={})
    assert accounting.to_record()["by_evidence_level"] == {}
    assert accounting.to_record()["overall"]["requested"] == 0


def test_variable_r_accounting_does_not_allow_an_extra_cartesian_level_axis():
    key = DecisionKey("p", "d")
    with pytest.raises(ValueError, match="exactly one|mutually exclusive"):
        build_decision_outcome_accounting([], requested_decisions=[key], evidence_levels=[1, 8],
            requested_evidence_levels={key: 1})
