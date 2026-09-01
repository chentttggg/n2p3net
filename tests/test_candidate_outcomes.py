from __future__ import annotations

import numpy as np

from transfer.evaluation import candidate_decision_outcomes
from transfer.outcomes import DecisionStatus, build_decision_outcome_accounting


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
