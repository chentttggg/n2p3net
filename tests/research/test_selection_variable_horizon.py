"""Prefix selection keeps each original decision's actual repetition depth."""

import numpy as np

from data.candidate_task import MembershipDecision
from data.identity import origin_subject_key
from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from research.calibration_selection import _metrics
from transfer.ensemble_views import CalibrationSelection, PredictionDecisionView
from transfer.outcomes import DecisionKey


def selection(repetitions, index):
    ids = np.tile(np.arange(12, dtype=np.int64), repetitions)
    decision = MembershipDecision.row_column(ids, n_rows=6, n_columns=6,
        event_ids=tuple(f"event-{index}-{row}" for row in range(len(ids))),
        repetition_indices=np.repeat(np.arange(repetitions), 12), repetitions=repetitions)
    samples = np.zeros((len(ids) + 1, 1), dtype=np.float64)
    view = PredictionDecisionView(f"record-{index}", origin_subject_key("BI2014a", "01"), f"selection-{index}",
        samples, np.arange(len(ids), dtype=np.int64), decision, "a" * 64,
        (0, len(samples)), 512., ("CZ",), ("EEG",), "right earlobe", "subject_01.mat", "BI2014a", "01", len(samples))
    group = CalibrationSelection(view, decision.membership[:, 0], decision.candidate_vocabulary[0])
    counts = tuple(map(int, decision.membership.sum(axis=0)))
    evidence = CandidateEvidence.from_log_evidence(np.linspace(-1, 1, 36),
        candidate_vocabulary=decision.candidate_vocabulary, participant_key=view.participant_key,
        decision_id=view.decision_id, physical_support_digest=view.physical_support_digest,
        expert_identities=(CandidateExpertIdentity("fixed-contract", view.raw_input_digest, "b" * 64,
            None, "c" * 64, "contract_evidence", False),), evidence_level=repetitions,
        expected_event_counts=counts, observed_event_counts=counts)
    return group, evidence


def test_prefix_selection_metrics_use_each_original_requested_repetition_depth():
    first, a = selection(1, 1)
    second, b = selection(3, 2)
    result = _metrics((first, second), (a, b))
    assert result.evidence_level is None
    assert dict(result.requested_evidence_levels) == {
        DecisionKey(first.view.participant_key, first.view.decision_id): 1,
        DecisionKey(second.view.participant_key, second.view.decision_id): 3}
    assert sum(row.requested_decisions for row in result.participants) == 2


def test_uniform_selection_metrics_keep_the_original_scalar_horizon_record():
    first, a = selection(2, 1)
    second, b = selection(2, 2)
    result = _metrics((first, second), (a, b))
    assert result.evidence_level == 2 and result.requested_evidence_levels is None
    assert "requested_evidence_levels" not in result.record()
