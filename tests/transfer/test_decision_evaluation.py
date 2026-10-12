from __future__ import annotations

from transfer.evaluation import candidate_evidence_endpoints


def test_all_evidence_reports_raw_bias_balanced_endpoint_and_missing_coverage() -> None:
    endpoints = candidate_evidence_endpoints(
        logits=[0.5, 0.5, 0.5, 1.0, 2.0],
        digits=[0, 0, 0, 1, 0],
        group_ids=["imbalanced", "imbalanced", "imbalanced", "imbalanced", "missing"],
        truth_by_group={"imbalanced": 1, "missing": 1},
        repetition_indices=[0, 1, 2, 0, 0],
        aggregation="sum",
        candidate_vocabulary=(0, 1),
    )

    assert endpoints["raw_predictions_by_group"] == {
        "imbalanced": 0,
        "missing": None,
    }
    assert endpoints["balanced_predictions_by_group"] == {
        "imbalanced": 1,
        "missing": None,
    }
    assert endpoints["raw_all_hit_rate"] == 0.0
    assert endpoints["balanced_all_hit_rate"] == 0.5
    assert endpoints["eligible_by_repetition"][1] == 1
    assert endpoints["correct_by_repetition"][1] == 1

    tempered = candidate_evidence_endpoints(
        logits=[0.5, 0.5, 0.5, 1.0],
        digits=[0, 0, 0, 1],
        group_ids=["imbalanced"] * 4,
        truth_by_group={"imbalanced": 1},
        repetition_indices=[0, 1, 2, 0],
        aggregation="tempered_evidence",
        evidence_count_power=0.5,
        candidate_vocabulary=(0, 1),
    )
    assert tempered["raw_predictions_by_group"] == {"imbalanced": 1}
    assert tempered["raw_all_hit_rate"] == 1.0
