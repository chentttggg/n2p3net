from __future__ import annotations

import numpy as np
import pytest

from transfer.evaluation import candidate_evidence_endpoints, hit_at_repetition


@pytest.mark.parametrize(
    "vocabulary,truth,amplitudes",
    [(range(1, 5), (2, 3), (1.0, 1.5)), (range(9), (0, 8), (1.0, 1.0))],
    ids=["one-based-four", "zero-based-nine"],
)
def test_hit_at_repetition_sums_target_logits(vocabulary, truth, amplitudes) -> None:
    digits = np.tile(np.repeat(vocabulary, 2), 2)
    groups = np.repeat(["a", "b"], len(vocabulary) * 2)
    repetitions = np.tile([0, 1], len(vocabulary) * 2)
    logits = np.zeros(len(digits))
    for group, target, amplitude in zip(("a", "b"), truth, amplitudes, strict=True):
        logits[(groups == group) & (digits == target)] = amplitude
    hits = hit_at_repetition(
        logits,
        digits,
        groups,
        dict(zip(("a", "b"), truth, strict=True)),
        repetitions,
        aggregation="sum",
        max_repetitions=2,
    )
    assert hits[1] == hits[2] == 1.0


def test_hit_at_repetition_rejects_nonfinite_logits() -> None:
    with pytest.raises(ValueError, match="NaN"):
        hit_at_repetition(
            [1.0, float("nan")],
            [1, 2],
            ["g", "g"],
            {"g": 1},
            [0, 0],
        )


@pytest.mark.parametrize("logits,expected", [([2.0, 1.0, 0.0, 0.0], 1.0), ([0.0] * 4, 0.0)])
def test_trim_two_repetitions_keeps_both_values_and_ties_abstain(logits, expected) -> None:
    hits = hit_at_repetition(
        logits,
        [1, 1, 2, 2],
        ["g"] * 4,
        {"g": 1},
        [0, 1, 0, 1],
        aggregation="trim0.2",
        max_repetitions=2,
        candidate_vocabulary=(1, 2),
    )
    assert hits[2] == expected


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


def test_precision_aggregation_requires_predictive_variance() -> None:
    logits = np.asarray([1.0, 2.0, 0.1, 0.1])
    digits = np.asarray([1, 1, 2, 2])
    groups = np.repeat("g", 4)
    repetitions = np.asarray([0, 1, 0, 1])

    with pytest.raises(ValueError, match="predictive variances"):
        hit_at_repetition(
            logits,
            digits,
            groups,
            {"g": 1},
            repetitions,
            aggregation="precision",
        )

    hits = hit_at_repetition(
        logits,
        digits,
        groups,
        {"g": 1},
        repetitions,
        aggregation="precision",
        logit_variances=np.asarray([1.0, 1.0, 0.01, 0.01]),
    )
    assert hits[2] == 1.0
