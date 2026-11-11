"""Original scheduled missingness and tie cases keep a scored denominator."""
import numpy as np

from experiments.analyze_brainsync_consistent import decision_records
from research.target_recognition import summarize_target_decisions
from tests.brainsync.test_real_matrix_driver import fixture


def test_single_round_missing_flash_is_unscored_and_does_not_shift_later_cycles():
    dataset, logits, _ = fixture(removed=(8,))
    dataset.n_epochs = len(logits)
    records, oracles = decision_records(dataset, logits, session_id="S", revealed_target="1", repetitions=1)
    metric = summarize_target_decisions(records)
    assert len(records) == 50 and oracles == 49
    assert records[0]["prediction"] is None
    assert records[1]["decision_id"] == "S:R1:block:1:cycle:2"
    assert records[1]["prediction"] == "1"
    assert metric["scored_decisions"] == 49
    assert metric["accuracy"] == metric["correct_decisions"] / 49
    assert metric["coverage"] == .98


def test_five_round_ties_are_excluded_from_accuracy_and_truth_cannot_change_predictions():
    dataset, logits, _ = fixture()
    dataset.n_epochs = len(logits)
    logits[:] = 0.
    first, _ = decision_records(dataset, logits, session_id="S", revealed_target="1", repetitions=5)
    second, _ = decision_records(dataset, logits, session_id="S", revealed_target="9", repetitions=5)
    assert all(r["prediction"] is None for r in first)
    assert summarize_target_decisions(first)["accuracy"] is None
    np.testing.assert_array_equal([r["posterior"] for r in first], [r["posterior"] for r in second])
