import pytest

from experiments.evaluate_brainsync_remaining import unavailable_records
from research.target_recognition import summarize_target_decisions


@pytest.mark.parametrize("repetitions,requested", [(1, 50), (5, 10)])
def test_complete_qc_rejection_preserves_requested_roster_without_accuracy_errors(repetitions, requested):
    records = unavailable_records(session_id="S1", target="7", repetitions=repetitions,
        reason="no_usable_EEG_after_frozen_QC", scheduled=450)
    metric = summarize_target_decisions(records)
    assert metric["requested_decisions"] == requested
    assert metric["scored_decisions"] == 0 and metric["accuracy"] is None
    assert metric["coverage"] == 0
    assert metric["unscored_reasons"] == {"no_usable_EEG_after_frozen_QC": requested}
