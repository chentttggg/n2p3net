from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from data.events import ScheduledEventTimeline, observed_only_timeline


def _timeline() -> ScheduledEventTimeline:
    return ScheduledEventTimeline(
        event_ids=np.asarray(["e0", "e1"]),
        group_ids=np.asarray(["g", "g"]),
        subject_ids=np.asarray(["s", "s"]),
        stimulus_ids=np.asarray([1, 2]),
        onset_samples=np.asarray([0, 100]),
        onset_times_s=np.asarray([0.0, 1.0]),
        evidence_available_times_s=np.asarray([0.8, np.nan]),
        evidence_indices=np.asarray([0, -1]),
        statuses=np.asarray(["available", "artifact_rejected"]),
        status_details=np.asarray(["", "abs_voltage_gt_threshold"]),
        dataset_ids=np.asarray(["d", "d"]),
        session_ids=np.asarray(["session", "session"]),
        run_ids=np.asarray(["run", "run"]),
        selection_ids=np.asarray(["selection", "selection"]),
        complete=True,
        online_causal=False,
        timing_source="synthetic",
    ).validate(n_epochs=1)


def test_scheduled_event_timeline_owns_read_only_arrays() -> None:
    source = np.asarray(["e0", "e1"])
    timeline = replace(_timeline(), event_ids=source)
    source[0] = "changed"
    assert timeline.event_ids[0] == "e0"
    with pytest.raises(ValueError, match="read-only"):
        timeline.statuses[0] = "missing"


def test_timeline_fingerprint_binds_rejection_detail() -> None:
    timeline = _timeline()
    changed = replace(
        timeline,
        status_details=np.asarray(["", "different_artifact_reason"]),
    ).validate(n_epochs=1)
    assert changed.fingerprint() != timeline.fingerprint()


def test_timeline_rejects_pre_onset_availability_and_mixed_recording_group() -> None:
    timeline = _timeline()
    with pytest.raises(ValueError, match="before its scheduled onset"):
        replace(
            timeline,
            evidence_available_times_s=np.asarray([-0.1, np.nan]),
        ).validate(n_epochs=1)
    with pytest.raises(ValueError, match="multiple run_ids"):
        replace(timeline, run_ids=np.asarray(["run-a", "run-b"])).validate(n_epochs=1)


def test_timeline_rejects_lossy_integer_and_boolean_coercion() -> None:
    timeline = _timeline()
    with pytest.raises(ValueError, match="coercion is forbidden"):
        replace(timeline, stimulus_ids=np.asarray([1.9, 2.0]))
    with pytest.raises(ValueError, match="strict booleans"):
        replace(timeline, complete="False")


def test_observed_timeline_rejects_fractional_stimulus_ids() -> None:
    with pytest.raises(ValueError, match="integer array"):
        observed_only_timeline(
            dataset_id="d",
            subject_ids=np.array(["s"]),
            stimulus_ids=np.array([1.5]),
        )


def test_candidate_contract_is_string_safe_and_keeps_unavailable_rows() -> None:
    timeline = replace(
        _timeline(),
        candidate_ids=np.asarray(["left", "right"]),
        target_candidate_ids=np.asarray(["right", "right"]),
        repetition_indices=np.asarray([0, 0], dtype=np.int64),
    ).validate(n_epochs=1)

    assert timeline.has_candidate_sets is True
    assert timeline.has_repetition_structure is True
    assert timeline.supports_full_candidate_chain is True
    scheduled = timeline.encoded_candidate_selection(available_only=False)
    available = timeline.encoded_candidate_selection()
    assert scheduled.vocabulary == ("left", "right")
    assert scheduled.candidate_codes.tolist() == [0, 1]
    assert scheduled.target_codes.tolist() == [1, 1]
    assert available.candidate_codes.tolist() == [0]
    assert scheduled.truth_by_group == {"g": 1}


def test_candidate_metadata_fails_closed_when_target_is_missing() -> None:
    timeline = replace(
        _timeline(),
        candidate_ids=np.asarray(["A", "B"]),
    ).validate(n_epochs=1)

    assert timeline.has_candidate_ids is True
    assert timeline.has_candidate_sets is False
    assert timeline.supports_full_candidate_chain is False


def test_candidate_contract_rejects_mixed_truth_and_repetition_order() -> None:
    with pytest.raises(ValueError, match="mixed target_candidate_ids"):
        replace(
            _timeline(),
            candidate_ids=np.asarray(["A", "B"]),
            target_candidate_ids=np.asarray(["A", "B"]),
        ).validate(n_epochs=1)

    four = ScheduledEventTimeline(
        event_ids=np.asarray(["e0", "e1", "e2", "e3"]),
        group_ids=np.repeat("g", 4),
        subject_ids=np.repeat("s", 4),
        stimulus_ids=np.asarray([10, 20, 10, 20]),
        onset_samples=np.arange(4),
        onset_times_s=np.arange(4, dtype=float),
        evidence_available_times_s=np.arange(4, dtype=float),
        evidence_indices=np.arange(4),
        statuses=np.repeat("available", 4),
        status_details=np.repeat("", 4),
        dataset_ids=np.repeat("d", 4),
        session_ids=np.repeat("session", 4),
        run_ids=np.repeat("run", 4),
        selection_ids=np.repeat("selection", 4),
        complete=True,
        online_causal=False,
        timing_source="synthetic",
        candidate_ids=np.asarray(["A", "B", "A", "B"]),
        target_candidate_ids=np.repeat("B", 4),
        repetition_indices=np.asarray([1, 0, 0, 1]),
    )
    with pytest.raises(ValueError, match="non-increasing repetition_indices"):
        four.validate(n_epochs=4)
