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
