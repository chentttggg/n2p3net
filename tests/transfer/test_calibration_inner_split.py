from __future__ import annotations

from dataclasses import replace

import numpy as np

from data.brainsync import load_brainsync_sessions
from tests.support.brainsync import write_bids_session
from tests.support.datasets import binary_dataset
from transfer.calibration_split import calibration_inner_split, dataset_calibration_inner_split


def test_single_recording_purges_raw_support_inside_authorized_calibration():
    onset = np.arange(30, dtype=float)
    allowed = np.arange(30) < 20
    split = calibration_inner_split(
        np.arange(30) % 2, onset, onset + 1.2, np.repeat("recording", 30),
        tmin_s=-0.2, calibration_mask=allowed,
    )
    assert split.legal and split.mode == "within_recording_purged_80_20"
    np.testing.assert_array_equal(np.flatnonzero(split.training_mask), np.arange(16))
    np.testing.assert_array_equal(np.flatnonzero(split.purged_mask), [16])
    np.testing.assert_array_equal(np.flatnonzero(split.validation_mask), [17, 18, 19])
    assert not ((split.training_mask | split.validation_mask) & ~allowed).any()


def test_real_builder_uses_session_timestamps_instead_of_identifiers(tmp_path):
    roots = [
        write_bids_session(
            tmp_path / name, session_id=name, target=target,
            started_utc=f"2026-09-06T08:{index * 3:02d}:00Z",
            blocks=(tuple(range(1, 10)) * 3,),
        )
        for index, (name, target) in enumerate((("z-earlier", 1), ("a-later", 3)))
    ]
    dataset = load_brainsync_sessions(roots)
    split = dataset_calibration_inner_split(dataset)
    assert split.legal
    assert split.training_recordings == ("z-earlier",)
    assert split.validation_recordings == ("a-later",)


def test_missing_class_has_no_selection_evidence_and_never_borrows_final_rows():
    onset = np.arange(20, dtype=float)
    labels = np.zeros(20, dtype=int)
    labels[[2, 18]] = 1
    split = calibration_inner_split(
        labels, onset, onset + 0.5, np.repeat("recording", 20), tmin_s=-0.2,
        calibration_mask=np.arange(20) < 10,
    )
    assert not split.legal
    assert "both binary classes" in split.reason


def test_two_single_class_sessions_cannot_create_a_legal_training_session_pool():
    split = calibration_inner_split(
        np.asarray([0, 0, 1, 1, 0, 1]), np.tile([0.0, 2.0], 3), np.tile([1.0, 3.0], 3),
        np.repeat(["a", "b", "c"], 2), tmin_s=-0.2, recording_order=("a", "b", "c"),
    )
    assert not split.legal
    assert "every complete recording" in split.reason


def test_public_multi_run_uses_declared_run_axis_and_ignores_outside_scope():
    from data.events import observed_only_timeline
    from transfer.calibration_split import dataset_calibration_recording_order

    dataset = binary_dataset()
    runs = np.repeat(["run-10", "run-2", "final"], [10, 10, 20])
    sessions = np.repeat("session-one", 40)
    onsets = np.concatenate((np.arange(10), np.arange(10), np.arange(20))).astype(float)
    timeline = observed_only_timeline(
        dataset_id=dataset.name, subject_ids=dataset.subject_ids, stimulus_ids=np.arange(40),
        onset_times_s=onsets * 2, evidence_available_times_s=onsets * 2 + 0.8,
        session_ids=sessions, run_ids=runs, group_ids=runs.copy(),
    )
    dataset = replace(dataset, event_timeline=timeline)
    protocol = {"session_order": ["session-one"], "run_order": ["run-10", "run-2", "final"]}
    allowed = np.arange(40) < 20
    assert dataset_calibration_recording_order(dataset, allowed, protocol) == ("run-10", "run-2", "final")
    split = dataset_calibration_inner_split(dataset, allowed, recording_order=protocol)
    assert split.legal
    np.testing.assert_array_equal(np.flatnonzero(split.training_mask), np.arange(10))
    np.testing.assert_array_equal(np.flatnonzero(split.validation_mask), np.arange(10, 20))
    runs[20:] = "other"
    sessions[20:] = "new-session"
    changed = replace(dataset, event_timeline=replace(timeline, run_ids=runs, session_ids=sessions))
    other = dataset_calibration_inner_split(changed, allowed, recording_order=protocol)
    assert other.record() == split.record()
