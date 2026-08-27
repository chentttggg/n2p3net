from __future__ import annotations

import json

import mne
import numpy as np

from data.brainsync import BRAIN_SYNC_PREPROCESSING, load_brainsync_session

CHANNELS = ("Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz")
POSITIONS = [
    [0.0003122, 0.058512, 0.066462],
    [0.0004009, -0.009167, 0.100244],
    [-0.0530073, -0.0787878, 0.05594],
    [0.0003247, -0.081115, 0.082615],
    [0.0556667, -0.0785602, 0.056561],
    [-0.0548404, -0.0975279, 0.002792],
    [0.0556666, -0.0976251, 0.00273],
    [0.0001076, -0.114892, 0.014657],
]


def test_brainsync_session_is_preprocessed_into_universal_dataset(tmp_path) -> None:
    session_dir = tmp_path / "session"
    raw_dir = session_dir / "raw"
    events_dir = session_dir / "events"
    raw_dir.mkdir(parents=True)
    events_dir.mkdir()

    info = mne.create_info(list(CHANNELS), 1000.0, ch_types="eeg")
    values = np.random.default_rng(7).normal(0.0, 5e-6, (len(CHANNELS), 5000))
    mne.io.RawArray(values, info, verbose=False).save(
        raw_dir / "recording_raw.fif", overwrite=True, verbose=False
    )
    session = {
        "schema": "brainsync-gtn-session/2",
        "session_id": "P001_test",
        "experiment": {"subject_id": "P001", "age": "19", "sex": "F"},
        "target_label": {"status": "confirmed", "thought_digit": 2},
        "recording": {"path": "raw/recording_raw.fif"},
        "montage": {
            "labels": list(CHANNELS),
            "active_mask": 0xFF,
            "channel_positions_m": POSITIONS,
            "coordinate_frame": "head",
            "units": "m",
        },
    }
    (session_dir / "session.json").write_text(json.dumps(session), encoding="utf-8")
    marker_rows = [
        {"trial_id": "B001-T001", "block_id": 1, "trial_index": 1, "digit": 1, "eeg_time_seconds": 1.0},
        {"trial_id": "B001-T002", "block_id": 1, "trial_index": 2, "digit": 2, "eeg_time_seconds": 2.0},
        {"trial_id": "B001-T003", "block_id": 1, "trial_index": 3, "digit": 1, "eeg_time_seconds": 3.0},
    ]
    (events_dir / "events.jsonl").write_text(
        "".join(
            json.dumps({"event": "recording_marker", "payload": {"kind": "onset", **row}}) + "\n"
            for row in marker_rows
        ),
        encoding="utf-8",
    )

    dataset = load_brainsync_session(session_dir, preprocessing=BRAIN_SYNC_PREPROCESSING)

    assert dataset.X.shape == (3, 8, 250)
    assert dataset.X.dtype == np.float32
    assert np.array_equal(dataset.y, np.array([0, 1, 0], dtype=np.int64))
    assert dataset.preprocessing.sfreq == 250.0
    assert np.allclose(dataset.channel_positions_m, np.asarray(POSITIONS, dtype=np.float32))
    assert dataset.metadata["repetition_index"].tolist() == [0, 0, 1]
    assert dataset.event_timeline.supports_full_candidate_chain is True
