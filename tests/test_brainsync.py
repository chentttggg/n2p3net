from __future__ import annotations

import json
from dataclasses import asdict, replace

import mne
import numpy as np
import pytest
import torch

from data.brainsync import (
    BRAIN_SYNC_PREPROCESSING,
    load_brainsync_session,
    load_brainsync_sessions,
)
from data.contract import (
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    SOURCE_COHORT_DATA_CONTRACTS,
    assert_causal_p300_input_contract,
)
from data.epochs import save_epoch_dataset
from experiments.run_brainsync_cross_decision import main as run_brainsync_cross_decision
from models.n2p3net import N2P3Net
from transfer.within_subject import calibration_decision_split

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


def test_brainsync_uses_the_canonical_causal_profile() -> None:
    assert BRAIN_SYNC_PREPROCESSING.name == SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.name
    assert BRAIN_SYNC_PREPROCESSING.l_freq == 0.1
    assert BRAIN_SYNC_PREPROCESSING.h_freq == 30.0
    assert BRAIN_SYNC_PREPROCESSING.tmax_ms == 1200.0
    assert BRAIN_SYNC_PREPROCESSING.filter_phase == "forward"
    assert BRAIN_SYNC_PREPROCESSING.causal_iir_initial_state == "steady_state_first_sample"
    assert BRAIN_SYNC_PREPROCESSING.n_times == SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times
    assert (
        SOURCE_COHORT_DATA_CONTRACTS["causal"]
        is SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
    )


def test_brainsync_rejects_the_removed_2hz_800ms_contract() -> None:
    removed = replace(
        BRAIN_SYNC_PREPROCESSING,
        name="removed_causal_2hz_800ms",
        l_freq=2.0,
        tmax_ms=800.0,
        n_times=128,
    )
    with pytest.raises(ValueError, match="Regenerate the cache"):
        assert_causal_p300_input_contract(removed)


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
            "ref_label": "A2",
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

    assert dataset.X.shape == (3, 8, BRAIN_SYNC_PREPROCESSING.n_times)
    assert dataset.X.dtype == np.float32
    assert np.array_equal(dataset.y, np.array([0, 1, 0], dtype=np.int64))
    assert dataset.preprocessing.sfreq == 128.0
    assert dataset.preprocessing.filter_phase == "forward"
    assert dataset.preprocessing.causal_iir_initial_state == "steady_state_first_sample"
    assert dataset.provenance["source_sample_rate_hz"] == 1000.0
    assert dataset.provenance["target_sample_rate_hz"] == 128.0
    baseline_samples = round(0.2 * dataset.preprocessing.sfreq)
    np.testing.assert_allclose(dataset.X[:, :, :baseline_samples].mean(axis=2), 0.0, atol=1e-10)
    assert np.allclose(dataset.channel_positions_m, np.asarray(POSITIONS, dtype=np.float32))
    assert dataset.metadata["repetition_index"].tolist() == [0, 0, 1]
    assert len(set(dataset.metadata["selection_id"])) == 1
    assert dataset.event_timeline.supports_full_candidate_chain is True


def _write_multi_decision_session(
    root,
    *,
    session_id: str,
    subject_id: str = "P001",
    block_targets: tuple[int, ...] = (1, 2),
) -> None:
    raw_dir = root / "raw"
    events_dir = root / "events"
    raw_dir.mkdir(parents=True)
    events_dir.mkdir()
    info = mne.create_info(list(CHANNELS), 1000.0, ch_types="eeg")
    n_samples = (len(block_targets) * 9 + 3) * 1000
    values = np.random.default_rng(len(session_id)).normal(
        0.0, 5e-6, (len(CHANNELS), n_samples)
    )
    mne.io.RawArray(values, info, verbose=False).save(
        raw_dir / "recording_raw.fif", overwrite=True, verbose=False
    )
    session = {
        "schema": "brainsync-gtn-session/2",
        "session_id": session_id,
        "started_utc": (
            "2026-09-01T00:00:00+00:00"
            if session_id.endswith("_a")
            else "2026-09-02T00:00:00+00:00"
        ),
        "experiment": {"subject_id": subject_id, "age": "19", "sex": "F"},
        "recording": {"path": "raw/recording_raw.fif"},
        "montage": {
            "labels": list(CHANNELS),
            "active_mask": 0xFF,
            "channel_positions_m": POSITIONS,
            "coordinate_frame": "head",
            "units": "m",
            "ref_label": "A2",
        },
    }
    (root / "session.json").write_text(json.dumps(session), encoding="utf-8")
    rows = []
    onset = 1.0
    for block, target in enumerate(block_targets, start=1):
        for digit in range(1, 10):
            rows.append(
                {
                    "trial_id": f"B{block:03d}-T{digit:03d}",
                    "block_id": block,
                    "trial_index": digit,
                    "digit": digit,
                    "target_digit": target,
                    "eeg_time_seconds": onset,
                }
            )
            onset += 1.0
    (events_dir / "events.jsonl").write_text(
        "".join(
            json.dumps({"event": "recording_marker", "payload": {"kind": "onset", **row}})
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_brainsync_blocks_form_distinct_target_changing_decisions(tmp_path) -> None:
    session_dir = tmp_path / "multi"
    _write_multi_decision_session(session_dir, session_id="P001_multi")

    dataset = load_brainsync_session(session_dir)

    assert len(np.unique(dataset.event_timeline.group_ids)) == 2
    assert len(np.unique(dataset.event_timeline.selection_ids)) == 2
    assert set(dataset.metadata["target_candidate_id"]) == {"1", "2"}
    assert dataset.metadata.groupby("selection_id")["repetition_index"].min().eq(0).all()


def test_brainsync_multi_session_loader_preserves_session_decisions(tmp_path) -> None:
    first = tmp_path / "session_a"
    second = tmp_path / "session_b"
    _write_multi_decision_session(first, session_id="P001_a", block_targets=(1,))
    _write_multi_decision_session(second, session_id="P001_b", block_targets=(2,))

    dataset = load_brainsync_sessions([first, second])

    assert dataset.name == "BrainSync-GTN-multisession"
    assert dataset.provenance["n_sessions"] == 2
    assert set(dataset.event_timeline.session_ids) == {"P001_a", "P001_b"}
    assert len(np.unique(dataset.event_timeline.group_ids)) == 2


def test_brainsync_cross_decision_split_enforces_real_time_embargo(tmp_path) -> None:
    session_dir = tmp_path / "three_decisions"
    _write_multi_decision_session(
        session_dir,
        session_id="P001_three",
        block_targets=(1, 2, 3),
    )
    dataset = load_brainsync_session(session_dir)

    split = calibration_decision_split(
        dataset,
        calibration_selections=1,
        test_repetitions=1,
        candidate_vocabulary=range(1, 10),
    )

    assert split.usable_subjects == ("P001",)
    assert len(split.requested_test_groups_by_subject["P001"]) == 2
    assert len(split.test_groups_by_subject["P001"]) == 1
    assert set(split.failed_test_groups_by_subject["P001"].values()) == {
        "selection_overlaps_calibration_evidence"
    }


def test_brainsync_cross_decision_runner_keeps_failed_decision_in_denominator(
    tmp_path, monkeypatch
) -> None:
    session_dir = tmp_path / "runner_decisions"
    _write_multi_decision_session(
        session_dir,
        session_id="P001_runner",
        block_targets=(1, 2, 3),
    )
    dataset = load_brainsync_session(session_dir)
    cache = save_epoch_dataset(tmp_path / "brainsync.npz", dataset)
    trunk = N2P3Net(
        dataset.n_channels,
        n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq,
        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
        pooling_mode="full_unfold",
        temporal_kernel_size=35,
    )
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "trunk_state_dict": trunk.state_dict(),
            "training_subject_keys": ["external-source\0other"],
            "training_subjects": ["other"],
            "source_dataset_name": "external-source",
            "input_channel_names": list(dataset.channel_names),
            "input_preprocessing": asdict(dataset.preprocessing),
            "input_source_reference": dataset.provenance["source_reference"],
            "classifier_trained": True,
            "input_mean": [0.0] * dataset.n_channels,
            "input_std": [1.0] * dataset.n_channels,
            "training_pos_weight": 8.0,
            "training_prior": 1.0 / 9.0,
            "architecture": trunk.architecture_record(),
            "n_channels": dataset.n_channels,
            "n_times": dataset.n_times,
            "input_sample_rate_hz": dataset.preprocessing.sfreq,
            "input_tmin_s": dataset.preprocessing.tmin_ms / 1000.0,
        },
        checkpoint,
    )
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_brainsync_cross_decision.py",
            "--dataset-cache",
            str(cache),
            "--checkpoint",
            str(checkpoint),
            "--calibration-selections",
            "1",
            "--test-reps",
            "1",
            "--head",
            "zero_shot",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )

    run_brainsync_cross_decision()

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["input_preprocessing"] == asdict(BRAIN_SYNC_PREPROCESSING)
    assert result["decision_accounting"]["requested"] == 2
    assert result["decision_accounting"]["eligible"] == 1
    assert result["decision_accounting"]["failed"] == 1
