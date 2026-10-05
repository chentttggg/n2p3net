"""Synthetic brainsync builders for contract tests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import mne
import numpy as np

from data.brainsync_contract import BRAIN_SYNC_RAW_TIME_BASE, BRAIN_SYNC_SESSION_SCHEMA
from data.contract import BRAIN_SYNC_ACQUISITION_CHANNELS, BRAIN_SYNC_ACQUISITION_TYPES

CHANNELS = BRAIN_SYNC_ACQUISITION_CHANNELS


POSITIONS = [
    [-0.001229276531, 0.093274451792, 0.102639295161],
    [-0.001374138752, 0.027617093176, 0.140199497342],
    [-0.055038236082, -0.044210303575, 0.099908977747],
    [-0.001709450386, -0.045212991536, 0.126672923565],
    [0.053636010736, -0.044334523380, 0.100516028702],
    [0.053557317704, -0.066416941583, 0.047850977629],
    None,
    [-0.002060251543, -0.082782991230, 0.060736630111],
]


def write_bids_session(
    root: Path,
    *,
    session_id: str,
    target: int,
    started_utc: str,
    subject_id: str = "P001",
    age: str = "19",
    blocks: tuple[tuple[int, ...], ...] = (tuple(range(1, 10)),),
    source_rate_hz: float = 250.0,
    eog_scale: float = 1.0,
) -> Path:
    """Write a self-contained workstation v4 BIDS raw session."""

    subject_label = "P001"
    session_label = "session"
    eeg_dir = root / f"sub-{subject_label}" / f"ses-{session_label}" / "eeg"
    eeg_dir.mkdir(parents=True)
    rest_duration_seconds = 2.0
    onset = 1.0
    event_rows: list[list[object]] = []
    rest_segments: list[dict[str, float]] = []
    for block_id, digits in enumerate(blocks, start=1):
        for trial_index, digit in enumerate(digits, start=1):
            trial_id = f"B{block_id:03d}-T{trial_index:03d}"
            event_rows.append(
                [
                    onset,
                    0.3,
                    round(onset * source_rate_hz),
                    "stimulus",
                    digit,
                    "stimulus",
                    trial_id,
                    block_id,
                    trial_index,
                    digit,
                    "n/a",
                ]
            )
            onset += 1.5
        if block_id < len(blocks):
            rest_start = onset
            rest_end = rest_start + rest_duration_seconds
            event_rows.append(
                [
                    rest_start,
                    rest_duration_seconds,
                    round(rest_start * source_rate_hz),
                    "rest",
                    "rest",
                    "rest",
                    "n/a",
                    block_id,
                    "n/a",
                    "n/a",
                    f"block-{block_id}",
                ]
            )
            rest_segments.append(
                {
                    "start_seconds": rest_start,
                    "end_seconds": rest_end,
                    "duration_seconds": rest_duration_seconds,
                }
            )
            onset = rest_end + 0.5
    duration_seconds = onset + 1.5
    n_samples = int(round(duration_seconds * source_rate_hz))
    info = mne.create_info(
        list(CHANNELS),
        source_rate_hz,
        ch_types=[kind.lower() for kind in BRAIN_SYNC_ACQUISITION_TYPES],
    )
    values = np.random.default_rng(sum(map(ord, session_id))).normal(
        0.0, 5e-6, (len(CHANNELS), n_samples)
    )
    values[CHANNELS.index("EOG")] *= eog_scale
    stem = f"sub-{subject_label}_ses-{session_label}_task-gtn_run-01"
    raw_path = eeg_dir / f"{stem}_eeg.edf"
    mne.io.RawArray(values, info, verbose=False).export(
        raw_path, fmt="edf", overwrite=True, physical_range="auto", verbose=False
    )
    started = datetime.fromisoformat(started_utc.replace("Z", "+00:00"))
    confirmed_utc = (started + timedelta(seconds=duration_seconds + 1.0)).isoformat()
    ended_utc = (started + timedelta(seconds=duration_seconds + 2.0)).isoformat()
    session = {
        "schema": BRAIN_SYNC_SESSION_SCHEMA,
        "session_id": session_id,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "status": "completed",
        "experiment": {
            "subject_id": subject_id,
            "age": age,
            "sex": "F",
            "blocks": len(blocks),
            "digits_per_block": len(blocks[0]),
            "sequence_policy": "balanced_random_permutation_cycles",
            "repetitions_per_digit": len(blocks[0]) // 9,
            "thought_digit": target,
            "target_label_status": "confirmed_post_experiment",
        },
        "target_label": {
            "status": "confirmed",
            "source": "post_experiment_confirmation",
            "thought_digit": target,
            "confirmed_utc": confirmed_utc,
        },
        "recording": {
            "path": raw_path.relative_to(root).as_posix(),
            "source_sample_rate_hz": source_rate_hz,
            "stage": "bids_raw",
            "preprocessing_status": "pending",
            "timeline": {
                "status": "finalized",
                "time_base": BRAIN_SYNC_RAW_TIME_BASE,
                "rest_segments_retained": True,
                "rest_segments": rest_segments,
                "source_sample_rate_hz": source_rate_hz,
                "output_duration_seconds": n_samples / source_rate_hz,
            },
            "bids": {
                "dataset_description": "dataset_description.json",
                "recording": raw_path.relative_to(root).as_posix(),
                "eeg_json": (eeg_dir / f"{stem}_eeg.json").relative_to(root).as_posix(),
                "channels_tsv": (eeg_dir / f"{stem}_channels.tsv").relative_to(root).as_posix(),
                "events_tsv": (eeg_dir / f"{stem}_events.tsv").relative_to(root).as_posix(),
                "electrodes_tsv": (
                    eeg_dir
                    / f"sub-{subject_label}_ses-{session_label}_space-BrainSyncHead_electrodes.tsv"
                )
                .relative_to(root)
                .as_posix(),
                "coordsystem_json": (
                    eeg_dir
                    / f"sub-{subject_label}_ses-{session_label}_space-BrainSyncHead_coordsystem.json"
                )
                .relative_to(root)
                .as_posix(),
            },
        },
        "quality": {"eeg_continuity": {"passed": True}},
    }
    (root / "session.json").write_text(json.dumps(session), encoding="utf-8")
    (root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "BrainSync test",
                "BIDSVersion": "1.11.0",
                "DatasetType": "raw",
            }
        ),
        encoding="utf-8",
    )
    (eeg_dir / f"{stem}_eeg.json").write_text(
        json.dumps(
            {
                "TaskName": "gtn",
                "SamplingFrequency": source_rate_hz,
                "RecordingDuration": n_samples / source_rate_hz,
                "EEGReference": "A2",
                "EOGDerivation": {
                    "EOG": {
                        "kind": "single_ended",
                        "placement": "left_eye_corner",
                        "reference": "A2",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (eeg_dir / f"{stem}_channels.tsv").write_text(
        "name\ttype\tunits\tsampling_frequency\tstatus\n"
        + "".join(
            f"{label}\t{kind}\tuV\t{source_rate_hz:g}\tgood\n"
            for label, kind in zip(CHANNELS, BRAIN_SYNC_ACQUISITION_TYPES, strict=True)
        ),
        encoding="utf-8",
    )
    (
        eeg_dir / f"sub-{subject_label}_ses-{session_label}_space-BrainSyncHead_electrodes.tsv"
    ).write_text(
        "name\tx\ty\tz\n"
        + "".join(
            (
                f"{label}\tn/a\tn/a\tn/a\n"
                if position is None
                else f"{label}\t{position[0]}\t{position[1]}\t{position[2]}\n"
            )
            for label, position in zip(CHANNELS, POSITIONS, strict=True)
        ),
        encoding="utf-8",
    )
    (
        eeg_dir / f"sub-{subject_label}_ses-{session_label}_space-BrainSyncHead_coordsystem.json"
    ).write_text(
        json.dumps(
            {
                "EEGCoordinateSystem": "Other",
                "EEGCoordinateUnits": "m",
                "EEGCoordinateSystemDescription": "Synthetic MNE head coordinates.",
            }
        ),
        encoding="utf-8",
    )
    event_rows.sort(key=lambda row: float(row[0]))
    event_header = (
        "onset\tduration\tsample\ttrial_type\tvalue\tevent_kind\ttrial_id\t"
        "block_id\ttrial_index\tdigit\trest_segment_id\n"
    )
    (eeg_dir / f"{stem}_events.tsv").write_text(
        event_header + "".join("\t".join(str(value) for value in row) + "\n" for row in event_rows),
        encoding="utf-8",
    )
    return root


def write_recording_clock(root: Path, *, acquired: int = 251, serialized: int = 500) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "recording.edf.clock.json"
    journal_path = root / "recording.edf.clock.jsonl"
    row = {
        "batch_id": 0,
        "received_start": 0,
        "n_received": acquired,
        "n_stored": acquired,
        "stored_start": 0,
        "segment_id": 0,
        "sequence_valid": True,
        "seq_num": list(range(acquired)),
        "delta_time_raw": [400] * acquired,
    }
    contents = (json.dumps(row) + "\n").encode()
    journal_path.write_bytes(contents)
    metadata = {
        "schema": "brainsync-sample-clock/1",
        "finalized": True,
        "integrity_failed": False,
        "sample_rate_hz": 250,
        "alignment_level": "delivery_anchored",
        "absolute_offset_bound_seconds": None,
        "stored_samples": acquired,
        "acquired_samples": acquired,
        "voltage_samples": acquired,
        "edf_samples": serialized,
        "padding_samples": serialized - acquired,
        "received_samples": acquired,
        "batch_count": 1,
        "segments": [{"segment_id": 0, "stored_start": 0, "stored_stop": acquired}],
        "journal_sha256": hashlib.sha256(contents).hexdigest(),
        "metadata_path": str(metadata_path),
        "journal_path": str(journal_path),
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return {
        "clock": {
            **metadata,
            "metadata_path": metadata_path.name,
            "journal_path": journal_path.name,
        }
    }
