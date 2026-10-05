"""Finalized sample-clock segments must agree with the BIDS pause boundaries."""

import hashlib
import json

import pytest

from data.brainsync import load_brainsync_session
from data.brainsync_contract import validate_brainsync_bids_session
from tests.support.brainsync import write_bids_session, write_recording_clock
from tests.support.io import read_json, read_tsv, write_json, write_tsv


@pytest.fixture
def recording(tmp_path):
    return write_bids_session(
        tmp_path / "session", session_id="clock-segments", target=1,
        started_utc="2026-09-06T08:00:00Z",
    )


def _write_clock_segments(root, boundaries):
    validated = validate_brainsync_bids_session(root)
    count = round(validated.bids.duration_seconds * validated.bids.sample_rate_hz)
    clock = write_recording_clock(root, acquired=count, serialized=count)["clock"]
    metadata_path, journal_path = root / clock["metadata_path"], root / clock["journal_path"]
    metadata = read_json(metadata_path)
    original = read_json(journal_path)
    edges = (0, *boundaries, count)
    segments = list(zip(edges[:-1], edges[1:], strict=True))
    rows = [
        {**original, "batch_id": index, "received_start": start, "stored_start": start,
         "n_received": stop-start, "n_stored": stop-start, "segment_id": index,
         "seq_num": original["seq_num"][start:stop],
         "delta_time_raw": original["delta_time_raw"][start:stop]}
        for index, (start, stop) in enumerate(segments)
    ]
    journal_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metadata.update(
        batch_count=len(segments), journal_sha256=hashlib.sha256(journal_path.read_bytes()).hexdigest(),
        segments=[{"segment_id": i, "stored_start": start, "stored_stop": stop}
                  for i, (start, stop) in enumerate(segments)],
    )
    write_json(metadata_path, metadata)
    manifest_path = root / "session.json"
    manifest = read_json(manifest_path)
    manifest["recording"]["clock"] = {
        **metadata, "metadata_path": clock["metadata_path"], "journal_path": clock["journal_path"],
    }
    write_json(manifest_path, manifest)


def _add_pause(root, boundary_sample):
    manifest_path = root / "session.json"
    manifest = read_json(manifest_path)
    recording = manifest["recording"]
    sidecar_path = root / recording["bids"]["eeg_json"]
    sidecar = read_json(sidecar_path)
    onset = boundary_sample / sidecar["SamplingFrequency"]
    pauses = [{"pause_id": 1, "boundary_sample": boundary_sample,
               "start_wall_seconds": onset, "end_wall_seconds": onset + 1.0}]
    recording["timeline"].update(
        time_base="segmented_recording", acquisition_pauses=pauses, pause_samples_retained=False,
    )
    sidecar.update(RecordingType="discontinuous", AcquisitionPauses=pauses)
    write_json(sidecar_path, sidecar)
    events_path = root / recording["bids"]["events_tsv"]
    rows = read_tsv(events_path)
    boundary = dict.fromkeys(rows[0], "n/a")
    boundary.update(onset=str(onset), duration="0", sample=str(boundary_sample),
                    trial_type="acquisition_boundary", value="pause", event_kind="acquisition_boundary", block_id="1")
    rows.append(boundary)
    rows.sort(key=lambda row: float(row["onset"]))
    write_tsv(events_path, rows)
    write_json(manifest_path, manifest)


@pytest.mark.parametrize("clock_boundaries,pause_boundaries", [
    ((300,), ()), ((), (300,)), ((300,), (400,)),
])
def test_clock_and_pause_sources_cannot_disagree(recording, clock_boundaries, pause_boundaries):
    _write_clock_segments(recording, clock_boundaries)
    for boundary in pause_boundaries:
        _add_pause(recording, boundary)
    with pytest.raises(ValueError, match="Clock acquisition segment boundaries"):
        validate_brainsync_bids_session(recording)


@pytest.fixture
def paused_recording(recording):
    _add_pause(recording, 300)
    return recording


def _assert_cross_segment_epoch_excluded(root):
    dataset = load_brainsync_session(root)
    assert dataset.event_timeline.evidence_indices[0] == -1
    assert dataset.n_epochs == 8
    assert dataset.event_timeline.n_events == 9


def test_matching_clock_boundaries_exclude_cross_segment_epochs(paused_recording):
    _write_clock_segments(paused_recording, (300,))
    _assert_cross_segment_epoch_excluded(paused_recording)


def test_legacy_pause_without_clock_excludes_cross_segment_epochs(paused_recording):
    _assert_cross_segment_epoch_excluded(paused_recording)
