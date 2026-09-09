"""Finalized sample-clock segments must agree with the BIDS pause boundaries."""

import csv
import hashlib
import json

import pytest

from data.brainsync import load_brainsync_session
from data.brainsync_contract import validate_brainsync_bids_session
from tests.test_brainsync import _write_bids_session
from tests.test_recording_clock import clock_fixture


def _recording(tmp_path, *, clock_boundary=None, pause_boundary=None, with_clock=True):
    root = _write_bids_session(tmp_path / "session", session_id="clock-segments", target=1,
                              started_utc="2026-09-06T08:00:00Z")
    validated = validate_brainsync_bids_session(root)
    count = round(validated.bids.duration_seconds * validated.bids.sample_rate_hz)
    manifest_path = root / "session.json"
    manifest = json.loads(manifest_path.read_text())
    if with_clock:
        clock = clock_fixture(root, acquired=count, serialized=count)["clock"]
        metadata_path, journal_path = root / clock["metadata_path"], root / clock["journal_path"]
        metadata = json.loads(metadata_path.read_text())
        original = json.loads(journal_path.read_text())
        if clock_boundary is not None:
            segments = [(0, clock_boundary), (clock_boundary, count)]
            rows = [{**original, "batch_id": index, "received_start": start, "stored_start": start,
                     "n_received": stop-start, "n_stored": stop-start, "segment_id": index,
                     "seq_num": list(range(start, stop)), "delta_time_raw": [400] * (stop-start)}
                    for index, (start, stop) in enumerate(segments)]
            journal_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            metadata.update(batch_count=2, journal_sha256=hashlib.sha256(journal_path.read_bytes()).hexdigest(),
                            segments=[{"segment_id": i, "stored_start": start, "stored_stop": stop}
                                      for i, (start, stop) in enumerate(segments)])
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        manifest["recording"]["clock"] = {**metadata, "metadata_path": clock["metadata_path"],
                                          "journal_path": clock["journal_path"]}
    if pause_boundary is not None:
        pauses = [{"pause_id": 1, "boundary_sample": pause_boundary,
                   "start_wall_seconds": pause_boundary / 250., "end_wall_seconds": pause_boundary / 250. + 1.}]
        recording = manifest["recording"]
        recording["timeline"].update(time_base="segmented_recording", acquisition_pauses=pauses, pause_samples_retained=False)
        sidecar_path = root / recording["bids"]["eeg_json"]
        sidecar = json.loads(sidecar_path.read_text())
        sidecar.update(RecordingType="discontinuous", AcquisitionPauses=pauses)
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        events_path = root / recording["bids"]["events_tsv"]
        with events_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            fields, rows = reader.fieldnames, list(reader)
        boundary = dict.fromkeys(fields, "n/a")
        boundary.update(onset=str(pause_boundary / 250.), duration="0", sample=str(pause_boundary),
                        trial_type="acquisition_boundary", value="pause", event_kind="acquisition_boundary", block_id="1")
        rows.append(boundary)
        rows.sort(key=lambda row: float(row["onset"]))
        with events_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.mark.parametrize("clock_boundary,pause_boundary", [(300, None), (None, 300), (300, 400)])
def test_clock_and_pause_sources_cannot_disagree(tmp_path, clock_boundary, pause_boundary):
    root = _recording(tmp_path, clock_boundary=clock_boundary, pause_boundary=pause_boundary)
    with pytest.raises(ValueError, match="Clock acquisition segment boundaries"):
        validate_brainsync_bids_session(root)


@pytest.mark.parametrize("with_clock", [True, False])
def test_matching_boundaries_exclude_cross_segment_epochs_and_preserve_legacy(tmp_path, with_clock):
    root = _recording(tmp_path, clock_boundary=300, pause_boundary=300, with_clock=with_clock)
    dataset = load_brainsync_session(root)
    assert dataset.event_timeline.evidence_indices[0] == -1
    assert dataset.n_epochs == 8
    assert dataset.event_timeline.n_events == 9
