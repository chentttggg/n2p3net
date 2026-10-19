"""Shared original EDF/BIDS requests for the adult programme contract tests."""
from __future__ import annotations

from datetime import datetime

from data import brainsync_raw_authority as authority
from data.contract import brainsync_reference_record
from research import adult_programme_input as module
from tests.support.brainsync import (
    write_bids_session,
    write_paused_recording,
    write_recording_clock,
)
from tests.support.io import read_json, write_json


def original_session(root, name, *, started, blocks=1):
    root = write_bids_session(root, session_id=name, target=3, started_utc=started,
        blocks=(tuple(range(1, 10))*5,)*blocks)
    manifest = read_json(root/'session.json')
    sidecar_path = root/manifest['recording']['bids']['eeg_json']
    sidecar = read_json(sidecar_path)
    reference = brainsync_reference_record()
    manifest['montage'] = {'ref_label': reference['label'], 'reference_details': reference}
    sidecar.update(EEGReference=reference['label'], EEGReferenceDetails=reference)
    sidecar['EOGDerivation']['EOG']['reference'] = reference['label']
    count = round(sidecar['RecordingDuration']*250)
    manifest['recording']['clock'] = write_recording_clock(root, acquired=count, serialized=count)['clock']
    write_json(sidecar_path, sidecar)
    write_json(root/'session.json', manifest)
    return root


def read_requests(root):
    record = read_json(root/'session.json')
    return tuple(module.NativeReadRequest(row['segment_id'], row['stored_start']+125,
        row['stored_stop'], row['stored_stop']) for row in record['recording']['clock']['segments'])


def build_complete_requests(tmp_path):
    # Reuse the actual canonical two-segment/pause/interrupted-exposure builder.
    c1 = write_paused_recording(tmp_path/'first-C')
    c2 = original_session(tmp_path/'second-C', 'adult-second-C', started='2026-10-01T08:10:00Z')
    held = original_session(tmp_path/'held', 'adult-held', started='2026-10-01T09:00:00Z', blocks=2)
    held_record = read_json(held/'session.json')
    # Valid public recording metadata; deliberately unusable supervised truth.
    held_record['experiment']['thought_digit'] = {'private': 'not supplied'}
    held_record['target_label'] = {'status': 'withheld', 'thought_digit': 'not supplied'}
    write_json(held/'session.json', held_record)
    rows = [read_json(path/'session.json') for path in (c1, c2, held)]
    calibration = tuple(module.CalibrationRecordingRequest(record['session_id'], path, read_requests(path), count,
        datetime.fromisoformat(record['ended_utc']).timestamp())
        for path, record, count in zip((c1, c2), rows[:2], (90, 45), strict=True))
    inference = (module.HeldRecordingRequest(rows[2]['session_id'], held, read_requests(held)),)
    scope = authority.BrainSyncRecordingScope(tuple(row.session_id for row in calibration),
        tuple(row.session_id for row in inference))
    return calibration, inference, scope

