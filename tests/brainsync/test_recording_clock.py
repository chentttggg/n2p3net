from __future__ import annotations

import subprocess
import textwrap

import pytest

from data.recording_clock import validate_recording_clock
from tests.support.brainsync import write_recording_clock
from tests.support.io import read_json, write_json


def test_absolute_sidecar_and_relative_manifest_paths_share_the_same_identity(tmp_path):
    recording = write_recording_clock(tmp_path)
    result = validate_recording_clock(tmp_path, recording, sample_rate_hz=250, duration_seconds=2)
    assert result["acquired_samples"] == 251
    assert result["edf_samples"] == 500
    assert result["padding_samples"] == 249


@pytest.mark.parametrize("location", ["metadata_path", "journal_path"])
def test_manifest_clock_paths_must_resolve_to_contained_files(tmp_path, location):
    recording = write_recording_clock(tmp_path / "session")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    recording["clock"][location] = str(outside)
    with pytest.raises(ValueError, match="outside|path|location"):
        validate_recording_clock(tmp_path / "session", recording,
                                 sample_rate_hz=250, duration_seconds=2)


def test_relocated_sidecar_location_hints_are_never_used_for_io(tmp_path):
    recording = write_recording_clock(tmp_path / "session")
    path = tmp_path / "session" / recording["clock"]["metadata_path"]
    metadata = read_json(path)
    metadata.update(metadata_path="Z:/old-workstation/not-present.clock.json",
                    journal_path="Z:/old-workstation/not-present.clock.jsonl")
    write_json(path, metadata)
    result = validate_recording_clock(tmp_path / "session", recording,
                                     sample_rate_hz=250, duration_seconds=2)
    assert result["acquired_samples"] == 251


def test_location_normalization_does_not_hide_semantic_metadata_changes(tmp_path):
    recording = write_recording_clock(tmp_path)
    path = tmp_path / recording["clock"]["metadata_path"]
    metadata = read_json(path)
    metadata["acquired_samples"] += 1
    write_json(path, metadata)
    with pytest.raises(ValueError, match="manifest and finalized sidecar disagree"):
        validate_recording_clock(tmp_path, recording, sample_rate_hz=250, duration_seconds=2)


@pytest.mark.parametrize("duration", [1.004, 1.996, 2.004])
def test_serialized_duration_must_match_exactly_without_one_sample_tolerance(tmp_path, duration):
    recording = write_recording_clock(tmp_path)
    with pytest.raises(ValueError, match="duration"):
        validate_recording_clock(tmp_path, recording, sample_rate_hz=250, duration_seconds=duration)


@pytest.mark.parametrize("change", [
    {"acquired_samples": 252}, {"padding_samples": 250}, {"edf_samples": 250},
    {"acquired_samples": True}, {"padding_samples": -1},
])
def test_acquired_and_serialized_counts_cannot_be_conflated(tmp_path, change):
    recording = write_recording_clock(tmp_path)
    path = tmp_path / recording["clock"]["metadata_path"]
    metadata = read_json(path)
    metadata.update(change)
    recording["clock"].update(change)
    write_json(path, metadata)
    with pytest.raises(ValueError, match="samples|duration|count"):
        validate_recording_clock(tmp_path, recording, sample_rate_hz=250, duration_seconds=2)


@pytest.fixture(scope="module")
def real_frontend_exports(tmp_path_factory, frontend_runtime):
    frontend, executable = frontend_runtime
    output = tmp_path_factory.mktemp("real_clock_exports")
    script = textwrap.dedent("""
        import json
        import shutil
        import sys
        from pathlib import Path
        from gtn_workstation.app import configure_runtime_paths
        configure_runtime_paths()
        import numpy as np
        import pyedflib
        from gtn_workstation.bids import (
            inspect_eeg_recording, make_bids_session_layout, write_bids_eeg_sidecars,
        )
        from gtn_workstation.models import ChannelMontage, ExperimentConfig
        from gtn_workstation.runtime import LocalEdfRecorder
        for count in (250, 251, 500, 501):
            root = Path(sys.argv[1]) / str(count)
            montage = ChannelMontage()
            recorder = LocalEdfRecorder(root / 'sourcedata', montage.eeg, 250,
                                        'P001', 'X', '', 'clock fixture')
            recorder.append(np.zeros((count, 8)), packet_metadata=[
                {'seq_num': i, 'delta_time': 400} for i in range(count)
            ], receive_monotonic_ns=10_000_000_000)
            source = Path(recorder.finalize())
            metadata = json.loads(recorder.clock_metadata_path.read_text(encoding='utf-8'))
            clock = {**metadata,
                     'metadata_path': recorder.clock_metadata_path.relative_to(root).as_posix(),
                     'journal_path': recorder.clock_journal_path.relative_to(root).as_posix()}
            layout = make_bids_session_layout(root, 'P001', str(count))
            layout.eeg_dir.mkdir(parents=True)
            raw_path = layout.raw_path('.edf')
            shutil.copyfile(source, raw_path)
            info = inspect_eeg_recording(raw_path)
            write_bids_eeg_sidecars(layout, info, montage, ExperimentConfig(),
                                    sdk_version='fixture', impedance_values=[1.0] * 8)
            bids = json.loads(layout.sidecar_path('eeg').read_text(encoding='utf-8'))
            reader = pyedflib.EdfReader(str(raw_path))
            try:
                actual_samples = reader.getNSamples().tolist()
            finally:
                reader.close()
            record = {'clock': clock, 'actual_edf_samples': actual_samples,
                      'duration_seconds': bids['RecordingDuration']}
            (root / 'recording.json').write_text(json.dumps(record), encoding='utf-8')
    """)
    subprocess.run([str(executable), "-c", script, str(output)], cwd=frontend,
                   capture_output=True, text=True, check=True, timeout=60)
    return output


@pytest.mark.parametrize("acquired,serialized", [(250, 250), (251, 500), (500, 500), (501, 750)])
def test_real_recorder_edf_bids_and_backend_clock_agree(real_frontend_exports, acquired, serialized):
    root = real_frontend_exports / str(acquired)
    recording = read_json(root / "recording.json")
    assert recording["actual_edf_samples"] == [serialized] * 8
    result = validate_recording_clock(root, recording, sample_rate_hz=250,
                                     duration_seconds=recording["duration_seconds"])
    assert result["acquired_samples"] == acquired
    assert result["edf_samples"] == serialized
    assert result["padding_samples"] == serialized - acquired
