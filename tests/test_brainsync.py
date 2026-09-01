from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path

import mne
import numpy as np
import pytest
import torch

from data.brainsync import (
    BRAIN_SYNC_PREPROCESSING,
    load_brainsync_session,
    load_brainsync_sessions,
)
from data.brainsync_contract import (
    BRAIN_SYNC_ANALYSIS_TIME_BASE,
    BRAIN_SYNC_SESSION_SCHEMA,
    DecisionTargetPolicy,
    PopulationScopePolicy,
    derive_brainsync_evidence_scope,
    derive_population_scope,
)
from data.contract import (
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    SOURCE_COHORT_DATA_CONTRACTS,
    assert_causal_p300_input_contract,
)
from data.epochs import EpochDataset, save_epoch_dataset
from data.identity import DatasetIdentityTable
from experiments.run_brainsync_cross_decision import main as run_brainsync_cross_decision
from models.n2p3net import N2P3Net
from research.contracts import TrainingRunContract
from transfer.checkpoint import CHECKPOINT_SCHEMA
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


def _write_v2_session(
    root: Path,
    *,
    session_id: str,
    target: int,
    started_utc: str,
    subject_id: str = "P001",
    age: str = "19",
    blocks: tuple[tuple[int, ...], ...] = (tuple(range(1, 10)),),
    source_rate_hz: float = 128.0,
) -> Path:
    """Write the current workstation v2 shape: target only in session.json."""

    raw_dir = root / "raw"
    events_dir = root / "events"
    raw_dir.mkdir(parents=True)
    events_dir.mkdir()
    n_trials = sum(len(block) for block in blocks)
    duration_seconds = 2.0 + 1.5 * n_trials
    n_samples = int(round(duration_seconds * source_rate_hz))
    info = mne.create_info(list(CHANNELS), source_rate_hz, ch_types="eeg")
    values = np.random.default_rng(sum(map(ord, session_id))).normal(
        0.0, 5e-6, (len(CHANNELS), n_samples)
    )
    raw_path = raw_dir / "recording_trimmed_raw.fif"
    mne.io.RawArray(values, info, verbose=False).save(
        raw_path, overwrite=True, verbose=False
    )
    started = datetime.fromisoformat(started_utc.replace("Z", "+00:00"))
    confirmed_utc = (
        started + timedelta(seconds=duration_seconds + 1.0)
    ).isoformat()
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
            "thought_digit": target,
            "target_label_status": "confirmed_post_experiment",
        },
        "target_label": {
            "status": "confirmed",
            "source": "post_experiment_confirmation",
            "thought_digit": target,
            "confirmed_utc": confirmed_utc,
        },
        "channels": list(CHANNELS),
        "recording": {
            "path": "raw/recording_trimmed_raw.fif",
            "source_sample_rate_hz": source_rate_hz,
            "analysis_ready": True,
            "timeline": {
                "status": "finalized",
                "time_base": BRAIN_SYNC_ANALYSIS_TIME_BASE,
                "source_path": "raw/recording_source_raw.fif",
                "output_path": "raw/recording_trimmed_raw.fif",
                "source_sample_rate_hz": source_rate_hz,
                "output_duration_seconds": n_samples / source_rate_hz,
            },
        },
        "quality": {"eeg_continuity": {"passed": True}},
        "montage": {
            "schema": "brainsync-channel-montage/2",
            "labels": list(CHANNELS),
            "active_mask": 0xFF,
            "channel_positions_m": POSITIONS,
            "coordinate_frame": "head",
            "units": "m",
            "ref_label": "A2",
            "gnd_label": "GND",
        },
    }
    (root / "session.json").write_text(json.dumps(session), encoding="utf-8")

    records: list[dict[str, object]] = []
    onset = 1.0
    for block_id, digits in enumerate(blocks, start=1):
        for trial_index, digit in enumerate(digits, start=1):
            common = {
                "trial_id": f"B{block_id:03d}-T{trial_index:03d}",
                "block_id": block_id,
                "trial_index": trial_index,
                "digit": digit,
            }
            records.append(
                {
                    "event": "recording_marker",
                    "payload": {
                        "kind": "onset",
                        **common,
                        "eeg_time_seconds": onset,
                        "eeg_time_base": BRAIN_SYNC_ANALYSIS_TIME_BASE,
                        "annotation": (
                            f"STIM_ONSET|trial={common['trial_id']}|digit={digit}"
                        ),
                    },
                }
            )
            records.append(
                {
                    "event": "recording_marker",
                    "payload": {
                        "kind": "offset",
                        **common,
                        "eeg_time_seconds": onset + 0.3,
                        "eeg_time_base": BRAIN_SYNC_ANALYSIS_TIME_BASE,
                        "annotation": (
                            f"STIM_OFFSET|trial={common['trial_id']}|digit={digit}"
                        ),
                    },
                }
            )
            onset += 1.5
    (events_dir / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return root


def _write_source_freeze_manifest(root: Path) -> tuple[Path, str]:
    archive = root / "source_snapshot.tar.gz"
    member = root / "snapshot.txt"
    member.write_text("frozen source fixture\n", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(member, arcname="snapshot.txt")
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "schema": "n2p3_source_freeze/1",
        "archive": archive.name,
        "archive_sha256": archive_sha256,
        "source_commit": "a" * 40,
        "member_count": 1,
        "byte_size": archive.stat().st_size,
        "scope": "synthetic source-freeze fixture",
        "contains_raw_eeg_cache_or_model_checkpoint": False,
        "replacement_policy": "test fixture only",
    }
    manifest_path = root / "source_snapshot.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, archive_sha256


def _write_current_checkpoint(
    path: Path,
    dataset: EpochDataset,
    *,
    source_snapshot_sha256: str,
    mismatch_architecture: bool = False,
) -> Path:
    trunk = N2P3Net(
        dataset.n_channels,
        n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq,
        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
        pooling_mode="full_unfold",
        temporal_kernel_size=35,
    )
    training_identity_ledger = DatasetIdentityTable.from_source_rows(
        ["other"], ["external-source"]
    )
    architecture = trunk.architecture_record()
    training_contract = TrainingRunContract(
        source_cache_sha256="a" * 64,
        source_identity_digest=training_identity_ledger.digest(),
        source_snapshot_sha256=source_snapshot_sha256,
        architecture=architecture,
        preprocessing={
            "epoch": asdict(dataset.preprocessing),
            "channel_names": list(dataset.channel_names),
            "source_reference": dataset.provenance["source_reference"],
        },
        optimizer={"kind": "synthetic_test_fixture"},
        validation={"kind": "synthetic_test_fixture"},
        objective={"kind": "supervised_binary_cross_entropy"},
        seed=0,
        training_participant_keys=training_identity_ledger.authority_keys("source"),
        holdout_participant_keys=(),
    )
    payload_architecture = dict(architecture)
    if mismatch_architecture:
        payload_architecture["st_temporal_filters"] = int(
            payload_architecture["st_temporal_filters"]
        ) + 1
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "trunk_state_dict": trunk.state_dict(),
            "training_identity_ledger": training_identity_ledger.payload(),
            "training_identity_ledger_digest": training_identity_ledger.digest(),
            "training_contract": training_contract.record(),
            "training_contract_digest": training_contract.digest(),
            "source_cache_sha256": training_contract.source_cache_sha256,
            "source_dataset_name": "external-source",
            "input_channel_names": list(dataset.channel_names),
            "input_preprocessing": asdict(dataset.preprocessing),
            "input_source_reference": dataset.provenance["source_reference"],
            "classifier_trained": True,
            "input_mean": [0.0] * dataset.n_channels,
            "input_std": [1.0] * dataset.n_channels,
            "source_calibration": {
                "pos_weight": 8.0,
                "train_prior": 1.0 / 9.0,
                "temperature": 1.0,
                "source": "synthetic_test_fixture",
            },
            "architecture": payload_architecture,
            "n_channels": dataset.n_channels,
            "n_times": dataset.n_times,
            "input_sample_rate_hz": dataset.preprocessing.sfreq,
            "input_tmin_s": dataset.preprocessing.tmin_ms / 1000.0,
        },
        path,
    )
    return path


def _read_manifest(session_dir: Path) -> dict[str, object]:
    return json.loads((session_dir / "session.json").read_text(encoding="utf-8"))


def _write_manifest(session_dir: Path, payload: dict[str, object]) -> None:
    (session_dir / "session.json").write_text(json.dumps(payload), encoding="utf-8")


def _set_nested(payload: dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    node = payload
    for part in parts[:-1]:
        child = node[part]
        assert isinstance(child, dict)
        node = child
    node[parts[-1]] = value


def test_brainsync_uses_the_canonical_causal_profile() -> None:
    assert BRAIN_SYNC_PREPROCESSING.name == SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.name
    assert BRAIN_SYNC_PREPROCESSING.l_freq == 0.1
    assert BRAIN_SYNC_PREPROCESSING.h_freq == 30.0
    assert BRAIN_SYNC_PREPROCESSING.tmax_ms == 1200.0
    assert BRAIN_SYNC_PREPROCESSING.filter_phase == "forward"
    assert BRAIN_SYNC_PREPROCESSING.causal_iir_initial_state == "steady_state_first_sample"
    assert BRAIN_SYNC_PREPROCESSING.n_times == SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times
    assert SOURCE_COHORT_DATA_CONTRACTS["causal"] is SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT


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


def test_v2_session_is_one_decision_and_blocks_only_preserve_schedule_metadata(
    tmp_path: Path,
) -> None:
    session_dir = _write_v2_session(
        tmp_path / "session",
        session_id="P001_s1",
        target=2,
        started_utc="2026-09-01T00:00:00+00:00",
        blocks=(tuple(range(1, 10)), tuple(range(1, 10))),
    )

    dataset = load_brainsync_session(session_dir)

    assert dataset.X.shape == (18, 8, BRAIN_SYNC_PREPROCESSING.n_times)
    assert dataset.X.dtype == np.float32
    assert len(np.unique(dataset.event_timeline.group_ids)) == 1
    assert set(dataset.event_timeline.selection_ids) == {"P001_s1"}
    assert set(dataset.event_timeline.target_candidate_ids) == {"2"}
    assert dataset.metadata.groupby("block_id").size().to_dict() == {1: 9, 2: 9}
    assert dataset.metadata["repetition_index"].tolist() == [0] * 9 + [1] * 9
    assert set(dataset.metadata["age_years"]) == {19.0}
    assert dataset.provenance["decision_unit"] == "session"
    assert dataset.provenance["event_time_base"] == BRAIN_SYNC_ANALYSIS_TIME_BASE
    assert dataset.event_timeline.supports_full_candidate_chain is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "brainsync-gtn-session/20", "Unsupported BrainSync session schema"),
        ("status", "running", "status is not analysis-ready"),
        ("target_label.status", "pending", "post-experiment confirmed"),
        ("target_label.source", "inline_label", "post_experiment_confirmation"),
        ("target_label.confirmed_utc", None, "must be a non-empty string"),
        (
            "experiment.target_label_status",
            "pending_post_experiment_confirmation",
            "confirmed_post_experiment",
        ),
        ("ended_utc", None, "must be a non-empty string"),
        ("recording.analysis_ready", False, "analysis-ready gate"),
        ("recording.timeline.status", "continuous_recording", "must be finalized"),
        ("recording.timeline.time_base", "continuous_recording", "rest-removed"),
        ("recording.timeline.output_path", None, "must be a non-empty string"),
        (
            "recording.source_sample_rate_hz",
            None,
            "must be a finite non-negative number",
        ),
        (
            "recording.timeline.source_sample_rate_hz",
            None,
            "must be a finite non-negative number",
        ),
        (
            "recording.timeline.output_duration_seconds",
            None,
            "must be a finite non-negative number",
        ),
        ("quality.eeg_continuity.passed", False, "continuity quality must pass"),
    ],
)
def test_analysis_ready_gate_fails_before_raw_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    session_dir = _write_v2_session(
        tmp_path / field.replace(".", "_"),
        session_id="P001_gate",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    manifest = _read_manifest(session_dir)
    _set_nested(manifest, field, value)
    _write_manifest(session_dir, manifest)
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *_args, **_kwargs: pytest.fail("raw reader ran before v2 gate"),
    )

    with pytest.raises(ValueError, match=message):
        load_brainsync_session(session_dir)


def test_analysis_ready_gate_binds_recording_to_finalized_output_before_raw_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = _write_v2_session(
        tmp_path / "wrong_output",
        session_id="P001_wrong_output",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    manifest = _read_manifest(session_dir)
    _set_nested(
        manifest,
        "recording.timeline.output_path",
        "events/events.jsonl",
    )
    _write_manifest(session_dir, manifest)
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *_args, **_kwargs: pytest.fail("raw reader ran before output-path gate"),
    )

    with pytest.raises(ValueError, match="must identify the finalized"):
        load_brainsync_session(session_dir)


def test_analysis_ready_gate_requires_ordered_lifecycle_timestamps_before_raw_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = _write_v2_session(
        tmp_path / "bad_timestamps",
        session_id="P001_bad_timestamps",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    manifest = _read_manifest(session_dir)
    manifest["ended_utc"] = manifest["started_utc"]
    _write_manifest(session_dir, manifest)
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *_args, **_kwargs: pytest.fail("raw reader ran before timestamp gate"),
    )

    with pytest.raises(ValueError, match="started_utc <= target confirmation <= ended_utc"):
        load_brainsync_session(session_dir)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("montage.schema", None, "montage schema"),
        ("montage.coordinate_frame", None, "head frame and metres"),
        ("montage.units", None, "head frame and metres"),
        ("montage.labels", None, "must provide montage.labels"),
        ("montage.active_mask", None, "active_mask must be an integer"),
        ("montage.channel_positions_m", None, "must align with montage.labels"),
        ("montage.gnd_label", None, "REF/GND labels"),
        ("channels", list(reversed(CHANNELS)), "must exactly match montage.labels"),
    ],
)
def test_v2_montage_fields_have_no_fallback_before_raw_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    session_dir = _write_v2_session(
        tmp_path / field.replace(".", "_"),
        session_id="P001_montage_gate",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    manifest = _read_manifest(session_dir)
    _set_nested(manifest, field, value)
    _write_manifest(session_dir, manifest)
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *_args, **_kwargs: pytest.fail("raw reader ran before montage gate"),
    )

    with pytest.raises(ValueError, match=message):
        load_brainsync_session(session_dir)


def test_v2_montage_rejects_wrong_channel_count_and_invalid_positions_before_raw_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong_count = _write_v2_session(
        tmp_path / "wrong_channel_count",
        session_id="P001_wrong_channel_count",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    manifest = _read_manifest(wrong_count)
    manifest["channels"] = list(CHANNELS[:-1])
    montage = manifest["montage"]
    assert isinstance(montage, dict)
    montage["labels"] = list(CHANNELS[:-1])
    montage["channel_positions_m"] = POSITIONS[:-1]
    montage["active_mask"] = 0x7F
    _write_manifest(wrong_count, manifest)
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *_args, **_kwargs: pytest.fail("raw reader ran before montage count gate"),
    )
    with pytest.raises(ValueError, match="must declare 8 EEG channels"):
        load_brainsync_session(wrong_count)

    invalid_position = _write_v2_session(
        tmp_path / "invalid_position",
        session_id="P001_invalid_position",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    manifest = _read_manifest(invalid_position)
    montage = manifest["montage"]
    assert isinstance(montage, dict)
    positions = montage["channel_positions_m"]
    assert isinstance(positions, list)
    positions[0] = ["not-a-number", 0.0, 0.1]
    _write_manifest(invalid_position, manifest)
    with pytest.raises(ValueError, match="finite numbers"):
        load_brainsync_session(invalid_position)


def test_v2_markers_must_fit_declared_output_duration_before_raw_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = _write_v2_session(
        tmp_path / "short_duration",
        session_id="P001_short_duration",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    manifest = _read_manifest(session_dir)
    _set_nested(manifest, "recording.timeline.output_duration_seconds", 1.1)
    _write_manifest(session_dir, manifest)
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *_args, **_kwargs: pytest.fail("raw reader ran before duration gate"),
    )

    with pytest.raises(ValueError, match="onset marker exceeds"):
        load_brainsync_session(session_dir)


@pytest.mark.parametrize("unsafe_path", ["../outside.fif", "D:/outside.fif"])
def test_analysis_ready_gate_rejects_noncontained_recording_paths_before_raw_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_path: str,
) -> None:
    session_dir = _write_v2_session(
        tmp_path / "unsafe",
        session_id="P001_unsafe",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    manifest = _read_manifest(session_dir)
    _set_nested(manifest, "recording.path", unsafe_path)
    _write_manifest(session_dir, manifest)
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *_args, **_kwargs: pytest.fail("raw reader ran before path gate"),
    )

    with pytest.raises(ValueError, match="relative|parent traversal"):
        load_brainsync_session(session_dir)


def test_v2_marker_rejects_label_leakage_before_raw_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = _write_v2_session(
        tmp_path / "leak",
        session_id="P001_leak",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    events_path = session_dir / "events" / "events.jsonl"
    records = [json.loads(line) for line in events_path.read_text().splitlines()]
    records[0]["payload"]["target_digit"] = 1
    events_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *_args, **_kwargs: pytest.fail("raw reader ran before marker gate"),
    )

    with pytest.raises(ValueError, match="cannot contain derived decision/label fields"):
        load_brainsync_session(session_dir)


def test_v2_marker_requires_the_finalized_time_base_before_raw_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = _write_v2_session(
        tmp_path / "marker_timebase",
        session_id="P001_marker_timebase",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    events_path = session_dir / "events" / "events.jsonl"
    records = [json.loads(line) for line in events_path.read_text().splitlines()]
    records[0]["payload"]["eeg_time_base"] = "continuous_recording"
    events_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *_args, **_kwargs: pytest.fail("raw reader ran before marker time-base gate"),
    )

    with pytest.raises(ValueError, match="must use eeg_time_base"):
        load_brainsync_session(session_dir)


def test_v2_marker_rejects_duplicate_trial_identity_before_raw_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = _write_v2_session(
        tmp_path / "duplicate",
        session_id="P001_duplicate",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    events_path = session_dir / "events" / "events.jsonl"
    records = [json.loads(line) for line in events_path.read_text().splitlines()]
    records.insert(1, records[0])
    events_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *_args, **_kwargs: pytest.fail("raw reader ran before marker gate"),
    )

    with pytest.raises(ValueError, match="Duplicate BrainSync onset trial_id"):
        load_brainsync_session(session_dir)


def test_v2_marker_rejects_distinct_onsets_mapping_to_one_source_sample(
    tmp_path: Path,
) -> None:
    session_dir = _write_v2_session(
        tmp_path / "duplicate_sample",
        session_id="P001_duplicate_sample",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    events_path = session_dir / "events" / "events.jsonl"
    onset_records = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if json.loads(line)["payload"]["kind"] == "onset"
    ]
    onset_records[1]["payload"]["eeg_time_seconds"] = 1.001
    events_path.write_text(
        "".join(json.dumps(record) + "\n" for record in onset_records), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate source EEG samples"):
        load_brainsync_session(session_dir)


def test_multisession_loader_preserves_one_decision_per_session(tmp_path: Path) -> None:
    first = _write_v2_session(
        tmp_path / "session_a",
        session_id="P001_a",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    second = _write_v2_session(
        tmp_path / "session_b",
        session_id="P001_b",
        target=2,
        started_utc="2026-09-02T00:00:00+00:00",
    )

    dataset = load_brainsync_sessions([second, first])

    assert dataset.name == "BrainSync-GTN-multisession"
    assert dataset.provenance["n_sessions"] == 2
    assert dataset.provenance["session_ids"] == ["P001_a", "P001_b"]
    assert set(dataset.event_timeline.session_ids) == {"P001_a", "P001_b"}
    assert len(np.unique(dataset.event_timeline.group_ids)) == 2
    assert dataset.provenance["population_scope"]["label"] == "age_descriptive"


def test_multisession_loader_rejects_duplicate_session_input(tmp_path: Path) -> None:
    session = _write_v2_session(
        tmp_path / "session",
        session_id="P001_duplicate_input",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="unique BrainSync session directories"):
        load_brainsync_sessions([session, session])


def test_forced_switch_policy_retains_violation_without_replacement(tmp_path: Path) -> None:
    sessions = [
        _write_v2_session(
            tmp_path / f"session_{index}",
            session_id=f"P001_s{index}",
            target=target,
            started_utc=f"2026-09-0{index}T00:00:00+00:00",
        )
        for index, target in enumerate((1, 1, 2, 3), start=1)
    ]
    dataset = load_brainsync_sessions(sessions)

    split = calibration_decision_split(
        dataset,
        calibration_selections=1,
        test_repetitions=1,
        target_policy=DecisionTargetPolicy.FORCED_SWITCH,
        max_test_selections=2,
        candidate_vocabulary=range(1, 10),
    )

    requested = split.requested_test_groups_by_subject["P001"]
    eligible = split.test_groups_by_subject["P001"]
    failures = split.failed_test_groups_by_subject["P001"]
    assert len(requested) == 2
    assert len(eligible) == 1
    assert failures == {requested[0]: "target_policy_same_as_previous_decision"}
    assert all("P001_s4" not in group for group in requested)


def test_forced_switch_returns_complete_failure_ledger_when_none_are_eligible(
    tmp_path: Path,
) -> None:
    sessions = [
        _write_v2_session(
            tmp_path / f"all_failed_{index}",
            session_id=f"P001_all_failed_{index}",
            target=1,
            started_utc=f"2026-09-0{index}T00:00:00+00:00",
        )
        for index in (1, 2)
    ]
    dataset = load_brainsync_sessions(sessions)

    split = calibration_decision_split(
        dataset,
        calibration_selections=1,
        test_repetitions=1,
        target_policy=DecisionTargetPolicy.FORCED_SWITCH,
        candidate_vocabulary=range(1, 10),
    )

    requested = split.requested_test_groups_by_subject["P001"]
    assert split.usable_subjects == ()
    assert split.test_groups_by_subject.get("P001", ()) == ()
    assert split.failed_test_groups_by_subject["P001"] == {
        requested[0]: "target_policy_same_as_previous_decision"
    }
    assert split.excluded_subjects["P001"] == "no_eligible_unknown_test_decision"


def test_split_rejects_legacy_cache_that_promotes_blocks_to_decisions(
    tmp_path: Path,
) -> None:
    session = _write_v2_session(
        tmp_path / "legacy_blocks",
        session_id="P001_legacy_blocks",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
        blocks=(tuple(range(1, 10)), tuple(range(1, 10))),
    )
    dataset = load_brainsync_session(session)
    legacy_groups = np.asarray(
        ["legacy-block-1"] * 9 + ["legacy-block-2"] * 9,
        dtype=str,
    )
    dataset.event_timeline = replace(
        dataset.event_timeline,
        group_ids=legacy_groups,
        selection_ids=legacy_groups,
        repetition_indices=np.zeros(18, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="one decision per session"):
        calibration_decision_split(
            dataset,
            calibration_selections=1,
            test_repetitions=1,
            target_policy=DecisionTargetPolicy.OBSERVED_SEQUENCE,
            candidate_vocabulary=range(1, 10),
        )


def test_observed_sequence_policy_does_not_claim_target_switch(tmp_path: Path) -> None:
    sessions = [
        _write_v2_session(
            tmp_path / f"observed_{index}",
            session_id=f"P001_observed_{index}",
            target=1,
            started_utc=f"2026-09-0{index}T00:00:00+00:00",
        )
        for index in (1, 2)
    ]
    dataset = load_brainsync_sessions(sessions)

    split = calibration_decision_split(
        dataset,
        calibration_selections=1,
        test_repetitions=1,
        target_policy=DecisionTargetPolicy.OBSERVED_SEQUENCE,
        candidate_vocabulary=range(1, 10),
    )
    population = derive_population_scope(
        ("P001", "P001"),
        (19.0, 19.0),
        policy=PopulationScopePolicy.DESCRIPTIVE,
    )
    scope = derive_brainsync_evidence_scope(
        population,
        target_policy=split.target_policy,
    )

    assert len(split.test_groups_by_subject["P001"]) == 1
    assert scope.decision_claim == "later_session_observed_target_sequence"
    assert scope.population.label == "age_descriptive"


def test_unseen_calibration_policy_uses_all_calibration_targets(tmp_path: Path) -> None:
    sessions = [
        _write_v2_session(
            tmp_path / f"unseen_{index}",
            session_id=f"P001_unseen_{index}",
            target=target,
            started_utc=f"2026-09-0{index}T00:00:00+00:00",
        )
        for index, target in enumerate((1, 2, 1, 3), start=1)
    ]
    dataset = load_brainsync_sessions(sessions)

    split = calibration_decision_split(
        dataset,
        calibration_selections=2,
        test_repetitions=1,
        target_policy=DecisionTargetPolicy.UNSEEN_CALIBRATION_CODES,
        candidate_vocabulary=range(1, 10),
    )

    requested = split.requested_test_groups_by_subject["P001"]
    assert split.failed_test_groups_by_subject["P001"] == {
        requested[0]: "target_policy_seen_in_calibration"
    }
    assert split.test_groups_by_subject["P001"] == (requested[1],)


def test_adult_scope_requires_explicit_policy_and_complete_age_evidence() -> None:
    descriptive = derive_population_scope(
        ("P001", "P001"),
        (19.0, 19.0),
        policy=PopulationScopePolicy.DESCRIPTIVE,
    )
    adult = derive_population_scope(
        ("P001", "P001"),
        (19.0, 20.0),
        policy=PopulationScopePolicy.ADULT_ONLY,
    )

    assert descriptive.label == "age_descriptive"
    assert adult.label == "adult"
    with pytest.raises(ValueError, match="requires age for every"):
        derive_population_scope(
            ("P001", "P001"),
            (19.0, None),
            policy=PopulationScopePolicy.ADULT_ONLY,
        )
    with pytest.raises(ValueError, match="below adult_min_age_years"):
        derive_population_scope(
            ("P001",),
            (17.0,),
            policy=PopulationScopePolicy.ADULT_ONLY,
        )


def test_brainsync_runner_derives_scope_and_keeps_policy_failure_in_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = [
        _write_v2_session(
            tmp_path / f"runner_{index}",
            session_id=f"P001_runner_{index}",
            target=target,
            started_utc=f"2026-09-0{index}T00:00:00+00:00",
        )
        for index, target in enumerate((1, 1, 2), start=1)
    ]
    dataset = load_brainsync_sessions(sessions)
    cache = save_epoch_dataset(tmp_path / "brainsync.npz", dataset)
    source_snapshot_manifest, source_snapshot_sha256 = _write_source_freeze_manifest(
        tmp_path
    )
    checkpoint = _write_current_checkpoint(
        tmp_path / "source.pt",
        dataset,
        source_snapshot_sha256=source_snapshot_sha256,
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
            "--arm-name",
            "zero_shot_source",
            "--source-snapshot-manifest",
            str(source_snapshot_manifest),
            "--identity-exclusion-policy",
            "source",
            "--calibration-selections",
            "1",
            "--test-reps",
            "1",
            "--max-test-selections",
            "2",
            "--target-policy",
            "forced_switch",
            "--population-policy",
            "adult_only",
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
    assert result["schema"] == "n2p3_brainsync_cross_decision_result/2"
    assert result["decision_accounting"]["requested"] == 2
    assert result["decision_accounting"]["eligible"] == 1
    assert result["decision_accounting"]["pre_evaluation_failed"] == 1
    assert result["decision_accounting"]["by_evidence_level"]["1"]["incomplete"] == 1
    assert result["evaluation_contract_digest"]
    assert {row["participant_key"] for row in result["decision_outcomes"]} == set(
        result["requested_participant_keys"]
    )
    assert result["target_policy"] == "forced_switch"
    assert result["evidence_scope"]["decision_unit"] == "session"
    assert result["evidence_scope"]["decision_claim"] == "later_session_target_switch"
    assert result["evidence_scope"]["population"]["label"] == "adult"
    assert result["source_snapshot_manifest"] == str(source_snapshot_manifest.resolve())


def test_brainsync_runner_validates_bad_checkpoint_when_policy_leaves_no_usable_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = [
        _write_v2_session(
            tmp_path / f"empty_runner_{index}",
            session_id=f"P001_empty_runner_{index}",
            target=1,
            started_utc=f"2026-09-0{index}T00:00:00+00:00",
        )
        for index in (1, 2)
    ]
    dataset = load_brainsync_sessions(sessions)
    cache = save_epoch_dataset(tmp_path / "empty_brainsync.npz", dataset)
    source_snapshot_manifest, source_snapshot_sha256 = _write_source_freeze_manifest(
        tmp_path
    )
    mismatched_snapshot = "b" * 64
    assert mismatched_snapshot != source_snapshot_sha256
    checkpoint = _write_current_checkpoint(
        tmp_path / "wrong_snapshot.pt",
        dataset,
        source_snapshot_sha256=mismatched_snapshot,
    )
    output = tmp_path / "must_not_exist.json"
    argv = [
        "run_brainsync_cross_decision.py",
        "--dataset-cache",
        str(cache),
        "--checkpoint",
        str(checkpoint),
        "--arm-name",
        "zero_shot_source",
        "--source-snapshot-manifest",
        str(source_snapshot_manifest),
        "--identity-exclusion-policy",
        "source",
        "--calibration-selections",
        "1",
        "--test-reps",
        "1",
        "--max-test-selections",
        "1",
        "--target-policy",
        "forced_switch",
        "--population-policy",
        "adult_only",
        "--head",
        "zero_shot",
        "--device",
        "cpu",
        "--output",
        str(output),
    ]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(ValueError, match="source snapshot disagrees"):
        run_brainsync_cross_decision()
    assert not output.exists()

    bad_architecture_checkpoint = _write_current_checkpoint(
        tmp_path / "bad_architecture.pt",
        dataset,
        source_snapshot_sha256=source_snapshot_sha256,
        mismatch_architecture=True,
    )
    argv[4] = str(bad_architecture_checkpoint)
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(ValueError, match="architecture disagrees"):
        run_brainsync_cross_decision()
    assert not output.exists()
