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

from data.bids_eeg import BidsRestInterval, epoch_rest_overlap_mask
from data.brainsync import (
    BRAIN_SYNC_PREPROCESSING,
    InvalidSessionPolicy,
    load_brainsync_session,
    load_brainsync_sessions,
    load_brainsync_sessions_resilient,
)
from data.brainsync_contract import (
    BRAIN_SYNC_RAW_TIME_BASE,
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
from data.epochs import EpochDataset, load_epoch_dataset, save_epoch_dataset
from data.identity import DatasetIdentityTable
from experiments.prepare_brainsync_cache import main as prepare_brainsync_cache
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


def _write_bids_session(
    root: Path,
    *,
    session_id: str,
    target: int,
    started_utc: str,
    subject_id: str = "P001",
    age: str = "19",
    blocks: tuple[tuple[int, ...], ...] = (tuple(range(1, 10)),),
    source_rate_hz: float = 250.0,
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
            event_rows.append([
                onset, 0.3, round(onset * source_rate_hz), "stimulus", digit,
                "stimulus", trial_id, block_id, trial_index, digit, "n/a",
            ])
            onset += 1.5
        if block_id < len(blocks):
            rest_start = onset
            rest_end = rest_start + rest_duration_seconds
            event_rows.append([
                rest_start, rest_duration_seconds, round(rest_start * source_rate_hz),
                "rest", "rest", "rest", "n/a", block_id, "n/a", "n/a",
                f"block-{block_id}",
            ])
            rest_segments.append({
                "start_seconds": rest_start,
                "end_seconds": rest_end,
                "duration_seconds": rest_duration_seconds,
            })
            onset = rest_end + 0.5
    duration_seconds = onset + 1.5
    n_samples = int(round(duration_seconds * source_rate_hz))
    info = mne.create_info(list(CHANNELS), source_rate_hz, ch_types="eeg")
    values = np.random.default_rng(sum(map(ord, session_id))).normal(
        0.0, 5e-6, (len(CHANNELS), n_samples)
    )
    stem = f"sub-{subject_label}_ses-{session_label}_task-gtn_run-01"
    raw_path = eeg_dir / f"{stem}_eeg.edf"
    mne.io.RawArray(values, info, verbose=False).export(
        raw_path, fmt="edf", overwrite=True, physical_range="auto", verbose=False
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
                "electrodes_tsv": (eeg_dir / f"sub-{subject_label}_ses-{session_label}_space-BrainSyncHead_electrodes.tsv").relative_to(root).as_posix(),
                "coordsystem_json": (eeg_dir / f"sub-{subject_label}_ses-{session_label}_space-BrainSyncHead_coordsystem.json").relative_to(root).as_posix(),
            },
        },
        "quality": {"eeg_continuity": {"passed": True}},
    }
    (root / "session.json").write_text(json.dumps(session), encoding="utf-8")
    (root / "dataset_description.json").write_text(json.dumps({
        "Name": "BrainSync test",
        "BIDSVersion": "1.11.0",
        "DatasetType": "raw",
    }), encoding="utf-8")
    (eeg_dir / f"{stem}_eeg.json").write_text(json.dumps({
        "TaskName": "gtn",
        "SamplingFrequency": source_rate_hz,
        "RecordingDuration": n_samples / source_rate_hz,
        "EEGReference": "A2",
    }), encoding="utf-8")
    (eeg_dir / f"{stem}_channels.tsv").write_text(
        "name\ttype\tunits\tsampling_frequency\tstatus\n"
        + "".join(f"{label}\tEEG\tuV\t{source_rate_hz:g}\tgood\n" for label in CHANNELS),
        encoding="utf-8",
    )
    (eeg_dir / f"sub-{subject_label}_ses-{session_label}_space-BrainSyncHead_electrodes.tsv").write_text(
        "name\tx\ty\tz\n"
        + "".join(
            f"{label}\t{position[0]}\t{position[1]}\t{position[2]}\n"
            for label, position in zip(CHANNELS, POSITIONS, strict=True)
        ),
        encoding="utf-8",
    )
    (eeg_dir / f"sub-{subject_label}_ses-{session_label}_space-BrainSyncHead_coordsystem.json").write_text(json.dumps({
        "EEGCoordinateSystem": "Other",
        "EEGCoordinateUnits": "m",
        "EEGCoordinateSystemDescription": "Synthetic MNE head coordinates.",
    }), encoding="utf-8")
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


def test_rest_overlap_uses_half_open_epoch_and_rest_intervals() -> None:
    rest = BidsRestInterval(2.0, 1.0, 500, "block-1", 1, 2)

    overlaps = epoch_rest_overlap_mask(
        [0.8, 0.8001, 3.1999, 3.2],
        tmin_seconds=-0.2,
        tmax_seconds=1.2,
        rest_intervals=[rest],
    )

    assert overlaps == (False, True, True, False)


def test_v4_bids_session_is_one_decision_and_blocks_preserve_schedule_metadata(
    tmp_path: Path,
) -> None:
    session_dir = _write_bids_session(
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
    assert dataset.provenance["event_time_base"] == BRAIN_SYNC_RAW_TIME_BASE
    assert dataset.provenance["retained_rest_interval_count"] == 1
    assert dataset.event_timeline.supports_full_candidate_chain is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "brainsync-gtn-session/3", "Unsupported BrainSync session schema"),
        ("status", "running", "status is not a model input"),
        ("status", "aborted", "status is not a model input"),
        ("target_label.status", "pending", "post-experiment confirmed"),
        ("experiment.sequence_policy", "independent_with_replacement", "sequence_policy"),
        ("experiment.digits_per_block", 10, "complete candidate cycles"),
        ("experiment.repetitions_per_digit", 2, "conflicts with digits_per_block"),
        ("recording.stage", "rest_removed_recording", "stage must be bids_raw"),
        ("recording.preprocessing_status", "completed", "must be pending"),
        ("recording.timeline.time_base", "rest_removed_recording", "continuous_recording"),
        ("recording.timeline.rest_segments_retained", False, "retain rest"),
        ("quality.eeg_continuity.passed", False, "continuity quality"),
    ],
)
def test_v4_manifest_gate_fails_before_binary_eeg_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    session_dir = _write_bids_session(
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
        lambda *_args, **_kwargs: pytest.fail("EEG binary reader ran before metadata gate"),
    )

    with pytest.raises(ValueError, match=message):
        load_brainsync_session(session_dir)


def test_v4_rejects_unbalanced_actual_events_before_epoch_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = _write_bids_session(
        tmp_path / "unbalanced",
        session_id="P001_unbalanced",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    manifest = _read_manifest(session_dir)
    events_path = session_dir / manifest["recording"]["bids"]["events_tsv"]
    lines = events_path.read_text(encoding="utf-8").splitlines()
    fields = lines[-1].split("\t")
    fields[4] = "1"
    fields[9] = "1"
    lines[-1] = "\t".join(fields)
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *_args, **_kwargs: pytest.fail("epoch EEG reader ran before schedule gate"),
    )

    with pytest.raises(ValueError, match="does not balance every candidate"):
        load_brainsync_session(session_dir)


def test_v4_rejects_block_balanced_but_non_permutation_cycles(tmp_path: Path) -> None:
    session_dir = _write_bids_session(
        tmp_path / "bad_cycles",
        session_id="P001_bad_cycles",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
        blocks=(tuple(value for digit in range(1, 10) for value in (digit, digit)),),
    )

    with pytest.raises(ValueError, match="not a complete candidate permutation"):
        load_brainsync_session(session_dir)


def test_bids_sidecars_reject_path_escape_and_bad_coordinates_before_raw_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    escaped = _write_bids_session(
        tmp_path / "escaped",
        session_id="P001_escaped",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    manifest = _read_manifest(escaped)
    _set_nested(manifest, "recording.bids.events_tsv", "../outside.tsv")
    _write_manifest(escaped, manifest)
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *_args, **_kwargs: pytest.fail("EEG binary reader ran before BIDS path gate"),
    )
    with pytest.raises(ValueError, match="inside the dataset root"):
        load_brainsync_session(escaped)

    invalid = _write_bids_session(
        tmp_path / "invalid_coordinates",
        session_id="P001_invalid_coordinates",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    invalid_manifest = _read_manifest(invalid)
    bids = invalid_manifest["recording"]["bids"]
    electrodes_path = invalid / bids["electrodes_tsv"]
    lines = electrodes_path.read_text(encoding="utf-8").splitlines()
    cells = lines[1].split("\t")
    cells[1] = "not-a-number"
    lines[1] = "\t".join(cells)
    electrodes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite number"):
        load_brainsync_session(invalid)


def test_bids_event_sample_must_match_onset_before_raw_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = _write_bids_session(
        tmp_path / "sample_mismatch",
        session_id="P001_sample_mismatch",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    manifest = _read_manifest(session_dir)
    bids = manifest["recording"]["bids"]
    events_path = session_dir / bids["events_tsv"]
    lines = events_path.read_text(encoding="utf-8").splitlines()
    cells = lines[1].split("\t")
    cells[2] = str(int(cells[2]) + 2)
    lines[1] = "\t".join(cells)
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *_args, **_kwargs: pytest.fail("EEG binary reader ran before event gate"),
    )

    with pytest.raises(ValueError, match="more than half a sample"):
        load_brainsync_session(session_dir)


def test_manifest_rest_ledger_must_match_bids_events_before_raw_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = _write_bids_session(
        tmp_path / "rest_mismatch",
        session_id="P001_rest_mismatch",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
        blocks=(tuple(range(1, 10)), tuple(range(1, 10))),
    )
    manifest = _read_manifest(session_dir)
    timeline = manifest["recording"]["timeline"]
    timeline["rest_segments"][0]["end_seconds"] += 0.5
    timeline["rest_segments"][0]["duration_seconds"] += 0.5
    _write_manifest(session_dir, manifest)
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *_args, **_kwargs: pytest.fail("EEG binary reader ran before rest ledger gate"),
    )

    with pytest.raises(ValueError, match="conflict with BIDS events.tsv"):
        load_brainsync_session(session_dir)


def test_rest_overlap_is_excluded_after_continuous_filtering_without_time_shift(
    tmp_path: Path,
) -> None:
    session_dir = _write_bids_session(
        tmp_path / "rest_overlap",
        session_id="P001_rest_overlap",
        target=2,
        started_utc="2026-09-01T00:00:00+00:00",
        blocks=(tuple(range(1, 10)), tuple(range(1, 10))),
    )
    manifest = _read_manifest(session_dir)
    bids = manifest["recording"]["bids"]
    events_path = session_dir / bids["events_tsv"]
    lines = events_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:]]
    second_block = next(row for row in rows if row[header.index("trial_id")] == "B002-T001")
    second_block[header.index("onset")] = "16.6"
    second_block[header.index("sample")] = str(round(16.6 * 250))
    lines = ["\t".join(header), *["\t".join(row) for row in rows]]
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dataset = load_brainsync_session(session_dir)

    timeline = dataset.event_timeline
    event_index = 9
    assert timeline.onset_times_s[event_index] == pytest.approx(16.6)
    assert timeline.evidence_indices[event_index] == -1
    assert "BAD_brainsync_rest" in timeline.status_details[event_index]
    assert dataset.X.shape[0] == 17
    assert dataset.provenance["event_time_base"] == BRAIN_SYNC_RAW_TIME_BASE
    assert dataset.provenance["rest_policy"] == "continuous_filter_then_exclude_intersecting_epochs"
    assert dataset.provenance["rest_overlapping_stimulus_count"] == 1


def test_duplicate_bids_source_samples_are_rejected(
    tmp_path: Path,
) -> None:
    session_dir = _write_bids_session(
        tmp_path / "duplicate_sample",
        session_id="P001_duplicate_sample",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    manifest = _read_manifest(session_dir)
    events_path = session_dir / manifest["recording"]["bids"]["events_tsv"]
    lines = events_path.read_text(encoding="utf-8").splitlines()
    first = lines[1].split("\t")
    second = lines[2].split("\t")
    second[0] = first[0]
    second[2] = first[2]
    lines[2] = "\t".join(second)
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate source EEG samples"):
        load_brainsync_session(session_dir)


def test_multisession_loader_preserves_one_decision_per_session(tmp_path: Path) -> None:
    first = _write_bids_session(
        tmp_path / "session_a",
        session_id="P001_a",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    second = _write_bids_session(
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
    session = _write_bids_session(
        tmp_path / "session",
        session_id="P001_duplicate_input",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="unique BrainSync session directories"):
        load_brainsync_sessions([session, session])


def test_resilient_loader_skips_invalid_session_and_records_failure(tmp_path: Path) -> None:
    valid = _write_bids_session(
        tmp_path / "valid",
        session_id="P001_valid",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    invalid = _write_bids_session(
        tmp_path / "invalid",
        session_id="P002_invalid",
        target=2,
        subject_id="P002",
        started_utc="2026-09-02T00:00:00+00:00",
    )
    manifest = _read_manifest(invalid)
    _set_nested(manifest, "quality.eeg_continuity.passed", False)
    _write_manifest(invalid, manifest)
    corrupt = _write_bids_session(
        tmp_path / "corrupt",
        session_id="P003_corrupt",
        target=3,
        subject_id="P003",
        started_utc="2026-09-03T00:00:00+00:00",
    )
    corrupt_manifest = _read_manifest(corrupt)
    corrupt_raw = corrupt / corrupt_manifest["recording"]["bids"]["recording"]
    corrupt_raw.write_bytes(b"not an EDF")

    with pytest.warns(RuntimeWarning, match="Invalid measurement date"):
        result = load_brainsync_sessions_resilient(
            [invalid, corrupt, valid], invalid_session_policy=InvalidSessionPolicy.SKIP
        )

    assert set(result.dataset.subject_ids) == {"P001"}
    assert len(result.failures) == 2
    assert result.failures[0].session_dir == str(invalid.resolve())
    assert result.failures[1].session_dir == str(corrupt.resolve())
    assert result.dataset.provenance["ingress_policy"] == "skip"
    assert result.dataset.provenance["skipped_sessions"][0]["error_type"] == "ValueError"


def test_brainsync_cache_cli_runs_the_v3_bids_preprocessing_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _write_bids_session(
        tmp_path / "cli_session",
        session_id="P001_cli",
        target=4,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    output = tmp_path / "output" / "brainsync_epochs.npz"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_brainsync_cache.py",
            "--session-dir",
            str(session),
            "--output",
            str(output),
            "--invalid-session",
            "error",
        ],
    )

    prepare_brainsync_cache()

    dataset = load_epoch_dataset(output)
    assert dataset.X.shape == (9, 8, BRAIN_SYNC_PREPROCESSING.n_times)
    assert output.with_suffix(".record.json").is_file()
    assert dataset.provenance["session_schema"] == BRAIN_SYNC_SESSION_SCHEMA


def test_forced_switch_policy_retains_violation_without_replacement(tmp_path: Path) -> None:
    sessions = [
        _write_bids_session(
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
        _write_bids_session(
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
    session = _write_bids_session(
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
        _write_bids_session(
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
        _write_bids_session(
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
        _write_bids_session(
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
        _write_bids_session(
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
