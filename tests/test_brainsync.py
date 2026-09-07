from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
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
    load_brainsync_inference_session,
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
    BRAIN_SYNC_ACQUISITION_CHANNELS,
    BRAIN_SYNC_ACQUISITION_TYPES,
    BRAIN_SYNC_EEG_CHANNELS,
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    SOURCE_COHORT_DATA_CONTRACTS,
    assert_causal_p300_input_contract,
)
from data.epochs import EpochDataset, load_epoch_dataset, save_epoch_dataset
from data.identity import DatasetIdentityTable
from data.ocular import load_eog_regression, validate_eog_calibration_budget
from experiments.fit_brainsync_eog import main as fit_brainsync_eog
from experiments.prepare_brainsync_cache import main as prepare_brainsync_cache
from experiments.run_brainsync_cross_decision import main as run_brainsync_cross_decision
from models.n2p3net import N2P3Net
from research.contracts import TrainingRunContract
from transfer.checkpoint import CHECKPOINT_SCHEMA
from transfer.within_subject import calibration_decision_split

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


def test_eog_is_preserved_in_raw_but_never_in_model_epochs(tmp_path: Path) -> None:
    session = _write_bids_session(
        tmp_path / "session", session_id="eog-layout", target=4, started_utc="2026-09-06T00:00:00Z"
    )
    dataset = load_brainsync_session(session)
    assert dataset.n_channels == 7
    assert dataset.channel_names == tuple(name.upper() for name in BRAIN_SYNC_EEG_CHANNELS)
    assert "EOG" not in dataset.channel_names and "PO7" not in dataset.channel_names
    assert dataset.provenance["acquisition_layout"]["channel_names"] == list(CHANNELS)
    assert dataset.provenance["ocular_artifact_processing"]["status"] == "not_applied"
    assert dataset.provenance["ocular_artifact_processing"]["eog_in_classifier"] is False
    raw = mne.io.read_raw_edf(dataset.provenance["recording_path"], verbose=False)
    assert len(raw.ch_names) == 8 and raw.ch_names[6] == "EOG"
    path = save_epoch_dataset(tmp_path / "seven.npz", dataset)
    assert load_epoch_dataset(path, validation="attested").n_channels == 7


def test_strong_eog_changes_do_not_change_uncleaned_eeg_epochs(tmp_path: Path) -> None:
    arguments = dict(session_id="same-waveforms", target=4, started_utc="2026-09-06T00:00:00Z")
    first = load_brainsync_session(_write_bids_session(tmp_path / "a", **arguments))
    second = load_brainsync_session(
        _write_bids_session(tmp_path / "b", eog_scale=200.0, **arguments)
    )
    np.testing.assert_array_equal(first.X, second.X)


def test_explicit_eog_classifier_request_is_rejected(tmp_path: Path) -> None:
    session = _write_bids_session(
        tmp_path / "session", session_id="eog-request", target=4, started_utc="2026-09-06T00:00:00Z"
    )
    with pytest.raises(ValueError, match="never EOG"):
        load_brainsync_session(session, channels=("Fz", "EOG"))


def test_inference_ingress_ignores_later_target_confirmation_without_fabricated_labels(
    tmp_path: Path,
) -> None:
    session = _write_bids_session(
        tmp_path / "unknown",
        session_id="unknown",
        target=4,
        started_utc="2026-09-06T00:00:00+00:00",
        blocks=(tuple(range(1, 10)) * 5,),
    )
    manifest = _read_manifest(session)
    manifest["target_label"] = {"status": "pending"}
    manifest["experiment"].pop("thought_digit")
    manifest["experiment"]["target_label_status"] = "pending_post_experiment_confirmation"
    _write_manifest(session, manifest)
    with pytest.raises(ValueError, match="post-experiment confirmed"):
        load_brainsync_session(session)
    pending = load_brainsync_inference_session(session)
    assert pending.y is None
    assert pending.n_epochs == 45
    assert not pending.event_timeline.has_candidate_sets
    assert pending.event_timeline.has_candidate_ids
    assert pending.provenance["candidate_vocabulary"] == [str(value) for value in range(1, 10)]
    assert "target_candidate_id" not in pending.metadata
    assert "target_digits" not in pending.provenance
    assert "calibration_available_timestamp_s" not in pending.metadata
    assert pending.metadata["repetition_index"].value_counts().to_dict() == dict.fromkeys(
        range(5), 9
    )
    assert pending.metadata["recording_available_timestamp_s"].notna().all()
    roundtrip = load_epoch_dataset(save_epoch_dataset(tmp_path / "unknown.npz", pending))
    assert roundtrip.y is None
    manifest["target_label"] = {
        "status": "confirmed",
        "source": "post_experiment_confirmation",
        "thought_digit": 8,
        "confirmed_utc": "2026-09-06T01:00:00+00:00",
    }
    manifest["experiment"]["thought_digit"] = 8
    manifest["experiment"]["target_label_status"] = "confirmed_post_experiment"
    _write_manifest(session, manifest)
    confirmed = load_brainsync_inference_session(session)
    np.testing.assert_array_equal(confirmed.X, pending.X)
    assert confirmed.metadata.equals(pending.metadata)
    assert confirmed.provenance == pending.provenance
    np.testing.assert_array_equal(
        confirmed.event_timeline.target_candidate_ids, pending.event_timeline.target_candidate_ids
    )


@pytest.mark.parametrize("invalid_kind", ["recording_error", "premature_raw", "eog_role"])
def test_unlabelled_inference_keeps_recording_and_physical_role_gates(
    tmp_path: Path, invalid_kind: str
) -> None:
    session = _write_bids_session(
        tmp_path / invalid_kind,
        session_id=invalid_kind,
        target=3,
        started_utc="2026-09-06T00:00:00+00:00",
    )
    manifest = _read_manifest(session)
    if invalid_kind == "recording_error":
        manifest["status"] = "recording_error"
    elif invalid_kind == "premature_raw":
        manifest["recording"]["timeline"]["status"] = "pending"
    else:
        channels = session / manifest["recording"]["bids"]["channels_tsv"]
        channels.write_text(
            channels.read_text(encoding="utf-8").replace("EOG\tEOG", "EOG\tEEG"), encoding="utf-8"
        )
    _write_manifest(session, manifest)
    with pytest.raises(ValueError):
        load_brainsync_inference_session(session)


@pytest.mark.parametrize("transport", ["mock", "ble", None])
def test_simulation_provenance_follows_declared_transport_for_both_loaders(
    tmp_path: Path, transport: str | None
) -> None:
    session = _write_bids_session(
        tmp_path / "mock_named_path",
        session_id="mock_named_id",
        target=3,
        started_utc="2026-09-06T00:00:00+00:00",
    )
    manifest = _read_manifest(session)
    if transport is not None:
        manifest["device"] = {"transport": transport}
    _write_manifest(session, manifest)
    expected = "declared_mock" if transport == "mock" else "not_declared"
    assert load_brainsync_session(session).provenance["simulation_status"] == expected
    assert load_brainsync_inference_session(session).provenance["simulation_status"] == expected


def test_personal_calibration_cannot_mix_declared_mock_and_other_records(tmp_path: Path) -> None:
    sessions = [
        _write_bids_session(
            tmp_path / f"mixed_{index}",
            session_id=f"mixed_{index}",
            target=index,
            started_utc=f"2026-09-0{index}T00:00:00+00:00",
        )
        for index in (1, 2)
    ]
    manifest = _read_manifest(sessions[0])
    manifest["device"] = {"transport": "mock"}
    _write_manifest(sessions[0], manifest)
    with pytest.raises(ValueError, match="cannot mix declared mock"):
        load_brainsync_sessions(sessions)


@pytest.mark.parametrize("replacement", ["EOG\tEEG", "PO8\tEOG", "PO7\tEEG"])
def test_brainsync_rejects_ambiguous_or_legacy_slot_roles(tmp_path: Path, replacement: str) -> None:
    session = _write_bids_session(
        tmp_path / "session", session_id="bad-roles", target=4, started_utc="2026-09-06T00:00:00Z"
    )
    path = next(session.rglob("*_channels.tsv"))
    original = "EOG\tEOG" if replacement.startswith("EOG") else "PO8\tEEG"
    path.write_text(path.read_text().replace(original, replacement), encoding="utf-8")
    with pytest.raises(ValueError):
        load_brainsync_session(session)


def test_eog_does_not_require_fabricated_scalp_coordinates(tmp_path: Path) -> None:
    session = _write_bids_session(
        tmp_path / "session",
        session_id="eog-no-coords",
        target=4,
        started_utc="2026-09-06T00:00:00Z",
    )
    path = next(session.rglob("*_electrodes.tsv"))
    path.write_text(
        "\n".join(line for line in path.read_text().splitlines() if not line.startswith("EOG\t"))
        + "\n",
        encoding="utf-8",
    )
    assert load_brainsync_session(session).n_channels == 7


def test_sessions_with_different_bad_channels_keep_trial_observation_masks(tmp_path):
    first = _write_bids_session(
        tmp_path / "a", session_id="a", target=1, started_utc="2026-09-06T00:00:00Z"
    )
    second = _write_bids_session(
        tmp_path / "b", session_id="b", target=2, started_utc="2026-09-06T01:00:00Z"
    )
    path = next(first.rglob("*_channels.tsv"))
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(row[:-4] + "bad" if row.startswith("P3\t") else row for row in rows) + "\n",
        encoding="utf-8",
    )
    dataset = load_brainsync_sessions([first, second])
    index = dataset.channel_names.index("P3")
    assert dataset.channel_mask[index]
    assert not dataset.trial_channel_mask[:9, index].any()
    assert dataset.trial_channel_mask[9:, index].all()
    assert np.all(dataset.X[:9, index] == 0.0)


def test_paused_producer_events_exclude_boundary_crossing_epochs(tmp_path, monkeypatch):
    producer_root = os.environ.get("BRAIN_SYNC_WORKSTATION_ROOT")
    if not producer_root:
        pytest.skip("set BRAIN_SYNC_WORKSTATION_ROOT for producer integration")
    monkeypatch.syspath_prepend(producer_root)
    from gtn_workstation.bids import make_bids_session_layout, write_bids_events

    root = _write_bids_session(
        tmp_path / "paused", session_id="paused", target=1,
        started_utc="2026-09-06T00:00:00Z",
    )
    manifest_path = root / "session.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recording = manifest["recording"]
    path = root / recording["bids"]["events_tsv"]
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    markers = []
    for row in rows:
        for kind, delta in (("onset", 0), ("offset", float(row["duration"]))):
            wall = float(row["onset"]) + delta
            wall += 60 if wall >= 8 else 0
            markers.append({
                "event": "recording_marker", "payload": {
                    "kind": kind, "eeg_time_seconds": wall,
                    "eeg_time_base": "continuous_recording", "trial_id": row["trial_id"],
                    "digit": int(row["digit"]), "block_id": int(row["block_id"]),
                    "trial_index": int(row["trial_index"]),
                },
            })
    ledger = root / "pause-events.jsonl"
    ledger.write_text("".join(json.dumps(marker) + "\n" for marker in markers), encoding="utf-8")
    pauses = [{"pause_id": 1, "boundary_sample": 2000, "start_wall_seconds": 8.0, "end_wall_seconds": 68.0}]
    layout = make_bids_session_layout(root, "P001", "session")
    generated = write_bids_events(layout, ledger, 250.0, recording["timeline"]["output_duration_seconds"], pauses=pauses)
    recording["bids"]["events_tsv"] = generated.relative_to(root).as_posix()
    recording["timeline"].update(time_base="segmented_recording", acquisition_pauses=pauses, pause_samples_retained=False)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sidecar_path = root / recording["bids"]["eeg_json"]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.update(RecordingType="discontinuous", AcquisitionPauses=pauses)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    dataset = load_brainsync_session(root)
    assert dataset.event_timeline.evidence_indices[4] == -1
    assert dataset.X.shape[0] == 8
    assert dataset.provenance["event_time_base"] == "segmented_recording"
    recording["timeline"]["acquisition_pauses"][0]["boundary_sample"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="boundaries conflict"):
        load_brainsync_session(root)


def test_workstation_manifest_matches_physical_channel_contract() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/brainsync_workstation.model.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["input_contract"]["session_schema"] == BRAIN_SYNC_SESSION_SCHEMA
    assert manifest["input_contract"]["acquisition_channels"] == list(
        BRAIN_SYNC_ACQUISITION_CHANNELS
    )
    assert manifest["input_contract"]["acquisition_channel_types"] == list(
        BRAIN_SYNC_ACQUISITION_TYPES
    )
    assert manifest["output_contract"]["model_eeg_channels"] == list(BRAIN_SYNC_EEG_CHANNELS)
    assert manifest["output_contract"]["eog_in_classifier"] is False


def test_collection_plan_matches_all_digit_calibration_and_independent_r5_test() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/brainsync_collection_plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    calibration, test = plan["calibration"], plan["test"]
    assert test["target_policy"] == DecisionTargetPolicy.OBSERVED_SEQUENCE.value
    assert set(calibration["target_sequence"]) == set(calibration["target_vocabulary"])
    assert len(calibration["target_sequence"]) == calibration["decisions_per_participant"]
    vocabulary = set(test["candidate_vocabulary"])
    assert set(calibration["target_vocabulary"]) == set(test["target_vocabulary"]) == vocabulary
    for phase in (calibration, test):
        experiment = phase["experiment"]
        assert experiment["digits_per_block"] % len(vocabulary) == 0
    experiment = calibration["experiment"]
    calibration_trials = (
        calibration["decisions_per_participant"]
        * experiment["blocks"]
        * experiment["digits_per_block"]
    )
    assert calibration_trials == calibration["stimulus_trials"] == 4050
    assert calibration_trials // len(vocabulary) == calibration["target_trials"] == 450
    assert (
        calibration_trials * experiment["soa_ms"] / 1000.0
        == calibration["nominal_soa_budget_seconds"]
    )
    assert test["experiment"]["blocks"] == 1
    assert (
        test["experiment"]["digits_per_block"] // len(vocabulary)
        == test["primary_repetitions"]
        == 5
    )
    assert (
        len(test["target_vocabulary"]) * test["decisions_per_target"]
        == test["decisions_per_participant"]
    )
    assert test["test_labels_for_scoring_only"]
    assert test["model_and_recipe_frozen_before_test"]
    assert test["labels_required_for_inference"] is False


@pytest.mark.parametrize("reference_channel", [None, "Cz"])
def test_heldout_brainsync_uses_only_past_same_participant_ocular_fit(
    tmp_path, monkeypatch, reference_channel
):
    calibration = _write_bids_session(
        tmp_path / "cal", session_id="known-cal", target=4, started_utc="2026-09-06T00:00:00Z"
    )
    later = _write_bids_session(
        tmp_path / "later", session_id="unknown-later", target=7, started_utc="2026-09-06T01:00:00Z"
    )
    earlier = _write_bids_session(
        tmp_path / "earlier",
        session_id="earlier-test",
        target=2,
        started_utc="2026-09-05T23:00:00Z",
    )
    another = _write_bids_session(
        tmp_path / "another",
        session_id="other-person",
        target=1,
        subject_id="P002",
        started_utc="2026-09-06T02:00:00Z",
    )
    model_path = tmp_path / "ocular.json"
    argv = [
        "fit_brainsync_eog",
        "--calibration-session-dir",
        str(calibration),
        "--output",
        str(model_path),
    ]
    if reference_channel is not None:
        argv += ["--reference-channel", reference_channel]
    monkeypatch.setattr("sys.argv", argv)
    fit_brainsync_eog()
    model = load_eog_regression(model_path)
    assert load_brainsync_session(calibration, ocular_model=model).n_channels == (
        7 if reference_channel is None else 6
    )
    cleaned = load_brainsync_session(later, ocular_model=model)
    unlabelled = load_brainsync_inference_session(later, ocular_model=model)
    assert unlabelled.y is None
    np.testing.assert_array_equal(unlabelled.X, cleaned.X)
    assert unlabelled.provenance["source_reference"] == cleaned.provenance["source_reference"]
    if reference_channel is not None:
        assert cleaned.provenance["source_reference"] == "measured_channel:CZ"
        assert cleaned.provenance["acquisition_source_reference"] == "A2"
        assert "CZ" not in cleaned.channel_names
    assert cleaned.provenance["ocular_artifact_processing"]["status"] == "applied"
    assert cleaned.provenance["ocular_artifact_processing"]["model"]["fit_provenance"][
        "calibration_session_ids"
    ] == ["known-cal"]
    validate_eog_calibration_budget(
        cleaned.provenance["ocular_artifact_processing"], {model.participant_key: ("known-cal",)}
    )
    with pytest.raises(ValueError, match="outside the declared calibration budget"):
        validate_eog_calibration_budget(
            cleaned.provenance["ocular_artifact_processing"], {model.participant_key: ()}
        )
    with pytest.raises(ValueError, match="future data"):
        load_brainsync_session(earlier, ocular_model=model)
    with pytest.raises(ValueError, match="different participant_key"):
        load_brainsync_session(another, ocular_model=model)


def test_actual_workstation_producer_roundtrips_seven_eeg_one_eog(tmp_path, monkeypatch):
    producer_root = os.environ.get("BRAIN_SYNC_WORKSTATION_ROOT")
    if not producer_root:
        pytest.skip("set BRAIN_SYNC_WORKSTATION_ROOT for the independent producer integration")
    monkeypatch.syspath_prepend(producer_root)
    from gtn_workstation.bids import BidsRawInfo, make_bids_session_layout, write_bids_eeg_sidecars
    from gtn_workstation.models import ChannelMontage, ExperimentConfig

    session = _write_bids_session(
        tmp_path / "session",
        session_id="producer-consumer",
        target=4,
        started_utc="2026-09-06T00:00:00Z",
    )
    manifest = _read_manifest(session)
    raw_path = session / manifest["recording"]["path"]
    raw = mne.io.read_raw_edf(raw_path, verbose=False)
    info = BidsRawInfo(
        tuple(raw.ch_names), float(raw.info["sfreq"]), raw.n_times / raw.info["sfreq"], "uV"
    )
    layout = make_bids_session_layout(session, "P001", "session")
    montage = ChannelMontage(ref_label="A2", gnd_label="M1")
    write_bids_eeg_sidecars(
        layout,
        info,
        montage,
        ExperimentConfig(),
        sdk_version="synthetic-integration",
        impedance_values=[],
    )
    sidecar = json.loads(layout.sidecar_path("eeg").read_text(encoding="utf-8"))
    assert sidecar["EEGChannelCount"] == 7 and sidecar["EOGChannelCount"] == 1
    loaded = load_brainsync_session(session)
    assert loaded.n_channels == 7 and "EOG" not in loaded.channel_names
    active = ChannelMontage.from_dict(
        json.loads((Path(producer_root) / "channel_assignments.json").read_text(encoding="utf-8"))
    )
    assert active.eeg == BRAIN_SYNC_ACQUISITION_CHANNELS
    assert tuple(kind.upper() for kind in active.channel_types) == BRAIN_SYNC_ACQUISITION_TYPES
    from data.channel import build_channel_identity

    expected = build_channel_identity(BRAIN_SYNC_EEG_CHANNELS)
    positions = [value for value in active.channel_positions_m if value is not None]
    np.testing.assert_allclose(positions, expected.coords, rtol=0, atol=1e-9)


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
    training_identity_ledger = DatasetIdentityTable.from_source_rows(["other"], ["external-source"])
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
        payload_architecture["st_temporal_filters"] = (
            int(payload_architecture["st_temporal_filters"]) + 1
        )
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

    assert dataset.X.shape == (18, len(BRAIN_SYNC_EEG_CHANNELS), BRAIN_SYNC_PREPROCESSING.n_times)
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


@pytest.mark.parametrize("row_change", ["truncated", "extra_field"])
def test_malformed_bids_table_is_reported_by_resilient_ingress(
    tmp_path: Path, row_change: str
) -> None:
    valid = _write_bids_session(
        tmp_path / "valid",
        session_id="valid",
        target=3,
        started_utc="2026-09-06T00:00:00Z",
    )
    malformed = _write_bids_session(
        tmp_path / "malformed",
        session_id="malformed",
        target=7,
        started_utc="2026-09-06T01:00:00Z",
    )
    manifest = _read_manifest(malformed)
    channels_path = malformed / manifest["recording"]["bids"]["channels_tsv"]
    rows = channels_path.read_text(encoding="utf-8").splitlines()
    fields = rows[1].split("\t")
    rows[1] = "\t".join(fields[:-1] if row_change == "truncated" else fields + ["extra"])
    channels_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row 2.*field count"):
        load_brainsync_session(malformed)
    result = load_brainsync_sessions_resilient([valid, malformed], invalid_session_policy="skip")
    assert result.dataset.provenance["session_ids"] == ["valid"]
    assert len(result.failures) == 1
    assert result.failures[0].error_type == "ValueError"
    assert "field count" in result.failures[0].message


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
    assert dataset.X.shape == (9, len(BRAIN_SYNC_EEG_CHANNELS), BRAIN_SYNC_PREPROCESSING.n_times)
    assert output.with_suffix(".record.json").is_file()
    assert dataset.provenance["session_schema"] == BRAIN_SYNC_SESSION_SCHEMA


def test_post_recording_confirmation_keeps_explicit_calibration_availability(
    tmp_path: Path,
) -> None:
    session = _write_bids_session(
        tmp_path / "late_label",
        session_id="late_label",
        target=4,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    manifest = _read_manifest(session)
    ended = datetime.fromisoformat(manifest["ended_utc"])
    confirmed = ended + timedelta(seconds=5)
    manifest["target_label"]["confirmed_utc"] = confirmed.isoformat()
    _write_manifest(session, manifest)

    dataset = load_brainsync_session(session)
    assert dataset.metadata["session_end_timestamp_s"].eq(ended.timestamp()).all()
    assert dataset.metadata["target_confirmed_timestamp_s"].eq(confirmed.timestamp()).all()
    assert dataset.metadata["calibration_available_timestamp_s"].eq(confirmed.timestamp()).all()
    assert dataset.provenance["session_ended_utc"] == ended.isoformat()


def test_late_calibration_label_cannot_retroactively_train_for_requested_test(
    tmp_path: Path,
) -> None:
    sessions = [
        _write_bids_session(
            tmp_path / f"label_time_{index}",
            session_id=f"label_time_{index}",
            target=index,
            started_utc=f"2026-09-0{index}T00:00:00+00:00",
        )
        for index in range(1, 5)
    ]
    calibration = _read_manifest(sessions[0])
    calibration["target_label"]["confirmed_utc"] = "2026-09-02T12:00:00+00:00"
    _write_manifest(sessions[0], calibration)
    test_label = _read_manifest(sessions[2])
    test_label["target_label"]["confirmed_utc"] = "2026-09-10T00:00:00+00:00"
    _write_manifest(sessions[2], test_label)
    dataset = load_brainsync_sessions(sessions)

    split = calibration_decision_split(
        dataset,
        calibration_selections=1,
        test_repetitions=1,
        target_policy=DecisionTargetPolicy.OBSERVED_SEQUENCE,
        max_test_selections=2,
        candidate_vocabulary=range(1, 10),
    )
    requested = split.requested_test_groups_by_subject["P001"]
    assert len(requested) == 2
    assert split.test_groups_by_subject["P001"] == (requested[1],)
    assert split.failed_test_groups_by_subject["P001"] == {
        requested[0]: "selection_precedes_calibration_availability"
    }
    assert all("label_time_4" not in group for group in requested)


@pytest.mark.parametrize("invalid_time", ["before_start", "before_last_stimulus"])
def test_post_recording_confirmation_does_not_relax_past_time_constraints(
    tmp_path: Path, invalid_time: str
) -> None:
    session = _write_bids_session(
        tmp_path / invalid_time,
        session_id=invalid_time,
        target=2,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    manifest = _read_manifest(session)
    manifest["target_label"]["confirmed_utc"] = (
        "2026-08-31T23:59:59+00:00"
        if invalid_time == "before_start"
        else "2026-09-01T00:00:01+00:00"
    )
    _write_manifest(session, manifest)
    with pytest.raises(ValueError, match="timestamps|confirmation precedes"):
        load_brainsync_session(session)


@pytest.mark.parametrize("bad_metadata", ["missing", "nonfinite", "inconsistent", "forged_earlier"])
def test_brainsync_split_requires_explicit_consistent_label_availability(
    tmp_path: Path, bad_metadata: str
) -> None:
    sessions = [
        _write_bids_session(
            tmp_path / f"metadata_{index}",
            session_id=f"metadata_{index}",
            target=index,
            started_utc=f"2026-09-0{index}T00:00:00+00:00",
        )
        for index in (1, 2)
    ]
    dataset = load_brainsync_sessions(sessions)
    field = "calibration_available_timestamp_s"
    if bad_metadata == "missing":
        dataset.metadata = dataset.metadata.drop(columns=[field])
    elif bad_metadata == "nonfinite":
        dataset.metadata.loc[0, field] = np.nan
    elif bad_metadata == "inconsistent":
        dataset.metadata.loc[0, field] += 1.0
    else:
        dataset.metadata[field] = dataset.metadata["session_start_timestamp_s"]
    with pytest.raises(
        ValueError, match="calibration.*availability|calibration_available_timestamp_s"
    ):
        calibration_decision_split(
            dataset,
            calibration_selections=1,
            test_repetitions=1,
            target_policy=DecisionTargetPolicy.OBSERVED_SEQUENCE,
            candidate_vocabulary=range(1, 10),
        )


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
    assert split.calibration_mask.any()
    assert len(split.calibration_groups_by_subject["P001"]) == 1


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


@pytest.mark.parametrize(
    ("head", "with_ocular"),
    [
        ("zero_shot", False),
        ("target_only_tuned", False),
        ("target_shrinkage_lda", False),
        ("zero_shot", True),
    ],
)
def test_brainsync_runner_derives_scope_and_keeps_policy_failure_in_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, head: str, with_ocular: bool
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
    ocular_model = None
    if with_ocular:
        model_path = tmp_path / "runner_ocular.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "fit_brainsync_eog",
                "--calibration-session-dir",
                str(sessions[0]),
                "--output",
                str(model_path),
            ],
        )
        fit_brainsync_eog()
        ocular_model = load_eog_regression(model_path)
    dataset = load_brainsync_sessions(sessions, ocular_model=ocular_model)
    cache = save_epoch_dataset(tmp_path / "brainsync.npz", dataset)
    source_snapshot_manifest, source_snapshot_sha256 = _write_source_freeze_manifest(tmp_path)
    evaluation_root = tmp_path / "evaluation_source"
    evaluation_root.mkdir()
    evaluation_manifest, evaluation_sha256 = _write_source_freeze_manifest(evaluation_root)
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
            "--evaluation-snapshot-manifest",
            str(evaluation_manifest),
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
            head,
            "--normalization",
            "source" if head == "zero_shot" else "target_prefix",
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
    assert result["source_snapshot_sha256"] == source_snapshot_sha256
    assert result["evaluation_contract"]["source_snapshot_sha256"] == evaluation_sha256
    if with_ocular:
        assert result["evidence_scope"]["target_preprocessing_calibrated"] is True
        assert result["evidence_scope"]["zero_target_fit_claim"] is False
        argv = list(sys.argv)
        argv[argv.index("--max-test-selections") + 1] = "1"
        monkeypatch.setattr("sys.argv", argv)
        run_brainsync_cross_decision()
        all_failed = json.loads(output.read_text(encoding="utf-8"))
        assert all_failed["decision_accounting"]["requested"] == 1
        assert all_failed["decision_accounting"]["eligible"] == 0
        assert all_failed["decision_accounting"]["pre_evaluation_failed"] == 1
    if head != "zero_shot":
        assert result["evaluation_contract"]["model_origin"]["kind"] == "scratch"
        assert result["evaluation_contract"]["adaptation"]["source_statistics_used"] is False
        assert result["records"][0]["calibration_trials"] == 9
        assert result["records"][0]["calibration_positives"] == 1
        if head == "target_only_tuned":
            assert result["epochs"] == 60
            assert result["lr"] == 1e-3
            assert result["evaluation_contract"]["adaptation"]["freeze_batchnorm_stats"] is False
        source = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for value in source["trunk_state_dict"].values():
            if value.is_floating_point():
                value.fill_(17.0)
        source["input_mean"] = [1e9] * dataset.n_channels
        source["input_std"] = [1e-3] * dataset.n_channels
        torch.save(source, checkpoint)
        run_brainsync_cross_decision()
        repeated = json.loads(output.read_text(encoding="utf-8"))
        np.testing.assert_array_equal(
            result["records"][0]["suffix_predictions"]["raw_logits"],
            repeated["records"][0]["suffix_predictions"]["raw_logits"],
        )


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
    source_snapshot_manifest, source_snapshot_sha256 = _write_source_freeze_manifest(tmp_path)
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
