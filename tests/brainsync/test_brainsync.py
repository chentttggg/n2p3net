from __future__ import annotations

import csv
import json
import os
from dataclasses import replace
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
    BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT,
    BRAIN_SYNC_CAUSAL_DATA_CONTRACT,
    BRAIN_SYNC_EEG_CHANNELS,
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    SOURCE_COHORT_DATA_CONTRACTS,
    assert_causal_p300_input_contract,
)
from data.epochs import load_epoch_dataset, save_epoch_dataset
from data.ocular import load_eog_regression, validate_eog_calibration_budget
from experiments.fit_brainsync_eog import main as fit_brainsync_eog
from experiments.prepare_brainsync_cache import main as prepare_brainsync_cache
from experiments.run_brainsync_cross_decision import main as run_brainsync_cross_decision
from tests import PROJECT_ROOT
from tests.support.brainsync import write_bids_session
from tests.support.checkpoints import write_checkpoint
from tests.support.io import read_json, read_tsv, write_json, write_tsv
from tests.support.snapshots import write_source_snapshot
from transfer.within_subject import calibration_decision_split

CHANNELS = BRAIN_SYNC_ACQUISITION_CHANNELS


def test_pending_confirmation_is_rejected_before_reading_raw(tmp_path):
    from data.brainsync_contract import (
        validate_brainsync_bids_recording,
        validate_brainsync_bids_session,
    )

    (tmp_path / ".target-confirmation.pending.json").write_text("{}", encoding="utf-8")
    for validate in (validate_brainsync_bids_recording, validate_brainsync_bids_session):
        with pytest.raises(ValueError, match="confirmation.*pending recovery"):
            validate(tmp_path)


def test_continuous_qc_preserves_scheduled_events_and_binds_cache(tmp_path):
    from data.brainsync_contract import validate_brainsync_bids_session
    from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc
    from data.epochs import materialize_dataset_identity, preprocessing_spec_from_contract

    root = write_bids_session(tmp_path / "continuous", session_id="qc-recording", target=4,
                               started_utc="2026-09-06T00:00:00Z")
    preprocessing = preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT)
    before = load_brainsync_session(root, preprocessing=preprocessing)
    session = validate_brainsync_bids_session(root)
    raw = mne.io.read_raw_edf(session.bids.raw_path, preload=True, verbose=False)
    raw.set_channel_types(dict(zip(raw.ch_names, [kind.lower() for kind in session.bids.channel_types], strict=True)))
    participant = materialize_dataset_identity(before).record_for("P001").authority_key("source_or_global")
    fitted = fit_continuous_qc(raw, spec=ContinuousQCSpec(12, 1e-7, 10),
        fit_sample_mask=np.arange(raw.n_times) < 1000,
        fit_provenance={"participant_key": participant, "source_reference": session.bids.source_reference,
                        "calibration_session_ids": [session.session_id],
                        "calibration_end_timestamp_s": session.ended_timestamp_s})
    raw._data[3, 2000:] += .0142
    raw.export(session.bids.raw_path, overwrite=True, physical_range="auto", verbose=False)
    dataset = load_brainsync_session(root, preprocessing=preprocessing, continuous_qc_model=fitted)
    assert dataset.n_epochs == 4
    assert len(dataset.event_timeline.event_ids) == 9
    assert np.sum(dataset.event_timeline.evidence_indices < 0) == 5
    assert np.all(dataset.event_timeline.evidence_available_times_s[:4] == raw.n_times / 250)
    cached = load_epoch_dataset(save_epoch_dataset(tmp_path / "qc.npz", dataset))
    np.testing.assert_array_equal(cached.X, dataset.X)
    assert cached.provenance["continuous_artifact_processing"]["model_digest"] == dataset.provenance["continuous_artifact_processing"]["model_digest"]


def test_eog_is_preserved_in_raw_but_never_in_model_epochs(tmp_path: Path) -> None:
    session = write_bids_session(
        tmp_path / "session", session_id="eog-layout", target=4, started_utc="2026-09-06T00:00:00Z"
    )
    dataset = load_brainsync_session(session)
    assert dataset.n_channels == 7
    assert dataset.channel_names == tuple(name.upper() for name in BRAIN_SYNC_EEG_CHANNELS)
    assert "EOG" not in dataset.channel_names and "PO7" not in dataset.channel_names
    assert dataset.provenance["acquisition_layout"]["channel_names"] == list(CHANNELS)
    assert dataset.provenance["ocular_artifact_processing"]["status"] == "not_applied"
    assert dataset.provenance["ocular_artifact_processing"]["eog_in_classifier"] is False
    receipt = dataset.provenance["preprocessing_execution"]
    assert receipt["declaration"] == dataset.preprocessing.record()
    assert receipt["output"]["scheduled_events"] == dataset.event_timeline.n_events
    raw = mne.io.read_raw_edf(dataset.provenance["recording_path"], verbose=False)
    assert len(raw.ch_names) == 8 and raw.ch_names[6] == "EOG"
    path = save_epoch_dataset(tmp_path / "seven.npz", dataset)
    reloaded = load_epoch_dataset(path, validation="attested")
    assert reloaded.n_channels == 7
    assert reloaded.provenance["preprocessing_execution"] == receipt


def test_strong_eog_changes_do_not_change_uncleaned_eeg_epochs(tmp_path: Path) -> None:
    arguments = dict(session_id="same-waveforms", target=4, started_utc="2026-09-06T00:00:00Z")
    first = load_brainsync_session(write_bids_session(tmp_path / "a", **arguments))
    second = load_brainsync_session(
        write_bids_session(tmp_path / "b", eog_scale=200.0, **arguments)
    )
    np.testing.assert_array_equal(first.X, second.X)


def test_explicit_eog_classifier_request_is_rejected(tmp_path: Path) -> None:
    session = write_bids_session(
        tmp_path / "session", session_id="eog-request", target=4, started_utc="2026-09-06T00:00:00Z"
    )
    with pytest.raises(ValueError, match="never EOG"):
        load_brainsync_session(session, channels=("Fz", "EOG"))


def test_inference_ingress_ignores_later_target_confirmation_without_fabricated_labels(
    tmp_path: Path,
) -> None:
    session = write_bids_session(
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
    session = write_bids_session(
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
    session = write_bids_session(
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
    sessions = _daily_sessions(tmp_path, (1, 2), prefix="mixed_")
    manifest = _read_manifest(sessions[0])
    manifest["device"] = {"transport": "mock"}
    _write_manifest(sessions[0], manifest)
    with pytest.raises(ValueError, match="cannot mix declared mock"):
        load_brainsync_sessions(sessions)


@pytest.mark.parametrize("replacement", ["EOG\tEEG", "PO8\tEOG", "PO7\tEEG"])
def test_brainsync_rejects_ambiguous_or_legacy_slot_roles(tmp_path: Path, replacement: str) -> None:
    session = write_bids_session(
        tmp_path / "session", session_id="bad-roles", target=4, started_utc="2026-09-06T00:00:00Z"
    )
    path = next(session.rglob("*_channels.tsv"))
    original = "EOG\tEOG" if replacement.startswith("EOG") else "PO8\tEEG"
    path.write_text(path.read_text().replace(original, replacement), encoding="utf-8")
    with pytest.raises(ValueError):
        load_brainsync_session(session)


def test_eog_does_not_require_fabricated_scalp_coordinates(tmp_path: Path) -> None:
    session = write_bids_session(
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
    first = write_bids_session(
        tmp_path / "a", session_id="a", target=1, started_utc="2026-09-06T00:00:00Z"
    )
    second = write_bids_session(
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

    root = write_bids_session(
        tmp_path / "paused", session_id="paused", target=1,
        started_utc="2026-09-06T00:00:00Z",
    )
    manifest_path = root / "session.json"
    manifest = read_json(manifest_path)
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
    write_json(manifest_path, manifest)
    sidecar_path = root / recording["bids"]["eeg_json"]
    sidecar = read_json(sidecar_path)
    sidecar.update(RecordingType="discontinuous", AcquisitionPauses=pauses)
    write_json(sidecar_path, sidecar)
    dataset = load_brainsync_session(root)
    assert dataset.event_timeline.evidence_indices[4] == -1
    assert dataset.X.shape[0] == 8
    assert dataset.provenance["event_time_base"] == "segmented_recording"
    recording["timeline"]["acquisition_pauses"][0]["boundary_sample"] += 1
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="boundaries conflict"):
        load_brainsync_session(root)


def test_workstation_manifest_matches_physical_channel_contract() -> None:
    path = PROJECT_ROOT / "configs/brainsync_workstation.model.json"
    manifest = read_json(path)
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
    path = PROJECT_ROOT / "configs/brainsync_collection_plan.json"
    plan = read_json(path)
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
    calibration = write_bids_session(
        tmp_path / "cal", session_id="known-cal", target=4, started_utc="2026-09-06T00:00:00Z"
    )
    later = write_bids_session(
        tmp_path / "later", session_id="unknown-later", target=7, started_utc="2026-09-06T01:00:00Z"
    )
    earlier = write_bids_session(
        tmp_path / "earlier",
        session_id="earlier-test",
        target=2,
        started_utc="2026-09-05T23:00:00Z",
    )
    another = write_bids_session(
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

    session = write_bids_session(
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
    sidecar = read_json(layout.sidecar_path("eeg"))
    assert sidecar["EEGChannelCount"] == 7 and sidecar["EOGChannelCount"] == 1
    loaded = load_brainsync_session(session)
    assert loaded.n_channels == 7 and "EOG" not in loaded.channel_names
    active = ChannelMontage.from_dict(
        read_json(Path(producer_root) / "channel_assignments.json")
    )
    assert active.eeg == BRAIN_SYNC_ACQUISITION_CHANNELS
    assert tuple(kind.upper() for kind in active.channel_types) == BRAIN_SYNC_ACQUISITION_TYPES
    from data.channel import build_channel_identity

    expected = build_channel_identity(BRAIN_SYNC_EEG_CHANNELS)
    positions = [value for value in active.channel_positions_m if value is not None]
    np.testing.assert_allclose(positions, expected.coords, rtol=0, atol=1e-9)


def _runner_argv(cache, checkpoint, snapshot, output, *, head="zero_shot", normalization="source", test_limit=1):
    return [
        "run_brainsync_cross_decision.py", "--dataset-cache", str(cache),
        "--checkpoint", str(checkpoint), "--arm-name", "zero_shot_source",
        "--source-snapshot-manifest", str(snapshot), "--identity-exclusion-policy", "source",
        "--calibration-selections", "1", "--test-reps", "1",
        "--max-test-selections", str(test_limit), "--target-policy", "forced_switch",
        "--population-policy", "adult_only", "--head", head,
        "--normalization", normalization,
        "--device", "cpu", "--output", str(output),
    ]


def _daily_sessions(root, targets, *, prefix):
    return [
        write_bids_session(
            root / f"{prefix}{index}", session_id=f"{prefix}{index}", target=target,
            started_utc=f"2026-09-{index:02d}T00:00:00+00:00",
        )
        for index, target in enumerate(targets, start=1)
    ]


def _read_manifest(session_dir: Path) -> dict[str, object]:
    return read_json(session_dir / "session.json")


def _write_manifest(session_dir: Path, payload: dict[str, object]) -> None:
    write_json(session_dir / "session.json", payload)


def _set_nested(payload: dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    node = payload
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


@pytest.fixture
def bids_session(tmp_path):
    return write_bids_session(
        tmp_path / "metadata", session_id="P001_gate", target=1,
        started_utc="2026-09-01T00:00:00+00:00",
        blocks=(tuple(range(1, 10)),) * 2,
    )


@pytest.fixture
def metadata_session(bids_session, monkeypatch):
    monkeypatch.setattr(
        "data.brainsync.read_raw",
        lambda *args, **kwargs: pytest.fail("EEG reader ran before the metadata gate"),
    )
    return bids_session


def test_brainsync_uses_the_canonical_bidirectional_profile() -> None:
    assert BRAIN_SYNC_PREPROCESSING.name == BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT.name
    assert BRAIN_SYNC_PREPROCESSING.notch_freqs_hz == (50.0,)
    assert BRAIN_SYNC_PREPROCESSING.notch_quality_factor == 30.0
    assert SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.notch_freqs_hz == ()
    assert BRAIN_SYNC_PREPROCESSING.l_freq == 0.1
    assert BRAIN_SYNC_PREPROCESSING.h_freq == 30.0
    assert BRAIN_SYNC_PREPROCESSING.tmax_ms == 1200.0
    assert BRAIN_SYNC_PREPROCESSING.filter_phase == "zero"
    assert BRAIN_SYNC_PREPROCESSING.causal_iir_initial_state == "not_applicable"
    assert BRAIN_SYNC_PREPROCESSING.zero_phase_edge_policy == "mne_reflect_limited"
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
    session_dir = write_bids_session(
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
        ("recording.bids.events_tsv", "../outside.tsv", "inside the dataset root"),
    ],
)
def test_v4_manifest_gate_fails_before_binary_eeg_read(metadata_session, field, value, message):
    manifest = _read_manifest(metadata_session)
    _set_nested(manifest, field, value)
    _write_manifest(metadata_session, manifest)
    with pytest.raises(ValueError, match=message):
        load_brainsync_session(metadata_session)


@pytest.mark.parametrize(
    "table,row,changes,message",
    [
        pytest.param("events_tsv", -1, {"value": "1", "digit": "1"},
                     "does not balance every candidate", id="unbalanced-candidates"),
        pytest.param("electrodes_tsv", 0, {"x": "not-a-number"},
                     "finite number", id="invalid-coordinate"),
        pytest.param("events_tsv", 0, {"sample": "252"},
                     "more than half a sample", id="sample-onset-mismatch"),
    ],
)
def test_bids_table_gates_precede_binary_read(metadata_session, table, row, changes, message):
    path = metadata_session / _read_manifest(metadata_session)["recording"]["bids"][table]
    rows = read_tsv(path)
    rows[row].update(changes)
    write_tsv(path, rows)
    with pytest.raises(ValueError, match=message):
        load_brainsync_session(metadata_session)


def test_duplicate_bids_source_samples_are_rejected(bids_session):
    path = bids_session / _read_manifest(bids_session)["recording"]["bids"]["events_tsv"]
    rows = read_tsv(path)
    rows[1].update(onset=rows[0]["onset"], sample=rows[0]["sample"])
    write_tsv(path, rows)
    with pytest.raises(ValueError, match="duplicate source EEG samples"):
        load_brainsync_session(bids_session)


def test_v4_rejects_block_balanced_but_non_permutation_cycles(tmp_path: Path) -> None:
    session_dir = write_bids_session(
        tmp_path / "bad_cycles",
        session_id="P001_bad_cycles",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
        blocks=(tuple(value for digit in range(1, 10) for value in (digit, digit)),),
    )

    with pytest.raises(ValueError, match="not a complete candidate permutation"):
        load_brainsync_session(session_dir)


def test_manifest_rest_ledger_must_match_bids_events_before_raw_read(metadata_session):
    manifest = _read_manifest(metadata_session)
    rest = manifest["recording"]["timeline"]["rest_segments"][0]
    rest["end_seconds"] += 0.5
    rest["duration_seconds"] += 0.5
    _write_manifest(metadata_session, manifest)
    with pytest.raises(ValueError, match="conflict with BIDS events.tsv"):
        load_brainsync_session(metadata_session)


def test_rest_overlap_is_excluded_after_continuous_filtering_without_time_shift(
    tmp_path: Path,
) -> None:
    session_dir = write_bids_session(
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


def test_multisession_loader_preserves_one_decision_per_session(tmp_path: Path) -> None:
    first = write_bids_session(
        tmp_path / "session_a",
        session_id="P001_a",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    second = write_bids_session(
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
    session = write_bids_session(
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
    valid = write_bids_session(
        tmp_path / "valid",
        session_id="valid",
        target=3,
        started_utc="2026-09-06T00:00:00Z",
    )
    malformed = write_bids_session(
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
    valid = write_bids_session(
        tmp_path / "valid",
        session_id="P001_valid",
        target=1,
        started_utc="2026-09-01T00:00:00+00:00",
    )
    invalid = write_bids_session(
        tmp_path / "invalid",
        session_id="P002_invalid",
        target=2,
        subject_id="P002",
        started_utc="2026-09-02T00:00:00+00:00",
    )
    manifest = _read_manifest(invalid)
    _set_nested(manifest, "quality.eeg_continuity.passed", False)
    _write_manifest(invalid, manifest)
    corrupt = write_bids_session(
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
    session = write_bids_session(
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
    session = write_bids_session(
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
    sessions = _daily_sessions(tmp_path, (1, 2, 3, 4), prefix="label_time_")
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
    session = write_bids_session(
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
    sessions = _daily_sessions(tmp_path, (1, 2), prefix="metadata_")
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


@pytest.mark.parametrize(
    "policy,targets,calibration,eligible,failures,excluded",
    [
        pytest.param(DecisionTargetPolicy.FORCED_SWITCH, (1, 1, 2, 3), 1, ("policy3",),
                     {"policy2": "target_policy_same_as_previous_decision"}, {}, id="forced-switch"),
        pytest.param(DecisionTargetPolicy.FORCED_SWITCH, (1, 1), 1, (),
                     {"policy2": "target_policy_same_as_previous_decision"},
                     {"P001": "no_eligible_unknown_test_decision"}, id="all-failed"),
        pytest.param(DecisionTargetPolicy.OBSERVED_SEQUENCE, (1, 1), 1, ("policy2",),
                     {}, {}, id="observed-sequence"),
        pytest.param(DecisionTargetPolicy.UNSEEN_CALIBRATION_CODES, (1, 2, 1, 3), 2, ("policy4",),
                     {"policy3": "target_policy_seen_in_calibration"}, {}, id="unseen-targets"),
    ],
)
def test_target_policy_keeps_requested_decisions_and_typed_failures(
    tmp_path, policy, targets, calibration, eligible, failures, excluded,
):
    dataset = load_brainsync_sessions(_daily_sessions(tmp_path, targets, prefix="policy"))
    split = calibration_decision_split(
        dataset, calibration_selections=calibration, test_repetitions=1,
        target_policy=policy, max_test_selections=2, candidate_vocabulary=range(1, 10),
    )
    timeline = dataset.event_timeline
    groups = dict(zip(timeline.selection_ids, timeline.group_ids, strict=True))
    requested = tuple(groups[f"policy{i}"] for i in range(calibration + 1, min(len(targets), calibration + 2) + 1))
    assert split.requested_test_groups_by_subject["P001"] == requested
    assert split.test_groups_by_subject.get("P001", ()) == tuple(groups[name] for name in eligible)
    assert split.failed_test_groups_by_subject["P001"] == {groups[name]: reason for name, reason in failures.items()}
    assert split.excluded_subjects == excluded
    assert split.usable_subjects == (("P001",) if eligible else ())
    assert len(split.calibration_groups_by_subject["P001"]) == calibration
    assert split.calibration_mask.sum() == 9 * calibration
    if policy == DecisionTargetPolicy.OBSERVED_SEQUENCE:
        scope = derive_brainsync_evidence_scope(
            derive_population_scope(("P001",) * 2, (19.0,) * 2,
                                    policy=PopulationScopePolicy.DESCRIPTIVE),
            target_policy=split.target_policy,
        )
        assert scope.decision_claim == "later_session_observed_target_sequence"
        assert scope.population.label == "age_descriptive"


def test_split_rejects_legacy_cache_that_promotes_blocks_to_decisions(
    tmp_path: Path,
) -> None:
    session = write_bids_session(
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
    ("head", "normalization", "with_ocular"),
    [
        ("zero_shot", "source", False),
        ("target_only_tuned", "target_prefix", False),
        ("target_shrinkage_lda", "target_prefix", False),
        ("zero_shot", "source", True),
    ],
)
def test_brainsync_runner_derives_scope_and_keeps_policy_failure_in_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, head: str, normalization: str, with_ocular: bool
) -> None:
    sessions = _daily_sessions(tmp_path, (1, 1, 2), prefix="P001_runner_")
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
    source_snapshot_manifest, source_snapshot_sha256, _ = write_source_snapshot(tmp_path)
    evaluation_root = tmp_path / "evaluation_source"
    evaluation_root.mkdir()
    evaluation_manifest, evaluation_sha256, _ = write_source_snapshot(evaluation_root)
    checkpoint = write_checkpoint(
        tmp_path / "source.pt",
        dataset,
        source_snapshot_sha256=source_snapshot_sha256,
    )
    output = tmp_path / "result.json"
    argv = _runner_argv(cache, checkpoint, source_snapshot_manifest, output, head=head,
                        normalization=normalization, test_limit=2)
    argv.extend(["--evaluation-snapshot-manifest", str(evaluation_manifest)])
    monkeypatch.setattr("sys.argv", argv)

    run_brainsync_cross_decision()

    result = read_json(output)
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
        argv = _runner_argv(cache, checkpoint, source_snapshot_manifest, output, test_limit=1)
        argv.extend(["--evaluation-snapshot-manifest", str(evaluation_manifest)])
        monkeypatch.setattr("sys.argv", argv)
        run_brainsync_cross_decision()
        all_failed = read_json(output)
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
        repeated = read_json(output)
        np.testing.assert_array_equal(
            result["records"][0]["suffix_predictions"]["raw_logits"],
            repeated["records"][0]["suffix_predictions"]["raw_logits"],
        )


def test_brainsync_runner_validates_bad_checkpoint_when_policy_leaves_no_usable_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = _daily_sessions(tmp_path, (1, 1), prefix="P001_empty_runner_")
    dataset = load_brainsync_sessions(sessions)
    cache = save_epoch_dataset(tmp_path / "empty_brainsync.npz", dataset)
    source_snapshot_manifest, source_snapshot_sha256, _ = write_source_snapshot(tmp_path)
    mismatched_snapshot = "b" * 64
    assert mismatched_snapshot != source_snapshot_sha256
    checkpoint = write_checkpoint(
        tmp_path / "wrong_snapshot.pt",
        dataset,
        source_snapshot_sha256=mismatched_snapshot,
    )
    output = tmp_path / "must_not_exist.json"
    argv = _runner_argv(cache, checkpoint, source_snapshot_manifest, output)
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(ValueError, match="source snapshot disagrees"):
        run_brainsync_cross_decision()
    assert not output.exists()

    bad_architecture_checkpoint = write_checkpoint(
        tmp_path / "bad_architecture.pt",
        dataset,
        source_snapshot_sha256=source_snapshot_sha256,
    )
    payload = torch.load(bad_architecture_checkpoint, weights_only=True)
    payload["architecture"]["st_temporal_filters"] += 1
    torch.save(payload, bad_architecture_checkpoint)
    monkeypatch.setattr("sys.argv", _runner_argv(cache, bad_architecture_checkpoint, source_snapshot_manifest, output))
    with pytest.raises(ValueError, match="architecture disagrees"):
        run_brainsync_cross_decision()
    assert not output.exists()
