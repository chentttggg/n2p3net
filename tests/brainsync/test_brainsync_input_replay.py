"""Frozen raw replay through the same interface used by inference and reports."""

from __future__ import annotations

import copy
import csv
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from types import SimpleNamespace

import mne
import numpy as np
import pytest

from data.brainsync import (
    BrainSyncNoUsableEvidenceError,
    load_brainsync_inference_session,
    load_brainsync_session,
)
from data.brainsync_contract import (
    validate_brainsync_bids_recording,
    validate_brainsync_bids_session,
)
from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc
from data.contract import BRAIN_SYNC_CAUSAL_DATA_CONTRACT
from data.domain import adapt_observed_montage
from data.epochs import materialize_dataset_identity, preprocessing_spec_from_contract
from experiments.brainsync_model_task import inference_record
from experiments.brainsync_model_task import main as inference_main
from experiments.fit_brainsync_eog import fit_brainsync_eog_sessions
from experiments.report_brainsync_quality import main as report_main
from research.recording_quality import file_sha256
from tests.support.brainsync import write_bids_session
from tests.support.io import read_json, write_json
from transfer.calibrated import (
    CalibratedPredictor,
    fit_calibrated_predictor,
    load_calibrated_predictor,
    save_calibrated_predictor,
)

CAUSAL = preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT)


@dataclass(frozen=True)
class RawCase:
    reference: str | None = None
    ocular: bool = False
    quality: bool = False
    quality_reference: str | None = None
    channels: tuple[str, ...] | None = None
    legacy_no_notch: bool = False
    flatline_min_duration_s: float | None = None


def _replay_case(root, case):
    calibration = write_bids_session(root / "calibration", session_id="calibration", target=4,
        started_utc="2026-09-06T00:00:00Z", blocks=(tuple(range(1, 10)) * 5,))
    testing = write_bids_session(root / "testing", session_id="testing", target=7,
        started_utc="2026-09-07T00:00:00Z", blocks=(tuple(range(1, 10)) * 5,))
    spec = replace(CAUSAL, notch_freqs_hz=()) if case.legacy_no_notch else CAUSAL
    source = load_brainsync_session(calibration, preprocessing=spec)
    qc = None
    if case.quality:
        recording = validate_brainsync_bids_session(calibration)
        raw = mne.io.read_raw_edf(recording.bids.raw_path, preload=True, verbose=False)
        raw.set_channel_types(dict(zip(raw.ch_names,
            [kind.lower() for kind in recording.bids.channel_types], strict=True)))
        qc = fit_continuous_qc(raw,
            spec=ContinuousQCSpec(12, 1e-7, 10, channel_local=True,
                reference_channel=case.quality_reference, minimum_channel_fraction=.5, include_eog=True,
                flatline_min_duration_s=case.flatline_min_duration_s),
            fit_sample_mask=np.ones(raw.n_times, dtype=bool), fit_provenance={
                "participant_key": materialize_dataset_identity(source).authority_keys()[0],
                "source_reference": recording.bids.source_reference,
                "calibration_session_ids": [recording.session_id],
                "calibration_end_timestamp_s": recording.ended_timestamp_s,
            })
    ocular = fit_brainsync_eog_sessions(
        [calibration], reference_channel=case.reference, continuous_qc_model=qc,
    ) if case.ocular else None
    post_reference = case.reference is not None and not case.ocular and case.quality_reference is None
    options = dict(preprocessing=spec, continuous_qc_model=qc, ocular_model=ocular,
                   channels=None if post_reference else case.channels)
    training = load_brainsync_session(calibration, **options)
    expected = load_brainsync_inference_session(testing, **options)
    if post_reference:
        channels = case.channels or tuple(name for name in training.channel_names if name != "CZ")
        training = adapt_observed_montage(training, channels, reference_channel=case.reference)
        expected = adapt_observed_montage(expected, channels, reference_channel=case.reference)
    predictor = fit_calibrated_predictor(training, declared_preprocessing=spec)
    artifact = save_calibrated_predictor(root / "model.pt", predictor)
    return SimpleNamespace(artifact=artifact, calibration=calibration, testing=testing,
                           expected=expected, preprocessing=spec, ocular=ocular, quality=qc)


@pytest.fixture(scope="module", params=[
    pytest.param(RawCase(), id="native"),
    pytest.param(RawCase(reference="Cz", channels=("PZ", "FZ")), id="post_epoch_cz_subset"),
    pytest.param(RawCase(reference="Cz", ocular=True, channels=("PZ", "FZ")), id="ocular_cz_subset"),
    pytest.param(RawCase(reference="Cz", quality=True, quality_reference="Cz", channels=("PZ", "FZ")), id="qc_cz_subset"),
    pytest.param(RawCase(reference="Cz", quality=True, channels=("PZ", "FZ")), id="native_qc_post_epoch_cz"),
    pytest.param(RawCase(reference="Cz", ocular=True, quality=True, quality_reference="Cz"), id="ocular_and_qc_cz"),
    pytest.param(RawCase(legacy_no_notch=True), id="legacy_no_notch"),
])
def replay_case(request, tmp_path_factory):
    return _replay_case(tmp_path_factory.mktemp("frozen-replay"), request.param)


def test_frozen_replay_matches_original_processing_without_learning(replay_case, monkeypatch):
    predictor = load_calibrated_predictor(replay_case.artifact)
    metadata = copy.deepcopy(predictor.metadata)
    before = file_sha256(replay_case.artifact)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("raw input replay cannot run the classifier")

    monkeypatch.setattr(CalibratedPredictor, "predict_logit", forbidden)
    actual = predictor.load_inference_session(replay_case.testing)
    assert actual.y is None
    assert actual.preprocessing == replay_case.preprocessing
    assert actual.channel_names == replay_case.expected.channel_names
    np.testing.assert_array_equal(actual.X, replay_case.expected.X)
    np.testing.assert_array_equal(actual.event_timeline.event_ids, replay_case.expected.event_timeline.event_ids)
    np.testing.assert_array_equal(actual.event_timeline.evidence_indices, replay_case.expected.event_timeline.evidence_indices)
    np.testing.assert_array_equal(actual.trial_channel_mask, replay_case.expected.trial_channel_mask)
    assert predictor.metadata == metadata
    assert file_sha256(replay_case.artifact) == before


def test_inference_and_report_consume_the_same_frozen_input(replay_case, tmp_path, monkeypatch):
    predictor = load_calibrated_predictor(replay_case.artifact)
    expected = inference_record(predictor, replay_case.expected)
    before = file_sha256(replay_case.artifact)
    inference_main(["--task", "infer", "--session-dir", str(replay_case.testing),
                    "--model-artifact", str(replay_case.artifact), "--output-dir", str(tmp_path / "infer"),
                    "--request-id", "frontend-request"])
    result = read_json(tmp_path / "infer/summary.json")
    assert result["request_id"] == "frontend-request"
    assert result["session_dirs"] == [str(replay_case.testing.resolve())]
    assert result["artifact_sha256"] == before
    assert result["subject_id"] == predictor.metadata["subject_id"]
    assert result["status"] == expected["status"]
    assert result["candidate_scores"] == expected["candidate_scores"]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the quality report cannot run the classifier")

    monkeypatch.setattr(CalibratedPredictor, "predict_logit", forbidden)
    report = report_main(["--session-dir", str(replay_case.testing), "--model-artifact", str(replay_case.artifact),
                          "--output-dir", str(tmp_path / "report")])
    assert report["status"] in {"completed", "attention"}, report.get("error")
    assert report["counts"]["scheduled_trials"] == expected["scheduled_trials"] == 45
    assert report["counts"]["preprocessed_epochs"] == replay_case.expected.n_epochs
    assert report["preprocessing"] == replay_case.preprocessing.record()
    assert file_sha256(replay_case.artifact) == before


@pytest.fixture(scope="module")
def native_case(tmp_path_factory):
    return _replay_case(tmp_path_factory.mktemp("admission"), RawCase())


@pytest.mark.parametrize("subject_id,session_id,started_utc,transport,error", [
    ("P002", "new", "2026-09-07T00:00:00Z", None, "different participant"),
    ("P001", "calibration", "2026-09-07T00:00:00Z", None, "calibration recordings"),
    ("P001", "new", "2026-09-05T00:00:00Z", None, "before calibration labels"),
    ("P001", "new", None, None, "before calibration labels"),
    ("P001", "new", "2026-09-07T00:00:00Z", "mock", "simulation status"),
])
def test_recording_admission_precedes_qc_and_empty_evidence(
    native_case, tmp_path, monkeypatch, subject_id, session_id, started_utc, transport, error,
):
    predictor = load_calibrated_predictor(native_case.artifact)
    if started_utc is None:
        started_utc = datetime.fromtimestamp(predictor.metadata["calibration_available_timestamp_s"], UTC).isoformat()
    root = write_bids_session(tmp_path / "input", session_id=session_id, target=7,
                             subject_id=subject_id, started_utc=started_utc)
    if transport is not None:
        manifest = read_json(root / "session.json")
        manifest["device"] = {"transport": transport}
        write_json(root / "session.json", manifest)
    processed = []

    def no_evidence(validated, **_kwargs):
        processed.append(validated.session_id)
        raise BrainSyncNoUsableEvidenceError(validated, {"intervals": []})

    monkeypatch.setattr("data.brainsync._build_validated_brainsync_session", no_evidence)
    with pytest.raises(ValueError, match=error):
        predictor.load_inference_session(root)
    with pytest.raises(SystemExit):
        inference_main(["--task", "infer", "--session-dir", str(root), "--model-artifact", str(native_case.artifact),
                        "--output-dir", str(tmp_path / "infer")])
    assert read_json(tmp_path / "infer/summary.json")["status"] == "failed"
    report = report_main(["--session-dir", str(root), "--model-artifact", str(native_case.artifact),
                          "--output-dir", str(tmp_path / "report")])
    assert report["status"] == "not_evaluable"
    assert error in report["error"]["message"]
    assert processed == []


def test_admitted_empty_recording_preserves_requested_denominator(native_case, tmp_path, monkeypatch):
    def no_evidence(validated, **_kwargs):
        raise BrainSyncNoUsableEvidenceError(validated, {"intervals": []})

    monkeypatch.setattr("data.brainsync._build_validated_brainsync_session", no_evidence)
    inference_main(["--task", "infer", "--session-dir", str(native_case.testing),
                    "--model-artifact", str(native_case.artifact), "--output-dir", str(tmp_path / "infer")])
    result = read_json(tmp_path / "infer/summary.json")
    assert result["status"] == "abstained"
    assert result["scheduled_trials"] == result["unavailable_trials"] == 45
    assert result["evidence_trials"] == 0
    assert result["artifact_sha256"] == file_sha256(native_case.artifact)
    assert result["subject_id"] == "P001"
    report = report_main(["--session-dir", str(native_case.testing), "--model-artifact", str(native_case.artifact),
                          "--output-dir", str(tmp_path / "report")])
    assert report["counts"]["scheduled_trials"] == report["counts"]["preprocessing_unavailable"] == 45
    assert report["counts"]["preprocessed_epochs"] == 0


@pytest.mark.parametrize("change", [
    {"channel_names": ["PZ", "FZ"]},
    {"reference_channel": "Cz"},
    {"ocular_model": {}},
    {"continuous_qc_model": {}},
    {"reference_mode": "guess"},
])
def test_inconsistent_frozen_request_is_rejected_before_raw_processing(native_case, monkeypatch, change):
    predictor = load_calibrated_predictor(native_case.artifact)
    predictor.metadata["input_request"].update(change)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid model input declaration must fail before raw processing")

    monkeypatch.setattr("data.brainsync._build_validated_brainsync_session", forbidden)
    with pytest.raises(ValueError, match="saved raw input|reference mode"):
        predictor.load_inference_session(native_case.testing)


@pytest.fixture(scope="module")
def geometry_case(tmp_path_factory):
    return _replay_case(tmp_path_factory.mktemp("geometry"), RawCase(quality=True, flatline_min_duration_s=1.0))


@pytest.mark.parametrize("empty", [False, True])
def test_channel_geometry_contract_also_applies_to_real_qc_total_rejection(geometry_case, tmp_path, monkeypatch, empty):
    predictor = load_calibrated_predictor(geometry_case.artifact)
    root = shutil.copytree(geometry_case.testing, tmp_path / "testing")
    recording = validate_brainsync_bids_recording(root)
    if empty:
        raw = mne.io.read_raw_edf(recording.bids.raw_path, preload=True, verbose=False)
        raw._data[:] = 0
        monkeypatch.setattr("data.brainsync.read_raw", lambda *_args, **_kwargs: raw.copy())
        with pytest.raises(BrainSyncNoUsableEvidenceError):
            predictor.load_inference_session(root)
    else:
        predictor.load_inference_session(root)
    with recording.bids.electrodes_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields, rows = reader.fieldnames, list(reader)
    row = next(row for row in rows if row["name"] == "Fz")
    x, y = float(row["x"]), float(row["y"])
    row["x"] = str(np.cos(.1) * x - np.sin(.1) * y)
    row["y"] = str(np.sin(.1) * x + np.cos(.1) * y)
    with recording.bids.electrodes_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    # BIDS allows explicit non-template geometry; the frozen model does not.
    validate_brainsync_bids_recording(root)
    with pytest.raises(ValueError, match="channel geometry"):
        predictor.load_inference_session(root)


def test_wrong_ocular_participant_cannot_be_hidden_by_total_qc_rejection(tmp_path, monkeypatch):
    case = _replay_case(tmp_path, RawCase(reference="Cz", ocular=True, quality=True,
                                        quality_reference="Cz", flatline_min_duration_s=1.0))
    recording = validate_brainsync_bids_recording(case.testing)
    raw = mne.io.read_raw_edf(recording.bids.raw_path, preload=True, verbose=False)
    raw._data[:] = 0
    monkeypatch.setattr("data.brainsync.read_raw", lambda *_args, **_kwargs: raw.copy())
    options = dict(preprocessing=case.preprocessing, continuous_qc_model=case.quality)
    with pytest.raises(BrainSyncNoUsableEvidenceError):
        load_brainsync_inference_session(case.testing, ocular_model=case.ocular, **options)
    wrong = replace(case.ocular, participant_key="other-participant")
    with pytest.raises(ValueError, match="ocular model calibration belongs to a different participant"):
        load_brainsync_inference_session(case.testing, ocular_model=wrong, **options)
