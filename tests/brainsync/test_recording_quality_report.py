from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import mne
import numpy as np
import pytest

from data.artifact import FoldLocalArtifactPolicy
from data.brainsync import (
    BrainSyncNoUsableEvidenceError,
    load_brainsync_inference_session,
    load_brainsync_session,
)
from data.brainsync_contract import validate_brainsync_bids_recording
from data.domain import adapt_observed_montage
from data.personal_quality import PersonalEpochQualityPolicy
from experiments.report_brainsync_quality import main
from research.recording_quality import (
    REPORT_SCHEMA,
    _block_counts,
    _raw_diagnostics,
    create_recording_quality_report,
    file_sha256,
)
from tests.support.brainsync import write_bids_session
from tests.support.io import read_json, write_json
from transfer.calibrated import (
    CalibratedPredictor,
    fit_calibrated_predictor,
    save_calibrated_predictor,
)
from transfer.quality import fit_epoch_quality


@pytest.fixture
def recording(tmp_path):
    return write_bids_session(tmp_path / "recording", session_id="quality-test", target=4,
                               started_utc="2026-09-06T00:00:00Z", blocks=(tuple(range(1, 10)) * 5,))


def test_target_free_report_preserves_raw_and_uses_shared_preprocessing(recording, tmp_path, monkeypatch):
    manifest = recording / "session.json"
    payload = read_json(manifest)
    payload["target_label"] = {"status": "pending"}
    payload["experiment"]["thought_digit"] = None
    write_json(manifest, payload)
    raw = recording / payload["recording"]["path"]
    before = {path: file_sha256(path) for path in (manifest, raw)}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("report must not fit QC or evaluate a classifier")

    monkeypatch.setattr(PersonalEpochQualityPolicy, "fit", forbidden)
    monkeypatch.setattr(FoldLocalArtifactPolicy, "fit", forbidden)
    monkeypatch.setattr(CalibratedPredictor, "predict_logit", forbidden)
    report = main(["--session-dir", str(recording), "--output-dir", str(tmp_path / "report"), "--reference-mode", "native"])
    expected = load_brainsync_inference_session(recording)
    assert report["schema"] == REPORT_SCHEMA
    assert report["status"] in {"completed", "attention"}
    assert report["counts"]["preprocessed_epochs"] == expected.n_epochs
    assert report["counts"]["scheduled_trials"] == expected.event_timeline.n_events
    assert report["counts"]["qc_retained_epochs"] is None
    assert not report["target_labels_used"] and not report["fit_performed"]
    assert len(report["plots"]) == 4
    assert all((tmp_path / "report" / plot["file"]).is_file() for plot in report["plots"])
    assert (tmp_path / "report" / "report.html").is_file()
    assert before == {path: file_sha256(path) for path in before}


def test_target_edits_do_not_change_quality_statistics(recording, tmp_path):
    first = create_recording_quality_report(recording, tmp_path / "a")
    manifest = recording / "session.json"
    payload = read_json(manifest)
    payload["target_label"] = {"status": "confirmed", "thought_digit": 8}
    payload["experiment"]["thought_digit"] = 8
    write_json(manifest, payload)
    second = create_recording_quality_report(recording, tmp_path / "b")
    assert first["status"] in {"completed", "attention"}
    assert first["raw"] == second["raw"]
    assert first["counts"] == second["counts"]
    assert first["epoch_channels"] == second["epoch_channels"]


def test_aborted_recording_gets_explanation_without_running_preprocessing(recording, tmp_path):
    manifest = recording / "session.json"
    payload = read_json(manifest)
    payload["status"] = "aborted"
    write_json(manifest, payload)

    def forbidden(_root):
        raise AssertionError("aborted recording cannot be an algorithm input")

    report = create_recording_quality_report(recording, tmp_path / "report", loader=forbidden)
    assert report["status"] == "not_evaluable"
    assert report["error"]
    assert report["plots"] == []
    assert (tmp_path / "report" / "report.html").is_file()


def test_exact_constant_diagnostics_do_not_claim_fitted_qc(recording, tmp_path):
    dataset = load_brainsync_inference_session(recording)
    constant = replace(dataset, X=np.full_like(dataset.X, 1.23e-6))
    report = create_recording_quality_report(recording, tmp_path / "report", loader=lambda _root: constant)
    assert report["status"] == "attention"
    assert report["counts"]["constant_channel_epochs"] == dataset.n_epochs * dataset.n_channels
    assert report["epoch_quality"]["status"] == "not_configured"


def test_frozen_qc_total_rejection_keeps_all_requested_events(recording, tmp_path):
    calibration = load_brainsync_session(recording)
    frozen = fit_epoch_quality(calibration, np.ones(calibration.n_epochs, dtype=bool),
                               FoldLocalArtifactPolicy(min_clean_epochs=2))
    testing = load_brainsync_inference_session(recording)
    testing = replace(testing, X=testing.X * 1e6)
    report = create_recording_quality_report(recording, tmp_path / "report", loader=lambda _root: testing,
                                             epoch_quality=frozen)
    assert report["status"] == "attention"
    assert report["counts"]["scheduled_trials"] == testing.event_timeline.n_events
    assert report["counts"]["qc_retained_epochs"] == 0
    assert report["epoch_quality"]["rejected_epochs"] == testing.n_epochs
    assert not report["epoch_quality"]["fit_performed"]


def test_reports_cannot_overwrite_existing_versions(recording, tmp_path):
    output = tmp_path / "report"
    create_recording_quality_report(recording, output)
    before = file_sha256(output / "report.json")
    with pytest.raises(FileExistsError, match="immutable"):
        create_recording_quality_report(recording, output)
    assert before == file_sha256(output / "report.json")


@pytest.mark.parametrize("reference_channel", [None, "Cz"])
def test_saved_personal_model_is_checked_without_predicting_or_updating(recording, tmp_path, monkeypatch, reference_channel):
    calibration = load_brainsync_session(recording)
    if reference_channel is not None:
        calibration = adapt_observed_montage(
            calibration, tuple(name for name in calibration.channel_names if name != "CZ"),
            reference_channel=reference_channel,
        )
    model = save_calibrated_predictor(tmp_path / "model.pt", fit_calibrated_predictor(calibration))
    testing = write_bids_session(tmp_path / "testing", session_id="new-test", target=7,
                                 started_utc="2026-09-07T00:00:00Z")
    before = file_sha256(model)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("quality report must not evaluate a classifier")

    monkeypatch.setattr(CalibratedPredictor, "predict_logit", forbidden)
    report = main(["--session-dir", str(testing), "--output-dir", str(tmp_path / "frozen-report"), "--model-artifact", str(model)])
    assert report["status"] in {"completed", "attention"}
    assert report["request"]["model_sha256"] == before == file_sha256(model)


def test_completely_missing_block_keeps_its_requested_denominator():
    blocks = _block_counts([1, 1, 2, 2], [0, 1, -1, -1], np.array([False, True]))
    assert blocks == [
        {"block_id": "1", "scheduled_trials": 2, "preprocessed_epochs": 2, "qc_retained_epochs": 1},
        {"block_id": "2", "scheduled_trials": 2, "preprocessed_epochs": 0, "qc_retained_epochs": 0},
    ]


def test_edf_padding_cannot_hide_constant_acquired_signal(tmp_path, monkeypatch):
    raw = mne.io.RawArray(np.array([[1e-6] * 5 + [0.] * 5, [2e-6] * 5 + [0.] * 5]),
                         mne.create_info(["Fz", "EOG"], 10, ["eeg", "eog"]), verbose=False)
    monkeypatch.setattr("research.recording_quality.read_raw", lambda *_args, **_kwargs: raw)
    validated = SimpleNamespace(manifest={"recording": {"clock": {"acquired_samples": 5}}},
        bids=SimpleNamespace(raw_path=tmp_path / "raw.edf", channel_names=("Fz", "EOG"), sample_rate_hz=10,
                             source_reference="left earlobe", acquisition_boundaries=(),
                             channel_types=("EEG", "EOG"), channel_statuses=("good", "good")))
    result, _ = _raw_diagnostics(validated, tmp_path)
    assert result["edf_padding_samples"] == 5
    assert all(channel["entire_recording_constant"] for channel in result["channels"])
    assert all(channel["ptp_uv"] == 0 for channel in result["channels"])


def test_continuous_qc_empty_evidence_has_visible_counts(recording, tmp_path):
    validated = validate_brainsync_bids_recording(recording)

    def unavailable(_root):
        raise BrainSyncNoUsableEvidenceError(validated, {"intervals": []})

    report = create_recording_quality_report(recording, tmp_path / "empty-report", loader=unavailable)
    assert report["status"] == "not_evaluable"
    assert report["counts"]["preprocessing_unavailable"] == len(validated.bids.stimuli)
    conclusion = dict(next(fields for title, fields in report["sections"] if title == "可用性结论"))
    assert conclusion["预处理保留epoch"] == 0
    assert conclusion["原排程闪烁"] == len(validated.bids.stimuli)


def test_input_change_during_generation_invalidates_the_report(recording, tmp_path):
    validated = validate_brainsync_bids_recording(recording)

    def changed_input(root):
        dataset = load_brainsync_inference_session(root)
        with validated.bids.events_path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        return dataset

    report = create_recording_quality_report(recording, tmp_path / "changed-report", loader=changed_input)
    assert report["status"] == "not_evaluable"
    assert "发生变化" in report["error"]["message"]
