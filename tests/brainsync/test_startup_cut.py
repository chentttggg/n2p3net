import hashlib
import json

import mne
import numpy as np
import pytest

from data.brainsync import load_brainsync_session
from data.preprocess import continuous_filter_segments
from tests.support.brainsync import write_bids_session


def test_complete_trial_rounding_keeps_the_first_full_baseline():
    from data.brainsync_startup import plan_startup_cut

    plan = plan_startup_cut(np.array([1250, 1625, 2000, 2375, 2750]), 250., 4000)
    assert plan["nominal_stop_sample"] == 2250
    assert plan["stop_sample"] == 2325
    assert plan["dropped_event_indices"] == [0, 1, 2]
    assert plan["first_retained_event_index"] == 3
    # A shorter actual SOA removes four flashes, rather than hard-coding three.
    faster = plan_startup_cut(np.array([1250, 1525, 1800, 2075, 2350, 2625]), 250., 4000)
    assert faster["dropped_event_indices"] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="complete.*trial"):
        plan_startup_cut(np.array([1250, 1625]), 250., 2000)


def _registered(monkeypatch, root):
    from data import brainsync_startup as startup
    from data.brainsync_contract import validate_brainsync_bids_session

    session = validate_brainsync_bids_session(root)
    try:
        digest = hashlib.sha256(session.bids.raw_path.read_bytes()).hexdigest()
        plan = startup.plan_startup_cut(np.array([s.sample for s in session.bids.stimuli]),
                                       session.bids.sample_rate_hz,
                                       round(session.bids.duration_seconds * session.bids.sample_rate_hz))
        entry = {**plan, "raw_sha256": digest, "session_id": session.session_id,
                 "authority": "test owner confirmed device startup fault"}
        monkeypatch.setattr(startup, "startup_cuts", lambda: {digest: entry})
        return session.bids.raw_path, entry
    finally:
        session.bids.close()


def test_real_loader_filters_only_the_suffix_and_keeps_requested_events(monkeypatch, tmp_path):
    from data.brainsync_startup import apply_startup_cut

    root = write_bids_session(tmp_path / "input", session_id="startup-cut", target=4,
                              started_utc="2026-09-06T00:00:00Z")
    path, entry = _registered(monkeypatch, root)
    actual = load_brainsync_session(root)
    assert actual.event_timeline.n_events == 9
    assert np.all(actual.event_timeline.evidence_indices[:3] == -1)
    assert actual.n_epochs == 6
    assert actual.provenance["raw_startup_cut"]["stop_sample"] == entry["stop_sample"]
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    before = raw.get_data().copy()
    apply_startup_cut(raw, entry)
    assert continuous_filter_segments(raw) == ((entry["stop_sample"], raw.n_times),)
    np.testing.assert_array_equal(raw.get_data(), before)
    raw.close()


def test_removed_prefix_cannot_change_the_retained_filter_outputs(monkeypatch, tmp_path):
    from data.brainsync_startup import apply_startup_cut, plan_startup_cut
    from data.contract import DEFAULT_P300_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract
    from data.preprocess import execute_preprocessing

    rate = 250.
    raw = mne.io.RawArray(np.random.default_rng(9).normal(0, 5e-6, (3, 6000)),
                         mne.create_info(["Fz", "Cz", "Pz"], rate, "eeg"), verbose=False)
    events = np.column_stack((np.arange(1250, 5000, 375), np.zeros(10, int), np.ones(10, int)))
    cut = plan_startup_cut(events[:, 0], rate, raw.n_times)
    changed = raw.copy()
    changed._data[:, :cut["stop_sample"]] += .1 * np.sin(np.arange(cut["stop_sample"]) / 3)
    spec = preprocessing_spec_from_contract(DEFAULT_P300_DATA_CONTRACT)
    control_a = execute_preprocessing(raw, events, spec)
    control_b = execute_preprocessing(changed, events, spec)
    assert not np.array_equal(control_a.data[3:], control_b.data[3:])
    for recording in (raw, changed):
        apply_startup_cut(recording, cut)
    a, b = (execute_preprocessing(r, events, spec) for r in (raw, changed))
    np.testing.assert_array_equal(a.data, b.data)
    assert a.event_indices.tolist() == list(range(3, 10))


def test_native_edf_cut_preserves_every_retained_digital_sample(tmp_path):
    import edfio

    from data.brainsync_startup import write_cut_edf

    root = write_bids_session(tmp_path / "input", session_id="cut-writer", target=4,
                              started_utc="2026-09-06T00:00:00Z")
    manifest = json.loads((root / "session.json").read_text())
    source = root / manifest["recording"]["path"]
    output = tmp_path / "cropped.edf"
    receipt = write_cut_edf(source, output, 1325)
    original, cropped = edfio.read_edf(source), edfio.read_edf(output)
    for left, right in zip(original.signals, cropped.signals, strict=True):
        np.testing.assert_array_equal(left.digital[1325:], right.digital)
        assert left.physical_range == right.physical_range
    assert receipt["digital_suffix_equal"] is True
    with pytest.raises(FileExistsError):
        write_cut_edf(source, output, 1325)


def test_qc_and_eog_fits_use_only_the_cut_original_support(monkeypatch, tmp_path):
    from dataclasses import replace

    from data.brainsync import BRAIN_SYNC_PREPROCESSING
    from data.brainsync_startup import POLICY
    from data.ocular import PRESERVE_ACQUIRED_EOG
    from experiments.fit_brainsync_continuous_qc import fit_brainsync_continuous_manifest
    from experiments.fit_brainsync_eog import fit_brainsync_eog_sessions

    root = write_bids_session(tmp_path / "input", session_id="cut-fit", target=4,
                              started_utc="2026-09-06T00:00:00Z")
    path, entry = _registered(monkeypatch, root)
    original = mne.io.read_raw_edf(path, preload=True, verbose=False)
    values = original.get_data()
    stop = entry["stop_sample"]
    manifest = tmp_path / "fit.json"
    manifest.write_text(json.dumps({"schema": "brainsync_continuous_qc_calibration/1",
        "preprocessing": BRAIN_SYNC_PREPROCESSING.record(),
        "spec": {"difference_z": 12., "difference_floor_v": 1e-7, "include_eog": True,
                 "settle_samples": 25, "channel_local": True},
        "epoch_recovery": {"tolerance_v": 1e-6, "inflation": 2., "warmup_samples": 250,
                           "probe_step_samples": 250},
        "sessions": [{"session_dir": str(root), "clean_intervals_samples": [[0, int(original.n_times)]],
                      "clean_interval_evidence": "whole explicit calibration"}]}))
    qc = fit_brainsync_continuous_manifest(manifest)
    assert qc.fit_provenance["startup_cut_policy"] == POLICY
    expected_differences = np.diff(values[:, stop:], axis=1)
    np.testing.assert_array_equal(qc.difference_center_v, np.median(expected_differences, axis=1))
    eog = fit_brainsync_eog_sessions([root], reference_channel="Cz",
                                    eog_reference_policy=PRESERVE_ACQUIRED_EOG, fit_domain="native_raw")
    assert eog.fit_provenance["startup_cut_policy"] == POLICY
    x = values[original.ch_names.index("EOG"), stop:]
    x = (x-x.mean())[:, None]
    y = values[[original.ch_names.index(n) for n in eog.eeg_channels], stop:]
    y = y-values[original.ch_names.index("Cz"), stop:]
    expected = np.linalg.lstsq(x, (y-y.mean(axis=1, keepdims=True)).T, rcond=None)[0].T
    np.testing.assert_allclose(eog.coefficients, expected, rtol=1e-12, atol=1e-12)
    old = replace(qc, fit_provenance={k: v for k,v in qc.fit_provenance.items() if k != "startup_cut_policy"})
    with pytest.raises(ValueError, match="refit QC/EOG"):
        load_brainsync_session(root, continuous_qc_model=old)
    original.close()


def test_batch_exports_raw_sources_and_accounts_for_failed_recordings(tmp_path):
    from data.source_records import file_sha256
    from experiments.cut_brainsync_startup import publish

    root = write_bids_session(tmp_path / "input", session_id="cut-batch", target=4,
                              started_utc="2026-09-06T00:00:00Z")
    session = json.loads((root / "session.json").read_text())
    source = root / session["recording"]["path"]
    before = file_sha256(source)
    quality, registry = tmp_path / "quality.json", tmp_path / "registry.json"
    quality.write_text(json.dumps({"sessions": [
        {"directory_id": "cut-batch", "status": "completed", "input_root": str(root)},
        {"directory_id": "failed", "status": "recording_error", "input_root": "not-a-complete-record"}]}))
    registry.write_text(json.dumps({"schema": "brainsync_startup_cuts/1", "recordings": []}))
    output = tmp_path / "cut"
    result = publish(quality, output, registry)
    assert result["status"] == "completed" and len(result["recordings"]) == 1
    assert len(result["unavailable"]) == 1
    record = result["recordings"][0]
    assert record["remaining_scheduled_events"] == 6
    assert record["digital_suffix_equal"] is True and record["filters_applied"] == []
    assert record["raw_sha256"] == before == file_sha256(source)
    assert json.loads(registry.read_text())["recordings"][0]["stop_sample"] == 1325
    assert (output / "index.html").is_file()


def test_native_raw_capability_cannot_grant_the_removed_prefix(monkeypatch, tmp_path):
    from contextlib import closing

    from data import brainsync_raw_authority as authority
    from tests.support.brainsync import PAUSE_BOUNDARY, write_paused_recording

    root = write_paused_recording(tmp_path)
    _, entry = _registered(monkeypatch, root)
    sid = json.loads((root / "session.json").read_text())["session_id"]
    scope = authority.BrainSyncRecordingScope((), (sid,))
    with closing(authority.open_brainsync_recording(root, scope=scope)) as issuer:
        with pytest.raises(ValueError, match="startup"):
            issuer.issue_raw(0, prefix_stop=250, full_stop=1500, available_stop=1500)
        cap = issuer.issue_raw(0, prefix_stop=entry["stop_sample"]+250,
                               full_stop=PAUSE_BOUNDARY, available_stop=PAUSE_BOUNDARY)
        assert cap.public_binding["sample_start"] == entry["stop_sample"]
        assert cap.public_binding["public_acquisition_segment"]["stored_start"] == 0
        assert cap.eeg.shape[0] == PAUSE_BOUNDARY-entry["stop_sample"]
        history = {e["event_id"]: e for e in cap.public_binding["public_physical_history"]["exposures"]}
        assert all(e.segment_sample == history[e.event_id]["segment_sample"] for e in cap.events)
    calibration_scope = authority.BrainSyncRecordingScope((sid,), ("other-record",))
    with closing(authority.open_brainsync_calibration(root, scope=calibration_scope)) as issuer:
        labels = issuer.issue_labels(3, ready_at_timestamp_s=10**12)
        # Physical stimulation and the original C permission remain real;
        # these labels do not restore their deleted EEG training rows.
        assert len(labels.labels) == 3
        assert len(labels.public_binding["startup_events_without_EEG_support"]) == 3
        from research.target_noise_operand import CalibrationNoiseRecord, _compile_c_residuals

        session = json.loads((root / "session.json").read_text())
        end = session["recording"]["clock"]["acquired_samples"]
        caps = (issuer.issue_raw(0, prefix_stop=entry["stop_sample"]+250,
                                  full_stop=PAUSE_BOUNDARY, available_stop=PAUSE_BOUNDARY),
                issuer.issue_raw(1, prefix_stop=PAUSE_BOUNDARY+250, full_stop=end, available_stop=end))
        labels = issuer.issue_labels(90, ready_at_timestamp_s=10**12)
        assert sum(len(authority.observed_event_indices(c)) for c in caps) == 87
        calls = []

        def fixed_mean(cap, granted):
            calls.append(cap.identity_sha256)
            exposures = cap.public_binding["public_physical_history"]["exposures"]
            considered = [e for e in exposures if e["sample"] < cap.public_binding["full_stop"]]
            return np.zeros_like(cap.eeg), np.zeros_like(cap.eeg), {
                "source_response_model_identity": "a"*64, "target_response_C_updates_before_noise_fit": 0,
                "mean_role": "caller_declared_analytic_fixed_mean",
                "used_label_event_ids": [e["event_id"] for e in considered if e["exposure_role"] == "completed_protocol_stimulus"],
                "considered_exposure_ids": [e["event_id"] for e in considered],
                "unknown_exposure_law": "declared_zero_mean_software_fixture"}

        residuals, _ = _compile_c_residuals((CalibrationNoiseRecord(caps, labels),), fixed_mean)
        assert len(residuals) == len(calls) == 2


def test_stale_partial_fit_rejected_when_another_fit_source_is_cut(monkeypatch):
    from data import brainsync_startup as startup

    a, b = {"policy": startup.POLICY, "raw_sha256": "a"}, {"policy": startup.POLICY, "raw_sha256": "b"}
    monkeypatch.setattr(startup, "startup_cuts", lambda: {"a": a, "b": b})
    provenance = {"startup_cut_policy": startup.POLICY, "sources": [
        {"raw_sha256": "a", "raw_startup_cut": a}, {"raw_sha256": "b", "raw_startup_cut": None}]}
    with pytest.raises(ValueError, match="refit QC/EOG"):
        startup.require_startup_fit(provenance, a)
