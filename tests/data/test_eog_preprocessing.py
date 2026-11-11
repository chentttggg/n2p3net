"""Source-only ocular fitting, truthful missing EOG and EEG-only saved inputs."""
from dataclasses import replace

import mne
import numpy as np
import pytest

from data.domain import adapt_observed_montage
from data.eog_preprocessing import EOGPreprocessingPolicy, prepare_source_eog
from data.epochs import load_epoch_dataset
from data.source_records import SIX, file_sha256, prepare_source_record

QC = EOGPreprocessingPolicy("QC_eog_ols_v1", True, "record_unavailable")


def _raw():
    rng = np.random.default_rng(20261031)
    eye = rng.normal(0, 90e-6, 5120)
    eeg = rng.normal(0, 8e-6, (3, len(eye))) + np.array([[.8], [.1], [.5]]) * eye
    return mne.io.RawArray(np.vstack([eeg, eye]),
        mne.create_info(["Fz", "Cz", "Pz", "EOG"], 256, ["eeg"] * 3 + ["eog"]), verbose=False)


def _prepare(raw, **kwargs):
    return prepare_source_eog(raw, policy=QC, eeg_names=["Fz", "Cz", "Pz"], source_reference="nose",
        participant_key="GTN:fixture", session_id="s1", fit_mask=np.arange(raw.n_times) < 2500, **kwargs)


def test_source_ols_uses_only_authorized_samples_and_preserves_acquired_eog():
    raw = _raw()
    before = raw.get_data().copy()
    corrected, reference, record = _prepare(raw)
    np.testing.assert_array_equal(raw.get_data(), before)
    np.testing.assert_array_equal(corrected.get_data(picks=["EOG"]), before[[3]])
    assert reference == "measured_channel:CZ"
    assert corrected.get_channel_types() == ["eeg", "misc", "eeg", "eog"]
    raw._data[:, 2500:] += np.arange(raw.n_times - 2500) * 1e-3
    _, _, after = _prepare(raw)
    assert record["fitted_model"] == after["fitted_model"]
    assert record["classifier_EOG_features"] == 0


def test_missing_eog_is_explicit_and_does_not_synthesize_a_regressor():
    raw = _raw().pick(["Fz", "Cz", "Pz"])
    unchanged, reference, record = _prepare(raw)
    assert unchanged is raw and reference == "nose"
    assert record["status"] == "unavailable_no_recorded_EOG"
    with pytest.raises(ValueError, match="real recorded EOG"):
        prepare_source_eog(raw, policy=replace(QC, missing_eog="error"), eeg_names=raw.ch_names,
            source_reference="nose", participant_key="fixture", session_id="s1", fit_mask=np.ones(raw.n_times, bool))


def test_policy_roundtrip_cannot_silently_change_method():
    assert EOGPreprocessingPolicy.from_record(QC.record()) == QC
    with pytest.raises(ValueError, match="inconsistent"):
        EOGPreprocessingPolicy.from_record({**QC.record(), "EOG_model_feature": True})


def test_actual_source_pipeline_saves_only_eeg_with_established_cz_and_reload_receipts(tmp_path, monkeypatch):
    source = tmp_path / "fixture.bin"
    source.write_bytes(b"source input identity")
    raw = _raw()
    labels = np.tile([0, 1], 6)
    events = np.column_stack(((np.arange(len(labels)) + 3) * 256, np.zeros(len(labels), int), labels + 1))
    monkeypatch.setattr("data.source_records._read_record", lambda _item, _path, **kw: (raw.copy(), events, labels))
    item = dict(study="BI2014a", subject="1", path=source.name, sha256=file_sha256(source),
        sample_rate_hz=256, reference="nose", session="s1", run="r1")
    prepared = prepare_source_record(item, root=tmp_path, output=tmp_path / "qc", eog_policy=QC)
    data = load_epoch_dataset(prepared["cache"])
    assert data.channel_names == tuple(c.upper() for c in SIX)
    assert data.provenance["source_reference"] == "measured_channel:CZ"
    assert data.channel_mask.tolist() == [True, False, True, False, False, False]
    assert data.provenance["artifact_preprocessing_policy"] == QC.record()
    assert data.provenance["ocular_artifact_processing"]["status"] == "applied"
    assert "EOG" not in data.channel_names
    assert adapt_observed_montage(data, SIX, preserve_reference_label=True) is data
    with pytest.raises(ValueError, match="final reference"):
        adapt_observed_montage(data, SIX, reference_channel="Cz")


def test_full_qc_preserves_original_ledger_and_propagates_reference_spike_to_all_channels(tmp_path, monkeypatch):
    from data.epochs import load_preprocessing_spec
    from tests import PROJECT_ROOT

    rate = 256
    rng = np.random.default_rng(17)
    samples = 250 * rate
    raw = mne.io.RawArray(rng.normal(0, 5e-6, (3, samples)),
        mne.create_info(["Fz", "Cz", "Pz"], rate, "eeg"), verbose=False)
    raw._data[1, 45*rate] += .01
    labels = np.tile([0, 1], 20)
    seconds = np.r_[np.arange(5, 45, 2), np.arange(45, 245, 10)]
    events = np.column_stack((seconds * rate, np.zeros(len(labels), int), labels + 1))
    source = tmp_path / "source.bin"
    source.write_bytes(b"QC reference-spike counterexample")
    monkeypatch.setattr("data.source_records._read_record", lambda _item, _path, **kw: (raw.copy(), events, labels))
    item = dict(study="BI2014a", subject="1", path=source.name, sha256=file_sha256(source),
        sample_rate_hz=rate, reference="nose", session="s1", run="r1")
    policy = replace(QC, continuous_qc=True)
    assert EOGPreprocessingPolicy.from_record(policy.record()) == policy
    result = prepare_source_record(item, root=tmp_path, output=tmp_path / "full_qc", eog_policy=policy,
        preprocessing=load_preprocessing_spec(PROJECT_ROOT / "configs/brainsync_bidirectional_preprocessing.json"))
    dataset = load_epoch_dataset(result["cache"])
    assert dataset.event_timeline.n_events == len(events)
    assert 0 < dataset.n_epochs < len(events)
    processing = dataset.provenance["continuous_artifact_processing"]
    assert processing["status"] == "applied"
    assert processing["model"]["spec"]["flatline_min_duration_s"] == 5.
    assert dataset.provenance["ocular_artifact_processing"]["status"] == "unavailable_no_recorded_EOG"
    assert dataset.provenance["source_reference"] == "measured_channel:CZ"
    # A corrupted reference changes every measured difference, even with clean Fz/Pz ADC data.
    row = np.flatnonzero(dataset.event_timeline.onset_samples == 45 * rate)[0]
    assert dataset.event_timeline.evidence_indices[row] == -1
    assert set(dataset.provenance["continuous_quality_ledger"]["channel_names"]) == {"Fz", "Pz"}


def test_short_source_recovery_warmup_uses_only_explicitly_authorized_context():
    from data.epochs import load_preprocessing_spec
    from data.source_qc import continuous_parameters, scope_recovery_parameters
    from tests import PROJECT_ROOT

    raw = _raw()
    spec = load_preprocessing_spec(PROJECT_ROOT / "configs/brainsync_bidirectional_preprocessing.json")
    _, parameters = continuous_parameters(replace(QC, continuous_qc=True), 256)
    mask = np.arange(raw.n_times) < 12 * 256
    scoped = scope_recovery_parameters(raw, mask, spec, parameters,
        rule="cap_by_longest_authorized_segment_quarter")
    assert scoped["warmup_samples"] < 3 * 256
    assert parameters["warmup_samples"] == 30 * 256
    raw._data[:, ~mask] = 1e6
    assert scope_recovery_parameters(raw, mask, spec, parameters,
        rule="cap_by_longest_authorized_segment_quarter") == scoped


def test_quiet_calibration_record_cannot_collapse_a_louder_records_native_qc_band():
    from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc
    from data.source_qc import calibration_group_envelope

    spec = ContinuousQCSpec(difference_z=12., difference_floor_v=1e-7, settle_samples=25)
    models = []
    for number, scale in enumerate([1., .03, .03]):
        raw = _raw()
        raw._data *= scale
        model = fit_continuous_qc(raw, spec=spec, fit_sample_mask=np.ones(raw.n_times, bool),
            fit_provenance={"group": number})
        models.append(model)
    combined = calibration_group_envelope(models, fit_provenance={"scope": "C_only"})
    singleton = calibration_group_envelope(models[:1], fit_provenance={"scope": "one_C_record"})
    np.testing.assert_array_equal(singleton.difference_center_v, models[0].difference_center_v)
    np.testing.assert_array_equal(singleton.difference_scale_v, models[0].difference_scale_v)
    c, s = np.asarray(combined.difference_center_v), np.asarray(combined.difference_scale_v)
    for model in models:
        ci, si = np.asarray(model.difference_center_v), np.asarray(model.difference_scale_v)
        assert np.all(c-12*s <= ci-12*si + 1e-18)
        assert np.all(c+12*s >= ci+12*si - 1e-18)
    reverse = calibration_group_envelope(models[::-1], fit_provenance={"scope": "C_only"})
    np.testing.assert_array_equal(reverse.difference_scale_v, combined.difference_scale_v)
