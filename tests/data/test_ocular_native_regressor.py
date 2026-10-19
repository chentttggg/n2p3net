"""EEG rereferencing commutes with OLS only when its EOG regressor is unchanged."""

import copy
import hashlib
import json
from dataclasses import replace

import mne
import numpy as np
import pytest

from data.brainsync import (
    IndependentRecordingScope,
    load_brainsync_inference_session,
    load_brainsync_session,
)
from data.brainsync_contract import validate_brainsync_bids_session
from data.continuous_qc import ContinuousQCSpec, FittedContinuousQC, signal_dependency_matrix
from data.ocular import (
    PRESERVE_ACQUIRED_EOG,
    FittedEOGRegression,
    eog_regression_from_record,
    fit_eog_regression,
    load_eog_regression,
    save_eog_regression,
    validate_eog_calibration_budget,
)
from experiments.fit_brainsync_eog import fit_brainsync_eog_sessions, main
from research.contracts import semantic_sha256
from tests.support.brainsync import write_bids_session
from tests.support.io import read_json, write_json


def test_centered_ols_reference_difference_identity_with_unchanged_regressor():
    rng = np.random.default_rng(20260908)
    eye = rng.normal(300e-6, 50e-6, (2400, 2))
    eeg = rng.normal(120e-6, 5e-6, (2400, 3)) + eye @ np.array([[.4, -.2, .9], [.1, .5, -.3]])
    selected = np.arange(len(eye)) < 1300
    design = eye[selected] - eye[selected].mean(axis=0)
    native = np.linalg.lstsq(design, eeg[selected] - eeg[selected].mean(axis=0), rcond=None)[0]
    rereferenced = eeg[:, [0, 2]] - eeg[:, [1]]
    direct = np.linalg.lstsq(design, rereferenced[selected] - rereferenced[selected].mean(axis=0), rcond=None)[0]
    np.testing.assert_allclose(direct, native[:, [0, 2]] - native[:, [1]], atol=1e-14)
    native_cleaned = eeg - (eye - eye[selected].mean(axis=0)) @ native
    direct_cleaned = rereferenced - (eye - eye[selected].mean(axis=0)) @ direct
    np.testing.assert_allclose(direct_cleaned, native_cleaned[:, [0, 2]] - native_cleaned[:, [1]], atol=1e-18)


def _raw(seed=1):
    rng = np.random.default_rng(seed)
    eye = rng.normal(300e-6, 50e-6, 2400)
    eeg = rng.normal(120e-6, 5e-6, (3, 2400)) + np.array([[.3], [-.2], [.7]]) * eye
    return mne.io.RawArray(np.vstack([eeg, eye]),
        mne.create_info(["Fz", "Cz", "Pz", "EOG"], 250, ["eeg"] * 3 + ["eog"]), verbose=False)


def _fit(raw=None, **changes):
    kwargs = {"participant_key": "test:P001", "source_reference": "A2",
              "fit_provenance": {"role": "calibration", "calibration_session_ids": ["R2", "R4"],
                                 "calibration_end_timestamp_s": 10.},
              "reference_channel": "Cz", "eog_reference_policy": PRESERVE_ACQUIRED_EOG}
    return fit_eog_regression(_raw() if raw is None else raw, **{**kwargs, **changes})


def test_preserved_regressor_fit_and_apply_equal_native_ols_then_eeg_rereference():
    calibration, test = _raw(), _raw(2)
    mask = np.arange(calibration.n_times) < 1300
    native = _fit(calibration, reference_channel=None, fit_sample_mask=mask)
    preserved = _fit(calibration, fit_sample_mask=mask)
    native_coefficients = np.asarray(native.coefficients)
    np.testing.assert_allclose(preserved.coefficients,
        native_coefficients[[0, 2]] - native_coefficients[[1]], atol=1e-14)
    native_corrected = native.apply(test, source_reference="A2").get_data(picks=["Fz", "Cz", "Pz"])
    before = test.get_data().copy()
    actual = preserved.apply(test, source_reference="A2")
    np.testing.assert_allclose(actual.get_data(picks=["Fz", "Pz"]),
        native_corrected[[0, 2]] - native_corrected[[1]], atol=1e-18)
    np.testing.assert_array_equal(actual.get_data(picks=["EOG"]), before[[3]])
    np.testing.assert_array_equal(test.get_data(), before)
    assert preserved.eog_derivation == {}
    assert preserved.fit_provenance["shared_eeg_eog_acquisition_reference"] is None


def test_unknown_eog_is_allowed_only_by_explicit_preserved_regressor_policy():
    with pytest.raises(ValueError, match="explicit EOG derivation"):
        _fit(eog_reference_policy="shared_measured_reference")
    model = _fit()
    assert model.eog_derivation == {}
    assert model.record()["schema"] == "n2p3_calibration_eog_regression/4"
    assert model.record()["eog_reference_policy"] == PRESERVE_ACQUIRED_EOG


def test_known_bipolar_derivation_is_preserved_without_claiming_shared_reference():
    derivation = {"EOG": {"kind": "bipolar", "placement": "horizontal_pair", "reference": "bipolar_pair"}}
    model = _fit(eog_derivation=derivation)
    assert model.eog_derivation == derivation
    assert model.fit_provenance["shared_eeg_eog_acquisition_reference"] is False
    model.apply(_raw(2), source_reference="A2", eog_derivation=derivation)
    with pytest.raises(ValueError, match="shared EEG acquisition reference"):
        _fit(eog_derivation=derivation, eog_reference_policy="shared_measured_reference")
    with pytest.raises(ValueError, match="derivation"):
        model.apply(_raw(2), source_reference="A2", eog_derivation={})


def test_preserved_regressor_keeps_acquisition_reference_and_derivation_gates():
    model = _fit()
    with pytest.raises(ValueError, match="acquisition reference"):
        model.apply(_raw(2), source_reference="different_reference")
    with pytest.raises(ValueError, match="derivation"):
        model.apply(_raw(2), source_reference="A2", eog_derivation={"EOG": {"kind": "unknown"}})
    with pytest.raises(ValueError, match="requires kind"):
        _fit(eog_derivation={"EOG": {"kind": "unknown"}})


def test_preserved_ols_fit_ignores_samples_outside_declared_calibration_mask():
    raw = _raw()
    mask = np.arange(raw.n_times) < 1300
    first = _fit(raw, fit_sample_mask=mask)
    raw._data[:, 1300:] += np.arange(1100)[None, :] * 1e-3
    second = _fit(raw, fit_sample_mask=mask)
    assert first.record() == second.record()
    assert first.fit_provenance["samples"] == 1300


def test_preserved_model_roundtrip_is_exact_and_unknown_policies_fail(tmp_path):
    model = _fit()
    before = copy.deepcopy(model.record())
    path = save_eog_regression(tmp_path / "preserved.json", model)
    restored = load_eog_regression(path)
    assert restored.record() == before
    assert semantic_sha256(restored.record()) == semantic_sha256(before)
    np.testing.assert_array_equal(restored.apply(_raw(2), source_reference="A2").get_data(),
                                  model.apply(_raw(2), source_reference="A2").get_data())
    for value in ("unknown", None):
        with pytest.raises(ValueError, match="policy"):
            _fit(eog_reference_policy=value)


@pytest.mark.parametrize("change", [
    {"schema": "n2p3_calibration_eog_regression/3"},
    {"schema": "n2p3_calibration_eog_regression/5"},
    {"eog_reference_policy": "shared_measured_reference"}, {"eog_reference_policy": "unknown"},
])
def test_schema_and_reference_policy_must_agree(change):
    with pytest.raises(ValueError, match="schema"):
        eog_regression_from_record({**_fit().record(), **change})


def test_legacy_record_preserves_exact_bytes_without_new_policy_field():
    derivation = {"EOG": {"kind": "single_ended", "placement": "left_eye_corner", "reference": "A2"}}
    provenance = {"role": "golden", "acquisition_source_reference": "A2"}
    model = FittedEOGRegression(("Pz",), ("EOG",), ((.25,),), (0.,), "fixture:P001",
        "measured_channel:CZ", provenance, reference_channel="Cz", eog_derivation=derivation)
    expected = {"schema": "n2p3_calibration_eog_regression/3", "eeg_channels": ["Pz"],
        "eog_channels": ["EOG"], "coefficients": [[.25]], "calibration_eog_mean_v": [0.],
        "participant_key": "fixture:P001", "source_reference": "measured_channel:CZ",
        "fit_provenance": provenance, "application": "pointwise_subtraction_with_frozen_calibration_center",
        "signal_unit": "V", "reference_changed": True, "reference_channel": "CZ", "eog_derivation": derivation}
    assert json.dumps(model.record(), separators=(",", ":")).encode() == json.dumps(expected, separators=(",", ":")).encode()
    assert eog_regression_from_record(expected).record() == expected
    assert "eog_reference_policy" not in model.record()


def test_preserved_regressor_matrix_keeps_full_cz_dependency_even_with_beta_one():
    model = replace(_fit(), coefficients=((1.,), (1.,)))
    raw = _raw(2)
    output_channels = ("Fz", "Pz")
    matrix = signal_dependency_matrix(raw.ch_names, output_channels, ocular_model=model)
    np.testing.assert_array_equal(matrix, [[1., -1., 0., -1.], [0., -1., 1., -1.]])
    mapped = matrix @ raw.get_data() + np.asarray(model.coefficients) @ np.asarray(model.calibration_eog_mean_v)[:, None]
    np.testing.assert_allclose(model.apply(raw, source_reference="A2").get_data(picks=list(output_channels)), mapped, atol=1e-18)
    continuous = FittedContinuousQC(
        ContinuousQCSpec(12, 1., 25, channel_local=True, include_eog=True, flatline_min_duration_s=.1),
        tuple(raw.ch_names), 250., (0.,) * 4, (1.,) * 4, {"scope": "synthetic"})
    raw._data[1, 500:750] = 0.
    projected = continuous.project_result(continuous.detect(raw), matrix, output_channels)
    observed = projected.epoch_channel_mask([550], channels=output_channels, tmin_s=0., tmax_s=0.)
    assert not observed.any()


def test_unknown_derivation_bids_cli_fit_and_independent_test_roundtrip(tmp_path, monkeypatch):
    sessions = {}
    raw_hashes = {}
    for digit in (2, 3, 4, 5):
        root = write_bids_session(tmp_path / f"R{digit}", session_id=f"R{digit}", target=digit,
                                   started_utc=f"2026-09-0{digit}T00:00:00Z")
        sidecar = next(root.rglob("*_eeg.json"))
        metadata = read_json(sidecar)
        metadata.pop("EOGDerivation")
        write_json(sidecar, metadata)
        sessions[f"R{digit}"] = root
        raw_path = validate_brainsync_bids_session(root).bids.raw_path
        raw_hashes[raw_path] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    output = tmp_path / "ocular.json"
    monkeypatch.setattr("sys.argv", ["fit_brainsync_eog", "--calibration-session-dir", str(sessions["R2"]),
        "--calibration-session-dir", str(sessions["R4"]), "--reference-channel", "Cz",
        "--eog-reference-policy", PRESERVE_ACQUIRED_EOG, "--output", str(output)])
    main()
    model = load_eog_regression(output)
    assert model.eog_derivation == {}
    assert model.fit_provenance["calibration_session_ids"] == ["R2", "R4"]
    with pytest.raises(ValueError, match="future data"):
        load_brainsync_session(sessions["R3"], ocular_model=model)
    scope = IndependentRecordingScope(("R2", "R4"), ("R3", "R5"))
    labelled = load_brainsync_session(sessions["R3"], ocular_model=model, recording_scope=scope)
    unlabelled = load_brainsync_inference_session(sessions["R3"], ocular_model=model, recording_scope=scope)
    np.testing.assert_array_equal(labelled.X, unlabelled.X)
    assert unlabelled.y is None and labelled.n_channels == 6
    native_model = fit_brainsync_eog_sessions([sessions["R2"], sessions["R4"]])
    native = load_brainsync_session(sessions["R3"], ocular_model=native_model, recording_scope=scope)
    cz = native.channel_names.index("CZ")
    indices = [native.channel_names.index(name) for name in labelled.channel_names]
    np.testing.assert_allclose(labelled.X, native.X[:, indices] - native.X[:, [cz]], rtol=1e-5, atol=1e-10)
    processing = labelled.provenance["ocular_artifact_processing"]
    validate_eog_calibration_budget(processing, {model.participant_key: ("R2", "R4")})
    with pytest.raises(ValueError, match="outside.*calibration budget"):
        validate_eog_calibration_budget(processing, {model.participant_key: ("R2",)})
    assert model.record() == load_eog_regression(output).record()
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in raw_hashes.items())
