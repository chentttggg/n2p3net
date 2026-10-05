from __future__ import annotations

from dataclasses import replace

import mne
import numpy as np
import pytest

from data.contract import BRAIN_SYNC_CAUSAL_DATA_CONTRACT
from data.dataset import build_subject
from data.epochs import preprocessing_spec_from_contract
from data.ocular import (
    PRESERVE_ACQUIRED_EOG,
    eog_regression_from_record,
    fit_eog_regression,
    load_eog_regression,
    save_eog_regression,
)


def _raw(seed=42, samples=2000):
    rng = np.random.default_rng(seed)
    eye = rng.normal(0.0, 100e-6, samples)
    neural = rng.normal(0.0, 2e-6, (2, samples))
    values = np.vstack([neural + np.array([[0.3], [-0.7]]) * eye, eye])
    return mne.io.RawArray(
        values, mne.create_info(["Cz", "Pz", "EOG"], 250.0, ["eeg", "eeg", "eog"]), verbose=False
    )


@pytest.fixture(scope="module")
def fitted_ocular():
    """One immutable calibration fit for application and persistence contracts."""
    model = fit_eog_regression(_raw(1), participant_key="test-authority:P001", source_reference="A2",
                               fit_provenance={"role": "calibration", "source_hash": "fixture"})
    before = model.record()
    yield model
    assert model.record() == before


def test_calibration_regression_reduces_known_artifact_and_preserves_eog(tmp_path, fitted_ocular):
    test = _raw(2)
    model = fitted_ocular
    before = test.get_data()
    corrected = model.apply(test, source_reference="A2")
    assert np.std(corrected.get_data()[:2]) < 0.1 * np.std(before[:2])
    np.testing.assert_array_equal(corrected.get_data()[2], before[2])
    np.testing.assert_array_equal(test.get_data(), before)
    path = save_eog_regression(tmp_path / "ocular.json", model)
    np.testing.assert_array_equal(
        load_eog_regression(path).apply(test, source_reference="A2").get_data(),
        corrected.get_data(),
    )


def test_future_eog_changes_cannot_change_earlier_correction(fitted_ocular):
    model = fitted_ocular
    test = _raw(2)
    changed = test.copy()
    changed._data[2, 1000:] += 0.1
    first = model.apply(test, source_reference="A2").get_data()
    second = model.apply(changed, source_reference="A2").get_data()
    np.testing.assert_array_equal(first[:, :1000], second[:, :1000])


def test_correction_strength_preserves_fit_and_scales_actual_dependency(tmp_path, fitted_ocular):
    from data.continuous_qc import signal_dependency_matrix

    raw = _raw(1)
    fitted = fitted_ocular
    half = replace(fitted, correction_strength=.5)
    zero = replace(fitted, correction_strength=0.)
    original = raw.get_data()
    complete = fitted.apply(raw, source_reference="A2").get_data()
    np.testing.assert_array_equal(zero.apply(raw, source_reference="A2").get_data(), original)
    np.testing.assert_allclose(half.apply(raw, source_reference="A2").get_data(), (original + complete)/2, atol=1e-18)
    assert half.coefficients == fitted.coefficients and half.fit_provenance == fitted.fit_provenance
    restored = load_eog_regression(save_eog_regression(tmp_path/"half.json", half))
    assert restored.record() == half.record()
    full_map = signal_dependency_matrix(raw.ch_names, fitted.eeg_channels, ocular_model=fitted)
    half_map = signal_dependency_matrix(raw.ch_names, fitted.eeg_channels, ocular_model=half)
    zero_map = signal_dependency_matrix(raw.ch_names, fitted.eeg_channels, ocular_model=zero)
    np.testing.assert_allclose(half_map[:, -1], full_map[:, -1]/2)
    np.testing.assert_array_equal(zero_map[:, -1], 0)
    for strength in (-1, 1.1, float("nan"), True):
        with pytest.raises(ValueError, match="strength"):
            replace(fitted, correction_strength=strength)


def test_embedded_eog_record_reuses_frozen_fit_and_rejects_changed_units(fitted_ocular):
    model = fitted_ocular
    restored = eog_regression_from_record(model.record())
    np.testing.assert_array_equal(
        model.apply(_raw(2), source_reference="A2").get_data(),
        restored.apply(_raw(2), source_reference="A2").get_data(),
    )
    bad_record = model.record()
    bad_record["signal_unit"] = "uV"
    with pytest.raises(ValueError, match="unsupported or inconsistent"):
        eog_regression_from_record(bad_record)


def test_flat_eog_and_reference_mismatch_are_not_silent_success(fitted_ocular):
    raw = _raw()
    model = fitted_ocular
    with pytest.raises(ValueError, match="references differ"):
        model.apply(raw, source_reference="average")
    raw._data[2] = 0.0
    with pytest.raises(ValueError, match="flat"):
        fit_eog_regression(
            raw,
            participant_key="test-authority:P001",
            source_reference="A2",
            fit_provenance={"role": "calibration"},
        )


def test_common_reference_precedes_regression_for_eeg_and_eog():
    first = _raw(1)
    second = first.copy()
    wave = np.sin(np.arange(first.n_times) / 20.0) * 100e-6
    second._data += wave[None, :]
    a = fit_eog_regression(
        first,
        participant_key="test-authority:P001",
        source_reference="A2",
        fit_provenance={"role": "calibration"},
        reference_channel="Cz",
        eog_derivation={"EOG": {"kind": "single_ended", "placement": "left_eye_corner", "reference": "A2"}},
    )
    b = fit_eog_regression(
        second,
        participant_key="test-authority:P001",
        source_reference="T7",
        fit_provenance={"role": "calibration"},
        reference_channel="Cz",
        eog_derivation={"EOG": {"kind": "single_ended", "placement": "left_eye_corner", "reference": "T7"}},
    )
    assert a.eeg_channels == ("Pz",)
    assert a.source_reference == b.source_reference == "measured_channel:CZ"
    np.testing.assert_allclose(a.coefficients, b.coefficients, atol=1e-12)
    np.testing.assert_allclose(
        a.apply(first, source_reference="A2", eog_derivation=a.eog_derivation).get_data(),
        b.apply(second, source_reference="T7", eog_derivation=b.eog_derivation).get_data(),
        atol=1e-12,
    )


def test_masked_ocular_fit_ignores_contaminated_samples_and_binds_heog():
    raw = _raw(1)
    mask = np.arange(raw.n_times) < 1000
    kwargs = dict(participant_key="test:P001", source_reference="A2",
                  fit_provenance={"role": "calibration"}, fit_sample_mask=mask,
                  eog_derivation={"EOG": {"kind": "single_ended", "placement": "left_eye_corner", "reference": "A2"}})
    first = fit_eog_regression(raw, **kwargs)
    raw._data[:, 1000:] += np.arange(1000)[None, :]
    second = fit_eog_regression(raw, **kwargs)
    assert first.record() == second.record()
    assert first.fit_provenance["samples"] == 1000
    assert first.eog_derivation["EOG"]["placement"] == "left_eye_corner"


def test_ocular_rereference_requires_shared_reference_declaration():
    with pytest.raises(ValueError, match="derivation"):
        fit_eog_regression(_raw(), participant_key="test:P001", source_reference="A2",
                           reference_channel="Cz", fit_provenance={"role": "calibration"})


def test_causal_notch_pipeline_independent_and_neural_correlated_eog(tmp_path):
    rate = 250
    times = np.arange(rate * 20) / rate
    eye = 20e-6 * np.sin(2 * np.pi * 3 * times)
    neural = 3e-6 * np.cos(2 * np.pi * 2 * times)
    reference = 1e-6 * np.sin(2 * np.pi * .7 * times)
    info = mne.create_info(["Fz", "Cz", "EOG"], rate, ["eeg", "eeg", "eog"])
    def raw(signal, eog):
        return mne.io.RawArray(np.stack([signal + reference, reference, eog]), info, verbose=False)
    model = fit_eog_regression(raw(neural + 2 * eye, eye), participant_key="toy/p", source_reference="left earlobe",
        reference_channel="Cz", eog_reference_policy=PRESERVE_ACQUIRED_EOG, fit_provenance={"role": "calibration"})
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT)
    events = np.column_stack([np.arange(2, 18) * rate, np.zeros(16, int), np.arange(16) % 2 + 1])
    def epochs(value):
        return build_subject(value, events, np.arange(16) % 2, subject_id="p",
                             channels=["Fz"], preprocessing=spec).data
    truth = epochs(raw(neural, eye).set_eeg_reference(["Cz"], verbose=False))
    processed = {}
    for alpha in (0., .5, 1.):
        fitted = replace(model, correction_strength=alpha)
        restored = load_eog_regression(save_eog_regression(tmp_path / f"alpha{alpha}.json", fitted))
        processed[alpha] = epochs(restored.apply(raw(neural + 2 * eye, eye), source_reference="left earlobe"))
    np.testing.assert_allclose(processed[1.], truth, atol=1e-10, rtol=1e-5)
    np.testing.assert_allclose(processed[.5], (processed[0.] + processed[1.]) / 2, atol=1e-10, rtol=1e-5)
    correlated = epochs(model.apply(raw(neural + 2 * eye, eye + .4 * neural), source_reference="left earlobe"))
    # Same beta removes 80% of neural ERP when that component also contaminates EOG.
    np.testing.assert_allclose(correlated, .2 * truth, atol=1e-10, rtol=1e-5)

