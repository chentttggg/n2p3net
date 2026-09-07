from __future__ import annotations

import mne
import numpy as np
import pytest

from data.ocular import (
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


def test_calibration_regression_reduces_known_artifact_and_preserves_eog(tmp_path):
    train, test = _raw(1), _raw(2)
    model = fit_eog_regression(
        train,
        participant_key="test-authority:P001",
        source_reference="A2",
        fit_provenance={"role": "calibration", "source_hash": "fixture"},
    )
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


def test_future_eog_changes_cannot_change_earlier_correction():
    model = fit_eog_regression(
        _raw(1),
        participant_key="test-authority:P001",
        source_reference="A2",
        fit_provenance={"role": "calibration"},
    )
    test = _raw(2)
    changed = test.copy()
    changed._data[2, 1000:] += 0.1
    first = model.apply(test, source_reference="A2").get_data()
    second = model.apply(changed, source_reference="A2").get_data()
    np.testing.assert_array_equal(first[:, :1000], second[:, :1000])


def test_embedded_eog_record_reuses_frozen_fit_and_rejects_changed_units():
    model = fit_eog_regression(
        _raw(1),
        participant_key="test-authority:P001",
        source_reference="A2",
        fit_provenance={"role": "calibration"},
    )
    restored = eog_regression_from_record(model.record())
    np.testing.assert_array_equal(
        model.apply(_raw(2), source_reference="A2").get_data(),
        restored.apply(_raw(2), source_reference="A2").get_data(),
    )
    bad_record = model.record()
    bad_record["signal_unit"] = "uV"
    with pytest.raises(ValueError, match="unsupported or inconsistent"):
        eog_regression_from_record(bad_record)


def test_flat_eog_and_reference_mismatch_are_not_silent_success():
    raw = _raw()
    model = fit_eog_regression(
        raw,
        participant_key="test-authority:P001",
        source_reference="A2",
        fit_provenance={"role": "calibration"},
    )
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
    )
    b = fit_eog_regression(
        second,
        participant_key="test-authority:P001",
        source_reference="T7",
        fit_provenance={"role": "calibration"},
        reference_channel="Cz",
    )
    assert a.eeg_channels == ("Pz",)
    assert a.source_reference == b.source_reference == "measured_channel:CZ"
    np.testing.assert_allclose(a.coefficients, b.coefficients, atol=1e-12)
    np.testing.assert_allclose(
        a.apply(first, source_reference="A2").get_data(),
        b.apply(second, source_reference="T7").get_data(),
        atol=1e-12,
    )
