"""Exact raw sample repetition is distinct from low physiological amplitude."""

import copy
import json
from dataclasses import replace

import mne
import numpy as np
import pytest
from scipy.signal import butter

from data.continuous_qc import (
    ContinuousQCSpec,
    FittedContinuousQC,
    continuous_qc_from_record,
    signal_dependency_matrix,
)
from data.contract import CAUSAL_IIR_INITIAL_STATE, SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from data.preprocess import filter_continuous
from data.recovery import EpochStateRecovery
from research.contracts import semantic_sha256

CHANNELS = ("Fz", "Cz", "Pz")
RATE = 250.0


BANDPASS_PREPROCESSING = preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)

def _raw(count=4000):
    time = np.arange(count) / RATE
    values = np.array([np.sin(2 * np.pi * frequency * time + .2) * 5e-6
                       for frequency in (3.1, 4.7, 8.3)])
    return mne.io.RawArray(values, mne.create_info(CHANNELS, RATE, "eeg"), verbose=False)


def _model(duration=5.0):
    # Large difference bounds isolate exact repetition from the existing gates.
    return FittedContinuousQC(
        ContinuousQCSpec(12, 1., 25, channel_local=True, flatline_min_duration_s=duration),
        CHANNELS, RATE, (0., 0., 0.), (1., 1., 1.), {"scope": "synthetic_raw_constant_control"},
    )


def _constant_rows(result):
    return [row for row in result.intervals if row.reason == "raw_exact_constant"]


def test_disabled_feature_retains_legacy_record_bytes_and_digest():
    expected = {
        "schema": "n2p3_continuous_qc/3",
        "spec": {"difference_z": 12, "difference_floor_v": 1., "settle_samples": 25,
                 "channel_local": True, "reference_channel": None, "include_eog": False,
                 "minimum_channel_fraction": 1., "range_window_samples": None,
                 "range_quantile": .995, "range_inflation": 2., "range_floor_v": 50e-6,
                 "artifact_pre_guard_samples": 0},
        "channel_names": list(CHANNELS), "sample_rate_hz": RATE,
        "difference_center_v": [0., 0., 0.], "difference_scale_v": [1., 1., 1.],
        "fit_provenance": {"scope": "synthetic_raw_constant_control"},
        "recovery_policy": "reject_unbounded_state_error", "recovery": None,
        "epoch_recovery": None, "range_threshold_v": None,
        "detection_lookahead_samples": 0, "signal_unit": "V",
    }
    model = _model(None)
    actual = model.record()
    assert json.dumps(actual, separators=(",", ":")).encode() == json.dumps(expected, separators=(",", ":")).encode()
    assert semantic_sha256(actual) == semantic_sha256(expected)
    assert continuous_qc_from_record(expected).record() == expected
    raw = _raw()
    raw._data[2, 500:2000] = 0
    assert model.detect(raw).intervals == ()


def test_enabled_feature_has_explicit_v4_record_and_detection_latency():
    model = _model()
    record = model.record()
    assert record["schema"] == "n2p3_continuous_qc/4"
    assert record["spec"]["flatline_min_duration_s"] == 5.0
    assert record["detection_lookahead_samples"] == 1250
    assert continuous_qc_from_record(record).record() == record
    assert semantic_sha256(_model(None).record()) != semantic_sha256(record)


@pytest.mark.parametrize("change", [
    {"schema": "n2p3_continuous_qc/2"}, {"schema": "n2p3_continuous_qc/5"},
    {"schema": "n2p3_continuous_qc/3"},
])
def test_unknown_or_mismatched_schema_is_rejected(change):
    with pytest.raises(ValueError, match="schema"):
        continuous_qc_from_record({**_model().record(), **change})


@pytest.mark.parametrize("remove", [True, False])
def test_v4_cannot_silently_disable_exact_constant_detection(remove):
    record = copy.deepcopy(_model().record())
    if remove:
        record["spec"].pop("flatline_min_duration_s")
    else:
        record["spec"]["flatline_min_duration_s"] = None
    with pytest.raises(ValueError, match="schema /4"):
        continuous_qc_from_record(record)


@pytest.mark.parametrize("duration", [0., -1., np.nan, np.inf, True, "5"])
def test_duration_must_be_explicit_finite_positive_number(duration):
    with pytest.raises(ValueError, match="exact-constant"):
        _model(duration)


def test_exact_constant_requires_local_channel_scope():
    with pytest.raises(ValueError, match="channel-local"):
        ContinuousQCSpec(12, 1e-7, 25, flatline_min_duration_s=5.)


@pytest.mark.parametrize("run_samples", [1250, 1251, 1252])
def test_declared_duration_uses_first_to_last_observed_sample(run_samples):
    raw = _raw()
    raw._data[2, 500:500 + run_samples] = 7e-6
    original = raw.get_data().copy()
    result = _model().detect(raw)
    rows = _constant_rows(result)
    assert len(rows) == int(run_samples >= 1251)
    if rows:
        row = rows[0]
        assert (row.start_sample, row.end_sample) == (500, 500 + run_samples)
        assert row.detected_sample == 1750
        assert row.confirmed_sample == 500 + run_samples
        assert row.channels == ("Pz",) and row.reset_filter
        assert result.record()["detection"] == "raw_anomaly_with_backdated_support_complete_recording"
    np.testing.assert_array_equal(raw.get_data(), original)


def test_fractional_sample_duration_rounds_up():
    model = _model(.0040001)
    assert model.flatline_min_samples == 3
    raw = _raw()
    raw._data[2, 500:502] = 0.
    assert not _constant_rows(model.detect(raw))
    raw._data[2, 502] = 0.
    assert _constant_rows(model.detect(raw))[0].detected_sample == 502


def test_finite_low_amplitude_dynamic_signal_is_never_called_exact_constant():
    raw = _raw()
    raw._data *= 1e-12
    assert np.max(np.std(raw.get_data(), axis=1)) < 1e-15
    assert not _constant_rows(_model().detect(raw))


@pytest.mark.parametrize("constant", [0., 1e-15, .03])
def test_constant_detection_does_not_depend_on_voltage_amplitude(constant):
    raw = _raw()
    raw._data[2, 500:2000] = constant
    rows = _constant_rows(_model().detect(raw))
    assert len(rows) == 1 and (rows[0].start_sample, rows[0].end_sample) == (500, 2000)


@pytest.mark.parametrize("description,duration", [("EDGE boundary", 0.), ("BAD_acq_skip", .4)])
def test_subthreshold_runs_cannot_accumulate_across_acquisition_boundary(description, duration):
    raw = _raw()
    raw._data[2, 1000:2600] = 0.
    raw.annotations.append(1800 / RATE, duration, description)
    assert not _constant_rows(_model().detect(raw))


def test_qualified_runs_stay_separate_across_boundary_and_recovery_is_bounded():
    raw = _raw(6000)
    raw._data[2, 100:3300] = 0.
    raw.annotations.append(1700 / RATE, 0., "EDGE boundary")
    result = _model().detect(raw)
    rows = _constant_rows(result)
    assert [(row.start_sample, row.end_sample) for row in rows] == [(100, 1700), (1700, 3300)]
    assert rows[0].confirmed_sample == 1699
    assert all(row.end_sample <= 1700 for row in result.intervals if row.start_sample < 1700)


@pytest.mark.parametrize("bad_channel,expected", [("Cz", (False, False)), ("Pz", (True, False))])
def test_reference_dependency_propagates_exact_constant_only_to_affected_outputs(bad_channel, expected):
    raw = _raw()
    raw._data[CHANNELS.index(bad_channel), 500:2000] = 0.
    model = _model()
    native = model.detect(raw)
    output_channels = ("Fz", "Pz")
    mixing = signal_dependency_matrix(CHANNELS, output_channels, reference_channel="Cz")
    projected = model.project_result(native, mixing, output_channels)
    observed = projected.epoch_channel_mask([1000], channels=output_channels, tmin_s=0., tmax_s=0.)
    assert tuple(observed[0]) == expected
    assert {row.channels[0] for row in _constant_rows(projected)} == {
        name for name, usable in zip(output_channels, expected, strict=True) if not usable
    }
    assert all(row.reason == "filter_state_uncertainty_unbounded"
               for row in projected.intervals if not row.reset_filter)


def test_empirical_recovery_applies_after_constant_and_scales_with_reference_dependency():
    raw = _raw(15000)
    raw._data[1, 500:2000] = 0.
    scope = {"participant_key": "synthetic", "source_reference": "A2",
             "calibration_session_ids": ["cal"], "calibration_end_timestamp_s": 1.}
    recovery = EpochStateRecovery.from_envelope(
        butter(4, [.1, 30], btype="bandpass", fs=RATE, output="sos"), RATE,
        BANDPASS_PREPROCESSING, CHANNELS, np.full((3, 8), 1e-5),
        tolerance_v=1e-6, fit_provenance=scope,
    )
    model = replace(_model(), fit_provenance=scope, epoch_recovery=recovery)
    mixing = signal_dependency_matrix(CHANNELS, ("Fz", "Pz"), reference_channel="Cz")
    projected = model.project_result(model.detect(raw), mixing, ("Fz", "Pz"))
    waits = recovery.project(mixing, ("Fz", "Pz")).recovery_samples()
    rows = [row for row in projected.intervals if not row.reset_filter]
    assert len(rows) == 2
    for row, wait in zip(rows, waits, strict=True):
        assert row.reason == "empirical_epoch_state_recovery"
        assert row.start_sample == 2000 and row.end_sample == min(15000, 2000 + wait)


def test_local_annotation_resets_only_constant_channel_and_preserves_raw_values():
    raw = _raw()
    raw._data[2, 500:2000] = 0.
    annotated = _model().detect(raw).annotate(raw)
    np.testing.assert_array_equal(raw.get_data(), annotated.get_data())
    options = {"l_freq": .1, "h_freq": 30, "phase": "forward", "copy": True,
               "causal_iir_initial_state": CAUSAL_IIR_INITIAL_STATE}
    original = filter_continuous(raw, **options).get_data()
    actual = filter_continuous(annotated, **options).get_data()
    np.testing.assert_array_equal(actual[:2], original[:2])
    tail = mne.io.RawArray(raw.get_data()[[2], 2000:], mne.create_info(["Pz"], RATE, "eeg"), verbose=False)
    np.testing.assert_array_equal(actual[2, 2000:], filter_continuous(tail, **options).get_data()[0])


def test_overlapping_jump_and_constant_keep_reasons_and_recover_after_union():
    raw = _raw()
    raw._data[2, 500:2000] = .03
    model = replace(_model(), difference_scale_v=(1e-7, 1e-7, 1e-7))
    result = model.detect(raw)
    rows = _constant_rows(result)
    assert len(rows) == 1 and (rows[0].start_sample, rows[0].end_sample) == (500, 2000)
    assert any(row.reset_filter and row.reason == "raw_dependency_outlier" for row in result.intervals)
    recoveries = [row for row in result.intervals if row.channels == ("Pz",) and not row.reset_filter]
    assert len(recoveries) == 1 and recoveries[0].start_sample == 2026
    observed = result.epoch_channel_mask([1999, 2000, 2025, 2026],
        channels=CHANNELS, tmin_s=0., tmax_s=0.)
    assert not observed[:, 2].any()
    assert observed[:, :2].all()
