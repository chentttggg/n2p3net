"""Continuous contamination must not propagate through excluded intervals."""

import mne
import numpy as np
import pytest

from data.contract import CAUSAL_IIR_INITIAL_STATE
from data.preprocess import filter_continuous


def _raw(values, sfreq=250.0):
    return mne.io.RawArray(values[None, :], mne.create_info(["Cz"], sfreq, "eeg"), verbose=False)


def _filter(raw, phase="forward"):
    return filter_continuous(
        raw, l_freq=0.1, h_freq=30.0, phase=phase, copy=True,
        causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE if phase == "forward" else "not_applicable",
    ).get_data()[0]


@pytest.mark.parametrize("description", ["BAD_acq_skip", "BAD_continuous_artifact", "EDGE gap"])
@pytest.mark.parametrize("phase", ["forward", "zero"])
def test_full_artifact_interval_is_excluded_from_filter_state(description, phase):
    sfreq = 250
    times = np.arange(12 * sfreq) / sfreq
    clean = np.sin(2*np.pi*3*times) * 5e-6
    values = clean.copy()
    values[4*sfreq:6*sfreq] = .02
    raw = _raw(values)
    raw.set_annotations(mne.Annotations([4.0], [2.0], [description]))
    actual = _filter(raw, phase)
    assert np.array_equal(actual[:4*sfreq], _filter(_raw(clean[:4*sfreq]), phase))
    assert np.array_equal(actual[6*sfreq:], _filter(_raw(clean[6*sfreq:]), phase))


@pytest.mark.parametrize("description", ["BAD_continuous_recovery", "BAD_brainsync_rest"])
def test_epoch_exclusion_does_not_reset_continuous_filter_state(description):
    values = np.sin(np.arange(2500) / 50) * 1e-5
    raw = _raw(values)
    raw.set_annotations(mne.Annotations([4], [2], [description]))
    assert np.array_equal(_filter(raw), _filter(_raw(values)))


def test_local_artifact_never_changes_an_unaffected_channels_filter_state():
    time = np.arange(6000) / 250
    values = np.vstack([np.sin(time * 8) * 2e-6, np.sin(time * 7) * 3e-6])
    raw = mne.io.RawArray(values, mne.create_info(["Cz", "Pz"], 250, "eeg"), verbose=False)
    expected = _filter(raw)
    changed = raw.copy()
    changed._data[1, 1500:1550] += .02
    changed.annotations.append(6., .2, "QUALITY_LOCAL_ARTIFACT", ch_names=[("Pz",)])
    actual = _filter(changed)
    np.testing.assert_array_equal(actual, expected)
    all_values = filter_continuous(changed, .1, 30, phase="forward", copy=True,
        causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE).get_data()
    np.testing.assert_allclose(all_values[1, 1550:], _filter(_raw(values[1, 1550:])), atol=1e-15)


def test_frozen_difference_detector_has_explicit_fit_scope_and_unknown_recovery():
    from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc

    calibration = _raw(np.sin(np.arange(1000) / 20) * 1e-6)
    selected = np.arange(1000) < 500
    spec = ContinuousQCSpec(difference_z=12, difference_floor_v=1e-7, settle_samples=10)
    fitted = fit_continuous_qc(calibration, spec=spec, fit_sample_mask=selected,
                               fit_provenance={"role": "calibration", "source": "synthetic"})
    changed = calibration.copy()
    changed._data[:, 500:] = 100
    same = fit_continuous_qc(changed, spec=spec, fit_sample_mask=selected,
                             fit_provenance={"role": "calibration", "source": "synthetic"})
    assert fitted.record() == same.record()
    contaminated = calibration.copy()
    contaminated._data[:, 300:] += .0142
    result = fitted.detect(contaminated)
    assert result.intervals[0].start_sample == 300
    assert result.intervals[0].reason == "raw_difference_outlier"
    assert not result.valid_sample_mask[300:].any()
    assert result.intervals[-1].reason == "filter_state_uncertainty_unbounded"
    assert result.intervals[-1].end_sample == 1000
    assert np.array_equal(contaminated.get_data(), result.annotate(contaminated).get_data())


def test_state_recovery_bound_covers_later_rebounds_and_requires_evidence():
    from scipy.signal import butter, sosfilt

    from data.continuous_qc import FilterStateRecovery

    sos = butter(4, [.1, 30], fs=250, btype="bandpass", output="sos")
    bounds = np.ones((len(sos), 2)) * .0142
    recovery = FilterStateRecovery(tuple(map(tuple, sos)), tuple(map(tuple, bounds)),
                                   1e-6, 5.0, "synthetic_known_state_error")
    count = recovery.required_samples()
    assert count is not None and count > 0
    for seed in range(5):
        state = np.random.default_rng(seed).uniform(-bounds, bounds)
        tail, _ = sosfilt(sos, np.zeros(count + 10000), zi=state)
        assert 5 * np.abs(tail[count:]).max() <= 1e-6
    with pytest.raises(ValueError, match="evidence"):
        FilterStateRecovery(tuple(map(tuple, sos)), tuple(map(tuple, bounds)), 1e-6, 5, "")


def test_range_calibration_never_crosses_a_declared_acquisition_boundary():
    from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc

    raw = _raw(np.r_[np.zeros(1000), np.full(1000, .01)])
    raw.annotations.append(4., 0., "EDGE boundary")
    model = fit_continuous_qc(raw,
        spec=ContinuousQCSpec(12, 1e-7, 25, channel_local=True, range_window_samples=351),
        fit_sample_mask=np.ones(raw.n_times, dtype=bool), fit_provenance={"scope": "two_flat_segments"})
    np.testing.assert_array_equal(model.range_threshold_v, [50e-6])


def test_detected_onset_guard_is_local_segment_bounded_and_preserves_confirmation():
    from dataclasses import replace

    from data.continuous_qc import ContinuousQCSpec, continuous_qc_from_record, fit_continuous_qc

    raw = _raw(np.zeros(2000))
    raw.annotations.append(4., 0., "EDGE boundary")
    fitted = fit_continuous_qc(raw,
        spec=ContinuousQCSpec(12, 1e-7, 25, channel_local=True, artifact_pre_guard_samples=350),
        fit_sample_mask=np.ones(raw.n_times, dtype=bool), fit_provenance={"scope": "synthetic"})
    raw._data[:, 1100:1120] += .01
    guarded = fitted.detect(raw)
    unguarded = replace(fitted, spec=replace(fitted.spec, artifact_pre_guard_samples=0)).detect(raw)
    left = next(row for row in guarded.intervals if row.reset_filter)
    right = next(row for row in unguarded.intervals if row.reset_filter)
    assert left.start_sample == 1000 and right.start_sample == 1100
    assert left.end_sample == right.end_sample == 1146
    assert left.detected_sample == right.detected_sample == 1100
    assert left.confirmed_sample == right.confirmed_sample
    assert guarded.detection_lookahead_samples == 350
    assert continuous_qc_from_record(fitted.record()).record() == fitted.record()
    with pytest.raises(ValueError, match="pre-guard"):
        ContinuousQCSpec(12, 1e-7, 25, artifact_pre_guard_samples=1)
    with pytest.raises(ValueError, match="pre-guard"):
        ContinuousQCSpec(12, 1e-7, 25, channel_local=True, artifact_pre_guard_samples=-1)
