"""Slow-trend diagnostics report observable structure, not artifact truth."""

import json

import mne
import numpy as np
import pytest
from scipy.stats import theilslopes

from data.slow_trend import _window_metrics, analyze_slow_trends


def _raw(values, sfreq=100, *, first_samp=0):
    values = np.atleast_2d(values)
    names = ["Cz", "Pz", "EOG"][:len(values)]
    types = ["eeg", "eeg", "eog"][:len(values)]
    return mne.io.RawArray(values, mne.create_info(names, sfreq, types),
                           first_samp=first_samp, verbose=False)


def test_vectorized_slopes_and_residuals_match_scipy_with_outliers():
    rng = np.random.default_rng(419)
    bins = rng.normal(size=1050) * 5e-6 + np.arange(1050) * 1e-7
    bins[[3, 25, 501]] += .01
    for width in (5, 15, 30):
        actual = _window_metrics(bins, width, .75)
        time = np.arange(width) * .75
        for index in (0, 4, 487, 511, 512, len(actual) - 1):
            values = bins[index:index + width]
            expected = theilslopes(values, time, method="joint")
            residual = values - expected.slope * time - expected.intercept
            np.testing.assert_allclose(actual[index],
                [expected.slope, np.median(np.abs(residual)), np.ptp(values)], rtol=1e-12, atol=1e-15)


def test_linear_ramp_has_physical_slope_and_complete_trailing_support():
    time = np.arange(1200) / 100
    raw = _raw(7e-6 * time + .03)
    before = raw.get_data().copy()
    result = analyze_slow_trends(raw, window_seconds=(5,))
    assert result["fit_performed"] is result["labels_used"] is result["mask_changes"] is False
    assert result["artifact_classification"] is False
    assert len(result["windows"]) == 8
    for row in result["windows"]:
        assert row["end_sample"] - row["start_sample"] == row["supported_samples"] == 500
        assert row["available_sample"] == row["end_sample"] - 1
        assert row["slope_v_per_s"] == pytest.approx(7e-6, abs=1e-17)
        assert row["residual_mad_v"] < 1e-16
    assert result["windows"][4]["slope_change_v_per_s"] is None
    assert result["windows"][5]["comparison_start_sample"] == 0
    assert result["windows"][5]["change_supported_samples"] == 1000
    assert result["windows"][5]["slope_change_v_per_s"] == pytest.approx(0, abs=1e-17)
    np.testing.assert_array_equal(raw.get_data(), before)
    assert len(raw.annotations) == 0
    json.dumps(result, allow_nan=False)


def test_change_feature_observes_ramp_onset_and_is_affine_trend_invariant():
    time = np.arange(2500) / 100
    values = 6e-6 * np.maximum(time - 10, 0)
    original = analyze_slow_trends(_raw(values), window_seconds=(5,))
    shifted = analyze_slow_trends(_raw(values + .02 - 40e-6 * time), window_seconds=(5,))
    original_changes = [row["slope_change_v_per_s"] for row in original["windows"] if row["slope_change_v_per_s"] is not None]
    shifted_changes = [row["slope_change_v_per_s"] for row in shifted["windows"] if row["slope_change_v_per_s"] is not None]
    assert max(original_changes) == pytest.approx(6e-6)
    np.testing.assert_allclose(original_changes, shifted_changes, atol=1e-17)


def test_added_ramp_can_cancel_a_larger_existing_trend():
    time = np.arange(3000) / 100
    baseline = analyze_slow_trends(_raw(-7e-6 * time), window_seconds=(30,))
    injected = analyze_slow_trends(_raw((-7e-6 + 200e-6 / 30) * time), window_seconds=(30,))
    before, after = baseline["windows"][0], injected["windows"][0]
    assert abs(after["slope_v_per_s"]) < abs(before["slope_v_per_s"])
    assert after["bin_ptp_v"] < before["bin_ptp_v"]
    assert injected["artifact_classification"] is False


def test_identical_neural_and_electrode_slow_waveforms_have_identical_diagnostics():
    time = np.arange(9000) / 100
    low_frequency_neural = 40e-6 * np.sin(2 * np.pi * .03 * time)
    wide_erp = 25e-6 * np.exp(-.5 * ((time - 40) / 2.5) ** 2)
    neural = _raw(low_frequency_neural + wide_erp)
    same_observed_voltage_from_another_cause = neural.copy()
    result = analyze_slow_trends(neural)
    assert result == analyze_slow_trends(same_observed_voltage_from_another_cause)
    assert result["channel_summaries"][0]["scales"][0]["max_abs_slope_v_per_s"] > 0
    assert result["mask_changes"] is False
    assert "neural activity" in result["interpretation"]


def test_gaps_restart_bins_and_zero_duration_boundaries_break_change_support():
    time = np.arange(2000) / 100
    raw = _raw(time * 3e-6)
    raw.annotations.append([6.25, 15], [1.5, 0], ["BAD_acq_skip", "EDGE boundary"])
    result = analyze_slow_trends(raw, window_seconds=(5,))
    assert [(row["start_sample"], row["end_sample"]) for row in result["segments"]] == [
        (0, 625), (775, 1500), (1500, 2000)]
    assert [row["unbinned_tail_samples"] for row in result["segments"]] == [25, 25, 0]
    for row in result["windows"]:
        assert any(start <= row["start_sample"] < row["end_sample"] <= stop
                   for start, stop in [(0, 625), (775, 1500), (1500, 2000)])
        assert row["comparison_start_sample"] is None
    assert result["channel_summaries"][0]["excluded_annotation_samples"] == 150


def test_nonfinite_samples_and_local_artifacts_only_split_the_affected_channel():
    values = np.tile(np.arange(2000) * 1e-8, (2, 1))
    values[0, 650:700] = np.nan
    values[0, 1400] = np.inf
    raw = _raw(values)
    raw.annotations.append(4, 2, "QUALITY_LOCAL_ARTIFACT", ch_names=[("Cz",)])
    together = analyze_slow_trends(raw, window_seconds=(5,))
    pz_alone = analyze_slow_trends(raw, channels=["Pz"], window_seconds=(5,))
    assert [row for row in together["windows"] if row["channel"] == "Pz"] == pz_alone["windows"]
    summary = together["channel_summaries"][0]
    assert summary["excluded_nonfinite_samples"] == 51
    assert summary["excluded_annotation_samples"] == 200
    assert summary["finite_supported_samples"] == 1749
    json.dumps(together, allow_nan=False)


def test_short_and_constant_signals_produce_finite_records_without_invented_windows():
    short = analyze_slow_trends(_raw(np.zeros(425)))
    assert short["windows"] == []
    assert short["segments"][0]["unbinned_tail_samples"] == 25
    assert short["channel_summaries"][0]["scales"][0]["max_abs_slope_v_per_s"] is None
    constant = analyze_slow_trends(_raw(np.full(3000, .012)))
    assert all(row["slope_v_per_s"] == row["residual_mad_v"] == row["bin_ptp_v"] == 0
               for row in constant["windows"])
    assert constant["artifact_classification"] is False
    json.dumps([short, constant], allow_nan=False)


def test_raw_relative_coordinates_and_explicit_eog_selection():
    raw = _raw(np.zeros((3, 2000)), first_samp=1000)
    raw.annotations.append(raw.first_time + 6, 0, "EDGE boundary")
    default = analyze_slow_trends(raw, window_seconds=(5,))
    assert default["channel_names"] == ["Cz", "Pz"]
    assert default["raw_first_sample"] == 1000
    assert [(row["start_sample"], row["end_sample"]) for row in default["segments"][:2]] == [(0, 600), (600, 2000)]
    eye = analyze_slow_trends(raw, channels=["EOG"], window_seconds=(5,))
    assert eye["channel_names"] == ["EOG"]


@pytest.mark.parametrize("kwargs", [
    {"bin_seconds": 0}, {"bin_seconds": True}, {"bin_seconds": np.bool_(True)},
    {"bin_seconds": np.nan},
    {"bin_seconds": .015}, {"window_seconds": ()}, {"window_seconds": (5, 5)},
    {"window_seconds": (1,)}, {"window_seconds": (5.5,)}, {"window_seconds": 5},
    {"channels": []}, {"channels": ["Pz"]}, {"channels": ["Cz", "Cz"]},
    {"channels": "Cz"}, {"channels": [["Cz"]]},
])
def test_invalid_support_or_channel_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        analyze_slow_trends(_raw(np.zeros(1000)), **kwargs)
