"""Descriptive raw-channel trends; no fit, artifact classification, or masking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mne
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from data.preprocess import continuous_filter_segments

SLOW_TREND_SCHEMA = "n2p3_slow_trend_diagnostics/1"


def _sample_count(seconds: float, rate: float, name: str) -> int:
    if isinstance(seconds, (bool, np.bool_)) or not np.isscalar(seconds):
        raise ValueError(f"{name} must be a positive finite duration")
    try:
        value = float(seconds)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite duration") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite duration")
    count = round(value * rate)
    if count < 1 or not np.isclose(count, value * rate, rtol=0, atol=1e-8):
        raise ValueError(f"{name} must span an integer number of raw samples")
    return count


def _window_metrics(bins: np.ndarray, width: int, bin_seconds: float) -> np.ndarray:
    """Vectorized Theil-Sen paired-slope median, checked against SciPy in tests."""
    windows = sliding_window_view(bins, width)
    left, right = np.triu_indices(width, k=1)
    elapsed = (right - left) * bin_seconds
    time = np.arange(width) * bin_seconds
    metrics = np.empty((len(windows), 3), dtype=float)
    # Bound the paired-slope workspace for long recordings and larger windows.
    batch_size = max(1, min(512, 1_000_000 // len(left)))
    for start in range(0, len(windows), batch_size):
        values = windows[start:start + batch_size]
        slopes = np.median((values[:, right] - values[:, left]) / elapsed, axis=1)
        residual = values - slopes[:, None] * time
        residual -= np.median(residual, axis=1, keepdims=True)
        metrics[start:start + len(values), 0] = slopes
        metrics[start:start + len(values), 1] = np.median(np.abs(residual), axis=1)
        metrics[start:start + len(values), 2] = np.ptp(values, axis=1)
    if not np.isfinite(metrics).all():
        raise ValueError("slow-trend metrics overflowed the finite physical signal range")
    return metrics


def analyze_slow_trends(
    raw: mne.io.BaseRaw,
    *,
    channels: Sequence[str] | None = None,
    window_seconds: Sequence[float] = (5.0, 15.0, 30.0),
    bin_seconds: float = 1.0,
) -> dict[str, Any]:
    """Describe complete raw windows without interpreting trends as artifacts.

    Default channels are all measured EEG channels, including already marked
    bad channels; EOG is included only when explicitly requested. Acquisition
    boundaries and channel-local artifact annotations follow the shared raw
    support helper. Nonfinite samples split support further. Bins restart at
    each boundary; incomplete tail bins/windows are not padded or joined.

    Each slope change compares the current window with its immediately
    preceding, non-overlapping window of the same duration. No future sample
    is used beyond the row's ``available_sample``. All sample coordinates and
    times are relative to the first sample of the supplied raw object.
    """
    rate = float(raw.info["sfreq"])
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("slow-trend diagnostics require a positive finite sample rate")
    bin_samples = _sample_count(bin_seconds, rate, "bin_seconds")
    bin_duration = bin_samples / rate
    if isinstance(window_seconds, (str, bytes)):
        raise ValueError("window_seconds must contain distinct complete-bin durations")
    try:
        scales = tuple(_sample_count(value, rate, "window_seconds") for value in window_seconds)
    except TypeError as error:
        raise ValueError("window_seconds must contain distinct complete-bin durations") from error
    if (not scales or len(set(scales)) != len(scales)
            or any(count % bin_samples or count < 2 * bin_samples for count in scales)):
        raise ValueError("window_seconds must contain distinct durations of at least two complete bins")
    if channels is None:
        names = tuple(raw.ch_names[index] for index in mne.pick_types(raw.info, eeg=True, exclude=[]))
    else:
        if isinstance(channels, (str, bytes)):
            raise ValueError("channels must be an explicit sequence of measured channel names")
        names = tuple(channels)
    if (not names or any(not isinstance(name, str) or name not in raw.ch_names for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("slow-trend channels must be distinct measured channel names")

    rows: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name in names:
        values = raw.get_data(picks=[name])[0]
        spans = continuous_filter_segments(raw, name)
        finite_samples = 0
        bins_count = 0
        tail_samples = 0
        channel_rows: list[dict[str, Any]] = []
        segment_index = 0
        for span_start, span_stop in spans:
            finite = np.isfinite(values[span_start:span_stop])
            changes = np.flatnonzero(np.diff(np.r_[False, finite, False].astype(np.int8)))
            for local_start, local_stop in zip(changes[::2], changes[1::2], strict=True):
                start, stop = int(span_start + local_start), int(span_start + local_stop)
                count, tail = divmod(stop - start, bin_samples)
                finite_samples += stop - start
                bins_count += count
                tail_samples += tail
                segments.append({"channel": name, "segment_index": segment_index,
                                 "start_sample": start, "end_sample": stop,
                                 "complete_bins": count, "unbinned_tail_samples": tail})
                bins = np.median(values[start:start + count * bin_samples].reshape(count, bin_samples), axis=1)
                for window_samples in scales:
                    width = window_samples // bin_samples
                    if count < width:
                        continue
                    metrics = _window_metrics(bins, width, bin_duration)
                    for index, (slope, residual_mad, excursion) in enumerate(metrics):
                        begin = start + index * bin_samples
                        end = begin + window_samples
                        comparison = begin - window_samples if index >= width else None
                        change = float(slope - metrics[index - width, 0]) if comparison is not None else None
                        channel_rows.append({
                            "channel": name, "segment_index": segment_index,
                            "window_seconds": window_samples / rate,
                            "start_sample": begin, "end_sample": end,
                            "start_time_s": begin / rate, "end_time_s": end / rate,
                            "available_sample": end - 1, "complete_bins": width,
                            "supported_samples": window_samples,
                            "slope_v_per_s": float(slope), "residual_mad_v": float(residual_mad),
                            "bin_ptp_v": float(excursion), "slope_change_v_per_s": change,
                            "comparison_start_sample": comparison,
                            "change_supported_samples": 2 * window_samples if comparison is not None else None,
                        })
                segment_index += 1
        scale_summaries = []
        for window_samples in scales:
            selected = [row for row in channel_rows if row["supported_samples"] == window_samples]
            changes = [abs(row["slope_change_v_per_s"]) for row in selected if row["slope_change_v_per_s"] is not None]
            scale_summaries.append({
                "window_seconds": window_samples / rate, "window_count": len(selected),
                "slope_change_count": len(changes),
                "max_abs_slope_v_per_s": max((abs(row["slope_v_per_s"]) for row in selected), default=None),
                "max_abs_slope_change_v_per_s": max(changes, default=None),
                "max_residual_mad_v": max((row["residual_mad_v"] for row in selected), default=None),
            })
        annotated_support = int(sum(stop - start for start, stop in spans))
        summaries.append({"channel": name, "finite_supported_samples": finite_samples,
                          "excluded_annotation_samples": int(raw.n_times) - annotated_support,
                          "excluded_nonfinite_samples": annotated_support - finite_samples,
                          "complete_bins": bins_count, "unbinned_supported_tail_samples": tail_samples,
                          "window_count": len(channel_rows), "scales": scale_summaries})
        rows.extend(channel_rows)
    return {
        "schema": SLOW_TREND_SCHEMA, "status": "diagnostic_only", "fit_performed": False,
        "labels_used": False, "mask_changes": False, "artifact_classification": False,
        "signal_domain": "supplied_raw", "required_input_stage": "native_before_reference_and_filter",
        "signal_unit": "V", "fit_scope": "none_descriptive_window_estimation_only",
        "slope_unit": "V/s", "residual_mad_consistency_factor": 1.0,
        "sample_coordinates": "zero_based_relative_to_raw_first_sample", "sample_interval": "half_open",
        "raw_first_sample": int(raw.first_samp), "n_samples": int(raw.n_times), "sample_rate_hz": rate,
        "channel_names": list(names), "bin_seconds": bin_duration, "bin_samples": bin_samples,
        "window_seconds": [count / rate for count in scales], "step_samples": bin_samples,
        "estimator": "one_second_median_bins_then_theil_sen_paired_slope_median"
        if bin_duration == 1 else "median_bins_then_theil_sen_paired_slope_median",
        "slope_change": "current_minus_immediately_preceding_nonoverlapping_same_duration_window",
        "support_policy": "shared_continuous_filter_segments_then_channel_local_finite_runs",
        "interpretation": "Slow trends and changes may be neural activity, electrode drift, or both; these statistics do not identify their cause.",
        "segments": segments, "windows": rows, "channel_summaries": summaries,
    }
