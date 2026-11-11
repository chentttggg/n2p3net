"""Exploratory remedies for measured BrainSync artifacts, without changing raw data.

Replay the exact production filter/epoch kernels. All alternative cutoffs, masks,
trend subtraction and robust averages are diagnostic comparisons, not fitted QC.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import trim_mean
from threadpoolctl import threadpool_limits

from data.brainsync import BRAIN_SYNC_PREPROCESSING, load_brainsync_inference_session
from data.brainsync_contract import validate_brainsync_bids_session
from data.preprocess import (
    design_continuous_sos,
    filter_bidirectional_segment,
    transform_epoch_samples,
)
from research.recording_quality import file_sha256

CUTOFFS = (.1, .3, .5, 1.)
GUARDS = (0., 2., 5., 10., 20., 30.)
REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def q(values):
    return {name: float(np.quantile(values, fraction)) for name, fraction in (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1.))}


def filter_values(values, fs, segments, spec):
    sos = design_continuous_sos(fs, l_freq=spec.l_freq, h_freq=spec.h_freq,
                                order=spec.filter_order, notch_freqs_hz=spec.notch_freqs_hz,
                                notch_quality_factor=spec.notch_quality_factor)
    pad = int(mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)["padlen"])
    result = np.zeros_like(values, dtype=np.float64)
    for a, b in segments:
        result[:, a:b] = filter_bidirectional_segment(sos, values[:, a:b], padding_limit=pad).data
    return result


def epoch_values(values, fs, onsets, spec):
    start = round(spec.tmin_ms / 1000 * fs)
    stop = round(spec.tmax_ms / 1000 * fs)
    extracted = np.stack([values[:, onset + start:onset + stop + 1] for onset in onsets])
    return transform_epoch_samples(extracted, fs, spec)


def trend_values(values, fs, segments, width_seconds):
    """Exploratory piecewise-linear interpolation of complete bin medians."""
    result = np.empty_like(values)
    width = round(width_seconds * fs)
    for a, b in segments:
        count = (b - a) // width
        if count < 2:
            result[:, a:b] = np.median(values[:, a:b], axis=1, keepdims=True)
            continue
        centers = a + np.arange(count) * width + (width - 1) / 2
        medians = np.median(values[:, a:a + count * width].reshape(len(values), count, width), axis=-1)
        result[:, a:b] = interp1d(centers, medians, axis=-1, kind="linear", fill_value="extrapolate")(np.arange(a, b))
    return result


def median_slopes(values, fs, segments):
    slopes = []
    width = round(fs)
    for a, b in segments:
        count = (b - a) // width
        if count <= 30:
            continue
        medians = np.median(values[:, a:a + count * width].reshape(len(values), count, width), axis=-1)
        slopes.append(np.abs(medians[:, 30:] - medians[:, :-30]) / 30)
    return np.concatenate(slopes, axis=1) * 1e6


def detect_large_changes(values, fs, segments, *, threshold_uv=500., bin_seconds=.2):
    """Fixed raw-bin sensitivity probe, independent of target labels.

    A 0.2 s bin spans ten nominal 50Hz cycles. This cannot certify motion or
    detect every artifact and does not become a production accept/reject rule.
    """
    width = round(bin_seconds * fs)
    intervals = []
    for a, b in segments:
        count = (b - a) // width
        bins = np.median(values[:, a:a + count * width].reshape(len(values), count, width), axis=-1)
        for channel in range(len(values)):
            indices = np.flatnonzero(np.abs(np.diff(bins[channel])) * 1e6 > threshold_uv)
            if not len(indices):
                continue
            clusters = np.split(indices, np.flatnonzero(np.diff(indices) * bin_seconds > 1.) + 1)
            for cluster in clusters:
                left = max(a, a + (int(cluster[0]) - 1) * width)
                right = min(b, a + (int(cluster[-1]) + 3) * width)
                intervals.append({"channel_index": channel, "start_sample": left, "stop_sample": right,
                                  "start_s": left / fs, "stop_s": right / fs,
                                  "segment_start_sample": a, "segment_stop_sample": b,
                                  "max_02s_median_change_uv": float(np.abs(np.diff(bins[channel]))[cluster].max() * 1e6)})
    return sorted(intervals, key=lambda row: (row["start_sample"], row["channel_index"]))


def masks_for_intervals(intervals, onsets, fs, n_channels, guard):
    starts = onsets + round(-.2 * fs)
    stops = onsets + round(1.2 * fs) + 1
    valid = np.ones((len(onsets), n_channels), dtype=bool)
    for interval in intervals:
        left = max(interval["segment_start_sample"], interval["start_sample"] - guard * fs)
        right = min(interval["segment_stop_sample"], interval["stop_sample"] + guard * fs)
        overlap = (stops > left) & (starts < right)
        valid[overlap, interval["channel_index"]] = False
    return valid


def summarize_epochs(epochs):
    ptp = np.ptp(epochs, axis=-1) * 1e6
    delta = (np.median(epochs[:, :, -26:], axis=-1) - np.median(epochs[:, :, :26], axis=-1)) * 1e6
    return {"channel_ptp_uv": [q(ptp[:, i]) for i in range(ptp.shape[1])],
            "channel_abs_end_minus_baseline_uv": [q(np.abs(delta[:, i])) for i in range(ptp.shape[1])],
            "epochs_any_channel_ptp_above_uv": {str(t): int((ptp > t).any(axis=1).sum()) for t in (100, 200, 500)}}


def synthetic_controls(directory):
    fs = 250.
    samples = round(240 * fs)
    center = round(120 * fs)
    t = (np.arange(samples) - center) / fs
    narrow = 5e-6 * np.exp(-.5 * ((t - .35) / .08) ** 2)
    broad = np.zeros(samples)
    positive = t > 0
    broad[positive] = 5e-6 * (t[positive] / .45) ** 3 * np.exp(3 * (1 - t[positive] / .45))
    pulses = np.stack((narrow, broad))
    filtered = {}
    epo = {}
    for cutoff in (None, *CUTOFFS):
        spec = replace(BRAIN_SYNC_PREPROCESSING, l_freq=cutoff)
        filtered[str(cutoff)] = filter_values(pulses, fs, [(0, samples)], spec)
        epo[str(cutoff)] = epoch_values(filtered[str(cutoff)], fs, [center], spec)[0]
    times = -.2 + np.arange(179) / 128
    response = (times >= .25) & (times < .6)
    controls = {"erp_shapes": []}
    for i, name in enumerate(("5uV Gaussian peak350ms sigma80ms", "5uV gamma peak450ms shape3")):
        base = epo["0.1"][i]
        rows = []
        for cutoff in CUTOFFS:
            current = epo[str(cutoff)][i]
            rows.append({"highpass_hz": cutoff, "peak_uv": float(current.max() * 1e6),
                         "mean_250_600_uv": float(current[response].mean() * 1e6),
                         "mean_ratio_to_01": float(current[response].mean() / base[response].mean()),
                         "pre_stimulus_max_abs_uv": float(np.abs(current[times < 0]).max() * 1e6),
                         "rms_difference_to_01_uv": float(np.sqrt(np.mean((current - base) ** 2)) * 1e6)})
        controls["erp_shapes"].append({"name": name, "methods": rows})
    artifact = np.zeros((2, samples))
    artifact[0, center:center + round(.2 * fs)] = .080
    artifact[1, center:] = .080
    all_onsets = np.arange(round(60 * fs), round(180 * fs), round(1.5 * fs))
    controls["80mv_artifacts"] = []
    for cutoff in CUTOFFS:
        spec = replace(BRAIN_SYNC_PREPROCESSING, l_freq=cutoff)
        f = filter_values(artifact, fs, [(0, samples)], spec)
        e = epoch_values(f, fs, all_onsets, spec)
        for channel, name in enumerate(("0.2s pulse", "persistent step")):
            row = {"kind": name, "highpass_hz": cutoff, "thresholds_uv": {}}
            for threshold in (1., 5.):
                bad = np.flatnonzero(np.abs(f[channel]) > threshold * 1e-6)
                bad_epochs = np.max(np.abs(e[:, channel]), axis=-1) > threshold * 1e-6
                row["thresholds_uv"][str(threshold)] = {
                    "continuous_influence_left_s": max(0., (center - int(bad[0])) / fs),
                    "continuous_influence_right_s": max(0., (int(bad[-1]) - center) / fs),
                    "affected_epoch_count_of80": int(bad_epochs.sum()),
                    "affected_epoch_onsets_relative_s": ((all_onsets[bad_epochs] - center) / fs).tolist()}
            controls["80mv_artifacts"].append(row)
        if cutoff == .1:
            np.savez_compressed(directory / "synthetic_artifact.npz", time_relative_s=t,
                                raw_artifact_v=artifact, filtered_artifact_v=f,
                                epoch_onsets_s=(all_onsets - center) / fs, epoch_values_v=e)
    # Baseline subtraction removes a constant, not an affine slope.
    linear = 100e-6 * np.arange(samples)[None] / fs
    baseline_only = epoch_values(linear, fs, [center], BRAIN_SYNC_PREPROCESSING)[0, 0]
    linear_f = filter_values(linear, fs, [(0, samples)], BRAIN_SYNC_PREPROCESSING)
    linear_processed = epoch_values(linear_f, fs, [center], BRAIN_SYNC_PREPROCESSING)[0, 0]
    controls["100uv_per_s_linear_ramp"] = {"baseline_only_max_abs_uv": float(np.max(np.abs(baseline_only)) * 1e6),
                                           "full_production_max_abs_uv_at_interior": float(np.max(np.abs(linear_processed)) * 1e6)}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for cutoff in CUTOFFS:
        axes[0].plot(times * 1000, epo[str(cutoff)][1] * 1e6, label=f"HP {cutoff:g} Hz")
    axes[0].set(xlabel="Nominal epoch time (ms)", ylabel="uV", title="Known broad 5 uV ERP, same production kernels")
    axes[0].legend(fontsize=8)
    with np.load(directory / "synthetic_artifact.npz") as arr:
        for i, name in enumerate(("80mV / 0.2s pulse", "80mV step")):
            axes[i + 1].plot(t, arr["filtered_artifact_v"][i] * 1e6)
            axes[i + 1].set(xlim=(-45, 45), ylim=(-150, 150), xlabel="Time from artifact (s)", ylabel="uV",
                            title=f"{name}: HP 0.1 Hz tail (central peak clipped)")
            axes[i + 1].axhline(5, color=".5", linestyle="--", linewidth=.8)
            axes[i + 1].axhline(-5, color=".5", linestyle="--", linewidth=.8)
    for axis in axes:
        axis.grid(alpha=.15)
    figure.savefig(directory / "filter_tradeoffs.png", dpi=145, facecolor="white")
    plt.close(figure)
    write_json(directory / "synthetic_controls.json", controls)
    return controls


def analyze_session(row, directory):
    root = Path(row["input_root"])
    d = load_brainsync_inference_session(root)
    validated = validate_brainsync_bids_session(root)
    try:
        raw = mne.io.read_raw_edf(validated.bids.raw_path, preload=True, verbose="ERROR")
        try:
            fs = float(raw.info["sfreq"])
            acquired = int((validated.manifest["recording"].get("clock") or {}).get("acquired_samples", raw.n_times))
            values = raw.get_data(stop=acquired)
            cuts = [0, *(round(t * fs) for t in validated.bids.acquisition_boundaries), acquired]
            segments = list(zip(cuts[:-1], cuts[1:], strict=True))
            eeg = [i for i, t in enumerate(validated.bids.channel_types) if t.lower() == "eeg"]
            eog = next(i for i, t in enumerate(validated.bids.channel_types) if t.lower() == "eog")
            available = np.asarray(d.event_timeline.evidence_indices) >= 0
            onsets = np.asarray(d.event_timeline.onset_samples)[available]
            assert np.array_equal(np.asarray(d.event_timeline.evidence_indices)[available], np.arange(d.n_epochs))
            markers = np.asarray(validated.bids.stimuli, dtype=object)[available]
            labels = np.asarray([int(m.candidate_id) == validated.thought_digit for m in markers])
            result = {"session": row["directory_id"], "scheduled": len(validated.bids.stimuli),
                      "basic_epochs": d.n_epochs, "channel_names": list(d.channel_names),
                      "methods": {}, "raw_slope_uv_per_s": [q(v) for v in median_slopes(values[eeg], fs, segments)]}
            production = filter_values(values, fs, segments, BRAIN_SYNC_PREPROCESSING)
            current_all = epoch_values(production, fs, onsets, BRAIN_SYNC_PREPROCESSING)
            current = current_all[:, eeg]
            assert np.array_equal(current.astype(d.X.dtype), d.X), "Custom replay differs from actual production X"
            result["production_epoch_replay_bitwise_equal"] = True
            result["filtered_slope_uv_per_s"] = [q(v) for v in median_slopes(production[eeg], fs, segments)]
            result["methods"]["hp_0.1"] = summarize_epochs(current)
            for cutoff in CUTOFFS[1:]:
                spec = replace(BRAIN_SYNC_PREPROCESSING, l_freq=cutoff)
                transformed = epoch_values(filter_values(values[eeg], fs, segments, spec), fs, onsets, spec)
                result["methods"][f"hp_{cutoff:g}"] = summarize_epochs(transformed)
            # One known additive broad ERP on Pz, at confirmed target flashes.
            injected = np.zeros_like(values[eeg])
            support = np.arange(round(1.2 * fs)) / fs
            pulse = np.zeros_like(support)
            positive = support > 0
            pulse[positive] = 5e-6 * (support[positive] / .45) ** 3 * np.exp(3 * (1 - support[positive] / .45))
            pz = d.channel_names.index("PZ")
            for onset in onsets[labels]:
                stop = min(acquired, int(onset) + len(pulse))
                injected[pz, onset:stop] += pulse[:stop - onset]
            injection_reference = epoch_values(filter_values(injected, fs, segments, BRAIN_SYNC_PREPROCESSING), fs, onsets,
                                               BRAIN_SYNC_PREPROCESSING)
            result["slow_trend_subtraction"] = []
            for width in (30., 2.):
                trend = trend_values(values[eeg], fs, segments, width)
                filtered_trend = filter_values(trend, fs, segments, BRAIN_SYNC_PREPROCESSING)
                trend_epoch = epoch_values(filtered_trend, fs, onsets, BRAIN_SYNC_PREPROCESSING)
                detrended = current - trend_epoch
                result["methods"][f"median_trend_{width:g}s_then_hp01"] = summarize_epochs(detrended)
                perturbed_trend = trend_values(values[eeg] + injected, fs, segments, width)
                trend_change_epoch = epoch_values(filter_values(perturbed_trend - trend, fs, segments, BRAIN_SYNC_PREPROCESSING),
                                                 fs, onsets, BRAIN_SYNC_PREPROCESSING)
                relative_error = np.linalg.norm(trend_change_epoch[:, pz]) / np.linalg.norm(injection_reference[:, pz])
                result["slow_trend_subtraction"].append({"median_bin_s": width,
                    "removed_component_epoch_max_abs_uv_quantiles": [q(np.max(np.abs(trend_epoch[:, i]), axis=-1) * 1e6) for i in range(len(eeg))],
                    "known_additive_erp_increment_relative_l2_change": float(relative_error)})
            # Full-epoch linear detrending is an explicitly aggressive negative control.
            grid = np.arange(current.shape[-1], dtype=float)
            grid -= grid.mean()
            slope = np.einsum("ect,t->ec", current, grid) / (grid @ grid)
            detrended = current - current.mean(axis=-1, keepdims=True) - slope[..., None] * grid
            result["methods"]["full_epoch_linear_detrend_negative_control"] = summarize_epochs(detrended)
            inj_slope = np.einsum("ect,t->ec", injection_reference, grid) / (grid @ grid)
            inj_delta = injection_reference.mean(axis=-1, keepdims=True) + inj_slope[..., None] * grid
            result["full_epoch_detrend_known_erp_relative_l2_change"] = float(np.linalg.norm(inj_delta[:, pz]) / np.linalg.norm(injection_reference[:, pz]))
            intervals = detect_large_changes(values[eeg], fs, segments)
            for item in intervals:
                item["channel_name"] = d.channel_names[item["channel_index"]]
            result["candidate_intervals"] = intervals
            result["candidate_threshold_uv_per_02s"] = 500.
            result["guard_sensitivity"] = []
            for guard in GUARDS:
                valid = masks_for_intervals(intervals, onsets, fs, len(eeg), guard)
                result["guard_sensitivity"].append({"guard_s": guard, "all7_available_epochs": int(valid.all(axis=1).sum()),
                    "at_least5_available_epochs": int((valid.sum(axis=1) >= 5).sum()),
                    "available_channel_epochs": int(valid.sum()), "total_channel_epochs": int(valid.size),
                    "per_channel_available_epochs": valid.sum(axis=0).tolist()})
            ptp = np.ptp(current, axis=-1) * 1e6
            result["ptp_mask_sensitivity"] = []
            for threshold in (100., 200., 500.):
                valid = ptp <= threshold
                result["ptp_mask_sensitivity"].append({"threshold_uv": threshold,
                    "all7_available_epochs": int(valid.all(axis=1).sum()),
                    "at_least5_available_epochs": int((valid.sum(axis=1) >= 5).sum()),
                    "per_channel_available_epochs": valid.sum(axis=0).tolist()})
            centered = current_all - current_all.mean(axis=-1, keepdims=True)
            eog_signal = centered[:, eog]
            correlations = np.einsum("ect,et->ec", centered[:, eeg], eog_signal) / np.sqrt(
                np.einsum("ect,ect->ec", centered[:, eeg], centered[:, eeg]) * np.einsum("et,et->e", eog_signal, eog_signal)[:, None])
            result["same_band_eog_abs_epoch_correlation"] = [q(np.abs(correlations[:, i])) for i in range(len(eeg))]
            means, robust, masked = [], [], []
            result["single_epoch_mean_erp_influence"] = []
            for channel in range(len(eeg)):
                target = current[labels, channel]
                other = current[~labels, channel]
                mean_difference = target.mean(axis=0) - other.mean(axis=0)
                means.append(mean_difference)
                robust.append(trim_mean(target, .2, axis=0) - trim_mean(other, .2, axis=0))
                valid = ptp[:, channel] <= 200
                masked.append(current[labels & valid, channel].mean(axis=0) - current[~labels & valid, channel].mean(axis=0))
                changes = np.empty(len(current))
                changes[labels] = np.max(np.abs((target - target.mean(axis=0)) / (len(target) - 1)), axis=-1) * 1e6
                changes[~labels] = np.max(np.abs((other - other.mean(axis=0)) / (len(other) - 1)), axis=-1) * 1e6
                maximum = int(np.argmax(changes))
                result["single_epoch_mean_erp_influence"].append({"channel_name": d.channel_names[channel],
                    "max_change_uv": float(changes[maximum]), "event_onset_s": float(onsets[maximum] / fs),
                    "event_is_target": bool(labels[maximum]),
                    "max_trimmed_vs_mean_difference_uv": float(np.max(np.abs(robust[-1] - means[-1])) * 1e6),
                    "max_masked_vs_mean_difference_uv": float(np.max(np.abs(masked[-1] - means[-1])) * 1e6)})
            means, robust, masked = np.asarray(means), np.asarray(robust), np.asarray(masked)
            times = -.2 + np.arange(179) / 128
            np.savez_compressed(directory / f"{root.name}_arrays.npz", time_s=times, onsets_s=onsets / fs,
                                labels=labels, epoch_ptp_uv=ptp, same_band_eog_correlations=correlations,
                                erp_mean_v=means, erp_trimmed20_v=robust, erp_channel_ptp200_mask_v=masked,
                                channel_names=np.asarray(d.channel_names))
            figure, axes = plt.subplots(1, 2, figsize=(12, 3.7), layout="constrained", sharex=True)
            for axis, name in zip(axes, ("FZ", "PZ"), strict=True):
                channel = d.channel_names.index(name)
                axis.plot(times * 1000, means[channel] * 1e6, label="All basic epochs")
                axis.plot(times * 1000, robust[channel] * 1e6, label="20% trimmed (diagnostic)", linestyle="--")
                axis.plot(times * 1000, masked[channel] * 1e6, label="Channel PTP<=200uV (diagnostic)", linestyle=":")
                axis.set(xlabel="Nominal model time (ms)", ylabel="Target - nontarget (uV)", title=f"{root.name} / {name}")
                axis.axhline(0, color=".5", linewidth=.6)
                axis.grid(alpha=.15)
            axes[0].legend(fontsize=8)
            figure.savefig(directory / f"{root.name}_erp_sensitivity.png", dpi=140, facecolor="white")
            plt.close(figure)
            # Local replacement is a sensitivity experiment, never recovered EEG.
            if intervals:
                strongest = max(intervals, key=lambda item: item["max_02s_median_change_uv"])
                left = max(0, strongest["start_sample"] - round(fs))
                right = min(acquired, strongest["stop_sample"] + round(fs))
                segment = next((a, b) for a, b in segments if a <= strongest["start_sample"] < b)
                left, right = max(left, segment[0]), min(right, segment[1])
                shadow = values[eeg].copy()
                for channel in range(len(eeg)):
                    shadow[channel, left:right] = np.linspace(shadow[channel, left], shadow[channel, right - 1], right - left)
                filtered_difference = filter_values(values[eeg] - shadow, fs, segments, BRAIN_SYNC_PREPROCESSING)
                epoch_difference = epoch_values(filtered_difference, fs, onsets, BRAIN_SYNC_PREPROCESSING)
                sensitivity = np.max(np.abs(epoch_difference), axis=(1, 2)) * 1e6
                result["strongest_interval_linear_bridge_sensitivity"] = {
                    "interval_s": [left / fs, right / fs], "trigger_channel": strongest["channel_name"],
                    "not_clean_ground_truth": True,
                    "affected_epochs_max_abs_change_above_uv": {str(threshold): int((sensitivity > threshold).sum()) for threshold in (1., 5., 20.)}}
                selected = (onsets / fs > left / fs - 50) & (onsets / fs < right / fs + 50)
                figure, axes = plt.subplots(2, 1, figsize=(11.5, 6), layout="constrained")
                fz = d.channel_names.index("FZ")
                lo, hi = max(segment[0], left - round(20 * fs)), min(segment[1], right + round(20 * fs))
                plot_t = np.arange(lo, hi) / fs
                axes[0].plot(plot_t, values[eeg[fz], lo:hi] * 1e6, linewidth=.65)
                axes[0].axvspan(left / fs, right / fs, color="#e69f00", alpha=.2)
                axes[0].set(ylabel="Raw Fz (uV)", title=f"{root.name}: candidate transient and conditional sensitivity")
                axes[1].scatter(onsets[selected] / fs, sensitivity[selected], s=15)
                axes[1].axhline(5, color=".5", linestyle="--", linewidth=.8)
                axes[1].set(xlabel="Original stored sample time (s)", ylabel="Epoch max change (uV)")
                axes[1].set_yscale("symlog", linthresh=1)
                for axis in axes:
                    axis.grid(alpha=.15)
                figure.savefig(directory / f"{root.name}_transient_sensitivity.png", dpi=140, facecolor="white")
                plt.close(figure)
            assert all(file_sha256(root / p) == digest for p, digest in row["input_sha256"].items())
            validated.bids.assert_content_unchanged()
            write_json(directory / f"{root.name}_results.json", result)
            return result
        finally:
            raw.close()
    finally:
        validated.bids.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    batch = args.batch_manifest.resolve()
    parent = json.loads(batch.read_text(encoding="utf-8"))
    sources = [Path(__file__), REPO / "src/data/preprocess.py", REPO / "src/data/brainsync.py"]
    sources_sha = {str(p.relative_to(REPO)): file_sha256(p) for p in sources}
    manifest = {"schema": "brainsync_artifact_remedies/1", "created_utc": datetime.now(UTC).isoformat(),
                "parent_manifest": str(batch), "parent_sha256": file_sha256(batch), "source_sha256": sources_sha,
                "role": "exploratory_quality_remedy_comparisons", "production_defaults_changed": False,
                "raw_files_modified": False, "classifier_fitted": False,
                "methods": {"cutoffs_hz": CUTOFFS, "raw_candidate_probe": "Adjacent 0.2s raw bin medians differ by >500uV, within original acquisition segments; cluster within 1s; not certified artifacts.",
                            "drift_estimators": "Piecewise-linear interpolation/extrapolation of complete nonoverlapping 30s or 2s bin medians, fitted only as exploratory full-record descriptive processing.",
                            "eog": "Single HEOG is separately filtered in the same band for descriptive correlation; this is not the production EEG feature pipeline or EOG regression.",
                            "masks_and_guards": "Fixed sensitivity probes; they do not establish clean intervals, validated recovery, or product decision availability.",
                            "erp": "Confirmed labels used only for descriptive mean/trimmed/masked ERP and synthetic incremental response test; no threshold/model selection on accuracy.",
                            "interpolation": "Shadow linear bridge only measures dependence on an interval; no filled sample is accepted as recovered EEG."}}
    with threadpool_limits(limits=2):
        manifest["synthetic"] = synthetic_controls(directory)
        rows = []
        for row in parent["sessions"]:
            if row["group"].startswith("Brain") and row.get("extra"):
                print("START", row["directory_id"], flush=True)
                result = analyze_session(row, directory)
                rows.append(result)
                print("DONE", result["session"], "candidate_channel_intervals", len(result["candidate_intervals"]), flush=True)
    manifest["sessions"] = rows
    assert all(file_sha256(REPO / p) == digest for p, digest in sources_sha.items())
    assert file_sha256(batch) == manifest["parent_sha256"]
    manifest["finished_utc"] = datetime.now(UTC).isoformat()
    with (directory / "candidate_intervals.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("session", "channel_name", "start_s", "stop_s", "max_02s_median_change_uv"))
        writer.writeheader()
        for row in rows:
            for item in row["candidate_intervals"]:
                writer.writerow({"session": row["session"], **{k: item[k] for k in writer.fieldnames if k != "session"}})
    manifest["output_sha256"] = {p.name: file_sha256(p) for p in directory.iterdir() if p.is_file()}
    write_json(directory / "manifest.json", manifest)
    print("COMPLETE", directory, flush=True)


if __name__ == "__main__":
    main()
