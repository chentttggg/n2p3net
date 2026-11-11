"""Same-operator descriptive comparison of the September and October recordings.

The common-rule replay is a retrospective numerical diagnostic. It does not
extend the admitted /5 recording roster or change any recording disposition.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import sys
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
from analyze_brainsync_artifact_remedies import (
    detect_large_changes,
    epoch_values,
    filter_values,
    median_slopes,
)
from scipy.stats import trim_mean
from threadpoolctl import threadpool_limits

from data.brainsync import BRAIN_SYNC_PREPROCESSING, load_brainsync_inference_session
from data.brainsync_contract import validate_brainsync_bids_session
from data.continuous_qc import ContinuousQCInterval, ContinuousQCResult, load_continuous_qc
from data.epochs import load_epoch_dataset
from data.preprocess import filter_continuous
from research.execution import atomic_write_json
from research.recording_quality import file_sha256

REPO = Path(__file__).resolve().parents[1]
EXCLUDED = "P001_20261030_144501_01"
BAD_FZ = {"P001_20261028_221030_01", "P001_20261028_222559_01"}
N_BOOT = 2000


def native_raw(validated):
    raw = mne.io.read_raw_edf(validated.bids.raw_path, preload=True, verbose=False)
    raw.set_channel_types(dict(zip(raw.ch_names, [t.lower() for t in validated.bids.channel_types], strict=True)))
    boundaries = list(validated.bids.acquisition_boundaries)
    if boundaries:
        raw.set_annotations(raw.annotations + mne.Annotations(boundaries, [0.] * len(boundaries),
            ["EDGE boundary"] * len(boundaries), orig_time=raw.annotations.orig_time))
    return raw


def compiled_intervals(values, fs, segments, base_epochs, onsets, names, *, persistent_fz=False):
    rows = []
    for r in detect_large_changes(values, fs, segments):
        left = max(r["segment_start_sample"], r["start_sample"] - round(.2 * fs))
        right = min(r["segment_stop_sample"], r["stop_sample"] + round(.2 * fs))
        rows.append(ContinuousQCInterval(left, right, left, right - 1,
                                        (names[r["channel_index"]],), "reviewed_transient", True))
    for trial, channel in np.argwhere(np.ptp(base_epochs, axis=-1) * 1e6 > 500):
        left = max(0, int(onsets[trial]) - round(.2 * fs))
        right = min(values.shape[1], int(onsets[trial]) + round(1.2 * fs) + 1)
        rows.append(ContinuousQCInterval(left, right, left, right - 1, (names[channel],), "reviewed_gross_spike_epoch", True))
    if persistent_fz:
        rows.append(ContinuousQCInterval(0, values.shape[1], 0, values.shape[1] - 1,
                                        (names[0],), "reviewed_persistent_unstable_channel", True))
    return tuple(rows)


def replay(raw, names, rows, recovery, full_onsets, base_available, spec):
    result = ContinuousQCResult(int(raw.n_times), float(raw.info["sfreq"]), rows,
                               "retrospective_same_rule_comparison", tuple(names), True,
                               detection_origin="fixed_comparison_rules_not_personal_model_admission")
    annotated = result.annotate(raw)
    mask = result.epoch_channel_mask(full_onsets, channels=names, tmin_s=-.2, tmax_s=1.2)
    bounds, _ = recovery.assess_epochs(annotated, full_onsets, boundary_action="artifact_edges")
    mask &= bounds
    mask &= base_available[:, None]
    keep = mask.sum(axis=1) >= 5
    filter_continuous(annotated, l_freq=spec.l_freq, h_freq=spec.h_freq, method=spec.filter_method,
        phase=spec.filter_phase, order=spec.filter_order, notch_freqs_hz=spec.notch_freqs_hz,
        notch_quality_factor=spec.notch_quality_factor, zero_phase_edge_policy=spec.zero_phase_edge_policy)
    epochs = epoch_values(annotated.get_data(picks=list(names)), raw.info["sfreq"], full_onsets[base_available], spec)
    return epochs, mask, keep


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None


def erp_summary(epochs, masks, labels, blocks, seed):
    times = -.2 + np.arange(epochs.shape[-1]) / 128
    window = (times >= .25) & (times < .6)
    response = times >= 0
    curves = np.full((3, epochs.shape[1], epochs.shape[2]), np.nan)
    summaries = []
    for channel in range(epochs.shape[1]):
        valid = masks[:, channel]
        target, other = epochs[valid & labels, channel], epochs[valid & ~labels, channel]
        row = {"target_epochs": len(target), "nontarget_epochs": len(other)}
        if not len(target) or not len(other):
            row.update(mean_250_600_uv=None, trimmed20_250_600_uv=None, block_bootstrap_ci95_uv=None,
                       odd_even_response_correlation=None)
            summaries.append(row)
            continue
        means = target.mean(axis=0) - other.mean(axis=0)
        trimmed = trim_mean(target, .2, axis=0) - trim_mean(other, .2, axis=0)
        curves[0, channel], curves[1, channel] = means * 1e6, trimmed * 1e6
        halves = []
        for odd in (True, False):
            selected = valid & ((blocks % 2 == 1) == odd)
            halves.append(epochs[selected & labels, channel].mean(axis=0)
                          - epochs[selected & ~labels, channel].mean(axis=0))
        curves[2, channel] = (halves[0] - halves[1]) * 1e6
        scalar = epochs[:, channel, window].mean(axis=-1) * 1e6
        block_stats = []
        for block in range(1, 11):
            t = valid & labels & (blocks == block)
            n = valid & ~labels & (blocks == block)
            block_stats.append((scalar[t].sum(), t.sum(), scalar[n].sum(), n.sum()))
        stats = np.asarray(block_stats)
        rng = np.random.default_rng(seed + channel)
        draws = rng.integers(0, 10, size=(N_BOOT, 10))
        sums = stats[draws].sum(axis=1)
        good = (sums[:, 1] > 0) & (sums[:, 3] > 0)
        bootstrap = sums[good, 0] / sums[good, 1] - sums[good, 2] / sums[good, 3]
        row.update(mean_250_600_uv=float(means[window].mean() * 1e6),
                   trimmed20_250_600_uv=float(trimmed[window].mean() * 1e6),
                   block_bootstrap_ci95_uv=np.quantile(bootstrap, [.025, .975]).tolist(),
                   valid_block_bootstrap_draws=int(good.sum()),
                   odd_even_response_correlation=corr(halves[0][response], halves[1][response]))
        summaries.append(row)
    return summaries, curves


def stage_summary(epochs, observed, labels, blocks, names, seed):
    ptp = np.ptp(epochs, axis=-1) * 1e6
    channels = []
    erp, curves = erp_summary(epochs, observed, labels, blocks, seed)
    for i, name in enumerate(names):
        values = ptp[observed[:, i], i]
        channels.append({"name": name.upper(), "observed_epochs": len(values),
                         "median_ptp_uv": float(np.median(values)) if len(values) else None,
                         "p95_ptp_uv": float(np.quantile(values, .95)) if len(values) else None,
                         "erp": erp[i]})
    return {"retained_epochs": int(observed.any(axis=1).sum()),
            "complete_seven_channel_epochs": int(observed.all(axis=1).sum()), "channels": channels,
            "observed_channel_epochs": int(observed.sum()),
            "epochs_any_observed_channel_ptp_gt_uv": {str(t): int(((ptp > t) & observed).any(axis=1).sum()) for t in (100, 200, 500)}}, curves


def analyze(entry, profile, current_entry, output, seed):
    root = Path(entry["input_root"])
    base = load_brainsync_inference_session(root)
    validated = validate_brainsync_bids_session(root)
    raw = native_raw(validated)
    try:
        fs = float(raw.info["sfreq"])
        names = tuple(raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, eog=False))
        picks = [raw.ch_names.index(n) for n in names]
        assert [n.upper() for n in names] == list(base.channel_names)
        recovery = profile.epoch_recovery.project(np.eye(7), names)
        full_onsets = np.asarray(base.event_timeline.onset_samples)
        available = np.asarray(base.event_timeline.evidence_indices) >= 0
        assert np.array_equal(np.asarray(base.event_timeline.evidence_indices)[available], np.arange(base.n_epochs))
        onsets = full_onsets[available]
        labels = np.asarray([int(m.candidate_id) == validated.thought_digit for m in validated.bids.stimuli])[available]
        blocks = np.asarray([m.block_id for m in validated.bids.stimuli])[available]
        clock = validated.manifest["recording"].get("clock")
        acquired = int(clock["acquired_samples"]) if clock else int(raw.n_times)
        assert acquired == raw.n_times, "Comparison must explicitly crop padding before filtering"
        cuts = [0, *(round(t * fs) for t in validated.bids.acquisition_boundaries), int(raw.n_times)]
        segments = list(zip(cuts[:-1], cuts[1:], strict=True))
        values = raw.get_data()
        filtered = filter_values(values, fs, segments, BRAIN_SYNC_PREPROCESSING)
        all_epochs = epoch_values(filtered, fs, onsets, BRAIN_SYNC_PREPROCESSING)
        epochs = all_epochs[:, picks]
        assert np.array_equal(epochs.astype(base.X.dtype), base.X)
        stage, curves_base = stage_summary(epochs, np.ones(epochs.shape[:2], bool), labels, blocks, names, seed)
        rows = compiled_intervals(values[picks], fs, segments, epochs, onsets, names)
        common_epochs, common_full_mask, common_keep = replay(raw, names, rows, recovery, full_onsets, available, BRAIN_SYNC_PREPROCESSING)
        observed = common_full_mask[available] & common_keep[available, None]
        common, curves_common = stage_summary(common_epochs, observed, labels, blocks, names, seed)
        # This comparison does not admit P003 to a P001 predictor or extend the shipped profile.
        result = {"session": entry["directory_id"], "original_session_id": validated.session_id,
                  "original_subject_id": validated.subject_id, "analysis_person_id": "same_person_01_user_confirmed",
                  "month": "September" if entry["group"].startswith("historical") else "October",
                  "target": validated.thought_digit, "scheduled": len(full_onsets),
                  "clock_grade": clock["alignment_level"] if clock else "legacy_host_only_no_sample_journal",
                  "known_missing_samples": clock.get("known_missing_samples") if clock else None,
                  "edf_padding_samples_certified": clock.get("padding_samples") if clock else None,
                  "basic_replay_bitwise_equal": True, "base": stage, "common_rule": common,
                  "reference_session_in_boundary_fit": entry["directory_id"] == "P001_20261028_224755_01",
                  "candidate_intervals": [{"start_sample": r.start_sample, "stop_sample": r.end_sample,
                                             "channels": list(r.channels), "reason": r.reason} for r in rows]}
        for name, array in (("raw", values[picks]), ("filtered", filtered[picks])):
            slopes = median_slopes(array, fs, segments)
            result[f"{name}_30s_abs_slope_p95_uv_per_s"] = np.quantile(slopes, .95, axis=1).tolist()
        centered = all_epochs - all_epochs.mean(axis=-1, keepdims=True)
        eog = next(i for i, kind in enumerate(validated.bids.channel_types) if kind == "EOG")
        result["same_band_eog_abs_correlation_median"] = []
        for index in picks:
            numerator = np.einsum("et,et->e", centered[:, index], centered[:, eog])
            denominator = np.linalg.norm(centered[:, index], axis=1) * np.linalg.norm(centered[:, eog], axis=1)
            result["same_band_eog_abs_correlation_median"].append(float(np.median(np.abs(numerator / denominator))))
        if current_entry:
            actual_rows = compiled_intervals(values[picks], fs, segments, epochs, onsets, names,
                                             persistent_fz=entry["directory_id"] in BAD_FZ)
            bound = next(r for r in profile.reviewed_recordings if r.raw_sha256 == entry["report"]["input_sha256"]["raw"])
            expected = [(r.start_sample, r.stop_sample, r.channels, r.reason) for r in bound.intervals]
            assert [(r.start_sample, r.end_sample, r.channels, r.reason) for r in actual_rows] == expected
            actual_epochs, actual_mask, actual_keep = replay(raw, names, actual_rows, recovery, full_onsets, available, BRAIN_SYNC_PREPROCESSING)
            accepted = load_epoch_dataset(current_entry["cache"])
            selected = actual_keep[available]
            actual_observed = actual_mask[available][selected]
            assert np.array_equal(np.where(actual_observed[..., None], actual_epochs[selected], 0).astype(accepted.X.dtype), accepted.X)
            assert np.array_equal(actual_observed, accepted.trial_channel_mask)
            actual, _ = stage_summary(actual_epochs, actual_mask[available] & actual_keep[available, None], labels, blocks, names, seed)
            result["october_accepted"] = actual
            result["october_current_mask_and_cache_exact"] = True
        np.savez_compressed(output / f"{entry['directory_id']}_diagnostics.npz", time_ms=(-.2 + np.arange(179) / 128) * 1000,
                            channel_names=np.asarray(base.channel_names), base_erp_uv=curves_base,
                            common_erp_uv=curves_common, common_full_channel_mask=common_full_mask,
                            common_full_epoch_kept=common_keep, full_onset_samples=full_onsets)
        validated.bids.assert_content_unchanged()
        assert all(file_sha256(root / p) == digest for p, digest in entry["input_sha256"].items())
        atomic_write_json(output / f"{entry['directory_id']}_result.json", result)
        return result
    finally:
        raw.close()
        validated.bids.close()


def summarize(rows):
    groups = {}
    for group in ("September", "October"):
        selected = [r for r in rows if r["month"] == group]
        stages = {}
        for stage in ("base", "common_rule"):
            stages[stage] = {"retained_epochs": sum(r[stage]["retained_epochs"] for r in selected),
                "complete_seven_channel_epochs": sum(r[stage]["complete_seven_channel_epochs"] for r in selected),
                "median_of_session_channel_ptp_medians_uv": [float(np.median([r[stage]["channels"][i]["median_ptp_uv"] for r in selected if r[stage]["channels"][i]["median_ptp_uv"] is not None])) for i in range(7)],
                "epochs_any_observed_channel_ptp_gt_uv": {str(t): sum(r[stage]["epochs_any_observed_channel_ptp_gt_uv"][str(t)] for r in selected) for t in (100, 200, 500)}}
        groups[group] = {"complete_eligible_sessions": len(selected), "scheduled_flashes": sum(r["scheduled"] for r in selected),
                         "stages": stages,
                         "median_of_session_filtered_30s_slope_p95_uv_per_s": np.median([r["filtered_30s_abs_slope_p95_uv_per_s"] for r in selected], axis=0).tolist()}
    return groups


def plots(rows, output):
    labels = [r["session"].removeprefix("P001_").removesuffix("_01") for r in rows]
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    y = np.arange(len(rows))
    axes[0].barh(y, [450] * len(rows), color="#dddddd", label="Scheduled")
    axes[0].barh(y, [r["common_rule"]["retained_epochs"] for r in rows], color="#0072b2", label="Same-rule diagnostic replay")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0, 530), xlabel="Flashes (single person, repeated sessions)", title="Common rule; no named Fz exclusions")
    axes[0].axhline(5.5, color=".4", linestyle="--", linewidth=.7)
    for i, r in enumerate(rows):
        axes[0].text(455, i, str(r["common_rule"]["retained_epochs"]) + "/450", va="center", fontsize=8)
    axes[0].legend(loc="lower left", fontsize=8)
    matrix = np.asarray([[c["median_ptp_uv"] for c in r["base"]["channels"]] for r in rows])
    im = axes[1].imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[1].set_yticks(y, labels)
    axes[1].set_xticks(range(7), [c["name"] for c in rows[0]["base"]["channels"]])
    axes[1].set_title("Identical basic processing: median epoch PTP (uV)")
    axes[1].axhline(5.5, color="white", linestyle="--", linewidth=.7)
    for i in range(len(rows)):
        for j in range(7):
            axes[1].text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center", fontsize=8,
                         color="white" if im.norm(matrix[i, j]) < .5 else "black")
    figure.colorbar(im, ax=axes[1], label="uV")
    figure.savefig(output / "comparison.png", dpi=145, facecolor="white")
    plt.close(figure)
    figure, axes = plt.subplots(2, 4, figsize=(16, 7.5), layout="constrained", sharex=True, sharey=True)
    for column, target in enumerate((2, 3, 4, 5)):
        for row_index, month in enumerate(("September", "October")):
            axis = axes[row_index, column]
            for r in rows:
                if r["month"] != month or r["target"] != target:
                    continue
                with np.load(output / f"{r['session']}_diagnostics.npz") as archive:
                    axis.plot(archive["time_ms"], archive["common_erp_uv"][0, 3], label=r["session"][5:20])
                    axis.plot(archive["time_ms"], archive["common_erp_uv"][1, 3], linestyle="--", alpha=.8)
            axis.axhline(0, color=".5", linewidth=.6)
            axis.axvline(0, color=".5", linewidth=.6)
            axis.grid(alpha=.15)
            axis.set(title=f"{month}, target {target}", xlabel="Nominal epoch time (ms)", ylabel="Pz target - nontarget (uV)")
            axis.legend(fontsize=7)
    figure.savefig(output / "matched_target_erp.png", dpi=140, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--accepted-execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    batch_path, accepted_path = args.batch_manifest.resolve(), args.accepted_execution.resolve()
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    profile_path = accepted_path.parent / "continuous_reviewed_native7.json"
    profile = load_continuous_qc(profile_path)
    source_paths = [Path(__file__), REPO / "scripts/analyze_brainsync_artifact_remedies.py",
                    REPO / "src/data/preprocess.py", REPO / "src/data/continuous_qc.py", REPO / "src/data/bidirectional.py",
                    REPO / "src/data/brainsync.py", REPO / "src/data/brainsync_contract.py"]
    sources = {p.relative_to(REPO).as_posix(): file_sha256(p) for p in source_paths}
    result = {"schema": "brainsync_month_quality_comparison/1", "started_utc": datetime.now(UTC).isoformat(),
        "role": "retrospective_same_person_descriptive_comparison",
        "bindings": {str(p): file_sha256(p) for p in (batch_path, accepted_path, profile_path)},
        "source_sha256": sources, "python": sys.version,
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "mne", "matplotlib")},
        "protocol": {"basic": BRAIN_SYNC_PREPROCESSING.record(),
            "common_rule": "Same frozen500uV coarse0.2s changes plus0.2s guards, PTP>500uV channel-epoch proposals, original-axis cuts, same frozen two-boundary1uV/inflation2 reference, >=5of7 channels; no named persistent Fz exclusions.",
            "identity": "User confirmed same person; original P001/P003 IDs retained. No predictor admission/identity gate or existing reviewed roster is changed.",
            "inference": "No classifier. Existing October profile unchanged. Same-rule September replay is diagnostic, not new admitted QC.",
            "erp": "Pz target minus nontarget250<=t<600ms; all observed retained channels,20percent trimmed diagnostic; odd/even block waveform correlation.2000 seeded whole-block bootstrap draws of pooled class sums/counts.",
            "bootstrap_scope": "Within-session repeatability conditional on these ten repeated blocks and selection; not independent human/choice or population uncertainty.",
            "independence": "One person; nonrandom months and target compositions. October reference session is in calibration; summarize sensitivity excluding it.",
            "exclusion": "14:45 remains unusable and stays in October operational denominator, but excluded from eligible-signal distribution summaries."}, "sessions": []}
    result["acquisition_roster"] = {}
    for month, prefix in (("September", "202609"), ("October", "202610")):
        selected = [r for r in batch["sessions"] if r["directory_id"][5:].startswith(prefix)]
        result["acquisition_roster"][month] = {"directories": len(selected),
            "complete": sum(r["status"] == "completed" for r in selected),
            "failed_or_incomplete": sum(r["status"] != "completed" for r in selected),
            "whole_record_excluded": sum(r["directory_id"] == EXCLUDED for r in selected)}
    entries = [r for r in batch["sessions"] if r.get("extra") and r["directory_id"] != EXCLUDED]
    with threadpool_limits(limits=2):
        for i, entry in enumerate(entries):
            print("COMPARE", entry["directory_id"], flush=True)
            current = next((r for r in accepted["sessions"] if r["session"] == entry["directory_id"] and r.get("cache")), None)
            row = analyze(entry, profile, current, output, 20261030 + 100 * i)
            result["sessions"].append(row)
            print("RESULT", row["session"], "common", row["common_rule"]["retained_epochs"], flush=True)
    result["summary"] = summarize(result["sessions"])
    result["sensitivity_excluding_october_reference"] = summarize([r for r in result["sessions"] if not r["reference_session_in_boundary_fit"]])
    result["october_actual"] = {"retained_epochs": accepted["retained_epochs_total"], "eligible_scheduled": 3150,
                                "all_requested_including_excluded": 3600, "excluded_session": EXCLUDED}
    plots(result["sessions"], output)
    with (output / "sessions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = ["session", "original_subject_id", "month", "target", "scheduled", "base", "common_rule", "october_accepted", "base_fz_median_ptp_uv", "base_pz_median_ptp_uv", "common_pz_250_600_uv", "common_pz_trimmed20_uv", "common_pz_odd_even_r"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in result["sessions"]:
            erp = r["common_rule"]["channels"][3]["erp"]
            writer.writerow({**{k: r[k] for k in columns[:5]}, "base": r["base"]["retained_epochs"],
                "common_rule": r["common_rule"]["retained_epochs"], "october_accepted": r.get("october_accepted", {}).get("retained_epochs"),
                "base_fz_median_ptp_uv": r["base"]["channels"][0]["median_ptp_uv"], "base_pz_median_ptp_uv": r["base"]["channels"][3]["median_ptp_uv"],
                "common_pz_250_600_uv": erp["mean_250_600_uv"], "common_pz_trimmed20_uv": erp["trimmed20_250_600_uv"],
                "common_pz_odd_even_r": erp["odd_even_response_correlation"]})
    assert all(file_sha256(REPO / p) == digest for p, digest in sources.items())
    result.update(finished_utc=datetime.now(UTC).isoformat(), original_inputs_unchanged=True,
                  october_current_data_and_masks_exact=True, output_sha256={p.name: file_sha256(p) for p in output.iterdir() if p.is_file()})
    atomic_write_json(output / "comparison.json", result)
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
