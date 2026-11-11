"""Replay proposed local artifact annotations on the unchanged native sample axis.

This is a diagnostic split-filter experiment. Guards are sensitivity probes, not
empirical recovery certificates or production QC acceptance decisions.
"""

from __future__ import annotations

import argparse
import json
import os
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
from analyze_brainsync_artifact_remedies import epoch_values, filter_values, q
from threadpoolctl import threadpool_limits

from data.brainsync import BRAIN_SYNC_PREPROCESSING
from data.brainsync_contract import validate_brainsync_bids_recording
from data.preprocess import continuous_filter_plan, filter_continuous
from research.recording_quality import file_sha256

REPO = Path(__file__).resolve().parents[1]
GUARDS = (0., 2., 5., 10., 20., 30.)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def synthetic_split():
    fs = 250.
    n = round(240 * fs)
    t = np.arange(n) / fs
    clean = 4e-3 * np.sin(2 * np.pi * t / 240) + 12e-6 * np.sin(2 * np.pi * 10 * t)
    onsets = np.arange(round(60 * fs), round(180 * fs), round(1.5 * fs))
    for onset in onsets[::9]:
        pulse_t = np.arange(round(1.2 * fs)) / fs
        clean[onset:onset + len(pulse_t)] += 5e-6 * np.exp(-.5 * ((pulse_t - .35) / .08) ** 2)
    bad = clean.copy()
    bad[round(120 * fs):round(120.2 * fs)] += .080
    spec = BRAIN_SYNC_PREPROCESSING
    clean_epoch = epoch_values(filter_values(clean[None], fs, [(0, n)], spec), fs, onsets, spec)[:, 0]
    bad_epoch = epoch_values(filter_values(bad[None], fs, [(0, n)], spec), fs, onsets, spec)[:, 0]
    a, b = round(119 * fs), round(121 * fs)
    isolated = filter_values(bad[None], fs, [(0, a), (b, n)], spec)
    isolated_epoch = epoch_values(isolated, fs, onsets, spec)[:, 0]
    rows = []
    for guard in GUARDS:
        valid = (onsets + round(1.2 * fs) + 1 <= a - guard * fs) | (onsets + round(-.2 * fs) >= b + guard * fs)
        errors = np.max(np.abs(isolated_epoch[valid] - clean_epoch[valid]), axis=-1) * 1e6
        old_errors = np.max(np.abs(bad_epoch[valid] - clean_epoch[valid]), axis=-1) * 1e6
        rows.append({"guard_s": guard, "available_epochs_of80": int(valid.sum()),
                     "split_vs_known_clean_epoch_max_error_uv": q(errors),
                     "unsplit_vs_known_clean_same_epochs_max_error_uv": q(old_errors),
                     "split_epochs_error_above1uv": int((errors > 1).sum())})
    assert rows[-1]["split_vs_known_clean_epoch_max_error_uv"]["max"] < rows[0]["split_vs_known_clean_epoch_max_error_uv"]["max"]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remedy-manifest", type=Path, required=True)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    remedy_path, batch_path = args.remedy_manifest.resolve(), args.batch_manifest.resolve()
    remedy = json.loads(remedy_path.read_text(encoding="utf-8"))
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    sources = [Path(__file__), REPO / "scripts/analyze_brainsync_artifact_remedies.py", REPO / "src/data/preprocess.py"]
    source_sha = {str(p.relative_to(REPO)): file_sha256(p) for p in sources}
    manifest = {"schema": "brainsync_split_filter_probe/1", "created_utc": datetime.now(UTC).isoformat(),
                "parent_manifest_sha256": {str(p): file_sha256(p) for p in (remedy_path, batch_path)},
                "source_sha256": source_sha,
                "role": "exploratory_split_filter_probe_not_recovery_certification",
                "annotation_contract": "Original sample indices retained. Channel-local QUALITY_LOCAL_ARTIFACT annotations; each actual clean segment filtered separately by production filter_continuous. No samples deleted, shifted, reconstructed, or accepted from the gaps.",
                "guards_s": GUARDS, "synthetic": synthetic_split(), "sessions": []}
    with threadpool_limits(limits=2):
        for result in remedy["sessions"]:
            entry = next(r for r in batch["sessions"] if r["directory_id"] == result["session"])
            root = Path(entry["input_root"])
            validated = validate_brainsync_bids_recording(root)
            try:
                raw = mne.io.read_raw_edf(validated.bids.raw_path, preload=True, verbose="ERROR")
                try:
                    raw.set_channel_types(dict(zip(raw.ch_names, (t.lower() for t in validated.bids.channel_types), strict=True)))
                    raw.pick("eeg")
                    fs = float(raw.info["sfreq"])
                    count = int(validated.manifest["recording"]["clock"]["acquired_samples"])
                    if count < raw.n_times:
                        raw.crop(tmax=(count - 1) / fs)
                    boundaries = list(validated.bids.acquisition_boundaries)
                    if boundaries:
                        raw.set_annotations(raw.annotations + mne.Annotations(boundaries, [0.] * len(boundaries), ["EDGE boundary"] * len(boundaries), orig_time=raw.annotations.orig_time))
                    raw_names = {name.upper(): name for name in raw.ch_names}
                    intervals = result["candidate_intervals"]
                    if intervals:
                        raw.annotations.append([r["start_s"] for r in intervals], [r["stop_s"] - r["start_s"] for r in intervals],
                                               ["QUALITY_LOCAL_ARTIFACT_diagnostic"] * len(intervals),
                                               ch_names=[(raw_names[r["channel_name"]],) for r in intervals])
                    plans = {name: continuous_filter_plan(raw, name) for name in raw.ch_names}
                    spec = BRAIN_SYNC_PREPROCESSING
                    filter_continuous(raw, l_freq=spec.l_freq, h_freq=spec.h_freq, method=spec.filter_method, phase=spec.filter_phase,
                                      causal_iir_initial_state=spec.causal_iir_initial_state, order=spec.filter_order,
                                      notch_freqs_hz=spec.notch_freqs_hz, notch_quality_factor=spec.notch_quality_factor,
                                      zero_phase_edge_policy=spec.zero_phase_edge_policy)
                    with np.load(remedy_path.parent / f"{result['session']}_arrays.npz") as arr:
                        onsets = np.rint(arr["onsets_s"] * fs).astype(int)
                        original_ptp = arr["epoch_ptp_uv"].copy()
                        labels = arr["labels"].copy()
                    epochs = epoch_values(raw.get_data(), fs, onsets, spec)
                    stats = []
                    mask20 = None
                    masks = {}
                    for guard in GUARDS:
                        valid = np.zeros(epochs.shape[:2], dtype=bool)
                        for channel, name in enumerate(raw.ch_names):
                            for segment in plans[name]:
                                left = segment.start_sample
                                right = segment.stop_sample
                                if "artifact" in segment.left_causes:
                                    left += round(guard * fs)
                                if "artifact" in segment.right_causes:
                                    right -= round(guard * fs)
                                valid[:, channel] |= ((onsets + round(-.2 * fs) >= left)
                                                      & (onsets + round(1.2 * fs) + 1 <= right))
                        ptp = np.ptp(epochs, axis=-1) * 1e6
                        channel_stats = [{"name": name.upper(), "available_epochs": int(valid[:, i].sum()),
                                          "unsplit_ptp_uv_on_same_valid_epochs": q(original_ptp[valid[:, i], i]),
                                          "split_ptp_uv_on_valid_epochs": q(ptp[valid[:, i], i])}
                                         for i, name in enumerate(raw.ch_names)]
                        stats.append({"guard_s": guard, "all7_available_epochs": int(valid.all(axis=1).sum()),
                                      "at_least5_available_epochs": int((valid.sum(axis=1) >= 5).sum()),
                                      "valid_channel_epochs_above200uv": int(((ptp > 200) & valid).sum()),
                                      "unsplit_same_valid_channel_epochs_above200uv": int(((original_ptp > 200) & valid).sum()),
                                      "channels": channel_stats})
                        masks[str(guard)] = valid
                        if guard == 20:
                            mask20 = valid.copy()
                    accepted_values = np.where(mask20[..., None], epochs, np.nan)
                    assert np.isnan(accepted_values[~mask20]).all()
                    np.savez_compressed(output / f"{result['session']}_diagnostic_epochs.npz",
                                        X_v=accepted_values, trial_channel_mask=mask20, labels=labels,
                                        channel_names=np.asarray(raw.ch_names), original_onset_samples=onsets,
                                        scheduled_trials=np.asarray(result["scheduled"]),
                                        use_status=np.asarray("diagnostic_only_unknown_boundary_recovery"))
                    row = {"session": result["session"], "scheduled": result["scheduled"], "basic_epochs": result["basic_epochs"],
                           "guard_comparisons": stats, "channel_plans": {name: [s.record() for s in spans] for name, spans in plans.items()},
                           "artifact_boundary_segments_requiring_recovery": sum(s.requires_recovery for spans in plans.values() for s in spans)}
                    manifest["sessions"].append(row)
                    write_json(output / f"{result['session']}_results.json", row)
                    assert all(file_sha256(root / p) == digest for p, digest in entry["input_sha256"].items())
                    validated.bids.assert_content_unchanged()
                    print(result["session"], [(s["guard_s"], s["unsplit_same_valid_channel_epochs_above200uv"], s["valid_channel_epochs_above200uv"]) for s in stats], flush=True)
                finally:
                    raw.close()
            finally:
                validated.bids.close()
    assert all(file_sha256(REPO / p) == sha for p, sha in source_sha.items())
    assert all(file_sha256(Path(p)) == sha for p, sha in manifest["parent_manifest_sha256"].items())
    manifest["finished_utc"] = datetime.now(UTC).isoformat()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    rows = manifest["sessions"]
    positions = np.arange(len(rows))
    selected = [r["guard_comparisons"][3] for r in rows]  # 10 s is one plotted sensitivity, not a selected rule.
    axes[0].barh(positions - .18, [r["unsplit_same_valid_channel_epochs_above200uv"] for r in selected], height=.35, color="#d55e00", label="Unsplit")
    axes[0].barh(positions + .18, [r["valid_channel_epochs_above200uv"] for r in selected], height=.35, color="#0072b2", label="Split before filtering")
    axes[0].set_yticks(positions, [r["session"].removeprefix("P001_").removesuffix("_01") for r in rows])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Channel-epochs above 200uV, same 10s-guard support", title="Measured split-filter effect; no clean-ground-truth claim")
    axes[0].legend()
    synthetic = manifest["synthetic"]
    axes[1].plot(GUARDS, [r["split_vs_known_clean_epoch_max_error_uv"]["max"] for r in synthetic], marker="o", label="Split: maximum error")
    axes[1].plot(GUARDS, [r["unsplit_vs_known_clean_same_epochs_max_error_uv"]["max"] for r in synthetic], marker="s", label="Unsplit: same retained epochs")
    axes[1].set(xlabel="Guard from annotated bad interval (s)", ylabel="Maximum epoch error (uV)", title="Known 80mV artifact + known clean signal")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.15)
    figure.savefig(output / "split_filter_comparison.png", dpi=145, facecolor="white")
    plt.close(figure)
    manifest["output_sha256"] = {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
