"""Follow up a batch-quality manifest with channel similarity and line-noise replay."""

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
from scipy.signal import welch
from threadpoolctl import threadpool_limits

from data.brainsync import BRAIN_SYNC_PREPROCESSING, load_brainsync_inference_session
from data.brainsync_contract import validate_brainsync_bids_recording
from data.preprocess import filter_continuous
from research.recording_quality import file_sha256


def spectrum(values, fs, segments):
    total, weight = None, 0
    size = round(2 * fs)
    for a, b in segments:
        if b - a < size:
            continue
        f, p = welch(values[:, a:b], fs=fs, nperseg=size, axis=-1)
        n = 1 + (b - a - size) // (size // 2)
        total = p * n if total is None else total + p * n
        weight += n
    return f, total / weight


def run_row(row, output):
    root = Path(row["input_root"])
    d = load_brainsync_inference_session(root)
    x = d.X - d.X.mean(axis=-1, keepdims=True)
    cross = np.einsum("ect,edt->ecd", x, x)
    energy = np.einsum("ect,ect->ec", x, x)
    correlations = cross / np.sqrt(energy[:, :, None] * energy[:, None, :])
    pairs = []
    for i, left in enumerate(d.channel_names):
        for j in range(i + 1, d.n_channels):
            corr = correlations[:, i, j]
            pairs.append({"left": left, "right": d.channel_names[j],
                          "median_r": float(np.median(corr)), "p95_r": float(np.quantile(corr, .95)),
                          "epochs_r_above_095": int((corr > .95).sum()),
                          "epochs_r_above_099": int((corr > .99).sum()),
                          "median_difference_rms_uv": float(np.median(np.sqrt(np.mean((x[:, i] - x[:, j]) ** 2, axis=-1))) * 1e6)})
    v = validate_brainsync_bids_recording(root)
    try:
        raw = mne.io.read_raw_edf(v.bids.raw_path, preload=True, verbose="ERROR")
        try:
            raw.set_channel_types(dict(zip(raw.ch_names, (t.lower() for t in v.bids.channel_types), strict=True)))
            fs = float(raw.info["sfreq"])
            count = int((v.manifest["recording"].get("clock") or {}).get("acquired_samples", raw.n_times))
            if count < raw.n_times:
                raw.crop(tmax=(count - 1) / fs)
            boundaries = list(v.bids.acquisition_boundaries)
            if boundaries:
                raw.set_annotations(raw.annotations + mne.Annotations(boundaries, [0.] * len(boundaries),
                                                                       ["EDGE boundary"] * len(boundaries), orig_time=raw.annotations.orig_time))
            picks = [i for i, t in enumerate(v.bids.channel_types) if t.lower() == "eeg"]
            cuts = [0, *(round(t * fs) for t in boundaries), count]
            segments = list(zip(cuts[:-1], cuts[1:], strict=True))
            f, before = spectrum(raw.get_data(picks=picks), fs, segments)
            spec = BRAIN_SYNC_PREPROCESSING
            filter_continuous(raw, l_freq=spec.l_freq, h_freq=spec.h_freq, method=spec.filter_method,
                              phase=spec.filter_phase, causal_iir_initial_state=spec.causal_iir_initial_state,
                              order=spec.filter_order, notch_freqs_hz=spec.notch_freqs_hz,
                              notch_quality_factor=spec.notch_quality_factor,
                              zero_phase_edge_policy=spec.zero_phase_edge_policy)
            _, after = spectrum(raw.get_data(picks=picks), fs, segments)
            df = f[1] - f[0]
            band = (f >= 49) & (f < 51)
            line = [{"name": raw.ch_names[index],
                     "raw_49_51_rms_uv": float(np.sqrt(before[k, band].sum() * df)) * 1e6,
                     "filtered_49_51_rms_uv": float(np.sqrt(after[k, band].sum() * df)) * 1e6}
                    for k, index in enumerate(picks)]
        finally:
            raw.close()
        v.bids.assert_content_unchanged()
    finally:
        v.bids.close()
    assert all(file_sha256(root / p) == digest for p, digest in row["input_sha256"].items())
    np.savez_compressed(output / f"{row['directory_id']}_correlations.npz", epoch_correlations=correlations,
                        channel_names=np.asarray(d.channel_names))
    print(row["directory_id"], [(p["left"], p["right"], round(p["median_r"], 4)) for p in pairs if p["median_r"] > .95], flush=True)
    return {"session": row["directory_id"], "group": row["group"], "epochs": d.n_epochs, "pairs": pairs, "line_noise": line}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.batch_manifest.resolve()
    parent_sha = file_sha256(manifest_path)
    source_sha = file_sha256(Path(__file__))
    parent = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    with threadpool_limits(limits=2):
        for row in parent["sessions"]:
            if row.get("extra"):
                rows.append(run_row(row, output))
    figure, axes = plt.subplots(1, 3, figsize=(13, 7.5), layout="constrained", sharey=True)
    labels = [r["session"].removeprefix("P001_").removesuffix("_01") for r in rows]
    for axis, pair in zip(axes, (("FZ", "CZ"), ("CZ", "PZ"), ("P3", "P4")), strict=True):
        values = [next(p["median_r"] for p in r["pairs"] if (p["left"], p["right"]) == pair) for r in rows]
        axis.scatter(values, range(len(rows)), marker="o", color="#0072b2")
        for i, value in enumerate(values):
            axis.annotate(f"{value:.3f}", (value, i), xytext=(4, 5), textcoords="offset points", fontsize=8)
        axis.set(xlim=(-.1, 1.15), xlabel="Median within-epoch Pearson r", title=" / ".join(pair))
        axis.axvline(.95, color=".5", linestyle="--", linewidth=.7)
        axis.grid(alpha=.15)
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].invert_yaxis()
    figure.savefig(output / "channel_similarity.png", dpi=150, facecolor="white")
    plt.close(figure)
    assert file_sha256(manifest_path) == parent_sha
    assert file_sha256(Path(__file__)) == source_sha
    result = {"schema": "brainsync_channel_followup/1", "created_utc": datetime.now(UTC).isoformat(),
              "batch_manifest": str(manifest_path), "batch_manifest_sha256": parent_sha,
              "script_sha256": source_sha, "fit_performed": False, "labels_used": False,
              "methods": {"correlations": "Each epoch/channel centered over time; Pearson r computed within each epoch; median across all available epochs. High correlation is diagnostic, not proof of electrode bridging or duplicated ADC channels.",
                          "line_noise": "Production native continuous 50Hz notch and 0.1-30Hz bidirectional EEG filtering, same actual acquisition boundaries; 2s/50%-overlap Welch bin sum in [49,51)Hz before epoching/resampling. EOG is not filtered or reclassified.",
                          "thresholds": "r>.95 and r>.99 describe sensitivity; no channel rejection or correction."},
              "input_files_unchanged": True, "sessions": rows,
              "outputs": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "channel_details.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
