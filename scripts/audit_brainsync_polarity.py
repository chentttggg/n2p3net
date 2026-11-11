"""Independent EDF decoding, preprocessing stages and polarity controls.

--backend-src selects an isolated frozen backend for same-input old/new replay.
All outputs are diagnostic; no raw file, QC roster or model is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode_edf(path):
    """Parse native signed little-endian words and affine physical map, without MNE."""
    import numpy as np

    with Path(path).open("rb") as stream:
        fixed = stream.read(256)
        header_length = int(fixed[184:192])
        records = int(fixed[236:244])
        duration = float(fixed[244:252])
        channels = int(fixed[252:256])
        rest = stream.read(header_length - 256)
        cursor = 0

        def fields(width):
            nonlocal cursor
            result = [rest[cursor + i * width:cursor + (i + 1) * width].decode("latin1").strip() for i in range(channels)]
            cursor += width * channels
            return result

        labels = fields(16)
        fields(80)
        units = fields(8)
        physical_min = np.asarray(list(map(float, fields(8))))
        physical_max = np.asarray(list(map(float, fields(8))))
        digital_min = np.asarray(list(map(float, fields(8))))
        digital_max = np.asarray(list(map(float, fields(8))))
        fields(80)
        sizes = np.asarray(list(map(int, fields(8))))
        fields(32)
        words = np.frombuffer(stream.read(), dtype="<i2")
    assert words.size == records * sizes.sum()
    words = words.reshape(records, sizes.sum())
    offsets = np.r_[0, np.cumsum(sizes)]
    selected = [i for i, label in enumerate(labels) if "Annotations" not in label]
    values, summary = [], []
    for i in selected:
        gain = (physical_max[i] - physical_min[i]) / (digital_max[i] - digital_min[i])
        offset = physical_min[i] - gain * digital_min[i]
        digital = words[:, offsets[i]:offsets[i + 1]].reshape(-1)
        unit = units[i].replace("µ", "u").replace("μ", "u")
        factor = {"uV": 1e-6, "mV": 1e-3, "V": 1.}[unit]
        values.append((digital.astype(float) * gain + offset) * factor)
        summary.append({"name": labels[i], "unit": units[i], "physical_min": physical_min[i],
                        "physical_max": physical_max[i], "digital_min": digital_min[i], "digital_max": digital_max[i],
                        "physical_units_per_digital_step": gain, "sample_rate_hz": sizes[i] / duration})
    return np.asarray(values), summary


def synthetic_filter_controls(output):
    import mne
    import numpy as np

    from data.brainsync import BRAIN_SYNC_PREPROCESSING
    from data.preprocess import apply_trial_baseline, filter_continuous

    fs = 250.
    spec = BRAIN_SYNC_PREPROCESSING
    n = round(120 * fs)
    center = round(60 * fs)
    time = (np.arange(n) - center) / fs
    positive = 5e-6 * np.exp(-.5 * ((time - .35) / .08) ** 2)
    late_positive = 5e-6 * np.exp(-.5 * ((time - .75) / .12) ** 2)
    raw = mne.io.RawArray(np.stack((positive, late_positive, -late_positive)),
                         mne.create_info(["Fz", "Cz", "Pz"], fs, ch_types="eeg"), verbose=False)
    options = dict(l_freq=spec.l_freq, h_freq=spec.h_freq, method=spec.filter_method, phase=spec.filter_phase,
        causal_iir_initial_state=spec.causal_iir_initial_state, order=spec.filter_order,
        notch_freqs_hz=spec.notch_freqs_hz, notch_quality_factor=spec.notch_quality_factor,
        zero_phase_edge_policy=spec.zero_phase_edge_policy)
    filter_continuous(raw, **options)
    filtered = raw.get_data()
    events = np.asarray([[center, 0, 1]])
    epochs = mne.Epochs(raw, events, event_id={"probe": 1}, tmin=-.2, tmax=1.2,
                        baseline=None, preload=True, verbose=False)
    epochs.resample(128., method="fft", npad="auto", window="auto", pad="edge", verbose=False)
    epoch = epochs.get_data()[..., :179]
    epoch = apply_trial_baseline(epoch, sfreq=128., tmin=-.2, baseline_mode="mean_only")
    t = -.2 + np.arange(179) / 128
    assert epoch[0, 0].max() > 0 and epoch[0, 1].max() > 0 and epoch[0, 2].min() < 0
    assert np.max(np.abs(epoch[0, 1] + epoch[0, 2])) < 1e-18
    flipped = mne.io.RawArray(-np.stack((positive, late_positive, -late_positive)),
                             mne.create_info(["Fz", "Cz", "Pz"], fs, ch_types="eeg"), verbose=False)
    filter_continuous(flipped, **options)
    assert np.max(np.abs(flipped.get_data() + filtered)) < 1e-18
    np.savez_compressed(output / "synthetic.npz", time_s=t, epoch_v=epoch,
                        raw_time_s=time, input_v=np.stack((positive, late_positive, -late_positive)), filtered_v=filtered)
    return {"positive_350ms_peak_uv": float(epoch[0, 0].max() * 1e6),
            "positive_750ms_peak_uv": float(epoch[0, 1].max() * 1e6),
            "negative_750ms_trough_uv": float(epoch[0, 2].min() * 1e6),
            "positive_350ms_response_mean_600_1000_uv": float(epoch[0, 0, (t >= .6) & (t < 1.)].mean() * 1e6),
            "known_positive_and_negative_polarities_preserved": True,
            "sign_inversion_negative_control_distinguished": True, "filter_odd_symmetry_verified": True}


def run(args):
    import importlib.metadata

    import mne
    import numpy as np
    from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt
    from threadpoolctl import threadpool_limits

    from data.brainsync import (
        BRAIN_SYNC_PREPROCESSING,
        load_brainsync_inference_session,
        load_brainsync_session,
    )
    from data.brainsync_contract import validate_brainsync_bids_session

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    batch_path = args.batch_manifest.resolve()
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    record = {"schema": "brainsync_polarity_audit/1", "backend_src": str(args.backend_src.resolve()),
              "batch_manifest_sha256": sha(batch_path), "script_sha256": sha(Path(__file__)),
              "source_sha256": {name: sha(Path(__import__(name, fromlist=["_"]).__file__)) for name in
                                ("data.brainsync", "data.brainsync_contract", "data.preprocess", "data.continuous_qc", "data.personal_quality")},
              "packages": {p: importlib.metadata.version(p) for p in ("mne", "numpy", "scipy")},
              "preprocessing": asdict(BRAIN_SYNC_PREPROCESSING),
              "role": "polarity_debugging_same_current_dependencies_not_historical_environment_reenactment", "sessions": []}
    with threadpool_limits(limits=2):
        record["synthetic"] = synthetic_filter_controls(output)
        for entry in batch["sessions"]:
            if not entry.get("extra") or entry["directory_id"] == "P001_20261030_144501_01":
                continue
            root = Path(entry["input_root"])
            print("POLARITY", root.name, flush=True)
            validated = validate_brainsync_bids_session(root)
            raw = mne.io.read_raw_edf(validated.bids.raw_path, preload=True, verbose=False)
            try:
                independent, header = decode_edf(validated.bids.raw_path)
                values = raw.get_data()
                maximum_decode_difference = float(np.max(np.abs(independent - values)))
                assert maximum_decode_difference < 1e-12
                assert all(row["physical_units_per_digital_step"] > 0 for row in header)
                eeg = [i for i, kind in enumerate(validated.bids.channel_types) if kind == "EEG"]
                labels = np.asarray([int(m.candidate_id) == validated.thought_digit for m in validated.bids.stimuli])
                onsets = np.asarray([m.sample for m in validated.bids.stimuli])
                fs = float(raw.info["sfreq"])
                d = load_brainsync_inference_session(root)
                labelled = load_brainsync_session(root)
                full_indices = np.asarray(d.event_timeline.evidence_indices)
                available = full_indices >= 0
                ordered_labels = labels[available]
                assert np.array_equal(full_indices[available], np.arange(d.n_epochs))
                assert np.array_equal(labelled.X, d.X)
                assert np.array_equal(labelled.y, ordered_labels.astype(int))
                cuts = [0, *(round(t * fs) for t in validated.bids.acquisition_boundaries), int(raw.n_times)]
                stages = {}
                native_width = round(1.2 * fs) - round(-.2 * fs) + 1
                source_t = np.arange(native_width) / fs - .2
                baseline = source_t < -1e-12
                coarse_before_fft = []
                for stage, cutoff, phase in (("raw_no_filter", None, None), ("lowpass30_only_zero", None, "zero"),
                                              ("band01_30_forward_no_notch", .1, "forward"), ("band01_30_zero_no_notch", .1, "zero")):
                    transformed = values[eeg].copy()
                    if phase is not None:
                        sos = butter(4, 30 if cutoff is None else [cutoff, 30], btype="lowpass" if cutoff is None else "bandpass", fs=fs, output="sos")
                        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
                            if phase == "zero":
                                transformed[:, a:b] = sosfiltfilt(sos, transformed[:, a:b], axis=-1)
                            else:
                                initial = sosfilt_zi(sos)[:, None, :] * transformed[None, :, a, None]
                                transformed[:, a:b] = sosfilt(sos, transformed[:, a:b], axis=-1, zi=initial)[0]
                    epochs = np.stack([transformed[:, onset - round(.2 * fs):onset - round(.2 * fs) + native_width] for onset in onsets[available]])
                    baselined = epochs - epochs[:, :, baseline].mean(axis=-1, keepdims=True)
                    means_t, means_n = baselined[ordered_labels].mean(axis=0), baselined[~ordered_labels].mean(axis=0)
                    diff = means_t - means_n
                    stages[stage] = {"target_late_uv": (means_t[:, (source_t >= .6) & (source_t < 1.)].mean(axis=-1) * 1e6).tolist(),
                                     "nontarget_late_uv": (means_n[:, (source_t >= .6) & (source_t < 1.)].mean(axis=-1) * 1e6).tolist(),
                                     "difference_late_uv": (diff[:, (source_t >= .6) & (source_t < 1.)].mean(axis=-1) * 1e6).tolist()}
                    coarse_before_fft.append(diff * 1e6)
                x = d.X.astype(float)
                model_time = -.2 + np.arange(d.n_times) / d.preprocessing.sfreq
                target = x[ordered_labels].mean(axis=0) * 1e6
                nontarget = x[~ordered_labels].mean(axis=0) * 1e6
                difference = target - nontarget
                reverse_difference = nontarget - target
                assert np.array_equal(reverse_difference, -difference)
                stages["production_basic"] = {"target_late_uv": target[:, (model_time >= .6) & (model_time < 1.)].mean(axis=-1).tolist(),
                                               "nontarget_late_uv": nontarget[:, (model_time >= .6) & (model_time < 1.)].mean(axis=-1).tolist(),
                                               "difference_late_uv": difference[:, (model_time >= .6) & (model_time < 1.)].mean(axis=-1).tolist()}
                common_path = Path(args.common_comparison).resolve() / f"{root.name}_diagnostics.npz"
                with np.load(common_path) as arr:
                    stages["common_qc"] = {"difference_late_uv": arr["common_erp_uv"][0, :, (model_time >= .6) & (model_time < 1.)].mean(axis=0).tolist()}
                np.savez_compressed(output / f"{root.name}.npz", model_time_s=model_time,
                    model_target_uv=target, model_nontarget_uv=nontarget, model_difference_uv=difference,
                    X=d.X, native_time_s=source_t, no_fft_stage_difference_uv=np.asarray(coarse_before_fft))
                source = root / validated.manifest["recording"]["source_path"]
                source_equal = source.is_file() and sha(source) == sha(validated.bids.raw_path)
                row = {"session": root.name, "target": validated.thought_digit, "subject": validated.subject_id,
                       "source_reference": validated.bids.source_reference, "header": header,
                       "independent_edf_decode_max_difference_v": maximum_decode_difference,
                       "all_edf_affine_gains_positive": True, "source_bids_bytes_equal": source_equal,
                       "actual_label_y_equals_digit_equal_confirmed_target": True,
                       "labelled_unlabelled_X_exact": True, "target_count_in_available": int(ordered_labels.sum()),
                       "nontarget_count_in_available": int((~ordered_labels).sum()), "stages": stages}
                record["sessions"].append(row)
                assert all(sha(root / path) == digest for path, digest in entry["input_sha256"].items())
                if hasattr(validated.bids, "assert_content_unchanged"):
                    validated.bids.assert_content_unchanged()
                print("LATE_PZ", root.name, {k: round(v["difference_late_uv"][3], 3) for k, v in stages.items()}, flush=True)
            finally:
                raw.close()
                if hasattr(validated.bids, "close"):
                    validated.bids.close()
    record["outputs"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    (output / "audit.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-src", type=Path, required=True)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--common-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.backend_src.resolve()))
    run(args)


if __name__ == "__main__":
    main()
