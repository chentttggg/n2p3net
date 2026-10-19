"""Challenge conditional correction using known ERP increments, not EEG labels."""
from __future__ import annotations

import hashlib
import json

import mne
import numpy as np
from probe import HERE, PLAN, freeze_dependencies
from scipy.fft import dct
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


def main():
    execution = freeze_dependencies()
    from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract
    from data.preprocess import (
        design_continuous_sos,
        filter_bidirectional_segment,
        source_epoch_support,
        transform_epoch_samples,
    )
    draws = np.load(HERE / "projected_draws.npz", allow_pickle=False)
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    fs = 256.
    sos = design_continuous_sos(fs, l_freq=spec.l_freq, h_freq=spec.h_freq,
        order=spec.filter_order, notch_freqs_hz=spec.notch_freqs_hz, notch_quality_factor=spec.notch_quality_factor)
    pad = int(mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)["padlen"])
    onset = round(100 * fs)
    samples = round(259.0625 * fs)
    starts, stops = source_epoch_support(np.array([[onset, 0, 1]]), fs, spec)
    start, stop = int(starts[0]), int(stops[0])

    def apply(x):
        return filter_bidirectional_segment(sos, x, padding_limit=pad).data

    def feature(x):
        return dct(transform_epoch_samples(x, fs, spec, require_linear=True), norm="ortho", axis=-1)[:, :8].reshape(-1)

    time = np.arange(samples) / fs - onset / fs
    rows = []
    for length in PLAN["cut_lengths_seconds"]:
        for distance in PLAN["epoch_distance_from_right_edge_seconds"]:
            end = stop + round(distance * fs)
            begin = end - round(length * fs)
            geometrical = (draws["lengths"] == length) & (draws["distances"] == distance)
            injections = []
            for latency in (.35, .8, 1.15):
                template = 10 * np.exp(-.5 * ((time - latency) / .12)**2) - 4 * np.exp(-.5 * ((time - latency + .18) / .06)**2)
                for topography_name, topography in (("posterior", np.array([.25, .5, 1., .3, .6, .6, .4, .4])),
                                                    ("pz_only", np.eye(8)[2])):
                    signal = topography[:, None] * template
                    reference = feature(apply(signal)[:, start:stop])
                    cut = signal[:, begin:end]
                    f = feature(apply(cut)[:, start - begin:stop - begin])
                    width = round(fs * .25)
                    left, right = cut[:, :width], cut[:, -width:]
                    context = np.concatenate([f, left.mean(axis=1), right.mean(axis=1),
                        (left[:, -1] - left[:, 0]) / .25,
                        (right[:, -1] - right[:, 0]) / .25])
                    injections.append((latency, topography_name, reference, f, context))
            for target in PLAN["subjects"]:
                training = geometrical & (draws["subjects"] != target)
                U = draws["context"][training]
                D = draws["finite"][training] - draws["long"][training]
                um, us = U.mean(axis=0), U.std(axis=0)
                us = np.maximum(us, max(float(np.median(us)), 1e-10) * .01)
                dm = D.mean(axis=0)
                model = Ridge(alpha=float(len(U)), fit_intercept=False)
                model.fit((U - um) / us, D - dm)
                for latency, topography, reference, f, context in injections:
                    # Counterfactual increment: intercept and baseline context cancel.
                    correction_increment = model.predict((context / us)[None])[0]
                    corrected = f - correction_increment
                    scale = np.linalg.norm(reference)
                    rows.append({"target": target, "length_s": length, "distance_s": distance,
                        "erp_peak_s": latency, "topography": topography,
                        "finite_relative_error": float(np.linalg.norm(f - reference) / scale),
                        "corrected_relative_error": float(np.linalg.norm(corrected - reference) / scale),
                        "corrected_reference_gain": float(corrected @ reference / (reference @ reference)),
                        "removed_increment_relative": float(np.linalg.norm(correction_increment) / scale)})
    summary = []
    for length in PLAN["cut_lengths_seconds"]:
        for distance in PLAN["epoch_distance_from_right_edge_seconds"]:
            for latency in (.35, .8, 1.15):
                selected = [r for r in rows if (r["length_s"], r["distance_s"], r["erp_peak_s"]) == (length, distance, latency)]
                summary.append({"length_s": length, "distance_s": distance, "erp_peak_s": latency,
                    "finite_mean_relative_error": float(np.mean([r["finite_relative_error"] for r in selected])),
                    "corrected_mean_relative_error": float(np.mean([r["corrected_relative_error"] for r in selected])),
                    "corrected_max_relative_error": max(r["corrected_relative_error"] for r in selected),
                    "minimum_reference_gain": min(r["corrected_reference_gain"] for r in selected)})
    report = {"scope": "known_ERP_increment_robustness_no_performance_claim", "execution": execution,
        "probe_sha256": hashlib.sha256((HERE / "probe_erp_increment.py").read_bytes()).hexdigest(),
        "draws_sha256": hashlib.sha256((HERE / "projected_draws.npz").read_bytes()).hexdigest(),
        "rows": rows, "summary": summary,
        "interpretation": "Low conditional boundary-difference MSE is insufficient; a correction must also preserve physically meaningful ERP increments across frozen topographies and latencies."}
    (HERE / "erp_increment_results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
