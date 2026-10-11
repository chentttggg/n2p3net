"""Equality-constrained conditional correction; frozen response protection."""
from __future__ import annotations

import hashlib
import json

import mne
import numpy as np
from probe import HERE, PLAN, freeze_dependencies
from scipy.fft import dct
from threadpoolctl import threadpool_limits


def run():
    execution = freeze_dependencies()
    from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract
    from data.preprocess import (
        design_continuous_sos,
        filter_bidirectional_segment,
        source_epoch_support,
        transform_epoch_samples,
    )
    config_bytes = (HERE / "constrained_plan.json").read_bytes()
    config = json.loads(config_bytes)
    draws = np.load(HERE / "projected_draws.npz", allow_pickle=False)
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    fs = 256.
    sos = design_continuous_sos(fs, l_freq=spec.l_freq, h_freq=spec.h_freq, order=spec.filter_order,
        notch_freqs_hz=spec.notch_freqs_hz, notch_quality_factor=spec.notch_quality_factor)
    pad = int(mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)["padlen"])
    onset, samples = round(100 * fs), round(259.0625 * fs)
    starts, stops = source_epoch_support(np.array([[onset, 0, 1]]), fs, spec)
    start, stop = int(starts[0]), int(stops[0])
    time = np.arange(samples) / fs - onset / fs
    dictionary = []
    for center in np.linspace(0., 1.2 - 1 / fs, 8):
        vector = 10 * np.exp(-.5 * ((time - center) / (1.2 / 7))**2)
        vector[(time < 0) | (time >= 1.2)] = 0
        dictionary.append(vector)
    pre = 10 * np.exp(-.5 * ((time + .1) / .08)**2)
    pre[(time < -.2) | (time >= 0)] = 0
    dictionary.extend([pre, np.full(samples, 10.), time / 25.90625])
    dictionary = np.stack(dictionary)
    rng = np.random.default_rng(20261004)

    def apply(values):
        return filter_bidirectional_segment(sos, values, padding_limit=pad).data

    def feature(values):
        return dct(transform_epoch_samples(values, fs, spec, require_linear=True), norm="ortho", axis=-1)[:, :8]

    reference_dictionary = feature(apply(dictionary)[:, start:stop])
    rows, robustness = [], []
    for length in PLAN["cut_lengths_seconds"]:
        for distance in PLAN["epoch_distance_from_right_edge_seconds"]:
            end, begin = stop + round(distance * fs), stop + round(distance * fs) - round(length * fs)
            width = round(fs * .25)

            def feature_context(values, begin=begin, width=width):
                # Input is arbitrary channel/batch rows. Output preserves leading row.
                f = feature(apply(values)[:, start - begin:stop - begin])
                left, right = values[:, :width], values[:, -width:]
                context = np.column_stack([f, left.mean(axis=1), right.mean(axis=1),
                    (left[:, -1] - left[:, 0]) / .25, (right[:, -1] - right[:, 0]) / .25])
                return f, context

            f_dictionary, c_dictionary = feature_context(dictionary[:, begin:end])
            # Scalar channel templates are lifted with exact native channel ordering.
            C, target = [], []
            for channel in range(8):
                for index in range(len(dictionary)):
                    feature_block = np.zeros((8, 8))
                    feature_block[channel] = c_dictionary[index, :8]
                    scalar_blocks = np.zeros((4, 8))
                    scalar_blocks[:, channel] = c_dictionary[index, 8:]
                    C.append(np.concatenate([feature_block.reshape(-1), scalar_blocks.reshape(-1)]))
                    desired = np.zeros((8, 8))
                    desired[channel] = f_dictionary[index] - reference_dictionary[index]
                    target.append(desired.reshape(-1))
            C, target = np.stack(C), np.stack(target)
            injections = []
            # Original six cases plus neighboring-flash, early and narrow waveform challenges.
            for latency, spread, label in ((.35, .12, "original_035"), (.8, .12, "original_080"),
                                           (1.15, .12, "original_115"), (-.15, .12, "neighbor_before"),
                                           (.85, .12, "neighbor_after"), (.05, .12, "early"),
                                           (.35, .04, "narrow")):
                wave = 10 * np.exp(-.5 * ((time - latency) / spread)**2) - 4 * np.exp(-.5 * ((time - latency + .18) / (spread / 2))**2)
                reference_scalar = feature(apply(wave[None])[:, start:stop])[0]
                f_scalar, c_scalar = feature_context(wave[None, begin:end])
                for topography_name, topography in (("posterior", np.array([.25, .5, 1., .3, .6, .6, .4, .4])),
                                                   ("pz_only", np.eye(8)[2])):
                    f = (topography[:, None] * f_scalar).reshape(-1)
                    ref = (topography[:, None] * reference_scalar).reshape(-1)
                    context = np.concatenate([f, (c_scalar[0, 8:, None] * topography).reshape(-1)])
                    injections.append((label, topography_name, f, ref, context))
            white = rng.normal(size=(32 * 8, end - begin))
            wf, wc = feature_context(white)
            wf = wf.reshape(32, 8, 8).reshape(32, 64)
            wc = wc.reshape(32, 8, 12)
            wc = np.concatenate([wc[:, :, :8].reshape(32, 64), wc[:, :, 8:].transpose(0, 2, 1).reshape(32, 32)], axis=1)
            geometry = (draws["lengths"] == length) & (draws["distances"] == distance)
            for participant in PLAN["subjects"]:
                train = geometry & (draws["subjects"] != participant)
                held = geometry & (draws["subjects"] == participant)
                U, D = draws["context"][train], draws["finite"][train] - draws["long"][train]
                ut, dt = draws["context"][held], draws["finite"][held] - draws["long"][held]
                um, us, dm = U.mean(axis=0), U.std(axis=0), D.mean(axis=0)
                us = np.maximum(us, max(float(np.median(us)), 1e-10) * .01)
                V = (U - um) / us
                H = V.T @ V + len(V) * np.eye(V.shape[1])
                L = np.linalg.cholesky(H)
                W = np.linalg.solve(L.T, np.linalg.solve(L, V.T @ (D - dm)))
                constraint = C / us
                lift = np.linalg.solve(L, constraint.T)
                a, singular, b = np.linalg.svd(lift, full_matrices=False)
                keep = singular > singular[0] * config["svd_relative_threshold"]
                objective_condition = float(np.linalg.cond(H))
                for name, desired in (("unconstrained", None), ("preserve_finite_ERP_increment", np.zeros_like(target)),
                                      ("match_long_reference_ERP_increment", target)):
                    if desired is None:
                        coefficient, residual = W, None
                    else:
                        mismatch = desired - constraint @ W
                        change = (a[:, keep] / singular[keep]) @ (b[keep] @ mismatch)
                        coefficient = W + np.linalg.solve(L.T, change)
                        constraint_error = np.linalg.norm(constraint @ coefficient - desired)
                        residual = float(constraint_error / max(np.linalg.norm(target), 1.))
                    accepted = residual is None or residual <= config["relative_constraint_residual_rejection"]
                    prediction = (ut - um) / us @ coefficient + dm
                    white_corrected = wf - wc / us @ coefficient
                    row = {"target": participant, "length_s": length, "distance_s": distance, "model": name,
                        "accepted": accepted, "constraint_relative_residual": residual,
                        "constraint_rank": int(keep.sum()), "constraint_count": len(C),
                        "constraint_condition": float(singular[0] / singular[keep][-1]),
                        "ridge_objective_condition": objective_condition,
                        "delta_mse_zero": float(np.mean(dt**2)),
                        "delta_mse_corrected": float(np.mean((dt - prediction)**2)),
                        "white_raw_noise_energy_gain": float(np.sum(white_corrected**2) / np.sum(wf**2)),
                        "zero_input_correction_feature_rms_uv": float(np.sqrt(np.mean((dm - um / us @ coefficient)**2))),
                        "standardized_mapping_spectral_norm": float(np.linalg.norm(coefficient, 2))}
                    rows.append(row)
                    for label, topography, f, reference, context in injections:
                        corrected = f - context / us @ coefficient
                        robustness.append({"target": participant, "length_s": length, "distance_s": distance,
                            "model": name, "injection": label, "topography": topography,
                            "finite_relative_error": float(np.linalg.norm(f - reference) / np.linalg.norm(reference)),
                            "corrected_relative_error": float(np.linalg.norm(corrected - reference) / np.linalg.norm(reference)),
                            "relative_change_from_finite": float(np.linalg.norm(corrected - f) / np.linalg.norm(reference)),
                            "reference_gain": float(corrected @ reference / (reference @ reference))})
            print(f"constraints completed {length}s, distance {distance}s", flush=True)
    summary = []
    for length in PLAN["cut_lengths_seconds"]:
        for distance in PLAN["epoch_distance_from_right_edge_seconds"]:
            for model in config["models"]:
                subset = [r for r in rows if (r["length_s"], r["distance_s"], r["model"]) == (length, distance, model)]
                erps = [r for r in robustness if (r["length_s"], r["distance_s"], r["model"]) == (length, distance, model)]
                original = [r for r in erps if r["injection"].startswith("original_")]
                summary.append({"length_s": length, "distance_s": distance, "model": model,
                    "accepted_folds": sum(r["accepted"] for r in subset),
                    "mse_reduction_macro": float(np.mean([1 - r["delta_mse_corrected"] / r["delta_mse_zero"] for r in subset])),
                    "improved_participants": sum(r["delta_mse_corrected"] < r["delta_mse_zero"] for r in subset),
                    "noise_gain_max": max(r["white_raw_noise_energy_gain"] for r in subset),
                    "constraint_condition_max": max(r["constraint_condition"] for r in subset),
                    "original_injections_mean_relative_error": float(np.mean([r["corrected_relative_error"] for r in original])),
                    "all_injections_max_relative_error": max(r["corrected_relative_error"] for r in erps)})
    report = {"execution": execution, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "script_sha256": hashlib.sha256((HERE / "probe_constrained.py").read_bytes()).hexdigest(),
        "draws_sha256": hashlib.sha256((HERE / "projected_draws.npz").read_bytes()).hexdigest(),
        "rows": rows, "robustness": robustness, "summary": summary}
    (HERE / "constrained_results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
