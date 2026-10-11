"""Fixed soft-constraint sweep for boundary correction.

The penalty is selected as a declared research grid, never from held EEG. It
reports a Pareto surface: boundary-difference MSE, ERP-increment fidelity,
constraint violation and white-noise gain. This is a diagnostic, not a
production cleaner.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mne
import numpy as np
from probe import HERE, PLAN, freeze_dependencies
from scipy.fft import dct
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

    config = json.loads((HERE / "constrained_plan.json").read_text(encoding="utf-8"))
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
        value = 10 * np.exp(-.5 * ((time-center) / (1.2/7))**2)
        value[(time < 0) | (time >= 1.2)] = 0
        dictionary.append(value)
    dictionary.extend([
        10*np.exp(-.5*((time+.1)/.08)**2) * ((time >= -.2) & (time < 0)),
        np.full(samples, 10.), time/25.90625,
    ])
    dictionary = np.stack(dictionary)

    def apply(values):
        return filter_bidirectional_segment(sos, values, padding_limit=pad).data

    def feature(values):
        return dct(transform_epoch_samples(values, fs, spec, require_linear=True), norm="ortho", axis=-1)[:, :8]

    reference_dictionary = feature(apply(dictionary)[:, start:stop])
    rows, robustness = [], []
    penalties = tuple(config.get("soft_constraint_penalties", [0., 1e-4, 1e-3, 1e-2, .1, 1., 10., 100., 1e4]))
    for length in PLAN["cut_lengths_seconds"]:
        for distance in PLAN["epoch_distance_from_right_edge_seconds"]:
            end = stop + round(distance*fs)
            begin = end - round(length*fs)
            width = round(fs*.25)

            def context(values, begin=begin, start=start, stop=stop, width=width):
                transformed = feature(apply(values)[:, start-begin:stop-begin])
                left, right = values[:, :width], values[:, -width:]
                return np.column_stack([transformed, left.mean(axis=1), right.mean(axis=1),
                    (left[:, -1]-left[:, 0])/.25, (right[:, -1]-right[:, 0])/.25])

            f_dictionary = context(dictionary[:, begin:end])[:, :8]
            c_dictionary = context(dictionary[:, begin:end])
            C, target = [], []
            for channel in range(8):
                for index in range(len(dictionary)):
                    block = np.zeros((8, 8))
                    block[channel] = c_dictionary[index, :8]
                    scalar = np.zeros((4, 8))
                    scalar[:, channel] = c_dictionary[index, 8:]
                    C.append(np.concatenate([block.reshape(-1), scalar.reshape(-1)]))
                    desired = np.zeros((8, 8))
                    desired[channel] = f_dictionary[index] - reference_dictionary[index]
                    target.append(desired.reshape(-1))
            C, target = np.stack(C), np.stack(target)
            injections = []
            for latency, spread, label in ((.35,.12,"p300_035"),(.8,.12,"p300_080"),(.05,.12,"early"),(.35,.04,"narrow")):
                wave = 10*np.exp(-.5*((time-latency)/spread)**2) - 4*np.exp(-.5*((time-latency+.18)/(spread/2))**2)
                reference = feature(apply(wave[None])[:, start:stop])[0]
                cut = wave[None, begin:end]
                f, c = context(cut)[0, :8], context(cut)[0]
                for topography_name, topography in (("posterior", np.array([.25,.5,1.,.3,.6,.6,.4,.4])), ("pz_only", np.eye(8)[2])):
                    injections.append((label, topography_name, (topography[:,None]*reference).reshape(-1),
                        (topography[:,None]*f).reshape(-1), np.concatenate([(c[:8,None]*topography).reshape(-1),
                            (c[8:,None]*topography).reshape(-1)])))
            geometry = (draws["lengths"] == length) & (draws["distances"] == distance)
            for participant in PLAN["subjects"]:
                train = geometry & (draws["subjects"] != participant)
                held = geometry & (draws["subjects"] == participant)
                U, D = draws["context"][train], draws["finite"][train]-draws["long"][train]
                ut, dt = draws["context"][held], draws["finite"][held]-draws["long"][held]
                um, us, dm = U.mean(axis=0), U.std(axis=0), D.mean(axis=0)
                us = np.maximum(us, max(float(np.median(us)), 1e-10)*.01)
                V = (U-um)/us
                H = V.T@V + len(V)*np.eye(V.shape[1])
                rhs = V.T@(D-dm)
                normalized = C/us
                row_norm = np.maximum(np.linalg.norm(normalized, axis=1), 1e-10)
                constraint = normalized/row_norm[:,None]
                gram_constraint = constraint.T@constraint
                for penalty in penalties:
                    if penalty == 0:
                        coefficient = np.linalg.solve(H, rhs)
                    else:
                        coefficient = np.linalg.solve(H + penalty*len(V)*gram_constraint, rhs)
                    prediction = (ut-um)/us@coefficient+dm
                    noise_gain = []
                    for _ in range(8):
                        noise = np.random.default_rng(20261004+int(penalty*100000)+int(length)).normal(size=(32,8,end-begin))
                        # Same linear correction, separated by channel, without saving raw draws.
                        nf = np.empty((len(noise), 64))
                        nc = np.empty((len(noise), 96))
                        for row_index, values in enumerate(noise):
                            value = context(values)
                            feature_block = value[:, :8]
                            scalar_block = value[:, 8:]
                            nf[row_index] = feature_block.reshape(-1)
                            nc[row_index] = np.concatenate([feature_block.reshape(-1), scalar_block.T.reshape(-1)])
                        corrected = nf - (nc/us@coefficient)
                        noise_gain.append(np.sum(corrected**2)/max(np.sum(nf**2), 1e-30))
                    rows.append({"length_s":length,"distance_s":distance,"participant":participant,"penalty":penalty,
                        "delta_mse":float(np.mean(dt**2)),"corrected_mse":float(np.mean((dt-prediction)**2)),
                        "mse_reduction":float(1-np.mean((dt-prediction)**2)/max(np.mean(dt**2),1e-30)),
                        "constraint_rms":float(np.linalg.norm(constraint@coefficient)/np.sqrt(len(constraint))),
                        "noise_gain_mean":float(np.mean(noise_gain)),"mapping_norm":float(np.linalg.norm(coefficient,2))})
                    for label, topography, reference, finite, context_vector in injections:
                        corrected = finite - context_vector/us@coefficient
                        robustness.append({"length_s":length,"distance_s":distance,"participant":participant,"penalty":penalty,
                            "injection":label,"topography":topography,"finite_error":float(np.linalg.norm(finite-reference)/max(np.linalg.norm(reference),1e-30)),
                            "corrected_error":float(np.linalg.norm(corrected-reference)/max(np.linalg.norm(reference),1e-30)),
                            "reference_gain":float(corrected@reference/max(reference@reference,1e-30))})
    summary=[]
    for length in PLAN["cut_lengths_seconds"]:
        for distance in PLAN["epoch_distance_from_right_edge_seconds"]:
            for penalty in penalties:
                values=[r for r in rows if (r["length_s"],r["distance_s"],r["penalty"])==(length,distance,penalty)]
                erp=[r for r in robustness if (r["length_s"],r["distance_s"],r["penalty"])==(length,distance,penalty)]
                summary.append({"length_s":length,"distance_s":distance,"penalty":penalty,
                    "mse_reduction_macro":float(np.mean([r["mse_reduction"] for r in values])),
                    "mse_improved_participants":sum(r["corrected_mse"]<r["delta_mse"] for r in values),
                    "noise_gain_max":max(r["noise_gain_mean"] for r in values),
                    "erp_error_mean":float(np.mean([r["corrected_error"] for r in erp])),
                    "erp_error_max":max(r["corrected_error"] for r in erp),
                    "erp_gain_min":min(r["reference_gain"] for r in erp)})
    report={"schema":"n2p3_boundary_constraint_pareto/1","execution":execution,
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "draws_sha256":hashlib.sha256((HERE/"projected_draws.npz").read_bytes()).hexdigest(),
        "penalties":list(penalties),"rows":rows,"robustness":robustness,"summary":summary,
        "selection":"no held EEG used for penalty selection; report full Pareto surface"}
    (HERE/"constraint_pareto_results.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
