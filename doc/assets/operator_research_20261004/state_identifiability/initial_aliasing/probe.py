"""Frozen legal-prefix research: output covariance versus boundary uncertainty.

This file and results are research artifacts. It does not register production
operators or fit using any T samples, truth, or long-T diagnostic oracle.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import sys
import tarfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
from scipy.linalg import toeplitz
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PLAN_BYTES = (HERE / "plan.json").read_bytes()
PLAN = json.loads(PLAN_BYTES)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def freeze_dependencies():
    """Execute historical verified source, not a potentially changing dirty tree."""
    folder = ROOT / "frozen/system_implementation_20261004_after"
    manifest = json.loads((folder / "source.manifest.json").read_bytes())
    archive = folder / manifest["archive"]
    payload = archive.read_bytes()
    assert sha(payload) == manifest["archive_sha256"]
    destination = HERE / "executed_source"
    destination.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as handle:
        for member in handle.getmembers():
            if not member.name.startswith("src/") or not member.isfile():
                continue
            target = destination / member.name
            assert target.resolve().is_relative_to(destination.resolve())
            data = handle.extractfile(member).read()
            if target.exists():
                assert target.read_bytes() == data
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    sys.path.insert(0, str(destination / "src"))
    return {"archive_sha256": sha(payload), "source_archive": str(archive),
            "execution_root": str(destination), "probe_sha256": sha(Path(__file__).read_bytes()),
            "plan_sha256": sha(PLAN_BYTES), "python": sys.version,
            "numpy": np.__version__, "platform": platform.platform()}


def counterexamples():
    """Exact examples independent from repository covariance implementation."""
    rng = np.random.default_rng(20261004)
    rows = {}
    # u is long-reference residual. d=-.8u is correlated finite-edge error.
    u = rng.normal(size=(100000, 2))
    d = -.8 * u
    z = u + d
    empirical = np.cov(z, rowvar=False, bias=True)
    full = np.cov(np.column_stack([u, d]), rowvar=False, bias=True)
    lift = np.column_stack([np.eye(2), np.eye(2)])
    np.testing.assert_allclose(lift @ full @ lift.T, empirical, atol=2e-14)
    base = np.cov(u, rowvar=False, bias=True)
    delta = np.cov(d, rowvar=False, bias=True)
    rows["negative_cross_term"] = {
        "true_variance": float(np.trace(empirical) / 2),
        "omit_cross_variance": float(np.trace(base + delta) / 2),
        "double_count_variance": float(np.trace(empirical + delta) / 2),
        "omit_cross_inflation": float(np.trace(base + delta) / np.trace(empirical)),
        "double_count_inflation": float(np.trace(empirical + delta) / np.trace(empirical)),
    }
    d2 = .8 * u
    z2 = u + d2
    rows["positive_cross_term"] = {
        "true_variance": float(np.trace(np.cov(z2, rowvar=False, bias=True)) / 2),
        "omit_cross_variance": float(np.trace(base + delta) / 2),
    }
    B = np.array([[1., 1.], [1., -1.], [.4, 2.]])
    G = np.array([[2., .2], [.2, .5]])
    T = np.diag([1000., .001])
    transformed_B = B @ np.linalg.inv(T)
    transformed_G = T @ G @ T.T
    np.testing.assert_allclose(B @ G @ B.T, transformed_B @ transformed_G @ transformed_B.T,
                               atol=1e-13)
    # An isotropic ridge in unknown state coordinates is not a physical prior.
    ridge = .1
    rows["state_coordinate_gauge"] = {
        "projected_covariance_difference": float(np.max(np.abs(B @ G @ B.T - transformed_B @ transformed_G @ transformed_B.T))),
        "isotropic_state_ridge_effect_ratio": float(np.trace(transformed_B @ transformed_B.T) / np.trace(B @ B.T)),
        "original_regularized_trace": float(np.trace(B @ (G + ridge * np.eye(2)) @ B.T)),
        "rescaled_regularized_trace": float(np.trace(transformed_B @ (transformed_G + ridge * np.eye(2)) @ transformed_B.T)),
    }
    # No unique latent decomposition: a'=a+Kx, A'=A-BK leaves observations.
    x = rng.normal(size=(1000, 2))
    a = rng.normal(size=(1000, 2))
    A = np.array([[.4, -.3], [.1, 1.], [-.2, .6]])
    K = np.array([[.2, 2.], [-1., .4]])
    np.testing.assert_allclose(x @ A.T + a @ B.T,
                               x @ (A - B @ K).T + (a + x @ K.T) @ B.T, atol=1e-14)
    rows["latent_decomposition_gauge"] = {
        "exact_observation_identity": True,
        "conclusion": "Gamma/Omega not identified from z without fixed A, B, residual definition and state origin",
    }
    # AR(1): full stationary covariance versus exact innovation whitening.
    n, rho, innovation_variance = 60, .85, 1.7
    covariance = innovation_variance / (1 - rho**2) * toeplitz(rho ** np.arange(n))
    white = np.eye(n)
    white[0, 0] = np.sqrt(1 - rho**2)
    white[np.arange(1, n), np.arange(n - 1)] = -rho
    np.testing.assert_allclose(white @ covariance @ white.T,
                               innovation_variance * np.eye(n), atol=2e-14)
    # s=x[-1] before observed x[0:n]. Its conditional variance shrinks by observing x0.
    state_var = innovation_variance / (1 - rho**2)
    omega = state_var * rho ** np.arange(1, n + 1)
    conditioned = state_var - omega @ np.linalg.solve(covariance, omega)
    np.testing.assert_allclose(conditioned, innovation_variance, atol=1e-13)
    rows["ar_startup_schur"] = {
        "stationary_state_variance": state_var,
        "state_variance_given_observed_segment": float(conditioned),
        "known_innovation_variance": innovation_variance,
        "whitening_identity_max_error": float(np.max(np.abs(white @ covariance @ white.T - innovation_variance * np.eye(n)))),
        "warning": "Gamma stationary is identifiable only under the fitted stable AR assumptions; add no independent startup covariance after exact stationary whitening",
    }
    # Coordinate s equal to observed x0 is exactly known conditionally.
    duplicated_omega = covariance[:, 0]
    schur = covariance[0, 0] - duplicated_omega @ np.linalg.solve(covariance, duplicated_omega)
    assert abs(schur) < 1e-13
    rows["observed_state_degeneracy"] = {"conditional_variance": float(schur)}
    # Dense within-record replication does not create participants.
    independent_groups, repeats, icc = 8, 100, .9
    effective = independent_groups * repeats / (1 + (repeats - 1) * icc)
    rows["pseudo_replication"] = {"rows": independent_groups * repeats,
                                  "physical_groups": independent_groups,
                                  "equicorrelated_mean_effective_n": effective,
                                  "naive_standard_error_underestimate_factor": np.sqrt(1 + (repeats - 1) * icc)}
    # Mean state can be signal-dependent, so covariance alone cannot correct its bias.
    classes = np.repeat([0, 1], 100)
    means = classes * 2.
    rows["mean_not_variance"] = {"delta_class_mean_difference": float(means[classes == 1].mean() - means[classes == 0].mean()),
                                "centred_delta_within_class_variance": 0.,
                                "implication": "zero within-class Gamma does not imply zero class-dependent boundary distortion"}
    return rows


def empirical_cov(values):
    centered = values - values.mean(axis=0)
    return centered.T @ centered / len(values)


def gaussian_diagnostics(residual, covariance):
    L = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(L, residual.T).T
    dimension = residual.shape[1]
    quadratic = np.sum(whitened**2, axis=1)
    logdet = 2 * np.log(np.diag(L)).sum()
    marginal = np.abs(residual) / np.sqrt(np.diag(covariance))
    return {"nll_per_dimension": float(np.mean(.5 * (quadratic + logdet + dimension * np.log(2 * np.pi))) / dimension),
            "mahalanobis_per_dimension": float(np.mean(quadratic) / dimension),
            "marginal_95_coverage": float(np.mean(marginal <= 1.959963984540054)),
            "covariance_trace": float(np.trace(covariance))}


def fit_evaluate(records):
    from sklearn.linear_model import Ridge

    output = []
    for length in PLAN["cut_lengths_seconds"]:
        for distance in PLAN["epoch_distance_from_right_edge_seconds"]:
            geometry = [r for r in records if r["length_s"] == length and r["distance_s"] == distance]
            for target in PLAN["subjects"]:
                train = [r for r in geometry if r["subject"] != target]
                test = [r for r in geometry if r["subject"] == target]
                assert train and test
                H = np.stack([r["long"] for r in train])
                F = np.stack([r["finite"] for r in train])
                U = np.stack([r["context"] for r in train])
                Y = np.array([r["label"] for r in train], dtype=int)
                ht = np.stack([r["long"] for r in test])
                ft = np.stack([r["finite"] for r in test])
                ut = np.stack([r["context"] for r in test])
                yt = np.array([r["label"] for r in test], dtype=int)
                D, dt = F - H, ft - ht
                mean_h = np.stack([H[Y == label].mean(axis=0) for label in (0, 1)])
                mean_d = np.stack([D[Y == label].mean(axis=0) for label in (0, 1)])
                rh, rd = H - mean_h[Y], D - mean_d[Y]
                rf = rh + rd
                joint = empirical_cov(np.column_stack([rh, rd]))
                dim = H.shape[1]
                cov_h, cov_d = joint[:dim, :dim], joint[dim:, dim:]
                cross = joint[:dim, dim:] + joint[dim:, :dim]
                direct = empirical_cov(rf)
                exact = cov_h + cov_d + cross
                np.testing.assert_allclose(exact, direct, rtol=1e-10, atol=1e-8)
                ridge = max(float(np.trace(direct) / dim * .001), 1e-12) * np.eye(dim)
                finite_residual = ft - (mean_h + mean_d)[yt]
                covariances = {"direct_finite": direct, "full_paired": exact,
                               "omit_cross": cov_h + cov_d, "double_count": direct + cov_d}
                scores = {name: gaussian_diagnostics(finite_residual, cov + ridge)
                          for name, cov in covariances.items()}
                # Conditional mean from observable input only, no labels or reference at prediction.
                # Residual covariance is measured out of participant within training to avoid fit residual optimism.
                conditional_predictions = np.zeros_like(D)
                subjects = np.array([r["subject"] for r in train])

                def predict_delta(fit_mask, value):
                    um = U[fit_mask].mean(axis=0)
                    us = U[fit_mask].std(axis=0)
                    us = np.maximum(us, max(float(np.median(us)), 1e-10) * .01)
                    dm = D[fit_mask].mean(axis=0)
                    model = Ridge(alpha=float(fit_mask.sum()), fit_intercept=False)
                    model.fit((U[fit_mask] - um) / us, D[fit_mask] - dm)
                    return model.predict((value - um) / us) + dm

                for subject in np.unique(subjects):
                    held = subjects == subject
                    conditional_predictions[held] = predict_delta(~held, U[held])
                predicted = predict_delta(np.ones(len(U), dtype=bool), ut)
                residual_d = D - conditional_predictions
                conditional_cov = empirical_cov(residual_d)
                unconditional_cov = empirical_cov(D)
                # delta-only scoring compares predictions of long-reference differences;
                # it does not label the covariance as physiological noise.
                delta_ridge = max(float(np.trace(unconditional_cov) / dim * .01), 1e-12) * np.eye(dim)
                zero_error = dt
                mean_error = dt - D.mean(axis=0)
                conditional_error = dt - predicted
                delta_scores = {
                    "unconditional": gaussian_diagnostics(mean_error, unconditional_cov + delta_ridge),
                    "conditional": gaussian_diagnostics(conditional_error, conditional_cov + delta_ridge),
                }
                output.append({"target": target, "length_s": length, "distance_s": distance,
                    "source_participants": sorted(set(subjects.tolist())),
                    "source_cut_count": len(train), "target_cut_count": len(test),
                    "source_label_count": np.bincount(Y, minlength=2).tolist(),
                    "target_label_count": np.bincount(yt, minlength=2).tolist(),
                    "paired_covariance_max_error": float(np.max(np.abs(exact - direct))),
                    "cross_trace_relative_to_finite": float(np.trace(cross) / np.trace(direct)),
                    "delta_trace_relative_to_finite": float(np.trace(cov_d) / np.trace(direct)),
                    "cross_frobenius_relative_to_finite": float(np.linalg.norm(cross) / np.linalg.norm(direct)),
                    "delta_class_mean_distance": float(np.linalg.norm(mean_d[1] - mean_d[0])),
                    "finite_covariance_scores": scores, "delta_scores": delta_scores,
                    "delta_mse_zero": float(np.mean(zero_error**2)),
                    "delta_mse_source_mean": float(np.mean(mean_error**2)),
                    "delta_mse_conditional": float(np.mean(conditional_error**2))})
    return output


def real_prefix_probe():
    import mne
    from scipy.fft import dct

    from data.bnci2014_008_candidate import read_bnci2014_008_candidate_record
    from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract
    from data.preprocess import (design_continuous_sos, filter_bidirectional_segment,
                                 source_epoch_support, transform_epoch_samples)

    inputs = json.loads((ROOT / "doc/assets/system_implementation_20261004/bnci_continuous_plan.json").read_bytes())
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    fs = 256.
    sos = design_continuous_sos(fs, l_freq=spec.l_freq, h_freq=spec.h_freq,
                                order=spec.filter_order, notch_freqs_hz=spec.notch_freqs_hz,
                                notch_quality_factor=spec.notch_quality_factor)
    pad = int(mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)["padlen"])
    records, ledger = [], []

    def apply(values):
        return filter_bidirectional_segment(sos, values, padding_limit=pad).data

    def feature(values):
        epoch = transform_epoch_samples(values, fs, spec, require_linear=True)
        # DCT has physical uV units; no data-fitted transform or channels.
        return dct(epoch, norm="ortho", axis=-1)[:, :8].reshape(-1)

    for input_record in inputs["records"]:
        path = Path(input_record["path"])
        payload = path.read_bytes()
        assert sha(payload) == input_record["sha256"]
        record = read_bnci2014_008_candidate_record(io.BytesIO(payload),
                    source_relative_path=path.name, source_sha256=sha(payload))
        # Physical authority is the first 800 labels and support before unknown flash801.
        stop = int(record.flash_sample[800])
        x = np.array(record.eeg_uv[:stop], copy=True).T
        onsets = np.array(record.flash_sample[:800], copy=True)
        labels = np.array(record.raw_is_target[:800], copy=True)
        # Explicitly discard every decoded later sample/label before numerical analysis.
        del record, payload
        reference = apply(x)
        events = np.column_stack([onsets, np.zeros(len(onsets), dtype=int), np.ones(len(onsets), dtype=int)])
        starts, stops = source_epoch_support(events, fs, spec)
        for length in PLAN["cut_lengths_seconds"]:
            for distance in PLAN["epoch_distance_from_right_edge_seconds"]:
                for index in range(128, 673, 24):
                    end = int(stops[index] + round(distance * fs))
                    start = end - round(length * fs)
                    if start < 0 or end > stop or starts[index] < start:
                        ledger.append({"subject": input_record["subject"], "event": index,
                                       "length_s": length, "distance_s": distance,
                                       "status": "geometrically_unavailable", "start": start, "stop": end,
                                       "authorized_stop": stop})
                        continue
                    values = x[:, start:end]
                    finite = apply(values)
                    f = feature(finite[:, starts[index] - start:stops[index] - start])
                    h = feature(reference[:, starts[index]:stops[index]])
                    # Fixed local physical features; no oracle values used as regressors.
                    width = round(fs * .25)
                    left, right = values[:, :width], values[:, -width:]
                    context = np.concatenate([f, left.mean(axis=1), right.mean(axis=1),
                        (left[:, -1] - left[:, 0]) / .25,
                        (right[:, -1] - right[:, 0]) / .25])
                    identity = {"subject": input_record["subject"], "event": index,
                                "length_s": length, "distance_s": distance,
                                "status": "included", "start": start, "stop": end,
                                "authorized_stop": stop, "parent_sha256": input_record["sha256"],
                                "cut_sha256": sha(np.ascontiguousarray(values).tobytes()),
                                "physical_group": input_record["subject"]}
                    ledger.append(identity)
                    records.append({**identity, "long": h, "finite": f, "context": context,
                                    "label": bool(labels[index])})
        print(f"legal C800: {input_record['subject']}, {stop / fs:.3f}s, cumulative cuts={len(records)}", flush=True)
    write_json(HERE / "raw_support_ledger.json", ledger)
    np.savez_compressed(HERE / "projected_draws.npz", long=np.stack([r["long"] for r in records]),
        finite=np.stack([r["finite"] for r in records]), context=np.stack([r["context"] for r in records]),
        labels=np.array([r["label"] for r in records]), subjects=np.array([r["subject"] for r in records]),
        lengths=np.array([r["length_s"] for r in records]), distances=np.array([r["distance_s"] for r in records]))
    folds = fit_evaluate(records)
    summary = []
    for length in PLAN["cut_lengths_seconds"]:
        for distance in PLAN["epoch_distance_from_right_edge_seconds"]:
            subset = [r for r in folds if r["length_s"] == length and r["distance_s"] == distance]
            summary.append({"length_s": length, "distance_s": distance, "physical_participants": len(subset),
                "conditional_mse_reduction_macro": float(np.mean([1 - r["delta_mse_conditional"] / r["delta_mse_zero"] for r in subset])),
                "conditional_better_than_zero_participants": sum(r["delta_mse_conditional"] < r["delta_mse_zero"] for r in subset),
                "mean_cross_trace_relative": float(np.mean([r["cross_trace_relative_to_finite"] for r in subset])),
                "mean_delta_trace_relative": float(np.mean([r["delta_trace_relative_to_finite"] for r in subset])),
                "finite_covariance_mean_nll": {name: float(np.mean([r["finite_covariance_scores"][name]["nll_per_dimension"] for r in subset])) for name in ("direct_finite", "full_paired", "omit_cross", "double_count")},
                "delta_mean_nll": {name: float(np.mean([r["delta_scores"][name]["nll_per_dimension"] for r in subset])) for name in ("unconditional", "conditional")}})
    return {"folds": folds, "summary": summary, "physical_participants": 8,
            "n_pseudo_cuts": len(records), "n_unavailable_cuts": sum(r["status"] != "included" for r in ledger),
            "limitation": "Source C800-only mechanism diagnostic. Class-centred EEG is not an identified deconvolved noise process. No prediction performance or K200 same-budget claim."}


def main():
    execution = freeze_dependencies()
    write_json(HERE / "execution.json", {**execution, "status": "running"})
    with threadpool_limits(limits=2):
        mathematics = counterexamples()
        write_json(HERE / "counterexamples.json", mathematics)
        public = real_prefix_probe()
    write_json(HERE / "results.json", {"execution": execution, "mathematics": mathematics, "public": public})
    write_json(HERE / "execution.json", {**execution, "status": "completed", "results_sha256": sha((HERE / "results.json").read_bytes())})
    print(json.dumps(public["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
