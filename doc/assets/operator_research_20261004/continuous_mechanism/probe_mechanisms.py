"""Independent mathematical probes; no public target evaluation is read.

Run with project venv, native threads bounded by threadpoolctl.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import mne  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import solve_triangular, toeplitz  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from data.preprocess import design_continuous_sos, filter_bidirectional_segment  # noqa: E402
from transfer.continuous_decision import _event_design, smooth_erp_basis  # noqa: E402


def gaussian_scores(x, means, covariance):
    root = np.linalg.cholesky(covariance)
    residual = np.asarray(x)[:, None, :] - np.asarray(means)[None, :, :]
    z = solve_triangular(root, residual.reshape(-1, len(root)).T, lower=True).T
    return -.5 * np.sum(z.reshape(residual.shape) ** 2, axis=-1)


def summary(scores, truth):
    logp = scores - logsumexp(scores, axis=1, keepdims=True)
    return {"accuracy": float(np.mean(np.argmax(scores, axis=1) == truth)),
            "nll": float(-np.mean(logp[np.arange(len(truth)), truth])),
            "mean_max_probability": float(np.exp(logp).max(axis=1).mean())}


def invertible_change_of_coordinates():
    rng = np.random.default_rng(472901)
    n = 36
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    transform = q @ np.diag(np.geomspace(.13, 1.7, n)) @ q.T
    raw_cov = toeplitz(.65 ** np.arange(n))
    means = rng.normal(size=(3, n)) * .21
    truth = rng.integers(0, 3, 400)
    x = means[truth] + rng.normal(size=(len(truth), n)) @ np.linalg.cholesky(raw_cov).T
    raw = gaussian_scores(x, means, raw_cov)
    exact = gaussian_scores(x @ transform.T, means @ transform.T,
                            transform @ raw_cov @ transform.T)
    only_data = gaussian_scores(x @ transform.T, means, raw_cov)
    data_and_mean = gaussian_scores(x @ transform.T, means @ transform.T, raw_cov)
    error = float(np.max(np.abs((raw - raw[:, :1]) - (exact - exact[:, :1]))))
    assert error < 1e-10
    return {"seed": 472901, "n_draws": len(truth), "dimension": n,
            "max_bayesfactor_log_error_full_transform": error,
            "full_transform_same_argmax": bool(np.array_equal(raw.argmax(1), exact.argmax(1))),
            "data_only_winner_changes": int(np.sum(raw.argmax(1) != only_data.argmax(1))),
            "data_and_mean_wrong_covariance_winner_changes": int(np.sum(raw.argmax(1) != data_and_mean.argmax(1))),
            "raw": summary(raw, truth), "exact_transformed": summary(exact, truth),
            "data_only": summary(only_data, truth),
            "data_and_mean_wrong_covariance": summary(data_and_mean, truth)}


def actual_filter_dictionary():
    sfreq = 256.
    basis = smooth_erp_basis(sfreq, 1.2, 8)
    sos = design_continuous_sos(sfreq, l_freq=.1, h_freq=30., order=4,
                              notch_freqs_hz=(50.,), notch_quality_factor=30.)
    pad = int(mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)["padlen"])
    n = 8500
    onsets = 100 + np.arange(120) * 64
    labels = np.arange(120) % 6 == 0
    design = _event_design(onsets, labels, n, basis)
    transformed = filter_bidirectional_segment(sos, design.T, padding_limit=pad).data.T
    nuisance = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
    q, _ = np.linalg.qr(nuisance, mode="reduced")
    def project(a):
        return a - q @ (q.T @ a)
    d = project(design)
    fd = project(transformed)
    reconstructed = d @ np.linalg.lstsq(d, fd, rcond=None)[0]
    relative = np.linalg.norm(reconstructed - fd, axis=0) / np.linalg.norm(fd, axis=0)
    # Smooth physically plausible kernel, not an intentionally discontinuous basis vector.
    time = np.arange(len(basis)) / sfreq
    waveform = np.exp(-.5 * ((time - .36) / .1) ** 2) - .4 * np.exp(-.5 * ((time - .16) / .045) ** 2)
    raw_mean = np.zeros(n)
    for onset in onsets[labels]:
        raw_mean[onset:onset + len(waveform)] += waveform
    filtered_mean = filter_bidirectional_segment(sos, raw_mean[None, :], padding_limit=pad).data[0]
    raw_mean_fit = np.column_stack([d, q]) @ np.linalg.lstsq(np.column_stack([d, q]), filtered_mean, rcond=None)[0]
    # Finite filtering does not commute with cropping, even if both operations are linear.
    a, b = 1100, 5000
    short = filter_bidirectional_segment(sos, raw_mean[None, a:b], padding_limit=pad).data[0]
    commutator = short - filtered_mean[a:b]
    return {"sfreq_hz": sfreq, "padding_limit": pad, "samples": n,
            "basis_length": len(basis), "basis_rank": basis.shape[1],
            "FD_outside_span_D_after_same_nuisance_projection_per_column": relative.tolist(),
            "FD_relative_frobenius_projection_residual": float(np.linalg.norm(fd-reconstructed)/np.linalg.norm(fd)),
            "smooth_mean_best_old_dictionary_relative_residual": float(np.linalg.norm(filtered_mean-raw_mean_fit)/np.linalg.norm(filtered_mean)),
            "crop_filter_noncommutation_relative_norm": float(np.linalg.norm(commutator)/np.linalg.norm(filtered_mean[a:b])),
            "crop_filter_noncommutation_max_abs": float(np.max(np.abs(commutator))),
            "interpretation": "A deterministic operator must transform the full event design on identical scope; learned effective kernels can absorb part, not all, of the discrepancy."}


def ar1_information_distortion():
    # A legal stationary AR(2) process, not a hidden covariance fitted from evaluation data.
    # Oscillatory poles are realistic as a rival to monotone AR(1), but this is synthetic.
    n = 384
    frequency = 10 / 256
    radius = .96
    a1, a2 = 2 * radius * np.cos(2*np.pi*frequency), -radius**2
    # Solve Yule-Walker from the known generating parameters.
    r1_over_r0 = a1 / (1-a2)
    r0 = 1 / (1 - a1*r1_over_r0 - a2*(a1*r1_over_r0+a2))
    r = np.empty(n)
    r[:2] = [r0, r1_over_r0*r0]
    for k in range(2, n):
        r[k] = a1*r[k-1] + a2*r[k-2]
    cov = toeplitz(r)
    # Population-optimal AR1. This probes model misspecification without estimation noise.
    phi_unclipped = r1_over_r0
    rng = np.random.default_rng(548200)
    truth = rng.integers(0, 2, 6000)
    noise = rng.normal(size=(len(truth), n)) @ np.linalg.cholesky(cov).T
    output = []
    for frequency_hz in (2., 6., 10., 16., 24.):
        direction = np.sin(2*np.pi*frequency_hz*np.arange(n)/256)
        direction *= np.hanning(n)
        direction *= 2 / np.sqrt(direction @ np.linalg.solve(cov, direction))
        means = np.stack([-direction/2, direction/2])
        x = means[truth] + noise
        exact = gaussian_scores(x, means, cov)
        variants = {}
        for name, phi in (("population_ar1", phi_unclipped), ("current_clip_ar1", min(phi_unclipped, .95))):
            innovations = r[0]*(1+phi**2)-2*phi*r[1]
            approximation = toeplitz(innovations/(1-phi**2)*phi**np.arange(n))
            weight = np.linalg.solve(approximation, direction)
            expectation = float(direction @ weight / 2)
            variance = float(weight @ cov @ weight)
            variants[name] = {"phi": float(phi), "expected_true_log_odds": expectation,
                              "actual_log_odds_variance": variance,
                              "optimal_linear_logit_calibration_slope": 2*expectation/variance,
                              **summary(gaussian_scores(x, means, approximation), truth)}
        output.append({"signal_frequency_hz": frequency_hz, "exact": summary(exact, truth), **variants})
    return {"seed": 548200, "n_draws": len(truth), "known_raw_process": {"a1": a1, "a2": a2, "innovation_variance": 1},
            "known_signal_mahalanobis_squared": 4., "per_frequency": output,
            "interpretation": "AR1 can produce both underconfidence and overconfidence; effect follows the discriminative spectral direction, not sample count alone."}


def uncertain_noise_scale_counterexample():
    # Predeclared normal/inverse-gamma predictive likelihood for unknown global scale.
    # Common variance integrates once per record, not independently once per sample.
    rng = np.random.default_rng(96000)
    n, draws, nu = 24, 12000, 6.
    mean_delta = np.linspace(-1, 1, n)
    mean_delta *= 2 / np.linalg.norm(mean_delta)
    means = np.stack([-mean_delta/2, mean_delta/2])
    truth = rng.integers(0, 2, draws)
    scales = nu / rng.chisquare(nu, draws)
    x = means[truth] + np.sqrt(scales[:, None]) * rng.normal(size=(draws, n))
    q = np.sum((x[:, None, :] - means[None, :, :])**2, axis=2)
    normal = -.5*q
    student = -.5*(nu+n)*np.log1p(q/nu)
    return {"seed": 96000, "n_draws": draws, "nu": nu, "dimension": n,
            "normal": summary(normal, truth), "correct_scale_marginal": summary(student, truth),
            "counterclaim": "The Student likelihood can improve NLL without changing winner here, but its prior degrees of freedom must come from legal data; software sample count is not independent scale evidence."}


def main():
    with threadpool_limits(limits=2):
        results = {"schema": "n2p3_continuous_operator_mechanisms/1",
                   "evidence_scope": "synthetic_mathematical_counterexample",
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                     ("src/transfer/continuous_decision.py", "src/data/preprocess.py")},
                   "invertible_transform": invertible_change_of_coordinates(),
                   "actual_finite_filter": actual_filter_dictionary(),
                   "AR1_information": ar1_information_distortion(),
                   "unknown_scale": uncertain_noise_scale_counterexample()}
    destination = Path(__file__).with_name("mechanism_results.json")
    destination.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
