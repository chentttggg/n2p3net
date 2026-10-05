"""Independent small-matrix counterexamples, with isolated-source identity.

This is mathematical evidence, not a public-data fit or a decoder experiment.
Run with the repository venv; it writes only beside this script.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.linalg import toeplitz  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from scipy.stats import multivariate_normal, norm  # noqa: E402

from data.finite_filter_operator import FiniteFilterOperator  # noqa: E402
from data.harmonic import HarmonicConfig, clean_harmonic_segments  # noqa: E402
from transfer.exact_covariance import ExactPropagatedCovariance  # noqa: E402
from transfer.temporal_noise import TemporalNoise  # noqa: E402

SEED = 2026100517
RNG = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parent
RESULTS = []


def report(name, evidence, condition):
    assert condition, (name, evidence)
    RESULTS.append({"name": name, "status": "PASS", "evidence": evidence})


def finite_operator(n=96, pad=24):
    sos = butter(2, [.6, 18.], btype="bandpass", fs=64., output="sos")
    op = FiniteFilterOperator(tuple(map(tuple, sos)), pad)
    identity = np.eye(n)
    F = op.apply(identity)
    Q, B, L = op.decompose(identity)
    return op, F, Q, B, L


def deterministic_covariance():
    op, F, Q, B, L = finite_operator()
    n = len(F)
    sigma = toeplitz(.73 ** np.arange(n))
    gamma = L @ sigma @ L.T
    omega = sigma @ L.T
    exact = F @ sigma @ F.T
    cross = Q @ omega @ B.T + B @ omega.T @ Q.T
    joint = Q @ sigma @ Q.T + B @ gamma @ B.T + cross
    naive = Q @ sigma @ Q.T + B @ gamma @ B.T
    doubled = exact + B @ gamma @ B.T
    schur = gamma - omega.T @ np.linalg.solve(sigma, omega)
    x = RNG.normal(size=(n, 2))
    scipy_y = sosfiltfilt(np.asarray(op.sos), x, axis=0,
                         padtype="odd", padlen=min(op.padding_limit, n - 1))
    error = np.linalg.norm(F - Q - B @ L)
    report("deterministic_padding_requires_cross_terms_and_no_extra_gamma", {
        "samples": n, "sos_sections": len(op.sos), "B_shape": list(B.shape),
        "L_shape": list(L.shape), "decomposition_error": float(error),
        "scipy_actual_padding_max_error": float(np.max(np.abs(op.apply(x) - scipy_y))),
        "joint_covariance_relative_error": float(np.linalg.norm(joint - exact) / np.linalg.norm(exact)),
        "omit_cross_terms_relative_error": float(np.linalg.norm(naive - exact) / np.linalg.norm(exact)),
        "double_count_gamma_relative_error": float(np.linalg.norm(doubled - exact) / np.linalg.norm(exact)),
        "deterministic_schur_norm": float(np.linalg.norm(schur)),
        "operator_record": op.record(),
    }, error < 1e-10 and np.max(np.abs(op.apply(x) - scipy_y)) < 1e-10
        and np.linalg.norm(joint - exact) / np.linalg.norm(exact) < 1e-11
        and np.linalg.norm(naive - exact) / np.linalg.norm(exact) > .01
        and np.linalg.norm(schur) < 1e-10)


def unseen_context_and_conditional_mean():
    op, _, _, _, _ = finite_operator(128, 96)
    full_a = np.zeros((384, 1))
    full_b = full_a.copy()
    full_b[100] = 1.
    a_short, b_short = full_a[128:256], full_b[128:256]
    delta_a = op.apply(full_a)[128:256] - op.apply(a_short)
    delta_b = op.apply(full_b)[128:256] - op.apply(b_short)
    report("same_short_segment_different_external_history_is_not_identifiable", {
        "short_equal": bool(np.array_equal(a_short, b_short)),
        "delta_difference_norm": float(np.linalg.norm(delta_a - delta_b)),
        "external_pulse_sample": 100, "short_scope": [128, 256],
    }, np.array_equal(a_short, b_short) and np.linalg.norm(delta_a - delta_b) > 1e-3)

    sigma, gamma, omega = 1., 2., .8
    x, a = .15, .4
    means_x = np.array([-1., 1.])
    means_a = np.zeros(2)
    conditional_means = means_a + omega / sigma * (x - means_x)
    conditional_variance = gamma - omega**2 / sigma
    covariance = np.array([[sigma, omega], [omega, gamma]])
    joint_ll = np.array([multivariate_normal.logpdf([x, a], [mx, ma], covariance)
                         for mx, ma in zip(means_x, means_a, strict=True)])
    conditioned_ll = norm.logpdf(x, means_x, np.sqrt(sigma)) + norm.logpdf(
        a, conditional_means, np.sqrt(conditional_variance))
    wrong_ll = norm.logpdf(x, means_x, np.sqrt(sigma)) + norm.logpdf(
        a, means_a, np.sqrt(conditional_variance))
    report("schur_covariance_requires_candidate_conditioned_mean", {
        "conditional_means": conditional_means.tolist(),
        "conditional_variance": conditional_variance,
        "factorization_max_error": float(np.max(np.abs(joint_ll - conditioned_ll))),
        "correct_logBF": float(joint_ll[1] - joint_ll[0]),
        "wrong_unconditional_mean_logBF": float(wrong_ll[1] - wrong_ll[0]),
    }, np.max(np.abs(joint_ll - conditioned_ll)) < 1e-12
        and abs((joint_ll[1] - joint_ll[0]) - (wrong_ll[1] - wrong_ll[0])) > .1)


def learned_uncertainty_and_double_counting():
    theta = RNG.normal(1., .5, size=100000)
    observed_x = 2.
    z = theta * observed_x
    variance = float(np.var(z))
    report("learned_parameter_uncertainty_survives_conditioning_on_raw", {
        "observed_x": observed_x, "theta_mean": 1., "theta_variance": .25,
        "posterior_predictive_variance_theory": 1., "variance_mc": variance,
        "plug_in_fixed_theta_variance": 0.,
    }, abs(variance - 1.) < .02)
    x = np.array([.2, -.7, 1.2])
    qmeans = [np.zeros(3), np.array([1., 0., 0.])]
    ll = np.array([multivariate_normal.logpdf(x, mean, np.eye(3)) for mean in qmeans])
    bf = ll[1] - ll[0]
    report("deterministic_postprocessing_is_not_a_second_independent_observation", {
        "raw_logBF": float(bf), "incorrect_duplicate_identity_logBF": float(2 * bf),
        "correct_joint_support": "Z=X; p(X,Z|q)=p(X|q)*Dirac(Z-X)",
    }, abs(bf) > .1 and abs(2 * bf - bf) > .1)


def overlap_noise_and_switch():
    h, x = np.array([1., 0.]), np.array([1., 1.])
    F = np.diag([.8, 1.])
    K = np.diag([.2, 0.])
    correction = K @ x
    long_error = np.linalg.norm((F + K) @ h - h)
    finite_error = np.linalg.norm(K @ h) / np.linalg.norm(F @ h)
    report("one_hundred_percent_output_overlap_can_restore_erp", {
        "output_overlap_fraction": float(correction[0]**2 / (correction @ correction)),
        "ERP_error_to_long_reference": float(long_error),
        "ERP_increment_error_to_short_reference": float(finite_error),
    }, long_error == 0 and finite_error > .2)
    K = np.diag([-1., .3])
    correction = K @ np.array([0., 1.])
    report("zero_output_overlap_does_not_protect_erp_derivative", {
        "output_overlap_fraction": float(correction[0]**2 / (correction @ correction)),
        "ERP_increment_error": float(np.linalg.norm(K @ h)),
        "complete_mapping_ERP_output_norm": float(np.linalg.norm((np.eye(2) + K) @ h)),
    }, correction[0] == 0 and np.linalg.norm(K @ h) == 1)
    noise = RNG.normal(size=(50000, 2))
    gain_minus, gain_plus = .1**2, 1.9**2
    empirical = [float(np.mean(np.sum((scale * noise)**2, axis=1))
                       / np.mean(np.sum(noise**2, axis=1))) for scale in [.1, 1.9]]
    report("correction_energy_does_not_measure_complete_mapping_noise_gain", {
        "same_correction_energy_ratio": .9**2,
        "gain_for_I_minus_point9I": gain_minus,
        "gain_for_I_plus_point9I": gain_plus,
        "white_noise_mc_gains": empirical,
    }, abs(empirical[0] - gain_minus) < 1e-12 and abs(empirical[1] - gain_plus) < 1e-12)
    base = np.array([-1e-6, 1.])
    perturbation = np.array([2e-6, 0.])

    def mapping(v):
        return v + (.5 if v[0] > 0 else -.5) * v

    increment_error = np.linalg.norm(mapping(base + perturbation) - mapping(base) - perturbation)
    report("data_dependent_penalty_switch_is_nonlinear_at_the_switch", {
        "perturbation_norm": float(np.linalg.norm(perturbation)),
        "increment_error_norm": float(increment_error),
        "relative_increment_error": float(increment_error / np.linalg.norm(perturbation)),
    }, increment_error > .9)


def actual_harmonic_superposition():
    fs = 250.
    t = np.arange(250) / fs
    a = 1e-3 * np.sin(2 * np.pi * 49.6 * t + .2)
    b = .8e-3 * np.sin(2 * np.pi * 50.4 * t - .6)
    config = HarmonicConfig(search_half_width_hz=.6, envelope_order=0,
                            robust_iterations=2, min_line_to_residual_rms=.1)
    results = [clean_harmonic_segments(v[None], fs, [(0, len(t))], config)
               for v in [a, b, a + b]]
    residual = results[2].cleaned - results[0].cleaned - results[1].cleaned
    ratio = np.linalg.norm(residual) / np.linalg.norm(a + b)
    report("actual_adaptive_harmonic_is_not_one_fixed_linear_operator", {
        "superposition_relative_error": float(ratio),
        "estimated_centers_hz": [r.windows[0]["frequency_hz"] for r in results],
        "application": [r.windows[0]["applied"] for r in results],
        "config": results[0].record()["config"],
    }, ratio > .01)


def rank_information_and_observability():
    P = np.diag([1., 0.])
    h = np.array([1., 0.])
    delta_mu = np.array([0., 2.])
    before = float(delta_mu @ delta_mu)
    after = float(np.linalg.norm(P @ delta_mu)**2)
    report("rank_reduction_protects_a_subspace_not_all_information", {
        "protected_ERP_error": float(np.linalg.norm(P @ h - h)),
        "raw_squared_discriminability": before,
        "projected_squared_discriminability": after,
        "transformation_rank": int(np.linalg.matrix_rank(P)),
    }, np.array_equal(P @ h, h) and before > 0 and after == 0)
    H = np.array([[1., 1.]])
    As = [np.eye(2), np.diag([.5, .9]), np.diag([.9, .90001])]
    obs = [np.vstack([H, H @ A]) for A in As]
    report("kalman_low_dimension_and_formal_rank_do_not_prove_identifiability", {
        "observability_ranks": [int(np.linalg.matrix_rank(value)) for value in obs],
        "observability_condition_numbers": [float(np.linalg.cond(value)) for value in obs[1:]],
        "unobservable_vector": [1., -1.],
        "single_sample_Q_R_pairs_same_total_variance": [[.1, .9], [.5, .5]],
    }, np.linalg.matrix_rank(obs[0]) == 1 and np.linalg.matrix_rank(obs[1]) == 2
        and np.linalg.cond(obs[2]) > 1e5)


def neighbor_marginal_and_baseline():
    y, sigma = .5, .2
    means_q = np.array([-1., 1.])
    means_neighbor = np.array([-1., 1.])
    marginal = np.array([logsumexp(norm.logpdf(y, mq + means_neighbor, sigma)) - np.log(2)
                         for mq in means_q])
    omitted = norm.logpdf(y, means_q, sigma)
    report("unknown_neighbor_is_marginalized_not_deleted", {
        "marginal_logBF": float(marginal[1] - marginal[0]),
        "omitted_neighbor_logBF": float(omitted[1] - omitted[0]),
        "neighbor_prior": [.5, .5], "noise_sd": sigma,
    }, abs(marginal[1] - marginal[0]) < 1e-5
        and abs(omitted[1] - omitted[0]) > 20)
    n = 20
    sigma_t = toeplitz(.65 ** np.arange(n))
    b = np.zeros(n)
    b[:4] = .25
    B = np.eye(n) - np.ones((n, 1)) @ b[None]
    sigma_b = B @ sigma_t @ B.T
    diagonal_spread = float(np.ptp(np.diag(sigma_b)))
    report("pre_stimulus_baseline_covariance_is_not_automatically_toeplitz", {
        "baseline_projection_rank": int(np.linalg.matrix_rank(B)),
        "output_variance_diagonal_spread": diagonal_spread,
    }, np.linalg.matrix_rank(B) == n - 1 and diagonal_spread > .1)


def independent_erp_mapping_panel():
    n, channels = 96, 2
    _, F_time, _, B_time, _ = finite_operator(n, 24)
    F = np.kron(F_time, np.eye(channels))
    B = np.kron(B_time, np.eye(channels))
    basis, singular, _ = np.linalg.svd(B, full_matrices=False)
    active = singular > singular[0] * 1e-10
    state_projector = basis[:, active] @ basis[:, active].T
    K = -.35 * state_projector @ F
    t = np.arange(n) / 64.
    templates = []
    for center in [.33, .50]:
        for topography in np.eye(channels):
            waveform = np.exp(-.5 * ((t - center) / .12)**2)
            templates.append((waveform[:, None] * topography).ravel())
    _, singular_u, U = np.linalg.svd(np.stack(templates), full_matrices=False)
    U = U[singular_u > singular_u[0] * 1e-10]
    P_erp = U.T @ U
    input_protected = K @ (np.eye(n * channels) - P_erp)
    output_soft = (np.eye(n * channels) - .8 * P_erp) @ K
    maps = {"finite_original": F, "state_projection_fixture": F + K,
            "input_hard_protection_fixture": F + input_protected,
            "output_soft_protection_fixture": F + output_soft}
    # Challenge morphology/topography differs from the protection templates.
    challenges = []
    for center in [.04, .21, .39, .71, 1.40]:
        for width in [.035, .095]:
            for topography in [np.array([1., -.4]), np.array([.25, 1.])]:
                waveform = np.exp(-.5 * ((t - center) / width)**2)
                challenges.append((f"gaussian_center{center}_width{width}_topo{topography.tolist()}",
                                   12e-6 * (waveform[:, None] * topography).ravel()))
    erp = np.exp(-.5 * ((t - .39) / .095)**2)
    challenges.extend([
        ("independent_first_derivative", 12e-6 * np.gradient(erp)[:, None]
         * np.array([[1., .3]])),
        ("independent_second_derivative", 12e-6 * np.gradient(np.gradient(erp))[:, None]
         * np.array([[1., -.3]])),
        ("adjacent_flash_superposition", 12e-6 * (erp + .8 * np.exp(-.5 * ((t - .72) / .12)**2))[:, None]
         * np.array([[.3, 1.]])),
    ])
    challenges = [(name, np.asarray(value).ravel()) for name, value in challenges]
    base = RNG.normal(scale=8e-6, size=n * channels)
    white = RNG.normal(size=(30000, n * channels))
    denominator = np.mean(np.sum((white @ F.T)**2, axis=1))
    panels = {}
    for name, T in maps.items():
        entries = []
        for challenge, h in challenges:
            expected = F @ h
            delta = T @ (base + h) - T @ base
            entries.append({"challenge": challenge,
                            "increment_relative_error_to_finite": float(np.linalg.norm(delta - expected)
                                                                       / np.linalg.norm(expected)),
                            "peak_absolute_increment_error_v": float(np.max(np.abs(delta - expected)))})
        gain = float(np.sum(T**2) / np.sum(F**2))
        empirical_gain = float(np.mean(np.sum((white @ T.T)**2, axis=1)) / denominator)
        panels[name] = {"injections": entries,
                        "worst_increment_error": max(v["increment_relative_error_to_finite"] for v in entries),
                        "white_gain_exact_to_finite": gain,
                        "white_gain_mc_to_finite": empirical_gain,
                        "worst_absolute_white_gain": float(np.linalg.norm(T, ord=2)**2)}
    report("independent_ERP_and_white_noise_use_complete_mapping", {
        "fixture_scope": "synthetic_challenges_not_a_proposed_trained_boundary_corrector",
        "sample_first_flattening": True, "channels": channels, "samples": n,
        "calibration_ERP_templates": {"centers_seconds": [.33, .5], "width_seconds": .12},
        "challenge_count": len(challenges), "panels": panels,
    }, panels["finite_original"]["worst_increment_error"] < 1e-10
        and panels["state_projection_fixture"]["worst_increment_error"] > .1
        and all(abs(v["white_gain_exact_to_finite"] - v["white_gain_mc_to_finite"]) < .015
                for v in panels.values()))


def covariance_null_whitening():
    op, _, _, _, _ = finite_operator(72, 24)
    noise = TemporalNoise.fit([RNG.normal(size=(1000, 1))], order=0,
                              record_ids=("synthetic_training_only",), fit_scope="target_calibration")
    exact = ExactPropagatedCovariance.compute(op, noise, 72)
    eigenvalues, eigenvectors = np.linalg.eigh(exact.K_propagated)
    direction = eigenvectors[:, :1]
    amplified_norm = float(np.linalg.norm(exact.whiten(direction)))
    model_variance = float((direction.T @ exact.K_propagated @ direction).item())
    report("global_ridge_whitens_null_space_as_added_model_noise", {
        "actual_isolated_module": "src/transfer/exact_covariance.py",
        "numerical_rank": exact.numerical_rank,
        "null_dimension": exact.null_dimension,
        "smallest_eigenvalue": float(eigenvalues[0]),
        "ridge": exact.regularization_ridge,
        "unit_null_direction_whitened_norm": amplified_norm,
        "raw_projected_null_variance": model_variance,
        "interpretation": "ridge covariance differs from exact PSD covariance; range-only likelihood is an alternative",
    }, exact.null_dimension > 0 and amplified_norm > 1e5
        and abs(model_variance) < 1e-10)


def source_identity():
    paths = ["src/data/finite_filter_operator.py", "src/data/preprocess.py",
             "src/data/harmonic.py", "src/transfer/noise_model.py",
             "src/transfer/exact_covariance.py", "src/transfer/raw_continuous.py",
             "experiments/boundary_state_research.py", "experiments/run_e2_prior_scale.py"]
    entries = []
    for name in paths:
        isolated = ROOT / name
        original = Path("D:/BCI/n2p3-net") / name
        executed_hash = hashlib.sha256(isolated.read_bytes()).hexdigest()
        original_hash = hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else None
        entries.append({"path": name, "isolated_source_sha256": executed_hash,
                        "original_readonly_comparison_sha256": original_hash,
                        "equal_at_read": executed_hash == original_hash})
    return entries


def main():
    deterministic_covariance()
    unseen_context_and_conditional_mean()
    learned_uncertainty_and_double_counting()
    overlap_noise_and_switch()
    actual_harmonic_superposition()
    rank_information_and_observability()
    neighbor_marginal_and_baseline()
    independent_erp_mapping_panel()
    covariance_null_whitening()
    result = {"evidence_scope": "mathematical_counterexamples_no_public_training_or_scoring",
              "created_utc": datetime.now(UTC).isoformat(), "seed": SEED,
              "executable": sys.executable, "python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__,
              "isolated_root": str(ROOT), "sys_path_first": sys.path[0],
              "loaded_finite_module": sys.modules["data.finite_filter_operator"].__file__,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_comparison_at_execution": source_identity(),
              "counterexamples": RESULTS, "all_passed": True}
    (OUT / "math_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                                           encoding="utf-8", newline="\n")
    print(json.dumps({"all_passed": True, "count": len(RESULTS),
                      "results": str(OUT / "math_results.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
