"""Independent latent-model counterexamples, with one isolated current-owner probe.

These are deterministic mechanism checks. They fit no public EEG or neural model.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKOUT = ROOT.parents[3]
ISOLATED_SRC = CHECKOUT / "src"
BLOCKED_IMPORT_ROOTS = {str(ROOT).lower(), str(CHECKOUT).lower(),
                        "d:\\bci\\n2p3-net", "d:\\bci\\n2p3-net\\src"}
sys.path[:] = [str(ISOLATED_SRC), *[p for p in sys.path if p and str(Path(p)).lower() not in BLOCKED_IMPORT_ROOTS]]

import numpy as np  # noqa: E402
from scipy.linalg import toeplitz  # noqa: E402
from scipy.stats import norm  # noqa: E402

from transfer.temporal_noise import TemporalNoise  # noqa: E402


def gaussian_logpdf(y, mean, covariance):
    root = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(root, y - mean)
    return float(-0.5 * (len(y) * np.log(2 * np.pi) + 2 * np.log(np.diag(root)).sum() + whitened @ whitened))


def g_prior_covariance(x, g):
    return g * np.linalg.solve(x.T @ x, np.eye(x.shape[1]))


def check_basis_invariance(rng):
    x = rng.normal(size=(60, 4))
    a = np.diag([0.01, 0.3, 3.0, 40.0]) @ np.array([[1, .2, 0, 0], [0, 1, .3, 0], [0, 0, 1, .4], [0, 0, 0, 1]])
    h = g_prior_covariance(x, 7)
    xp = x @ a
    hp = g_prior_covariance(xp, 7)
    induced = x @ h @ x.T
    changed = xp @ hp @ xp.T
    error = float(np.max(np.abs(induced - changed)))
    ridge_error = float(np.linalg.norm(x @ x.T - xp @ xp.T) / np.linalg.norm(x @ x.T))
    assert error < 1e-9
    assert ridge_error > 10
    return {"g_prior_predictive_max_error": error, "fixed_ridge_predictive_relative_change": ridge_error,
            "basis_transform_condition": float(np.linalg.cond(a))}


def check_rank_and_weak_directions():
    u = np.arange(1.0, 11.0)
    x = np.column_stack([u, u])
    a = np.diag([2.0, 1.0])
    h = np.linalg.pinv(x.T @ x)
    hp = np.linalg.pinv((x @ a).T @ (x @ a))
    z = np.array([1.0, 0.0])
    train_error = float(np.max(np.abs(x @ h @ x.T - (x @ a) @ hp @ (x @ a).T)))
    held_variance = float(z @ h @ z)
    held_changed = float((z @ a) @ hp @ (z @ a))
    null = np.array([1.0, -1.0]) / np.sqrt(2)
    assert np.linalg.norm(x @ null) < 1e-12
    assert abs(z @ null) > .5
    assert train_error < 1e-12 and abs(held_variance - held_changed) > .0005
    weak = np.diag([1.0, 1e-5])
    weak_h = g_prior_covariance(weak, 9)
    # A pure g-prior shrinks fitted function directions equally, not coefficients.
    posterior = np.linalg.solve(weak.T @ weak + np.linalg.inv(weak_h), weak.T)
    fitted = weak @ posterior
    assert np.max(np.abs(fitted - .9 * np.eye(2))) < 1e-12
    return {"rank": int(np.linalg.matrix_rank(x)), "training_projector_max_error": train_error,
            "held_variance_pinv_original": held_variance, "held_variance_pinv_changed_basis": held_changed,
            "null_direction_held_response": float(z @ null), "weak_coefficient_prior_variance": float(weak_h[1, 1]),
            "weak_fitted_function_shrinkage": np.diag(fitted).tolist(),
            "implication": "pinv g-prior is invariant on training column space but not arbitrary held extrapolation"}


def check_gauge_and_units(rng):
    n, c = 14, 2
    k = toeplitz(.6 ** np.arange(n))
    spatial = np.array([[1.5, .2], [.2, .7]])
    d = rng.normal(size=(n, 3))
    dp = d.copy()
    dp[:, 1] += np.sin(np.arange(n))
    y = rng.normal(size=n * c)

    def evidence(design, temporal, channel, ridge=2.0, g=None):
        if g is None:
            h = np.eye(design.shape[1]) / ridge
        else:
            gram = design.T @ np.linalg.solve(temporal, design)
            h = g * np.linalg.inv(gram)
        cov = np.kron(temporal + design @ h @ design.T, channel)
        return gaussian_logpdf(y, np.zeros_like(y), cov)

    base = evidence(d, k, spatial) - evidence(dp, k, spatial)
    alpha = 50.0
    correct = evidence(d, alpha * k, spatial / alpha, ridge=2 / alpha) - evidence(dp, alpha * k, spatial / alpha, ridge=2 / alpha)
    wrong = evidence(d, alpha * k, spatial / alpha) - evidence(dp, alpha * k, spatial / alpha)
    g_base = evidence(d, k, spatial, g=4) - evidence(dp, k, spatial, g=4)
    g_changed = evidence(d, alpha * k, spatial / alpha, g=4) - evidence(dp, alpha * k, spatial / alpha, g=4)
    scale = 1e6
    cov0 = np.kron(k + d @ d.T / 2, spatial)
    cov1 = np.kron(k + dp @ dp.T / 2, spatial)
    unit_bf = gaussian_logpdf(scale * y, np.zeros_like(y), scale**2 * cov0) - gaussian_logpdf(scale * y, np.zeros_like(y), scale**2 * cov1)
    bad0 = np.kron(k + d @ d.T / (2 * scale**2), scale**2 * spatial)
    bad1 = np.kron(k + dp @ dp.T / (2 * scale**2), scale**2 * spatial)
    bad_bf = gaussian_logpdf(scale * y, np.zeros_like(y), bad0) - gaussian_logpdf(scale * y, np.zeros_like(y), bad1)
    assert abs(base - correct) < 1e-10 and abs(g_base - g_changed) < 1e-10
    assert abs(base - unit_bf) < 1e-10 and abs(base - wrong) > .01 and abs(base - bad_bf) > .01
    return {"ridge_gauge_log_bf_error": abs(base - correct), "g_prior_gauge_log_bf_error": abs(g_base - g_changed),
            "voltage_unit_log_bf_error": abs(base - unit_bf), "unchanged_ridge_gauge_log_bf_change": abs(base - wrong),
            "noise_scaled_ridge_unit_log_bf_change": abs(base - bad_bf),
            "distinction": "gauge changes factorization; voltage units change observed physical numbers; changing AR changes the model"}


def check_fixed_support_and_prefix(rng):
    n, hold_back = 180, 32
    y = rng.normal(size=(n, 2))
    d = rng.normal(size=(n, 3))
    coefficients = [.72, -.15, .04, .02]
    model = TemporalNoise.from_parameters(coefficients, innovation_variance=1,
                                          reference_id="independent_fixed_support", startup_mode="conditional_prefix")
    p = model.order
    wy = model.whiten(y)[hold_back-p:]
    wd = model.whiten(d)[hold_back-p:]
    mean = rng.normal(size=(3, 2))
    direct = model.whiten(y - d @ mean)[hold_back-p:]
    linear_error = float(np.max(np.abs(direct - (wy - wd @ mean))))
    assert wy.shape[0] == n-hold_back and wd.shape[0] == n-hold_back and linear_error < 1e-12
    sizes = {}
    for order in (0, 1, 4, 8, 16, 32):
        coef = [] if order == 0 else [.4] + [0.0] * (order-1)
        ar = TemporalNoise.from_parameters(coef, innovation_variance=1,
                                          reference_id=f"order_{order}", startup_mode="conditional_prefix")
        result = ar.whiten(y)[hold_back-order:]
        sizes[str(order)] = len(result)
        assert len(result) == n-hold_back
    modified = y.copy()
    modified[hold_back-1] += 7
    changed = model.whiten(modified)[hold_back-p:]
    prefix_effect = float(np.linalg.norm(changed - wy))
    assert prefix_effect > 1
    # Independently verify conditional Gaussian for AR1 over the identical tail.
    phi = .72
    k = toeplitz(phi ** np.arange(n) / (1-phi**2))
    tail = k[hold_back:, hold_back:]
    cross = k[hold_back:, :hold_back]
    prefix = k[:hold_back, :hold_back]
    cm = cross @ np.linalg.solve(prefix, y[:hold_back, 0])
    ck = tail - cross @ np.linalg.solve(prefix, cross.T)
    conditional_ll = gaussian_logpdf(y[hold_back:, 0], cm, ck)
    innovations = y[hold_back:, 0] - phi*y[hold_back-1:-1, 0]
    innovation_ll = float(-.5 * (len(innovations)*np.log(2*np.pi) + innovations @ innovations))
    marginal_ll = gaussian_logpdf(y[hold_back:, 0], np.zeros(n-hold_back), tail)
    assert abs(conditional_ll - innovation_ll) < 1e-10
    assert abs(conditional_ll - marginal_ll) > .01
    # Startup safety cannot be inferred from first in-segment onset alone.
    previous_onset, response_length = -10, 50
    assert previous_onset + response_length > hold_back
    return {"orders_scored_rows": sizes, "whiten_data_and_design_residual_max_error": linear_error,
            "actual_prefix_perturbation_effect": prefix_effect, "conditional_dense_vs_innovation_logpdf_error": abs(conditional_ll - innovation_ll),
            "stationary_marginal_vs_conditional_logpdf_difference": abs(conditional_ll - marginal_ll),
            "previous_event_outside_segment_has_response_until_sample": previous_onset + response_length,
            "budget": {"read_samples": n, "conditioning_samples": hold_back, "scored_samples": n-hold_back}}


def check_channel_ar_is_not_kronecker():
    n = 8
    phi = [.8, -.4]
    matrices = [toeplitz(value**np.arange(n)/(1-value**2)) for value in phi]
    # If K_time x Sigma_channel holds, each channel has the same normalized ACF.
    normalized_lag = [float(k[0, 1]/k[0, 0]) for k in matrices]
    assert abs(normalized_lag[0]-normalized_lag[1]) > 1
    # Correlated innovation streams create lag-dependent cross-channel matrices.
    innovation = np.array([[1, .3], [.3, 1]])
    transition = np.diag(phi)
    stationary = innovation / (1-np.outer(phi, phi))
    lag1 = transition @ stationary
    least_shared_phi = float(np.sum(lag1*stationary)/np.sum(stationary**2))
    error = float(np.linalg.norm(lag1-least_shared_phi*stationary)/np.linalg.norm(lag1))
    assert error > .3
    return {"channel_normalized_lag1": normalized_lag, "best_separable_lag1_relative_error": error,
            "implication": "channel-specific AR with correlated innovations is a restricted VAR, not scalar time x space MN"}


def check_whitening_can_hurt_discrimination(rng):
    n = 20000
    truth = rng.choice([-1.0, 1.0], size=n)
    noise = rng.normal(size=(n, 2))
    data = noise.copy()
    data[:, 0] += 1.5 * truth
    before = float(np.mean(np.sign(data[:, 0]) == truth))
    nuisance_projection = np.diag([0.0, 1.0])
    after_data = data @ nuisance_projection
    # No surviving coordinate contains the class bit; declare the tie as failure.
    after = 0.0
    variance_before = float(np.mean(data[:, 0]**2))
    variance_after = float(np.mean(after_data[:, 0]**2))
    expected_bayes_before = float(norm.cdf(1.5))
    assert before > .9 and after < before and variance_after < variance_before
    # Invertible correct whitening cannot erase information; approximate weighting can.
    sigma = np.diag([4.0, .1])
    delta = np.array([1.0, .2])
    root = np.linalg.cholesky(sigma)
    transformed_delta = np.linalg.solve(root, delta)
    d2 = float(delta @ np.linalg.solve(sigma, delta))
    preserved = float(transformed_delta @ transformed_delta)
    assert abs(d2-preserved) < 1e-12
    # Overweight noise-only component: an incorrect precision can decrease hit.
    after_score = data[:, 0] + 20*data[:, 1]
    wrong_precision_hit = float(np.mean(np.sign(after_score) == truth))
    assert wrong_precision_hit < before - .3
    return {"signal_projection_hit_before": before, "signal_projection_hit_after_tie_as_failure": after,
            "expected_bayes_hit_before": expected_bayes_before, "candidate_component_energy_before": variance_before,
            "candidate_component_energy_after": variance_after, "misweighted_classifier_hit": wrong_precision_hit,
            "invertible_correct_whitening_mahalanobis_error": abs(d2-preserved)}


def check_psd_acf_do_not_imply_hit(rng):
    n = 24000
    noise = rng.normal(size=(n, 2))
    truth_a = np.where(noise[:, 0] > 0, 1, -1)
    truth_b = -truth_a
    score = np.sign(noise[:, 0])
    # The same frozen residual sequence has identical ACF and PSD in both worlds.
    psd = abs(np.fft.rfft(noise[:, 1]))**2
    acf = float(noise[1:, 1] @ noise[:-1, 1] / (noise[:, 1] @ noise[:, 1]))
    hits = [float(np.mean(score == truth_a)), float(np.mean(score == truth_b))]
    assert hits == [1.0, 0.0]
    return {"same_residual_sha256": hashlib.sha256(noise.tobytes()).hexdigest(), "same_residual_lag1_acf": acf,
            "same_residual_psd_sha256": hashlib.sha256(psd.tobytes()).hexdigest(), "different_candidate_hits": hits,
            "implication": "noise diagnostics contain no candidate-label correctness certificate"}


def check_latency_derivatives():
    time = np.linspace(-.25, .85, 1000)
    center, width, delay = .35, .09, .004
    h = np.exp(-.5*((time-center)/width)**2)
    first = -(time-center)/width**2*h
    second = (((time-center)**2/width**4)-1/width**2)*h
    shifted = np.exp(-.5*((time-center-delay)/width)**2)
    first_approx = h-delay*first
    second_approx = first_approx + delay**2/2*second
    err1 = float(np.linalg.norm(shifted-first_approx)/np.linalg.norm(shifted))
    err2 = float(np.linalg.norm(shifted-second_approx)/np.linalg.norm(shifted))
    assert err2 < err1 / 20
    random_latency_sd = .02
    random_cov = random_latency_sd**2 * np.outer(first, first)
    rank = int(np.linalg.matrix_rank(random_cov, tol=np.max(np.linalg.eigvalsh(random_cov))*1e-8))
    assert rank == 1
    return {"latency_seconds": delay, "first_order_relative_error": err1, "second_order_relative_error": err2,
            "first_order_random_latency_covariance_rank": rank,
            "limitation": "Taylor basis is local in delay; random latency covariance is signal-dependent"}


def check_common_selection_identifiability():
    basis = np.array([[1.0], [2.0], [1.0]])
    design = np.kron(np.eye(3), basis)
    common = design.sum(axis=1, keepdims=True)
    full = np.column_stack([common, design])
    null = np.array([1.0, -1.0, -1.0, -1.0])
    assert np.linalg.norm(full @ null) < 1e-12
    # Sum-to-zero selection effects remove the exact duplicate common direction.
    contrast = np.array([[1, 0], [0, 1], [-1, -1]])
    identified = np.column_stack([common, design @ contrast])
    assert np.linalg.matrix_rank(identified) == 3
    return {"unconstrained_design_rank": int(np.linalg.matrix_rank(full)), "unconstrained_columns": full.shape[1],
            "sum_to_zero_design_rank": int(np.linalg.matrix_rank(identified)),
            "implication": "common plus free selection offsets is unidentified without centering or a proper hierarchical prior"}


def check_probability_mixture_is_not_bayes_model_average():
    likelihood = np.array([[.8, .2], [.004, .016]])
    prior_models = np.array([.5, .5])
    prior_candidates = np.array([.5, .5])
    normalized_experts = likelihood / likelihood.sum(axis=1, keepdims=True)
    fixed_probability_mixture = prior_models @ normalized_experts
    evidence_models = likelihood @ prior_candidates
    weights = prior_models*evidence_models
    weights /= weights.sum()
    bayes_average = weights @ normalized_experts
    joint = (prior_models @ likelihood)*prior_candidates
    joint /= joint.sum()
    assert np.max(np.abs(bayes_average-joint)) < 1e-12
    assert np.linalg.norm(fixed_probability_mixture-bayes_average) > .3
    # Lambda fixed-mixture and calibration-evidence weights are two declared estimators.
    return {"fixed_equal_probability_mixture": fixed_probability_mixture.tolist(),
            "bayes_model_weights": weights.tolist(), "bayes_model_average": bayes_average.tolist(),
            "implication": "softmax each lambda then average is a probability ensemble, not marginal likelihood integration"}


def check_reference_rank_and_mean_prior():
    c = 7
    car = np.eye(c)-np.ones((c, c))/c
    sigma = car @ np.eye(c) @ car.T
    rank = int(np.linalg.matrix_rank(sigma))
    assert rank == c-1 and np.linalg.norm(sigma @ np.ones(c)) < 1e-12
    gain = np.diag([1, 1, 1, 3, 1, 1, 1])
    delta = np.arange(1.0, c+1)
    precision = np.eye(c)
    weighted_precision = gain @ precision @ gain
    base = float(delta @ precision @ delta)
    changed = float(delta @ weighted_precision @ delta)
    # Mean prior changes signal hypothesis, not observed noise calibration.
    spatial_mean_prior = np.diag([1, 1, 1, 3, 1, 1, 1])
    assert changed > base and np.trace(spatial_mean_prior) == 9
    return {"car_noise_rank": rank, "car_null_vector_residual": float(np.linalg.norm(sigma @ np.ones(c))),
            "arbitrary_pz_precision_changes_mahalanobis": [base, changed],
            "implication": "rank is owned by observation/reference coordinates; ERP mean prior must be separate from noise precision"}


def check_candidate_adaptive_prior_and_prefix_kernel_update():
    d0, d1 = np.array([[1.0], [0.0]]), np.array([[4.0], [0.0]])
    frozen_prior = np.array([[.5]])
    y = np.array([1.0, 0.0])
    frozen_bf = gaussian_logpdf(y, np.zeros(2), np.eye(2)+d0@frozen_prior@d0.T) - gaussian_logpdf(y, np.zeros(2), np.eye(2)+d1@frozen_prior@d1.T)
    adaptive_bf = gaussian_logpdf(y, np.zeros(2), np.eye(2)+d0@g_prior_covariance(d0, 2)@d0.T) - gaussian_logpdf(y, np.zeros(2), np.eye(2)+d1@g_prior_covariance(d1, 2)@d1.T)
    assert abs(frozen_bf) > .1 and abs(adaptive_bf) < 1e-12
    # The prefix has no candidate contrast but informs the shared uncertain ERP coefficient.
    prior_variance, prefix_value = 2.0, 3.0
    posterior_variance = 1/(1/prior_variance+1)
    posterior_mean = posterior_variance*prefix_value
    target_value = np.array([1.0])
    prefix_conditioned = gaussian_logpdf(target_value, np.array([posterior_mean]), np.array([[1+posterior_variance]]))
    prefix_ignored = gaussian_logpdf(target_value, np.zeros(1), np.array([[1+prior_variance]]))
    # Joint density ratio independently yields the exact posterior-predictive conditional.
    joint_cov = np.array([[1+prior_variance, prior_variance], [prior_variance, 1+prior_variance]])
    ratio = gaussian_logpdf(np.array([prefix_value, target_value[0]]), np.zeros(2), joint_cov) - gaussian_logpdf(np.array([prefix_value]), np.zeros(1), np.array([[1+prior_variance]]))
    assert abs(prefix_conditioned-ratio) < 1e-12 and abs(prefix_conditioned-prefix_ignored) > .01
    return {"frozen_physical_amplitude_prior_log_bf": frozen_bf,
            "candidate_adaptive_gprior_log_bf": adaptive_bf,
            "prefix_shared_kernel_conditional_logpdf_error": abs(prefix_conditioned-ratio),
            "prefix_ignored_logpdf_difference": abs(prefix_conditioned-prefix_ignored),
            "implication": "freeze g-prior on calibration design; candidate-free prefix still updates uncertain shared coefficients"}


def check_student_t_scale_hierarchy():
    # Independent local scales and one global scale both have t-like marginal tails;
    # they imply different posterior downweighting of a clean coordinate.
    degrees, dimensions = 5.0, 2
    residual = np.array([12.0, .3])
    global_precision_mean = (degrees+dimensions)/(degrees+residual@residual)
    local_precision_mean = (degrees+1)/(degrees+residual**2)
    assert local_precision_mean[1] > 1 and global_precision_mean < .1
    assert local_precision_mean[1]/global_precision_mean > 20
    return {"selection_global_precision_posterior_mean": float(global_precision_mean),
            "independent_row_precision_posterior_means": local_precision_mean.tolist(),
            "implication": "a selection-wide scale shock downweights all samples; local t scales downweight the outlier only"}


def check_residual_acf_improves_while_signal_is_removed(rng):
    records, times, phi, amplitude = 8000, 64, .9, 2.0
    innovation = rng.normal(size=(records, times))
    noise = np.empty_like(innovation)
    noise[:, 0] = innovation[:, 0]/np.sqrt(1-phi**2)
    for t in range(1, times):
        noise[:, t] = phi*noise[:, t-1]+innovation[:, t]
    truth = rng.choice([-1.0, 1.0], size=records)
    observed = noise + amplitude*truth[:, None]
    before_hit = float(np.mean(np.sign(observed.mean(axis=1)) == truth))
    whitened_noise = noise.copy()
    whitened_noise[:, 0] *= np.sqrt(1-phi**2)
    whitened_noise[:, 1:] -= phi*noise[:, :-1]
    signal = np.ones(times)
    signal[0] *= np.sqrt(1-phi**2)
    signal[1:] *= 1-phi
    direction = signal/np.linalg.norm(signal)
    projected_noise = whitened_noise - (whitened_noise @ direction)[:, None]*direction
    remaining_signal = signal-(signal @ direction)*direction
    acf_before = float(np.sum(noise[:, 1:]*noise[:, :-1])/np.sum(noise[:, :-1]**2))
    acf_after = float(np.sum(projected_noise[:, 1:]*projected_noise[:, :-1])/np.sum(projected_noise[:, :-1]**2))
    assert before_hit > .9 and abs(acf_before) > .85 and abs(acf_after) < .05
    assert np.linalg.norm(remaining_signal) < 1e-12
    return {"held_noise_lag1_acf_before": acf_before, "held_noise_lag1_acf_after": acf_after,
            "classification_hit_before": before_hit, "remaining_candidate_signal_norm": float(np.linalg.norm(remaining_signal)),
            "classification_hit_after_all_candidate_ties": 0.0,
            "implication": "correct whitening plus an overbroad nuisance projection improves held residual ACF but removes the ERP contrast"}


def main():
    rng = np.random.default_rng(20261005)
    checks = {
        "full_rank_gprior_basis_invariance_and_ridge_failure": lambda: check_basis_invariance(rng),
        "rank_deficiency_and_weak_directions": check_rank_and_weak_directions,
        "matrix_normal_gauge_vs_physical_voltage_units": lambda: check_gauge_and_units(rng),
        "fixed_holdback32_and_actual_conditional_prefix": lambda: check_fixed_support_and_prefix(rng),
        "channel_ar_nonseparability": check_channel_ar_is_not_kronecker,
        "whitening_and_candidate_discrimination_separate": lambda: check_whitening_can_hurt_discrimination(rng),
        "residual_psd_acf_cannot_replace_hit": lambda: check_psd_acf_do_not_imply_hit(rng),
        "latency_derivative_local_approximation": check_latency_derivatives,
        "common_selection_identifiability": check_common_selection_identifiability,
        "probability_mixture_vs_bayes_evidence_weights": check_probability_mixture_is_not_bayes_model_average,
        "reference_rank_and_mean_prior_vs_precision": check_reference_rank_and_mean_prior,
        "candidate_frozen_prior_and_prefix_shared_kernel_conditioning": check_candidate_adaptive_prior_and_prefix_kernel_update,
        "student_t_scale_hierarchy_changes_inference": check_student_t_scale_hierarchy,
        "held_acf_improves_while_candidate_contrast_is_removed": lambda: check_residual_acf_improves_while_signal_is_removed(rng),
    }
    results = {}
    for name, check in checks.items():
        results[name] = {"status": "PASS", **check()}
        print(name, "PASS")
    owner_path = Path(sys.modules[TemporalNoise.__module__].__file__).resolve()
    assert owner_path.is_relative_to(ISOLATED_SRC)
    source_paths = ["src/transfer/temporal_noise.py", "src/transfer/continuous_decision.py", "src/transfer/noise_model.py",
                    "src/baselines/structured_erp.py", "src/baselines/temporal_transfer.py"]
    report = {"schema": "n2p3_latent_research_math/1", "evidence_scope": "independent_synthetic_mechanism_checks_no_public_or_adult_performance",
              "seed": 20261005, "python_executable": sys.executable, "isolated_checkout": str(CHECKOUT),
              "current_owner_import": str(owner_path), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_sha256": {path: hashlib.sha256((CHECKOUT/path).read_bytes()).hexdigest() for path in source_paths},
              "check_count": len(results), "results": results}
    (ROOT / "math_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
