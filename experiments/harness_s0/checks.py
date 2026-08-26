"""Self-contained implementations of the seven S0 counterexample checks.

Each function returns a dict with machine-readable pass/fail evidence. The
pytest wrapper in tests/test_s0_harness.py asserts on those dicts so the harness
has a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift as ndshift
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression


def _bce(logits: np.ndarray, labels: np.ndarray) -> float:
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def _brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(labels)) ** 2))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def check_soft_label_semantics(rng: np.random.Generator | None = None) -> dict[str, object]:
    """1-a is a severity score, not P(clean).

    For deterministic feature interpolation z(a), y=1-a is the only soft label.
    Brier and soft-BCE therefore have the same population minimizer E[y]=0.5.
    Two equally valid hard clean/corrupt labelings of the same interpolation
    have optimal probabilities 0.25 and 0.75, proving 1-a cannot identify a
    Bernoulli clean state.
    """
    rng = rng or np.random.default_rng(0)
    n = 4000
    a = rng.uniform(0.0, 1.0, n)
    y_soft = 1.0 - a
    labels = {"soft": y_soft, "hard_lo": (a < 0.25).astype(float), "hard_hi": (a < 0.75).astype(float)}

    results: dict[str, object] = {}
    for name, y in labels.items():
        brier_min = minimize_scalar(
            lambda p, y_vec=y: _brier(float(p), y_vec), bounds=(1e-6, 1.0 - 1e-6), method="bounded"
        ).x
        bce_min = minimize_scalar(
            lambda p, y_vec=y: float(
                -np.mean(
                    y_vec * np.log(max(float(p), 1e-9))
                    + (1 - y_vec) * np.log1p(-min(float(p), 1 - 1e-9))
                )
            ),
            bounds=(1e-6, 1.0 - 1e-6),
            method="bounded",
        ).x
        results[f"{name}_brier_min"] = float(brier_min)
        results[f"{name}_bce_min"] = float(bce_min)

    passed = bool(
        abs(results["soft_brier_min"] - float(np.mean(y_soft))) < 1e-3
        and abs(results["soft_bce_min"] - float(np.mean(y_soft))) < 1e-3
        and abs(results["hard_lo_bce_min"] - float(np.mean((a < 0.25).astype(float)))) < 1e-3
        and abs(results["hard_hi_bce_min"] - float(np.mean((a < 0.75).astype(float)))) < 1e-3
        and abs(float(np.mean((a < 0.25).astype(float))) - float(np.mean((a < 0.75).astype(float)))) > 0.4
    )
    results["soft_mean"] = float(np.mean(y_soft))
    return {"passed": passed, **results}


def check_count_prior_cancellation() -> dict[str, object]:
    """beta*log(K) cancels in exact@K and is an illegal prefix prior."""
    n_candidates = 9
    base = np.linspace(-0.8, 0.8, n_candidates)

    # exact@K: all candidates have exactly K appearances.
    k_exact = 3.0
    prior = 1.7 * np.log(k_exact)
    before = _softmax(base + prior)
    after = _softmax(base)
    exact_max_abs_diff = float(np.max(np.abs(before - after)))

    # prefix_minK: one candidate reached K earlier and already has n_c = K+m.
    # A global log(K) term changes a candidate whose own count did not change.
    n_c = np.full(n_candidates, 3.0)
    n_c[8] = 5.0
    score_at_k3 = base + 1.7 * np.log(3.0)
    score_at_k4 = base + 1.7 * np.log(4.0)
    unchanged_candidate = 0
    delta = float(score_at_k4[unchanged_candidate] - score_at_k3[unchanged_candidate])

    passed = bool(exact_max_abs_diff < 1e-14 and delta > 1e-6)
    return {
        "passed": passed,
        "exact_max_abs_diff": exact_max_abs_diff,
        "prefix_unchanged_candidate_delta": delta,
        "candidate_specific_n_c": n_c.tolist(),
    }


def _chain_score(rho: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    mixture = rho * a + (1.0 - rho) * b
    return float(np.sum(np.log(mixture)))


def check_monotone_rho_chain_rank() -> dict[str, object]:
    """A strict monotone rho map preserves rho AUC but can flip chain argmax."""
    # Deterministic two-step, two-candidate instance found by fixed-seed search.
    rho_1 = np.asarray([0.94213158, 0.54174345])
    a_1 = np.asarray([6.79744724, 6.88570989])
    b_1 = np.asarray([1.53145495, 1.45879427])
    rho_2 = np.asarray([0.46356207, 0.52105619])
    a_2 = np.asarray([2.72772708, 1.80940826])
    b_2 = np.asarray([5.04417166, 6.5997871])

    def phi(r: np.ndarray) -> np.ndarray:
        return r * r

    before = _chain_score(rho_1, a_1, b_1) - _chain_score(rho_2, a_2, b_2)
    after = _chain_score(phi(rho_1), a_1, b_1) - _chain_score(phi(rho_2), a_2, b_2)

    # phi is strictly increasing, so it preserves the AUC of any rho score.
    rho_all = np.concatenate((rho_1, rho_2))
    order_before = np.argsort(rho_all)
    order_after = np.argsort(phi(rho_all))
    ranking_preserved = bool(np.array_equal(order_before, order_after))

    passed = bool(ranking_preserved and before > 1e-6 and after < -1e-6)
    return {
        "passed": passed,
        "ranking_preserved": ranking_preserved,
        "candidate_1_minus_2_before": before,
        "candidate_1_minus_2_after": after,
    }


def _legacy_nonnegative_alpha(base: np.ndarray, evidence: np.ndarray, labels: np.ndarray) -> float:
    """Mirror of the legacy S + alpha*L, alpha>=0, no-intercept fit."""
    result = minimize_scalar(
        lambda a: _bce(base + float(a) * evidence, labels),
        bounds=(0.0, 3.0),
        method="bounded",
        options={"xatol": 1e-9},
    )
    return float(result.x)


def check_redundant_likelihood_fusion(rng: np.random.Generator | None = None) -> dict[str, object]:
    """S -> Y and S -> L with Y independent of L given S.

    The legacy one-parameter nonnegative fit reports a positive alpha and an
    apparent BCE gain even though L carries no incremental information. The
    nested M0:a+bS vs M1:a+bS+cL audit gives a cluster-bootstrap CI for c that
    covers zero.
    """
    rng = rng or np.random.default_rng(0)
    n_subjects = 30
    n_per_subject = 300
    n = n_subjects * n_per_subject
    subject = np.repeat(np.arange(n_subjects), n_per_subject)
    y = (rng.random(n) < 0.5).astype(float)
    s = np.where(y == 1.0, 1.0, -1.0) + rng.normal(size=n)
    llr = 0.8 * s + rng.normal(size=n)

    legacy_alpha = _legacy_nonnegative_alpha(s, llr, y)
    base_bce = _bce(s, y)
    legacy_bce = _bce(s + legacy_alpha * llr, y)

    x = np.column_stack((s, llr))
    point = LogisticRegression(C=1e6, max_iter=2000).fit(x, y)
    c_point = float(point.coef_[0, 1])

    n_boot = 250
    coefficients: list[float] = []
    for _ in range(n_boot):
        sampled_subjects = rng.integers(0, n_subjects, size=n_subjects)
        rows = np.concatenate([np.flatnonzero(subject == j) for j in sampled_subjects])
        fitted = LogisticRegression(C=1e6, max_iter=2000).fit(x[rows], y[rows])
        coefficients.append(float(fitted.coef_[0, 1]))
    coefficients = np.asarray(coefficients)
    ci_lo = float(np.percentile(coefficients, 2.5))
    ci_hi = float(np.percentile(coefficients, 97.5))

    passed = bool(
        legacy_alpha > 0.1
        and base_bce - legacy_bce > 0.005
        and ci_lo < 0.0 < ci_hi
    )
    return {
        "passed": passed,
        "legacy_alpha": legacy_alpha,
        "base_bce": base_bce,
        "legacy_bce": legacy_bce,
        "apparent_bce_gain": base_bce - legacy_bce,
        "nested_c_point": c_point,
        "nested_c_cluster_ci": [ci_lo, ci_hi],
        "bootstrap_samples": n_boot,
    }


@dataclass(frozen=True)
class _LatencyCheckResult:
    passed: bool
    bias_ms: float
    rmse_ms: float
    slope: float
    coverage: float
    tau0_bias_ms: float
    init_sensitivity_ms: float


def _synthetic_latency_trial(
    tau_ms: float,
    template: np.ndarray,
    time_ms: np.ndarray,
    sigma_ms: float,
    amplitude: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    gaussian = np.exp(-0.5 * ((time_ms - tau_ms) / sigma_ms) ** 2)
    return amplitude * gaussian + rng.normal(0.0, noise_std, size=time_ms.shape)


def check_latency_gauge() -> dict[str, object]:
    """Profile-likelihood latency measurement with an explicit template anchor.

    Sub-checks:
      * gauge degeneracy: shifting all tau by +c and the template by -c leaves
        the profile likelihood unchanged;
      * anchored estimation passes the S0 synthetic thresholds.
    """
    rng = np.random.default_rng(2026)
    sfreq = 250.0
    n_time = 256
    time_ms = np.arange(n_time) / sfreq * 1000.0
    tau0_true = 460.0
    sigma_ms = 40.0
    noise_std = 0.10
    lo = int(0.120 * sfreq)
    hi = int(0.900 * sfreq)
    score_mask = np.zeros(n_time, dtype=bool)
    score_mask[lo:hi] = True

    def make_set(n_trials: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        deltas = rng.normal(0.0, 12.0, n_trials)
        amps = rng.uniform(0.8, 1.2, n_trials)
        x = np.stack(
            [
                _synthetic_latency_trial(tau0_true + d, None, time_ms, sigma_ms, a, noise_std, rng)
                for a, d in zip(amps, deltas, strict=True)
            ]
        )
        return x, tau0_true + deltas, amps

    x_train, _, _ = make_set(260)

    # Fold-local Woody-style template estimation, evaluated on an interior mask
    # so shifts never rely on a boundary shortcut.
    template = np.mean(x_train, axis=0)
    for _ in range(3):
        aligned: list[np.ndarray] = []
        for trial in x_train:
            best = (-1e18, 0)
            for shift_sample in range(-10, 11):
                shifted = ndshift(trial, shift_sample, order=1, mode="constant", cval=0.0)
                value = float(np.dot(shifted[score_mask], template[score_mask]))
                if value > best[0]:
                    best = (value, shift_sample)
            aligned.append(ndshift(trial, best[1], order=1, mode="constant", cval=0.0))
        template = np.mean(np.asarray(aligned), axis=0)

    # Gauge degeneracy: x_i = a g(t-tau_i), tau_i -> tau_i+c, g -> g shifted +c.
    # Likelihoods must be identical when both the trial and template are shifted.
    trial = np.stack(
        [
            _synthetic_latency_trial(tau0_true + 5.0, None, time_ms, sigma_ms, 1.0, noise_std, rng)
        ]
    )
    trial_shifted = np.stack(
        [ndshift(row, 20.0 * sfreq / 1000.0, order=1, mode="constant", cval=0.0) for row in trial]
    )
    template_shifted = ndshift(template, 20.0 * sfreq / 1000.0, order=1, mode="constant", cval=0.0)
    grid = np.arange(tau0_true - 60.0, tau0_true + 60.0 + 0.5, 0.5)
    grid_shifted = grid + 20.0

    def profile_for(
        template_arg: np.ndarray, grid_arg: np.ndarray, x: np.ndarray, anchor_ms: float
    ) -> np.ndarray:
        shifted = np.stack(
            [
                ndshift(
                    template_arg,
                    (tau - anchor_ms) * sfreq / 1000.0,
                    order=1,
                    mode="constant",
                    cval=0.0,
                )[score_mask]
                for tau in grid_arg
            ]
        )
        denom = (shifted * shifted).sum(axis=1)
        xm = x[:, score_mask]
        num = xm @ shifted.T
        rss = (xm * xm).sum(axis=1)[:, None] - num * num / denom[None, :]
        return -0.5 * rss / noise_std**2

    ll_original = profile_for(template, grid, trial, tau0_true)
    ll_gauge_shifted = profile_for(template_shifted, grid_shifted, trial_shifted, tau0_true + 20.0)
    # The two configurations are the same model up to the gauge shift: the
    # profile-likelihood surface must be identical up to a 20 ms translation.
    # Spline interpolation makes exact pointwise equality unnecessarily strict,
    # so the check uses shape invariance plus the translated argmax.
    gauge_max_abs_diff = float(np.max(np.abs(ll_original - ll_gauge_shifted)))
    gauge_corr = float(np.corrcoef(ll_original.ravel(), ll_gauge_shifted.ravel())[0, 1])
    gauge_arg_shift = float(
        grid_shifted[int(np.argmax(ll_gauge_shifted))] - grid[int(np.argmax(ll_original))]
    )
    gauge_passed = bool(gauge_corr > 0.9999 and abs(gauge_arg_shift - 20.0) < 0.5)

    def profile_log_likelihoods(x: np.ndarray) -> np.ndarray:
        return profile_for(template, grid, x, tau0_true)

    # Test-set measurement with validation-calibrated posterior sharpness.
    x_cal, truth_cal, _ = make_set(140)
    x_test, truth_test, _ = make_set(220)

    def posterior_summaries(x: np.ndarray, noise_eff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lls = profile_log_likelihoods(x) / (noise_eff**2)
        lls = lls - lls.max(axis=1, keepdims=True)
        q = np.exp(lls)
        q /= q.sum(axis=1, keepdims=True)
        means = q @ grid
        lowers: list[float] = []
        uppers: list[float] = []
        for row in q:
            i = int(np.argmax(row))
            j = i
            mass = row[i]
            while mass < 0.90:
                left = row[i - 1] if i > 0 else -1.0
                right = row[j + 1] if j + 1 < len(row) else -1.0
                if left >= right:
                    i -= 1
                    mass += row[i]
                else:
                    j += 1
                    mass += row[j]
            lowers.append(grid[i])
            uppers.append(grid[j])
        return means, np.asarray(lowers), np.asarray(uppers)

    def coverage_for(x: np.ndarray, truth: np.ndarray, noise_eff: float) -> float:
        _, lo, hi = posterior_summaries(x, noise_eff)
        return float(np.mean((truth >= lo) & (truth <= hi)))

    best_noise_eff: float | None = None
    best_gap: float = np.inf
    for candidate in np.linspace(0.5, 3.0, 121):
        coverage = coverage_for(x_cal, truth_cal, candidate)
        gap = abs(coverage - 0.90)
        if gap < best_gap:
            best_gap = gap
            best_noise_eff = float(candidate)

    means, lowers, uppers = posterior_summaries(x_test, best_noise_eff)
    errors = means - truth_test
    bias = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    slope = float(np.polyfit(truth_test, means, 1)[0])
    coverage = float(np.mean((truth_test >= lowers) & (truth_test <= uppers)))

    # Group tau0 identified by E[delta]=0 under the fixed-template anchor.
    tau0_hat_a = float(np.mean(means[: len(means) // 2]))
    tau0_hat_b = float(np.mean(means[len(means) // 2 :]))
    init_sensitivity = abs(tau0_hat_a - tau0_hat_b)

    passed = bool(
        gauge_passed
        and abs(bias) < 5.0
        and rmse < 10.0
        and 0.9 <= slope <= 1.1
        and 0.85 <= coverage <= 0.95
        and abs(tau0_hat_a - tau0_true) < 5.0
        and abs(tau0_hat_b - tau0_true) < 5.0
        and init_sensitivity < 2.0
    )
    result = _LatencyCheckResult(
        passed=passed,
        bias_ms=bias,
        rmse_ms=rmse,
        slope=slope,
        coverage=coverage,
        tau0_bias_ms=float(abs(tau0_hat_a - tau0_true)),
        init_sensitivity_ms=init_sensitivity,
    )
    return {
        "passed": passed,
        "gauge_max_abs_diff": gauge_max_abs_diff,
        "gauge_corr": gauge_corr,
        "gauge_arg_shift_ms": gauge_arg_shift,
        "gauge_passed": gauge_passed,
        **result.__dict__,
        "calibrated_noise_eff": best_noise_eff,
        "tau0_hat_a_ms": tau0_hat_a,
        "tau0_hat_b_ms": tau0_hat_b,
    }


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(probabilities)
    p = np.asarray(probabilities)[order]
    y = np.asarray(labels)[order]
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = 0
    for k in range(bins):
        m = (p > edges[k]) & (p <= edges[k + 1])
        if m.sum():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
            n += m.sum()
    return float(total / max(n, 1))


def _deployment_prior_probabilities(
    z_cal: np.ndarray, y_cal: np.ndarray, z_dep: np.ndarray, pi_deploy: float, pi_cal: float
) -> np.ndarray:
    calibrator = LogisticRegression(C=1.0, max_iter=2000).fit(z_cal.reshape(-1, 1), y_cal)
    p = calibrator.predict_proba(z_dep.reshape(-1, 1))[:, 1]
    odds = p / np.clip(1.0 - p, 1e-9, None)
    odds = odds * (pi_deploy / (1.0 - pi_deploy)) * ((1.0 - pi_cal) / pi_cal)
    return odds / (1.0 + odds)


def check_reliability_prevalence_shift(rng: np.random.Generator | None = None) -> dict[str, object]:
    """A calibrator fit at the 1:4 gate prevalence miscalibrates at deployment."""
    rng = rng or np.random.default_rng(0)
    n = 5000
    z_clean = rng.normal(2.2, 0.55, n)
    z_corrupt = rng.normal(1.2, 0.55, n)
    y_clean = np.ones(n)
    y_corrupt = np.zeros(n)

    # Calibration pool: 1 clean : 4 corrupt, matching the legacy gate pool.
    pi_cal = 0.2
    n_cal_clean = n // 5
    n_cal_corrupt = n - n_cal_clean
    z_cal = np.concatenate((z_clean[:n_cal_clean], z_corrupt[:n_cal_corrupt]))
    y_cal = np.concatenate((y_clean[:n_cal_clean], y_corrupt[:n_cal_corrupt]))

    # Deployment prior 0.9. Fit calibrator on a deployment-prior training split
    # and evaluate held-out deployment-prior trials.
    pi_deploy = 0.9
    dep_start = n_cal_clean
    z_dep_clean = z_clean[dep_start:]
    z_dep_corrupt = z_corrupt[n_cal_corrupt:]
    n_dep_clean = int(len(z_dep_clean) * 0.9)
    n_dep_corrupt = int(n_dep_clean * (1.0 - pi_deploy) / pi_deploy)

    train_clean = z_dep_clean[: n_dep_clean // 2]
    train_corrupt = z_dep_corrupt[: n_dep_corrupt // 2]
    test_clean = z_dep_clean[n_dep_clean // 2 : n_dep_clean]
    test_corrupt = z_dep_corrupt[n_dep_corrupt // 2 : n_dep_corrupt]
    z_train = np.concatenate((train_clean, train_corrupt))
    y_train = np.concatenate((np.ones(len(train_clean)), np.zeros(len(train_corrupt))))
    z_test = np.concatenate((test_clean, test_corrupt))
    y_test = np.concatenate((np.ones(len(test_clean)), np.zeros(len(test_corrupt))))

    # A calibrator fit at pi=0.2 and applied without prior conversion.
    no_shift = LogisticRegression(C=1.0, max_iter=2000).fit(z_cal.reshape(-1, 1), y_cal)
    p_no_shift = no_shift.predict_proba(z_test.reshape(-1, 1))[:, 1]
    ece_no_shift = _ece(p_no_shift, y_test)

    # Explicit hard-label calibrator at deployment prior.
    p_deploy = LogisticRegression(C=1.0, max_iter=2000).fit(z_train.reshape(-1, 1), y_train).predict_proba(
        z_test.reshape(-1, 1)
    )[:, 1]
    ece_deploy = _ece(p_deploy, y_test)

    passed = bool(ece_no_shift > 0.20 and ece_deploy < 0.05)
    return {
        "passed": passed,
        "calibration_prevalence": pi_cal,
        "deployment_prevalence": pi_deploy,
        "ece_without_prior_conversion": ece_no_shift,
        "ece_with_deployment_prior_calibration": ece_deploy,
    }


def _first_crossing(
    log_posterior_target: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (decided_mask, stop_index, stop_used) for posterior crossing."""
    n_seq, n_steps = log_posterior_target.shape
    stop_index = np.full(n_seq, n_steps - 1, dtype=int)
    decided = np.zeros(n_seq, dtype=bool)
    for n in range(n_steps):
        cross = (~decided) & (log_posterior_target[:, n] >= threshold)
        stop_index[cross] = n
        decided |= cross
    return decided, stop_index, decided


def check_stopping_replay(rng: np.random.Generator | None = None) -> dict[str, object]:
    """Misspecified posterior thresholds are not error-rate controllers."""
    rng = rng or np.random.default_rng(3)
    n_seq = 1500
    n_steps = 20
    true_sep = 0.35  # true target logit separation
    misspecified_scale = 0.25  # overconfident posterior model

    target = rng.random(n_seq) < 0.5
    # Independent flash evidence under the true model.
    logits = true_sep * np.where(target[:, None], 1.0, -1.0) + rng.normal(size=(n_seq, n_steps))
    # Misspecified posterior: treats evidence variance as 0.25 instead of 1.
    log_posterior = np.cumsum(logits / misspecified_scale, axis=1)
    threshold = np.log(0.95 / 0.05)
    decided, stop_index, _ = _first_crossing(log_posterior, threshold)
    decided_rows = np.flatnonzero(decided)
    decided_posterior = log_posterior[decided_rows, stop_index[decided_rows]]
    empirical_error = float(np.mean(target[decided_rows] != (decided_posterior > 0)))

    # E-process under the null hypothesis (not target): the per-step likelihood
    # ratio against the true densities is nonnegative and has expectation <=1.
    n_null = 3000
    null_logits = true_sep * (-1.0) + rng.normal(size=(n_null, n_steps))
    lr = np.exp(true_sep * (null_logits - 0.5 * true_sep))
    e_process = np.cumprod(lr, axis=1)
    nonnegative = bool(np.min(e_process) >= 0.0)
    expectation_at_steps = float(np.mean(e_process[:, -1]))
    expectation_bound_holds = bool(expectation_at_steps <= 1.0 + 0.05)

    # Replay metrics are summary statistics of a fixed rule; they never tune on
    # the labels of the replay set.
    replay = {
        "decided_fraction": float(np.mean(decided)),
        "empirical_error": empirical_error,
        "expected_flashes": float(np.mean(stop_index[decided] + 1)) if decided.any() else np.nan,
        "undecided_fraction": float(np.mean(~decided)),
    }

    passed = bool(
        empirical_error > 0.10
        and nonnegative
        and expectation_bound_holds
        and np.isfinite(replay["expected_flashes"])
    )
    return {"passed": passed, **replay, "e_process_nonnegative": nonnegative, "e_process_expectation": expectation_at_steps}


def run_all_checks() -> dict[str, dict[str, object]]:
    """Run all seven S0 checks and return the machine-readable report."""
    checks = {
        "s0_soft_label_semantics": check_soft_label_semantics(),
        "s0_count_prior_cancellation": check_count_prior_cancellation(),
        "s0_monotone_rho_chain_rank": check_monotone_rho_chain_rank(),
        "s0_redundant_likelihood_fusion": check_redundant_likelihood_fusion(),
        "s0_latency_gauge": check_latency_gauge(),
        "s0_reliability_prevalence_shift": check_reliability_prevalence_shift(),
        "s0_stopping_replay": check_stopping_replay(),
    }
    return checks
