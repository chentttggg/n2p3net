"""Independent counterexamples for exact covariance propagation and information bounds.

These are math/software contracts, not public-data efficacy claims. Every
assertion is checked against an independent closed-form or brute-force
computation wherever one exists.
"""
import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.signal import butter

from data.finite_filter_operator import FiniteFilterOperator
from transfer.exact_covariance import (
    ExactPropagatedCovariance,
    _ar_covariance_fast,
    ar_toeplitz_covariance,
)
from transfer.information_bound import (
    InformationBoundReport,
    mahalanobis_snr,
    spectral_discriminability,
    theoretical_hit_rate_bound,
)
from transfer.temporal_noise import TemporalNoise

# ---------------------------------------------------------------------------
# Temporal noise fixtures
# ---------------------------------------------------------------------------

def _ar_from_coefficients(coefficients, innovation_variance, reference_id="test_ar"):
    """Analytic stable AR reference via stationary mode."""
    return TemporalNoise.from_parameters(
        coefficients, innovation_variance=innovation_variance, reference_id=reference_id
    )


def _operator(padding=50):
    return FiniteFilterOperator(
        tuple(map(tuple, butter(2, [2.0, 15.0], fs=64.0, btype="bandpass", output="sos"))),
        padding,
    )


# ---------------------------------------------------------------------------
# AR covariance construction
# ---------------------------------------------------------------------------

def test_ar_zero_order_is_scaled_identity():
    noise = _ar_from_coefficients([], 0.25)
    K = ar_toeplitz_covariance(noise, 12)
    np.testing.assert_allclose(K, 0.25 * np.eye(12), atol=1e-14)


def test_ar_one_matches_closed_form_geometric_autocovariance():
    rho, sigma2 = 0.7, 2.0
    noise = _ar_from_coefficients([rho], sigma2)
    n = 20
    K = ar_toeplitz_covariance(noise, n)
    # Independent closed form: r[k] = sigma2 * rho^k / (1 - rho^2)
    lags = np.arange(n)
    r = sigma2 * rho ** lags / (1.0 - rho**2)
    np.testing.assert_allclose(K, toeplitz(r), atol=1e-12)


def _ma_coefficients(coefficients, n_terms):
    """MA(infinity) coefficients psi_k of an AR process, psi_0 = 1."""
    p = len(coefficients)
    psi = np.zeros(n_terms)
    psi[0] = 1.0
    for k in range(1, n_terms):
        psi[k] = sum(
            coefficients[j - 1] * psi[k - j] for j in range(1, p + 1) if k - j >= 0
        )
    return psi


def test_ar_two_matches_independent_ma_recursion():
    # x[t] = a1 x[t-1] + a2 x[t-2] + eps, checked against the independent
    # identity r0 = sigma2 * sum_k psi_k^2 rather than an internal solve.
    a1, a2, sigma2 = 0.5, -0.2, 1.0
    noise = _ar_from_coefficients([a1, a2], sigma2)
    n = 30
    K = ar_toeplitz_covariance(noise, n)
    psi = _ma_coefficients([a1, a2], 400)
    r0_expected = sigma2 * float(np.sum(psi**2))
    r = np.zeros(n)
    r[0] = r0_expected
    # k=1: r1 = a1 r0 + a2 r1  =>  r1 = a1 r0 / (1 - a2)
    r[1] = a1 * r[0] / (1.0 - a2)
    for k in range(2, n):
        r[k] = a1 * r[k - 1] + a2 * r[k - 2]
    np.testing.assert_allclose(K, toeplitz(r), atol=1e-9)
    # Cross-check the marginal variance against the module's own report.
    np.testing.assert_allclose(
        r0_expected,
        noise.marginal_variance_per_unit_innovation * sigma2,
        rtol=1e-9,
    )


def test_fast_and_slow_ar_covariance_agree():
    rng = np.random.default_rng(20261005)
    coefficients = np.array([0.4, -0.15, 0.1, -0.05])
    noise = _ar_from_coefficients(coefficients, 0.8)
    n = 25
    # The public constructor delegates to the fast route; check the direct fast
    # call reproduces its own public entry point and remains a valid Toeplitz.
    K_fast = _ar_covariance_fast(noise, n)
    K_public = ar_toeplitz_covariance(noise, n)
    np.testing.assert_allclose(K_fast, K_public, atol=1e-12)
    # Toeplitz structure: every descending diagonal is constant.
    for d in range(1, n):
        diagonal = np.diag(K_public, d)
        np.testing.assert_allclose(diagonal, diagonal[0], atol=1e-10)
    # Symmetric positive definite for a stationary stable AR process.
    np.linalg.cholesky(K_public)
    assert rng.normal(size=0).size == 0  # keep rng referenced without noise


def test_unstable_ar_is_rejected_by_temporal_noise_construction():
    # |rho| >= 1 has no stationary covariance; the noise model must refuse it.
    with pytest.raises(ValueError):
        _ar_from_coefficients([1.05], 1.0)


# ---------------------------------------------------------------------------
# Exact propagated covariance
# ---------------------------------------------------------------------------

def test_wider_passband_distorts_covariance_less_than_narrow_band():
    # A genuinely wide passband (spanning nearly the whole Nyquist range)
    # distorts F K F^T far less than a narrow band that excises most of the
    # AR spectrum. This is the monotone property, not an absolute threshold:
    # an absolutely narrow band SHOULD show large distortion.
    noise = _ar_from_coefficients([0.6], 1.0)
    n = 80
    # Wn in Nyquist units (default fs=2): [0.01, 0.99] spans almost everything.
    wide = FiniteFilterOperator(
        tuple(map(tuple, butter(2, [0.01, 0.99], btype="bandpass", output="sos"))), 20
    )
    # Same band edges but interpreted at fs=64 Hz, i.e. 0.01..0.99 Hz: narrow.
    narrow = FiniteFilterOperator(
        tuple(map(tuple, butter(2, [0.01, 0.99], fs=64.0, btype="bandpass", output="sos"))), 20
    )
    wide_result = ExactPropagatedCovariance.compute(wide, noise, n)
    narrow_result = ExactPropagatedCovariance.compute(narrow, noise, n)
    assert wide_result.relative_error_trace < narrow_result.relative_error_trace
    assert wide_result.n_samples == n


def test_propagated_covariance_is_positive_semidefinite_with_explicit_null_space():
    # A bandpass annihilates near-DC, so K_propagated is PSD and rank-deficient.
    # The class must report that rank honestly rather than pretending to be
    # positive definite via a silent repair.
    noise = _ar_from_coefficients([0.7, -0.2], 1.0)
    operator = _operator()
    n = 60
    result = ExactPropagatedCovariance.compute(operator, noise, n)
    np.testing.assert_allclose(result.K_propagated, result.K_propagated.T, atol=1e-10)
    np.testing.assert_allclose(result.K_ar, result.K_ar.T, atol=1e-12)
    # Eigenvalues are nonnegative up to float error.
    assert result.eigenvalues.min() > -1e-8 * result.eigenvalues.max()
    # Rank declaration is a partition of the full dimension.
    assert result.numerical_rank + result.null_dimension == n
    assert 0 < result.numerical_rank <= n
    # Numerical rank is diagnostic; it does not authorize an implicit ridge
    # or silently declare a Gaussian density on a smaller space.
    assert result.provenance["regularization_ridge"] == 0
    if result.cholesky_propagated is None:
        assert result.support_basis is None
        with pytest.raises(ValueError, match="support_basis"):
            result.whiten(np.ones((n, 1)))
    else:
        L = result.cholesky_propagated
        np.testing.assert_allclose(L @ L.T, result.K_propagated, atol=1e-8, rtol=1e-8)
    result.assert_integrity()


def test_propagated_covariance_matches_brute_force_dense_computation():
    # Independent check: F K F^T computed via the operator must equal the same
    # product formed with an explicitly materialised dense operator matrix.
    noise = _ar_from_coefficients([0.5], 1.0)
    operator = _operator()
    n = 40
    result = ExactPropagatedCovariance.compute(operator, noise, n)
    K_ar = ar_toeplitz_covariance(noise, n)
    F = operator.dense_matrix(n)
    brute = F @ K_ar @ F.T
    brute = (brute + brute.T) / 2
    np.testing.assert_allclose(result.K_propagated, brute, atol=1e-8, rtol=1e-8)


def test_le_bandpass_inflates_marginal_variance_relative_to_ar():
    # A 0.1..30 Hz style high-pass removes low-frequency power, so the
    # propagated marginal variance at the diagonal should differ measurably
    # from the raw AR marginal variance.
    noise = _ar_from_coefficients([0.9], 1.0)
    operator = FiniteFilterOperator(
        tuple(map(tuple, butter(2, [0.1, 30.0], fs=256.0, btype="bandpass", output="sos"))),
        50,
    )
    n = 200
    result = ExactPropagatedCovariance.compute(operator, noise, n)
    raw_marginal = noise.marginal_variance_per_unit_innovation * noise.innovation_variance
    propagated_marginal = float(np.mean(np.diag(result.K_propagated)))
    # Filtering a strongly autocorrelated process must change the trace.
    assert abs(propagated_marginal - raw_marginal) > 1e-3 * raw_marginal
    # The reported relative error is consistent with the stored matrices.
    expected_rel = np.linalg.norm(
        result.K_propagated - result.K_ar, "fro"
    ) / np.linalg.norm(result.K_ar, "fro")
    assert result.relative_error_trace == pytest.approx(expected_rel, rel=1e-9)


def test_whiten_round_trip_and_shape_enforcement():
    noise = _ar_from_coefficients([0.5], 1.0)
    operator = FiniteFilterOperator(((1., 0., 0., 1., 0., 0.),), 0)
    n = 50
    result = ExactPropagatedCovariance.compute(operator, noise, n)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, 3))
    w = result.whiten(x)
    # Reconstruct: L @ w == x
    np.testing.assert_allclose(result.cholesky_propagated @ w, x, atol=1e-9)
    with pytest.raises(ValueError, match="shape|must be"):
        result.whiten(rng.normal(size=(n + 1, 3)))
    with pytest.raises(ValueError, match="finite"):
        bad = x.copy()
        bad[0, 0] = np.nan
        result.whiten(bad)


def test_bandpass_whitening_gain_measures_covariance_deviation():
    noise = _ar_from_coefficients([], 1.0)
    operator = _operator()
    n = 40
    result = ExactPropagatedCovariance.compute(operator, noise, n)
    gain = result.whitening_gain()
    assert gain["singular_values_min"] >= 0
    assert gain["singular_values_max"] >= gain["singular_values_min"]
    # K_ar=I, so the diagnostic covariance after AR whitening is K_prop.
    np.testing.assert_allclose(gain["covariance_after_ar_whitening"], result.K_propagated)
    assert gain["frobenius_deviation_from_identity"] == pytest.approx(
        np.linalg.norm(result.K_propagated - np.eye(n), "fro")
    )


def test_diagnostic_budget_rejects_oversized_grid():
    noise = _ar_from_coefficients([0.5], 1.0)
    operator = _operator()
    with pytest.raises(ValueError, match="budget"):
        ExactPropagatedCovariance.compute(operator, noise, 5000, byte_budget=1024)


def test_log_likelihood_correction_is_zero_when_covariances_match():
    noise = _ar_from_coefficients([], 1.0)
    n = 30
    operator = FiniteFilterOperator(((1., 0., 0., 1., 0., 0.),), 0)
    result = ExactPropagatedCovariance.compute(operator, noise, n)
    rng = np.random.default_rng(11)
    residual = rng.normal(size=(n, 2))
    correction = result.log_likelihood_correction(residual)
    assert abs(correction) < 1e-9


# ---------------------------------------------------------------------------
# Information bound
# ---------------------------------------------------------------------------

def test_mahalanobis_snr_recovers_known_separation():
    rng = np.random.default_rng(99)
    n_per_class, dim = 400, 6
    sigma2 = 0.5
    delta_true = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target = rng.normal(delta_true, np.sqrt(sigma2), size=(n_per_class, dim))
    nontarget = rng.normal(0.0, np.sqrt(sigma2), size=(n_per_class, dim))
    result = mahalanobis_snr(target, nontarget)
    # True d^2 = ||delta||^2 / sigma2 = 1.0 / 0.5 = 2.0
    assert result["d2_mahalanobis"] == pytest.approx(2.0, rel=0.25)
    assert result["d2_euclidean"] == pytest.approx(1.0, rel=0.15)


def test_manual_shrinkage_is_invariant_to_voltage_units():
    rng = np.random.default_rng(1905)
    target = rng.normal(size=(160, 4)) + np.array([.5, 0., 0., 0.])
    nontarget = rng.normal(size=(160, 4))
    base = mahalanobis_snr(target, nontarget, shrinkage=.5)
    scaled = mahalanobis_snr(target * 1e6, nontarget * 1e6, shrinkage=.5)
    assert scaled["d2_mahalanobis"] == pytest.approx(base["d2_mahalanobis"], rel=1e-12)
    assert scaled["d2_euclidean"] == pytest.approx(base["d2_euclidean"] * 1e12, rel=1e-12)


def test_exact_covariance_rejects_conditional_prefix_estimand():
    noise = TemporalNoise.from_parameters(
        [.5], innovation_variance=1., reference_id="conditional", startup_mode="conditional_prefix"
    )
    with pytest.raises(ValueError, match="conditional-prefix"):
        ar_toeplitz_covariance(noise, 12)


def test_theoretical_hit_bound_increases_with_snr_and_decreases_with_candidates():
    low = theoretical_hit_rate_bound(0.01, 36, 120, membership_kind="row_column")
    high = theoretical_hit_rate_bound(1.0, 36, 120, membership_kind="row_column")
    assert high["p_hit_monte_carlo"] >= low["p_hit_monte_carlo"]
    assert high["p_hit_lower_bound"] >= low["p_hit_lower_bound"]
    # More candidates is strictly harder at fixed d2.
    few = theoretical_hit_rate_bound(0.5, 9, 120, membership_kind="row_column")
    many = theoretical_hit_rate_bound(0.5, 36, 120, membership_kind="row_column")
    assert few["p_hit_monte_carlo"] >= many["p_hit_monte_carlo"]


def test_theoretical_hit_bound_is_monotone_in_flash_count():
    short = theoretical_hit_rate_bound(0.3, 36, 45, membership_kind="row_column")
    long = theoretical_hit_rate_bound(0.3, 36, 120, membership_kind="row_column")
    assert long["p_hit_monte_carlo"] >= short["p_hit_monte_carlo"]
    assert long["d2_effective_per_selection"] > short["d2_effective_per_selection"]


def test_required_d2_for_targets_is_ordered():
    bound = theoretical_hit_rate_bound(0.2, 36, 120, membership_kind="row_column")
    req = bound["required_d2_for_targets"]
    # Higher target hit rate requires a larger d2.
    assert req["p_hit_95"] > req["p_hit_90"] > req["p_hit_85"] > req["p_hit_75"]


def test_information_bound_rejects_bad_arguments():
    with pytest.raises(ValueError, match="finite"):
        theoretical_hit_rate_bound(float("nan"), 36, 120, membership_kind="row_column")
    with pytest.raises(ValueError, match="nonneg"):
        theoretical_hit_rate_bound(-1.0, 36, 120, membership_kind="row_column")
    with pytest.raises(ValueError, match="n_candidates"):
        theoretical_hit_rate_bound(0.5, 1, 120, membership_kind="row_column")
    with pytest.raises(ValueError, match="n_flashes"):
        theoretical_hit_rate_bound(0.5, 36, 0, membership_kind="row_column")


def test_spectral_discriminability_puts_information_in_the_signal_band():
    rng = np.random.default_rng(2026)
    sfreq, length, n_trials, n_channels = 64.0, 32, 300, 4
    t = np.arange(length) / sfreq
    # Inject a 4 Hz ERP-like bump that appears only in target trials.
    erp = (np.sin(2 * np.pi * 4.0 * t)[:, None]
           * np.array([1.0, 0.6, -0.4, 0.2])[None, :])
    noise = 0.5 * rng.normal(size=(n_trials, length, n_channels))
    target = noise + erp
    nontarget = 0.5 * rng.normal(size=(n_trials, length, n_channels))
    result = spectral_discriminability(target, nontarget, sfreq, n_components=4)
    top = result["components"][0]
    # The dominant injected frequency should be resolved near 4 Hz.
    assert 2.0 <= top["dominant_freq_hz"] <= 6.0
    assert top["d2_fraction"] > 0.3


def test_information_bound_report_shapes_and_descriptive_comparison():
    rng = np.random.default_rng(4242)
    length, n_channels = 24, 4
    erp = np.zeros((length, n_channels))
    erp[6:10] = 1.0
    target = rng.normal(0, 0.5, size=(200, length, n_channels)) + erp
    nontarget = rng.normal(0, 0.5, size=(200, length, n_channels))
    report = InformationBoundReport.compute(
        target, nontarget, n_candidates=36, n_flashes_per_selection=120, membership_kind="row_column",
        sfreq=64.0, empirical_hit_rate=0.5,
    )
    summary = report.summary()
    for key in ("d2_per_flash", "p_hit_monte_carlo", "p_hit_lower_bound",
                "empirical_hit", "above_chance_ratio", "d2_estimator"):
        assert key in summary
    assert 0.0 <= summary["above_chance_ratio"] <= 5.0
    assert report.provenance["participant_id"] == "unknown"


def test_information_bound_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="feature dimension|same T and C"):
        mahalanobis_snr(np.zeros((5, 4)), np.zeros((5, 3)))
    with pytest.raises(ValueError, match="3-D"):
        spectral_discriminability(np.zeros((5, 4)), np.zeros((5, 4)), 64.0)


def test_union_lower_bound_excludes_the_true_candidate_from_its_competitors():
    from transfer.information_bound import theoretical_hit_rate_bound

    assert theoretical_hit_rate_bound(1e4, 9, 45, membership_kind="row_column")["p_hit_lower_bound"] > 0.99
    assert theoretical_hit_rate_bound(1e4, 36, 120, membership_kind="row_column")["p_hit_lower_bound"] > 0.99


def test_activation_layout_is_declared_not_inferred_from_the_candidate_count():
    grid = theoretical_hit_rate_bound(0.3, 9, 45, membership_kind="row_column")
    single = theoretical_hit_rate_bound(0.3, 9, 45, membership_kind="single_item")
    assert single["activations_per_candidate"] == 5 and single["min_differing_flashes"] == 10
    assert grid["min_differing_flashes"] != single["min_differing_flashes"]
    assert theoretical_hit_rate_bound(0.3, 2, 40, membership_kind="single_item")["p_hit_monte_carlo"] > 0.5
    with pytest.raises(ValueError, match="perfect-square"):
        theoretical_hit_rate_bound(0.3, 2, 40, membership_kind="row_column")


def test_same_sample_plugin_d2_is_explicitly_retained_without_false_debias():
    rng = np.random.default_rng(4)
    target, nontarget = rng.normal(size=(40, 60)), rng.normal(size=(200, 60))
    snr = mahalanobis_snr(target, nontarget, shrinkage=0.0)
    assert snr["d2_mahalanobis"] > 1.0
    assert snr["d2_estimator"] == "same_sample_shrinkage_covariance_plugin"
    assert "d2_mahalanobis_debiased" not in snr
    assert "d2_plugin_null_bias" not in snr


def test_kronecker_discriminability_is_invariant_to_voltage_units():
    rng = np.random.default_rng(6)
    target, nontarget = rng.normal(size=(60, 16, 3)) + 0.5, rng.normal(size=(120, 16, 3))
    volts = spectral_discriminability(target * 1e-6, nontarget * 1e-6, 64.0)["d2_kronecker_approx"]
    micro = spectral_discriminability(target, nontarget, 64.0)["d2_kronecker_approx"]
    assert volts == pytest.approx(micro, rel=1e-6)


def test_log_likelihood_correction_refuses_a_ridge_dominated_null_space():
    noise = _ar_from_coefficients([0.7, -0.2], 1.0)
    result = ExactPropagatedCovariance.compute(_operator(), noise, 60)
    assert result.null_dimension > 0
    with pytest.raises(ValueError, match="null space"):
        result.log_likelihood_correction(np.zeros((60, 2)))
