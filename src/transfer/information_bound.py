"""Calibration plug-in discriminability and declared Gaussian-model hit diagnostics.

For independent flashes with common known covariance Sigma and binary activation
A[i,q], let delta=mu1-mu0 and d²=delta.T @ inv(Sigma) @ delta. The log likelihood
of candidate q, up to a candidate-independent term, is

    L_q = sum_i A[i,q] * (delta.T @ inv(Sigma) @ (y_i-mu0) - d²/2).

Under true candidate t, L_q-L_t has mean -d²*h[t,q]/2 and variance d²*h[t,q],
where h[t,q] is the number of flashes on which the two activation columns differ.
Thus the pairwise error is Phi(-sqrt(d²*h[t,q])/2). Its union gives a lower
bound on optimal model hit, and its maximum gives an upper bound. The exact
model rate lies between them; finite Monte Carlo estimates it with a recorded
standard error and may fluctuate beyond those bounds. Identical columns use uniform
random tie resolution. Unequal activation counts are retained exactly.

The calibration helper estimates both means and a shrinkage covariance on the
same rows. Its raw plug-in d² is a fitted diagnostic, with same-sample optimism
and covariance-estimation error. Subtracting p*(1/n1+1/n0) is unbiased only with
the fixed true covariance, not this inverse fitted shrinkage covariance. The
module supplies no debiased population estimator and no Shannon MI identity.

The known-parameter model optimum is an upper limit for decoders in that model;
the Monte Carlo point estimate itself is finite-sample and is not an upper bound.
Calibration plug-in and factored spectral diagnostics have their own estimation
scope. This module has no production imports and preserves original provenance
when a saved report is read without refitting.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.covariance import LedoitWolf

from transfer.noise_model import (
    array_digest,
    finite_real_array,
    freeze_metadata,
    metadata_digest,
    require_sha256,
)

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_SCHEMA = "n2p3_information_bound/2"


# ---------------------------------------------------------------------------
# ERP discriminability
# ---------------------------------------------------------------------------

def mahalanobis_snr(
    target_epochs: np.ndarray,
    nontarget_epochs: np.ndarray,
    *,
    shrinkage: float | None = None,
) -> dict:
    """Estimate the per-flash Mahalanobis SNR d^2 = (mu1-mu0)^T Sigma^{-1} (mu1-mu0).

    Parameters
    ----------
    target_epochs : ndarray (n_target, T*C) or (n_target, T, C)
        Flattened or 3-D target-class epochs.
    nontarget_epochs : ndarray (n_nontarget, T*C) or (n_nontarget, T, C)
        Non-target-class epochs.
    shrinkage : float or None
        LedoitWolf shrinkage override. None = auto.

    Returns
    -------
    dict with:
        d2_mahalanobis   : float — the squared Mahalanobis distance
        d2_euclidean     : float — plain ||mu1-mu0||^2 (no whitening)
        erp_snr_db       : float — 10 * log10(d2_mahalanobis / n_features)
        n_target         : int
        n_nontarget      : int
        n_features       : int
        noise_shrinkage  : float — actual LW shrinkage used
        noise_cond       : float — condition number of noise covariance
        erp_mean_norm    : float — ||mu1 - mu0||_2
    """
    tgt = finite_real_array(target_epochs, name="target_epochs")
    non = finite_real_array(nontarget_epochs, name="nontarget_epochs")
    if tgt.ndim == non.ndim == 3 and tgt.shape[1:] != non.shape[1:]:
        raise ValueError("epochs must have the same T and C")
    if tgt.ndim == 3:
        tgt = tgt.reshape(len(tgt), -1)
    if non.ndim == 3:
        non = non.reshape(len(non), -1)
    if tgt.ndim != 2 or non.ndim != 2 or tgt.shape[1] != non.shape[1]:
        raise ValueError("epochs must be 2-D or 3-D arrays with the same feature dimension")
    if len(tgt) < 2 or len(non) < 2:
        raise ValueError("at least 2 epochs of each class are required")

    n_features = tgt.shape[1]
    if n_features < 1:
        raise ValueError("epochs need a nonempty feature dimension")
    mu1 = tgt.mean(axis=0)
    mu0 = non.mean(axis=0)
    delta = mu1 - mu0

    # Pool covariance from within-class residuals
    pooled = np.vstack([tgt - mu1, non - mu0])  # (n_tgt + n_non, n_features)
    if shrinkage is None:
        lw = LedoitWolf(assume_centered=True).fit(pooled)
        Sigma = lw.covariance_
        actual_shrinkage = float(lw.shrinkage_)
    else:
        if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must be in [0, 1]")
        n = len(pooled)
        empirical = pooled.T @ pooled / n
        # Shrink toward the average variance, not unit volts.  A unit target
        # would make this public helper depend on whether the caller uses V or
        # µV whenever shrinkage > 0.
        target_variance = float(np.trace(empirical) / n_features)
        Sigma = (1 - shrinkage) * empirical + shrinkage * target_variance * np.eye(n_features)
        actual_shrinkage = float(shrinkage)

    # Regularize for numerical stability
    min_eig = float(np.linalg.eigvalsh(Sigma).min())
    ridge_variance = 0.0
    if min_eig < np.finfo(float).tiny * float(np.trace(Sigma)):
        eps = max(float(np.trace(Sigma)) * 1e-10, np.finfo(float).tiny)
        Sigma = Sigma + eps * np.eye(n_features)
        ridge_variance = eps

    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"noise covariance is not positive definite: {exc}") from exc

    whitened_delta = solve_triangular(L, delta, lower=True)
    d2 = float(whitened_delta @ whitened_delta)
    d2_euclidean = float(delta @ delta)
    noise_cond = float(np.linalg.cond(Sigma))
    erp_snr_db = 10.0 * np.log10(max(d2 / n_features, np.finfo(float).tiny))

    return {
        "d2_mahalanobis": d2,
        "d2_estimator": "same_sample_shrinkage_covariance_plugin",
        "estimation_scope": (
            "Means and inverse shrinkage covariance use the same calibration rows; "
            "the plug-in value includes same-sample optimism and covariance estimation error."
        ),
        "d2_euclidean": d2_euclidean,
        "erp_snr_db": erp_snr_db,
        "n_target": len(tgt),
        "n_nontarget": len(non),
        "n_features": n_features,
        "noise_shrinkage": actual_shrinkage,
        "noise_cond": noise_cond,
        "noise_ridge_variance": ridge_variance,
        "covariance_interpretation": "fitted_positive_definite_diagnostic",
        "erp_mean_norm": float(np.linalg.norm(delta)),
    }


# ---------------------------------------------------------------------------
# Declared activation geometry and Gaussian-model hit probability
# ---------------------------------------------------------------------------


def _activation_matrix(n_candidates: int, n_flashes: int, *, membership_kind: str):
    """Synthetic canonical schedule; incomplete cycles retain unequal counts."""
    Q, F = n_candidates, n_flashes
    if membership_kind == "single_item":
        return np.eye(Q, dtype=float)[np.arange(F) % Q], None
    if membership_kind != "row_column":
        raise ValueError("membership_kind must be 'row_column' or 'single_item'")
    root = int(np.sqrt(Q))
    if root * root != Q or root < 2:
        raise ValueError("a row/column layout needs a perfect-square candidate count of at least 4")
    templates = np.zeros((2 * root, Q), dtype=float)
    for line in range(root):
        templates[line, line * root:(line + 1) * root] = 1
        templates[root + line, line::root] = 1
    counts = np.full(2 * root, F // (2 * root))
    counts[:F % (2 * root)] += 1
    return np.repeat(templates, counts, axis=0), 2 * root


def _declared_activation(Q, F, membership_kind, activation_matrix):
    canonical, lines = _activation_matrix(Q, F, membership_kind=membership_kind)
    if activation_matrix is None:
        return canonical, lines, "synthetic_canonical_schedule"
    raw = np.asarray(activation_matrix)
    if raw.dtype.kind not in "biuf":
        raise ValueError("activation_matrix must contain binary real membership")
    A = np.asarray(raw, dtype=float)
    if A.shape != (F, Q) or not np.all((A == 0) | (A == 1)):
        raise ValueError("activation_matrix must be binary with shape (n_flashes, n_candidates)")
    if membership_kind == "single_item":
        if not np.all(A.sum(axis=1) == 1):
            raise ValueError("single_item must activate exactly one candidate per flash")
    else:
        root = int(np.sqrt(Q))
        templates, _ = _activation_matrix(Q, 2 * root, membership_kind=membership_kind)
        if not np.all(np.any(np.all(A[:, None, :] == templates[None, :, :], axis=2), axis=1)):
            raise ValueError("row_column flashes must be complete declared rows or columns")
    return A, lines, "caller_supplied_schedule"


def theoretical_hit_rate_bound(
    d2_per_flash: float,
    n_candidates: int,
    n_flashes_per_selection: int,
    *,
    membership_kind: str,
    activation_matrix: np.ndarray | None = None,
    mc_trials: int = 4000,
    mc_seed: int = 20261005,
) -> dict:
    """Known equal-covariance, independent-flash, uniform-prior Gaussian diagnostic.

    Supply actual membership rows to bind an observed schedule; otherwise the
    declared membership kind produces a synthetic canonical schedule. Bounds
    use all candidate pairs, including indistinguishable activation columns.
    Model hits use uniform random tie resolution, independently of production
    tie/abstention policy. MC trials share draws across true candidates; its
    reported standard error therefore uses trial-level averages, not Q*N iid
    observations. Required d² values invert this finite seeded MC curve.
    """
    if not np.isfinite(d2_per_flash) or d2_per_flash < 0:
        raise ValueError("d2_per_flash must be finite and nonneg")
    if type(n_candidates) is not int or n_candidates < 2:
        raise ValueError("n_candidates must be integer >= 2")
    if type(n_flashes_per_selection) is not int or n_flashes_per_selection < 1:
        raise ValueError("n_flashes_per_selection must be positive integer")
    if type(mc_trials) is not int or mc_trials < 2:
        raise ValueError("mc_trials must be an integer >= 2")
    if type(mc_seed) is not int or mc_seed < 0:
        raise ValueError("mc_seed must be a nonnegative integer")
    Q, F = n_candidates, n_flashes_per_selection
    d2 = float(d2_per_flash)
    A, lines, activation_scope = _declared_activation(Q, F, membership_kind, activation_matrix)
    counts = A.sum(axis=0).astype(int)
    pair_diff = np.abs(A[:, :, None] - A[:, None, :]).sum(axis=0)
    identical = pair_diff == 0
    indistinguishable_counts = identical.sum(axis=1)

    # Under true t, Lq-Lt = sqrt(d²)*(Zq-Zt) - d²*h[t,q]/2.
    # For h>0, winning is equivalent to sqrt(d²) > 2*(Zq-Zt)/h.
    # One normal draw and one threshold per trial/true candidate exactly define
    # the monotone finite MC curve, without regenerating noise during bisection.
    rng = np.random.default_rng(mc_seed)
    Z = rng.standard_normal((mc_trials, F)) @ A
    thresholds = np.empty((mc_trials, Q))
    for true in range(Q):
        competitors = pair_diff[true] > 0
        if np.any(competitors):
            thresholds[:, true] = np.max(
                2 * (Z[:, competitors] - Z[:, [true]]) / pair_diff[true, competitors], axis=1
            )
        else:
            thresholds[:, true] = -np.inf

    def hit_samples(value):
        if value == 0:
            return np.full((mc_trials, Q), 1 / Q)
        return (np.sqrt(value) > thresholds) / indistinguishable_counts[None, :]

    samples = hit_samples(d2)
    per_true_mc = samples.mean(axis=0)
    trial_means = samples.mean(axis=1)
    p_hit_mc = float(trial_means.mean()) if d2 > 0 else 1 / Q
    standard_error = float(trial_means.std(ddof=1) / np.sqrt(mc_trials)) if d2 > 0 else 0.

    from scipy.special import ndtr

    pair_error = ndtr(-0.5 * np.sqrt(d2 * pair_diff))
    np.fill_diagonal(pair_error, 0)
    lower_per_true = np.maximum(0, 1 - pair_error.sum(axis=1))
    upper_per_true = 1 - pair_error.max(axis=1)
    # A uniform-prior Bayes decoder can always achieve chance; the exact
    # identical-column ceiling also improves the pairwise upper bound.
    ceiling_per_true = 1 / indistinguishable_counts
    p_lower = max(1 / Q, float(lower_per_true.mean()))
    p_upper = float(np.minimum(upper_per_true, ceiling_per_true).mean())
    required_d2 = {}
    for target in (.75, .85, .90, .95):
        lo, hi = 0., 50.
        key = f"p_hit_{int(target * 100)}"
        if float(hit_samples(hi).mean()) < target:
            required_d2[key] = None  # finite search bracket, not an infinity in JSON
            continue
        for _ in range(32):
            mid = (lo + hi) / 2
            if float(hit_samples(mid).mean()) < target:
                lo = mid
            else:
                hi = mid
        required_d2[key] = hi

    balanced = bool(np.all(counts == counts[0]))
    if membership_kind == "row_column":
        templates, _ = _activation_matrix(Q, lines, membership_kind=membership_kind)
        line_counts = np.all(A[:, None, :] == templates[None, :, :], axis=2).sum(axis=0)
    else:
        line_counts = counts
    return {
        "p_hit_lower_bound": p_lower,
        "p_hit_upper_bound": p_upper,
        "p_hit_monte_carlo": p_hit_mc,
        "p_hit_model_estimate": p_hit_mc,
        "p_hit_per_candidate_monte_carlo": per_true_mc.tolist(),
        "p_hit_per_candidate_pairwise_lower_bound": lower_per_true.tolist(),
        "p_hit_per_candidate_upper_bound": np.minimum(upper_per_true, ceiling_per_true).tolist(),
        "d2_per_flash": d2,
        "d2_effective_per_selection": float(counts.mean() * d2),
        "d2_effective_per_candidate": (counts * d2).tolist(),
        "n_candidates": Q,
        "n_flashes": F,
        "membership_kind": membership_kind,
        "grid_lines": lines,
        "activation_scope": activation_scope,
        "activation_matrix": A.astype(int).tolist(),
        "activation_sha256": array_digest(A),
        "activation_counts_per_candidate": counts.tolist(),
        "activations_per_candidate": float(counts.mean()),
        "p_activation": float(counts.mean() / F),
        "p_activation_per_candidate": (counts / F).tolist(),
        "activation_counts_balanced": balanced,
        "line_flash_counts": line_counts.tolist(),
        "schedule_cycles_balanced": bool(np.all(line_counts == line_counts[0])),
        "pairwise_differing_flashes": pair_diff.astype(int).tolist(),
        "indistinguishable_candidates_per_candidate": indistinguishable_counts.tolist(),
        "identifiable_hit_ceiling": float(ceiling_per_true.mean()),
        "min_differing_flashes": float(pair_diff[np.triu_indices(Q, 1)].min()),
        "max_differing_flashes": float(pair_diff.max()),
        "required_d2_for_targets": required_d2,
        "required_d2_search_max": 50.,
        "required_d2_interpretation": "seeded_finite_MC_inversion; null means outside search bracket",
        "monte_carlo": {
            "trials": mc_trials,
            "seed": mc_seed,
            "bit_generator": "PCG64",
            "standard_error": standard_error,
            "standard_error_scope": "trial_average_across_correlated_true_candidate_draws",
            "per_candidate_standard_error": (samples.std(axis=0, ddof=1) / np.sqrt(mc_trials)).tolist(),
        },
        "model": {
            "candidate_prior": "uniform",
            "noise": "equal_covariance_independent_flashes_known_parameters",
            "decoder": "maximum_likelihood_uniform_random_ties",
            "d2_role": "supplied_parameter_for_declared_model",
        },
        "interpretation": (
            f"{'Equal' if balanced else 'Unequal'} activation counts for {membership_kind}: "
            f"{counts.min()}..{counts.max()} of {F} flashes per candidate. "
            f"Gaussian-model optimal hit MC={p_hit_mc:.3f}, "
            f"pairwise lower={p_lower:.3f}, model upper={p_upper:.3f}; "
            f"{mc_trials} trials, seed={mc_seed}."
        ),
    }


# ---------------------------------------------------------------------------
# Spectral discriminability decomposition
# ---------------------------------------------------------------------------

def spectral_discriminability(
    target_epochs: np.ndarray,
    nontarget_epochs: np.ndarray,
    sfreq: float,
    *,
    n_components: int = 8,
) -> dict:
    """SVD of a fitted Kronecker-whitened mean contrast.

    Fractions partition the factored plug-in d² across singular components.
    Frequencies describe each unwhitened temporal pattern, rather than an
    additive frequency-band information decomposition.
    """
    tgt = finite_real_array(target_epochs, name="target_epochs")
    non = finite_real_array(nontarget_epochs, name="nontarget_epochs")
    if tgt.ndim != 3 or non.ndim != 3:
        raise ValueError("epochs must be 3-D: (n_epochs, T, C)")
    T, C = tgt.shape[1], tgt.shape[2]
    if non.shape[1:] != (T, C):
        raise ValueError("target and nontarget epochs must have the same T and C")
    if len(tgt) < 2 or len(non) < 2 or T < 1 or C < 1:
        raise ValueError("at least two epochs of each class and nonempty T and C are required")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be positive")
    if type(n_components) is not int or n_components < 1:
        raise ValueError("n_components must be a positive integer")

    mu1 = tgt.mean(axis=0)  # (T, C)
    mu0 = non.mean(axis=0)  # (T, C)
    delta = mu1 - mu0        # (T, C)

    # Noise covariance in (T, C) space — use pooled within-class residuals
    pooled = np.vstack([
        (tgt - mu1).reshape(len(tgt), T * C),
        (non - mu0).reshape(len(non), T * C),
    ])
    # A full (T*C)×(T*C) covariance is not formed: the factored Kronecker
    # path below whitens time and space separately, which is the only
    # tractable route for realistic T*C and is the declared approximation.

    # For large T*C, use a factored approach:
    # whiten separately in time (pooled over channels) and space
    # This is an approximation valid when the time-space covariance is
    # approximately Kronecker: Sigma ≈ K_time ⊗ Sigma_space

    # Time covariance (pooled over channels)
    pooled_t = pooled.reshape(len(pooled), T, C).transpose(0, 2, 1).reshape(-1, T)
    lw_t = LedoitWolf(assume_centered=True).fit(pooled_t)
    # Both factors are pooled covariances, so each carries the noise scale;
    # unit-trace-per-sample time keeps the product equal to Sigma, not Sigma*scale.
    K_time = lw_t.covariance_ / (np.trace(lw_t.covariance_) / T)  # (T, T)

    # Space covariance (pooled over time)
    pooled_s = pooled.reshape(len(pooled), T, C).reshape(-1, C)
    lw_s = LedoitWolf(assume_centered=True).fit(pooled_s)
    Sigma_space = lw_s.covariance_  # (C, C)

    # Whiten delta in time and space separately
    try:
        L_t = np.linalg.cholesky(K_time)
        L_s = np.linalg.cholesky(Sigma_space)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"noise covariance factorization failed: {exc}") from exc

    # Whitened delta: L_t^{-1} delta L_s^{-T} shape (T, C)
    delta_wt = solve_triangular(L_t, delta, lower=True)
    delta_ws = solve_triangular(L_s, delta_wt.T, lower=True).T

    # SVD of whitened delta matrix: each singular value gives discriminability
    U, s, Vt = np.linalg.svd(delta_ws, full_matrices=False)
    # s^2 are the eigenvalues of the whitened outer product — summing gives d^2_approx
    d2_approx = float(np.sum(s**2))
    keep = min(n_components, len(s))
    fractions = (s[:keep]**2 / max(d2_approx, np.finfo(float).tiny)).tolist()

    # Temporal profile: project each component back to time domain
    # U columns in whitened time coords; unwhiten: L_t U
    L_t_U = L_t @ U[:, :keep]  # (T, keep) — temporal patterns in signal units
    temporal_profiles = L_t_U.tolist()

    # Frequency content of each temporal component
    freqs = np.fft.rfftfreq(T, d=1.0 / sfreq)
    spectral_profiles = []
    for k in range(keep):
        psd = np.abs(np.fft.rfft(L_t_U[:, k]))**2
        dominant_freq = float(freqs[np.argmax(psd)])
        centroid_freq = float(np.sum(freqs * psd) / max(np.sum(psd), np.finfo(float).tiny))
        spectral_profiles.append({
            "component": k,
            "singular_value": float(s[k]),
            "d2_fraction": fractions[k],
            "dominant_freq_hz": dominant_freq,
            "centroid_freq_hz": centroid_freq,
        })

    return {
        "d2_kronecker_approx": d2_approx,
        "n_components_shown": keep,
        "components": spectral_profiles,
        "temporal_profiles": temporal_profiles,
        "spatial_patterns": (L_s @ Vt[:keep].T).tolist(),  # (C, keep)
        "time_noise_shrinkage": float(lw_t.shrinkage_),
        "space_noise_shrinkage": float(lw_s.shrinkage_),
        "estimation_scope": "same_sample_factored_shrinkage_covariance_plugin",
        "interpretation": (
            "Singular components partition fitted Kronecker-whitened contrast energy. "
            "Temporal spectra describe patterns; selecting preprocessing from these "
            "calibration diagnostics requires a separate evaluation."
        ),
    }


# ---------------------------------------------------------------------------
# Full bound report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InformationBoundReport:
    """Immutable schema-2 calibration diagnostic, with all output content bound.

    Save/load restores original numerical diagnostics and source identities,
    without estimating covariance or drawing fresh Monte Carlo samples.
    """
    d2_per_flash: float
    n_candidates: int
    n_flashes: int
    membership_kind: str
    empirical_hit_rate: float | None
    bound: dict
    snr_analysis: dict
    spectral_analysis: dict | None
    provenance: dict
    _integrity_sha256: str

    def __post_init__(self):
        for field in ("bound", "snr_analysis", "spectral_analysis", "provenance"):
            object.__setattr__(self, field, freeze_metadata(getattr(self, field)))
        self._assert_integrity()
        if self.provenance.get("schema") != _SCHEMA:
            raise ValueError("information report schema must be n2p3_information_bound/2")
        for key in ("implementation_sha256", "metadata_contract_sha256", "target_sha256",
                    "nontarget_sha256", "activation_sha256"):
            require_sha256(self.provenance[key], key)
        if (self.bound["d2_per_flash"] != self.d2_per_flash
                or self.snr_analysis["d2_mahalanobis"] != self.d2_per_flash
                or self.snr_analysis["d2_estimator"] != "same_sample_shrinkage_covariance_plugin"):
            raise ValueError("information report plug-in d2 relation mismatch")
        for field, key in ((self.n_candidates, "n_candidates"), (self.n_flashes, "n_flashes"),
                           (self.membership_kind, "membership_kind")):
            if self.bound[key] != field or self.provenance[key] != field:
                raise ValueError(f"information report {key} relation mismatch")
        if (self.empirical_hit_rate is not None
                and (not np.isfinite(self.empirical_hit_rate)
                     or not 0 <= self.empirical_hit_rate <= 1)):
            raise ValueError("empirical_hit_rate must be in [0, 1]")
        A, _, _ = _declared_activation(
            self.n_candidates, self.n_flashes, self.membership_kind, self.bound["activation_matrix"]
        )
        if (array_digest(A) != self.bound["activation_sha256"]
                or self.bound["activation_sha256"] != self.provenance["activation_sha256"]
                or tuple(A.sum(axis=0).astype(int)) != self.bound["activation_counts_per_candidate"]):
            raise ValueError("information report activation relation mismatch")

    def _payload(self):
        return {
            "d2_per_flash": self.d2_per_flash,
            "n_candidates": self.n_candidates,
            "n_flashes": self.n_flashes,
            "membership_kind": self.membership_kind,
            "empirical_hit_rate": self.empirical_hit_rate,
            "bound": self.bound,
            "snr_analysis": self.snr_analysis,
            "spectral_analysis": self.spectral_analysis,
            "provenance": self.provenance,
        }

    def _assert_integrity(self):
        require_sha256(self._integrity_sha256, "integrity_sha256")
        if metadata_digest(self._payload()) != self._integrity_sha256:
            raise ValueError("information report content integrity mismatch")

    @classmethod
    def compute(
        cls,
        target_epochs: np.ndarray,
        nontarget_epochs: np.ndarray,
        *,
        n_candidates: int,
        n_flashes_per_selection: int,
        membership_kind: str,
        sfreq: float,
        empirical_hit_rate: float | None = None,
        compute_spectral: bool = True,
        n_spectral_components: int = 8,
        participant_id: str = "unknown",
        activation_matrix: np.ndarray | None = None,
        mc_trials: int = 4000,
        mc_seed: int = 20261005,
    ) -> InformationBoundReport:
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError("sfreq must be finite and positive")
        if not isinstance(participant_id, str) or not participant_id.strip():
            raise ValueError("participant_id must be nonempty text")
        if empirical_hit_rate is not None:
            if not np.isfinite(empirical_hit_rate) or not 0 <= empirical_hit_rate <= 1:
                raise ValueError("empirical_hit_rate must be in [0, 1]")
            empirical_hit_rate = float(empirical_hit_rate)
        snr = mahalanobis_snr(target_epochs, nontarget_epochs)
        d2 = snr["d2_mahalanobis"]
        bound = theoretical_hit_rate_bound(
            d2, n_candidates, n_flashes_per_selection, membership_kind=membership_kind,
            activation_matrix=activation_matrix, mc_trials=mc_trials, mc_seed=mc_seed,
        )
        spec = None
        tgt = finite_real_array(target_epochs, name="target_epochs")
        non = finite_real_array(nontarget_epochs, name="nontarget_epochs")
        if compute_spectral and tgt.ndim == 3:
            spec = spectral_discriminability(tgt, non, sfreq, n_components=n_spectral_components)
        if empirical_hit_rate is not None:
            model_hit = bound["p_hit_model_estimate"]
            chance = 1 / n_candidates
            room = model_hit - chance
            bound["empirical_comparison"] = {
                "empirical_hit": empirical_hit_rate,
                "gaussian_model_hit_mc": model_hit,
                "chance_hit": chance,
                "model_minus_empirical_hit": model_hit - empirical_hit_rate,
                "above_chance_ratio": ((empirical_hit_rate - chance) / room if room > 0 else None),
                "interpretation": "descriptive comparison with fitted model; ratio is not efficiency",
            }
        prov = {
            "schema": _SCHEMA,
            "participant_id": participant_id,
            "n_candidates": n_candidates,
            "n_flashes": n_flashes_per_selection,
            "membership_kind": membership_kind,
            "activation_scope": bound["activation_scope"],
            "activation_sha256": bound["activation_sha256"],
            "sfreq": float(sfreq),
            "compute_spectral": compute_spectral,
            "n_spectral_components": n_spectral_components,
            "n_target_epochs": int(len(tgt)),
            "n_nontarget_epochs": int(len(non)),
            "target_sha256": array_digest(tgt),
            "nontarget_sha256": array_digest(non),
            "implementation_sha256": _IMPLEMENTATION_SHA256,
            "metadata_contract_sha256": hashlib.sha256(
                Path(__file__).with_name("noise_model.py").read_bytes()
            ).hexdigest(),
            "d2_estimator": snr["d2_estimator"],
            "evidence_scope": "calibration_plugin_gaussian_model_diagnostic",
        }
        payload = {
            "d2_per_flash": d2, "n_candidates": n_candidates,
            "n_flashes": n_flashes_per_selection, "membership_kind": membership_kind,
            "empirical_hit_rate": empirical_hit_rate, "bound": bound,
            "snr_analysis": snr, "spectral_analysis": spec, "provenance": prov,
        }
        return cls(**payload, _integrity_sha256=metadata_digest(payload))

    def to_dict(self) -> dict:
        """Detached JSON data; mutating it cannot change this report."""
        self._assert_integrity()
        return json.loads(json.dumps(
            {**self._payload(), "integrity_sha256": self._integrity_sha256}, allow_nan=False
        ))

    @classmethod
    def from_dict(cls, record: dict) -> InformationBoundReport:
        """Read schema-2 output without refitting or relabelling old producer identity."""
        if not isinstance(record, dict) or record.get("provenance", {}).get("schema") != _SCHEMA:
            raise ValueError("information report schema must be n2p3_information_bound/2")
        payload = dict(record)
        integrity = payload.pop("integrity_sha256", None)
        return cls(**payload, _integrity_sha256=integrity)

    def save(self, path: str | Path) -> None:
        """Write one complete JSON report, refusing to replace an existing artifact."""
        record = self.to_dict()
        with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> InformationBoundReport:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def replay_model(self) -> dict:
        """Recompute saved Gaussian diagnostics only; no calibration covariance fit.

        Exact seeded replay uses the original producer implementation. A report
        from different source bytes remains readable through load/to_dict.
        """
        self._assert_integrity()
        if self.provenance["implementation_sha256"] != _IMPLEMENTATION_SHA256:
            raise ValueError("model replay requires the original implementation bytes")
        actual = (self.bound["activation_matrix"]
                  if self.bound["activation_scope"] == "caller_supplied_schedule" else None)
        replayed = theoretical_hit_rate_bound(
            self.d2_per_flash, self.n_candidates, self.n_flashes,
            membership_kind=self.membership_kind, activation_matrix=actual,
            mc_trials=self.bound["monte_carlo"]["trials"],
            mc_seed=self.bound["monte_carlo"]["seed"],
        )
        saved = dict(self.bound)
        saved.pop("empirical_comparison", None)
        if metadata_digest(replayed) != metadata_digest(saved):
            raise ValueError("saved Gaussian-model diagnostics do not replay")
        return replayed

    def summary(self) -> dict:
        """Detached reporting view, retaining the model and estimator scope."""
        self._assert_integrity()
        s = {
            "schema": _SCHEMA,
            "d2_per_flash": self.d2_per_flash,
            "d2_estimator": self.snr_analysis["d2_estimator"],
            "erp_snr_db": self.snr_analysis["erp_snr_db"],
            "n_candidates": self.n_candidates,
            "n_flashes": self.n_flashes,
            "membership_kind": self.membership_kind,
            "p_hit_lower_bound": self.bound["p_hit_lower_bound"],
            "p_hit_upper_bound": self.bound["p_hit_upper_bound"],
            "p_hit_monte_carlo": self.bound["p_hit_monte_carlo"],
            "d2_effective_per_selection": self.bound["d2_effective_per_selection"],
            "activation_counts_per_candidate": self.bound["activation_counts_per_candidate"],
            "activation_scope": self.bound["activation_scope"],
            "monte_carlo": self.bound["monte_carlo"],
        }
        if "empirical_comparison" in self.bound:
            s.update(self.bound["empirical_comparison"])
        if self.spectral_analysis:
            top = self.spectral_analysis["components"][0]
            s["top_component_d2_fraction"] = top["d2_fraction"]
            s["top_component_freq_hz"] = top["dominant_freq_hz"]
        return json.loads(json.dumps(s, allow_nan=False))
