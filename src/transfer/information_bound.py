"""Gaussian-model hit estimates and conservative pairwise error bounds.

First-principles derivation
----------------------------
From Shannon's mutual information, the maximum number of bits that can be
recovered from K independent observations of a Gaussian channel is:

    I(q; Y_1..K) = (K/2) * log2(1 + d^2 / K_noise)

where d^2 = (mu_1 - mu_0)^T K_noise^{-1} (mu_1 - mu_0) is the squared
Mahalanobis distance between target and non-target ERP means, and K_noise is
the noise covariance.

For the P300 row/column paradigm with Q candidates and R flashes per
selection, the mutual information available per selection is:

    I_sel = sum_{flash i} I(q; y_i | q activates or not in flash i)

Under the linear Gaussian signal model:
    y_i | target_i=1 ~ N(mu_1, K_noise)
    y_i | target_i=0 ~ N(mu_0, K_noise)

    I(q; y_i) = (1/2) * [E_p(y|1)[log p(y|1)/p(y|0)] * P(target=1)
                        + E_p(y|0)[log p(y|1)/p(y|0)] * P(target=0)]

For equal priors P(1) = P(0) = 0.5 this simplifies to half the KL divergence:
    I(q; y_i) = (1/8) * d_i^2   (in nats, for scalar case)

where d_i^2 is the Mahalanobis distance for flash i.

For the multi-candidate selection with Q candidates and F flashes per
selection, each flash activates exactly 1/Q of all flashes as targets for
the true candidate. The expected Mahalanobis information per flash is:

    d_bar^2 = d^2 * (1/Q)

where d^2 = || mu_1 - mu_0 ||^2_{K_noise^{-1}} is the per-flash SNR.

    A model-based Monte Carlo estimate for Q-choice identification from F
    flashes (continuous Gaussian, optimal Bayesian decoder, known parameters)
    can be computed for a declared activation matrix. It is not a universal
    information-theoretic upper bound: it inherits the equal-covariance,
    independent-flash, known-parameter model and the estimated d².

    P_hit_theoretical >= 1 - (Q-1) * exp(-F * d_bar^2 / 8)

    The pairwise union calculation is a conservative lower bound on the
    model hit probability. The Monte Carlo value is an estimate of the same
    declared model, not a bound on arbitrary decoders or future data.

For Q=36, F=120, this gives us a curve: as d^2 increases, P_hit approaches 1.
Comparing the theoretical bound with the empirical 0.771 tells us how far
the current method is from the information-theoretic optimum.

This module computes:
1. The empirical d^2 (signal detectability) from calibration data
    2. The declared Gaussian-model hit estimate and pairwise lower bound
    3. The gap between current performance and that model estimate
4. Spectral decomposition to identify which frequency/spatial components
   carry most of the ERP discriminability

Scope
------
Research diagnostic, no production import.
Results bind to declared calibration data digests.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.covariance import LedoitWolf

from transfer.noise_model import array_digest, metadata_digest

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_SCHEMA = "n2p3_information_bound/1"


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
    tgt = np.asarray(target_epochs, dtype=float)
    non = np.asarray(nontarget_epochs, dtype=float)
    if tgt.ndim == 3:
        tgt = tgt.reshape(len(tgt), -1)
    if non.ndim == 3:
        non = non.reshape(len(non), -1)
    if tgt.ndim != 2 or non.ndim != 2 or tgt.shape[1] != non.shape[1]:
        raise ValueError("epochs must be 2-D or 3-D arrays with the same feature dimension")
    if len(tgt) < 2 or len(non) < 2:
        raise ValueError("at least 2 epochs of each class are required")

    n_features = tgt.shape[1]
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
    if min_eig < np.finfo(float).tiny * float(np.trace(Sigma)):
        eps = max(float(np.trace(Sigma)) * 1e-10, np.finfo(float).tiny)
        Sigma = Sigma + eps * np.eye(n_features)

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
        "d2_euclidean": d2_euclidean,
        "erp_snr_db": erp_snr_db,
        "n_target": len(tgt),
        "n_nontarget": len(non),
        "n_features": n_features,
        "noise_shrinkage": actual_shrinkage,
        "noise_cond": noise_cond,
        "erp_mean_norm": float(np.linalg.norm(delta)),
    }


# ---------------------------------------------------------------------------
# Theoretical upper bound on hit rate
# ---------------------------------------------------------------------------

def _activation_matrix(n_candidates: int, n_flashes: int, n_lines: int | None = None):
    """Balanced row/column activation matrix A of shape (n_flashes, n_candidates).

    For a Q = L×L row/column paradigm, there are 2L lines (L rows + L columns)
    and n_flashes total flashes, evenly split as n_flashes/(2L) per line.
    Candidate q is activated whenever its row or its column flashes, so
    |activations of any candidate| = n_flashes / L = n_flashes / sqrt(Q).

    If n_candidates is not a perfect square, fall back to an injective
    assignment where each flash activates exactly round(sqrt(Q)) candidates
    in a cyclic pattern; this keeps the total activation count well-defined
    without pretending a physical grid exists.
    """
    Q = n_candidates
    F = n_flashes
    root = int(round(np.sqrt(Q)))
    if root * root == Q and (n_lines is None or n_lines == 2 * root):
        L = root
        lines = 2 * L
        per_line = F // lines
        remainder = F - per_line * lines
        A = np.zeros((F, Q), dtype=float)
        flash = 0
        for line in range(lines):
            count = per_line + (1 if line < remainder else 0)
            if line < L:
                members = np.arange(line * L, (line + 1) * L)  # a row
            else:
                col = line - L
                members = np.arange(col, Q, L)                 # a column
            A[flash:flash + count, members] = 1.0
            flash += count
        return A, L
    # Non-square fallback: cyclic groups of size round(sqrt(Q)).
    group = max(2, root)
    A = np.zeros((F, Q), dtype=float)
    for i in range(F):
        start = (i * group) % Q
        members = [(start + k) % Q for k in range(group)]
        A[i, members] = 1.0
    return A, group


def theoretical_hit_rate_bound(
    d2_per_flash: float,
    n_candidates: int,
    n_flashes_per_selection: int,
) -> dict:
    """Gaussian-model hit estimate and pairwise lower bound.

    Model
    -----
    Each flash i yields a per-candidate unit log-evidence contribution
    t_i ~ N(0, d2) under the null, and the true candidate q* additionally
    receives a shift of +d2/2 on the flashes that activate it (each of its
    L = sqrt(Q) lines). Writing A[i, q] ∈ {0, 1} for "flash i activates q",
    the log-evidence difference of competitor q against q* is

        S_q = sum_i (A[i,q] - A[i,q*]) t_i
              - (d2/2) * sum_i |A[i,q] - A[i,q*]|

    and the optimal decoder is correct iff S_q < 0 for all q ≠ q*.

    Because a competitor sharing a row or a column with q* differs on fewer
    flashes than one sharing neither, the competitors split into two groups
    with different Mahalanobis separations. Both groups are treated exactly by
    the vectorised Monte Carlo below; no single-d' approximation is imposed.

    Parameters
    ----------
    d2_per_flash : float
        Squared Mahalanobis distance per flash (from mahalanobis_snr).
    n_candidates : int
        Number of candidates (Q = 36 for the BNCI 6×6 row/column grid).
    n_flashes_per_selection : int
        Total flashes per selection (120 for one BNCI C800 selection).

    Returns
    -------
    dict with the Monte Carlo hit estimate, a conservative union lower bound,
    and the per-flash d2 required to reach reference hit rates by bisection on
    the Monte Carlo estimator.
    """
    if not np.isfinite(d2_per_flash) or d2_per_flash < 0:
        raise ValueError("d2_per_flash must be finite and nonneg")
    if type(n_candidates) is not int or n_candidates < 2:
        raise ValueError("n_candidates must be integer >= 2")
    if type(n_flashes_per_selection) is not int or n_flashes_per_selection < 1:
        raise ValueError("n_flashes_per_selection must be positive integer")

    Q = n_candidates
    F = n_flashes_per_selection
    A, lines = _activation_matrix(Q, F)
    base_activation = A[:, 0]
    activations_per_candidate = float(base_activation.sum())
    p_act = activations_per_candidate / F

    # Pairwise differing-flash counts for every possible true candidate.  For
    # F divisible by the number of lines this reduces to the usual two groups;
    # for an uneven synthetic grid it avoids silently privileging candidate 0.
    pair_diff = np.abs(A[:, :, None] - A[:, None, :]).sum(axis=0).astype(float)
    base_diff_counts = pair_diff[0, 1:]
    unique_counts, multiplicity = np.unique(base_diff_counts, return_counts=True)
    n_shared = int(multiplicity[0]) if len(multiplicity) else 0
    n_disjoint = int(Q - 1 - n_shared)
    min_diff = float(pair_diff[np.triu_indices(Q, 1)].min()) if Q > 1 else 0.0
    max_diff = float(pair_diff[np.triu_indices(Q, 1)].max()) if Q > 1 else 0.0

    def _hit_monte_carlo(d2, n_trials=4000, seed=20261005):
        if d2 <= 0:
            return 1.0 / Q
        rng = np.random.default_rng(seed)
        # Per-flash discriminant noise shared across candidates.
        noise = rng.normal(0.0, np.sqrt(d2), size=(n_trials, F))
        T = noise @ A                       # (n_trials, Q)
        hits = []
        for true_candidate in range(Q):
            competitors = np.arange(Q) != true_candidate
            S = (T[:, competitors] - T[:, [true_candidate]])
            S -= 0.5 * d2 * pair_diff[true_candidate, competitors][None, :]
            hits.append(np.all(S < 0.0, axis=1))
        return float(np.mean(np.stack(hits, axis=1)))

    # Union bound. For competitor q, S_q ~ N(-0.5 d2 n_diff, d2 n_diff), so
    # P(q beats q*) = Phi(-0.5 sqrt(d2 n_diff)) and the error probability is
    # bounded by the sum over all competitors. Computing the sum directly over
    # every diff_count avoids any two-group approximation.
    from scipy.special import ndtr
    dprime_sq = 0.25 * d2_per_flash * pair_diff
    per_true_fail_bound = np.sum(ndtr(-np.sqrt(np.maximum(dprime_sq, 0.0))), axis=1)
    p_hit_lower = float(np.mean(np.maximum(0.0, 1.0 - per_true_fail_bound)))

    p_hit_mc = _hit_monte_carlo(d2_per_flash)

    # Required d2 for reference hit rates: bisection on the MC estimator.
    target_hit_rates = [0.75, 0.85, 0.90, 0.95]
    required_d2 = {}
    for p_target in target_hit_rates:
        lo, hi = 0.0, 50.0
        if _hit_monte_carlo(hi) < p_target:
            required_d2[f"p_hit_{int(p_target * 100)}"] = float("inf")
            continue
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if _hit_monte_carlo(mid) < p_target:
                lo = mid
            else:
                hi = mid
        required_d2[f"p_hit_{int(p_target * 100)}"] = float(hi)

    return {
        "p_hit_lower_bound": p_hit_lower,
        "p_hit_monte_carlo": p_hit_mc,
        "p_hit_model_estimate": p_hit_mc,
        "d2_per_flash": float(d2_per_flash),
        "d2_effective_per_selection": float(activations_per_candidate * d2_per_flash),
        "n_candidates": Q,
        "n_flashes": F,
        "grid_lines": lines,
        "activations_per_candidate": activations_per_candidate,
        "p_activation": float(p_act),
        "n_competitors_sharing_line": n_shared,
        "n_competitors_disjoint": n_disjoint,
        "min_differing_flashes": min_diff,
        "max_differing_flashes": max_diff,
        "required_d2_for_targets": required_d2,
        "interpretation": (
            f"Balanced {lines}-line row/column grid: each candidate activates on "
            f"{activations_per_candidate:.0f}/{F} flashes (p={p_act:.3f}). "
            f"With d2_per_flash={d2_per_flash:.4f}, the declared Gaussian-model "
            f"hit estimate is {p_hit_mc:.3f} (MC); the pairwise lower bound is "
            f"{p_hit_lower:.3f}."
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
    """Decompose d^2 into temporal and spatial components.

    Returns the fraction of total Mahalanobis SNR carried by each
    frequency band and each spatial direction, using the whitened
    SVD of the ERP mean difference.

    This identifies WHERE in time-frequency space the P300 information lives,
    which informs which AR order or filter design would preserve most of it.
    """
    tgt = np.asarray(target_epochs, dtype=float)
    non = np.asarray(nontarget_epochs, dtype=float)
    if tgt.ndim != 3 or non.ndim != 3:
        raise ValueError("epochs must be 3-D: (n_epochs, T, C)")
    T, C = tgt.shape[1], tgt.shape[2]
    if non.shape[1:] != (T, C):
        raise ValueError("target and nontarget epochs must have the same T and C")
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
    K_time = lw_t.covariance_  # (T, T)

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
        "interpretation": (
            "Components are ranked by discriminability fraction. "
            "AR whitening preserves information best when it targets the "
            "frequency bands with lowest d2_fraction (noise-dominated). "
            "High-d2_fraction components near P300 latency must not be "
            "attenuated by the whitening operator."
        ),
    }


# ---------------------------------------------------------------------------
# Full bound report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InformationBoundReport:
    """Full information-theoretic analysis of one participant's calibration.

    Computes and freezes:
      - Mahalanobis SNR from empirical calibration epochs
      - Declared Gaussian-model hit estimate and pairwise lower bound
      - Spectral discriminability decomposition
      - Efficiency gap: (empirical_hit - bound) / (1 - bound)
    """
    d2_per_flash: float
    n_candidates: int
    n_flashes: int
    empirical_hit_rate: float | None
    bound: dict
    snr_analysis: dict
    spectral_analysis: dict | None
    provenance: dict
    _integrity_sha256: str

    @classmethod
    def compute(
        cls,
        target_epochs: np.ndarray,
        nontarget_epochs: np.ndarray,
        *,
        n_candidates: int,
        n_flashes_per_selection: int,
        sfreq: float,
        empirical_hit_rate: float | None = None,
        compute_spectral: bool = True,
        n_spectral_components: int = 8,
        participant_id: str = "unknown",
    ) -> InformationBoundReport:
        snr = mahalanobis_snr(target_epochs, nontarget_epochs)
        d2 = snr["d2_mahalanobis"]

        bound = theoretical_hit_rate_bound(d2, n_candidates, n_flashes_per_selection)

        spec = None
        tgt = np.asarray(target_epochs, dtype=float)
        non = np.asarray(nontarget_epochs, dtype=float)
        if compute_spectral and tgt.ndim == 3:
            spec = spectral_discriminability(
                tgt, non, sfreq, n_components=n_spectral_components
            )

        # Efficiency gap: how much of the remaining room does empirical hit use?
        # The Monte Carlo estimate is only the declared Gaussian-model rate;
        # it is not a universal achievable rate or an information-theoretic cap.
        efficiency = None
        if empirical_hit_rate is not None:
            best_possible = float(bound["p_hit_monte_carlo"])
            chance = 1.0 / n_candidates
            gap = best_possible - float(empirical_hit_rate)
            efficiency = {
                "empirical_hit": float(empirical_hit_rate),
                "achievable_hit_mc": best_possible,
                "chance_hit": chance,
                "absolute_gap": gap,
                "relative_gap_pct": 100.0 * gap / max(1.0 - float(empirical_hit_rate), np.finfo(float).tiny),
                "fraction_of_room_used": max(
                    0.0,
                    (float(empirical_hit_rate) - chance)
                    / max(best_possible - chance, np.finfo(float).tiny),
                ),
            }
        if efficiency is not None:
            bound["efficiency"] = efficiency

        prov = {
            "schema": _SCHEMA,
            "participant_id": participant_id,
            "n_candidates": n_candidates,
            "n_flashes": n_flashes_per_selection,
            "sfreq": float(sfreq),
            "n_target_epochs": int(len(target_epochs)),
            "n_nontarget_epochs": int(len(nontarget_epochs)),
            "target_sha256": array_digest(np.asarray(target_epochs, dtype=float)),
            "nontarget_sha256": array_digest(np.asarray(nontarget_epochs, dtype=float)),
            "implementation_sha256": _IMPLEMENTATION_SHA256,
            "evidence_scope": "diagnostic_research_no_production_claim",
        }
        integrity = metadata_digest({**prov, "d2": d2})

        obj = object.__new__(cls)
        object.__setattr__(obj, "d2_per_flash", d2)
        object.__setattr__(obj, "n_candidates", n_candidates)
        object.__setattr__(obj, "n_flashes", n_flashes_per_selection)
        object.__setattr__(obj, "empirical_hit_rate", empirical_hit_rate)
        object.__setattr__(obj, "bound", bound)
        object.__setattr__(obj, "snr_analysis", snr)
        object.__setattr__(obj, "spectral_analysis", spec)
        object.__setattr__(obj, "provenance", prov)
        object.__setattr__(obj, "_integrity_sha256", integrity)
        return obj

    def summary(self) -> dict:
        """JSON-serialisable summary for reporting."""
        s = {
            "d2_per_flash": self.d2_per_flash,
            "erp_snr_db": self.snr_analysis["erp_snr_db"],
            "n_candidates": self.n_candidates,
            "n_flashes": self.n_flashes,
            "p_hit_lower_bound": self.bound["p_hit_lower_bound"],
            "p_hit_monte_carlo": self.bound["p_hit_monte_carlo"],
            "d2_effective_per_selection": self.bound["d2_effective_per_selection"],
        }
        if self.empirical_hit_rate is not None and "efficiency" in self.bound:
            s.update(self.bound["efficiency"])
        if self.spectral_analysis:
            s["top_component_d2_fraction"] = (
                self.spectral_analysis["components"][0]["d2_fraction"]
                if self.spectral_analysis["components"] else None
            )
            s["top_component_freq_hz"] = (
                self.spectral_analysis["components"][0]["dominant_freq_hz"]
                if self.spectral_analysis["components"] else None
            )
        return s
