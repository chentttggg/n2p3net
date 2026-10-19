"""Exact propagated covariance F K_AR F^T for finite bidirectional SOS operators.

First-principles derivation
----------------------------
The observed signal X satisfies a (filtered) AR(p) noise model:

    X = D_q W + E,   vec(E) ~ N(0, K_AR ⊗ Σ)

where K_AR is the N×N Toeplitz covariance of the AR(p) process and Σ is the
C×C channel covariance. After applying a finite bidirectional SOS operator F:

    Z = F X = F D_q W + F E

The noise covariance in the Z domain is:

    Cov(vec(FE)) = (F ⊗ I_C)(K_AR ⊗ Σ)(F^T ⊗ I_C) = (F K_AR F^T) ⊗ Σ

because the channels are mixed only by Σ, not by F.

Current approximation vs this module
--------------------------------------
ContinuousDecisionModel and OperatorGLS apply F to the data and then whiten
using the AR(p) structure of the whitened residuals, implicitly treating
K_propagated ≈ K_AR. This is only correct when F K_AR F^T ≈ K_AR, which
holds well in the flat passband but fails near the filter's cutoff frequencies.

This module provides the exact K_propagated = F K_AR F^T for diagnostic
comparison and for use in a reference GLS that bypasses the approximation.

The computation is O(N^2 p) for the AR covariance construction and O(N^3)
for the Cholesky of the N×N propagated covariance. For the BNCI case
(N ≈ 307, p ≤ 32) this is approximately 3×10^7 operations — fully tractable
as a diagnostic/research arm.

Scope
------
Research-only diagnostic. No production import. Results are bound to the
declared AR coefficients, operator digest, and grid length.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov, solve_triangular

from data.finite_filter_operator import FiniteFilterOperator
from transfer.noise_model import array_digest, frozen_array, metadata_digest
from transfer.temporal_noise import TemporalNoise

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_SCHEMA = "n2p3_exact_propagated_covariance/1"

# ---------------------------------------------------------------------------
# AR(p) covariance construction
# ---------------------------------------------------------------------------

def _autocovariance_sequence(coefficients, innovation_variance, n_samples):
    """Exact autocovariance r[0..n-1] of a stationary AR(p) process.

    For x[t] = sum_{j=1}^{p} a[j-1] x[t-j] + eps[t], the state z_t =
    [x_t, ..., x_{t-p+1}]^T obeys z_t = C z_{t-1} + e_1 eps_t with the
    companion matrix C. The stationary state covariance solves the discrete
    Lyapunov equation P = C P C^T + sigma2 e_1 e_1^T, and r[k] = P[k, 0].

    Only the first p lags come from P; later lags follow the forward recursion
    r[k] = sum_{j=1}^{p} a[j-1] r[k-j]. The recursion alone is *not* a valid
    construction for k < p because it references not-yet-solved unknowns.
    """
    p = len(coefficients)
    if p == 0:
        r = np.zeros(max(n_samples, 1))
        r[0] = innovation_variance
        return r
    a = np.asarray(coefficients, dtype=float)
    companion = np.zeros((p, p))
    companion[0] = a
    companion[1:, :-1] = np.eye(p - 1)
    driving = np.zeros((p, p))
    driving[0, 0] = innovation_variance
    state_covariance = solve_discrete_lyapunov(companion, driving)
    state_covariance = (state_covariance + state_covariance.T) / 2
    r = np.zeros(n_samples)
    leading = min(p, n_samples)
    r[:leading] = state_covariance[np.arange(leading), 0]
    for k in range(p, n_samples):
        r[k] = float(a @ r[k - p:k][::-1])
    return r


def ar_toeplitz_covariance(noise: TemporalNoise, n_samples: int) -> np.ndarray:
    """Build the exact N×N Toeplitz AR(p) covariance matrix.

    Uses the exact stationary autocovariance sequence (Lyapunov-solved leading
    lags plus the forward recursion), then fills a symmetric Toeplitz matrix.
    The scale is noise.innovation_variance, matching TemporalNoise.
    """
    K = _ar_covariance_fast(noise, n_samples)
    return K


def _ar_covariance_fast(noise: TemporalNoise, n_samples: int) -> np.ndarray:
    """O(N p) exact construction after one p×p Lyapunov solve."""
    if not isinstance(noise, TemporalNoise):
        raise ValueError("AR covariance requires a fitted TemporalNoise")
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if noise.startup_mode == "conditional_prefix":
        raise ValueError(
            "conditional-prefix noise has a different estimand; use its "
            "innovation covariance after explicit prefix conditioning"
        )
    noise.assert_integrity()

    if noise.order == 0:
        return noise.innovation_variance * np.eye(n_samples)

    r = _autocovariance_sequence(
        noise.coefficients, noise.innovation_variance, n_samples
    )
    rows = np.arange(n_samples)
    lag = np.abs(rows[:, None] - rows[None, :])
    K = r[lag]
    return (K + K.T) / 2


# ---------------------------------------------------------------------------
# Exact propagated covariance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExactPropagatedCovariance:
    """F K_AR F^T for a declared finite operator and AR noise model.

    Attributes
    ----------
    K_propagated : ndarray (N, N)
        The exact propagated temporal covariance in the filtered domain.
    K_ar : ndarray (N, N)
        The original AR(p) temporal covariance before filtering.
    cholesky_propagated : ndarray (N, N)
        Lower-triangular Cholesky factor of K_propagated (for whitening).
    n_samples : int
    operator_digest : str
        Digest of the FiniteFilterOperator record.
    noise_digest : str
        Digest of the TemporalNoise record.
    relative_error_trace : float
        Relative Frobenius-norm difference between K_propagated and K_ar,
        i.e. ||F K F^T - K||_F / ||K||_F. A large value means the
        current approximation (treating filtered noise as still AR) is poor.
    provenance : dict
    """
    K_propagated: np.ndarray
    K_ar: np.ndarray
    cholesky_propagated: np.ndarray
    n_samples: int
    operator_digest: str
    noise_digest: str
    relative_error_trace: float
    provenance: dict
    _integrity_sha256: str

    @classmethod
    def compute(
        cls,
        operator: FiniteFilterOperator,
        noise: TemporalNoise,
        n_samples: int,
        *,
        byte_budget: int = 64 * 1024 * 1024,  # 64 MiB
    ) -> ExactPropagatedCovariance:
        """Compute and freeze F K_AR F^T.

        Parameters
        ----------
        operator : FiniteFilterOperator
            The finite bidirectional SOS operator.
        noise : TemporalNoise
            Fitted AR(p) noise, assumed to describe the pre-filter signal.
        n_samples : int
            Number of output samples (the grid length).
        byte_budget : int
            Upper bound on memory for N×N matrices (2 of them).
            Default 64 MiB handles N up to ~2900.
        """
        if not isinstance(operator, FiniteFilterOperator):
            raise ValueError("operator must be a FiniteFilterOperator")
        if not isinstance(noise, TemporalNoise):
            raise ValueError("noise must be a TemporalNoise")
        if type(n_samples) is not int or n_samples < 3:
            raise ValueError("n_samples must be an integer >= 3")
        if 2 * 8 * n_samples ** 2 > byte_budget:
            raise ValueError(
                f"two {n_samples}×{n_samples} float64 matrices exceed the "
                f"{byte_budget // (1024**2)} MiB diagnostic budget"
            )
        noise.assert_integrity()

        # Step 1: Build AR covariance K on a grid of n_samples columns
        K_ar = _ar_covariance_fast(noise, n_samples)

        # Step 2: Compute F applied column-by-column to the identity (= rows of F)
        # F @ K @ F^T = (F @ I) @ K @ (F @ I)^T
        # We need the N×N matrix F. Rather than materialising it fully,
        # compute F @ K_ar directly: F has rows f_i = F e_i = operator applied
        # to the i-th standard basis vector.
        #
        # Memory-efficient: apply operator to K_ar column by column.
        # K_ar is symmetric, so F K F^T = F (F K^T)^T = F (F K)^T.
        # Compute FK = F @ K_ar row by row (operator applied to each column of K_ar^T = K_ar).
        FK = operator.apply(K_ar)          # shape (n_samples, n_samples)
        K_prop = operator.apply(FK.T).T    # F K F^T

        # Symmetrize
        K_prop = (K_prop + K_prop.T) / 2

        # Relative error vs original AR covariance
        norm_K = float(np.linalg.norm(K_ar, "fro"))
        rel_err = float(np.linalg.norm(K_prop - K_ar, "fro")) / max(norm_K, np.finfo(float).tiny)

        # A bandpass operator annihilates DC (and, with finite odd-reflection
        # padding, near-DC), so F K F^T is positive SEMI-definite, not
        # definite. Straight Cholesky therefore fails on a rank-deficient but
        # mathematically valid covariance. Use the eigendecomposition to
        # separate the genuine range from the numerical null space, then add a
        # relative ridge that is many orders of magnitude below the smallest
        # retained eigenvalue before factoring.
        eigenvalues, eigenvectors = np.linalg.eigh(K_prop)
        max_eig = float(eigenvalues[-1])
        min_eig = float(eigenvalues[0])
        if max_eig <= 0:
            raise ValueError("propagated covariance has no positive direction")
        # Numerical rank: directions with meaningful variance relative to the top.
        relative_floor = max_eig * 1e-10
        rank = int(np.count_nonzero(eigenvalues > relative_floor))
        null_dimension = int(n_samples - rank)
        # Ridge is set far below the retained spectrum so it does not distort
        # any identified direction, matching the relative-floor convention used
        # elsewhere in the project (e.g. OperatorAwareLDA identifiable support).
        ridge = max_eig * 1e-12
        K_prop_regularized = K_prop + ridge * np.eye(n_samples)
        try:
            L = np.linalg.cholesky(K_prop_regularized)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"regularized propagated covariance failed Cholesky "
                f"(rel_err={rel_err:.4f}, rank={rank}/{n_samples}, ridge={ridge:.3e}): {exc}"
            ) from exc

        op_digest = operator.digest
        noise_digest = array_digest(noise.coefficients, np.asarray(noise.innovation_variance))

        prov = {
            "schema": _SCHEMA,
            "n_samples": n_samples,
            "ar_order": noise.order,
            "ar_innovation_variance": float(noise.innovation_variance),
            "ar_marginal_variance": float(noise.marginal_variance_per_unit_innovation
                                          * noise.innovation_variance),
            "operator_digest": op_digest,
            "noise_digest": noise_digest,
            "relative_frobenius_error_FKF_vs_K": rel_err,
            "numerical_rank": rank,
            "null_dimension": null_dimension,
            "relative_rank_floor": 1e-10,
            "min_eigenvalue_raw": min_eig,
            "max_eigenvalue": max_eig,
            "regularization_ridge": ridge,
            "positivity_note": (
                "bandpass F annihilates near-DC directions; F K F^T is PSD and "
                "rank-deficient, not indefinite. The null space is a genuine "
                "feature of the filtered noise model, not a numerical defect."
            ),
            "implementation_sha256": _IMPLEMENTATION_SHA256,
            "evidence_scope": "diagnostic_research_no_production_claim",
        }

        integrity = metadata_digest({
            **prov,
            "K_propagated_sha256": array_digest(K_prop),
            "K_ar_sha256": array_digest(K_ar),
            "cholesky_sha256": array_digest(L),
        })

        obj = object.__new__(cls)
        object.__setattr__(obj, "K_propagated", frozen_array(K_prop))
        object.__setattr__(obj, "K_ar", frozen_array(K_ar))
        object.__setattr__(obj, "cholesky_propagated", frozen_array(L))
        object.__setattr__(obj, "eigenvalues", frozen_array(eigenvalues))
        object.__setattr__(obj, "eigenvectors", frozen_array(eigenvectors))
        object.__setattr__(obj, "numerical_rank", rank)
        object.__setattr__(obj, "null_dimension", null_dimension)
        object.__setattr__(obj, "regularization_ridge", ridge)
        object.__setattr__(obj, "n_samples", n_samples)
        object.__setattr__(obj, "operator_digest", op_digest)
        object.__setattr__(obj, "noise_digest", noise_digest)
        object.__setattr__(obj, "relative_error_trace", rel_err)
        object.__setattr__(obj, "provenance", prov)
        object.__setattr__(obj, "_integrity_sha256", integrity)
        return obj

    def assert_integrity(self) -> None:
        check = metadata_digest({
            **self.provenance,
            "K_propagated_sha256": array_digest(self.K_propagated),
            "K_ar_sha256": array_digest(self.K_ar),
            "cholesky_sha256": array_digest(self.cholesky_propagated),
        })
        if check != self._integrity_sha256:
            raise ValueError("ExactPropagatedCovariance integrity check failed")

    def whiten(self, values: np.ndarray) -> np.ndarray:
        """Apply the whitening factor to (samples, channels).

        Uses the Cholesky factor of the ridge-regularized propagated
        covariance. Directions in the numerical null space carry no noise
        variance; they receive the ridge and are thus heavily (but finitely)
        whitened rather than rejected.
        """
        self.assert_integrity()
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or x.shape[0] != self.n_samples:
            raise ValueError(
                f"values must be ({self.n_samples}, C); got {x.shape}"
            )
        if not np.isfinite(x).all():
            raise ValueError("values must be finite for exact whitening")
        return solve_triangular(self.cholesky_propagated, x, lower=True)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def spectral_comparison(self, sfreq: float) -> dict:
        """Compare the eigen-spectra of K_ar and K_propagated.

        K_ar is Toeplitz, so its eigenvalues approximate the AR power spectral
        density. K_propagated = F K F^T is NOT Toeplitz, so its first row is
        not an autocovariance sequence; its spectrum is reported via its own
        eigenvalues rather than a row FFT.
        """
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError("sfreq must be positive")
        self.assert_integrity()
        eig_ar = np.linalg.eigvalsh(self.K_ar)
        eig_prop = self.eigenvalues
        # Frequency axis is nominal: eigenvalue index maps to the sorted
        # spectral support, not to a physical frequency bin for K_propagated.
        return {
            "eigenvalues_ar": eig_ar.tolist(),
            "eigenvalues_propagated": eig_prop.tolist(),
            "max_eigenvalue_ar": float(eig_ar[-1]),
            "max_eigenvalue_propagated": float(eig_prop[-1]),
            "trace_ar": float(np.trace(self.K_ar)),
            "trace_propagated": float(np.trace(self.K_propagated)),
            "note": (
                "K_ar eigenvalue index approximates frequency order; "
                "K_propagated is non-Toeplitz so its index is a sorted rank, "
                "not a frequency."
            ),
        }

    def whitening_gain(self) -> dict:
        """Ratio of whitening gains: exact vs AR approximation.

        When OperatorGLS whittens with the AR covariance instead of F K F^T,
        the effective noise in the GLS is L_AR^{-1} F E rather than white.
        This method quantifies the deviation as the singular values of
        L_AR^{-1} @ cholesky_propagated, which should all be 1 if K_prop = K_ar.
        """
        self.assert_integrity()
        # Approximate whitening uses cholesky(K_ar)
        try:
            L_ar = np.linalg.cholesky(self.K_ar)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"K_ar is not positive definite: {exc}") from exc

        # M = L_ar^{-1} @ L_prop
        M = solve_triangular(L_ar, self.cholesky_propagated, lower=True)
        sv = np.linalg.svd(M, compute_uv=False)
        return {
            "singular_values_min": float(sv.min()),
            "singular_values_max": float(sv.max()),
            "condition_number": float(sv.max() / max(sv.min(), np.finfo(float).tiny)),
            "frobenius_deviation_from_identity": float(np.linalg.norm(M - np.eye(self.n_samples), "fro")),
            "relative_frobenius_error": self.relative_error_trace,
            "interpretation": (
                "If singular values are all near 1.0, the AR approximation is accurate. "
                "Condition >> 1 means the AR whitening is misspecified in some directions."
            ),
        }

    def log_likelihood_correction(
        self, whitened_residuals: np.ndarray
    ) -> float:
        """Return log p_prop(r) - log p_ar(r) for the same supplied residual.

        L_prop factors K_propagated + rho I, including the ridge used by
        whiten(), while L_AR factors K_ar. For C residual columns with a
        common identity spatial covariance, or already spatial-whitened
        columns, the diagnostic is

            delta_ll = -0.5 * [||L_prop^{-1} r||^2 - ||L_AR^{-1} r||^2]
                       - C * [sum(log(diag(L_prop))) - sum(log(diag(L_AR)))]

        The determinant term means the difference of sums of log diagonals.
        Positive delta_ll assigns greater density under the regularized
        propagated covariance for this same residual.

        whitened_residuals is the historical parameter name. Both factors
        solve against the array as passed; this method does not undo earlier
        temporal whitening or update a fitted mean or posterior coefficient
        uncertainty. Refitted or integrated GLS needs those extra changes.
        """
        self.assert_integrity()
        r = np.asarray(whitened_residuals, dtype=float)
        if r.ndim != 2 or r.shape[0] != self.n_samples:
            raise ValueError(
                f"residuals must be ({self.n_samples}, C); got {r.shape}"
            )
        try:
            L_ar = np.linalg.cholesky(self.K_ar)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"K_ar is not positive definite: {exc}") from exc

        r_approx = solve_triangular(L_ar, r, lower=True)
        r_exact = self.whiten(r)

        quad_approx = float(np.sum(r_approx ** 2))
        quad_exact = float(np.sum(r_exact ** 2))
        logdet_approx = float(np.log(np.diag(L_ar)).sum())
        logdet_exact = float(np.log(np.diag(self.cholesky_propagated)).sum())
        n_channels = r.shape[1]

        # delta_ll per-candidate (already summed over samples and channels)
        delta_ll = -0.5 * (quad_exact - quad_approx) - n_channels * (logdet_exact - logdet_approx)
        return delta_ll
