"""Finite covariance propagation and explicitly declared Gaussian measures.

For PRE-FILTER E with row-major Cov(vec(E))=K_AR ⊗ Sigma and fixed sample
operator F, Cov(vec(F E))=(F K_AR F.T) ⊗ Sigma. The declared stationary AR
model must describe that pre-filter noise; an AR fitted after filtering is a
different input law. Conditional-prefix covariance is not stationary Toeplitz.

Raw propagation is retained independently of whether an ambient SPD density
is numerically resolved. Singular laws need an explicit support declaration
and its own Euclidean measure. Regularized laws are separate diagnostics.
No production caller is installed by this research-only module. State binds
the original noise/operator records, grid, factors and support declaration.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov, solve_triangular

from data.finite_filter_operator import FiniteFilterOperator
from transfer.noise_model import array_digest, finite_real_array, frozen_array, metadata_digest
from transfer.temporal_noise import TemporalNoise

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_SCHEMA = "n2p3_exact_propagated_covariance/2"

# ---------------------------------------------------------------------------
# AR(p) covariance construction
# ---------------------------------------------------------------------------

def _autocovariance_sequence(coefficients, innovation_variance, n_samples):
    """Exact autocovariance r[0..n-1] of a stationary AR(p) process.

    For x[t] = sum_{j=1}^{p} a[j-1] x[t-j] + eps[t], the state z_t =
    [x_t, ..., x_{t-p+1}]^T obeys z_t = C z_{t-1} + e_1 eps_t with the
    companion matrix C. The stationary state covariance solves the discrete
    Lyapunov equation P = C P C^T + sigma2 e_1 e_1^T, and r[k] = P[k, 0]
    only for the leading lags 0 <= k < p.

    Only the first p lags come from P; later lags follow the forward recursion
    r[k] = sum_{j=1}^{p} a[j-1] r[k-j]. The recursion alone is *not* a valid
    construction for k < p because it references not-yet-solved unknowns.
    These are model identities; floating Lyapunov arithmetic is not a
    certificate for an arbitrary high-order or ill-conditioned AR model.
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
    """Raw F K_AR F.T and a declared Gaussian measure, without a ridge.

    SPD laws use ambient Lebesgue density. A singular or unresolved operator
    requires explicit orthonormal support_basis Q for the identified law
    Q (Q.T K Q) Q.T. Its density uses Euclidean volume on that support.
    Numerical rank never silently selects or truncates the statistical law.
    """
    K_propagated: np.ndarray
    K_ar: np.ndarray
    cholesky_propagated: np.ndarray | None
    support_basis: np.ndarray | None
    cholesky_support: np.ndarray | None
    eigenvalues: np.ndarray
    numerical_rank: int
    n_samples: int
    operator_digest: str
    noise_digest: str
    relative_error_trace: float
    provenance: dict
    _integrity_sha256: str

    @property
    def null_dimension(self) -> int:
        return self.n_samples - self.numerical_rank

    @property
    def support_dimension(self) -> int | None:
        return None if self.support_basis is None else self.support_basis.shape[1]

    @classmethod
    def compute(cls, operator: FiniteFilterOperator, noise: TemporalNoise,
                n_samples: int, *, support_basis: np.ndarray | None = None,
                byte_budget: int = 64 * 1024 * 1024) -> ExactPropagatedCovariance:
        """Compile finite covariance; support declaration is explicit.

        Q has shape (N,r), orthonormal columns and contains the actual filter
        image to float64 backward-error scale. r=0 is a point mass. The budget
        reserves eight dense float64 payloads; LAPACK-specific workspace may
        add allocations. No spectral floor becomes an automatic support law.
        """
        if not isinstance(operator, FiniteFilterOperator):
            raise ValueError("operator must be a FiniteFilterOperator")
        if not isinstance(noise, TemporalNoise):
            raise ValueError("noise must be a TemporalNoise")
        if type(n_samples) is not int or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        if type(byte_budget) is not int or byte_budget < 1:
            raise ValueError("byte_budget must be a positive integer")
        if 64 * n_samples**2 > byte_budget:
            raise ValueError("propagated covariance exceeds the explicit dense diagnostic budget")
        K_ar = _ar_covariance_fast(noise, n_samples)
        # apply consumes samples by columns: apply(I) is F, not F.T.
        F = operator.apply(np.eye(n_samples))
        K_prop = F @ K_ar @ F.T
        K_prop = (K_prop + K_prop.T) / 2
        eigenvalues = np.linalg.eigvalsh(K_prop)
        singular_values = np.linalg.svd(F, compute_uv=False)
        rank_floor = float(singular_values[0] * n_samples * np.finfo(float).eps)
        rank = int(np.count_nonzero(singular_values > rank_floor))
        tolerance = 64 * n_samples * np.finfo(float).eps
        L = Q = L_support = None
        mode = "raw_covariance_support_unresolved"
        ambient_factorization_status = "not_attempted_operator_rank_unresolved"
        image_error = None
        noise_image_error = None
        if support_basis is not None:
            Q = finite_real_array(support_basis, name="support_basis")
            if Q.ndim != 2 or Q.shape[0] != n_samples or Q.shape[1] > n_samples:
                raise ValueError("support_basis must have shape (n_samples, rank)")
            if np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), "fro") > tolerance:
                raise ValueError("support_basis must have orthonormal columns")
            filter_scale = float(np.max(np.abs(F)))
            scaled_F = F / filter_scale if filter_scale else F
            image_error = float(np.linalg.norm(scaled_F - Q @ (Q.T @ scaled_F), "fro"))
            if image_error > tolerance * float(np.linalg.norm(scaled_F, "fro")):
                raise ValueError("support_basis does not contain the actual filter image")
            noise_image = F @ np.linalg.cholesky(K_ar)
            noise_scale = float(np.max(np.abs(noise_image)))
            scaled_noise_image = noise_image / noise_scale if noise_scale else noise_image
            noise_image_error = float(np.linalg.norm(
                scaled_noise_image - Q @ (Q.T @ scaled_noise_image), "fro"
            ))
            if noise_image_error > tolerance * float(np.linalg.norm(scaled_noise_image, "fro")):
                raise ValueError("support_basis does not contain the propagated noise image")
            B = Q.T @ noise_image
            try:
                singular_support = np.linalg.svd(B, compute_uv=False)
                if Q.shape[1] and (
                    singular_support[-1] <= singular_support[0] * n_samples * np.finfo(float).eps
                ):
                    raise ValueError("declared support includes an unresolved zero-variance direction")
                L_support = np.linalg.cholesky(B @ B.T)
            except np.linalg.LinAlgError as exc:
                raise ValueError("declared support covariance is not positive definite") from exc
            mode = "declared_identified_subspace"
            ambient_factorization_status = "not_requested_declared_support_measure"
        elif rank == n_samples:
            try:
                L = np.linalg.cholesky(K_prop)
            except np.linalg.LinAlgError:
                # The finite operator may be invertible while forming F K F.T
                # loses its tiny positive eigenvalues to roundoff. Preserve the
                # raw propagation diagnostic, without inventing a support or
                # adding a ridge to make a different law factorable.
                ambient_factorization_status = "raw_covariance_cholesky_unresolved"
            else:
                Q = np.eye(n_samples)
                L_support = L
                mode = "ambient_spd"
                ambient_factorization_status = "resolved_without_regularization"
        covariance_scale = float(np.max(np.abs(K_ar)))
        rel_err = float(np.linalg.norm((K_prop - K_ar) / covariance_scale, "fro")
                        / np.linalg.norm(K_ar / covariance_scale, "fro"))
        prov = {
            "schema": _SCHEMA, "n_samples": n_samples,
            "operator_record": operator.record(), "operator_digest": operator.digest,
            "noise_record": noise.record(), "noise_digest": metadata_digest(noise.record()),
            "relative_frobenius_error_FKF_vs_K": rel_err,
            "numerical_operator_rank": rank, "operator_singular_value_floor": rank_floor,
            "rank_role": "resolution_diagnostic_not_automatic_support_declaration",
            "measure": mode, "support_dimension": None if Q is None else Q.shape[1],
            "ambient_factorization_status": ambient_factorization_status,
            "support_max_scaled_filter_image_error_frobenius": image_error,
            "support_max_scaled_noise_image_error_frobenius": noise_image_error,
            "support_relative_roundoff_tolerance": tolerance,
            "support_measure": "Euclidean_volume_in_declared_orthonormal_chart",
            "regularization_ridge": 0.0, "implementation_sha256": _IMPLEMENTATION_SHA256,
            "evidence_scope": "diagnostic_research_no_production_claim",
        }
        obj = cls(frozen_array(K_prop), frozen_array(K_ar),
                  None if L is None else frozen_array(L),
                  None if Q is None else frozen_array(Q),
                  None if L_support is None else frozen_array(L_support),
                  frozen_array(eigenvalues), rank, n_samples, operator.digest,
                  prov["noise_digest"], rel_err, prov, "")
        object.__setattr__(obj, "_integrity_sha256", obj._state_digest())
        return obj

    def _state_digest(self) -> str:
        return metadata_digest({
            "provenance": self.provenance, "n_samples": self.n_samples,
            "operator_digest": self.operator_digest, "noise_digest": self.noise_digest,
            "relative_error_trace": self.relative_error_trace, "numerical_rank": self.numerical_rank,
            "arrays": {name: None if getattr(self, name) is None else array_digest(getattr(self, name))
                       for name in ("K_propagated", "K_ar", "cholesky_propagated", "support_basis",
                                    "cholesky_support", "eigenvalues")},
        })

    def assert_integrity(self) -> None:
        if self._state_digest() != self._integrity_sha256:
            raise ValueError("ExactPropagatedCovariance integrity check failed")

    def _values(self, values: np.ndarray) -> np.ndarray:
        self.assert_integrity()
        x = finite_real_array(values, name="values")
        if x.ndim != 2 or x.shape[0] != self.n_samples or x.shape[1] < 1:
            raise ValueError(f"values must be ({self.n_samples}, C) with C >= 1; got {x.shape}")
        return x

    def whiten(self, values: np.ndarray) -> np.ndarray:
        """Whiten raw on-support residuals into (support dimension, channels)."""
        x = self._values(values)
        if self.support_basis is None:
            raise ValueError("declare support_basis for singular or unresolved covariance whitening")
        coordinates = self.support_basis.T @ x
        residual = x - self.support_basis @ coordinates
        tolerance = self.provenance["support_relative_roundoff_tolerance"]
        # Each channel has its own scale: a large channel cannot hide a weak
        # channel's off-support component.
        for column in range(x.shape[1]):
            scale = float(np.max(np.abs(x[:, column])))
            if scale and (
                np.linalg.norm(residual[:, column] / scale)
                > tolerance * np.linalg.norm(x[:, column] / scale)
            ):
                raise ValueError("values lie outside the declared covariance support")
        return solve_triangular(self.cholesky_support, coordinates, lower=True)

    def log_density_on_support(self, residuals: np.ndarray) -> float:
        """Density wrt Euclidean support volume (Dirac at r=0).

        Columns are independent unit-channel Gaussian residuals. Correlated
        channels require separately declared whitening and volume correction.
        Different supports define different measures, not ambient Bayes factors.
        """
        w = self.whiten(residuals)
        log_volume = float(np.log(np.diag(self.cholesky_support)).sum())
        return float(-0.5 * np.sum(w**2) - w.shape[1] * log_volume
                     - 0.5 * w.size * np.log(2 * np.pi))

    def log_density(self, residuals: np.ndarray) -> float:
        """Ordinary ambient Gaussian density, available only for the SPD law."""
        self.assert_integrity()
        if self.cholesky_propagated is None:
            raise ValueError("ambient density requires an SPD covariance; use declared support measure")
        return self.log_density_on_support(residuals)

    def regularized_diagnostic(self, values: np.ndarray, *, ridge: float) -> dict:
        """Explicit full-rank diagnostic law K_prop + ridge I, never exact."""
        x = self._values(values)
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("diagnostic ridge must be finite and positive")
        covariance = self.K_propagated + ridge * np.eye(self.n_samples)
        try:
            L = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("requested ridge does not yield a resolved SPD diagnostic") from exc
        w = solve_triangular(L, x, lower=True)
        density = float(-0.5 * np.sum(w**2) - x.shape[1] * np.log(np.diag(L)).sum()
                        - 0.5 * x.size * np.log(2 * np.pi))
        return {"measure": "regularized_ambient_diagnostic", "ridge": float(ridge),
                "source_state_sha256": self._integrity_sha256,
                "covariance_sha256": array_digest(covariance),
                "cholesky": frozen_array(L), "whitened": frozen_array(w), "log_density": density}

    def spectral_comparison(self, sfreq: float) -> dict:
        """Sorted covariance eigenvalues; their index is not a frequency bin."""
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError("sfreq must be positive")
        self.assert_integrity()
        eig_ar = np.linalg.eigvalsh(self.K_ar)
        return {"eigenvalues_ar": eig_ar.tolist(), "eigenvalues_propagated": self.eigenvalues.tolist(),
                "max_eigenvalue_ar": float(eig_ar[-1]),
                "max_eigenvalue_propagated": float(self.eigenvalues[-1]),
                "trace_ar": float(np.trace(self.K_ar)),
                "trace_propagated": float(np.trace(self.K_propagated)),
                "note": "Sorted eigenvalue indices of either covariance are not physical frequencies."}

    def whitening_gain(self) -> dict:
        """Actual covariance M=L_AR^-1 K_prop L_AR^-T after AR whitening.

        Eigenvalues are variance gains; their square roots are std gains. The
        deviation is covariance error, never a difference of square-root charts.
        """
        self.assert_integrity()
        L_ar = np.linalg.cholesky(self.K_ar)
        left = solve_triangular(L_ar, self.K_propagated, lower=True)
        M = solve_triangular(L_ar, left.T, lower=True).T
        M = (M + M.T) / 2
        eigenvalues = np.linalg.eigvalsh(M)
        std = np.sqrt(np.maximum(eigenvalues, 0))
        minimum = 0.0 if self.null_dimension else float(std[0])
        condition = float(std[-1] / minimum) if minimum > 0 else float("inf")
        return {"covariance_after_ar_whitening": M.tolist(), "variance_gains": eigenvalues.tolist(),
                "singular_values_min": minimum, "singular_values_max": float(std[-1]),
                "condition_number": condition,
                "frobenius_deviation_from_identity": float(np.linalg.norm(M - np.eye(self.n_samples), "fro")),
                "relative_frobenius_error": self.relative_error_trace,
                "deviation_role": "covariance_not_square_root_chart"}

    def log_likelihood_correction(self, residuals: np.ndarray) -> float:
        """Ambient log p_prop(r)-log p_ar(r), RAW residuals and SPD only.

        Both laws use the same Lebesgue measure and independent unit channels.
        A support density cannot be subtracted from an ambient density.
        """
        self.assert_integrity()
        if self.cholesky_propagated is None:
            raise ValueError("likelihood correction requires SPD: null space or unresolved support")
        r = self._values(residuals)
        L_ar = np.linalg.cholesky(self.K_ar)
        w_ar = solve_triangular(L_ar, r, lower=True)
        w_prop = self.whiten(r)
        log_ratio = np.log(np.diag(self.cholesky_propagated)).sum() - np.log(np.diag(L_ar)).sum()
        return float(-0.5 * (np.sum(w_prop**2) - np.sum(w_ar**2)) - r.shape[1] * log_ratio)
