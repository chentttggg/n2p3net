"""Data-level alignment for heterogeneous EEG feature spaces.

The adapter in this module operates on already constructed feature vectors,
not on physical :class:`data.epochs.EpochDataset` channel semantics.  A caller
must therefore provide an explicit per-domain lift when raw montages differ.
GALIA is represented by a shared approximate joint diagonalizer and one
domain-specific diagonal scale, with all fit scope and numerical diagnostics
kept in a serializable state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

ACW_ALIGNMENT_SCHEMA = "n2p3_ajd_congruence_whitening/1"
FAST_ALIGNMENT_SCHEMA = "n2p3_galia_fast_alignment/1"
VALID_AJD_METHODS = frozenset({"ajd_pham", "rjd", "uwedge"})
VALID_FAST_SCOPES = frozenset({"target_calibration", "source_weighted"})


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _readonly(value: np.ndarray, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    if dtype is None:
        dtype = np.dtype(np.float64)
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _validate_feature_matrix(value: object, *, name: str, min_samples: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, n_features), got {array.shape}.")
    if array.shape[0] < min_samples or array.shape[1] < 1:
        raise ValueError(
            f"{name} needs at least {min_samples} samples and one feature; got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return array


def _validate_domains(
    domains: Mapping[str, object],
    *,
    min_samples: int,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    if not isinstance(domains, Mapping) or len(domains) < 2:
        raise ValueError("ACW requires at least two named domains.")
    order: list[str] = []
    matrices: list[np.ndarray] = []
    feature_dim: int | None = None
    for raw_name, value in domains.items():
        name = str(raw_name).strip()
        if not name or "\0" in name:
            raise ValueError("domain names must be non-empty strings without NUL.")
        if name in order:
            raise ValueError(f"domain names must be unique; repeated {name!r}.")
        matrix = _validate_feature_matrix(value, name=f"domain[{name!r}]", min_samples=min_samples)
        if feature_dim is None:
            feature_dim = matrix.shape[1]
        elif matrix.shape[1] != feature_dim:
            raise ValueError(
                "ACW requires equal feature dimensions after explicit domain lifts; "
                f"domain {name!r} has {matrix.shape[1]} and expected {feature_dim}."
            )
        order.append(name)
        matrices.append(matrix)
    return tuple(order), tuple(matrices)


def apply_feature_lift(
    epochs: np.ndarray,
    spatial_lift: np.ndarray,
    *,
    temporal_basis: np.ndarray | None = None,
) -> np.ndarray:
    """Apply an explicit spatial lift, optionally followed by a temporal basis.

    ``epochs`` remains in ``(N, C, T)`` order.  ``spatial_lift`` is ``(P, C)``
    and ``temporal_basis`` is ``(T, Q)``.  The output is ``(N, P, T)`` or
    ``(N, P, Q)``.  This function does not invent channel coordinates or pad a
    montage; the caller owns the physical meaning and provenance of the lift.
    """

    values = np.asarray(epochs, dtype=np.float64)
    lift = np.asarray(spatial_lift, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"epochs must have shape (N,C,T), got {values.shape}.")
    if lift.ndim != 2 or lift.shape[1] != values.shape[1] or lift.shape[0] < 1:
        raise ValueError(
            "spatial_lift must have shape (P,C) with C matching epochs; "
            f"got {lift.shape} for C={values.shape[1]}."
        )
    if feature_lift_rank(lift) == 0:
        raise ValueError("spatial_lift must have positive numerical rank.")
    if not np.isfinite(values).all() or not np.isfinite(lift).all():
        raise ValueError("epochs and spatial_lift must be finite.")
    projected = np.einsum("pc,nct->npt", lift, values, optimize=True)
    if temporal_basis is None:
        return projected
    temporal = np.asarray(temporal_basis, dtype=np.float64)
    if temporal.ndim != 2 or temporal.shape[0] != values.shape[2] or temporal.shape[1] < 1:
        raise ValueError(
            "temporal_basis must have shape (T,Q) with T matching epochs; "
            f"got {temporal.shape} for T={values.shape[2]}."
        )
    if not np.isfinite(temporal).all():
        raise ValueError("temporal_basis must be finite.")
    if np.linalg.matrix_rank(temporal, tol=1e-10) == 0:
        raise ValueError("temporal_basis must have positive numerical rank.")
    return np.einsum("npt,tq->npq", projected, temporal, optimize=True)


def flatten_feature_epochs(values: np.ndarray) -> np.ndarray:
    """Flatten ``(N,P,T)`` or ``(N,P,Q)`` lifted epochs into feature vectors."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"lifted epochs must have shape (N,P,T), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("lifted epochs contain NaN or infinity.")
    return np.ascontiguousarray(array.reshape(array.shape[0], -1))


def feature_lift_rank(spatial_lift: np.ndarray, *, tolerance: float = 1e-10) -> int:
    """Return the numerical rank of a domain-specific spatial lift."""

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    lift = np.asarray(spatial_lift, dtype=np.float64)
    if lift.ndim != 2 or not np.isfinite(lift).all():
        raise ValueError("spatial_lift must be a finite two-dimensional matrix.")
    return int(np.linalg.matrix_rank(lift, tol=tolerance))


@dataclass(frozen=True)
class AcwConfig:
    """Numerical and provenance settings for one GALIA fit."""

    regularization: float = 1e-6
    method: str = "ajd_pham"
    tolerance: float = 1e-6
    max_iter: int = 100
    min_samples: int = 2
    max_condition_number: float = 1e12

    def __post_init__(self) -> None:
        if not np.isfinite(self.regularization) or self.regularization <= 0.0:
            raise ValueError("regularization must be finite and positive.")
        if self.method not in VALID_AJD_METHODS:
            raise ValueError(f"method must be one of {sorted(VALID_AJD_METHODS)}.")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if isinstance(self.max_iter, bool) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if isinstance(self.min_samples, bool) or self.min_samples < 2:
            raise ValueError("min_samples must be at least 2.")
        if not np.isfinite(self.max_condition_number) or self.max_condition_number <= 1.0:
            raise ValueError("max_condition_number must be finite and greater than one.")


@dataclass(frozen=True)
class FastAlignmentState:
    """A frozen target transform derived from an explicitly declared scope."""

    parent_state_digest: str
    domain_id: str
    fit_scope: str
    mean: np.ndarray
    scales: np.ndarray
    covariance_condition_number: float
    target_offdiag_residual: float
    buffer_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.parent_state_digest, str) or len(self.parent_state_digest) != 64:
            raise ValueError("parent_state_digest must be a SHA-256 digest.")
        domain = str(self.domain_id).strip()
        if not domain or "\0" in domain:
            raise ValueError("domain_id must be non-empty and cannot contain NUL.")
        if self.fit_scope not in VALID_FAST_SCOPES:
            raise ValueError(f"fit_scope must be one of {sorted(VALID_FAST_SCOPES)}.")
        mean = np.asarray(self.mean, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        if mean.ndim != 1 or scales.ndim != 1 or mean.shape != scales.shape or not len(mean):
            raise ValueError("mean and scales must be aligned non-empty vectors.")
        if not np.isfinite(mean).all() or not np.isfinite(scales).all() or bool((scales <= 0).any()):
            raise ValueError("mean/scales must be finite and scales must be positive.")
        if not np.isfinite(self.covariance_condition_number) or self.covariance_condition_number <= 0:
            raise ValueError("covariance_condition_number must be finite and positive.")
        if (
            not np.isfinite(self.target_offdiag_residual)
            or self.target_offdiag_residual < 0.0
        ):
            raise ValueError("target_offdiag_residual must be finite and non-negative.")
        if not isinstance(self.buffer_digest, str) or len(self.buffer_digest) != 64:
            raise ValueError("buffer_digest must be a SHA-256 digest.")
        object.__setattr__(self, "domain_id", domain)
        object.__setattr__(self, "mean", _readonly(mean))
        object.__setattr__(self, "scales", _readonly(scales))

    def record(self) -> dict[str, object]:
        semantic = {
            "schema": FAST_ALIGNMENT_SCHEMA,
            "parent_state_digest": self.parent_state_digest,
            "domain_id": self.domain_id,
            "fit_scope": self.fit_scope,
            "mean": self.mean.tolist(),
            "scales": self.scales.tolist(),
            "covariance_condition_number": float(self.covariance_condition_number),
            "target_offdiag_residual": float(self.target_offdiag_residual),
            "buffer_digest": self.buffer_digest,
        }
        return {**semantic, "state_digest": _digest(semantic)}

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> FastAlignmentState:
        if not isinstance(value, Mapping):
            raise ValueError("Fast Alignment state record must be a mapping.")
        expected = {
            "schema", "parent_state_digest", "domain_id", "fit_scope", "mean",
            "scales", "covariance_condition_number", "target_offdiag_residual",
            "buffer_digest", "state_digest",
        }
        if set(value) != expected or value.get("schema") != FAST_ALIGNMENT_SCHEMA:
            raise ValueError("Fast Alignment state record fields or schema are invalid.")
        semantic = {key: value[key] for key in expected - {"state_digest"}}
        if _digest(semantic) != value["state_digest"]:
            raise ValueError("Fast Alignment state digest does not match its semantic record.")
        return cls(
            parent_state_digest=str(value["parent_state_digest"]),
            domain_id=str(value["domain_id"]),
            fit_scope=str(value["fit_scope"]),
            mean=np.asarray(value["mean"], dtype=np.float64),
            scales=np.asarray(value["scales"], dtype=np.float64),
            covariance_condition_number=float(value["covariance_condition_number"]),
            target_offdiag_residual=float(value["target_offdiag_residual"]),
            buffer_digest=str(value["buffer_digest"]),
        )


@dataclass(frozen=True)
class AcwState:
    """Frozen AJD congruence-whitening state shared by source domains."""

    domain_order: tuple[str, ...]
    means: np.ndarray
    covariances: np.ndarray
    joint_diagonalizer: np.ndarray
    diagonal_scales: np.ndarray
    sample_counts: tuple[int, ...]
    domain_weights: tuple[float, ...]
    method: str
    regularization: float
    tolerance: float
    max_iter: int
    ajd_offdiag_residual: float
    covariance_condition_numbers: tuple[float, ...]
    max_condition_number: float = 1e12
    lift_ranks: tuple[int, ...] | None = None
    solver_name: str = "pyriemann"
    solver_version: str = "unknown"

    def __post_init__(self) -> None:
        domains = tuple(str(value).strip() for value in self.domain_order)
        if len(domains) < 2 or any(not value for value in domains) or len(set(domains)) != len(domains):
            raise ValueError("domain_order must contain at least two unique non-empty domains.")
        means = np.asarray(self.means, dtype=np.float64)
        covariances = np.asarray(self.covariances, dtype=np.float64)
        diagonalizer = np.asarray(self.joint_diagonalizer, dtype=np.float64)
        scales = np.asarray(self.diagonal_scales, dtype=np.float64)
        domain_count = len(domains)
        if means.ndim != 2 or means.shape[0] != domain_count:
            raise ValueError("means must have shape (domain_count, feature_dim).")
        feature_dim = means.shape[1]
        if covariances.shape != (domain_count, feature_dim, feature_dim):
            raise ValueError("covariances must have shape (domain_count, feature_dim, feature_dim).")
        if diagonalizer.shape != (feature_dim, feature_dim):
            raise ValueError("joint_diagonalizer must be square with feature_dim rows.")
        if scales.shape != (domain_count, feature_dim):
            raise ValueError("diagonal_scales must have shape (domain_count, feature_dim).")
        arrays = (means, covariances, diagonalizer, scales)
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("ACW state arrays must be finite.")
        if bool((scales <= 0.0).any()):
            raise ValueError("ACW diagonal scales must be positive.")
        if len(self.sample_counts) != domain_count or any(int(value) < 2 for value in self.sample_counts):
            raise ValueError("sample_counts must contain at least two rows per domain.")
        weights = tuple(float(value) for value in self.domain_weights)
        if len(weights) != domain_count or any(not np.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("domain_weights must be finite, non-negative and domain-aligned.")
        if not np.isclose(sum(weights), 1.0, atol=1e-9):
            raise ValueError("domain_weights must sum to one.")
        if self.method not in VALID_AJD_METHODS:
            raise ValueError(f"unsupported GALIA method {self.method!r}.")
        if not np.isfinite(self.regularization) or self.regularization <= 0.0:
            raise ValueError("regularization must be finite and positive.")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if isinstance(self.max_iter, bool) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not np.isfinite(self.ajd_offdiag_residual) or self.ajd_offdiag_residual < 0.0:
            raise ValueError("ajd_offdiag_residual must be finite and non-negative.")
        conditions = tuple(float(value) for value in self.covariance_condition_numbers)
        if len(conditions) != domain_count or any(
            not np.isfinite(value) or value <= 0.0 for value in conditions
        ):
            raise ValueError("covariance_condition_numbers must be finite and positive.")
        if not np.isfinite(self.max_condition_number) or self.max_condition_number <= 1.0:
            raise ValueError("max_condition_number must be finite and greater than one.")
        if any(value > self.max_condition_number for value in conditions):
            raise ValueError("covariance condition number exceeds the declared maximum.")
        if np.linalg.matrix_rank(diagonalizer, tol=self.tolerance) < feature_dim:
            raise ValueError("joint_diagonalizer must be full rank at the declared tolerance.")
        if not np.allclose(
            covariances,
            covariances.transpose(0, 2, 1),
            atol=10 * self.tolerance,
        ):
            raise ValueError("ACW covariances must be symmetric.")
        if any(float(np.linalg.eigvalsh(covariance).min()) <= 0.0 for covariance in covariances):
            raise ValueError("ACW covariances must be positive definite after regularization.")
        ranks = (
            tuple(int(value) for value in self.lift_ranks)
            if self.lift_ranks is not None
            else tuple(feature_dim for _ in domains)
        )
        if len(ranks) != domain_count or any(value < 1 or value > feature_dim for value in ranks):
            raise ValueError("lift_ranks must be positive and bounded by feature_dim.")
        solver_name = str(self.solver_name).strip()
        solver_version = str(self.solver_version).strip()
        if not solver_name or not solver_version or "\0" in solver_name + solver_version:
            raise ValueError("solver_name and solver_version must be non-empty strings.")
        object.__setattr__(self, "solver_name", solver_name)
        object.__setattr__(self, "solver_version", solver_version)
        object.__setattr__(self, "domain_order", domains)
        object.__setattr__(self, "means", _readonly(means))
        object.__setattr__(self, "covariances", _readonly(covariances))
        object.__setattr__(self, "joint_diagonalizer", _readonly(diagonalizer))
        object.__setattr__(self, "diagonal_scales", _readonly(scales))
        object.__setattr__(self, "sample_counts", tuple(int(value) for value in self.sample_counts))
        object.__setattr__(self, "domain_weights", weights)
        object.__setattr__(self, "covariance_condition_numbers", conditions)
        object.__setattr__(self, "max_condition_number", float(self.max_condition_number))
        object.__setattr__(self, "lift_ranks", ranks)

    @property
    def feature_dim(self) -> int:
        return int(self.means.shape[1])

    def _domain_index(self, domain_id: str) -> int:
        try:
            return self.domain_order.index(str(domain_id).strip())
        except ValueError as error:
            raise ValueError(f"domain {domain_id!r} is absent from the frozen GALIA state.") from error

    def transform(self, features: object, domain_id: str) -> np.ndarray:
        values = _validate_feature_matrix(features, name="features", min_samples=1)
        if values.shape[1] != self.feature_dim:
            raise ValueError(
                f"features have dimension {values.shape[1]}, expected {self.feature_dim}."
            )
        index = self._domain_index(domain_id)
        centered = values - self.means[index]
        rotated = centered @ self.joint_diagonalizer.T
        return np.ascontiguousarray(rotated / np.sqrt(self.diagonal_scales[index]))

    def fast_alignment(
        self,
        features: object,
        *,
        domain_id: str = "target",
        fit_scope: str,
        regularization: float | None = None,
    ) -> FastAlignmentState:
        """Fit target scales from an explicitly permitted calibration scope."""

        if fit_scope not in VALID_FAST_SCOPES:
            raise ValueError(
                f"fit_scope must be one of {sorted(VALID_FAST_SCOPES)}; target test is forbidden."
            )
        values = _validate_feature_matrix(features, name="fast_alignment_features", min_samples=2)
        if values.shape[1] != self.feature_dim:
            raise ValueError(
                f"fast-alignment features have dimension {values.shape[1]}, expected {self.feature_dim}."
            )
        if fit_scope == "source_weighted":
            mean = np.average(self.means, axis=0, weights=np.asarray(self.domain_weights))
            covariance = np.zeros((self.feature_dim, self.feature_dim), dtype=np.float64)
            for weight, source_mean, source_cov in zip(
                self.domain_weights,
                self.means,
                self.covariances,
                strict=True,
            ):
                delta = source_mean - mean
                covariance += weight * (source_cov + np.outer(delta, delta))
        else:
            mean = values.mean(axis=0)
            centered = values - mean
            covariance = centered.T @ centered / values.shape[0]
        epsilon = self.regularization if regularization is None else float(regularization)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("regularization must be finite and positive.")
        covariance = covariance + epsilon * np.eye(self.feature_dim)
        transformed = self.joint_diagonalizer @ covariance @ self.joint_diagonalizer.T
        scales = np.diag(transformed)
        if not np.isfinite(scales).all() or bool((scales <= 0.0).any()):
            raise ValueError("Fast Alignment produced non-positive diagonal scales.")
        condition_number = float(np.linalg.cond(covariance))
        if not np.isfinite(condition_number) or condition_number > self.max_condition_number:
            raise ValueError(
                "Fast Alignment covariance condition number exceeds the parent ACW limit."
            )
        offdiag = transformed - np.diag(scales)
        transformed_norm = float(np.linalg.norm(transformed, ord="fro"))
        target_offdiag_residual = (
            0.0
            if transformed_norm == 0.0
            else float(np.linalg.norm(offdiag, ord="fro") / transformed_norm)
        )
        return FastAlignmentState(
            parent_state_digest=self.state_digest,
            domain_id=domain_id,
            fit_scope=fit_scope,
            mean=mean,
            scales=scales,
            covariance_condition_number=condition_number,
            target_offdiag_residual=target_offdiag_residual,
            buffer_digest=_array_digest(values),
        )

    def transform_fast(self, features: object, fast_state: FastAlignmentState) -> np.ndarray:
        if fast_state.parent_state_digest != self.state_digest:
            raise ValueError("Fast Alignment state belongs to a different ACW state.")
        values = _validate_feature_matrix(features, name="features", min_samples=1)
        if values.shape[1] != self.feature_dim:
            raise ValueError(
                f"features have dimension {values.shape[1]}, expected {self.feature_dim}."
            )
        rotated = (values - fast_state.mean) @ self.joint_diagonalizer.T
        return np.ascontiguousarray(rotated / np.sqrt(fast_state.scales))

    @property
    def state_digest(self) -> str:
        return _digest(self._semantic_record())

    def _semantic_record(self) -> dict[str, object]:
        return {
            "schema": ACW_ALIGNMENT_SCHEMA,
            "domain_order": list(self.domain_order),
            "means": self.means.tolist(),
            "covariances": self.covariances.tolist(),
            "joint_diagonalizer": self.joint_diagonalizer.tolist(),
            "diagonal_scales": self.diagonal_scales.tolist(),
            "sample_counts": list(self.sample_counts),
            "domain_weights": list(self.domain_weights),
            "method": self.method,
            "regularization": self.regularization,
            "tolerance": self.tolerance,
            "max_iter": self.max_iter,
            "ajd_offdiag_residual": self.ajd_offdiag_residual,
            "covariance_condition_numbers": list(self.covariance_condition_numbers),
            "max_condition_number": self.max_condition_number,
            "lift_ranks": list(self.lift_ranks or ()),
            "solver_name": self.solver_name,
            "solver_version": self.solver_version,
        }

    def record(self) -> dict[str, object]:
        semantic = self._semantic_record()
        return {**semantic, "state_digest": self.state_digest}

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> AcwState:
        if not isinstance(value, Mapping):
            raise ValueError("ACW state record must be a mapping.")
        expected_keys = set(cls._record_keys()) | {"state_digest"}
        if set(value) != expected_keys:
            raise ValueError("ACW state record fields are invalid.")
        semantic = {key: value[key] for key in cls._record_keys()}
        expected_digest = _digest(semantic)
        if value["state_digest"] != expected_digest:
            raise ValueError("ACW state digest does not match its semantic record.")
        if semantic["schema"] != ACW_ALIGNMENT_SCHEMA:
            raise ValueError("ACW state schema is unsupported.")
        state = cls(
            domain_order=tuple(str(item) for item in semantic["domain_order"]),
            means=np.asarray(semantic["means"], dtype=np.float64),
            covariances=np.asarray(semantic["covariances"], dtype=np.float64),
            joint_diagonalizer=np.asarray(semantic["joint_diagonalizer"], dtype=np.float64),
            diagonal_scales=np.asarray(semantic["diagonal_scales"], dtype=np.float64),
            sample_counts=tuple(int(item) for item in semantic["sample_counts"]),
            domain_weights=tuple(float(item) for item in semantic["domain_weights"]),
            method=str(semantic["method"]),
            regularization=float(semantic["regularization"]),
            tolerance=float(semantic["tolerance"]),
            max_iter=int(semantic["max_iter"]),
            ajd_offdiag_residual=float(semantic["ajd_offdiag_residual"]),
            covariance_condition_numbers=tuple(
                float(item) for item in semantic["covariance_condition_numbers"]
            ),
            max_condition_number=float(semantic["max_condition_number"]),
            lift_ranks=tuple(int(item) for item in semantic["lift_ranks"]),
            solver_name=str(semantic["solver_name"]),
            solver_version=str(semantic["solver_version"]),
        )
        if state.state_digest != value["state_digest"]:
            raise ValueError("ACW state digest changed while decoding.")
        return state

    @staticmethod
    def _record_keys() -> tuple[str, ...]:
        return (
            "schema",
            "domain_order",
            "means",
            "covariances",
            "joint_diagonalizer",
            "diagonal_scales",
            "sample_counts",
            "domain_weights",
            "method",
            "regularization",
            "tolerance",
            "max_iter",
            "ajd_offdiag_residual",
            "covariance_condition_numbers",
            "max_condition_number",
            "lift_ranks",
            "solver_name",
            "solver_version",
        )


class AjdCongruenceWhitening:
    """Fit and apply a GALIA state over explicit same-dimensional feature vectors."""

    def __init__(self, config: AcwConfig | None = None) -> None:
        self.config = AcwConfig() if config is None else config
        if not isinstance(self.config, AcwConfig):
            raise TypeError("config must be a AcwConfig.")

    def fit(
        self,
        domains: Mapping[str, object],
        *,
        weights: Mapping[str, float] | None = None,
        lift_ranks: Mapping[str, int] | None = None,
    ) -> AcwState:
        order, matrices = _validate_domains(domains, min_samples=self.config.min_samples)
        domain_count = len(order)
        feature_dim = matrices[0].shape[1]
        if lift_ranks is None:
            ranks = tuple(feature_dim for _ in order)
        else:
            if set(lift_ranks) != set(order):
                raise ValueError("lift_ranks must cover exactly the frozen domain vocabulary.")
            ranks = tuple(int(lift_ranks[name]) for name in order)
            if any(value < 1 or value > feature_dim for value in ranks):
                raise ValueError("each lift rank must be in [1, feature_dim].")
        means = np.stack([matrix.mean(axis=0) for matrix in matrices])
        covariances = []
        conditions = []
        eye = np.eye(feature_dim, dtype=np.float64)
        for matrix, mean in zip(matrices, means, strict=True):
            centered = matrix - mean
            covariance = centered.T @ centered / matrix.shape[0]
            covariance = (covariance + covariance.T) / 2.0
            covariance += self.config.regularization * eye
            condition = float(np.linalg.cond(covariance))
            if not np.isfinite(condition) or condition > self.config.max_condition_number:
                raise ValueError(
                    f"domain covariance condition number {condition:g} exceeds the configured "
                    f"limit {self.config.max_condition_number:g}."
                )
            covariances.append(covariance)
            conditions.append(condition)
        covariance_stack = np.stack(covariances)
        if weights is None:
            domain_weights = np.full(domain_count, 1.0 / domain_count, dtype=np.float64)
        else:
            if self.config.method != "ajd_pham":
                raise ValueError(
                    "explicit domain weights require method='ajd_pham'; "
                    f"{self.config.method!r} ignores sample weights."
                )
            if set(weights) != set(order):
                raise ValueError("domain weights must have exactly the frozen domain vocabulary.")
            domain_weights = np.asarray([float(weights[name]) for name in order], dtype=np.float64)
            if not np.isfinite(domain_weights).all() or bool((domain_weights < 0.0).any()):
                raise ValueError("domain weights must be finite and non-negative.")
            total = float(domain_weights.sum())
            if total <= 0.0:
                raise ValueError("domain weights must have positive total mass.")
            domain_weights /= total
        joint_diagonalizer, transformed = self._fit_ajd(covariance_stack, domain_weights)
        diagonal_scales = np.diagonal(transformed, axis1=1, axis2=2)
        if not np.isfinite(diagonal_scales).all() or bool((diagonal_scales <= 0.0).any()):
            raise ValueError("AJD produced non-positive diagonal scales.")
        diagonal = np.diagonal(transformed, axis1=1, axis2=2)
        offdiag = transformed - diagonal[:, :, None] * np.eye(feature_dim, dtype=np.float64)[None]
        numerator = float(np.sum(offdiag * offdiag))
        denominator = float(np.sum(transformed * transformed))
        residual = 0.0 if denominator == 0.0 else numerator / denominator
        try:
            from importlib.metadata import PackageNotFoundError, version

            solver_version = version("pyriemann")
        except PackageNotFoundError:
            solver_version = "unknown"
        return AcwState(
            domain_order=order,
            means=means,
            covariances=covariance_stack,
            joint_diagonalizer=joint_diagonalizer,
            diagonal_scales=diagonal_scales,
            sample_counts=tuple(matrix.shape[0] for matrix in matrices),
            domain_weights=tuple(float(value) for value in domain_weights),
            method=self.config.method,
            regularization=self.config.regularization,
            tolerance=self.config.tolerance,
            max_iter=self.config.max_iter,
            ajd_offdiag_residual=residual,
            covariance_condition_numbers=tuple(conditions),
            max_condition_number=self.config.max_condition_number,
            lift_ranks=ranks,
            solver_name=f"pyriemann.{self.config.method}",
            solver_version=solver_version,
        )

    def _fit_ajd(
        self,
        covariance_stack: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            from pyriemann.geometry.ajd import ajd_pham, rjd, uwedge
        except ImportError as error:
            raise RuntimeError(
                "GALIA requires pyriemann; install the project 'baselines' extra."
            ) from error
        functions = {"ajd_pham": ajd_pham, "rjd": rjd, "uwedge": uwedge}
        function = functions[self.config.method]
        uniform = np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
        if self.config.method != "ajd_pham" and not np.allclose(weights, uniform, atol=1e-12):
            raise ValueError(
                f"AJD method {self.config.method!r} cannot consume non-uniform domain weights; "
                "use ajd_pham or declare uniform weights."
            )
        kwargs: dict[str, object] = {
            "eps": self.config.tolerance,
            "n_iter_max": self.config.max_iter,
        }
        if self.config.method == "ajd_pham":
            kwargs["sample_weight"] = weights
        try:
            diagonalizer, transformed = function(covariance_stack, **kwargs)
        except TypeError as error:
            if kwargs.get("sample_weight") is not None and not np.allclose(
                weights, uniform, atol=1e-12
            ):
                raise RuntimeError(
                    "ajd_pham rejected sample_weight; refusing to silently "
                    "downgrade a weighted AJD fit to a uniform fit."
                ) from error
            kwargs.pop("sample_weight", None)
            diagonalizer, transformed = function(covariance_stack, **kwargs)
        diagonalizer = np.asarray(diagonalizer, dtype=np.float64)
        transformed = np.asarray(transformed, dtype=np.float64)
        if diagonalizer.shape != (covariance_stack.shape[1], covariance_stack.shape[1]):
            raise ValueError("AJD solver returned an invalid diagonalizer shape.")
        if transformed.shape != covariance_stack.shape:
            raise ValueError("AJD solver returned invalid transformed covariance shapes.")
        scale = np.linalg.norm(diagonalizer, ord="fro")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("AJD solver returned a degenerate diagonalizer.")
        diagonalizer = diagonalizer * (np.sqrt(covariance_stack.shape[1]) / scale)
        transformed = np.einsum(
            "ab,dbc,ec->dae", diagonalizer, covariance_stack, diagonalizer, optimize=True
        )
        transformed = (transformed + transformed.transpose(0, 2, 1)) / 2.0
        if not np.isfinite(transformed).all():
            raise ValueError("AJD solver produced NaN or infinity.")
        return diagonalizer, transformed


__all__ = [
    "FAST_ALIGNMENT_SCHEMA",
    "ACW_ALIGNMENT_SCHEMA",
    "FastAlignmentState",
    "AjdCongruenceWhitening",
    "AcwConfig",
    "AcwState",
    "apply_feature_lift",
    "feature_lift_rank",
    "flatten_feature_epochs",
]
