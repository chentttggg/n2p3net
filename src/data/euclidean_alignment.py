"""Euclidean Alignment (EA) for shape-preserving EEG trial alignment.

Implements the primary-source form ``x_tilde = Rbar^{-1/2} x`` (He & Wu
2020, Eq. 10-11; Wu 2025, Eq. 1-3).  The transform keeps ``(N,C,T)`` shape,
so it can feed the N2P3 trunk directly.  No target-test statistics may be
used; the declared fit scope is frozen in the state record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

EA_ALIGNMENT_SCHEMA = "n2p3_euclidean_alignment/1"
VALID_EA_SCOPES = frozenset(
    {"source_rows", "target_calibration", "unlabelled_buffer", "per_domain"}
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _validate_epochs(epochs: np.ndarray, *, name: str, min_samples: int) -> np.ndarray:
    array = np.asarray(epochs, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] < min_samples or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"{name} must have shape (N,C,T), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return array


@dataclass(frozen=True)
class EuclideanAlignmentConfig:
    shrinkage: float = 0.0
    regularization: float = 1e-6
    max_condition_number: float = 1e12
    min_samples: int = 2

    def __post_init__(self) -> None:
        if not np.isfinite(self.shrinkage) or not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must be finite and in [0,1].")
        if not np.isfinite(self.regularization) or self.regularization <= 0.0:
            raise ValueError("regularization must be finite and positive.")
        if not np.isfinite(self.max_condition_number) or self.max_condition_number <= 1.0:
            raise ValueError("max_condition_number must be finite and greater than one.")
        if isinstance(self.min_samples, bool) or self.min_samples < 2:
            raise ValueError("min_samples must be at least two.")


@dataclass(frozen=True)
class EuclideanAlignmentState:
    domain_id: str | None
    fit_scope: str
    mean_trial_covariance: np.ndarray
    inverse_sqrt_reference: np.ndarray
    sample_count: int
    shrinkage: float
    regularization: float
    covariance_condition_number: float
    buffer_digest: str

    def __post_init__(self) -> None:
        if self.fit_scope not in VALID_EA_SCOPES:
            raise ValueError(f"fit_scope must be one of {sorted(VALID_EA_SCOPES)}.")
        if self.fit_scope == "per_domain" and not str(self.domain_id or "").strip():
            raise ValueError("per_domain EA states require a non-empty domain_id.")
        covariance = np.asarray(self.mean_trial_covariance, dtype=np.float64)
        inverse = np.asarray(self.inverse_sqrt_reference, dtype=np.float64)
        shape = covariance.shape[0]
        if covariance.shape != (shape, shape) or inverse.shape != (shape, shape):
            raise ValueError("EA covariance/inverse matrices must be square and aligned.")
        if not np.isfinite(covariance).all() or not np.isfinite(inverse).all():
            raise ValueError("EA state matrices must be finite.")
        if not np.allclose(covariance, covariance.T, atol=1e-9):
            raise ValueError("EA mean trial covariance must be symmetric.")
        if isinstance(self.sample_count, bool) or self.sample_count < 2:
            raise ValueError("sample_count must be at least two.")
        if not np.isfinite(self.covariance_condition_number) or self.covariance_condition_number <= 0:
            raise ValueError("covariance_condition_number must be finite and positive.")
        if not isinstance(self.buffer_digest, str) or len(self.buffer_digest) != 64:
            raise ValueError("buffer_digest must be a SHA-256 digest.")
        object.__setattr__(self, "domain_id", self.domain_id.strip() if self.domain_id else None)
        object.__setattr__(self, "mean_trial_covariance", _readonly(covariance))
        object.__setattr__(self, "inverse_sqrt_reference", _readonly(inverse))

    def record(self) -> dict[str, object]:
        semantic = {
            "schema": EA_ALIGNMENT_SCHEMA,
            "domain_id": self.domain_id,
            "fit_scope": self.fit_scope,
            "mean_trial_covariance": self.mean_trial_covariance.tolist(),
            "inverse_sqrt_reference": self.inverse_sqrt_reference.tolist(),
            "sample_count": int(self.sample_count),
            "shrinkage": float(self.shrinkage),
            "regularization": float(self.regularization),
            "covariance_condition_number": float(self.covariance_condition_number),
            "buffer_digest": self.buffer_digest,
        }
        return {**semantic, "state_digest": _digest(semantic)}

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> EuclideanAlignmentState:
        expected = {
            "schema",
            "domain_id",
            "fit_scope",
            "mean_trial_covariance",
            "inverse_sqrt_reference",
            "sample_count",
            "shrinkage",
            "regularization",
            "covariance_condition_number",
            "buffer_digest",
            "state_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("EA alignment record fields are invalid.")
        if value.get("schema") != EA_ALIGNMENT_SCHEMA:
            raise ValueError("EA alignment schema is unsupported.")
        semantic = {key: value[key] for key in expected - {"state_digest"}}
        if _digest(semantic) != value["state_digest"]:
            raise ValueError("EA state digest does not match its semantic record.")
        return cls(
            domain_id=None if semantic["domain_id"] is None else str(semantic["domain_id"]),
            fit_scope=str(semantic["fit_scope"]),
            mean_trial_covariance=np.asarray(semantic["mean_trial_covariance"], dtype=np.float64),
            inverse_sqrt_reference=np.asarray(
                semantic["inverse_sqrt_reference"], dtype=np.float64
            ),
            sample_count=int(semantic["sample_count"]),
            shrinkage=float(semantic["shrinkage"]),
            regularization=float(semantic["regularization"]),
            covariance_condition_number=float(semantic["covariance_condition_number"]),
            buffer_digest=str(semantic["buffer_digest"]),
        )


def _inverse_sqrt_spd(
    covariance: np.ndarray,
    *,
    regularization: float,
    max_condition_number: float,
) -> tuple[np.ndarray, float]:
    dim = covariance.shape[0]
    eye = np.eye(dim, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues.min()), 0.0)
    if floor < regularization:
        covariance = covariance + (regularization - floor) * eye
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    condition = float(eigenvalues.max() / max(eigenvalues.min(), np.finfo(np.float64).tiny))
    if not np.isfinite(condition) or condition > max_condition_number:
        raise ValueError(
            f"EA reference covariance condition number {condition:g} exceeds limit "
            f"{max_condition_number:g}."
        )
    inverse_sqrt = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    return inverse_sqrt, condition


def fit_euclidean_alignment(
    epochs: np.ndarray,
    *,
    fit_scope: str,
    domain_id: str | None = None,
    config: EuclideanAlignmentConfig | None = None,
) -> EuclideanAlignmentState:
    """Fit one EA reference from a declared scope."""
    if fit_scope not in VALID_EA_SCOPES:
        raise ValueError(f"fit_scope must be one of {sorted(VALID_EA_SCOPES)}.")
    if fit_scope == "per_domain" and not str(domain_id or "").strip():
        raise ValueError("per_domain scope requires a non-empty domain_id.")
    cfg = EuclideanAlignmentConfig() if config is None else config
    values = _validate_epochs(epochs, name="epochs", min_samples=cfg.min_samples)
    # EA primary form: average trial covariance, no mean subtraction.
    trial_covariances = np.einsum("nct,nut->ncu", values, values) / values.shape[2]
    mean_covariance = trial_covariances.mean(axis=0)
    if cfg.shrinkage:
        trace = np.trace(mean_covariance) / mean_covariance.shape[0]
        mean_covariance = (1.0 - cfg.shrinkage) * mean_covariance + cfg.shrinkage * trace * np.eye(
            mean_covariance.shape[0]
        )
    inverse_sqrt, condition = _inverse_sqrt_spd(
        mean_covariance,
        regularization=cfg.regularization,
        max_condition_number=cfg.max_condition_number,
    )
    return EuclideanAlignmentState(
        domain_id=domain_id,
        fit_scope=fit_scope,
        mean_trial_covariance=mean_covariance,
        inverse_sqrt_reference=inverse_sqrt,
        sample_count=values.shape[0],
        shrinkage=cfg.shrinkage,
        regularization=cfg.regularization,
        covariance_condition_number=condition,
        buffer_digest=_array_digest(values),
    )


def fit_per_domain_euclidean_alignment(
    epochs: np.ndarray,
    domain_ids: np.ndarray,
    *,
    config: EuclideanAlignmentConfig | None = None,
) -> dict[str, EuclideanAlignmentState]:
    cfg = EuclideanAlignmentConfig() if config is None else config
    values = _validate_epochs(epochs, name="epochs", min_samples=cfg.min_samples)
    domains = np.asarray(domain_ids).astype(str)
    if domains.ndim != 1 or len(domains) != len(values) or bool((domains == "").any()):
        raise ValueError("domain_ids must be non-empty strings aligned with epochs.")
    states: dict[str, EuclideanAlignmentState] = {}
    for domain in sorted(np.unique(domains).tolist()):
        rows = domains == domain
        states[domain] = fit_euclidean_alignment(
            values[rows],
            fit_scope="per_domain",
            domain_id=domain,
            config=cfg,
        )
    return states


def transform_euclidean_alignment(
    epochs: np.ndarray,
    state: EuclideanAlignmentState,
) -> np.ndarray:
    values = _validate_epochs(epochs, name="epochs", min_samples=1)
    if values.shape[1] != state.inverse_sqrt_reference.shape[0]:
        raise ValueError("epoch channels disagree with the EA reference.")
    transformed = np.einsum("cp,npt->nct", state.inverse_sqrt_reference, values, optimize=True)
    return np.ascontiguousarray(transformed, dtype=np.float32)


__all__ = [
    "EA_ALIGNMENT_SCHEMA",
    "VALID_EA_SCOPES",
    "EuclideanAlignmentConfig",
    "EuclideanAlignmentState",
    "fit_euclidean_alignment",
    "fit_per_domain_euclidean_alignment",
    "transform_euclidean_alignment",
]
