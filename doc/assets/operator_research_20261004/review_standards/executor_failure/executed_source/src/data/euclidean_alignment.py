"""Numerical Euclidean alignment into dimensionless component coordinates.

This owner implements the covariance operator. Scientific fit authorization,
physical input contracts and persisted feature artifacts belong to ea_artifact.
An aligned array is never an EpochDataset or voltage at native electrodes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

EA_ALIGNMENT_SCHEMA = "n2p3_euclidean_alignment/2"
VALID_EA_SCOPES = frozenset(
    {"source_rows", "target_calibration", "unlabelled_buffer", "per_domain"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
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
    array = np.asarray(epochs)
    if (array.ndim != 3 or array.shape[0] < min_samples or min(array.shape[1:]) < 1
            or not np.issubdtype(array.dtype, np.floating)):
        raise ValueError(f"{name} must be a floating array with shape (N,C,T), got {array.shape}.")
    for start in range(0, len(array), 4096):
        if not np.isfinite(array[start:start + 4096]).all():
            raise ValueError(f"{name} contains NaN or infinity.")
    return array


@dataclass(frozen=True)
class EuclideanAlignmentConfig:
    shrinkage: float = 0.0
    relative_eigenvalue_floor: float = 1e-6
    max_condition_number: float = 1e12
    min_samples: int = 2

    def __post_init__(self) -> None:
        for name in ("shrinkage", "relative_eigenvalue_floor", "max_condition_number"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
        if not 0 <= self.shrinkage <= 1:
            raise ValueError("shrinkage must be in [0,1].")
        if not 0 < self.relative_eigenvalue_floor < 1:
            raise ValueError("relative_eigenvalue_floor must be in (0,1).")
        if self.max_condition_number <= 1:
            raise ValueError("max_condition_number must be greater than one.")
        if type(self.min_samples) is not int or self.min_samples < 2:
            raise ValueError("min_samples must be an integer of at least two.")


def _reference_operator(covariance: np.ndarray, config: EuclideanAlignmentConfig):
    """Regularize relative to trace/C, without introducing a voltage-scale constant."""
    scale = float(np.trace(covariance) / len(covariance))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("EA requires positive finite observed signal energy.")
    normalized = covariance / scale
    empirical_eigenvalues = np.linalg.eigvalsh(normalized)
    tolerance = np.finfo(float).eps * len(covariance) * max(float(np.abs(normalized).max()), 1.) * 32
    if empirical_eigenvalues.min() < -tolerance:
        raise ValueError("EA empirical second moment must be positive semidefinite.")
    shrunk = (1 - config.shrinkage) * normalized + config.shrinkage * np.eye(len(covariance))
    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    regularized = np.maximum(eigenvalues, config.relative_eigenvalue_floor)
    condition = float(regularized.max() / regularized.min())
    if not np.isfinite(condition) or condition > config.max_condition_number:
        raise ValueError(
            f"EA reference covariance condition number {condition:g} exceeds limit "
            f"{config.max_condition_number:g}."
        )
    reference = (eigenvectors * (regularized * scale)) @ eigenvectors.T
    inverse = (eigenvectors / np.sqrt(regularized * scale)) @ eigenvectors.T
    rank = int(np.sum(empirical_eigenvalues > config.relative_eigenvalue_floor))
    return reference, inverse, condition, rank


@dataclass(frozen=True)
class EuclideanAlignmentState:
    domain_id: str | None
    fit_scope: str
    mean_trial_covariance: np.ndarray
    reference_covariance: np.ndarray
    inverse_sqrt_reference: np.ndarray
    sample_count: int
    shrinkage: float
    relative_eigenvalue_floor: float
    covariance_condition_number: float
    effective_rank: int
    buffer_digest: str

    def __post_init__(self) -> None:
        if self.fit_scope not in VALID_EA_SCOPES:
            raise ValueError(f"fit_scope must be one of {sorted(VALID_EA_SCOPES)}.")
        if self.domain_id is not None and (not isinstance(self.domain_id, str) or not self.domain_id.strip()):
            raise ValueError("domain_id must be nonempty text when supplied.")
        if self.fit_scope == "per_domain" and self.domain_id is None:
            raise ValueError("per_domain EA states require a non-empty domain_id.")
        config = EuclideanAlignmentConfig(
            shrinkage=self.shrinkage, relative_eigenvalue_floor=self.relative_eigenvalue_floor,
        )
        covariance = np.asarray(self.mean_trial_covariance, dtype=np.float64)
        reference = np.asarray(self.reference_covariance, dtype=np.float64)
        inverse = np.asarray(self.inverse_sqrt_reference, dtype=np.float64)
        if (covariance.ndim != 2 or covariance.shape[0] < 1 or covariance.shape[0] != covariance.shape[1]
                or reference.shape != covariance.shape or inverse.shape != covariance.shape):
            raise ValueError("EA covariance/inverse matrices must be square and aligned.")
        if not all(np.isfinite(value).all() for value in (covariance, reference, inverse)):
            raise ValueError("EA state matrices must be finite.")
        scale = max(float(np.abs(covariance).max()), np.finfo(float).tiny)
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=scale * 1e-12):
            raise ValueError("EA mean trial covariance must be symmetric.")
        expected_reference, expected_inverse, condition, rank = _reference_operator(covariance, config)
        if (not np.allclose(reference, expected_reference, rtol=1e-9, atol=scale * 1e-11)
                or not np.allclose(inverse, expected_inverse, rtol=1e-9, atol=np.abs(expected_inverse).max() * 1e-11)):
            raise ValueError("EA state matrices disagree with the declared relative operator.")
        if type(self.sample_count) is not int or self.sample_count < 2:
            raise ValueError("sample_count must be an integer of at least two.")
        if type(self.effective_rank) is not int or self.effective_rank != rank:
            raise ValueError("EA effective_rank disagrees with its empirical second moment.")
        if (isinstance(self.covariance_condition_number, (bool, np.bool_))
                or not isinstance(self.covariance_condition_number, Real)
                or not np.isclose(self.covariance_condition_number, condition, rtol=1e-9, atol=0)):
            raise ValueError("EA condition number disagrees with its regularized reference.")
        if not isinstance(self.buffer_digest, str) or not _SHA256.fullmatch(self.buffer_digest):
            raise ValueError("buffer_digest must be a lowercase SHA-256 digest.")
        object.__setattr__(self, "domain_id", self.domain_id.strip() if self.domain_id else None)
        object.__setattr__(self, "mean_trial_covariance", _readonly(covariance))
        object.__setattr__(self, "reference_covariance", _readonly(reference))
        object.__setattr__(self, "inverse_sqrt_reference", _readonly(inverse))

    def record(self) -> dict[str, object]:
        semantic = {
            "schema": EA_ALIGNMENT_SCHEMA, "domain_id": self.domain_id, "fit_scope": self.fit_scope,
            "mean_trial_covariance": self.mean_trial_covariance.tolist(),
            "reference_covariance": self.reference_covariance.tolist(),
            "inverse_sqrt_reference": self.inverse_sqrt_reference.tolist(),
            "sample_count": self.sample_count, "shrinkage": float(self.shrinkage),
            "relative_eigenvalue_floor": float(self.relative_eigenvalue_floor),
            "covariance_condition_number": float(self.covariance_condition_number),
            "effective_rank": self.effective_rank, "buffer_digest": self.buffer_digest,
            "output_unit": "dimensionless", "regularization_policy": "eigenvalue_floor_times_empirical_trace_per_channel",
        }
        return {**semantic, "state_digest": _digest(semantic)}

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> EuclideanAlignmentState:
        fields = set(cls.__dataclass_fields__)
        expected = fields | {"schema", "state_digest", "output_unit", "regularization_policy"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("EA alignment record fields are invalid; legacy absolute-floor states are not admitted.")
        if (value.get("schema") != EA_ALIGNMENT_SCHEMA or value.get("output_unit") != "dimensionless"
                or value.get("regularization_policy") != "eigenvalue_floor_times_empirical_trace_per_channel"):
            raise ValueError("EA alignment schema or coordinate contract is unsupported.")
        semantic = {key: value[key] for key in expected - {"state_digest"}}
        if _digest(semantic) != value["state_digest"]:
            raise ValueError("EA state digest does not match its semantic record.")
        return cls(**{name: value[name] for name in fields})


def fit_euclidean_alignment(
    epochs: np.ndarray, *, fit_scope: str, domain_id: str | None = None,
    config: EuclideanAlignmentConfig | None = None,
) -> EuclideanAlignmentState:
    """Fit a numerical operator; ea_artifact binds scientific fit authorization."""
    if fit_scope not in VALID_EA_SCOPES:
        raise ValueError(f"fit_scope must be one of {sorted(VALID_EA_SCOPES)}.")
    if fit_scope == "per_domain" and not str(domain_id or "").strip():
        raise ValueError("per_domain scope requires a non-empty domain_id.")
    cfg = EuclideanAlignmentConfig() if config is None else config
    values = _validate_epochs(epochs, name="epochs", min_samples=cfg.min_samples)
    covariance = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    block_rows = max(1, 16 * 1024**2 // (8 * values.shape[1] * values.shape[2]))
    for start in range(0, len(values), block_rows):
        block = np.asarray(values[start:start + block_rows], dtype=np.float64)
        covariance += np.einsum("nct,nut->cu", block, block)
    covariance /= len(values) * values.shape[2]
    reference, inverse, condition, rank = _reference_operator(covariance, cfg)
    return EuclideanAlignmentState(
        domain_id=domain_id, fit_scope=fit_scope, mean_trial_covariance=covariance,
        reference_covariance=reference, inverse_sqrt_reference=inverse, sample_count=len(values),
        shrinkage=cfg.shrinkage, relative_eigenvalue_floor=cfg.relative_eigenvalue_floor,
        covariance_condition_number=condition, effective_rank=rank, buffer_digest=_array_digest(values),
    )


def fit_per_domain_euclidean_alignment(
    epochs: np.ndarray, domain_ids: np.ndarray, *, config: EuclideanAlignmentConfig | None = None,
) -> dict[str, EuclideanAlignmentState]:
    cfg = EuclideanAlignmentConfig() if config is None else config
    values = _validate_epochs(epochs, name="epochs", min_samples=cfg.min_samples)
    domains = np.asarray(domain_ids)
    if (domains.dtype.kind not in {"U", "S"} or domains.shape != (len(values),)
            or np.any(np.char.strip(domains.astype(str)) == "")):
        raise ValueError("domain_ids must be non-empty strings aligned with epochs.")
    return {
        domain: fit_euclidean_alignment(values[domains == domain], fit_scope="per_domain", domain_id=domain, config=cfg)
        for domain in sorted(np.unique(domains.astype(str)).tolist())
    }


def transform_euclidean_alignment(epochs: np.ndarray, state: EuclideanAlignmentState) -> np.ndarray:
    """Return dimensionless component values, never physical electrode voltages."""
    if not isinstance(state, EuclideanAlignmentState):
        raise TypeError("EA transform requires a validated EuclideanAlignmentState.")
    values = _validate_epochs(epochs, name="epochs", min_samples=1)
    if values.shape[1] != state.inverse_sqrt_reference.shape[0]:
        raise ValueError("epoch channels disagree with the EA reference.")
    output = np.empty(values.shape, dtype=np.float32)
    for start in range(0, len(values), 4096):
        output[start:start + 4096] = np.einsum(
            "cp,npt->nct", state.inverse_sqrt_reference, values[start:start + 4096], optimize=True,
        )
    if not np.isfinite(output).all():
        raise ValueError("EA output contains non-finite component values.")
    return output


__all__ = [
    "EA_ALIGNMENT_SCHEMA", "VALID_EA_SCOPES", "EuclideanAlignmentConfig", "EuclideanAlignmentState",
    "fit_euclidean_alignment", "fit_per_domain_euclidean_alignment", "transform_euclidean_alignment",
]
