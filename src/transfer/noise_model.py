"""Fit and propagate joint raw-residual / boundary-state uncertainty.

The caller supplies paired draws from declared training support. This module
does not infer unobserved filter states or certify physical data authority.
It estimates the joint covariance once; separate marginal additions would lose
Omega or count the same fluctuation twice.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class _FrozenMetadata(dict):
    """JSON-compatible deeply frozen provenance (copies are safe shared views)."""

    def _reject(self, *args, **kwargs):
        raise TypeError("fitted metadata is frozen")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _reject

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def freeze_metadata(value):
    if isinstance(value, dict):
        return _FrozenMetadata({key: freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_metadata(item) for item in value)
    return value


def require_sha256(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def require_identifiers(values, name):
    if (not isinstance(values, (tuple, list)) or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{name} must be unique nonempty textual identities")


def metadata_digest(metadata):
    """Bind all serialized numerical/configuration state, not just arrays."""
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def frozen_array(value):
    """Copy into immutable bytes so setflags(write=True) cannot unfreeze state."""
    array = np.ascontiguousarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def validate_covariance(value, *, name):
    covariance = np.asarray(value, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite")
    tolerance = max(np.linalg.norm(covariance, ord=2), np.finfo(float).tiny) * 1e-10
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=tolerance):
        raise ValueError(f"{name} must be symmetric")
    if np.linalg.eigvalsh(covariance).min() < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return (covariance + covariance.T) / 2


@dataclass(frozen=True)
class PropagatedNoise:
    mean: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class JointNoiseModel:
    mean: np.ndarray
    covariance: np.ndarray
    raw_dimension: int
    provenance: dict
    _integrity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=float)
        covariance = validate_covariance(self.covariance, name="joint noise covariance")
        if mean.ndim != 1 or covariance.shape != (len(mean), len(mean)):
            raise ValueError("noise mean and covariance dimension mismatch")
        if (not np.all(np.isfinite(mean)) or type(self.raw_dimension) is not int
                or not 0 < self.raw_dimension <= len(mean)):
            raise ValueError("invalid raw noise dimension or mean")
        object.__setattr__(self, "mean", frozen_array(mean))
        object.__setattr__(self, "covariance", frozen_array(covariance))
        provenance = json.loads(json.dumps(self.provenance, allow_nan=False))
        required = {"fit_scope", "scope_id", "draw_ids", "n_draws", "input_sha256", "shrinkage",
                    "implementation_sha256", "estimator"}
        if (not isinstance(provenance, dict) or set(provenance) != required
                or provenance["fit_scope"] not in {"source_training", "target_calibration"}
                or not isinstance(provenance["scope_id"], str) or not provenance["scope_id"].strip()
                or provenance["estimator"] != "joint_standardized_covariance_with_cross_terms"):
            raise ValueError("joint noise requires its complete immutable training scope ledger")
        require_identifiers(provenance["draw_ids"], "noise draw IDs")
        if type(provenance["n_draws"]) is not int or provenance["n_draws"] != len(provenance["draw_ids"]) or provenance["n_draws"] < 3:
            raise ValueError("noise draw count disagrees with its fit ledger")
        require_sha256(provenance["input_sha256"], "noise fit input")
        require_sha256(provenance["implementation_sha256"], "noise implementation")
        if (isinstance(provenance["shrinkage"], bool) or not np.isfinite(provenance["shrinkage"])
                or not 0 <= provenance["shrinkage"] <= 1):
            raise ValueError("noise shrinkage must be finite in [0, 1]")
        object.__setattr__(self, "provenance", freeze_metadata(provenance))
        object.__setattr__(self, "_integrity_sha256", metadata_digest(self._metadata()))

    @classmethod
    def fit(cls, raw_residual_draws, state_draws, *, fit_scope, scope_id, draw_ids,
            shrinkage=None):
        """Estimate covariance from paired rows, not from oracle target states.

        Scope declarations and unique physical draw IDs are mandatory. They are
        an audit record, not a substitute for the caller's split authority.
        Shrinkage acts in standardized joint coordinates, preserving different
        physical units of raw residuals and filter states.
        """
        if fit_scope not in {"source_training", "target_calibration"} or not isinstance(scope_id, str) or not scope_id.strip():
            raise ValueError("noise fit scope must be explicit training or calibration")
        raw = np.asarray(raw_residual_draws, dtype=float)
        state = np.asarray(state_draws, dtype=float)
        if raw.ndim != 2 or state.ndim != 2 or raw.shape[0] != state.shape[0]:
            raise ValueError("raw and state draws must be paired two-dimensional arrays")
        if len(raw) < 3 or raw.shape[1] < 1:
            raise ValueError("at least three paired noise draws are required")
        ids = tuple(draw_ids)
        require_identifiers(ids, "noise draw IDs")
        if len(ids) != len(raw) or len(set(ids)) != len(ids) or any(not v for v in ids):
            raise ValueError("noise draws require unique physical draw IDs")
        joint = np.column_stack([raw, state])
        if not np.all(np.isfinite(joint)):
            raise ValueError("noise draws must be finite")
        mean = joint.mean(axis=0)
        centered = joint - mean
        scale = np.sqrt(np.mean(centered**2, axis=0))
        observed = scale > 0
        safe_scale = np.where(observed, scale, 1)
        standardized = centered / safe_scale
        if shrinkage is None:
            covariance = np.zeros((joint.shape[1], joint.shape[1]))
            actual_shrinkage = 0.
            if np.any(observed):
                estimator = LedoitWolf(assume_centered=True).fit(standardized[:, observed])
                covariance[np.ix_(observed, observed)] = estimator.covariance_
                actual_shrinkage = float(estimator.shrinkage_)
        else:
            if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
                raise ValueError("shrinkage must be in [0, 1]")
            empirical = standardized.T @ standardized / len(raw)
            covariance = (1 - shrinkage) * empirical + shrinkage * np.diag(observed)
            actual_shrinkage = float(shrinkage)
        # Constant dimensions remain exactly constant rather than fabricated noise.
        covariance *= np.outer(scale, scale)
        return cls(mean, covariance, raw.shape[1], {
            "fit_scope": fit_scope, "scope_id": str(scope_id), "draw_ids": list(ids),
            "n_draws": len(raw), "input_sha256": array_digest(raw, state),
            "shrinkage": actual_shrinkage,
            "implementation_sha256": _IMPLEMENTATION_SHA256,
            "estimator": "joint_standardized_covariance_with_cross_terms",
        })

    def _metadata(self):
        return {"schema": "n2p3_joint_noise/1", "raw_dimension": self.raw_dimension,
                "provenance": self.provenance,
                "state_sha256": array_digest(self.mean, self.covariance)}

    def assert_integrity(self):
        if metadata_digest(self._metadata()) != self._integrity_sha256:
            raise ValueError("frozen joint noise integrity changed")

    def project(self, raw_operator, state_operator):
        """Push forward the *joint* model, including all raw/state cross terms.

        Stack epoch operators by rows to retain overlap and common state effects.
        There is no independence assumption between epochs or raw and state.
        """
        self.assert_integrity()
        A = np.asarray(raw_operator, dtype=float)
        B = np.asarray(state_operator, dtype=float)
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[0]:
            raise ValueError("operators must share the output sample dimension")
        if A.shape[1] != self.raw_dimension or B.shape[1] != len(self.mean) - self.raw_dimension:
            raise ValueError("operator dimensions do not match fitted noise coordinates")
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
            raise ValueError("operators must be finite")
        operator = np.column_stack([A, B])
        cov = operator @ self.covariance @ operator.T
        return PropagatedNoise(frozen_array(operator @ self.mean),
                               frozen_array((cov + cov.T) / 2))

    def save(self, path):
        self.assert_integrity()
        metadata = self._metadata()
        metadata["artifact_sha256"] = metadata_digest(metadata)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, mean=self.mean, covariance=self.covariance,
                                metadata=np.asarray(json.dumps(metadata, sort_keys=True)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            mean, covariance = archive["mean"], archive["covariance"]
        if metadata.get("schema") != "n2p3_joint_noise/1":
            raise ValueError("unsupported joint noise schema")
        expected = metadata.pop("artifact_sha256", None)
        if expected != metadata_digest(metadata):
            raise ValueError("joint noise metadata digest mismatch")
        if metadata.get("state_sha256") != array_digest(mean, covariance):
            raise ValueError("joint noise state digest mismatch")
        return cls(mean, covariance, metadata["raw_dimension"], metadata["provenance"])
