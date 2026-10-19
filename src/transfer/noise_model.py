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


def finite_real_array(value, *, name):
    """Normalize external numerical data once without lossy complex coercion."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain finite real numbers")
    array = np.asarray(raw, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array


def validate_covariance(value, *, name):
    covariance = finite_real_array(value, name=name)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be square")
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
class ConditionedNoise:
    mean: np.ndarray
    covariance: np.ndarray
    rank_ledger: dict


@dataclass(frozen=True)
class ConditionalProjection:
    mean_gain: np.ndarray
    covariance: np.ndarray
    rank_ledger: dict


def _support_residual(centered, scales, basis, *, name):
    standardized = centered / scales
    residual = standardized - (standardized @ basis) @ basis.T
    relative = 64*np.finfo(float).eps*len(scales)
    if np.any(np.linalg.norm(residual, axis=-1) > relative*np.linalg.norm(standardized, axis=-1)):
        raise ValueError(f"{name} outside identified conditioning support")


def _conditional_factors(covariance, raw_dimension):
    """Solve on the standardized raw support, with an explicit roundoff ledger.

    Scaling coordinates changes neither their physical support nor the
    conditional distribution. The retained inverse is a solve on that support;
    observations and Omega must belong to it before it can be used.
    """
    r = raw_dimension
    sx, omega, gamma = covariance[:r, :r], covariance[:r, r:], covariance[r:, r:]
    raw_scale = np.sqrt(np.diag(sx))
    fallback = float(raw_scale.max()) or 1.
    raw_scale = np.where(raw_scale > 0, raw_scale, fallback)
    raw_correlation = sx / np.outer(raw_scale, raw_scale)
    values, vectors = np.linalg.eigh(raw_correlation)
    threshold = np.finfo(float).eps*r*max(float(values[-1]), 0.)
    if values[0] < -32*max(threshold, np.finfo(float).tiny):
        raise ValueError("raw covariance is not PSD at its conditioning resolution")
    keep = values > threshold
    basis = vectors[:, keep]
    s = len(gamma)
    state_scale = np.sqrt(np.diag(gamma))
    state_scale = np.where(state_scale > 0, state_scale, float(state_scale.max(initial=0)) or 1.)
    cross = omega / raw_scale[:, None] / state_scale[None, :]
    _support_residual(cross.T, np.ones(r), basis, name="raw/state cross covariance")
    gain_standard = ((cross.T @ basis) / values[keep]) @ basis.T
    gain = gain_standard * state_scale[:, None] / raw_scale[None, :]
    if s:
        state_correlation = gamma / np.outer(state_scale, state_scale)
        correction = gain_standard @ cross
        schur = state_correlation-correction
        schur = (schur+schur.T)/2
        reference = max(float(np.linalg.norm(state_correlation, ord=2)), float(np.linalg.norm(correction, ord=2)))
        resolution = 32*np.finfo(float).eps*(r+s)*reference
        eigen, modes = np.linalg.eigh(schur)
        if eigen[0] < -resolution:
            raise ValueError("joint covariance has an invalid conditional Schur complement")
        identified = eigen > resolution
        conditional = ((modes[:, identified]*eigen[identified]) @ modes[:, identified].T)*np.outer(state_scale, state_scale)
    else:
        eigen, identified, resolution = np.empty(0), np.empty(0, dtype=bool), 0.
        conditional = np.empty((0, 0))
    ledger = {"method": "standardized_raw_identified_support_eigensolve",
              "raw_dimension": r, "raw_rank": int(keep.sum()),
              "raw_correlation_eigenvalues": values.tolist(), "raw_eigenvalue_threshold": float(threshold),
              "raw_support_relative_tolerance": float(64*np.finfo(float).eps*r),
              "conditional_state_rank": int(identified.sum()),
              "conditional_state_correlation_eigenvalues": eigen.tolist(),
              "conditional_roundoff_threshold": float(resolution),
              "roundoff_modes_removed": int((~identified).sum()), "statistical_identifiability_claimed": False}
    return raw_scale, basis, gain, conditional, ledger


@dataclass(frozen=True)
class JointNoiseModel:
    mean: np.ndarray
    covariance: np.ndarray
    raw_dimension: int
    provenance: dict
    conditioning_gain: np.ndarray = field(init=False, repr=False, compare=False)
    conditional_state_covariance: np.ndarray = field(init=False, repr=False, compare=False)
    conditioning_rank_ledger: dict = field(init=False, repr=False, compare=False)
    _raw_scales: np.ndarray = field(init=False, repr=False, compare=False)
    _raw_support_basis: np.ndarray = field(init=False, repr=False, compare=False)
    _integrity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        mean = finite_real_array(self.mean, name="joint noise mean")
        covariance = validate_covariance(self.covariance, name="joint noise covariance")
        if mean.ndim != 1 or covariance.shape != (len(mean), len(mean)):
            raise ValueError("noise mean and covariance dimension mismatch")
        if (type(self.raw_dimension) is not int
                or not 0 < self.raw_dimension <= len(mean)):
            raise ValueError("invalid raw noise dimension or mean")
        object.__setattr__(self, "mean", frozen_array(mean))
        object.__setattr__(self, "covariance", frozen_array(covariance))
        provenance = json.loads(json.dumps(self.provenance, allow_nan=False))
        required = {"fit_scope", "scope_id", "draw_ids", "n_draws", "input_sha256", "shrinkage",
                    "implementation_sha256", "estimator"}
        if (not isinstance(provenance, dict) or set(provenance) not in (required, required | {"empirical_resolution"})
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
        empirical = provenance.get("empirical_resolution")
        if empirical is not None and (
                empirical.get("joint_dimension") != len(mean)
                or empirical.get("n_draws") != provenance["n_draws"]
                or empirical.get("centered_rank_upper_bound")
                    != min(provenance["n_draws"]-1, empirical.get("active_dimensions", -1))
                or not 0 <= empirical.get("active_dimensions", -1) <= len(mean)):
            raise ValueError("noise empirical resolution differs from the actual paired fit dimensions")
        object.__setattr__(self, "provenance", freeze_metadata(provenance))
        scales, basis, gain, conditional, ledger = _conditional_factors(covariance, self.raw_dimension)
        object.__setattr__(self, "_raw_scales", frozen_array(scales))
        object.__setattr__(self, "_raw_support_basis", frozen_array(basis))
        object.__setattr__(self, "conditioning_gain", frozen_array(gain))
        object.__setattr__(self, "conditional_state_covariance", frozen_array(conditional))
        object.__setattr__(self, "conditioning_rank_ledger", freeze_metadata(ledger))
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
        raw = finite_real_array(raw_residual_draws, name="paired raw residual draws")
        state = finite_real_array(state_draws, name="paired state draws")
        if raw.ndim != 2 or state.ndim != 2 or raw.shape[0] != state.shape[0]:
            raise ValueError("raw and state draws must be paired two-dimensional arrays")
        if len(raw) < 3 or raw.shape[1] < 1:
            raise ValueError("at least three paired noise draws are required")
        ids = tuple(draw_ids)
        require_identifiers(ids, "noise draw IDs")
        if len(ids) != len(raw) or len(set(ids)) != len(ids) or any(not v for v in ids):
            raise ValueError("noise draws require unique physical draw IDs")
        joint = np.column_stack([raw, state])
        mean = joint.mean(axis=0)
        centered = joint - mean
        scale = np.sqrt(np.mean(centered**2, axis=0))
        observed = scale > 0
        safe_scale = np.where(observed, scale, 1)
        standardized = centered / safe_scale
        empirical_resolution = {
            "n_draws": len(raw), "joint_dimension": joint.shape[1],
            "active_dimensions": int(observed.sum()),
            "centered_rank_upper_bound": min(len(raw)-1, int(observed.sum())),
        }
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
            "empirical_resolution": empirical_resolution,
        })

    def estimation_diagnostics(self):
        """Sample/rank disclosure, not a certificate of covariance accuracy.

        Shrinkage defines a usable fitted covariance even when centered draws
        do not span all active coordinates. A numerical support or a shrinkage
        coefficient does not establish statistical identification or fit quality.
        Old producers retain their scope and do not acquire an invented rank.
        """
        self.assert_integrity()
        n, dimension = self.provenance["n_draws"], len(self.mean)
        empirical = self.provenance.get("empirical_resolution")
        return freeze_metadata({
            "schema": "joint_noise_estimation_diagnostic/1", "n_draws": n,
            "raw_dimension": self.raw_dimension, "state_dimension": dimension-self.raw_dimension,
            "joint_dimension": dimension, "draws_per_joint_dimension": n/dimension,
            "centered_rank_upper_bound": min(n-1, dimension),
            "empirical_resolution": empirical,
            "empirical_resolution_role": ("recorded_at_original_fit" if empirical is not None
                                           else "not_recorded_by_original_producer"),
            "shrinkage": self.provenance["shrinkage"],
            "empirical_null_directions_at_least": (None if empirical is None else
                empirical["active_dimensions"]-empirical["centered_rank_upper_bound"]),
            "shrinkage_fills_empirical_null_directions": (
                False if self.provenance["shrinkage"] == 0 else
                True if empirical is not None and empirical["centered_rank_upper_bound"] < empirical["active_dimensions"]
                else None),
            "statistical_accuracy_certified": False,
            "interpretation": "sample size, empirical resolution and shrinkage are distinct from physical support",
        })

    def _metadata(self):
        return {"schema": "n2p3_joint_noise/2", "raw_dimension": self.raw_dimension,
                "provenance": self.provenance,
                "state_sha256": array_digest(self.mean, self.covariance),
                "conditioning_rank_ledger": self.conditioning_rank_ledger,
                "conditioning_state_sha256": array_digest(*self._conditional_arrays().values())}

    def _conditional_arrays(self):
        return {"raw_scales": self._raw_scales, "raw_support_basis": self._raw_support_basis,
                "conditioning_gain": self.conditioning_gain,
                "conditional_state_covariance": self.conditional_state_covariance}

    def assert_integrity(self):
        if metadata_digest(self._metadata()) != self._integrity_sha256:
            raise ValueError("frozen joint noise integrity changed")

    def digest(self):
        self.assert_integrity()
        return self._integrity_sha256

    def project(self, raw_operator, state_operator):
        """Push forward the *joint* model, including all raw/state cross terms.

        Stack epoch operators by rows to retain overlap and common state effects.
        There is no independence assumption between epochs or raw and state.
        """
        self.assert_integrity()
        A = finite_real_array(raw_operator, name="raw operator")
        B = finite_real_array(state_operator, name="state operator")
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[0]:
            raise ValueError("operators must share the output sample dimension")
        if A.shape[1] != self.raw_dimension or B.shape[1] != len(self.mean) - self.raw_dimension:
            raise ValueError("operator dimensions do not match fitted noise coordinates")
        operator = np.column_stack([A, B])
        cov = operator @ self.covariance @ operator.T
        return PropagatedNoise(frozen_array(operator @ self.mean),
                               frozen_array((cov + cov.T) / 2))

    def validate_conditioning_support(self, centered):
        """Check all declared candidate residuals, without silently projecting."""
        value = finite_real_array(centered, name="conditioning residuals")
        if value.ndim < 2 or value.shape[-1] != self.raw_dimension:
            raise ValueError("finite raw residuals must match conditioning support coordinates")
        _support_residual(value, self._raw_scales, self._raw_support_basis, name="raw observation/candidate")

    def _require_conditional_producer(self):
        if self.provenance["implementation_sha256"] != _IMPLEMENTATION_SHA256:
            raise ValueError("conditional noise producer differs from its executed implementation")

    def condition(self, raw_values, *, raw_means, state_means):
        """Return a|x,q for full candidate expectations, including noise mean.

        This is a conditional distribution, not extra independent evidence.
        All candidates must share an affine raw support in this first slice.
        The fitted residual mean is already part of the supplied expectations.
        """
        self.assert_integrity()
        self._require_conditional_producer()
        raw = finite_real_array(raw_values, name="observed raw conditioners")
        mx = finite_real_array(raw_means, name="full candidate raw means")
        ma = finite_real_array(state_means, name="full candidate state means")
        if (raw.ndim != 2 or raw.shape[1] != self.raw_dimension or mx.ndim != 2
                or mx.shape[1] != self.raw_dimension or ma.shape != (len(mx), len(self.mean)-self.raw_dimension)
                or len(mx) == 0):
            raise ValueError("raw observations and full candidate raw/state means must match fitted dimensions")
        self.validate_conditioning_support(mx-mx[:1])
        centered = raw[:, None, :]-mx[None, :, :]
        self.validate_conditioning_support(centered)
        means = ma[None, :, :]+centered @ self.conditioning_gain.T
        return ConditionedNoise(frozen_array(means), self.conditional_state_covariance, self.conditioning_rank_ledger)

    def conditional_project(self, raw_operator, state_operator):
        """Compile y=A x+B a given x; callers supply full output expectations."""
        self.assert_integrity()
        self._require_conditional_producer()
        A = finite_real_array(raw_operator, name="conditional raw operator")
        B = finite_real_array(state_operator, name="conditional state operator")
        if (A.ndim != 2 or B.shape != (len(A), len(self.mean)-self.raw_dimension)
                or A.shape[1] != self.raw_dimension):
            raise ValueError("conditional operators must match fitted raw/state coordinates")
        covariance = B @ self.conditional_state_covariance @ B.T
        return ConditionalProjection(frozen_array(A+B @ self.conditioning_gain),
                                     frozen_array((covariance+covariance.T)/2), self.conditioning_rank_ledger)

    def save(self, path):
        self.assert_integrity()
        metadata = self._metadata()
        metadata["artifact_sha256"] = metadata_digest(metadata)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, mean=self.mean, covariance=self.covariance,
                                **self._conditional_arrays(),
                                metadata=np.asarray(json.dumps(metadata, sort_keys=True)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            mean, covariance = archive["mean"], archive["covariance"]
            conditional = ({k: finite_real_array(archive[k], name=f"saved {k}") for k in ("raw_scales", "raw_support_basis", "conditioning_gain", "conditional_state_covariance")}
                           if metadata.get("schema") == "n2p3_joint_noise/2" else None)
        if metadata.get("schema") not in {"n2p3_joint_noise/1", "n2p3_joint_noise/2"}:
            raise ValueError("unsupported joint noise schema")
        expected = metadata.pop("artifact_sha256", None)
        if expected != metadata_digest(metadata):
            raise ValueError("joint noise metadata digest mismatch")
        if metadata.get("state_sha256") != array_digest(mean, covariance):
            raise ValueError("joint noise state digest mismatch")
        model = cls(mean, covariance, metadata["raw_dimension"], metadata["provenance"])
        if conditional is not None:
            model._require_conditional_producer()
            if (metadata.get("conditioning_state_sha256") != array_digest(*conditional.values())
                    or metadata_digest(model.conditioning_rank_ledger) != metadata_digest(metadata.get("conditioning_rank_ledger"))
                    or any(not np.array_equal(v, model._conditional_arrays()[k]) for k, v in conditional.items())):
                raise ValueError("saved conditional noise state differs from its actual joint covariance")
        return model
