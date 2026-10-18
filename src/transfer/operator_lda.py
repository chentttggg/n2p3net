"""Research LDA using frozen propagated joint raw/state uncertainty.

The Interface consumes explicitly estimated JointNoiseModel, not oracle states
at prediction. The caller must supply authorized paired noise draws and the
actual feature operators. It does not solve physical state identifiability.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from transfer.noise_model import (
    JointNoiseModel,
    array_digest,
    finite_real_array,
    freeze_metadata,
    frozen_array,
    metadata_digest,
    require_identifiers,
    require_sha256,
)

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ConditioningObservationSupport:
    """Declared physical read support; caller attests the actual feature maps.

    A response wholly computed from the conditioner is not an extra
    observation. These intervals do not themselves certify a raw-data parser.
    """
    conditioning_sample_support: tuple[int, int]
    response_sample_support: tuple[int, int]
    available_samples: int

    def __post_init__(self):
        for name in ("conditioning_sample_support", "response_sample_support"):
            span = tuple(getattr(self, name))
            if len(span) != 2 or any(type(v) is not int for v in span) or not 0 <= span[0] < span[1]:
                raise ValueError("conditioning/response sample support must be an explicit nonempty interval")
            object.__setattr__(self, name, span)
        c, y = self.conditioning_sample_support, self.response_sample_support
        if type(self.available_samples) is not int or max(c[1], y[1]) > self.available_samples:
            raise ValueError("conditioning/response support exceeds available observed samples")
        if c[0] <= y[0] and y[1] <= c[1]:
            raise ValueError("conditional response needs observed information outside the conditioner")

    def record(self):
        return {"conditioning_sample_support": list(self.conditioning_sample_support),
                "response_sample_support": list(self.response_sample_support), "available_samples": self.available_samples}

    @classmethod
    def from_record(cls, record):
        return cls(**record)


@dataclass(frozen=True)
class OperatorAwareLDA:
    class_means: np.ndarray
    noise: JointNoiseModel
    raw_operator: np.ndarray
    state_operator: np.ndarray
    ridge_relative: float
    input_identity: str
    provenance: dict
    conditioning_class_means: np.ndarray | None = None
    observation_support: ConditioningObservationSupport | None = None
    conditioning_gain: np.ndarray | None = field(init=False, repr=False, compare=False)
    weights: np.ndarray = field(init=False)
    covariance: np.ndarray = field(init=False)
    identifiable_projector: np.ndarray = field(init=False)
    intercept: float = field(init=False)
    _integrity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not self.input_identity:
            raise ValueError("LDA requires an explicit feature input identity")
        if not np.isfinite(self.ridge_relative) or self.ridge_relative < 0:
            raise ValueError("relative regularization must be finite and nonnegative")
        means = finite_real_array(self.class_means, name="full class output expectations")
        conditional = self.conditioning_class_means is not None
        raw_means = None
        gain = None
        if conditional:
            if not isinstance(self.observation_support, ConditioningObservationSupport):
                raise ValueError("conditional operator requires declared observed sample support")
            raw_means = finite_real_array(self.conditioning_class_means, name="full class raw expectations")
            if raw_means.shape != (2, self.noise.raw_dimension):
                raise ValueError("full class raw expectations must match conditioning coordinates")
            self.noise.validate_conditioning_support(raw_means-raw_means[:1])
            projected = self.noise.conditional_project(self.raw_operator, self.state_operator)
            gain = projected.mean_gain
        else:
            if self.observation_support is not None:
                raise ValueError("observation support requires an actual conditional operator program")
            projected = self.noise.project(self.raw_operator, self.state_operator)
        covariance = projected.covariance
        if means.shape != (2, len(covariance)):
            raise ValueError("class means must match propagated output coordinates")
        values, vectors = np.linalg.eigh(covariance)
        if values[-1] <= 0:
            raise ValueError("no identifiable stochastic covariance directions")
        relative = np.finfo(float).eps*len(values) if conditional else 1e-10
        keep = values > values[-1]*relative
        V = vectors[:, keep]
        projector = V @ V.T
        adjusted_means = means if not conditional else means-raw_means @ gain.T
        delta = adjusted_means[1]-adjusted_means[0]
        null_contrast = delta - projector @ delta
        support_relative = 64*np.finfo(float).eps*len(values) if conditional else 1e-6
        if np.linalg.norm(null_contrast) > max(np.linalg.norm(delta), np.finfo(float).tiny)*support_relative:
            raise ValueError("class contrast outside noise-identifiable support")
        # Add uncertainty only on physically represented output directions.
        regularized = values[keep] + self.ridge_relative * values[keep].mean()
        weights = V @ ((V.T @ delta) / regularized)
        object.__setattr__(self, "class_means", frozen_array(means))
        object.__setattr__(self, "raw_operator", frozen_array(self.raw_operator))
        object.__setattr__(self, "state_operator", frozen_array(self.state_operator))
        object.__setattr__(self, "covariance", frozen_array((V * regularized) @ V.T))
        object.__setattr__(self, "identifiable_projector", frozen_array(projector))
        object.__setattr__(self, "weights", frozen_array(weights))
        object.__setattr__(self, "conditioning_class_means", None if raw_means is None else frozen_array(raw_means))
        object.__setattr__(self, "conditioning_gain", gain)
        object.__setattr__(self, "intercept", -float((adjusted_means[0]+adjusted_means[1]) @ weights)/2)
        provenance = json.loads(json.dumps(self.provenance, allow_nan=False))
        if (provenance.get("fit_scope") != "target_calibration" or not provenance.get("scope_id")
                or provenance.get("noise_estimation_scope") != json.loads(json.dumps(self.noise.provenance))):
            raise ValueError("operator LDA fit and noise ledger disagree")
        require_identifiers(provenance.get("calibration_ids"), "LDA calibration IDs")
        counts = provenance.get("class_counts")
        if (not isinstance(counts, list) or len(counts) != 2
                or any(type(count) is not int or count < 1 for count in counts)
                or sum(counts) != len(provenance["calibration_ids"])):
            raise ValueError("LDA class count and calibration ledger disagree")
        require_sha256(provenance.get("fit_sha256"), "LDA calibration input")
        require_sha256(provenance.get("implementation_sha256"), "LDA implementation")
        if (self.noise.provenance["fit_scope"] == "target_calibration"
                and self.noise.provenance["scope_id"] != provenance["scope_id"]):
            raise ValueError("target noise fit exceeds the classifier calibration scope")
        if (self.noise.provenance["fit_scope"] == "target_calibration"
                and not set(self.noise.provenance["draw_ids"]) <= set(provenance["calibration_ids"])):
            raise ValueError("target noise draws exceed actual authorized calibration IDs")
        if conditional:
            if (provenance.get("implementation_sha256") != _IMPLEMENTATION_SHA256
                    or provenance.get("conditioning_program") != "conditional_tail_with_fixed_class_prior"
                    or provenance.get("candidate_mean_definition") != "full_class_expectation_including_noise_mean"
                    or metadata_digest(provenance.get("observation_support")) != metadata_digest(self.observation_support.record())
                    or provenance.get("conditional_noise_sha256") != self.noise.digest()):
                raise ValueError("conditional operator producer/support differs from its actual frozen program")
        elif "conditioning_program" in provenance:
            raise ValueError("conditional operator cannot be relabeled as unconditional LDA")
        object.__setattr__(self, "provenance", freeze_metadata(provenance))
        object.__setattr__(self, "_integrity_sha256", self._integrity_digest())

    @classmethod
    def fit(cls, features, labels, *, noise, raw_operator, state_operator,
            calibration_ids, scope_id, input_identity, ridge_relative=1e-3,
            conditioning_features=None, observation_support=None):
        """Fit class means on target calibration, retain the supplied noise fit.

        A calibration-derived noise model must share this scope identity. Source
        noise may be borrowed explicitly; its fit rows/scope remain in the model.
        The covariance is not silently re-estimated from held-out features.
        """
        X, y = finite_real_array(features, name="calibration features"), np.asarray(labels)
        if not scope_id or not isinstance(noise, JointNoiseModel):
            raise ValueError("calibration scope and a fitted joint noise model are required")
        if X.ndim != 2 or y.shape != (len(X),) or len(X) < 4:
            raise ValueError("finite calibration features and one label per row are required")
        if set(y.tolist()) != {0, 1}:
            raise ValueError("both binary calibration classes are required")
        noise.assert_integrity()
        ids = tuple(calibration_ids)
        require_identifiers(ids, "LDA calibration IDs")
        if len(ids) != len(X) or len(set(ids)) != len(ids) or any(not v for v in ids):
            raise ValueError("calibration IDs must be complete and unique")
        noise_scope = noise.provenance.get("fit_scope")
        if noise_scope not in {"source_training", "target_calibration"}:
            raise ValueError("joint noise was not fitted on a declared training scope")
        if noise_scope == "target_calibration" and noise.provenance.get("scope_id") != scope_id:
            raise ValueError("target noise fit exceeds the classifier calibration scope")
        means = np.stack([X[y == label].mean(axis=0) for label in (0, 1)])
        raw_means = None
        raw = None
        conditional_provenance = {}
        if conditioning_features is not None:
            raw = finite_real_array(conditioning_features, name="observed raw calibration conditioners")
            if raw.shape != (len(X), noise.raw_dimension):
                raise ValueError("observed raw calibration conditioners must match rows and fitted coordinates")
            if not isinstance(observation_support, ConditioningObservationSupport):
                raise ValueError("conditional calibration requires its observed sample support")
            raw_means = np.stack([raw[y == label].mean(axis=0) for label in (0, 1)])
            noise.validate_conditioning_support(raw-raw_means[y.astype(int)])
            conditional_provenance = {"conditioning_program": "conditional_tail_with_fixed_class_prior",
                "candidate_mean_definition": "full_class_expectation_including_noise_mean",
                "observation_support": observation_support.record(), "conditional_noise_sha256": noise.digest(),
                "conditioning_input_sha256": array_digest(raw), "raw_likelihood_multiplied": False}
        model = cls(means, noise, raw_operator, state_operator, ridge_relative,
                    input_identity, {"fit_scope": "target_calibration", "scope_id": str(scope_id),
                                     "calibration_ids": list(ids), "class_counts": [int(sum(y == k)) for k in (0, 1)],
                                     "fit_sha256": array_digest(X, y, *( () if raw is None else (raw,))),
                                     "implementation_sha256": _IMPLEMENTATION_SHA256,
                                     "noise_estimation_scope": noise.provenance,
                                     "noise_parameters_reestimated_by_classifier": False,
                                     "class_prior": "equal", "oracle_state_at_predict": False, **conditional_provenance},
                    conditioning_class_means=raw_means, observation_support=observation_support)
        residual = X-means[y.astype(int)]
        if raw is not None:
            residual -= (raw-raw_means[y.astype(int)]) @ model.conditioning_gain.T
        model._validate_support(residual)
        return model

    def _validate_support(self, centered):
        # Scale each row before projection and norm evaluation: the relative
        # support geometry is unchanged without overflow/underflow in squares.
        tiny = np.finfo(float).tiny
        scale = np.maximum(np.max(np.abs(centered), axis=-1, keepdims=True), tiny)
        normalized = centered / scale
        residual = normalized - normalized @ self.identifiable_projector
        relative = 64*np.finfo(float).eps*len(self.weights) if self.conditioning_gain is not None else 1e-6
        limit = np.maximum(np.linalg.norm(normalized, axis=-1), (tiny / scale)[..., 0]) * relative
        if np.any(np.linalg.norm(residual, axis=-1) > limit):
            raise ValueError("features outside frozen covariance identifiable support")

    def predict_logit(self, features, *, input_identity, observed_raw=None):
        """Read-only prediction: no labels, raw states or covariance updates."""
        self.assert_integrity()
        if input_identity != self.input_identity:
            raise ValueError("prediction input identity differs from frozen feature operator")
        X = finite_real_array(features, name="prediction features")
        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise ValueError("finite prediction features must match frozen output coordinates")
        means = self.class_means.mean(axis=0)
        if self.conditioning_gain is not None:
            raw = finite_real_array(observed_raw, name="observed raw conditioners") if observed_raw is not None else None
            if raw is None or raw.shape != (len(X), self.noise.raw_dimension):
                raise ValueError("conditional prediction requires finite observed raw conditioners")
            self.noise.validate_conditioning_support(raw-self.conditioning_class_means.mean(axis=0))
            X = X-raw @ self.conditioning_gain.T
            means = means-self.conditioning_class_means.mean(axis=0) @ self.conditioning_gain.T
        elif observed_raw is not None:
            raise ValueError("observed raw conditioners require a frozen conditional operator program")
        self._validate_support(X-means)
        return X @ self.weights + self.intercept

    def _arrays(self):
        arrays = {"class_means": self.class_means, "joint_mean": self.noise.mean,
                "joint_covariance": self.noise.covariance, "raw_operator": self.raw_operator,
                "state_operator": self.state_operator}
        if self.conditioning_gain is not None:
            arrays.update(conditioning_class_means=self.conditioning_class_means,
                          conditioning_gain=self.conditioning_gain, conditional_covariance=self.covariance)
        return arrays

    def _metadata(self):
        return {"schema": "n2p3_operator_aware_lda/2" if self.conditioning_gain is not None else "n2p3_operator_aware_lda/1", "ridge_relative": self.ridge_relative,
                "input_identity": self.input_identity, "provenance": self.provenance,
                "noise_raw_dimension": self.noise.raw_dimension, "noise_provenance": self.noise.provenance,
                "arrays_sha256": array_digest(*self._arrays().values())}

    def _integrity_digest(self):
        return metadata_digest({**self._metadata(), "derived_sha256": array_digest(
            self.weights, self.covariance, self.identifiable_projector, np.asarray(self.intercept))})

    def assert_integrity(self):
        self.noise.assert_integrity()
        if self._integrity_digest() != self._integrity_sha256:
            raise ValueError("frozen operator LDA integrity changed")

    def save(self, path):
        self.assert_integrity()
        arrays = self._arrays()
        metadata = self._metadata()
        metadata["artifact_sha256"] = metadata_digest(metadata)
        from research.execution import publish_exclusive

        publish_exclusive(path, lambda handle: np.savez_compressed(
            handle, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True))))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            keys = ["class_means", "joint_mean", "joint_covariance", "raw_operator", "state_operator"]
            conditional = metadata.get("schema") == "n2p3_operator_aware_lda/2"
            if conditional:
                keys += ["conditioning_class_means", "conditioning_gain", "conditional_covariance"]
            if set(archive.files) != {*keys, "metadata"}:
                raise ValueError("operator LDA numeric payload differs from its declared program")
            arrays = [finite_real_array(archive[key], name=f"saved {key}") for key in keys]
        if metadata.get("schema") not in {"n2p3_operator_aware_lda/1", "n2p3_operator_aware_lda/2"}:
            raise ValueError("unsupported operator LDA schema")
        digest = metadata.pop("artifact_sha256", None)
        if digest != metadata_digest(metadata) or metadata.get("arrays_sha256") != array_digest(*arrays):
            raise ValueError("operator LDA artifact digest mismatch")
        means, joint_mean, joint_cov, A, B = arrays[:5]
        noise = JointNoiseModel(joint_mean, joint_cov, metadata["noise_raw_dimension"], metadata["noise_provenance"])
        model = cls(means, noise, A, B, metadata["ridge_relative"], metadata["input_identity"], metadata["provenance"],
                    conditioning_class_means=arrays[5] if conditional else None,
                    observation_support=ConditioningObservationSupport.from_record(metadata["provenance"]["observation_support"]) if conditional else None)
        if conditional and (not np.array_equal(arrays[6], model.conditioning_gain) or not np.array_equal(arrays[7], model.covariance)):
            raise ValueError("operator conditional mean/covariance differs from actual joint noise")
        return model
