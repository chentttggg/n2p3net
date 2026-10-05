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
    freeze_metadata,
    frozen_array,
    metadata_digest,
    require_identifiers,
    require_sha256,
)

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class OperatorAwareLDA:
    class_means: np.ndarray
    noise: JointNoiseModel
    raw_operator: np.ndarray
    state_operator: np.ndarray
    ridge_relative: float
    input_identity: str
    provenance: dict
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
        means = np.asarray(self.class_means, dtype=float)
        projected = self.noise.project(self.raw_operator, self.state_operator)
        covariance = projected.covariance
        if means.shape != (2, len(covariance)) or not np.all(np.isfinite(means)):
            raise ValueError("class means must match propagated output coordinates")
        values, vectors = np.linalg.eigh(covariance)
        if values[-1] <= 0:
            raise ValueError("no identifiable stochastic covariance directions")
        keep = values > values[-1] * 1e-10
        V = vectors[:, keep]
        projector = V @ V.T
        delta = means[1] - means[0]
        null_contrast = delta - projector @ delta
        if np.linalg.norm(null_contrast) > max(np.linalg.norm(delta), np.finfo(float).tiny) * 1e-6:
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
        object.__setattr__(self, "intercept", -float((means[0] + means[1]) @ weights) / 2)
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
        object.__setattr__(self, "provenance", freeze_metadata(provenance))
        object.__setattr__(self, "_integrity_sha256", self._integrity_digest())

    @classmethod
    def fit(cls, features, labels, *, noise, raw_operator, state_operator,
            calibration_ids, scope_id, input_identity, ridge_relative=1e-3):
        """Fit class means on target calibration, retain the supplied noise fit.

        A calibration-derived noise model must share this scope identity. Source
        noise may be borrowed explicitly; its fit rows/scope remain in the model.
        The covariance is not silently re-estimated from held-out features.
        """
        X, y = np.asarray(features, dtype=float), np.asarray(labels)
        if not scope_id or not isinstance(noise, JointNoiseModel):
            raise ValueError("calibration scope and a fitted joint noise model are required")
        if X.ndim != 2 or y.shape != (len(X),) or len(X) < 4 or not np.all(np.isfinite(X)):
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
        model = cls(means, noise, raw_operator, state_operator, ridge_relative,
                    input_identity, {"fit_scope": "target_calibration", "scope_id": str(scope_id),
                                     "calibration_ids": list(ids), "class_counts": [int(sum(y == k)) for k in (0, 1)],
                                     "fit_sha256": array_digest(X, y),
                                     "implementation_sha256": _IMPLEMENTATION_SHA256,
                                     "noise_estimation_scope": noise.provenance,
                                     "noise_parameters_estimated_from_calibration_features": False,
                                     "class_prior": "equal", "oracle_state_at_predict": False})
        model._validate_support(X - means[y.astype(int)])
        return model

    def _validate_support(self, centered):
        residual = centered - centered @ self.identifiable_projector
        if np.linalg.norm(residual) > max(np.linalg.norm(centered), np.finfo(float).tiny) * 1e-6:
            raise ValueError("features outside frozen covariance identifiable support")

    def predict_logit(self, features, *, input_identity):
        """Read-only prediction: no labels, raw states or covariance updates."""
        self.assert_integrity()
        if input_identity != self.input_identity:
            raise ValueError("prediction input identity differs from frozen feature operator")
        X = np.asarray(features, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.weights) or not np.all(np.isfinite(X)):
            raise ValueError("finite prediction features must match frozen output coordinates")
        self._validate_support(X - self.class_means.mean(axis=0))
        return X @ self.weights + self.intercept

    def _arrays(self):
        return {"class_means": self.class_means, "joint_mean": self.noise.mean,
                "joint_covariance": self.noise.covariance, "raw_operator": self.raw_operator,
                "state_operator": self.state_operator}

    def _metadata(self):
        return {"schema": "n2p3_operator_aware_lda/1", "ridge_relative": self.ridge_relative,
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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            arrays = [archive[key] for key in ("class_means", "joint_mean", "joint_covariance",
                                              "raw_operator", "state_operator")]
        if metadata.get("schema") != "n2p3_operator_aware_lda/1":
            raise ValueError("unsupported operator LDA schema")
        digest = metadata.pop("artifact_sha256", None)
        if digest != metadata_digest(metadata) or metadata.get("arrays_sha256") != array_digest(*arrays):
            raise ValueError("operator LDA artifact digest mismatch")
        means, joint_mean, joint_cov, A, B = arrays
        noise = JointNoiseModel(joint_mean, joint_cov, metadata["noise_raw_dimension"], metadata["noise_provenance"])
        return cls(means, noise, A, B, metadata["ridge_relative"], metadata["input_identity"], metadata["provenance"])
