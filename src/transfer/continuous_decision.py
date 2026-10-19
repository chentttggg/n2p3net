"""Frozen continuous candidate likelihood with estimated kernel uncertainty.

Each raw sample is used once. A matrix-normal ridge response model is fitted
only on labeled calibration/source records. AR(1) temporal noise and a shrunk
channel covariance are estimated from calibration residuals. Prediction includes
the conditional kernel posterior covariance and candidate-dependent logdet.

This is a research likelihood, not a certified posterior: AR/noise hyperparameter
uncertainty and nonlinear/refractory ERP behavior are not integrated. Constant
and linear drift are explicit shared diffuse nuisance directions, projected out
in whitened coordinates. They can remove genuine ERP information.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf

from transfer.noise_model import (
    array_digest,
    freeze_metadata,
    frozen_array,
    metadata_digest,
    require_identifiers,
    require_sha256,
    validate_covariance,
)

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class CalibrationRecord:
    record_id: str
    samples: np.ndarray  # (continuous samples, observed channels)
    onsets: np.ndarray   # exact sample indices within this independent record
    target_mask: np.ndarray  # binary *calibration* event labels
    right_censored: bool = False  # explicit authorized support ends before the final response tail


@dataclass(frozen=True)
class ContinuousDecisionResult:
    candidate_ids: tuple[str, ...]
    log_evidence: np.ndarray
    probabilities: np.ndarray
    winner: str | None
    reason: str
    n_unique_samples: int
    input_sha256: str

    def record(self):
        return {
            "schema": "n2p3_continuous_decision_result/1",
            "candidate_ids": list(self.candidate_ids),
            "log_evidence": self.log_evidence.tolist(),
            "probabilities": self.probabilities.tolist(),
            "winner": self.winner, "reason": self.reason,
            "n_unique_samples": self.n_unique_samples,
            "input_sha256": self.input_sha256,
            "evidence_scope": "research_conditional_gaussian_likelihood",
        }


def smooth_erp_basis(sfreq, response_seconds=0.8, n_basis=8):
    """Fixed smooth time dictionary; this is not a learned source prior."""
    if not np.isfinite(sfreq) or sfreq <= 0 or not np.isfinite(response_seconds):
        raise ValueError("invalid ERP basis sampling or response duration")
    length = int(round(sfreq * response_seconds))
    if not isinstance(n_basis, int) or not 1 <= n_basis <= length or length < 3:
        raise ValueError("ERP basis needs 1 <= n_basis <= response samples")
    centers = np.linspace(0, length - 1, n_basis)
    width = max(1., length / max(n_basis - 1, 1))
    return np.exp(-0.5 * ((np.arange(length)[:, None] - centers) / width) ** 2)


def _validate_record(samples, onsets, n_channels, kernel_length, *, allow_right_censor=False):
    X = np.asarray(samples, dtype=float)
    onset = np.asarray(onsets)
    if X.ndim != 2 or X.shape[1] != n_channels or len(X) < kernel_length + 3:
        raise ValueError("continuous samples must match observed channels and response support")
    if not np.all(np.isfinite(X)):
        raise ValueError("continuous samples must be finite")
    if onset.ndim != 1 or not len(onset) or onset.dtype.kind not in "iu":
        raise ValueError("onsets must be nonempty integer sample indices")
    if (type(allow_right_censor) is not bool or np.any(onset[1:] <= onset[:-1]) or onset[0] < 0
            or onset[-1] >= len(X) or (not allow_right_censor and int(onset[-1]) + kernel_length > len(X))):
        raise ValueError("events must be ordered, distinct and have complete response tails")
    return X, onset.astype(np.int64, copy=False)


def _event_design(onsets, labels, n_samples, basis):
    width, rank = basis.shape
    design = np.zeros((n_samples, 2 * rank))
    for onset, target in zip(onsets, labels, strict=True):
        retained = min(width, n_samples - int(onset))
        design[onset:onset + retained, :rank] += basis[:retained]
        design[onset:onset + retained, rank:] += float(target) * basis[:retained]
    return design


def _whiten_and_project(value, phi):
    array = np.asarray(value, dtype=float)
    whitened = array.copy()
    whitened[0] *= np.sqrt(1 - phi**2)
    whitened[1:] = array[1:] - phi * array[:-1]
    nuisance = np.column_stack([np.ones(len(array)), np.linspace(-1, 1, len(array))])
    nuisance[1:] -= phi * nuisance[:-1].copy()
    nuisance[0] *= np.sqrt(1 - phi**2)
    Q, _ = np.linalg.qr(nuisance, mode="reduced")
    return whitened - Q @ (Q.T @ whitened)


@dataclass(frozen=True)
class ContinuousDecisionModel:
    basis: np.ndarray
    kernel_mean: np.ndarray
    kernel_row_covariance: np.ndarray
    innovation_covariance: np.ndarray
    phi: float
    sfreq: float
    channels: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    operator_identity: str
    provenance: dict
    _integrity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        basis = np.asarray(self.basis, dtype=float)
        mean = np.asarray(self.kernel_mean, dtype=float)
        row_cov = validate_covariance(self.kernel_row_covariance, name="kernel covariance")
        noise = validate_covariance(self.innovation_covariance, name="innovation covariance")
        channels, candidates = tuple(self.channels), tuple(self.candidate_ids)
        if basis.ndim != 2 or min(basis.shape) == 0 or not np.all(np.isfinite(basis)):
            raise ValueError("invalid ERP basis")
        p = basis.shape[1] * 2
        if mean.shape != (p, len(channels)) or row_cov.shape != (p, p):
            raise ValueError("kernel mean/covariance do not match basis and channels")
        if noise.shape != (len(channels), len(channels)) or not np.all(np.isfinite(mean)):
            raise ValueError("innovation covariance does not match channels")
        if not len(channels) or len(set(channels)) != len(channels) or any(not c for c in channels):
            raise ValueError("observed channel names must be distinct and nonempty")
        if len(candidates) < 2 or len(set(candidates)) != len(candidates) or any(not c for c in candidates):
            raise ValueError("candidate vocabulary must be complete, unique and nonempty")
        if not np.isfinite(self.phi) or abs(self.phi) >= 1 or not np.isfinite(self.sfreq) or self.sfreq <= 0:
            raise ValueError("invalid AR parameter or sampling frequency")
        if not self.operator_identity:
            raise ValueError("explicit materialized input operator identity is required")
        np.linalg.cholesky(noise)
        np.linalg.cholesky(row_cov)
        for name, value in [("basis", basis), ("kernel_mean", mean),
                            ("kernel_row_covariance", row_cov), ("innovation_covariance", noise)]:
            object.__setattr__(self, name, frozen_array(value))
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "candidate_ids", candidates)
        provenance = json.loads(json.dumps(self.provenance, allow_nan=False))
        scope = provenance.get("fit_scope")
        if scope == "analytic_reference":
            if set(provenance) != {"fit_scope", "reference_id", "records"} or provenance["records"] or not provenance["reference_id"]:
                raise ValueError("analytic numerical reference must be explicitly identified without training claims")
        else:
            if scope not in {"source_training", "target_calibration"} or not provenance.get("records"):
                raise ValueError("continuous model requires a nonempty complete training ledger")
            required = {"record_id", "n_samples", "n_events", "n_targets", "signal_sha256", "input_sha256",
                        "right_censored", "right_censored_events"}
            require_identifiers([record["record_id"] for record in provenance["records"]], "training record IDs")
            for record in provenance["records"]:
                if (set(record) != required or any(type(record[key]) is not int for key in ("n_samples", "n_events", "n_targets"))
                        or record["n_samples"] < len(basis) + 3 or not 1 <= record["n_events"] <= record["n_samples"]
                        or not 0 <= record["n_targets"] <= record["n_events"]):
                    raise ValueError("invalid continuous training record ledger")
                if (type(record["right_censored"]) is not bool or type(record["right_censored_events"]) is not int
                        or not 0 <= record["right_censored_events"] <= record["n_events"]
                        or (not record["right_censored"] and record["right_censored_events"])):
                    raise ValueError("invalid continuous calibration censoring declaration")
                require_sha256(record["signal_sha256"], "continuous fit signal")
                require_sha256(record["input_sha256"], "continuous fit labels/input")
            require_sha256(provenance.get("implementation_sha256"), "continuous implementation")
            if not np.isfinite(provenance.get("ridge", float("nan"))) or provenance["ridge"] <= 0:
                raise ValueError("continuous fit ledger lacks positive ridge")
        object.__setattr__(self, "provenance", freeze_metadata(provenance))
        object.__setattr__(self, "_integrity_sha256", metadata_digest(self._metadata()))

    @classmethod
    def fit(cls, records, *, sfreq, channels, candidate_ids, operator_identity,
            basis=None, ridge=1., fit_scope="target_calibration"):
        """Fit all response and noise state from declared labeled records only.

        Records are independent legal scopes with complete stimulus histories.
        Explicit right_censored calibration ends before an unlabelled flash;
        response design rows beyond that authorized support are unobserved.
        Every supplied event retains its authorized label. Scoring still requires
        complete response tails and never inherits this calibration permission.
        The caller must enforce physical participant/time/label authorization;
        record IDs and digests retain the concrete input used here. Ridge and
        dictionary must be fixed before held-out scoring.
        """
        if fit_scope not in {"target_calibration", "source_training"}:
            raise ValueError("continuous model fit scope must be training/calibration")
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be strictly positive")
        basis = smooth_erp_basis(sfreq) if basis is None else np.asarray(basis, dtype=float)
        if basis.ndim != 2 or not np.all(np.isfinite(basis)) or min(basis.shape) == 0:
            raise ValueError("basis must be finite and nonempty")
        records = tuple(records)
        ids = [r.record_id for r in records]
        if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("distinct calibration record IDs are required")
        designs, samples, ledger, all_labels = [], [], [], []
        for record in records:
            X, onsets = _validate_record(record.samples, record.onsets, len(channels), len(basis),
                                        allow_right_censor=record.right_censored)
            labels = np.asarray(record.target_mask)
            if labels.shape != onsets.shape or not np.all(np.isin(labels, [0, 1])):
                raise ValueError("one binary calibration label is required per event")
            designs.append(_event_design(onsets, labels, len(X), basis))
            samples.append(X)
            all_labels.extend(labels.tolist())
            ledger.append({"record_id": record.record_id, "n_samples": len(X),
                           "n_events": len(onsets), "n_targets": int(labels.sum()),
                           "right_censored": record.right_censored,
                           "right_censored_events": int(np.sum(onsets + len(basis) > len(X))),
                           "signal_sha256": array_digest(X, onsets),
                           "input_sha256": array_digest(X, onsets, labels)})
        if set(all_labels) != {0, 1}:
            raise ValueError("both target and non-target calibration events are required")
        p = 2 * basis.shape[1]

        def solve(phi):
            D = np.vstack([_whiten_and_project(d, phi) for d in designs])
            Y = np.vstack([_whiten_and_project(x, phi) for x in samples])
            gram = D.T @ D
            row_cov = np.linalg.inv(gram + ridge * np.eye(p))
            mean = row_cov @ D.T @ Y
            return D, Y, mean, (row_cov + row_cov.T) / 2, gram

        D, Y, initial, _, _ = solve(0.)
        initial_residual = Y - D @ initial
        offsets = np.cumsum([0] + [len(x) for x in samples])
        pairs = [initial_residual[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
        numerator = sum(float(np.sum(r[1:] * r[:-1])) for r in pairs)
        denominator = sum(float(np.sum(r[:-1] ** 2)) for r in pairs)
        phi = float(np.clip(numerator / max(denominator, np.finfo(float).tiny), -0.95, 0.95))
        D, Y, mean, row_cov, gram = solve(phi)
        residual = Y - D @ mean
        effective_parameters = float(np.trace(gram @ row_cov))
        residual_df = len(Y) - 2 * len(records) - effective_parameters
        if residual_df <= 1:
            raise ValueError("insufficient residual degrees of freedom for noise estimation")
        estimator = LedoitWolf(assume_centered=True).fit(residual)
        noise = estimator.covariance_ * len(Y) / residual_df
        # A tiny *relative* floor only ensures a usable density in observed coordinates.
        scale = max(float(np.trace(noise) / len(channels)),
                    float(np.mean(Y**2)) * np.finfo(float).eps, np.finfo(float).tiny)
        noise += np.eye(len(channels)) * scale * 1e-8
        return cls(basis, mean, row_cov, noise, phi, sfreq, tuple(channels),
                   tuple(candidate_ids), operator_identity, {
                       "fit_scope": fit_scope, "records": ledger, "ridge": float(ridge),
                       "implementation_sha256": _IMPLEMENTATION_SHA256,
                       "design_rank": int(np.linalg.matrix_rank(D)), "kernel_parameters_per_channel": p,
                       "effective_parameters_per_channel": effective_parameters,
                       "residual_degrees_of_freedom": residual_df,
                       "channel_shrinkage": float(estimator.shrinkage_),
                       "kernel_uncertainty": "matrix_normal_posterior_conditional_on_fitted_noise",
                       "noise": "calibration_estimated_AR1_times_channel_covariance",
                       "nuisance": "shared_diffuse_constant_and_linear_trend",
                       "hyperparameter_uncertainty_integrated": False,
                       "likelihood_assumptions": ["linear_event_superposition", "shared_class_noise",
                                                  "stationary_AR1", "complete_event_history"],
                   })

    def score(self, samples, onsets, membership, *, operator_identity, sfreq, channels,
              candidate_ids, record_id):
        """Score a complete fixed vocabulary using public membership, never truth.

        The supplied record is one independent decision scope. Onsets must cover
        all response-generating events; unknown adjacent decisions must be
        handled by the caller or excluded by predeclared support. No state is
        updated. No oracle kernel or Gamma is accepted by this interface.
        """
        self.assert_integrity()
        if operator_identity != self.operator_identity:
            raise ValueError("prediction operator identity differs from frozen model")
        if sfreq != self.sfreq or tuple(channels) != self.channels:
            raise ValueError("prediction sample rate or channel order differs from frozen model")
        if tuple(candidate_ids) != self.candidate_ids:
            raise ValueError("prediction candidate vocabulary/order differs from frozen model")
        training_records = self.provenance["records"]
        if not record_id or record_id in {r["record_id"] for r in training_records}:
            raise ValueError("prediction requires a distinct held-out record identity")
        X, onsets = _validate_record(samples, onsets, len(self.channels), len(self.basis))
        if array_digest(X, onsets) in {r["signal_sha256"] for r in training_records}:
            raise ValueError("prediction signal repeats an exact calibration record")
        member = np.asarray(membership)
        if member.shape != (len(onsets), len(self.candidate_ids)) or not np.all(np.isin(member, [0, 1])):
            raise ValueError("membership must include the complete frozen candidate vocabulary")
        if np.any(member.sum(axis=1) == 0) or np.any(member.sum(axis=1) == len(self.candidate_ids)):
            raise ValueError("every event must flash a proper nonempty candidate subset")
        if np.any(member.sum(axis=0) == 0):
            raise ValueError("missing candidate evidence: refuse incomplete decision")
        Y = _whiten_and_project(X, self.phi)
        spatial_root = np.linalg.cholesky(self.innovation_covariance)
        kernel_root = np.linalg.cholesky(self.kernel_row_covariance)
        evidence = []
        for candidate in range(len(self.candidate_ids)):
            design = _event_design(onsets, member[:, candidate], len(X), self.basis)
            D = _whiten_and_project(design, self.phi)
            residual = Y - D @ self.kernel_mean
            residual = solve_triangular(spatial_root, residual.T, lower=True).T
            U = D @ kernel_root
            small = np.eye(U.shape[1]) + U.T @ U
            root = np.linalg.cholesky(small)
            projected = solve_triangular(root, U.T @ residual, lower=True)
            mahalanobis = float(np.sum(residual**2) - np.sum(projected**2))
            if mahalanobis < -1e-8 * max(1., float(np.sum(residual**2))):
                raise ValueError("unstable Gaussian covariance solve")
            logdet = len(self.channels) * 2 * float(np.log(np.diag(root)).sum())
            evidence.append(-0.5 * (max(0., mahalanobis) + logdet))
        evidence = np.asarray(evidence)
        probability = np.exp(evidence - logsumexp(evidence))
        best = int(np.argmax(evidence))
        tied = np.count_nonzero(np.isclose(evidence, evidence[best], rtol=0, atol=1e-10)) > 1
        return ContinuousDecisionResult(self.candidate_ids, frozen_array(evidence),
                                        frozen_array(probability),
                                        None if tied else self.candidate_ids[best],
                                        "tied_evidence" if tied else "ok", len(X),
                                        array_digest(X, onsets, member))

    def _metadata(self):
        return {"schema": "n2p3_continuous_decision_model/2", "phi": self.phi,
                "sfreq": self.sfreq, "channels": self.channels,
                "candidate_ids": self.candidate_ids, "operator_identity": self.operator_identity,
                "provenance": self.provenance,
                "state_sha256": array_digest(self.basis, self.kernel_mean,
                                              self.kernel_row_covariance, self.innovation_covariance)}

    def assert_integrity(self):
        if metadata_digest(self._metadata()) != self._integrity_sha256:
            raise ValueError("frozen continuous model integrity changed")

    def save(self, path):
        self.assert_integrity()
        state = {"basis": self.basis, "kernel_mean": self.kernel_mean,
                 "kernel_row_covariance": self.kernel_row_covariance,
                 "innovation_covariance": self.innovation_covariance}
        metadata = self._metadata()
        metadata["artifact_sha256"] = metadata_digest(metadata)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, **state, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            state = [archive[key] for key in ("basis", "kernel_mean", "kernel_row_covariance",
                                             "innovation_covariance")]
        if metadata.get("schema") != "n2p3_continuous_decision_model/2":
            raise ValueError("unsupported continuous decision model schema")
        expected = metadata.pop("artifact_sha256", None)
        if expected != metadata_digest(metadata):
            raise ValueError("continuous model metadata digest mismatch")
        if array_digest(*state) != metadata.get("state_sha256"):
            raise ValueError("continuous model state digest mismatch")
        return cls(*state, metadata["phi"], metadata["sfreq"], tuple(metadata["channels"]),
                   tuple(metadata["candidate_ids"]), metadata["operator_identity"], metadata["provenance"])
