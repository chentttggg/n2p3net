"""Frozen continuous candidate likelihood with estimated kernel uncertainty.

Each raw sample is used once. A matrix-normal ridge response model is fitted
only on labeled calibration/source records. The original default retains its
clipped AR(1) estimator. Explicit research orders use the shared segmented Burg
owner and exact stationary PARCOR innovations. Prediction includes the
conditional kernel posterior covariance and candidate-dependent logdet.

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
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf

from research.execution import UnidentifiableFitError
from transfer.noise_model import (
    array_digest,
    freeze_metadata,
    frozen_array,
    metadata_digest,
    require_identifiers,
    require_sha256,
    validate_covariance,
)
from transfer.temporal_noise import TemporalNoise

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_STATE_KEYS = ("basis", "kernel_mean", "kernel_row_covariance", "innovation_covariance")
_COMPUTE_DTYPE = np.dtype(np.float64).str
_CONDITIONAL_SOLVER_SETTINGS = freeze_metadata({"eta_max": 8., "maxiter": 1000, "maxcor": 32, "ftol": 1e-10, "gtol": 1e-6})


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
    conditioning_samples: int = 0

    @property
    def log_probability(self):
        """Normalized log mass before finite-precision probability underflow."""
        return frozen_array(self.log_evidence - logsumexp(self.log_evidence))

    def record(self):
        result = {
            "schema": "n2p3_continuous_decision_result/2",
            "candidate_ids": list(self.candidate_ids),
            "log_evidence": self.log_evidence.tolist(),
            "log_probability": self.log_probability.tolist(),
            "probabilities": self.probabilities.tolist(),
            "winner": self.winner, "reason": self.reason,
            "n_unique_samples": self.n_unique_samples,
            "input_sha256": self.input_sha256,
            "evidence_scope": "research_conditional_gaussian_likelihood",
        }
        if self.conditioning_samples:
            result.update(schema="n2p3_continuous_decision_result/3",
                conditioning_sample_support=[0, self.conditioning_samples],
                score_sample_support=[self.conditioning_samples, self.n_unique_samples],
                n_scored_samples=self.n_unique_samples - self.conditioning_samples,
                score_estimand="conditional_tail_with_fixed_candidate_prior",
                prefix_updates="kernel_and_per_record_diffuse_drift",
                gaussian_normalization_constants_included=True)
        return result


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


def _real_values(value, name):
    supplied = np.asarray(value)
    if supplied.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real numeric values")
    return np.asarray(supplied, dtype=float)


def _validate_record(samples, onsets, n_channels, kernel_length, *, allow_right_censor=False):
    X = _real_values(samples, "continuous samples")
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


def _stationary_transform(noise, n_samples):
    """One within-record noise geometry shared by Y, every design and drift."""
    nuisance = np.column_stack([np.ones(n_samples), np.linspace(-1, 1, n_samples)])
    Q, _ = np.linalg.qr(noise.whiten(nuisance), mode="reduced")

    def transform(value):
        whitened = noise.whiten(value)
        return whitened - Q @ (Q.T @ whitened)

    return transform


def _joint_posterior(design, samples, prior_mean, prior_root):
    """One shared kernel and independent record drifts, updated by exact QR."""
    width = design.shape[1]
    prior = np.zeros((len(prior_root), width))
    prior[:, :len(prior_root)] = prior_root
    Q, root = np.linalg.qr(np.vstack((design, prior)), mode="reduced")
    rhs = np.vstack((samples, prior_root @ prior_mean))
    mean = solve_triangular(root, Q.T @ rhs)
    return mean, root


def _conditional_solver_record(noise):
    optimization = noise.provenance["optimization"]
    method = ("none" if optimization["strategy"] == "burg_only" else
              "no_AR_shape_parameters" if not noise.order else "L-BFGS-B")
    gradient_source = None
    if method == "L-BFGS-B":
        from transfer.continuous_gradient import IMPLEMENTATION_SHA256
        gradient_source = IMPLEMENTATION_SHA256
    return {"method": method, "parameterization": "actual_PARCOR_tanh_eta",
            "eta_max": optimization["eta_max"], "max_iterations": _CONDITIONAL_SOLVER_SETTINGS["maxiter"],
            "curvature_history": _CONDITIONAL_SOLVER_SETTINGS["maxcor"],
            "ftol": _CONDITIONAL_SOLVER_SETTINGS["ftol"], "gtol": _CONDITIONAL_SOLVER_SETTINGS["gtol"],
            "jacobian": "torch_CPU_float64_exact_autograd" if gradient_source else None,
            "gradient_implementation_sha256": gradient_source,
            "optimized_objective_scale": "per_scored_row_centered_at_Burg"}


def _conditional_prediction(samples, designs, noise, prefix, prior_mean, prior_covariance, *, posterior=False):
    """Condition all record prefixes jointly, retaining the same drift measure.

    The first coordinates are one shared response kernel. Each independent
    record has its own diffuse intercept/trend. The actual full-record AR map
    acts on Y, D and Z together; only subsequent rows enter this likelihood.
    """
    kernel_width, channels = prior_mean.shape
    width = kernel_width + 2 * len(samples)
    prefix_design, prefix_y, tail_design, tail_y = [], [], [], []
    for index, (y, design) in enumerate(zip(samples, designs, strict=True)):
        # One affine coordinate system for both prefix and tail. Centering its
        # observed prefix keeps drift information well scaled on long records.
        # The same diffuse measure change cancels in the conditional density.
        drift = np.column_stack((np.ones(len(y)), (2 * np.arange(len(y)) - (prefix - 1)) / (prefix - 1)))
        transformed = noise.whiten(np.column_stack((y, design, drift)))
        matrix = np.zeros((len(y), width))
        matrix[:, :kernel_width] = transformed[:, channels:channels + kernel_width]
        matrix[:, kernel_width + 2 * index:kernel_width + 2 * index + 2] = transformed[:, -2:]
        prefix_design.append(matrix[:prefix])
        prefix_y.append(transformed[:prefix, :channels])
        tail_design.append(matrix[prefix:])
        tail_y.append(transformed[prefix:, :channels])
    A_u, Y_u, A_s, Y_s = map(np.vstack, (prefix_design, prefix_y, tail_design, tail_y))
    prior_root = solve_triangular(np.linalg.cholesky(prior_covariance), np.eye(kernel_width), lower=True)
    prefix_mean, prefix_information_root = _joint_posterior(A_u, Y_u, prior_mean, prior_root)
    tail_q, full_information_root = np.linalg.qr(np.vstack((A_s, prefix_information_root)), mode="reduced")
    full_mean = solve_triangular(full_information_root,
        tail_q.T @ np.vstack((Y_s, prefix_information_root @ prefix_mean)))
    remainder = Y_s - A_s @ full_mean
    parameter_residual = prefix_information_root @ (full_mean - prefix_mean)
    scatter = remainder.T @ remainder + parameter_residual.T @ parameter_residual
    logdet = 2 * float(np.log(np.abs(np.diag(full_information_root))).sum()
                      - np.log(np.abs(np.diag(prefix_information_root))).sum())
    mean = covariance = None
    if posterior:
        full_root = solve_triangular(full_information_root, np.eye(width))
        mean = full_mean[:kernel_width]
        covariance = full_root[:kernel_width] @ full_root[:kernel_width].T
    return scatter, logdet, len(Y_s), mean, covariance


def _fit_fixed_prefix(samples, designs, residuals, ids, *, order, prefix, ridge, strategy, fit_scope):
    """Exact conditional objective; Burg shape or locally optimized PARCOR.

    Prefix data updates the shared kernel and diffuse drift posterior. Spatial
    profiling and the AR objective use the common scored tail only. Burg has
    different lag-pair counts by order and is recorded solely as initialization.
    """
    initializer = TemporalNoise.fit(residuals, order=order, record_ids=ids,
        fit_scope=fit_scope, startup_mode="stationary_innovations")
    kernel_width, channels = designs[0].shape[1], samples[0].shape[1]
    prior_mean, prior_covariance = np.zeros((kernel_width, channels)), np.eye(kernel_width) / ridge
    training_digest = array_digest(*[value for pair in zip(samples, designs, strict=True) for value in pair])
    evaluations = 0

    def evaluate(reflections, *, posterior=False):
        nonlocal evaluations
        evaluations += 1
        noise = TemporalNoise.from_reflections(reflections, innovation_variance=1., reference_id="conditional_objective")
        scatter, logdet, n_tail, mean, covariance = _conditional_prediction(
            samples, designs, noise, prefix, prior_mean, prior_covariance, posterior=posterior)
        spatial = (scatter + scatter.T) / (2 * n_tail)
        try:
            spatial_root = np.linalg.cholesky(spatial)
        except np.linalg.LinAlgError as error:
            raise UnidentifiableFitError("conditional innovation covariance is not identifiable") from error
        objective = .5 * (n_tail * 2 * np.log(np.diag(spatial_root)).sum() + channels * logdet)
        if not np.isfinite(objective):
            raise UnidentifiableFitError("conditional Gaussian objective is nonfinite")
        return float(objective), spatial, n_tail, mean, covariance

    reflections = np.asarray(initializer.reflection_coefficients)
    initial_result = evaluate(reflections)
    initial_objective, _, objective_rows, _, _ = initial_result
    eta_max = _CONDITIONAL_SOLVER_SETTINGS["eta_max"]
    optimization = {"strategy": "burg_only" if strategy == "burg" else "conditional_profile",
                    "initial_objective": initial_objective, "final_objective": initial_objective,
                    "converged": True, "iterations": 0, "evaluations": 1, "eta_max": eta_max,
                    "status": 0, "message": "actual Burg reflections retained"}
    if strategy == "conditional_profile" and order:
        from transfer.continuous_gradient import ConditionalGaussianObjective

        eta = np.arctanh(reflections)
        if np.any(np.abs(eta) > eta_max):
            raise UnidentifiableFitError("Burg initializer is outside the declared conditional PARCOR search domain")
        gradient_objective = ConditionalGaussianObjective(samples, designs, conditioning_samples=prefix, ridge=ridge)
        evaluations += 1
        initial_torch, _ = gradient_objective.value_gradient(eta)
        if gradient_objective.objective_rows != objective_rows or not np.isclose(
            initial_torch * objective_rows, initial_objective, rtol=1e-11, atol=1e-8
        ):
            raise UnidentifiableFitError("exact-gradient and frozen Gaussian objectives disagree at initialization")

        def value_gradient(point):
            nonlocal evaluations
            evaluations += 1
            value, gradient = gradient_objective.value_gradient(point)
            return value - initial_torch, gradient

        result = minimize(value_gradient, eta, jac=True,
            method="L-BFGS-B", bounds=[(-eta_max, eta_max)] * order,
            options={name: _CONDITIONAL_SOLVER_SETTINGS[name] for name in ("maxiter", "maxcor", "ftol", "gtol")})
        if not result.success:
            raise UnidentifiableFitError(f"conditional PARCOR optimization did not converge: {result.message}")
        reflections = np.tanh(result.x)
        evaluations += 1
        final_torch, _ = gradient_objective.value_gradient(result.x)
        final_numpy = evaluate(reflections)[0]
        if not np.isclose(final_torch * objective_rows, final_numpy, rtol=1e-11, atol=1e-8):
            raise UnidentifiableFitError("exact-gradient optimum differs from the frozen Gaussian objective")
        optimization.update(converged=bool(result.success), iterations=int(result.nit),
                            status=int(result.status), message=str(result.message))
    elif strategy == "conditional_profile":
        optimization["message"] = "no AR shape parameters"
    objective, spatial, n_tail, mean, covariance = evaluate(reflections, posterior=True)
    if objective > initial_objective + 1e-9 * max(1., abs(initial_objective)):
        raise UnidentifiableFitError("conditional PARCOR optimization increased the exact Gaussian objective")
    optimization.update(final_objective=objective, evaluations=evaluations)
    noise = TemporalNoise.from_conditional_fit(initializer, reflections,
        innovation_variance=float(np.trace(spatial) / channels), conditioning_samples=prefix,
        objective_rows=n_tail, optimization=optimization, training_input_sha256=training_digest)
    return mean, covariance, spatial, noise, training_digest


@dataclass(frozen=True)
class ContinuousDecisionModel:
    basis: np.ndarray
    kernel_mean: np.ndarray
    kernel_row_covariance: np.ndarray
    innovation_covariance: np.ndarray
    phi: float | None
    sfreq: float
    channels: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    operator_identity: str
    provenance: dict
    noise: TemporalNoise | None = None
    conditioning_samples: int = 0
    _integrity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        basis = _real_values(self.basis, "ERP basis")
        mean = _real_values(self.kernel_mean, "kernel mean")
        row_cov = validate_covariance(_real_values(self.kernel_row_covariance, "kernel covariance"), name="kernel covariance")
        noise = validate_covariance(_real_values(self.innovation_covariance, "innovation covariance"), name="innovation covariance")
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
        if not np.isfinite(self.sfreq) or self.sfreq <= 0:
            raise ValueError("invalid sampling frequency")
        if self.noise is None:
            if self.phi is None or not np.isfinite(self.phi) or abs(self.phi) >= 1:
                raise ValueError("invalid legacy AR parameter")
        else:
            if (not isinstance(self.noise, TemporalNoise) or self.phi is not None
                    or self.noise.startup_mode != "stationary_innovations"):
                raise ValueError("explicit AR expert requires stationary innovations and no legacy phi")
            self.noise.assert_integrity()
        if (type(self.conditioning_samples) is not int or self.conditioning_samples < 0
                or (self.conditioning_samples and (self.noise is None or self.conditioning_samples < max(2, self.noise.order)))):
            raise ValueError("conditioning prefix must identify diffuse drift and cover the AR history")
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
            if self.noise is not None and self.noise.provenance["fit_scope"] != scope:
                raise ValueError("analytic continuous reference requires analytic temporal noise")
        else:
            if scope not in {"source_training", "target_calibration"} or not provenance.get("records"):
                raise ValueError("continuous model requires a nonempty complete training ledger")
            required = {"record_id", "n_samples", "n_events", "n_targets", "signal_sha256", "input_sha256",
                        "right_censored", "right_censored_events"}
            if self.noise is not None:
                required |= {"read_sample_support", "input_dtypes", "raw_input_sha256"}
            if self.conditioning_samples:
                required |= {"conditioning_sample_support", "score_sample_support"}
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
                if self.noise is not None:
                    if record["read_sample_support"] != [0, record["n_samples"]]:
                        raise ValueError("continuous AR training support differs from the recorded observations")
                    dtypes = record["input_dtypes"]
                    if (set(dtypes) != {"samples", "onsets", "target_mask"}
                            or np.dtype(dtypes["samples"]).kind not in "fiu"
                            or np.dtype(dtypes["onsets"]).kind not in "iu"
                            or np.dtype(dtypes["target_mask"]).kind not in "bfiu"):
                        raise ValueError("continuous AR ledger lacks exact numeric input dtypes")
                    require_sha256(record["raw_input_sha256"], "continuous raw fit input")
                if self.conditioning_samples and (
                    record["conditioning_sample_support"] != [0, self.conditioning_samples]
                    or record["score_sample_support"] != [self.conditioning_samples, record["n_samples"]]
                    or record["n_samples"] <= self.conditioning_samples
                ):
                    raise ValueError("continuous conditioning/score support differs from the actual record")
            require_sha256(provenance.get("implementation_sha256"), "continuous implementation")
            if not np.isfinite(provenance.get("ridge", float("nan"))) or provenance["ridge"] <= 0:
                raise ValueError("continuous fit ledger lacks positive ridge")
            if self.noise is not None:
                noise_provenance = self.noise.provenance
                if (noise_provenance["fit_scope"] != scope
                        or noise_provenance["n_columns"] != len(channels)
                        or [(r["record_id"], r["n_samples"]) for r in noise_provenance["records"]]
                        != [(r["record_id"], r["n_samples"]) for r in provenance["records"]]):
                    raise ValueError("continuous response and temporal noise training ledgers differ")
                if (provenance["implementation_sha256"] != _IMPLEMENTATION_SHA256
                        or provenance.get("compute_dtype") != _COMPUTE_DTYPE
                        or provenance.get("noise_order") != self.noise.order
                        or provenance.get("startup_mode") != self.noise.startup_mode
                        or provenance.get("kernel_prior_scale") != "innovation"):
                    raise ValueError("continuous AR source, dtype or noise/prior contract differs")
                if self.conditioning_samples and (
                    noise_provenance.get("conditioning_samples") != self.conditioning_samples
                    or provenance.get("conditioning_samples") != self.conditioning_samples
                    or provenance.get("conditional_training_input_sha256") != noise_provenance.get("training_input_sha256")
                    or provenance.get("kernel_observation_support") != "all_authorized_prefix_and_tail"
                    or noise_provenance.get("objective_rows") != sum(r["n_samples"] - self.conditioning_samples for r in provenance["records"])
                    or not np.isclose(self.noise.innovation_variance, float(np.trace(noise) / len(channels)), rtol=1e-13, atol=0)
                ):
                    raise ValueError("continuous conditional objective, covariance and training support differ")
                if not self.conditioning_samples and noise_provenance.get("conditioning_samples"):
                    raise ValueError("conditional noise cannot be relabeled as a stationary model")
                if self.conditioning_samples:
                    actual_strategy = noise_provenance["optimization"]["strategy"]
                    expected_estimator = "burg" if actual_strategy == "burg_only" else "conditional_profile"
                    if (provenance.get("noise_estimator") != expected_estimator
                            or provenance.get("noise_solver") != _conditional_solver_record(self.noise)):
                        raise ValueError("conditional estimator and solver differ from the actual noise fit")
        object.__setattr__(self, "provenance", freeze_metadata(provenance))
        object.__setattr__(self, "_integrity_sha256", metadata_digest(self._metadata()))

    @classmethod
    def fit(cls, records, *, sfreq, channels, candidate_ids, operator_identity,
            basis=None, ridge=1., fit_scope="target_calibration", noise_order=None,
            conditioning_samples=0, noise_estimator="burg"):
        """Fit all response and noise state from declared labeled records only.

        Records are independent legal scopes with complete stimulus histories.
        Explicit right_censored calibration ends before an unlabelled flash;
        response design rows beyond that authorized support are unobserved.
        Every supplied event retains its authorized label. Scoring still requires
        complete response tails and never inherits this calibration permission.
        The caller must enforce physical participant/time/label authorization;
        record IDs and digests retain the concrete input used here. Ridge and
        dictionary must be fixed before held-out scoring.

        Omitting noise_order executes the original clipped AR1 fit exactly.
        An explicit integer order selects one feasible GLS update using pooled
        Burg residual noise and a stationary latent startup. Ridge refers to
        the unit-innovation row prior; the channel covariance owns voltage
        scale. A fixed conditioning prefix instead integrates one shared kernel
        and per-record diffuse drift jointly; its spatial profile and optional
        PARCOR optimization use only the common tail. Kernel information comes
        from all authorized prefix and tail data. Optimizing that exact objective
        reports local convergence, without a global-MLE claim.
        """
        if fit_scope not in {"target_calibration", "source_training"}:
            raise ValueError("continuous model fit scope must be training/calibration")
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be strictly positive")
        if noise_order is not None and (type(noise_order) is not int or not 0 <= noise_order <= 128):
            raise ValueError("explicit AR order must be an integer in 0..128")
        if (type(conditioning_samples) is not int or conditioning_samples < 0
                or (conditioning_samples and (noise_order is None or conditioning_samples < max(2, noise_order)))):
            raise ValueError("conditioning prefix must cover explicit AR order and identify diffuse drift")
        if noise_estimator not in {"burg", "conditional_profile"} or (not conditioning_samples and noise_estimator != "burg"):
            raise ValueError("conditional profile noise estimation requires a fixed conditioning prefix")
        basis = smooth_erp_basis(sfreq) if basis is None else _real_values(basis, "ERP basis")
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
            if conditioning_samples and len(X) <= conditioning_samples:
                raise ValueError("conditioning prefix must leave observed tail samples")
            all_labels.extend(labels.tolist())
            ledger.append({"record_id": record.record_id, "n_samples": len(X),
                           "n_events": len(onsets), "n_targets": int(labels.sum()),
                           "right_censored": record.right_censored,
                           "right_censored_events": int(np.sum(onsets + len(basis) > len(X))),
                           "signal_sha256": array_digest(X, onsets),
                           "input_sha256": array_digest(X, onsets, labels)})
            if noise_order is not None:
                original = {"samples": np.asarray(record.samples), "onsets": np.asarray(record.onsets),
                            "target_mask": np.asarray(record.target_mask)}
                ledger[-1].update(read_sample_support=[0, len(X)],
                                  input_dtypes={key: value.dtype.str for key, value in original.items()},
                                  raw_input_sha256=array_digest(*original.values()))
            if conditioning_samples:
                ledger[-1].update(conditioning_sample_support=[0, conditioning_samples],
                                  score_sample_support=[conditioning_samples, len(X)])
        if set(all_labels) != {0, 1}:
            raise ValueError("both target and non-target calibration events are required")
        p = 2 * basis.shape[1]

        def solve(phi):
            if isinstance(phi, TemporalNoise):
                transforms = [_stationary_transform(phi, len(x)) for x in samples]
                D = np.vstack([transform(d) for transform, d in zip(transforms, designs, strict=True)])
                Y = np.vstack([transform(x) for transform, x in zip(transforms, samples, strict=True)])
            else:
                D = np.vstack([_whiten_and_project(d, phi) for d in designs])
                Y = np.vstack([_whiten_and_project(x, phi) for x in samples])
            gram = D.T @ D
            if noise_order is None:
                row_cov = np.linalg.inv(gram + ridge * np.eye(p))
                mean = row_cov @ D.T @ Y
            else:
                # Augmented QR does not square the conditioning of the design.
                Q, root = np.linalg.qr(np.vstack([D, np.sqrt(ridge)*np.eye(p)]), mode="reduced")
                inverse_root = solve_triangular(root, np.eye(p))
                row_cov = inverse_root @ inverse_root.T
                mean = solve_triangular(root, Q[:len(Y)].T @ Y)
            return D, Y, mean, (row_cov + row_cov.T) / 2, gram

        D, Y, initial, _, _ = solve(0.)
        initial_residual = Y - D @ initial
        offsets = np.cumsum([0] + [len(x) for x in samples])
        pairs = [initial_residual[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
        if conditioning_samples:
            mean, row_cov, noise, noise_model, training_digest = _fit_fixed_prefix(
                samples, designs, pairs, ids, order=noise_order, prefix=conditioning_samples,
                ridge=ridge, strategy=noise_estimator, fit_scope=fit_scope)
            provenance = {"fit_scope": fit_scope, "records": ledger, "ridge": float(ridge),
                "implementation_sha256": _IMPLEMENTATION_SHA256, "compute_dtype": _COMPUTE_DTYPE,
                "noise_order": noise_model.order, "startup_mode": noise_model.startup_mode,
                "kernel_prior_scale": "innovation", "conditioning_samples": conditioning_samples,
                "conditional_training_input_sha256": training_digest,
                "kernel_observation_support": "all_authorized_prefix_and_tail",
                "kernel_uncertainty": "joint_matrix_normal_posterior_given_all_authorized_C",
                "noise": "fixed_prefix_conditional_spatial_profile_after_Burg_initialization",
                "noise_estimator": noise_estimator, "nuisance": "per_record_diffuse_constant_and_prefix_centered_linear_trend",
                "noise_solver": _conditional_solver_record(noise_model),
                "score_estimand": "conditional_tail_with_fixed_candidate_prior",
                "hyperparameter_uncertainty_integrated": False, "global_MLE_claimed": False,
                "likelihood_assumptions": ["linear_event_superposition", "shared_class_noise",
                    "stationary_shared_AR_reference", "complete_event_history", "fixed_conditioning_prefix"]}
            return cls(basis, mean, row_cov, noise, None, sfreq, tuple(channels), tuple(candidate_ids),
                       operator_identity, provenance, noise=noise_model, conditioning_samples=conditioning_samples)
        if noise_order is None:
            numerator = sum(float(np.sum(r[1:] * r[:-1])) for r in pairs)
            denominator = sum(float(np.sum(r[:-1] ** 2)) for r in pairs)
            phi = float(np.clip(numerator / max(denominator, np.finfo(float).tiny), -0.95, 0.95))
            noise_model = None
            D, Y, mean, row_cov, gram = solve(phi)
        else:
            phi = None
            noise_model = TemporalNoise.fit(pairs, order=noise_order, record_ids=ids,
                                            fit_scope=fit_scope, startup_mode="stationary_innovations")
            D, Y, mean, row_cov, gram = solve(noise_model)
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
        provenance = {
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
                   }
        if noise_model is not None:
            provenance.update(noise="segmented_Burg_AR_times_channel_covariance",
                              noise_order=noise_model.order, startup_mode=noise_model.startup_mode,
                              kernel_prior_scale="innovation", compute_dtype=_COMPUTE_DTYPE,
                              noise_fit_passes=1, monotone_likelihood_optimization_claimed=False,
                              likelihood_assumptions=["linear_event_superposition", "shared_class_noise",
                                                      "stationary_shared_AR_spectrum", "complete_event_history"])
        return cls(basis, mean, row_cov, noise, phi, sfreq, tuple(channels),
                   tuple(candidate_ids), operator_identity, provenance, noise=noise_model)

    @property
    def noise_order(self):
        return 1 if self.noise is None else self.noise.order

    def score(self, samples, onsets, membership, *, operator_identity, sfreq, channels,
              candidate_ids, record_id):
        """Score a complete fixed vocabulary using public membership, never truth.

        The supplied record is one independent decision scope. Onsets must cover
        all response-generating events; unknown adjacent decisions must be
        handled by the caller or excluded by predeclared support. No state is
        updated. No oracle kernel or Gamma is accepted by this interface.
        log_evidence omits constants shared by this record's candidates; it
        supports candidate normalization, not cross-model marginal evidence.
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
        if self.conditioning_samples and len(X) <= self.conditioning_samples:
            raise ValueError("conditioning prefix must leave observed tail samples")
        if array_digest(X, onsets) in {r["signal_sha256"] for r in training_records}:
            raise ValueError("prediction signal repeats an exact calibration record")
        member = np.asarray(membership)
        if member.shape != (len(onsets), len(self.candidate_ids)) or not np.all(np.isin(member, [0, 1])):
            raise ValueError("membership must include the complete frozen candidate vocabulary")
        if np.any(member.sum(axis=1) == 0) or np.any(member.sum(axis=1) == len(self.candidate_ids)):
            raise ValueError("every event must flash a proper nonempty candidate subset")
        if np.any(member.sum(axis=0) == 0):
            raise ValueError("missing candidate evidence: refuse incomplete decision")
        if self.conditioning_samples:
            spatial_root = np.linalg.cholesky(self.innovation_covariance)
            logdet_spatial = 2 * float(np.log(np.diag(spatial_root)).sum())
            evidence = []
            for candidate in range(len(self.candidate_ids)):
                design = _event_design(onsets, member[:, candidate], len(X), self.basis)
                scatter, logdet, n_tail, _, _ = _conditional_prediction(
                    [X], [design], self.noise, self.conditioning_samples, self.kernel_mean, self.kernel_row_covariance)
                spatial_precision_scatter = solve_triangular(spatial_root, scatter, lower=True)
                spatial_precision_scatter = solve_triangular(spatial_root.T, spatial_precision_scatter, lower=False)
                evidence.append(-.5 * (float(np.trace(spatial_precision_scatter)) + len(self.channels) * logdet
                    + n_tail * (logdet_spatial + len(self.channels) * np.log(2 * np.pi))))
            evidence = np.asarray(evidence)
            probability = np.exp(evidence - logsumexp(evidence))
            best = int(np.argmax(evidence))
            tied = np.count_nonzero(np.isclose(evidence, evidence[best], rtol=0, atol=1e-10)) > 1
            return ContinuousDecisionResult(self.candidate_ids, frozen_array(evidence), frozen_array(probability),
                None if tied else self.candidate_ids[best], "tied_evidence" if tied else "ok", len(X),
                array_digest(X, onsets, member), conditioning_samples=self.conditioning_samples)
        transform = (_stationary_transform(self.noise, len(X)) if self.noise is not None
                     else lambda value: _whiten_and_project(value, self.phi))
        Y = transform(X)
        spatial_root = np.linalg.cholesky(self.innovation_covariance)
        kernel_root = np.linalg.cholesky(self.kernel_row_covariance)
        evidence = []
        for candidate in range(len(self.candidate_ids)):
            design = _event_design(onsets, member[:, candidate], len(X), self.basis)
            D = transform(design)
            residual = Y - D @ self.kernel_mean
            residual = solve_triangular(spatial_root, residual.T, lower=True).T
            U = D @ kernel_root
            small = np.eye(U.shape[1]) + U.T @ U
            root = np.linalg.cholesky(small)
            projected = solve_triangular(root, U.T @ residual, lower=True)
            if self.noise is None:
                mahalanobis = float(np.sum(residual**2) - np.sum(projected**2))
                if mahalanobis < -1e-8 * max(1., float(np.sum(residual**2))):
                    raise ValueError("unstable Gaussian covariance solve")
                mahalanobis = max(0., mahalanobis)
            else:
                # Equivalent Woodbury quadratic as a sum of squares avoids
                # subtracting two nearly equal, large signal energies.
                coefficient = solve_triangular(root.T, projected, lower=False)
                mahalanobis = float(np.sum((residual - U @ coefficient)**2) + np.sum(coefficient**2))
            logdet = len(self.channels) * 2 * float(np.log(np.diag(root)).sum())
            evidence.append(-0.5 * (mahalanobis + logdet))
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
        state = {key: getattr(self, key) for key in _STATE_KEYS}
        metadata = {"sfreq": self.sfreq, "channels": self.channels,
                "candidate_ids": self.candidate_ids, "operator_identity": self.operator_identity,
                "provenance": self.provenance,
                "state_sha256": array_digest(*state.values())}
        if self.noise is None:
            metadata.update(schema="n2p3_continuous_decision_model/2", phi=self.phi)
        else:
            metadata.update(schema="n2p3_continuous_decision_model/3", noise=self.noise.record(),
                            state_dtypes={key: value.dtype.str for key, value in state.items()})
            if self.conditioning_samples:
                metadata.update(schema="n2p3_continuous_decision_model/4", conditioning_samples=self.conditioning_samples)
        return metadata

    def digest(self):
        self.assert_integrity()
        return self._integrity_sha256

    def assert_integrity(self):
        if metadata_digest(self._metadata()) != self._integrity_sha256:
            raise ValueError("frozen continuous model integrity changed")

    def save(self, path):
        self.assert_integrity()
        state = {key: getattr(self, key) for key in _STATE_KEYS}
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
            state = [archive[key] for key in _STATE_KEYS]
        schema = metadata.get("schema")
        if schema not in {"n2p3_continuous_decision_model/2", "n2p3_continuous_decision_model/3", "n2p3_continuous_decision_model/4"}:
            raise ValueError("unsupported continuous decision model schema")
        expected = metadata.pop("artifact_sha256", None)
        if expected != metadata_digest(metadata):
            raise ValueError("continuous model metadata digest mismatch")
        if array_digest(*state) != metadata.get("state_sha256"):
            raise ValueError("continuous model state digest mismatch")
        explicit_ar = schema != "n2p3_continuous_decision_model/2"
        if explicit_ar and (metadata.get("state_dtypes") != dict.fromkeys(_STATE_KEYS, _COMPUTE_DTYPE)
                            or any(value.dtype.str != _COMPUTE_DTYPE for value in state)):
            raise ValueError("continuous AR artifact requires its exact float64 state dtype")
        noise = TemporalNoise.from_record(metadata["noise"]) if explicit_ar else None
        prefix = metadata["conditioning_samples"] if schema == "n2p3_continuous_decision_model/4" else 0
        if schema == "n2p3_continuous_decision_model/4" and not prefix:
            raise ValueError("conditional model schema requires its actual conditioning prefix")
        return cls(*state, None if explicit_ar else metadata["phi"], metadata["sfreq"], tuple(metadata["channels"]),
                   tuple(metadata["candidate_ids"]), metadata["operator_identity"], metadata["provenance"], noise=noise,
                   conditioning_samples=prefix)
