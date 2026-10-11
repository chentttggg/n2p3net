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

Actual fitted predictions consume the fixed-prior/all-C QR precision factor
and response. The serialized dense kernel covariance is a rounded diagnostic;
its small eigenvalues need not survive an ill-conditioned factor Gram product.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from math import ulp
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf

from research.execution import UnidentifiableFitError
from transfer.kernel_prior import (
    CompiledKernelPrior,
    calibration_reference_roots,
    compile_kernel_prior,
    resolve_kernel_prior,
)
from transfer.noise_model import (
    array_digest,
    freeze_metadata,
    frozen_array,
    metadata_digest,
    require_identifiers,
    require_sha256,
    validate_covariance,
)
from transfer.predictive_prior import PhysicalResponseMetric
from transfer.response_dictionary import ResponseDictionary
from transfer.temporal_noise import TemporalNoise

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_STATE_KEYS = ("basis", "kernel_mean", "kernel_row_covariance", "innovation_covariance")
_PRIOR_STATE_KEYS = ("prior_precision_root", "prior_reference_root", "prior_design_root", "prior_fit_information_root", "prior_fit_score")
_FIT_STATISTIC_STATE_KEYS = ("prior_fit_information_root", "prior_fit_score")
_POSTERIOR_STATE_KEYS = ("posterior_precision_root", "posterior_response", "posterior_orthogonal")
_DICTIONARY_ARRAY_KEYS = ("times", "ambient_matrix", "endpoint_rows", "constraint_map")
_DICTIONARY_GEOMETRY_KEYS = ("reference_information_root", "fitted_information_root",
                             "ambient_reference_information_root", "ambient_prior_precision", "original_reference_root")
def _dictionary_state_keys(calibration_prior):
    roots = ("ambient_reference_information_root", "ambient_prior_precision")
    if not calibration_prior:
        roots += ("reference_information_root", "original_reference_root")
    return tuple("dictionary_"+key for key in (*_DICTIONARY_ARRAY_KEYS, *roots))
_CALIBRATION_PRIOR_SCHEMAS = {
    "calibration_g": "n2p3_continuous_decision_model/5",
    "calibration_trace_isotropic": "n2p3_continuous_decision_model/6",
    "calibration_completed_g": "n2p3_continuous_decision_model/7",
}
_COMPUTE_DTYPE = np.dtype(np.float64).str
_CONDITIONAL_SOLVER_SETTINGS = freeze_metadata({"eta_max": 8., "maxiter": 1000, "maxcor": 32, "ftol": 1e-10, "gtol": 1e-6})
_POSTERIOR_BINDING = freeze_metadata({
    "schema": "n2p3_continuous_posterior_binding/1",
    "authority": "fixed_prior_and_all_C_actual_noise_QR_statistics",
    "prediction": "frozen_producer_triangular_precision_and_kernel_mean",
    "numerical_state": "producer_Q_R_response_preserved_without_QR_regeneration",
    "qr_admission": "column_scaled_factorization_backward_error_plus_entrywise_product_roundoff",
    "saved_covariance": "rounded_derived_factor_Gram_diagnostic",
    "covariance_admission": "entrywise_equilibrated_QR_replay_and_factor_product_resolution",
})


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


def _conditional_prediction(samples, designs, noise, prefix, prior_mean, prior_covariance, *, posterior=False,
                            prior_precision_root=None):
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
    prior_root = (solve_triangular(np.linalg.cholesky(prior_covariance), np.eye(kernel_width), lower=True)
                  if prior_precision_root is None else prior_precision_root)
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


def _fit_fixed_prefix(samples, designs, residuals, ids, *, order, prefix, ridge, strategy, fit_scope, kernel_prior=None):
    """Exact conditional objective; Burg shape or locally optimized PARCOR.

    Prefix data updates the shared kernel and diffuse drift posterior. Spatial
    profiling and the AR objective use the common scored tail only. Burg has
    different lag-pair counts by order and is recorded solely as initialization.
    """
    initializer = TemporalNoise.fit(residuals, order=order, record_ids=ids,
        fit_scope=fit_scope, startup_mode="stationary_innovations")
    kernel_width, channels = designs[0].shape[1], samples[0].shape[1]
    prior_mean = np.zeros((kernel_width, channels))
    prior_covariance = np.eye(kernel_width)/ridge if kernel_prior is None else None
    prior_arguments = {} if kernel_prior is None else {"prior_precision_root": kernel_prior.precision_root}
    training_digest = array_digest(*[value for pair in zip(samples, designs, strict=True) for value in pair])
    evaluations = 0

    def evaluate(reflections, *, posterior=False):
        nonlocal evaluations
        evaluations += 1
        noise = TemporalNoise.from_reflections(reflections, innovation_variance=1., reference_id="conditional_objective")
        scatter, logdet, n_tail, mean, covariance = _conditional_prediction(
            samples, designs, noise, prefix, prior_mean, prior_covariance, posterior=posterior, **prior_arguments)
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
        gradient_arguments = {"ridge": ridge} if kernel_prior is None else {"kernel_prior": kernel_prior}
        gradient_objective = ConditionalGaussianObjective(samples, designs, conditioning_samples=prefix, **gradient_arguments)
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
    objective, spatial, n_tail, _, _ = evaluate(reflections)
    if objective > initial_objective + 1e-9 * max(1., abs(initial_objective)):
        raise UnidentifiableFitError("conditional PARCOR optimization increased the exact Gaussian objective")
    optimization.update(final_objective=objective, evaluations=evaluations)
    noise = TemporalNoise.from_conditional_fit(initializer, reflections,
        innovation_variance=float(np.trace(spatial) / channels), conditioning_samples=prefix,
        objective_rows=n_tail, optimization=optimization, training_input_sha256=training_digest)
    statistics = _kernel_fit_statistics(samples, designs, noise, prefix)
    precision = np.sqrt(ridge)*np.eye(kernel_width) if kernel_prior is None else kernel_prior.precision_root
    root, response, covariance, _, orthogonal = _kernel_posterior_factors(precision,
        np.asarray(statistics["information_root"]), np.asarray(statistics["score"]), with_factor=True)
    mean = solve_triangular(root, response)
    posterior_state = {"precision_root": root, "response": response, "orthogonal": orthogonal}
    return mean, covariance, spatial, noise, training_digest, statistics, posterior_state


def _prior_binding(sfreq, channels, candidate_ids, operator_identity, basis, ledger, fit_scope, dictionary=None):
    result = {"sample_rate_hz": sfreq, "channels": list(channels), "candidate_ids": list(candidate_ids),
            "operator_identity": operator_identity, "basis_sha256": array_digest(basis),
            "fit_scope": fit_scope, "calibration_inputs": ledger}
    if dictionary is not None:
        result["dictionary_digest"] = dictionary.identity_digest
    return result


def _kernel_fit_statistics(samples, designs, noise, prefix):
    """Small actual C sufficient statistics bind the saved prior to its posterior."""
    width, channels = designs[0].shape[1], samples[0].shape[1]
    root = np.empty((0, width + channels))
    for y, d in zip(samples, designs, strict=True):
        drift = np.column_stack((np.ones(len(y)), (2*np.arange(len(y))-(prefix-1))/(prefix-1)))
        transformed = noise.whiten(np.column_stack((d, y, drift)))
        # Eliminate nuisance with orthogonal transformations and retain Q_D.T Y.
        # Forming D.T Y then solving normal equations squares the conditioning.
        quotient = np.linalg.qr(np.column_stack((transformed[:, -2:], transformed[:, :-2])), mode="r")[2:, 2:]
        root = np.linalg.qr(np.vstack((root, quotient)), mode="r")
    return _qr_fit_statistics(root, width, _fitted_noise_digest(noise))


def _fitted_noise_digest(noise, phi=None):
    return (noise.record()["artifact_sha256"] if noise is not None else
            metadata_digest({"program": "legacy_clipped_AR1_stationary_transform", "phi": phi}))


def _qr_fit_statistics(root, width, noise_digest):
    if len(root) < width:
        root = np.vstack((root, np.zeros((width-len(root), root.shape[1]))))
    information, response = root[:width, :width], root[:width, width:]
    return {"information_root": information.tolist(), "score": response.tolist(),
            "state_sha256": array_digest(information, response), "noise_digest": noise_digest,
            "method": "all_C_actual_noise_per_record_diffuse_QR_response"}


def _kernel_posterior_factors(prior_root, information, score, *, with_factor=False):
    """One full-C quotient posterior for both fitting and persisted relations."""
    q, root = np.linalg.qr(np.vstack((information, prior_root)), mode="reduced")
    response = q.T @ np.vstack((score, np.zeros_like(score)))
    inverse = solve_triangular(root, np.eye(len(root)))
    covariance = inverse@inverse.T
    return ((root, response, covariance, inverse, q) if with_factor
            else (root, response, covariance))


class KernelPosteriorResolutionError(UnidentifiableFitError):
    """The declared posterior cannot be certified at its numerical resolution."""

    def __init__(self, ledger):
        self._evidence = freeze_metadata({"kind": "kernel_posterior_numerical_resolution_failure", **ledger})
        super().__init__("posterior factor is unresolved after kernel-coordinate column equilibration")

    def record(self):
        return json.loads(json.dumps(self._evidence, allow_nan=False))


def _validate_prior_posterior(prior_root, statistics, mean, covariance, noise_digest, *, posterior_state=None):
    information = _real_values(statistics["information_root"], "fitted kernel information root")
    score = _real_values(statistics["score"], "fitted kernel score")
    width = len(prior_root)
    if (information.shape != (width, width) or score.shape != mean.shape
            or not np.isfinite(information).all() or not np.isfinite(score).all()
            or statistics.get("method") != "all_C_actual_noise_per_record_diffuse_QR_response"
            or statistics.get("state_sha256") != array_digest(information, score)
            or statistics.get("noise_digest") != noise_digest):
        raise ValueError("kernel prior posterior statistics differ from the fitted noise and coordinates")
    if posterior_state is None:
        root, response, expected_covariance, factor, orthogonal = _kernel_posterior_factors(
            prior_root, information, score, with_factor=True)
    else:
        if set(posterior_state) != {"precision_root", "response", "orthogonal"}:
            raise ValueError("saved posterior QR state differs from its declared numerical program")
        root, response, orthogonal = (_real_values(posterior_state[key], f"posterior {key}")
                                     for key in ("precision_root", "response", "orthogonal"))
        if (root.shape != (width, width) or response.shape != mean.shape
                or orthogonal.shape != (2*width, width)
                or not all(np.isfinite(value).all() for value in (root, response, orthogonal))
                or np.any(np.tril(root, -1)) or np.any(np.diag(root) == 0)):
            raise ValueError("saved posterior QR state differs from actual kernel coordinates")
        factor = solve_triangular(root, np.eye(width))
        expected_covariance = factor@factor.T
    # H is a rounded factor Gram, not a numerically trustworthy inverse in
    # every direction. Check each stored entry at its own product scale: a
    # huge weak variance cannot provide the tolerance for another diagonal.
    # Prediction consumes R and its bound producer mean, avoiding Cholesky(H)
    # and the ill-conditioned identity R H R.T = I altogether.
    covariance_scale = np.abs(factor)@np.abs(factor).T
    mean_scale = np.abs(root)@np.abs(mean)+np.abs(response)
    resolution = 128*np.finfo(float).eps*max(1, width)
    stacked = np.vstack((information, prior_root))
    column_scales = np.max(np.abs(stacked), axis=0)
    if np.any(column_scales == 0):
        raise KernelPosteriorResolutionError({"kernel_dimensions": width, "zero_precision_column": True})
    # Raw condition numbers would wrongly reject diag(1, 1e-100), which is an
    # exactly representable coordinate scale change. Only the column-equilibrated
    # precision system enters this replay error budget. Dense H is diagnostic;
    # this is not a claim to recover its tiny eigenvalues from rounded entries.
    singular = np.linalg.svd(stacked/column_scales, compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else np.inf
    replay_resolution = 128*np.finfo(float).eps*max(stacked.shape)*condition
    if not np.isfinite(replay_resolution) or replay_resolution >= .125:
        raise KernelPosteriorResolutionError({"kernel_dimensions": width,
            "equilibrated_precision_condition": condition if np.isfinite(condition) else None,
            "replay_resolution": replay_resolution if np.isfinite(replay_resolution) else None,
            "criterion": "equilibrated_QR_relative_perturbation_budget_below_one_eighth"})
    covariance_resolution = max(resolution, 2*replay_resolution/(1-replay_resolution))
    # Authenticate producer Q/R/response against actual prior+C. Orthogonality
    # has dimensionless scale; each matrix equation has its own product scale.
    # Neither a full Gram norm nor R H R.T can supply another direction's bound.
    qr_scale = np.abs(orthogonal)@np.abs(root)+np.abs(stacked)
    target = np.vstack((score, np.zeros_like(score)))
    response_scale = np.abs(orthogonal.T)@np.abs(target)+np.abs(response)
    qr_resolution = 128*np.finfo(float).eps*max(stacked.shape)
    # Explicit Q is itself computed from Householder reflectors. Its factor-
    # generation error is columnwise; |Q||R| alone only bounds the subsequent
    # matrix product and can assign eps**2 to a spurious near-zero Q entry.
    # Keep each authorized A column's own scale. A large unrelated column
    # cannot supply the budget for a genuinely small coordinate direction.
    column_backward = (qr_resolution*column_scales)*np.linalg.norm(stacked/column_scales, axis=0)
    factorization_bound = qr_resolution*qr_scale+column_backward[None, :]
    if (not np.isfinite(qr_scale).all() or not np.isfinite(response_scale).all()
            or not np.isfinite(factorization_bound).all()
            or np.any(np.abs(orthogonal.T@orthogonal-np.eye(width)) > qr_resolution)
            or np.any(np.abs(orthogonal@root-stacked) > factorization_bound)
            or np.any(np.abs(orthogonal.T@target-response) > qr_resolution*response_scale)):
        raise ValueError("saved posterior QR differs from actual prior and fitted C information/response")
    if (not np.isfinite(expected_covariance).all() or not np.isfinite(covariance_scale).all()
            or not np.isfinite(mean_scale).all()):
        raise UnidentifiableFitError("posterior diagnostic relation is not representable in float64")
    if np.any(np.abs(covariance-expected_covariance) > covariance_resolution*covariance_scale):
        raise ValueError("saved kernel posterior covariance differs from its declared producer factor Gram")
    if np.any(np.abs(root@mean-response) > resolution*mean_scale):
        raise ValueError("saved kernel posterior mean differs from its frozen producer QR response")
    return root, response, orthogonal


def _fraction_gram(array):
    """Exact real Gram of the declared binary64 factor, without float squaring."""
    rows = [[Fraction.from_float(float(value)) for value in row] for row in array]
    width = array.shape[1]
    return [[sum((row[i]*row[j] for row in rows), Fraction()) for j in range(width)] for i in range(width)]


def _fraction_ldlt(matrix):
    """Exact SPD elimination; no numeric rank truncation or ridge is introduced."""
    width = len(matrix)
    lower = [[Fraction(i == j) for j in range(width)] for i in range(width)]
    diagonal = []
    for j in range(width):
        pivot = matrix[j][j]-sum((lower[j][k]**2*diagonal[k] for k in range(j)), Fraction())
        if pivot <= 0:
            raise UnidentifiableFitError("dictionary diagnostic requires its actual proper precision")
        diagonal.append(pivot)
        for i in range(j+1, width):
            lower[i][j] = (matrix[i][j]-sum((lower[i][k]*lower[j][k]*diagonal[k]
                                           for k in range(j)), Fraction()))/pivot
    return lower, diagonal


def _fraction_trace_solve(matrix, numerator):
    lower, diagonal = _fraction_ldlt(matrix)
    width = len(matrix)
    trace = Fraction()
    for column in range(width):
        forward = []
        for i in range(width):
            forward.append(numerator[i][column]-sum((lower[i][j]*forward[j] for j in range(i)), Fraction()))
        solution = [forward[i]/diagonal[i] for i in range(width)]
        for i in range(width-1, -1, -1):
            solution[i] -= sum((lower[j][i]*solution[j] for j in range(i+1, width)), Fraction())
        trace += solution[column]
    return trace


def _capacity_from_binary_factors(reference, fitted, prior):
    precision = _fraction_gram(prior)
    reference_information, fitted_information = _fraction_gram(reference), _fraction_gram(fitted)

    def df(information):
        total = [[p+g for p, g in zip(prior_row, data_row, strict=True)]
                 for prior_row, data_row in zip(precision, information, strict=True)]
        return _fraction_trace_solve(total, information)

    values = (df(reference_information), df(fitted_information),
              _fraction_trace_solve(precision, reference_information))
    try:
        rounded = tuple(float(value) for value in values)
    except OverflowError:
        rounded = tuple(None if value > Fraction.from_float(np.finfo(float).max) else float(value) for value in values)
    return rounded


@lru_cache(maxsize=32)
def _cached_capacity_binary_factors(reference_shape, reference_bytes, fitted_shape, fitted_bytes, prior_shape, prior_bytes):
    arrays = [np.frombuffer(raw, dtype="<f8").reshape(shape) for shape, raw in (
        (reference_shape, reference_bytes), (fitted_shape, fitted_bytes), (prior_shape, prior_bytes))]
    return _capacity_from_binary_factors(*arrays)


def _exact_capacity_diagnostics(reference, fitted, prior):
    """Correctly rounded scalar diagnostics; never controls posterior/score.

    Exact Fraction arithmetic is isolated to this small-matrix fit/load seam.
    A byte-bound LRU reuses fit/admission/replay summaries; larger payloads are
    calculated without retaining them, keeping cache memory bounded.
    """
    arrays = [np.ascontiguousarray(value, dtype="<f8") for value in (reference, fitted, prior)]
    if sum(value.nbytes for value in arrays) > 65536:
        return _capacity_from_binary_factors(*arrays)
    arguments = tuple(item for value in arrays for item in (value.shape, value.tobytes()))
    return _cached_capacity_binary_factors(*arguments)


def _dictionary_geometry_from_roots(dictionary, roots, prior_root, n_rows, n_records,
                                    *, restricted_precision_extension=False):
    p = dictionary.matrix.shape[1]*2
    ambient = dictionary.spec.n_basis*2
    arrays = {key: _real_values(roots[key], f"dictionary {key}") for key in _DICTIONARY_GEOMETRY_KEYS}
    for key, width in (("reference_information_root", p), ("fitted_information_root", p),
                       ("original_reference_root", p), ("ambient_reference_information_root", ambient)):
        if arrays[key].ndim != 2 or arrays[key].shape[1] != width or not np.isfinite(arrays[key]).all():
            raise ValueError("dictionary information roots differ from their actual function coordinates")
    ambient_precision = arrays["ambient_prior_precision"]
    if ambient_precision.shape != (ambient, ambient) or not np.isfinite(ambient_precision).all():
        raise ValueError("dictionary ambient prior precision differs from its function coordinates")
    embedding = np.kron(np.eye(2), dictionary.constraint_map)
    reference, actual, parent = (arrays[key] for key in ("reference_information_root", "fitted_information_root", "ambient_reference_information_root"))
    precision = prior_root.T@prior_root
    reference_information = reference.T@reference
    reference_scale = max(float(np.linalg.norm(reference_information)), float(np.linalg.norm(parent.T@parent)), np.finfo(float).tiny)
    precision_scale = max(float(np.linalg.norm(precision)), float(np.linalg.norm(ambient_precision)), np.finfo(float).tiny)
    if (not np.allclose(reference_information, embedding.T@(parent.T@parent)@embedding, rtol=2e-12, atol=reference_scale*2e-13)
            or not np.allclose(precision, embedding.T@ambient_precision@embedding, rtol=2e-12, atol=precision_scale*2e-13)):
        raise ValueError("dictionary constraints and prior precision do not obey the actual ambient pullback")
    if restricted_precision_extension and not np.allclose(
            ambient_precision, embedding@precision@embedding.T, rtol=2e-12, atol=precision_scale*2e-13):
        raise ValueError("dictionary completion precision extension is not zero on the constraint complement")
    df_ref, df_fit, prior_energy = _exact_capacity_diagnostics(reference, actual, prior_root)
    reference_scale = float(np.linalg.norm(arrays["original_reference_root"], ord=2))
    threshold = np.finfo(float).eps*max(n_rows, p)*reference_scale
    rank = int(np.sum(np.linalg.svd(reference, compute_uv=False) > threshold))
    return {"schema": "n2p3_dictionary_fit_geometry/3", "dictionary_digest": dictionary.identity_digest,
            "ambient_dimensions": ambient, "kernel_dimensions": p, "n_rows": n_rows, "n_records": n_records,
            "reference_geometry": "all_C_AR0_per_record_diffuse_affine", "fitted_geometry": "all_C_actual_noise_per_record_diffuse_affine",
            "df_ref": df_ref, "df_fit": df_fit, "reference_rank": rank, "rank_threshold": threshold,
            "reference_prior_energy": prior_energy,
            "capacity_changed_by_hard_endpoint": dictionary.spec.endpoint == "hard_zero",
            "capacity_matching_claimed": False,
            "prior_pullback": ("restricted_precision_extension_zero_on_constraint_complement"
                               if restricted_precision_extension else "embedding_T_ambient_precision_embedding"),
            "proper_on_actual_function_space": True,
            "diagnostic_resolution_rule": "exact_binary64_operand_Fraction_trace_with_two_ULP_scalar_storage",
            "diagnostic_status": {"df_ref": "certified_binary64_operands", "df_fit": "certified_binary64_operands",
                                  "reference_prior_energy": "unrepresentable_diagnostic" if prior_energy is None else "certified_binary64_operands"},
            "roots_sha256": array_digest(*arrays.values()), **{key: value.tolist() for key, value in arrays.items()}}


def _validate_dictionary_geometry(saved, actual):
    """Admit frozen producer diagnostics without rewriting their float64 values.

    Exact rational trace diagnostics refer to the actual saved binary64 roots.
    Two ULPs cover scalar transport rounding, not an ill-conditioned matrix
    calculation. An unrepresentable summary has an explicit diagnostic status
    and never changes an otherwise valid posterior/score. Spectral rank threshold
    remains a separately declared float64 numerical-resolution diagnostic.
    """
    scalar_keys = {"df_ref", "df_fit", "rank_threshold", "reference_prior_energy"}
    if (set(saved) != set(actual)
            or metadata_digest({k: v for k, v in saved.items() if k not in scalar_keys})
            != metadata_digest({k: v for k, v in actual.items() if k not in scalar_keys})):
        raise ValueError("dictionary capacity/constraint ledger differs from its actual numerical roots")
    resolution = 128*np.finfo(np.float64).eps*max(actual["n_rows"], actual["kernel_dimensions"])
    for key in scalar_keys:
        value, expected = saved[key], actual[key]
        if key == "reference_prior_energy" and expected is None:
            if value is not None:
                raise ValueError("dictionary capacity diagnostic differs from its declared unrepresentable state")
            continue
        tolerance = (resolution*max(abs(expected), np.finfo(float).tiny) if key == "rank_threshold"
                     else 2*ulp(expected))
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
                or value < 0 or abs(value-expected) > tolerance):
            raise ValueError("dictionary capacity diagnostic differs from its actual numerical relation")


def _fit_dictionary_geometry(dictionary, records, designs, actual_root, ridge, prior):
    reference, original = calibration_reference_roots(designs)
    ambient_designs = [_event_design(r.onsets, r.target_mask, len(r.samples), dictionary.ambient_matrix) for r in records]
    ambient_reference, _ = calibration_reference_roots(ambient_designs)
    width = 2*dictionary.spec.n_basis
    completion = prior is not None and prior.spec.kind == "calibration_completed_g"
    if completion:
        embedding = np.kron(np.eye(2), dictionary.constraint_map)
        ambient_precision = embedding@(prior.precision_root.T@prior.precision_root)@embedding.T
    elif prior is not None and prior.spec.kind == "calibration_g":
        ambient_precision = ambient_reference.T@ambient_reference/prior.spec.g
    else:
        # trace isotropic is calibrated in the declared restricted model. The
        # same actual scalar lifts to ambient coordinates; this is not a claim
        # that an unconstrained trace-control energy budget was preserved.
        scalar = ridge if prior is None else prior.derived_precision
        ambient_precision = scalar*np.eye(width)
    prior_root = np.sqrt(ridge)*np.eye(designs[0].shape[1]) if prior is None else prior.precision_root
    return _dictionary_geometry_from_roots(dictionary, {"reference_information_root": reference,
        "original_reference_root": original, "fitted_information_root": actual_root,
        "ambient_reference_information_root": ambient_reference, "ambient_prior_precision": ambient_precision},
        prior_root, sum(len(r.samples) for r in records), len(records),
        restricted_precision_extension=completion)


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
    kernel_prior: CompiledKernelPrior | None = None
    dictionary: ResponseDictionary | None = None
    dictionary_geometry: dict | None = None
    posterior_state: dict | None = field(default=None, repr=False, compare=False)
    _integrity_sha256: str = field(init=False, repr=False, compare=False)
    _posterior_precision_root: np.ndarray | None = field(init=False, repr=False, compare=False)
    _predictive_kernel_mean: np.ndarray = field(init=False, repr=False, compare=False)
    _fit_information_root: np.ndarray | None = field(init=False, repr=False, compare=False)
    _fit_score: np.ndarray | None = field(init=False, repr=False, compare=False)

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
        actual_posterior = self.provenance.get("fit_scope") in {"source_training", "target_calibration"}
        if not actual_posterior:
            np.linalg.cholesky(row_cov)
        for name, value in [("basis", basis), ("kernel_mean", mean),
                            ("kernel_row_covariance", row_cov), ("innovation_covariance", noise)]:
            object.__setattr__(self, name, frozen_array(value))
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "candidate_ids", candidates)
        provenance = json.loads(json.dumps(self.provenance, allow_nan=False))
        posterior_root = None
        predictive_mean = mean
        fit_information = fit_score = None
        posterior_state = None
        if self.dictionary is not None:
            if (not isinstance(self.dictionary, ResponseDictionary) or self.dictionary.sample_rate_hz != self.sfreq
                    or not np.array_equal(self.dictionary.matrix, basis) or self.dictionary_geometry is None):
                raise ValueError("owned physical dictionary differs from the actual response basis/grid")
            roots = {**self.dictionary_geometry,
                     "fitted_information_root": provenance["kernel_prior_fit_statistics"]["information_root"]}
            if self.kernel_prior is not None:
                roots.update(reference_information_root=self.kernel_prior.reference_root.tolist(),
                             original_reference_root=self.kernel_prior.design_root.tolist())
            prior_root = (np.sqrt(provenance["ridge"])*np.eye(p) if self.kernel_prior is None else self.kernel_prior.precision_root)
            geometry = _dictionary_geometry_from_roots(self.dictionary, roots, prior_root,
                sum(r["n_samples"] for r in provenance["records"]), len(provenance["records"]),
                restricted_precision_extension=(self.kernel_prior is not None
                                                and self.kernel_prior.spec.kind == "calibration_completed_g"))
            _validate_dictionary_geometry(self.dictionary_geometry, geometry)
            object.__setattr__(self, "dictionary_geometry", freeze_metadata(self.dictionary_geometry))
        elif self.dictionary_geometry is not None:
            raise ValueError("dictionary geometry requires its actual owned response function")
        scope = provenance.get("fit_scope")
        if self.kernel_prior is not None:
            if (not isinstance(self.kernel_prior, CompiledKernelPrior)
                    or not self.kernel_prior.spec.uses_calibration_reference or not self.conditioning_samples):
                raise ValueError("compiled calibration prior requires the explicit fixed-prefix program")
            self.kernel_prior.digest()
        if scope == "analytic_reference":
            if self.posterior_state is not None:
                raise ValueError("analytic reference cannot claim actual fitted posterior QR state")
            if self.kernel_prior is not None:
                raise ValueError("calibration prior models require an actual C ledger, not an analytic relabeling")
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
            if self.kernel_prior is not None:
                required |= {"design_sha256"}
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
                if self.kernel_prior is not None:
                    require_sha256(record["design_sha256"], "actual calibration design")
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
            if self.kernel_prior is None:
                if not np.isfinite(provenance.get("ridge", float("nan"))) or provenance["ridge"] <= 0:
                    raise ValueError("continuous fit ledger lacks positive ridge")
                if "kernel_prior_spec" in provenance:
                    raise ValueError("calibration prior program cannot be relabeled as the fixed ridge model")
            else:
                expected_binding = _prior_binding(self.sfreq, channels, candidates, self.operator_identity,
                                                  basis, provenance["records"], scope, self.dictionary)
                self.kernel_prior.validate_binding(expected_binding)
                if ("ridge" in provenance or provenance.get("kernel_prior_spec") != self.kernel_prior.spec.record()
                        or provenance.get("kernel_prior_digest") != self.kernel_prior.digest()
                        or self.kernel_prior.provenance["fit_scope"] != scope):
                    raise ValueError("compiled kernel prior differs from its actual C/basis/program binding")
            prior_root = (np.sqrt(provenance["ridge"])*np.eye(p) if self.kernel_prior is None
                          else self.kernel_prior.precision_root)
            if "kernel_prior_fit_statistics" not in provenance:
                raise ValueError("actual fitted posterior requires its complete C QR statistics")
            posterior_root, posterior_response, posterior_orthogonal = _validate_prior_posterior(
                prior_root, provenance["kernel_prior_fit_statistics"], mean, row_cov,
                _fitted_noise_digest(self.noise, self.phi), posterior_state=self.posterior_state)
            posterior_state = {"precision_root": frozen_array(posterior_root),
                "response": frozen_array(posterior_response), "orthogonal": frozen_array(posterior_orthogonal)}
            # Preserve the producer's already verified posterior mean. Re-solving
            # on another host could replace its numeric predictive state silently.
            predictive_mean = mean
            fit_information = np.asarray(provenance["kernel_prior_fit_statistics"]["information_root"], dtype=float)
            fit_score = np.asarray(provenance["kernel_prior_fit_statistics"]["score"], dtype=float)
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
        object.__setattr__(self, "posterior_state", None if posterior_state is None else freeze_metadata(posterior_state))
        object.__setattr__(self, "_posterior_precision_root", None if posterior_state is None else posterior_state["precision_root"])
        object.__setattr__(self, "_predictive_kernel_mean", frozen_array(predictive_mean))
        object.__setattr__(self, "_fit_information_root", None if fit_information is None else frozen_array(fit_information))
        object.__setattr__(self, "_fit_score", None if fit_score is None else frozen_array(fit_score))
        object.__setattr__(self, "_integrity_sha256", metadata_digest(self._metadata()))

    @classmethod
    def fit(cls, records, *, sfreq, channels, candidate_ids, operator_identity,
            basis=None, ridge=None, fit_scope="target_calibration", noise_order=None,
            conditioning_samples=0, noise_estimator="burg", kernel_prior=None, dictionary=None):
        """Fit all response and noise state from declared labeled records only.

        Records are independent legal scopes with complete stimulus histories.
        Explicit right_censored calibration ends before an unlabelled flash;
        response design rows beyond that authorized support are unobserved.
        Every supplied event retains its authorized label. Scoring still requires
        complete response tails and never inherits this calibration permission.
        The caller must enforce physical participant/time/label authorization;
        record IDs and digests retain the concrete input used here. Ridge and
        dictionary must be fixed before held-out scoring.

        Omitting noise_order retains the clipped AR1 estimation program. All
        stationary variants use the same producer QR for the numerical posterior.
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
        prior_spec = resolve_kernel_prior(kernel_prior, ridge)
        ridge = prior_spec.precision
        if noise_order is not None and (type(noise_order) is not int or not 0 <= noise_order <= 128):
            raise ValueError("explicit AR order must be an integer in 0..128")
        if (type(conditioning_samples) is not int or conditioning_samples < 0
                or (conditioning_samples and (noise_order is None or conditioning_samples < max(2, noise_order)))):
            raise ValueError("conditioning prefix must cover explicit AR order and identify diffuse drift")
        if noise_estimator not in {"burg", "conditional_profile"} or (not conditioning_samples and noise_estimator != "burg"):
            raise ValueError("conditional profile noise estimation requires a fixed conditioning prefix")
        if prior_spec.uses_calibration_reference and not conditioning_samples:
            raise ValueError("calibration priors are implemented only for fixed-prefix conditional inference")
        if dictionary is not None:
            if basis is not None:
                raise ValueError("owned dictionary and a separate numeric basis are mutually exclusive")
            if not isinstance(dictionary, ResponseDictionary) or dictionary.sample_rate_hz != sfreq:
                raise ValueError("owned dictionary sampling differs from the actual native input")
            dictionary.digest()
            basis = dictionary.matrix
        else:
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
            if prior_spec.uses_calibration_reference:
                ledger[-1]["design_sha256"] = array_digest(designs[-1])
        if set(all_labels) != {0, 1}:
            raise ValueError("both target and non-target calibration events are required")
        p = 2 * basis.shape[1]
        compiled_prior = None
        if prior_spec.uses_calibration_reference:
            binding = _prior_binding(sfreq, channels, candidate_ids, operator_identity, basis, ledger, fit_scope, dictionary)
            physical_metric = (PhysicalResponseMetric.from_basis(basis, sample_rate_hz=sfreq, binding=binding)
                               if prior_spec.kind == "calibration_completed_g" else None)
            compiled_prior = compile_kernel_prior(prior_spec, designs, record_ids=ids, fit_scope=fit_scope,
                binding=binding, physical_metric=physical_metric)

        def solve(phi):
            if isinstance(phi, TemporalNoise):
                transforms = [_stationary_transform(phi, len(x)) for x in samples]
                D = np.vstack([transform(d) for transform, d in zip(transforms, designs, strict=True)])
                Y = np.vstack([transform(x) for transform, x in zip(transforms, samples, strict=True)])
            else:
                D = np.vstack([_whiten_and_project(d, phi) for d in designs])
                Y = np.vstack([_whiten_and_project(x, phi) for x in samples])
            statistics = _qr_fit_statistics(np.linalg.qr(np.column_stack((D, Y)), mode="r"), p,
                _fitted_noise_digest(phi if isinstance(phi, TemporalNoise) else None,
                                     None if isinstance(phi, TemporalNoise) else phi))
            prior_root = np.sqrt(ridge)*np.eye(p) if compiled_prior is None else compiled_prior.precision_root
            root, response, row_cov, _, orthogonal = _kernel_posterior_factors(prior_root,
                np.asarray(statistics["information_root"]), np.asarray(statistics["score"]), with_factor=True)
            mean = solve_triangular(root, response)
            posterior_state = {"precision_root": root, "response": response, "orthogonal": orthogonal}
            # Fit, noise residuals and persistence consume this same posterior.
            # A second raw-row QR can differ in a tiny cancelling mean entry.
            return D, Y, mean, row_cov, statistics, posterior_state

        D, Y, initial, _, _, _ = solve(0.)
        initial_residual = Y - D @ initial
        offsets = np.cumsum([0] + [len(x) for x in samples])
        pairs = [initial_residual[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
        if conditioning_samples:
            mean, row_cov, noise, noise_model, training_digest, statistics, posterior_state = _fit_fixed_prefix(
                samples, designs, pairs, ids, order=noise_order, prefix=conditioning_samples,
                ridge=ridge, strategy=noise_estimator, fit_scope=fit_scope, kernel_prior=compiled_prior)
            provenance = {"fit_scope": fit_scope, "records": ledger,
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
            provenance["kernel_prior_fit_statistics"] = statistics
            if compiled_prior is None:
                provenance["ridge"] = float(ridge)
            else:
                provenance.update(kernel_prior_spec=prior_spec.record(), kernel_prior_digest=compiled_prior.digest(),
                    kernel_prior_fit_statistics=statistics)
            geometry = (None if dictionary is None else _fit_dictionary_geometry(dictionary, records, designs,
                np.asarray(statistics["information_root"]), ridge, compiled_prior))
            return cls(basis, mean, row_cov, noise, None, sfreq, tuple(channels), tuple(candidate_ids),
                       operator_identity, provenance, noise=noise_model, conditioning_samples=conditioning_samples,
                        kernel_prior=compiled_prior, dictionary=dictionary, dictionary_geometry=geometry,
                        posterior_state=posterior_state)
        if noise_order is None:
            numerator = sum(float(np.sum(r[1:] * r[:-1])) for r in pairs)
            denominator = sum(float(np.sum(r[:-1] ** 2)) for r in pairs)
            phi = float(np.clip(numerator / max(denominator, np.finfo(float).tiny), -0.95, 0.95))
            noise_model = None
            D, Y, mean, row_cov, statistics, posterior_state = solve(phi)
        else:
            phi = None
            noise_model = TemporalNoise.fit(pairs, order=noise_order, record_ids=ids,
                                            fit_scope=fit_scope, startup_mode="stationary_innovations")
            D, Y, mean, row_cov, statistics, posterior_state = solve(noise_model)
        residual = Y - D @ mean
        # tr(D.T D H) = ||R_D R_posterior^-1||_F^2. The producer's
        # information root avoids cancelling products of a large Gram and H,
        # which can invent degrees of freedom for duplicate basis columns.
        information = np.asarray(statistics["information_root"])
        smoothing_factor = solve_triangular(posterior_state["precision_root"].T, information.T, lower=True).T
        effective_parameters = float(np.sum(smoothing_factor**2))
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
        provenance["kernel_prior_fit_statistics"] = statistics
        geometry = (None if dictionary is None else _fit_dictionary_geometry(dictionary, records, designs,
            np.asarray(provenance["kernel_prior_fit_statistics"]["information_root"]), ridge, compiled_prior))
        return cls(basis, mean, row_cov, noise, phi, sfreq, tuple(channels),
                    tuple(candidate_ids), operator_identity, provenance, noise=noise_model,
                    dictionary=dictionary, dictionary_geometry=geometry, posterior_state=posterior_state)

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
                posterior_arguments = ({} if self._posterior_precision_root is None else
                                       {"prior_precision_root": self._posterior_precision_root})
                scatter, logdet, n_tail, _, _ = _conditional_prediction(
                    [X], [design], self.noise, self.conditioning_samples, self._predictive_kernel_mean,
                    self.kernel_row_covariance if self._posterior_precision_root is None else None,
                    **posterior_arguments)
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
        kernel_root = (np.linalg.cholesky(self.kernel_row_covariance)
                       if self._posterior_precision_root is None else None)
        evidence = []
        for candidate in range(len(self.candidate_ids)):
            design = _event_design(onsets, member[:, candidate], len(X), self.basis)
            D = transform(design)
            residual = Y - D @ self._predictive_kernel_mean
            residual = solve_triangular(spatial_root, residual.T, lower=True).T
            U = (D@kernel_root if self._posterior_precision_root is None else
                 solve_triangular(self._posterior_precision_root.T, D.T, lower=True).T)
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
        state = self._state_arrays()
        provenance = self.provenance
        if self._posterior_precision_root is not None:
            provenance = json.loads(json.dumps(self.provenance))
            provenance["kernel_prior_fit_statistics"].pop("information_root")
            provenance["kernel_prior_fit_statistics"].pop("score")
        metadata = {"sfreq": self.sfreq, "channels": self.channels,
                "candidate_ids": self.candidate_ids, "operator_identity": self.operator_identity,
                "provenance": provenance,
                "state_sha256": array_digest(*state.values())}
        if self._posterior_precision_root is not None:
            metadata.update(posterior_binding=_POSTERIOR_BINDING,
                            state_dtypes={key: value.dtype.str for key, value in state.items()})
        if self.noise is None:
            metadata.update(schema="n2p3_continuous_decision_model/2", phi=self.phi)
        else:
            metadata.update(schema="n2p3_continuous_decision_model/3", noise=self.noise.record(),
                            state_dtypes={key: value.dtype.str for key, value in state.items()})
            if self.conditioning_samples:
                metadata.update(schema="n2p3_continuous_decision_model/4", conditioning_samples=self.conditioning_samples)
            if self.kernel_prior is not None:
                prior = self.kernel_prior.record()
                prior.pop("precision_root")
                prior.pop("reference_root")
                prior.pop("design_root")
                metadata.update(schema=_CALIBRATION_PRIOR_SCHEMAS[self.kernel_prior.spec.kind], kernel_prior=prior)
        if self.dictionary is not None:
            record = self.dictionary.metadata()
            geometry = json.loads(json.dumps(self.dictionary_geometry))
            for key in _DICTIONARY_GEOMETRY_KEYS:
                geometry.pop(key)
            metadata.update(base_model_schema=metadata["schema"], schema="n2p3_continuous_dictionary_model/2",
                            dictionary=record, dictionary_geometry=geometry,
                            state_dtypes={key: value.dtype.str for key, value in state.items()})
        return metadata

    def _state_arrays(self):
        state = {key: getattr(self, key) for key in _STATE_KEYS}
        if self.kernel_prior is not None:
            state.update(prior_precision_root=self.kernel_prior.precision_root,
                prior_reference_root=self.kernel_prior.reference_root,
                prior_design_root=self.kernel_prior.design_root,
                prior_fit_information_root=self._fit_information_root,
                prior_fit_score=self._fit_score)
        elif self._posterior_precision_root is not None:
            state.update(prior_fit_information_root=self._fit_information_root, prior_fit_score=self._fit_score)
        if self.dictionary is not None:
            state.update({"dictionary_"+key: getattr(self.dictionary, key) for key in _DICTIONARY_ARRAY_KEYS})
            state.update({key: np.asarray(self.dictionary_geometry[key.removeprefix("dictionary_")], dtype=np.float64)
                          for key in _dictionary_state_keys(self.kernel_prior is not None) if key.removeprefix("dictionary_") in _DICTIONARY_GEOMETRY_KEYS})
        if self.posterior_state is not None:
            state.update({"posterior_"+key: value for key, value in self.posterior_state.items()})
        return state

    def digest(self):
        self.assert_integrity()
        return self._integrity_sha256

    def assert_integrity(self):
        if metadata_digest(self._metadata()) != self._integrity_sha256:
            raise ValueError("frozen continuous model integrity changed")

    def save(self, path):
        self.assert_integrity()
        state = self._state_arrays()
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
            schema = metadata.get("schema")
            if schema == "n2p3_continuous_dictionary_model/1":
                raise ValueError("unsupported frozen dictionary model schema; use its original execution source")
            owned_dictionary = schema == "n2p3_continuous_dictionary_model/2"
            if owned_dictionary:
                schema = metadata.get("base_model_schema")
            calibration_prior = schema in _CALIBRATION_PRIOR_SCHEMAS.values()
            posterior_bound = metadata.get("provenance", {}).get("fit_scope") in {"source_training", "target_calibration"}
            if posterior_bound and metadata.get("posterior_binding") != _POSTERIOR_BINDING:
                raise ValueError("unsupported fitted posterior binding protocol; use its original execution source")
            if not posterior_bound and "posterior_binding" in metadata:
                raise ValueError("analytic reference cannot declare an actual fitted posterior binding")
            state_keys = (_STATE_KEYS + (_PRIOR_STATE_KEYS if calibration_prior else ())
                          + (_FIT_STATISTIC_STATE_KEYS if (posterior_bound or owned_dictionary) and not calibration_prior else ())
                          + (_dictionary_state_keys(calibration_prior) if owned_dictionary else ())
                          + (_POSTERIOR_STATE_KEYS if posterior_bound else ()))
            if set(archive.files) != {*state_keys, "metadata"}:
                raise ValueError("continuous model numeric payload differs from its declared prior schema")
            all_state = {key: archive[key] for key in state_keys}
            state = [all_state[key] for key in _STATE_KEYS]
        schema = metadata.get("base_model_schema") if owned_dictionary else metadata.get("schema")
        if schema not in {"n2p3_continuous_decision_model/2", "n2p3_continuous_decision_model/3",
                          "n2p3_continuous_decision_model/4", *_CALIBRATION_PRIOR_SCHEMAS.values()}:
            raise ValueError("unsupported continuous decision model schema")
        expected = metadata.pop("artifact_sha256", None)
        if expected != metadata_digest(metadata):
            raise ValueError("continuous model metadata digest mismatch")
        explicit_ar = schema != "n2p3_continuous_decision_model/2"
        if (explicit_ar or owned_dictionary or posterior_bound) and (metadata.get("state_dtypes") != dict.fromkeys(state_keys, _COMPUTE_DTYPE)
                            or any(value.dtype.str != _COMPUTE_DTYPE for value in all_state.values())):
            raise ValueError("continuous AR artifact requires its exact float64 state dtype")
        if array_digest(*all_state.values()) != metadata.get("state_sha256"):
            raise ValueError("continuous model state digest mismatch")
        noise = TemporalNoise.from_record(metadata["noise"]) if explicit_ar else None
        conditional = schema == "n2p3_continuous_decision_model/4" or calibration_prior
        prefix = metadata["conditioning_samples"] if conditional else 0
        if conditional and not prefix:
            raise ValueError("conditional model schema requires its actual conditioning prefix")
        prior = None
        if calibration_prior:
            prior_record = metadata["kernel_prior"]
            prior_record.update(precision_root=all_state["prior_precision_root"].tolist(),
                                reference_root=all_state["prior_reference_root"].tolist(),
                                design_root=all_state["prior_design_root"].tolist())
            prior = CompiledKernelPrior.from_record(prior_record)
            if _CALIBRATION_PRIOR_SCHEMAS.get(prior.spec.kind) != schema:
                raise ValueError("continuous model prior kind differs from its declared schema")
        if posterior_bound:
            metadata["provenance"]["kernel_prior_fit_statistics"].update(
                information_root=all_state["prior_fit_information_root"].tolist(),
                score=all_state["prior_fit_score"].tolist())
        dictionary = geometry = None
        if owned_dictionary:
            dictionary_record = metadata["dictionary"]
            dictionary_record.update(matrix=all_state["basis"].tolist(),
                **{key: all_state["dictionary_"+key].tolist() for key in _DICTIONARY_ARRAY_KEYS})
            dictionary = ResponseDictionary.from_record(dictionary_record)
            geometry = {**metadata["dictionary_geometry"],
                        **{key.removeprefix("dictionary_"): all_state[key].tolist()
                           for key in _dictionary_state_keys(calibration_prior) if key.removeprefix("dictionary_") in _DICTIONARY_GEOMETRY_KEYS},
                        "fitted_information_root": metadata["provenance"]["kernel_prior_fit_statistics"]["information_root"]}
            if calibration_prior:
                geometry.update(reference_information_root=prior.reference_root.tolist(), original_reference_root=prior.design_root.tolist())
        return cls(*state, None if explicit_ar else metadata["phi"], metadata["sfreq"], tuple(metadata["channels"]),
                   tuple(metadata["candidate_ids"]), metadata["operator_identity"], metadata["provenance"], noise=noise,
                    conditioning_samples=prefix, kernel_prior=prior, dictionary=dictionary, dictionary_geometry=geometry,
                    posterior_state=({key.removeprefix("posterior_"): all_state[key] for key in _POSTERIOR_STATE_KEYS}
                                     if posterior_bound else None))
