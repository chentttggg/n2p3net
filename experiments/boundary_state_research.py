"""Boundary redesign mathematics and experiment-plan scaffold, research only.

Long-context differences require actually observed context. The dense Gaussian
reference below tests covariance gauges; it does not estimate an AR model or
claim a calibrated physiological posterior. No production consumer imports it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.finite_filter_operator import FiniteFilterOperator  # noqa: E402
from transfer.noise_model import array_digest, frozen_array  # noqa: E402


def _finite(value, ndim, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real numeric data")
    result = np.asarray(raw, dtype=np.float64)
    if result.ndim != ndim or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be nonempty finite {ndim}-dimensional data")
    return result


def erp_row_basis(calibration_templates, *, relative_tolerance=1e-10):
    """Orthonormal rows on a declared time-major output grid, calibration only.

    Callers build templates using authorized calibration records, freeze their
    labels/support and align them to the output grid. Classifier coefficients
    and xDAWN spatial filters are not automatically such signal templates.
    """
    templates = _finite(calibration_templates, 2, "ERP templates")
    if not 0 < relative_tolerance < 1:
        raise ValueError("relative tolerance must lie in (0, 1)")
    _, singular, rows = np.linalg.svd(templates, full_matrices=False)
    if singular[0] <= np.finfo(float).tiny:
        raise ValueError("ERP templates have no identifiable nonzero direction")
    rows = rows[singular > relative_tolerance * singular[0]].copy()
    rows.setflags(write=False)
    return rows


def _erp_rows(rows, dimension):
    rows = _finite(rows, 2, "ERP row basis")
    if rows.shape[1] != dimension or not np.allclose(
        rows @ rows.T, np.eye(len(rows)), rtol=1e-10, atol=1e-12
    ):
        raise ValueError("ERP rows must be orthonormal on the exact output grid")
    return rows


def protect_delta(delta, erp_rows, *, penalty):
    """Solve min ||u-delta||^2 + penalty ||U u||^2 for orthonormal rows U.

    Output protection is distinct from K U_input = 0, which protects the
    response of a learned correction map to ERP input increments.
    """
    values = _finite(delta, 2, "delta")
    rows = _erp_rows(erp_rows, values.size)
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    vector = values.reshape(-1)  # sample first, then physical channel
    return _protect_output(vector, rows, penalty).reshape(values.shape)


def _protect_output(value, rows, penalty):
    return value - penalty / (1 + penalty) * (rows.T @ (rows @ value))


@dataclass(frozen=True)
class FrozenShortBoundaryMap:
    """Bounded research reference for T(x)=R F_short x + K x + b.

    Samples are time-major native voltages on the declared short scope. The
    output is the declared native crop; downstream resampling/reference or
    baseline is not implicitly performed. K/b are provided frozen parameters,
    not fitted or authorized by this reference. Source/C authorization belongs
    to their producer. Prediction has no long-context or target-label input.
    """

    operator: FiniteFilterOperator
    correction_matrix: np.ndarray
    short_scope: tuple[int, int]
    output_scope: tuple[int, int]
    sfreq: float
    channel_names: tuple[str, ...]
    reference: str
    signal_unit: str
    parameter_source_id: str
    correction_offset: np.ndarray | None = None
    byte_budget: int = 8 * 1024 * 1024
    baseline_matrix: np.ndarray = field(init=False, repr=False, compare=False)
    complete_matrix: np.ndarray = field(init=False, repr=False, compare=False)
    _protection: tuple | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self):
        short, output = tuple(self.short_scope), tuple(self.output_scope)
        if (not isinstance(self.operator, FiniteFilterOperator) or len(short) != 2 or len(output) != 2
                or any(type(v) is not int for v in (*short, *output))
                or not 0 <= short[0] <= output[0] < output[1] <= short[1]):
            raise ValueError("short/output physical sample scopes must share a contained native grid")
        names = tuple(self.channel_names)
        if (not names or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name.strip() for name in names)
                or not np.isfinite(self.sfreq) or self.sfreq <= 0
                or any(not isinstance(value, str) or not value.strip()
                       for value in (self.reference, self.signal_unit, self.parameter_source_id))):
            raise ValueError("mapping requires named channels, sample rate, reference, unit and parameter source")
        n, m, channels = short[1] - short[0], output[1] - output[0], len(names)
        required_bytes = 8 * (n * n + 3 * m * n * channels * channels + m * channels)
        if type(self.byte_budget) is not int or self.byte_budget < required_bytes:
            raise ValueError("complete dense boundary map exceeds the explicit diagnostic byte budget")
        correction = _finite(self.correction_matrix, 2, "short-only correction matrix")
        if correction.shape != (m * channels, n * channels):
            raise ValueError("correction matrix must read only the declared short grid and write the output grid")
        offset = (np.zeros(m * channels) if self.correction_offset is None
                  else _finite(self.correction_offset, 1, "correction offset"))
        if offset.shape != (m * channels,):
            raise ValueError("correction offset must match the declared output grid")
        matrix = self.operator.dense_matrix(n, byte_budget=self.byte_budget)
        baseline = np.kron(matrix[output[0] - short[0]:output[1] - short[0]], np.eye(channels))
        for name, value in (("correction_matrix", correction), ("correction_offset", offset),
                            ("baseline_matrix", baseline), ("complete_matrix", baseline + correction)):
            object.__setattr__(self, name, frozen_array(value))
        object.__setattr__(self, "short_scope", short)
        object.__setattr__(self, "output_scope", output)
        object.__setattr__(self, "channel_names", names)

    def apply(self, short_samples):
        """Apply frozen T to observed short samples; no parameter updates."""
        values = _finite(short_samples, 2, "observed short samples")
        if values.shape != (self.short_scope[1] - self.short_scope[0], len(self.channel_names)):
            raise ValueError("prediction must contain only samples on the frozen short input grid")
        result = self.complete_matrix @ values.ravel() + self.correction_offset
        return result.reshape(self.output_scope[1] - self.output_scope[0], len(self.channel_names))

    def with_output_protection(self, erp_rows, *, penalty):
        """Derive an explicitly declared output-soft arm, without fitting."""
        rows = _erp_rows(erp_rows, self.complete_matrix.shape[0])
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("penalty must be finite and nonnegative")
        protected = replace(self,
            correction_matrix=_protect_output(self.correction_matrix, rows, penalty),
            correction_offset=_protect_output(self.correction_offset, rows, penalty))
        object.__setattr__(protected, "_protection", (float(penalty), array_digest(rows), self.digest))
        return protected

    def record(self):
        return {"schema": "n2p3_frozen_short_boundary_map/1", "mapping_role": "short_only_frozen_affine",
                "short_scope": list(self.short_scope), "output_scope": list(self.output_scope),
                "sfreq": self.sfreq, "channel_names": list(self.channel_names),
                "reference": self.reference, "signal_unit": self.signal_unit,
                "flattening": "sample_then_physical_channel", "output_stage": "native_crop",
                "operator": self.operator.record(), "parameter_source_id": self.parameter_source_id,
                "parameter_source_authorized_here": False,
                "numeric_sha256": array_digest(self.correction_matrix, self.correction_offset,
                                                self.baseline_matrix, self.complete_matrix),
                "output_protection": None if self._protection is None else {
                    "penalty": self._protection[0], "erp_rows_sha256": self._protection[1],
                    "parent_map_sha256": self._protection[2]}}

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _increment_metrics(increment, reference, sfreq):
    norm = float(np.linalg.norm(reference))
    error = increment - reference
    relative = None if norm <= np.finfo(float).tiny else float(np.linalg.norm(error) / norm)
    peaks = np.argmax(np.abs(increment), axis=0)
    reference_peaks = np.argmax(np.abs(reference), axis=0)
    observed_peak = (np.any(increment != 0, axis=0) & np.any(reference != 0, axis=0))
    latency = [float((peak - target) / sfreq) if defined else None
               for peak, target, defined in zip(peaks, reference_peaks, observed_peak, strict=True)]
    return {"relative_error": relative, "reference_norm": norm,
            "undefined_reason": "zero_reference_increment" if relative is None else None,
            "max_absolute_error": float(np.max(np.abs(error))),
            "peak_signed_channels": increment[peaks, np.arange(increment.shape[1])].tolist(),
            "peak_latency_error_seconds": latency,
            "area_error_channels": (error.sum(axis=0) / sfreq).tolist()}


def evaluate_complete_map(observed_full, mapping, operator_long, *, context_scope,
                          erp_injections, injection_design_id):
    """Full T increments and exact gains; long data is only an observed oracle.

    Injection shapes are independently declared from protection/fitting
    dictionaries by their producer. All injections retain this observed full
    context grid; the applier sees only their short crop. No policy is selected.
    This linear/affine reference does not certify nonlinear adaptive pipelines.
    """
    if not isinstance(mapping, FrozenShortBoundaryMap) or not isinstance(operator_long, FiniteFilterOperator):
        raise ValueError("complete-map evaluation requires frozen short and explicit long operators")
    x = _finite(observed_full, 2, "observed oracle context")
    scope = tuple(context_scope)
    if (len(scope) != 2 or any(type(v) is not int for v in scope)
            or scope[1] - scope[0] != len(x) or not 0 <= scope[0] <= mapping.short_scope[0]
            or mapping.short_scope[1] > scope[1] or x.shape[1] != len(mapping.channel_names)):
        raise ValueError("observed context support must contain the mapping's physical short/output grids")
    if not isinstance(injection_design_id, str) or not injection_design_id.strip():
        raise ValueError("independent ERP injection design must have an explicit identity")
    if not isinstance(erp_injections, dict) or not erp_injections:
        raise ValueError("complete-map evaluation requires independent named ERP injections")
    begin, end = (value - scope[0] for value in mapping.short_scope)
    out_begin, out_end = (value - scope[0] for value in mapping.output_scope)
    short = x[begin:end]
    unperturbed = mapping.apply(short)
    entries = []
    for name, value in erp_injections.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("ERP injections require nonempty names")
        full_h = _finite(value, 2, "independent full-context ERP increment")
        if full_h.shape != x.shape:
            raise ValueError("ERP injection must use the same actually observed full context grid")
        short_h = full_h[begin:end]
        increment = mapping.apply(short + short_h) - unperturbed
        short_reference = (mapping.baseline_matrix @ short_h.ravel()).reshape(increment.shape)
        long_reference = operator_long.apply(full_h)[out_begin:out_end]
        entries.append({"injection_id": name, "increment_sha256": array_digest(full_h),
                        "short_input_increment_norm": float(np.linalg.norm(short_h)),
                        "short_reference": _increment_metrics(increment, short_reference, mapping.sfreq),
                        "observed_long_reference": _increment_metrics(increment, long_reference, mapping.sfreq)})
    matrix, baseline = mapping.complete_matrix, mapping.baseline_matrix
    baseline_energy = float(np.sum(baseline**2))
    singular = np.linalg.svd(matrix, compute_uv=False)
    keep = singular > max(matrix.shape) * np.finfo(float).eps * singular[0]
    rank = int(keep.sum())
    mean_gain = None if baseline_energy <= np.finfo(float).tiny else float(np.sum(matrix**2) / baseline_energy)
    return {"schema": "n2p3_boundary_complete_map_diagnostics/1",
            "context_scope": list(scope), "observed_long_role": "teacher_reference_only",
            "deployment_input_scope": list(mapping.short_scope), "mapping": mapping.record(),
            "mapping_sha256": mapping.digest, "injection_design_id": injection_design_id,
            "independence_authorized_by_evaluator": False, "requested_injections": len(entries),
            "defined_short_relative_errors": sum(row["short_reference"]["relative_error"] is not None for row in entries),
            "defined_long_relative_errors": sum(row["observed_long_reference"]["relative_error"] is not None for row in entries),
            "erp_increments": entries,
            "white_noise_gain": {"mode": "exact_affine_increment_linear_map", "input_white_variance": 1.,
                "mean_relative_to_short": mean_gain,
                "undefined_reason": "zero_baseline_white_energy" if mean_gain is None else None,
                "worst_absolute": float(singular[0]**2), "rank": rank,
                "condition_in_identified_range": None if not rank else float(singular[0] / singular[keep][-1]),
                "offset_included_as_noise": False},
            "policy_selected": False, "external_state_uniquely_recoverable_claim": False,
            "production_or_public_performance_claim": False}


def distance_adaptive_erp_penalty(distance_seconds, *, d_pivot=1., tau=.5, lambda_max=100.):
    """Unvalidated decreasing protection schedule; d is nearest output-support edge."""
    if (not np.isfinite([distance_seconds, d_pivot, tau, lambda_max]).all()
            or distance_seconds < 0 or d_pivot < 0 or tau <= 0 or lambda_max < 0):
        raise ValueError("invalid distance/transition/penalty")
    return float(lambda_max * expit((d_pivot - distance_seconds) / tau))


@dataclass(frozen=True)
class BoundaryStateAnalysis:
    """Observed-context diagnostic, without selecting a policy from held data."""

    distance_seconds: float
    erp_fraction: float | None
    overlap_flag: bool
    zero_difference: bool
    pareto_diagnostics: tuple[dict, ...]
    input_sha256: str
    reference_role: str = "observed_long_context_diagnostic_only"
    grid_support: dict = field(default_factory=dict)

    @classmethod
    def compute(cls, x_observed_full, operator_short, operator_long, erp_rows, *,
                short_scope, output_scope, sfreq, penalty_grid=(0., .01, .1, 1., 10., 100.),
                overlap_threshold=.5, frozen_map=None, erp_injections=None,
                injection_design_id=None, context_scope=None):
        """Compare R F_long x to R F_short C x on a common output grid.

        Scopes are absolute half-open native sample intervals; context_scope
        declares the supplied array's origin. A supplied frozen map reads only
        short X. Without one, output protection acts on teacher delta and cannot
        certify full-map ERP/gain. No ERP challenge or penalty is inferred here.
        """
        x = _finite(x_observed_full, 2, "observed full context")
        if not isinstance(operator_short, FiniteFilterOperator) or not isinstance(
            operator_long, FiniteFilterOperator
        ):
            raise ValueError("explicit finite operators required")
        if (len(short_scope) != 2 or len(output_scope) != 2
                or any(type(v) is not int for v in (*short_scope, *output_scope))):
            raise ValueError("scopes must contain integer half-open boundaries")
        begin, end = short_scope
        out_begin, out_end = output_scope
        if frozen_map is not None and context_scope is None:
            raise ValueError("supplied map requires explicit context support on the physical sample grid")
        context = (0, len(x)) if context_scope is None else tuple(context_scope)
        if (len(context) != 2 or any(type(value) is not int for value in context)
                or context[1] - context[0] != len(x)
                or not 0 <= context[0] <= begin <= out_begin < out_end <= end <= context[1]):
            raise ValueError("output must be inside short scope and observed full context")
        if not np.isfinite(sfreq) or sfreq <= 0 or not 0 < overlap_threshold < 1:
            raise ValueError("invalid sample rate or diagnostic overlap threshold")
        penalties = tuple(penalty_grid)
        if not penalties or not np.isfinite(penalties).all() or min(penalties) < 0:
            raise ValueError("nonempty finite nonnegative penalty grid required")
        if frozen_map is not None:
            if (not isinstance(frozen_map, FrozenShortBoundaryMap) or context_scope is None
                    or frozen_map.short_scope != tuple(short_scope) or frozen_map.output_scope != tuple(output_scope)
                    or frozen_map.sfreq != sfreq or len(frozen_map.channel_names) != x.shape[1]
                    or frozen_map.operator.digest != operator_short.digest):
                raise ValueError("supplied map must share explicit context support and the actual short/output grid")
            if erp_injections is None or injection_design_id is None:
                raise ValueError("supplied map requires independent ERP injections and their design identity")
        elif erp_injections is not None or injection_design_id is not None:
            raise ValueError("ERP/gain diagnostics require an explicitly supplied frozen short-only map")
        observed_short = x[begin - context[0]:end - context[0]]
        short = operator_short.apply(observed_short)[out_begin - begin:out_end - begin]
        long = operator_long.apply(x)[out_begin - context[0]:out_end - context[0]]
        delta = long - short
        rows = _erp_rows(erp_rows, delta.size)
        energy = float(np.sum(delta**2))
        zero = energy <= np.finfo(float).tiny
        fraction = None if zero else float(np.sum((rows @ delta.reshape(-1))**2) / energy)
        diagnostics = []
        for penalty in penalties:
            full = None
            if frozen_map is None:
                correction = protect_delta(delta, rows, penalty=penalty)
            else:
                deployed = frozen_map.with_output_protection(rows, penalty=penalty)
                correction = deployed.apply(observed_short) - short
                full = evaluate_complete_map(x, deployed, operator_long, context_scope=context,
                    erp_injections=erp_injections, injection_design_id=injection_design_id)
            residual = delta - correction
            diagnostics.append({
                "penalty": float(penalty),
                "reference_difference_mse": float(np.mean(residual**2)),
                "difference_mse_improvement": None if zero else 1 - float(np.sum(residual**2)) / energy,
                "erp_aligned_correction_norm": float(np.linalg.norm(rows @ correction.reshape(-1))),
                "protection_map_condition": float(1 + penalty) if len(rows) < delta.size else 1.,
                "correction_role": "observed_long_teacher" if full is None else "frozen_short_only_predictor",
                "erp_increment_error": None if full is None else {
                    "requested": full["requested_injections"],
                    "defined_short": full["defined_short_relative_errors"],
                    "defined_long": full["defined_long_relative_errors"],
                    "injection_design_id": injection_design_id,
                    "cases": full["erp_increments"]},
                "white_noise_gain": None if full is None else full["white_noise_gain"],
                "complete_map_sha256": None if full is None else full["mapping_sha256"],
                "missing_metrics_reason": "requires_frozen_input_correction_map_and_independent_injections" if full is None else None,
            })
        digest = hashlib.sha256()
        digest.update(x.tobytes())
        digest.update(rows.tobytes())
        digest.update(json.dumps([operator_short.record(), operator_long.record(),
                                  short_scope, output_scope, sfreq, context]).encode())
        if frozen_map is not None:
            digest.update(frozen_map.digest.encode())
            digest.update(injection_design_id.encode())
            for name, value in erp_injections.items():
                digest.update(name.encode())
                digest.update(array_digest(value).encode())
        return cls(min(out_begin - begin, end - out_end) / sfreq, fraction,
                   fraction is not None and fraction >= overlap_threshold, zero,
                   tuple(diagnostics), digest.hexdigest(), grid_support={
                       "context_scope": list(context), "short_scope": list(short_scope),
                       "output_scope": list(output_scope), "sample_rate_hz": sfreq,
                       "flattening": "sample_then_physical_channel",
                       "mapping_role": "teacher_only" if frozen_map is None else "short_only",
                       "observed_long_used_at_deployment": False})


@dataclass(frozen=True)
class ScaleInvariantContinuousModel:
    """Small dense matrix-normal reference with an explicit covariance gauge.

    The canonical prior is W ~ MN(0, I / ridge_relative, Sigma). Covariance
    K -> alpha K, Sigma -> Sigma / alpha requires lambda -> lambda / alpha.
    K and Sigma must be independently frozen from calibration. This reference
    neither fits AR1/AR8 nor promotes a new physiological prior.
    """

    kernel_mean: np.ndarray
    kernel_row_covariance: np.ndarray
    channel_covariance: np.ndarray
    covariance_factor: float
    provenance: dict

    @classmethod
    def fit_scale_invariant(cls, design, samples, temporal_covariance, innovation_covariance, *,
                            ridge_relative=1., covariance_factor=1., byte_budget=8 * 1024 * 1024):
        design = _finite(design, 2, "calibration design")
        y = _finite(samples, 2, "calibration samples")
        temporal = _finite(temporal_covariance, 2, "canonical temporal covariance")
        spatial = _finite(innovation_covariance, 2, "canonical channel covariance")
        if (len(design) != len(y) or temporal.shape != (len(y), len(y))
                or spatial.shape != (y.shape[1], y.shape[1])):
            raise ValueError("design/sample/covariance grids do not match")
        if (not np.isfinite([ridge_relative, covariance_factor]).all()
                or ridge_relative <= 0 or covariance_factor <= 0):
            raise ValueError("positive finite ridge and covariance factor required")
        if type(byte_budget) is not int or byte_budget <= 0 or temporal.nbytes > byte_budget:
            raise ValueError("dense temporal reference exceeds diagnostic byte budget")
        for matrix in (temporal, spatial):
            if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0):
                raise ValueError("covariance must be symmetric")
            np.linalg.cholesky(matrix)
        root = np.linalg.cholesky(covariance_factor * temporal)
        d = solve_triangular(root, design, lower=True)
        observed = solve_triangular(root, y, lower=True)
        effective_ridge = ridge_relative / covariance_factor
        augmented = np.vstack([d, np.sqrt(effective_ridge) * np.eye(d.shape[1])])
        _, factor = np.linalg.qr(augmented, mode="reduced")
        inverse = solve_triangular(factor, np.eye(len(factor)))
        row_cov = inverse @ inverse.T
        mean = row_cov @ d.T @ observed
        arrays = [mean, row_cov, spatial / covariance_factor]
        for value in arrays:
            value.setflags(write=False)
        return cls(*arrays, covariance_factor, {
            "scope": "synthetic_or_calibration_frozen_noise_dense_reference",
            "ridge_relative": float(ridge_relative), "effective_ridge": float(effective_ridge),
            "scale_invariant": True, "gauge_rule": "lambda_divided_by_temporal_covariance_factor",
            "ar_estimated": False,
        })

    def log_evidence(self, samples, candidate_designs, temporal_covariance, *, byte_budget=8 * 1024 * 1024):
        """Exact small dense Gaussian, including candidate-dependent logdet."""
        y = _finite(samples, 2, "held samples")
        candidates = _finite(candidate_designs, 3, "candidate designs")
        temporal = _finite(temporal_covariance, 2, "held canonical temporal covariance")
        if (temporal.shape != (len(y), len(y)) or candidates.shape[1:] != (len(y), len(self.kernel_mean))
                or y.shape[1] != len(self.channel_covariance)):
            raise ValueError("candidate/sample/covariance grids do not match")
        if (type(byte_budget) is not int or byte_budget <= 0
                or 8 * y.size**2 > byte_budget):
            raise ValueError("dense predictive reference exceeds diagnostic byte budget")
        if not np.allclose(temporal, temporal.T, rtol=1e-12, atol=0):
            raise ValueError("held temporal covariance must be symmetric")
        np.linalg.cholesky(temporal)
        evidence = []
        for design in candidates:
            temporal_predictive = self.covariance_factor * temporal + design @ self.kernel_row_covariance @ design.T
            covariance = np.kron(temporal_predictive, self.channel_covariance)
            residual = (y - design @ self.kernel_mean).reshape(-1)
            root = np.linalg.cholesky(covariance)
            whitened = solve_triangular(root, residual, lower=True)
            evidence.append(-.5 * (float(whitened @ whitened) + 2 * np.log(np.diag(root)).sum()))
        return np.asarray(evidence)


def make_plan(baseline_plan):
    """Validate historical input identity without reading new test selections."""
    path = Path(baseline_plan)
    payload = path.read_bytes()
    base = json.loads(payload)
    if (base.get("authorized_labels_per_participant") != 800 or base.get("requested_fold_count") != 48
            or base.get("group_selection") != "first_six_complete_120_flash_selections"
            or base.get("sample_rate_hz") != 256 or base.get("channels") != "BNCI_native_8"):
        raise ValueError("baseline plan is not the declared C800 group protocol")
    sources = base["records"]
    if len(sources) != 8 or len({r["subject"] for r in sources}) != 8:
        raise ValueError("eight distinct physical participants required")
    for row in sources:
        if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("BNCI distributed MAT bytes changed")
    return {
        "schema": "n2p3_boundary_redesign_plan/1", "status": "framework_only_not_executed",
        "evidence_scope": "previously_observed_public_calibration_development",
        "baseline_plan_sha256": hashlib.sha256(payload).hexdigest(),
        "research_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": sources, "authorized_labels_per_participant": 800,
        "group_selection": base["group_selection"], "requested_fold_count": 48,
        "fit_labels_per_outer_fold": 600, "held_labels_per_outer_fold": 120,
        "unused_authorized_labels": 80, "physical_channels": base["channels"],
        "sample_rate_hz": base["sample_rate_hz"],
        "pre_roll_seconds": base["pre_roll_seconds"],
        "response_seconds": base["response_seconds"], "basis_rank": base["basis_rank"],
        "raw_rebuild_owner": "doc/assets/operator_research_20261004/run_calibration_study.py:selection_views",
        "erp_basis": "calibration_only_signal_templates_on_explicit_time_major_grid",
        "E1": {"distances_seconds": [0., .5, 1., 2., 4., 8.],
               "overlap_threshold_candidate": .5, "overlap_alone_proves_damage": False,
               "long_context_role": "diagnostic_reference_never_unknown_test_context"},
        "E2": {"ar_orders": [1, 8], "ridge_relative_grid": [.01, .1, 1., 10.],
               "arms": ["innovation_relative", "marginal_relative_existing_prior_change"],
               "gauge_check": "K_alpha_Sigma_over_alpha_lambda_over_alpha",
               "unit_check": "Y_times_s_Sigma_times_s_squared_same_lambda",
               "noise_estimation": "existing_research_OperatorGLS_frozen_calibration_only",
               "metrics": ["requested_hit@all", "coverage", "conditional_NLL", "maximum_confidence"]},
        "E3": {"distances_seconds": [0., .5, 1., 2., 4., 8.],
               "penalties": [0., .01, .1, 1., 10., 100.],
               "adaptive_schedule": {"d_pivot": 1., "tau": .5, "lambda_max": 100., "validated": False},
               "required_metrics": ["difference_MSE", "ERP_increment_error", "white_noise_gain", "condition_number"]},
        "selection": "inner_training_groups_only_or_fixed_grid_report_all_no_held_selection_choice",
        "unavailable_geometry": "keep_requested_case_with_reason_do_not_read_outside_C800",
        "acceptance": {"log_BF_gauge_atol": 1e-8, "ERP_increment_relative_error_candidate": .05,
                       "hit_floor_candidate": .75, "NLL_smoke_limit": float(2 * np.log(36)),
                       "promotion": "requires_paired_anchor_differences_coverage_uncertainty_and_independent_public_confirmation"},
        "kalman": "deferred_pending_observability_and_prediction_vs_retrospective_smoothing_contract",
        "full_E1_E2_E3_executed": False, "production_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = make_plan(args.baseline_plan)
    # Never overwrite a previous plan/result or its execution identity.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "participants": len(result["records"]), "full_experiments_executed": False}))


if __name__ == "__main__":
    main()
