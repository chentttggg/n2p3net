"""Segmented Burg AR noise with exact stationary startup whitening.

The temporal shape is shared across observed channels. The whitening operator
preserves innovation units: only the temporal covariance for *unit* innovation
is removed. A downstream channel covariance owns voltage scale. The pooled
innovation variance saved here is a diagnostic, never a second normalization.

Fitting receives residual records from an authorized caller; it neither removes
an intercept nor joins records. Stationary startup and conditioning on an
observed prefix are explicit, different probability models. The latter retains
the prefix as actual AR history but scores only subsequent innovations. Callers
own whether that prefix carries candidate/response information. Work is O(N p)
plus p-sized state algebra; no record-sized covariance matrix is formed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import InitVar, dataclass, field
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from research.execution import UnidentifiableFitError
from transfer.dictionary_artifact import COMPILER_SOURCE_SHA256, _binary64
from transfer.noise_model import (
    array_digest,
    freeze_metadata,
    frozen_array,
    metadata_digest,
    require_identifiers,
    require_sha256,
)

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_SCHEMA = "n2p3_temporal_noise/5"
_SCALE_CONVENTION = "unit_innovation_temporal_shape_preserve_data_units"
_MODES = {"stationary", "conditional_prefix", "stationary_innovations"}
_BURG_METHOD = "segmented_multichannel_burg_no_pole_clipping"
_CONDITIONAL_METHOD = "fixed_prefix_conditional_profile_after_segmented_burg"
_COMPILER_PROGRAM = "canonical_PARCOR_prefix_inverse_decimal96_and_binary64_Gram/1"
_SCALAR_CONTEXT = Context(prec=96, rounding=ROUND_HALF_EVEN, Emin=-999999999, Emax=999999999,
                          capitals=1, clamp=0, flags=[], traps=[InvalidOperation, DivisionByZero, Overflow])
_ARRAY_FIELDS = ("startup_covariance", "startup_root", "startup_whitening_factor",
                 "startup_log_variances", "startup_innovation_variances", "startup_scales")
_FIT_FIELDS = {"fit_scope", "records", "method", "assume_centered", "available_lag_rows",
               "minimum_lag_rows", "minimum_samples_per_parameter", "n_columns", "implementation_sha256"}


def _real_vector(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != 1:
        raise ValueError(f"{name} must be a real numeric vector")
    result = np.asarray(raw, dtype=np.float64)
    if len(result) > 128 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of order at most 128")
    return result


def _real_scalar(value, name, *, positive=False):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu" or not np.isfinite(raw):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _optimization_record(value):
    required = {"strategy", "initial_objective", "final_objective", "converged",
                "iterations", "evaluations", "eta_max", "status", "message"}
    if (not isinstance(value, dict) or set(value) != required
            or value["strategy"] not in {"burg_only", "conditional_profile"}):
        raise ValueError("conditional optimization requires its complete strategy and result")
    result = dict(value)
    for name in ("initial_objective", "final_objective", "eta_max"):
        result[name] = _real_scalar(value[name], f"optimization {name}", positive=name == "eta_max")
    if result["converged"] is not True:
        raise ValueError("conditional optimization must have converged before state publication")
    if (type(result["iterations"]) is not int or result["iterations"] < 0
            or type(result["evaluations"]) is not int or result["evaluations"] < 1
            or type(result["status"]) is not int or result["status"] < 0
            or not isinstance(result["message"], str) or not result["message"].strip()):
        raise ValueError("conditional optimization requires actual counts, status and message")
    tolerance = 1e-10*max(1., abs(result["initial_objective"]), abs(result["final_objective"]))
    if result["final_objective"] > result["initial_objective"] + tolerance:
        raise ValueError("conditional optimization objective worsened")
    if result["strategy"] == "burg_only" and (
            result["iterations"] != 0 or result["initial_objective"] != result["final_objective"]):
        raise ValueError("retained Burg shape requires an unchanged objective and zero iterations")
    return result


def _values(value, *, minimum_samples=1):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2 or raw.shape[1] < 1:
        raise ValueError("noise observations must be a real numeric samples-by-columns array")
    result = np.asarray(raw, dtype=np.float64)
    if len(result) < minimum_samples or not np.isfinite(result).all():
        raise ValueError("noise observations require sufficient samples and finite values")
    return result


def _reflections(coefficients):
    """Step down the prediction polynomial, independently checking Burg stability."""
    # Compile from the exact binary64 polynomial. Float stepdown near a unit
    # reflection loses enough digits to violate its own later step-up relation.
    current = [Decimal.from_float(float(value)).copy_negate() for value in coefficients]
    reflection = np.empty(len(current))
    with localcontext(_SCALAR_CONTEXT):
        for order in range(len(current), 0, -1):
            k = current[-1]
            if not k.is_finite() or abs(k) >= Decimal.from_float(1-1e-12):
                raise ValueError("AR polynomial is not strictly stable in reflection coordinates")
            reflection[order-1] = _canonical_scalar(k)
            prior = current[:-1]
            current = [(left-k*right)/(1-k*k) for left, right in zip(prior, reversed(prior), strict=True)]
    return reflection


def _startup(coefficients, *, conditional=False, reflections=None):
    order = len(coefficients)
    if not order:
        return np.empty((0, 0)), np.empty((0, 0)), 0., None if conditional else 1.
    companion = np.zeros((order, order))
    companion[0] = coefficients
    companion[1:, :-1] = np.eye(order-1)
    radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
    if radius >= 1 - 1e-12:
        raise ValueError("AR poles are not strictly stable")
    if conditional:
        # The caller conditions on actual prefix samples. There is no latent
        # stationary startup covariance to form or numerically regularize.
        return np.empty((0, 0)), np.empty((0, 0)), radius, None
    reflection = _reflections(coefficients) if reflections is None else reflections
    _, _, scales = _prefix_scalars(reflection)
    covariance, root = _canonical_stationary(_prefix_factor(reflection, scales))
    condition = float(np.linalg.cond(covariance))
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError("stationary startup covariance is numerically unidentifiable")
    # State ordering reverses time; a stationary scalar AR covariance is symmetric Toeplitz.
    return covariance, root, radius, condition


def _parcor_coefficients(reflections):
    polynomial = np.empty(0)
    for reflection in reflections:
        polynomial = np.r_[polynomial+reflection*polynomial[::-1], reflection]
    return -polynomial


def _canonical_scalar(value):
    numerator, denominator = value.as_integer_ratio()
    try:
        return _binary64(numerator, denominator)
    except ValueError as error:
        raise UnidentifiableFitError("temporal compiled scalar is not representable in float64") from error


def _prefix_scalars(reflections):
    """Fixed scalar expressions; no host log/exp/sqrt defines their identity."""
    product = Fraction(1)
    logs, variances, scales = [], [], []
    with localcontext(_SCALAR_CONTEXT):
        for value in reversed(reflections):
            reflection = Fraction.from_float(float(value))
            product *= 1-reflection*reflection
            decimal_product = Decimal(product.numerator)/Decimal(product.denominator)
            logs.append(_canonical_scalar(-decimal_product.ln()))
            variances.append(_canonical_scalar(1/product))
            scales.append(_canonical_scalar(decimal_product.sqrt()))
    result = tuple(np.asarray(values[::-1], dtype=np.float64) for values in (logs, variances, scales))
    if any(not np.isfinite(values).all() or np.any(values <= 0) for values in result[1:]):
        raise UnidentifiableFitError("temporal prefix variance/scale is not finitely invertible")
    return result


def _reference_variance(reflections):
    product = Fraction(1)
    for value in reflections:
        reflection = Fraction.from_float(float(value))
        product *= 1-reflection*reflection
    return _canonical_scalar(1/product)


def _prefix_factor(reflections, scales):
    factor = np.zeros((len(reflections), len(reflections)))
    polynomial = np.empty(0)
    for row, reflection in enumerate(reflections):
        factor[row, row] = scales[row]
        factor[row, :row] = polynomial[::-1]*scales[row]
        polynomial = np.r_[polynomial+reflection*polynomial[::-1], reflection]
    return factor


def _canonical_stationary(factor):
    """Publish the actual prefix root; no platform Lyapunov/Cholesky authority."""
    order = len(factor)
    with localcontext(_SCALAR_CONTEXT):
        decimal_factor = [[Decimal.from_float(float(value)) for value in row] for row in factor]
        inverse = [[Decimal(0) for _ in range(order)] for _ in range(order)]
        for row in range(order):
            for column in range(row+1):
                total = sum((decimal_factor[row][index]*inverse[index][column]
                             for index in range(column, row)), Decimal(0))
                inverse[row][column] = ((Decimal(int(row == column))-total)/decimal_factor[row][row])
        root = np.asarray([[_canonical_scalar(value) for value in row] for row in inverse])
    exact = [[Fraction.from_float(float(value)) for value in row] for row in root]
    covariance = np.empty_like(root)
    for row in range(order):
        for column in range(row+1):
            value = sum((exact[row][index]*exact[column][index]
                         for index in range(column+1)), Fraction())
            covariance[row, column] = covariance[column, row] = _canonical_scalar(value)
    return covariance, root


def _innovation_factor(reflections):
    """Levinson prefix factor; no covariance inversion or pseudo-PSD repair."""
    order = len(reflections)
    log_variances, variances, scales = _prefix_scalars(reflections)
    factor = _prefix_factor(reflections, scales)
    if not np.isfinite(factor).all() or np.any(np.diag(factor) <= 0):
        raise ValueError("stationary innovations factor is not finitely invertible")
    condition = float(np.linalg.cond(factor)) if order else 1.
    if not np.isfinite(condition):
        raise ValueError("stationary innovations factor condition is nonfinite")
    return factor, log_variances, variances, scales, condition


def _lattice_innovations(values, reflections, scales):
    """Stable PARCOR recursion for prefix and full-order innovations alike."""
    order = len(reflections)
    if not order:
        return values.copy()
    output = np.empty_like(values)
    output[0] = values[0]*scales[0]
    forward, backward = values[1:].copy(), values[:-1].copy()
    for m, reflection in enumerate(reflections, 1):
        next_forward = forward+reflection*backward
        next_backward = backward+reflection*forward
        if m < order:
            output[m] = next_forward[0]*scales[m]
            forward, backward = next_forward[1:], next_backward[:-1]
        else:
            output[order:] = next_forward
    if not np.isfinite(output).all():
        raise ValueError("stationary lattice innovations overflowed")
    return output


def _validate_provenance(provenance, coefficients, reflection, startup_mode, producer_sha256):
    if not isinstance(provenance, dict):
        raise ValueError("temporal noise requires its provenance record")
    provenance = dict(provenance)
    conditional_fit = provenance.get("method") == _CONDITIONAL_METHOD
    if conditional_fit:
        provenance["optimization"] = _optimization_record(provenance.get("optimization"))
    provenance = json.loads(json.dumps(provenance, allow_nan=False))
    if provenance.get("fit_scope") == "analytic_reference" and not conditional_fit:
        if set(provenance) != {"fit_scope", "reference_id", "records"} or not isinstance(provenance["reference_id"], str) or not provenance["reference_id"].strip() or provenance["records"]:
            raise ValueError("analytic temporal reference must be named without a training ledger")
    elif conditional_fit:
        required = _FIT_FIELDS | {"initialization", "conditioning_samples", "objective_rows",
                                  "optimization", "training_input_sha256"}
        if (set(provenance) != required
                or provenance.get("fit_scope") not in {"source_training", "target_calibration"}
                or startup_mode != "stationary_innovations"):
            raise ValueError("conditional temporal noise requires its training scope and complete ledger")
        initialization = provenance["initialization"]
        if (not isinstance(initialization, dict)
                or initialization.get("schema") != _SCHEMA
                or initialization.get("startup_mode") != "stationary_innovations"
                or initialization.get("provenance", {}).get("method") != _BURG_METHOD):
            raise ValueError("conditional noise initialization must be an actual fitted segmented Burg lattice")
        initializer = TemporalNoise.from_record(initialization)
        if initializer.order != len(coefficients) or any(
                provenance[name] != initialization["provenance"][name]
                for name in _FIT_FIELDS - {"method"}):
            raise ValueError("conditional noise scope, records and columns must preserve its Burg initialization")
        h = provenance["conditioning_samples"]
        if (type(h) is not int or h < max(2, len(coefficients))
                or any(record["n_samples"] <= h for record in provenance["records"])):
            raise ValueError("conditioning samples must leave a positive tail in every actual fit record")
        actual_rows = sum(record["n_samples"]-h for record in provenance["records"])
        if (type(provenance["objective_rows"]) is not int
                or provenance["objective_rows"] != actual_rows or actual_rows <= 0):
            raise ValueError("conditional objective rows disagree with the fixed-prefix training tails")
        require_sha256(provenance["training_input_sha256"], "conditional training input")
        optimization = provenance["optimization"]
        if optimization["strategy"] == "burg_only":
            if array_digest(reflection) != array_digest(initializer.reflection_coefficients):
                raise ValueError("retained Burg reflections differ from the actual initialization")
        else:
            eta_max = optimization["eta_max"]
            tolerance = 2e-14*max(1., eta_max)
            if any(np.any(np.abs(np.arctanh(values)) > eta_max+tolerance)
                   for values in (reflection, initializer.reflection_coefficients)):
                raise ValueError("conditional optimization PARCOR is outside its declared eta domain")
    else:
        if (set(provenance) != _FIT_FIELDS or provenance.get("fit_scope") not in {"source_training", "target_calibration"}
                or provenance.get("method") != _BURG_METHOD
                or provenance.get("assume_centered") is not True or not provenance.get("records")):
            raise ValueError("temporal noise requires its complete training scope and records ledger")
        require_identifiers([r["record_id"] for r in provenance["records"]], "temporal fit record IDs")
        if type(provenance["n_columns"]) is not int or provenance["n_columns"] < 1:
            raise ValueError("temporal noise fit requires positive observed column count")
        for record in provenance["records"]:
            if (set(record) != {"record_id", "n_samples", "input_sha256"}
                    or type(record["n_samples"]) is not int or record["n_samples"] < max(3, len(coefficients)+2)):
                raise ValueError("invalid temporal noise fit record")
            require_sha256(record["input_sha256"], "temporal fit input")
        actual = sum(r["n_samples"]-len(coefficients) for r in provenance["records"])
        per_parameter = provenance["minimum_samples_per_parameter"]
        if type(per_parameter) is not int or per_parameter < 2:
            raise ValueError("temporal fit minimum samples per parameter is invalid")
        minimum = max(32, per_parameter*max(1, len(coefficients)))
        if (type(provenance["available_lag_rows"]) is not int or type(provenance["minimum_lag_rows"]) is not int
                or provenance["available_lag_rows"] != actual or provenance["minimum_lag_rows"] != minimum
                or actual < minimum):
            raise ValueError("temporal noise ledger does not establish sufficient lag rows")
        require_sha256(provenance["implementation_sha256"], "temporal noise implementation")
        if provenance["implementation_sha256"] != producer_sha256:
            raise ValueError("temporal noise artifact was fitted by a different implementation")
    return freeze_metadata(provenance)


def _check_parameter_relation(coefficients, reflection):
    if (reflection.shape != coefficients.shape or np.any(np.abs(reflection) >= 1-1e-12)
            or not np.allclose(coefficients, _parcor_coefficients(reflection), rtol=2e-14, atol=2e-14)):
        raise ValueError("AR coefficients disagree with the actual stable PARCOR sequence")


def _validate_compiled_arrays(coefficients, reflection, mode, arrays, marginal):
    """Check saved expressions and matrix equations; never solve new startup state."""
    order = len(coefficients)
    stationary, lattice = mode == "stationary", mode == "stationary_innovations"
    shapes = {"startup_covariance": (order, order) if stationary else (0, 0),
              "startup_root": (order, order) if stationary else (0, 0),
              "startup_whitening_factor": (order, order) if lattice else (0, 0),
              **{key: (order,) if lattice else (0,) for key in
                 ("startup_log_variances", "startup_innovation_variances", "startup_scales")}}
    if set(arrays) != set(_ARRAY_FIELDS):
        raise ValueError("temporal compiled state requires its complete numerical arrays")
    for key, shape in shapes.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != np.dtype(np.float64) or not np.isfinite(value).all():
            raise ValueError(f"temporal compiled {key} shape/dtype/state differs")
    if lattice:
        logs, variances, scales = _prefix_scalars(reflection)
        for key, expected in zip(("startup_log_variances", "startup_innovation_variances", "startup_scales"),
                                 (logs, variances, scales), strict=True):
            if array_digest(arrays[key]) != array_digest(expected):
                raise ValueError(f"temporal compiled {key} disagrees with its fixed PARCOR scalar expression")
        factor = arrays["startup_whitening_factor"]
        polynomial = np.empty(0)
        for row in range(order):
            if (factor[row, row] != scales[row] or np.any(factor[row, row+1:] != 0)
                    or not np.array_equal(factor[row, :row], polynomial[::-1]*scales[row])):
                raise ValueError("temporal compiled whitening factor disagrees with its prefix recursion")
            polynomial = np.r_[polynomial+reflection[row]*polynomial[::-1], reflection[row]]
        expected_marginal = float(variances[0]) if order else 1.
    elif stationary and order:
        covariance, root = arrays["startup_covariance"], arrays["startup_root"]
        if (not np.array_equal(covariance, covariance.T) or np.any(np.triu(root, 1) != 0)
                or np.any(np.diag(root) <= 0)):
            raise ValueError("stationary compiled covariance/root must be symmetric and positive triangular")
        eps = np.finfo(float).eps
        gamma = (4*order+4)*eps/(1-(4*order+4)*eps)
        # Cholesky's componentwise backward relation plus the verification dot
        # error. A huge marginal variance cannot hide an unrelated small entry.
        with np.errstate(over="ignore", invalid="ignore"):
            factor_scale = np.abs(root)@np.abs(root.T)
            reconstructed = root@root.T
        if not np.isfinite(factor_scale).all() or not np.isfinite(reconstructed).all():
            raise ValueError("stationary compiled factor relation is not numerically representable")
        if np.any(np.abs(reconstructed-covariance) > gamma*factor_scale+eps*np.abs(covariance)):
            raise ValueError("stationary compiled root differs from its covariance relation")
        companion = np.zeros((order, order))
        companion[0] = coefficients
        companion[1:, :-1] = np.eye(order-1)
        driving = np.zeros((order, order))
        driving[0, 0] = 1.
        with np.errstate(over="ignore", invalid="ignore"):
            predicted = companion@covariance@companion.T
            scale = np.abs(covariance)+np.abs(companion)@np.abs(covariance)@np.abs(companion.T)+driving
            residual = covariance-predicted-driving
        if any(not np.isfinite(value).all() for value in (predicted, scale, residual)):
            raise ValueError("stationary compiled AR relation is not numerically representable")
        # SciPy's Lyapunov solver is normwise, not componentwise backward
        # stable. Nearly zero lag entries cannot normalize their own residual.
        # The prefix density gate below separately protects small directions.
        norm_scale = float(np.linalg.norm(covariance) +
            np.linalg.norm(companion, ord="fro")**2*np.linalg.norm(covariance) + np.linalg.norm(driving))
        residual_norm = float(np.linalg.norm(residual))
        if not np.isfinite(norm_scale) or not np.isfinite(residual_norm):
            raise ValueError("stationary compiled AR backward relation is not numerically representable")
        if residual_norm > gamma*norm_scale:
            raise ValueError("stationary compiled covariance differs from its unit-innovation AR equation")
        # A small Cholesky pivot can disappear from a large C entry. Admit its
        # actual density scale independently through the PARCOR prefix map.
        _, _, scales = _prefix_scalars(reflection)
        prefix = _prefix_factor(reflection, scales)
        with np.errstate(over="ignore", invalid="ignore"):
            density = prefix@root
            density_scale = np.abs(prefix)@np.abs(root)+np.eye(order)
        if not np.isfinite(density).all() or not np.isfinite(density_scale).all():
            raise ValueError("stationary compiled prefix density is not numerically representable")
        if np.any(np.abs(density-np.eye(order)) > 2*gamma*density_scale):
            raise ValueError("stationary compiled root differs in a resolved prefix density direction")
        expected_marginal = float(covariance[0, 0])
    else:
        expected_marginal = _reference_variance(reflection)
    if marginal != expected_marginal or not np.isfinite(marginal) or marginal <= 0:
        raise ValueError("compiled marginal variance differs from its actual reference state")


@dataclass(frozen=True)
class TemporalNoise:
    coefficients: np.ndarray
    innovation_variance: float
    provenance: dict
    startup_mode: str = "stationary"
    reflection_state: InitVar[np.ndarray | None] = None
    startup_covariance: np.ndarray = field(init=False, repr=False, compare=False)
    reflection_coefficients: np.ndarray = field(init=False, repr=False, compare=False)
    pole_radius: float = field(init=False)
    startup_condition: float | None = field(init=False)
    _startup_root: np.ndarray = field(init=False, repr=False, compare=False)
    startup_whitening_factor: np.ndarray = field(init=False, repr=False, compare=False)
    startup_innovation_variances: np.ndarray = field(init=False, repr=False, compare=False)
    startup_factor_condition: float | None = field(init=False)
    _startup_log_variances: np.ndarray = field(init=False, repr=False, compare=False)
    _startup_scales: np.ndarray = field(init=False, repr=False, compare=False)
    _marginal_variance_value: float = field(init=False, repr=False, compare=False)
    _compiler: dict = field(init=False, repr=False, compare=False)
    _runtime_binding: tuple = field(init=False, repr=False, compare=False)
    _integrity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, reflection_state):
        if self.startup_mode not in _MODES:
            raise ValueError("unsupported temporal startup mode")
        if not isinstance(self.provenance, dict):
            raise ValueError("temporal noise requires its provenance record")
        coefficients = _real_vector(self.coefficients, "AR coefficients")
        innovation_variance = _real_scalar(self.innovation_variance, "innovation variance", positive=True)
        factor, log_variances, factor_condition = np.empty((0, 0)), np.empty(0), None
        variances, scales = np.empty(0), np.empty(0)
        if self.startup_mode == "stationary_innovations":
            if reflection_state is None and self.provenance.get("fit_scope") != "analytic_reference":
                raise ValueError("fitted stationary innovations requires the actual Burg reflection sequence")
            reflection = (_reflections(coefficients) if reflection_state is None
                          else _real_vector(reflection_state, "actual PARCOR sequence"))
            if (reflection.shape != coefficients.shape or not np.isfinite(reflection).all()
                    or np.any(np.abs(reflection) >= 1-1e-12)):
                raise ValueError("actual PARCOR sequence does not establish strict stability")
            expected = _parcor_coefficients(reflection)
            if not np.allclose(coefficients, expected, rtol=2e-14, atol=2e-14):
                raise ValueError("AR coefficients disagree with the actual PARCOR sequence")
            factor, log_variances, variances, scales, factor_condition = _innovation_factor(reflection)
            covariance, startup_root, condition = np.empty((0, 0)), np.empty((0, 0)), None
            # Root finding on expanded high-order polynomials can falsely report
            # instability. This mode executes the strictly stable PARCOR lattice;
            # the expanded polynomial radius is retained only as a diagnostic.
            radius = float(np.max(np.abs(np.roots(np.r_[1., -coefficients])))) if len(coefficients) else 0.
            if not np.isfinite(radius):
                raise ValueError("expanded polynomial radius diagnostic is nonfinite")
        else:
            if reflection_state is not None:
                raise ValueError("explicit PARCOR state belongs to stationary innovations mode")
            reflection = _reflections(coefficients)
            covariance, startup_root, radius, condition = _startup(coefficients,
                conditional=self.startup_mode == "conditional_prefix", reflections=reflection)
        provenance = _validate_provenance(self.provenance, coefficients, reflection, self.startup_mode, _IMPLEMENTATION_SHA256)
        arrays = {"startup_covariance": covariance, "startup_root": startup_root,
                  "startup_whitening_factor": factor, "startup_log_variances": log_variances,
                  "startup_innovation_variances": variances, "startup_scales": scales}
        marginal = (float(covariance[0, 0]) if len(coefficients) and self.startup_mode == "stationary"
                    else _reference_variance(reflection))
        _check_parameter_relation(coefficients, reflection)
        _validate_compiled_arrays(coefficients, reflection, self.startup_mode, arrays, marginal)
        self._install(coefficients, reflection, innovation_variance, provenance, arrays,
            {"pole_radius": radius, "startup_condition": condition, "startup_factor_condition": factor_condition},
            marginal, {"program": _COMPILER_PROGRAM, "source_sha256": _IMPLEMENTATION_SHA256,
                       "scalar_rounding_source_sha256": COMPILER_SOURCE_SHA256,
                       "state_origin": "PARCOR_prefix_factor_inverse_then_exact_binary64_root_Gram",
                       "scalar_arithmetic": "decimal96_and_integer_binary64_nearest_even"})

    def _install(self, coefficients, reflection, innovation_variance, provenance, arrays, diagnostics, marginal, compiler):
        for name, value in (("coefficients", coefficients), ("reflection_coefficients", reflection),
                ("startup_covariance", arrays["startup_covariance"]), ("_startup_root", arrays["startup_root"]),
                ("startup_whitening_factor", arrays["startup_whitening_factor"]),
                ("_startup_log_variances", arrays["startup_log_variances"]),
                ("startup_innovation_variances", arrays["startup_innovation_variances"]),
                ("_startup_scales", arrays["startup_scales"])):
            object.__setattr__(self, name, frozen_array(value))
        object.__setattr__(self, "innovation_variance", innovation_variance)
        object.__setattr__(self, "provenance", freeze_metadata(provenance))
        object.__setattr__(self, "_marginal_variance_value", marginal)
        object.__setattr__(self, "_compiler", freeze_metadata(compiler))
        for name, value in diagnostics.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_integrity_sha256", metadata_digest(self._state()))
        object.__setattr__(self, "_runtime_binding", self._runtime_signature())

    def _runtime_signature(self):
        # Array buffers and nested metadata are immutable. Verify ownership at
        # consumption without serializing/hashing the entire initializer tree.
        return (self.startup_mode, type(self.innovation_variance), self.innovation_variance,
                self._marginal_variance_value, self.pole_radius, self.startup_condition,
                self.startup_factor_condition, id(self.provenance), id(self._compiler),
                id(self.coefficients), id(self.reflection_coefficients),
                *(id(value) for value in self._arrays().values()))

    @property
    def order(self):
        return len(self.coefficients)

    def _marginal_variance(self):
        return self._marginal_variance_value

    @property
    def marginal_variance_per_unit_innovation(self):
        """Auditable r0 for an explicitly chosen kernel-prior scale convention.

        Stationary mode uses the solved covariance; conditional mode uses the
        reflection-coefficient identity. Neither changes the whitening scale.
        This scalar is a stationary AR reference even when the likelihood is
        conditional; it does not assert a stationary actual observed prefix.
        """
        self.assert_integrity()
        return self._marginal_variance()

    @classmethod
    def fit(cls, records, *, order, record_ids, fit_scope, minimum_samples_per_parameter=20,
            startup_mode="stationary"):
        """Fit pooled Burg shape without crossing independent record starts.

        Inputs must already be residualized by their authorized mean/nuisance
        model. Channels pool the Burg objective but do not multiply the minimum
        independent temporal row count. No labels or evaluation signal is read.
        Degenerate reflections or an ill-conditioned stationary state fail
        explicitly; poles are never clipped into an apparently fitted model.
        """
        if fit_scope not in {"source_training", "target_calibration"}:
            raise ValueError("temporal noise scope must be training or calibration")
        if startup_mode not in _MODES:
            raise ValueError("unsupported temporal startup mode")
        if type(order) is not int or not 0 <= order <= 128:
            raise ValueError("AR order must be an integer in 0..128")
        if type(minimum_samples_per_parameter) is not int or minimum_samples_per_parameter < 2:
            raise ValueError("minimum samples per parameter must be an integer >= 2")
        ids = tuple(record_ids)
        require_identifiers(ids, "temporal fit record IDs")
        arrays = tuple(_values(x, minimum_samples=max(3, order+2)) for x in records)
        if len(arrays) != len(ids) or len({x.shape[1] for x in arrays}) != 1:
            raise ValueError("one distinct record ID per equally shaped observed channel set is required")
        available = sum(len(x)-order for x in arrays)
        minimum = max(32, minimum_samples_per_parameter*max(1, order))
        if available < minimum:
            raise UnidentifiableFitError("insufficient lag rows to identify requested AR order")
        polynomial = np.empty(0)
        actual_reflections = []
        forward, backward = [x[1:].copy() for x in arrays], [x[:-1].copy() for x in arrays]
        for _ in range(order):
            numerator = 2*sum(float(np.sum(f*b)) for f, b in zip(forward, backward, strict=True))
            denominator = sum(float(np.sum(f*f)+np.sum(b*b)) for f, b in zip(forward, backward, strict=True))
            if not np.isfinite(denominator) or denominator <= 0:
                raise UnidentifiableFitError("degenerate Burg noise energy")
            reflection = -numerator/denominator
            if not np.isfinite(reflection) or abs(reflection) >= 1-1e-12:
                raise UnidentifiableFitError("degenerate Burg reflection cannot establish strict stability")
            actual_reflections.append(reflection)
            polynomial = np.r_[polynomial + reflection*polynomial[::-1], reflection]
            pairs = [(f+reflection*b, b+reflection*f) for f, b in zip(forward, backward, strict=True)]
            forward, backward = [f[1:] for f, _ in pairs], [b[:-1] for _, b in pairs]
        coefficients = -polynomial
        squared = 0.
        count = 0
        for index, x in enumerate(arrays):
            if startup_mode == "stationary_innovations" and order:
                innovation = pairs[index][0]  # Actual final Burg lattice errors.
            else:
                innovation = x[order:].copy()
                for lag, coefficient in enumerate(coefficients, start=1):
                    innovation -= coefficient*x[order-lag:len(x)-lag]
            squared += float(np.sum(innovation**2))
            count += innovation.size
        variance = squared/count
        if not np.isfinite(variance) or variance <= 0:
            raise UnidentifiableFitError("degenerate pooled innovation variance")
        return cls(coefficients, variance, {
            "fit_scope": fit_scope, "method": _BURG_METHOD,
            "assume_centered": True, "available_lag_rows": available, "minimum_lag_rows": minimum,
            "minimum_samples_per_parameter": minimum_samples_per_parameter, "n_columns": arrays[0].shape[1],
            "implementation_sha256": _IMPLEMENTATION_SHA256,
            "records": [{"record_id": name, "n_samples": len(x), "input_sha256": array_digest(x)}
                        for name, x in zip(ids, arrays, strict=True)],
        }, startup_mode=startup_mode,
            reflection_state=np.asarray(actual_reflections) if startup_mode == "stationary_innovations" else None)

    @classmethod
    def from_parameters(cls, coefficients, *, innovation_variance, reference_id, startup_mode="stationary"):
        """An explicitly analytic reference; does not claim a data fit."""
        return cls(coefficients, innovation_variance,
                   {"fit_scope": "analytic_reference", "reference_id": reference_id, "records": []},
                   startup_mode=startup_mode)

    @classmethod
    def from_reflections(cls, reflections, *, innovation_variance, reference_id):
        """Analytic stable lattice reference without inverse polynomial stepdown."""
        values = _real_vector(reflections, "analytic PARCOR")
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(np.abs(values) >= 1-1e-12):
            raise ValueError("analytic PARCOR requires a finite strictly stable vector")
        return cls(_parcor_coefficients(values), innovation_variance,
                   {"fit_scope": "analytic_reference", "reference_id": reference_id, "records": []},
                   startup_mode="stationary_innovations", reflection_state=values)

    @classmethod
    def from_conditional_fit(cls, initializer, reflections, *, innovation_variance,
                             conditioning_samples, objective_rows, optimization,
                             training_input_sha256):
        """Publish a caller-owned fixed-prefix conditional fit as immutable shape.

        The caller owns residual means, prefix/tail Gaussian objectives and the
        channel covariance whose trace supplies ``innovation_variance``. This
        owner retains the actual segmented Burg initializer and final PARCOR;
        it performs no label access, mean fit or likelihood optimization. Its
        whitening remains the full unit-innovation lattice, leaving conditional
        predictive calculations and fixed-prefix slicing to that caller.

        A published conditional-profile result must be converged and no worse
        than initialization. ``burg_only`` instead retains the actual Burg
        sequence with an unchanged objective and zero optimizer iterations.
        Unconverged numerical attempts belong to the caller's failure ledger.
        """
        if not isinstance(initializer, TemporalNoise):
            raise ValueError("conditional noise initialization must be a fitted TemporalNoise")
        if initializer._compiler["source_sha256"] != _IMPLEMENTATION_SHA256:
            raise ValueError("new conditional fit requires the current declared Burg producer implementation")
        initialization = initializer.record()
        values = _real_vector(reflections, "conditional fitted PARCOR")
        provenance = dict(initializer.provenance)
        provenance.update(method=_CONDITIONAL_METHOD, initialization=initialization,
                          conditioning_samples=conditioning_samples, objective_rows=objective_rows,
                          optimization=optimization, training_input_sha256=training_input_sha256)
        return cls(_parcor_coefficients(values), innovation_variance, provenance,
                   startup_mode="stationary_innovations", reflection_state=values)

    def whiten(self, values):
        """Return innovations under the declared startup probability model.

        The first p samples use the p-by-p stationary covariance for unit
        innovation. Remaining rows use their actual within-record history. Each
        call starts a new independent record. Conditional-prefix mode instead
        uses actual first-p observations as history and returns only n-p rows.
        Candidate-free prefix admission belongs to the scientific caller, not
        this residual-array interface. Conditioning is not marginalization;
        callers cannot silently discard labels or refit unknown nuisance priors.
        """
        self.assert_integrity()
        minimum = self.order+1 if self.startup_mode == "conditional_prefix" else max(1, self.order)
        x = _values(values, minimum_samples=minimum)
        if self.startup_mode == "stationary_innovations":
            return _lattice_innovations(x, self.reflection_coefficients, self._startup_scales)
        if self.startup_mode == "conditional_prefix":
            result = x[self.order:].copy()
            for lag, coefficient in enumerate(self.coefficients, start=1):
                result -= coefficient*x[self.order-lag:len(x)-lag]
            return result
        result = x.copy()
        if self.order:
            result[:self.order] = solve_triangular(self._startup_root, x[:self.order], lower=True)
            for lag, coefficient in enumerate(self.coefficients, start=1):
                result[self.order:] -= coefficient*x[self.order-lag:len(x)-lag]
        return result

    def _arrays(self):
        return {"startup_covariance": self.startup_covariance, "startup_root": self._startup_root,
                "startup_whitening_factor": self.startup_whitening_factor,
                "startup_log_variances": self._startup_log_variances,
                "startup_innovation_variances": self.startup_innovation_variances,
                "startup_scales": self._startup_scales}

    @property
    def identity_digest(self):
        """Identity admitted once; immutable internal consumers need no recompile."""
        return self._integrity_sha256

    def _state(self):
        stationary = self.startup_mode == "stationary"
        lattice = self.startup_mode == "stationary_innovations"
        conditional_fit = self.provenance.get("method") == _CONDITIONAL_METHOD
        arrays = self._arrays()
        origin = ("actual_Burg_recursion_retained" if conditional_fit and self.provenance["optimization"]["strategy"] == "burg_only"
                  else "conditional_profile_optimized_PARCOR" if conditional_fit
                  else "analytic_reference" if self.provenance["fit_scope"] == "analytic_reference"
                  else "actual_Burg_recursion" if lattice else "expanded_AR_stepdown_stability_certificate")
        return {"schema": _SCHEMA, "compiler": self._compiler, "startup_mode": self.startup_mode,
                "coefficients": self.coefficients.tolist(), "reflection_coefficients": self.reflection_coefficients.tolist(),
                "innovation_variance": self.innovation_variance, "provenance": self.provenance,
                "scale_convention": _SCALE_CONVENTION,
                "startup_assumption": "stationary_AR_latent_initial_state" if stationary or lattice else "conditional_on_observed_first_p_samples",
                "conditioning_prefix_samples": self.order if self.startup_mode == "conditional_prefix" else None,
                "marginal_variance_per_unit_innovation": self._marginal_variance_value,
                "pole_radius": self.pole_radius, "startup_condition": self.startup_condition,
                "startup_factor_condition": self.startup_factor_condition,
                "floating_precision_condition_indicator": (None if self.startup_factor_condition is None else
                    float(np.finfo(float).eps*self.startup_factor_condition)),
                "expanded_polynomial_radius_role": "producer_diagnostic_not_compiled_state_authority",
                "stability_certificate": "actual_PARCOR_strictly_inside_unit_interval",
                "reflection_origin": origin,
                "compiled_state": {"dtype": "float64", "shapes": {key: list(value.shape) for key, value in arrays.items()},
                    "arrays": {key: value.tolist() for key, value in arrays.items()},
                    "state_sha256": array_digest(self.coefficients, self.reflection_coefficients, *arrays.values())}}

    def assert_integrity(self):
        if self._runtime_signature() != self._runtime_binding:
            raise ValueError("frozen temporal noise integrity changed")

    def record(self):
        self.assert_integrity()
        result = self._state()
        result["artifact_sha256"] = metadata_digest(result)
        return result

    @classmethod
    def from_record(cls, record):
        try:
            payload = json.loads(json.dumps(record, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("temporal compiled artifact must contain finite JSON numeric state") from error
        if payload.get("schema") != _SCHEMA:
            raise ValueError("unsupported temporal compiled contract; use the original execution source for schemas 1-4")
        expected = payload.pop("artifact_sha256", None)
        if expected != metadata_digest(payload):
            raise ValueError("temporal noise artifact digest mismatch")
        mode, compiler = payload.get("startup_mode"), payload.get("compiler")
        if (mode not in _MODES or not isinstance(compiler, dict)
                or set(compiler) != {"program", "source_sha256", "scalar_rounding_source_sha256", "state_origin", "scalar_arithmetic"}
                or compiler["program"] != _COMPILER_PROGRAM
                or compiler["state_origin"] != "PARCOR_prefix_factor_inverse_then_exact_binary64_root_Gram"
                or compiler["scalar_arithmetic"] != "decimal96_and_integer_binary64_nearest_even"):
            raise ValueError("unsupported temporal compiled program contract")
        require_sha256(compiler["source_sha256"], "temporal compiler source")
        require_sha256(compiler["scalar_rounding_source_sha256"], "scalar rounding compiler source")
        coefficients = _real_vector(payload["coefficients"], "AR coefficients")
        reflection = _real_vector(payload["reflection_coefficients"], "actual PARCOR sequence")
        _check_parameter_relation(coefficients, reflection)
        variance = _real_scalar(payload["innovation_variance"], "innovation variance", positive=True)
        provenance = _validate_provenance(payload["provenance"], coefficients, reflection, mode, compiler["source_sha256"])
        compiled = payload.get("compiled_state")
        if (not isinstance(compiled, dict) or set(compiled) != {"dtype", "shapes", "arrays", "state_sha256"}
                or compiled["dtype"] != "float64" or set(compiled["arrays"]) != set(_ARRAY_FIELDS)
                or set(compiled["shapes"]) != set(_ARRAY_FIELDS)):
            raise ValueError("temporal compiled state lacks its complete numerical payload")
        arrays = {}
        for key in _ARRAY_FIELDS:
            shape = compiled["shapes"][key]
            if not isinstance(shape, list) or any(type(value) is not int or value < 0 for value in shape):
                raise ValueError("temporal compiled array shape is invalid")
            raw = np.asarray(compiled["arrays"][key])
            if raw.dtype != np.dtype(np.float64) or not np.isfinite(raw).all():
                raise ValueError("temporal compiled state must use finite real float64 values")
            if raw.size == 0 and 0 in shape:
                raw = raw.reshape(shape)
            if list(raw.shape) != shape:
                raise ValueError("temporal compiled array payload differs from its declared shape")
            arrays[key] = raw
        marginal = _real_scalar(payload["marginal_variance_per_unit_innovation"], "marginal variance", positive=True)
        _validate_compiled_arrays(coefficients, reflection, mode, arrays, marginal)
        if compiled["state_sha256"] != array_digest(coefficients, reflection, *arrays.values()):
            raise ValueError("temporal compiled numerical state digest differs from its payload")
        diagnostics = {key: payload[key] for key in ("pole_radius", "startup_condition", "startup_factor_condition")}
        for key, value in diagnostics.items():
            if value is None:
                if key == "pole_radius" or key == "startup_condition" and mode == "stationary" or key == "startup_factor_condition" and mode == "stationary_innovations":
                    raise ValueError("temporal producer diagnostic is missing")
            else:
                diagnostics[key] = _real_scalar(value, key, positive=key != "pole_radius")
                if key == "pole_radius" and value < 0:
                    raise ValueError("temporal pole radius diagnostic must be nonnegative")
        if (mode != "stationary" and diagnostics["startup_condition"] is not None
                or mode != "stationary_innovations" and diagnostics["startup_factor_condition"] is not None):
            raise ValueError("temporal producer diagnostics differ from their startup mode")
        result = object.__new__(cls)
        object.__setattr__(result, "startup_mode", mode)
        result._install(coefficients, reflection, variance, provenance, arrays, diagnostics, marginal, compiler)
        # Exactness applies to the saved record and program, after independent
        # relation admission. Diagnostics stay the producer's values: no eig,
        # cond, Lyapunov solver or startup compiler replaces them on load.
        if metadata_digest(result._state()) != metadata_digest(payload):
            raise ValueError("temporal compiled contract replay differs from its saved record")
        return result

    def save(self, path):
        payload = json.dumps(self.record(), sort_keys=True, allow_nan=False)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            np.savez_compressed(handle, metadata=np.asarray(payload))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["metadata"]:
                raise ValueError("unexpected temporal noise archive members")
            record = json.loads(str(archive["metadata"]))
        return cls.from_record(record)
