"""Calibration quotients and explicit physical completion of unseen kernels.

The quotient defines distributions for estimable response functionals only.
Completion is a separate proper matrix-normal prior: its unseen-direction
variance is a declared physical assumption, never an implicit pseudoinverse.
Both use the equal-class mean squared ERP metric on the actual response grid.
No signal, noise fit, held label or held design enters compilation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from research.execution import UnidentifiableFitError
from transfer.noise_model import array_digest, freeze_metadata, frozen_array, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _matrix(value, name):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a finite real matrix")
    return np.asarray(raw, dtype=np.float64)


def _positive(value, name):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu" or not np.isfinite(raw) or raw <= 0:
        raise ValueError(f"{name} must be a positive finite real scalar")
    return float(raw)


def _same_factor_relation(saved, calculated, *, precision, tolerance):
    """Compare in expected prior scales, so weak directions cannot hide errors.

    QR/SVD factor signs and rotations are immaterial. A single unscaled Gram
    norm could miss a large relative change to a small-variance direction when
    another direction is huge. The saved source arrays remain authoritative
    only after this relation to the declared program has been checked.
    """
    if not calculated.shape[1]:
        return np.array_equal(saved, calculated)
    if precision:
        normalized = solve_triangular(calculated.T, saved.T, lower=True).T
        expected = np.eye(len(calculated))
    else:
        q, root = np.linalg.qr(calculated, mode="reduced")
        image = q.T@saved
        outside = saved-q@image
        column_scales = np.maximum(np.linalg.norm(saved, axis=0), np.finfo(float).tiny)
        if np.any(np.linalg.norm(outside, axis=0) > tolerance*column_scales):
            return False
        normalized = solve_triangular(root, image)
        baseline = solve_triangular(root, q.T@calculated)
        expected = baseline@baseline.T
    return (np.isfinite(normalized).all()
            and np.linalg.norm(normalized@normalized.T-expected)
            <= tolerance*max(float(np.linalg.norm(expected)), np.finfo(float).tiny))


def rank_ledger_matches(saved, actual, *, float_fields, rtol):
    """One external ledger admission: real scalar/vector shapes precede allclose."""
    if set(saved) != set(actual):
        return False
    for key, expected in actual.items():
        value = saved[key]
        if key in float_fields:
            raw, reference = np.asarray(value), np.asarray(expected)
            if (raw.dtype.kind not in "fiu" or raw.shape != reference.shape
                    or not np.isfinite(raw).all() or not np.allclose(raw, reference, rtol=rtol, atol=0)):
                return False
        elif type(value) is not type(expected) or value != expected:
            return False
    return True


@dataclass(frozen=True)
class PhysicalResponseMetric:
    """Equal-class mean response energy, including common/target cross terms.

    ``basis`` owns the dimensionless physical function grid. ``coordinate_map``
    maps actual kernel coordinates into its common/increment coordinates; an
    arbitrary invertible change must transform this map too. ``binding`` is
    the same authoritative C/basis/input identity used by the model compiler.
    A singular physical function map is rejected, rather than ridge repaired.
    """

    basis: np.ndarray
    sample_rate_hz: float
    binding: dict
    coordinate_map: np.ndarray | None = None
    precision_root: np.ndarray = field(init=False, repr=False, compare=False)
    _integrity: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        basis = _matrix(self.basis, "physical ERP basis")
        rate = _positive(self.sample_rate_hz, "physical sample rate")
        if min(basis.shape) == 0:
            raise ValueError("physical response metric needs a nonempty actual function grid")
        width = 2*basis.shape[1]
        change = np.eye(width) if self.coordinate_map is None else _matrix(self.coordinate_map, "kernel coordinate map")
        if change.shape != (width, width):
            raise ValueError("physical metric coordinate map must match both response blocks")
        zero = np.zeros_like(basis)
        response = np.block([[basis, zero], [basis, basis]])@change
        # dt/(2*Tactual) = 1/(2*L): a mean energy, not sample-count energy.
        weighted = response/np.sqrt(2*len(basis))
        singular = np.linalg.svd(weighted, compute_uv=False)
        threshold = np.finfo(float).eps*max(weighted.shape)*singular[0]
        if len(singular) != width or np.any(singular <= threshold):
            raise UnidentifiableFitError("physical ERP metric has unresolved function coordinates")
        root = np.linalg.qr(weighted, mode="r")
        binding = json.loads(json.dumps(self.binding, allow_nan=False))
        if not isinstance(binding, dict) or not binding:
            raise ValueError("physical response metric requires its authoritative model binding")
        if "basis_sha256" in binding:
            if (not np.array_equal(change, np.eye(width))
                    or binding["basis_sha256"] != array_digest(basis)):
                raise ValueError("physical response metric differs from the actual basis/common-increment coordinate binding")
        if "sample_rate_hz" in binding and binding["sample_rate_hz"] != rate:
            raise ValueError("physical response metric differs from the actual time grid binding")
        for name, value in (("basis", basis), ("coordinate_map", change), ("precision_root", root)):
            object.__setattr__(self, name, frozen_array(value))
        object.__setattr__(self, "sample_rate_hz", rate)
        object.__setattr__(self, "binding", freeze_metadata(binding))
        object.__setattr__(self, "_integrity", metadata_digest(self._record()))

    @classmethod
    def from_basis(cls, basis, *, sample_rate_hz, binding):
        return cls(basis, sample_rate_hz, binding)

    def pullback(self, change, *, binding):
        """Preserve the same physical functions under D'=D S, beta'=S^-1 beta."""
        self.digest()
        return type(self)(self.basis, self.sample_rate_hz, binding,
                          self.coordinate_map@_matrix(change, "kernel basis change"))

    def validate_binding(self, binding):
        if metadata_digest(self.binding) != metadata_digest(binding):
            raise ValueError("physical response metric differs from the actual C/basis/input binding")

    def _record(self):
        return {"schema": "n2p3_physical_response_metric/1",
                "program": "equal_class_mean_squared_response",
                "response_roles": ["non_target_common", "target_common_plus_increment"],
                "basis": self.basis.tolist(), "coordinate_map": self.coordinate_map.tolist(),
                "precision_root": self.precision_root.tolist(), "sample_rate_hz": self.sample_rate_hz,
                "support_samples": [0, len(self.basis)], "actual_seconds": len(self.basis)/self.sample_rate_hz,
                "quadrature_weight": 1/(2*len(self.basis)), "column_units": "dimensionless",
                "coefficient_units": "volts", "row_scale": "relative_to_innovation_covariance",
                "binding": self.binding, "implementation_sha256": IMPLEMENTATION_SHA256}

    def digest(self):
        if metadata_digest(self._record()) != self._integrity:
            raise ValueError("frozen physical response metric changed")
        return self._integrity

    def record(self):
        result = json.loads(json.dumps(self._record(), allow_nan=False))
        result["artifact_sha256"] = self.digest()
        return result

    @classmethod
    def from_record(cls, record):
        values = json.loads(json.dumps(record, allow_nan=False))
        expected = values.pop("artifact_sha256", None)
        if (values.get("schema") != "n2p3_physical_response_metric/1"
                or metadata_digest(values) != expected):
            raise ValueError("physical metric record identity changed")
        result = cls(values["basis"], values["sample_rate_hz"], values["binding"], values["coordinate_map"])
        # Recompute the physical relation; an outer checksum alone is insufficient.
        actual = result._record()
        saved_root = np.asarray(values["precision_root"])
        actual_root = np.asarray(actual.pop("precision_root"))
        other = {k: v for k, v in values.items() if k != "precision_root"}
        actual_gram = actual_root.T@actual_root
        scale = max(float(np.linalg.norm(actual_gram)), np.finfo(float).tiny)
        if (metadata_digest(actual) != metadata_digest(other) or saved_root.shape != actual_root.shape
                or not np.isfinite(saved_root).all() or np.any(np.tril(saved_root, -1))
                or np.any(np.diag(saved_root) == 0)
                or np.linalg.norm(saved_root.T@saved_root-actual_gram) > 2e-12*scale
                or not _same_factor_relation(saved_root, actual_root, precision=True, tolerance=2e-10)):
            raise ValueError("physical metric numerical state differs from its actual functions")
        object.__setattr__(result, "precision_root", frozen_array(saved_root))
        object.__setattr__(result, "_integrity", metadata_digest(result._record()))
        return result


class PredictivePriorSupportError(UnidentifiableFitError):
    """An admissible held design asks the C quotient about an unseen response."""

    def __init__(self, ledger):
        self._evidence = freeze_metadata({"kind": "kernel_quotient_prediction_failure", **ledger})
        super().__init__("prediction functional leaves the calibration identified quotient; explicit completion required")

    def record(self):
        return json.loads(json.dumps(self._evidence, allow_nan=False))


@dataclass(frozen=True)
class CompiledPredictivePrior:
    """A fixed-C quotient, optionally completed in the physical response metric.

    ``reference_root`` is the all-C AR0 diffuse-affine residual design root;
    ``design_root`` retains its unprojected scale to resolve projection error.
    ``completion_variance=None`` leaves unseen responses unsupported. A positive
    value explicitly supplies their variance relative to innovation covariance.
    Near-rank truncation is recorded as numerical model resolution; it is not a
    statistical identifiability claim or an automatically selected regularizer.
    """

    metric: PhysicalResponseMetric
    reference_root: np.ndarray
    design_root: np.ndarray
    observed_rows: int
    n_records: int
    g: float
    completion_variance: float | None = None
    identified_vectors: np.ndarray = field(init=False, repr=False, compare=False)
    null_vectors: np.ndarray = field(init=False, repr=False, compare=False)
    covariance_factor: np.ndarray = field(init=False, repr=False, compare=False)
    precision_root: np.ndarray | None = field(init=False, repr=False, compare=False)
    rank_ledger: dict = field(init=False, compare=False)
    _integrity: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.metric, PhysicalResponseMetric):
            raise ValueError("predictive prior requires its declared physical metric")
        self.metric.digest()
        reference = _matrix(self.reference_root, "calibration reference root")
        original = _matrix(self.design_root, "original calibration design root")
        width = len(self.metric.precision_root)
        if (reference.shape[1] != width or original.shape[1] != width
                or type(self.observed_rows) is not int or type(self.n_records) is not int
                or self.n_records < 1 or self.observed_rows < 3*self.n_records):
            raise ValueError("predictive geometry differs from actual C rows and kernel coordinates")
        g = _positive(self.g, "g variance multiplier")
        tau = (None if self.completion_variance is None
               else _positive(self.completion_variance, "unidentified physical response variance"))
        metric_root = self.metric.precision_root
        whitened = solve_triangular(metric_root.T, reference.T, lower=True).T
        unprojected = solve_triangular(metric_root.T, original.T, lower=True).T
        _, singular, right = np.linalg.svd(whitened, full_matrices=True)
        original_scale = float(np.linalg.norm(unprojected, ord=2))
        residual_scale = float(singular[0]) if len(singular) else 0.
        relative = float(np.finfo(float).eps*max(self.observed_rows, width))
        threshold = relative*max(original_scale, residual_scale)
        rank = int(np.sum(singular > threshold))
        identified, null = right[:rank].T, right[rank:].T
        factors = [identified*(np.sqrt(g)/singular[:rank])]
        if tau is not None:
            factors.append(null*np.sqrt(tau))
        factor = solve_triangular(metric_root, np.column_stack(factors))
        precision = None
        if tau is not None:
            # QR of physical row roots avoids forming/inverting a squared Gram.
            stacked = np.vstack((singular[:rank, None]*right[:rank]/np.sqrt(g), right[rank:]/np.sqrt(tau)))@metric_root
            precision = np.linalg.qr(stacked, mode="r")
            if not np.isfinite(precision).all() or np.any(np.diag(precision) == 0):
                raise UnidentifiableFitError("physical completion precision is not numerically resolved")
        if not np.isfinite(factor).all():
            raise UnidentifiableFitError("predictive prior variance is not representable in float64")
        ledger = {"requested_dimensions": width, "rank": rank, "unidentified_dimensions": width-rank,
                  "observed_rows": self.observed_rows, "nuisance_dimensions": 2*self.n_records,
                  "residual_rows": self.observed_rows-2*self.n_records,
                  "singular_values": singular.tolist(), "relative_threshold": relative,
                  "absolute_threshold": threshold, "original_design_scale": original_scale,
                  "residual_design_scale": residual_scale,
                  "criterion": "float64_physical_metric_original_design_projection_resolution",
                  "near_rank_boundary": bool(threshold > 0 and np.any((singular > threshold/16)&(singular < threshold*16))),
                  "discarded_reference_prior_energy": (0. if tau is None else float(tau*np.sum(singular[rank:]**2))),
                  "reference_prior_energy": float(rank*g+(0. if tau is None else tau*np.sum(singular[rank:]**2))),
                  "statistical_identifiability_claimed": False,
                  "completion_assumption": "none" if tau is None else "isotropic_in_unseen_physical_response_metric"}
        for name, value in (("reference_root", reference), ("design_root", original),
                            ("identified_vectors", identified), ("null_vectors", null), ("covariance_factor", factor)):
            object.__setattr__(self, name, frozen_array(value))
        object.__setattr__(self, "g", g)
        object.__setattr__(self, "completion_variance", tau)
        object.__setattr__(self, "precision_root", None if precision is None else frozen_array(precision))
        object.__setattr__(self, "rank_ledger", freeze_metadata(ledger))
        object.__setattr__(self, "_integrity", metadata_digest(self._record()))

    def prediction_support(self, design):
        """Inspect a functional after the caller's actual prediction nuisance map.

        Raw held designs are a conservative check. Conditional prefix likelihood
        needs its complete prefix/tail/diffuse geometry; checking the tail alone
        would omit the kernel update in the conditioning prefix.
        """
        value = _matrix(design, "prediction functional")
        if value.shape[1] != len(self.metric.precision_root):
            raise ValueError("prediction functional differs from frozen kernel coordinates")
        physical = solve_triangular(self.metric.precision_root.T, value.T, lower=True).T
        if not np.isfinite(physical).all():
            raise UnidentifiableFitError("prediction functional is not representable in the physical metric")
        # Each row is an independently requested response functional. Normalize
        # its own nonzero scale: another large identified row cannot hide a
        # small unseen row, and tiny squares cannot underflow into fake support.
        row_scales = np.max(np.abs(physical), axis=1, initial=0.)
        normalized = np.divide(physical, row_scales[:, None], out=np.zeros_like(physical),
                               where=row_scales[:, None] != 0)
        outside_relative = np.linalg.norm(normalized@self.null_vectors, axis=1)
        relative_tolerance = 64*np.finfo(float).eps*max(1, value.shape[1])*np.linalg.norm(normalized, axis=1)
        unsupported = outside_relative > relative_tolerance
        return {"identified_rank": self.rank_ledger["rank"], "requested_dimensions": value.shape[1],
                "functional_rows": len(value),
                "functional_row_scales": row_scales.tolist(),
                "outside_relative_norm_by_row": outside_relative.tolist(),
                "relative_tolerance_by_row": relative_tolerance.tolist(),
                "unsupported_functional_rows": np.flatnonzero(unsupported).tolist(),
                "supported": not np.any(unsupported),
                "support_space": "every_functional_in_physical_metric_resolved_C_row_space"}

    def predictive_covariance(self, design):
        """Return response row covariance, rejecting unsupported quotient use."""
        support = self.prediction_support(design)
        if self.completion_variance is None and not support["supported"]:
            raise PredictivePriorSupportError(support)
        image = _matrix(design, "prediction functional")@self.covariance_factor
        covariance = image@image.T
        if not np.isfinite(covariance).all():
            raise UnidentifiableFitError("predictive covariance is not representable in float64")
        return frozen_array(covariance)

    def _record(self):
        return {"schema": "n2p3_predictive_kernel_prior/1", "physical_metric": self.metric.record(),
                "reference_root": self.reference_root.tolist(), "design_root": self.design_root.tolist(),
                "observed_rows": self.observed_rows, "n_records": self.n_records, "g": self.g,
                "completion_variance": self.completion_variance, "rank_ledger": self.rank_ledger,
                "identified_vectors": self.identified_vectors.tolist(), "null_vectors": self.null_vectors.tolist(),
                "covariance_factor": self.covariance_factor.tolist(),
                "precision_root": None if self.precision_root is None else self.precision_root.tolist(),
                "implementation_sha256": IMPLEMENTATION_SHA256}

    def digest(self):
        if metadata_digest(self._record()) != self._integrity:
            raise ValueError("frozen predictive kernel prior changed")
        return self._integrity

    def record(self):
        result = json.loads(json.dumps(self._record(), allow_nan=False))
        result["artifact_sha256"] = self.digest()
        return result

    @classmethod
    def from_record(cls, record):
        values = json.loads(json.dumps(record, allow_nan=False))
        expected = values.pop("artifact_sha256", None)
        if values.get("schema") != "n2p3_predictive_kernel_prior/1" or metadata_digest(values) != expected:
            raise ValueError("predictive prior record identity changed")
        result = cls(PhysicalResponseMetric.from_record(values["physical_metric"]), values["reference_root"],
                     values["design_root"], values["observed_rows"], values["n_records"], values["g"], values["completion_variance"])
        actual = result._record()
        arrays = ("identified_vectors", "null_vectors", "covariance_factor", "precision_root")
        # Saved roots remain the producer's numeric state. Recompute scientific
        # relations, allowing QR/SVD roundoff and eigenvector signs on replay.
        for name in arrays:
            saved, calculated = values[name], actual[name]
            if saved is None or calculated is None:
                if saved is not calculated:
                    raise ValueError("predictive prior completion program changed")
                continue
            saved, calculated = np.asarray(saved), np.asarray(calculated)
            if saved.shape != calculated.shape or not np.isfinite(saved).all():
                raise ValueError("predictive prior numerical dimensions changed")
            if name == "precision_root" and (np.any(np.tril(saved, -1)) or np.any(np.diag(saved) == 0)):
                raise ValueError("predictive prior precision root must remain full-rank upper triangular")
            saved_gram = saved.T@saved if name == "precision_root" else saved@saved.T
            actual_gram = calculated.T@calculated if name == "precision_root" else calculated@calculated.T
            scale = max(float(np.linalg.norm(actual_gram)), np.finfo(float).tiny)
            if np.linalg.norm(saved_gram-actual_gram) > 2e-11*scale:
                raise ValueError("predictive prior numerical relation changed")
            if (name in {"covariance_factor", "precision_root"}
                    and not _same_factor_relation(saved, calculated, precision=name == "precision_root", tolerance=2e-9)):
                raise ValueError("predictive prior numerical relation changed in a resolved direction")
            object.__setattr__(result, name, frozen_array(saved))
            actual.pop(name)
        other = {k: v for k, v in values.items() if k not in arrays}
        for name in arrays:
            actual.pop(name, None)
        # Rank thresholds are scientific ledger state; do not accept re-signed
        # arbitrary values. Numeric roots carry the tolerance, metadata exactness
        # is reserved for program/identity/dimensions.
        actual_ledger = actual.pop("rank_ledger")
        saved_ledger = other.pop("rank_ledger")
        float_fields = {"singular_values", "relative_threshold", "absolute_threshold",
                        "original_design_scale", "residual_design_scale",
                        "discarded_reference_prior_energy", "reference_prior_energy"}
        if (not rank_ledger_matches(saved_ledger, actual_ledger, float_fields=float_fields, rtol=2e-12)
                or metadata_digest(actual) != metadata_digest(other)):
            raise ValueError("predictive prior rank, source or physical program changed")
        object.__setattr__(result, "rank_ledger", freeze_metadata(saved_ledger))
        object.__setattr__(result, "_integrity", metadata_digest(result._record()))
        return result
