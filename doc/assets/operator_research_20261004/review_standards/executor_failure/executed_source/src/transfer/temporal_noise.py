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
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov, solve_triangular

from transfer.noise_model import (
    array_digest,
    freeze_metadata,
    frozen_array,
    metadata_digest,
    require_identifiers,
    require_sha256,
)

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_SCHEMA = "n2p3_temporal_noise/1"
_SCALE_CONVENTION = "unit_innovation_temporal_shape_preserve_data_units"
_MODES = {"stationary", "conditional_prefix", "stationary_innovations"}


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
    current = -np.asarray(coefficients, dtype=float).copy()
    reflection = np.empty(len(current))
    for order in range(len(current), 0, -1):
        k = float(current[-1])
        if not np.isfinite(k) or abs(k) >= 1 - 1e-12:
            raise ValueError("AR polynomial is not strictly stable in reflection coordinates")
        reflection[order-1] = k
        current = (current[:-1] - k * current[:-1][::-1]) / (1 - k*k)
    return reflection


def _startup(coefficients, *, conditional=False):
    order = len(coefficients)
    if not order:
        return np.empty((0, 0)), 0., None if conditional else 1.
    companion = np.zeros((order, order))
    companion[0] = coefficients
    companion[1:, :-1] = np.eye(order-1)
    radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
    if radius >= 1 - 1e-12:
        raise ValueError("AR poles are not strictly stable")
    if conditional:
        # The caller conditions on actual prefix samples. There is no latent
        # stationary startup covariance to form or numerically regularize.
        return np.empty((0, 0)), radius, None
    driving = np.zeros((order, order))
    driving[0, 0] = 1
    covariance = solve_discrete_lyapunov(companion, driving)
    covariance = (covariance + covariance.T) / 2
    residual = covariance - companion @ covariance @ companion.T - driving
    if not np.isfinite(covariance).all() or np.linalg.norm(residual) > 1e-8 * max(1., np.linalg.norm(covariance)):
        raise ValueError("stationary startup covariance solve failed")
    condition = float(np.linalg.cond(covariance))
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError("stationary startup covariance is numerically unidentifiable")
    # State ordering reverses time; a stationary scalar AR covariance is symmetric Toeplitz.
    np.linalg.cholesky(covariance)
    return covariance, radius, condition


def _parcor_coefficients(reflections):
    polynomial = np.empty(0)
    for reflection in reflections:
        polynomial = np.r_[polynomial+reflection*polynomial[::-1], reflection]
    return -polynomial


def _innovation_factor(reflections):
    """Levinson prefix factor; no covariance inversion or pseudo-PSD repair."""
    order = len(reflections)
    log_variances = -np.cumsum(np.log1p(-reflections[::-1]**2))[::-1]
    factor = np.zeros((order, order))
    polynomial = np.empty(0)
    for t in range(order):
        scale = np.exp(-.5*log_variances[t])
        factor[t, t] = scale
        factor[t, :t] = polynomial[::-1]*scale
        polynomial = np.r_[polynomial+reflections[t]*polynomial[::-1], reflections[t]]
    if not np.isfinite(factor).all() or np.any(np.diag(factor) <= 0):
        raise ValueError("stationary innovations factor is not finitely invertible")
    condition = float(np.linalg.cond(factor)) if order else 1.
    if not np.isfinite(condition):
        raise ValueError("stationary innovations factor condition is nonfinite")
    return factor, log_variances, condition


def _lattice_innovations(values, reflections, log_variances):
    """Stable PARCOR recursion for prefix and full-order innovations alike."""
    order = len(reflections)
    if not order:
        return values.copy()
    output = np.empty_like(values)
    output[0] = values[0]*np.exp(-.5*log_variances[0])
    forward, backward = values[1:].copy(), values[:-1].copy()
    for m, reflection in enumerate(reflections, 1):
        next_forward = forward+reflection*backward
        next_backward = backward+reflection*forward
        if m < order:
            output[m] = next_forward[0]*np.exp(-.5*log_variances[m])
            forward, backward = next_forward[1:], next_backward[:-1]
        else:
            output[order:] = next_forward
    if not np.isfinite(output).all():
        raise ValueError("stationary lattice innovations overflowed")
    return output


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
    _integrity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, reflection_state):
        if self.startup_mode not in _MODES:
            raise ValueError("unsupported temporal startup mode")
        coefficients = np.asarray(self.coefficients, dtype=float)
        if coefficients.ndim != 1 or len(coefficients) > 128 or not np.isfinite(coefficients).all():
            raise ValueError("AR coefficients must be a finite vector of order at most 128")
        if (isinstance(self.innovation_variance, bool) or not np.isfinite(self.innovation_variance)
                or self.innovation_variance <= 0):
            raise ValueError("innovation variance must be positive and finite")
        factor, log_variances, factor_condition = np.empty((0, 0)), np.empty(0), None
        if self.startup_mode == "stationary_innovations":
            if reflection_state is None and self.provenance.get("fit_scope") != "analytic_reference":
                raise ValueError("fitted stationary innovations requires the actual Burg reflection sequence")
            reflection = _reflections(coefficients) if reflection_state is None else np.asarray(reflection_state, dtype=float)
            if (reflection.shape != coefficients.shape or not np.isfinite(reflection).all()
                    or np.any(np.abs(reflection) >= 1-1e-12)):
                raise ValueError("actual PARCOR sequence does not establish strict stability")
            expected = _parcor_coefficients(reflection)
            if not np.allclose(coefficients, expected, rtol=2e-14, atol=2e-14):
                raise ValueError("AR coefficients disagree with the actual PARCOR sequence")
            factor, log_variances, factor_condition = _innovation_factor(reflection)
            covariance, condition = np.empty((0, 0)), None
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
            covariance, radius, condition = _startup(coefficients, conditional=self.startup_mode == "conditional_prefix")
        provenance = json.loads(json.dumps(self.provenance, allow_nan=False))
        if provenance.get("fit_scope") == "analytic_reference":
            if set(provenance) != {"fit_scope", "reference_id", "records"} or not isinstance(provenance["reference_id"], str) or not provenance["reference_id"].strip() or provenance["records"]:
                raise ValueError("analytic temporal reference must be named without a training ledger")
        else:
            required = {"fit_scope", "records", "method", "assume_centered", "available_lag_rows",
                        "minimum_lag_rows", "minimum_samples_per_parameter", "n_columns", "implementation_sha256"}
            if (set(provenance) != required or provenance.get("fit_scope") not in {"source_training", "target_calibration"}
                    or provenance.get("method") != "segmented_multichannel_burg_no_pole_clipping"
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
            if provenance["implementation_sha256"] != _IMPLEMENTATION_SHA256:
                raise ValueError("temporal noise artifact was fitted by a different implementation")
        object.__setattr__(self, "coefficients", frozen_array(coefficients))
        object.__setattr__(self, "reflection_coefficients", frozen_array(reflection))
        object.__setattr__(self, "innovation_variance", float(self.innovation_variance))
        object.__setattr__(self, "startup_covariance", frozen_array(covariance))
        object.__setattr__(self, "_startup_root", frozen_array(np.linalg.cholesky(covariance)))
        object.__setattr__(self, "startup_whitening_factor", frozen_array(factor))
        object.__setattr__(self, "startup_innovation_variances", frozen_array(np.exp(log_variances)))
        object.__setattr__(self, "_startup_log_variances", frozen_array(log_variances))
        object.__setattr__(self, "startup_factor_condition", factor_condition)
        object.__setattr__(self, "pole_radius", radius)
        object.__setattr__(self, "startup_condition", condition)
        object.__setattr__(self, "provenance", freeze_metadata(provenance))
        if not np.isfinite(self._marginal_variance()) or self._marginal_variance() <= 0:
            raise ValueError("stationary marginal variance scalar is not finite and positive")
        object.__setattr__(self, "_integrity_sha256", metadata_digest(self._state()))

    @property
    def order(self):
        return len(self.coefficients)

    def _marginal_variance(self):
        if not self.order:
            return 1.
        if self.startup_mode == "stationary":
            return float(self.startup_covariance[0, 0])
        # Prediction-error recursion: innovation variance / marginal variance
        # equals product(1-k_j^2). This needs no ill-conditioned covariance.
        return float(np.exp(-np.log1p(-self.reflection_coefficients**2).sum()))

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
            raise ValueError("insufficient lag rows to identify requested AR order")
        polynomial = np.empty(0)
        actual_reflections = []
        forward, backward = [x[1:].copy() for x in arrays], [x[:-1].copy() for x in arrays]
        for _ in range(order):
            numerator = 2*sum(float(np.sum(f*b)) for f, b in zip(forward, backward, strict=True))
            denominator = sum(float(np.sum(f*f)+np.sum(b*b)) for f, b in zip(forward, backward, strict=True))
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError("degenerate Burg noise energy")
            reflection = -numerator/denominator
            if not np.isfinite(reflection) or abs(reflection) >= 1-1e-12:
                raise ValueError("degenerate Burg reflection cannot establish strict stability")
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
            raise ValueError("degenerate pooled innovation variance")
        return cls(coefficients, variance, {
            "fit_scope": fit_scope, "method": "segmented_multichannel_burg_no_pole_clipping",
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
        values = np.asarray(reflections, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(np.abs(values) >= 1-1e-12):
            raise ValueError("analytic PARCOR requires a finite strictly stable vector")
        return cls(_parcor_coefficients(values), innovation_variance,
                   {"fit_scope": "analytic_reference", "reference_id": reference_id, "records": []},
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
            return _lattice_innovations(x, self.reflection_coefficients, self._startup_log_variances)
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

    def _state(self):
        stationary = self.startup_mode == "stationary"
        lattice = self.startup_mode == "stationary_innovations"
        schema = _SCHEMA if stationary else "n2p3_temporal_noise/3" if lattice else "n2p3_temporal_noise/2"
        result = {"schema": schema, "coefficients": self.coefficients.tolist(),
                "innovation_variance": self.innovation_variance, "provenance": self.provenance,
                "scale_convention": _SCALE_CONVENTION,
                "startup_assumption": "stationary_AR_latent_initial_state" if stationary or lattice else "conditional_on_observed_first_p_samples",
                "startup_covariance_sha256": array_digest(self.startup_covariance),
                "startup_root_sha256": array_digest(self._startup_root),
                "reflection_coefficients": self.reflection_coefficients.tolist(),
                "pole_radius": self.pole_radius, "startup_condition": self.startup_condition}
        if lattice:
            result.update(startup_mode=self.startup_mode, marginal_variance_per_unit_innovation=self._marginal_variance(),
                startup_whitening_factor_sha256=array_digest(self.startup_whitening_factor),
                startup_innovation_variances=self.startup_innovation_variances.tolist(),
                startup_log_variances_sha256=array_digest(self._startup_log_variances),
                startup_factor_condition=self.startup_factor_condition,
                floating_precision_condition_indicator=float(np.finfo(float).eps*self.startup_factor_condition),
                stability_certificate="actual_PARCOR_strictly_inside_unit_interval",
                reflection_origin="analytic_reference" if self.provenance["fit_scope"] == "analytic_reference" else "actual_Burg_recursion",
                expanded_polynomial_radius_role="diagnostic_only_not_lattice_stability_test")
        elif not stationary:
            result.update(startup_mode=self.startup_mode, conditioning_prefix_samples=self.order,
                          marginal_variance_per_unit_innovation=self._marginal_variance())
        return result

    def assert_integrity(self):
        if metadata_digest(self._state()) != self._integrity_sha256:
            raise ValueError("frozen temporal noise integrity changed")

    def record(self):
        self.assert_integrity()
        result = self._state()
        result["artifact_sha256"] = metadata_digest(result)
        return result

    @classmethod
    def from_record(cls, record):
        payload = json.loads(json.dumps(record, allow_nan=False))
        expected = payload.pop("artifact_sha256", None)
        if expected != metadata_digest(payload):
            raise ValueError("temporal noise artifact digest mismatch")
        mode = "stationary" if payload.get("schema") == _SCHEMA else payload.get("startup_mode")
        expected_assumption = {"stationary": "stationary_AR_latent_initial_state",
                               "conditional_prefix": "conditional_on_observed_first_p_samples",
                               "stationary_innovations": "stationary_AR_latent_initial_state"}.get(mode)
        if (payload.get("schema") not in {_SCHEMA, "n2p3_temporal_noise/2", "n2p3_temporal_noise/3"}
                or payload.get("scale_convention") != _SCALE_CONVENTION or expected_assumption is None
                or payload.get("startup_assumption") != expected_assumption
                or (payload.get("schema") == "n2p3_temporal_noise/2" and mode != "conditional_prefix")
                or (payload.get("schema") == "n2p3_temporal_noise/3" and mode != "stationary_innovations")):
            raise ValueError("unsupported temporal noise contract")
        model = cls(payload["coefficients"], payload["innovation_variance"], payload["provenance"], startup_mode=mode,
                    reflection_state=payload["reflection_coefficients"] if mode == "stationary_innovations" else None)
        state = model._state()
        if set(state) != set(payload) or any(state[key] != payload[key] for key in
                ("reflection_coefficients", "startup_covariance_sha256", "startup_root_sha256", "pole_radius", "startup_condition")):
            raise ValueError("temporal noise numeric replay differs from its saved state")
        if mode == "conditional_prefix" and any(state[key] != payload[key] for key in
                ("conditioning_prefix_samples", "marginal_variance_per_unit_innovation")):
            raise ValueError("conditional temporal noise numeric replay differs")
        if mode == "stationary_innovations" and any(state[key] != payload[key] for key in (
                "marginal_variance_per_unit_innovation", "startup_whitening_factor_sha256", "startup_innovation_variances",
                "startup_log_variances_sha256", "startup_factor_condition", "floating_precision_condition_indicator",
                "stability_certificate", "reflection_origin", "expanded_polynomial_radius_role")):
            raise ValueError("stationary innovations numeric replay differs")
        return model

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
