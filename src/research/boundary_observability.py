"""Identifiable boundary functionals and complete native affine-map diagnostics.

An observation O z identifies L z iff ker(O) is contained in ker(L). A noise
prior can predict an unobserved direction; it cannot make that direction an
observation. Native crop maps have an exact, explicit missing-sample kernel.
General state/nuisance maps instead report their declared numerical resolution.
These diagnostics neither fit a correction nor authorize its training support.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from data.finite_filter_operator import FiniteFilterOperator
from research.contracts import semantic_sha256
from transfer.noise_model import (
    array_digest,
    finite_real_array,
    freeze_metadata,
    frozen_array,
    require_sha256,
)


class BoundaryIdentifiabilityError(ValueError):
    """Requested recovery is unsupported; ``report`` explains the null direction."""

    def __init__(self, report):
        self.report = freeze_metadata(report)
        super().__init__(f"boundary functional cannot be recovered: {report['status']}")


class BoundaryObservationRangeError(ValueError):
    """The supplied observation is outside the represented noiseless map range."""


def _matrix(value, name):
    result = finite_real_array(value, name=name)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{name} must be a nonempty matrix")
    return result


def _scope(value, name):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(item) is not int for item in value)
            or not 0 <= value[0] < value[1]):
        raise ValueError(f"{name} requires absolute integer half-open sample bounds")
    return tuple(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} requires an explicit nonempty identity")
    return value


def _spectrum(matrix, relative_tolerance):
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = float(relative_tolerance * singular[0])
    keep = singular > threshold
    return {"singular_values": singular.tolist(), "rank": int(keep.sum()),
            "rank_relative_tolerance": relative_tolerance, "rank_threshold": threshold,
            "condition_in_identified_range": (
                float(singular[0] / singular[keep][-1]) if keep.any() else None),
            "discarded_nonzero_singular_values": singular[(singular > 0) & ~keep].tolist()}


def _stable_norm(values):
    """Physical norm without square underflow; None marks unrepresentable size."""
    if not np.isfinite(values).all():
        return None
    scale = float(np.max(np.abs(values)))
    if not scale:
        return 0.
    normalized_norm = float(np.linalg.norm(values / scale))
    if scale > 1 and normalized_norm > np.finfo(float).max / scale:
        return None
    return scale * normalized_norm


@dataclass(frozen=True)
class LinearFunctionalObservability:
    """Small explicit O/L audit, including nuisance columns in the same latent z.

For y = D u + H a and requested J a, supply O=[D,H], L=[0,J].
This tests identifiable *functionals*, rather than demanding all latent state.
Only selected singular directions are inverted. A tolerance-based null witness
    is labelled unresolved unless it uses structurally zero observation columns.
    A floating dot product rounded to zero is insufficient proof of an exact
    null direction. Recovery is a noiseless diagnostic, not a Bayesian estimator.
"""

    observation_matrix: np.ndarray = field(repr=False, compare=False)
    functional_matrix: np.ndarray = field(repr=False, compare=False)
    latent_identity: str
    observation_identity: str
    functional_identity: str
    rank_relative_tolerance: float = 1e-12
    relation_relative_tolerance: float = 1e-10
    byte_budget: int = 64 * 1024 * 1024
    _gain: np.ndarray = field(init=False, repr=False, compare=False)
    _range_basis: np.ndarray = field(init=False, repr=False, compare=False)
    _row_scales: np.ndarray = field(init=False, repr=False, compare=False)
    _functional_scales: np.ndarray = field(init=False, repr=False, compare=False)
    _report: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        observation = _matrix(self.observation_matrix, "actual observation map")
        L = _matrix(self.functional_matrix, "requested linear functional")
        if observation.shape[1] != L.shape[1]:
            raise ValueError("observation and functional must share the same latent columns")
        for name in ("latent_identity", "observation_identity", "functional_identity"):
            _text(getattr(self, name), name)
        if (not np.isfinite([self.rank_relative_tolerance, self.relation_relative_tolerance]).all()
                or not 0 < self.rank_relative_tolerance < 1
                or not 0 < self.relation_relative_tolerance < 1):
            raise ValueError("explicit rank/relation tolerances must lie in (0, 1)")
        # Include input copies, reduced SVD factors, residual/gain and work arrays.
        required = 8 * (5 * observation.size + 4 * L.size + 3 * observation.shape[0] * L.shape[0])
        if type(self.byte_budget) is not int or self.byte_budget < required:
            raise ValueError("observability reference exceeds its explicit dense byte budget")
        # Maximum-absolute row scaling is overflow/underflow-safe and avoids
        # treating a voltage-unit rescaling as a changed numerical rank.
        scales = np.max(np.abs(observation), axis=1)
        scales[scales == 0] = 1.
        normalized = observation / scales[:, None]
        U, singular, Vt = np.linalg.svd(normalized, full_matrices=False)
        threshold = self.rank_relative_tolerance * singular[0]
        keep = singular > threshold
        rows, columns = Vt[keep], U[:, keep]
        functional_scales = np.max(np.abs(L), axis=1)
        functional_scales[functional_scales == 0] = 1.
        normalized_functional = L / functional_scales[:, None]
        residual = normalized_functional - (normalized_functional @ rows.T) @ rows
        residual_by_row = np.linalg.norm(residual, axis=1)
        target_by_row = np.linalg.norm(normalized_functional, axis=1)
        unresolved = residual_by_row > self.relation_relative_tolerance * target_by_row
        relative_by_row = np.divide(residual_by_row, target_by_row,
                                   out=np.zeros_like(target_by_row), where=target_by_row > 0)
        relation_ok = not unresolved.any()
        # Keep the gain between normalized coordinate systems. A physical gain
        # of 1e400 can be unrepresentable while its actual recovered value is
        # finite, e.g. O=1e-200, L=1e200 and observed=2e-200.
        gain = ((normalized_functional @ rows.T) / singular[keep]) @ columns.T
        witness = None
        if not relation_ok:
            row_index = int(np.argmax(relative_by_row))
            direction = residual[row_index].copy()
            direction /= np.linalg.norm(direction)
            with np.errstate(over="ignore", invalid="ignore"):
                observed_change, target_change = observation @ direction, L @ direction
            normalized_target_change = normalized_functional @ direction
            computed_zero = bool(np.all(observed_change == 0))
            structural_zero = bool(np.all(observation[:, direction != 0] == 0))
            witness = {"latent_direction": direction.tolist(),
                       "functional_row_index": row_index,
                       "observation_change": [float(value) if np.isfinite(value) else None
                                              for value in observed_change],
                       "functional_change": [float(value) if np.isfinite(value) else None
                                             for value in target_change],
                       "functional_change_in_normalized_rows": normalized_target_change.tolist(),
                       "observation_change_norm": _stable_norm(observed_change),
                       "functional_change_norm": _stable_norm(target_change),
                       "computed_observation_change_zero": computed_zero,
                       "exactly_zero_stored_map_observation": structural_zero,
                       "structural_proof": "zero_observation_columns" if structural_zero else None}
            status = ("structurally_unidentifiable" if structural_zero and np.any(normalized_target_change != 0)
                      else "unresolved_at_declared_resolution")
        else:
            status = "identified_at_declared_resolution"
        report = {"schema": "n2p3_linear_functional_observability/1", "status": status,
                  "criterion": "ker_observation_subset_ker_functional",
                  "latent_identity": self.latent_identity,
                  "observation_identity": self.observation_identity,
                  "functional_identity": self.functional_identity,
                  "latent_dimension": observation.shape[1], "observation_dimension": observation.shape[0],
                  "functional_dimension": L.shape[0], "observation_rank": int(keep.sum()),
                  "observation_singular_values_after_row_normalization": singular.tolist(),
                  "observation_row_scales": scales.tolist(),
                  "observation_row_normalization": "maximum_absolute_entry",
                  "rank_relative_tolerance": self.rank_relative_tolerance,
                  "rank_threshold": float(threshold),
                  "discarded_nonzero_singular_values": singular[(singular > 0) & ~keep].tolist(),
                  "condition_in_identified_range": (
                      float(singular[0] / singular[keep][-1]) if keep.any() else None),
                  "functional_range_residual_norm": _stable_norm(residual * functional_scales[:, None]),
                  "functional_range_relative_residual": (
                      float(relative_by_row.max()) if np.any(target_by_row > 0) else None),
                  "functional_range_relative_residual_by_row": [float(value) if norm else None
                      for value, norm in zip(relative_by_row, target_by_row, strict=True)],
                  "functional_row_scales": functional_scales.tolist(),
                  "functional_row_normalization": "maximum_absolute_entry_each_nonzero_task_independently",
                  "physical_norm_unrepresentable_uses_null": True,
                  "relation_relative_tolerance": self.relation_relative_tolerance,
                  "witness": witness, "maps_sha256": array_digest(observation, L),
                  "inference_scope": "noiseless_linear_maps_at_declared_numerical_resolution",
                  "prior_makes_unobserved_state_an_observation": False,
                  "physical_state_identification_claim": False}
        for name, value in (("observation_matrix", observation), ("functional_matrix", L),
                            ("_gain", gain), ("_range_basis", columns), ("_row_scales", scales),
                            ("_functional_scales", functional_scales)):
            object.__setattr__(self, name, frozen_array(value))
        object.__setattr__(self, "_report", freeze_metadata(report))

    def report(self):
        return self._report

    def recover(self, observed):
        """Recover only an identified functional from feasible noiseless observations."""
        if self._report["status"] != "identified_at_declared_resolution":
            raise BoundaryIdentifiabilityError(self._report)
        values = finite_real_array(observed, name="actual noiseless observation")
        if values.ndim not in (1, 2) or values.shape[0] != self.observation_matrix.shape[0]:
            raise ValueError("observations must match the actual observation-map rows")
        normalized = values / (self._row_scales if values.ndim == 1 else self._row_scales[:, None])
        if not np.isfinite(normalized).all():
            raise ValueError("observations exceed finite normalized coordinates for noiseless recovery")
        range_scales = np.max(np.abs(normalized), axis=0)
        range_scales = np.where(range_scales == 0, 1., range_scales)
        range_values = normalized / range_scales
        residual = range_values - self._range_basis @ (self._range_basis.T @ range_values)
        residual_norm = np.linalg.norm(residual, axis=0)
        observed_norm = np.linalg.norm(range_values, axis=0)
        if np.any(residual_norm > self.relation_relative_tolerance * observed_norm):
            raise BoundaryObservationRangeError("observation lies outside the declared noiseless range")
        result = self._gain @ normalized
        result *= self._functional_scales if values.ndim == 1 else self._functional_scales[:, None]
        if not np.isfinite(result).all():
            raise ValueError("identified functional recovery exceeds finite physical output coordinates")
        return result


@dataclass(frozen=True)
class NativeBoundaryGrid:
    """Actual source/native coordinate identity; does not issue read authorization."""

    context_scope: tuple[int, int]
    sample_rate_hz: float
    channel_names: tuple[str, ...]
    reference: str
    source_sha256: str
    source_identity: str
    signal_unit: str = "V"

    def __post_init__(self):
        object.__setattr__(self, "context_scope", _scope(self.context_scope, "native context"))
        if not isinstance(self.channel_names, (tuple, list)):
            raise ValueError("physical channels require their explicit ordered sequence")
        names = tuple(self.channel_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("actual native physical channel order must be nonempty and unique")
        for name in names:
            _text(name, "physical channel")
        for name in ("reference", "source_identity"):
            _text(getattr(self, name), name)
        if (isinstance(self.sample_rate_hz, (bool, np.bool_))
                or not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0
                or self.signal_unit != "V"):
            raise ValueError("native boundary diagnostics require positive native rate and volt units")
        require_sha256(self.source_sha256, "actual source bytes")
        object.__setattr__(self, "channel_names", names)

    @property
    def sample_count(self):
        return self.context_scope[1] - self.context_scope[0]

    def record(self):
        return {"schema": "n2p3_native_boundary_grid/1", "context_scope": list(self.context_scope),
                "sample_rate_hz": self.sample_rate_hz, "channel_names": list(self.channel_names),
                "reference": self.reference, "signal_unit": self.signal_unit,
                "source_sha256": self.source_sha256, "source_identity": self.source_identity,
                "flattening": "sample_then_physical_channel", "stage": "native_finite_filter_crop",
                "read_authorization_issued_here": False}

    def values(self, value, scope, name):
        values = _matrix(value, name)
        if values.shape != (scope[1] - scope[0], len(self.channel_names)):
            raise ValueError(f"{name} must match the frozen native sample/channel grid")
        return values


@dataclass(frozen=True)
class NativeBoundaryObservability:
    """Real finite-filter target and exact crop observation on a common native grid.

Temporal maps are shared by the physical channels, so no large Kronecker/SVD
is needed. Observation scopes explicitly enumerate legal observed context.
The long-context teacher is a diagnostic target, never an extra predictor input.
"""

    grid: NativeBoundaryGrid
    short_operator: FiniteFilterOperator
    context_operator: FiniteFilterOperator
    short_scope: tuple[int, int]
    output_scope: tuple[int, int]
    observed_scopes: tuple[tuple[int, int], ...]
    available_at_sample: int
    byte_budget: int = 64 * 1024 * 1024
    _target: np.ndarray = field(init=False, repr=False, compare=False)
    _baseline: np.ndarray = field(init=False, repr=False, compare=False)
    _indices: np.ndarray = field(init=False, repr=False, compare=False)
    _missing: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (not isinstance(self.grid, NativeBoundaryGrid)
                or not isinstance(self.short_operator, FiniteFilterOperator)
                or not isinstance(self.context_operator, FiniteFilterOperator)):
            raise TypeError("boundary observability requires the actual finite operators/native grid")
        short, output = _scope(self.short_scope, "short input"), _scope(self.output_scope, "output")
        begin, end = self.grid.context_scope
        if not begin <= short[0] <= output[0] < output[1] <= short[1] <= end:
            raise ValueError("short/output must be contained in the same native context")
        scopes = tuple(_scope(value, "legal observation") for value in self.observed_scopes)
        if (not scopes or any(not begin <= left < right <= end for left, right in scopes)
                or any(left[1] > right[0] for left, right in zip(scopes[:-1], scopes[1:], strict=True))
                or type(self.available_at_sample) is not int
                or self.available_at_sample < max(right for _, right in scopes)):
            raise ValueError("ordered disjoint observations require their actual availability cutoff")
        mask = np.zeros(self.grid.sample_count, dtype=bool)
        for left, right in scopes:
            mask[left - begin:right - begin] = True
        if not mask[short[0] - begin:short[1] - begin].all():
            raise ValueError("the actual short input must be included in legal observations")
        n, m, k = self.grid.sample_count, output[1] - output[0], short[1] - short[0]
        required = 8 * (2 * n * n + 3 * m * n + 2 * k * k)
        if type(self.byte_budget) is not int or self.byte_budget < required:
            raise ValueError("native observability exceeds its explicit dense byte budget")
        target = self.context_operator.dense_matrix(n, byte_budget=self.byte_budget)[
            output[0] - begin:output[1] - begin]
        baseline = self.short_operator.dense_matrix(k, byte_budget=self.byte_budget)[
            output[0] - short[0]:output[1] - short[0]]
        for name, value in (("_target", target), ("_baseline", baseline)):
            object.__setattr__(self, name, frozen_array(value))
        for name, value in (("_indices", np.flatnonzero(mask)), ("_missing", np.flatnonzero(~mask))):
            immutable = np.frombuffer(value.astype(np.int64).tobytes(), dtype=np.int64)
            object.__setattr__(self, name, immutable)
        object.__setattr__(self, "short_scope", short)
        object.__setattr__(self, "output_scope", output)
        object.__setattr__(self, "observed_scopes", scopes)

    def record(self):
        return {"schema": "n2p3_native_boundary_observability/1", "grid": self.grid.record(),
                "short_scope": list(self.short_scope), "output_scope": list(self.output_scope),
                "observed_scopes": [list(value) for value in self.observed_scopes],
                "available_at_sample": self.available_at_sample,
                "short_operator": self.short_operator.record(),
                "context_operator": self.context_operator.record(),
                "target_and_baseline_sha256": array_digest(self._target, self._baseline),
                "target_role": "common_output_grid_context_filter_teacher",
                "context_teacher_available_at_sample": self.grid.context_scope[1]}

    def report(self, *, temporal_functionals=None, rank_relative_tolerance=1e-12):
        """Audit all output samples or declared time-linear functionals per channel.

Missing-sample dependence is tested exactly for the stored finite map. Rank
tolerance describes the sensitivity spectrum only; it never zeroes exterior
dependence to grant recovery. Arbitrary channel-coupled state/nuisance maps use
LinearFunctionalObservability with explicit latent columns.
"""
        if not np.isfinite(rank_relative_tolerance) or not 0 < rank_relative_tolerance < 1:
            raise ValueError("rank relative tolerance must lie in (0, 1)")
        L = self._target
        if temporal_functionals is not None:
            rows = _matrix(temporal_functionals, "declared native time-linear functionals")
            if rows.shape[1] != L.shape[0]:
                raise ValueError("functional columns must match the actual output time grid")
            L = rows @ L
        missing = L[:, self._missing]
        # A crop's kernel is the coordinate complement, known without an SVD.
        identified = bool(np.all(missing == 0))
        spectrum = _spectrum(missing, rank_relative_tolerance) if missing.shape[1] else {
            "singular_values": [], "rank": 0, "rank_relative_tolerance": rank_relative_tolerance,
            "rank_threshold": 0., "condition_in_identified_range": None,
            "discarded_nonzero_singular_values": []}
        target_norm, missing_norm = float(np.linalg.norm(L)), float(np.linalg.norm(missing))
        return {**self.record(), "status": "identified" if identified else "structurally_unidentifiable",
                "criterion": "requested_map_zero_on_exact_unobserved_native_coordinates",
                "functional_role": "native_time_linear_per_physical_channel",
                "functional_map_sha256": array_digest(L),
                "observation_rank": len(self._indices) * len(self.grid.channel_names),
                "missing_native_samples": (self._missing + self.grid.context_scope[0]).tolist(),
                "missing_functional_norm": missing_norm,
                "missing_functional_relative_norm": missing_norm / target_norm if target_norm else None,
                "exterior_functional_spectrum_per_channel": spectrum,
                "physical_state_dimension_inferred": False,
                "padding_state_role": "deterministic_function_of_actual_short_samples",
                "prior_makes_unobserved_state_an_observation": False,
                "upstream_acquisition_filter_inverted": False}

    def observe(self, full_context):
        context = self.grid.values(full_context, self.grid.context_scope, "diagnostic observed context")
        return context[self._indices].copy()

    def recover(self, observed_samples, *, temporal_functionals=None):
        report = self.report(temporal_functionals=temporal_functionals)
        if report["status"] != "identified":
            raise BoundaryIdentifiabilityError(report)
        values = _matrix(observed_samples, "legal native observations")
        if values.shape != (len(self._indices), len(self.grid.channel_names)):
            raise ValueError("observation samples must match the ordered actual crop map")
        L = self._target if temporal_functionals is None else np.asarray(temporal_functionals) @ self._target
        return L[:, self._indices] @ values

    def counterexample(self, full_context, *, amplitude_volts=1e-6, channel_name=None):
        """Actual same-observation, different-exterior pair; no reconstruction is fitted."""
        context = self.grid.values(full_context, self.grid.context_scope, "diagnostic observed context")
        report = self.report()
        if report["status"] == "identified":
            raise ValueError("there is no unobserved native direction affecting this target map")
        if not np.isfinite(amplitude_volts) or amplitude_volts == 0:
            raise ValueError("counterexample requires a finite nonzero voltage increment")
        name = self.grid.channel_names[0] if channel_name is None else channel_name
        if name not in self.grid.channel_names:
            raise ValueError("counterexample channel must be an actually measured physical channel")
        norms = np.linalg.norm(self._target[:, self._missing], axis=0)
        sample = int(self._missing[int(np.argmax(norms))])
        delta = np.zeros_like(context)
        delta[sample, self.grid.channel_names.index(name)] = amplitude_volts
        alternate = context + delta
        out = slice(self.output_scope[0] - self.grid.context_scope[0],
                    self.output_scope[1] - self.grid.context_scope[0])
        first, second = self.context_operator.apply(context)[out], self.context_operator.apply(alternate)[out]
        linear_delta = self.context_operator.apply(delta)[out]
        actual_delta = second - first
        unchanged = bool(np.array_equal(self.observe(context), self.observe(alternate)))
        row = {"schema": "n2p3_boundary_exterior_counterexample/1",
               "status": "verified" if unchanged and np.any(actual_delta != 0) else "numerically_unresolved",
               "same_legal_observation_bytes": unchanged,
               "outside_sample": sample + self.grid.context_scope[0], "physical_channel": name,
               "amplitude_volts": amplitude_volts,
               "actual_target_change_norm": float(np.linalg.norm(actual_delta)),
               "linear_increment_target_norm": float(np.linalg.norm(linear_delta)),
               "finite_difference_roundoff_norm": float(np.linalg.norm(actual_delta - linear_delta)),
               "context_sha256": array_digest(context), "alternate_context_sha256": array_digest(alternate),
               "observation_sha256": array_digest(self.observe(context)),
               "exterior_increment_sha256": array_digest(delta),
               "problem_sha256": semantic_sha256(self.record()),
               "meaning": "a_short_only_function_returns_the_same_value_for_both_contexts",
               "physiological_exterior_distribution_proved": False}
        return row, frozen_array(delta)

    def padding_state_report(self, short_samples):
        short = self.grid.values(short_samples, self.short_scope, "actual short samples")
        Qx, B, state = self.short_operator.decompose(short)
        return {"schema": "n2p3_deterministic_padding_state/1",
                "operator_sha256": self.short_operator.digest, "short_sha256": array_digest(short),
                "state_sha256": array_digest(state), "state_dimension_per_channel": len(state),
                "decomposition_residual_norm": float(np.linalg.norm(self.short_operator.apply(short) - Qx - B @ state)),
                "state_role": "deterministic_padding_not_unknown_external_state",
                "conditional_on_short_extra_independent_covariance": False}


@dataclass(frozen=True)
class NativeAffineBoundaryMap:
    """The full frozen T(x)=R F_short x+Kx+b on actual native coordinates.

The operator is really executed by apply; complete_matrix is its dense affine
derivative for white-gain diagnostics. A producer must supply/freeze K,b from
authorized source/C. A textual parameter identity does not certify that fit.
No oracle, label, outside sample, resampling or reference change enters apply.
"""

    problem: NativeBoundaryObservability
    correction_matrix: np.ndarray = field(repr=False, compare=False)
    parameter_identity: str
    correction_offset: np.ndarray | None = field(default=None, repr=False, compare=False)
    byte_budget: int = 64 * 1024 * 1024
    baseline_matrix: np.ndarray = field(init=False, repr=False, compare=False)
    complete_matrix: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.problem, NativeBoundaryObservability):
            raise TypeError("complete T requires the actual native boundary problem")
        _text(self.parameter_identity, "frozen correction producer")
        channels = len(self.problem.grid.channel_names)
        m, n = self.problem._baseline.shape
        required = 8 * (4 * m * n * channels**2 + m * channels)
        if type(self.byte_budget) is not int or self.byte_budget < required:
            raise ValueError("complete native T exceeds its explicit dense byte budget")
        K = _matrix(self.correction_matrix, "frozen short-only correction")
        if K.shape != (m * channels, n * channels):
            raise ValueError("correction must read only the actual short sample/channel grid")
        b = (np.zeros(m * channels) if self.correction_offset is None else
             finite_real_array(self.correction_offset, name="frozen correction offset"))
        if b.shape != (m * channels,):
            raise ValueError("correction offset must match the actual native output grid")
        baseline = np.kron(self.problem._baseline, np.eye(channels))
        for name, value in (("correction_matrix", K), ("correction_offset", b),
                            ("baseline_matrix", baseline), ("complete_matrix", baseline + K)):
            object.__setattr__(self, name, frozen_array(value))

    def record(self):
        return {"schema": "n2p3_native_affine_boundary_map/1", "problem": self.problem.record(),
                "parameter_identity": self.parameter_identity,
                "parameter_training_authorized_here": False,
                "deployment_input_scope": list(self.problem.short_scope),
                "deployment_available_at_sample": self.problem.short_scope[1],
                "output_stage": "native_finite_filter_crop",
                "affine_arrays_sha256": array_digest(self.correction_matrix, self.correction_offset,
                                                      self.baseline_matrix, self.complete_matrix),
                "observed_context_used_at_deployment": False}

    @property
    def digest(self):
        return semantic_sha256(self.record())

    def apply(self, short_samples):
        short = self.problem.grid.values(short_samples, self.problem.short_scope, "actual short-only input")
        left, right = (sample - self.problem.short_scope[0] for sample in self.problem.output_scope)
        native = self.problem.short_operator.apply(short)[left:right]
        correction = self.correction_matrix @ short.ravel() + self.correction_offset
        return native + correction.reshape(native.shape)

    def measure(self, diagnostic_context, injections: Mapping, *, injection_design_identity):
        """Execute T(x+h)-T(x); compare both native-short and observed-context teachers."""
        _text(injection_design_identity, "independent ERP injection design")
        if not isinstance(injections, Mapping) or not injections:
            raise ValueError("complete T measurements require named independent injections")
        p = self.problem
        context = p.grid.values(diagnostic_context, p.grid.context_scope, "actually observed diagnostic context")
        a, b = (sample - p.grid.context_scope[0] for sample in p.short_scope)
        c, d = (sample - p.grid.context_scope[0] for sample in p.output_scope)
        short, entries = context[a:b], []
        initial = self.apply(short)
        rate = p.grid.sample_rate_hz
        for name, value in injections.items():
            _text(name, "ERP injection")
            h = p.grid.values(value, p.grid.context_scope, "independent native-context injection")
            increment = self.apply(short + h[a:b]) - initial
            derivative = (self.complete_matrix @ h[a:b].ravel()).reshape(increment.shape)
            short_reference = p.short_operator.apply(h[a:b])[c - a:d - a]
            long_reference = p.context_operator.apply(h)[c:d]
            entries.append({"injection_identity": name, "injection_sha256": array_digest(h),
                            "short_input_increment_norm": float(np.linalg.norm(h[a:b])),
                            "actual_complete_increment_sha256": array_digest(increment),
                            "actual_vs_affine_derivative_roundoff_norm": float(np.linalg.norm(increment - derivative)),
                            "short_reference": _increment_metrics(increment, short_reference, rate),
                            "context_reference": _increment_metrics(increment, long_reference, rate)})
        spectrum = _spectrum(self.complete_matrix, max(self.complete_matrix.shape) * np.finfo(float).eps)
        baseline_energy = float(np.sum(self.baseline_matrix**2))
        complete_energy = float(np.sum(self.complete_matrix**2))
        return {"schema": "n2p3_native_complete_T_measurement/1", "mapping": self.record(),
                "mapping_sha256": self.digest, "diagnostic_context_sha256": array_digest(context),
                "diagnostic_context_read_scope": list(p.grid.context_scope),
                "diagnostic_context_available_at_sample": p.grid.context_scope[1],
                "injection_design_identity": injection_design_identity,
                "injection_independence_authorized_here": False,
                "requested_injections": len(entries),
                "defined_short_relative_errors": sum(row["short_reference"]["relative_error"] is not None for row in entries),
                "defined_context_relative_errors": sum(row["context_reference"]["relative_error"] is not None for row in entries),
                "erp_increments": entries,
                "white_noise_gain": {"input_white_variance": 1.,
                    "complete_frobenius_squared": complete_energy,
                    "baseline_frobenius_squared": baseline_energy,
                    "mean_relative_to_short": complete_energy / baseline_energy if baseline_energy else None,
                    "relative_gain_undefined_reason": "zero_short_baseline" if not baseline_energy else None,
                    "worst_absolute": spectrum["singular_values"][0]**2,
                    **spectrum, "offset_included_as_noise": False},
                "measurement_scope": "actual_native_affine_T_only_no_downstream_resample_reference",
                "unknown_state_recovered": False, "policy_selected": False,
                "production_or_public_performance_claim": False}


def _increment_metrics(increment, reference, rate):
    error, norm = increment - reference, float(np.linalg.norm(reference))
    peak, ref_peak = np.argmax(np.abs(increment), axis=0), np.argmax(np.abs(reference), axis=0)
    valid = np.any(increment != 0, axis=0) & np.any(reference != 0, axis=0)
    return {"relative_error": float(np.linalg.norm(error) / norm) if norm else None,
            "undefined_reason": "zero_reference_increment" if not norm else None,
            "reference_norm": norm, "maximum_absolute_error": float(np.max(np.abs(error))),
            "signed_peak_volts_per_channel": increment[peak, np.arange(increment.shape[1])].tolist(),
            "peak_latency_error_seconds_per_channel": [float((x - y) / rate) if ok else None
                for x, y, ok in zip(peak, ref_peak, valid, strict=True)],
            "area_error_volt_seconds_per_channel": (error.sum(axis=0) / rate).tolist()}
