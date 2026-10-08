"""Reference-aware ERP signal priors, separate from fitted residual noise.

The coefficients B have shape (response coordinates, measured native EEG).
All vectorization is column-major: vec(B) has covariance S_signal x H_time.
An actual observation matrix A maps native acquired voltages to output
coordinates. Reference nulls require a structural declaration; unresolved
numerical dependence is rejected rather than signed as physical information loss.
The posterior is a general Gaussian; a matrix-normal row covariance is not
assumed. The caller owns temporal whitening, response/event design and legal C.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import InitVar, dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from research.execution import UnidentifiableFitError, atomic_write_json
from transfer.noise_model import (
    array_digest,
    finite_real_array,
    freeze_metadata,
    frozen_array,
    metadata_digest,
    require_identifiers,
    require_sha256,
)

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_PRIOR_SCHEMA = "n2p3_spatial_signal_prior/2"
_POSTERIOR_SCHEMA = "n2p3_spatial_signal_posterior/3"
_OBSERVATION_SCHEMA = "n2p3_spatial_observation/2"
_IDENTITY_RESOLUTION = float(np.sqrt(np.finfo(float).eps))


def _matrix(value, name):
    result = finite_real_array(value, name=name)
    if result.ndim != 2 or min(result.shape) == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    return result


def _positive(value, name):
    raw = np.asarray(value)
    if raw.ndim or raw.dtype.kind not in "fiu" or not np.isfinite(raw) or raw <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return float(raw)


def _spd(value, name):
    matrix = _matrix(value, name)
    if matrix.shape[0] != matrix.shape[1] or np.any(np.diag(matrix) <= 0):
        raise ValueError(f"{name} must be symmetric positive definite")
    scales = np.sqrt(np.diag(matrix))
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = (matrix / scales[:, None]) / scales[None, :]
    tolerance = 64 * np.finfo(float).eps * len(matrix) * np.maximum(
        np.abs(normalized), np.abs(normalized.T))
    if not np.isfinite(normalized).all() or np.any(np.abs(normalized - normalized.T) > tolerance):
        raise ValueError(f"{name} must be symmetric positive definite")
    normalized = normalized / 2 + normalized.T / 2
    matrix = matrix / 2 + matrix.T / 2
    try:
        root = scales[:, None] * np.linalg.cholesky(normalized)
    except np.linalg.LinAlgError as error:
        raise SpatialPriorIdentificationError("covariance_not_certified_positive_definite", name=name) from error
    return matrix, root


def _column_scales(matrix):
    scales = np.max(np.abs(matrix), axis=0)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise SpatialPriorIdentificationError("precision_coordinate_not_resolved")
    return scales


def _gamma(operations):
    value = np.finfo(float).eps * max(1, operations)
    return value / (1 - value)


def _unit_information_relation(actual, saved, *, lower=False):
    """Certify every precision direction, with column units removed.

    Right whitening turns a fourfold small precision change into a unit-scale
    discrepancy. A componentwise triangular-solve error enclosure determines
    whether that assertion is numerically resolved; uncertainty never licenses
    a wider state-admission gate.
    """
    scales = _column_scales(actual)
    normalized, root = actual / scales, saved / scales
    try:
        whitened = solve_triangular(root.T, normalized.T, lower=not lower).T
        inverse = solve_triangular(root, np.eye(len(root)), lower=lower)
    except np.linalg.LinAlgError as error:
        raise SpatialPriorIdentificationError("precision_relation_resolution_unresolved") from error
    if not np.isfinite(whitened).all() or not np.isfinite(inverse).all():
        raise SpatialPriorIdentificationError("precision_relation_resolution_unresolved")
    residual = np.abs(normalized - whitened @ root)
    rounding = _gamma(len(root)) * (np.abs(normalized) + np.abs(whitened) @ np.abs(root))
    forward = (residual + rounding) @ np.abs(inverse)
    bound = (forward.T @ np.abs(whitened) + np.abs(whitened).T @ forward
             + forward.T @ forward + _gamma(len(actual)) * (np.abs(whitened).T @ np.abs(whitened)))
    if not np.isfinite(bound).all() or np.max(bound) > _IDENTITY_RESOLUTION:
        raise SpatialPriorIdentificationError("precision_relation_resolution_unresolved",
            relative_identity_error_bound=float(np.max(bound)), identity_resolution=_IDENTITY_RESOLUTION)
    tolerance = 256 * _gamma(max(actual.shape))
    if np.any(np.abs(whitened.T @ whitened - np.eye(len(root))) > tolerance + bound):
        raise ValueError("saved precision differs from actual resolved likelihood directions")
    return scales


class SpatialPriorIdentificationError(UnidentifiableFitError):
    """A valid requested spatial mechanism has no identified implementation."""

    def __init__(self, reason, **evidence):
        self._record = {"schema": "n2p3_spatial_prior_failure/1", "reason": reason, **evidence}
        super().__init__(reason)

    def record(self):
        return json.loads(json.dumps(self._record))


@dataclass(frozen=True)
class SpatialLayout:
    """Actual measured native EEG, with an unchanged acquisition reference."""

    channels: tuple[str, ...]
    acquisition_reference: str
    authority_sha256: str
    channel_types: tuple[str, ...]
    signal_unit: str = "V"
    positions_m: np.ndarray | None = field(default=None, repr=False, compare=False)
    coordinate_frame: str | None = None

    def __post_init__(self):
        require_identifiers(self.channels, "native measured EEG channels")
        require_sha256(self.authority_sha256, "physical layout authority")
        if (not isinstance(self.acquisition_reference, str) or not self.acquisition_reference.strip()
                or self.signal_unit not in {"V", "uV"}
                or tuple(self.channel_types) != ("EEG",) * len(self.channels)
                or any(name.upper() in {"EOG", "HEOG", "VEOG", "GND", "REF"} for name in self.channels)):
            raise ValueError("spatial signal layout requires measured EEG, reference and voltage units")
        if len({name.upper() for name in self.channels}) != len(self.channels):
            raise ValueError("native channel identities cannot differ only by case")
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "channel_types", tuple(self.channel_types))
        if self.positions_m is not None:
            positions = _matrix(self.positions_m, "actual electrode positions")
            if positions.shape != (len(self.channels), 3) or not self.coordinate_frame:
                raise ValueError("actual positions need one 3-D metre coordinate per EEG and a frame")
            object.__setattr__(self, "positions_m", frozen_array(positions))
        elif self.coordinate_frame is not None:
            raise ValueError("coordinate frame cannot certify absent electrode positions")

    def record(self):
        return {"channels": list(self.channels), "acquisition_reference": self.acquisition_reference,
                "channel_types": list(self.channel_types),
                "authority_sha256": self.authority_sha256, "signal_unit": self.signal_unit,
                "positions_m": None if self.positions_m is None else self.positions_m.tolist(),
                "coordinate_frame": self.coordinate_frame}

    @classmethod
    def from_record(cls, record):
        return cls(**record)


def _null_complement(null):
    """Orthonormal complement of an explicitly declared unit null vector."""
    vector = np.asarray(null, dtype=float).copy()
    vector[-1] += np.copysign(1., vector[-1])
    vector /= np.linalg.norm(vector)
    return (np.eye(len(vector)) - 2 * np.outer(vector, vector))[:, :-1]


@dataclass(frozen=True)
class SpatialObservation:
    """Actual coordinates with certified row units and declared reference nulls."""

    layout: SpatialLayout
    matrix: np.ndarray = field(repr=False, compare=False)
    output_names: tuple[str, ...]
    output_reference: str
    declaration: dict | None = None
    support_basis: np.ndarray = field(init=False, repr=False, compare=False)
    normalized_support_basis: np.ndarray = field(init=False, repr=False, compare=False)
    normalized_matrix: np.ndarray = field(init=False, repr=False, compare=False)
    equilibration_scales: np.ndarray = field(init=False, repr=False, compare=False)
    coordinate_log_volume: float = field(init=False, repr=False, compare=False)
    rank_ledger: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.layout, SpatialLayout):
            raise TypeError("spatial observation needs an owned physical layout")
        matrix = _matrix(self.matrix, "actual observation matrix")
        require_identifiers(self.output_names, "observed output coordinates")
        if (matrix.shape != (len(self.output_names), len(self.layout.channels))
                or not isinstance(self.output_reference, str) or not self.output_reference.strip()):
            raise ValueError("actual observation dimensions/reference differ from the measured layout")
        if np.any(~np.any(matrix != 0, axis=1)):
            raise SpatialPriorIdentificationError("unmeasured_or_zero_filled_output")
        declaration = {"kind": "full_row_coordinates"} if self.declaration is None else dict(self.declaration)
        kind = declaration.get("kind")
        if kind not in {"full_row_coordinates", "native_EEG", "measured_reference", "average_actual_EEG"}:
            raise ValueError("unknown actual observation declaration")
        scales = np.max(np.abs(matrix), axis=1)
        normalized = matrix / scales[:, None]
        count = len(self.layout.channels)
        rows = len(matrix)
        if kind != "full_row_coordinates":
            gains = finite_real_array(declaration.get("row_gains"), name="declared output row gains")
            if gains.shape != (rows,) or np.any(gains == 0):
                raise ValueError("observation declaration needs one finite nonzero gain per actual output")
            if declaration.get("native_channels") != list(self.layout.channels):
                raise ValueError("observation declaration differs from actual measured native EEG")
            if kind == "native_EEG":
                expected = np.eye(count)
            elif kind == "measured_reference":
                anchor = declaration.get("reference_channel")
                if anchor not in self.layout.channels:
                    raise ValueError("observation declaration requests an unmeasured reference")
                index = self.layout.channels.index(anchor)
                selected = [i for i in range(count) if i != index]
                expected = np.eye(count)[selected]
                expected[:, index] -= 1
            else:
                expected = np.eye(count) - np.ones((count, count)) / count
            if expected.shape != matrix.shape:
                raise ValueError("observation declaration has different actual dimensions")
            # Check source-generation relations in native row units. Float CAR
            # roundoff is part of this declared operator, not a seventh signal.
            generated = matrix / gains[:, None]
            bound = 128 * _gamma(count) * np.maximum(np.abs(expected), np.abs(generated))
            if not np.isfinite(generated).all() or np.any(np.abs(generated - expected) > bound):
                raise ValueError("actual observation matrix differs from its structural declaration")
        if kind == "average_actual_EEG":
            rank = count - 1
            null = np.sign(gains) / np.sqrt(count)
            support = _null_complement(null)
            # Use the declared rank. Numeric remnants of the reference null
            # are bounded above; they cannot become new likelihood dimensions.
            normalized = support @ (support.T @ normalized)
            logs = np.log(scales)
            volume = float(logs.sum() + .5 * logsumexp(2 * np.log(np.abs(null)) - 2 * logs))
            physical_null = np.sign(null) * np.exp(np.log(np.abs(null)) - logs
                - np.max(np.log(np.abs(null)) - logs))
            physical_null /= np.linalg.norm(physical_null)
            physical_support = _null_complement(physical_null)
            singular = np.linalg.svd(support.T @ normalized, compute_uv=False)
        else:
            rank = rows
            support = np.eye(rows)
            physical_support = support
            volume = float(np.log(scales).sum())
            singular = np.linalg.svd(normalized, compute_uv=False)
        threshold = 64 * _gamma(max(matrix.shape)) * singular[0]
        if rank > count or len(singular) != rank or singular[-1] <= threshold:
            raise SpatialPriorIdentificationError("observation_resolution_unresolved",
                declared_output_dimensions=rows, normalized_singular_values=singular.tolist(),
                normalized_resolution_threshold=float(threshold), declaration=declaration)
        object.__setattr__(self, "matrix", frozen_array(matrix))
        object.__setattr__(self, "output_names", tuple(self.output_names))
        object.__setattr__(self, "declaration", freeze_metadata(declaration))
        object.__setattr__(self, "support_basis", frozen_array(physical_support))
        object.__setattr__(self, "normalized_support_basis", frozen_array(support))
        object.__setattr__(self, "normalized_matrix", frozen_array(normalized))
        object.__setattr__(self, "equilibration_scales", frozen_array(scales))
        object.__setattr__(self, "coordinate_log_volume", volume)
        object.__setattr__(self, "rank_ledger", freeze_metadata({
            "rank": rank, "structural_rank": rank, "output_dimensions": len(self.output_names),
            "native_measured_dimensions": len(self.layout.channels),
            "normalized_singular_values": singular.tolist(), "normalized_resolution_threshold": float(threshold),
            "policy": "declared_reference_or_certified_complete_output_range",
            "row_units_removed_before_resolution": True,
            "structural_evidence": kind, "lost_native_directions": matrix.shape[1] - rank}))

    @classmethod
    def native(cls, layout):
        return cls(layout, np.eye(len(layout.channels)), layout.channels, layout.acquisition_reference,
                   {"kind": "native_EEG", "native_channels": list(layout.channels),
                    "row_gains": [1.] * len(layout.channels)})

    @classmethod
    def measured_reference(cls, layout, reference_channel):
        if reference_channel not in layout.channels:
            raise SpatialPriorIdentificationError("reference_not_measured", channel=reference_channel)
        anchor = layout.channels.index(reference_channel)
        indices = [i for i in range(len(layout.channels)) if i != anchor]
        matrix = np.eye(len(layout.channels))[indices]
        matrix[:, anchor] -= 1
        names = tuple(f"{layout.channels[i]}-{reference_channel}" for i in indices)
        return cls(layout, matrix, names, "measured_channel:" + reference_channel,
                   {"kind": "measured_reference", "native_channels": list(layout.channels),
                    "reference_channel": reference_channel, "row_gains": [1.] * len(names)})

    @classmethod
    def average_reference(cls, layout):
        count = len(layout.channels)
        matrix = np.eye(count) - np.ones((count, count)) / count
        return cls(layout, matrix, tuple(name + "-CAR" for name in layout.channels), "CAR_actual_EEG",
                   {"kind": "average_actual_EEG", "native_channels": list(layout.channels),
                    "row_gains": [1.] * count})

    def with_output_gains(self, gains):
        gains = finite_real_array(gains, name="actual output row gains")
        if gains.shape != (len(self.matrix),) or np.any(gains == 0):
            raise ValueError("one finite nonzero gain is required per actual output")
        declaration = json.loads(json.dumps(self.declaration))
        if declaration["kind"] != "full_row_coordinates":
            declaration["row_gains"] = (np.asarray(declaration["row_gains"]) * gains).tolist()
        return SpatialObservation(self.layout, self.matrix * gains[:, None], self.output_names,
                                  self.output_reference, declaration)

    def record(self):
        return {"schema": _OBSERVATION_SCHEMA, "layout": self.layout.record(), "matrix": self.matrix.tolist(),
                "output_names": list(self.output_names), "output_reference": self.output_reference,
                "declaration": json.loads(json.dumps(self.declaration)),
                "equilibration_scales": self.equilibration_scales.tolist(),
                "coordinate_log_volume": self.coordinate_log_volume,
                "rank_ledger": json.loads(json.dumps(self.rank_ledger)),
                "matrix_sha256": array_digest(self.matrix)}

    def project(self, native_samples):
        values = _matrix(native_samples, "native measured samples")
        if values.shape[1] != len(self.layout.channels):
            raise ValueError("native samples differ from actual measured channels")
        return values @ self.matrix.T

    def _noise_root(self, native_root):
        """Push the frozen native noise factor through the actual range map.

        Forming M Sigma M.T and testing each near-zero off-diagonal relatively
        confuses cancellation roundoff with failed positive definiteness. QR
        of (M L_native).T preserves the generative factor instead. Row units
        are removed before certifying its complete observed noise directions.
        """
        mapping = self.normalized_support_basis.T @ self.normalized_matrix
        factor = mapping @ native_root
        scales = np.max(np.abs(factor), axis=1)
        if not np.isfinite(factor).all() or np.any(scales <= 0):
            raise SpatialPriorIdentificationError("observed_noise_resolution_unresolved")
        normalized = factor / scales[:, None]
        _, upper = np.linalg.qr(normalized.T, mode="reduced")
        singular = np.linalg.svd(upper, compute_uv=False)
        threshold = 64 * _gamma(max(normalized.shape)) * singular[0]
        if singular[-1] <= threshold:
            raise SpatialPriorIdentificationError("observed_noise_resolution_unresolved",
                normalized_singular_values=singular.tolist(), normalized_resolution_threshold=float(threshold))
        # A column sign change leaves L L.T intact and fixes a positive diagonal.
        upper = np.sign(np.diag(upper))[:, None] * upper
        _unit_information_relation(normalized.T, upper)
        root = scales[:, None] * upper.T
        if not np.isfinite(root).all() or np.any(np.diag(root) <= 0):
            raise SpatialPriorIdentificationError("observed_noise_resolution_unresolved")
        return mapping, root

    def _whiten(self, samples, native_noise, native_root=None):
        # Pure output gains change a Jacobian, never the measured information.
        normalized = samples / self.equilibration_scales
        if not np.isfinite(normalized).all():
            raise SpatialPriorIdentificationError("observed_coordinate_resolution_unresolved")
        support = self.normalized_support_basis
        values = normalized @ support
        outside = normalized - values @ support.T
        bound = 128 * _gamma(max(self.matrix.shape)) * (
            np.abs(normalized) + np.abs(normalized) @ np.abs(support) @ np.abs(support.T))
        if np.any(np.abs(outside) > bound):
            raise SpatialPriorIdentificationError("sample_outside_observation_range",
                normalized_max_residual=float(np.max(np.abs(outside))),
                maximum_componentwise_rounding_bound=float(np.max(bound)))
        if native_root is None:
            _, native_root = _spd(native_noise, "native residual noise")
        mapping, root = self._noise_root(native_root)
        return (solve_triangular(root, values.T, lower=True).T,
                solve_triangular(root, mapping, lower=True),
                2 * float(np.log(np.diag(root)).sum()) + 2 * self.coordinate_log_volume)


@dataclass(frozen=True)
class SpatialSignalPrior:
    """Proper physical response prior; H_time is independent of residual noise."""

    layout: SpatialLayout
    mean: np.ndarray = field(repr=False, compare=False)
    temporal_covariance: np.ndarray = field(repr=False, compare=False)
    signal_covariance: np.ndarray = field(repr=False, compare=False)
    design_binding: dict
    source: dict
    precision_root: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.layout, SpatialLayout):
            raise TypeError("signal prior requires owned actual layout")
        mean = _matrix(self.mean, "physical response prior mean")
        temporal, temporal_root = _spd(self.temporal_covariance, "temporal signal prior")
        signal, signal_root = _spd(self.signal_covariance, "spatial signal prior")
        if mean.shape != (len(temporal), len(self.layout.channels)) or signal.shape != (mean.shape[1],) * 2:
            raise ValueError("signal prior dimensions differ from physical response coordinates")
        if not isinstance(self.design_binding, dict) or not isinstance(self.source, dict):
            raise ValueError("signal prior needs response-design and source bindings")
        require_sha256(self.design_binding.get("dictionary_sha256"), "response dictionary identity")
        if self.design_binding.get("vectorization") != "column_major_response_by_native_EEG":
            raise ValueError("signal prior requires explicit column-major response coordinates")
        if self.source.get("role") not in {"predeclared", "source_training"}:
            raise ValueError("signal prior source must be predeclared or source-only training")
        require_sha256(self.source.get("authority_sha256"), "signal prior source authority")
        root = np.kron(solve_triangular(signal_root, np.eye(len(signal)), lower=True),
                       solve_triangular(temporal_root, np.eye(len(temporal)), lower=True))
        for name, value in (("mean", mean), ("temporal_covariance", temporal),
                            ("signal_covariance", signal), ("precision_root", root)):
            object.__setattr__(self, name, frozen_array(value))
        object.__setattr__(self, "design_binding", freeze_metadata(self.design_binding))
        object.__setattr__(self, "source", freeze_metadata(self.source))

    @classmethod
    def graph(cls, *, layout, mean, temporal_covariance, variance, length_scale_m,
              graph_strength, design_binding, source):
        if layout.positions_m is None:
            raise SpatialPriorIdentificationError("graph_requires_actual_positions")
        variance = _positive(variance, "native signal variance in squared voltage units")
        length = _positive(length_scale_m, "graph spatial length scale in metres")
        strength = _positive(graph_strength, "predeclared graph strength")
        delta = layout.positions_m[:, None, :] - layout.positions_m[None, :, :]
        weights = np.exp(-np.sum(delta**2, axis=2) / (2 * length**2))
        np.fill_diagonal(weights, 0)
        laplacian = np.diag(weights.sum(axis=1)) - weights
        covariance = variance * np.linalg.solve(np.eye(len(weights)) + strength * laplacian,
                                                np.eye(len(weights)))
        source = {**source, "graph": {"construction": "actual_distance_gaussian_weights_proper_I_plus_L",
                  "variance": variance, "length_scale_m": length, "strength": strength,
                  "laplacian_sha256": array_digest(laplacian)}}
        return cls(layout, mean, temporal_covariance, covariance, design_binding, source)

    @classmethod
    def learn(cls, *, mechanism, independent_groups=None, **kwargs):
        # No unvalidated d-prime/graph estimator can silently replace this prior.
        if mechanism not in {"dprime", "graph"}:
            raise ValueError("unknown learned spatial signal prior")
        groups = () if independent_groups is None else tuple(independent_groups)
        reason = "independent_training_groups_required" if len(set(groups)) < 2 else "learned_spatial_prior_not_implemented"
        raise SpatialPriorIdentificationError(reason, mechanism=mechanism,
                                              independent_training_groups=len(set(groups)))

    def record(self):
        state = {"schema": _PRIOR_SCHEMA, "layout": self.layout.record(), "mean": self.mean.tolist(),
                 "temporal_covariance": self.temporal_covariance.tolist(),
                 "signal_covariance": self.signal_covariance.tolist(),
                 "design_binding": json.loads(json.dumps(self.design_binding)),
                 "source": json.loads(json.dumps(self.source)),
                 "precision_root": self.precision_root.tolist(),
                 "prior_covariance_semantics": "S_signal_kronecker_H_time_independent_of_noise",
                 "arrays_sha256": array_digest(self.mean, self.temporal_covariance, self.signal_covariance,
                                                self.precision_root)}
        return {**state, "artifact_sha256": metadata_digest(state)}

    def digest(self):
        return self.record()["artifact_sha256"]

    @classmethod
    def from_record(cls, record):
        payload = json.loads(json.dumps(record, allow_nan=False))
        digest = payload.pop("artifact_sha256", None)
        if payload.pop("schema", None) != _PRIOR_SCHEMA or digest != metadata_digest({"schema": _PRIOR_SCHEMA, **payload}):
            raise ValueError("spatial signal prior artifact identity mismatch")
        expected_arrays = payload.pop("arrays_sha256")
        saved_root = _matrix(payload.pop("precision_root"), "producer signal prior precision root")
        if payload.pop("prior_covariance_semantics") != "S_signal_kronecker_H_time_independent_of_noise":
            raise ValueError("spatial signal prior covariance semantics mismatch")
        prior = cls(layout=SpatialLayout.from_record(payload.pop("layout")), **payload)
        if (saved_root.shape != prior.precision_root.shape or np.any(np.triu(saved_root, 1) != 0)
                or np.any(np.diag(saved_root) <= 0)):
            raise ValueError("producer signal prior root differs from its declared triangular coordinates")
        _unit_information_relation(prior.precision_root, saved_root, lower=True)
        if array_digest(prior.mean, prior.temporal_covariance, prior.signal_covariance, saved_root) != expected_arrays:
            raise ValueError("spatial signal prior arrays differ from their frozen record")
        object.__setattr__(prior, "precision_root", frozen_array(saved_root))
        return prior


@dataclass(frozen=True)
class SpatialResponseBlock:
    """Actual rows after caller-owned temporal whitening of Y, D and Z.

    Prediction records contain an unlabeled candidate design. Their explicit
    read-only role cannot be submitted to posterior fitting.
    """

    samples: np.ndarray = field(repr=False, compare=False)
    design: np.ndarray = field(repr=False, compare=False)
    observation: SpatialObservation
    authorization: dict
    nuisance: np.ndarray | None = field(default=None, repr=False, compare=False)
    temporal_log_abs_det: float = 0.0

    def __post_init__(self):
        samples, design = _matrix(self.samples, "C samples"), _matrix(self.design, "actual C response design")
        if (not isinstance(self.observation, SpatialObservation) or len(samples) != len(design)
                or samples.shape[1] != len(self.observation.output_names)):
            raise ValueError("C design/sample dimensions differ from actual observed coordinates")
        authorization = self.authorization
        if (not isinstance(authorization, dict)
                or authorization.get("fit_scope") not in {"target_calibration", "source_training", "read_only_prediction"}
                or not authorization.get("record_id")):
            raise ValueError("spatial response rows require actual authority and an explicit information role")
        for key in ("source_sha256", "dictionary_sha256", "temporal_program_sha256"):
            require_sha256(authorization.get(key), "C " + key)
        support = authorization.get("raw_sample_support")
        if (not isinstance(support, (tuple, list)) or len(support) != 2
                or any(type(value) is not int for value in support) or support[0] < 0
                or support[1] - support[0] != len(samples)):
            raise ValueError("C actual raw support must match the complete temporally transformed rows")
        if not np.isfinite(self.temporal_log_abs_det):
            raise ValueError("temporal whitening determinant must be finite")
        if self.nuisance is not None:
            nuisance = _matrix(self.nuisance, "temporally whitened diffuse nuisance")
            if len(nuisance) != len(samples):
                raise ValueError("nuisance must share the same actual observed rows")
            object.__setattr__(self, "nuisance", frozen_array(nuisance))
        object.__setattr__(self, "samples", frozen_array(samples))
        object.__setattr__(self, "design", frozen_array(design))
        object.__setattr__(self, "authorization", freeze_metadata(authorization))

    def _compressed(self, native_noise, native_root=None):
        y, mapping, logdet_noise = self.observation._whiten(self.samples, native_noise, native_root)
        design = self.design
        nuisance_rank, nuisance_logdet = 0, 0.0
        if self.nuisance is not None:
            q, r = np.linalg.qr(self.nuisance, mode="reduced")
            singular = np.linalg.svd(r, compute_uv=False)
            threshold = np.finfo(float).eps * max(self.nuisance.shape) * singular[0]
            nuisance_rank = int(np.sum(singular > threshold))
            if nuisance_rank != self.nuisance.shape[1] or nuisance_rank >= len(y):
                raise SpatialPriorIdentificationError("diffuse_nuisance_not_identified",
                                                      rank=nuisance_rank, threshold=float(threshold))
            nuisance_logdet = 2 * float(np.log(np.abs(np.diag(r))).sum())
            design = design - q @ (q.T @ design)
            y = y - q @ (q.T @ y)
        q, root = np.linalg.qr(design, mode="reduced")
        response = q.T @ y
        outside = y - q @ response
        return (np.kron(mapping, root), response.flatten(order="F"),
                float(np.sum(outside**2)), len(y) - nuisance_rank, mapping.shape[0],
                logdet_noise, nuisance_logdet)

    def record(self):
        return {"authorization": json.loads(json.dumps(self.authorization)),
                "observation": self.observation.record(), "samples": len(self.samples),
                "response_dimensions": self.design.shape[1],
                "actual_input_sha256": array_digest(self.samples, self.design,
                    np.empty((len(self.samples), 0)) if self.nuisance is None else self.nuisance),
                "nuisance_dimensions": 0 if self.nuisance is None else self.nuisance.shape[1],
                "temporal_log_abs_det": self.temporal_log_abs_det}


def _posterior_qr(prior, roots, responses):
    stacked = np.vstack((prior.precision_root, *roots))
    rhs = np.concatenate((prior.precision_root @ prior.mean.flatten(order="F"), *responses))
    scales = _column_scales(stacked)
    q, normalized_root = np.linalg.qr(stacked / scales, mode="reduced")
    root = normalized_root * scales
    response = q.T @ rhs
    singular = np.linalg.svd(normalized_root, compute_uv=False)
    threshold = 64 * _gamma(max(stacked.shape)) * singular[0]
    if singular[-1] <= threshold:
        raise SpatialPriorIdentificationError("posterior_precision_numerically_unresolved",
            singular_values=singular.tolist(), threshold=float(threshold))
    vector = solve_triangular(normalized_root, response) / scales
    inverse = solve_triangular(normalized_root, np.eye(len(root))) / scales[:, None]
    return root, response, vector.reshape(prior.mean.shape, order="F"), inverse @ inverse.T, q, scales


def _posterior_state(prior, roots, responses, state):
    """Admit producer QR arrays without constructing a new platform-specific QR."""
    width = prior.mean.size
    root = _matrix(state["precision_root"], "producer posterior precision root")
    mean = _matrix(state["mean"], "producer posterior mean")
    covariance = _matrix(state["covariance"], "producer posterior covariance")
    q = _matrix(state["orthogonal_factor"], "producer actual-C QR orthogonal factor")
    response = finite_real_array(state["response"], name="producer posterior response")
    scales = finite_real_array(state["qr_column_scales"], name="actual-C QR coefficient scales")
    stacked = np.vstack((prior.precision_root, *roots))
    rhs = np.concatenate((prior.precision_root @ prior.mean.flatten(order="F"), *responses))
    if (root.shape != (width, width) or mean.shape != prior.mean.shape or covariance.shape != (width, width)
            or q.shape != stacked.shape or response.shape != (width,) or scales.shape != (width,)
            or np.any(np.tril(root, -1) != 0) or np.any(np.diag(root) == 0)
            or not np.array_equal(scales, _column_scales(stacked))):
        raise ValueError("producer posterior factors differ from actual C dimensions or coordinate scales")
    normalized_root, normalized_stack = root / scales, stacked / scales
    tolerance = 256 * _gamma(max(stacked.shape))
    q_gram_bound = _gamma(len(q)) * (np.abs(q).T @ np.abs(q))
    if np.any(np.abs(q.T @ q - np.eye(width)) > tolerance + q_gram_bound):
        raise ValueError("producer actual-C QR factor is not orthonormal in every direction")
    qr_bound = 256 * _gamma(width) * (np.abs(normalized_stack) + np.abs(q) @ np.abs(normalized_root))
    if np.any(np.abs(normalized_stack - q @ normalized_root) > qr_bound):
        raise ValueError("producer posterior QR root differs from actual C likelihood information")
    _unit_information_relation(stacked, root)
    response_bound = 256 * _gamma(len(q)) * (np.abs(q).T @ np.abs(rhs))
    if np.any(np.abs(response - q.T @ rhs) > response_bound):
        raise ValueError("producer posterior response differs from actual C likelihood directions")
    coefficient = scales * mean.flatten(order="F")
    mean_bound = 256 * _gamma(width) * (np.abs(normalized_root) @ np.abs(coefficient) + np.abs(response))
    if np.any(np.abs(normalized_root @ coefficient - response) > mean_bound):
        raise ValueError("saved spatial posterior mean violates its actual C information")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_covariance = (covariance * scales[:, None]) * scales[None, :]
    if not np.isfinite(normalized_covariance).all():
        raise SpatialPriorIdentificationError("posterior_covariance_resolution_unresolved")
    _spd(normalized_covariance, "producer posterior covariance in coefficient units")
    gram = normalized_root @ normalized_covariance @ normalized_root.T
    gram_bound = (3 * _gamma(width) + 2 * np.finfo(float).eps) * (
        np.abs(normalized_root) @ np.abs(normalized_covariance) @ np.abs(normalized_root.T))
    if not np.isfinite(gram_bound).all() or np.max(gram_bound) > _IDENTITY_RESOLUTION:
        raise SpatialPriorIdentificationError("posterior_covariance_resolution_unresolved",
            relative_identity_error_bound=float(np.max(gram_bound)), identity_resolution=_IDENTITY_RESOLUTION)
    if np.any(np.abs(gram - np.eye(width)) > tolerance + gram_bound):
        raise ValueError("saved spatial posterior covariance violates its actual C information")
    return root, response, mean, covariance, q, scales


@dataclass(frozen=True)
class SpatialSignalPosterior:
    """Frozen exact general-Gaussian posterior, conditional on actual fitted noise."""

    prior: SpatialSignalPrior
    native_noise_covariance: np.ndarray = field(repr=False, compare=False)
    information_roots: tuple[np.ndarray, ...] = field(repr=False, compare=False)
    responses: tuple[np.ndarray, ...] = field(repr=False, compare=False)
    fit_records: tuple[dict, ...]
    noise_binding: dict
    restore_state: InitVar[dict | None] = None
    mean: np.ndarray = field(init=False, repr=False, compare=False)
    covariance: np.ndarray = field(init=False, repr=False, compare=False)
    precision_root: np.ndarray = field(init=False, repr=False, compare=False)
    response: np.ndarray = field(init=False, repr=False, compare=False)
    orthogonal_factor: np.ndarray = field(init=False, repr=False, compare=False)
    qr_column_scales: np.ndarray = field(init=False, repr=False, compare=False)
    native_noise_root: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self, restore_state):
        noise, native_root = _spd(self.native_noise_covariance, "native residual noise")
        width = self.prior.mean.size
        if noise.shape != self.prior.signal_covariance.shape:
            raise ValueError("residual noise does not share native physical EEG coordinates")
        if not self.information_roots or len(self.information_roots) != len(self.responses) or len(self.responses) != len(self.fit_records):
            raise ValueError("posterior needs one actual compressed likelihood per C record")
        roots, responses = [], []
        for root, response in zip(self.information_roots, self.responses, strict=True):
            matrix = _matrix(root, "C likelihood information root")
            vector = finite_real_array(response, name="C compressed response")
            if matrix.shape[1] != width or vector.shape != (len(matrix),):
                raise ValueError("C compressed likelihood differs from response coordinates")
            roots.append(frozen_array(matrix))
            responses.append(frozen_array(vector))
        for record, root, response in zip(self.fit_records, roots, responses, strict=True):
            authorization = record["authorization"]
            if (record["observation"]["layout"] != self.prior.layout.record()
                    or authorization["dictionary_sha256"] != self.prior.design_binding["dictionary_sha256"]
                    or authorization["fit_scope"] not in {"source_training", "target_calibration"}):
                raise ValueError("C layout or response design differs from signal prior authority")
            if record["compression"]["native_noise_sha256"] != array_digest(noise):
                raise ValueError("C compressed likelihood differs from its actual noise coordinates")
            if record["compression"]["likelihood_sha256"] != array_digest(root, response):
                raise ValueError("C likelihood arrays differ from their actual compression receipt")
        ids = [record["authorization"]["record_id"] for record in self.fit_records]
        require_identifiers(ids, "C record identities")
        by_source = {}
        for record in self.fit_records:
            authority = record["authorization"]
            support = authority["raw_sample_support"]
            source = authority["source_sha256"]
            for start, stop in by_source.get(source, ()):
                if max(start, support[0]) < min(stop, support[1]):
                    raise ValueError("independent spatial C records reuse actual raw sample positions")
            by_source.setdefault(source, []).append(tuple(support))
        if not isinstance(self.noise_binding, dict):
            raise ValueError("posterior needs actual fitted noise provenance")
        require_sha256(self.noise_binding.get("authority_sha256"), "fitted noise authority")
        if restore_state is not None:
            saved_native_root = _matrix(restore_state["native_noise_root"], "producer native noise factor")
            if (saved_native_root.shape != noise.shape or np.any(np.triu(saved_native_root, 1) != 0)
                    or np.any(np.diag(saved_native_root) <= 0)):
                raise ValueError("producer native noise factor differs from actual physical noise dimensions")
            # L_saved^-1 Sigma_native L_saved^-T is checked in unit directions,
            # retaining the producer factor instead of re-Cholesky publication.
            _unit_information_relation(native_root.T, saved_native_root.T)
            native_root = saved_native_root
        native_root_digest = array_digest(native_root)
        for record in self.fit_records:
            if (record["compression"]["native_noise_root_sha256"] != native_root_digest
                    or record["compression"]["noise_factorization"] != "actual_range_square_root_pushforward_QR"):
                raise ValueError("actual C likelihood differs from its frozen native noise factorization")
        state = (_posterior_qr(self.prior, roots, responses) if restore_state is None
                 else _posterior_state(self.prior, roots, responses, restore_state))
        root, response, mean, covariance, q, scales = state
        if restore_state is None:
            # Publication uses exactly the relation contract subsequently used
            # at load. An unresolved numerical state is not saved as authority.
            _posterior_state(self.prior, roots, responses, {
                "precision_root": root, "response": response, "mean": mean, "covariance": covariance,
                "orthogonal_factor": q, "qr_column_scales": scales})
        for name, value in (("native_noise_covariance", noise), ("precision_root", root),
                            ("response", response), ("mean", mean), ("covariance", covariance),
                            ("orthogonal_factor", q), ("qr_column_scales", scales),
                            ("native_noise_root", native_root)):
            object.__setattr__(self, name, frozen_array(value))
        object.__setattr__(self, "information_roots", tuple(roots))
        object.__setattr__(self, "responses", tuple(responses))
        object.__setattr__(self, "fit_records", tuple(freeze_metadata(r) for r in self.fit_records))
        object.__setattr__(self, "noise_binding", freeze_metadata(self.noise_binding))

    @classmethod
    def fit(cls, prior, blocks, *, native_noise_covariance, noise_binding):
        noise, native_root = _spd(native_noise_covariance, "native residual noise")
        roots, responses, records = [], [], []
        for block in blocks:
            if (not isinstance(block, SpatialResponseBlock)
                    or block.authorization["fit_scope"] not in {"target_calibration", "source_training"}
                    or block.observation.layout.record() != prior.layout.record()
                    or block.design.shape[1] != prior.mean.shape[0]
                    or block.authorization["dictionary_sha256"] != prior.design_binding["dictionary_sha256"]):
                raise ValueError("C actual input differs from the bound spatial signal prior")
            root, response, *_ = block._compressed(noise, native_root)
            roots.append(root)
            responses.append(response)
            record = block.record()
            record["compression"] = {"native_noise_sha256": array_digest(noise),
                                     "native_noise_root_sha256": array_digest(native_root),
                                     "noise_factorization": "actual_range_square_root_pushforward_QR",
                                     "likelihood_sha256": array_digest(root, response)}
            records.append(record)
        return cls(prior, noise, tuple(roots), tuple(responses), tuple(records), noise_binding)

    def log_density(self, block):
        """Exact predictive likelihood on actual spatial range.

        Without Z this is a normalized Gaussian predictive density. With Z,
        it integrates a fixed flat nuisance measure, not a proper unconditional
        raw-data probability. Its full/prefix ratio is a proper conditional
        density when the actual prefix identifies Z. The determinant of Z is
        retained; a projected quadratic alone is wrong.
        No candidate signal or label changes the prior or fitted residual noise.
        """
        if (block.observation.layout.record() != self.prior.layout.record()
                or block.design.shape[1] != self.mean.shape[0]
                or block.authorization["dictionary_sha256"] != self.prior.design_binding["dictionary_sha256"]):
            raise ValueError("prediction input differs from frozen response/physical coordinates")
        design, values, outside, rows, channels, logdet_noise, logdet_nuisance = block._compressed(
            self.native_noise_covariance, self.native_noise_root)
        centered = values - design @ self.mean.flatten(order="F")
        stacked = np.vstack((design, self.precision_root))
        scales = _column_scales(stacked)
        normalized = stacked / scales
        q, root = np.linalg.qr(normalized, mode="reduced")
        rhs = np.concatenate((centered, np.zeros(self.mean.size)))
        coefficient = solve_triangular(root, q.T @ rhs)
        residual = rhs - normalized @ coefficient
        quadratic = float(residual @ residual) + outside
        logdet = 2 * float(np.log(np.abs(np.diag(root))).sum()
                          - np.log(np.abs(np.diag(self.precision_root / scales))).sum())
        return float(-0.5 * (rows * channels * np.log(2 * np.pi) + rows * logdet_noise
                            + channels * logdet_nuisance + logdet + quadratic)
                     + channels * block.temporal_log_abs_det)

    def conditional_log_density(self, full, prefix):
        """p(full tail | actual prefix,C), retaining response and nuisance updates."""
        if (prefix.observation.record() != full.observation.record()
                or any(prefix.authorization[key] != full.authorization[key]
                       for key in ("source_sha256", "dictionary_sha256", "temporal_program_sha256"))
                or prefix.authorization["raw_sample_support"][0] != full.authorization["raw_sample_support"][0]
                or len(prefix.samples) >= len(full.samples)
                or not np.array_equal(prefix.samples, full.samples[:len(prefix.samples)])
                or not np.array_equal(prefix.design, full.design[:len(prefix.samples)])
                or ((prefix.nuisance is None) != (full.nuisance is None))
                or (prefix.nuisance is not None and not np.array_equal(prefix.nuisance, full.nuisance[:len(prefix.samples)]))):
            raise ValueError("conditional prefix must be the same actual rows and coordinates")
        return self.log_density(full) - self.log_density(prefix)

    def record(self):
        state = {"schema": _POSTERIOR_SCHEMA, "prior": self.prior.record(),
                 "native_noise_covariance": self.native_noise_covariance.tolist(),
                 "native_noise_root": self.native_noise_root.tolist(),
                 "noise_binding": json.loads(json.dumps(self.noise_binding)),
                 "information_roots": [root.tolist() for root in self.information_roots],
                 "responses": [response.tolist() for response in self.responses],
                 "fit_records": json.loads(json.dumps(self.fit_records)),
                 "mean": self.mean.tolist(), "covariance": self.covariance.tolist(),
                 "precision_root": self.precision_root.tolist(), "response": self.response.tolist(),
                 "orthogonal_factor": self.orthogonal_factor.tolist(),
                 "qr_column_scales": self.qr_column_scales.tolist(),
                 "numeric_sha256": array_digest(self.mean, self.covariance, self.native_noise_covariance,
                                                self.precision_root, self.response,
                                                self.orthogonal_factor, self.qr_column_scales, self.native_noise_root),
                 "observed_noise_factorization": "actual_range_square_root_pushforward_QR",
                 "posterior_semantics": "exact_vec_Gaussian_conditional_on_frozen_noise",
                 "precision_binding": {"method": "actual_C_column_equilibrated_QR_and_unit_directions",
                    "identity_resolution": _IDENTITY_RESOLUTION,
                    "actual_stacked_root_sha256": array_digest(np.vstack((self.prior.precision_root, *self.information_roots))),
                    "actual_stacked_response_sha256": array_digest(np.concatenate((
                        self.prior.precision_root @ self.prior.mean.flatten(order="F"), *self.responses)))},
                 "fit_sha256": metadata_digest(self.fit_records)}
        return {**state, "artifact_sha256": metadata_digest(state)}

    @classmethod
    def from_record(cls, record):
        payload = json.loads(json.dumps(record, allow_nan=False))
        artifact = payload.pop("artifact_sha256", None)
        if artifact != metadata_digest(payload) or payload.get("schema") != _POSTERIOR_SCHEMA:
            raise ValueError("spatial signal posterior artifact identity mismatch")
        if payload["posterior_semantics"] != "exact_vec_Gaussian_conditional_on_frozen_noise":
            raise ValueError("spatial signal posterior semantics mismatch")
        if payload["observed_noise_factorization"] != "actual_range_square_root_pushforward_QR":
            raise ValueError("spatial observed noise factorization differs from the frozen program")
        state_names = ("precision_root", "response", "mean", "covariance", "orthogonal_factor", "qr_column_scales",
                       "native_noise_root")
        state = {name: payload[name] for name in state_names}
        model = cls(SpatialSignalPrior.from_record(payload["prior"]), payload["native_noise_covariance"],
                    tuple(payload["information_roots"]), tuple(payload["responses"]),
                    tuple(payload["fit_records"]), payload["noise_binding"], restore_state=state)
        if payload["numeric_sha256"] != array_digest(model.mean, model.covariance, model.native_noise_covariance,
            model.precision_root, model.response, model.orthogonal_factor, model.qr_column_scales, model.native_noise_root):
            raise ValueError("saved spatial posterior numeric identity mismatch")
        if payload["fit_sha256"] != metadata_digest(model.fit_records):
            raise ValueError("saved spatial posterior C identity mismatch")
        if payload["precision_binding"] != model.record()["precision_binding"]:
            raise ValueError("saved spatial precision differs from its actual prior/C/source binding")
        return model

    def save(self, path):
        return atomic_write_json(path, self.record())

    @classmethod
    def load(cls, path, *, expected_artifact_sha256=None):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if expected_artifact_sha256 is not None and record.get("artifact_sha256") != expected_artifact_sha256:
            raise ValueError("spatial signal posterior differs from external artifact authority")
        return cls.from_record(record)
