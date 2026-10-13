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
from dataclasses import InitVar, dataclass, field, replace
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls
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
from transfer.temporal_noise import TemporalNoise

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_PRIOR_SCHEMA = "n2p3_spatial_signal_prior/2"
_POSTERIOR_SCHEMA = "n2p3_spatial_signal_posterior/4"
_OBSERVATION_SCHEMA = "n2p3_spatial_observation/2"
_IDENTITY_RESOLUTION = float(np.sqrt(np.finfo(float).eps))
_STRUCTURAL_MAX_SCALARS = 250_000
_STRUCTURAL_MAX_COLUMNS = 512


def _matrix(value, name):
    result = finite_real_array(value, name=name)
    if result.ndim != 2 or min(result.shape) == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    return result


def _exact_structural_null(operators, width):
    """Exact binary64 structural null of actual design plus record nuisances.

    The operators are physical `X|Z` rows in the actual declared observation
    chart before noise/QR compression. Declared CAR uses exact EEG differences;
    this preserves its known reference null through floating chart generation.
    Every stored finite operand is promoted to its exact dyadic Fraction; no
    tolerance can turn an exact nonzero functional on this null into signal.
    """
    operators = tuple(_matrix(value, "actual structural C operator") for value in operators)
    if not operators or any(operator.shape[1] < width for operator in operators):
        raise ValueError("structural C operators require shared coefficients and local nuisance columns")
    nuisance_width = sum(operator.shape[1] - width for operator in operators)
    rows = sum(len(operator) for operator in operators)
    columns = width + nuisance_width
    if rows * columns > _STRUCTURAL_MAX_SCALARS or columns > _STRUCTURAL_MAX_COLUMNS:
        raise SpatialPriorIdentificationError("structural_identifiability_unresolved",
            rows=rows, columns=columns, scalar_budget=_STRUCTURAL_MAX_SCALARS,
            column_budget=_STRUCTURAL_MAX_COLUMNS)
    matrix = []
    offset = width
    for operator in operators:
        local = operator.shape[1] - width
        for row in operator:
            values = [Fraction.from_float(float(value)) for value in row[:width]]
            values.extend(Fraction(0) for _ in range(nuisance_width))
            values[offset:offset + local] = (Fraction.from_float(float(value)) for value in row[width:])
            matrix.append(values)
        offset += local
    rank = 0
    pivots = []
    for column in range(columns):
        pivot = next((row for row in range(rank, rows) if matrix[row][column]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        divisor = matrix[rank][column]
        matrix[rank] = [value / divisor for value in matrix[rank]]
        for row in range(rows):
            if row == rank or not matrix[row][column]:
                continue
            scale = matrix[row][column]
            matrix[row] = [left - scale * right for left, right in zip(matrix[row], matrix[rank], strict=True)]
        pivots.append(column)
        rank += 1
        if rank == rows:
            break
    free = [column for column in range(columns) if column not in pivots]
    basis = []
    for free_column in free:
        vector = [Fraction(0)] * columns
        vector[free_column] = Fraction(1)
        for row, pivot in enumerate(pivots):
            vector[pivot] = -matrix[row][free_column]
        beta = tuple(vector[:width])
        if any(beta):
            basis.append(beta)
    return tuple(tuple(value for value in beta) for beta in basis), {
        "rows": rows, "columns": columns, "rank": rank, "nullity": len(free),
        "beta_nullity": len(basis), "arithmetic": "exact_binary64_fraction_rref",
    }


def _structural_operator_digest(operators):
    return array_digest(*(np.asarray(operator, dtype=np.float64) for operator in operators))


def _validate_source_learning_semantics(source):
    learning = source.get("source_learning")
    if learning is None:
        return
    if (learning.get("mechanism") != "source_group"
            or learning.get("completion") not in {"mean_only", "separable"}
            or learning.get("variance_uncertainty_integrated") is not False
            or learning.get("unseen_spatial_directions") != "proper_reference_conditional_completion"
            or learning.get("source_forecast_uncertainty_propagated") is not (learning.get("completion") == "separable")):
        raise ValueError("source-learned prior semantic contract differs from its canonical completion")
    require_sha256(learning.get("effect_definition_sha256"), "source-learned effect definition")
    require_identifiers(learning.get("source_participant_keys", ()), "source-learned source participants")
    target = learning.get("target_participant_key")
    if not isinstance(target, str) or not target.strip() or target in learning["source_participant_keys"]:
        raise ValueError("source-learned prior target participant must be an independent declared identity")


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


def _qr_column_relation(actual, q, root):
    """Householder QR reconstruction in each original coefficient column.

    QR is backward stable at column scale, including when an input entry is
    exactly zero. The orthogonal factor's construction and the subsequent dot
    products can mix all rows of that column; an entry's own near-zero result
    is therefore not their roundoff scale. Independent unit-information and
    orthogonality certificates still resolve every weak precision direction.
    """
    residual = actual - q @ root
    original_norms = np.sqrt(np.sum(actual**2, axis=0))
    product_norms = np.sqrt(np.sum((np.abs(q) @ np.abs(root))**2, axis=0))
    rows, width = actual.shape
    q_generation = _gamma(6 * rows * width) * original_norms
    multiplication = _gamma(width) * product_norms
    bounds = q_generation + multiplication
    residual_norms = np.sqrt(np.sum(residual**2, axis=0))
    if np.any(residual_norms > bounds):
        raise ValueError("producer posterior QR root differs from actual C likelihood information")


def _binary_integer_matrix(matrix):
    """Represent finite binary64 operands over one exact power-of-two denominator."""
    ratios = [float(value).as_integer_ratio() for value in np.asarray(matrix).flat]
    powers = [denominator.bit_length() - 1 for _, denominator in ratios]
    power = max(powers, default=0)
    values = [numerator << (power - item_power)
              for (numerator, _), item_power in zip(ratios, powers, strict=True)]
    return np.asarray(values, dtype=object).reshape(matrix.shape), power


def _exact_covariance_relation(root, covariance):
    """Rare exact check of the stored R H R.T operands, with no tolerance conversion.

    Each finite float is a rational with a power-of-two denominator. Two Python
    integer matrix products therefore give the exact quadratic; comparing its
    integer residual to the original resolution avoids an inconclusive float
    cancellation enclosure. This is ingress/publication work, never scoring.
    """
    integer_root, root_power = _binary_integer_matrix(root)
    integer_covariance, covariance_power = _binary_integer_matrix(covariance)
    quadratic = (integer_root @ integer_covariance) @ integer_root.T
    denominator = 1 << (2 * root_power + covariance_power)
    numerator = max(abs(int(quadratic[i, j]) - (denominator if i == j else 0))
                    for i in range(len(root)) for j in range(len(root)))
    resolution_numerator, resolution_denominator = _IDENTITY_RESOLUTION.as_integer_ratio()
    if numerator * resolution_denominator > denominator * resolution_numerator:
        raise SpatialPriorIdentificationError("posterior_covariance_resolution_unresolved",
            exact_unit_error_numerator=str(numerator), exact_unit_error_denominator=str(denominator),
            identity_resolution_numerator=str(resolution_numerator),
            identity_resolution_denominator=str(resolution_denominator))
    return {"arithmetic": "exact_binary64_power_of_two_integer_quadratic",
            "unit_error_numerator": str(numerator), "unit_error_denominator": str(denominator),
            "identity_resolution_numerator": str(resolution_numerator),
            "identity_resolution_denominator": str(resolution_denominator)}


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
            if tuple(declaration.get("native_channels", ())) != self.layout.channels:
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

    def _structural_matrix(self):
        if self.declaration["kind"] == "average_actual_EEG":
            count = len(self.layout.channels)
            return np.column_stack((np.eye(count - 1), -np.ones(count - 1)))
        return self.normalized_support_basis.T @ self.normalized_matrix

    def _exact_effect_map(self):
        """Physical covectors and their actual declared generation roundoff.

        Q is the stored Euclidean support chart. For declared CAR the physical
        map is exactly Q.T diag(g) (I - 11.T/C), not a tiny reference component
        created by floating matrix multiplication. An arbitrary output matrix
        has no such authority and retains every stored nonzero coefficient.
        """
        numerical = self.support_basis.T @ self.matrix
        if self.declaration["kind"] != "average_actual_EEG":
            return tuple(tuple(Fraction.from_float(float(value)) for value in row) for row in numerical), {
                "authority": "actual_binary64_functional", "generation_rounding_bound": np.zeros_like(numerical).tolist()}
        count = len(self.layout.channels)
        gains = tuple(Fraction.from_float(float(value)) for value in self.declaration["row_gains"])
        q = [[Fraction.from_float(float(value)) for value in row] for row in self.support_basis]
        nominal = [[gains[i] * (Fraction(int(i == j)) - Fraction(1, count))
                    for j in range(count)] for i in range(count)]
        stored = [[Fraction.from_float(float(value)) for value in row] for row in self.matrix]
        exact, bounds = [], []
        for column in range(self.support_basis.shape[1]):
            values, row_bounds = [], []
            for j in range(count):
                ideal = sum((q[i][column] * nominal[i][j] for i in range(count)), Fraction(0))
                defect = sum((abs(q[i][column]) * abs(stored[i][j] - nominal[i][j])
                              for i in range(count)), Fraction(0))
                magnitude = sum((abs(q[i][column] * stored[i][j]) for i in range(count)), Fraction(0))
                bound = defect + Fraction.from_float(_gamma(count)) * magnitude
                actual = Fraction.from_float(float(numerical[column, j]))
                if abs(actual - ideal) > bound:
                    raise SpatialPriorIdentificationError("declared_CAR_effect_generation_unresolved",
                        support_coordinate=column, native_electrode=j)
                values.append(ideal)
                row_bounds.append(float(bound))
            exact.append(tuple(values))
            bounds.append(row_bounds)
        return tuple(exact), {"authority": "declared_Q_transpose_gain_exact_average_reference",
                              "generation_rounding_bound": bounds}

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
        _validate_source_learning_semantics(self.source)
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

    def _c_energy_roots(self, blocks, noise):
        blocks = tuple(blocks)
        if not blocks:
            raise ValueError("energy matching requires actual authorized C")
        roots, receipts, supports = [], [], {}
        for block in blocks:
            if not isinstance(block, SpatialResponseBlock):
                raise TypeError("signal energy requires owned spatial response blocks")
            authority = block.authorization
            if (authority["fit_scope"] not in {"target_calibration", "source_training"}
                    or block.observation.layout.record() != self.layout.record()
                    or block.design.shape[1] != self.mean.shape[0]
                    or authority["dictionary_sha256"] != self.design_binding["dictionary_sha256"]):
                raise ValueError("energy matching consumes only the bound C/design/layout")
            start, stop = authority["raw_sample_support"]
            source = authority["source_sha256"]
            if any(max(start, a) < min(stop, b) for a, b in supports.get(source, ())):
                raise ValueError("C energy matching cannot duplicate physical observation rows")
            supports.setdefault(source, []).append((start, stop))
            root = block._compressed(noise)[0]
            roots.append(root)
            receipts.append({"actual_C": block.record(), "root_sha256": array_digest(root)})
        if len({b.authorization["fit_scope"] for b in blocks}) != 1:
            raise ValueError("one energy program cannot mix source training and target calibration")
        return tuple(roots), receipts

    def _root_energy(self, roots):
        factor = solve_triangular(self.precision_root, np.eye(self.mean.size), lower=True)
        return float(sum(np.sum((root @ factor) ** 2) for root in roots))

    def expected_whitened_signal_energy(self, blocks, *, native_noise_covariance):
        """Centered prior signal energy on actual C, with its nuisance removed.

        The bound physical prior and frozen residual noise pass through the
        same actual observation A. Sample amplitudes do not select this prior
        energy; the existing response owner supplies its likelihood roots.
        """
        roots, _ = self._c_energy_roots(blocks, native_noise_covariance)
        return self._root_energy(roots)

    def energy_matched_graph(self, blocks, *, native_noise_covariance,
                             length_scale_m, graph_strength, source):
        """Fixed graph shape with C-plugin variance matched to this prior.

        Keep actual H/basis, mean, native layout and noise. Match centered
        expected whitened signal energy, not effective response df/capacity.
        Reference/gain/missing maps subsequently push the SAME S and Sigma;
        rebuilding a graph or rematching its projected energy is another
        program. Graph shape is declared externally, never learned here.
        """
        roots, receipts = self._c_energy_roots(blocks, native_noise_covariance)
        variance = float(np.trace(self.signal_covariance) / len(self.layout.channels))
        arguments = {"layout": self.layout, "mean": self.mean,
                     "temporal_covariance": self.temporal_covariance,
                     "length_scale_m": length_scale_m, "graph_strength": graph_strength,
                     "design_binding": dict(self.design_binding), "source": dict(source)}
        unit_graph = self.graph(variance=variance, **arguments)
        target, initial = self._root_energy(roots), unit_graph._root_energy(roots)
        if not np.isfinite([target, initial]).all() or target <= 0 or initial <= 0:
            raise ValueError("C/noise prior energy must be finite and nonzero; no fallback strength")
        multiplier = target / initial
        receipt = {"schema": "n2p3_B09_graph_C_energy_match/1",
                   "shape_role": "external_predeclared_layout_graph",
                   "variance_role": "C_derived_plugin_reference",
                   "centered_signal_energy": True, "effective_response_df_matched": False,
                   "noise_sha256": array_digest(native_noise_covariance),
                   "reference_prior_sha256": self.digest(), "actual_C": receipts,
                   "target_energy": target, "unmatched_graph_energy": initial,
                   "variance_multiplier": multiplier,
                   "reference_native_variance": variance,
                   "matched_native_variance": variance * multiplier,
                   "projected_graph_rebuilt": False, "projected_energy_rematched": False,
                   "noise_refitted": False, "shape_learned_from_C": False,
                   "held_data_or_labels_used": False}
        matched = self.graph(variance=variance * multiplier, **arguments)
        actual = matched._root_energy(roots)
        if not np.isclose(actual, target, rtol=1e-12, atol=0):
            raise ArithmeticError("same-C/noise graph energy failed its declared match")
        receipt["actual_matched_energy"] = actual
        receipt["relative_energy_error"] = abs(actual / target - 1)
        receipt["match_binding_sha256"] = metadata_digest(receipt)
        # Publish the complete receipt once through the existing frozen owner.
        matched = replace(matched, source={**matched.source, "C_plugin_energy_matching": receipt})
        return matched, receipt

    @classmethod
    def learn(cls, *, mechanism, estimates, policy, source_authority):
        """Learn a source population program from owned physical measurements."""
        if mechanism != "source_group":
            raise ValueError("unknown source-learning mechanism")
        return SourceGroupPrior.fit(estimates, policy=policy, source_authority=source_authority)

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
    _qr_column_relation(normalized_stack, q, normalized_root)
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
        _exact_covariance_relation(root, covariance)
    elif np.any(np.abs(gram - np.eye(width)) > tolerance + gram_bound):
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
    structural_operators: tuple[np.ndarray, ...] = field(repr=False, compare=False, default=())
    structural_null_basis: tuple[tuple[Fraction, ...], ...] = field(repr=False, compare=False, default=())
    structural_binding: dict = field(default_factory=dict)
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
            learning = self.prior.source.get("source_learning")
            if learning is not None:
                participant = authorization.get("original_view", {}).get("participant_key")
                if (authorization["fit_scope"] != "target_calibration"
                        or participant != learning["target_participant_key"]
                        or participant in learning["source_participant_keys"]):
                    raise ValueError("source-learned prior requires its independent declared target C")
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
        operators = tuple(_matrix(operator, "saved structural C operator") for operator in self.structural_operators)
        if len(operators) != len(roots):
            raise ValueError("posterior needs one exact structural C operator per actual fit record")
        null_basis, structural_resolution = _exact_structural_null(operators, width)
        saved_basis = tuple(tuple(Fraction(value) for value in vector) for vector in self.structural_null_basis)
        if saved_basis != null_basis:
            raise ValueError("saved structural C null basis differs from its exact actual design relation")
        if (not isinstance(self.structural_binding, dict)
                or self.structural_binding.get("operator_sha256") != _structural_operator_digest(operators)
                or self.structural_binding.get("arithmetic") != "exact_binary64_fraction_rref"
                or self.structural_binding.get("resolution") != structural_resolution
                or tuple(self.structural_binding.get("observation_coordinates", ())) != tuple(
                    "declared_CAR_exact_EEG_differences" if record["observation"]["declaration"]["kind"] == "average_actual_EEG"
                    else "actual_normalized_observation" for record in self.fit_records)):
            raise ValueError("saved structural C identifiability binding differs from actual operators")
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
        object.__setattr__(self, "structural_operators", tuple(frozen_array(operator) for operator in operators))
        object.__setattr__(self, "structural_null_basis", null_basis)
        object.__setattr__(self, "structural_binding", freeze_metadata(self.structural_binding))
        object.__setattr__(self, "fit_records", tuple(freeze_metadata(r) for r in self.fit_records))
        object.__setattr__(self, "noise_binding", freeze_metadata(self.noise_binding))

    @classmethod
    def fit(cls, prior, blocks, *, native_noise_covariance, noise_binding):
        noise, native_root = _spd(native_noise_covariance, "native residual noise")
        roots, responses, records, structural_operators = [], [], [], []
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
            mapping = block.observation._structural_matrix()
            operator = np.kron(mapping, block.design)
            nuisance = (np.kron(np.eye(mapping.shape[0]), block.nuisance)
                        if block.nuisance is not None else np.empty((len(operator), 0)))
            structural_operators.append(np.column_stack((operator, nuisance)))
        null_basis, structural_resolution = _exact_structural_null(structural_operators, prior.mean.size)
        structural_binding = {"operator_sha256": _structural_operator_digest(structural_operators),
            "arithmetic": "exact_binary64_fraction_rref", "resolution": structural_resolution,
            "observation_coordinates": ["declared_CAR_exact_EEG_differences"
                if record["observation"]["declaration"]["kind"] == "average_actual_EEG"
                else "actual_normalized_observation" for record in records]}
        return cls(prior, noise, tuple(roots), tuple(responses), tuple(records), noise_binding,
                   tuple(structural_operators), null_basis, structural_binding)

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

    def estimate_effect(self, effect, dependence, *, temporal_noise):
        """Estimate an identifiable physical functional from joint actual C."""
        return ERPChannelEstimate.from_posterior(self, effect=effect, dependence=dependence,
                                                temporal_noise=temporal_noise)

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
                 "structural_operators": [operator.tolist() for operator in self.structural_operators],
                 "structural_null_basis": [[[value.numerator, value.denominator] for value in vector]
                                            for vector in self.structural_null_basis],
                 "structural_binding": json.loads(json.dumps(self.structural_binding)),
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
        structural_basis = tuple(tuple(Fraction(numerator, denominator) for numerator, denominator in vector)
                                 for vector in payload["structural_null_basis"])
        model = cls(SpatialSignalPrior.from_record(payload["prior"]), payload["native_noise_covariance"],
                    tuple(payload["information_roots"]), tuple(payload["responses"]),
                    tuple(payload["fit_records"]), payload["noise_binding"],
                    tuple(payload["structural_operators"]), structural_basis,
                    payload["structural_binding"], restore_state=state)
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


@dataclass(frozen=True)
class ERPPhysicalEffect:
    """Explicit physical window and actual observed spatial functionals."""

    layout: SpatialLayout
    design_binding: dict
    basis: np.ndarray = field(repr=False, compare=False)
    waveform_weights: np.ndarray = field(repr=False, compare=False)
    grid_seconds: np.ndarray = field(repr=False, compare=False)
    observation: SpatialObservation
    response_functional: np.ndarray = field(init=False, repr=False, compare=False)
    spatial_map: np.ndarray = field(init=False, repr=False, compare=False)
    functional: np.ndarray = field(init=False, repr=False, compare=False)
    structural_spatial_map: tuple[tuple[Fraction, ...], ...] = field(init=False, repr=False, compare=False)
    structural_functional: tuple[tuple[Fraction, ...], ...] = field(init=False, repr=False, compare=False)
    definition: dict = field(init=False)

    def __post_init__(self):
        basis = _matrix(self.basis, "effect actual physical basis")
        weights = finite_real_array(self.waveform_weights, name="effect window weights")
        grid = finite_real_array(self.grid_seconds, name="effect physical time grid")
        binding = self.design_binding
        rate, operator = binding.get("sample_rate_hz"), binding.get("operator_identity")
        if (isinstance(rate, bool) or not np.isscalar(rate) or not np.isfinite(rate) or rate <= 0
                or not isinstance(operator, str) or not operator.strip()):
            raise SpatialPriorIdentificationError("effect_requires_explicit_rate_operator")
        if (weights.shape != (len(basis),) or grid.shape != weights.shape
                or not np.any(weights != 0)
                or (len(grid) > 1 and not np.allclose(np.diff(grid), 1 / rate, rtol=1e-12, atol=0))
                or binding.get("actual_basis_sha256") != array_digest(basis)
                or tuple(binding.get("coefficient_groups", ())) != ("common", "target_increment")
                or not isinstance(self.observation, SpatialObservation)
                or self.observation.layout.record() != self.layout.record()):
            raise ValueError("effect basis/window/grid/observation differs from its actual design")
        require_sha256(binding.get("dictionary_sha256"), "effect dictionary identity")
        f = np.r_[np.zeros(basis.shape[1]), weights @ basis]
        mapping = self.observation.support_basis.T @ self.observation.matrix
        functional = np.kron(mapping, f[None, :])
        exact_mapping, map_authority = self.observation._exact_effect_map()
        exact_f = tuple(Fraction.from_float(float(value)) for value in f)
        exact_functional = tuple(tuple(a * value for a in row for value in exact_f) for row in exact_mapping)
        # Participant-specific layout authority and noise are measurement law;
        # actual electrodes/reference/units and A are the common effect chart.
        physical_layout = {key: self.layout.record()[key] for key in
                           ("channels", "channel_types", "acquisition_reference", "signal_unit")}
        definition = {"schema": "n2p3_physical_ERP_effect/2", "layout": physical_layout,
            "dictionary_sha256": binding["dictionary_sha256"], "actual_basis_sha256": array_digest(basis),
            "waveform_weights": weights.tolist(), "grid_seconds": grid.tolist(),
            "sample_rate_hz": float(rate), "operator_identity": operator,
            "coefficient_groups": ["common", "target_increment"],
            "vectorization": "column_major_response_by_native_EEG",
            "output_names": list(self.observation.output_names),
            "output_reference": self.observation.output_reference,
            "spatial_map": mapping.tolist(), "observed_rank": len(mapping),
            "structural_spatial_map": [[str(value) for value in row] for row in exact_mapping],
            "structural_map_generation": map_authority}
        for name, value in (("basis", basis), ("waveform_weights", weights), ("grid_seconds", grid),
                            ("response_functional", f), ("spatial_map", mapping), ("functional", functional)):
            object.__setattr__(self, name, frozen_array(value))
        object.__setattr__(self, "design_binding", freeze_metadata(binding))
        object.__setattr__(self, "definition", freeze_metadata(definition))
        object.__setattr__(self, "structural_spatial_map", exact_mapping)
        object.__setattr__(self, "structural_functional", exact_functional)

    @classmethod
    def from_prior(cls, prior, *, basis, waveform_weights, grid_seconds, observation):
        if not isinstance(prior, SpatialSignalPrior):
            raise TypeError("effect definition requires an owned physical prior")
        effect = cls(prior.layout, dict(prior.design_binding), basis, waveform_weights,
                     grid_seconds, observation)
        effect.validate_prior(prior)
        return effect

    def validate_prior(self, prior):
        if (prior.layout.record() != self.layout.record()
                or prior.mean.shape[0] != len(self.response_functional)
                or any(prior.design_binding.get(key) != self.design_binding.get(key) for key in
                       ("dictionary_sha256", "actual_basis_sha256", "sample_rate_hz", "operator_identity"))):
            raise ValueError("physical effect differs from the actual prior/design destination")
        if tuple(prior.design_binding.get("coefficient_groups", ())) != tuple(self.design_binding["coefficient_groups"]):
            raise ValueError("physical effect differs from the actual prior/design coefficient groups")

    def record(self):
        payload = {"schema": "n2p3_physical_ERP_effect_state/2", "layout": self.layout.record(),
            "design_binding": dict(self.design_binding), "basis": self.basis.tolist(),
            "waveform_weights": self.waveform_weights.tolist(), "grid_seconds": self.grid_seconds.tolist(),
            "observation": self.observation.record(), "definition": dict(self.definition)}
        return {**payload, "artifact_sha256": metadata_digest(payload)}

    @classmethod
    def from_record(cls, record):
        payload = json.loads(json.dumps(record, allow_nan=False))
        digest = payload.pop("artifact_sha256", None)
        if payload.get("schema") != "n2p3_physical_ERP_effect_state/2" or metadata_digest(payload) != digest:
            raise ValueError("physical effect state identity mismatch")
        layout = SpatialLayout.from_record(payload["layout"])
        observed = payload["observation"]
        observation = SpatialObservation(layout, observed["matrix"], tuple(observed["output_names"]),
                                         observed["output_reference"], observed["declaration"])
        effect = cls(layout, payload["design_binding"], payload["basis"], payload["waveform_weights"],
                     payload["grid_seconds"], observation)
        if metadata_digest(effect.record()) != metadata_digest(record):
            raise ValueError("physical effect state differs from actual physical relations")
        return effect


@dataclass(frozen=True, init=False)
class ERPPhysicalTransport:
    """A reference-free destination functional embedded in actual source EEG.

    The source effect supplies its frozen temporal definition, not a measured
    native effect or a prior-based completion. The destination's covectors are
    placed in source coefficient coordinates; the real source likelihood must
    subsequently identify them. Separate acquisition references are independent
    unknown potentials, even when their provenance strings happen to match.
    """

    source_effect: ERPPhysicalEffect
    destination_effect: ERPPhysicalEffect
    local_effect: ERPPhysicalEffect
    channel_map: np.ndarray
    binding: dict
    local_structural_functional: tuple[tuple[Fraction, ...], ...]

    @classmethod
    def between(cls, source_effect, destination_effect):
        from data.channel import canonical_channel_name

        if not isinstance(source_effect, ERPPhysicalEffect) or not isinstance(destination_effect, ERPPhysicalEffect):
            raise TypeError("physical transport needs explicit source and destination effects")
        source, destination = source_effect, destination_effect
        temporal_fields = ("dictionary_sha256", "actual_basis_sha256", "sample_rate_hz",
                           "operator_identity", "coefficient_groups", "vectorization")
        if (any(source.definition[key] != destination.definition[key] for key in temporal_fields)
                or not np.array_equal(source.basis, destination.basis)
                or not np.array_equal(source.waveform_weights, destination.waveform_weights)
                or not np.array_equal(source.grid_seconds, destination.grid_seconds)):
            raise ValueError("source and destination temporal effect definitions differ")
        if source.layout.signal_unit != destination.layout.signal_unit:
            raise ValueError("physical transport requires the same explicit voltage unit")
        source_channels = tuple(map(canonical_channel_name, source.layout.channels))
        destination_channels = tuple(map(canonical_channel_name, destination.layout.channels))
        if (len(set(source_channels)) != len(source_channels)
                or len(set(destination_channels)) != len(destination_channels)):
            raise ValueError("physical transport needs unique shared canonical electrode identities")
        # An exact binary64 nonzero coefficient on the unknown acquisition
        # reference is information, however small. A tolerant sum would erase it.
        reference_coefficients = tuple(-sum(row, Fraction(0)) for row in destination.structural_spatial_map)
        if any(reference_coefficients):
            raise SpatialPriorIdentificationError("transport_reference_component_not_observed",
                exact_reference_coefficients=[str(value) for value in reference_coefficients],
                capability="reference_free_actual_electrode_functionals")
        channel_map = np.zeros((len(destination_channels), len(source_channels)))
        missing = []
        for index, channel in enumerate(destination_channels):
            if channel in source_channels:
                channel_map[index, source_channels.index(channel)] = 1.
            elif any(row[index] for row in destination.structural_spatial_map):
                missing.append(channel)
        if missing:
            raise SpatialPriorIdentificationError("transport_required_electrode_not_measured",
                required_missing_electrodes=missing, actual_source_electrodes=list(source_channels))
        local_map = destination.spatial_map @ channel_map
        # These are effect-support coordinates, not invented measured channels.
        names = tuple(f"physical_effect_support:{index}" for index in range(len(local_map)))
        local_observation = SpatialObservation(source.layout, local_map, names,
                                                destination.observation.output_reference)
        local = ERPPhysicalEffect(source.layout, dict(source.design_binding), source.basis,
                                  source.waveform_weights, source.grid_seconds, local_observation)
        exact_local_map = tuple(tuple(sum((row[j] * int(channel_map[j, i]) for j in range(len(row))), Fraction(0))
                                     for i in range(len(source_channels)))
                                for row in destination.structural_spatial_map)
        exact_f = tuple(Fraction.from_float(float(value)) for value in source.response_functional)
        exact_local_functional = tuple(tuple(a * value for a in row for value in exact_f) for row in exact_local_map)
        # Embedding is a selector/permutation: every used coefficient survives
        # exactly, including coefficients on channels absent from the other layout.
        if not np.array_equal(local.spatial_map @ channel_map.T, destination.spatial_map):
            raise SpatialPriorIdentificationError("transport_physical_embedding_unresolved")
        result = object.__new__(cls)
        for name, value in (("source_effect", source), ("destination_effect", destination), ("local_effect", local)):
            object.__setattr__(result, name, value)
        object.__setattr__(result, "channel_map", frozen_array(channel_map))
        object.__setattr__(result, "local_structural_functional", exact_local_functional)
        object.__setattr__(result, "binding", freeze_metadata({
            "physical_relation": "destination_electrode_covectors_embedded_in_actual_source",
            "source_canonical_electrodes": list(source_channels),
            "destination_canonical_electrodes": list(destination_channels),
            "exact_reference_coefficients": [str(value) for value in reference_coefficients],
            "local_structural_spatial_map": [[str(value) for value in row] for row in exact_local_map],
            "destination_structural_map_generation": dict(destination.definition["structural_map_generation"]),
            "reference_policy": "separate_unknown_acquisition_potentials_cancel_exactly",
            "observation_row_space_authority": "actual_source_joint_likelihood_structural_null_and_BLUE_witness",
            "posterior_or_dprime_transport_used": False}))
        return result

    def record(self):
        payload = {"schema": "n2p3_physical_ERP_transport/1", "source_effect": self.source_effect.record(),
            "destination_effect_artifact_sha256": self.destination_effect.record()["artifact_sha256"],
            "local_effect": self.local_effect.record(), "channel_map": self.channel_map.tolist(),
            "binding": dict(self.binding)}
        return {**payload, "artifact_sha256": metadata_digest(payload)}


def _likelihood_frame(posterior):
    return {"posterior_artifact_sha256": posterior.record()["artifact_sha256"],
        "coordinates": "block_then_column_major_observed_response_QR_rows",
        "blocks": [{"record_id": record["authorization"]["record_id"], "rows": len(root),
                    "root_response_sha256": array_digest(root, response)}
                   for root, response, record in zip(posterior.information_roots, posterior.responses,
                                                    posterior.fit_records, strict=True)]}


@dataclass(frozen=True)
class SpatialLikelihoodDependence:
    """External full error factor in the exact owned compressed-response frame."""

    frame: dict
    factor: np.ndarray = field(repr=False, compare=False)
    authority: dict

    def __post_init__(self):
        factor = _matrix(self.factor, "complete compressed error factor")
        rows = sum(block["rows"] for block in self.frame["blocks"])
        if (factor.shape != (rows, rows) or np.any(np.triu(factor, 1) != 0)
                or np.any(np.diag(factor) <= 0)):
            raise ValueError("dependency factor must be proper lower triangular in its actual full frame")
        singular = np.linalg.svd(factor / np.max(np.abs(factor), axis=1)[:, None], compute_uv=False)
        if singular[-1] <= 64 * _gamma(rows) * singular[0]:
            raise SpatialPriorIdentificationError("dependency_factor_numerically_unresolved")
        require_sha256(self.authority.get("authority_sha256"), "dependency law authority")
        if self.authority.get("law") not in {"independent_compressed_errors", "external_full_compressed_errors"}:
            raise ValueError("dependency law requires an explicit complete frame declaration")
        object.__setattr__(self, "factor", frozen_array(factor))
        object.__setattr__(self, "frame", freeze_metadata(self.frame))
        object.__setattr__(self, "authority", freeze_metadata(self.authority))

    @classmethod
    def from_factor(cls, posterior, factor, *, authority):
        if not isinstance(posterior, SpatialSignalPosterior):
            raise TypeError("dependency frame requires an owned spatial posterior")
        return cls(_likelihood_frame(posterior), factor, authority)

    @classmethod
    def independent(cls, posterior, *, authority_sha256):
        frame = _likelihood_frame(posterior)
        return cls(frame, np.eye(sum(b["rows"] for b in frame["blocks"])),
                   {"authority_sha256": authority_sha256, "law": "independent_compressed_errors"})

    def record(self):
        return {"frame": dict(self.frame), "factor": self.factor.tolist(), "authority": dict(self.authority)}


def _effect_measurement(posterior, effect, dependence, *, physical_transport=None):
    """Joint identifiable functional BLUE; proper prior is only its numeric chart."""
    structural_functional = effect.structural_functional
    if physical_transport is not None:
        if (not isinstance(physical_transport, ERPPhysicalTransport)
                or metadata_digest(physical_transport.local_effect.record()) != metadata_digest(effect.record())):
            raise ValueError("effect structural transport differs from its owned local functional")
        structural_functional = physical_transport.local_structural_functional
    structural_witness = []
    for basis in posterior.structural_null_basis:
        witness = []
        for row in structural_functional:
            value = sum((coefficient * coefficient_null
                         for coefficient, coefficient_null in zip(row, basis, strict=True)), Fraction(0))
            witness.append(value)
        structural_witness.append(witness)
    nonzero = [(index, row_index, value) for index, witness in enumerate(structural_witness)
               for row_index, value in enumerate(witness) if value]
    if nonzero:
        raise SpatialPriorIdentificationError("physical_effect_not_identified_by_structural_C",
            structural_null_witness=[{ "basis_index": index, "effect_row": row, "exact_value": str(value)}
                                     for index, row, value in nonzero],
            structural_resolution=posterior.structural_binding["resolution"])
    x = np.vstack(posterior.information_roots)
    y = np.concatenate(posterior.responses)
    x = solve_triangular(dependence.factor, x, lower=True)
    y = solve_triangular(dependence.factor, y, lower=True)
    chart = solve_triangular(posterior.prior.precision_root, np.eye(posterior.mean.size), lower=True)
    with np.errstate(over="ignore", invalid="ignore"):
        actual, functional = x @ chart, effect.functional @ chart
    if not np.isfinite(actual).all() or not np.isfinite(functional).all():
        raise SpatialPriorIdentificationError("joint_effect_physical_chart_not_representable")
    _, singular, right = np.linalg.svd(actual, full_matrices=False)
    threshold = 64 * _gamma(max(actual.shape)) * singular[0]
    rank = int(np.sum(singular > threshold))
    vectors = right[:rank].T
    outside = functional - (functional @ vectors) @ vectors.T
    row_norm = np.linalg.norm(functional, axis=1)
    relative_outside = np.divide(np.linalg.norm(outside, axis=1), row_norm,
                                 out=np.zeros_like(row_norm), where=row_norm > 0)
    if rank == 0 or not np.any(row_norm):
        mean, factor = np.zeros(len(functional)), np.empty((len(functional), 0))
        weights = np.zeros((len(functional), len(x)))
    else:
        reduced = actual @ vectors
        q, root = np.linalg.qr(reduced, mode="reduced")
        _qr_column_relation(reduced, q, root)
        _unit_information_relation(reduced, root)
        inverse = solve_triangular(root, np.eye(rank))
        factor = (functional @ vectors) @ inverse
        weights = factor @ q.T
        mean = factor @ (q.T @ y)
    # A numerical row-space projection can hide a small genuine null component
    # behind another coefficient. Certify the candidate witness F = W X in the
    # original physical coefficient units, separately for every entry. The
    # construction allowance uses the W row and original X column norms:
    # orthogonal factors/triangular solves can mix rows even when F_ij is zero.
    # An exactly zero X column has no allowance from other columns, regardless
    # of the size of its nonzero F component.
    reconstructed = weights @ x
    magnitude = np.abs(weights) @ np.abs(x)
    construction = _gamma(6 * (x.size + x.shape[1] ** 2)) * np.outer(
        np.linalg.norm(weights, axis=1), np.linalg.norm(x, axis=0))
    relation_bound = (construction + 4 * _gamma(len(x)) * magnitude
                      + np.finfo(float).eps * (np.abs(effect.functional) + np.abs(reconstructed)))
    relation_error = np.abs(effect.functional - reconstructed)
    physical_norm = np.linalg.norm(effect.functional, axis=1)
    relative_bound = np.divide(np.linalg.norm(relation_bound, axis=1), physical_norm,
        out=np.zeros_like(physical_norm), where=physical_norm > 0)
    if (not np.isfinite(relation_bound).all()
            or np.any(relation_error > relation_bound)):
        raise SpatialPriorIdentificationError("physical_effect_not_identified_by_joint_C",
            functional_relative_null=relative_outside.tolist(), joint_rank=rank,
            singular_values=singular.tolist(), resolution_threshold=float(threshold),
            functional_relation_error=relation_error.tolist(),
            functional_relation_rounding_bound=relation_bound.tolist())
    if np.any(relative_bound > _IDENTITY_RESOLUTION):
        raise SpatialPriorIdentificationError("physical_effect_relation_resolution_unresolved",
            functional_relative_error_bound=relative_bound.tolist(), identity_resolution=_IDENTITY_RESOLUTION)
    if not np.isfinite(mean).all() or not np.isfinite(factor).all():
        raise SpatialPriorIdentificationError("effect_measurement_not_representable")
    return mean, factor, {"joint_rank": rank, "coefficient_dimensions": x.shape[1],
        "functional_relative_null": relative_outside.tolist(), "singular_values": singular.tolist(),
        "functional_relative_error_bound": relative_bound.tolist(),
        "structural_null_basis_count": len(posterior.structural_null_basis),
        "resolution_threshold": float(threshold), "prior_information_used": False,
        "method": "joint_owned_likelihood_thin_QR_componentwise_physical_F_equals_WX"}


@dataclass(frozen=True, init=False)
class ERPChannelEstimate:
    """Factory-owned identifiable physical effect, independent of posterior shrinkage."""

    effect: ERPPhysicalEffect
    mean: np.ndarray
    covariance_factor: np.ndarray
    noise_sd: np.ndarray
    receipt: dict

    @classmethod
    def from_transported_posterior(cls, posterior, *, source_effect, destination_effect, dependence, temporal_noise):
        """Measure a common physical contrast directly in the actual source law.

        No native BLUE is required: it may contain unobserved reference modes.
        The local destination functional uses the existing joint-likelihood BLUE
        and full native noise factor, then receives its common effect identity.
        """
        source_effect.validate_prior(posterior.prior)
        source_kinds = {record["observation"]["declaration"]["kind"] for record in posterior.fit_records}
        if not source_kinds <= {"native_EEG", "measured_reference", "average_actual_EEG"}:
            raise SpatialPriorIdentificationError("transport_source_physical_structure_requires_declared_observation",
                actual_source_observation_kinds=sorted(source_kinds),
                capability="native_measured_reference_or_declared_CAR_source_likelihood")
        # The posterior's stored likelihood is immutable. Its observation
        # metadata must still prove this new physical capability at ingress;
        # a re-signed kind string cannot authorize arbitrary generic matrices.
        for record, operator, information in zip(posterior.fit_records, posterior.structural_operators,
                                                  posterior.information_roots, strict=True):
            saved = record["observation"]
            observation = SpatialObservation(SpatialLayout.from_record(saved["layout"]), saved["matrix"],
                tuple(saved["output_names"]), saved["output_reference"], dict(saved["declaration"]))
            if metadata_digest(observation.record()) != metadata_digest(saved):
                raise ValueError("source physical observation differs from its actual declared relations")
            mapping = observation._structural_matrix()
            samples, width, rank = record["samples"], record["response_dimensions"], len(mapping)
            nuisance_width = record["nuisance_dimensions"]
            if (operator.shape != (rank * samples, width * len(observation.layout.channels) + rank * nuisance_width)
                    or information.shape != (rank * min(samples, width), posterior.prior.mean.size)
                    or width != posterior.prior.mean.shape[0] or np.any(~np.isin(mapping, (-1., 0., 1.)))):
                raise ValueError("source physical observation differs from its actual structural operator dimensions")
            row, channel = np.argwhere(mapping != 0)[0]
            design = operator[row * samples:(row + 1) * samples, channel * width:(channel + 1) * width] / mapping[row, channel]
            nuisance = operator[:samples, posterior.prior.mean.size:posterior.prior.mean.size + nuisance_width]
            expected = np.column_stack((np.kron(mapping, design), np.kron(np.eye(rank), nuisance)))
            if not np.array_equal(operator, expected):
                raise ValueError("source physical observation differs from its actual structural operator relation")
        transport = ERPPhysicalTransport.between(source_effect, destination_effect)
        local = cls._from_owned_posterior(posterior, effect=transport.local_effect,
            dependence=dependence, temporal_noise=temporal_noise, physical_transport=transport)
        result = object.__new__(cls)
        object.__setattr__(result, "effect", destination_effect)
        for name in ("mean", "covariance_factor", "noise_sd"):
            object.__setattr__(result, name, getattr(local, name))
        object.__setattr__(result, "receipt", freeze_metadata({**local.receipt,
            "physical_transport": transport.record()}))
        return result

    @classmethod
    def from_posterior(cls, posterior, *, effect, dependence, temporal_noise):
        return cls._from_owned_posterior(posterior, effect=effect, dependence=dependence,
                                        temporal_noise=temporal_noise, physical_transport=None)

    @classmethod
    def _from_owned_posterior(cls, posterior, *, effect, dependence, temporal_noise, physical_transport):
        if (not isinstance(posterior, SpatialSignalPosterior) or not isinstance(effect, ERPPhysicalEffect)
                or not isinstance(dependence, SpatialLikelihoodDependence)):
            raise TypeError("effect estimation needs owned posterior/effect/dependency inputs")
        effect.validate_prior(posterior.prior)
        if metadata_digest(dependence.frame) != metadata_digest(_likelihood_frame(posterior)):
            raise ValueError("dependency factor belongs to different actual likelihood rows/frame")
        if (not isinstance(temporal_noise, TemporalNoise) or temporal_noise.innovation_variance != 1.
                or temporal_noise.startup_mode == "conditional_prefix"):
            raise SpatialPriorIdentificationError("effect_noise_functional_requires_complete_unit_innovation_map")
        records = posterior.fit_records
        scopes = {r["authorization"]["fit_scope"] for r in records}
        participants = {r["authorization"].get("original_view", {}).get("participant_key") for r in records}
        temporal_sha = temporal_noise.record()["artifact_sha256"]
        if (len(scopes) != 1 or len(participants) != 1 or None in participants
                or any(r["authorization"]["temporal_program_sha256"] != temporal_sha for r in records)):
            raise ValueError("effect C scope/participant or functional noise differs from its actual producer")
        mean, factor, resolution = _effect_measurement(posterior, effect, dependence, physical_transport=physical_transport)
        w = effect.waveform_weights
        inverse_white = np.linalg.solve(temporal_noise.whiten(np.eye(len(w))), np.eye(len(w)))
        temporal_variance = float(np.sum((w @ inverse_white) ** 2))
        noise_factor = effect.spatial_map @ posterior.native_noise_root
        noise_sd = np.sqrt(temporal_variance * np.sum(noise_factor ** 2, axis=1))
        if not np.isfinite(noise_sd).all() or np.any(noise_sd <= 0):
            raise SpatialPriorIdentificationError("effect_noise_functional_SD_unresolved")
        result = object.__new__(cls)
        object.__setattr__(result, "effect", effect)
        for name, value in (("mean", mean), ("covariance_factor", factor), ("noise_sd", noise_sd)):
            object.__setattr__(result, name, frozen_array(value))
        receipt = {"fit_scope": next(iter(scopes)), "participant_key": next(iter(participants)),
            "producer_implementation_sha256": IMPLEMENTATION_SHA256,
            "posterior_artifact_sha256": posterior.record()["artifact_sha256"],
            "actual_C": list(records), "dependency": dependence.record(), "resolution": resolution,
            "temporal_noise_sha256": temporal_sha,
            "native_noise_sha256": array_digest(posterior.native_noise_covariance),
            "noise_or_dependency_uncertainty_integrated": False, "posterior_mean_used_as_measurement": False}
        object.__setattr__(result, "receipt", freeze_metadata(receipt))
        return result

    @property
    def covariance(self):
        return self.covariance_factor @ self.covariance_factor.T

    @property
    def dprime_covariance(self):
        factor = self.covariance_factor / self.noise_sd[:, None]
        return factor @ factor.T

    def record(self):
        schema = "n2p3_ERP_effect_measurement/2" if "physical_transport" in self.receipt else "n2p3_ERP_effect_measurement/1"
        payload = {"schema": schema, "effect": self.effect.record(),
            "mean": self.mean.tolist(), "covariance_factor": self.covariance_factor.tolist(),
            "noise_sd": self.noise_sd.tolist(), "dprime_mean": (self.mean / self.noise_sd).tolist(),
            "receipt": dict(self.receipt)}
        return {**payload, "artifact_sha256": metadata_digest(payload)}

    def save(self, path):
        atomic_write_json(path, self.record())

    @classmethod
    def load(cls, path, *, posterior, temporal_noise, expected_artifact_sha256):
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = {key: value for key, value in saved.items() if key != "artifact_sha256"}
        if (payload.get("schema") not in {"n2p3_ERP_effect_measurement/1", "n2p3_ERP_effect_measurement/2"}
                or saved.get("artifact_sha256") != expected_artifact_sha256
                or metadata_digest(payload) != expected_artifact_sha256):
            raise ValueError("effect measurement differs from its external artifact identity")
        effect = ERPPhysicalEffect.from_record(saved["effect"])
        dependency = saved["receipt"]["dependency"]
        dependence = SpatialLikelihoodDependence(dependency["frame"], dependency["factor"], dependency["authority"])
        if payload["schema"] == "n2p3_ERP_effect_measurement/2":
            transport = saved["receipt"]["physical_transport"]
            source_effect = ERPPhysicalEffect.from_record(transport["source_effect"])
            actual = cls.from_transported_posterior(posterior, source_effect=source_effect,
                destination_effect=effect, dependence=dependence, temporal_noise=temporal_noise)
            if metadata_digest(transport) != metadata_digest(actual.receipt["physical_transport"]):
                raise ValueError("saved effect transport differs from its actual physical source relation")
        else:
            actual = cls.from_posterior(posterior, effect=effect, dependence=dependence, temporal_noise=temporal_noise)
        if (metadata_digest(actual.receipt) != metadata_digest(saved["receipt"])
                or not np.array_equal(actual.noise_sd, saved["noise_sd"])):
            raise ValueError("effect measurement differs from its actual likelihood producer")
        mean = finite_real_array(saved["mean"], name="producer effect mean")
        factor = finite_real_array(saved["covariance_factor"], name="producer effect covariance factor")
        if mean.shape != actual.mean.shape or factor.shape != actual.covariance_factor.shape:
            raise ValueError("producer effect dimensions differ from its actual functionals")
        if not np.array_equal(mean / actual.noise_sd, saved["dprime_mean"]):
            raise ValueError("saved physical d-prime differs from its effect mean and frozen noise SD")
        if factor.shape[1] == 0:
            if not np.array_equal(mean, actual.mean) or not np.array_equal(factor, actual.covariance_factor):
                raise ValueError("zero physical effect differs from its exact zero measurement")
            return actual
        _, upper = np.linalg.qr(factor.T, mode="reduced")
        _unit_information_relation(actual.covariance_factor.T, upper)
        _, covariance_root = _spd(actual.covariance, "resolved physical effect covariance")
        error = solve_triangular(covariance_root, mean - actual.mean, lower=True)
        scale = np.linalg.norm(solve_triangular(covariance_root, actual.mean, lower=True))
        if np.linalg.norm(error) > 256 * _gamma(len(dependence.factor)) * max(1., scale):
            raise ValueError("saved physical effect mean differs from its actual measurement")
        object.__setattr__(actual, "mean", frozen_array(mean))
        object.__setattr__(actual, "covariance_factor", frozen_array(factor))
        return actual


@dataclass(frozen=True)
class SourceGroupPolicy:
    """External physical templates, strong anchors and whole-program bounds."""

    template_names: tuple[str, ...]
    templates: np.ndarray = field(repr=False, compare=False)
    metric_covariance: np.ndarray = field(repr=False, compare=False)
    component_anchor: np.ndarray = field(repr=False, compare=False)
    component_ceiling: np.ndarray = field(repr=False, compare=False)
    component_anchor_df: float
    mean_anchor: np.ndarray = field(repr=False, compare=False)
    mean_anchor_covariance: np.ndarray = field(repr=False, compare=False)
    maximum_mean_metric_energy: float
    authority_sha256: str

    def __post_init__(self):
        require_identifiers(self.template_names, "source group templates")
        require_sha256(self.authority_sha256, "external group policy authority")
        metric, _ = _spd(self.metric_covariance, "external physical covariance-fit metric")
        mean_covariance, _ = _spd(self.mean_anchor_covariance, "external population mean anchor")
        templates = finite_real_array(self.templates, name="external group covariance templates")
        anchor = finite_real_array(self.component_anchor, name="external variance components")
        ceiling = finite_real_array(self.component_ceiling, name="external variance-component ceilings")
        mean = finite_real_array(self.mean_anchor, name="external population mean")
        width, count = len(metric), len(self.template_names)
        if (templates.shape != (count, width, width) or anchor.shape != (count,)
                or ceiling.shape != anchor.shape or mean.shape != (width,)
                or mean_covariance.shape != metric.shape or np.any(anchor < 0)
                or anchor[0] <= 0 or np.any(ceiling < anchor)):
            raise ValueError("external group policy dimensions/anchors/ceilings disagree")
        _spd(templates[0], "proper independent signal template")
        for template in templates:
            if (not np.array_equal(template, template.T)
                    or np.linalg.eigvalsh(template).min() < -64 * _gamma(width) * np.linalg.norm(template)):
                raise ValueError("signal variance components require positive semidefinite templates")
        for name, value in (("templates", templates), ("metric_covariance", metric),
                            ("component_anchor", anchor), ("component_ceiling", ceiling),
                            ("mean_anchor", mean), ("mean_anchor_covariance", mean_covariance)):
            object.__setattr__(self, name, frozen_array(value))
        object.__setattr__(self, "template_names", tuple(self.template_names))
        object.__setattr__(self, "component_anchor_df", _positive(self.component_anchor_df, "external anchor df"))
        object.__setattr__(self, "maximum_mean_metric_energy",
                           _positive(self.maximum_mean_metric_energy, "external mean-scale bound"))

    def record(self):
        return {"template_names": list(self.template_names), "templates": self.templates.tolist(),
            "metric_covariance": self.metric_covariance.tolist(), "component_anchor": self.component_anchor.tolist(),
            "component_ceiling": self.component_ceiling.tolist(), "component_anchor_df": self.component_anchor_df,
            "mean_anchor": self.mean_anchor.tolist(), "mean_anchor_covariance": self.mean_anchor_covariance.tolist(),
            "maximum_mean_metric_energy": self.maximum_mean_metric_energy, "authority_sha256": self.authority_sha256}


def _population_rows(measurements, covariances, between, policy):
    _, anchor_root = _spd(policy.mean_anchor_covariance, "source mean anchor")
    anchor_precision = solve_triangular(anchor_root, np.eye(len(between)), lower=True)
    rows, rhs = [anchor_precision], [anchor_precision @ policy.mean_anchor]
    for mean, covariance in zip(measurements, covariances, strict=True):
        _, factor = _spd(between + covariance, "between-person plus measurement covariance")
        precision = solve_triangular(factor, np.eye(len(between)), lower=True)
        rows.append(precision)
        rhs.append(precision @ mean)
    return np.vstack(rows), np.concatenate(rhs)


def _validate_source_authority(authority, ids):
    require_sha256(authority.get("authority_sha256"), "source participant authority")
    if (authority.get("fit_scope") != "source_training"
            or authority.get("independent_unit") != "participant"
            or tuple(authority.get("participant_keys", ())) != tuple(ids)):
        raise ValueError("source participant request/scope must match the complete actual source estimates")


def _validate_source_group_binding(binding, policy, means, errors, between, components):
    """Recompute source-program derived numbers and immutable semantic roles."""
    if (binding.get("whole_program_policy") != "external_actual_unit_bounds_no_person_drop_no_capping"
            or binding.get("variance_noise_dependency_uncertainty_integrated") is not False
            or binding.get("forecast") != "between_person_plus_conditional_source_mean_uncertainty"
            or binding.get("centered_energy_or_effective_df_matched") is not False):
        raise ValueError("saved source population semantic contract differs from its canonical program")
    if binding.get("component_fit_residual_resolution") != "64_gamma_columns_plus_8_eps_norm_scale":
        raise ValueError("saved source population residual resolution contract differs")
    count = len(means)
    alpha = (count - 1) / (count - 1 + policy.component_anchor_df)
    raw = finite_real_array(binding.get("raw_components"), name="saved raw source components")
    expected_components = (1 - alpha) * policy.component_anchor + alpha * raw
    expected_between = np.einsum("k,kij->ij", expected_components, policy.templates)
    if (not np.array_equal(float(binding.get("component_source_weight")), alpha)
            or not np.array_equal(components, expected_components)
            or not np.array_equal(between, expected_between)):
        raise ValueError("saved source population component weight/covariance differs from its actual program")
    _, metric_root = _spd(policy.metric_covariance, "saved source metric")
    inverse_metric = solve_triangular(metric_root, np.eye(len(metric_root)), lower=True)
    moment = np.cov(means, rowvar=False, ddof=1).reshape(errors.shape[1:]) - errors.mean(axis=0)
    columns = np.column_stack([(inverse_metric @ template @ inverse_metric.T).ravel()
                               for template in policy.templates])
    target = (inverse_metric @ moment @ inverse_metric.T).ravel()
    residual_vector = columns @ raw - target
    expected_residual = float(np.linalg.norm(residual_vector))
    residual_scale = float(np.linalg.norm(np.abs(columns) @ np.abs(raw)) + np.linalg.norm(np.abs(target)))
    residual_bound = (64 * _gamma(max(columns.shape)) * residual_scale
                      + 8 * np.finfo(float).eps * max(1., expected_residual))
    if abs(float(binding.get("component_fit_residual")) - expected_residual) > residual_bound:
        raise ValueError("saved source population NNLS residual differs from its actual moment relation")


@dataclass(frozen=True, init=False)
class SourceGroupPrior:
    """Source-only learned physical forecast, conditional on variance plug-ins."""

    effect: ERPPhysicalEffect
    policy: SourceGroupPolicy
    mean: np.ndarray
    between_covariance: np.ndarray
    mean_precision_root: np.ndarray
    components: np.ndarray
    measurements: np.ndarray
    measurement_covariances: np.ndarray
    fit_binding: dict

    @classmethod
    def fit(cls, estimates, *, policy, source_authority):
        estimates = tuple(estimates)
        if (len(estimates) < 2 or any(not isinstance(e, ERPChannelEstimate) for e in estimates)
                or not isinstance(policy, SourceGroupPolicy)):
            raise ValueError("source-group learning needs owned effects for at least two source participants")
        ids = tuple(e.receipt["participant_key"] for e in estimates)
        require_identifiers(ids, "independent source participants")
        _validate_source_authority(source_authority, ids)
        if any(e.receipt["fit_scope"] != "source_training" for e in estimates):
            raise ValueError("source participant request/scope must match the complete actual source estimates")
        effect = estimates[0].effect
        if any(metadata_digest(e.effect.definition) != metadata_digest(effect.definition) for e in estimates):
            raise ValueError("source estimates have different physical effect definitions; explicit transport required")
        supports = {}
        for estimate in estimates:
            for record in estimate.receipt["actual_C"]:
                authority = record["authorization"]
                start, stop = authority["raw_sample_support"]
                source = authority["source_sha256"]
                if any(max(start, a) < min(stop, b) for a, b in supports.get(source, ())):
                    raise ValueError("independent source participants duplicate actual physical C support")
                supports.setdefault(source, []).append((start, stop))
        means = np.asarray([e.mean for e in estimates])
        errors = np.asarray([e.covariance for e in estimates])
        if policy.metric_covariance.shape != errors.shape[1:]:
            raise ValueError("source group policy differs from the actual observed effect coordinates")
        moment = np.cov(means, rowvar=False, ddof=1).reshape(errors.shape[1:]) - errors.mean(axis=0)
        _, metric_root = _spd(policy.metric_covariance, "source component metric")
        inverse_metric = solve_triangular(metric_root, np.eye(len(moment)), lower=True)
        columns = np.column_stack([(inverse_metric @ q @ inverse_metric.T).ravel() for q in policy.templates])
        singular = np.linalg.svd(columns, compute_uv=False)
        if len(singular) < columns.shape[1] or singular[-1] <= 64 * _gamma(max(columns.shape)) * singular[0]:
            raise SpatialPriorIdentificationError("source_variance_templates_not_identified")
        raw_components, residual = nnls(columns, (inverse_metric @ moment @ inverse_metric.T).ravel())
        if np.any(raw_components > policy.component_ceiling):
            raise SpatialPriorIdentificationError("whole_source_program_exceeds_component_scale",
                participant_keys=list(ids), actual_components=raw_components.tolist(),
                external_ceilings=policy.component_ceiling.tolist())
        weight = (len(estimates) - 1) / (len(estimates) - 1 + policy.component_anchor_df)
        components = (1 - weight) * policy.component_anchor + weight * raw_components
        between = np.einsum("k,kij->ij", components, policy.templates)
        _spd(between, "proper learned between-person signal covariance")
        rows, rhs = _population_rows(means, errors, between, policy)
        scales = _column_scales(rows)
        q, normalized_root = np.linalg.qr(rows / scales, mode="reduced")
        _qr_column_relation(rows / scales, q, normalized_root)
        _unit_information_relation(rows / scales, normalized_root)
        mean = solve_triangular(normalized_root, q.T @ rhs) / scales
        root = normalized_root * scales
        mean_energy = float(np.sum((inverse_metric @ (mean - policy.mean_anchor)) ** 2))
        if mean_energy > policy.maximum_mean_metric_energy:
            raise SpatialPriorIdentificationError("whole_source_program_exceeds_mean_scale",
                participant_keys=list(ids), actual_metric_energy=mean_energy,
                external_maximum=policy.maximum_mean_metric_energy)
        binding = {"schema": "n2p3_source_group_fit/1", "fit_scope": "source_training",
            "source_authority": source_authority, "source_participant_keys": list(ids),
            "source_estimates": [{"artifact_sha256": e.record()["artifact_sha256"],
                "physical_moments_sha256": array_digest(e.mean, e.covariance),
                "effect_definition_sha256": metadata_digest(e.effect.definition),
                "fit_scope": e.receipt["fit_scope"], "participant_key": e.receipt["participant_key"],
                "posterior_artifact_sha256": e.receipt["posterior_artifact_sha256"],
                "dependency_frame_factor_sha256": metadata_digest(e.receipt["dependency"]),
                "producer_implementation_sha256": e.receipt["producer_implementation_sha256"]}
                for e in estimates],
            "effect_definition_sha256": metadata_digest(effect.definition),
            "producer_implementation_sha256": IMPLEMENTATION_SHA256,
            "raw_components": raw_components.tolist(), "component_source_weight": weight,
            "component_fit_residual": float(residual),
            "component_fit_residual_resolution": "64_gamma_columns_plus_8_eps_norm_scale",
            "whole_program_policy": "external_actual_unit_bounds_no_person_drop_no_capping",
            "variance_noise_dependency_uncertainty_integrated": False,
            "forecast": "between_person_plus_conditional_source_mean_uncertainty",
            "centered_energy_or_effective_df_matched": False}
        _validate_source_group_binding(binding, policy, means, errors, between, components)
        return cls._install(effect, policy, mean, between, root, components, means, errors, binding)

    @classmethod
    def _install(cls, effect, policy, mean, between, root, components, means, errors, binding):
        if not np.any(effect.response_functional != 0):
            raise SpatialPriorIdentificationError("zero_effect_has_no_proper_population_forecast")
        result = object.__new__(cls)
        object.__setattr__(result, "effect", effect)
        object.__setattr__(result, "policy", policy)
        for name, value in (("mean", mean), ("between_covariance", between), ("mean_precision_root", root),
                            ("components", components), ("measurements", means), ("measurement_covariances", errors)):
            object.__setattr__(result, name, frozen_array(value))
        object.__setattr__(result, "fit_binding", freeze_metadata(binding))
        return result

    @property
    def mean_covariance(self):
        inverse = solve_triangular(self.mean_precision_root, np.eye(len(self.mean)))
        return inverse @ inverse.T

    @property
    def forecast_covariance(self):
        return self.between_covariance + self.mean_covariance

    def spatial_prior(self, reference, *, effect, completion, target_participant_key):
        if not isinstance(reference, SpatialSignalPrior) or not isinstance(effect, ERPPhysicalEffect):
            raise TypeError("source completion requires owned prior and explicit destination effect")
        effect.validate_prior(reference)
        if metadata_digest(effect.definition) != metadata_digest(self.effect.definition):
            raise ValueError("source completion destination physical effect definition differs")
        if completion not in {"mean_only", "separable"}:
            raise ValueError("choose an explicit source mean-only or separable completion")
        if (not isinstance(target_participant_key, str) or not target_participant_key.strip()
                or target_participant_key in self.fit_binding["source_participant_keys"]):
            raise ValueError("source forecast requires an independent declared target participant")
        f, a, h, s = effect.response_functional, effect.spatial_map, reference.temporal_covariance, reference.signal_covariance
        variance = float(f @ h @ f)
        if variance <= 0:
            raise SpatialPriorIdentificationError("effect_has_no_proper_temporal_completion")
        _, native_factor = _spd(s, "reference spatial completion")
        mapped_factor = a @ native_factor
        _, upper = np.linalg.qr(mapped_factor.T, mode="reduced")
        upper = np.sign(np.diag(upper))[:, None] * upper
        _unit_information_relation(mapped_factor.T, upper)
        covariance_inverse = solve_triangular(upper, solve_triangular(upper.T, np.eye(len(a)), lower=True))
        spatial_lift = s @ a.T @ covariance_inverse
        temporal_lift = h @ f / variance
        mean = reference.mean + np.outer(temporal_lift, spatial_lift @ (self.mean - a @ (f @ reference.mean)))
        signal = s
        if completion == "separable":
            _, forecast_factor = _spd(self.forecast_covariance / variance, "new-person effect forecast")
            conditional_factor = (np.eye(len(s)) - spatial_lift @ a) @ native_factor
            factor = np.column_stack((conditional_factor, spatial_lift @ forecast_factor))
            signal = factor @ factor.T
        source = {"role": "source_training", "authority_sha256": self.record()["artifact_sha256"],
            "source_learning": {"mechanism": "source_group", "completion": completion,
                "source_participant_keys": list(self.fit_binding["source_participant_keys"]),
                "target_participant_key": target_participant_key,
                "effect_definition_sha256": metadata_digest(effect.definition),
                "variance_uncertainty_integrated": False,
                "unseen_spatial_directions": "proper_reference_conditional_completion",
                "source_forecast_uncertainty_propagated": completion == "separable"}}
        return SpatialSignalPrior(reference.layout, mean, h, signal, dict(reference.design_binding), source)

    def record(self):
        payload = {"schema": "n2p3_source_group_prior/1", "effect": self.effect.record(),
            "policy": self.policy.record(), "mean": self.mean.tolist(),
            "between_covariance": self.between_covariance.tolist(),
            "mean_precision_root": self.mean_precision_root.tolist(), "components": self.components.tolist(),
            "measurements": self.measurements.tolist(), "measurement_covariances": self.measurement_covariances.tolist(),
            "fit_binding": dict(self.fit_binding)}
        return {**payload, "artifact_sha256": metadata_digest(payload)}

    def save(self, path):
        atomic_write_json(path, self.record())

    @classmethod
    def load(cls, path, *, expected_artifact_sha256):
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = {key: value for key, value in saved.items() if key != "artifact_sha256"}
        if (payload.get("schema") != "n2p3_source_group_prior/1"
                or saved.get("artifact_sha256") != expected_artifact_sha256
                or metadata_digest(payload) != expected_artifact_sha256):
            raise ValueError("source population prior differs from its external artifact identity")
        effect = ERPPhysicalEffect.from_record(saved["effect"])
        policy = SourceGroupPolicy(**saved["policy"])
        mean, between, root, components, means, errors = (finite_real_array(saved[key], name=key) for key in
            ("mean", "between_covariance", "mean_precision_root", "components", "measurements", "measurement_covariances"))
        binding = saved["fit_binding"]
        ids, count = binding["source_participant_keys"], len(means)
        require_identifiers(ids, "saved source participants")
        _validate_source_authority(binding["source_authority"], ids)
        if (count < 2 or len(ids) != count or binding["fit_scope"] != "source_training"
                or binding["source_authority"]["participant_keys"] != ids
                or any(e["fit_scope"] != "source_training"
                       or e["participant_key"] != identity for e, identity in
                       zip(binding["source_estimates"], ids, strict=True))):
            raise ValueError("saved source population producer scope/participant relationship differs")
        if (metadata_digest(effect.definition) != binding["effect_definition_sha256"]
                or components.shape != policy.component_anchor.shape or mean.shape != policy.mean_anchor.shape
                or root.shape != policy.metric_covariance.shape or np.any(np.tril(root, -1) != 0)
                or np.any(np.diag(root) == 0)):
            raise ValueError("saved population physical definition or numerical dimensions differ")
        _validate_source_group_binding(binding, policy, means, errors, between, components)
        weight = (count - 1) / (count - 1 + policy.component_anchor_df)
        actual_components = (1 - weight) * policy.component_anchor + weight * np.asarray(binding["raw_components"])
        if (not np.array_equal(components, actual_components)
                or not np.array_equal(between, np.einsum("k,kij->ij", components, policy.templates))
                or np.any(np.asarray(binding["raw_components"]) > policy.component_ceiling)
                or any(e["physical_moments_sha256"] != array_digest(m, v)
                       or e["effect_definition_sha256"] != metadata_digest(effect.definition)
                       for e, m, v in zip(binding["source_estimates"], means, errors, strict=True))):
            raise ValueError("saved source population covariance differs from its actual source summaries")
        moment = np.cov(means, rowvar=False, ddof=1).reshape(errors.shape[1:]) - errors.mean(axis=0)
        _, metric_root = _spd(policy.metric_covariance, "saved source metric")
        inverse_metric = solve_triangular(metric_root, np.eye(len(moment)), lower=True)
        columns = np.column_stack([(inverse_metric @ q @ inverse_metric.T).ravel() for q in policy.templates])
        raw = np.asarray(binding["raw_components"])
        residual = columns @ raw - (inverse_metric @ moment @ inverse_metric.T).ravel()
        gradient = columns.T @ residual
        bound = 256 * _gamma(len(columns)) * (np.abs(columns).T @ (
            np.abs(columns) @ np.abs(raw) + np.abs((inverse_metric @ moment @ inverse_metric.T).ravel())))
        if np.any(raw < 0) or np.any(gradient < -bound) or np.any(np.abs(raw * gradient) > np.abs(raw) * bound):
            raise ValueError("saved source components violate their actual moment NNLS relationship")
        rows, rhs = _population_rows(means, errors, between, policy)
        _unit_information_relation(rows, root)
        residual = rows.T @ (rows @ mean - rhs)
        budget = 256 * _gamma(len(rows)) * (np.abs(rows).T @ (np.abs(rows) @ np.abs(mean) + np.abs(rhs)))
        if np.any(np.abs(residual) > budget):
            raise ValueError("saved population mean violates actual source information")
        if float(np.sum((inverse_metric @ (mean - policy.mean_anchor)) ** 2)) > policy.maximum_mean_metric_energy:
            raise ValueError("saved source mean exceeds its external whole-program scale")
        return cls._install(effect, policy, mean, between, root, components, means, errors, binding)
