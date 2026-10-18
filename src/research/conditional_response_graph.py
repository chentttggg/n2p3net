"""Exact named Gaussian completion; binary64 arrays remain original operands.

No rounded projected source factor or conditional-gain product defines this law.
The consumer requests the actual linear functional before numerical compilation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from transfer.noise_model import array_digest, freeze_metadata, frozen_array, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def exact(value):
    return value if isinstance(value, Fraction) else Fraction.from_float(float(value))


def rows(matrix):
    return tuple(tuple(exact(v) for v in row) for row in matrix)


def transpose(matrix):
    return tuple(zip(*matrix, strict=True))


def product(left, right):
    columns = transpose(right)
    result = []
    for row in left:
        support = [(i, value) for i, value in enumerate(row) if value]
        result.append(tuple(sum((value * exact(column[i]) for i, value in support), Fraction())
                            for column in columns))
    return tuple(result)


def inverse(matrix):
    width = len(matrix)
    work = [list(row) + [Fraction(i == j) for j in range(width)] for i, row in enumerate(matrix)]
    for k in range(width):
        pivot = next((i for i in range(k, width) if work[i][k]), None)
        if pivot is None:
            raise ValueError("exact conditional covariance is singular")
        work[k], work[pivot] = work[pivot], work[k]
        scale = work[k][k]
        work[k] = [value / scale for value in work[k]]
        for i in range(width):
            if i != k and work[i][k]:
                scale = work[i][k]
                work[i] = [a - scale * b for a, b in zip(work[i], work[k], strict=True)]
    return tuple(tuple(row[width:]) for row in work)


def proper(matrix):
    width = len(matrix)
    if any(len(row) != width for row in matrix):
        raise ValueError("proper covariance must be square")
    if any(matrix[i][j] != matrix[j][i] for i in range(width) for j in range(width)):
        raise ValueError("proper covariance must be exactly symmetric")
    work = [list(row) for row in matrix]
    for k in range(width):
        pivot = work[k][k]
        if pivot <= 0:
            raise ValueError("reference conditional law must be proper in exact saved-operand arithmetic")
        for i in range(k + 1, width):
            for j in range(k + 1, width):
                work[i][j] -= work[i][k] * work[k][j] / pivot


def conditional(covariance, common):
    proper(covariance)
    width = len(covariance)
    cc = tuple(tuple(covariance[i][j] for j in range(common)) for i in range(common))
    uc = tuple(tuple(covariance[i][j] for j in range(common)) for i in range(common, width))
    gain = product(uc, inverse(cc)) if uc else ()
    correction = product(gain, transpose(uc)) if uc else ()
    residual = tuple(tuple(covariance[i + common][j + common] - correction[i][j]
                           for j in range(width - common)) for i in range(width - common))
    if residual:
        proper(residual)
    return gain, residual


def kronecker(left, right):
    return tuple(tuple(a * b for a in left_row for b in right_row)
                 for left_row in left for right_row in right)


def identity(width):
    return tuple(tuple(Fraction(i == j) for j in range(width)) for i in range(width))


def string_rows(matrix):
    return [[str(value) for value in row] for row in matrix]


@dataclass(frozen=True)
class ExactGaussianImage:
    mean: tuple[Fraction, ...]
    covariance: tuple[tuple[Fraction, ...], ...]

    def record(self):
        return {"schema": "n2p3_exact_gaussian_linear_image/1", "mean": [str(v) for v in self.mean],
                "covariance": string_rows(self.covariance),
                "arithmetic": "exact_rational_moments_no_binary64_materialization"}


@dataclass(frozen=True)
class ExactConditionalResponseGraph:
    source_mean: np.ndarray = field(repr=False, compare=False)
    source_factor: np.ndarray = field(repr=False, compare=False)
    source_map: tuple = field(repr=False)
    reference_chart_mean: tuple = field(repr=False)
    conditional_gain: tuple = field(repr=False)
    conditional_covariance: tuple = field(repr=False)
    chart: object
    coefficient_count: int
    binding: dict

    def __post_init__(self):
        object.__setattr__(self, "source_mean", frozen_array(self.source_mean))
        object.__setattr__(self, "source_factor", frozen_array(self.source_factor))
        for name in ("source_map", "conditional_gain", "conditional_covariance"):
            object.__setattr__(self, name, rows(getattr(self, name)))
        object.__setattr__(self, "reference_chart_mean", tuple(exact(v) for v in self.reference_chart_mean))
        object.__setattr__(self, "binding", freeze_metadata(self.binding))
        if (self.source_mean.ndim != 1 or self.source_factor.ndim != 2
                or self.source_factor.shape[0] != len(self.source_mean)
                or not np.isfinite(self.source_mean).all() or not np.isfinite(self.source_factor).all()):
            raise ValueError("graph requires the finite original source Gaussian operands")
        common, rest = len(self.common_rows), len(self.completion_rows)
        if (len(self.source_map) != common or any(len(row) != len(self.source_mean) for row in self.source_map)
                or len(self.reference_chart_mean) != self.dimension
                or len(self.conditional_gain) != rest or any(len(row) != common for row in self.conditional_gain)
                or len(self.conditional_covariance) != rest
                or any(len(row) != rest for row in self.conditional_covariance)):
            raise ValueError("exact graph dimensions differ from the coefficientwise reversible chart")

    @property
    def dimension(self):
        return self.coefficient_count * self.chart.native_dimension

    @property
    def common_rows(self):
        return tuple(i * self.chart.native_dimension + j for i in range(self.coefficient_count)
                     for j in range(self.chart.common_dimension))

    @property
    def completion_rows(self):
        common = set(self.common_rows)
        return tuple(i for i in range(self.dimension) if i not in common)

    def exact_linear_image(self, functionals):
        """Exact moments of physical functionals in the declared chart.

        Supply Fraction for rational mathematical coefficients such as 1/3;
        a supplied float retains its exact stored binary64 meaning.
        """
        query = rows(functionals)
        if not query or any(len(row) != self.dimension for row in query):
            raise ValueError("linear image needs nonempty rows in the complete response chart")
        common, rest = self.common_rows, self.completion_rows
        w_common = tuple(tuple(row[i] for i in common) for row in query)
        w_rest = tuple(tuple(row[i] for i in rest) for row in query)
        if rest:
            adjustment = product(w_rest, self.conditional_gain)
            effective = tuple(tuple(a + b for a, b in zip(row, extra, strict=True))
                              for row, extra in zip(w_common, adjustment, strict=True))
        else:
            effective = w_common
        source_weights = product(effective, self.source_map)
        source_image = product(source_weights, self.source_factor)
        means = [sum((weight * exact(value) for weight, value in zip(row, self.source_mean, strict=True)), Fraction())
                 for row in source_weights]
        covariance = product(source_image, transpose(source_image))
        if rest:
            z0 = tuple((self.reference_chart_mean[i],) for i in common)
            adjustment = product(self.conditional_gain, z0)
            intercept = tuple((self.reference_chart_mean[i] - adjustment[j][0],) for j, i in enumerate(rest))
            shift = product(w_rest, intercept)
            residual = product(product(w_rest, self.conditional_covariance), transpose(w_rest))
            means = [value + shift[i][0] for i, value in enumerate(means)]
            covariance = tuple(tuple(value + residual[i][j] for j, value in enumerate(row))
                               for i, row in enumerate(covariance))
        return ExactGaussianImage(tuple(means), covariance)

    def record(self):
        return {"schema": "n2p3_exact_conditional_response_graph/1", "binding": self.binding,
                "chart": self.chart.record(), "coefficient_count": self.coefficient_count,
                "original_source_mean_factor_sha256": array_digest(self.source_mean, self.source_factor),
                "exact_source_map_sha256": metadata_digest(string_rows(self.source_map)),
                "reference_chart_mean_sha256": metadata_digest([str(v) for v in self.reference_chart_mean]),
                "conditional_gain": string_rows(self.conditional_gain),
                "conditional_covariance_sha256": metadata_digest(string_rows(self.conditional_covariance)),
                "law_role": "canonical_exact_named_source_map_and_reference_conditional_graph",
                "materialized_binary64_mean_or_factor": False,
                "graph_implementation_sha256": IMPLEMENTATION_SHA256}

    def digest(self):
        return metadata_digest(self.record())

    @classmethod
    def from_record(cls, record, *, source, **factory_arguments):
        # Actual owners are required: self-consistent metadata is not origin.
        from transfer.source_response_transport import transport_response_prior
        regenerated = transport_response_prior(source, **factory_arguments)
        if metadata_digest(regenerated.record()) != metadata_digest(record):
            raise ValueError("saved exact graph differs from actual source/completion/transport owners")
        return regenerated

    def validate_origin(self, source, **factory_arguments):
        return self.from_record(self.record(), source=source, **factory_arguments)
