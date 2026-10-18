"""Compile an actual functional of the exact graph into one shared affine chart.

The original source latents and independent exact-LDL conditional latents are
global across requests. Rounding occurs after the requested functional contracts
the named operators; each returned entry carries an outward absolute error.
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from research.conditional_response_graph import exact, kronecker, product, rows, string_rows
from transfer.noise_model import array_digest, freeze_metadata, frozen_array, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _up_fraction(value):
    result = float(value)
    if Fraction.from_float(result) < value:
        result = float(np.nextafter(result, np.inf))
    return result


def _round_rational(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("actual Gaussian functional is outside binary64 range")
    return result, _up_fraction(abs(value - Fraction.from_float(result)))


def _round_surd(coefficient, radicand):
    """Correctly round a*sqrt(d), with exact adjacent-float square comparisons."""
    if not coefficient:
        return 0., 0.
    square = coefficient * coefficient * radicand
    exponent = square.numerator.bit_length() - square.denominator.bit_length()
    scale = Fraction(2)**(2 * (exponent // 2))
    estimate = math.ldexp(math.sqrt(float(square / scale)), exponent // 2)
    if not math.isfinite(estimate):
        raise ValueError("actual Gaussian functional is outside binary64 range")
    while Fraction.from_float(estimate)**2 > square:
        estimate = float(np.nextafter(estimate, -np.inf))
    while True:
        upper = float(np.nextafter(estimate, np.inf))
        if not math.isfinite(upper) or Fraction.from_float(upper)**2 > square:
            break
        estimate = upper
    lower_exact = Fraction.from_float(estimate)
    if lower_exact**2 == square:
        rounded, error = estimate, 0.
    else:
        if not math.isfinite(upper):
            raise ValueError("actual Gaussian functional has no finite upper binary64 enclosure")
        upper_exact = Fraction.from_float(upper)
        midpoint = (lower_exact + upper_exact) / 2
        even_lower = struct.unpack(">Q", struct.pack(">d", estimate))[0] % 2 == 0
        rounded = estimate if square < midpoint**2 or square == midpoint**2 and even_lower else upper
        rounded_exact = Fraction.from_float(rounded)
        denominator = rounded_exact + lower_exact
        # For the smallest subnormal bracket, use its outward endpoint width.
        bound = abs(square - rounded_exact**2) / denominator if denominator else upper_exact
        error = _up_fraction(bound)
    return (-rounded if coefficient < 0 else rounded), error


def exact_ldl(covariance):
    covariance = rows(covariance)
    size = len(covariance)
    lower = [[Fraction(i == j) for j in range(size)] for i in range(size)]
    diagonal = []
    for i in range(size):
        pivot = covariance[i][i] - sum((lower[i][k]**2 * diagonal[k] for k in range(i)), Fraction())
        if pivot <= 0:
            raise ValueError("conditional innovation covariance must be exact proper")
        diagonal.append(pivot)
        for j in range(i + 1, size):
            if covariance[j][i] != covariance[i][j]:
                raise ValueError("conditional innovation covariance must be exact symmetric")
            lower[j][i] = (covariance[j][i] - sum((lower[j][k]*lower[i][k]*diagonal[k]
                                                 for k in range(i)), Fraction())) / pivot
    return rows(lower), tuple(diagonal)


@dataclass(frozen=True)
class CompiledAffineImage:
    values: np.ndarray
    absolute_error: np.ndarray
    binding: dict

    def __post_init__(self):
        if (self.values.shape != self.absolute_error.shape or not np.isfinite(self.values).all()
                or not np.isfinite(self.absolute_error).all() or np.any(self.absolute_error < 0)):
            raise ValueError("compiled affine values and outward errors differ")
        object.__setattr__(self, "values", frozen_array(self.values))
        object.__setattr__(self, "absolute_error", frozen_array(self.absolute_error))
        object.__setattr__(self, "binding", freeze_metadata(self.binding))

    def gaussian_response_images(self, coordinate_ids):
        from research.channel_noise_mechanism import GaussianResponseImages
        return GaussianResponseImages(self.values[:, :, 0], self.values[:, :, 1:],
            self.absolute_error[:, :, 0], self.absolute_error[:, :, 1:], coordinate_ids, self.binding)


class AffineResponseSource:
    def __init__(self, graph, *, completion_factorization=None):
        self.graph = graph
        rest = len(graph.completion_rows)
        if completion_factorization is None:
            lower, diagonal = exact_ldl(graph.conditional_covariance)
        else:
            temporal, spatial = map(rows, completion_factorization)
            if kronecker(temporal, spatial) != graph.conditional_covariance:
                raise ValueError("conditional factorization differs from the exact graph covariance")
            time_lower, time_diagonal = exact_ldl(temporal)
            space_lower, space_diagonal = exact_ldl(spatial)
            lower = kronecker(time_lower, space_lower)
            diagonal = tuple(a*b for a in time_diagonal for b in space_diagonal)
        if len(diagonal) != rest:
            raise ValueError("conditional innovations differ from the complete proper residual space")
        self.lower, self.diagonal = lower, diagonal
        source_rank = graph.source_factor.shape[1]
        self.rank = source_rank + rest
        self.coordinate_ids = tuple(f"original_source:{i}" for i in range(source_rank)) + tuple(
            f"independent_conditional_LDL:{i}" for i in range(rest))
        self.source_chart = metadata_digest({"graph": graph.digest(), "source_factor": array_digest(graph.source_factor),
            "conditional_unit_lower": string_rows(lower), "conditional_D": [str(v) for v in diagonal],
            "coordinate_ids": self.coordinate_ids, "law": "independent_original_source_and_conditional_standard_normals"})

    def compile(self, functionals, *, query_binding):
        raw = np.asarray(functionals, dtype=object)
        if raw.ndim != 3 or raw.shape[-1] != self.graph.dimension or not raw.shape[0] or not raw.shape[1]:
            raise ValueError("actual functionals require sample by channel by complete response coordinates")
        query = rows(raw.reshape(-1, self.graph.dimension))
        common, rest = self.graph.common_rows, self.graph.completion_rows
        wc = tuple(tuple(row[i] for i in common) for row in query)
        wr = tuple(tuple(row[i] for i in rest) for row in query)
        adjustment = product(wr, self.graph.conditional_gain) if rest else tuple(() for _ in query)
        effective = tuple(tuple(a+b for a,b in zip(w, extra, strict=True)) for w,extra in zip(wc,adjustment,strict=True)) if rest else wc
        source_weights = product(effective, self.graph.source_map)
        source_image = product(source_weights, self.graph.source_factor)
        mean = [sum((weight*exact(value) for weight,value in zip(row,self.graph.source_mean,strict=True)), Fraction())
                for row in source_weights]
        if rest:
            z0 = tuple((self.graph.reference_chart_mean[i],) for i in common)
            lifted = product(self.graph.conditional_gain, z0)
            offset = tuple((self.graph.reference_chart_mean[i]-lifted[j][0],) for j,i in enumerate(rest))
            shift = product(wr, offset)
            mean = [value+shift[i][0] for i,value in enumerate(mean)]
            residual_image = product(wr, self.lower)
        else:
            residual_image = tuple(() for _ in query)
        values, errors = np.zeros((len(query), 1+self.rank)), np.zeros((len(query), 1+self.rank))
        source_rank = self.graph.source_factor.shape[1]
        for i, value in enumerate(mean):
            values[i, 0], errors[i, 0] = _round_rational(value)
            for j, coefficient in enumerate(source_image[i]):
                values[i, 1+j], errors[i, 1+j] = _round_rational(coefficient)
            for j, coefficient in enumerate(residual_image[i]):
                values[i, 1+source_rank+j], errors[i, 1+source_rank+j] = _round_surd(coefficient, self.diagonal[j])
        shape = raw.shape[:2]+(1+self.rank,)
        binding = {"schema": "n2p3_actual_functional_affine_Gaussian/1", "graph_digest": self.graph.digest(),
            "source_chart": self.source_chart, "proper_latent_dimensions": self.rank,
            "exact_query_sha256": metadata_digest(string_rows(query)), "actual_query_binding": query_binding,
            "source_mean_factor_first_materialized": False, "independent_conditional_latents": True,
            "rounding": "correct_round_rational_and_rational_times_sqrt_rational_after_actual_query",
            "absolute_error": "outward_entry_error_relative_to_exact_declared_query_no_noise_or_design_error_assumed_zero",
            "cross_row_contrasts": "compose_with_exact_query_or_propagate_returned_entry_errors",
            "compiler_sha256": IMPLEMENTATION_SHA256, "source_fits": 0, "target_fits": 0}
        return CompiledAffineImage(values.reshape(shape), errors.reshape(shape), binding)


@dataclass(frozen=True)
class GraphAffineResponse:
    functional_plan: object
    source: AffineResponseSource
    channels: int

    @property
    def scope(self):
        return self.functional_plan.scope

    @property
    def source_chart(self):
        return self.source.source_chart

    @property
    def rank(self):
        return self.source.rank

    @property
    def identity(self):
        return metadata_digest({"functional_plan": self.functional_plan.identity,
            "source_chart": self.source_chart, "compiler": IMPLEMENTATION_SHA256, "channels": self.channels})

    def support(self, *, native_start, native_stop):
        return self.functional_plan.support(native_start=native_start, native_stop=native_stop)

    def affine(self, selected_states, *, native_start, native_stop):
        query, ledger = self.functional_plan.functionals(selected_states, native_start=native_start, native_stop=native_stop)
        result = self.source.compile(query, query_binding=ledger)
        if result.values.shape[:2] != (native_stop-native_start, self.channels):
            raise ValueError("actual functional plan changed native sample/channel coordinates")
        return result.values, result.absolute_error, result.binding

    def affine_difference(self, selected_states, reference_states, *, native_start, native_stop):
        """Contract state minus reference before either affine image is rounded."""
        actual, actual_ledger = self.functional_plan.functionals(selected_states,
            native_start=native_start, native_stop=native_stop)
        reference, reference_ledger = self.functional_plan.functionals(reference_states,
            native_start=native_start, native_stop=native_stop)
        actual, reference = np.asarray(actual, dtype=object), np.asarray(reference, dtype=object)
        if actual.shape != reference.shape:
            raise ValueError("actual/reference functionals differ in their original physical coordinates")
        difference = np.empty(actual.shape, dtype=object)
        for index in np.ndindex(actual.shape):
            difference[index] = exact(actual[index])-exact(reference[index])
        result = self.source.compile(difference, query_binding={"actual": actual_ledger,
            "reference": reference_ledger, "operation": "exact_actual_functional_state_minus_reference"})
        if result.values.shape[:2] != (native_stop-native_start, self.channels):
            raise ValueError("functional difference changed native sample/channel coordinates")
        return result.values, result.absolute_error, result.binding
