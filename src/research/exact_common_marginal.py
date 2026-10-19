"""Exact pushforward of the original source Gaussian onto the shared z space.

The original source codec remains the origin.  Only Gaussian directions absent
from every target functional are analytically integrated out, with integral one.
No magnitude cutoff, SVD, factor refit, or rounded covariance defines this chart.
"""
from __future__ import annotations

import hashlib
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from research.conditional_response_graph import exact, product, rows, string_rows
from research.numeric_response_consumer import CompiledAffineImage, _round_rational, _round_surd
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
KIND = 'exact_original_source_shared_common_gaussian_marginal/1'


def integer_rows(matrix):
    denominators = tuple(math.lcm(*(v.denominator for v in row)) for row in matrix)
    integers = np.array([[v.numerator * (d // v.denominator) for v in row]
                         for row, d in zip(matrix, denominators, strict=True)], dtype=object)
    return integers, denominators


def exact_psd_ldl_from_integer_gram(gram, denominators):
    """Symmetric fraction-free Bareiss elimination; retain every positive pivot.

    At each step the active integer matrix equals previous_pivot times the
    Schur complement.  A zero PSD pivot has an exactly zero Schur row.
    Row denominators are applied algebraically after integer elimination.
    """
    a = [list(map(int, row)) for row in gram]
    n = len(a)
    if any(len(row) != n for row in a) or len(denominators) != n:
        raise ValueError('complete square common covariance and row denominators required')
    if any(a[i][j] != a[j][i] for i in range(n) for j in range(i)):
        raise ValueError('exact common Gram must be symmetric')
    columns, diagonal, pivots, zero_pivots = [], [], [], []
    previous = 1
    for k in range(n):
        pivot = a[k][k]
        if pivot < 0:
            raise ValueError('exact common covariance is not PSD')
        if pivot == 0:
            if any(a[j][k] != 0 for j in range(k + 1, n)):
                raise ValueError('zero PSD pivot requires an exactly zero Schur row')
            zero_pivots.append(k)
            continue
        column = [Fraction() for _ in range(n)]
        column[k] = Fraction(1)
        for j in range(k + 1, n):
            column[j] = Fraction(a[j][k] * denominators[k], pivot * denominators[j])
        columns.append(column)
        diagonal.append(Fraction(pivot, previous * denominators[k] ** 2))
        pivots.append(k)
        for j in range(k + 1, n):
            for i in range(j, n):
                numerator = pivot * a[i][j] - a[i][k] * a[j][k]
                value, remainder = divmod(numerator, previous)
                if remainder:
                    raise ArithmeticError('fraction-free exact Gram elimination lost divisibility')
                a[i][j] = a[j][i] = value
        previous = pivot
    lower = tuple(tuple(column[i] for column in columns) for i in range(n))
    return lower, tuple(diagonal), tuple(pivots), tuple(zero_pivots)


class ExactCommonMarginalSource:
    def __init__(self, reference):
        started = time.perf_counter()
        self.graph, self.reference = reference.graph, reference
        self.compiler_sha256 = IMPLEMENTATION_SHA256
        self.lower, self.diagonal = reference.lower, reference.diagonal
        self.original_source_rank = self.graph.source_factor.shape[1]
        self.common_image = product(self.graph.source_map, self.graph.source_factor)
        self.common_mean = tuple(row[0] for row in product(self.graph.source_map,
            tuple((exact(v),) for v in self.graph.source_mean)))
        projected = time.perf_counter()
        integer, denominators = integer_rows(self.common_image)
        gram_integer = integer @ integer.T
        grammed = time.perf_counter()
        self.common_factor, self.common_diagonal, pivots, zeros = exact_psd_ldl_from_integer_gram(
            gram_integer, denominators)
        factorized = time.perf_counter()
        self.source_rank = len(self.common_diagonal)
        self.rank = self.source_rank + len(self.diagonal)
        self.coordinate_ids = tuple(f'exact_common_PSD_LDL:{i}' for i in pivots) + tuple(
            f'independent_conditional_LDL:{i}' for i in range(len(self.diagonal)))
        self.origin = {'kind': KIND, 'original_graph_digest': self.graph.digest(),
            'original_source_factor_sha256': array_digest(self.graph.source_factor),
            'original_reference_source_chart': reference.source_chart,
            'exact_common_mean': list(map(str, self.common_mean)),
            'exact_common_factor': string_rows(self.common_factor),
            'exact_common_diagonal': list(map(str, self.common_diagonal)),
            'common_coordinate_pivots': pivots, 'exact_zero_pivots': zeros,
            'conditional_unit_lower': string_rows(self.lower),
            'conditional_diagonal': list(map(str, self.diagonal)),
            'coordinate_ids': self.coordinate_ids,
            'source_null_directions_integrated': self.original_source_rank-self.source_rank,
            'source_null_standard_gaussian_integral': 1,
            'measure': 'independent_standard_normal_in_declared_marginal_chart',
            'compiler_sha256': IMPLEMENTATION_SHA256}
        self.source_chart = metadata_digest(self.origin)
        self.diagnostics = {'source_projection_seconds': projected-started,
            'integer_gram_seconds': grammed-projected, 'exact_psd_ldl_seconds': factorized-grammed,
            'setup_seconds': time.perf_counter()-started,
            'common_dimensions': len(denominators), 'original_source_dimensions': self.original_source_rank,
            'exact_common_rank': self.source_rank, 'conditional_dimensions': len(self.diagonal),
            'proper_latent_dimensions': self.rank, 'exact_zero_pivots': zeros,
            'maximum_L_integer_bits': max((max(v.numerator.bit_length(), v.denominator.bit_length())
                for row in self.common_factor for v in row), default=0),
            'maximum_D_integer_bits': max((max(v.numerator.bit_length(), v.denominator.bit_length())
                for v in self.common_diagonal), default=0)}

    def compile(self, functionals, *, query_binding):
        raw = np.asarray(functionals, dtype=object)
        if raw.ndim != 3 or raw.shape[-1] != self.graph.dimension or not raw.shape[0] or not raw.shape[1]:
            raise ValueError('actual functionals require complete response coordinates')
        query = rows(raw.reshape(-1, self.graph.dimension))
        common, rest = self.graph.common_rows, self.graph.completion_rows
        wc = tuple(tuple(row[i] for i in common) for row in query)
        wr = tuple(tuple(row[i] for i in rest) for row in query)
        adjustment = product(wr, self.graph.conditional_gain) if rest else tuple(() for _ in query)
        effective = tuple(tuple(a+b for a, b in zip(w, extra, strict=True))
            for w, extra in zip(wc, adjustment, strict=True)) if rest else wc
        source_image = product(effective, self.common_factor)
        mean = [sum((a*b for a, b in zip(row, self.common_mean, strict=True)), Fraction())
                for row in effective]
        if rest:
            lifted = product(self.graph.conditional_gain, tuple((self.graph.reference_chart_mean[i],) for i in common))
            offset = tuple((self.graph.reference_chart_mean[i]-lifted[j][0],) for j, i in enumerate(rest))
            shift = product(wr, offset)
            mean = [value+shift[i][0] for i, value in enumerate(mean)]
            residual_image = product(wr, self.lower)
        else:
            residual_image = tuple(() for _ in query)
        values = np.empty((len(query), 1+self.rank))
        errors = np.empty_like(values)
        for i, value in enumerate(mean):
            values[i, 0], errors[i, 0] = _round_rational(value)
            for j, coefficient in enumerate(source_image[i]):
                values[i, 1+j], errors[i, 1+j] = _round_surd(coefficient, self.common_diagonal[j])
            for j, coefficient in enumerate(residual_image[i]):
                values[i, 1+self.source_rank+j], errors[i, 1+self.source_rank+j] = _round_surd(coefficient, self.diagonal[j])
        binding = {'schema': KIND, 'graph_digest': self.graph.digest(), 'source_chart': self.source_chart,
            'proper_latent_dimensions': self.rank, 'exact_query_sha256': metadata_digest(string_rows(query)),
            'actual_query_binding': query_binding, 'compiler_sha256': IMPLEMENTATION_SHA256,
            'source_null_standard_gaussian_integral': 1,
            'rounding': 'after_exact_effective_query_then_rational_times_sqrt_rational'}
        shape = raw.shape[:2]+(1+self.rank,)
        return CompiledAffineImage(values.reshape(shape), errors.reshape(shape), binding)
