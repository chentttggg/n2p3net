"""Portable physical dictionary state and exact expression admission.

The numeric program is independent of libm, NumPy transcendental functions,
BLAS, the decimal context of its caller, and the host float rounding mode.
Admission verifies the saved scalars against that program and retains them;
it never substitutes a freshly compiled matrix for the saved matrix. Source
hashes identify producers, while expression checks establish numeric relations.
"""
from __future__ import annotations

import hashlib
import re
import struct
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

from transfer.noise_model import array_digest, frozen_array, metadata_digest

DICTIONARY_SCHEMA = "n2p3_response_dictionary/2"
COMPILER_SEMANTIC_ID = "n2p3_physical_dictionary_decimal96_binary64/1"
COMPILER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ARRAY_KEYS = ("times", "ambient_matrix", "matrix", "endpoint_rows", "constraint_map")
_PI = "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647"
_CONTEXT = Context(prec=96, rounding=ROUND_HALF_EVEN, Emin=-999999999, Emax=999999999,
                   capitals=1, clamp=0, flags=[], traps=[InvalidOperation, DivisionByZero, Overflow])
_COMPILER = {
    "semantic_id": COMPILER_SEMANTIC_ID,
    "decimal_precision": 96,
    "decimal_rounding": "ROUND_HALF_EVEN",
    "cosine_terms": 64,
    "pi_decimal": _PI,
    "binary64_rounding": "integer_nearest_ties_even",
    "lift": "exact_rational_dot_of_binary64_operands_then_one_binary64_round",
}


class DictionaryArtifactError(ValueError):
    """A contract error, never an expected scientific fit/rank failure."""

    def __init__(self, reason, *, field=None, index=None):
        self.reason, self.field, self.index = reason, field, index
        location = "" if field is None else f" ({field}{'' if index is None else index})"
        super().__init__(f"response dictionary numerical artifact: {reason}{location}")

    def record(self):
        return {"schema": "n2p3_dictionary_artifact_error/1", "reason": self.reason,
                "field": self.field, "index": None if self.index is None else list(self.index)}


def _nearest_integer(numerator, denominator):
    quotient, remainder = divmod(numerator, denominator)
    return quotient + (2*remainder > denominator or
                       (2*remainder == denominator and quotient % 2 == 1))


def _binary64(numerator, denominator=1):
    """Round an exact rational to IEEE binary64 with integer arithmetic."""
    if denominator <= 0:
        raise ValueError("positive rational denominator required")
    sign = int(numerator < 0) << 63
    numerator = abs(numerator)
    if not numerator:
        bits = 0
    else:
        exponent = numerator.bit_length() - denominator.bit_length()
        if (numerator < denominator << exponent if exponent >= 0
                else numerator << -exponent < denominator):
            exponent -= 1
        shift = 1074 if exponent < -1022 else 52-exponent
        scaled_n = numerator << max(shift, 0)
        scaled_d = denominator << max(-shift, 0)
        significand = _nearest_integer(scaled_n, scaled_d)
        if exponent < -1022:
            bits = sign | significand
        else:
            if significand == 1 << 53:
                exponent += 1
                significand >>= 1
            if exponent > 1023:
                raise DictionaryArtifactError("nonfinite_physical_scalar")
            bits = sign | ((exponent+1023) << 52) | (significand-(1 << 52))
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _rounded(value):
    numerator, denominator = value.as_integer_ratio()
    return _binary64(numerator, denominator)


def _cosine(phase):
    """Evaluate cos(pi*phase); exact symmetry reduces the series to [0, pi/2]."""
    phase %= 2
    if phase > 1:
        phase = 2-phase
    sign = 1 if phase < Fraction(1, 2) else -1
    phase = phase if sign == 1 else 1-phase
    if phase == Fraction(1, 2):
        return 0.
    if phase == 0:
        return float(sign)
    x = Decimal(_PI)*Decimal(phase.numerator)/Decimal(phase.denominator)
    square, term, total = x*x, Decimal(1), Decimal(1)
    for order in range(1, 64):
        term = -term*square/Decimal((2*order-1)*(2*order))
        total += term
    return _rounded(sign*total)


def _program_dimensions(spec, sample_rate_hz):
    if (isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, (int, float))
            or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0):
        raise DictionaryArtifactError("invalid_native_sampling_rate", field="sample_rate_hz")
    rate = float(sample_rate_hz)
    rate_ratio = Fraction.from_float(rate)
    # Preserve L=round(binary64(fs*T)), including the original ties-to-even rule.
    product = rate_ratio*Fraction.from_float(spec["response_seconds"])
    length = round(_rounded(product))
    width = spec["n_basis"]
    if length < 3 or width > length:
        raise DictionaryArtifactError("ambient_dimension_outside_sample_support")
    if spec["endpoint"] == "hard_zero" and (spec["family"] != "cosine" or width < 3):
        raise DictionaryArtifactError("unsupported_or_empty_hard_endpoint_space")
    return rate, rate_ratio, length, width


def _chart(width, hard):
    if not hard:
        return np.eye(width, dtype=np.float64)
    result = np.zeros((width, width-2), dtype=np.float64)
    column = 0
    for parity in (0, 1):
        indices = tuple(range(parity, width, 2))
        for order in range(1, len(indices)):
            denominator = Decimal(order*(order+1)).sqrt()
            one = _rounded(Decimal(1)/denominator)
            result[list(indices[:order]), column] = one
            result[indices[order], column] = _rounded(-Decimal(order)/denominator)
            column += 1
    return result


def _ambient(spec, length, width, sample, column):
    if spec["family"] == "cosine":
        return _cosine(Fraction(sample*column, length))
    denominator = max(width-1, 1)
    center = Fraction(column*(length-1), denominator)
    sigma = Fraction(length, denominator)
    exponent = -Fraction(1, 2)*((Fraction(sample)-center)/sigma)**2
    decimal_exponent = Decimal(exponent.numerator)/Decimal(exponent.denominator)
    return _rounded(decimal_exponent.exp())


def _dot(left, right):
    # No BLAS reduction order and no rounded intermediate binary64 products.
    total = sum((Fraction.from_float(float(a))*Fraction.from_float(float(b))
                 for a, b in zip(left, right, strict=True) if b != 0), Fraction())
    return _rounded(total)


def _generation(spec, rate_ratio, length, width):
    if spec["family"] == "gaussian":
        denominator = max(width-1, 1)
        return {"centers_seconds": [_rounded(Fraction(j*(length-1), denominator)/rate_ratio)
                                    for j in range(width)],
                "width_seconds": _rounded(Fraction(length, denominator)/rate_ratio),
                "evaluation": "exact_rational_native_coordinate_then_decimal96_exp"}
    return {"angular_frequencies_per_second": [
                _rounded(Decimal(_PI)*Decimal(j*rate_ratio.numerator)/
                         Decimal(length*rate_ratio.denominator)) for j in range(width)],
            "evaluation": "exact_rational_phase_then_decimal96_reduced_cosine"}


def _metadata(spec, rate, rate_ratio, length, width, arrays, producer, compiler_source):
    hard = spec["endpoint"] == "hard_zero"
    return {"schema": DICTIONARY_SCHEMA, "spec": spec, "sample_rate_hz": rate,
            "actual_seconds": _rounded(Fraction(length)/rate_ratio),
            "support_samples": [0, length],
            "last_sample_seconds": _rounded(Fraction(length-1)/rate_ratio),
            "endpoint_evaluation": "u=0_and_actual_L_over_sample_rate",
            "column_units": "dimensionless", "coefficient_units": "volts",
            "generation": _generation(spec, rate_ratio, length, width),
            "ambient_dimension": width, "function_dimension": width-2 if hard else width,
            "coordinate_system": {"ambient_order": "ascending_center_or_harmonic",
                "coefficient_chart": "helmert_even_then_odd" if hard else "identity",
                "constraints": "analytic_cosine_endpoints" if hard else "unrestricted"},
            "compiler": {**_COMPILER, "source_sha256": compiler_source},
            "implementation_sha256": producer,
            "state_sha256": array_digest(*(arrays[key] for key in ARRAY_KEYS))}


def compile_state(spec, sample_rate_hz, *, implementation_sha256):
    """Compile one new /2 state. Spec validation belongs to ResponseDictionarySpec."""
    rate, rate_ratio, length, width = _program_dimensions(spec, sample_rate_hz)
    hard = spec["endpoint"] == "hard_zero"
    with localcontext(_CONTEXT):
        chart = _chart(width, hard)
        times = np.asarray([_rounded(Fraction(i)/rate_ratio) for i in range(length)])
        ambient = np.asarray([[_ambient(spec, length, width, i, j) for j in range(width)]
                              for i in range(length)])
        endpoints = np.asarray([[_ambient(spec, length, width, i, j) for j in range(width)]
                                for i in (0, length)])
        matrix = np.asarray([[_dot(row, column) for column in chart.T] for row in ambient]) if hard else ambient
        arrays = {key: frozen_array(value) for key, value in zip(ARRAY_KEYS,
                  (times, ambient, matrix, endpoints, chart), strict=True)}
        metadata = _metadata(spec, rate, rate_ratio, length, width, arrays,
                             implementation_sha256, COMPILER_SOURCE_SHA256)
    return arrays, metadata


def _same_scalar(actual, expected, field, index):
    if struct.pack(">d", float(actual)) != struct.pack(">d", expected):
        raise DictionaryArtifactError("physical_expression_relation_mismatch", field=field, index=index)


def admit_record(record, spec):
    """Validate one external /2 artifact and retain its exact saved numeric state.

    This is not producer authentication. The package/executor must separately
    bind producer hashes to its accepted execution freeze. Rehashing a forged
    payload cannot bypass the independent physical expression checks here.
    """
    if not isinstance(record, dict) or record.get("schema") != DICTIONARY_SCHEMA:
        raise DictionaryArtifactError("unsupported_dictionary_schema", field="schema")
    rate, rate_ratio, length, width = _program_dimensions(spec, record.get("sample_rate_hz"))
    hard = spec["endpoint"] == "hard_zero"
    dimension = width-2 if hard else width
    shapes = ((length,), (length, width), (length, dimension), (2, width), (width, dimension))
    arrays = {}
    for key, shape in zip(ARRAY_KEYS, shapes, strict=True):
        try:
            value = np.asarray(record[key])
        except (KeyError, TypeError, ValueError) as error:
            raise DictionaryArtifactError("invalid_numerical_payload", field=key) from error
        if (value.dtype.kind != "f" or value.dtype.itemsize != 8 or value.shape != shape
                or not np.all(np.isfinite(value))):
            raise DictionaryArtifactError("invalid_numerical_payload", field=key)
        arrays[key] = frozen_array(value)
    compiler = record.get("compiler")
    for key, value in (("implementation_sha256", record.get("implementation_sha256")),
                       ("compiler.source_sha256", None if not isinstance(compiler, dict) else compiler.get("source_sha256"))):
        if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
            raise DictionaryArtifactError("invalid_producer_source_identity", field=key)
    with localcontext(_CONTEXT):
        expected = _metadata(spec, rate, rate_ratio, length, width, arrays,
                             record["implementation_sha256"], compiler["source_sha256"])
        metadata = {key: value for key, value in record.items() if key not in ARRAY_KEYS}
        try:
            metadata_identity = metadata_digest(metadata)
        except (TypeError, ValueError) as error:
            raise DictionaryArtifactError("invalid_physical_metadata") from error
        if metadata_identity != metadata_digest(expected):
            raise DictionaryArtifactError("physical_metadata_or_payload_binding_mismatch")
        chart = _chart(width, hard)
        for row in range(width):
            for column in range(dimension):
                _same_scalar(arrays["constraint_map"][row, column], float(chart[row, column]),
                             "constraint_map", (row, column))
        for row in range(length):
            _same_scalar(arrays["times"][row], _rounded(Fraction(row)/rate_ratio), "times", (row,))
            for column in range(width):
                _same_scalar(arrays["ambient_matrix"][row, column],
                             _ambient(spec, length, width, row, column), "ambient_matrix", (row, column))
            for column in range(dimension):
                value = (_dot(arrays["ambient_matrix"][row], arrays["constraint_map"][:, column])
                         if hard else float(arrays["ambient_matrix"][row, column]))
                _same_scalar(arrays["matrix"][row, column], value, "matrix", (row, column))
        for row, sample in enumerate((0, length)):
            for column in range(width):
                _same_scalar(arrays["endpoint_rows"][row, column],
                             _ambient(spec, length, width, sample, column), "endpoint_rows", (row, column))
    return arrays, expected
