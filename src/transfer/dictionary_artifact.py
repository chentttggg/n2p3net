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
    DecimalException,
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
BSPLINE_DICTIONARY_SCHEMA = "n2p3_response_dictionary/3"
FAMILY_DICTIONARY_SCHEMA = "n2p3_response_dictionary/4"
_FAMILY_SCHEMAS = {"gaussian": DICTIONARY_SCHEMA, "cosine": DICTIONARY_SCHEMA,
                   "bspline": BSPLINE_DICTIONARY_SCHEMA, "raised_cosine": FAMILY_DICTIONARY_SCHEMA,
                   "multiscale_bspline": FAMILY_DICTIONARY_SCHEMA}
DICTIONARY_SCHEMAS = frozenset(_FAMILY_SCHEMAS.values())
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
_BSPLINE_COMPILER = {
    "semantic_id": "n2p3_physical_cubic_bspline_rational_binary64/1",
    "degree": 3,
    "knot_program": "open_clamped_uniform_normalized_exact_rational",
    "evaluation": "exact_rational_Cox_de_Boor_then_one_binary64_round",
    "binary64_rounding": "integer_nearest_ties_even",
    "lift": "exact_rational_dot_of_binary64_operands_then_one_binary64_round",
}
_RAISED_COMPILER = {**_COMPILER, "semantic_id": "n2p3_regular_raised_cosine_decimal96_composed_binary64/1",
                    "evaluation": "exact_rational_phase_then_Decimal96_(1+cos)/2_then_one_binary64_round"}
_MULTISCALE_COMPILER = {**_BSPLINE_COMPILER, "semantic_id": "n2p3_coarse6_detail3_cubic_rational_binary64/1",
                        "evaluation": "independent_coarse6_and_fine9_columns_exact_rational_then_one_binary64_round"}
_FAMILY_COMPILERS = {"gaussian": _COMPILER, "cosine": _COMPILER, "bspline": _BSPLINE_COMPILER,
                     "raised_cosine": _RAISED_COMPILER, "multiscale_bspline": _MULTISCALE_COMPILER}
EVALUATION_SCHEMA = "n2p3_response_dictionary_evaluation/1"
EVALUATION_PROGRAM = {
    "semantic_id": "n2p3_physical_dictionary_exact_query_and_saved_chart/1",
    "coordinate_arithmetic": "exact_integer_or_stored_binary64_or_rational; seconds_times_exact_stored_fs",
    "scalar_expression": "original_family_compiler_expression_at_exact_sample_coordinate",
    "chart_lift": "exact_rational_dot_of_binary64_ambient_and_saved_constraint_map_then_one_binary64_round",
    "derivative": 0,
    "column_units": "dimensionless",
    "sample_site": "n_exact; seconds_site=n/exact_stored_fs; rounded_native_times_are_not_exact_sites",
    "right_endpoint": "analytic_value_at_exact_L; cropped_function_excludes_L",
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


def _cosine_decimal(phase):
    """Shared Decimal96 cosine core; composition precedes binary64 rounding."""
    phase %= 2
    if phase > 1:
        phase = 2-phase
    sign = 1 if phase < Fraction(1, 2) else -1
    phase = phase if sign == 1 else 1-phase
    if phase == Fraction(1, 2):
        return Decimal(0)
    if phase == 0:
        return Decimal(sign)
    x = Decimal(_PI)*Decimal(phase.numerator)/Decimal(phase.denominator)
    square, term, total = x*x, Decimal(1), Decimal(1)
    for order in range(1, 64):
        term = -term*square/Decimal((2*order-1)*(2*order))
        total += term
    return sign*total


def _cosine(phase):
    """Original /2 one-final-round cos(pi*phase) wrapper, with its exact anchors."""
    return _rounded(_cosine_decimal(phase))


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
    if spec["family"] == "bspline" and width < 4:
        raise DictionaryArtifactError("cubic_bspline_requires_four_ambient_functions")
    if spec["family"] == "raised_cosine" and width < 2:
        raise DictionaryArtifactError("regular_raised_cosine_requires_two_ambient_functions")
    if spec["family"] == "multiscale_bspline" and width != 9:
        raise DictionaryArtifactError("fixed_coarse6_detail3_requires_nine_ambient_functions")
    if spec["endpoint"] == "hard_zero" and (spec["family"] not in {"cosine", "bspline", "raised_cosine", "multiscale_bspline"} or width < 3):
        raise DictionaryArtifactError("unsupported_or_empty_hard_endpoint_space")
    return rate, rate_ratio, length, width


def _chart(width, hard, family="cosine"):
    if not hard:
        return np.eye(width, dtype=np.float64)
    if family in {"bspline", "raised_cosine"}:
        # Open-clamped cubic endpoint rows are e_0 and e_(width-1).
        return np.eye(width, dtype=np.float64)[:, 1:-1]
    if family == "multiscale_bspline":
        # Coarse endpoint functions are columns0/5; the three fine details
        # vanish at both physical endpoints. Restrict this actual ambient chart.
        return np.eye(9, dtype=np.float64)[:, (1, 2, 3, 4, 6, 7, 8)]
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


def _cubic_bspline_fraction(length, width, sample, column):
    """Exact uniform/open-clamped cubic value; q=1 is the inner left limit."""
    coordinate = Fraction(sample, length)
    values = {}

    def knot(index):
        return Fraction(0) if index < 4 else Fraction(1) if index >= width else Fraction(index-3, width-3)

    def basis(index, degree):
        key = index, degree
        if key not in values:
            if degree == 0:
                values[key] = Fraction(int(knot(index) < coordinate <= knot(index+1) if sample == length
                                           else knot(index) <= coordinate < knot(index+1)))
            else:
                left, right = knot(index+degree)-knot(index), knot(index+degree+1)-knot(index+1)
                values[key] = (((coordinate-knot(index))*basis(index, degree-1)/left if left else 0)
                               +((knot(index+degree+1)-coordinate)*basis(index+1, degree-1)/right if right else 0))
        return values[key]

    # Only the fixed cubic triangle (at most ten nodes), not every column's
    # full ambient row. This local memo dies after the one scalar expression.
    return basis(column, 3)


def _cubic_bspline_value(length, width, sample, column):
    value = _cubic_bspline_fraction(length, width, sample, column)
    return _binary64(value.numerator, value.denominator)


def _ambient(spec, length, width, sample, column):
    if spec["family"] == "bspline":
        return _cubic_bspline_value(length, width, sample, column)
    if spec["family"] == "multiscale_bspline":
        # Direct independent functions, not rounded fine-matrix products or
        # all coarse/fine columns duplicated in the coefficient prior.
        return (_cubic_bspline_value(length, 6, sample, column) if column < 6 else
                _cubic_bspline_value(length, 9, sample, (2, 4, 6)[column-6]))
    if spec["family"] == "raised_cosine":
        phase = Fraction(sample*(width-1), length)-column
        if abs(phase) >= 1:
            return 0.
        return _rounded((1+_cosine_decimal(phase))/2)
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
    if spec["family"] == "bspline":
        intervals = width-3
        knots = [Fraction()] * 4 + [Fraction(i, intervals) for i in range(1, intervals)] + [Fraction(1)] * 4
        return {"degree": 3, "normalized_knots_exact_ratio": [[v.numerator, v.denominator] for v in knots],
                "time_coordinate": "q=t/(L/fs); native q=n/L",
                "endpoint_side": "left_at_q=1; right_at_other_knots",
                "evaluation": "exact_rational_Cox_de_Boor_then_one_binary64_round"}
    if spec["family"] == "raised_cosine":
        return {"normalized_centers_exact_ratio": [[j, width-1] for j in range(width)],
                "normalized_radius_exact_ratio": [1, width-1],
                "evaluation": "exact_rational_phase_then_Decimal96_(1+cos)/2_then_one_binary64_round",
                "atom_support": "abs(phase)<1; exact_zero_at_plus_minus_one_and_outside",
                "atom_zero_extension_smoothness": "C1; second_derivative_jumps_at_support_boundary",
                "global_crop": "half_open_0_T; free_endpoint_values_may_jump; hard_value_zero_is_C1"}
    if spec["family"] == "multiscale_bspline":
        refinement = ((1, 0, 0, 0, 0, 0), (Fraction(1, 2), Fraction(1, 2), 0, 0, 0, 0),
            (0, Fraction(3, 4), Fraction(1, 4), 0, 0, 0),
            (0, Fraction(3, 16), Fraction(11, 16), Fraction(1, 8), 0, 0),
            (0, 0, Fraction(1, 2), Fraction(1, 2), 0, 0),
            (0, 0, Fraction(1, 8), Fraction(11, 16), Fraction(3, 16), 0),
            (0, 0, 0, Fraction(1, 4), Fraction(3, 4), 0),
            (0, 0, 0, 0, Fraction(1, 2), Fraction(1, 2)), (0, 0, 0, 0, 0, 1))
        return {"degree": 3, "coarse_uniform_intervals": 3, "fine_uniform_intervals": 6,
                "coarse_width": 6, "fine_width": 9, "fine_detail_indices": [2, 4, 6],
                "coarse_in_fine_exact_ratio": [[[v.numerator, v.denominator] for v in row] for row in refinement],
                "evaluation": "direct_exact_coarse6_and_fine_detail3_then_one_binary64_round",
                "span": "independent_nine_column_chart_of_uniform_fine9_space; not_orthogonal_wavelets",
                "endpoint_coordinate_indices": [0, 5], "endpoint_side": "left_at_q=1; right_at_other_knots"}
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
    family = spec["family"]
    cubic = family == "bspline"
    order = ("coarse6_then_fine_details_2_4_6" if family == "multiscale_bspline" else
             "ascending_normalized_center" if family == "raised_cosine" else
             "ascending_cubic_knot_index" if cubic else "ascending_center_or_harmonic")
    chart = ("omit_coarse_endpoint_coefficients_0_5" if family == "multiscale_bspline" else
             "interior_center_coefficients" if family == "raised_cosine" else
             "interior_clamped_coefficients" if cubic else "helmert_even_then_odd")
    constraint = ("coarse_detail_endpoint_values" if family == "multiscale_bspline" else
                  "regular_raised_cosine_endpoint_values" if family == "raised_cosine" else
                  "clamped_cubic_endpoint_values" if cubic else "analytic_cosine_endpoints")
    return {"schema": _FAMILY_SCHEMAS[family], "spec": spec, "sample_rate_hz": rate,
            "actual_seconds": _rounded(Fraction(length)/rate_ratio),
            "support_samples": [0, length],
            "last_sample_seconds": _rounded(Fraction(length-1)/rate_ratio),
            "endpoint_evaluation": "u=0_and_actual_L_over_sample_rate",
            "column_units": "dimensionless", "coefficient_units": "volts",
            "generation": _generation(spec, rate_ratio, length, width),
            "ambient_dimension": width, "function_dimension": width-2 if hard else width,
            "coordinate_system": {"ambient_order": order, "coefficient_chart": chart if hard else "identity",
                                  "constraints": constraint if hard else "unrestricted"},
            "compiler": {**_FAMILY_COMPILERS[family], "source_sha256": compiler_source},
            "implementation_sha256": producer,
            "state_sha256": array_digest(*(arrays[key] for key in ARRAY_KEYS))}


def compile_state(spec, sample_rate_hz, *, implementation_sha256):
    """Compile the exact declared family/schema state using one owner."""
    rate, rate_ratio, length, width = _program_dimensions(spec, sample_rate_hz)
    hard = spec["endpoint"] == "hard_zero"
    with localcontext(_CONTEXT):
        chart = _chart(width, hard, spec["family"])
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
    """Validate the exact declared family/schema program and retain saved state.

    This is not producer authentication. The package/executor must separately
    bind producer hashes to its accepted execution freeze. Rehashing a forged
    payload cannot bypass the independent physical expression checks here.
    """
    if (not isinstance(record, dict) or not isinstance(record.get("schema"), str)
            or record["schema"] not in DICTIONARY_SCHEMAS):
        raise DictionaryArtifactError("unsupported_dictionary_schema", field="schema")
    schema = _FAMILY_SCHEMAS[spec["family"]]
    if record["schema"] != schema:
        raise DictionaryArtifactError("dictionary_schema_and_family_program_mismatch", field="schema")
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
        chart = _chart(width, hard, spec["family"])
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


def exact_query(query):
    """Keep every query value as a real rational, never snap it to a grid."""
    try:
        values = np.asarray(query, dtype=object)
    except (TypeError, ValueError) as error:
        raise DictionaryArtifactError("invalid_exact_query") from error
    if values.size == 0:
        raise DictionaryArtifactError("empty_exact_query")
    ratios = []
    for value in values.flat:
        if isinstance(value, Fraction):
            exact = value
        elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            exact = Fraction(int(value))
        elif isinstance(value, (float, np.floating)) and np.asarray(value).dtype.itemsize <= 8 and np.isfinite(value):
            exact = Fraction.from_float(float(value))
        else:
            raise DictionaryArtifactError("query_requires_finite_binary64_integer_or_Fraction")
        ratios.append(exact)
    return values.shape, tuple(ratios)


def _evaluation_metadata(dictionary_record, dictionary_digest, shape, ratios, coordinate, support, arrays,
                         evaluator_source, compiler_source):
    rate = Fraction.from_float(dictionary_record["sample_rate_hz"])
    samples = tuple(value*rate if coordinate == "seconds" else value for value in ratios)
    return {"schema": EVALUATION_SCHEMA, "program": EVALUATION_PROGRAM,
        "query_shape": list(shape), "query_exact_ratio": [[v.numerator, v.denominator] for v in ratios],
        "query_coordinate": coordinate, "query_units": "s" if coordinate == "seconds" else "native_sample_coordinate",
        "sample_coordinates_exact_ratio": [[v.numerator, v.denominator] for v in samples],
        "support_program": support, "support_exact_sample_interval": [0, dictionary_record["support_samples"][1]],
        "dictionary": {"identity_digest": dictionary_digest, "schema": dictionary_record["schema"],
            "spec": dictionary_record["spec"], "sample_rate_hz": dictionary_record["sample_rate_hz"],
            "state_sha256": dictionary_record["state_sha256"], "implementation_sha256": dictionary_record["implementation_sha256"],
            "compiler": dictionary_record["compiler"]},
        "consumer": {"implementation_sha256": evaluator_source, "compiler_source_sha256": compiler_source},
        "state_sha256": array_digest(arrays["ambient_matrix"], arrays["matrix"], arrays["support_mask"])}


def evaluate_state(dictionary_record, dictionary_digest, query, *, coordinate, support, implementation_sha256):
    """Evaluate the same scalar family expression and admitted saved chart.

    The caller owns an admitted immutable dictionary. Native samples remain the
    original scalar expression, including Gaussian centers at L-1. Analytic
    evaluation and causal [0,L) cropping are separate named functions.
    """
    if coordinate not in {"seconds", "sample"} or support not in {"analytic", "half_open"}:
        raise DictionaryArtifactError("unknown_query_coordinate_or_support")
    shape, ratios = exact_query(query)
    spec = dictionary_record["spec"]
    _, rate, length, width = _program_dimensions(spec, dictionary_record["sample_rate_hz"])
    chart = np.asarray(dictionary_record["constraint_map"], dtype=np.float64)
    samples = tuple(value*rate if coordinate == "seconds" else value for value in ratios)
    mask = np.asarray([0 <= value < length for value in samples], dtype=bool).reshape(shape)
    ambient = np.zeros((len(samples), width), dtype=np.float64)
    matrix = np.zeros((len(samples), chart.shape[1]), dtype=np.float64)
    with localcontext(_CONTEXT):
        for index, value in enumerate(samples):
            if support == "half_open" and not 0 <= value < length:
                continue
            try:
                ambient[index] = [_ambient(spec, length, width, value, column) for column in range(width)]
                matrix[index] = ([_dot(ambient[index], column) for column in chart.T]
                                 if spec["endpoint"] == "hard_zero" else ambient[index])
            except (DecimalException, OverflowError) as error:
                raise DictionaryArtifactError("query_scalar_expression_not_representable", index=(index,)) from error
    if not np.isfinite(ambient).all() or not np.isfinite(matrix).all():
        raise DictionaryArtifactError("query_scalar_expression_not_representable")
    arrays = {"ambient_matrix": frozen_array(ambient.reshape((*shape, width))),
              "matrix": frozen_array(matrix.reshape((*shape, chart.shape[1]))),
              "support_mask": np.frombuffer(mask.tobytes(), dtype=bool).reshape(shape)}
    metadata = _evaluation_metadata(dictionary_record, dictionary_digest, shape, ratios, coordinate, support,
                                    arrays, implementation_sha256, COMPILER_SOURCE_SHA256)
    return arrays, metadata


def admit_evaluation_record(record, dictionary_record, dictionary_digest):
    """Check query, program, parent and exact output; retain the saved values."""
    if not isinstance(record, dict) or record.get("schema") != EVALUATION_SCHEMA:
        raise DictionaryArtifactError("unsupported_evaluation_schema")
    try:
        shape = record["query_shape"]
        if (not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape)):
            raise ValueError("invalid shape")
        raw_ratios = record["query_exact_ratio"]
        if any(not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair)
               or pair[1] <= 0 for pair in raw_ratios):
            raise ValueError("invalid ratio")
        ratios = tuple(Fraction(*pair) for pair in raw_ratios)
        query = np.asarray(ratios, dtype=object).reshape(shape)
        source = record["consumer"]
        if any(not isinstance(source.get(key), str) or re.fullmatch("[0-9a-f]{64}", source[key]) is None
               for key in ("implementation_sha256", "compiler_source_sha256")):
            raise ValueError("invalid evaluator source")
        expected_arrays, _ = evaluate_state(dictionary_record, dictionary_digest, query,
            coordinate=record["query_coordinate"], support=record["support_program"],
            implementation_sha256=source["implementation_sha256"])
        expected_metadata = _evaluation_metadata(dictionary_record, dictionary_digest, tuple(shape), ratios,
            record["query_coordinate"], record["support_program"], expected_arrays,
            source["implementation_sha256"], source["compiler_source_sha256"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise DictionaryArtifactError("invalid_evaluation_query_or_program") from error
    arrays = {}
    for key, expected in expected_arrays.items():
        value = np.asarray(record.get(key))
        required_kind = "b" if key == "support_mask" else "f"
        if (value.shape != expected.shape or value.dtype.kind != required_kind
                or (required_kind == "f" and (value.dtype.itemsize != 8 or not np.isfinite(value).all()))):
            raise DictionaryArtifactError("invalid_evaluation_payload", field=key)
        if value.tobytes() != expected.tobytes():
            raise DictionaryArtifactError("query_expression_relation_mismatch", field=key)
        arrays[key] = (np.frombuffer(value.tobytes(), dtype=bool).reshape(value.shape)
                       if key == "support_mask" else frozen_array(value).reshape(value.shape))
    metadata = {key: value for key, value in record.items() if key not in arrays}
    if metadata_digest(metadata) != metadata_digest(expected_metadata):
        raise DictionaryArtifactError("evaluation_query_program_parent_or_binding_mismatch")
    return arrays, expected_metadata
