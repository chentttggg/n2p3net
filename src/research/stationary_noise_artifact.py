"""Canonical joint stationary covariance of exact stored PARCOR processes.

Small pairwise rational Yule--Walker equations define the law, independently
of the numerical Cholesky compiler. Admission compares the saved root to that
law in its own weak directions; no covariance jitter or parameter repair is
permitted. The actual finite tail lattice remains its original owner's map.
"""
from __future__ import annotations

import hashlib
import math
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
from scipy.linalg import solve_triangular

from research.execution import UnidentifiableFitError
from transfer.dictionary_artifact import COMPILER_SOURCE_SHA256, DictionaryArtifactError, _binary64
from transfer.noise_model import array_digest, metadata_digest

PROGRAM = "exact_PARCOR_pairwise_rational_YW_joint_root_compile_and_admission/1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_PRECISION = 96
_CONTEXT = Context(prec=_PRECISION, rounding=ROUND_HALF_EVEN,
    Emin=-999999999, Emax=999999999, capitals=1, clamp=0, flags=[],
    traps=[InvalidOperation, DivisionByZero, Overflow])


def compiler_record(reflections, covariance, prefix, root):
    return {"program": PROGRAM, "source_sha256": SOURCE_SHA256,
        "scalar_rounding_source_sha256": COMPILER_SOURCE_SHA256,
        "law": "exact_real_step_up_of_stored_binary64_PARCOR_and_stored_innovation_covariance",
        "row_order": "native_time_then_observed_channel",
        "root_state_origin": "actual_numeric_root_law_admission_preserves_fit_producer_provenance",
        "new_root_creation": "Decimal96_joint_Cholesky_then_integer_binary64_nearest_ties_even",
        "root_admission": "exact_rational_law_covariance_pullback_without_parameter_or_PSD_repair",
        "decimal_precision": _PRECISION, "binary64_rounding": "integer_nearest_ties_even",
        "input_law_sha256": metadata_digest({"PARCOR": np.asarray(reflections).tolist(),
            "innovation_covariance": np.asarray(covariance).tolist(), "prefix_samples": prefix}),
        "compiled_root_sha256": array_digest(root)}


def _coefficients(reflections):
    polynomial = []
    for value in reflections:
        reflection = Fraction.from_float(float(value))
        polynomial = [left+reflection*right for left, right in
                      zip(polynomial, reversed(polynomial), strict=True)] + [reflection]
    return tuple(-value for value in polynomial)


def _rational_solve(matrix, rhs):
    """Fraction-free Bareiss elimination; exact divisions certify each step."""
    size = len(rhs)
    rows = []
    for row, value in zip(matrix, rhs, strict=True):
        operands = [*row, value]
        # PARCOR polynomials are dyadic. Use the largest denominator, then
        # assert exact lifting rather than assuming arbitrary rational inputs.
        denominator = max(item.denominator for item in operands)
        if any(denominator % item.denominator for item in operands):
            raise ValueError("stationary Yule-Walker lifting requires dyadic coefficients")
        rows.append([item.numerator*(denominator//item.denominator) for item in operands])
    previous = 1
    for column in range(size-1):
        pivot_row = next((row for row in range(column, size) if rows[row][column]), None)
        if pivot_row is None:
            raise UnidentifiableFitError("stationary pairwise law is singular")
        if pivot_row != column:
            rows[column], rows[pivot_row] = rows[pivot_row], rows[column]
        pivot = rows[column][column]
        for row in range(column+1, size):
            multiplier = rows[row][column]
            for index in range(column+1, size+1):
                value, remainder = divmod(pivot*rows[row][index]-multiplier*rows[column][index], previous)
                if remainder:
                    raise ValueError("stationary exact elimination lost its integer relation")
                rows[row][index] = value
            rows[row][column] = 0
        previous = pivot
    if not rows[-1][-2]:
        raise UnidentifiableFitError("stationary pairwise law is singular")
    solution = [Fraction(0)]*size
    for row in reversed(range(size)):
        solution[row] = (rows[row][-1]-sum((rows[row][column]*solution[column]
            for column in range(row+1, size)), Fraction(0)))/rows[row][row]
    return solution


def _pair_lags(left, right, prefix):
    order = len(left)
    if not order:
        return {lag: Fraction(int(lag == 0)) for lag in range(1-prefix, prefix)}
    size = 2*order+1
    matrix = [[Fraction(int(row == column)) for column in range(size)] for row in range(size)]
    rhs = [Fraction(0)]*size
    rhs[order] = Fraction(1)
    for lag in range(1, order+1):
        for index, coefficient in enumerate(left, 1):
            matrix[order+lag][order+lag-index] -= coefficient
        for index, coefficient in enumerate(right, 1):
            matrix[order-lag][order-lag+index] -= coefficient
    for index, coefficient in enumerate(left, 1):
        matrix[order][order-index] -= coefficient
    initial = _rational_solve(matrix, rhs)
    values = {lag: initial[order+lag] for lag in range(-order, order+1)}
    for lag in range(order+1, prefix):
        values[lag] = sum((coefficient*values[lag-index] for index, coefficient in enumerate(left, 1)), Fraction(0))
        values[-lag] = sum((coefficient*values[-lag+index] for index, coefficient in enumerate(right, 1)), Fraction(0))
    return values


def _law_covariance(reflections, covariance, prefix):
    coefficients = tuple(_coefficients(row) for row in reflections)
    channels = len(coefficients)
    cache, pairs = {}, {}
    for left in range(channels):
        for right in range(left, channels):
            key = (coefficients[left], coefficients[right])
            if key not in cache:
                cache[key] = _pair_lags(*key, prefix)
            variance = Fraction.from_float(float(covariance[left, right]))
            pairs[left, right] = {lag: variance*value for lag, value in cache[key].items()}
    size = prefix*channels
    result = [[None]*size for _ in range(size)]
    for row in range(size):
        time, left = divmod(row, channels)
        for column in range(row+1):
            previous, right = divmod(column, channels)
            lag = time-previous
            value = pairs[left, right][lag] if left <= right else pairs[right, left][-lag]
            result[row][column] = result[column][row] = value
    return result


def _decimal(value):
    return Decimal(value.numerator)/Decimal(value.denominator)


def _round(value):
    numerator, denominator = value.as_integer_ratio()
    return _binary64(numerator, denominator)


def _upper(value):
    if not value:
        return 0.
    return np.nextafter(_round(value), np.inf)


def _frobenius_upper(values):
    """Positive scalar upper bound, independent of BLAS/SVD norm rounding."""
    if not np.isfinite(values).all():
        raise UnidentifiableFitError("stationary root admission arithmetic is nonfinite")
    operands = [Decimal.from_float(float(value)) for value in np.ravel(values)]
    total = sum((value*value for value in operands), Decimal(0))
    rho = Decimal(10)**(1-_PRECISION)
    rounding = (len(operands)+1)*rho/(1-(len(operands)+1)*rho)
    return _upper((total/(1-rounding)).sqrt())


def _cholesky(covariance):
    size = len(covariance)
    root = [[Decimal(0)]*size for _ in range(size)]
    for row in range(size):
        for column in range(row+1):
            value = _decimal(covariance[row][column])-sum((root[row][index]*root[column][index]
                for index in range(column)), Decimal(0))
            if row == column:
                if value <= 0:
                    raise UnidentifiableFitError("stationary joint covariance has unresolved observed support")
                root[row][column] = value.sqrt()
            else:
                root[row][column] = value/root[column][column]
    return np.asarray([[_round(value) for value in row] for row in root])


def _admit(root, covariance):
    """Bound the exact covariance discrepancy after the stored root inverse.

    A Decimal residual follows the small covariance defect, rather than the
    cancelling binary64 Gram operands. Its absolute rounding bound is carried
    through a checked inverse and the normwise Neumann relation. Thus the
    compiler's own Cholesky is not the numerical admission oracle.
    """
    size = len(root)
    # A common exact power-of-two chart prevents units alone from underflowing
    # the law residual. The actual saved root and scoring map stay unchanged.
    exponent = math.frexp(float(np.max(np.abs(root))))[1]
    normalized_root = np.ldexp(root, -exponent)
    if np.any((root != 0) & (normalized_root == 0)):
        raise UnidentifiableFitError("stationary covariance coordinate dynamic range is unresolved")
    scale = Fraction(2)**(-2*exponent)
    root = normalized_root
    decimal_root = [[Decimal.from_float(float(value)) for value in row] for row in root]
    residual, residual_error = np.empty_like(root), np.empty_like(root)
    rho = Decimal(10)**(1-_PRECISION)
    for row in range(size):
        for column in range(row+1):
            products = [decimal_root[row][index]*decimal_root[column][index] for index in range(column+1)]
            expected = _decimal(covariance[row][column]*scale)
            value = expected-sum(products, Decimal(0))
            # Conservative relative bounds for rational conversion, products,
            # sequential reduction, subtraction and final binary64 rounding.
            absolute = (column+4)*rho*(abs(expected)+sum(map(abs, products), Decimal(0)))
            actual = _round(value)
            conversion = abs(value-Decimal.from_float(actual))
            bound = _round(absolute+conversion)
            residual[row, column] = residual[column, row] = actual
            residual_error[row, column] = residual_error[column, row] = np.nextafter(bound, np.inf)
    inverse = solve_triangular(root, np.eye(size), lower=True)
    epsilon = Decimal.from_float(float(np.finfo(float).eps))
    gamma = (size+1)*epsilon/(1-(size+1)*epsilon)
    # Standard relative dot bounds need an absolute term for gradual
    # underflow. This is the Frobenius bound of a uniform (size+1)*u entry
    # error matrix, where u is the smallest binary64 subnormal.
    underflow = size*(size+1)*Decimal.from_float(float(np.nextafter(0., 1.)))
    defect = np.eye(size)-inverse@root
    positive_product = np.abs(inverse)@np.abs(root)
    eta = _upper(Decimal.from_float(_frobenius_upper(defect))
        +(gamma*Decimal.from_float(_frobenius_upper(positive_product))+underflow)/(1-gamma))
    if not np.isfinite(eta) or eta >= 1:
        raise UnidentifiableFitError("stationary joint root inverse is numerically unresolved")
    image = inverse@residual@inverse.T
    magnitudes = np.abs(inverse)@np.abs(residual)@np.abs(inverse.T)
    chain_underflow = (1+gamma)*underflow*Decimal.from_float(_frobenius_upper(inverse))+underflow
    magnitude_bound = (Decimal.from_float(_frobenius_upper(magnitudes))+chain_underflow)/(1-gamma)**2
    arithmetic = (2*gamma+gamma*gamma)*magnitude_bound+chain_underflow
    source = (Decimal.from_float(_frobenius_upper(
        np.abs(inverse)@residual_error@np.abs(inverse.T)))+chain_underflow)/(1-gamma)**2
    bound = _upper((Decimal.from_float(_frobenius_upper(image))+arithmetic+source)
                   /(1-Decimal.from_float(eta))**2)
    if not np.isfinite(bound) or bound > 1e-8*np.sqrt(size):
        raise UnidentifiableFitError("compiled prefix state does not obey the declared stationary noise model")
    return bound


def stationary_root(reflections, covariance, prefix, *, supplied_root=None):
    """Compile or admit one joint prefix, keeping actual parameters untouched."""
    try:
        with localcontext(_CONTEXT):
            law = _law_covariance(reflections, covariance, prefix)
            root = _cholesky(law) if supplied_root is None else np.asarray(supplied_root)
            if not np.isfinite(root).all():
                raise UnidentifiableFitError("stationary joint root is not finitely representable")
            _admit(root, law)
    except (InvalidOperation, DivisionByZero, Overflow, OverflowError, DictionaryArtifactError) as error:
        if supplied_root is not None:
            raise ValueError("saved stationary prefix law cannot be admitted") from error
        raise UnidentifiableFitError("stationary joint law is not representable") from error
    except UnidentifiableFitError as error:
        if supplied_root is not None:
            raise ValueError(str(error)) from error
        raise
    return root
