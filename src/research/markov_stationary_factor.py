"""Exact Markov innovation prefix root and small-factor law admission.

The actual channel-diagonal AR(p) law, full innovation covariance, initial
stationary joint law and native time ordering are unchanged. Only a new,
explicit root producer and a consumer-side mathematical witness are provided.
"""
from __future__ import annotations

import hashlib
import math
from decimal import Decimal, DecimalException, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

from research.execution import UnidentifiableFitError
from research.shared_stationary_factor import (
    _approximate_inverse,
    _fraction_matrix,
    _sqrt_upper,
    _upper,
)
from research.stationary_noise_artifact import (
    _CONTEXT,
    _PRECISION,
    _cholesky,
    _coefficients,
    _decimal,
    _law_covariance,
    _round,
)
from research.stationary_noise_artifact import SOURCE_SHA256 as LAW_SOURCE_SHA256
from transfer.dictionary_artifact import COMPILER_SOURCE_SHA256, DictionaryArtifactError, _binary64
from transfer.noise_model import array_digest, finite_real_array, metadata_digest, require_sha256

PROGRAM = "exact_diagonal_AR_Markov_innovation_prefix_root/1"
SCHEMA = "n2p3_markov_stationary_root_factor/1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_LAW = "exact_real_step_up_of_stored_binary64_PARCOR_and_stored_innovation_covariance"
_KEYS = {"schema", "program", "source_sha256", "factor_law_source_sha256",
    "scalar_rounding_source_sha256", "law", "row_order", "prefix_samples", "channels",
    "order", "input_law_sha256", "initial_root", "innovation_root", "compiled_root_sha256", "certificate"}


def _law(reflections, covariance, prefix):
    values = finite_real_array(reflections, name="stored channel PARCOR")
    sigma = finite_real_array(covariance, name="actual innovation covariance")
    if (values.ndim != 2 or not len(values) or np.any(np.abs(values) >= 1)
            or type(prefix) is not int or prefix < 1 or sigma.shape != (len(values), len(values))
            or not np.array_equal(sigma, sigma.T)):
        raise ValueError("Markov stationary law needs stable equal-order PARCOR and exact symmetric full Q")
    if values.shape[1] >= prefix:
        raise UnidentifiableFitError("no innovation tail; use the original exact stationary entry")
    order = values.shape[1]
    initial = _law_covariance(values, sigma, order) if order else []
    return values, sigma, initial, _fraction_matrix(sigma)


def _binding(values, sigma, prefix):
    return metadata_digest({"PARCOR": values.tolist(), "innovation_covariance": sigma.tolist(),
                            "prefix_samples": prefix})


def _positive_lower(values, size, name):
    root = finite_real_array(values, name=name)
    if root.shape != (size, size) or np.any(np.diag(root) <= 0) or np.any(np.triu(root, 1) != 0):
        raise ValueError(f"{name} requires its complete positive lower-triangular support")
    return root


def _rounded_recurrence(values, initial, innovation, prefix):
    """Exact dyadic lag recursion; one integer binary64 rounding per output.

    Small Cholesky roots have already been rounded. Their error is part of
    the law certificate, not silently treated as an exact covariance root.
    """
    channels, order = values.shape
    coefficients = [_coefficients(row) for row in values]
    first, eps = _fraction_matrix(initial), _fraction_matrix(innovation)
    rows = []
    output = np.zeros((prefix*channels, prefix*channels))
    for time in range(prefix):
        for channel in range(channels):
            index = time*channels+channel
            if time < order:
                row = first[index][:index+1]
            else:
                row = [Fraction() for _ in range(index+1)]
                for lag, coefficient in enumerate(coefficients[channel], 1):
                    if coefficient:
                        for column, previous in enumerate(rows[index-lag*channels]):
                            row[column] += coefficient*previous
                row[time*channels:] = eps[channel][:channel+1]
            rows.append(row)
            for column, value in enumerate(row):
                output[index, column] = _binary64(value.numerator, value.denominator)
    return output


def _row_chart(initial, innovation, root, initial_law, sigma):
    """One exact power-of-two scale per observed channel; AR lags commute."""
    channels = len(innovation)
    exponents = [math.frexp(float(np.max(np.abs(row))))[1] for row in innovation]
    def chart(value):
        powers = np.resize(exponents, len(value))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            normalized = np.ldexp(value, -np.asarray(powers)[:, None])
            restored = np.ldexp(normalized, np.asarray(powers)[:, None])
        if not np.isfinite(normalized).all() or not np.array_equal(restored, value):
            raise UnidentifiableFitError("Markov factor channel chart is not exactly representable")
        return normalized
    first = chart(initial) if len(initial) else initial
    eps, saved = chart(innovation), chart(root)
    scales = [Fraction(2)**(-exponents[channel]) for channel in range(channels)]
    initial_covariance = [[value*scales[i % channels]*scales[j % channels]
        for j, value in enumerate(row)] for i, row in enumerate(initial_law)]
    covariance = [[value*scales[i]*scales[j] for j, value in enumerate(row)] for i, row in enumerate(sigma)]
    return first, eps, saved, initial_covariance, covariance


def _integer_matrix(values):
    """Lift dyadic matrices exactly; matrix multiplication then uses integers."""
    denominator = max((value.denominator for row in values for value in row), default=1)
    if denominator & (denominator-1):
        raise ValueError("Markov certificate expected exact dyadic operands")
    result = np.asarray([[value.numerator*(denominator//value.denominator) for value in row]
                         for row in values], dtype=object)
    return result, denominator


def _product_squared(left, right):
    """Exact Frobenius squared, without a full-prefix floating matrix product."""
    a, ad = left
    b, bd = _integer_matrix(right)
    product = a @ b
    return Fraction(sum((int(value)**2 for value in product.flat), 0), (ad*bd)**2)


def _inverse_witness(root):
    """Exact dyadic check of a small inverse, independent of its construction."""
    inverse = _approximate_inverse(root)
    a, ad = _integer_matrix(inverse)
    b, bd = _integer_matrix(_fraction_matrix(root))
    product = a @ b
    denominator = ad*bd
    for i in range(len(root)):
        product[i, i] -= denominator
    squared = Fraction(sum((int(value)**2 for value in product.flat), 0), denominator**2)
    defect = Fraction.from_float(_sqrt_upper(squared))
    if defect >= 1:
        raise UnidentifiableFitError("Markov small factor inverse is unresolved")
    return inverse, defect


def _pushed_norm(inverse, values):
    """Exact norm of U values U^T, with all binary64 operands dyadic."""
    a, ad = _integer_matrix(inverse)
    b, bd = _integer_matrix(_fraction_matrix(values))
    product = (a @ b) @ a.T
    squared = Fraction(sum((int(value)**2 for value in product.flat), 0), (ad*ad*bd)**2)
    return Fraction.from_float(_sqrt_upper(squared))


def _small_factor_certificate(root, covariance):
    """Original Decimal96 residual bound, deterministic exact inverse pullback.

    This carries the canonical stationary owner's conversion/dot/subtraction
    bound. The small-matrix inverse and both residual images are checked in
    exact dyadic arithmetic; no platform BLAS value enters the certificate.
    """
    size = len(root)
    residual, error = np.empty_like(root), np.empty_like(root)
    with localcontext(_CONTEXT):
        actual = [[Decimal.from_float(float(value)) for value in row] for row in root]
        rho = Decimal(10)**(1-_PRECISION)
        for row in range(size):
            for column in range(row+1):
                products = [actual[row][index]*actual[column][index] for index in range(column+1)]
                expected = _decimal(covariance[row][column])
                value = expected-sum(products, Decimal())
                absolute = (column+4)*rho*(abs(expected)+sum(map(abs, products), Decimal()))
                rounded = _round(value)
                conversion = abs(value-Decimal.from_float(rounded))
                bound = float(np.nextafter(_round(absolute+conversion), np.inf))
                residual[row, column] = residual[column, row] = rounded
                error[row, column] = error[column, row] = bound
    inverse, defect = _inverse_witness(root)
    magnitude = [[abs(value) for value in row] for row in inverse]
    law_error = (_pushed_norm(inverse, residual)+_pushed_norm(magnitude, error))/(1-defect)**2
    return law_error, inverse, defect


def _recurrence_residual_squared(values, initial, innovation, root, inverse_initial, inverse_innovation):
    channels, order = values.shape
    coefficients = [_coefficients(row) for row in values]
    saved = _fraction_matrix(root)
    first, eps = _fraction_matrix(initial), _fraction_matrix(innovation)
    squared = Fraction()
    if order:
        size = order*channels
        residual = [[saved[i][j]-first[i][j] for j in range(size)] for i in range(size)]
        squared += _product_squared(_integer_matrix(inverse_initial), residual)
    inverse = _integer_matrix(inverse_innovation)
    for time in range(order, len(root)//channels):
        width = (time+1)*channels
        residual = []
        for channel in range(channels):
            index = time*channels+channel
            row = list(saved[index][:width])
            for lag, coefficient in enumerate(coefficients[channel], 1):
                if coefficient:
                    for column in range(width):
                        row[column] -= coefficient*saved[index-lag*channels][column]
            for column, value in enumerate(eps[channel]):
                row[time*channels+column] -= value
            residual.append(row)
        squared += _product_squared(inverse, residual)
    return squared


def _certificate(values, initial_law, sigma, prefix, initial, innovation, root):
    first, eps, saved, initial_law, sigma = _row_chart(initial, innovation, root, initial_law, sigma)
    channels, order = values.shape
    if order:
        initial_error, inverse_initial, initial_defect = _small_factor_certificate(first, initial_law)
    else:
        initial_error = initial_defect = Fraction()
        inverse_initial = []
    innovation_error, inverse_innovation, innovation_defect = _small_factor_certificate(eps, sigma)
    squared = _recurrence_residual_squared(values, first, eps, saved, inverse_initial, inverse_innovation)
    defect = max(initial_defect, innovation_defect)
    rho = Fraction.from_float(_sqrt_upper(squared))/(1-defect)
    if rho >= 1:
        raise UnidentifiableFitError("Markov saved root innovation relation is unresolved")
    error = Fraction.from_float(_sqrt_upper(initial_error**2+(prefix-order)*innovation_error**2))
    bound = _upper((error+2*rho+rho*rho)/(1-rho)**2)
    policy = float(1e-8*np.sqrt(prefix*channels))
    if bound > policy:
        raise UnidentifiableFitError("Markov stationary root does not resolve the original law policy")
    return {"proof": "exact_dyadic_HS_minus_D_small_factor_inverse_defects_and_Markov_congruence",
        "initial_samples": order, "innovation_blocks": prefix-order,
        "initial_law_pullback_F_bound": _upper(initial_error),
        "innovation_law_pullback_F_bound": _upper(innovation_error),
        "initial_inverse_defect_F_bound": _upper(initial_defect),
        "innovation_inverse_defect_F_bound": _upper(innovation_defect),
        "innovation_relation_pullback_F_bound": _upper(rho),
        "block_diagonal_law_pullback_F_bound": _upper(error),
        "joint_law_pullback_F_bound": bound, "original_joint_policy_F_bound": policy}


def compile_markov_stationary_root(reflections, covariance, prefix):
    """Create a new root producer; original saved roots are never replaced."""
    values, sigma, initial_law, law_q = _law(reflections, covariance, prefix)
    try:
        with localcontext(_CONTEXT):
            initial = _cholesky(initial_law) if initial_law else np.empty((0, 0))
            innovation = _cholesky(law_q)
        root = _rounded_recurrence(values, initial, innovation, prefix)
        certificate = _certificate(values, initial_law, law_q, prefix, initial, innovation, root)
    except (DecimalException, OverflowError, DictionaryArtifactError) as error:
        raise UnidentifiableFitError("Markov stationary root is not representable") from error
    state = {"schema": SCHEMA, "program": PROGRAM, "source_sha256": SOURCE_SHA256,
        "factor_law_source_sha256": LAW_SOURCE_SHA256,
        "scalar_rounding_source_sha256": COMPILER_SOURCE_SHA256, "law": _LAW,
        "row_order": "native_time_then_observed_channel", "prefix_samples": prefix,
        "channels": len(sigma), "order": values.shape[1], "input_law_sha256": _binding(values, sigma, prefix),
        "initial_root": initial.tolist(), "innovation_root": innovation.tolist(),
        "compiled_root_sha256": array_digest(root), "certificate": certificate}
    return root, state


def admit_markov_stationary_root(reflections, covariance, prefix, root, state):
    """Cold admission uses saved small roots, never a Cholesky compiler."""
    values, sigma, initial_law, law_q = _law(reflections, covariance, prefix)
    if (set(state) != _KEYS or state["schema"] != SCHEMA or state["program"] != PROGRAM
            or state["law"] != _LAW or state["row_order"] != "native_time_then_observed_channel"
            or state["prefix_samples"] != prefix or state["channels"] != len(sigma)
            or state["order"] != values.shape[1] or state["input_law_sha256"] != _binding(values, sigma, prefix)):
        raise ValueError("saved Markov factor state differs from its actual stationary law")
    for key in ("source_sha256", "factor_law_source_sha256", "scalar_rounding_source_sha256"):
        require_sha256(state[key], key)
    saved = _positive_lower(root, prefix*len(sigma), "saved Markov prefix root")
    initial = (_positive_lower(state["initial_root"], values.shape[1]*len(sigma), "saved initial root")
               if values.shape[1] else np.asarray(state["initial_root"], dtype=float).reshape(0, 0))
    innovation = _positive_lower(state["innovation_root"], len(sigma), "saved innovation root")
    if array_digest(saved) != state["compiled_root_sha256"]:
        raise ValueError("saved Markov prefix root bytes differ from its producer state")
    certificate = _certificate(values, initial_law, law_q, prefix, initial, innovation, saved)
    if metadata_digest(certificate) != metadata_digest(state["certificate"]):
        raise ValueError("saved Markov root certificate differs from actual root-law arithmetic")
    return certificate


def admit_saved_markov_stationary_root(reflections, covariance, prefix, root):
    """Consumer-only witness for any unchanged original diagonal-AR /3 root."""
    values, sigma, initial_law, law_q = _law(reflections, covariance, prefix)
    saved = _positive_lower(root, prefix*len(sigma), "original saved private root")
    try:
        with localcontext(_CONTEXT):
            initial = _cholesky(initial_law) if initial_law else np.empty((0, 0))
            innovation = _cholesky(law_q)
        certificate = _certificate(values, initial_law, law_q, prefix, initial, innovation, saved)
    except (DecimalException, OverflowError, DictionaryArtifactError) as error:
        raise UnidentifiableFitError("saved private root Markov witness is not representable") from error
    return {"program": "diagonal_AR_PARCOR_saved_root_Markov_admission/1",
        "consumer_source_sha256": SOURCE_SHA256, "law_source_sha256": LAW_SOURCE_SHA256,
        "input_law_sha256": _binding(values, sigma, prefix), "saved_root_sha256": array_digest(saved),
        "state_role": "consumer_witness_only_original_saved_root_and_codec_unchanged",
        "certificate": certificate}
