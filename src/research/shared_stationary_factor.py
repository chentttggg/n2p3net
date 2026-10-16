"""Exact shared-AR stationary law with two small certified root factors.

For time-major/channel-fast native coordinates, K = Gamma_P tensor Sigma.
The stored dense root is the explicitly rounded product of two small roots.
Its certificate includes factor-law errors and actual product-rounding error.
"""
from __future__ import annotations

import hashlib
import math
from decimal import Decimal, DecimalException, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

from research.execution import UnidentifiableFitError
from research.stationary_noise_artifact import _CONTEXT, _cholesky, _law_covariance, _round
from research.stationary_noise_artifact import (
    SOURCE_SHA256 as LAW_SOURCE_SHA256,
)
from transfer.dictionary_artifact import COMPILER_SOURCE_SHA256, DictionaryArtifactError, _binary64
from transfer.noise_model import array_digest, finite_real_array, metadata_digest, require_sha256

PROGRAM = "shared_PARCOR_exact_Kronecker_prefix_root/1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SCHEMA = "n2p3_shared_stationary_root_factor/1"
_LAW = "exact_real_step_up_of_stored_binary64_PARCOR_and_stored_innovation_covariance"
_PROOF = "exact_rational_small_factor_residuals_inverse_defects_and_Kronecker_product_rounding"
_STATE_KEYS = {"schema", "program", "source_sha256", "factor_law_source_sha256",
    "scalar_rounding_source_sha256", "law", "row_order", "prefix_samples", "channels",
    "input_law_sha256", "temporal_root", "spatial_root", "compiled_root_sha256", "certificate"}


def _fraction_matrix(values):
    return [[Fraction.from_float(float(v)) for v in row] for row in values]


def _transpose(values):
    return list(map(list, zip(*values, strict=True)))


def _multiply(left, right):
    columns = _transpose(right)
    return [[sum((a*b for a, b in zip(row, column, strict=True)), Fraction())
             for column in columns] for row in left]


def _upper(value):
    result = _binary64(value.numerator, value.denominator)
    if Fraction.from_float(result) < value:
        result = float(np.nextafter(result, np.inf))
    if not math.isfinite(result):
        raise UnidentifiableFitError("shared stationary bound is not finitely representable")
    return result


def _sqrt_upper(value):
    if not value:
        return 0.
    with localcontext(_CONTEXT):
        result = _round((Decimal(value.numerator)/Decimal(value.denominator)).sqrt())
    # Decimal is only a witness here. Exact comparisons certify the final
    # upper bound, including halfway cases and subnormal results.
    while Fraction.from_float(result)**2 < value:
        result = float(np.nextafter(result, np.inf))
        if not math.isfinite(result):
            raise UnidentifiableFitError("shared stationary norm is not finitely representable")
    return result


def _norm_upper(values):
    return Fraction.from_float(_sqrt_upper(sum((v*v for row in values for v in row), Fraction())))


def _approximate_inverse(root):
    """A deterministic witness; all its inverse defects are then checked exactly."""
    size = len(root)
    with localcontext(_CONTEXT):
        actual = [[Decimal.from_float(float(v)) for v in row] for row in root]
        inverse = [[Decimal(0)]*size for _ in range(size)]
        for row in range(size):
            for column in range(row+1):
                total = sum((actual[row][j]*inverse[j][column] for j in range(column, row)), Decimal())
                inverse[row][column] = (Decimal(int(row == column))-total)/actual[row][row]
        return _fraction_matrix([[_round(v) for v in row] for row in inverse])


def _factor_certificate(root, covariance):
    """Exact small-matrix residual with a checked approximate inverse B.

    D=I-BA implies A^-1=(I-D)^-1 B. Thus eta>=||D||_F<1 bounds
    both inverse corrections and the two-sided covariance pullback. No
    accuracy claim about the Decimal inverse witness is assumed.
    """
    size = len(root)
    actual = _fraction_matrix(root)
    inverse = _approximate_inverse(root)
    product = _multiply(inverse, actual)
    defect = [[Fraction(int(i == j))-product[i][j] for j in range(size)] for i in range(size)]
    eta = _norm_upper(defect)
    if eta >= 1:
        raise UnidentifiableFitError("shared stationary factor inverse is unresolved")
    gram = _multiply(actual, _transpose(actual))
    residual = [[covariance[i][j]-gram[i][j] for j in range(size)] for i in range(size)]
    pulled = _multiply(_multiply(inverse, residual), _transpose(inverse))
    error = _norm_upper(pulled)/(1-eta)**2
    inverse_norm = _norm_upper(inverse)/(1-eta)
    absolute = _multiply([[abs(v) for v in row] for row in inverse], [[abs(v) for v in row] for row in actual])
    condition_product = _norm_upper(absolute)+eta/(1-eta)*_norm_upper(inverse)*_norm_upper(actual)
    return error, inverse_norm, condition_product, eta


def _normalize(root):
    exponent = math.frexp(float(np.max(np.abs(root))))[1]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = np.ldexp(root, -exponent)
        restored = np.ldexp(normalized, exponent)
    if not np.isfinite(normalized).all() or not np.array_equal(restored, root):
        raise UnidentifiableFitError("shared stationary factor coordinate range is unresolved")
    return normalized, exponent


def _rounded_product(temporal, spatial):
    """Each stored product has one explicit integer binary64 rounding."""
    time_size, channels = len(temporal), len(spatial)
    left, time_exponent = _normalize(temporal)
    right, spatial_exponent = _normalize(spatial)
    left_q, right_q = _fraction_matrix(left), _fraction_matrix(right)
    result = np.zeros((time_size*channels, time_size*channels))
    cache = {}
    for time in range(time_size):
        for previous in range(time+1):
            a = left_q[time][previous]
            if not a:
                continue
            for channel in range(channels):
                for source in range(channel+1):
                    b = right_q[channel][source]
                    key = a, b
                    if key not in cache:
                        value = a*b
                        cache[key] = _binary64(value.numerator, value.denominator)
                    result[time*channels+channel, previous*channels+source] = cache[key]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        physical = np.ldexp(result, time_exponent+spatial_exponent)
        restored = np.ldexp(physical, -(time_exponent+spatial_exponent))
    if (not np.isfinite(physical).all() or np.any(np.diag(physical) <= 0)
            or not np.array_equal(restored, result)):
        raise UnidentifiableFitError("shared stationary root products are not representable in native units")
    return physical, left, right, time_exponent, spatial_exponent


def _law(reflections, covariance, prefix):
    values = finite_real_array(reflections, name="shared stored PARCOR")
    sigma = finite_real_array(covariance, name="shared innovation covariance")
    if (values.ndim != 1
            or np.any(np.abs(values) >= 1) or type(prefix) is not int or prefix < 1
            or sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1] or not len(sigma)
            or not np.array_equal(sigma, sigma.T)):
        raise ValueError("shared stationary law requires one actual stable PARCOR and exact symmetric innovation covariance")
    temporal = _law_covariance([values], np.ones((1, 1)), prefix)
    return values, np.asarray(sigma, dtype=float), temporal, _fraction_matrix(sigma)


def _certificate(temporal_root, spatial_root, temporal_covariance, covariance, root):
    expected, left, right, time_exponent, spatial_exponent = _rounded_product(temporal_root, spatial_root)
    if array_digest(root) != array_digest(expected):
        raise ValueError("saved shared stationary root differs from its original rounded factor products")
    scale_time = Fraction(2)**(-2*time_exponent)
    scale_spatial = Fraction(2)**(-2*spatial_exponent)
    temporal_covariance = [[v*scale_time for v in row] for row in temporal_covariance]
    covariance = [[v*scale_spatial for v in row] for row in covariance]
    temporal_error, temporal_inverse, temporal_condition, time_defect = _factor_certificate(left, temporal_covariance)
    spatial_error, spatial_inverse, spatial_condition, spatial_defect = _factor_certificate(right, covariance)
    prefix, channels = len(temporal_root), len(spatial_root)
    size = prefix*channels
    epsilon = Fraction.from_float(float(np.finfo(float).eps))
    underflow = Fraction.from_float(float(np.nextafter(0., 1.)))
    rounding = epsilon*temporal_condition*spatial_condition+size*underflow*temporal_inverse*spatial_inverse
    if rounding >= 1:
        raise UnidentifiableFitError("shared stationary product rounding is unresolved")
    factor_error = (Fraction.from_float(_sqrt_upper(Fraction(channels)))*temporal_error
        +Fraction.from_float(_sqrt_upper(Fraction(prefix)))*spatial_error+temporal_error*spatial_error)
    bound = _upper((factor_error+2*rounding+rounding**2)/(1-rounding)**2)
    policy = 1e-8*np.sqrt(size)
    if not np.isfinite(bound) or bound > policy:
        raise UnidentifiableFitError("compiled prefix state does not obey the declared stationary noise model")
    return {"temporal_law_pullback_F_bound": _upper(temporal_error),
        "spatial_law_pullback_F_bound": _upper(spatial_error), "temporal_inverse_defect_F_bound": _upper(time_defect),
        "spatial_inverse_defect_F_bound": _upper(spatial_defect), "product_rounding_F_bound": _upper(rounding),
        "joint_law_pullback_F_bound": bound, "original_joint_policy_F_bound": float(policy),
        "proof": _PROOF}


def compile_shared_stationary_root(reflections, covariance, prefix):
    """Create a new producer's root; never substitute this for a saved root."""
    values, sigma, temporal_law, spatial_law = _law(reflections, covariance, prefix)
    try:
        with localcontext(_CONTEXT):
            temporal_root, spatial_root = _cholesky(temporal_law), _cholesky(spatial_law)
        root, *_ = _rounded_product(temporal_root, spatial_root)
        certificate = _certificate(temporal_root, spatial_root, temporal_law, spatial_law, root)
    except (DecimalException, OverflowError, DictionaryArtifactError) as error:
        raise UnidentifiableFitError("shared stationary law is not representable") from error
    state = {"schema": SCHEMA, "program": PROGRAM, "source_sha256": SOURCE_SHA256,
        "factor_law_source_sha256": LAW_SOURCE_SHA256,
        "scalar_rounding_source_sha256": COMPILER_SOURCE_SHA256,
        "law": _LAW,
        "row_order": "native_time_then_observed_channel", "prefix_samples": prefix, "channels": len(sigma),
        "input_law_sha256": metadata_digest({"PARCOR": [values.tolist()]*len(sigma),
            "innovation_covariance": sigma.tolist(), "prefix_samples": prefix}),
        "temporal_root": temporal_root.tolist(), "spatial_root": spatial_root.tolist(),
        "compiled_root_sha256": array_digest(root), "certificate": certificate}
    return root, state


def admit_shared_stationary_root(reflections, covariance, prefix, root, state):
    """Verify saved factors/products/law without generating any new factor.

    Source digests identify the original producer. They need not equal this
    consumer's source bytes; fixed semantics and exact witnesses are checked.
    """
    values, sigma, temporal_law, spatial_law = _law(reflections, covariance, prefix)
    if not isinstance(state, dict) or set(state) != _STATE_KEYS:
        raise ValueError("saved shared stationary factor fields are invalid")
    for key in ("source_sha256", "factor_law_source_sha256", "scalar_rounding_source_sha256"):
        require_sha256(state[key], "original shared stationary compiler")
    expected_law = metadata_digest({"PARCOR": [values.tolist()]*len(sigma),
        "innovation_covariance": sigma.tolist(), "prefix_samples": prefix})
    if (state["schema"] != SCHEMA or state["program"] != PROGRAM or state["law"] != _LAW
            or state["row_order"] != "native_time_then_observed_channel"
            or type(state["prefix_samples"]) is not int or type(state["channels"]) is not int
            or state["prefix_samples"] != prefix or state["channels"] != len(sigma)
            or state["input_law_sha256"] != expected_law or state["compiled_root_sha256"] != array_digest(root)):
        raise ValueError("saved shared stationary factor differs from its original parameters, coordinates or root")
    temporal_root = finite_real_array(state["temporal_root"], name="saved temporal factor")
    spatial_root = finite_real_array(state["spatial_root"], name="saved spatial factor")
    root = finite_real_array(root, name="saved shared stationary root")
    for factor, size in ((temporal_root, prefix), (spatial_root, len(sigma)), (root, prefix*len(sigma))):
        if (factor.shape != (size, size) or not np.isfinite(factor).all()
                or np.any(np.diag(factor) <= 0) or np.any(np.triu(factor, 1) != 0)):
            raise ValueError("saved shared stationary component root has invalid observed support")
    try:
        actual = _certificate(temporal_root, spatial_root, temporal_law, spatial_law, root)
    except (UnidentifiableFitError, DecimalException, OverflowError, DictionaryArtifactError) as error:
        raise ValueError("saved shared stationary factor law cannot be admitted") from error
    if metadata_digest(state["certificate"]) != metadata_digest(actual):
        raise ValueError("saved shared stationary factor certificate differs from its exact input/root witnesses")
    return root


def _difference_norm_upper(left, right):
    """Bound the exact Frobenius distance between two binary64 arrays.

    For RN subtraction, |x-y| <= (|fl(x-y)|+u)/(1-eps). A product and
    nonnegative reduction have at most m+1 relative rounding factors along
    any path, and at most 2m absolute gradual-underflow errors. No accuracy
    of BLAS or a floating-point norm is assumed.
    """
    count = left.size
    epsilon = Fraction.from_float(float(np.finfo(float).eps))
    underflow = Fraction.from_float(float(np.nextafter(0., 1.)))
    relative = (count+1)*epsilon
    if relative >= Fraction(1, 2):
        raise UnidentifiableFitError("saved stationary root distance is unresolved")
    gamma = relative/(1-relative)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        difference = left-right
        squared_sum = float(np.sum(difference*difference))
    if not math.isfinite(squared_sum):
        raise UnidentifiableFitError("saved stationary root distance is not finitely representable")
    squared_upper = (Fraction.from_float(squared_sum)+2*count*underflow)/(1-gamma)
    norm = Fraction.from_float(_sqrt_upper(squared_upper))
    return (norm+Fraction.from_float(_sqrt_upper(Fraction(count)))*underflow)/(1-epsilon)


def admit_saved_shared_stationary_root(reflections, covariance, prefix, root):
    """Certify an unchanged saved /3 root using only small law witnesses.

    A and B are newly computed numerical witnesses, not replacement state.
    With R0=A tensor B, W=round(R0), S=the saved root, certify
    rho >= ||R0^-1(S-R0)||F from the rounding of W and ||S-W||F.
    The returned consumer certificate never enters the original /3 record.
    An unresolved conservative bound leaves canonical admission available.
    """
    values, sigma, temporal_law, spatial_law = _law(reflections, covariance, prefix)
    saved = finite_real_array(root, name="original saved shared stationary root")
    size = prefix*len(sigma)
    if (saved.shape != (size, size) or np.any(np.diag(saved) <= 0)
            or np.any(np.triu(saved, 1) != 0)):
        raise ValueError("original saved stationary root requires its positive lower observed support")
    try:
        with localcontext(_CONTEXT):
            temporal_root, spatial_root = _cholesky(temporal_law), _cholesky(spatial_law)
        witness, left, right, time_exponent, spatial_exponent = _rounded_product(temporal_root, spatial_root)
        exponent = time_exponent+spatial_exponent
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            normalized_saved = np.ldexp(saved, -exponent)
            restored = np.ldexp(normalized_saved, exponent)
            normalized_witness = np.ldexp(witness, -exponent)
        if not np.isfinite(normalized_saved).all() or not np.array_equal(restored, saved):
            raise UnidentifiableFitError("saved stationary root witness coordinates are unresolved")
        temporal_law = [[v*Fraction(2)**(-2*time_exponent) for v in row] for row in temporal_law]
        spatial_law = [[v*Fraction(2)**(-2*spatial_exponent) for v in row] for row in spatial_law]
        time_error, time_inverse, time_condition, _ = _factor_certificate(left, temporal_law)
        spatial_error, spatial_inverse, spatial_condition, _ = _factor_certificate(right, spatial_law)
        epsilon = Fraction.from_float(float(np.finfo(float).eps))
        underflow = Fraction.from_float(float(np.nextafter(0., 1.)))
        inverse_product = time_inverse*spatial_inverse
        product_rounding = epsilon*time_condition*spatial_condition+size*underflow*inverse_product
        distance = _difference_norm_upper(normalized_saved, normalized_witness)
        perturbation = product_rounding+inverse_product*distance
        if perturbation >= 1:
            raise UnidentifiableFitError("saved stationary root relative perturbation is unresolved")
        factor_error = (Fraction.from_float(_sqrt_upper(Fraction(len(sigma))))*time_error
            +Fraction.from_float(_sqrt_upper(Fraction(prefix)))*spatial_error+time_error*spatial_error)
        bound = _upper((factor_error+2*perturbation+perturbation**2)/(1-perturbation)**2)
    except (DecimalException, OverflowError, DictionaryArtifactError) as error:
        raise UnidentifiableFitError("saved shared stationary root witness is not representable") from error
    policy = float(1e-8*np.sqrt(size))
    if bound > policy:
        raise UnidentifiableFitError("saved stationary root witness does not resolve the original law policy")
    return {"program": "shared_PARCOR_saved_root_perturbation_admission/1",
        "consumer_source_sha256": SOURCE_SHA256, "law_source_sha256": LAW_SOURCE_SHA256,
        "input_law_sha256": metadata_digest({"PARCOR": [values.tolist()]*len(sigma),
            "innovation_covariance": sigma.tolist(), "prefix_samples": prefix}),
        "saved_root_sha256": array_digest(saved), "witness_root_sha256": array_digest(witness),
        "normalized_root_distance_F_bound": _upper(distance),
        "factor_law_pullback_F_bound": _upper(factor_error),
        "small_inverse_F_product_bound": _upper(inverse_product),
        "witness_product_rounding_F_bound": _upper(product_rounding),
        "saved_root_relative_perturbation_F_bound": _upper(perturbation),
        "saved_root_law_pullback_F_bound": bound, "original_joint_policy_F_bound": policy,
        "state_role": "consumer_witness_only_original_saved_root_and_codec_unchanged"}
