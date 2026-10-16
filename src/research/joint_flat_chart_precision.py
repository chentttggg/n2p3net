"""Certify fixed saved source coordinates without changing the prior factor.

For K=T tensor S, choose the small orthogonal witness Q=Q_T tensor I.
With E=K^-1(L_old-KQ), Cov(U)=(Q+E)(Q+E)^T. Accurate stored-input
residuals certify E; the source's original L_old, mean and noise stay fixed.
"""
from __future__ import annotations

import math

import numpy as np

from research.channel_noise_mechanism import (
    DiffuseProjectionResolutionError,
    _inverse_root_geometry,
    _roundoff_gamma,
)


def _product_remainder(left, right, product):
    if not np.isfinite(product).all():
        raise DiffuseProjectionResolutionError('saved chart witness products are not representable')
    if hasattr(math, 'fma'):
        remainder = np.asarray([math.fma(float(a), float(b), -float(c)) for a, b, c in zip(
            np.broadcast_to(left, product.shape).ravel(), np.broadcast_to(right, product.shape).ravel(),
            product.ravel(), strict=True)]).reshape(product.shape)
    else:
        splitter = 134217729.
        sl, sr = splitter*left, splitter*right
        if not np.isfinite(sl).all() or not np.isfinite(sr).all():
            raise DiffuseProjectionResolutionError('saved chart witness splitting is not representable')
        hl, hr = sl-(sl-left), sr-(sr-right)
        ll, lr = left-hl, right-hr
        remainder = ((hl*hr-product)+hl*lr+ll*hr)+ll*lr
    if not np.isfinite(remainder).all():
        raise DiffuseProjectionResolutionError('saved chart witness product remainder is not finite')
    return remainder


def _stored_product_residual(left, right, expected):
    """expected-left@right, using both product limbs and an fsum reduction."""
    result, error = np.empty_like(expected), np.empty_like(expected)
    tiny, eps = np.finfo(float).tiny, np.finfo(float).eps
    for row in range(len(left)):
        for column in range(right.shape[1]):
            a, b = left[row], right[:, column]
            product = a*b
            remainder = _product_remainder(a, b, product)
            try:
                value = math.fsum([float(expected[row, column]), *map(float, -product), *map(float, -remainder)])
            except OverflowError as cause:
                raise DiffuseProjectionResolutionError('saved chart witness residual is not representable') from cause
            result[row, column] = value
            error[row, column] = 2*eps*abs(value)+2*len(a)*tiny
    if not np.isfinite(result).all() or not np.isfinite(error).all():
        raise DiffuseProjectionResolutionError('saved chart witness residual or bound is not finite')
    return result, error


def _upper_frobenius(values):
    if values.size == 0:
        return 0.
    scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale):
        raise DiffuseProjectionResolutionError('saved chart witness norm is not finite')
    if scale == 0:
        return 0.
    if scale < np.finfo(float).tiny:
        raise DiffuseProjectionResolutionError('saved chart witness norm is not normally representable')
    relative = np.abs(values)/scale
    squared = (float(np.sum(relative*relative))+values.size*np.finfo(float).tiny)
    squared /= 1-_roundoff_gamma(2*values.size+3)
    value = scale*float(np.nextafter(np.sqrt(squared), np.inf))
    if not np.isfinite(value):
        raise DiffuseProjectionResolutionError('saved chart witness norm is not representable')
    return float(np.nextafter(value, np.inf))


def balanced_gram_certificate(values):
    """Stored Q.T Q with one product and a balanced row sum on every path."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise DiffuseProjectionResolutionError('balanced Gram requires finite nonempty stored operands')
    rows, width = values.shape
    result, absolute = np.empty((width, width)), np.empty((width, width))
    for start in range(0, width, 8):
        product = values[:, :, None]*values[:, None, start:start+8]
        positive = np.abs(product)
        while len(product) > 1:
            count = len(product)//2
            signed = product[:2*count:2]+product[1:2*count:2]
            unsigned = positive[:2*count:2]+positive[1:2*count:2]
            if len(product)%2:
                signed = np.concatenate((signed, product[-1:]))
                unsigned = np.concatenate((unsigned, positive[-1:]))
            product, positive = signed, unsigned
        result[:, start:start+8], absolute[:, start:start+8] = product[0], positive[0]
    gamma = _roundoff_gamma(1+(rows-1).bit_length())
    # Both sums contain the same rounded products. Include multiplication,
    # the actual sum depth, and possible product underflow on each scalar.
    error = (gamma*absolute+2*rows*np.finfo(float).tiny)/(1-gamma)
    error = np.nextafter(error/(1-_roundoff_gamma(4)), np.inf)
    if not np.isfinite(result).all() or not np.isfinite(error).all():
        raise DiffuseProjectionResolutionError('balanced Gram witness is not representable')
    return result, error


def saved_source_chart_certificate(temporal, spatial, prior_root):
    """A consumer certificate of the original stored K/L relation, without fits."""
    temporal, spatial, prior_root = (np.asarray(value, dtype=float) for value in (temporal, spatial, prior_root))
    k, c = len(temporal), len(spatial)
    if (temporal.shape != (k, k) or spatial.shape != (c, c) or prior_root.shape != (k*c, k*c)
            or not k or not c or any(not np.isfinite(value).all() for value in (temporal, spatial, prior_root))
            or np.any(np.tril(temporal, -1) != 0) or np.any(np.triu(spatial, 1) != 0)
            or np.any(np.diag(temporal) == 0) or np.any(np.diag(spatial) == 0)):
        raise ValueError('original saved prior chart requires finite invertible temporal/spatial triangular factors')
    q, upper = np.linalg.qr(temporal.T)
    q *= np.where(np.diag(upper) < 0, -1., 1.)[None, :]
    residual_q, error_q = _stored_product_residual(q, q.T, np.eye(k))
    orthogonal_error = _upper_frobenius(np.abs(residual_q)+error_q)
    temporal_image = temporal@q
    residual_t, error_t = _stored_product_residual(temporal, q, temporal_image)
    left, right = temporal_image[:, None, :, None], spatial[None, :, None, :]
    product = left*right
    remainder = _product_remainder(left, right, product)
    correction = np.kron(residual_t, spatial)
    arithmetic_error = np.kron(error_t, np.abs(spatial))+_roundoff_gamma(1)*np.abs(correction)
    operands = (prior_root.ravel(), -product.ravel(), -remainder.ravel(), correction.ravel())
    try:
        difference = np.asarray([math.fsum(map(float, values)) for values in zip(*operands, strict=True)]).reshape(k*c, k*c)
    except OverflowError as cause:
        raise DiffuseProjectionResolutionError('saved prior minus chart witness is not representable') from cause
    arithmetic_error += 2*np.finfo(float).eps*np.abs(difference)+4*np.finfo(float).tiny
    inverse_temporal, _ = _inverse_root_geometry(temporal)
    inverse_spatial, _ = _inverse_root_geometry(spatial, lower=True)
    bound = (np.abs(difference)+arithmetic_error).reshape(k, c, k*c)
    spatial_error = np.einsum('cd,kdn->kcn', inverse_spatial, bound)/(1-_roundoff_gamma(c))
    full_error = np.einsum('ik,kcn->icn', inverse_temporal, spatial_error)/(1-_roundoff_gamma(k))
    pulled_error = _upper_frobenius(full_error)
    # ||Q+E||^2 differs from I by QQ^T-I+QE^T+EQ^T+EE^T.
    relation = orthogonal_error+2*np.sqrt(1+orthogonal_error)*pulled_error+pulled_error*pulled_error
    relation = float(np.nextafter(relation/(1-_roundoff_gamma(8)), np.inf))
    if not np.isfinite(relation) or relation >= np.sqrt(np.finfo(float).eps):
        raise DiffuseProjectionResolutionError('original saved source prior is unresolved in its fixed chart')
    return {'schema': 'n2p3_joint_flat_saved_prior_chart_certificate/1',
        'method': 'small_orthogonal_Kronecker_witness_and_compensated_stored_operand_residual',
        'covariance_relation_error': relation, 'orthogonal_witness_error': orthogonal_error,
        'pulled_back_prior_factor_error': pulled_error,
        'prior_minus_KQ_residual_norm': _upper_frobenius(difference),
        'prior_minus_KQ_arithmetic_error_norm': _upper_frobenius(arithmetic_error),
        'original_prior_root_preserved': True, 'new_parameter_fits': 0}
