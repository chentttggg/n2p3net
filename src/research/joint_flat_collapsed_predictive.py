"""Rao-Blackwellize response and drift from the captured P-only chain.

Retain original lambda/all-other-target draws, integrate the ONE proper beta
prior and every physical flat drift through augmented roots. The original
joint startup stays in the flat block; no Kronecker startup replacement.
"""
from __future__ import annotations

import hashlib
import sys
from collections import OrderedDict

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from research.channel_noise_mechanism import (
    DiffuseProjectionResolutionError,
    _column_norms,
    _inverse_root_geometry,
    _roundoff_gamma,
)
from research.joint_flat_factor_history import FactoredQRTree
from research.joint_response_scale import _augmented_qr
from transfer.noise_model import frozen_array


class _UnweightedQRCache:
    """One component call's bounded cache of deterministic, unweighted QR operands."""

    def __init__(self, capacity_bytes):
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError('unweighted QR cache requires a nonnegative byte capacity')
        self.capacity_bytes, self.used_bytes = capacity_bytes, 0
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def __call__(self, rows, chunk_rows, *, with_error):
        if not self.capacity_bytes:
            self.misses += 1
            return _augmented_qr(rows, chunk_rows, with_error=with_error)
        content = hashlib.sha256(memoryview(np.ascontiguousarray(rows)).cast('B')).hexdigest()
        key = (rows.shape, rows.dtype.str, content, chunk_rows, with_error)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        result = _augmented_qr(rows, chunk_rows, with_error=with_error)
        arrays = result if with_error else (result,)
        used = sum(value.nbytes for value in arrays)
        if used <= self.capacity_bytes:
            for value in arrays:
                value.setflags(write=False)
            while self.used_bytes+used > self.capacity_bytes:
                _, old = self.entries.popitem(last=False)
                self.used_bytes -= sum(value.nbytes for value in (old if isinstance(old, tuple) else (old,)))
            self.entries[key] = result
            self.used_bytes += used
        return result


def _retained_size(value):
    size = sys.getsizeof(value)
    if isinstance(value, tuple):
        size += sum(_retained_size(item) for item in value)
    return size


def _marginal_array_key(values):
    if type(values) is not np.ndarray or values.dtype.hasobject:
        return None
    content = hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast('B')).digest()
    return values.shape, values.dtype.str, values.strides, content


class _PrefixMarginalCache:
    """One component call's P-only scalar results under frozen native arithmetic.

    Ordered numerical operands and startup context are observed on each call.
    No entry survives the component call or substitutes a weighted F integral.
    The retained-byte charge includes keys, results and a mapping allowance.
    """

    def __init__(self, capacity_bytes):
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError('P marginal cache requires a nonnegative byte capacity')
        self.capacity_bytes, self.used_bytes = capacity_bytes, 0
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def __call__(self, trees, precision, *, prior_covariance_error=0.):
        # These bypasses preserve the original argument handling and named
        # startup rejection. In particular, hashing never consumes an iterator.
        if (not self.capacity_bytes or type(precision) is not np.ndarray
                or precision.dtype.hasobject
                or any(getattr(tree.record, 'startup_response_design', False) is not None for tree in trees)):
            self.misses += 1
            return _marginal(trees, precision, prior_covariance_error=prior_covariance_error)
        context = []
        for tree in trees:
            record = tree.record
            arrays = tuple(_marginal_array_key(value) for value in (
                tree.root, tree.errors[1], record.startup_rows, record.startup_errors))
            if any(key is None for key in arrays):
                self.misses += 1
                return _marginal(trees, precision, prior_covariance_error=prior_covariance_error)
            record_id = getattr(record, 'record_id', None)
            context.append((record_id if isinstance(record_id, str) else None,
                record.response_width, record.channels, record.drift_width, arrays))
        prior = np.asarray(prior_covariance_error)
        if prior.ndim or prior.dtype.hasobject:
            self.misses += 1
            return _marginal(trees, precision, prior_covariance_error=prior_covariance_error)
        key = (tuple(context), _marginal_array_key(precision), _marginal_array_key(prior))
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key][0]
        self.misses += 1
        result = _marginal(trees, precision, prior_covariance_error=prior_covariance_error)
        # Conservatively charge repeated shared Python objects separately.
        # The allowance covers each live OrderedDict node and its hash slot.
        used = _retained_size(key)+_retained_size(result)+sys.getsizeof((result, 0))+sys.getsizeof(0)+512
        if used <= self.capacity_bytes:
            while self.used_bytes+used > self.capacity_bytes:
                _, (_, retired_bytes) = self.entries.popitem(last=False)
                self.used_bytes -= retired_bytes
            self.entries[key] = result, used
            self.used_bytes += used
        return result


def _marginal(trees, precision, *, extra=None, prior_covariance_error=0., unweighted_qr=None):
    k, c = trees[0].record.response_width, trees[0].record.channels
    flat_width = sum(tree.record.drift_width for tree in trees)
    prior = np.column_stack((np.eye(k), np.zeros((k, flat_width+c))))
    pieces, source_errors = [prior], np.zeros(k+flat_width+c)
    flat_start = 0
    for index, (tree, lam) in enumerate(zip(trees, precision, strict=True)):
        r, root, errors = tree.record.drift_width, tree.root, tree.errors[1]
        if extra is not None and index == extra[0]:
            values, bounds = extra[1:]
            compress = _augmented_qr if unweighted_qr is None else unweighted_qr
            tail, tail_errors = compress(values, max(1, len(values)), with_error=True)
            root, qr_errors = compress(np.vstack((root, tail)), max(len(root), len(tail)), with_error=True)
            errors = errors+_column_norms(bounds)+tail_errors+qr_errors
        rows = np.zeros((len(root), k+flat_width+c))
        rows[:, :k], rows[:, k+flat_start:k+flat_start+r], rows[:, k+flat_width:] = root[:, r:r+k], root[:, :r], root[:, r+k:]
        scale = np.sqrt(lam)
        weighted = scale*rows
        pieces.append(weighted)
        source_errors[:k] += scale*errors[r:r+k]
        source_errors[k+flat_start:k+flat_start+r] += scale*errors[:r]
        source_errors[k+flat_width:] += scale*errors[r+k:]
        source_errors += _roundoff_gamma(2)*_column_norms(weighted)
        flat_start += r
    beta_root, beta_errors = _augmented_qr(np.vstack(pieces), k+flat_width+c, with_error=True)
    source_errors += beta_errors
    rb = beta_root[:k, :k]
    inverse_beta, _ = _inverse_root_geometry(rb)
    beta_metric = float(np.linalg.norm(source_errors[:k]@inverse_beta))
    if not np.isfinite(beta_metric) or beta_metric >= 1:
        raise DiffuseProjectionResolutionError('collapsed proper response root is unresolved')
    # Integrate beta first. Each whitened channel is one response column; its
    # prior root occurs once. Its residual flat block need not be independent
    # across channels after the original joint startup has been included.
    flat_tail = np.column_stack((np.kron(beta_root[k:, k:k+flat_width], np.eye(c)),
                                beta_root[k:, k+flat_width:].ravel()))
    flat_errors = np.r_[np.repeat(source_errors[k:k+flat_width], c),
                        float(np.linalg.norm(source_errors[k+flat_width:]))]
    rows = [flat_tail]
    flat_start = 0
    for tree, lam in zip(trees, precision, strict=True):
        record = tree.record
        if record.startup_response_design is not None:
            raise ValueError('this collapsed shared-source route requires structural startup response zero')
        leading = np.zeros((len(record.startup_rows), flat_width*c+1))
        leading[:, flat_start*c:(flat_start+record.drift_width)*c] = record.startup_rows[:, :-1]
        leading[:, -1] = record.startup_rows[:, -1]
        scale = np.sqrt(lam)
        rows.append(scale*leading)
        errors = _column_norms(record.startup_errors)
        flat_errors[flat_start*c:(flat_start+record.drift_width)*c] += scale*errors[:-1]
        flat_errors[-1] += scale*errors[-1]
        flat_errors += _roundoff_gamma(2)*_column_norms(scale*leading)
        flat_start += record.drift_width
    flat_root, qr_errors = _augmented_qr(np.vstack(rows), flat_width*c+1, with_error=True)
    flat_errors += qr_errors
    width = flat_width*c
    ra, rhs = flat_root[:width, :width], flat_root[:width, width]
    inverse_a, _ = _inverse_root_geometry(ra)
    metric = float(np.linalg.norm(flat_errors[:width]@inverse_a))
    if not np.isfinite(metric) or metric >= 1:
        raise DiffuseProjectionResolutionError('collapsed original joint-startup drift root is unresolved')
    drift = solve_triangular(ra, rhs, lower=False)
    drifts = drift.reshape(flat_width, c)
    beta_mean = solve_triangular(rb, beta_root[:k, k+flat_width:]-beta_root[:k, k:k+flat_width]@drifts,
                                 lower=False)
    beta_direction = float(np.linalg.norm(source_errors[:k]@np.abs(beta_mean)))
    error = float(flat_errors[:width]@np.abs(drift)+flat_errors[-1])+beta_direction
    error += _roundoff_gamma(width+1)*float(np.linalg.norm(np.abs(flat_root)@np.r_[np.abs(drift), 1.]))
    energy = float(np.linalg.norm(flat_root@np.r_[drift, -1.])**2)
    # Full triangular geometry is [Rb⊗I, Rba⊗I; 0, Ra]. Apply its inverse
    # error bound in blocks, without creating a pC×pC inverse/SVD per q.
    beta_inverse_error = np.repeat(source_errors[:k]@inverse_beta, c)
    flat_inverse_error = (flat_errors[:width]+beta_inverse_error@np.kron(
        np.abs(beta_root[:k, k:k+flat_width]), np.eye(c)))@inverse_a
    full_metric = float(np.linalg.norm(np.r_[beta_inverse_error, flat_inverse_error]))
    if not np.isfinite(full_metric) or full_metric >= 1:
        raise DiffuseProjectionResolutionError('collapsed full proper/flat posterior source metric is unresolved')
    solve_error = float(np.linalg.norm(ra@drift-rhs))
    solve_error += _roundoff_gamma(width)*float(np.linalg.norm(np.abs(ra)@np.abs(drift)))
    beta_rhs = beta_root[:k, k+flat_width:]-beta_root[:k, k:k+flat_width]@drifts
    solve_error += float(np.linalg.norm(rb@beta_mean-beta_rhs))
    solve_error += _roundoff_gamma(k+flat_width+1)*float(np.linalg.norm(
        np.abs(rb)@np.abs(beta_mean)+np.abs(beta_root[:k, k:k+flat_width])@np.abs(drifts)
        +np.abs(beta_root[:k, k+flat_width:])))
    displacement = ((1+full_metric)*error+full_metric*np.sqrt(energy)+solve_error)/(1-full_metric)**2
    error += full_metric*displacement+solve_error
    quadratic_error = 2*np.sqrt(energy)*error+error**2+_roundoff_gamma(len(flat_root))*energy
    lb, la = np.log(np.abs(np.diag(rb))), np.log(np.abs(np.diag(ra)))
    logdet = c*float(lb.sum())+float(la.sum())
    determinant_error = -(c*k+width)*np.log1p(-full_metric)
    determinant_error += _roundoff_gamma(k+width+3)*(c*float(np.abs(lb).sum())+float(np.abs(la).sum()))
    value = -.5*energy-logdet
    nats_error = .5*quadratic_error+determinant_error+_roundoff_gamma(2)*(.5*energy+abs(logdet))
    if prior_covariance_error:
        delta = prior_covariance_error
        t = delta/(1-delta)
        if not t < 1:
            raise DiffuseProjectionResolutionError('integrated original proper-prior covariance relation is unresolved')
        # Marginal proper posterior covariance in unit U coordinates is <=I.
        nats_error += .5*k*c*(-np.log1p(-delta)-np.log1p(-t))
        drift_solve = np.abs(ra@drift-rhs)+_roundoff_gamma(width)*(np.abs(ra)@np.abs(drift))
        drift_error = inverse_a@drift_solve
        beta_solve = np.abs(rb@beta_mean-beta_rhs)
        beta_solve += _roundoff_gamma(k+flat_width+1)*(
            np.abs(rb)@np.abs(beta_mean)+np.abs(beta_root[:k, k:k+flat_width])@np.abs(drifts)
            +np.abs(beta_root[:k, k+flat_width:]))
        beta_solve += np.abs(beta_root[:k, k:k+flat_width])@drift_error.reshape(flat_width, c)
        beta_mean_error = float(np.linalg.norm(inverse_beta@beta_solve))
        nats_error += .5*t*(float(np.linalg.norm(beta_mean))+beta_mean_error)**2/(1-t)
    if not np.isfinite(nats_error):
        raise DiffuseProjectionResolutionError('collapsed original Gaussian marginal error is not finite')
    return value, nats_error


def collapsed_tail_components(records, tail_rows, tail_errors, axes, samples, *, current_axis,
        tail_native_start, tail_noise_logdet, tail_noise_logdet_error, leaf_rows, cache_bytes,
        prior_covariance_error=0., cache_report=None):
    """P-only marginal lambda/other-q measure; all current q have actual mass."""
    local_axes = [tuple(axis for axis in axes if axis.record_index == index) for index in range(len(records))]
    unweighted_qr = _UnweightedQRCache(cache_bytes)
    prefix_marginal = _PrefixMarginalCache(cache_bytes)
    trees = [FactoredQRTree(record, local, leaf_rows=leaf_rows, cache_bytes=cache_bytes)
             for record, local in zip(records, local_axes, strict=True)]
    local = {index: sum(axis.record_index == axes[index].record_index for axis in axes[:index]) for index in range(len(axes))}
    current = axes[current_axis]
    positive = np.flatnonzero(current.prior_mass > 0)
    shape = (*samples.coefficients.shape[:2], len(current.prior_mass))
    weights, densities = np.full(shape, -np.inf), np.full(shape, -np.inf)
    k, r, c = records[current.record_index].response_width, records[current.record_index].drift_width, records[current.record_index].channels
    for chain in range(shape[0]):
        for draw in range(shape[1]):
            for index, axis in enumerate(axes):
                if index != current_axis:
                    trees[axis.record_index].set_target(local[index], int(samples.targets[chain, draw, index]))
            precision = samples.precisions[chain, draw]
            lambda_current = precision[current.record_index]
            logits = np.full(len(current.prior_mass), -np.inf)
            prefix_errors = np.zeros(len(current.prior_mass))
            for q in positive:
                trees[current.record_index].set_target(local[current_axis], int(q))
                value_p, error_p = prefix_marginal(trees, precision, prior_covariance_error=prior_covariance_error)
                actual_tail, actual_errors = tail_rows.copy(), tail_errors.copy()
                for index, axis in enumerate(axes):
                    if axis.record_index != current.record_index:
                        continue
                    state = q if index == current_axis else int(samples.targets[chain, draw, index])
                    value, error = axis.tail_images[state].block(tail_native_start, tail_native_start+len(actual_tail))
                    actual_errors[:, r:r+k] += error[:, :k]+_roundoff_gamma(1)*(
                        np.abs(actual_tail[:, r:r+k])+np.abs(value[:, :k]))
                    actual_errors[:, r+k:] += error[:, k:]+_roundoff_gamma(1)*(
                        np.abs(actual_tail[:, r+k:])+np.abs(value[:, k:]))
                    actual_tail[:, r:r+k] += value[:, :k]
                    actual_tail[:, r+k:] -= value[:, k:]
                value_f, error_f = _marginal(trees, precision,
                    extra=(current.record_index, actual_tail, actual_errors), prior_covariance_error=prior_covariance_error,
                    unweighted_qr=unweighted_qr)
                normalizer = len(actual_tail)*c*(np.log(2*np.pi)-np.log(lambda_current))+tail_noise_logdet
                normalizer_error = tail_noise_logdet_error+_roundoff_gamma(4)*(
                    len(actual_tail)*c*(abs(np.log(2*np.pi))+abs(np.log(lambda_current)))+abs(tail_noise_logdet))
                error = error_p+error_f+.5*normalizer_error+_roundoff_gamma(3)*(
                    abs(value_p)+abs(value_f)+abs(normalizer))
                if not np.isfinite(error) or error > np.sqrt(np.finfo(float).eps)*len(actual_tail)*c:
                    raise DiffuseProjectionResolutionError('collapsed original raw-T conditional density is unresolved in nats')
                logits[q] = np.log(current.prior_mass[q])+value_p
                prefix_errors[q] = error_p+_roundoff_gamma(3)*(abs(np.log(current.prior_mass[q]))+abs(value_p))
                densities[chain, draw, q] = value_f-value_p-.5*normalizer
            weights[chain, draw] = logits-logsumexp(logits)
            weight_error = 2*float(np.max(prefix_errors[positive]))+_roundoff_gamma(len(positive)+4)*(
                float(np.max(np.abs(logits[positive])))+abs(float(logsumexp(logits)))+np.log(len(positive)))
            budget = np.sqrt(np.finfo(float).eps)*sum(record.scalar_rows for record in records)
            if not np.isfinite(weight_error) or weight_error > budget:
                raise DiffuseProjectionResolutionError('complete normalized collapsed current-q weights are unresolved in nats')
    if cache_report is not None:
        cache_report.update({'scope': 'unweighted_tail_and_merge_QR_only_this_component_call',
            'hits': unweighted_qr.hits, 'misses': unweighted_qr.misses,
            'entries': len(unweighted_qr.entries), 'payload_bytes': unweighted_qr.used_bytes,
            'capacity_bytes': unweighted_qr.capacity_bytes})
    return frozen_array(weights), frozen_array(densities)
