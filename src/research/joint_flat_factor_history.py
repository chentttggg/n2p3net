"""Streamed shared-source history without native x channel x beta storage.

The original source law has one temporal response covariance factor and the
same spatial innovation covariance. Its proper unit chart is therefore a
temporal coefficient matrix with measured channels as columns. Original joint
startup rows are kept separately; no Kronecker replacement of their saved map
is made. The fast kernel requires structurally zero response in startup.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from research.channel_noise_mechanism import (
    DiffuseProjectionResolutionError,
    _column_norms,
    _inverse_root_geometry,
    _response_resolved_root,
    _roundoff_gamma,
)
from research.joint_flat_chain import JointHistorySamples
from research.joint_flat_chart_precision import _upper_frobenius, balanced_gram_certificate
from research.joint_response_scale import _augmented_qr as _original_augmented_qr
from research.joint_response_scale import _degrees, _numeric
from transfer.noise_model import finite_real_array, frozen_array


def _positive_add(left, right):
    value = np.asarray(left)+right
    return np.where(value == 0, 0., np.nextafter(value, np.inf))


def _augmented_qr(rows, chunk_rows, *, with_error=False):
    """The original Q/R, with a short balanced Gram witness for its polar error."""
    if not with_error:
        return _original_augmented_qr(rows, chunk_rows)
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError('QR work-block size must be a positive integer')
    root, errors = np.empty((0, rows.shape[1])), np.zeros(rows.shape[1])
    for start in range(0, len(rows), chunk_rows):
        block = np.vstack((root, rows[start:start+chunk_rows]))
        q, root = np.linalg.qr(block, mode='reduced')
        reconstruction = _column_norms(block-q@root)
        reconstruction += _roundoff_gamma(root.shape[0]+1)*_column_norms(np.abs(q)@np.abs(root))
        gram, gram_error = balanced_gram_certificate(q)
        defect = _upper_frobenius(gram-np.eye(q.shape[1]))+_upper_frobenius(gram_error)
        if not np.isfinite(defect) or defect >= 1:
            raise DiffuseProjectionResolutionError('augmented QR has unresolved balanced orthogonality')
        polar_error = defect/(1+np.sqrt(1-defect))
        error = reconstruction+polar_error*_column_norms(root)
        errors = _positive_add(errors, error/(1-_roundoff_gamma(2*root.size+8)))
    _numeric(root, name='augmented QR root')
    return root, errors


@dataclass(frozen=True)
class PulseTargetImage:
    """A physical event subset and one shared finite white pulse, not J images."""
    start: int
    stop: int
    onsets: tuple[int, ...]
    pulse: np.ndarray
    pulse_errors: np.ndarray
    coefficient_map: np.ndarray
    coefficient_map_errors: np.ndarray
    mean_map: np.ndarray
    mean_map_errors: np.ndarray

    def __post_init__(self):
        for name in ('pulse', 'pulse_errors', 'coefficient_map', 'coefficient_map_errors',
                     'mean_map', 'mean_map_errors'):
            object.__setattr__(self, name, frozen_array(finite_real_array(getattr(self, name), name=name)))
        if (not 0 <= self.start <= self.stop or self.pulse.ndim != 2
                or self.pulse_errors.shape != self.pulse.shape
                or self.coefficient_map.shape != self.coefficient_map_errors.shape
                or self.mean_map.shape != self.mean_map_errors.shape
                or self.coefficient_map.shape[0] != self.pulse.shape[1]
                or self.mean_map.shape[0] != self.pulse.shape[1]
                or any(onset < self.start for onset in self.onsets)):
            raise ValueError('pulse images require their exact original finite temporal and source chart')

    def block(self, lower, upper):
        """Generate only a leaf's beta/mean columns, in fixed original order."""
        temporal = np.zeros((upper-lower, self.pulse.shape[1]))
        errors = np.zeros_like(temporal)
        for onset in self.onsets:
            left, right = max(lower, onset), min(upper, onset+len(self.pulse), self.stop)
            if left < right:
                image = self.pulse[left-onset:right-onset]
                errors[left-lower:right-lower] += self.pulse_errors[left-onset:right-onset]
                errors[left-lower:right-lower] += _roundoff_gamma(1)*(
                    np.abs(temporal[left-lower:right-lower])+np.abs(image))
                temporal[left-lower:right-lower] += image
        values, bounds = [], []
        for mapping, mapping_error in ((self.coefficient_map, self.coefficient_map_errors),
                                       (self.mean_map, self.mean_map_errors)):
            value = temporal@mapping
            bound = errors@np.abs(mapping)+np.abs(temporal)@mapping_error
            bound += _roundoff_gamma(temporal.shape[1])*(np.abs(temporal)@np.abs(mapping))
            values.append(value)
            bounds.append(bound)
        return np.column_stack(values), np.column_stack(bounds)

    def prediction(self, lower, upper, coefficients):
        """Collapse source factors before time expansion at an actual draw."""
        mapping = self.coefficient_map@coefficients+self.mean_map
        bound = self.coefficient_map_errors@np.abs(coefficients)+self.mean_map_errors
        bound += _roundoff_gamma(len(coefficients)+1)*(
            np.abs(self.coefficient_map)@np.abs(coefficients)+np.abs(self.mean_map))
        value = np.zeros((upper-lower, mapping.shape[1]))
        error = np.zeros_like(value)
        for onset in self.onsets:
            left, right = max(lower, onset), min(upper, onset+len(self.pulse), self.stop)
            if left < right:
                pulse = self.pulse[left-onset:right-onset]
                image = pulse@mapping
                local = self.pulse_errors[left-onset:right-onset]@np.abs(mapping)+np.abs(pulse)@bound
                local += _roundoff_gamma(pulse.shape[1])*(np.abs(pulse)@np.abs(mapping))
                local += _roundoff_gamma(1)*(np.abs(value[left-lower:right-lower])+np.abs(image))
                error[left-lower:right-lower] += local
                value[left-lower:right-lower] += image
        return value, float(np.linalg.norm(error))


@dataclass(frozen=True)
class FactoredTargetAxis:
    selection_id: str
    candidate_vocabulary: tuple[str, ...]
    prior_mass: np.ndarray
    prefix_images: tuple[PulseTargetImage, ...]
    tail_images: tuple[PulseTargetImage, ...]
    record_index: int

    def __post_init__(self):
        mass = finite_real_array(self.prior_mass, name='complete original target prior')
        if (mass.shape != (len(self.candidate_vocabulary),) or np.any(mass < 0)
                or not np.isclose(mass.sum(), 1.) or len(self.prefix_images) != len(mass)
                or len(self.tail_images) != len(mass)):
            raise ValueError('each original selection retains its complete C-compatible vocabulary')
        object.__setattr__(self, 'prior_mass', frozen_array(mass))


@dataclass(frozen=True)
class SharedFactorRecord:
    """Original startup a rows; tail [Z_temporal, beta_temporal, Y_white]."""
    record_id: str
    channels: int
    startup_native_rows: int
    startup_rows: np.ndarray
    startup_errors: np.ndarray
    tail_rows: np.ndarray
    tail_errors: np.ndarray
    response_width: int
    drift_width: int
    startup_response_design: np.ndarray | None = None
    startup_response_errors: np.ndarray | None = None

    def __post_init__(self):
        for name in ('startup_rows', 'startup_errors', 'tail_rows', 'tail_errors'):
            object.__setattr__(self, name, frozen_array(finite_real_array(getattr(self, name), name=name)))
        if (self.channels < 1 or self.startup_native_rows < 1 or self.response_width < 1 or self.drift_width < 1
                or self.startup_rows.shape != (self.startup_native_rows*self.channels,
                                              self.drift_width*self.channels+1)
                or self.startup_errors.shape != self.startup_rows.shape
                or self.tail_rows.ndim != 2 or self.tail_rows.shape[1] !=
                    self.drift_width+self.response_width+self.channels
                or self.tail_errors.shape != self.tail_rows.shape
                or np.any(self.startup_errors < 0) or np.any(self.tail_errors < 0)):
            raise ValueError('native channel count and original startup/tail charts are required')
        if self.startup_response_design is not None:
            design = finite_real_array(self.startup_response_design, name='original startup response design')
            error = finite_real_array(self.startup_response_errors, name='original startup response errors')
            if (design.shape != (self.startup_native_rows*self.channels, self.response_width*self.channels)
                    or error.shape != design.shape or np.any(error < 0)):
                raise ValueError('the original startup must retain every proper coefficient and source error')
            object.__setattr__(self, 'startup_response_design', frozen_array(design))
            object.__setattr__(self, 'startup_response_errors', frozen_array(error))

    @property
    def native_rows(self):
        return self.startup_native_rows+len(self.tail_rows)

    @property
    def scalar_rows(self):
        return self.native_rows*self.channels


class FactoredQRTree:
    """One small temporal/multi-response root per leaf, cached by local q state."""
    def __init__(self, record, axes, *, leaf_rows, cache_bytes):
        self.record, self.axes = record, tuple(axes)
        self.states = {index: None for index in range(len(self.axes))}
        self.leaf_rows, self.cache_bytes = leaf_rows, cache_bytes
        count = max(1, (len(record.tail_rows)+leaf_rows-1)//leaf_rows)
        self.offset = 1 << (count-1).bit_length()
        self.nodes = [None]*(2*self.offset)
        self.errors = [None]*(2*self.offset)
        self.cache = OrderedDict()
        self.cache_used, self.qr_rebuilds, self.cache_hits = 0, 0, 0
        self.leaf_axes = []
        width = record.tail_rows.shape[1]
        for leaf in range(self.offset):
            left = record.startup_native_rows+leaf*leaf_rows
            right = min(record.native_rows, left+leaf_rows)
            touched = tuple(index for index, axis in enumerate(self.axes)
                if any(onset < right and min(image.stop, onset+len(image.pulse)) > left
                    for q, image in enumerate(axis.prefix_images) if axis.prior_mass[q] > 0
                    for onset in image.onsets))
            self.leaf_axes.append(touched)
            if right > left:
                self._leaf(leaf)
            else:
                self.nodes[self.offset+leaf] = np.empty((0, width))
                self.errors[self.offset+leaf] = np.zeros(width)
        for index in range(self.offset-1, 0, -1):
            self._merge(index)

    def _leaf(self, leaf):
        local = tuple((index, self.states[index]) for index in self.leaf_axes[leaf])
        key = leaf, local
        if key in self.cache:
            root, errors = self.cache.pop(key)
            self.cache[key] = root, errors
            self.cache_hits += 1
        else:
            record = self.record
            lower = leaf*self.leaf_rows
            upper = min(len(record.tail_rows), lower+self.leaf_rows)
            rows = record.tail_rows[lower:upper].copy()
            bound = record.tail_errors[lower:upper].copy()
            k, r = record.response_width, record.drift_width
            for index, state in local:
                if state is not None:
                    image = self.axes[index].prefix_images[state]
                    values, errors = image.block(lower+record.startup_native_rows, upper+record.startup_native_rows)
                    bound[:, r:r+k] = _positive_add(bound[:, r:r+k], _positive_add(errors[:, :k],
                        _roundoff_gamma(1)*(np.abs(rows[:, r:r+k])+np.abs(values[:, :k]))))
                    bound[:, r+k:] = _positive_add(bound[:, r+k:], _positive_add(errors[:, k:],
                        _roundoff_gamma(1)*(np.abs(rows[:, r+k:])+np.abs(values[:, k:]))))
                    rows[:, r:r+k] += values[:, :k]
                    rows[:, r+k:] -= values[:, k:]
            root, qr_error = _augmented_qr(rows, self.leaf_rows, with_error=True)
            errors = _positive_add(_column_norms(bound), qr_error)
            self.qr_rebuilds += 1
            used = root.nbytes+errors.nbytes
            if used <= self.cache_bytes:
                while self.cache and self.cache_used+used > self.cache_bytes:
                    _, (old_root, old_errors) = self.cache.popitem(last=False)
                    self.cache_used -= old_root.nbytes+old_errors.nbytes
                self.cache[key] = root, errors
                self.cache_used += used
        self.nodes[self.offset+leaf], self.errors[self.offset+leaf] = root, errors

    def _merge(self, index):
        operands = np.vstack((self.nodes[2*index], self.nodes[2*index+1]))
        root, qr_error = _augmented_qr(operands, max(1, len(operands)), with_error=True)
        self.nodes[index] = root
        # Child certificates belong to disjoint row blocks. Their orthogonal
        # embeddings stack, so column errors combine by Euclidean norm;
        # the current QR acts on those same rows and is then added.
        inherited = np.hypot(self.errors[2*index], self.errors[2*index+1])
        inherited = np.where(inherited == 0, 0., np.nextafter(inherited/(1-_roundoff_gamma(4)), np.inf))
        self.errors[index] = _positive_add(inherited, qr_error)

    @property
    def root(self):
        return self.nodes[1]

    def set_target(self, index, q):
        if self.states[index] == q:
            return
        self.states[index] = q
        parents = set()
        for leaf in range(self.offset):
            if index in self.leaf_axes[leaf]:
                self._leaf(leaf)
                parents.add((self.offset+leaf)//2)
        while parents:
            for index in parents:
                if index:
                    self._merge(index)
            parents = {index//2 for index in parents if index > 1}

    def residual(self, coefficients, drift, *, exclude_axis=None, lower=None, upper=None):
        record = self.record
        lower = record.startup_native_rows if lower is None else lower
        upper = record.native_rows if upper is None else upper
        rows = record.tail_rows[lower-record.startup_native_rows:upper-record.startup_native_rows]
        errors = record.tail_errors[lower-record.startup_native_rows:upper-record.startup_native_rows]
        r, k = record.drift_width, record.response_width
        theta = np.vstack((drift, coefficients))
        values = rows[:, r+k:]-rows[:, :r+k]@theta
        bound = errors[:, r+k:]+errors[:, :r+k]@np.abs(theta)
        bound += _roundoff_gamma(r+k+1)*(
            np.abs(rows[:, r+k:])+np.abs(rows[:, :r+k])@np.abs(theta))
        error = float(np.linalg.norm(bound))
        for index in sorted(self.states):
            if index != exclude_axis and self.states[index] is not None:
                image, ie = self.axes[index].prefix_images[self.states[index]].prediction(lower, upper, coefficients)
                error += ie+_roundoff_gamma(1)*float(np.linalg.norm(np.abs(values)+np.abs(image)))
                values -= image
        return values, error

    def energy(self, coefficients, drift, *, precision):
        record = self.record
        r, k, c = record.drift_width, record.response_width, record.channels
        vector = np.vstack((drift, coefficients, -np.eye(c)))
        residual = self.root@vector
        error = float(np.linalg.norm(self.errors[1]@np.abs(vector)))
        error += _roundoff_gamma(r+k+c)*float(np.linalg.norm(np.abs(self.root)@np.abs(vector)))
        startup_vector = np.r_[drift.ravel(), -1.]
        startup = record.startup_rows@startup_vector
        startup_error = float(_column_norms(record.startup_errors)@np.abs(startup_vector))
        startup_error += _roundoff_gamma(len(startup_vector))*float(np.linalg.norm(
            np.abs(record.startup_rows)@np.abs(startup_vector)))
        if record.startup_response_design is not None:
            value = record.startup_response_design@coefficients.ravel()
            startup_error += float(_column_norms(record.startup_response_errors)@np.abs(coefficients.ravel()))
            startup_error += _roundoff_gamma(coefficients.size)*float(np.linalg.norm(
                np.abs(record.startup_response_design)@np.abs(coefficients.ravel())))
            startup_error += _roundoff_gamma(1)*float(np.linalg.norm(np.abs(startup)+np.abs(value)))
            startup += value
        total = float(np.linalg.norm(residual)**2+startup@startup)
        error += startup_error
        nats = .5*precision*(2*np.sqrt(total)*error+error*error)
        nats += .5*precision*_roundoff_gamma(residual.size+len(startup)+2)*total
        if not np.isfinite(nats) or nats > np.sqrt(np.finfo(float).eps)*record.scalar_rows:
            raise DiffuseProjectionResolutionError('original startup/tail point is unresolved at the actual draw')
        return total


def _draw_resolved_gaussian(design, response, errors, rng, *, proper_prior_covariance_error=0.):
    rows = np.column_stack((design, response))
    augmented, qr_errors = _augmented_qr(rows, max(1, len(rows)), with_error=True)
    width = design.shape[1]
    root, rhs = augmented[:width, :width], augmented[:width, width:]
    _response_resolved_root(root, 'factored original coefficient conditional')
    inverse, inverse_norm = _inverse_root_geometry(root)
    total_errors = _positive_add(errors, qr_errors)
    metric_columns = (total_errors[:width]@inverse)/(1-_roundoff_gamma(width))
    metric = _upper_frobenius(metric_columns)
    if not np.isfinite(metric) or metric > np.sqrt(np.finfo(float).eps):
        raise DiffuseProjectionResolutionError('factored coefficient conditional is unresolved in its posterior metric')
    mean = solve_triangular(root, rhs, lower=False)
    solve_error = float(np.linalg.norm(root@mean-rhs))
    solve_error += _roundoff_gamma(width)*float(np.linalg.norm(np.abs(root)@np.abs(mean)))
    data_error = _upper_frobenius(total_errors[width:])
    remaining = _upper_frobenius(augmented[width:, width:])
    # In the actual posterior chart v=R beta, perturb A R^-1 by D.
    # The normal-equation right side is A'.T(f-D R mean)+D.T residual.
    # Column error bounds may be tiny in the actual mean direction even when
    # a different coefficient direction is weak. Both contractions below
    # bound the same delta_b-delta_A*mu0 term. mu0 is the exact stored-root
    # solve, and ||R(mu0-mu)|| <= solve_error covers its remaining displacement.
    mean_columns = (total_errors[:width]@np.abs(mean))/(1-_roundoff_gamma(width))
    directional_error = _upper_frobenius(_positive_add(total_errors[width:], mean_columns))
    directional_error = float(_positive_add(directional_error, metric*solve_error))
    global_error = data_error+metric*_upper_frobenius(rhs)
    effective_rhs_error = min(global_error, directional_error)
    mean_error = ((1+metric)*effective_rhs_error
                  +metric*remaining)/(1-metric)**2+solve_error
    budget = np.sqrt(np.finfo(float).eps)*max(1., np.sqrt(rhs.size))
    prior_mean_error = 0.
    if proper_prior_covariance_error:
        # This error changes only the ONE zero-mean proper prior precision.
        # It does not perturb a likelihood row and therefore cannot multiply
        # the unrelated full-data residual. Bound its precision action on the
        # actual mean after accounting for the certified likelihood errors.
        delta = proper_prior_covariance_error
        if not np.isfinite(delta) or not 0 <= delta < 1:
            raise ValueError('original proper-prior covariance relation requires a finite spectral bound')
        precision_error = delta/(1-delta)
        prior_metric = precision_error*inverse_norm**2/(1-metric)**2
        # The equivalent positive root's singular values lie between
        # sqrt((1-rho)^2-t*vInv^2) and sqrt((1+rho)^2+t*vInv^2).
        # Both deviations from one are bounded by rho+prior_metric.
        if not np.isfinite(prior_metric) or metric+prior_metric > np.sqrt(np.finfo(float).eps):
            raise DiffuseProjectionResolutionError('original proper prior is unresolved in its conditional posterior metric')
        prior_mean_error = (precision_error*inverse_norm*(float(np.linalg.norm(mean))+inverse_norm*mean_error)
                            /((1-metric)**2*(1-prior_metric)))
        mean_error += prior_mean_error
    if not np.isfinite(mean_error) or mean_error > budget:
        evidence = {'likelihood_metric': metric, 'data_error': data_error,
            'global_rhs_error': global_error, 'directional_rhs_error': directional_error,
            'remaining_norm': remaining, 'solve_error': solve_error,
            'proper_prior_covariance_error': proper_prior_covariance_error,
            'prior_mean_error': prior_mean_error, 'mean_error': mean_error, 'budget': budget}
        raise DiffuseProjectionResolutionError('factored Gaussian conditional mean is unresolved in its posterior metric: '
                                              +json.dumps(evidence, sort_keys=True))
    random = rng.normal(size=rhs.shape)
    draw_rhs = rhs+random
    value = solve_triangular(root, draw_rhs, lower=False)
    draw_error = float(np.linalg.norm(root@value-draw_rhs))
    draw_error += _roundoff_gamma(width)*float(np.linalg.norm(np.abs(root)@np.abs(value)))
    draw_error += _roundoff_gamma(1)*float(np.linalg.norm(np.abs(rhs)+np.abs(random)))
    if not np.isfinite(draw_error) or draw_error > budget:
        raise DiffuseProjectionResolutionError('factored Gaussian draw solve is unresolved in its posterior metric')
    _numeric(value, name='original factored coefficient conditional draw')
    return value


def _draw_beta(trees, drifts, precisions, rng, prior_covariance_error):
    k = trees[0].record.response_width
    c = trees[0].record.channels
    designs, responses = [np.eye(k)], [np.zeros((k, c))]
    errors = np.zeros(k+c)
    for tree, drift, precision in zip(trees, drifts, precisions, strict=True):
        r = tree.record.drift_width
        root, bound = tree.root, tree.errors[1]
        scale = np.sqrt(precision)
        design = scale*root[:, r:r+k]
        response = scale*(root[:, r+k:]-root[:, :r]@drift)
        designs.append(design)
        responses.append(response)
        local = np.r_[scale*bound[r:r+k]+_roundoff_gamma(2)*_column_norms(design),
            scale*(bound[r+k:]+bound[:r]@np.abs(drift))+_roundoff_gamma(r+3)*_column_norms(
                scale*(np.abs(root[:, r+k:])+np.abs(root[:, :r])@np.abs(drift)))]
        combined = np.hypot(errors, local)
        errors = np.where(combined == 0, 0., np.nextafter(combined/(1-_roundoff_gamma(4)), np.inf))
    return _draw_resolved_gaussian(np.vstack(designs), np.vstack(responses), errors, rng,
                                  proper_prior_covariance_error=prior_covariance_error)


def _draw_drift(tree, coefficients, precision, rng):
    record = tree.record
    r, k, c = record.drift_width, record.response_width, record.channels
    root, bound = tree.root, tree.errors[1]
    tail_design = np.kron(root[:, :r], np.eye(c))
    response = (root[:, r+k:]-root[:, r:r+k]@coefficients).ravel()
    design = np.vstack((record.startup_rows[:, :-1], tail_design))
    startup_response = record.startup_rows[:, -1].copy()
    if record.startup_response_design is not None:
        startup_response -= record.startup_response_design@coefficients.ravel()
    response = np.r_[startup_response, response][:, None]
    errors = _column_norms(record.startup_errors)
    if record.startup_response_design is not None:
        errors[-1] += float(_column_norms(record.startup_response_errors)@np.abs(coefficients.ravel()))
        errors[-1] += _roundoff_gamma(coefficients.size+1)*float(np.linalg.norm(
            np.abs(record.startup_rows[:, -1])+np.abs(record.startup_response_design)@np.abs(coefficients.ravel())))
    errors[:-1] += np.repeat(bound[:r], c)
    errors[-1] += float(np.linalg.norm(bound[r+k:]+bound[r:r+k]@np.abs(coefficients)))
    errors[-1] += _roundoff_gamma(k+1)*float(np.linalg.norm(
        np.abs(root[:, r+k:])+np.abs(root[:, r:r+k])@np.abs(coefficients)))
    scale = np.sqrt(precision)
    errors = scale*errors+_roundoff_gamma(2)*_column_norms(np.column_stack((scale*design, scale*response)))
    return _draw_resolved_gaussian(scale*design, scale*response, errors, rng)[:, 0].reshape(r, c)


def _draw_full_prior_beta(trees, drifts, precisions, prior_root, prior_errors, rng):
    """Fixed full physical prior, thin record roots and exact joint startup."""
    k, c = trees[0].record.response_width, trees[0].record.channels
    designs, responses = [prior_root], [np.zeros((len(prior_root), 1))]
    errors = np.r_[prior_errors.copy(), 0.]
    for tree, drift, precision in zip(trees, drifts, precisions, strict=True):
        record, root, bound = tree.record, tree.root, tree.errors[1]
        r = record.drift_width
        scale = np.sqrt(precision)
        tail_design = scale*np.kron(root[:, r:r+k], np.eye(c))
        tail_response = scale*(root[:, r+k:]-root[:, :r]@drift).ravel()[:, None]
        designs.append(tail_design)
        responses.append(tail_response)
        errors[:k*c] += scale*np.repeat(bound[r:r+k], c)+_roundoff_gamma(2)*_column_norms(tail_design)
        errors[-1] += scale*float(np.linalg.norm(bound[r+k:]+bound[:r]@np.abs(drift)))
        errors[-1] += _roundoff_gamma(r+3)*float(np.linalg.norm(
            scale*(np.abs(root[:, r+k:])+np.abs(root[:, :r])@np.abs(drift))))
        if record.startup_response_design is not None:
            leading_design = scale*record.startup_response_design
            leading_response = scale*(record.startup_rows[:, -1]-record.startup_rows[:, :-1]@drift.ravel())[:, None]
            designs.append(leading_design)
            responses.append(leading_response)
            errors[:k*c] += scale*_column_norms(record.startup_response_errors)
            errors[:k*c] += _roundoff_gamma(2)*_column_norms(leading_design)
            errors[-1] += scale*float(_column_norms(record.startup_errors)@np.r_[np.abs(drift.ravel()), 1.])
            errors[-1] += _roundoff_gamma(r*c+3)*float(np.linalg.norm(
                scale*(np.abs(record.startup_rows[:, -1])+np.abs(record.startup_rows[:, :-1])@np.abs(drift.ravel()))))
    return _draw_resolved_gaussian(np.vstack(designs), np.vstack(responses), errors, rng)[:, 0].reshape(k, c)


def run_factored_history_gibbs(records, axes, *, degrees_of_freedom, initial_coefficients,
        initial_drifts, initial_precisions, initial_targets, seeds, burnin, draws,
        sweeps_per_draw, leaf_rows, cache_bytes, proper_precision_root=None, proper_precision_errors=None,
        proper_prior_covariance_error=0.):
    """Exact startup; beta|a, a|beta and explicit-a raw-row Gamma, P only."""
    records, axes, seeds = tuple(records), tuple(axes), tuple(seeds)
    nu = _degrees(degrees_of_freedom)
    if (not records or any((record.channels, record.response_width) !=
            (records[0].channels, records[0].response_width) for record in records)):
        raise ValueError('all actual recordings must share one original response/channel chart')
    k, c = records[0].response_width, records[0].channels
    if not np.isfinite(proper_prior_covariance_error) or not 0 <= proper_prior_covariance_error < 1:
        raise ValueError('proper source covariance relation must have a finite resolved spectral bound')
    if proper_precision_root is not None:
        proper_precision_root = finite_real_array(proper_precision_root, name='fixed physical coefficient prior root')
        proper_precision_errors = finite_real_array(proper_precision_errors, name='physical prior root column errors')
        if (proper_precision_root.shape != (k*c, k*c) or proper_precision_errors.shape != (k*c,)
                or np.any(proper_precision_errors < 0)):
            raise ValueError('one original full proper prior and its actual chart errors are required')
    elif any(record.startup_response_design is not None for record in records):
        raise ValueError('nonzero original startup response requires the full physical-prior conditional')
    initial = finite_real_array(initial_coefficients, name='explicit overdispersed original response starts')
    lambdas = finite_real_array(initial_precisions, name='original true-record precision starts')
    states = np.asarray(initial_targets)
    if (initial.shape != (len(seeds), k, c) or lambdas.shape != (len(seeds), len(records))
            or states.shape != (len(seeds), len(axes)) or states.dtype.kind not in 'iu'
            or np.any(lambdas <= 0) or len(set(seeds)) != len(seeds) or len(seeds) < 2
            or len(initial_drifts) != len(records) or any(np.asarray(d).shape != (len(seeds), r.drift_width, c)
                for d, r in zip(initial_drifts, records, strict=True))
            or type(draws) is not int or draws < 8 or type(burnin) is not int or burnin < 0
            or type(sweeps_per_draw) is not int or sweeps_per_draw < 1
            or type(leaf_rows) is not int or leaf_rows < 1 or type(cache_bytes) is not int or cache_bytes < 0):
        raise ValueError('complete original factored Gibbs state and explicit resource programme are required')
    if np.isposinf(nu) and np.any(lambdas != 1):
        raise ValueError('Gaussian infinity has one deterministic unit precision per true record')
    coefficient_width = k*c+sum(r.drift_width*c for r in records)
    theta_draws = np.empty((len(seeds), draws, coefficient_width))
    precision_draws = np.empty((len(seeds), draws, len(records)))
    target_draws = np.empty((len(seeds), draws, len(axes)), dtype=np.int64)
    started, rebuilt = perf_counter(), 0
    per_record_axes = [tuple(axis for axis in axes if axis.record_index == index) for index in range(len(records))]
    local_indices = {global_index: sum(axis.record_index == axes[global_index].record_index
        for axis in axes[:global_index]) for global_index in range(len(axes))}
    for chain, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        beta = initial[chain].copy()
        drifts = [np.asarray(d)[chain].copy() for d in initial_drifts]
        precision, state = lambdas[chain].copy(), states[chain].copy()
        trees = [FactoredQRTree(record, local, leaf_rows=leaf_rows, cache_bytes=cache_bytes)
            for record, local in zip(records, per_record_axes, strict=True)]
        for index, q in enumerate(state):
            axis = axes[index]
            if not 0 <= q < len(axis.prior_mass) or axis.prior_mass[q] <= 0:
                raise ValueError('joint starts must retain the original C constraints')
            trees[axis.record_index].set_target(local_indices[index], int(q))
        for sweep in range(burnin+draws*sweeps_per_draw):
            for index, axis in enumerate(axes):
                tree = trees[axis.record_index]
                positive = np.flatnonzero(axis.prior_mass > 0)
                if len(positive) == 1:
                    # The original C constraint already fixed this whole q
                    # coordinate. Its response still enters every record root.
                    continue
                support = [axis.prefix_images[q] for q in positive
                           if axis.prefix_images[q].stop > axis.prefix_images[q].start]
                lower = min((image.start for image in support), default=tree.record.native_rows)
                upper = max((image.stop for image in support), default=lower)
                logits = np.full(len(axis.prior_mass), -np.inf)
                if upper <= lower:
                    logits[positive] = np.log(axis.prior_mass[positive])
                else:
                    residual, error = tree.residual(beta, drifts[axis.record_index],
                        exclude_axis=local_indices[index], lower=lower, upper=upper)
                    for q in positive:
                        value, bound = axis.prefix_images[q].prediction(lower, upper, beta)
                        candidate = residual-value
                        total_error = error+bound+_roundoff_gamma(1)*float(np.linalg.norm(np.abs(residual)+np.abs(value)))
                        energy = float(np.linalg.norm(candidate)**2)
                        nats = .5*precision[axis.record_index]*(2*np.sqrt(energy)*total_error+total_error**2)
                        nats += .5*precision[axis.record_index]*_roundoff_gamma(candidate.size+2)*energy
                        if not np.isfinite(nats) or nats > np.sqrt(np.finfo(float).eps)*max(1, candidate.size):
                            raise DiffuseProjectionResolutionError('complete factored q conditional is unresolved in original nats')
                        logits[q] = np.log(axis.prior_mass[q])-.5*precision[axis.record_index]*energy
                chosen = int(np.argmax(logits+rng.gumbel(size=len(logits))))
                if chosen != state[index]:
                    tree.set_target(local_indices[index], chosen)
                    state[index] = chosen
            beta = (_draw_beta(trees, drifts, precision, rng, proper_prior_covariance_error) if proper_precision_root is None else
                    _draw_full_prior_beta(trees, drifts, precision, proper_precision_root, proper_precision_errors, rng))
            prior_error = .5*(-k*c*np.log1p(-proper_prior_covariance_error)
                +proper_prior_covariance_error/(1-proper_prior_covariance_error)*float(np.linalg.norm(beta)**2))
            if not np.isfinite(prior_error) or prior_error > np.sqrt(np.finfo(float).eps)*k*c:
                raise DiffuseProjectionResolutionError('original source proper prior potential is unresolved at the actual draw')
            drifts = [_draw_drift(tree, beta, lam, rng) for tree, lam in zip(trees, precision, strict=True)]
            for index, tree in enumerate(trees):
                energy = tree.energy(beta, drifts[index], precision=precision[index])
                if not np.isposinf(nu):
                    precision[index] = rng.gamma((nu+tree.record.scalar_rows)/2, 2/(nu+energy))
            _numeric(precision, name='original same-record precision draws')
            if np.any(precision <= 0):
                raise DiffuseProjectionResolutionError('original Gamma draw is not representable and positive')
            if sweep >= burnin and (sweep-burnin+1)%sweeps_per_draw == 0:
                draw = (sweep-burnin)//sweeps_per_draw
                theta_draws[chain, draw] = np.r_[beta.ravel(), *(drift.ravel() for drift in drifts)]
                precision_draws[chain, draw], target_draws[chain, draw] = precision, state
        rebuilt += sum(tree.qr_rebuilds for tree in trees)
    return JointHistorySamples(theta_draws, precision_draws, target_draws, seeds, burnin,
        sweeps_per_draw, nu, sum(r.scalar_rows for r in records), perf_counter()-started,
        rebuilt, tuple(r.scalar_rows for r in records))


def _factor_residual(rows, errors, coefficients, drift):
    theta = np.vstack((drift, coefficients))
    width = len(theta)
    value = rows[:, width:]-rows[:, :width]@theta
    bound = errors[:, width:]+errors[:, :width]@np.abs(theta)
    bound += _roundoff_gamma(width+1)*(
        np.abs(rows[:, width:])+np.abs(rows[:, :width])@np.abs(theta))
    return value, float(np.linalg.norm(bound))


def _factor_subtract(value, error, image, coefficients, lower, upper):
    prediction, bound = image.prediction(lower, upper, coefficients)
    error += bound+_roundoff_gamma(1)*float(np.linalg.norm(np.abs(value)+np.abs(prediction)))
    value -= prediction
    return error


def _factor_energy(value, error, precision):
    energy = float(np.linalg.norm(value)**2)
    nats = .5*precision*(2*np.sqrt(energy)*error+error*error)
    nats += .5*precision*_roundoff_gamma(value.size+2)*energy
    if not np.isfinite(nats) or nats > np.sqrt(np.finfo(float).eps)*max(1, value.size):
        raise DiffuseProjectionResolutionError('factored actual prefix/tail component is unresolved in original nats')
    return energy, nats


def factored_tail_components(records, tail_rows, tail_errors, axes, samples, *, current_axis,
        tail_native_start, tail_noise_logdet, tail_noise_logdet_error):
    """Replay the captured P states; keep every current-q conditional component."""
    axis = axes[current_axis]
    record = records[axis.record_index]
    if (samples.record_scalar_rows != tuple(r.scalar_rows for r in records)
            or samples.targets.shape[-1] != len(axes)
            or len(tail_rows) == 0 or tail_native_start != record.native_rows
            or tail_rows.shape != tail_errors.shape
            or tail_rows.shape[1] != record.tail_rows.shape[1]):
        raise ValueError('captured states must retain original P rows, real-record axes and adjacent raw T')
    positive = np.flatnonzero(axis.prior_mass > 0)
    support = [axis.prefix_images[q] for q in positive
               if axis.prefix_images[q].stop > axis.prefix_images[q].start]
    lower = min((image.start for image in support), default=record.native_rows)
    upper = max((image.stop for image in support), default=lower)
    k, c = record.response_width, record.channels
    drift_offset = k*c+sum(r.drift_width*c for r in records[:axis.record_index])
    shape = (*samples.coefficients.shape[:2], len(axis.prior_mass))
    logweights, logdensity = np.full(shape, -np.inf), np.full(shape, -np.inf)
    for chain in range(shape[0]):
        for draw in range(shape[1]):
            theta = samples.coefficients[chain, draw]
            coefficients = theta[:k*c].reshape(k, c)
            drift = theta[drift_offset:drift_offset+record.drift_width*c].reshape(record.drift_width, c)
            precision = samples.precisions[chain, draw, axis.record_index]
            rows = record.tail_rows[lower-record.startup_native_rows:upper-record.startup_native_rows]
            errors = record.tail_errors[lower-record.startup_native_rows:upper-record.startup_native_rows]
            prefix, pe = _factor_residual(rows, errors, coefficients, drift)
            tail, te = _factor_residual(tail_rows, tail_errors, coefficients, drift)
            for index, other in enumerate(axes):
                if index == current_axis or other.record_index != axis.record_index:
                    continue
                q = samples.targets[chain, draw, index]
                pe = _factor_subtract(prefix, pe, other.prefix_images[q], coefficients, lower, upper)
                te = _factor_subtract(tail, te, other.tail_images[q], coefficients,
                                      tail_native_start, tail_native_start+len(tail_rows))
            logits = np.full(len(axis.prior_mass), -np.inf)
            for q in positive:
                rp, rt = prefix.copy(), tail.copy()
                qpe = _factor_subtract(rp, pe, axis.prefix_images[q], coefficients, lower, upper)
                qte = _factor_subtract(rt, te, axis.tail_images[q], coefficients,
                                      tail_native_start, tail_native_start+len(tail_rows))
                ep, _ = _factor_energy(rp, qpe, precision)
                et, energy_error = _factor_energy(rt, qte, precision)
                normalizer = rt.size*(np.log(2*np.pi)-np.log(precision))+tail_noise_logdet
                normalizer_error = tail_noise_logdet_error+_roundoff_gamma(4)*(
                    rt.size*(abs(np.log(2*np.pi))+abs(np.log(precision)))+abs(tail_noise_logdet))
                total_error = energy_error+.5*normalizer_error+_roundoff_gamma(2)*(
                    abs(normalizer)+precision*et)
                if not np.isfinite(total_error) or total_error > np.sqrt(np.finfo(float).eps)*rt.size:
                    raise DiffuseProjectionResolutionError('normalized factored raw T density is unresolved in nats')
                logits[q] = np.log(axis.prior_mass[q])-.5*precision*ep
                logdensity[chain, draw, q] = -.5*(normalizer+precision*et)
            logweights[chain, draw] = logits-logsumexp(logits)
    _numeric(logweights[:, :, positive], logdensity[:, :, positive], name='complete factored current-q components')
    return frozen_array(logweights), frozen_array(logdensity)
