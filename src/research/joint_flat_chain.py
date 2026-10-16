"""Same-record joint discrete-target / proper-response / flat-drift Gibbs.

The observation matrix comes from an original source authority and its full
causal whitening map. This numerical engine grants no sample or label access.
One prior on response coefficients, one flat drift chart and one precision
belong to the entire record. Keeping a explicitly makes the conditional Gamma
shape depend on raw rows; the existing collapsed-a sampler uses effective rows.
"""

from __future__ import annotations

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
from research.joint_response_sampling import (
    SamplingResolutionError,
    _batch_mcse,
    _functional_diagnostics,
)
from research.joint_response_scale import _augmented_qr, _degrees, _numeric
from transfer.noise_model import finite_real_array, frozen_array


class AugmentedQRTree:
    """Private mutable original rows; update only the leaves touched by q."""

    def __init__(self, rows, *, leaf_rows=2048, source_error_columns=None):
        values = finite_real_array(rows, name="whole-prefix augmented rows")
        if values.ndim != 2 or not len(values) or values.shape[1] < 3:
            raise ValueError("proper response/flat drift and the residual column are required")
        if type(leaf_rows) is not int or leaf_rows < 1:
            raise ValueError("QR work leaf must have a positive row count")
        self.base_rows = frozen_array(values)
        self.rows = values.copy()
        self.active_images = {}
        self.base_error = (np.zeros(values.shape[1]) if source_error_columns is None else
                           finite_real_array(source_error_columns, name="original prefix source error columns"))
        if self.base_error.shape != (values.shape[1],) or np.any(self.base_error < 0):
            raise ValueError("actual prefix source errors must preserve all coefficient and residual columns")
        self.leaf_rows = leaf_rows
        count = (len(values)+leaf_rows-1)//leaf_rows
        self.offset = 1 << (count-1).bit_length()
        self.nodes = [None]*(2*self.offset)
        self.node_errors = [None]*(2*self.offset)
        self.assembly_errors = np.zeros((count, values.shape[1]))
        empty = np.empty((0, values.shape[1]))
        for index in range(self.offset):
            start = index*leaf_rows
            if start < len(values):
                root, errors = _augmented_qr(self.rows[start:start+leaf_rows], leaf_rows, with_error=True)
            else:
                root, errors = empty, np.zeros(values.shape[1])
            self.nodes[self.offset+index], self.node_errors[self.offset+index] = root, errors
        for index in range(self.offset-1, 0, -1):
            self._merge(index)
        self.updated_leaves = 0

    def _merge(self, index):
        operands = np.vstack((self.nodes[2*index], self.nodes[2*index+1]))
        root, errors = _augmented_qr(operands, max(1, len(operands)), with_error=True)
        self.nodes[index] = root
        self.node_errors[index] = self.node_errors[2*index]+self.node_errors[2*index+1]+errors
        _numeric(self.nodes[index], name="same-record augmented QR tree")

    @property
    def root(self):
        return self.nodes[1]

    @property
    def source_errors(self):
        result = self.base_error.copy()
        result += _column_norms(self.assembly_errors)
        for image in self.active_images.values():
            if image.error_columns is not None:
                result[:len(image.error_columns)-1] += image.error_columns[:-1]
                result[-1] += image.error_columns[-1]
        return result

    @property
    def root_errors(self):
        return self.source_errors+self.node_errors[1]

    def set_image(self, axis_index, image, *, response_width):
        """Rebuild affected leaves from original operands in fixed axis order.

        Returning to the same q state restores the same numerical rows. q
        transitions do not accumulate subtraction/addition history in the data.
        """
        start, stop = image.start, image.start+len(image.values)
        if (type(axis_index) is not int or axis_index < 0 or not 0 <= start <= stop <= len(self.rows)
                or image.values.shape[1] != response_width+1):
            raise ValueError("a target change must retain original scalar row coordinates")
        previous = self.active_images.get(axis_index)
        self.active_images[axis_index] = image
        leaves = set()
        for value in (previous, image):
            if value is not None and len(value.values):
                leaves.update(range(value.start//self.leaf_rows,
                                    (value.start+len(value.values)-1)//self.leaf_rows+1))
        parents = set()
        for leaf in leaves:
            lower = leaf*self.leaf_rows
            upper = min(len(self.rows), lower+self.leaf_rows)
            block = self.base_rows[lower:upper].copy()
            arithmetic = np.zeros_like(block)
            for index in sorted(self.active_images):
                current = self.active_images[index]
                left, right = max(lower, current.start), min(upper, current.start+len(current.values))
                if left < right:
                    operand = current.values[left-current.start:right-current.start]
                    arithmetic[left-lower:right-lower, :response_width] += _roundoff_gamma(1)*(
                        np.abs(block[left-lower:right-lower, :response_width])+np.abs(operand[:, :response_width]))
                    arithmetic[left-lower:right-lower, -1] += _roundoff_gamma(1)*(
                        np.abs(block[left-lower:right-lower, -1])+np.abs(operand[:, -1]))
                    block[left-lower:right-lower, :response_width] += operand[:, :response_width]
                    block[left-lower:right-lower, -1] -= operand[:, -1]
            self.assembly_errors[leaf] = _column_norms(arithmetic)
            self.rows[lower:upper] = block
            self.nodes[self.offset+leaf], self.node_errors[self.offset+leaf] = _augmented_qr(
                block, len(block), with_error=True)
            parents.add((self.offset+leaf)//2)
            self.updated_leaves += 1
        while parents:
            for node in parents:
                if node:
                    self._merge(node)
            parents = {node//2 for node in parents if node > 1}

    def energy(self, coefficients):
        residual = self.root@np.r_[coefficients, -1.]
        value = float(residual@residual)
        _numeric(value, name="original raw-prefix residual energy")
        return value

    def check_point(self, coefficients, precision):
        vector = np.r_[coefficients, -1.]
        residual = self.root@vector
        bound = float(np.abs(vector)@self.root_errors)
        bound += _roundoff_gamma(len(vector))*float(np.linalg.norm(np.abs(self.root)@np.abs(vector)))
        error = .5*precision*(2*float(np.linalg.norm(residual))*bound+bound*bound)
        error += .5*precision*_roundoff_gamma(len(residual)+1)*float(residual@residual)
        if not np.isfinite(error) or error > np.sqrt(np.finfo(float).eps)*len(self.rows):
            raise DiffuseProjectionResolutionError("joint history point is unresolved in original prefix noise units")
        return error


@dataclass(frozen=True)
class SparseResponseImage:
    start: int
    values: np.ndarray
    error_columns: np.ndarray | None = None

    def __post_init__(self):
        values = finite_real_array(self.values, name="original sparse white response image")
        if type(self.start) is not int or self.start < 0 or values.ndim != 2:
            raise ValueError("sparse images need original scalar start and actual matrix columns")
        object.__setattr__(self, "values", frozen_array(values))
        if self.error_columns is not None:
            errors = finite_real_array(self.error_columns, name="original sparse image source errors")
            if errors.shape != (values.shape[1],) or np.any(errors < 0):
                raise ValueError("sparse source errors must retain its original column coordinates")
            object.__setattr__(self, "error_columns", frozen_array(errors))


@dataclass(frozen=True)
class DiscreteTargetAxis:
    selection_id: str
    candidate_vocabulary: tuple[str, ...]
    prior_mass: np.ndarray
    prefix_images: tuple[SparseResponseImage, ...]
    tail_images: tuple[SparseResponseImage, ...]
    record_index: int = 0

    def __post_init__(self):
        mass = finite_real_array(self.prior_mass, name="complete protocol target prior")
        width = len(self.candidate_vocabulary)
        if (not self.selection_id or mass.shape != (width,) or not width or np.any(mass < 0)
                or not np.isclose(mass.sum(), 1.) or len(self.prefix_images) != width
                or len(self.tail_images) != width or type(self.record_index) is not int or self.record_index < 0):
            raise ValueError("full target vocabularies, C-compatible prior and actual P/T images are required")
        object.__setattr__(self, "prior_mass", frozen_array(mass))


@dataclass(frozen=True)
class JointHistorySamples:
    coefficients: np.ndarray
    precisions: np.ndarray
    targets: np.ndarray
    seeds: tuple[int, ...]
    burnin: int
    sweeps_per_draw: int
    degrees_of_freedom: float
    raw_prefix_rows: int
    elapsed_seconds: float
    qr_updated_leaves: int
    record_scalar_rows: tuple[int, ...]

    def __post_init__(self):
        theta = finite_real_array(self.coefficients, name="retained joint response/drift draws")
        precision = finite_real_array(self.precisions, name="retained same-record precision")
        targets = np.asarray(self.targets)
        if (theta.ndim != 3 or precision.ndim != 3 or precision.shape[:2] != theta.shape[:2] or np.any(precision <= 0)
                or targets.ndim != 3 or targets.shape[:2] != theta.shape[:2] or targets.dtype.kind not in "iu"):
            raise ValueError("every retained state must keep response/drift, same lambda and all target axes")
        if (precision.shape[-1] != len(self.record_scalar_rows)
                or sum(self.record_scalar_rows) != self.raw_prefix_rows):
            raise ValueError("retained precision axes must match every original true-record row count")
        object.__setattr__(self, "coefficients", frozen_array(theta))
        object.__setattr__(self, "precisions", frozen_array(precision))
        immutable = np.frombuffer(np.ascontiguousarray(targets, dtype=np.int64).tobytes(), dtype=np.int64).reshape(targets.shape)
        object.__setattr__(self, "targets", immutable)


def _original_residual(rows, theta, source_errors=None):
    """Evaluate the stored original operands, including actual dot/subtract work."""
    residual = rows[:, -1]-rows[:, :-1]@theta
    error = 0. if source_errors is None else float(np.r_[np.abs(theta), 1.]@source_errors)
    error += _roundoff_gamma(len(theta)+1)*float(np.linalg.norm(
        np.abs(rows[:, :-1])@np.abs(theta)+np.abs(rows[:, -1])))
    return residual, error


def _subtract_image(residual, start, image, vector, error):
    lower = max(start, image.start)
    upper = min(start+len(residual), image.start+len(image.values))
    if lower < upper:
        rows = image.values[lower-image.start:upper-image.start]
        value = rows@vector
        if image.error_columns is not None:
            error += float(np.abs(vector)@image.error_columns)
        error += _roundoff_gamma(len(vector))*float(np.linalg.norm(np.abs(rows)@np.abs(vector)))
        before = residual[lower-start:upper-start]
        error += _roundoff_gamma(1)*float(np.linalg.norm(np.abs(before)+np.abs(value)))
        residual[lower-start:upper-start] -= value
    return error


def _point_energy(residual, error, precision, *, rows, name):
    energy = float(residual@residual)
    nats_error = .5*precision*(2*np.sqrt(energy)*error+error*error)
    nats_error += .5*precision*_roundoff_gamma(len(residual)+2)*energy
    budget = np.sqrt(np.finfo(float).eps)*max(1, rows)
    if not np.isfinite(energy) or not np.isfinite(nats_error) or nats_error > budget:
        raise DiffuseProjectionResolutionError(f"{name} is unresolved at its actual coefficients and precision")
    return energy, nats_error


def _axis_logits(tree, axis, axis_index, theta, precision, response_width):
    positive = np.flatnonzero(axis.prior_mass > 0)
    logits = np.full(len(axis.prior_mass), -np.inf)
    images = axis.prefix_images
    starts = [images[index].start for index in positive if len(images[index].values)]
    if not starts:
        return np.log(axis.prior_mass, where=axis.prior_mass > 0, out=logits)
    start = min(starts)
    stop = max(images[index].start+len(images[index].values) for index in positive)
    # Rebuild from original operands, excluding the current axis. Removing a
    # previous image from an already rounded row can amplify cancellation.
    residual, source_bound = _original_residual(tree.base_rows[start:stop], theta, tree.base_error)
    vector = np.r_[theta[:response_width], 1.]
    for other in sorted(tree.active_images):
        if other != axis_index:
            source_bound = _subtract_image(residual, start, tree.active_images[other], vector, source_bound)
    for index in positive:
        candidate = residual.copy()
        image = images[index]
        candidate_error = _subtract_image(candidate, start, image, vector, source_bound)
        energy, _ = _point_energy(candidate, candidate_error, precision, rows=len(candidate),
                                 name="complete target conditional")
        logits[index] = np.log(axis.prior_mass[index])-.5*precision*energy
    finite = logits[positive]
    _numeric(finite, name="complete target conditional log mass")
    return logits


def _draw_coefficients(trees, precision, response_width, rng):
    width = trees[0].rows.shape[1]-1
    prior = np.zeros((response_width, width+1))
    prior[:, :response_width] = np.eye(response_width)
    # Shared beta prior once, independent true-record likelihood scales.
    rows = [prior]
    source_errors = np.zeros(width+1)
    for lam, tree in zip(precision, trees, strict=True):
        scale = np.sqrt(lam)
        weighted = scale*tree.root
        rows.append(weighted)
        source_errors += scale*tree.root_errors+_roundoff_gamma(2)*_column_norms(weighted)
    augmented, qr_errors = _augmented_qr(np.vstack(rows), width+1, with_error=True)
    source_errors += qr_errors
    root, rhs = augmented[:width, :width], augmented[:width, width]
    _response_resolved_root(root, "joint proper response / flat drift posterior")
    inverse_bound, _ = _inverse_root_geometry(root)
    metric_error = float(np.linalg.norm(source_errors[:width]@inverse_bound))
    if not np.isfinite(metric_error) or metric_error > np.sqrt(np.finfo(float).eps):
        raise DiffuseProjectionResolutionError("joint coefficient covariance is unresolved in its posterior metric")
    mean = solve_triangular(root, rhs, lower=False)
    value = mean+solve_triangular(root, rng.normal(size=width), lower=False)
    _numeric(value, name="joint response/drift Gaussian draw")
    return value


def run_joint_history_gibbs(base_rows, axes, *, response_width, degrees_of_freedom,
        initial_coefficients, initial_precisions, initial_targets, seeds, burnin, draws,
        record_spans=None, source_error_columns=None, sweeps_per_draw=1, leaf_rows=2048):
    """All target axes, one Gaussian coefficient block, then one raw-row Gamma.

    base_rows excludes every axis' target image; C-known target effects and
    common ERP, observed prefix and original flat chart remain in its rows.
    Each axis retains the whole candidate vocabulary, including exact C zeros.
    T-only axes have zero P images and are sampled from their original prior.
    """
    rows = finite_real_array(base_rows, name="original full-prefix likelihood rows")
    nu, axes, seeds = _degrees(degrees_of_freedom), tuple(axes), tuple(seeds)
    spans = ((0, len(rows)),) if record_spans is None else tuple(tuple(pair) for pair in record_spans)
    if (not spans or spans[0][0] != 0 or spans[-1][1] != len(rows)
            or any(len(pair) != 2 or any(type(value) is not int for value in pair) or not pair[0] < pair[1]
                   for pair in spans)
            or any(left[1] != right[0] for left, right in zip(spans, spans[1:], strict=False))
            or any(axis.record_index >= len(spans) for axis in axes)):
        raise ValueError("original true-record scalar row spans must cover the supplied observations once")
    starts = finite_real_array(initial_coefficients, name="declared overdispersed joint starts")
    precisions = finite_real_array(initial_precisions, name="declared precision starts")
    targets = np.asarray(initial_targets)
    if (rows.ndim != 2 or type(response_width) is not int or not 0 < response_width < rows.shape[1]-1
            or starts.shape != (len(seeds), rows.shape[1]-1) or precisions.shape != (len(seeds), len(spans))
            or targets.shape != (len(seeds), len(axes)) or targets.dtype.kind not in "iu"
            or np.any(precisions <= 0) or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or type(burnin) is not int or burnin < 0 or type(draws) is not int or draws < 8
            or type(sweeps_per_draw) is not int or sweeps_per_draw < 1):
        raise ValueError("explicit complete Gibbs programme and original joint chart are required")
    if np.isposinf(nu) and np.any(precisions != 1):
        raise ValueError("Gaussian infinity has the deterministic unit noise precision")
    theta_draws = np.empty((len(seeds), draws, rows.shape[1]-1))
    scale_draws = np.empty((len(seeds), draws, len(spans)))
    target_draws = np.empty((len(seeds), draws, len(axes)), dtype=np.int64)
    start_time, updated = perf_counter(), 0
    for chain, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        trees = tuple(AugmentedQRTree(rows[start:stop], leaf_rows=leaf_rows,
            source_error_columns=None if source_error_columns is None else source_error_columns[index])
            for index, (start, stop) in enumerate(spans))
        theta, precision, state = starts[chain].copy(), precisions[chain].copy(), targets[chain].copy()
        for axis_index, (axis, index) in enumerate(zip(axes, state, strict=True)):
            if not 0 <= index < len(axis.prior_mass) or axis.prior_mass[index] <= 0:
                raise ValueError("initial targets must preserve every original C constraint")
            image = axis.prefix_images[index]
            trees[axis.record_index].set_image(axis_index, image, response_width=response_width)
        for sweep in range(burnin+draws*sweeps_per_draw):
            for axis_index, axis in enumerate(axes):
                tree = trees[axis.record_index]
                logits = _axis_logits(tree, axis, axis_index, theta,
                                      precision[axis.record_index], response_width)
                # Gumbel-max works in log mass. Small positive states remain
                # in the declared chain even when exp(logmass) underflows.
                chosen = int(np.argmax(logits+rng.gumbel(size=len(logits))))
                if chosen != state[axis_index]:
                    following = axis.prefix_images[chosen]
                    tree.set_image(axis_index, following, response_width=response_width)
                    state[axis_index] = chosen
            theta = _draw_coefficients(trees, precision, response_width, rng)
            for lam, tree in zip(precision, trees, strict=True):
                tree.check_point(theta, lam)
            if not np.isposinf(nu):
                # Each true recording owns a raw-row Gamma draw; beta is global.
                for record_index, tree in enumerate(trees):
                    shape, rate = (nu+len(tree.rows))/2, (nu+tree.energy(theta))/2
                    precision[record_index] = float(rng.gamma(shape, 1/rate))
                if not np.isfinite(precision).all() or np.any(precision <= 0):
                    raise SamplingResolutionError("same-record Gamma draw is not finite and positive")
            if sweep >= burnin and (sweep-burnin+1)%sweeps_per_draw == 0:
                index = (sweep-burnin)//sweeps_per_draw
                theta_draws[chain, index], scale_draws[chain, index], target_draws[chain, index] = theta, precision, state
        updated += sum(tree.updated_leaves for tree in trees)
    return JointHistorySamples(theta_draws, scale_draws, target_draws, seeds, burnin,
        sweeps_per_draw, nu, len(rows), perf_counter()-start_time, updated,
        tuple(stop-start for start, stop in spans))


def rao_blackwell_tail_components(base_prefix, base_tail, axes, samples, *, current_axis,
                                  response_width, tail_noise_logdet, record_spans=None,
                                  prefix_source_errors=None, tail_source_errors=None,
                                  tail_noise_logdet_error=0.):
    """P-only posterior → complete current-q weights and normalized raw-tail laws.

    Ignore the sampled current q, retaining its conditional *whole* vocabulary.
    Other target axes, including T-only prior states, remain paired with their
    response/drift/precision draw. Tail values never change prefix weights.
    """
    p, t = (finite_real_array(value, name=name) for value, name in
        ((base_prefix, "actual prefix rows"), (base_tail, "actual tail rows")))
    if (p.ndim != 2 or t.ndim != 2 or p.shape[1] != t.shape[1] or not len(t)
            or len(p) != samples.raw_prefix_rows or not 0 <= current_axis < len(axes)
            or samples.coefficients.shape[-1] != p.shape[1]-1
            or samples.targets.shape[-1] != len(axes)):
        raise ValueError("same P/T chart, all retained q axes and original raw tail rows are required")
    axis = axes[current_axis]
    spans = ((0, len(p)),) if record_spans is None else tuple(record_spans)
    record_start, record_stop = spans[axis.record_index]
    positive = np.flatnonzero(axis.prior_mass > 0)
    prefix_start = min((axis.prefix_images[q].start for q in positive if len(axis.prefix_images[q].values)), default=0)
    prefix_stop = max((axis.prefix_images[q].start+len(axis.prefix_images[q].values) for q in positive), default=0)
    shape = (*samples.coefficients.shape[:2], len(axis.prior_mass))
    logweights = np.full(shape, -np.inf)
    logdensity = np.full(shape, -np.inf)
    for chain in range(shape[0]):
        for draw in range(shape[1]):
            theta = samples.coefficients[chain, draw]
            precision = samples.precisions[chain, draw, axis.record_index]
            vector = np.r_[theta[:response_width], 1.]
            prefix, pe = _original_residual(p[record_start+prefix_start:record_start+prefix_stop], theta,
                None if prefix_source_errors is None else prefix_source_errors[axis.record_index])
            tail, te = _original_residual(t, theta, tail_source_errors)
            for index, other in enumerate(axes):
                if index == current_axis or other.record_index != axis.record_index:
                    continue
                q = samples.targets[chain, draw, index]
                pe = _subtract_image(prefix, prefix_start, other.prefix_images[q], vector, pe)
                te = _subtract_image(tail, 0, other.tail_images[q], vector, te)
            logits = np.full(len(axis.prior_mass), -np.inf)
            for q in positive:
                rp, rt = prefix.copy(), tail.copy()
                qpe = _subtract_image(rp, prefix_start, axis.prefix_images[q], vector, pe)
                qte = _subtract_image(rt, 0, axis.tail_images[q], vector, te)
                ep, _ = _point_energy(rp, qpe, precision, rows=len(rp), name="Rao-Blackwell prefix mass")
                et, energy_error = _point_energy(rt, qte, precision, rows=len(rt), name="raw tail Gaussian")
                logits[q] = np.log(axis.prior_mass[q])-.5*precision*ep
                normalizer = len(t)*(np.log(2*np.pi)-np.log(precision))+tail_noise_logdet
                normalizer_error = tail_noise_logdet_error+_roundoff_gamma(4)*(
                    len(t)*(abs(np.log(2*np.pi))+abs(np.log(precision)))+abs(tail_noise_logdet))
                total_error = energy_error+.5*normalizer_error+_roundoff_gamma(2)*(
                    abs(normalizer)+precision*et)
                if not np.isfinite(total_error) or total_error > np.sqrt(np.finfo(float).eps)*len(t):
                    raise DiffuseProjectionResolutionError("normalized raw tail density is unresolved in nats")
                logdensity[chain, draw, q] = -.5*(
                    normalizer+precision*et)
            logweights[chain, draw] = logits-logsumexp(logits)
    _numeric(logweights[:, :, positive], logdensity[:, :, positive], name="complete Rao-Blackwell current target law")
    return frozen_array(logweights), frozen_array(logdensity)


def summarize_joint_tail(logweights, logdensity, *, current_prior, scoring_law, batch_size):
    """Proper finite posterior mixture with paired same-chain probability MCSE."""
    weights = np.asarray(logweights)
    density = np.asarray(logdensity)
    prior = finite_real_array(current_prior, name="declared current target prior")
    if (weights.ndim != 3 or weights.shape != density.shape or prior.shape != weights.shape[2:]
            or np.any(prior < 0) or not np.isclose(prior.sum(), 1.)
            or scoring_law not in {"whole_observed_current_q", "fixed_current_prior_conditional_tail"}):
        raise ValueError("complete paired current-q law and explicit scoring estimand are required")
    positive = np.flatnonzero(prior > 0)
    count = weights.shape[0]*weights.shape[1]
    joint = weights+density
    log_a = logsumexp(joint, axis=(0, 1))-np.log(count)
    log_b = logsumexp(weights, axis=(0, 1))-np.log(count)
    scores = np.full(len(prior), -np.inf)
    scores[positive] = log_a[positive]
    if scoring_law == "fixed_current_prior_conditional_tail":
        scores[positive] += np.log(prior[positive])-log_b[positive]
    centered = scores-np.max(scores)
    logprobability = centered-logsumexp(centered)
    probability = np.exp(logprobability)
    influence = np.zeros(weights.shape)
    raw_a = np.exp(joint[:, :, positive]-log_a[positive])
    influence[:, :, positive] = raw_a
    if scoring_law == "fixed_current_prior_conditional_tail":
        influence[:, :, positive] -= np.exp(weights[:, :, positive]-log_b[positive])
    influence -= np.sum(influence*probability[None, None, :], axis=2, keepdims=True)
    mcse, functional = [], []
    for q in range(len(prior)):
        mcse.append(0. if prior[q] == 0 else _batch_mcse(influence[:, :, q], batch_size))
        functional.append(_functional_diagnostics(influence[:, :, q]))
    prefix_mass = np.exp(log_b)
    return {"log_evidence": frozen_array(scores), "log_probability": frozen_array(logprobability),
        "probability": frozen_array(probability), "prefix_current_q_probability": frozen_array(prefix_mass),
        "log_A": frozen_array(log_a), "log_B": frozen_array(log_b),
        "log_probability_paired_batch_mcse": mcse, "functional_diagnostics": functional,
        "scoring_law": scoring_law, "finite_prefix_posterior_measure_proper": True,
        "prefix_weights_frozen_before_tail": True,
        "posterior_program": "same_record_joint_explicit_drift_discrete_target_Gibbs",
        "current_prior_occurrences": 1 if scoring_law == "whole_observed_current_q" else
            "one_P_Bayes_update_then_explicit_fixed_prior_conditional_tail_program",
        "actual_retained_states": count, "current_vocabulary_width": len(prior)}
