"""Reference outer integration for the general structured frontier.

One original source Gaussian prior, original physical per-record flat measure,
complete discrete sum, all current candidates, common positive P/F integration.
No fitter or permission issuer. Binary64 numerical reference, not a nats-certified
or finite-sample-certified producer. Input authority and fitted-model adapters have independent owners.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, solve_triangular
from scipy.special import logsumexp

from research.joint_flat_sparse import whiten_record_prefix
from research.latent_response_reference import selection_design
from research.response_frontier import sum_product
from research.structured_response_frontier import combine_record_factors
from transfer.noise_model import array_digest, metadata_digest


class FlatEndpointUnidentifiedError(ValueError):
    pass


def admit_physical_flat_root(root, columns):
    """Original physical-flat QR admission, shared by native and grouped charts."""
    if (root.shape != (columns, columns) or np.any(np.diag(root) == 0)
            or not np.isfinite(root).all()):
        raise FlatEndpointUnidentifiedError('physical_flat_reference_qr_unresolved')
    return float(np.log(np.abs(np.diag(root))).sum())


def physical_flat_geometry_from_white_blocks(blocks, columns):
    """One QR/volume for a physical group; individual segments may lack rank."""
    root = np.empty((0, columns))
    for block in blocks:
        root = np.linalg.qr(np.vstack((root, block)), mode='reduced')[1]
    return root, admit_physical_flat_root(root, columns)


@dataclass(frozen=True)
class RecordEndpoints:
    model: object
    prefix_samples: int
    full_samples: int


def _white(model, values):
    value, _ = whiten_record_prefix(model.carrier.noise, values)
    return value/model.carrier.scale


def _flat_geometry(model, stop):
    c = model.carrier.channels
    if stop < 2:
        raise FlatEndpointUnidentifiedError('physical_intercept_slope_not_identified_by_endpoint')
    t = (model.origin+np.arange(stop))/model.dictionary.sample_rate_hz
    raw = np.zeros((stop, c, 2*c))
    for channel in range(c):
        raw[:, channel, channel] = 1.
        raw[:, channel, c+channel] = t
    white = _white(model, raw).reshape(stop*c, 2*c)
    q, r = np.linalg.qr(white, mode='reduced')
    return q, r, admit_physical_flat_root(r, 2*c)


class ContinuousFrontierTarget:
    """Direct general pointwise caller; all record likelihoods precede one sum."""

    def __init__(self, records):
        self.records = tuple(records)
        if not self.records:
            raise ValueError('outer integration requires original structured record endpoints')
        first = self.records[0].model
        self.rank = first.factor.shape[2]
        self.dimension = self.rank+sum(2*r.model.carrier.channels for r in self.records)
        self.q_values = first.q_values
        self.budget = first.budget
        current_roster = first.rosters[first.current]
        self.log_prior = np.array([current_roster.log_mass[
            next(i for i, state in enumerate(current_roster.states) if state.q == q)]
            for q in self.q_values])
        if len(self.log_prior) != len(self.q_values):
            raise ValueError('current variable must be one complete q axis')
        ids = [r.model.record_id for r in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError('one original record/segment may contribute only once')
        self.flat_geometry = {}
        failures = {}
        for endpoint in ('P', 'F'):
            geometry = []
            try:
                for record in self.records:
                    model = record.model
                    if (not 0 < record.prefix_samples <= record.full_samples <= len(model.y)
                            or model.factor.shape[2] != self.rank
                            or model.q_values != self.q_values
                            or array_digest(model.mean, model.factor) != array_digest(first.mean, first.factor)
                            or model.current != first.current or model.variable_ids != first.variable_ids
                            or len(model.rosters) != len(first.rosters)
                            or any(a.states != b.states or not np.array_equal(a.log_mass, b.log_mass)
                                for a, b in zip(model.rosters, first.rosters, strict=True))):
                        raise ValueError('records must share one original source chart and nested endpoints')
                    geometry.append(_flat_geometry(model,
                        record.prefix_samples if endpoint == 'P' else record.full_samples))
                self.flat_geometry[endpoint] = tuple(geometry)
            except FlatEndpointUnidentifiedError as error:
                failures[endpoint] = str(error)
        if 'F' not in self.flat_geometry:
            raise FlatEndpointUnidentifiedError(failures['F'])
        self.endpoints = tuple(e for e in ('P', 'F') if e in self.flat_geometry)
        self.refusals = failures
        self.identity = metadata_digest({'records': [{'model': r.model.identity,
            'P': r.prefix_samples, 'F': r.full_samples} for r in self.records],
            'source_prior': 'one_standard_normal_u', 'flat_measure': 'original_physical_da',
            'endpoints': self.endpoints, 'refusals': failures})

    def _split(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape != (self.dimension,) or not np.isfinite(z).all():
            raise ValueError('continuous point must preserve all shared and physical coordinates')
        start, drifts = self.rank, []
        for record in self.records:
            end = start+2*record.model.carrier.channels
            drifts.append(z[start:end].reshape(2, record.model.carrier.channels))
            start = end
        return z[:self.rank], drifts

    def log_joint(self, z):
        u, drifts = self._split(z)
        source_prior = -.5*(self.rank*np.log(2*np.pi)+u@u)
        results = []
        for endpoint in self.endpoints:
            factors = [r.model.factors(u, drift,
                r.prefix_samples if endpoint == 'P' else r.full_samples)
                for r, drift in zip(self.records, drifts, strict=True)]
            point = factors[0] if len(factors) == 1 else combine_record_factors(factors)
            partition = sum_product(point, maximum_intermediate_entries=self.budget)
            results.append(partition.log_joint_mass_by_q+source_prior)
        return np.array(results)

    def evaluate_many(self, points):
        return np.array([self.log_joint(z) for z in points])


@dataclass(frozen=True)
class GaussianBranch:
    q_index: int
    root: np.ndarray
    rhs: np.ndarray
    transform: np.ndarray
    log_constant: float
    residual: float
    log_integral: float


class GaussianAnchor:
    """Exact Gaussian branch integral; only for a declared enumerable anchor.

    It is an oracle/short-state target, not the long-history solver. Construction
    enumerates complete global variable assignments and refuses excess work.
    """

    def __init__(self, target, *, maximum_branches, progress=None):
        self.dimension, self.rank = target.dimension, target.rank
        self.q_values, self.log_prior = target.q_values, target.log_prior
        self.endpoints, self.refusals = target.endpoints, target.refusals
        self.original_target_identity = target.identity
        first = target.records[0].model
        rosters = first.rosters
        count = int(np.prod([len(r.states) for r in rosters], dtype=object))
        if count > maximum_branches:
            raise ValueError(f'complete analytic anchor needs {count} branches; no pruning permitted')
        self.branches = {}
        self.maximum_explicit_arrays_bytes = 0
        for endpoint in self.endpoints:
            branches = []
            geometry = target.flat_geometry[endpoint]
            transforms = [np.eye(target.rank)]+[g[1] for g in geometry]
            transform = block_diag(*transforms)
            log_measure = -sum(g[2] for g in geometry)
            for indices in product(*(range(len(r.states)) for r in rosters)):
                chosen = dict(zip(first.variable_ids, indices, strict=True))
                current_state = rosters[first.current].states[indices[first.current]]
                q_index = self.q_values.index(current_state.q)
                lp = sum(r.log_mass[i] for r, i in zip(rosters, indices, strict=True))
                prior_rows = np.zeros((target.rank, target.dimension+1))
                prior_rows[:, :target.rank] = np.eye(target.rank)
                compressed = prior_rows
                raw_bytes = 0
                flat_offset = target.rank
                likelihood_constant = 0.
                for record, (qflat, _, _ ) in zip(target.records, geometry, strict=True):
                    model = record.model
                    stop = record.prefix_samples if endpoint == 'P' else record.full_samples
                    design = np.zeros((stop, 2*model.dictionary.matrix.shape[1]))
                    for selection in model.selections:
                        q = rosters[model.variable_ids.index(selection.candidate_variable)].states[
                            chosen[selection.candidate_variable]].q
                        h = rosters[model.variable_ids.index(selection.shape_variable)].states[
                            chosen[selection.shape_variable]].shape
                        column = selection.membership.candidate_vocabulary.index(q)
                        part, _ = selection_design(model.dictionary, selection.onsets,
                            selection.membership.membership[:, column], origin_sample=model.origin,
                            observed_samples=stop, state=selection.shapes[h])
                        design += part
                    raw = np.einsum('nk,kcr->ncr', design, model.factor)
                    proper = _white(model, raw).reshape(stop*model.carrier.channels, target.rank)
                    b = _white(model, (model.y[:stop]-design@model.mean)[:, :, None]).ravel()
                    for start in range(0, len(b), 4096):
                        end = min(len(b), start+4096)
                        augmented = np.zeros((end-start, target.dimension+1))
                        augmented[:, :target.rank] = proper[start:end]
                        augmented[:, flat_offset:flat_offset+qflat.shape[1]] = qflat[start:end]
                        augmented[:, -1] = b[start:end]
                        stacked = np.vstack((compressed, augmented))
                        compressed = np.linalg.qr(stacked, mode='reduced')[1]
                        self.maximum_explicit_arrays_bytes = max(self.maximum_explicit_arrays_bytes,
                            raw.nbytes+proper.nbytes+b.nbytes+design.nbytes+qflat.nbytes
                            +augmented.nbytes+stacked.nbytes+compressed.nbytes)
                    flat_offset += qflat.shape[1]
                    raw_bytes = max(raw_bytes, raw.nbytes+proper.nbytes+b.nbytes+design.nbytes)
                    likelihood_constant += (model.constant+model.carrier.leading_log_jacobian(stop)
                        -.5*len(b)*np.log(2*np.pi))
                r = compressed
                root, rhs = r[:target.dimension, :target.dimension], r[:target.dimension, -1]
                if root.shape != (target.dimension, target.dimension) or np.any(np.diag(root) == 0):
                    raise FlatEndpointUnidentifiedError('joint_source_physical_flat_anchor_not_identified')
                residual = float(r[target.dimension:, -1]@r[target.dimension:, -1])
                constant = float(lp+likelihood_constant-.5*target.rank*np.log(2*np.pi))
                integral = (constant+.5*target.dimension*np.log(2*np.pi)+log_measure
                    -np.log(np.abs(np.diag(root))).sum()-.5*residual)
                branches.append(GaussianBranch(q_index, root, rhs, transform, constant, residual, float(integral)))
                self.maximum_explicit_arrays_bytes = max(self.maximum_explicit_arrays_bytes, raw_bytes+r.nbytes)
                if progress is not None:
                    progress(endpoint, len(branches), count)
            self.branches[endpoint] = tuple(branches)
        self.branch_count = count
        self.identity = metadata_digest({'target': target.identity, 'enumerable_branches': count,
            'purpose': 'analytic_gaussian_anchor_not_long_history_solver',
            'gaussian_branch_integrations': count*len(self.endpoints)})

    def evaluate_many(self, points):
        points = np.asarray(points)
        result = np.full((len(points), len(self.endpoints), len(self.q_values)), -np.inf)
        for endpoint_index, endpoint in enumerate(self.endpoints):
            for branch in self.branches[endpoint]:
                residual = (points@branch.transform.T)@branch.root.T-branch.rhs
                value = branch.log_constant-.5*(np.sum(residual*residual, axis=1)+branch.residual)
                q = branch.q_index
                result[:, endpoint_index, q] = np.logaddexp(result[:, endpoint_index, q], value)
        return result

    def log_joint(self, point):
        return self.evaluate_many(np.asarray(point)[None, :])[0]

    def integrated_joint(self):
        return np.array([[logsumexp([b.log_integral for b in self.branches[e] if b.q_index == q])
            for q in range(len(self.q_values))] for e in self.endpoints])

    def oracle_moment_proposal(self):
        """Numerical oracle proposal; not an original C posterior or production fit."""
        branches = [b for e in self.endpoints for b in self.branches[e]]
        endpoint_scales = logsumexp(self.integrated_joint(), axis=1)
        log_weights = np.array([b.log_integral-endpoint_scales[i]
            for i, e in enumerate(self.endpoints) for b in self.branches[e]])
        weights = np.exp(log_weights-logsumexp(log_weights))
        means, factors = [], []
        for b in branches:
            inverse = solve_triangular(b.root, np.eye(self.dimension), lower=False)
            factors.append(np.linalg.solve(b.transform, inverse))
            means.append(np.linalg.solve(b.transform, solve_triangular(b.root, b.rhs, lower=False)))
        mean = weights@np.array(means)
        covariance = sum(w*(f@f.T+np.outer(m-mean, m-mean))
            for w, m, f in zip(weights, means, factors, strict=True))
        return GaussianProposal(mean, np.linalg.cholesky(covariance)), endpoint_scales

    def oracle_candidate_balanced_proposal(self):
        """Known proper reference, one Gaussian per endpoint/q full shape mixture.

        This oracle uses exact anchor integrals as pilot scales. Production must
        replace this pilot with its separately recorded, truth-free estimate.
        """
        scales = self.integrated_joint()
        components = []
        for endpoint_index, endpoint in enumerate(self.endpoints):
            for q in range(len(self.q_values)):
                branches = [b for b in self.branches[endpoint] if b.q_index == q]
                weights = np.exp([b.log_integral-scales[endpoint_index, q] for b in branches])
                means, factors = [], []
                for b in branches:
                    inverse = solve_triangular(b.root, np.eye(self.dimension), lower=False)
                    factors.append(np.linalg.solve(b.transform, inverse))
                    means.append(np.linalg.solve(b.transform,
                        solve_triangular(b.root, b.rhs, lower=False)))
                mean = weights@np.array(means)
                covariance = sum(w*(f@f.T+np.outer(m-mean, m-mean))
                    for w, m, f in zip(weights, means, factors, strict=True))
                components.append(GaussianProposal(mean, np.linalg.cholesky(covariance)))
        return GaussianMixtureProposal(tuple(components),
            np.full(len(components), -np.log(len(components)))), scales


@dataclass(frozen=True)
class GaussianProposal:
    mean: np.ndarray
    root: np.ndarray

    def __post_init__(self):
        mean, root = np.array(self.mean, dtype=float, copy=True), np.array(self.root, dtype=float, copy=True)
        if (mean.ndim != 1 or not len(mean) or root.shape != (len(mean), len(mean))
                or not np.isfinite(mean).all() or not np.isfinite(root).all()
                or np.any(np.triu(root, 1) != 0) or np.any(np.diag(root) <= 0)):
            raise ValueError('proper Gaussian proposal requires finite lower Cholesky with positive diagonal')
        mean.flags.writeable = root.flags.writeable = False
        object.__setattr__(self, 'mean', mean)
        object.__setattr__(self, 'root', root)

    def draw(self, rng, count):
        return self.mean+rng.normal(size=(count, len(self.mean)))@self.root.T

    def log_density(self, points):
        white = solve_triangular(self.root, (points-self.mean).T, lower=True).T
        return (-.5*len(self.mean)*np.log(2*np.pi)-np.log(np.diag(self.root)).sum()
            -.5*np.sum(white*white, axis=1))

    def reversible_move(self, rng, points, scale):
        return (self.mean+np.sqrt(1-scale**2)*(points-self.mean)
            +scale*rng.normal(size=points.shape)@self.root.T)

    def record(self):
        return {'kind': 'proper_gaussian', 'mean': self.mean.tolist(), 'root': self.root.tolist()}


@dataclass(frozen=True)
class GaussianMixtureProposal:
    components: tuple[GaussianProposal, ...]
    log_weights: np.ndarray

    def __post_init__(self):
        components = tuple(self.components)
        weights = np.asarray(self.log_weights, dtype=float)
        if (not components or weights.shape != (len(components),) or not np.isfinite(weights).all()
                or any(len(c.mean) != len(components[0].mean) for c in components)):
            raise ValueError('proper mixture requires all finite positive components in one coordinate chart')
        weights = weights-logsumexp(weights)
        weights.flags.writeable = False
        object.__setattr__(self, 'components', components)
        object.__setattr__(self, 'log_weights', weights)

    def _component_logs(self, points):
        return np.column_stack([component.log_density(points)+weight
            for component, weight in zip(self.components, self.log_weights, strict=True)])

    def log_density(self, points):
        return logsumexp(self._component_logs(points), axis=1)

    def draw(self, rng, count):
        selected = rng.choice(len(self.components), size=count, p=np.exp(self.log_weights))
        result = np.empty((count, len(self.components[0].mean)))
        for index, component in enumerate(self.components):
            mask = selected == index
            result[mask] = component.draw(rng, int(mask.sum()))
        return result

    def reversible_move(self, rng, points, scale):
        # Gibbs-select a component conditional on x, then a component-reversible
        # pCN. g(x)p(k|x)K_k(x,y)=w_k g_k(x)K_k(x,y) is symmetric in x,y.
        logs = self._component_logs(points)
        mass = np.exp(logs-logsumexp(logs, axis=1)[:, None])
        cdf = np.cumsum(mass, axis=1)
        cdf[:, -1] = 1.
        selected = np.sum(rng.random(len(points))[:, None] > cdf, axis=1)
        result = np.empty_like(points)
        for index, component in enumerate(self.components):
            mask = selected == index
            result[mask] = component.reversible_move(rng, points[mask], scale)
        return result

    def record(self):
        return {'kind': 'proper_finite_gaussian_mixture', 'log_weights': self.log_weights.tolist(),
            'components': [component.record() for component in self.components]}


def probabilities(log_joint, endpoints, log_prior, log_component_scales=None):
    if log_component_scales is not None:
        log_joint = log_joint+np.asarray(log_component_scales)
    f = log_joint[endpoints.index('F')]
    result = {'whole_log_probability': f-logsumexp(f)}
    if 'P' in endpoints:
        conditional = log_prior+f-log_joint[endpoints.index('P')]
        result['conditional_log_probability'] = conditional-logsumexp(conditional)
    return result


def _psi(joint, scales):
    if scales.ndim == 1:
        scales = scales[:, None]
    return logsumexp(joint-scales[None, :, :], axis=(1, 2))


def run_smc(target, proposal, endpoint_log_scales, *, inference_seed, particles,
            replicates, temperatures, moves, pcn_scale, observer=None):
    """Fixed-schedule tempered SMC with shared positive P/F and complete current q.

    Resampling each stage is unbiased systematic resampling. Each production
    replicate has an independent SeedSequence stream. MC errors use independent
    replicates; neither ESS nor a delta-method SE is a finite-sample certificate.
    """
    temperatures = np.asarray(temperatures, dtype=float)
    scales = np.asarray(endpoint_log_scales, dtype=float)
    if scales.shape == (len(target.endpoints),):
        scales = np.repeat(scales[:, None], len(target.q_values), axis=1)
    if (temperatures[0] != 0 or temperatures[-1] != 1 or np.any(np.diff(temperatures) <= 0)
            or scales.shape != (len(target.endpoints), len(target.q_values))
            or not np.isfinite(scales).all() or particles < 2 or replicates < 2
            or not 0 < pcn_scale <= 1 or moves < 0):
        raise ValueError('positive complete fixed inference configuration is required')
    streams = np.random.SeedSequence(inference_seed).spawn(replicates)
    output, arrays, component_estimates = [], {}, []
    for replicate, stream in enumerate(streams):
        rng = np.random.default_rng(stream)
        if observer is not None:
            observer('initial_draw', replicate=replicate, rng_state=copy.deepcopy(rng.bit_generator.state),
                bit_generator=type(rng.bit_generator), particles=particles)
        points = proposal.draw(rng, particles)
        joint = target.evaluate_many(points)
        log_g = proposal.log_density(points)
        log_psi = _psi(joint, scales)
        ell = log_psi-log_g
        log_z, trace = 0., []
        for stage, (previous, gamma) in enumerate(zip(temperatures[:-1], temperatures[1:], strict=True)):
            log_w = (gamma-previous)*ell
            norm = logsumexp(log_w)
            log_z += norm-np.log(particles)
            weights = np.exp(log_w-norm)
            ess = 1/float(weights@weights)
            cdf = np.cumsum(weights)
            cdf[-1] = 1.
            ancestors = np.searchsorted(cdf, (rng.random()+np.arange(particles))/particles)
            if observer is not None:
                observer('resample', replicate=replicate, stage=stage, gamma=float(gamma),
                    ancestors=ancestors.copy())
            points, joint = points[ancestors], joint[ancestors]
            log_g, log_psi, ell = log_g[ancestors], log_psi[ancestors], ell[ancestors]
            accepted = 0
            for move in range(moves):
                if observer is not None:
                    move_rng_state = copy.deepcopy(rng.bit_generator.state)
                trial = proposal.reversible_move(rng, points, pcn_scale)
                tj = target.evaluate_many(trial)
                tg = proposal.log_density(trial)
                tp = _psi(tj, scales)
                te = tp-tg
                accept = np.log(rng.random(particles)) < gamma*(te-ell)
                accepted += int(accept.sum())
                points[accept], joint[accept] = trial[accept], tj[accept]
                log_g[accept], log_psi[accept], ell[accept] = tg[accept], tp[accept], te[accept]
                if observer is not None:
                    observer('move', replicate=replicate, stage=stage, move=move,
                        rng_state=move_rng_state, rho=pcn_scale, accept=accept.copy())
            trace.append({'gamma': float(gamma), 'log_normalizer': float(log_z),
                'increment_ess_diagnostic': ess, 'accepted_moves': accepted,
                'proposed_moves': moves*particles, 'ancestors': ancestors.tolist()})
        component = (log_z+logsumexp(joint-scales[None, :, :]-log_psi[:, None, None], axis=0)
            -np.log(particles))
        component_estimates.append(component)
        key = f'replicate_{replicate}'
        arrays[key+'_points'], arrays[key+'_joint'] = points, joint
        arrays[key+'_log_g'], arrays[key+'_log_psi'] = log_g, log_psi
        arrays[key+'_scaled_log_component'] = component
        if observer is not None:
            observer('replicate_complete', replicate=replicate, points_sha256=array_digest(points),
                points_shape=points.shape)
        output.append({'replicate': replicate, 'spawn_key': list(stream.spawn_key),
            'log_normalizer': float(log_z), 'trace': trace})
    components = np.array(component_estimates)
    pooled = logsumexp(components, axis=0)-np.log(replicates)
    result = probabilities(pooled, target.endpoints, target.log_prior, scales)
    # Relative component estimates have mean 1 and are <=R, even when P/F
    # absolute potentials or different candidate masses differ by >1000 nats.
    values = np.exp(components-pooled[None, :, :]).reshape(replicates, -1)
    covariance = np.cov(values, rowvar=False, ddof=1)/replicates
    errors = {}
    j = len(target.q_values)
    if np.isfinite(covariance).all():
        fidx = target.endpoints.index('F')
        fp = np.exp(result['whole_log_probability'])
        whole_gradient = np.zeros((j, covariance.shape[0]))
        whole_gradient[:, fidx*j:(fidx+1)*j] = np.eye(j)-np.ones((j, 1))*fp[None, :]
        errors['whole_log_probability_se'] = np.sqrt(np.maximum(0,
            np.diag(whole_gradient@covariance@whole_gradient.T))).tolist()
        if 'P' in target.endpoints:
            pidx = target.endpoints.index('P')
            logits_gradient = np.zeros((j, covariance.shape[0]))
            logits_gradient[:, fidx*j:(fidx+1)*j] = np.eye(j)
            logits_gradient[:, pidx*j:(pidx+1)*j] = -np.eye(j)
            cp = np.exp(result['conditional_log_probability'])
            gradient = (np.eye(j)-np.ones((j, 1))*cp[None, :])@logits_gradient
            errors['conditional_log_probability_se'] = np.sqrt(np.maximum(0,
                np.diag(gradient@covariance@gradient.T))).tolist()
    return {'schema': 'n2p3_collapsed_continuous_reference_result/2', 'target_identity': target.identity,
        'quality_kind': 'replicated_mc_estimate', 'finite_sample_certificate': False,
        'numerical_error_role': 'binary64_reference_not_absolute_nats_certificate',
        'endpoints': list(target.endpoints), 'endpoint_refusals': target.refusals,
        'q_values': list(target.q_values), 'log_current_prior': target.log_prior.tolist(),
        'component_log_scales': scales.tolist(), 'inference_seed': inference_seed,
        'proper_proposal': proposal.record(),
        'source_seed': None, 'particles': particles, 'replicates': replicates,
        'temperatures': temperatures.tolist(), 'moves': moves, 'pcn_scale': pcn_scale,
        'replicate_results': output, 'pooled_scaled_log_components': pooled.tolist(),
        'probabilities': {k: v.tolist() for k, v in result.items()}, 'estimated_mc_error': errors,
        'source_fits': 0, 'target_fits': 0}, arrays


def save_cloud(path, result, arrays):
    path = Path(path)
    np.savez_compressed(path, **arrays)
    result['saved_cloud'] = {'path': str(path), 'bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def replay_saved_cloud(result):
    path = Path(result['saved_cloud']['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != result['saved_cloud']['sha256']:
        raise ValueError('saved common cloud changed actual bytes')
    scales = np.array(result.get('component_log_scales', result.get('endpoint_log_scales')))
    if scales.ndim == 1:
        scales = np.repeat(scales[:, None], len(result['q_values']), axis=1)
    components = []
    with np.load(path, allow_pickle=False) as data:
        for row in result['replicate_results']:
            key = f"replicate_{row['replicate']}"
            joint, psi = data[key+'_joint'], data[key+'_log_psi']
            expected_psi = _psi(joint, scales)
            if not np.array_equal(psi, expected_psi):
                raise ValueError('saved P/F common target differs from complete joint components')
            component = (row['log_normalizer']+logsumexp(joint-scales[None, :, :]
                -psi[:, None, None], axis=0)-np.log(result['particles']))
            if not np.array_equal(component, data[key+'_scaled_log_component']):
                raise ValueError('saved SMC component estimator changed common normalizer or weights')
            components.append(component)
    pooled = logsumexp(components, axis=0)-np.log(result['replicates'])
    if not np.array_equal(pooled, np.array(result['pooled_scaled_log_components'])):
        raise ValueError('saved replicate pooling changed unnormalized estimator')
    replay = probabilities(pooled, tuple(result['endpoints']), np.array(result['log_current_prior']), scales)
    if any(not np.array_equal(value, np.array(result['probabilities'][key])) for key, value in replay.items()):
        raise ValueError('saved inference output differs from actual common-cloud replay')
    return replay


def save_gaussian_anchor(path, anchor):
    """Save numeric integration oracle without source/C fit or future sampling."""
    import json

    arrays, endpoint_rows = {}, {}
    for endpoint in anchor.endpoints:
        rows = []
        for index, branch in enumerate(anchor.branches[endpoint]):
            key = f'{endpoint}_{index}'
            arrays[key+'_root'], arrays[key+'_rhs'] = branch.root, branch.rhs
            arrays[key+'_transform'] = branch.transform
            rows.append({'q_index': branch.q_index, 'log_constant': branch.log_constant,
                'residual': branch.residual, 'log_integral': branch.log_integral, 'key': key})
        endpoint_rows[endpoint] = rows
    path = Path(path)
    np.savez_compressed(path, **arrays)
    metadata = {'schema': 'n2p3_complete_gaussian_integration_anchor/1',
        'identity': anchor.identity, 'original_target_identity': anchor.original_target_identity,
        'dimension': anchor.dimension, 'rank': anchor.rank, 'q_values': list(anchor.q_values),
        'log_prior': anchor.log_prior.tolist(), 'endpoints': list(anchor.endpoints),
        'refusals': anchor.refusals, 'branch_count': anchor.branch_count,
        'maximum_explicit_arrays_bytes': anchor.maximum_explicit_arrays_bytes,
        'branches': endpoint_rows, 'npz': {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return metadata


def load_gaussian_anchor(path, *, expected_metadata_sha256):
    import json

    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_metadata_sha256:
        raise ValueError('Gaussian anchor metadata differs from external expected SHA256')
    metadata = json.loads(content)
    numerical = Path(metadata['npz']['path'])
    if hashlib.sha256(numerical.read_bytes()).hexdigest() != metadata['npz']['sha256']:
        raise ValueError('original Gaussian anchor numerical bytes changed')
    result = object.__new__(GaussianAnchor)
    for name in ('identity', 'original_target_identity', 'dimension', 'rank', 'refusals',
            'branch_count', 'maximum_explicit_arrays_bytes'):
        setattr(result, name, metadata[name])
    result.endpoints, result.q_values = tuple(metadata['endpoints']), tuple(metadata['q_values'])
    result.log_prior = np.array(metadata['log_prior'])
    result.branches = {}
    with np.load(numerical, allow_pickle=False) as data:
        for endpoint in result.endpoints:
            result.branches[endpoint] = tuple(GaussianBranch(row['q_index'],
                data[row['key']+'_root'].copy(), data[row['key']+'_rhs'].copy(),
                data[row['key']+'_transform'].copy(), row['log_constant'], row['residual'],
                row['log_integral']) for row in metadata['branches'][endpoint])
    return result
