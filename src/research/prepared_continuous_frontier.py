"""Prepared local quadratics for the unchanged structured Gaussian likelihood.

Compile native row blocks once; score complete scopes many times. Cache matrices
are per record, never all-record d² and never N×r. The current implementation is
a binary64 reference producer with explicit resource accounting.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

from research.joint_flat_sparse import compile_sparse_response_image
from research.latent_response_reference import selection_design
from research.response_frontier import LogFactor, PointFactors, Roster, StateKey, sum_product
from research.structured_continuous_proposal import proposal_from_record
from research.structured_response_frontier import _elimination_order, combine_record_factors
from transfer.noise_model import array_digest, metadata_digest


def _digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _white_block(base, raw, raw_start, begin, end):
    owned = compile_sparse_response_image(base.carrier.noise, raw,
        start_native=raw_start, total_native=end)
    c = base.carrier.channels
    start = begin*c-owned.image.start
    value = owned.image.values[start:start+(end-begin)*c]/base.carrier.scale
    if len(value) != (end-begin)*c:
        raise ValueError('actual original noise map did not cover the requested row block')
    return value


def _raw_affine(base, selection, state, start, stop):
    column = selection.membership.candidate_vocabulary.index(state.q)
    design, _ = selection_design(base.dictionary, selection.onsets,
        selection.membership.membership[:, column], origin_sample=base.origin+start,
        observed_samples=stop-start, state=selection.shapes[state.shape])
    value = np.empty((stop-start, base.carrier.channels, 1+base.factor.shape[2]))
    value[:, :, 0] = design@base.mean
    value[:, :, 1:] = np.einsum('nk,kcr->ncr', design, base.factor)
    return value


@dataclass(frozen=True)
class QuadraticFactor:
    scope: tuple[int, ...]
    coefficients: np.ndarray
    proper_only: bool

    def values(self, coordinates):
        vector = coordinates[:self.coefficients.shape[-1]] if self.proper_only else coordinates
        return np.einsum('i,...ij,j->...', vector, self.coefficients, vector, optimize=False)


@dataclass(frozen=True)
class PreparedEndpoint:
    stop: int
    common: np.ndarray
    factors: tuple[QuadraticFactor, ...]
    log_normalizer: float


class PreparedRecordFrontier:
    """Per-record finite-scope quadratic cache, with original operands binding."""

    def __init__(self, base, stops, *, block_rows, maximum_cache_bytes):
        stops = tuple(sorted(set(stops)))
        self.record_id, self.origin = base.record_id, base.origin
        self.rank, self.channels = base.factor.shape[2], base.carrier.channels
        self.dimension = self.rank+2*self.channels
        self.rosters, self.variable_ids = base.rosters, base.variable_ids
        self.current, self.q_values, self.budget = base.current, base.q_values, base.budget
        self.base_identity = base.identity
        self.source_chart = metadata_digest({'mean': array_digest(base.mean), 'factor': array_digest(base.factor)})
        # Numerical flat admission stays with the original continuous owner.
        # Save its witness/status rather than infer admission from the row count.
        from research.collapsed_continuous_inference import (
            FlatEndpointUnidentifiedError,
            _flat_geometry,
        )

        self.flat_admission = {}
        for stop in stops:
            try:
                _, root, volume = _flat_geometry(base, stop)
                self.flat_admission[stop] = {'identified': True, 'physical_root': root.tolist(),
                    'log_chart_volume': volume, 'owner': 'original_continuous_flat_geometry/1',
                    'original_operands_identity': base.identity}
            except FlatEndpointUnidentifiedError as error:
                self.flat_admission[stop] = {'identified': False, 'reason': str(error),
                    'owner': 'original_continuous_flat_geometry/1', 'original_operands_identity': base.identity}
        if type(block_rows) is not int or block_rows < 1:
            raise ValueError('prepared row block must be explicitly positive')
        stops = tuple(sorted(set(stops)))
        self.block_rows, self.maximum_cache_bytes = block_rows, maximum_cache_bytes
        plans, estimated_bytes = {}, 0
        for stop in stops:
            if not 1 <= stop <= len(base.y):
                raise ValueError('prepared endpoint exceeds original observation support')
            selections = []
            for selection in base.selections:
                scope, widths, states = base._local_states(selection)
                left, right = base._support(selection, stop)
                if left < right:
                    white_left = 0 if left < base.carrier.noise.conditioning_samples else left
                    white_right = min(stop, max(right+base.carrier.noise.order,
                        base.carrier.noise.conditioning_samples if white_left == 0 else 0))
                else:
                    white_left = white_right = stop
                selections.append((selection, scope, widths, states, white_left, white_right))
                estimated_bytes += prod(widths)*(self.dimension+1)**2*8
            pairs = []
            for right, current in enumerate(selections):
                for left, old in enumerate(selections[:right]):
                    if max(old[4], current[4]) >= min(old[5], current[5]):
                        continue
                    scope = tuple(sorted(set(old[1]) | set(current[1])))
                    widths = tuple(len(self.rosters[index].states) for index in scope)
                    if prod(widths) > self.budget:
                        raise ValueError('complete prepared overlap scope exceeds table budget')
                    pairs.append((left, right, scope, widths))
                    estimated_bytes += prod(widths)*(self.rank+1)**2*8
            estimated_bytes += (self.dimension+1)**2*8
            plans[stop] = (tuple(selections), tuple(pairs))
        if estimated_bytes > maximum_cache_bytes:
            raise ValueError(f'complete prepared cache needs {estimated_bytes} bytes; no state pruning')
        self.estimated_cache_bytes = estimated_bytes
        self.endpoints = {}
        self.maximum_named_workspace_bytes = 0
        for stop, (selections, pairs) in plans.items():
            self.endpoints[stop] = self._compile(base, stop, selections, pairs)
        self.identity = self._identity()

    def _compile(self, base, stop, selections, pairs):
        d, m, c = self.dimension+1, self.rank+1, self.channels
        common = np.zeros((d, d))
        unary = [np.zeros((*row[2], d, d)) for row in selections]
        pair = [np.zeros((*row[3], m, m)) for row in pairs]
        prefix, order = base.carrier.noise.conditioning_samples, base.carrier.noise.order
        for begin in range(0, stop, self.block_rows):
            end = min(stop, begin+self.block_rows)
            raw_start = 0 if begin < prefix+order else begin-order
            raw_common = np.zeros((end-raw_start, c, d))
            raw_common[:, :, 0] = base.y[raw_start:end]
            times = (base.origin+np.arange(raw_start, end))/base.dictionary.sample_rate_hz
            for channel in range(c):
                raw_common[:, channel, m+channel] = -1.
                raw_common[:, channel, m+c+channel] = -times
            active, native_references = [], {}
            for index, row in enumerate(selections):
                if row[4] >= end or row[5] <= begin:
                    continue
                reference = _raw_affine(base, row[0], row[3][0], raw_start, end)
                raw_common[:, :, :m] -= reference
                native_references[index] = reference
                active.append(index)
            b = _white_block(base, raw_common, raw_start, begin, end)
            common -= .5*(b.T@b)
            images = {}
            for index in active:
                row = selections[index]
                values = []
                for state in row[3]:
                    raw = _raw_affine(base, row[0], state, raw_start, end)-native_references[index]
                    values.append(_white_block(base, raw, raw_start, begin, end))
                image = np.stack(values)
                images[index] = image
                table = unary[index].reshape(-1, d, d)
                for state_index, value in enumerate(image):
                    cross = b.T@value
                    table[state_index, :, :m] += .5*cross
                    table[state_index, :m, :] += .5*cross.T
                    table[state_index, :m, :m] -= .5*(value.T@value)
            for pair_index, (left, right, scope, widths) in enumerate(pairs):
                if left not in images or right not in images:
                    continue
                lrow, rrow = selections[left], selections[right]
                for coordinates in np.ndindex(widths):
                    selected = dict(zip(scope, coordinates, strict=True))
                    li = np.ravel_multi_index(tuple(selected[v] for v in lrow[1]), lrow[2])
                    ri = np.ravel_multi_index(tuple(selected[v] for v in rrow[1]), rrow[2])
                    cross = images[left][li].T@images[right][ri]
                    pair[pair_index][coordinates] -= .5*(cross+cross.T)
            self.maximum_named_workspace_bytes = max(self.maximum_named_workspace_bytes,
                raw_common.nbytes+b.nbytes+sum(x.nbytes for x in images.values())
                +sum(x.nbytes for x in native_references.values()))
        factors = [QuadraticFactor(row[1], value, False) for row, value in zip(selections, unary, strict=True)]
        factors.extend(QuadraticFactor(row[2], value, True) for row, value in zip(pairs, pair, strict=True))
        constant = base.constant+base.carrier.leading_log_jacobian(stop)-.5*stop*c*np.log(2*np.pi)
        return PreparedEndpoint(stop, common, tuple(factors), float(constant))

    def _identity(self):
        return metadata_digest({'base_identity': self.base_identity, 'block_rows': self.block_rows,
            'source_chart': self.source_chart, 'producer_sha256': _digest_file(__file__),
            'record': [self.record_id, self.origin, self.rank, self.channels],
            'flat_admission': self.flat_admission,
            'current': self.current, 'q_values': self.q_values,
            'rosters': [{'id': r.selection_id, 'states': [vars(s) for s in r.states],
                'log_mass': array_digest(r.log_mass)} for r in self.rosters],
            'endpoints': {str(stop): {'common': array_digest(endpoint.common),
                'normalizer': endpoint.log_normalizer, 'factors': [{'array': array_digest(f.coefficients),
                    'scope': f.scope, 'proper_only': f.proper_only} for f in endpoint.factors]}
                for stop, endpoint in self.endpoints.items()}})

    def factors(self, u, drift, stop):
        endpoint = self.endpoints[stop]
        u, drift = np.asarray(u), np.asarray(drift)
        if u.shape != (self.rank,) or drift.shape != (2, self.channels):
            raise ValueError('prepared point must use the original proper and physical drift chart')
        vector = np.r_[1., u, drift.ravel()]
        factors = tuple(LogFactor(f.scope, f.values(vector)) for f in endpoint.factors)
        return PointFactors(self.rosters, factors, self.current, self.q_values,
            float(endpoint.log_normalizer+vector@endpoint.common@vector),
            _elimination_order(self.rosters, factors, self.current),
            {'physical_record_segment_id': self.record_id, 'source_coefficient_chart_sha256': self.source_chart,
             'fixed_source_point_sha256': array_digest(u), 'physical_drift_sha256': array_digest(drift),
             'prepared_model_identity': self.identity, 'original_operands_sha256': self.base_identity,
             'original_native_support': [self.origin, self.origin+stop],
             'numerical_error_role': 'binary64_prepared_quadratic_reference_not_nats_certificate'})

    def save(self, path):
        path = Path(path)
        arrays, endpoints = {}, {}
        for stop, endpoint in self.endpoints.items():
            common_key = f'endpoint_{stop}_common'
            arrays[common_key] = endpoint.common
            rows = []
            for index, factor in enumerate(endpoint.factors):
                key = f'endpoint_{stop}_factor_{index}'
                arrays[key] = factor.coefficients
                rows.append({'key': key, 'scope': list(factor.scope), 'proper_only': factor.proper_only})
            endpoints[str(stop)] = {'common_key': common_key, 'log_normalizer': endpoint.log_normalizer, 'factors': rows}
        np.savez_compressed(path, **arrays)
        metadata = {'schema': 'n2p3_prepared_record_quadratic_frontier/1',
            'identity': self.identity, 'base_identity': self.base_identity, 'source_chart': self.source_chart,
            'producer_sha256': _digest_file(__file__), 'record_id': self.record_id, 'origin': self.origin,
            'rank': self.rank, 'channels': self.channels, 'dimension': self.dimension,
            'current': self.current, 'q_values': list(self.q_values), 'budget': self.budget,
            'variable_ids': list(self.variable_ids), 'rosters': [{'id': r.selection_id,
                'states': [vars(s) for s in r.states], 'log_mass': r.log_mass.tolist()} for r in self.rosters],
            'block_rows': self.block_rows, 'maximum_cache_bytes': self.maximum_cache_bytes,
            'estimated_cache_bytes': self.estimated_cache_bytes,
            'maximum_named_workspace_bytes': self.maximum_named_workspace_bytes, 'endpoints': endpoints,
            'flat_admission': self.flat_admission,
            'payload': {'path': str(path), 'sha256': _digest_file(path), 'bytes': path.stat().st_size}}
        path.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        return metadata

    @classmethod
    def load(cls, metadata_path, *, expected_metadata_sha256):
        content = Path(metadata_path).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_metadata_sha256:
            raise ValueError('prepared model metadata differs from external admission')
        metadata = json.loads(content)
        if _digest_file(metadata['payload']['path']) != metadata['payload']['sha256']:
            raise ValueError('prepared model numeric cache changed original bytes')
        if metadata['producer_sha256'] != _digest_file(__file__):
            raise ValueError('prepared model requires its exact numerical compiler source')
        obj = object.__new__(cls)
        for key in ('identity', 'base_identity', 'source_chart', 'record_id', 'origin', 'rank', 'channels',
                'dimension', 'current', 'budget', 'block_rows', 'maximum_cache_bytes',
                'estimated_cache_bytes', 'maximum_named_workspace_bytes'):
            setattr(obj, key, metadata[key])
        obj.q_values, obj.variable_ids = tuple(metadata['q_values']), tuple(metadata['variable_ids'])
        obj.flat_admission = {int(stop): row for stop, row in metadata['flat_admission'].items()}
        obj.rosters = tuple(Roster(row['id'], tuple(StateKey(**s) for s in row['states']),
            np.array(row['log_mass'])) for row in metadata['rosters'])
        obj.endpoints = {}
        with np.load(metadata['payload']['path'], allow_pickle=False) as data:
            for stop, endpoint in metadata['endpoints'].items():
                obj.endpoints[int(stop)] = PreparedEndpoint(int(stop), data[endpoint['common_key']].copy(),
                    tuple(QuadraticFactor(tuple(f['scope']), data[f['key']].copy(), f['proper_only'])
                        for f in endpoint['factors']), endpoint['log_normalizer'])
        if obj._identity() != obj.identity:
            raise ValueError('prepared cache differs from original operand/compiler identity')
        return obj


class PreparedJointTarget:
    """Cold-loadable numeric model: no native input reads, design or whitening."""

    def __init__(self, records):
        self.records = tuple(records)  # (prepared model, P stop, F stop)
        first = self.records[0][0]
        if len({m.record_id for m, _, _ in self.records}) != len(self.records):
            raise ValueError('one physical record/segment cannot enter prepared target twice')
        for model, p, f in self.records:
            if (model.rank != first.rank or model.source_chart != first.source_chart
                    or model.q_values != first.q_values or model.current != first.current
                    or model.variable_ids != first.variable_ids or p not in model.endpoints or f not in model.endpoints
                    or not 1 <= p <= f
                    or any(a.states != b.states or not np.array_equal(a.log_mass, b.log_mass)
                        for a, b in zip(model.rosters, first.rosters, strict=True))):
                raise ValueError('prepared records require the same complete source/prior/group chart')
        self.rank, self.q_values = first.rank, first.q_values
        self.dimension = self.rank+sum(2*m.channels for m, _, _ in self.records)
        self.budget = min(m.budget for m, _, _ in self.records)
        self.log_prior = np.array([first.rosters[first.current].log_mass[
            next(i for i, s in enumerate(first.rosters[first.current].states) if s.q == q)] for q in self.q_values])
        self.endpoints = tuple(e for e in ('P', 'F') if all(
            m.flat_admission[p if e == 'P' else f]['identified'] for m, p, f in self.records))
        self.refusals = {e: [{'record': m.record_id, 'admission': m.flat_admission[p if e == 'P' else f]}
            for m, p, f in self.records if not m.flat_admission[p if e == 'P' else f]['identified']]
            for e in ('P', 'F') if e not in self.endpoints}
        if 'F' not in self.endpoints:
            raise ValueError('full endpoint physical flat measure is unidentified')
        self.identity = metadata_digest({'models': [m.identity for m, _, _ in self.records],
            'cuts': [[p, f] for _, p, f in self.records], 'endpoints': self.endpoints})

    def log_joint(self, z):
        u, offset, drifts = z[:self.rank], self.rank, []
        for model, _, _ in self.records:
            drifts.append(z[offset:offset+2*model.channels].reshape(2, model.channels))
            offset += 2*model.channels
        source = -.5*(self.rank*np.log(2*np.pi)+u@u)
        result = []
        for endpoint in self.endpoints:
            points = [model.factors(u, drift, p if endpoint == 'P' else f)
                for (model, p, f), drift in zip(self.records, drifts, strict=True)]
            point = points[0] if len(points) == 1 else combine_record_factors(points)
            result.append(sum_product(point, maximum_intermediate_entries=self.budget).log_joint_mass_by_q+source)
        return np.array(result)

    def evaluate_many(self, points):
        return np.array([self.log_joint(z) for z in points])

    def numerical_reference(self):
        from research.collapsed_continuous_inference import GaussianProposal

        means, covariances = [np.zeros(self.rank)], [np.eye(self.rank)]
        for model, _, stop in self.records:
            matrix = model.endpoints[stop].common
            flat = slice(1+self.rank, None)
            precision = -2*matrix[flat, flat]
            linear = 2*matrix[flat, 0]
            means.append(np.linalg.solve(precision, linear))
            covariances.append(np.linalg.solve(precision, np.eye(len(precision))))
        return GaussianProposal(np.concatenate(means), np.linalg.cholesky(block_diag(*covariances)))




def replay_model_cloud(target, result):
    """Second layer: reevaluate saved points -> complete J, g and Psi."""
    from research.collapsed_continuous_inference import _psi, replay_saved_cloud

    replay_saved_cloud(result)
    if target.identity != result['target_identity']:
        raise ValueError('saved inference requires its original compiled model identity')
    proposal = proposal_from_record(result['proper_proposal'])
    scales = np.array(result['component_log_scales'])
    with np.load(result['saved_cloud']['path'], allow_pickle=False) as data:
        for row in result['replicate_results']:
            key = f"replicate_{row['replicate']}"
            points = data[key+'_points']
            joint, log_g = target.evaluate_many(points), proposal.log_density(points)
            if (not np.array_equal(joint, data[key+'_joint']) or not np.array_equal(log_g, data[key+'_log_g'])
                    or not np.array_equal(_psi(joint, scales), data[key+'_log_psi'])):
                raise ValueError('saved continuous cloud differs from actual compiled-model/proposal replay')
    return {'layer': 'model_only', 'bitexact': True, 'source_fits': 0, 'resampling_calls': 0}


def replay_original_parent(original_target, saved_models, result):
    """Third numeric layer: caller supplies actual original-authority target.

    The real data issuer remains the caller's owner; this function never creates
    authority from a JSON manifest. It verifies original native model identities
    before re-compiling the same complete quadratics and re-evaluating the cloud.
    """
    models = []
    if len(original_target.records) != len(saved_models):
        raise ValueError('original parent changed its complete physical record set')
    for record, saved in zip(original_target.records, saved_models, strict=True):
        model, p, f = saved
        if (record.model.identity != model.base_identity or record.prefix_samples != p or record.full_samples != f):
            raise ValueError('original parent changed native inputs/source/prior/record support')
        rebuilt = PreparedRecordFrontier(record.model, tuple(model.endpoints),
            block_rows=model.block_rows, maximum_cache_bytes=model.maximum_cache_bytes)
        if rebuilt.identity != model.identity:
            raise ValueError('recompiled original parent changed complete prepared factors')
        models.append((rebuilt, p, f))
    replay_model_cloud(PreparedJointTarget(models), result)
    return {'layer': 'original_parent', 'original_records_recompiled': len(models),
        'bitexact': True, 'source_fits': 0, 'resampling_calls': 0}
