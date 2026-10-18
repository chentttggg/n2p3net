"""Physical affine callbacks, local quadratics and explicit shared drift groups.

Noise is segment-local. Physical time, response variables and drift coordinates
are shared only by their declared owners. This module grants no raw authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from math import prod
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from research.collapsed_continuous_inference import (
    FlatEndpointUnidentifiedError,
    physical_flat_geometry_from_white_blocks,
)
from research.joint_flat_sparse import compile_sparse_response_image
from research.prepared_continuous_frontier import PreparedEndpoint, QuadraticFactor
from research.response_frontier import LogFactor, PointFactors, Roster, StateKey, sum_product
from research.structured_response_frontier import _elimination_order, combine_record_factors
from transfer.noise_model import array_digest, frozen_array, metadata_digest

_RAW_SEGMENT_ADMISSIONS = WeakKeyDictionary()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class PhysicalDriftGroup:
    group_id: str
    channels: tuple[str, ...]
    reference_origin_seconds: Fraction
    clock_program: str
    clock_domain: str

    def __post_init__(self):
        if (not self.group_id or not self.channels or len(set(self.channels)) != len(self.channels)
                or self.clock_program not in {'timeline_wall', 'delivery_anchor'} or not self.clock_domain):
            raise ValueError('physical drift group requires its unique coordinate channel names')
        object.__setattr__(self, 'reference_origin_seconds', Fraction(self.reference_origin_seconds))

    @property
    def identity(self):
        ref = self.reference_origin_seconds
        return metadata_digest({'group': self.group_id, 'channels': self.channels,
            'origin_seconds': [ref.numerator, ref.denominator], 'units': ['V', 'V/s'],
            'clock_program': self.clock_program, 'clock_domain': self.clock_domain,
            'measure': 'one_physical_da_per_group'})

    def local_columns(self, channels):
        indices = [self.channels.index(channel) for channel in channels]
        return tuple(indices + [len(self.channels) + i for i in indices])


class PhysicalResponseAffine:
    """Native physical design -> original mean/factor image in shared u."""

    def __init__(self, response, mean, factor):
        self.response = response
        self.mean, self.factor = frozen_array(mean), frozen_array(factor)
        if self.factor.ndim != 3 or self.factor.shape[:2] != self.mean.shape:
            raise ValueError('response source must retain its original rectangular factor chart')
        self.rank, self.channels = self.factor.shape[2], self.mean.shape[1]
        self.scope = response.scope
        self.physical_capability_identity = response.plan.chart.capability_identity_sha256
        self.physical_target_lock = response.plan.binding['target_lock_group']
        self.physical_candidate_variable = response.candidate_variable
        self.physical_clock_program = response.plan.clock_program
        self.physical_shape_block_id = response.block_id
        self.physical_shape_variable = response.shape_variable
        self.physical_shape_parameters = tuple(tuple(vars(s).items()) for s in response.shapes)
        self.physical_exposure_ids = tuple(e['event_id'] for e in response.plan.exposures if e['block_id'] == response.block_id)
        self.physical_unknown_identity_bindings = {}
        for event in response.plan.exposures:
            if event['block_id'] == response.block_id and event['membership'] is None:
                law = response.exposure_assignments
                self.physical_unknown_identity_bindings[event['event_id']] = {'variable': law.variable_id,
                    'state_to_candidate': {state: assignments[event['event_id']]
                        for state, assignments in law.assignments_by_state_id.items()}}
        self.source_chart = metadata_digest({'mean': array_digest(self.mean), 'factor': array_digest(self.factor)})
        self.identity = metadata_digest({'plan': response.plan.identity, 'block': response.block_id,
            'scope': self.scope, 'shape_parameters': [vars(s) for s in response.shapes],
            'source_chart': self.source_chart,
            'assignments': None if response.exposure_assignments is None else
                response.exposure_assignments.assignments_by_state_id,
            'compiler_sha256': file_sha(__file__)})

    def support(self, *, native_start, native_stop):
        return self.response.support(native_start=native_start, native_stop=native_stop)

    def affine(self, selected_states, *, native_start, native_stop):
        design, error, ledger = self.response.design(selected_states,
            native_start=native_start, native_stop=native_stop)
        return self._image(design, error, {'physical_design_identity': metadata_digest(ledger),
            'physical_plan': self.response.plan.identity, 'source_chart': self.source_chart,
            'original_input_error_role': ledger['numerical_error_role'],
            'actual_exposures_used': ledger['actual_exposures_used']})

    def affine_difference(self, selected_states, reference_states, *, native_start, native_stop):
        design, error, ledger = self.response.design(selected_states,
            native_start=native_start, native_stop=native_stop)
        reference, reference_error, reference_ledger = self.response.design(reference_states,
            native_start=native_start, native_stop=native_stop)
        difference = design - reference
        bound = error + reference_error + np.finfo(float).eps * (abs(design) + abs(reference))
        return self._image(difference, bound, {'state_design': metadata_digest(ledger),
            'reference_design': metadata_digest(reference_ledger), 'source_chart': self.source_chart,
            'difference_order': 'subtract_physical_design_before_original_source_affine_image'})

    def _image(self, design, error, binding):
        values = np.empty((len(design), self.channels, self.rank + 1))
        values[:, :, 0] = design @ self.mean
        values[:, :, 1:] = np.einsum('nk,kcr->ncr', design, self.factor)
        errors = np.empty_like(values)
        gamma = len(self.mean) * np.finfo(float).eps / (1 - len(self.mean) * np.finfo(float).eps)
        errors[:, :, 0] = error @ abs(self.mean) + gamma * (abs(design) @ abs(self.mean))
        errors[:, :, 1:] = (np.einsum('nk,kcr->ncr', error, abs(self.factor))
            + gamma * np.einsum('nk,kcr->ncr', abs(design), abs(self.factor)))
        return values, errors, {**binding, 'affine_error_role': 'binary64_dot_and_original_design_error_estimate'}


class PhysicalAffineSegment:
    """One original noise segment with callback means and physical nuisance."""

    def __init__(self, observed, carrier, clock, responses, rosters, drift_group, *,
                 record_id, origin_sample, clock_program, current_variable, q_values,
                 maximum_intermediate_entries, original_input_identity, noise_segment_declaration):
        self.y = frozen_array(observed)
        self.carrier, self.clock, self.responses = carrier, clock, tuple(responses)
        self.group = drift_group
        self.record_id, self.origin, self.clock_program = record_id, origin_sample, clock_program
        if clock_program != drift_group.clock_program:
            raise ValueError('a shared physical drift group must use one original clock coordinate programme')
        self.rosters = tuple(rosters)
        self.variable_ids = tuple(r.selection_id for r in self.rosters)
        self.current, self.q_values = self.variable_ids.index(current_variable), tuple(q_values)
        self.budget = maximum_intermediate_entries
        self.channels = len(carrier.noise.geometry.channels)
        self.channel_names = tuple(carrier.noise.geometry.channels)
        self.group_columns = drift_group.local_columns(self.channel_names)
        self.original_input_identity = original_input_identity
        self.noise_segment_declaration = noise_segment_declaration
        if (noise_segment_declaration['law'] != 'independent_noise_segments_original_stationary_startups/1'
                or not noise_segment_declaration['noise_segment_id']):
            raise ValueError('this carrier requires an explicit original independent-noise-segment law')
        if (self.y.shape != (len(self.y), self.channels) or not len(self.y)
                or not np.isfinite(self.y).all() or not self.responses):
            raise ValueError('physical segment requires its original measured rows/channels and response callbacks')
        self.rank, self.source_chart = self.responses[0].rank, self.responses[0].source_chart
        self.physical_scope_bindings = tuple({'target_lock': response.physical_target_lock,
            'shape_block': response.physical_shape_block_id, 'shape_variable': response.physical_shape_variable,
            'shape_parameters': response.physical_shape_parameters,
            'unknown_identity_bindings': response.physical_unknown_identity_bindings}
            for response in self.responses if hasattr(response, 'physical_shape_variable'))
        for response in self.responses:
            if (response.rank != self.rank or response.source_chart != self.source_chart
                    or response.channels != self.channels or len(set(response.scope)) != len(response.scope)
                    or not set(response.scope) <= set(self.variable_ids)):
                raise ValueError('all physical affine callbacks must share the actual source and declared complete scopes')
        current_states = self.rosters[self.current].states
        if (len(set(self.q_values)) != len(self.q_values) or len(current_states) != len(self.q_values)
                or len({s.q for s in current_states}) != len(current_states)
                or set(s.q for s in current_states) != set(self.q_values)):
            raise ValueError('physical current roster must retain every requested candidate')
        self.identity = metadata_digest({'schema': 'physical_affine_segment/1',
            'record': record_id, 'origin': origin_sample, 'observed': array_digest(self.y),
            'noise': carrier.identity, 'clock': clock.capability_identity_sha256,
            'clock_program': clock_program, 'drift_group': drift_group.identity,
            'group_columns': self.group_columns, 'source_chart': self.source_chart,
            'responses': [response.identity for response in self.responses],
            'rosters': [{'id': r.selection_id, 'states': [vars(s) for s in r.states],
                'mass': array_digest(r.log_mass)} for r in self.rosters],
            'current': self.current, 'q_values': self.q_values,
            'original_input_identity': original_input_identity, 'compiler_sha256': file_sha(__file__)})
        self.identity = metadata_digest({'numerical_segment': self.identity,
            'explicit_noise_segment_declaration': noise_segment_declaration,
            'physical_scope_bindings': self.physical_scope_bindings})

    def nuisance(self, start, stop):
        times = self.physical_times(start, stop)
        seconds = np.array([float(t) for t in times])
        error = np.array([float(abs(Fraction.from_float(value) - exact))
            for value, exact in zip(seconds, times, strict=True)])
        values = np.zeros((stop - start, self.channels, 2 * self.channels))
        errors = np.zeros_like(values)
        for channel in range(self.channels):
            values[:, channel, channel] = 1.
            values[:, channel, self.channels + channel] = seconds
            errors[:, channel, self.channels + channel] = error
        return values, errors

    def physical_times(self, start, stop):
        return self.clock.sample_time_seconds(tuple(range(self.origin + start, self.origin + stop)),
            clock_program=self.clock_program, reference_origin=self.group.reference_origin_seconds)


def physical_segment_from_raw_capability(cap, carrier, responses, rosters, drift_group, *,
                                        noise_segment_declaration, clock_program, current_variable,
                                        q_values, maximum_intermediate_entries):
    """Trusted parent seam: one canonical raw registry and complete history."""
    from data.brainsync_raw_authority import compile_physical_clock, require_raw_capability

    cap = require_raw_capability(cap)
    binding, geometry = cap.public_binding, carrier.noise.geometry
    if drift_group.clock_domain != binding['record_id']:
        raise ValueError('raw-bound drift requires its original recording clock domain or an explicit separate coordinate transform')
    if (geometry.channels != tuple(binding['eeg_channels']) or geometry.unit != binding['signal_unit']
            or geometry.reference != binding['reference'] or geometry.sfreq != binding['sample_rate_hz']
            or geometry.operator_identity != binding['operator']):
        raise ValueError('physical noise/observation geometry differs from original canonical raw capability')
    responses = tuple(responses)
    exposures, blocks = set(), set()
    for response in responses:
        if (response.physical_capability_identity != cap.identity_sha256
                or response.physical_target_lock != binding['target_lock_group']
                or response.physical_candidate_variable != binding['target_lock_group']
                or response.physical_clock_program != clock_program
                or response.physical_target_lock not in response.scope
                or response.physical_shape_block_id in blocks):
            raise ValueError('physical callback changed the actual cap/clock/block/target response scope')
        blocks.add(response.physical_shape_block_id)
        exposures.update(response.physical_exposure_ids)
        shape_roster = next(roster for roster in rosters if roster.selection_id == response.physical_shape_variable)
        if (len(shape_roster.states) != len(response.physical_shape_parameters)
                or {state.shape for state in shape_roster.states} != set(range(len(response.physical_shape_parameters)))):
            raise ValueError('physical shape variable must retain its complete original response-state axis')
    if exposures != {event['event_id'] for event in binding['public_physical_history']['exposures']}:
        raise ValueError('physical affine model must retain all actual completed/interrupted exposures')
    if noise_segment_declaration['acquisition_segment_id'] != binding['segment_id']:
        raise ValueError('explicit noise boundary declaration belongs to another original acquisition segment')
    segment = PhysicalAffineSegment(cap.eeg, carrier, compile_physical_clock(cap), responses, rosters, drift_group,
        record_id=binding['record_segment_id'], origin_sample=binding['sample_start'], clock_program=clock_program,
        current_variable=current_variable, q_values=q_values, maximum_intermediate_entries=maximum_intermediate_entries,
        original_input_identity=cap.identity_sha256, noise_segment_declaration=noise_segment_declaration)
    _RAW_SEGMENT_ADMISSIONS[segment] = (segment.identity, cap.identity_sha256)
    return segment


def require_raw_physical_segment(segment, cap):
    if _RAW_SEGMENT_ADMISSIONS.get(segment) != (segment.identity, cap.identity_sha256):
        raise ValueError('physical publication requires the original canonical-cap callback compilation admission')
    return segment


def _white(segment, raw, error, raw_start, begin, end):
    owned = compile_sparse_response_image(segment.carrier.noise, raw,
        start_native=raw_start, total_native=end, source_error=error)
    c = segment.channels
    first = begin * c - owned.image.start
    value = owned.image.values[first:first + (end - begin) * c] / segment.carrier.scale
    return value, owned.column_error / segment.carrier.scale, metadata_digest(owned.binding)


class PreparedPhysicalSegment:
    """Generic full-scope quadratics; physical flat admission belongs to group."""

    def __init__(self, base, stops, *, block_rows, maximum_cache_bytes):
        self.base_identity, self.record_id, self.origin = base.identity, base.record_id, base.origin
        self.rank, self.channels = base.rank, base.channels
        self.dimension, self.source_chart = base.rank + 2 * base.channels, base.source_chart
        self.rosters, self.variable_ids = base.rosters, base.variable_ids
        self.current, self.q_values, self.budget = base.current, base.q_values, base.budget
        self.group, self.group_columns = base.group, base.group_columns
        self.channel_names, self.clock_program = base.channel_names, base.clock_program
        self.block_rows, self.maximum_cache_bytes = block_rows, maximum_cache_bytes
        self.original_input_identity = base.original_input_identity
        self.noise_segment_declaration = base.noise_segment_declaration
        self.physical_scope_bindings = base.physical_scope_bindings
        self.noise_identity = base.carrier.identity
        self.endpoints, self.flat_roots, self.compilation_ledgers, self.flat_time_witness = {}, {}, {}, {}
        self.estimated_cache_bytes, self.maximum_named_workspace_bytes = 0, 0
        plans = {}
        for stop in sorted(set(stops)):
            if type(stop) is not int or not 0 <= stop <= len(base.y):
                raise ValueError('endpoint includes only actually available original segment rows')
            selections = []
            for response in base.responses:
                scope = tuple(sorted(base.variable_ids.index(name) for name in response.scope))
                widths = tuple(len(base.rosters[i].states) for i in scope)
                if prod(widths) > self.budget:
                    raise ValueError('complete physical response scope exceeds declared table budget')
                states = tuple({base.variable_ids[i]: base.rosters[i].states[j]
                    for i, j in zip(scope, coordinates, strict=True)} for coordinates in np.ndindex(widths))
                if stop:
                    proof = response.support(native_start=base.origin, native_stop=base.origin + stop)
                    left, right = (v - base.origin for v in proof['raw_native_hull'])
                else:
                    left = right = 0
                prefix, order = base.carrier.noise.conditioning_samples, base.carrier.noise.order
                if left < right:
                    left = 0 if left < prefix else left
                    right = min(stop, max(right + order, prefix if left == 0 else 0))
                selections.append((response, scope, widths, states, left, right))
                self.estimated_cache_bytes += prod(widths) * (self.dimension + 1) ** 2 * 8
            pairs = []
            for right, row in enumerate(selections):
                for left, previous in enumerate(selections[:right]):
                    if max(previous[4], row[4]) < min(previous[5], row[5]):
                        scope = tuple(sorted(set(previous[1]) | set(row[1])))
                        widths = tuple(len(self.rosters[i].states) for i in scope)
                        if prod(widths) > self.budget:
                            raise ValueError('complete physical overlap scope exceeds declared table budget')
                        pairs.append((left, right, scope, widths))
                        self.estimated_cache_bytes += prod(widths) * (self.rank + 1) ** 2 * 8
            self.estimated_cache_bytes += (self.dimension + 1) ** 2 * 8
            plans[stop] = (selections, pairs)
        if self.estimated_cache_bytes > maximum_cache_bytes:
            raise ValueError(f'complete physical cache needs {self.estimated_cache_bytes} bytes; no state/column pruning')
        for stop, (selections, pairs) in plans.items():
            times = base.physical_times(0, stop)
            extrema = (min(times), max(times)) if times else ()
            self.flat_time_witness[stop] = {channel: [[t.numerator, t.denominator] for t in extrema]
                for channel in self.channel_names}
            self.endpoints[stop] = self._compile(base, stop, selections, pairs)
        self.identity = self._identity()

    def _compile(self, base, stop, selections, pairs):
        d, m, c = self.dimension + 1, self.rank + 1, self.channels
        common = np.zeros((d, d))
        unary = [np.zeros((*row[2], d, d)) for row in selections]
        pair = [np.zeros((*row[3], m, m)) for row in pairs]
        flat_root, ledger, maximum_error = np.empty((0, 2 * c)), [], np.zeros(d)
        prefix, order = base.carrier.noise.conditioning_samples, base.carrier.noise.order
        for begin in range(0, stop, self.block_rows):
            end = min(stop, begin + self.block_rows)
            start = 0 if begin < prefix + order else begin - order
            raw, raw_error = np.zeros((end - start, c, d)), np.zeros((end - start, c, d))
            raw[:, :, 0] = base.y[start:end]
            nuisance, nuisance_error = base.nuisance(start, end)
            raw[:, :, m:], raw_error[:, :, m:] = -nuisance, nuisance_error
            references, reference_errors, active = {}, {}, []
            query_bindings = []
            for index, row in enumerate(selections):
                if row[4] >= end or row[5] <= begin:
                    continue
                reference, error, binding = row[0].affine(row[3][0],
                    native_start=base.origin + start, native_stop=base.origin + end)
                raw[:, :, :m] -= reference
                raw_error[:, :, :m] += error
                references[index], reference_errors[index] = reference, error
                active.append(index)
                query_bindings.append(metadata_digest(binding))
            b, error, owned = _white(base, raw, raw_error, start, begin, end)
            maximum_error = np.maximum(maximum_error, error)
            common -= .5 * (b.T @ b)
            flat_root = np.linalg.qr(np.vstack((flat_root, b[:, m:])), mode='reduced')[1]
            images, image_errors = {}, {}
            for index in active:
                row, values, errors = selections[index], [], []
                for state in row[3]:
                    if hasattr(row[0], 'affine_difference'):
                        difference, err, binding = row[0].affine_difference(state, row[3][0],
                            native_start=base.origin + start, native_stop=base.origin + end)
                    else:
                        response, response_error, binding = row[0].affine(state,
                            native_start=base.origin + start, native_stop=base.origin + end)
                        difference = response - references[index]
                        err = response_error + reference_errors[index] + np.finfo(float).eps * (abs(response) + abs(references[index]))
                    value, white_error, response_owned = _white(base, difference, err, start, begin, end)
                    values.append(value)
                    errors.append(white_error)
                    query_bindings.append(metadata_digest({'affine': binding, 'white': response_owned}))
                image = np.stack(values)
                images[index], image_errors[index] = image, np.max(errors, axis=0)
                table = unary[index].reshape(-1, d, d)
                for state_index, value in enumerate(image):
                    cross = b.T @ value
                    table[state_index, :, :m] += .5 * cross
                    table[state_index, :m, :] += .5 * cross.T
                    table[state_index, :m, :m] -= .5 * (value.T @ value)
            for pair_index, (left, right, scope, widths) in enumerate(pairs):
                if left not in images or right not in images:
                    continue
                lrow, rrow = selections[left], selections[right]
                for coordinates in np.ndindex(widths):
                    chosen = dict(zip(scope, coordinates, strict=True))
                    li = np.ravel_multi_index(tuple(chosen[i] for i in lrow[1]), lrow[2])
                    ri = np.ravel_multi_index(tuple(chosen[i] for i in rrow[1]), rrow[2])
                    cross = images[left][li].T @ images[right][ri]
                    pair[pair_index][coordinates] -= .5 * (cross + cross.T)
            ledger.append({'native_block': [base.origin + begin, base.origin + end],
                'raw_halo_start': base.origin + start, 'original_white_map': owned,
                'affine_query_bindings': metadata_digest(query_bindings),
                'maximum_white_affine_column_error_by_scope': {str(i): e.tolist() for i, e in image_errors.items()}})
            self.maximum_named_workspace_bytes = max(self.maximum_named_workspace_bytes,
                raw.nbytes + raw_error.nbytes + b.nbytes + sum(v.nbytes for v in images.values())
                + sum(v.nbytes for v in references.values()) + sum(v.nbytes for v in reference_errors.values()))
        self.flat_roots[stop] = flat_root
        self.compilation_ledgers[stop] = {'blocks': ledger, 'maximum_white_common_column_error': maximum_error.tolist(),
            'error_role': 'original_affine_and_noise_errors_carried; binary64_Q_dot_and_integral_reference',
            'extra_observation_rows': 0, 'noise_startups': 1 if stop else 0}
        factors = [QuadraticFactor(row[1], value, False) for row, value in zip(selections, unary, strict=True)]
        factors += [QuadraticFactor(row[2], value, True) for row, value in zip(pairs, pair, strict=True)]
        normalizer = base.carrier.leading_log_jacobian(stop) - .5 * stop * c * np.log(2 * np.pi) if stop else 0.
        return PreparedEndpoint(stop, common, tuple(factors), float(normalizer))

    def _identity(self):
        return metadata_digest({'schema': 'prepared_physical_affine_segment/1', 'base': self.base_identity,
            'producer': file_sha(__file__), 'source_chart': self.source_chart, 'group': self.group.identity,
            'group_columns': self.group_columns, 'block_rows': self.block_rows,
            'record': [self.record_id, self.origin, self.rank, self.channels, self.dimension],
            'channels': self.channel_names, 'clock_program': self.clock_program,
            'original_input_identity': self.original_input_identity, 'noise_identity': self.noise_identity,
            'noise_segment_declaration': self.noise_segment_declaration,
            'physical_scope_bindings': self.physical_scope_bindings,
            'current': self.current, 'q_values': self.q_values, 'budget': self.budget,
            'rosters': [{'id': r.selection_id, 'states': [vars(s) for s in r.states], 'mass': array_digest(r.log_mass)}
                for r in self.rosters],
            'endpoints': {str(stop): {'common': array_digest(e.common), 'normalizer': e.log_normalizer,
                'flat_root': array_digest(self.flat_roots[stop]),
                'flat_time_witness': self.flat_time_witness[stop],
                'factors': [{'scope': f.scope, 'proper_only': f.proper_only, 'q': array_digest(f.coefficients)}
                    for f in e.factors], 'compilation_ledger': metadata_digest(self.compilation_ledgers[stop])}
                for stop, e in self.endpoints.items()}})

    def factors(self, u, drift, stop):
        vector = np.r_[1., u, drift]
        endpoint = self.endpoints[stop]
        factors = tuple(LogFactor(f.scope, f.values(vector)) for f in endpoint.factors)
        return PointFactors(self.rosters, factors, self.current, self.q_values,
            float(endpoint.log_normalizer + vector @ endpoint.common @ vector),
            _elimination_order(self.rosters, factors, self.current),
            {'physical_record_segment_id': self.record_id,
             'source_coefficient_chart_sha256': self.source_chart,
             'fixed_source_point_sha256': array_digest(u), 'physical_drift_sha256': array_digest(drift),
             'physical_drift_group': self.group.identity, 'prepared_model_identity': self.identity})

    def save(self, path):
        path, arrays, endpoints = Path(path), {}, {}
        for stop, endpoint in self.endpoints.items():
            stem = f'e{stop}'
            arrays[stem + '_common'], arrays[stem + '_flat_root'] = endpoint.common, self.flat_roots[stop]
            rows = []
            for index, factor in enumerate(endpoint.factors):
                key = f'{stem}_factor_{index}'
                arrays[key] = factor.coefficients
                rows.append({'key': key, 'scope': factor.scope, 'proper_only': factor.proper_only})
            endpoints[str(stop)] = {'stem': stem, 'normalizer': endpoint.log_normalizer, 'factors': rows}
        np.savez_compressed(path, **arrays)
        reference = self.group.reference_origin_seconds
        metadata = {'schema': 'prepared_physical_affine_segment/1', 'identity': self.identity,
            'producer_sha256': file_sha(__file__), 'base_identity': self.base_identity,
            'source_chart': self.source_chart, 'rank': self.rank, 'channels': self.channels,
            'dimension': self.dimension, 'record_id': self.record_id, 'origin': self.origin,
            'channel_names': self.channel_names, 'clock_program': self.clock_program,
            'original_input_identity': self.original_input_identity, 'noise_identity': self.noise_identity,
            'noise_segment_declaration': self.noise_segment_declaration,
            'physical_scope_bindings': self.physical_scope_bindings,
            'group': {'id': self.group.group_id, 'channels': self.group.channels,
                'reference_origin_seconds': [reference.numerator, reference.denominator],
                'clock_program': self.group.clock_program, 'clock_domain': self.group.clock_domain},
            'group_columns': self.group_columns, 'current': self.current, 'q_values': self.q_values,
            'budget': self.budget, 'block_rows': self.block_rows, 'maximum_cache_bytes': self.maximum_cache_bytes,
            'estimated_cache_bytes': self.estimated_cache_bytes,
            'maximum_named_workspace_bytes': self.maximum_named_workspace_bytes,
            'rosters': [{'id': r.selection_id, 'states': [vars(s) for s in r.states], 'mass': r.log_mass.tolist()}
                for r in self.rosters], 'endpoints': endpoints, 'compilation_ledgers': self.compilation_ledgers,
            'flat_time_witness': self.flat_time_witness,
            'payload': {'path': str(path), 'sha256': file_sha(path)}}
        path.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        return metadata

    @classmethod
    def load(cls, metadata_path, *, expected_metadata_sha256):
        if file_sha(metadata_path) != expected_metadata_sha256:
            raise ValueError('physical affine metadata differs from external admission')
        metadata = json.loads(Path(metadata_path).read_bytes())
        if metadata['producer_sha256'] != file_sha(__file__) or file_sha(metadata['payload']['path']) != metadata['payload']['sha256']:
            raise ValueError('physical affine numeric payload/compiler differs from original')
        obj = object.__new__(cls)
        for key in ('identity', 'base_identity', 'source_chart', 'rank', 'channels', 'dimension', 'record_id',
                    'origin', 'clock_program', 'original_input_identity', 'noise_identity', 'current', 'budget',
                    'noise_segment_declaration',
                    'physical_scope_bindings',
                    'block_rows', 'maximum_cache_bytes', 'estimated_cache_bytes', 'maximum_named_workspace_bytes'):
            setattr(obj, key, metadata[key])
        group = metadata['group']
        obj.group = PhysicalDriftGroup(group['id'], tuple(group['channels']), Fraction(*group['reference_origin_seconds']),
            group['clock_program'], group['clock_domain'])
        obj.group_columns, obj.channel_names, obj.q_values = map(tuple,
            (metadata['group_columns'], metadata['channel_names'], metadata['q_values']))
        if obj.group_columns != obj.group.local_columns(obj.channel_names):
            raise ValueError('physical group channel mapping changed original coordinates')
        obj.rosters = tuple(Roster(row['id'], tuple(StateKey(**s) for s in row['states']), np.array(row['mass']))
            for row in metadata['rosters'])
        obj.variable_ids = tuple(r.selection_id for r in obj.rosters)
        obj.compilation_ledgers = {int(k): v for k, v in metadata['compilation_ledgers'].items()}
        obj.flat_time_witness = {int(k): v for k, v in metadata['flat_time_witness'].items()}
        obj.endpoints, obj.flat_roots = {}, {}
        with np.load(metadata['payload']['path'], allow_pickle=False) as data:
            for key, row in metadata['endpoints'].items():
                stop = int(key)
                obj.flat_roots[stop] = data[row['stem'] + '_flat_root'].copy()
                obj.endpoints[stop] = PreparedEndpoint(stop, data[row['stem'] + '_common'].copy(),
                    tuple(QuadraticFactor(tuple(f['scope']), data[f['key']].copy(), f['proper_only'])
                        for f in row['factors']), row['normalizer'])
        if obj._identity() != obj.identity:
            raise ValueError('physical affine cold cache changed original operands or group map')
        return obj


class GroupedPhysicalTarget:
    """One source u and one physical da per explicit drift group."""

    def __init__(self, records):
        self.records = tuple(records)
        first = self.records[0][0]
        self.rank, self.q_values, self.current = first.rank, first.q_values, first.current
        self.rosters, self.source_chart = first.rosters, first.source_chart
        if len({m.record_id for m, _, _ in self.records}) != len(self.records):
            raise ValueError('the same original noise segment cannot contribute twice')
        noise_ids = [m.noise_segment_declaration['noise_segment_id'] for m, _, _ in self.records]
        if len(set(noise_ids)) != len(noise_ids):
            raise ValueError('one declared noise segment cannot restart independently in multiple record factors')
        groups, physical_shapes, physical_unknowns = {}, {}, {}
        for model, p, f in self.records:
            if (model.source_chart != self.source_chart or model.rank != self.rank
                    or model.q_values != self.q_values or model.current != self.current
                    or model.variable_ids != first.variable_ids or not 0 <= p <= f
                    or p not in model.endpoints or f not in model.endpoints
                    or any(a.states != b.states or not np.array_equal(a.log_mass, b.log_mass)
                        for a, b in zip(model.rosters, self.rosters, strict=True))):
                raise ValueError('physical segments must share the original source and full variable priors')
            if model.group.group_id in groups and model.group.identity != groups[model.group.group_id].identity:
                raise ValueError('one physical drift group cannot use inconsistent physical charts')
            groups[model.group.group_id] = model.group
            for scope in model.physical_scope_bindings:
                key = scope['target_lock'], scope['shape_block']
                descriptor = metadata_digest({'variable': scope['shape_variable'], 'parameters': scope['shape_parameters']})
                if key in physical_shapes and physical_shapes[key] != descriptor:
                    raise ValueError('one physical block must retain the same response shape variable and parameters across segments')
                physical_shapes[key] = descriptor
                for event, law in scope['unknown_identity_bindings'].items():
                    descriptor = metadata_digest(law)
                    if event in physical_unknowns and physical_unknowns[event] != descriptor:
                        raise ValueError('one actual unknown exposure must retain its shared assignment variable and state-to-digit law across segments')
                    physical_unknowns[event] = descriptor
        self.drift_groups = tuple(groups.values())
        self.group_offsets, offset = {}, self.rank
        for group in self.drift_groups:
            self.group_offsets[group.group_id] = offset
            offset += 2 * len(group.channels)
        self.dimension, self.budget = offset, min(m.budget for m, _, _ in self.records)
        self.log_prior = np.array([self.rosters[self.current].log_mass[
            next(i for i, s in enumerate(self.rosters[self.current].states) if s.q == q)] for q in self.q_values])
        self.flat_group_admission, self.refusals = {}, {}
        for endpoint in ('P', 'F'):
            admitted = {}
            for group in self.drift_groups:
                width = 2 * len(group.channels)
                blocks, exact_times = [], {channel: set() for channel in group.channels}
                for model, p, f in self.records:
                    if model.group.group_id == group.group_id:
                        stop = p if endpoint == 'P' else f
                        root = model.flat_roots[stop]
                        embedded = np.zeros((len(root), width))
                        embedded[:, model.group_columns] = root
                        blocks.append(embedded)
                        for channel, times in model.flat_time_witness[stop].items():
                            exact_times[channel].update(Fraction(*t) for t in times)
                if any(len(times) < 2 for times in exact_times.values()):
                    self.refusals.setdefault(endpoint, []).append({'group': group.group_id,
                        'reason': 'physical_flat_group_exact_time_rank_unidentified'})
                    continue
                try:
                    root, volume = physical_flat_geometry_from_white_blocks(blocks, width)
                    admitted[group.group_id] = {'physical_root': root.tolist(), 'log_chart_volume': volume,
                        'group_identity': group.identity, 'measure_count': 1,
                        'exact_rank_witness': {channel: [[t.numerator, t.denominator] for t in sorted(times)]
                            for channel, times in exact_times.items()}}
                except FlatEndpointUnidentifiedError as error:
                    self.refusals.setdefault(endpoint, []).append({'group': group.group_id, 'reason': str(error)})
            self.flat_group_admission[endpoint] = admitted
        self.endpoints = tuple(e for e in ('P', 'F') if e not in self.refusals)
        if 'F' not in self.endpoints:
            raise FlatEndpointUnidentifiedError(str(self.refusals['F']))
        self.identity = metadata_digest({'schema': 'grouped_physical_continuous_target/1',
            'models': [m.identity for m, _, _ in self.records], 'cuts': [[p, f] for _, p, f in self.records],
            'groups': [g.identity for g in self.drift_groups], 'source_prior': 'one_standard_normal_u',
            'flat_group_admission': self.flat_group_admission, 'refusals': self.refusals})

    def local_drift(self, point, model):
        offset = self.group_offsets[model.group.group_id]
        return np.asarray(point)[offset + np.array(model.group_columns)]

    def log_joint(self, z):
        z = np.asarray(z)
        if z.shape != (self.dimension,) or not np.isfinite(z).all():
            raise ValueError('physical point must retain original source and explicit group coordinates')
        source = -.5 * (self.rank * np.log(2 * np.pi) + z[:self.rank] @ z[:self.rank])
        values = []
        for endpoint in self.endpoints:
            factors = [m.factors(z[:self.rank], self.local_drift(z, m), p if endpoint == 'P' else f)
                for m, p, f in self.records]
            point = factors[0] if len(factors) == 1 else combine_record_factors(factors)
            values.append(sum_product(point, maximum_intermediate_entries=self.budget).log_joint_mass_by_q + source)
        return np.array(values)

    def evaluate_many(self, points):
        return np.array([self.log_joint(point) for point in points])


def replay_physical_parent(original_segments, saved_target, result):
    """Rebuild actual callback/noise/time Q, then replay the same finite result."""
    from research.finite_quadratic_integration import replay_finite_model

    if len(original_segments) != len(saved_target.records):
        raise ValueError('physical parent changed the full noise-segment roster')
    records = []
    for original, (saved, p, f) in zip(original_segments, saved_target.records, strict=True):
        if original.identity != saved.base_identity:
            raise ValueError('physical parent differs from original source/raw/clock/group operands')
        rebuilt = PreparedPhysicalSegment(original, tuple(saved.endpoints), block_rows=saved.block_rows,
            maximum_cache_bytes=saved.maximum_cache_bytes)
        if rebuilt.identity != saved.identity:
            raise ValueError('physical parent recompile changed actual complete quadratics')
        records.append((rebuilt, p, f))
    replay_finite_model(GroupedPhysicalTarget(records), result)
    return {'layer': 'original_physical_parent', 'bitexact': True,
        'noise_segments_recompiled': len(records), 'source_fits': 0, 'target_model_fits': 0}
