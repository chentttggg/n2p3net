"""Complete history inference through exact discrete pushforward and local QR.

Source and auxiliary Gaussian measures occur once; original physical drift has
flat measure. Numeric construction creates no data, source or fitted authority.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp

from research.collapsed_continuous_inference import GaussianProposal
from research.frozen_metadata import freeze_metadata, plain_metadata
from research.gaussian_local_factor import (
    GaussianCandidateFactor,
    GaussianLocalPotential,
    SharedResponseChart,
    require_once_record_partition,
)
from research.joint_response_noise_calibration import joint_calibration_marginal
from research.joint_response_scale import NumericalResolutionError
from research.response_frontier import LogFactor, PointFactors, Roster, StateKey, sum_product
from research.structured_response_frontier import _elimination_order
from transfer.noise_model import array_digest, frozen_array, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_LOG2PI = float(np.log(2*np.pi))


def _normal_category(coordinate, log_mass):
    """Normal-CDF intervals; the reverse tail preserves tiny positive masses."""
    mass = np.asarray(log_mass)-logsumexp(log_mass)
    order = range(len(mass)) if coordinate <= 0 else range(len(mass)-1, -1, -1)
    log_u = float(log_ndtr(coordinate if coordinate <= 0 else -coordinate))
    cumulative = -np.inf
    for index in order:
        cumulative = np.logaddexp(cumulative, mass[index])
        if log_u < cumulative:
            return index
    raise ArithmeticError('normal-CDF categorical interval is numerically unresolved')


class LabelConditionedHistory:
    """Each current q has its own exact C-bit mass and history pushforward."""

    def __init__(self, rosters, current_index, q_values, label_factors):
        self.rosters, self.current = tuple(rosters), current_index
        self.q_values, self.label_factors = tuple(q_values), tuple(label_factors)
        if (not 0 <= self.current < len(self.rosters)
                or tuple(state.q for state in self.rosters[self.current].states) != self.q_values
                or len(set(self.q_values)) != len(self.q_values)):
            raise ValueError('the current roster retains every candidate once in original order')
        self.variable_ids = tuple(roster.selection_id for roster in self.rosters)
        if len(set(self.variable_ids)) != len(self.variable_ids):
            raise ValueError('each independent discrete variable has one original named prior')
        self.auxiliary_indices = tuple(i for i in range(len(self.rosters)) if i != self.current)
        self.constrained = set().union(*(factor.scope for factor in self.label_factors))
        # This bound is the mathematical full table size, not a state cutoff.
        # Only C-label graph scopes are contracted; EEG history is sampled.
        self.label_table_bound = prod(len(roster.states) for roster in self.rosters)
        self.index_rosters = tuple(Roster(roster.selection_id,
            tuple(StateKey(i, i) for i in range(len(roster.states))), roster.log_mass)
            for roster in self.rosters)
        partition = self._partition(self.current, ())
        self.log_C_mass = frozen_array(partition.log_joint_mass_by_q)
        self.log_prior = frozen_array(self.rosters[self.current].log_mass)
        self._conditionals = {}
        self.identity = metadata_digest(self.record())

    @property
    def dimension(self):
        return len(self.auxiliary_indices)

    def _partition(self, query, fixed):
        factors = list(self.label_factors)
        for index, chosen in fixed:
            values = np.full(len(self.rosters[index].states), -np.inf)
            values[chosen] = 0.
            factors.append(LogFactor((index,), values))
        point = PointFactors(self.index_rosters, tuple(factors), query,
            tuple(range(len(self.rosters[query].states))), 0.,
            _elimination_order(self.index_rosters, factors, query),
            {'role': 'exact_C_label_prior_relation_no_EEG', 'producer': IMPLEMENTATION_SHA256})
        return sum_product(point, maximum_intermediate_entries=self.label_table_bound)

    def decode(self, coordinates, q_index):
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != (self.dimension,) or not np.isfinite(coordinates).all():
            raise ValueError('one finite standard-Gaussian coordinate per other variable is required')
        if not np.isfinite(self.log_C_mass[q_index]):
            return None
        chosen = {self.current: q_index}
        conditioned = [(self.current, q_index)] if self.current in self.constrained else []
        for coordinate, index in zip(coordinates, self.auxiliary_indices, strict=True):
            if index in self.constrained:
                key = index, tuple(conditioned)
                mass = self._conditionals.get(key)
                if mass is None:
                    mass = self._partition(index, tuple(conditioned)).log_joint_mass_by_q
                    self._conditionals[key] = frozen_array(mass-logsumexp(mass))
                selected = _normal_category(coordinate, mass)
                conditioned.append((index, selected))
            else:
                selected = _normal_category(coordinate, self.rosters[index].log_mass)
            chosen[index] = selected
        return {self.variable_ids[i]: roster.states[chosen[i]] for i, roster in enumerate(self.rosters)}

    def record(self):
        return {'schema': 'per_current_candidate_C_label_conditioned_history_pushforward/1',
            'rosters': [{'id': roster.selection_id, 'states': [vars(state) for state in roster.states],
                'log_mass': roster.log_mass.tolist()} for roster in self.rosters],
            'current_index': self.current, 'q_values': self.q_values,
            'C_label_factors': [{'scope': factor.scope, 'shape': factor.values.shape,
                'values_sha256': array_digest(factor.values)} for factor in self.label_factors],
            'log_Z_C_by_q_including_current_prior': [None if np.isneginf(v) else float(v) for v in self.log_C_mass],
            'exact_zero_C_mass_mask': np.isneginf(self.log_C_mass).tolist(),
            'auxiliary_coordinates': [self.variable_ids[i] for i in self.auxiliary_indices],
            'pushforward': 'standard_normal_log_CDF_and_reverse_log_survival_exact_category_intervals',
            'interval_boundary_measure': 0, 'EEG_likelihood_in_constraint_transform': False}


@dataclass(frozen=True)
class EndpointComponents:
    potentials: tuple
    binding: dict


class JointHistoryTarget:
    """One shared u, one physical da, all current q; history jointly sampled."""

    def __init__(self, source, rosters, current_index, q_values, C_label_factors,
            component_factory, endpoint_records, drift_ids_by_record, *, factory_identity, input_binding):
        if type(source) is not SharedResponseChart:
            raise TypeError('history target consumes the complete explicit frozen shared chart')
        self.source = source
        self.discrete = LabelConditionedHistory(rosters, current_index, q_values, C_label_factors)
        self.rosters, self.current, self.q_values = self.discrete.rosters, current_index, tuple(q_values)
        self.log_prior = self.discrete.log_prior
        self.rank = source.rank
        self.endpoint_records = {name: dict(records) for name, records in endpoint_records.items()}
        if 'F' not in self.endpoint_records or set(self.endpoint_records)-{'P', 'F'}:
            raise ValueError('the complete full endpoint is required; P is an explicit comparable endpoint')
        self.endpoints = tuple(name for name in ('P', 'F') if name in self.endpoint_records)
        records = tuple(self.endpoint_records['F'])
        self.drift_ids_by_record = {record: tuple(drift_ids_by_record[record]) for record in records}
        self.drift_ids = tuple(name for record in records for name in self.drift_ids_by_record[record])
        if len(set(self.drift_ids)) != len(self.drift_ids):
            raise ValueError('different original recording groups retain distinct physical drift coordinates')
        if any(set(self.endpoint_records[name]) != set(records) or any(
                not 0 < self.endpoint_records[name][record] <= self.endpoint_records['F'][record]
                for record in records) for name in self.endpoints):
            raise ValueError('P/F retain the same physical record measures and nested original row unions')
        self.base_dimension = self.rank+len(self.drift_ids)
        self.dimension = self.base_dimension+self.discrete.dimension
        self.component_factory, self.factory_identity = component_factory, factory_identity
        self.input_binding = freeze_metadata(input_binding)
        self.refusals = {}
        self.binding = freeze_metadata({'schema': 'complete_joint_history_auxiliary_target/1',
            'source': source.identity, 'discrete_law': self.discrete.identity,
            'endpoint_original_records': self.endpoint_records,
            'physical_flat_coordinate_ids': self.drift_ids, 'factory': factory_identity,
            'input': input_binding, 'source_prior_count': 1, 'auxiliary_Gaussian_base_count': 1,
            'C_observation_union_count': 1, 'C_posterior_multiplied_into_target': False,
            'discrete_history': 'joint_auxiliary_integration_with_shared_response_and_record_drifts',
            'producer': IMPLEMENTATION_SHA256})
        self.identity = metadata_digest(self.binding)

    def components(self, endpoint, selected_states):
        bundle = self.component_factory(endpoint, selected_states)
        values = tuple(bundle.potentials)
        for record, count in self.endpoint_records[endpoint].items():
            pieces = tuple(value for value in values if value.binding['record_id'] == record)
            require_once_record_partition(pieces, record_id=record, original_native_samples=count)
            if any(tuple(value.binding['drift_coordinate_ids']) != self.drift_ids_by_record[record]
                    or value.binding['shared_response_chart'] != self.source.identity for value in pieces):
                raise ValueError('components preserve the one shared source and original physical flat chart')
        if {value.binding['record_id'] for value in values} != set(self.endpoint_records[endpoint]):
            raise ValueError('the endpoint includes every original record and no extra observation factor')
        return bundle

    def log_joint(self, point):
        point = np.asarray(point, dtype=float)
        if point.shape != (self.dimension,) or not np.isfinite(point).all():
            raise ValueError('continuous points retain source, physical drift and auxiliary coordinates')
        u, auxiliary = point[:self.rank], point[self.base_dimension:]
        prior = -.5*((self.rank+len(auxiliary))*_LOG2PI+u@u+auxiliary@auxiliary)
        result = np.full((len(self.endpoints), len(self.q_values)), -np.inf)
        for iq in range(len(self.q_values)):
            selected = self.discrete.decode(auxiliary, iq)
            if selected is None:
                continue
            for ie, endpoint in enumerate(self.endpoints):
                bundle = self.components(endpoint, selected)
                total = prior+self.discrete.log_C_mass[iq]
                for potential in bundle.potentials:
                    indices = [self.rank+self.drift_ids.index(name)
                        for name in potential.binding['drift_coordinate_ids']]
                    total += potential.log_value(u, point[indices])
                result[ie, iq] = total
        return result

    def evaluate_many(self, points):
        return np.array([self.log_joint(point) for point in points])

    def discrete_record(self):
        return {'algorithm': 'named_auxiliary_CDF_history_strategy', **self.discrete.record()}

    def branch_posterior(self, endpoint, selected_states):
        """A truth-free Gaussian numerical proposal branch, never a new prior."""
        return joint_calibration_marginal(self.components(endpoint, selected_states).potentials,
            source_chart=self.source, record_samples=self.endpoint_records[endpoint],
            input_binding={'target': self.identity, 'role': 'proper_proposal_reference_branch'})


class ConstrainedGaussianTable:
    """Full local state table; only original C-bit logical zeros skip compilation."""

    def __init__(self, rosters, scope, entries, *, exact_C_zero_binding):
        representative = next((value for value in entries.values() if value is not None), None)
        if representative is None:
            raise ValueError('an original C constraint left no feasible state in a local component')
        self.factor = GaussianCandidateFactor(rosters, scope,
            {key: representative if value is None else value for key, value in entries.items()})
        mask = np.array([entries[key] is None for key in self.factor.keys], dtype=bool)
        self.zero_mask = np.frombuffer(mask.tobytes(), dtype=bool).reshape(self.factor.shape)
        self.scope, self.representative = self.factor.scope, representative
        self.exact_C_zero_binding = freeze_metadata(exact_C_zero_binding)
        self.identity = metadata_digest({'complete_local_table': self.factor.identity,
            'exact_C_zero_mask': self.zero_mask.tolist(), 'constraint_proof': self.exact_C_zero_binding,
            'zero_state_role': 'original_C_label_logical_zero; no_numeric_likelihood_claim'})

    def at(self, shared, drift):
        factor = self.factor.at(shared, drift)
        return LogFactor(factor.scope, np.where(self.zero_mask, -np.inf, factor.values))

    def branch(self, selected, variable_ids):
        key = tuple(selected[variable_ids[index]] for index in self.scope)
        position = self.factor.keys.index(key)
        if self.zero_mask.ravel()[position]:
            raise ValueError('the proposal branch contradicts an original C bit constraint')
        return self.factor.potentials[position]


class RaoBlackwellHistoryTarget(JointHistoryTarget):
    """Default: actual component scopes -> exact LogFactor contraction at u,a."""

    def __init__(self, source, rosters, current_index, q_values, C_label_factors,
            component_tables, endpoint_records, drift_ids_by_record, *, factory_identity, input_binding):
        super().__init__(source, rosters, current_index, q_values, C_label_factors, None,
            endpoint_records, drift_ids_by_record, factory_identity=factory_identity, input_binding=input_binding)
        self.tables = {endpoint: tuple(component_tables[endpoint]) for endpoint in self.endpoints}
        self.dimension = self.base_dimension
        self.orders, self.frontier_layout = {}, {}
        table_identities = {}
        for endpoint, tables in self.tables.items():
            representatives, scopes, identities = [], [], []
            for table in tables:
                if type(table) is ConstrainedGaussianTable:
                    representatives.append(table.representative)
                    scopes.append(table.scope)
                elif type(table) is GaussianCandidateFactor:
                    representatives.append(table.potentials[0])
                    scopes.append(table.scope)
                elif type(table) is GaussianLocalPotential:
                    representatives.append(table)
                else:
                    raise TypeError('history components use the existing Gaussian-local owner')
                identities.append(table.identity)
            self._admit_potentials(endpoint, representatives)
            factors = tuple(C_label_factors)+tuple(LogFactor(scope,
                np.zeros(tuple(len(rosters[i].states) for i in scope))) for scope in scopes)
            order = _elimination_order(self.rosters, factors, self.current)
            self.orders[endpoint] = order
            live_scopes = [set(factor.scope) for factor in factors]
            widths, trace = tuple(len(roster.states) for roster in self.rosters), []
            for variable in order:
                bucket = [scope for scope in live_scopes if variable in scope]
                combined = set().union(*bucket) if bucket else {variable}
                trace.append({'variable': self.rosters[variable].selection_id,
                    'scope': [self.rosters[i].selection_id for i in sorted(combined)],
                    'entries': prod(widths[i] for i in combined)})
                live_scopes = [scope for scope in live_scopes if variable not in scope]
                if combined-{variable}:
                    live_scopes.append(combined-{variable})
            self.frontier_layout[endpoint] = {'component_scopes': scopes, 'elimination': trace,
                'maximum_intermediate_entries': max([1, *[row['entries'] for row in trace],
                    *[prod(widths[i] for i in scope) for scope in scopes]]),
                'global_prior_cartesian_states': prod(widths),
                'separate_global_history_cartesian_enumeration': False,
                'component_scope_equals_full_roster': [len(scope) == len(self.rosters) for scope in scopes],
                'component_table_entries': [prod(widths[i] for i in scope) for scope in scopes]}
            table_identities[endpoint] = identities
        self.binding = freeze_metadata({**plain_metadata(self.binding),
            'schema': 'actual_component_Rao_Blackwell_joint_history_target/1',
            'auxiliary_Gaussian_base_count': 0, 'discrete_history': 'exact_actual_component_LogFactor_sum_product',
            'numeric_evaluation_program': 'bounded_32_point_float64_BLAS_then_original_LogFactor_sum_product',
            'component_table_identities': table_identities, 'actual_frontier_layout': self.frontier_layout})
        self.identity = metadata_digest(self.binding)

    def numeric_snapshot(self):
        """Ingress/publication identity over the actual scientific operands."""
        tables = {}
        for endpoint, values in self.tables.items():
            records = []
            for table in values:
                if type(table) is GaussianLocalPotential:
                    records.append({'constant_potential': table.identity})
                    continue
                factor = table.factor if type(table) is ConstrainedGaussianTable else table
                record = {'scope': factor.scope, 'shape': factor.shape,
                    'state_keys': [[vars(state) for state in key] for key in factor.keys],
                    'actual_potentials': [value.identity for value in factor.potentials]}
                if type(table) is ConstrainedGaussianTable:
                    record['actual_exact_C_zero_mask'] = table.zero_mask.tolist()
                records.append(record)
            tables[endpoint] = records
        return {'source': self.source.identity, 'rank': self.rank, 'dimension': self.dimension,
            'rosters': [{'id': roster.selection_id, 'states': [vars(state) for state in roster.states],
                'log_mass': roster.log_mass.tolist()} for roster in self.rosters],
            'current': self.current, 'q_values': self.q_values, 'log_prior': self.log_prior.tolist(),
            'C_label_factors': [{'scope': factor.scope, 'values': array_digest(factor.values)}
                for factor in self.discrete.label_factors],
            'exact_C_mass': self.discrete.record()['log_Z_C_by_q_including_current_prior'],
            'endpoints': self.endpoints, 'endpoint_records': self.endpoint_records,
            'drift_ids': self.drift_ids, 'drift_ids_by_record': self.drift_ids_by_record,
            'elimination_orders': self.orders, 'component_tables': tables}

    def _admit_potentials(self, endpoint, values):
        for record, count in self.endpoint_records[endpoint].items():
            pieces = tuple(value for value in values if value.binding['record_id'] == record)
            require_once_record_partition(pieces, record_id=record, original_native_samples=count)
            if any(value.binding['shared_response_chart'] != self.source.identity
                    or tuple(value.binding['drift_coordinate_ids']) != self.drift_ids_by_record[record] for value in pieces):
                raise ValueError('every component uses the one complete source and physical record chart')
        if {value.binding['record_id'] for value in values} != set(self.endpoint_records[endpoint]):
            raise ValueError('actual endpoint components cover exactly its declared original records')

    def components(self, endpoint, selected_states):
        values = []
        for table in self.tables[endpoint]:
            if type(table) is ConstrainedGaussianTable:
                value = table.branch(selected_states, self.discrete.variable_ids)
            elif type(table) is GaussianCandidateFactor:
                key = tuple(selected_states[self.discrete.variable_ids[index]] for index in table.scope)
                value = table.potentials[table.keys.index(key)]
            else:
                value = table
            values.append(value)
        return EndpointComponents(tuple(values), {'role': 'actual_local_component_reference_branch'})

    def log_joint(self, point):
        point = np.asarray(point, dtype=float)
        if point.shape != (self.dimension,) or not np.isfinite(point).all():
            raise ValueError('Rao-Blackwell points retain shared source and original physical da only')
        u = point[:self.rank]
        source_prior = -.5*(self.rank*_LOG2PI+u@u)
        result = []
        for endpoint in self.endpoints:
            factors, common = list(self.discrete.label_factors), source_prior
            for table in self.tables[endpoint]:
                representative = (table.representative if type(table) is ConstrainedGaussianTable
                    else table.potentials[0] if type(table) is GaussianCandidateFactor else table)
                indices = [self.rank+self.drift_ids.index(name) for name in representative.binding['drift_coordinate_ids']]
                if type(table) is GaussianLocalPotential:
                    common += table.log_value(u, point[indices])
                else:
                    factors.append(table.at(u, point[indices]))
            partition = sum_product(PointFactors(self.rosters, tuple(factors), self.current,
                self.q_values, common, self.orders[endpoint], {'target': self.identity}),
                maximum_intermediate_entries=self.discrete.label_table_bound)
            result.append(partition.log_joint_mass_by_q)
        return np.array(result)

    def evaluate_many(self, points):
        """Batch only Gaussian products; retain the original exact contraction.

        A tile holds at most 32 common continuous points. Every original local
        table remains complete; each point enters the same sum_product owner.
        The scalar log_joint method remains an independent arithmetic control.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.dimension or not np.isfinite(points).all():
            raise ValueError('batched history queries retain every original finite shared and physical coordinate')
        output = np.empty((len(points), len(self.endpoints), len(self.q_values)))
        for begin in range(0, len(points), 32):
            tile = points[begin:begin+32]
            u = tile[:, :self.rank]
            prior = np.array([-.5*(self.rank*_LOG2PI+value@value) for value in u])
            cache = {}

            def values(potential, cache=cache, u=u, tile=tile):
                key = id(potential)
                if key not in cache:
                    indices = [self.rank+self.drift_ids.index(name) for name in potential.binding['drift_coordinate_ids']]
                    augmented = np.column_stack((u, tile[:, indices], -np.ones(len(tile))))
                    try:
                        with np.errstate(over='raise', invalid='raise'):
                            residual = potential.residual_root@augmented.T
                            cache[key] = potential.log_normalizer-.5*np.einsum('ij,ij->j', residual, residual)
                    except FloatingPointError as error:
                        raise NumericalResolutionError('batched Gaussian history potential is outside binary64 range') from error
                return cache[key]

            for endpoint_index, endpoint in enumerate(self.endpoints):
                common, local_tables = prior.copy(), []
                for table in self.tables[endpoint]:
                    if type(table) is GaussianLocalPotential:
                        common += values(table)
                        continue
                    constrained = type(table) is ConstrainedGaussianTable
                    factor = table.factor if constrained else table
                    columns = [np.full(len(tile), -np.inf) if constrained and table.zero_mask.ravel()[i]
                        else values(potential) for i, potential in enumerate(factor.potentials)]
                    local_tables.append((factor.scope, np.column_stack(columns).reshape((len(tile), *factor.shape))))
                for row in range(len(tile)):
                    factors = tuple(self.discrete.label_factors)+tuple(LogFactor(scope, table[row]) for scope, table in local_tables)
                    partition = sum_product(PointFactors(self.rosters, factors, self.current, self.q_values,
                        float(common[row]), self.orders[endpoint], {'target': self.identity}),
                        maximum_intermediate_entries=self.discrete.label_table_bound)
                    output[begin+row, endpoint_index] = partition.log_joint_mass_by_q
        return output

    def discrete_record(self):
        return {'algorithm': 'exact_actual_component_Rao_Blackwell_sum_product',
            'law': self.discrete.record(), 'auxiliary_Gaussian_coordinates_used': 0,
            'actual_frontier_layout': self.frontier_layout}


def gaussian_proposal_from_c(target, C_posterior, extra_flat_proposals):
    """Embed the full C posterior, new-record flat proposal laws and aux N(0,I)."""
    extra_flat_proposals = tuple(extra_flat_proposals)
    if (C_posterior.shared_dimension != target.rank
            or C_posterior.binding['source_chart_identity'] != target.source.identity):
        raise ValueError('the C proposal preserves all original named source coordinates')
    c_ids = tuple(C_posterior.drift_coordinate_ids)
    if not set(c_ids) <= set(target.drift_ids):
        raise ValueError('the C proposal cannot introduce a different physical record chart')
    index = list(range(target.rank))+[target.rank+target.drift_ids.index(name) for name in c_ids]
    mean, factor = np.zeros(target.dimension), np.zeros((target.dimension, target.dimension))
    mean[index] = C_posterior.mean
    factor[np.ix_(index, range(len(index)))] = C_posterior.factor
    offset = len(index)
    used = set(c_ids)
    for coordinate_ids, proposal in extra_flat_proposals:
        if used & set(coordinate_ids) or not set(coordinate_ids) <= set(target.drift_ids):
            raise ValueError('each new physical drift proposal appears once in the original chart')
        rows = [target.rank+target.drift_ids.index(name) for name in coordinate_ids]
        mean[rows] = proposal.mean
        factor[np.ix_(rows, range(offset, offset+len(rows)))] = proposal.root
        offset += len(rows)
        used.update(coordinate_ids)
    if used != set(target.drift_ids) or offset != target.base_dimension:
        raise ValueError('a proper proposal reference is supplied for every remaining original drift')
    auxiliary_width = target.dimension-target.base_dimension
    factor[target.base_dimension:, target.base_dimension:] = np.eye(auxiliary_width)
    # Rotate independent normal coordinates, retaining full support without Gram.
    lower = np.linalg.qr(factor.T, mode='r').T
    lower *= np.where(np.diag(lower) < 0, -1., 1.)[None, :]
    return GaussianProposal(mean, lower), freeze_metadata({'role': 'C_posterior_and_new_record_flat_proposal_only',
        'target': target.identity, 'C_posterior_binding': plain_metadata(C_posterior.binding),
        'C_mean_factor_sha256': array_digest(C_posterior.mean, C_posterior.factor),
        'extra_physical_proposal_ids': [list(ids) for ids, _ in extra_flat_proposals],
        'Gaussian_auxiliary_reference': 'independent_standard_normal', 'target_prior_changed': False})
