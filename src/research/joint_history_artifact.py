"""Complete local-factor numeric target codec; cold state grants no live use."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from research.frozen_metadata import plain_metadata
from research.gaussian_local_factor import (
    GaussianCandidateFactor,
    GaussianLocalPotential,
    SharedResponseChart,
)
from research.joint_history_inference import load_log_record, save_log_record
from research.joint_history_target import ConstrainedGaussianTable, RaoBlackwellHistoryTarget
from research.response_frontier import LogFactor, Roster, StateKey
from transfer.noise_model import metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def save_joint_history_target(target, path):
    if type(target) is not RaoBlackwellHistoryTarget:
        raise TypeError('the production codec saves the complete actual-scope Rao-Blackwell target')
    path = Path(path)
    recipes = getattr(target, 'native_component_recipes', None)
    store = None
    if recipes is not None:
        from research.joint_history_numeric_store import SharedHistoryNumericStore

        store = SharedHistoryNumericStore(target.native_numeric_store)
    arrays = {} if store is not None else {'source_mean': target.source.mean, 'source_factor': target.source.factor}
    potentials, by_identity, endpoints = {}, {}, {}
    def intern(potential):
        identity = potential.identity
        if identity not in by_identity:
            key = f'potential_{len(potentials)}'
            if store is not None:
                if identity not in recipes:
                    raise ValueError('compact history target requires every actual local component recipe')
                potentials[key] = {'identity': identity, 'native_recipe': plain_metadata(recipes[identity])}
            else:
                arrays[key] = potential.residual_root
                potentials[key] = {'identity': identity, 'log_normalizer': potential.log_normalizer,
                    'shared_dimension': potential.shared_dimension, 'drift_dimension': potential.drift_dimension,
                    'local_dimensions': potential.local_dimensions, 'binding': plain_metadata(potential.binding)}
            by_identity[identity] = key
        return by_identity[identity]
    for endpoint, tables in target.tables.items():
        rows = []
        for table in tables:
            if type(table) is GaussianLocalPotential:
                rows.append({'kind': 'constant', 'potential': intern(table)})
                continue
            constrained = type(table) is ConstrainedGaussianTable
            factor = table.factor if constrained else table
            row = {'kind': 'constrained' if constrained else 'complete', 'scope': factor.scope,
                'potentials': [None if constrained and table.zero_mask.ravel()[i] else intern(potential)
                    for i, potential in enumerate(factor.potentials)]}
            if constrained:
                row['exact_C_zero_binding'] = plain_metadata(table.exact_C_zero_binding)
            rows.append(row)
        endpoints[endpoint] = rows
    label_rows = []
    for i, factor in enumerate(target.discrete.label_factors):
        key = f'C_label_{i}'
        arrays[key] = factor.values
        label_rows.append({'scope': factor.scope, 'values': key})
    numerical = path.with_suffix('.npz')
    np.savez_compressed(numerical, **arrays)
    record = {'schema': ('native_sufficient_Rao_Blackwell_history_numeric_target/2' if store is not None
        else 'complete_Rao_Blackwell_history_numeric_target/1'), 'target_identity': target.identity,
        'source': {'identity': target.source.identity, 'coordinates': target.source.coordinate_ids,
            'channels': target.source.channel_ids, 'binding': plain_metadata(target.source.parameter_binding)},
        'rosters': [{'id': roster.selection_id, 'states': [vars(state) for state in roster.states],
            'log_mass': roster.log_mass.tolist()} for roster in target.rosters],
        'current': target.current, 'q_values': target.q_values, 'C_label_factors': label_rows,
        'endpoint_records': target.endpoint_records, 'drift_ids_by_record': target.drift_ids_by_record,
        'factory_identity': target.factory_identity, 'input_binding': plain_metadata(target.input_binding),
        'potentials': potentials, 'endpoint_tables': endpoints,
        'actual_numeric_snapshot': metadata_digest(target.numeric_snapshot()),
        'numerical_payload': {'path': numerical.name, 'sha256': hashlib.sha256(numerical.read_bytes()).hexdigest(),
            'bytes': numerical.stat().st_size}, 'numeric_only_grants_live_raw_source_or_fit_authority': False,
        'producer_sha256': IMPLEMENTATION_SHA256}
    if store is not None:
        record['shared_numeric_store'] = str(store.directory)
        record['source_arrays'] = {'mean': store.put(target.source.mean), 'factor': store.put(target.source.factor)}
        record['cold_operations'] = 'original_native_component_QR_and_original_Gaussian_local_integration_zero_fits'
        record['candidate_dense_residual_roots_stored'] = False
    return save_log_record(path, record)


def load_joint_history_target(path, *, expected_sha256):
    path = Path(path)
    record = load_log_record(path, expected_sha256=expected_sha256)
    compact = record['schema'] == 'native_sufficient_Rao_Blackwell_history_numeric_target/2'
    if (record['schema'] not in {'complete_Rao_Blackwell_history_numeric_target/1',
            'native_sufficient_Rao_Blackwell_history_numeric_target/2'}
            or record['numeric_only_grants_live_raw_source_or_fit_authority'] is not False
            or record['producer_sha256'] != IMPLEMENTATION_SHA256):
        raise ValueError('history cold replay requires its original numeric compiler and saved role')
    numerical = path.parent/record['numerical_payload']['path']
    if hashlib.sha256(numerical.read_bytes()).hexdigest() != record['numerical_payload']['sha256']:
        raise ValueError('complete local history numeric factors changed saved bytes')
    with np.load(numerical, allow_pickle=False) as arrays:
        source_record = record['source']
        store = None
        if compact:
            from research.joint_history_numeric_store import SharedHistoryNumericStore
            from research.native_joint_numeric_store import recompile_native_component

            store = SharedHistoryNumericStore(record['shared_numeric_store'])
        mean, factor = ((store.get(record['source_arrays'][name]) for name in ('mean', 'factor'))
            if compact else (arrays['source_mean'], arrays['source_factor']))
        source = SharedResponseChart(mean, factor,
            tuple(source_record['coordinates']), tuple(source_record['channels']), source_record['binding'])
        if source.identity != source_record['identity']:
            raise ValueError('cold history source changed a complete named source direction')
        potentials = {}
        for key, row in record['potentials'].items():
            potential = (recompile_native_component(row['native_recipe'], store, source) if compact
                else GaussianLocalPotential(arrays[key], row['log_normalizer'], row['shared_dimension'],
                    row['drift_dimension'], tuple(tuple(item) for item in row['local_dimensions']), row['binding']))
            if potential.identity != row['identity']:
                raise ValueError('cold local Gaussian changed its original rows, prior, physical measure or normalizer')
            potentials[key] = potential
        rosters = tuple(Roster(row['id'], tuple(StateKey(**state) for state in row['states']),
            np.array(row['log_mass'])) for row in record['rosters'])
        labels = tuple(LogFactor(tuple(row['scope']), arrays[row['values']].copy()) for row in record['C_label_factors'])
    tables = {}
    from itertools import product

    for endpoint, rows in record['endpoint_tables'].items():
        values = []
        for row in rows:
            if row['kind'] == 'constant':
                values.append(potentials[row['potential']])
                continue
            scope = tuple(row['scope'])
            keys = product(*(rosters[i].states for i in scope))
            entries = {key: None if reference is None else potentials[reference]
                for key, reference in zip(keys, row['potentials'], strict=True)}
            if row['kind'] == 'constrained':
                values.append(ConstrainedGaussianTable(rosters, scope, entries,
                    exact_C_zero_binding=row['exact_C_zero_binding']))
            elif row['kind'] == 'complete':
                values.append(GaussianCandidateFactor(rosters, scope, entries))
            else:
                raise ValueError('cold history requires its explicit original component kind')
        tables[endpoint] = tuple(values)
    target = RaoBlackwellHistoryTarget(source, rosters, record['current'], tuple(record['q_values']), labels,
        tables, record['endpoint_records'], record['drift_ids_by_record'],
        factory_identity=record['factory_identity'], input_binding=record['input_binding'])
    if (target.identity != record['target_identity']
            or metadata_digest(target.numeric_snapshot()) != record['actual_numeric_snapshot']):
        raise ValueError('cold history changed its complete scientific state')
    if compact:
        target.native_component_recipes = {row['identity']: row['native_recipe'] for row in record['potentials'].values()}
        target.native_numeric_store = record['shared_numeric_store']
    return target
