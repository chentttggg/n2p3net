"""Exact per-cell file closure, with CAS identities suitable for group union."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from research.joint_history_inference import load_log_record
from research.joint_history_numeric_store import SharedHistoryNumericStore
from research.joint_history_target import ConstrainedGaussianTable
from research.native_joint_numeric_store import get_numeric_record


def history_storage_closure(target_path, *, target_sha256, inference_path, inference_sha256):
    target_path, inference_path = Path(target_path), Path(inference_path)
    target = load_log_record(target_path, expected_sha256=target_sha256)
    inference = load_log_record(inference_path, expected_sha256=inference_sha256)
    if (target['schema'] != 'native_sufficient_Rao_Blackwell_history_numeric_target/2'
            or target['target_identity'] != inference['target_identity']
            or target['candidate_dense_residual_roots_stored'] is not False
            or inference['proper_proposal']['candidate_mean_or_root_arrays_stored'] is not False):
        raise ValueError('storage closure requires the original compact target and proposal recipes')
    store = SharedHistoryNumericStore(target['shared_numeric_store'])
    if str(store.directory) != inference['proper_proposal']['store']:
        raise ValueError('storage closure keeps the original shared target/C/source numeric dependencies')
    dependencies = {}

    def visit(value):
        if isinstance(value, dict):
            kind = value.get('kind')
            if kind == 'native_numeric_record/1':
                name = value['sha256']+'.json'
                if name not in dependencies:
                    dependencies[name] = {'sha256': value['sha256'], 'bytes': value['bytes'], 'kind': kind}
                    visit(get_numeric_record(store, value))
            elif {'sha256', 'numeric_identity', 'shape', 'bytes'} <= value.keys():
                name = value['sha256']+'.npz'
                if name not in dependencies:
                    store.get(value)
                    dependencies[name] = {key: value[key] for key in ('sha256', 'numeric_identity', 'shape', 'bytes')}
            else:
                for item in value.values():
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(target)
    visit(inference['proper_proposal'])
    numerical = target_path.parent/target['numerical_payload']['path']
    if hashlib.sha256(numerical.read_bytes()).hexdigest() != target['numerical_payload']['sha256']:
        raise ValueError('storage closure changed the complete target label payload')
    cloud = inference['saved_cloud']
    metadata_path = Path(cloud['path'])
    content = metadata_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != cloud['sha256']:
        raise ValueError('storage closure changed the original common-cloud descriptor')
    trace = json.loads(content)
    payload = metadata_path.parent/trace['payload']['name']
    if hashlib.sha256(payload.read_bytes()).hexdigest() != trace['payload']['sha256']:
        raise ValueError('storage closure changed original cloud science or transition operands')
    own = {str(path.resolve()): {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (target_path, numerical, inference_path, metadata_path, payload)}
    return {'schema': 'compact_history_cell_storage_closure/1', 'target_identity': target['target_identity'],
        'cell_files': own, 'cell_file_bytes': sum(row['bytes'] for row in own.values()),
        'shared_numeric_store': str(store.directory), 'shared_dependencies': dependencies,
        'shared_dependency_bytes_once': sum(row['bytes'] for row in dependencies.values()),
        'union_rule': 'union_shared_dependency_names_within_the_same_store_before_summing_matrix_bytes',
        'actual_distinct_native_potentials': len(target['potentials']),
        'dense_candidate_potential_or_proposal_roots_stored': False,
        'whole_matrix_storage_acceptance': False,
        'matrix_acceptance_requires': 'actual_4914_cell_file_closures_and_group_union_with_free_disk_margin'}


def history_target_cost(target):
    """Count live dense roots without allocating, copying or rehashing them."""
    roots = {}
    for tables in target.tables.values():
        for table in tables:
            factor = table.factor if type(table) is ConstrainedGaussianTable else table
            potentials = getattr(factor, 'potentials', (factor,))
            for potential in potentials:
                roots[id(potential.residual_root)] = potential.residual_root
    recipes = getattr(target, 'native_component_recipes', {})
    return {'schema': 'actual_history_numeric_target_cost/1', 'dimension': target.dimension,
        'source_rank': target.rank, 'endpoint_count': len(target.endpoints), 'candidate_count': len(target.q_values),
        'dense_residual_roots_in_memory': len(roots), 'dense_residual_root_bytes_in_memory': sum(root.nbytes for root in roots.values()),
        'maximum_dense_root_shape': max((root.shape for root in roots.values()), key=lambda shape: shape[0]*shape[1], default=(0, 0)),
        'distinct_native_potential_identities': len(recipes),
        'distinct_native_residual_root_values': len({row['expected_residual_root_sha256'] for row in recipes.values()}),
        'component_frontier_layout': target.frontier_layout,
        'cost_role': 'counts_from_this_actual_numeric_target_not_a_matrix_projection'}
