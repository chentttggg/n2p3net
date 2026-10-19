"""Short native likelihood operands; the original compiler owns every QR.

Cold holders expose component likelihood compilation only. They carry the
original leading rows and complete temporal roots, never a raw-data capability
or the full-record posterior-moment interface.
"""
from __future__ import annotations

import hashlib
import json
import uuid

from research.channel_noise_mechanism import NoiseGeometry
from research.frozen_metadata import freeze_metadata, plain_metadata
from research.native_joint_sufficient import IMPLEMENTATION_SHA256 as NATIVE_OWNER_SHA256
from research.native_joint_sufficient import NativeJointRecordCompiler, original_white_components
from research.private_ar1_noise_point import PrivateAR1NoisePoint
from transfer.noise_model import array_digest, frozen_array, metadata_digest


def put_numeric_record(store, record):
    content = (json.dumps(plain_metadata(record), sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    digest = hashlib.sha256(content).hexdigest()
    path = store.directory/(digest+'.json')
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('an immutable native numeric dependency changed its saved bytes')
    else:
        staging = store.directory/(digest+'.'+uuid.uuid4().hex+'.staging')
        staging.write_bytes(content)
        staging.replace(path)
    return freeze_metadata({'sha256': digest, 'bytes': len(content), 'kind': 'native_numeric_record/1'})


def get_numeric_record(store, reference):
    content = (store.directory/(reference['sha256']+'.json')).read_bytes()
    if (reference['kind'] != 'native_numeric_record/1' or len(content) != reference['bytes']
            or hashlib.sha256(content).hexdigest() != reference['sha256']):
        raise ValueError('a short native numeric dependency changed its external identity')
    return json.loads(content)


class _PrefixDesign:
    def __init__(self, design, error, binding):
        self.design, self.error, self.binding = frozen_array(design), frozen_array(error), freeze_metadata(binding)

    def __call__(self, start, stop):
        if not 0 <= start < stop <= len(self.design):
            raise ValueError('cold native likelihood holds only its original leading design rows')
        return self.design[start:stop], self.error[start:stop], self.binding


class NativeComponentNumericHolder:
    """Explicit numeric-only operand for the unique native component kernel."""

    _feature_width = NativeJointRecordCompiler._feature_width
    _prefix_physical = NativeJointRecordCompiler._prefix_physical
    _tail_map = NativeJointRecordCompiler._tail_map
    _compile_selected = NativeJointRecordCompiler._compile_selected
    compile_component = NativeJointRecordCompiler.compile_component


def _identity_input(compiler):
    return {'record': compiler.record_id, 'samples': compiler.observation_identity,
        'geometry': compiler.geometry.record(), 'p': compiler.p, 'drift': compiler.drift_ids,
        'specs': [{'selections': spec.selection_ids, 'intervals': spec.native_intervals,
            'tail_root': None if tail is None else array_digest(tail),
            'moment_root': None if moment is None else array_digest(moment)}
            for spec, tail, moment in zip(compiler.specs, compiler.tail_roots, compiler.moment_roots, strict=True)],
        'branch': compiler.branch_binding, 'producer': NATIVE_OWNER_SHA256}


def store_native_component(compiler, index, noise, within_factor, source, potential, store):
    """Export enough to call original compile_component with zero new fits."""
    if type(compiler) is not NativeJointRecordCompiler or type(noise) is not PrivateAR1NoisePoint:
        raise TypeError('short native export requires its explicit original compiler and noise programme')
    identity_input = _identity_input(compiler)
    if metadata_digest(identity_input) != compiler.identity or potential.binding['native_compiler'] != compiler.identity:
        raise ValueError('short native export requires the original unchanged complete component compiler')
    prefix = min(compiler.n, compiler.conditioning_samples)
    designs = {}
    for name, callback in compiler.designs.items():
        design, error, binding = callback(0, prefix)
        designs[name] = {'design': store.put(design), 'error': store.put(error), 'binding': plain_metadata(binding)}
    state = {'schema': 'native_component_likelihood_sufficient_state/1',
        'native_owner_sha256': NATIVE_OWNER_SHA256, 'identity': compiler.identity,
        'identity_input': identity_input, 'record_id': compiler.record_id, 'drift_group_id': compiler.drift_group_id,
        'n': compiler.n, 'p': compiler.p, 'c': compiler.c, 'geometry': compiler.geometry.record(),
        'drift_ids': compiler.drift_ids, 'supports': compiler.supports, 'branch_binding': compiler.branch_binding,
        'block_rows': compiler.block_rows, 'conditioning_samples': compiler.conditioning_samples,
        'compiled_component_indices': compiler.compiled_component_indices,
        'observation_identity': compiler.observation_identity, 'prefix_samples': store.put(compiler.samples[:prefix]),
        'prefix_designs': designs, 'tail_roots': [None if root is None else store.put(root) for root in compiler.tail_roots],
        'full_native_intervals': compiler.intervals,
        'posterior_moment_operations_available': False, 'numeric_only_no_live_grant': True}
    noise_record = noise.record()
    return freeze_metadata({'kind': 'native_component_likelihood_recompile/1',
        'compiler': put_numeric_record(store, state), 'component_index': index,
        'noise': put_numeric_record(store, noise_record),
        'noise_programme': {'schema': noise_record['schema'], 'producer': noise_record['producer']},
        'within_factor': store.put(within_factor), 'source_identity': source.identity,
        'expected_potential_identity': potential.identity,
        'expected_residual_root_sha256': array_digest(potential.residual_root)})


def _restore_compiler(record, store):
    if (record['schema'] != 'native_component_likelihood_sufficient_state/1'
            or record['native_owner_sha256'] != NATIVE_OWNER_SHA256
            or record['posterior_moment_operations_available'] is not False
            or record['numeric_only_no_live_grant'] is not True
            or metadata_digest(record['identity_input']) != record['identity']):
        raise ValueError('cold native likelihood requires its exact original numeric kernel and role')
    holder = NativeComponentNumericHolder()
    for name in ('record_id', 'drift_group_id', 'n', 'p', 'c', 'block_rows', 'conditioning_samples',
            'observation_identity', 'identity'):
        setattr(holder, name, record[name])
    holder.geometry = NoiseGeometry(**record['geometry'])
    holder.drift_ids, holder.supports = tuple(record['drift_ids']), {name: tuple(value) for name, value in record['supports'].items()}
    holder.branch_binding = freeze_metadata(record['branch_binding'])
    holder.compiled_component_indices = tuple(record['compiled_component_indices'])
    holder.samples = frozen_array(store.get(record['prefix_samples']))
    prefix = min(holder.n, holder.conditioning_samples)
    if holder.samples.shape != (prefix, holder.c) or holder.c != len(holder.geometry.channels):
        raise ValueError('short native observations lost complete original startup rows or channels')
    holder.specs = original_white_components(holder.supports, native_samples=holder.n, conditioning_samples=holder.conditioning_samples)
    holder.intervals = tuple(sorted((start, stop, i) for i, spec in enumerate(holder.specs) for start, stop in spec.native_intervals))
    if holder.intervals != tuple(map(tuple, record['full_native_intervals'])):
        raise ValueError('short native state changed the full original component row ledger')
    expected = record['identity_input']
    if (metadata_digest(expected['geometry']) != metadata_digest(holder.geometry.record())
            or expected['record'] != holder.record_id or expected['samples'] != holder.observation_identity
            or expected['p'] != holder.p or tuple(expected['drift']) != holder.drift_ids
            or metadata_digest(expected['branch']) != metadata_digest(holder.branch_binding)):
        raise ValueError('short native metadata differs from its complete original compiler identity')
    holder.tail_roots = []
    for spec, reference, original in zip(holder.specs, record['tail_roots'], expected['specs'], strict=True):
        root = None if reference is None else frozen_array(store.get(reference))
        if (spec.selection_ids != tuple(original['selections']) or spec.native_intervals != tuple(map(tuple, original['intervals']))
                or (None if root is None else array_digest(root)) != original['tail_root']
                or (root is not None and root.shape[1] != holder._feature_width(spec))):
            raise ValueError('short native temporal root changed a complete original component feature')
        holder.tail_roots.append(root)
    holder.designs = {}
    for name, row in record['prefix_designs'].items():
        design, error = store.get(row['design']), store.get(row['error'])
        if design.shape != (prefix, holder.p) or error.shape != design.shape:
            raise ValueError('short native design lost the complete original leading coefficient rows')
        holder.designs[name] = _PrefixDesign(design, error, row['binding'])
    return holder


def recompile_native_component(recipe, store, source):
    if recipe['kind'] != 'native_component_likelihood_recompile/1' or source.identity != recipe['source_identity']:
        raise ValueError('short native recipe requires the original complete shared source law')
    noise_record = get_numeric_record(store, recipe['noise'])
    if recipe['noise_programme'] != {name: noise_record[name] for name in ('schema', 'producer')}:
        raise ValueError('short native recipe changed its explicit noise programme')
    # The explicit programme retains its own schema, source pin and cold gate.
    noise = PrivateAR1NoisePoint.from_record(noise_record)
    compiler = _restore_compiler(get_numeric_record(store, recipe['compiler']), store)
    reduction = compiler.compile_component(noise, store.get(recipe['within_factor']), source, recipe['component_index'])
    potential = reduction.potential
    if (potential.identity != recipe['expected_potential_identity']
            or array_digest(potential.residual_root) != recipe['expected_residual_root_sha256']):
        raise ValueError('short native recompile changed its original complete potential or physical measure')
    return potential
