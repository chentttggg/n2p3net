"""Lossless final common-cloud reconstruction from actual SMC transitions.

The original SMC owns weights, resampling and accept decisions. This codec
records them and replays only proposal draws/moves; it has no target callback.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import marshal
import sys
import textwrap
from functools import cache
from pathlib import Path
from types import CodeType

import numpy as np

from research.collapsed_continuous_inference import GaussianMixtureProposal, GaussianProposal
from research.structured_continuous_proposal import (
    RecordConditionalGaussian,
    RecordConditionalStudent,
)
from transfer.noise_model import array_digest, metadata_digest

SCHEMA = 'n2p3_common_cloud_transition_trace/2'


def _bytes_identity(value):
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise TypeError('transition traces require pickle-free numeric arrays')
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(str(array.shape).encode())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def _portable_code(code):
    # Hash trusted live code, including nested code and exact constants. Source
    # paths and line locations are not part of the numerical method. No marshal
    # payload is saved or loaded by the artifact codec.
    constants = tuple(_portable_code(item) if isinstance(item, CodeType) else item for item in code.co_consts)
    return code.replace(co_filename='', co_firstlineno=0, co_linetable=b'', co_consts=constants)


@cache
def _function_program(function, actual_code):
    source = textwrap.dedent(inspect.getsource(function))
    syntax = ast.dump(ast.parse(source), include_attributes=False)
    return metadata_digest({'syntax': syntax,
        'live_bytecode_sha256': hashlib.sha256(marshal.dumps(_portable_code(actual_code))).hexdigest()})


def _program(proposal):
    methods = ['draw', 'log_density', 'reversible_move']
    if type(proposal) is GaussianMixtureProposal:
        methods.append('_component_logs')
    if type(proposal) is RecordConditionalGaussian:
        methods.append('_points')
    result = {}
    for name in methods:
        function = getattr(proposal, name).__func__
        result[name] = _function_program(function, function.__code__)
    properties = ()
    if type(proposal) is RecordConditionalGaussian:
        properties = ('mean', 'rank', 'dimension')
    elif type(proposal) is RecordConditionalStudent:
        properties = ('mean',)
    for name in properties:
        function = getattr(type(proposal), name).fget
        result[name] = _function_program(function, function.__code__)
    return result


def proposal_fingerprint(proposal):
    """Hash every actual proper parameter array, without a dense JSON record.

    Serialization wrappers pass their original proposal. Numerical methods and
    named factors are bound separately; no covariance equivalence substitutes
    a different factor, component order or positive mixture mass.
    """
    actual = type(proposal)
    result = {'type': actual.__module__+'.'+actual.__qualname__, 'program': _program(proposal)}
    if actual is GaussianProposal:
        result['parameters'] = _bytes_identity(proposal.mean), _bytes_identity(proposal.root)
    elif actual is GaussianMixtureProposal:
        result['weights'] = _bytes_identity(proposal.log_weights)
        result['components'] = [proposal_fingerprint(value) for value in proposal.components]
    elif actual is RecordConditionalGaussian:
        result['head'] = _bytes_identity(proposal.head_mean), _bytes_identity(proposal.head_root)
        result['blocks'] = [{'record_id': b.record_id, 'arrays': [_bytes_identity(a) for a in
            (b.chart_root, b.mean, b.regression, b.root)]} for b in proposal.blocks]
    elif actual is RecordConditionalStudent:
        result['base'] = proposal_fingerprint(proposal.base)
        result['degrees_of_freedom'] = float(proposal.degrees_of_freedom)
    else:
        raise TypeError('trace consumes an explicit original proper proposal owner')
    return metadata_digest(result)


@cache
def _binary_program(kind):
    module = sys.modules[kind.__module__]
    return {'class': kind.__name__, 'module': kind.__module__,
        'binary_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}


def _rng_program(kind):
    return {**_binary_program(kind), 'numpy_version': np.__version__,
        'generator': _binary_program(np.random.Generator),
        'seed_sequence': _binary_program(np.random.SeedSequence)}


def _encode_state(value, arrays, prefix):
    if isinstance(value, np.ndarray):
        key = prefix+'_'+str(len(arrays))
        arrays[key] = value.copy()
        return {'rng_array': key}
    if isinstance(value, np.generic):
        return {'rng_scalar': value.dtype.str, 'value': value.item()}
    if isinstance(value, dict):
        return {key: _encode_state(item, arrays, prefix) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode_state(item, arrays, prefix) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('unsupported complete BitGenerator state operand')


def _decode_state(value, arrays):
    if isinstance(value, dict):
        if set(value) == {'rng_array'}:
            return arrays[value['rng_array']].copy()
        if set(value) == {'rng_scalar', 'value'}:
            return np.array(value['value'], dtype=value['rng_scalar'])[()]
        return {key: _decode_state(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_state(item, arrays) for item in value]
    return value


def _restore_rng(state, arrays, program):
    decoded = _decode_state(state, arrays)
    kind = getattr(np.random, program['class'])
    if _rng_program(kind) != program or decoded['bit_generator'] != program['class']:
        raise ValueError('trace requires its actual BitGenerator program and version')
    generator = kind(0)
    generator.state = decoded
    return np.random.Generator(generator)


def _state_identity(state):
    arrays = {}
    structure = _encode_state(state, arrays, 'state')
    return metadata_digest({'structure': structure,
        'arrays': {name: _bytes_identity(value) for name, value in arrays.items()}})


class CommonCloudTrace:
    """Optional observer of the one original SMC loop; consumes no random draw."""

    def __init__(self, proposal):
        self.proposal_identity = proposal_fingerprint(proposal)
        self.replicates = {}
        self.arrays = {}

    def __call__(self, event, *, replicate, **fields):
        key = f'trace_r{replicate}'
        if event == 'initial_draw':
            if replicate in self.replicates:
                raise ValueError('trace observed a repeated replicate start')
            self.replicates[replicate] = {'replicate': replicate, 'particles': fields['particles'],
                'rng_program': _rng_program(fields['bit_generator']),
                'initial_state': _encode_state(fields['rng_state'], self.arrays, key+'_initial'), 'events': []}
            return
        row = self.replicates[replicate]
        if event == 'resample':
            ancestors = np.asarray(fields['ancestors'])
            dtype = np.min_scalar_type(row['particles']-1)
            name = key+f"_s{fields['stage']}_ancestors"
            self.arrays[name] = ancestors.astype(dtype)
            record = {'event': event, 'stage': fields['stage'], 'gamma': fields['gamma'], 'ancestors': name}
        elif event == 'move':
            prefix = key+f"_s{fields['stage']}_m{fields['move']}"
            name = prefix+'_accept'
            self.arrays[name] = np.packbits(fields['accept'], bitorder='little')
            record = {'event': event, 'stage': fields['stage'], 'move': fields['move'],
                'rng_state': _encode_state(fields['rng_state'], self.arrays, prefix+'_rng'),
                'rho': float(fields['rho']), 'accept': name}
        elif event == 'replicate_complete':
            row['final_points_sha256'] = fields['points_sha256']
            row['final_points_shape'] = tuple(fields['points_shape'])
            return
        else:
            raise ValueError('unknown common-cloud observation event')
        row['events'].append(record)


def _configuration(result):
    names = ('target_identity', 'endpoints', 'q_values', 'component_log_scales', 'inference_seed',
             'particles', 'replicates', 'temperatures', 'moves', 'pcn_scale')
    return {name: result[name] for name in names}


def _check_order(row, configuration):
    stages = len(configuration['temperatures'])-1
    expected = [('resample', stage, None) if move is None else ('move', stage, move)
        for stage in range(stages) for move in (None, *range(configuration['moves']))]
    actual = [(event['event'], event['stage'], event.get('move')) for event in row['events']]
    if actual != expected or row['particles'] != configuration['particles']:
        raise ValueError('trace changed its complete ordered stage/move/particle programme')
    for event in row['events']:
        if event['event'] == 'resample' and event['gamma'] != configuration['temperatures'][event['stage']+1]:
            raise ValueError('trace resampling temperature changed')
        if event['event'] == 'move' and event['rho'] != configuration['pcn_scale']:
            raise ValueError('trace changed the actual proposal move scale')


def save_common_cloud_trace(path, result, arrays, trace):
    """Save every scientific array except reconstructible final points."""
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('trace numerical payload requires an explicit npz path')
    configuration = _configuration(result)
    if sorted(trace.replicates) != list(range(result['replicates'])):
        raise ValueError('trace dropped or added a production replicate')
    payload = dict(trace.arrays)
    science = {}
    for key, value in arrays.items():
        science[key] = {'shape': value.shape, 'dtype': value.dtype.str, 'sha256': _bytes_identity(value)}
        if not key.endswith('_points'):
            payload['science_'+key] = value
    for index, row in sorted(trace.replicates.items()):
        _check_order(row, configuration)
        if array_digest(arrays[f'replicate_{index}_points']) != row['final_points_sha256']:
            raise ValueError('observed final point cloud differs from actual original output')
    with path.open('xb') as stream:
        np.savez_compressed(stream, **payload)
    metadata = {'schema': SCHEMA, 'proposal_identity': trace.proposal_identity,
        'configuration': configuration, 'replicate_traces': [trace.replicates[i] for i in sorted(trace.replicates)],
        'scientific_arrays': science, 'payload_arrays': {key: {'dtype': value.dtype.str, 'shape': value.shape,
            'sha256': _bytes_identity(value)} for key, value in payload.items()},
        'payload': {'name': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'bytes': path.stat().st_size},
        'reconstruction': 'original_proposal_draw_and_moves_with_saved_ancestry_and_accept_mask',
        'target_evaluations_during_reconstruction': 0, 'accept_decisions_recomputed': 0,
        'numeric_only_grants_live_authority': False}
    metadata_path = path.with_suffix('.json')
    with metadata_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n')
    return {'schema': SCHEMA, 'path': str(metadata_path.resolve()),
        'sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        'bytes': metadata_path.stat().st_size+path.stat().st_size}


def load_common_cloud_trace(descriptor, proposal):
    """Restore final arrays without target calls or another accept decision."""
    if descriptor['schema'] != SCHEMA:
        raise ValueError('not an explicit transition-trace cloud descriptor')
    metadata_path = Path(descriptor['path'])
    content = metadata_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != descriptor['sha256']:
        raise ValueError('common-cloud transition trace changed its externally pinned bytes')
    metadata = json.loads(content)
    if metadata['schema'] != SCHEMA or proposal_fingerprint(proposal) != metadata['proposal_identity']:
        raise ValueError('trace requires its complete actual original proper proposal')
    payload_path = metadata_path.parent / metadata['payload']['name']
    payload_bytes = payload_path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != metadata['payload']['sha256']:
        raise ValueError('transition ancestry, masks, RNG or scientific arrays changed')
    with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as archive:
        if set(archive.files) != set(metadata['payload_arrays']):
            raise ValueError('trace payload changed its complete numerical operand set')
        payload = {name: archive[name] for name in archive.files}
    for name, array in payload.items():
        expected = metadata['payload_arrays'][name]
        if (array.dtype.str != expected['dtype'] or list(array.shape) != expected['shape']
                or _bytes_identity(array) != expected['sha256']):
            raise ValueError('transition trace changed complete exact numeric bytes')
    configuration = metadata['configuration']
    rows = metadata['replicate_traces']
    if [row['replicate'] for row in rows] != list(range(configuration['replicates'])):
        raise ValueError('trace dropped or reordered a complete production replicate')
    streams = np.random.SeedSequence(configuration['inference_seed']).spawn(configuration['replicates'])
    output = {key.removeprefix('science_'): value for key, value in payload.items() if key.startswith('science_')}
    for row in rows:
        _check_order(row, configuration)
        count, program = row['particles'], row['rng_program']
        rng = _restore_rng(row['initial_state'], payload, program)
        expected_rng = np.random.default_rng(streams[row['replicate']])
        if _state_identity(rng.bit_generator.state) != _state_identity(expected_rng.bit_generator.state):
            raise ValueError('trace initial RNG state differs from its declared inference seed and replicate stream')
        points = proposal.draw(rng, count)
        for event in row['events']:
            if event['event'] == 'resample':
                ancestors = payload[event['ancestors']]
                if ancestors.shape != (count,) or ancestors.dtype.kind != 'u' or np.any(ancestors >= count):
                    raise ValueError('trace resampling ancestry changed its full particle scope')
                points = points[ancestors.astype(np.intp)]
            else:
                packed = payload[event['accept']]
                if packed.dtype != np.dtype('uint8') or packed.shape != ((count+7)//8,):
                    raise ValueError('trace move changed its complete packed acceptance mask')
                accept = np.unpackbits(packed, bitorder='little', count=count).astype(bool)
                rng = _restore_rng(event['rng_state'], payload, program)
                trial = proposal.reversible_move(rng, points, event['rho'])
                points[accept] = trial[accept]
        key = f"replicate_{row['replicate']}_points"
        if list(points.shape) != row['final_points_shape'] or array_digest(points) != row['final_points_sha256']:
            raise ValueError('trace failed to reconstruct original final points bit for bit')
        output[key] = points
    if set(output) != set(metadata['scientific_arrays']):
        raise ValueError('trace replay changed the complete scientific array set')
    for name, array in output.items():
        expected = metadata['scientific_arrays'][name]
        if (array.dtype.str != expected['dtype'] or list(array.shape) != expected['shape']
                or _bytes_identity(array) != expected['sha256']):
            raise ValueError('reconstructed scientific array differs from original exact bytes')
    return output
