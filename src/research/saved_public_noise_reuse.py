"""Join an original saved C-noise execution to genuine current public grants.

The saved witness and mathematical comparisons are numeric. Publication owns a
new live relation; the original fitted operand and its provenance stay intact.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from weakref import WeakKeyDictionary

from research.frozen_metadata import freeze_metadata
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_WITNESSES, _REUSES = WeakKeyDictionary(), WeakKeyDictionary()
SEGMENT_LAW = 'original_record_stationary_startup_C_history_physical_union/1'


def _pinned_json(reference):
    content = Path(reference['path']).read_bytes()
    if hashlib.sha256(content).hexdigest() != reference['sha256']:
        raise ValueError('saved execution metadata changed its pinned bytes')
    return json.loads(content)


@dataclass(frozen=True, eq=False)
class SavedPublicNoiseWitness:
    record: dict
    programme: dict
    descriptor: dict
    identity: str

    @classmethod
    def load(cls, path, *, expected_sha256):
        record = _pinned_json({'path': str(path), 'sha256': expected_sha256})
        if (record['schema'] != 'saved_public_C_noise_execution_witness/1'
                or record['numeric_witness_grants_data_authority'] is not False):
            raise ValueError('saved public noise requires its numeric execution witness')
        programme = _pinned_json(record['original_programme'])
        descriptor = _pinned_json(record['original_operand'])
        if (metadata_digest(programme) != metadata_digest(descriptor['programme'])
                or descriptor['identity'] != record['original_operand_identity']
                or descriptor['noise_record_sha256'] != record['original_noise_record_sha256']
                or programme['source_graph'] != record['source_law']['original_graph_identity']
                or programme['source_prior_sha256'] != record['source_law']['source_mean_factor_sha256']):
            raise ValueError('saved execution witness must retain its original operand/programme/law')
        result = cls(freeze_metadata(record), freeze_metadata(programme),
                     freeze_metadata(descriptor), expected_sha256)
        _WITNESSES[result] = result.record
        return result


def _require_witness(value):
    if type(value) is not SavedPublicNoiseWitness or _WITNESSES.get(value) is not value.record:
        raise TypeError('saved execution requires its pinned numeric witness loader')
    return value


def _is_identity(matrix, width):
    return len(matrix) == width and all(len(row) == width and all(
        value == int(i == j) for j, value in enumerate(row)) for i, row in enumerate(matrix))


def prove_saved_native_law(saved, model):
    """Compare actual named generators with the saved full native Gaussian law.

    This pure numeric proof accepts numeric test objects and grants no live use.
    The admission entry point separately requires the genuine source owner.
    """
    graph, parameters, dictionary = model.graph(), model.source.parameters, model.continuation
    dimensions = saved['coefficient_count'] * len(saved['geometry']['channels'])
    if (array_digest(graph.source_mean, graph.source_factor) != saved['source_mean_factor_sha256']
            or array_digest(parameters.prior.mean, parameters.prior.root) != saved['source_mean_factor_sha256']
            or array_digest(graph.source_mean) != saved['source_mean_sha256']
            or model.source.digest != saved['source_codec_sha256']
            or parameters.digest != saved['source_parameter_authority_sha256']
            or array_digest(parameters.basis) != saved['source_basis_sha256']
            or dictionary.digest() != saved['dictionary_identity']
            or dictionary.sample_rate_hz != saved['source_clock_hz']
            or array_digest(dictionary.matrix) != saved['source_basis_sha256']
            or model.target_participant_key != saved['target_participant_key']
            or metadata_digest(model.target_observation_geometry.record()) != metadata_digest(saved['geometry'])
            or metadata_digest(parameters.noise.geometry.record()) != metadata_digest(saved['geometry'])):
        raise ValueError('current native source changed a complete saved Gaussian/function/physical generator')
    if (graph.source_mean.shape != (dimensions,)
            or graph.coefficient_count != saved['coefficient_count']
            or not _is_identity(graph.source_map, dimensions)
            or tuple(graph.reference_chart_mean) != tuple(Fraction.from_float(float(v)) for v in graph.source_mean)
            or graph.conditional_gain or graph.conditional_covariance
            or not _is_identity(graph.chart.exact_matrix, len(saved['geometry']['channels']))
            or not _is_identity(graph.chart.exact_inverse, len(saved['geometry']['channels']))):
        raise ValueError('saved native law requires its exact full identity chart and all original directions')
    expected = {'source_codec_sha256': saved['source_codec_sha256'],
        'source_parameter_authority_sha256': saved['source_parameter_authority_sha256'],
        'source_original_native_geometry': saved['geometry'], 'target_observation_geometry': saved['geometry'],
        'target_participant_key': saved['target_participant_key'],
        'function_dictionary_sha256': saved['dictionary_identity'],
        'function_source_clock_hz': saved['source_clock_hz'], 'source_basis_sha256': saved['source_basis_sha256'],
        'coefficient_order': saved['coefficient_order'], 'coefficient_roles': saved['coefficient_roles'],
        'source_prior_preserved_sha256': saved['source_mean_factor_sha256'], 'completion_dimensions': 0}
    if any(metadata_digest(graph.binding[key]) != metadata_digest(value) for key, value in expected.items()):
        raise ValueError('current actual graph binding disagrees with its named saved native law')
    return {'original_graph_identity': saved['original_graph_identity'], 'current_graph_identity': graph.digest(),
        'source_mean_factor_sha256': array_digest(graph.source_mean, graph.source_factor),
        'source_basis_sha256': array_digest(parameters.basis), 'same_complete_named_parameter_law': True,
        'source_map': 'identity', 'reference_mean': 'original_source_mean',
        'chart': 'full_native_identity', 'conditional_completion_dimensions': 0,
        'source_directions_removed': 0, 'numeric_relation_grants_data_authority': False}


def _scope_semantics(scope):
    return {key: value for key, value in scope.items() if key not in
            ('original_grant_identity', 'original_source_owner_sha256', 'bridge_source_sha256')}


def _mean_semantics(binding):
    excluded = {'source_graph', 'C_input_identity', 'C_fixed_mean_programme',
                'mean_assembly_producer_sha256', 'mean_owner_producer_sha256'}
    result = {key: value for key, value in binding.items() if key not in excluded}
    compiler = {key: value for key, value in binding['source_functional_mean_compiler'].items()
                if key not in ('source_graph', 'producer_sha256')}
    compiler['actual_query_binding'] = {key: value for key, value in compiler['actual_query_binding'].items()
        if key not in ('C_input_identity', 'fixed_source_mean_programme')}
    result['source_functional_mean_compiler'] = compiler
    return result


def _residual_semantics(ledger):
    return {**ledger, 'provenance': {key: value for key, value in ledger['provenance'].items()
        if key not in ('source_model_sha256', 'source_input_sha256')}}


def compare_saved_residual_programmes(saved, current):
    """Pure relation over actual compiler outputs; no permission is conferred."""
    excluded = {'source_admission_identity', 'source_graph', 'fixed_mean_programme',
                'C_inputs', 'adapter_producer_sha256'}
    if (metadata_digest({k: v for k, v in saved.items() if k not in excluded}) !=
            metadata_digest({k: v for k, v in current.items() if k not in excluded})
            or len(saved['C_inputs']) != len(current['C_inputs'])):
        raise ValueError('noise reuse requires the complete original C recipe, scope and denominator')
    relations = []
    for old, new in zip(saved['C_inputs'], current['C_inputs'], strict=True):
        fields = ('granted_label_count', 'positive_label_count', 'source_mean_sha256',
            'source_mean_error_sha256', 'source_mean_error_max', 'residual_subtraction_bound_sha256',
            'residual_subtraction_bound_max')
        if (any(metadata_digest(old[key]) != metadata_digest(new[key]) for key in fields)
                or metadata_digest(_scope_semantics(old['original_public_scope'])) !=
                   metadata_digest(_scope_semantics(new['original_public_scope']))
                or metadata_digest(_mean_semantics(old['source_mean_binding'])) !=
                   metadata_digest(_mean_semantics(new['source_mean_binding']))
                or metadata_digest(_residual_semantics(old['residual'])) !=
                   metadata_digest(_residual_semantics(new['residual']))):
            raise ValueError('current C observations/bits/history/mean/error/residuals differ from the saved fit')
        for programme, row in ((saved, old), (current, new)):
            provenance, mean = row['residual']['provenance'], row['source_mean_binding']
            if (provenance['source_model_sha256'] != programme['source_graph']
                    or provenance['source_input_sha256'] != row['C_input_identity']
                    or mean['source_graph'] != programme['source_graph']
                    or mean['C_input_identity'] != row['C_input_identity']
                    or mean['C_fixed_mean_programme'] != programme['fixed_mean_programme']
                    or provenance['design_sha256'] != row['source_mean_sha256']):
                raise ValueError('C residual and mean must be owned by their actual side of the relation')
        relations.append({'record_id': old['residual']['record_id'],
            'original_input_identity': old['C_input_identity'], 'current_input_identity': new['C_input_identity'],
            'original_residual_sha256': old['residual']['residual_sha256'],
            'same_Y_bits_support_mean_error_and_residual': True,
            'original_ledger_sha256': metadata_digest(old['residual']),
            'current_ledger_sha256': metadata_digest(new['residual'])})
    return relations


@dataclass(frozen=True, eq=False)
class SavedPublicCNoiseReuse:
    witness: SavedPublicNoiseWitness
    operand: object
    compiled: object
    record: dict
    identity: str

    @property
    def source_admission(self):
        return self.compiled.source_admission

    @property
    def prefix(self):
        return self.compiled.prefix

    @property
    def noise(self):
        return self.operand.noise

    @property
    def eta(self):
        return self.operand.eta

    def carrier(self):
        return self.operand.carrier()

    def segment_binding(self, physical_input):
        from research.public_record_physical_bridge import _INPUTS, require_public_physical_input
        from transfer.ensemble_views import calibration_prefix_identity

        require_saved_public_c_noise_reuse(self)
        cap = require_public_physical_input(physical_input)
        owner = _INPUTS[cap]
        b, geometry, admission = cap.public_binding, self.noise.geometry, self.source_admission
        roster = admission.record['new_target_union']['source_record_roster']
        if (owner[0] is not admission.compiler or owner[1] is not self.prefix
                or b['C_prefix_identity'] != calibration_prefix_identity(self.prefix)
                or b['participant_key'] != admission.model.target_participant_key
                or b['record_id'] not in roster
                or b['source_sha256'] != roster[b['record_id']]['source_sha256']
                or (tuple(b['eeg_channels']), b['sample_rate_hz'], b['signal_unit'], b['reference'], b['operator']) !=
                   (geometry.channels, geometry.sfreq, geometry.unit, geometry.reference, geometry.operator_identity)):
            raise ValueError('saved noise application requires its current original acquisition/prefix/native grant')
        segment = {key: b[key] for key in ('dataset_id', 'participant_key', 'record_id', 'source_sha256')}
        return {'law': SEGMENT_LAW, 'original_record_noise_segment': metadata_digest(segment),
            'original_acquisition': segment, 'C_prefix_identity': b['C_prefix_identity'],
            'original_noise_operand': self.record['original_noise_operand'], 'saved_execution_reuse': self.identity,
            'observation_union': b['observation_union'], 'consume_each_original_sample_once': True,
            'startup_and_drift_once_per_original_record': True}


def admit_saved_public_c_noise(witness, operand, source_admission, prefix, calibration_inputs, *, fixed_mean):
    from research.public_native_source_admission import require_public_native_source
    from research.public_native_target_noise import (
        PublicNativeCNoiseOperand,
        compile_public_c_residuals,
    )

    witness = _require_witness(witness)
    admission = require_public_native_source(source_admission)
    if type(operand) is not PublicNativeCNoiseOperand:
        raise TypeError('saved C-noise reuse consumes the complete original numeric operand')
    operand.noise.assert_integrity()
    saved = witness.record
    if (operand.identity != saved['original_operand_identity']
            or metadata_digest(operand.programme) != metadata_digest(witness.programme)
            or metadata_digest(operand.noise.record()) != saved['original_noise_record_sha256']
            or array_digest(operand.initial_state_root()) != witness.descriptor['initial_state_root_sha256']
            or array_digest(operand.noise.compiled_prefix_root) != saved['conditioning_prefix_root_sha256']):
        raise ValueError('saved reuse must preserve all original fitted parameters, roots and ledgers')
    law = prove_saved_native_law(saved['source_law'], admission.model)
    compiled = compile_public_c_residuals(admission, prefix, calibration_inputs, mean=fixed_mean)
    rows = compare_saved_residual_programmes(witness.programme, compiled.programme)
    record = freeze_metadata({'schema': 'saved_public_C_noise_current_live_reuse/1',
        'saved_execution_witness': witness.identity, 'original_noise_operand': operand.identity,
        'original_complete_noise_record': saved['original_noise_record_sha256'],
        'original_C_programme': metadata_digest(witness.programme),
        'current_C_programme': metadata_digest(compiled.programme),
        'current_source_admission': metadata_digest(admission.record), 'exact_source_parameter_law': law,
        'current_C_records': rows, 'noise_refits': 0, 'source_fits': 0, 'response_C_updates_here': 0,
        'old_process_local_admission_recreated': False, 'original_fit_provenance_rewritten': False,
        'producer': IMPLEMENTATION_SHA256})
    result = SavedPublicCNoiseReuse(witness, operand, compiled, record, metadata_digest(record))
    _REUSES[result] = (compiled, operand, record)
    return result


def require_saved_public_c_noise_reuse(value):
    from research.public_native_source_admission import require_public_native_source
    from research.public_native_target_noise import _COMPILED
    from research.public_record_physical_bridge import require_public_physical_input

    if (type(value) is not SavedPublicCNoiseReuse
            or _REUSES.get(value) != (value.compiled, value.operand, value.record)):
        raise TypeError('public saved noise requires its genuine current source/C execution relation')
    _require_witness(value.witness)
    require_public_native_source(value.compiled.source_admission)
    if _COMPILED.get(value.compiled) is not value.compiled.programme:
        raise TypeError('saved reuse lost its live current C compiler relation')
    for cap in value.compiled.inputs:
        require_public_physical_input(cap)
    value.noise.assert_integrity()
    return value
