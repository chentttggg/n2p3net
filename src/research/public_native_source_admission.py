"""Original public source/cohort and native target-union consumption relation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from research.brainsync_source_admission import require_original_source
from research.channel_noise_mechanism import NoiseGeometry
from transfer.ensemble_views import RecordCalibrationPrefixCompiler, calibration_prefix_identity
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.native_source_response_model import NativeSourceResponseModel
from transfer.noise_model import array_digest, freeze_metadata, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_ADMISSIONS = WeakKeyDictionary()


def _owned_target(compiler, prefixes, calibration_inputs):
    from research.public_record_physical_bridge import require_public_physical_input

    if type(compiler) is not RecordCalibrationPrefixCompiler:
        raise TypeError('native source target requires its original public record compiler')
    prefixes = tuple(compiler.assert_original_prefix(prefix) for prefix in prefixes)
    if not prefixes or len({p.budget for p in prefixes}) != len(prefixes):
        raise ValueError('native source target requires its unique original requested C prefixes')
    inputs = tuple(require_public_physical_input(value) for value in calibration_inputs)
    owned = {(calibration_prefix_identity(p), raw.record_id): raw for p in prefixes for raw in p.calibration_records}
    supplied = {}
    for value in inputs:
        binding = value.public_binding
        key = binding['C_prefix_identity'], binding['record_id']
        if (binding['role'] != 'calibration' or key not in owned or key in supplied
                or value.samples is not owned[key].samples):
            raise ValueError('native source target requires every exact original C record adaptation once')
        supplied[key] = value
    if supplied.keys() != owned.keys():
        raise ValueError('native source target dropped an actual C record scope')
    first = inputs[0].public_binding
    geometry = NoiseGeometry(tuple(first['eeg_channels']), first['sample_rate_hz'], first['signal_unit'],
        first['reference'], first['operator'], len(first['eeg_channels']))
    target = prefixes[0].target_contract
    expected = (target['participant_key'], tuple(target['channel_names']), target['sample_rate_hz'],
                target['reference'], target['signal_unit'])
    if any((v.public_binding['participant_key'], tuple(v.public_binding['eeg_channels']),
            v.public_binding['sample_rate_hz'], v.public_binding['reference'], v.public_binding['signal_unit']) != expected
           or v.public_binding['operator'] != geometry.operator_identity for v in inputs):
        raise ValueError('native source target changed its original participant or native observation contract')
    held_ids = compiler.adapter.requested_future_record_ids
    if not held_ids:
        raise ValueError('native source development declares its complete original held record roster')
    held = tuple({'record_id': s.record_id, 'source_sha256': s.source_sha256,
        'selection_ids': tuple(selection.selection_id for selection in s.selections),
        'candidate_vocabulary': tuple(s.selections[0].decision.candidate_vocabulary),
        'native_sample_count': s.native_sample_count, 'session_id': s.session_id, 'run_id': s.run_id}
        for s in compiler.sources if s.record_id in held_ids)
    if {row['record_id'] for row in held} != set(held_ids):
        raise ValueError('native source target lost an original held acquisition')
    c_scopes = {f'C{p.budget}:{name}': scope for p in prefixes
                for name, scope in p.target_contract['record_scopes'].items()}
    c_ids = {name for p in prefixes for name in p.target_contract['record_scopes']}
    roster = {name: row for name, row in target['source_record_roster'].items() if name in c_ids or name in held_ids}
    contract = freeze_metadata({**target, 'record_scopes': c_scopes, 'source_record_roster': roster,
        'operator_identity': geometry.operator_identity,
        'native_source_target_role': 'original_public_C_and_complete_held_native_parameter_consumption',
        'C_prefixes': tuple({'requested_K': p.budget, 'identity': calibration_prefix_identity(p),
            'label_use': p.label_use,
            'records': tuple({'record_id': r.record_id, 'right_censored': r.right_censored,
                'input_sha256': array_digest(r.samples, r.onsets, r.target_mask)} for r in p.calibration_records)}
            for p in prefixes), 'held_record_roster': held,
        'requested_held_selections': sum(len(row['selection_ids']) for row in held),
        'public_C_adaptation_identities': tuple(v.identity for v in inputs), 'held_truth_consumed': False})
    return prefixes, inputs, geometry, contract


@dataclass(frozen=True, eq=False)
class PublicNativeSourceAdmission:
    original_source: object
    compiler: object
    prefixes: tuple
    calibration_inputs: tuple
    model: NativeSourceResponseModel
    record: dict


def admit_public_native_source(original_source, compiler, prefixes, calibration_inputs, *, terminal_values):
    original_source = require_original_source(original_source)
    prefixes, inputs, geometry, contract = _owned_target(compiler, prefixes, calibration_inputs)
    source = original_source.source
    source.assert_target(contract, exclusion_scope='target_participant')
    continuation = NativeResponseDictionary.from_source_matrix(source.parameters.basis,
        sample_rate_hz=source.parameters.noise.geometry.sfreq,
        source_basis_sha256=array_digest(source.parameters.basis), terminal_values=terminal_values)
    model = NativeSourceResponseModel(source, continuation, geometry, contract['participant_key'])
    record = freeze_metadata({'schema': 'n2p3_live_original_public_native_source_consumption/1',
        'source_admission_identity': metadata_digest(original_source.record),
        'original_source_entry': original_source.entry, 'original_source_record': original_source.record,
        'original_source_fit_stage_sha256': original_source.record['original_fit_stage_sha256'],
        'original_source_exclusion': original_source.entry['exclusion'],
        'new_target_exclusion_scope': 'target_participant', 'new_target_union': contract,
        'source_codec_sha256': source.digest, 'numeric_model_identity': metadata_digest(model.dependency_record()),
        'source_prior_preserved_sha256': array_digest(source.parameters.prior.mean, source.parameters.prior.root),
        'source_role': 'existing_single_participant_C68_development_prior',
        'population_prior_or_LOSO_claim': False, 'source_fits': 0, 'target_noise_fits': 0,
        'bridge_producer_sha256': IMPLEMENTATION_SHA256, 'old_fit_or_entry_resigned': False})
    admission = PublicNativeSourceAdmission(original_source, compiler, prefixes, inputs, model, record)
    _ADMISSIONS[admission] = record
    return admission


def require_public_native_source(admission):
    if type(admission) is not PublicNativeSourceAdmission or _ADMISSIONS.get(admission) is not admission.record:
        raise TypeError('native source use requires its live original-source/target-union admission')
    require_original_source(admission.original_source)
    _, _, _, target = _owned_target(admission.compiler, admission.prefixes, admission.calibration_inputs)
    if metadata_digest(target) != metadata_digest(admission.record['new_target_union']):
        raise ValueError('native source original target-union relation changed')
    admission.original_source.source.assert_target(target, exclusion_scope='target_participant')
    return admission
