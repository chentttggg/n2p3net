"""One actual C/held programme and its original EEG-byte union exclusion.

The canonical C issuer owns labels; the source-admission owner keeps the held
target grant and geometry. This module joins their live relations explicitly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from data.brainsync_raw_authority import (
    allowed_candidates,
    open_brainsync_calibration,
    require_calibration_labels,
    require_raw_capability,
)
from research import brainsync_source_admission as source_admission
from research.channel_noise_mechanism import NoiseGeometry
from research.frozen_metadata import freeze_metadata, plain_metadata
from transfer.noise_model import metadata_digest

_PROGRAMMES, _TRANSPORTS = WeakKeyDictionary(), WeakKeyDictionary()


@dataclass(frozen=True, eq=False)
class BrainSyncProgrammeAdmission:
    calibration_caps: tuple
    calibration_labels: object
    held_target: object
    union_contract: dict
    record: dict
    identity: str


def _require_independent_eeg_bytes(calibration_sha256s, held_sha256s):
    if set(calibration_sha256s) & set(held_sha256s):
        raise ValueError('independent C/held recordings cannot consume the same original EEG bytes twice')


def open_actual_brainsync_programme(calibration_path, *, scope, calibration_requests,
        label_count, ready_at_timestamp_s, inference_path, inference_requests):
    """Issue C and held from their original parents under the same scope.

    Only actual EEG reader filenames enter the exclusion roster. Held caps stay
    exclusively in ``held_target.caps``; C caps/labels have separate fields.
    """
    held = source_admission.open_actual_brainsync_target(inference_path,
        scope=scope, requests=inference_requests)
    issuer = None
    try:
        issuer = open_brainsync_calibration(calibration_path, scope=scope)
        caps = tuple(issuer.issue_raw(**request) for request in calibration_requests)
        if not caps:
            raise ValueError('the C/held programme requires its actual C raw scope')
        labels = issuer.issue_labels(label_count, ready_at_timestamp_s=ready_at_timestamp_s)
        first = caps[0].public_binding
        held_first = held.caps[0].public_binding
        geometry = NoiseGeometry(tuple(first['eeg_channels']), first['sample_rate_hz'],
            first['signal_unit'], first['reference'], first['operator'], len(first['eeg_channels']))
        if (first['participant_id'] != held_first['participant_id']
                or metadata_digest(geometry.record()) != metadata_digest(held.geometry.record())
                or first['independent_recording_scope_sha256'] != held_first['independent_recording_scope_sha256']
                or any(cap.public_binding['record_id'] != first['record_id'] for cap in caps)):
            raise ValueError('C/held requires one actual participant, geometry and recording scope')
        allowed_candidates(caps, labels)
        receipt = issuer.private_admission_receipt()
        actual_paths = tuple(str(Path(name).resolve()) for name in issuer._raw.filenames)
        actual_files = {name: receipt['original_files'][name]['sha256'] for name in actual_paths}
        if not actual_files:
            raise ValueError('actual C EEG reader has no original admitted EEG byte source')
        _require_independent_eeg_bytes(actual_files.values(),
            (row['source_sha256'] for row in held.contract['source_record_roster'].values()))
        contract = plain_metadata(held.contract)
        primary = actual_files[actual_paths[0]]
        c_roster = {f"{first['record_id']}:actual-eeg-file:{index}": {
            'source_sha256': actual_files[name], 'source_dataset_id': 'BrainSync-GTN',
            'participant_key': contract['participant_key']}
            for index, name in enumerate(actual_paths)}
        if set(c_roster) & set(contract['source_record_roster']):
            raise ValueError('C and held must retain different actual recording identities')
        held_roster_keys = tuple(contract['source_record_roster'])
        contract['source_record_roster'].update(c_roster)
        contract['record_scopes'].update({cap.public_binding['record_segment_id']: {
            'source_sha256': primary,
            'read_sample_support': [cap.public_binding['sample_start'], cap.public_binding['full_stop']],
            'prefix_stop': cap.public_binding['prefix_stop'],
            'available_stop': cap.public_binding['available_stop']}
            for cap in caps})
        contract.update({'schema': 'n2p3_brainsync_C_held_transport_input_union/1',
            'held_target_identity': held.identity,
            'calibration_record_ids': (first['record_id'],),
            'held_record_ids': (held_first['record_id'],),
            'calibration_source_roster_keys': tuple(c_roster),
            'held_source_roster_keys': held_roster_keys,
            'canonical_capability_identities': tuple(cap.identity_sha256 for cap in held.caps + caps),
            'authority_kind': 'actual_canonical_C_and_held_issuers_original_EEG_byte_union'})
        contract = freeze_metadata(contract)
        record = freeze_metadata({'schema': 'n2p3_actual_brainsync_C_held_programme_admission/1',
            'union_contract_sha256': metadata_digest(contract),
            'held_target_identity': held.identity,
            'recording_scope': scope.record(),
            'calibration_raw_capability_identities': tuple(cap.identity_sha256 for cap in caps),
            'calibration_label_capability_identity': labels.identity_sha256,
            'authorized_C_label_count': len(labels.labels),
            'held_raw_capability_identities': tuple(cap.identity_sha256 for cap in held.caps),
            'source_fits': 0, 'target_fits': 0,
            'held_labels_read': 0,
            'programme_source_sha256': source_admission.sha(__file__)})
        programme = BrainSyncProgrammeAdmission(caps, labels, held, contract, record, metadata_digest(record))
        _PROGRAMMES[programme] = (contract, issuer, freeze_metadata(receipt), caps, labels, held)
        return programme
    except BaseException:
        if issuer is not None:
            issuer.close()
        source_admission.close_actual_target(held)
        raise


def require_actual_programme(programme):
    registered = _PROGRAMMES.get(programme) if type(programme) is BrainSyncProgrammeAdmission else None
    if (registered is None or registered[0] is not programme.union_contract
            or registered[3] is not programme.calibration_caps
            or registered[4] is not programme.calibration_labels
            or registered[5] is not programme.held_target):
        raise TypeError('C/held requires its original live canonical programme admission')
    for cap in programme.calibration_caps:
        require_raw_capability(cap)
    require_calibration_labels(programme.calibration_labels)
    source_admission.require_actual_target(programme.held_target)
    return programme


def close_actual_programme(programme):
    """Close both EEG readers; original immutable capabilities remain valid."""
    programme = require_actual_programme(programme)
    try:
        _PROGRAMMES[programme][1].close()
    finally:
        source_admission.close_actual_target(programme.held_target)


@dataclass(frozen=True, eq=False)
class ProgrammeTransportAdmission:
    programme: object
    held_transport: object
    record: dict

    @property
    def model(self):
        return self.held_transport.model

    @property
    def graph(self):
        return self.held_transport.graph


def admit_programme_transported_source(original_source, programme, *, exclusion_scope, template):
    original_source = source_admission.require_original_source(original_source)
    programme = require_actual_programme(programme)
    original_source.source.assert_target_exclusion(programme.union_contract, scope=exclusion_scope)
    held = source_admission.admit_transported_source(original_source, programme.held_target,
        exclusion_scope=exclusion_scope, template=template)
    record = freeze_metadata({'schema': 'n2p3_source_to_actual_C_held_transport_programme/1',
        'programme_identity': programme.identity,
        'union_contract_sha256': metadata_digest(programme.union_contract),
        'union_exclusion_scope': exclusion_scope,
        'original_source_admission_sha256': metadata_digest(original_source.record),
        'source_codec_sha256': original_source.source.digest,
        'held_transport_record_sha256': metadata_digest(held.record),
        'held_target_identity': programme.held_target.identity,
        'actual_graph_identity': held.graph.digest(),
        'source_prior_occurrences': 1,
        'source_fits': 0, 'target_fits': 0,
        'programme_source_sha256': source_admission.sha(__file__)})
    result = ProgrammeTransportAdmission(programme, held, record)
    _TRANSPORTS[result] = (programme, held, record)
    return result


def require_programme_transport(admission):
    registered = _TRANSPORTS.get(admission) if type(admission) is ProgrammeTransportAdmission else None
    if (registered is None or registered[0] is not admission.programme
            or registered[1] is not admission.held_transport or registered[2] is not admission.record):
        raise TypeError('C/held transport requires its original live source/programme relation')
    require_actual_programme(admission.programme)
    source_admission.require_transported_admission(admission.held_transport)
    return admission


def save_programme_transport_binding(admission, path, *, core_binding_path):
    admission = require_programme_transport(admission)
    core_sha = source_admission.save_transport_binding(admission.held_transport, core_binding_path)
    core_ref = source_admission.FrozenArtifactReference(Path(core_binding_path), core_sha)
    document = {'schema': 'n2p3_compact_frozen_C_held_transport_binding/1',
        'record': plain_metadata(admission.record),
        'programme_record': plain_metadata(admission.programme.record),
        'union_contract': plain_metadata(admission.programme.union_contract),
        'core_numeric_binding': core_ref.record(),
        'raw_label_source_or_live_programme_authority_granted': False}
    Path(path).write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    return source_admission.sha(path)


def cold_numeric_programme_transport(path, *, expected_sha256):
    reference = source_admission.FrozenArtifactReference(Path(path), expected_sha256)
    document = json.loads(reference.read_bytes())
    core = document['core_numeric_binding']
    core_reference = source_admission.FrozenArtifactReference(Path(core['path']), core['sha256'])
    core_record = json.loads(core_reference.read_bytes())['record']
    model = source_admission.cold_numeric_transport(core['path'], expected_sha256=core['sha256'])
    record, programme, union = document['record'], document['programme_record'], document['union_contract']
    model.source.assert_target_exclusion(union, scope=record['union_exclusion_scope'])
    if (metadata_digest(programme) != record['programme_identity']
            or metadata_digest(union) != programme['union_contract_sha256']
            or metadata_digest(union) != record['union_contract_sha256']
            or union['held_target_identity'] != record['held_target_identity']
            or metadata_digest(core_record) != record['held_transport_record_sha256']
            or core_record['new_target_identity'] != record['held_target_identity']
            or core_record['original_source_admission_sha256'] != record['original_source_admission_sha256']
            or core_record['source_codec_sha256'] != record['source_codec_sha256']
            or any(union['source_record_roster'][key] != row
                for key, row in core_record['new_target_contract']['source_record_roster'].items())
            or any(union['record_scopes'][key] != row
                for key, row in core_record['new_target_contract']['record_scopes'].items())
            or model.target_participant_key != union['participant_key']
            or model.graph().digest() != record['actual_graph_identity']):
        raise ValueError('cold numeric graph differs from the saved C/held programme relation')
    return model
