"""Live original public C68 authority plus a separate transported target grant.

Old fit/export identities remain native and unchanged. Canonical BrainSync
issuer capabilities supply the new participant, EEG bytes and target geometry.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from data.brainsync_raw_authority import (
    open_brainsync_recording,
    require_raw_capability,
)
from data.epochs import PreprocessingSpec
from data.huebner2017_candidate import read_local_marker_member
from data.identity import DatasetIdentityTable
from research.channel_noise_mechanism import NoiseGeometry
from research.contracts import semantic_sha256
from research.frozen_metadata import freeze_metadata, plain_metadata
from research.joint_readset_transition import admit_joint_readset_transition
from research.joint_source_catalog import (
    JointFlatSourceCatalog,
    _entry_identity,
    _verify_original_cohort,
    admit_huebner_joint_source_data,
    huebner_original_source_metadata,
)
from transfer.continuous_decision import ContinuousDecisionModel
from transfer.ensemble_views import CalibrationPrefixCompiler, calibration_prefix_identity
from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation
from transfer.noise_model import array_digest, metadata_digest, require_sha256
from transfer.transport_graph_model import FrozenTransportGraphModel

_TARGETS, _SOURCES, _TRANSPORTS = WeakKeyDictionary(), WeakKeyDictionary(), WeakKeyDictionary()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class FrozenArtifactReference:
    """An explicit caller-selected artifact; its owner still checks semantics."""

    path: Path
    expected_sha256: str

    def __post_init__(self):
        path = Path(self.path)
        if not path.is_absolute():
            raise ValueError('artifact references require explicit absolute paths')
        require_sha256(self.expected_sha256, 'explicit source-admission artifact')
        object.__setattr__(self, 'path', path)

    def read_bytes(self):
        content = self.path.read_bytes()
        if hashlib.sha256(content).hexdigest() != self.expected_sha256:
            raise ValueError('configured artifact differs from its external expected bytes')
        return content

    def record(self):
        return {'path': str(self.path), 'sha256': self.expected_sha256}


@dataclass(frozen=True)
class HuebnerC68SourceConfiguration:
    """Named single-record C68 bridge, including its original marker storage.

    The registered Huebner metadata owner retains its own trusted roster. These
    references select the preserved fit entry, native target and old-reader
    relation; they do not replace that data authority.
    """

    original_entry: FrozenArtifactReference
    original_native_target: FrozenArtifactReference
    original_reader: FrozenArtifactReference
    original_fit_declaration: FrozenArtifactReference
    catalog_base_directory: Path
    source_requests: tuple
    authorization_preprocessing: PreprocessingSpec
    response_seconds: float
    pre_roll_seconds: float

    def __post_init__(self):
        base = Path(self.catalog_base_directory)
        if not base.is_absolute():
            raise ValueError('catalog base directory requires an explicit absolute path')
        object.__setattr__(self, 'catalog_base_directory', base)
        requests = freeze_metadata(tuple(self.source_requests))
        if (len(requests) != 1 or requests[0]['label_budget'] != 68
                or len(requests[0]['record_selectors']) != 1):
            raise ValueError('this named Huebner C68 bridge accepts one original C68 record programme')
        object.__setattr__(self, 'source_requests', requests)

    def record(self):
        return {'schema': 'n2p3_huebner_C68_source_configuration/1',
            'original_entry': self.original_entry.record(),
            'original_native_target': self.original_native_target.record(),
            'original_reader': self.original_reader.record(),
            'original_fit_declaration': self.original_fit_declaration.record(),
            'catalog_base_directory': str(self.catalog_base_directory),
            'source_requests': plain_metadata(self.source_requests),
            'authorization_preprocessing': self.authorization_preprocessing.record(),
            'response_seconds': self.response_seconds, 'pre_roll_seconds': self.pre_roll_seconds}


@dataclass(frozen=True, eq=False)
class BrainSyncTargetAdmission:
    caps: tuple
    contract: dict
    geometry: object
    identity_table: object
    identity: str


def open_actual_brainsync_target(path, *, scope, requests):
    issuer = open_brainsync_recording(path, scope=scope)
    try:
        caps = tuple(issuer.issue_raw(**request) for request in requests)
        first = caps[0].public_binding
        if any((cap.public_binding['participant_id'], cap.public_binding['record_id']) !=
                (first['participant_id'], first['record_id']) for cap in caps):
            raise ValueError('one target admission requires its actual same-record participant and raw programme')
        table = DatasetIdentityTable.from_source_rows([first['participant_id']], ['BrainSync-GTN'])
        participant = table.record_for(first['participant_id']).authority_key('source_or_global')
        receipt = issuer.private_admission_receipt()
        actual_paths = tuple(str(Path(name).resolve()) for name in issuer._raw.filenames)
        actual_files = {name: receipt['original_files'][name]['sha256'] for name in actual_paths}
        if not actual_files:
            raise ValueError('actual target EEG reader has no original admitted EEG byte source')
        primary = actual_paths[0]
        geometry = NoiseGeometry(tuple(first['eeg_channels']), first['sample_rate_hz'], first['signal_unit'],
            first['reference'], first['operator'], len(first['eeg_channels']))
        roster = {f"{first['record_id']}:actual-eeg-file:{i}": {'source_sha256': actual_files[name],
            'source_dataset_id': 'BrainSync-GTN', 'participant_key': participant}
            for i, name in enumerate(actual_paths)}
        contract = freeze_metadata({'schema': 'n2p3_brainsync_transport_target_input/1',
            'participant_key': participant, 'source_dataset_id': 'BrainSync-GTN',
            'source_subject_id': first['participant_id'], 'source_sha256': actual_files[primary],
            'source_record_roster': roster,
            'record_scopes': {cap.public_binding['record_segment_id']: {'source_sha256': actual_files[primary],
                'read_sample_support': [cap.public_binding['sample_start'], cap.public_binding['full_stop']],
                'prefix_stop': cap.public_binding['prefix_stop'], 'available_stop': cap.public_binding['available_stop']}
                for cap in caps},
            'channel_names': list(geometry.channels), 'channel_types': ['EEG'] * len(geometry.channels),
            'sample_rate_hz': geometry.sfreq, 'reference': geometry.reference, 'signal_unit': geometry.unit,
            'operator_identity': geometry.operator_identity, 'candidate_vocabulary': caps[0].candidate_vocabulary,
            'target_lock_group': first['target_lock_group'], 'identity_table': table.payload(),
            'canonical_capability_identities': [cap.identity_sha256 for cap in caps],
            'authority_kind': 'actual_canonical_BrainSync_issuer_with_private_EEG_byte_receipt'})
        identity = metadata_digest(contract)
        result = BrainSyncTargetAdmission(caps, contract, geometry, table, identity)
        _TARGETS[result] = (contract, issuer, freeze_metadata(receipt))
        return result
    except BaseException:
        issuer.close()
        raise


def require_actual_target(target):
    registered = _TARGETS.get(target) if type(target) is BrainSyncTargetAdmission else None
    if registered is None or registered[0] is not target.contract:
        raise TypeError('transport target requires its original live canonical issuer admission')
    for cap in target.caps:
        require_raw_capability(cap)
    return target


def close_actual_target(target):
    """Close the parent EEG reader; issued immutable caps/token remain valid."""
    require_actual_target(target)
    _TARGETS[target][1].close()


def _difference_paths(before, after, stem=()):
    if isinstance(before, dict) and isinstance(after, dict):
        paths = []
        for key in sorted(set(before) | set(after)):
            paths.extend(_difference_paths(before.get(key), after.get(key), (*stem, key)))
        return paths
    return [] if before == after else [list(stem)]


def verify_huebner_c68_marker_storage_transition(compiler, prefix, source, *, anchored_declaration,
                                               configuration):
    """Named original/current issuers over the SAME already authorized source C."""
    reader_bytes = configuration.original_reader.read_bytes()
    reader = {**configuration.original_reader.record(), 'bytes': len(reader_bytes)}
    owner = admit_joint_readset_transition(compiler, reader)
    old = owner.original_module
    anchored_old_prefix = owner.original_compiler.compile(prefix.budget)
    anchored_old_prefix = owner.original_compiler.assert_original_prefix(anchored_old_prefix)
    current_prefix = compiler.assert_original_prefix(prefix)
    fitted_prefix = source.parameters.binding['source_prefixes'][0]
    expected = fitted_prefix['prefix_digest']
    declaration_bytes = configuration.original_fit_declaration.read_bytes()
    declaration = json.loads(declaration_bytes)
    selector = configuration.source_requests[0]['record_selectors'][0]
    selected = [record for record in declaration['record_sources']
        if all(record[key] == selector[key] for key in ('block', 'run'))]
    if len(selected) != 1:
        raise ValueError('configured source selector does not name one original fitted Huebner record')
    declaration['record_sources'] = selected
    declaration.pop('requested_future_record_ids', None)
    old_record, new_record = declaration['record_sources'][0], anchored_declaration['record_sources'][0]
    if {k: v for k, v in old_record.items() if k != 'marker'} != {k: v for k, v in new_record.items() if k != 'marker'}:
        raise ValueError('source declaration transition changed a non-marker original acquisition input')
    old_marker, new_marker = old_record['marker'], new_record['marker']
    original_marker_bytes = read_local_marker_member(old_marker['archive_path'], old_marker['member'])
    extracted_marker_bytes = Path(new_marker['path']).read_bytes()
    if (original_marker_bytes != extracted_marker_bytes or sha(new_marker['path']) != new_marker['sha256']
            or len(extracted_marker_bytes) != new_marker['bytes']):
        raise ValueError('old ZIP and current extracted original VMRK marker bytes differ')
    header = new_marker['source_header']
    if sha(header['path']) != header['sha256']:
        raise ValueError('current independently linked original source header changed')
    fit_compiler = old.CalibrationPrefixCompiler.from_dataset_source(declaration, budgets=compiler.budgets,
        profiles=(), authorization_preprocessing=compiler.authorization_preprocessing,
        response_seconds=compiler._history_ingress['response_seconds'],
        pre_roll_seconds=compiler._history_ingress['pre_roll_seconds'])
    original_prefix = fit_compiler.assert_original_prefix(fit_compiler.compile(prefix.budget))
    original_digest = old.calibration_prefix_identity(original_prefix)
    anchored_old_digest, current_digest = old.calibration_prefix_identity(anchored_old_prefix), calibration_prefix_identity(current_prefix)
    if (original_digest != expected or semantic_sha256(original_prefix.target_contract) != semantic_sha256(fitted_prefix['target_contract'])
            or semantic_sha256(original_prefix.label_use['raw']) != semantic_sha256(fitted_prefix['label_use'])):
        raise ValueError('actual original declaration/reader did not reproduce the saved source-fit prefix and permissions')
    if anchored_old_digest != current_digest or semantic_sha256(anchored_old_prefix.record()) != semantic_sha256(current_prefix.record()):
        raise ValueError('actual original/current source C permission fields changed')
    differences = _difference_paths(plain_metadata(original_prefix.record()), plain_metadata(current_prefix.record()))
    if differences != [['target_contract', 'source_sha256']]:
        raise ValueError('source marker storage transition changed additional complete C permission fields')
    rows = []
    for before, after in zip(original_prefix.calibration_records, current_prefix.calibration_records, strict=True):
        if before.record_id != after.record_id or before.right_censored != after.right_censored:
            raise ValueError('actual source C recording identity or censor status changed')
        for key in ('samples', 'onsets', 'target_mask'):
            a, b = getattr(before, key), getattr(after, key)
            if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
                raise ValueError('actual source C samples/onsets/labels changed across the named readers')
        rows.append({'record_id': after.record_id, 'samples': list(after.samples.shape),
            'input_sha256': array_digest(after.samples, after.onsets, after.target_mask),
            'labels': len(after.onsets), 'targets': int(after.target_mask.sum())})
    record = freeze_metadata({'schema': 'n2p3_original_5e62_current_b39_source_C_prefix_relation/1',
        'original_reader': reader, 'current_reader': {'path': str(Path(__import__('transfer.ensemble_views', fromlist=['__file__']).__file__)),
            'sha256': compiler.history_owner_source},
        'original_prefix_digest': original_digest, 'current_prefix_digest': current_digest,
        'anchored_5e62_prefix_digest': anchored_old_digest,
        'saved_fitted_source_prefix_digest': expected, 'original_source_fit_prefix_reproduced': True,
        'anchored_5e62_and_current_b39_complete_prefix_equal': True,
        'fit_vs_current_named_complete_permission_differences': differences,
        'fit_vs_current_raw_label_permission_equal': True,
        'original_source_declaration': configuration.original_fit_declaration.record(),
        'marker_storage_relation': {'original_zip': old_marker, 'current_extracted_marker': new_marker,
            'actual_marker_bytes_equal': True, 'marker_sha256': hashlib.sha256(original_marker_bytes).hexdigest(),
            'marker_bytes': len(original_marker_bytes), 'new_original_header_link_verified': True},
        'original_fit_adapter_sha256': fit_compiler.adapter.digest,
        'current_anchored_adapter_sha256': compiler.adapter.digest,
        'actual_original_and_current_prefixes_owned_by_respective_live_issuers': True,
        'actual_C_arrays': rows, 'read_permission_granted_by_relation': False,
        'source_fits': 0, 'new_target_labels_read': 0})
    return (owner, fit_compiler), original_prefix, record


@dataclass(frozen=True, eq=False)
class OriginalPublicSourceAdmission:
    source: object
    live_data_admission: object
    entry: dict
    cohort: tuple
    transition_owner: object
    original_prefix: object
    record: dict


def admit_huebner_c68_original_source(configuration):
    entry_content = configuration.original_entry.read_bytes()
    entry = json.loads(entry_content)
    old_target = json.loads(configuration.original_native_target.read_bytes())['target_contract']
    preparation = FrozenJointFlatSourcePreparation()
    initial = JointFlatSourceCatalog([entry], base_directory=configuration.catalog_base_directory,
        evidence_scope='software_correctness', _source_preparation=preparation)
    source = initial.resolve(entry['source_id'], target_contract=old_target,
        exclusion_scope=entry['exclusion']['scope']).source
    declarations, anchor = huebner_original_source_metadata(configuration.source_requests)
    preprocessing = configuration.authorization_preprocessing
    cohort = []
    for declaration, budget in declarations:
        compiler = CalibrationPrefixCompiler.from_dataset_source(declaration, budgets=(budget,), profiles=(),
            authorization_preprocessing=preprocessing, response_seconds=configuration.response_seconds,
            pre_roll_seconds=configuration.pre_roll_seconds)
        cohort.append((compiler, compiler.compile(budget)))
    cohort = tuple(cohort)
    model = ContinuousDecisionModel.load(io.BytesIO(source.original_model_content))
    summary = _verify_original_cohort(source, model, cohort, entry['exclusion']['target_participant_key'])
    transition_owner, original_prefix, transition = verify_huebner_c68_marker_storage_transition(
        cohort[0][0], cohort[0][1], source, anchored_declaration=declarations[0][0], configuration=configuration)
    live = admit_huebner_joint_source_data(entry, target_contract=old_target,
        exclusion_scope=entry['exclusion']['scope'], source_requests=configuration.source_requests,
        authorization_preprocessing=preprocessing, response_seconds=configuration.response_seconds,
        pre_roll_seconds=configuration.pre_roll_seconds,
        base_directory=configuration.catalog_base_directory, _source_preparation=preparation)
    public = JointFlatSourceCatalog([entry], base_directory=configuration.catalog_base_directory, evidence_scope='public_calibration_development',
        data_admissions=(live,), _source_preparation=preparation)
    resolved = public.resolve(entry['source_id'], target_contract=old_target, exclusion_scope=entry['exclusion']['scope'])
    if resolved.source is not source:
        raise ValueError('original public source admission changed the exact decoded source parameter owner')
    record = freeze_metadata({'schema': 'n2p3_original_Huebner_source_live_bridge/1',
        'explicit_configuration': configuration.record(),
        'explicit_configuration_sha256': metadata_digest(configuration.record()),
        'entry_sha256': hashlib.sha256(entry_content).hexdigest(), 'entry_semantic_identity': _entry_identity(entry),
        'original_fit_stage_sha256': semantic_sha256(entry['fit_stage']),
        'original_exclusion': entry['exclusion'], 'source_codec_sha256': source.digest,
        'original_model_sha256': entry['original_model']['sha256'],
        'source_parameter_authority_sha256': source.parameters.digest,
        'source_summary_sha256': semantic_sha256(summary), 'source_records': summary['source_records'],
        'original_data_anchor': anchor, 'live_public_data_admission': live.record,
        'source_C_prefix_transition': transition,
        'source_programme_evidence_scope': entry['source_evidence_scope'],
        'new_data_admission_scope': live.record['data_admission_scope'],
        'new_consumer_scope': resolved.receipt['consumer_evidence_scope'],
        'source_fits': 0, 'target_fits': 0, 'new_target_labels_read': 0})
    result = OriginalPublicSourceAdmission(source, live, freeze_metadata(entry), cohort,
        transition_owner, original_prefix, record)
    _SOURCES[result] = record
    return result


def require_original_source(admission):
    if type(admission) is not OriginalPublicSourceAdmission or _SOURCES.get(admission) is not admission.record:
        raise TypeError('transported public source requires actual live original-data/cohort admission')
    return admission


@dataclass(frozen=True, eq=False)
class TransportedSourceAdmission:
    original_source: object
    actual_target: object
    model: object
    graph: object
    record: dict


def admit_transported_source(original_source, actual_target, *, exclusion_scope, template):
    original_source, actual_target = require_original_source(original_source), require_actual_target(actual_target)
    source = original_source.source
    source.assert_target_exclusion(actual_target.contract, scope=exclusion_scope)
    original_template = template
    template = FrozenTransportGraphModel.load(original_template.path, expected_sha256=original_template.expected_sha256)
    if template.source.digest != source.digest:
        raise ValueError('B09 transport template belongs to another original fitted source')
    model = FrozenTransportGraphModel(source, template.spatial_transport, template.completion_prior,
        template.continuation, actual_target.geometry, actual_target.contract['participant_key'])
    graph = model.graph()
    if (graph.binding['source_codec_sha256'] != source.digest
            or graph.binding['target_participant_key'] != actual_target.contract['participant_key']
            or metadata_digest(graph.binding['target_observation_geometry']) != metadata_digest(actual_target.geometry.record())):
        raise ValueError('actual source/target input and B09/exact graph relation disagree')
    record = freeze_metadata({'schema': 'n2p3_original_public_source_to_actual_transport_target/1',
        'original_source_admission_sha256': metadata_digest(original_source.record),
        'original_entry_identity': original_source.record['entry_semantic_identity'],
        'original_fit_stage_sha256': original_source.record['original_fit_stage_sha256'],
        'original_excluded_target': original_source.entry['exclusion'],
        'new_target_contract': actual_target.contract, 'new_target_identity': actual_target.identity,
        'new_target_exclusion_scope': exclusion_scope, 'source_codec_sha256': source.digest,
        'template': original_template.record(),
        'B09_transport_sha256': metadata_digest(model.spatial_transport.record()),
        'completion_prior_sha256': model.completion_prior.record()['artifact_sha256'],
        'source_continuation_sha256': model.continuation.digest(), 'actual_graph_identity': graph.digest(),
        'actual_target_geometry': actual_target.geometry.record(),
        'source_prior_preserved_sha256': array_digest(source.parameters.prior.mean, source.parameters.prior.root),
        'source_fits': 0, 'target_fits': 0, 'old_entry_or_fit_stage_resigned': False,
        'bridge_source_sha256': sha(__file__)})
    result = TransportedSourceAdmission(original_source, actual_target, model, graph, record)
    _TRANSPORTS[result] = record
    return result


def require_transported_admission(admission):
    if type(admission) is not TransportedSourceAdmission or _TRANSPORTS.get(admission) is not admission.record:
        raise TypeError('scientific source transport requires the original live source/target relation')
    require_original_source(admission.original_source)
    require_actual_target(admission.actual_target)
    return admission


def save_transport_binding(admission, path):
    admission = require_transported_admission(admission)
    payload = {'schema': 'n2p3_compact_frozen_transport_target_binding/1',
        'record': plain_metadata(admission.record), 'numeric_model_authority': False,
        'raw_or_label_authority_granted': False}
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return sha(path)


def cold_numeric_transport(path, *, expected_sha256):
    if sha(path) != expected_sha256:
        raise ValueError('compact transport target binding differs from external bytes')
    document = json.loads(Path(path).read_bytes())
    record = document['record']
    template = FrozenTransportGraphModel.load(record['template']['path'], expected_sha256=record['template']['sha256'])
    target = record['new_target_contract']
    template.source.assert_target_exclusion(target, scope=record['new_target_exclusion_scope'])
    model = FrozenTransportGraphModel(template.source, template.spatial_transport, template.completion_prior,
        template.continuation, NoiseGeometry(**record['actual_target_geometry']), target['participant_key'])
    if (model.graph().digest() != record['actual_graph_identity']
            or template.source.digest != record['source_codec_sha256']
            or metadata_digest(target) != record['new_target_identity']):
        raise ValueError('cold transported graph differs from its original numeric source/target relation')
    return model
