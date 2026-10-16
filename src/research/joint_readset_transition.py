"""One named original-issuer/current-issuer relation for completed joint C/P.

The relation never grants samples. Both live issuers must already own the same
readset; a prepared reader transports that issued relation with its sealed input
capsule. A public parent independently reissues both sides from original data.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research.contracts import semantic_sha256
from research.frozen_metadata import freeze_metadata, plain_metadata

ORIGINAL_READER_SHA256 = '5e62ec75adb5884799968f8fdaa0a0e7f7125f6275a0280fb9f5577933753b26'
CURRENT_READER_SHA256 = 'b39ff1ce4df291290357c2fe0ae318fe90dc1936398964bf6ec3e9013c3df1cc'
SCHEMA = 'n2p3_joint_original_5e62_current_b39_readset/1'
_OWNERS = weakref.WeakKeyDictionary()


def _owner(compiler):
    reference = _OWNERS.get(compiler)
    return None if reference is None else reference()


def _retain_owner(compiler, owner):
    compiler._joint_readset_transition_owner = owner
    _OWNERS[compiler] = weakref.ref(owner)


def _content(receipt):
    path = Path(receipt['path'])
    content = path.read_bytes()
    if len(content) != receipt['bytes'] or hashlib.sha256(content).hexdigest() != receipt['sha256']:
        raise ValueError('original joint reader code differs from its declared bytes')
    if receipt['sha256'] != ORIGINAL_READER_SHA256:
        raise ValueError('joint readset transition requires its named original 5e62 reader')
    return content


def _original_module(receipt):
    content = _content(receipt)
    name = '_n2p3_original_joint_readset_'+ORIGINAL_READER_SHA256
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, receipt['path'])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        exec(compile(content, receipt['path'], 'exec'), module.__dict__)
    if (hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != ORIGINAL_READER_SHA256
            or module._SELECTION_EXECUTION_SOURCE_SHA256 != ORIGINAL_READER_SHA256):
        raise ValueError('original private joint reader changed after its actual import')
    return module


def _bindings(original, current):
    if set(original) != set(current):
        raise ValueError('joint readset transition changed its complete permission fields')
    for key in original:
        if key == 'source_owner_sha256':
            if original[key] != ORIGINAL_READER_SHA256 or current[key] != CURRENT_READER_SHA256:
                raise ValueError('joint readset transition changed its two named actual issuer identities')
        elif semantic_sha256(original[key]) != semantic_sha256(current[key]):
            raise ValueError('joint readset transition changed permission field '+key)


def _arrays(original, current):
    if original.dtype != current.dtype or original.shape != current.shape or not np.array_equal(original, current):
        raise ValueError('joint readset transition changed actual sample, clock or authorized C label arrays')


@dataclass(frozen=True, init=False)
class _IssuedJointReadsetTransition:
    owner: object
    capability: object
    original_capability: object
    record: object

    @property
    def digest(self):
        return semantic_sha256(self.record)

    def input_relation(self, capability, capture, current_input):
        self.owner.assert_issued(self, capability, capture)
        if semantic_sha256(current_input['history_capability']) != semantic_sha256(capability.binding):
            raise ValueError('joint transition consumer changed its actual current permission')
        original_input = {**current_input, 'history_capability': self.record['original_history_capability']}
        if (semantic_sha256(original_input) != capture.binding['prefix_input_sha256']
                or semantic_sha256(original_input) != semantic_sha256(capture.binding['input'])):
            raise ValueError('joint transition changed original numerical input, source, model, q axes or operands')
        if semantic_sha256(current_input) != self.record['current_numeric_input_sha256']:
            raise ValueError('joint transition changed its current complete numerical input relation')
        return original_input

    def confirm_capture(self, capability, original, recomputed):
        self.owner.assert_issued(self, capability, original)
        if recomputed.digest != original.digest:
            raise ValueError('joint transition did not preserve the complete original numerical capture')
        self.owner.score_admitted.add((capability.digest, original.digest))


class _JointReadsetOwner:
    def __init__(self, compiler, reader, *, prepared=False, transported_reader=None):
        from transfer import ensemble_views as current

        self.compiler, self.reader = compiler, freeze_metadata(reader)
        self.issued = {}
        self.score_admitted = set()
        self.transport_requested = set()
        self.prepared = prepared
        if prepared:
            if (type(compiler) is not current._PreparedRecordCalibrationPrefixCompiler
                    or compiler.producer_source_sha256 != CURRENT_READER_SHA256):
                raise TypeError('joint transition transport requires its original current-issuer input capsule')
            if (not isinstance(transported_reader, bytes) or len(transported_reader) != reader['bytes']
                    or hashlib.sha256(transported_reader).hexdigest() != ORIGINAL_READER_SHA256
                    or reader['sha256'] != ORIGINAL_READER_SHA256):
                raise ValueError('prepared joint transition changed its transported original reader source bytes')
            self.reader_content = transported_reader
            self.original_module = None
            self.original_compiler = None
        else:
            self.reader_content = _content(reader)
            self.original_module = _original_module(reader)
            if (type(compiler) is not current.RecordCalibrationPrefixCompiler or compiler.profiles
                    or compiler.history_owner_source != CURRENT_READER_SHA256
                    or hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest() != CURRENT_READER_SHA256):
                raise TypeError('joint transition requires the named actual current record issuer without NN profiles')
            ingress = compiler._history_ingress
            self.original_compiler = self.original_module.RecordCalibrationPrefixCompiler(compiler.adapter,
                budgets=compiler.budgets, profiles=(), authorization_preprocessing=compiler.authorization_preprocessing,
                response_seconds=ingress['response_seconds'], pre_roll_seconds=ingress['pre_roll_seconds'])
            if (semantic_sha256(self.original_compiler.target) != semantic_sha256(compiler.target)
                    or self.original_compiler.actual_budgets != compiler.actual_budgets):
                raise ValueError('original and current joint issuers changed actual target, source roster or split')

    def _validate_current(self, capability):
        from transfer.ensemble_views import AuthorizedRecordPublicHistory, _record_history_key
        from transfer.noise_model import array_digest

        if (type(capability) is not AuthorizedRecordPublicHistory
                or self.compiler._record_history_scopes.get(_record_history_key(capability.binding)) is not capability
                or capability.binding['source_owner_sha256'] != CURRENT_READER_SHA256
                or capability.binding['samples_sha256'] != array_digest(capability.samples)):
            raise ValueError('joint transition requires its real registered immutable current history capability')
        prefix = self.compiler.compile(self._prefix_k(capability))
        self.compiler.assert_original_prefix(prefix)
        if capability.binding['C_record_input_sha256'] != {
                raw.record_id: array_digest(raw.samples, raw.onsets, raw.target_mask) for raw in prefix.calibration_records}:
            raise ValueError('joint transition changed actual authorized C samples or labels')

    def _prefix_k(self, capability):
        from transfer.ensemble_views import calibration_prefix_identity

        matches = [p.budget for p in self.compiler._prefixes.values()
                   if calibration_prefix_identity(p) == capability.binding['prefix_identity']]
        if len(matches) != 1:
            raise ValueError('joint transition has no unique actual original C prefix')
        return matches[0]

    def issue(self, capability, capture):
        self._validate_current(capability)
        old_binding = capture.binding['input']['history_capability']
        _bindings(old_binding, capability.binding)
        key = capability.digest, capture.digest
        if key in self.issued and self.issued[key].capability is capability:
            relation = self.issued[key]
            self.assert_issued(relation, capability, capture)
            return relation
        if self.prepared:
            raise ValueError('prepared joint reader cannot invent an unexported original readset transition')
        old = self.original_module
        old_prefix = self.original_compiler.compile(self._prefix_k(capability))
        view = old.PredictionDecisionView.from_record(capability.binding['current_view'],
            samples=capability.samples[slice(*capability.binding['current_view']['raw_scope'])])
        # The original issuer validates the original decision/event set, native
        # clocks, C scope, and actual original read against the supplied view.
        original = self.original_compiler.record_read_scope(view, prefix=old_prefix,
            conditioning_stop=capability.binding['conditioning_sample_support'][1],
            available_stop=capability.binding['available_sample'], program=capability.binding['program'])
        if (type(original) is not old.AuthorizedRecordPublicHistory
                or self.original_compiler._record_history_scopes.get(old._record_history_key(original.binding)) is not original
                or semantic_sha256(original.binding) != semantic_sha256(old_binding)):
            raise ValueError('old joint issuer did not actually reissue the captured original permission')
        _arrays(original.samples, capability.samples)
        if len(original.calibration_records) != len(capability.calibration_records):
            raise ValueError('joint transition changed the actual C recording roster')
        for left, right in zip(original.calibration_records, capability.calibration_records, strict=True):
            if left.record_id != right.record_id or left.right_censored != right.right_censored:
                raise ValueError('joint transition changed original C acquisition identity')
            for attr in ('samples', 'onsets', 'target_mask'):
                _arrays(getattr(left, attr), getattr(right, attr))
        record = {'schema': SCHEMA, 'original_reader_sha256': ORIGINAL_READER_SHA256,
            'current_reader_sha256': CURRENT_READER_SHA256, 'original_reader': plain_metadata(self.reader),
            'original_history_capability': old_binding, 'current_history_capability': capability.binding,
            'original_capability_sha256': original.digest, 'current_capability_sha256': capability.digest,
            'capture_sha256': capture.digest, 'original_numeric_input_sha256': capture.binding['prefix_input_sha256'],
            'current_numeric_input_sha256': semantic_sha256({**capture.binding['input'],
                'history_capability': capability.binding}),
            'permitted_identity_relation': 'actual_5e62_and_b39_issuer_source_owner_only',
            'actual_arrays_equal': True, 'read_permission_granted_by_transition': False}
        return self._store(capability, original, record)

    def _store(self, capability, original, record):
        relation = object.__new__(_IssuedJointReadsetTransition)
        for key, value in {'owner': self, 'capability': capability,
                           'original_capability': original, 'record': freeze_metadata(record)}.items():
            object.__setattr__(relation, key, value)
        self.issued[capability.digest, record['capture_sha256']] = relation
        return relation

    def assert_issued(self, relation, capability, capture):
        if (type(relation) is not _IssuedJointReadsetTransition or relation.owner is not self
                or _owner(self.compiler) is not self or relation.capability is not capability
                or self.issued.get((capability.digest, capture.digest)) is not relation):
            raise TypeError('joint readset relation was not issued by its actual private input owner')
        self._validate_current(capability)
        _bindings(relation.record['original_history_capability'], capability.binding)
        if (relation.record['capture_sha256'] != capture.digest
                or relation.record['original_numeric_input_sha256'] != capture.binding['prefix_input_sha256']
                or semantic_sha256(capture.binding['input']['history_capability']) !=
                    relation.record['original_capability_sha256']):
            raise ValueError('joint readset relation changed its original capture or numerical input')


def admit_joint_readset_transition(compiler, reader):
    """Instantiate the verified archived code and its actual original issuer."""
    owner = _owner(compiler)
    if owner is None:
        owner = _JointReadsetOwner(compiler, reader)
        _retain_owner(compiler, owner)
    elif semantic_sha256(owner.reader) != semantic_sha256(reader):
        raise ValueError('one joint input owner cannot change its original reader dependency')
    return owner


def joint_readset_relation(compiler, capability, capture):
    if capture is None or semantic_sha256(capture.binding['input']['history_capability']) == capability.digest:
        return None
    owner = _owner(compiler)
    return None if owner is None else owner.issue(capability, capture)


def joint_readset_record(compiler, capability, capture):
    relation = joint_readset_relation(compiler, capability, capture)
    if relation is None:
        return None
    if (capability.digest, capture.digest) not in relation.owner.score_admitted:
        raise ValueError('joint readset record requires its completed numerical first-use verification')
    return plain_metadata(relation.record)


def bind_joint_readset_descriptor(descriptor, *, base_directory):
    if set(descriptor) != {'schema', 'original_reader'} or descriptor['schema'] != SCHEMA:
        raise ValueError('joint completed readset requires its exact named descriptor')
    reader = descriptor['original_reader']
    reader = {**reader, 'path': str((Path(base_directory)/reader['path']).resolve())}
    _content(reader)
    return {'schema': SCHEMA, 'original_reader': reader}


def check_joint_readset_record(record):
    if (record['schema'] != SCHEMA or record['original_reader_sha256'] != ORIGINAL_READER_SHA256
            or record['current_reader_sha256'] != CURRENT_READER_SHA256
            or record['permitted_identity_relation'] != 'actual_5e62_and_b39_issuer_source_owner_only'
            or record['actual_arrays_equal'] is not True or record['read_permission_granted_by_transition'] is not False
            or record['original_reader']['sha256'] != ORIGINAL_READER_SHA256):
        raise ValueError('joint readset record differs from its named permission relation')
    _bindings(record['original_history_capability'], record['current_history_capability'])
    if (semantic_sha256(record['original_history_capability']) != record['original_capability_sha256']
            or semantic_sha256(record['current_history_capability']) != record['current_capability_sha256']):
        raise ValueError('joint readset record changed one complete issuer permission digest')


def publish_joint_readset_records(directory, records):
    """Save the exact code dependency; saved records themselves grant nothing."""
    directory = Path(directory)
    published = []
    for record in records:
        check_joint_readset_record(record)
        item = plain_metadata(freeze_metadata(record))
        content = _content(item['original_reader'])
        relative = 'original_readers/'+ORIGINAL_READER_SHA256+'.py'
        path = directory/relative
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
        if path.read_bytes() != content:
            raise ValueError('joint readset dependency differs from its original issuer code')
        item['original_reader'] = {**item['original_reader'], 'path': relative}
        published.append(item)
    return published


def reissue_joint_readset_record(compiler, capability, capture, record, *, directory):
    """Live parent reissues both sides; prepared transport must already own it."""
    expected = plain_metadata(record)
    check_joint_readset_record(expected)
    reader = {**expected['original_reader'], 'path': str(Path(directory)/expected['original_reader']['path'])}
    if _owner(compiler) is None:
        admit_joint_readset_transition(compiler, reader)
    relation = joint_readset_relation(compiler, capability, capture)
    if relation is None:
        raise ValueError('saved original joint capture lost its named readset transition')
    actual = plain_metadata(relation.record)
    actual['original_reader'] = expected['original_reader']
    if semantic_sha256(actual) != semantic_sha256(expected):
        raise ValueError('saved joint readset relation differs from actual two-issuer reissue')
    relation.owner.transport_requested.add((capability.digest, capture.digest))
    return relation


def export_joint_readset_transitions(compiler, capabilities):
    owner = _owner(compiler)
    if owner is None:
        return None
    # The immutable capsule transports only already issued relations and source
    # code. The parent still performs a fresh actual two-issuer reissue.
    allowed = {cap.digest for cap in capabilities}
    return {'schema': SCHEMA, 'original_reader': plain_metadata(owner.reader),
        'original_reader_code': owner.reader_content.decode('utf-8'),
        'relations': [plain_metadata(relation.record) for relation in owner.issued.values()
                      if relation.record['current_capability_sha256'] in allowed
                      and ((relation.record['current_capability_sha256'], relation.record['capture_sha256'])
                           in owner.score_admitted or
                           (relation.record['current_capability_sha256'], relation.record['capture_sha256'])
                           in owner.transport_requested)]}


def _restore_joint_readset_transitions(compiler, document):
    """Called only after the original capsule request/blob have been admitted."""
    from transfer.ensemble_views import _record_history_key

    if document['schema'] != SCHEMA:
        raise ValueError('prepared joint relation changed its named schema')
    receipt = document['original_reader']
    owner = _JointReadsetOwner(compiler, receipt, prepared=True,
        transported_reader=document['original_reader_code'].encode('utf-8'))
    _retain_owner(compiler, owner)
    for record in document['relations']:
        check_joint_readset_record(record)
        original, current = record['original_history_capability'], record['current_history_capability']
        _bindings(original, current)
        capability = compiler._record_history_scopes.get(_record_history_key(current))
        if (capability is None or semantic_sha256(capability.binding) != semantic_sha256(current)
                or semantic_sha256(original) != record['original_capability_sha256']
                or capability.digest != record['current_capability_sha256']
                or record['original_reader_sha256'] != ORIGINAL_READER_SHA256
                or record['current_reader_sha256'] != CURRENT_READER_SHA256
                or record['read_permission_granted_by_transition'] is not False):
            raise ValueError('prepared joint relation differs from its actual transported current permission')
        owner._validate_current(capability)
        owner._store(capability, None, record)
