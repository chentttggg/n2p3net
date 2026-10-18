"""Original public C/history grants adapted to native-clock physical inputs.

Every live require returns to the original compiler/registry. Numeric charts
and cold descriptors carry no raw or label granting authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from data.huebner2017_candidate import BLANK_POSITIONS, HuebnerNativeSource
from research.contracts import semantic_sha256
from research.frozen_metadata import freeze_metadata, plain_metadata
from research.history_authority import HistoryUnavailableError
from transfer.ensemble_views import (
    AuthorizedRecordPublicHistory,
    RecordCalibrationPrefixCompiler,
    _lookup_record_history_refusal,
    _prepared_record_read_scope,
    _record_history_key,
    calibration_prefix_identity,
)
from transfer.noise_model import array_digest, frozen_array, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_INPUTS, _UNAVAILABLE = WeakKeyDictionary(), WeakKeyDictionary()
NATIVE_CLOCK = 'native_sample_index'


class PublicPhysicalWitnessUnavailableError(ValueError):
    pass


def _owned_prefix(compiler, prefix):
    if type(compiler) is not RecordCalibrationPrefixCompiler:
        raise TypeError('this live bridge requires the original registered record compiler; prepared physical witnesses have another admission')
    prefix = compiler.assert_original_prefix(prefix)
    if compiler.adapter.digest != compiler._history_ingress['original_adapter_digest']:
        raise ValueError('original public record clock, membership or metadata changed after compiler admission')
    return prefix


def _source(compiler, record_id):
    matches = [source for source in compiler.sources if source.record_id == record_id]
    if len(matches) != 1:
        raise ValueError('physical input requires one original acquisition record')
    return matches[0]


def _selection_rows(source):
    lookup = {}
    for selection in source.selections:
        decision = selection.decision
        for local, index in enumerate(selection.event_indices):
            lookup[decision.event_ids[local]] = (selection.selection_id, int(index),
                decision.stimulus_vocabulary[int(decision.stimulus_indices[local])],
                tuple(bool(value) for value in decision.membership[local]), decision.candidate_vocabulary)
    return lookup


def _huebner_public_positions(source, index, stimulus_id):
    owner = getattr(source.read_native, '__self__', None)
    if type(owner) is not HuebnerNativeSource or owner.record_id != source.record_id:
        raise PublicPhysicalWitnessUnavailableError('original_Huebner_live_public_marker_witness_unavailable')
    flash = owner.marker.flashes[index]
    public = flash.public_record()
    if stimulus_id != 'physical_set:'+semantic_sha256(public):
        raise ValueError('Huebner public physical-set identity differs from its original admitted marker owner')
    return {'original_public_marker': public,
        'physical_positions': public['physical_positions'],
        'physical_blank_positions': tuple(position for position in public['physical_positions']
            if position in BLANK_POSITIONS),
        'original_marker_sha256': owner.marker.marker_sha256}


def _exposures(source, public_selections, full_stop, *, dataset_id):
    original = _selection_rows(source)
    exposures, vocabularies = [], set()
    for selection in public_selections:
        vocabulary = tuple(selection['candidate_vocabulary'])
        vocabularies.add(vocabulary)
        for event_id, onset, membership in zip(selection['event_ids'], selection['onsets'],
                selection['membership'], strict=True):
            name, index, stimulus, original_membership, original_vocabulary = original[event_id]
            onset = int(onset)
            if (name != selection['selection_id'] or vocabulary != original_vocabulary
                    or tuple(membership) != original_membership or onset != int(source.native_onsets[index])
                    or not 0 <= onset < full_stop):
                raise ValueError('physical exposure changed the original granted event, native clock, vocabulary or membership')
            extra = _huebner_public_positions(source, index, stimulus) if dataset_id == 'Huebner2017' else {}
            exposures.append({'event_id': event_id, 'block_id': name, 'target_lock_group': name,
                'sample': onset, 'native_sample': onset, 'stimulus_id': stimulus,
                'membership': original_membership,
                'duration_seconds_exact_ratio': None,
                'duration_evidence': 'original_public_owner_did_not_establish_exposure_duration',
                'exposure_role': 'completed_protocol_stimulus', **extra})
    if len(vocabularies) != 1 or len({row['event_id'] for row in exposures}) != len(exposures):
        raise ValueError('one physical history requires its complete ordered vocabulary and unique original events')
    return tuple(exposures), next(iter(vocabularies))


def observation_interval_union(calibration_scopes, history_scope=None):
    """Numeric interval relation only; it grants no observation permissions."""
    grouped = {}
    for role, scopes in (('C', calibration_scopes), ('history', () if history_scope is None else (history_scope,))):
        for scope in scopes:
            record, digest = scope['record_id'], scope['source_sha256']
            support = tuple(scope['read_sample_support'])
            if (len(support) != 2 or any(not isinstance(value, Integral) or isinstance(value, bool) for value in support)
                    or not 0 <= support[0] < support[1]):
                raise ValueError('original observation union requires a nonempty integer sample interval')
            previous = grouped.setdefault(record, {'record_id': record, 'source_sha256': digest, 'roles': []})
            if previous['source_sha256'] != digest:
                raise ValueError('one original record cannot change its acquisition source bytes')
            previous['roles'].append({'role': role, 'sample_support': support})
    records = []
    for row in grouped.values():
        intervals = sorted(entry['sample_support'] for entry in row['roles'])
        union = []
        for start, stop in intervals:
            if union and start <= union[-1][1]:
                union[-1][1] = max(union[-1][1], stop)
            else:
                union.append([start, stop])
        intersections = []
        for c in (entry['sample_support'] for entry in row['roles'] if entry['role'] == 'C'):
            for h in (entry['sample_support'] for entry in row['roles'] if entry['role'] == 'history'):
                start, stop = max(c[0], h[0]), min(c[1], h[1])
                if start < stop:
                    intersections.append([start, stop])
        records.append({**row, 'observation_union_sample_support': union,
            'C_history_intersection_sample_support': intersections,
            'consume_each_original_sample_once': True,
            'noise_startup_and_physical_drift_owned_per_original_record': True})
    return freeze_metadata({'schema': 'original_public_observation_interval_union/1', 'records': records})


def _c_scopes(prefix):
    return tuple({'record_id': raw.record_id,
        'source_sha256': prefix.target_contract['record_scopes'][raw.record_id]['source_sha256'],
        'read_sample_support': prefix.target_contract['record_scopes'][raw.record_id]['read_sample_support']}
        for raw in prefix.calibration_records)


def _binding(compiler, prefix, source, samples, public, *, role, current, P, original_identity):
    exposures, vocabulary = _exposures(source, public, len(samples), dataset_id=compiler.adapter.dataset_id)
    history_scope = None if role == 'calibration' else {'record_id': source.record_id,
        'source_sha256': source.source_sha256, 'read_sample_support': [0, len(samples)]}
    mapping = compiler.calibration_record_physical_map(prefix)
    return freeze_metadata({'schema': 'original_public_physical_input/1', 'role': role,
        'dataset_id': compiler.adapter.dataset_id, 'participant_key': source.participant_key,
        'record_id': source.record_id, 'source_sha256': source.source_sha256,
        'sample_start': 0, 'full_stop': len(samples), 'prefix_stop': P, 'available_stop': len(samples),
        'sample_rate_hz': source.sample_rate_hz,
        'sample_rate_hz_exact_ratio': tuple(Fraction.from_float(float(source.sample_rate_hz)).as_integer_ratio()),
        'signal_unit': 'V', 'eeg_channels': source.channel_names, 'reference': source.reference,
        'operator': 'native_unfiltered_volts/1', 'target_lock_group': current,
        'candidate_vocabulary': vocabulary, 'samples_shape': samples.shape,
        'samples_sha256': array_digest(samples), 'requested_C_label_budget': prefix.budget,
        'C_prefix_identity': calibration_prefix_identity(prefix),
        'C_label_use_sha256': metadata_digest(prefix.label_use),
        'C_record_scopes': prefix.target_contract['record_scopes'],
        'C_record_to_original_record': {key: row['physical_record_id'] for key, row in mapping['records'].items()},
        'C_record_input_sha256': {key: row['input_sha256'] for key, row in mapping['records'].items()},
        'observation_union': observation_interval_union(_c_scopes(prefix), history_scope),
        'original_grant_identity': original_identity,
        'public_physical_history': {'schema': 'original_public_native_record_history/1',
            'membership_candidate_vocabulary': vocabulary, 'selections': public, 'exposures': exposures,
            'native_sample_rate_hz': source.sample_rate_hz,
            'clock_error_role': 'original_native_sample_axis_and_admitted_event_join_no_wall_or_delivery_error_bound_added',
            'clock_program': NATIVE_CLOCK, 'record_clock_domain': source.record_id,
            'wall_or_delivery_anchor_asserted': False},
        'original_source_owner_sha256': compiler.history_owner_source,
        'bridge_source_sha256': IMPLEMENTATION_SHA256, 'future_truth_used': False})


@dataclass(frozen=True)
class PublicNativeClockChart:
    public_binding: dict
    capability_identity_sha256: str

    def _rows(self, rows, clock_program):
        if clock_program != NATIVE_CLOCK:
            raise ValueError('public native records require their original native_sample_index clock programme')
        scalar = isinstance(rows, Integral) and not isinstance(rows, bool)
        values = (rows,) if scalar else tuple(rows)
        if any(not isinstance(row, Integral) or isinstance(row, bool)
                or not self.public_binding['sample_start'] <= row < self.public_binding['full_stop'] for row in values):
            raise ValueError('clock rows must be actual granted integer native sample indices')
        return scalar, tuple(int(row) for row in values)

    def sample_time_seconds(self, native_samples, *, clock_program, reference_origin=Fraction(0)):
        scalar, rows = self._rows(native_samples, clock_program)
        rate = Fraction(*self.public_binding['sample_rate_hz_exact_ratio'])
        values = tuple(Fraction(row)/rate-Fraction(reference_origin) for row in rows)
        return values[0] if scalar else values

    def elapsed_seconds(self, native_samples, exposure_id, *, clock_program):
        scalar, rows = self._rows(native_samples, clock_program)
        event = next((event for event in self.public_binding['public_physical_history']['exposures']
            if event['event_id'] == exposure_id), None)
        if event is None:
            raise ValueError('native elapsed time requires an original granted public exposure identity')
        rate = Fraction(*self.public_binding['sample_rate_hz_exact_ratio'])
        values = tuple(Fraction(row-event['native_sample'])/rate for row in rows)
        return values[0] if scalar else values


def _chart(binding, identity, selection_id):
    selections = binding['public_physical_history']['selections']
    if selection_id not in {selection['selection_id'] for selection in selections}:
        raise ValueError('a physical target lock must name an original occurred selection')
    return PublicNativeClockChart(freeze_metadata({**binding, 'target_lock_group': selection_id}), identity)


@dataclass(frozen=True, eq=False)
class PublicPhysicalInput:
    samples: np.ndarray
    label_rows: tuple
    public_binding: dict
    identity: str

    def clock_for_selection(self, selection_id):
        require_public_physical_input(self)
        return _chart(self.public_binding, self.identity, selection_id)


def _publish(compiler, prefix, source, samples, labels, binding, *, raw=None, cap=None, view=None, P=None):
    identity = metadata_digest({'binding': binding, 'label_rows': labels})
    result = PublicPhysicalInput(samples, labels, binding, identity)
    _INPUTS[result] = (compiler, prefix, source.record_id, samples, labels, binding, raw, cap, view, P,
        semantic_sha256(source.public_record()))
    return result


def admit_public_calibration(compiler, prefix):
    prefix = _owned_prefix(compiler, prefix)
    result = []
    for raw in prefix.calibration_records:
        source = _source(compiler, raw.record_id)
        count = len(raw.onsets)
        if not np.array_equal(raw.onsets, source.native_onsets[:count]):
            raise ValueError('C physical clock differs from the original source-owned raw prefix')
        public, labels = [], []
        for selection in source.selections:
            rows = [local for local, index in enumerate(selection.event_indices) if index < count]
            if not rows:
                continue
            decision = selection.decision
            indices = [selection.event_indices[index] for index in rows]
            public.append({'selection_id': selection.selection_id, 'candidate_vocabulary': decision.candidate_vocabulary,
                'event_ids': tuple(decision.event_ids[index] for index in rows),
                'onsets': tuple(int(raw.onsets[index]) for index in indices),
                'membership': tuple(tuple(bool(v) for v in decision.membership[index]) for index in rows),
                'complete_requested_current': len(rows) == len(selection.event_indices)})
            labels.extend((decision.event_ids[local], int(raw.onsets[index]), int(raw.target_mask[index]))
                for local, index in zip(rows, indices, strict=True))
        labels = tuple(labels)
        binding = _binding(compiler, prefix, source, raw.samples, tuple(public), role='calibration',
            current=None, P=len(raw.samples), original_identity=array_digest(raw.samples, raw.onsets, raw.target_mask))
        result.append(_publish(compiler, prefix, source, raw.samples, labels, binding, raw=raw))
    return tuple(result)


@dataclass(frozen=True, eq=False)
class PublicPhysicalUnavailable:
    record: dict


def _unavailable(compiler, prefix, view, P, source, error, *, refusal=None):
    record = freeze_metadata({'schema': 'original_public_physical_unavailable/1',
        'status': 'unavailable', 'current_decision_id': view.decision_id,
        'record_id': source.record_id, 'source_sha256': source.source_sha256,
        'C_prefix_identity': calibration_prefix_identity(prefix), 'requested_C_label_budget': prefix.budget,
        'conditioning_stop': P, 'available_stop': view.available_sample,
        'original_view_sha256': semantic_sha256(view.record()),
        'error_type': type(error).__name__, 'reason': getattr(error, 'reason', str(error)),
        'original_registered_refusal': None if refusal is None else refusal.digest,
        'raw_or_label_permission_granted': False})
    result = PublicPhysicalUnavailable(record)
    _UNAVAILABLE[result] = (compiler, prefix, view, P, source.record_id, refusal, record,
        semantic_sha256(source.public_record()))
    return result


def require_original_history_scope(compiler, prefix, view, cap, *, conditioning_stop):
    if type(cap) is not AuthorizedRecordPublicHistory:
        raise TypeError('public physical history requires the original registered history scope')
    original = _prepared_record_read_scope(compiler, view, prefix=prefix, conditioning_stop=conditioning_stop)
    if original is not cap:
        raise ValueError('the physical bridge must retain the exact original registered history object')
    return cap


def admit_public_history(compiler, prefix, view, *, conditioning_stop):
    prefix = _owned_prefix(compiler, prefix)
    matches = [source for source in compiler.sources if source.source_sha256 == view.source_sha256
        and source.source_relative_path == view.source_relative_path]
    if len(matches) != 1:
        raise ValueError('public history requires its original unique acquisition source')
    source, P = matches[0], conditioning_stop
    key = _record_history_key({'source_sha256': view.source_sha256, 'record_id': source.record_id,
        'current_decision_id': view.decision_id, 'prefix_identity': calibration_prefix_identity(prefix),
        'conditioning_sample_support': [0, P], 'score_sample_support': [P, view.available_sample]})
    existing = key in getattr(compiler, '_record_history_scopes', {})
    try:
        # An existing but invalid scope must propagate its error, never reread.
        cap = (_prepared_record_read_scope(compiler, view, prefix=prefix, conditioning_stop=P) if existing else
            compiler.record_read_scope(view, prefix=prefix, conditioning_stop=P))
    except HistoryUnavailableError as error:
        refusal = _lookup_record_history_refusal(compiler, view, prefix=prefix, conditioning_stop=P)
        return _unavailable(compiler, prefix, view, P, source, error, refusal=refusal)
    except ValueError as error:
        selection = [selection for selection in source.selections if selection.selection_id == view.decision_id]
        if (existing or len(selection) != 1
                or str(error) != 'current history view differs from original event-set/native-clock authority'
                or not np.any(source.native_onsets[np.asarray(selection[0].event_indices)] < 0)):
            raise
        c_scope = prefix.target_contract['record_scopes'].get(source.record_id)
        if c_scope is not None and c_scope['read_sample_support'][1] > P:
            raise
        ordinal = next(index for index, value in enumerate(source.selections) if value is selection[0])
        # The already established missing native onset keeps this original
        # current-view request unavailable, so its owner performs no EEG read.
        original = compiler._current_view(source, ordinal).physical_record()
        actual = view.physical_record()
        if actual != original or view.samples is not None:
            raise
        return _unavailable(compiler, prefix, view, P, source, error)
    cap = require_original_history_scope(compiler, prefix, view, cap, conditioning_stop=P)
    binding = _binding(compiler, prefix, source, cap.samples, cap.binding['public_selections'], role='history',
        current=view.decision_id, P=P, original_identity=cap.digest)
    return _publish(compiler, prefix, source, cap.samples, (), binding, cap=cap, view=view, P=P)


def require_public_physical_input(value):
    stored = _INPUTS.get(value) if type(value) is PublicPhysicalInput else None
    if (stored is None or value.samples is not stored[3] or value.label_rows is not stored[4]
            or value.public_binding is not stored[5]):
        raise TypeError('public physical input requires its live original-compiler adaptation')
    compiler, prefix, record_id = stored[:3]
    prefix = _owned_prefix(compiler, prefix)
    source = _source(compiler, record_id)
    if semantic_sha256(source.public_record()) != stored[10]:
        raise ValueError('the original adapted source clock or public physical programme changed')
    if stored[6] is not None:
        if not any(raw is stored[6] for raw in prefix.calibration_records):
            raise ValueError('C adaptation lost its exact original raw/label grant')
    else:
        cap = _prepared_record_read_scope(compiler, stored[8], prefix=prefix, conditioning_stop=stored[9])
        if cap is not stored[7] or cap.samples is not value.samples:
            raise ValueError('history adaptation lost its exact original registered scope')
    return value


def require_public_unavailable(value):
    stored = _UNAVAILABLE.get(value) if type(value) is PublicPhysicalUnavailable else None
    if stored is None or value.record is not stored[6]:
        raise TypeError('public unavailable evidence requires its original live request relation')
    compiler, prefix, view, P, record_id, refusal = stored[:6]
    if semantic_sha256(view.record()) != value.record['original_view_sha256']:
        raise ValueError('the exact original unavailable public-view request changed')
    prefix = _owned_prefix(compiler, prefix)
    if semantic_sha256(_source(compiler, record_id).public_record()) != stored[7]:
        raise ValueError('the original unavailable clock programme changed')
    if refusal is not None:
        if _lookup_record_history_refusal(compiler, view, prefix=prefix, conditioning_stop=P) is not refusal:
            raise ValueError('original history refusal identity changed')
    else:
        try:
            compiler.record_read_scope(view, prefix=prefix, conditioning_stop=P)
        except ValueError as error:
            if type(error).__name__ != value.record['error_type'] or str(error) != value.record['reason']:
                raise ValueError('the original current-clock refusal changed') from error
        else:
            raise ValueError('the original current-clock refusal is no longer present')
    return value


@dataclass(frozen=True)
class NumericPublicPhysicalInput:
    samples: np.ndarray
    label_rows: tuple
    public_binding: dict
    identity: str

    def clock_for_selection(self, selection_id):
        return _chart(self.public_binding, self.identity, selection_id)


def save_numeric_adapter(value, path):
    value = require_public_physical_input(value)
    document = {'schema': 'numeric_public_physical_descriptor/1', 'identity': value.identity,
        'public_binding': plain_metadata(value.public_binding), 'label_rows': plain_metadata(value.label_rows),
        'numeric_arrays_supplied_separately': True, 'raw_or_label_or_live_authority_granted': False}
    Path(path).write_text(json.dumps(document, indent=2)+'\n', encoding='utf-8')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cold_numeric_adapter(path, *, expected_sha256, samples):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('numeric public descriptor differs from its external expected bytes')
    document, samples = json.loads(content), np.asarray(samples)
    binding, labels = freeze_metadata(document['public_binding']), freeze_metadata(document['label_rows'])
    if (tuple(samples.shape) != tuple(binding['samples_shape']) or array_digest(samples) != binding['samples_sha256']
            or metadata_digest({'binding': binding, 'label_rows': labels}) != document['identity']):
        raise ValueError('numeric public input differs from its original saved numeric relation')
    return NumericPublicPhysicalInput(frozen_array(samples), labels, binding, document['identity'])
