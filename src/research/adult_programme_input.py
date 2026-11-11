"""Canonical adult multi-record C/held inputs, independent of a source model.

Only the original raw issuer reads EEG and grants C labels. This owner joins
the full requested roster, original physical clocks and EEG-byte exclusion
union. A saved manifest is metadata; fresh live admission reopens every parent.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from data.brainsync_contract import PopulationScopePolicy, derive_population_scope
from data.brainsync_raw_authority import (
    BrainSyncRecordingScope,
    allowed_candidates,
    compile_physical_clock,
    open_brainsync_calibration,
    open_brainsync_recording,
    require_calibration_labels,
    require_raw_capability,
)
from data.identity import DatasetIdentityTable
from research.channel_noise_mechanism import NoiseGeometry
from research.execution import publish_exclusive
from research.frozen_metadata import freeze_metadata, plain_metadata
from transfer.noise_model import metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_LIVE = WeakKeyDictionary()


def _save_new(path, record):
    path = Path(path).resolve()
    payload = (json.dumps(plain_metadata(record), sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    publish_exclusive(path, lambda stream: stream.write(payload), byte_size=len(payload))
    return {'path': str(path), 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


@dataclass(frozen=True)
class NativeReadRequest:
    segment_id: int
    prefix_stop: int
    full_stop: int
    available_stop: int

    def record(self):
        return asdict(self)


@dataclass(frozen=True)
class CalibrationRecordingRequest:
    session_id: str
    path: Path
    reads: tuple[NativeReadRequest, ...]
    label_count: int
    ready_at_timestamp_s: float

    def __post_init__(self):
        object.__setattr__(self, 'path', Path(self.path).resolve())
        object.__setattr__(self, 'reads', tuple(self.reads))
        if type(self.label_count) is not int or self.label_count < 0:
            raise ValueError('C requests declare a nonnegative integer original label count')

    def record(self):
        return {'session_id': self.session_id, 'path': str(self.path),
            'reads': [read.record() for read in self.reads], 'label_count': self.label_count,
            'ready_at_timestamp_s': self.ready_at_timestamp_s}


@dataclass(frozen=True)
class HeldRecordingRequest:
    session_id: str
    path: Path
    reads: tuple[NativeReadRequest, ...]

    def __post_init__(self):
        object.__setattr__(self, 'path', Path(self.path).resolve())
        object.__setattr__(self, 'reads', tuple(self.reads))

    def record(self):
        return {'session_id': self.session_id, 'path': str(self.path),
            'reads': [read.record() for read in self.reads]}


class AdultProgrammeInputError(ValueError):
    """Input failure retains every request; it is not a scientific fit failure."""

    def __init__(self, record, error):
        self.record = freeze_metadata(record)
        self.original_error = error
        super().__init__(str(error))

    def save(self, path):
        return _save_new(path, self.record)


@dataclass(frozen=True)
class AdultSegmentInput:
    """Original cap and exact physical chart; no new noise or drift law."""

    capability: object
    clock: object

    @property
    def native_origin(self):
        return self.capability.public_binding['sample_start']

    @property
    def observation(self):
        return self.capability.eeg


@dataclass(frozen=True)
class AdultRecordingInput:
    role: str
    session_id: str
    record_id: str
    segments: tuple[AdultSegmentInput, ...]
    calibration_labels: object | None
    record: dict


@dataclass(frozen=True, eq=False)
class AdultProgrammeInputs:
    calibration: tuple[AdultRecordingInput, ...]
    held: tuple[AdultRecordingInput, ...]
    geometry: NoiseGeometry
    identity_table: object
    record: dict

    @property
    def identity(self):
        return metadata_digest(self.record)

    @property
    def exclusion_contract(self):
        return self.record['source_exclusion']

    def save(self, path):
        require_adult_programme_inputs(self)
        state = _LIVE[self]
        return _save_new(path, {'schema': 'frozen_adult_programme_input_manifest/1',
            'programme_identity': self.identity, 'record': self.record,
            'reopen_requests': state['requests'],
            'reopen_requests_sha256': metadata_digest(state['requests']),
            'private_original_file_refs': state['file_refs'],
            'raw_or_label_or_source_or_fit_authority_granted': False})

    def close(self):
        """Close all original readers; their issued immutable caps stay valid."""
        state = _LIVE.get(self)
        if state is None:
            raise TypeError('adult programme close requires its original live admission')
        state['stack'].close()

    def __enter__(self):
        return require_adult_programme_inputs(self)

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class FrozenAdultProgrammeInputs:
    """A hash-bound metadata view; it contains no live capability registry token."""

    record: dict
    requests: dict
    original_file_refs: tuple

    @property
    def identity(self):
        return metadata_digest(self.record)

    @classmethod
    def load(cls, path, *, expected_sha256):
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError('adult input manifest changed its externally pinned bytes')
        record = json.loads(payload)
        if (record['schema'] != 'frozen_adult_programme_input_manifest/1'
                or record['raw_or_label_or_source_or_fit_authority_granted'] is not False
                or metadata_digest(record['record']) != record['programme_identity']
                or metadata_digest(record['reopen_requests']) != record['reopen_requests_sha256']):
            raise ValueError('adult input manifest changed its original metadata/request relation')
        return cls(freeze_metadata(record['record']), freeze_metadata(record['reopen_requests']),
            freeze_metadata(record['private_original_file_refs']))

    def re_admit(self):
        """Reopen every original parent and reproduce the complete input identity."""
        requests = self.requests
        scope = BrainSyncRecordingScope(tuple(requests['scope']['calibration_session_ids']),
            tuple(requests['scope']['evaluation_session_ids']))
        calibration = tuple(CalibrationRecordingRequest(row['session_id'], Path(row['path']),
            tuple(NativeReadRequest(**read) for read in row['reads']), row['label_count'], row['ready_at_timestamp_s'])
            for row in requests['calibration'])
        held = tuple(HeldRecordingRequest(row['session_id'], Path(row['path']),
            tuple(NativeReadRequest(**read) for read in row['reads'])) for row in requests['held'])
        programme = open_adult_programme_inputs(calibration, held, scope=scope)
        state = _LIVE[programme]
        if (programme.identity != self.identity
                or metadata_digest(state['file_refs']) != metadata_digest(self.original_file_refs)):
            programme.close()
            raise ValueError('actual original raw/C re-admission differs from the complete saved programme')
        return programme


def _requests(calibration, held, scope):
    if (any(type(row) is not CalibrationRecordingRequest for row in calibration)
            or any(type(row) is not HeldRecordingRequest for row in held)):
        raise TypeError('adult input requests must retain explicit separate C/held request types')
    return freeze_metadata({'scope': scope.record(),
        'calibration': [row.record() for row in calibration], 'held': [row.record() for row in held]})


def _check_roster(calibration, held, scope):
    for requests, expected, role in ((calibration, scope.calibration_session_ids, 'calibration'),
            (held, scope.evaluation_session_ids, 'held')):
        names = tuple(row.session_id for row in requests)
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('the complete declared '+role+' session roster is required')
        if any(not row.reads or any(type(read) is not NativeReadRequest for read in row.reads) for row in requests):
            raise ValueError('every requested original recording requires its actual native read requests')
    if not held:
        raise ValueError('adult programme requires its declared independent held recording roster')


def _original_files(issuer):
    """Trusted-parent audit, using the same actual-reader proof as ProgrammeAdmission."""
    receipt = issuer.private_admission_receipt()['original_files']
    readers = tuple(str(Path(path).resolve()) for path in issuer._raw.filenames)
    if not readers:
        raise ValueError('the actual original EEG reader has no EEG byte source')
    return tuple({'path': path, 'sha256': receipt[path]['sha256']} for path in readers), tuple(
        {'path': path, 'sha256': value['sha256']} for path, value in sorted(receipt.items()))


def _record_inputs(role, request, issuer, participant_key, files, ledger):
    granted = []
    for read in request.reads:
        granted.append(issuer.issue_raw(**read.record()))
        ledger.update(raw_reads_available=len(granted),
            raw_capability_identities=[cap.identity_sha256 for cap in granted])
    caps = tuple(granted)
    first = caps[0].public_binding
    if (len({cap.public_binding['segment_id'] for cap in caps}) != len(caps)
            or any(cap.public_binding['record_id'] != first['record_id'] for cap in caps)):
        raise ValueError('one original segment is granted once inside its actual recording')
    labels = None
    coverage = None
    startup_history_ids = set()
    if role == 'calibration':
        labels = issuer.issue_labels(request.label_count, ready_at_timestamp_s=request.ready_at_timestamp_s)
        ledger.update(actual_C_labels=len(labels.labels), calibration_label_capability_identity=labels.identity_sha256)
        coverage = allowed_candidates(caps, labels)
        observed = {event.event_id for cap in caps for event in cap.events
            if cap.public_binding['sample_start'] <= event.sample < cap.public_binding['full_stop']}
        attested_startup_ids = set(labels.public_binding.get('startup_events_without_EEG_support', ()))
        startup_history_ids = {event.event_id for cap in caps for event in cap.events
            if cap.public_binding.get('raw_startup_cut') is not None
            and event.event_id in attested_startup_ids
            and event.sample < cap.public_binding['raw_startup_cut']['stop_sample']
            and event.sample < cap.public_binding['full_stop']}
        if any(label.event_id not in observed | startup_history_ids for label in labels.labels):
            raise ValueError('every requested C bit needs its original granted EEG observation support')
        if startup_history_ids:
            ledger.update(startup_history_labels_without_EEG_support=sorted(startup_history_ids),
                actual_C_labels_with_EEG_support=sum(label.event_id in observed for label in labels.labels))
    history_cap = max(caps, key=lambda cap: cap.public_binding['available_stop'])
    history = history_cap.public_binding['public_physical_history']
    blocks = tuple(row['block_id'] for row in first['block_requests'])
    segments = tuple(AdultSegmentInput(cap, compile_physical_clock(cap)) for cap in caps)
    public = freeze_metadata({'role': role, 'session_id': request.session_id,
        'record_id': first['record_id'], 'participant_key': participant_key,
        'requested_reads': [read.record() for read in request.reads],
        'segments': [{'raw_capability_identity': cap.identity_sha256,
            'segment_id': cap.public_binding['segment_id'],
            'record_segment_id': cap.public_binding['record_segment_id'],
            'native_origin': cap.public_binding['sample_start'],
            'read_sample_support': [cap.public_binding['sample_start'], cap.public_binding['full_stop']],
            'prefix_stop': cap.public_binding['prefix_stop'], 'available_stop': cap.public_binding['available_stop'],
            'public_acquisition_segment': cap.public_binding['public_acquisition_segment']}
            for cap in caps],
        'actual_original_EEG_files': [{'sha256': value['sha256']} for value in files],
        'geometry': {'channels': first['eeg_channels'], 'sfreq': first['sample_rate_hz'],
            'unit': first['signal_unit'], 'reference': first['reference'], 'operator_identity': first['operator'],
            'observation_rank': len(first['eeg_channels'])},
        'target_lock': {'id': first['target_lock_group'], 'candidate_vocabulary': caps[0].candidate_vocabulary,
            'origin': first['target_lock_program'], 'occurrences_in_record_roster': 1},
        'blocks': [{'block_id': block, 'shape_variable_id': f"{first['record_id']}:block:{block}:shape",
            'target_lock_id': first['target_lock_group'], 'shape_or_local_law_supplied': False} for block in blocks],
        'public_history': history, 'public_clock': first['public_clock'],
        'block_requests': first['block_requests'],
        'requested_completed_flashes': history['requested_completed_flashes'],
        'actual_observed_completed_flashes': sum(1 for cap in caps for event in cap.events
            if cap.public_binding['sample_start'] <= event.sample < cap.public_binding['full_stop']),
        **({'startup_history_labels_without_EEG_support': sorted(startup_history_ids)} if startup_history_ids else {}),
        'requested_C_labels': request.label_count if role == 'calibration' else None,
        'actual_C_labels': None if labels is None else len(labels.labels),
        'calibration_label_capability_identity': None if labels is None else labels.identity_sha256,
        'C_candidate_constraint': coverage, 'held_label_authority': False,
        'raw_requests_available': len(caps), 'all_requested_reads_available': True,
        'boundary_role': 'original_acquisition_segments_and_record_IDs_only; caller_owns_noise_and_drift_laws'})
    return AdultRecordingInput(role, request.session_id, first['record_id'], segments, labels, public)


def open_adult_programme_inputs(calibration_requests, held_requests, *, scope):
    """Admit the whole declared C/held roster; never return a readable subset."""
    calibration, held = tuple(calibration_requests), tuple(held_requests)
    request_record = _requests(calibration, held, scope)
    requested = [('calibration', row) for row in calibration]+[('held', row) for row in held]
    ledger = []
    for role, names in (('calibration', scope.calibration_session_ids), ('held', scope.evaluation_session_ids)):
        for name in names:
            matches = [row for requested_role, row in requested if requested_role == role and row.session_id == name]
            row = matches[0] if len(matches) == 1 else None
            ledger.append({'role': role, 'session_id': name,
                'requested_reads': None if row is None else len(row.reads),
                'requested_C_labels': row.label_count if role == 'calibration' and row is not None else None,
                'raw_reads_available': 0, 'actual_C_labels': None,
                'status': 'not_opened' if row is not None else 'missing_or_duplicate_request_description'})
    declared = {(row['role'], row['session_id']) for row in ledger}
    for role, row in requested:
        if (role, row.session_id) not in declared:
            ledger.append({'role': role, 'session_id': row.session_id,
                'requested_reads': len(row.reads), 'requested_C_labels': row.label_count if role == 'calibration' else None,
                'raw_reads_available': 0, 'actual_C_labels': None, 'status': 'outside_declared_scope'})
    positions = {(row['role'], row['session_id']): index for index, row in enumerate(ledger)}
    stack, opened, private_refs = ExitStack(), [], []
    stage, active_index = 'complete_requested_roster', None
    try:
        _check_roster(calibration, held, scope)
        for role, request in requested:
            index = positions[role, request.session_id]
            active_index = index
            stage = 'original_record_admission'
            opener = open_brainsync_calibration if role == 'calibration' else open_brainsync_recording
            issuer = opener(request.path, scope=scope)
            stack.callback(issuer.close)
            validated = issuer._validated
            if validated.session_id != request.session_id:
                raise ValueError('the actual original session differs from its requested identity')
            files, all_files = _original_files(issuer)
            private_refs.extend(all_files)
            opened.append((role, request, issuer, validated, files))
            ledger[index].update(status='original_metadata_admitted', age_years=validated.age_years,
                subject_id=validated.subject_id, started_utc=validated.started_utc,
                ended_utc=validated.ended_utc)
        stage, active_index = 'adult_complete_population_scope', None
        participants = [row[3].subject_id for row in opened]
        if len(set(participants)) != 1:
            raise ValueError('the complete adult programme requires one actual participant identity')
        population = derive_population_scope(participants, [row[3].age_years for row in opened],
            policy=PopulationScopePolicy.ADULT_ONLY)
        table = DatasetIdentityTable.from_source_rows([participants[0]], ['BrainSync-GTN'])
        participant_key = table.record_for(participants[0]).authority_key('source_or_global')
        stage = 'original_EEG_file_union'
        eeg_owners = {}
        for role, request, _, _, files in opened:
            active_index = positions[role, request.session_id]
            for file in files:
                if file['sha256'] in eeg_owners and eeg_owners[file['sha256']] != (role, request.session_id):
                    raise ValueError('independent requested recordings share the same original EEG bytes')
                eeg_owners[file['sha256']] = (role, request.session_id)
        records, geometry = [], None
        for role, request, issuer, _validated, files in opened:
            index = positions[role, request.session_id]
            active_index = index
            stage = 'complete_raw_and_C_label_requests'
            record = _record_inputs(role, request, issuer, participant_key, files, ledger[index])
            current_geometry = NoiseGeometry(**plain_metadata(record.record['geometry']))
            if geometry is not None and current_geometry != geometry:
                raise ValueError('all requested C/held recordings retain the same actual native observation geometry')
            geometry = current_geometry
            ledger[index].update(status='all_requests_admitted', raw_reads_available=len(record.segments),
                actual_C_labels=record.record['actual_C_labels'], record_id=record.record_id)
            records.append(record)
        c_ready = max((row[3].calibration_available_timestamp_s for row in opened if row[0] == 'calibration'), default=None)
        facts = [{'role': role, 'session_id': request.session_id, 'age_years': validated.age_years,
            'started_timestamp_s': validated.started_timestamp_s, 'ended_timestamp_s': validated.ended_timestamp_s,
            'original_C_labels_ready_timestamp_s': validated.calibration_available_timestamp_s if role == 'calibration' else None,
            'held_started_after_all_C_labels_ready': None if role == 'calibration' or c_ready is None
                else validated.started_timestamp_s >= c_ready} for role, request, _, validated, _ in opened]
        roster, record_scopes = {}, {}
        for record, source in zip(records, opened, strict=True):
            for index, file in enumerate(source[4]):
                roster[f'{record.record_id}:actual-eeg-file:{index}'] = {
                    'source_sha256': file['sha256'], 'source_dataset_id': 'BrainSync-GTN',
                    'participant_key': participant_key, 'session_id': record.session_id, 'role': record.role}
            for segment in record.record['segments']:
                record_scopes[segment['record_segment_id']] = {
                    'record_id': record.record_id, 'source_sha256': source[4][0]['sha256'],
                    **{key: segment[key] for key in ('read_sample_support', 'prefix_stop', 'available_stop')}}
        exclusion = freeze_metadata({'schema': 'actual_adult_C_held_original_file_union/1',
            'participant_key': participant_key, 'source_dataset_id': 'BrainSync-GTN',
            'source_subject_id': participants[0], 'source_record_roster': roster,
            'record_scopes': record_scopes, 'identity_table': table.payload(),
            'calibration_record_ids': [row.record_id for row in records if row.role == 'calibration'],
            'held_record_ids': [row.record_id for row in records if row.role == 'held'],
            'original_EEG_bytes_only_in_source_exclusion': True})
        public = freeze_metadata({'schema': 'actual_adult_multirecord_programme_inputs/1',
            'producer_sha256': IMPLEMENTATION_SHA256, 'scope': scope.record(),
            'participant_key': participant_key, 'population': population.to_dict(),
            'recording_facts': facts, 'requested_roster': ledger, 'records': [row.record for row in records],
            'requested_C_labels': sum(row.label_count for row in calibration),
            'actual_C_labels': sum(len(row.calibration_labels.labels) for row in records if row.role == 'calibration'),
            'requested_raw_reads': sum(len(row.reads) for _, row in requested),
            'actual_raw_reads': sum(len(row.segments) for row in records),
            'source_exclusion': exclusion, 'all_requested_inputs_admitted': True,
            'held_target_labels_read': 0, 'source_models_or_fits_issued': 0,
            'candidate_prior_shape_noise_and_drift_laws_issued': False})
        programme = AdultProgrammeInputs(tuple(row for row in records if row.role == 'calibration'),
            tuple(row for row in records if row.role == 'held'), geometry, table, public)
        _LIVE[programme] = {'record': public, 'records': tuple(records), 'stack': stack,
            'requests': request_record, 'file_refs': freeze_metadata(private_refs)}
        return programme
    except Exception as error:
        if active_index is not None:
            ledger[active_index].update(status='admission_failed', failed_stage=stage)
        failed = freeze_metadata({'schema': 'adult_multirecord_input_admission_failure/1',
            'stage': stage, 'error_type': type(error).__name__, 'reason': str(error),
            'failure_scope': 'complete_programme' if active_index is None else 'named_original_record',
            'requested_scope': scope.record(), 'requested_roster': ledger,
            'requested_C_labels': sum(row.label_count for row in calibration),
            'provided_C_label_request_counts_complete': all(row['requested_C_labels'] is not None
                for row in ledger if row['role'] == 'calibration'),
            'actual_raw_reads_granted_before_failure': sum(row['raw_reads_available'] for row in ledger),
            'actual_C_labels_granted_before_failure': sum(row['actual_C_labels'] or 0 for row in ledger),
            'all_requested_inputs_admitted': False, 'partial_inputs_grant_live_programme': False,
            'scientific_fit_failure': False, 'held_target_labels_read': 0})
        stack.close()
        raise AdultProgrammeInputError(failed, error) from error
    except BaseException:
        stack.close()
        raise


def require_adult_programme_inputs(programme):
    state = _LIVE.get(programme) if type(programme) is AdultProgrammeInputs else None
    if state is None or programme.record is not state['record']:
        raise TypeError('adult inputs require their original complete canonical live admission')
    records = programme.calibration+programme.held
    if len(records) != len(state['records']) or any(a is not b for a, b in zip(records, state['records'], strict=True)):
        raise ValueError('adult programme changed its complete original recording objects')
    if (programme.geometry.record() != plain_metadata(records[0].record['geometry'])
            or programme.identity_table.payload() != plain_metadata(programme.exclusion_contract['identity_table'])):
        raise ValueError('adult programme changed its actual observation or participant identity table')
    for record, issued_record in zip(records, programme.record['records'], strict=True):
        if record.record is not issued_record:
            raise ValueError('adult recording metadata changed its original programme binding')
        original = record.record
        if (record.role != original['role'] or record.session_id != original['session_id']
                or record.record_id != original['record_id'] or len(record.segments) != len(original['segments'])):
            raise ValueError('adult recording differs from its complete original role/identity/segment roster')
        for segment, declaration in zip(record.segments, original['segments'], strict=True):
            cap = require_raw_capability(segment.capability)
            expected_clock = compile_physical_clock(cap)
            if (cap.identity_sha256 != declaration['raw_capability_identity']
                    or type(segment.clock) is not type(expected_clock)
                    or segment.clock.capability_identity_sha256 != cap.identity_sha256
                    or metadata_digest(segment.clock.public_binding) != metadata_digest(expected_clock.public_binding)):
                raise ValueError('adult segment clock differs from its actual original raw capability')
        if record.role == 'calibration':
            labels = require_calibration_labels(record.calibration_labels)
            if labels.identity_sha256 != original['calibration_label_capability_identity']:
                raise ValueError('adult C labels differ from their original recording grant')
        elif record.calibration_labels is not None:
            raise ValueError('held inputs have no calibration-label authority')
    return programme
