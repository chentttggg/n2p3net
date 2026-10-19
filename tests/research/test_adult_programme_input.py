"""Original synthetic EDF/BIDS controls for the complete adult input owner."""
from __future__ import annotations

import json
import shutil
from fractions import Fraction

import pytest

from data import brainsync_raw_authority as authority
from research import adult_programme_input as module
from tests.support.adult_programme import build_complete_requests, original_session, read_requests
from tests.support.io import read_json, write_json


@pytest.fixture
def complete_requests(tmp_path):
    return build_complete_requests(tmp_path)

def test_complete_original_multirecord_inputs_and_cold_actual_readmission(complete_requests, tmp_path, monkeypatch):
    calibration, held, scope = complete_requests
    original_validator = authority.validate_brainsync_bids_session

    def only_calibration(path, **kwargs):
        assert path != held[0].path, 'held reached supervised target validation'
        return original_validator(path, **kwargs)

    monkeypatch.setattr(authority, 'validate_brainsync_bids_session', only_calibration)
    with module.open_adult_programme_inputs(calibration, held, scope=scope) as programme:
        assert programme.record['population']['label'] == 'adult'
        assert programme.record['population']['session_count'] == 3
        assert programme.record['requested_C_labels'] == programme.record['actual_C_labels'] == 135
        assert programme.record['requested_raw_reads'] == programme.record['actual_raw_reads'] == 4
        assert len(programme.calibration) == 2 and len(programme.held) == 1
        assert len(programme.exclusion_contract['source_record_roster']) == 3
        assert len(programme.exclusion_contract['record_scopes']) == 4
        assert programme.geometry.sfreq == 250 and len(programme.geometry.channels) == 7
        c1, c2 = programme.calibration
        assert c1.record['geometry'] == c2.record['geometry']
        assert c1.record_id != c2.record_id
        assert c1.segments[0].capability.identity_sha256 != c2.segments[0].capability.identity_sha256
        assert len({row.record['target_lock']['id'] for row in programme.calibration+programme.held}) == 3
        assert sum(len(row.record['blocks']) for row in programme.calibration+programme.held) == 5
        assert c1.record['target_lock']['occurrences_in_record_roster'] == 1
        assert len(c1.segments) == 2 and c1.segments[1].native_origin == 7625
        assert c1.segments[1].clock.sample_time_seconds(7750, clock_program='timeline_wall') == Fraction(33)
        history = c1.record['public_history']
        assert len(history['acquisition_pauses']) == 1 and history['unknown_exposure_count'] == 1
        assert history['prior_segment_responses_reset_to_zero'] is False
        interrupted = next(row for row in history['exposures'] if row['membership'] is None)
        assert interrupted['candidate_options'] == tuple(map(str, range(1, 10)))
        assert programme.held[0].calibration_labels is None
        assert programme.held[0].record['requested_completed_flashes'] == 90
        assert programme.held[0].record['actual_C_labels'] is None
        assert 'thought_digit' not in json.dumps(programme.record)
        reference = programme.save(tmp_path/'input-manifest.json')
        identity = programme.identity
        original_cap = c1.segments[0].capability
    # Closing the readers does not invent or revoke the original immutable cap.
    assert module.require_adult_programme_inputs(programme) is programme
    with monkeypatch.context() as patcher:
        patcher.setattr(module, 'open_brainsync_calibration', lambda *a, **k: pytest.fail('cold opened C'))
        patcher.setattr(module, 'open_brainsync_recording', lambda *a, **k: pytest.fail('cold opened held'))
        cold = module.FrozenAdultProgrammeInputs.load(reference['path'], expected_sha256=reference['sha256'])
    with pytest.raises(TypeError, match='original complete canonical'):
        module.require_adult_programme_inputs(cold)
    with cold.re_admit() as restored:
        assert restored.identity == identity
        assert restored.calibration[0].segments[0].capability is not original_cap
        assert restored.calibration[0].segments[0].capability.identity_sha256 == original_cap.identity_sha256
        assert restored.record['actual_C_labels'] == 135
        assert restored.exclusion_contract == programme.exclusion_contract


def test_missing_declared_c_request_keeps_the_whole_scope_ledger(complete_requests, tmp_path, monkeypatch):
    calibration, held, scope = complete_requests
    monkeypatch.setattr(module, 'open_brainsync_calibration', lambda *a, **k: pytest.fail('short roster read data'))
    with pytest.raises(module.AdultProgrammeInputError, match='complete declared calibration') as caught:
        module.open_adult_programme_inputs(calibration[:1], held, scope=scope)
    failure = caught.value.record
    assert [row['session_id'] for row in failure['requested_roster']] == list(scope.calibration_session_ids+scope.evaluation_session_ids)
    assert failure['requested_roster'][1]['status'] == 'missing_or_duplicate_request_description'
    assert failure['provided_C_label_request_counts_complete'] is False
    assert failure['actual_raw_reads_granted_before_failure'] == 0
    assert failure['partial_inputs_grant_live_programme'] is False and failure['scientific_fit_failure'] is False
    caught.value.save(tmp_path/'missing-C-request.json')


@pytest.mark.parametrize('age', [None, '17'])
def test_actual_adult_evidence_gate_precedes_eeg_grants(complete_requests, monkeypatch, age):
    calibration, held, scope = complete_requests
    path = held[0].path/'session.json'
    manifest = read_json(path)
    manifest['experiment']['age'] = age
    write_json(path, manifest)

    def forbidden_read(*args, **kwargs):
        pytest.fail('adult evidence failure reached numerical raw read')

    monkeypatch.setattr(authority._RecordingIssuer, 'issue_raw', forbidden_read)
    with pytest.raises(module.AdultProgrammeInputError, match='adult_only') as caught:
        module.open_adult_programme_inputs(calibration, held, scope=scope)
    assert caught.value.record['stage'] == 'adult_complete_population_scope'
    assert caught.value.record['failure_scope'] == 'complete_programme'
    assert all(row['status'] == 'original_metadata_admitted' for row in caught.value.record['requested_roster'])
    assert len(caught.value.record['requested_roster']) == 3
    assert caught.value.record['actual_raw_reads_granted_before_failure'] == 0
    assert caught.value.record['actual_C_labels_granted_before_failure'] == 0


def test_same_geometry_different_original_c_cannot_swap_capabilities(complete_requests):
    calibration, held, scope = complete_requests
    with module.open_adult_programme_inputs(calibration, held, scope=scope) as programme:
        first, second = programme.calibration
        original = first.segments
        object.__setattr__(first, 'segments', (second.segments[0], original[1]))
        try:
            with pytest.raises(ValueError, match='actual original raw capability'):
                module.require_adult_programme_inputs(programme)
        finally:
            object.__setattr__(first, 'segments', original)
        # Matching geometry and a second genuine record do not authorize
        # rewriting the first record's issued metadata/label identity as well.
        names = ('role', 'session_id', 'record_id', 'segments', 'calibration_labels', 'record')
        saved = {name: getattr(first, name) for name in names}
        for name in names:
            object.__setattr__(first, name, getattr(second, name))
        try:
            with pytest.raises(ValueError, match='metadata changed its original programme binding'):
                module.require_adult_programme_inputs(programme)
        finally:
            for name, value in saved.items():
                object.__setattr__(first, name, value)


def test_independent_held_copy_of_c_eeg_bytes_is_rejected_before_numerical_reads(complete_requests, monkeypatch):
    calibration, _, scope = complete_requests
    held_path = original_session(calibration[1].path.parent/'equal-size-held', scope.evaluation_session_ids[0],
        started='2026-10-01T09:00:00Z')
    before = read_json(calibration[1].path/'session.json')
    after = read_json(held_path/'session.json')
    shutil.copyfile(calibration[1].path/before['recording']['path'], held_path/after['recording']['path'])
    held = (module.HeldRecordingRequest(scope.evaluation_session_ids[0], held_path, read_requests(held_path)),)
    monkeypatch.setattr(authority._RecordingIssuer, 'issue_raw', lambda *a, **k: pytest.fail('overlap read numerical data'))
    with pytest.raises(module.AdultProgrammeInputError, match='same original EEG bytes') as caught:
        module.open_adult_programme_inputs(calibration, held, scope=scope)
    assert caught.value.record['stage'] == 'original_EEG_file_union'
    assert len(caught.value.record['requested_roster']) == 3
    assert caught.value.record['actual_raw_reads_granted_before_failure'] == 0


def test_label_available_but_original_eeg_support_missing_retains_both_facts(complete_requests):
    calibration, held, scope = complete_requests
    original = calibration[0]
    short = module.CalibrationRecordingRequest(original.session_id, original.path,
        (module.NativeReadRequest(0, 250, 1500, 7625),), 20, original.ready_at_timestamp_s)
    with pytest.raises(module.AdultProgrammeInputError, match='original granted EEG observation support') as caught:
        module.open_adult_programme_inputs((short, calibration[1]), held, scope=scope)
    failure = caught.value.record
    assert failure['requested_C_labels'] == 65
    assert failure['actual_raw_reads_granted_before_failure'] == 1
    assert failure['actual_C_labels_granted_before_failure'] == 20
    assert failure['provided_C_label_request_counts_complete'] is True
    assert len(failure['requested_roster']) == 3
    assert failure['partial_inputs_grant_live_programme'] is False
