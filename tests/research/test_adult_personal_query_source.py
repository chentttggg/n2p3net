"""Read-only new-session exclusion, using existing tiny source13 evidence."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data.brainsync_raw_authority import BrainSyncRecordingScope
from research import adult_population_source as module
from research import population_response_forecast as population
from research import public_population_source as source_owner
from research import source_data_factor
from research import source_likelihood_export as export
from research.adult_programme_input import HeldRecordingRequest, open_adult_programme_inputs
from tests.support.adult_programme import build_complete_requests, original_session, read_requests
from tests.support.io import read_json, write_json


def artifact(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    return {'path': str(path), 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


@pytest.fixture(scope='module')
def original_software_source():
    configured = os.environ.get('N2P3_FROZEN_ADULT_SOURCE')
    if configured is None:
        pytest.skip('explicit original tiny software adult-source artifact is required')
    reference = artifact(configured)
    record = json.loads(Path(configured).read_bytes())
    acceptance = json.loads(Path(record['independent_acceptance']['path']).read_bytes())
    assert acceptance['not_a_scientific_acceptance'] is True
    forecast_record = json.loads(Path(record['forecast']['path']).read_bytes())
    assert len(forecast_record['state']['chart']['coefficient_names']) <= 16, 'software controls do not load scientific payloads'
    store = export.SharedLikelihoodStore(forecast_record['shared_store'])
    execution = source_owner.load_source_population_execution(record['accepted_source_execution'],
        trusted_acceptance_reference=record['independent_acceptance'], expected_matrix_reference=record['source_matrix'])
    frozen = module.FrozenAdultPopulationSource.load(reference, shared_store=store)
    return frozen, execution, store


@pytest.fixture
def new_session_requests(tmp_path):
    calibration, _, _ = build_complete_requests(tmp_path/'old-C-builders')
    held = original_session(tmp_path/'new-query', 'adult-new-personal-query', started='2026-10-20T10:00:00Z')
    manifest = read_json(held/'session.json')
    manifest['experiment']['thought_digit'] = {'private': 'unavailable'}
    manifest['target_label'] = {'status': 'withheld', 'thought_digit': 'unavailable'}
    write_json(held/'session.json', manifest)
    inference = (HeldRecordingRequest(manifest['session_id'], held, read_requests(held)),)
    scope = BrainSyncRecordingScope(tuple(row.session_id for row in calibration), tuple(row.session_id for row in inference))
    return calibration, inference, scope


def fold_files(directory):
    return {str(path.relative_to(directory)): artifact(path) for path in Path(directory).glob('*.fold.json')}


def forbid_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('read-only personal query attempted a new source/C/population fit')

    for owner, names in ((population, ('fit_population_variance', 'infer_population')),
            (source_owner, ('prepare_fold_response_reference', 'fit_shared_population_fold', 'fit_and_export_source_person')),
            (source_data_factor, ('fit_source_data_factor',))):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)


def test_actual_new_held_same_person_readonly_relation(original_software_source, new_session_requests, monkeypatch):
    frozen, execution, store = original_software_source
    before = fold_files(store.directory)
    original_record = module.metadata_digest(frozen.record)
    original_producer = frozen.native.forecast.producer_sha256
    forbid_fitting(monkeypatch)
    monkeypatch.setattr(store, '_write_once', lambda *a, **k: pytest.fail('read-only query reached the store writer'))
    monkeypatch.setattr(store, 'put_fold_references', lambda *a, **k: pytest.fail('read-only query published a new fold'))
    calibration, held, scope = new_session_requests
    with open_adult_programme_inputs(calibration, held, scope=scope) as inputs:
        relation = module.admit_adult_personal_query_source(frozen, source_execution=execution,
            adult_inputs=inputs, shared_store=store)
        assert module.require_adult_personal_query_source(relation) is relation
        assert relation.adult_inputs is inputs
        assert relation.original_source.record == frozen.record
        assert module.metadata_digest(relation.original_source.record) == original_record
        assert relation.original_source.native.forecast.producer_sha256 == original_producer
        assert relation.original_source.native.forecast.input_identity == frozen.native.forecast.input_identity
        np.testing.assert_array_equal(relation.original_source.native.forecast.mean, frozen.native.forecast.mean)
        np.testing.assert_array_equal(relation.original_source.native.forecast.between_factor, frozen.native.forecast.between_factor)
        np.testing.assert_array_equal(relation.original_source.native.forecast.mean_factor, frozen.native.forecast.mean_factor)
        assert relation.record['original_adult_programme_identity'] != inputs.identity
        assert relation.record['validation_only_fold']['fold_id'] != frozen.native.forecast.fold.fold_id
        assert set(inputs.exclusion_contract['held_record_ids']).isdisjoint(frozen.record['source_exclusion']['held_record_ids'])
        assert relation.record['new_source_fits'] == relation.record['new_C_fits'] == relation.record['new_fold_metadata_publications'] == 0
        assert relation.record['raw_or_label_or_source_or_C_fit_authority_granted'] is False
        with pytest.raises(TypeError, match='original source and actual fresh'):
            module.require_adult_personal_query_source(copy.copy(relation))
        with pytest.raises(TypeError, match='original source and actual fresh'):
            module.require_adult_personal_query_source(SimpleNamespace(record=relation.record))
        with pytest.raises(TypeError, match='original adult full-source'):
            module.require_adult_population_source(relation)
        with pytest.raises(TypeError, match='original frozen'):
            module.admit_adult_personal_query_source(frozen.record, source_execution=execution,
                adult_inputs=inputs, shared_store=store)
        with pytest.raises(TypeError, match='original complete canonical'):
            module.admit_adult_personal_query_source(frozen, source_execution=execution,
                adult_inputs=copy.copy(inputs), shared_store=store)
    assert fold_files(store.directory) == before


def test_other_actual_person_is_rejected(original_software_source, new_session_requests):
    frozen, execution, store = original_software_source
    calibration, held, scope = new_session_requests
    for request in (*calibration, *held):
        path = request.path/'session.json'
        manifest = read_json(path)
        manifest['experiment']['subject_id'] = 'P002'
        write_json(path, manifest)
    before = fold_files(store.directory)
    with open_adult_programme_inputs(calibration, held, scope=scope) as inputs:
        assert inputs.exclusion_contract['source_subject_id'] == 'P002'
        with pytest.raises(ValueError, match='same original adult participant'):
            module.admit_adult_personal_query_source(frozen, source_execution=execution, adult_inputs=inputs, shared_store=store)
    assert fold_files(store.directory) == before


def numeric_control(store, reference, directory, *, alias=False):
    """An explicit numeric collision fixture, never a source publication."""
    original = store.get(reference)
    metadata = copy.deepcopy(original.metadata)
    metadata['numeric_control_only'] = True
    metadata['exporter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if alias:
        metadata['identity_keys'].append('adult-query-alias')
    value = export.ExportedSourceLikelihood(original.matrix, original.response, original.log_constant, json.dumps(metadata))
    isolated = export.SharedLikelihoodStore(directory)
    numeric_reference = isolated.put(value)
    expected = tuple((numeric_reference.participant_key, row['record_id'], row['source_sha256'], row['labels'])
        for row in metadata['block_constants'])
    fold = population.ParticipantFold('numeric-query-control', ('adult-query-alias',), (numeric_reference.participant_key,),
        expected, (), (), '1'*64)
    return isolated, numeric_reference, fold


@pytest.mark.parametrize('contamination', ('alias', 'EEG', 'record'))
def test_shared_pure_validator_rejects_source_contamination(original_software_source, tmp_path, contamination):
    _, execution, store = original_software_source
    reference = export.SharedLikelihoodReference(**module.plain_metadata(execution.record['persons'][0]['shared_likelihood']))
    isolated, numeric_reference, fold = numeric_control(store, reference, tmp_path/'numeric', alias=contamination == 'alias')
    first = isolated.get(numeric_reference).metadata['block_constants'][0]
    if contamination == 'EEG':
        fold = replace(fold, excluded_eeg_sha256s=(first['source_sha256'],))
    elif contamination == 'record':
        fold = replace(fold, excluded_record_ids=(first['record_id'],))
    before = fold_files(isolated.directory)
    with pytest.raises(ValueError, match='excluded target' if contamination == 'alias' else 'excluded original target'):
        isolated.validate_fold_references(fold, (numeric_reference,))
    assert fold_files(isolated.directory) == before == {}


def test_original_put_record_bytes_and_errors_are_preserved(original_software_source, tmp_path):
    _, execution, store = original_software_source
    preimage = Path(os.environ['N2P3_SOURCE_EXPORT_PREIMAGE'])
    specification = importlib.util.spec_from_file_location('original_source_export_readonly_test', preimage)
    original_module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = original_module
    specification.loader.exec_module(original_module)
    reference = export.SharedLikelihoodReference(**module.plain_metadata(execution.record['persons'][0]['shared_likelihood']))
    current, current_ref, fold = numeric_control(store, reference, tmp_path/'current')
    legacy = original_module.SharedLikelihoodStore(tmp_path/'legacy')
    legacy_ref = legacy.put(current.get(current_ref))
    assert current_ref.record() == legacy_ref.record()
    pure = current.validate_fold_references(fold, (current_ref,))
    current_fold = current.put_fold_references(fold, (current_ref,))
    original_fold = legacy.put_fold_references(fold, (legacy_ref,))
    assert current_fold == original_fold
    assert (current.directory/current_fold['file']).read_bytes() == (legacy.directory/original_fold['file']).read_bytes()
    assert json.loads((current.directory/current_fold['file']).read_bytes()) == pure
    bad = replace(fold, excluded_eeg_sha256s=(fold.expected_source_records[0][2],))
    with pytest.raises(ValueError) as old_error:
        legacy.put_fold_references(bad, (legacy_ref,))
    with pytest.raises(ValueError) as new_error:
        current.put_fold_references(bad, (current_ref,))
    assert str(new_error.value) == str(old_error.value)


def test_original_13_people_and_13_forecasts_cold_restore_without_fit_or_write(original_software_source, monkeypatch):
    frozen, execution, clone_store = original_software_source
    original_store_path = json.loads(Path(execution.record['folds'][0]['forecast']['path']).read_bytes())['shared_store']
    original_store = export.SharedLikelihoodStore(original_store_path)
    before = fold_files(original_store.directory), fold_files(clone_store.directory)
    forbid_fitting(monkeypatch)
    monkeypatch.setattr(export.SharedLikelihoodStore, '_write_once', lambda *a, **k: pytest.fail('cold restore reached the store writer'))
    for person in execution.record['persons']:
        cold = source_data_factor.load_source_data_factor(person['source_product'], shared_store=original_store)
        assert cold.likelihood.metadata['exporter_sha256'] != export.IMPLEMENTATION_SHA256
        assert cold.likelihood.metadata['source_producer_sha256'] == cold.product['producer_sha256']
    for fold in execution.record['folds']:
        forecast = source_owner.load_shared_forecast(fold['forecast'], shared_store=original_store)
        assert forecast.producer_sha256 == population.IMPLEMENTATION_SHA256
        assert tuple(v.participant_key for v in forecast.likelihoods) == tuple(fold['source_participant_keys'])
        assert all(row.original_producer_sha256 != export.IMPLEMENTATION_SHA256 for row in forecast.likelihoods)
    restored = module.FrozenAdultPopulationSource.load(frozen.source_reference, shared_store=clone_store)
    assert restored.record == frozen.record
    assert (fold_files(original_store.directory), fold_files(clone_store.directory)) == before
