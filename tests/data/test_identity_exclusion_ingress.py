"""Real identity consumers without optimizers, scientific fits or EEG files."""
from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from data.epochs import load_epoch_dataset, save_epoch_dataset
from data.exclusions import assert_identity_participants_allowed
from data.identity import (
    DatasetIdentityTable,
    ParticipantIdentityRecord,
    assert_target_identity_excluded,
    origin_subject_key,
    source_dataset_ids_from_identity,
)
from models.n2p3net import N2P3ArchitectureConfig, N2P3Net
from tests.support.checkpoints import checkpoint_identity_fields
from tests.support.datasets import binary_dataset, epoch_dataset
from transfer.calibrated import (
    CalibratedPredictor,
    load_calibrated_predictor,
    save_calibrated_predictor,
)
from transfer.checkpoint import checkpoint_training_identity_ledger, load_checkpoint_payload


def _opaque_table(subjects):
    return DatasetIdentityTable(tuple(
        ParticipantIdentityRecord(subject, (f'producer-authority:{subject}',), (), 'source_verified')
        for subject in subjects
    ))


@pytest.mark.parametrize('validation', ['full', 'attested'])
def test_general_dataset_opaque_identity_roundtrips_through_real_cache(tmp_path, validation):
    dataset = epoch_dataset()
    dataset.identity_table = _opaque_table(['s1', 's2', 's3'])
    dataset.validate()
    path = save_epoch_dataset(tmp_path / 'opaque.npz', dataset)
    loaded = load_epoch_dataset(path, validation=validation)
    assert loaded.identity_table == dataset.identity_table
    np.testing.assert_array_equal(loaded.X, dataset.X)
    with pytest.raises(ValueError, match='canonical origin'):
        source_dataset_ids_from_identity(loaded.identity_table)


def test_general_checkpoint_opaque_ledger_roundtrips_with_authority_identity_intact(tmp_path):
    dataset = binary_dataset(count=4)
    ledger = _opaque_table(['source-person'])
    payload = {'trunk_state_dict': {'fixture': torch.ones(1)},
               **checkpoint_identity_fields(ledger, dataset)}
    path = tmp_path / 'source.pt'
    torch.save(payload, path)
    loaded = load_checkpoint_payload(path)
    assert checkpoint_training_identity_ledger(loaded) == ledger
    assert loaded['training_contract']['training_participant_keys'] == list(ledger.authority_keys())
    target = ParticipantIdentityRecord('new-local-name', ('producer-authority:source-person',), (), 'source_verified')
    with pytest.raises(ValueError, match='shared source identity'):
        assert_target_identity_excluded(ledger, target)


def _predictor_material(authority):
    dataset = binary_dataset(count=4)
    configuration = N2P3ArchitectureConfig()
    trunk = N2P3Net(dataset.n_channels, n_times=dataset.n_times,
                    sfreq=dataset.preprocessing.sfreq, tmin_s=dataset.preprocessing.tmin_ms / 1000,
                    pooling_mode='ms_flatten', **configuration.model_kwargs())
    ledger = DatasetIdentityTable((ParticipantIdentityRecord(
        'source-person', (origin_subject_key('independent-study', 'source-person'),),
        (authority,), 'global_verified'),))
    contract = checkpoint_identity_fields(ledger, dataset)['training_contract']
    metadata = {
        'participant_key': origin_subject_key('BrainSync-GTN', 'target'),
        'model_kind': 'source_initialized_n2p3', 'source_pretraining_used': True,
        'source_training_contract': contract, 'selection': {},
        'architecture_configuration': asdict(configuration), 'pooling_mode': 'ms_flatten',
        'architecture': trunk.architecture_record(),
        'input_contract': {'n_channels': dataset.n_channels, 'n_times': dataset.n_times,
                           'preprocessing': dataset.preprocessing.record()},
    }
    state = {'trunk.' + name: value.clone() for name, value in trunk.state_dict().items()}
    state.update(input_mean=torch.zeros(1, dataset.n_channels, 1),
                 input_std=torch.ones(1, dataset.n_channels, 1))
    return metadata, state


def test_personal_predictor_accepts_real_global_training_authority_and_roundtrips(tmp_path):
    metadata, state = _predictor_material('registry:person-17')
    predictor = CalibratedPredictor(metadata, state)
    restored = load_calibrated_predictor(save_calibrated_predictor(tmp_path / 'personal.pt', predictor))
    assert restored.metadata['source_training_contract']['training_participant_keys'] == ['registry:person-17']
    assert restored.metadata == predictor.metadata
    for name, tensor in predictor.state.items():
        assert torch.equal(restored.state[name], tensor)


def test_personal_predictor_still_rejects_explicit_excluded_source_key():
    metadata, state = _predictor_material('registry:person-17')
    metadata = copy.deepcopy(metadata)
    metadata['source_training_contract']['training_participant_keys'] = [origin_subject_key('BI2014a', '62')]
    with pytest.raises(ValueError, match='excluded'):
        CalibratedPredictor(metadata, state)


def test_opaque_keys_preserve_exact_source_and_global_overlap_semantics():
    source = ParticipantIdentityRecord('source', ('opaque-source-1',), ('registry:person-17',), 'global_verified')
    table = DatasetIdentityTable((source,))
    assert_identity_participants_allowed(table)
    with pytest.raises(ValueError, match='shared source identity'):
        assert_target_identity_excluded(table, ParticipantIdentityRecord('local-alias', ('opaque-source-1',), (), 'source_verified'), policy='source')
    with pytest.raises(ValueError, match='shared global identity'):
        assert_target_identity_excluded(table, ParticipantIdentityRecord('other-stream', ('opaque-source-2',), ('registry:person-17',), 'global_verified'))
