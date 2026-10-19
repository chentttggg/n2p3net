"""Generic cache saves attest the representation actually serialized."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import data.epochs as owner
from tests.support.datasets import reference_dataset


@pytest.mark.parametrize('field', ('X', 'y', 'channel_positions_m', 'qc_features', 'empty_metadata'))
@pytest.mark.parametrize('validation', ('full', 'attested'))
def test_generic_writer_stored_state_roundtrip_without_mutating_input_arrays(tmp_path, field, validation):
    dataset = reference_dataset(np.random.default_rng(817).normal(0., 1e-6, (12, 3, 4)).astype(np.float32),
                                reference='nose')
    if field in ('X', 'channel_positions_m'):
        setattr(dataset, field, np.asarray(getattr(dataset, field)).astype(np.float64))
    elif field == 'y':
        dataset.y = dataset.y.astype(np.uint8)
    elif field == 'qc_features':
        from data.qc_features import compute_epoch_qc_features

        qc = compute_epoch_qc_features(dataset.X, channel_mask=dataset.channel_mask)
        dataset.qc_features = replace(qc, relative_ptp=qc.relative_ptp.astype(np.float64),
                                     channel_std_v=qc.channel_std_v.astype(np.float64),
                                     epoch_scale_v=qc.epoch_scale_v.astype(np.float64))
    else:
        dataset.metadata = pd.DataFrame()
    arrays = {name: (getattr(dataset, name), np.asarray(getattr(dataset, name)).copy())
              for name in ('X', 'y', 'channel_positions_m')}
    path = owner.save_epoch_dataset(tmp_path / 'generic.npz', dataset)
    loaded = owner.load_epoch_dataset(path, validation=validation)
    with np.load(path, allow_pickle=False) as archive:
        embedded = str(archive['decoded_dataset_sha256'].item())
        for name in ('X', 'y', 'subject_ids', 'channel_positions_m', 'channel_mask'):
            np.testing.assert_array_equal(getattr(loaded, name), archive[name])
    assert embedded == owner._decoded_dataset_sha256(loaded)
    if validation == 'attested':
        assert owner.loaded_epoch_cache_attestation(loaded)['decoded_dataset_sha256'] == embedded
    for name, (original, values) in arrays.items():
        assert getattr(dataset, name) is original
        assert np.asarray(original).dtype == values.dtype
        np.testing.assert_array_equal(original, values)
    assert list(loaded.metadata['subject']) == list(dataset.subject_ids)


def test_generic_roundtrip_retains_decoded_content_tamper_detection(tmp_path):
    dataset = reference_dataset(np.zeros((12, 3, 4), dtype=np.float32), reference='nose')
    dataset.channel_positions_m = dataset.channel_positions_m.astype(np.float64)
    path = owner.save_epoch_dataset(tmp_path / 'generic.npz', dataset)
    loaded = owner.load_epoch_dataset(path, validation='attested')
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload['X'][0, 0, 0] = np.float32(1e-6)
    np.savez(path, **payload)
    owner.write_epoch_dataset_record(path, loaded)
    with pytest.raises(ValueError, match='decoded dataset changed'):
        owner.load_epoch_dataset(path, validation='attested')


@pytest.mark.parametrize('field', ('X', 'channel_positions_m', 'qc_features'))
def test_generic_writer_validates_values_after_storage_conversion_before_publication(tmp_path, field):
    from data.qc_features import compute_epoch_qc_features

    dataset = reference_dataset(np.zeros((12, 3, 4), dtype=np.float32), reference='nose')
    qc = compute_epoch_qc_features(dataset.X, channel_mask=dataset.channel_mask)
    dataset.qc_features = qc
    if field == 'X':
        dataset.X = np.full(dataset.X.shape, 1e100, dtype=np.float64)
    elif field == 'channel_positions_m':
        dataset.channel_positions_m = np.full(dataset.channel_positions_m.shape, 1e-100, dtype=np.float64)
    else:
        dataset.qc_features = replace(qc, relative_ptp=np.full(qc.relative_ptp.shape, 1e100, dtype=np.float64))
    dataset.validate()
    path = tmp_path / 'invalid.npz'
    with np.errstate(over='ignore'):
        with pytest.raises(ValueError, match='finite|non-zero'):
            owner.save_epoch_dataset(path, dataset)
    assert not path.exists()
    assert not path.with_suffix('.record.json').exists()
