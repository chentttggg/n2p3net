from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from data.bi2014a_candidate import (
    BI2014A_SOURCE_BIDIRECTIONAL_PROFILE,
    assert_bi2014a_candidate_input_contract,
    build_bi2014a_prepared_participant,
    build_bi2014a_subject_dataset,
)
from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.epochs import load_epoch_dataset, preprocessing_spec_from_contract, save_epoch_dataset


def subject(tmp_path, *, gap=900, tail=0, offset=0.):
    directory = tmp_path / 'subject_01'
    directory.mkdir()
    onsets, codes, markers = [], [], []
    position = 1024
    for episode in range(7):
        row, col = episode % 6, (episode + 2) % 6
        block = [60 + row, 80 + col,
                 *[20 + i for i in range(6) if i != row],
                 *[40 + i for i in range(6) if i != col]]
        for code in block:
            onsets.append(position)
            codes.append(code)
            position += 96
        markers.append(position - 48)
        position += gap
    if tail:
        for code in [60, 21][:tail]:
            onsets.append(position)
            codes.append(code)
            position += 96
    rows = np.zeros((position + (10 if tail else 900), 19))
    rows[:, 1:17] = offset + np.random.default_rng(2026).normal(0, 3, (len(rows), 16))
    rows[onsets, 17] = codes
    rows[onsets, 18] = np.asarray(codes) >= 60
    rows[onsets, 18] += 1
    rows[markers, 17] = 104
    path = directory / 'subject_01.csv'
    pd.DataFrame(rows).to_csv(path, header=False, index=False)
    return directory, rows


def spec():
    return preprocessing_spec_from_contract(BI2014A_SOURCE_BIDIRECTIONAL_PROFILE)


def test_native512_model128_actual_cz_projection_and_absent_slots(tmp_path, monkeypatch):
    directory, rows = subject(tmp_path, offset=1e8)
    import data.bi2014a_candidate as owner
    original = owner.execute_preprocessing
    checked = []

    def executed(raw, events, preprocessing, **kwargs):
        assert raw.get_data().dtype == np.float64
        original_values = pd.read_csv(directory / 'subject_01.csv', header=None).iloc[:, 1:17].to_numpy(dtype=np.float64).T * 1e-6
        wanted = original_values[[9, 10, 11, 14]] - original_values[6]
        np.testing.assert_array_equal(raw.get_data(), wanted)
        checked.append(True)
        return original(raw, events, preprocessing, **kwargs)

    monkeypatch.setattr(owner, 'execute_preprocessing', executed)
    prepared = build_bi2014a_prepared_participant(directory, preprocessing=spec())
    assert checked
    for dataset in [prepared.calibration, prepared.future]:
        assert dataset.X.shape[1:] == (6, 179)
        np.testing.assert_array_equal(dataset.channel_mask, [False, True, True, True, False, True])
        assert not dataset.X[:, [0, 4]].any()
        assert dataset.provenance['source_reference'] == 'measured_channel:CZ'
        assert dataset.provenance['acquisition_reference'] == 'right earlobe'
        assert dataset.provenance['source_sample_rate_hz'] == 512
        assert dataset.preprocessing.sfreq == 128
        assert dataset.preprocessing.baseline_mode == 'none'
    assert prepared.calibration.y is not None and prepared.future.y is None
    assert not {'raw_target_label', 'target_row', 'target_col', 'raw_is_target'} & set(prepared.future.metadata)


def test_scope_purge_mapping_and_future_perturbation_do_not_change_c(tmp_path):
    directory, rows = subject(tmp_path, gap=0)
    first = build_bi2014a_prepared_participant(directory, preprocessing=spec())
    assert first.ledger['calibration_episode_count'] == 5
    assert first.ledger['calibration_last_raw_sample_exclusive'] <= first.ledger['common_future_first_raw_sample']
    assert (first.calibration.event_timeline.evidence_indices < 0).any()
    for dataset in [first.calibration, first.future]:
        evidence = dataset.event_timeline.evidence_indices
        available = evidence >= 0
        np.testing.assert_array_equal(np.sort(evidence[available]), np.arange(dataset.n_epochs))
        np.testing.assert_array_equal(dataset.metadata.flash_sample.to_numpy()[evidence[available]],
                                      dataset.event_timeline.onset_samples[available])
    scopes = first.ledger['calibration_raw_scopes'] + first.ledger['future_raw_scopes']
    assert all(left['stop_sample'] <= right['start_sample'] for left, right in zip(scopes[:-1], scopes[1:], strict=True))
    start = first.ledger['common_future_first_raw_sample']
    rows[start:, 1:17] += np.random.default_rng(91).normal(0, 10000, rows[start:, 1:17].shape)
    pd.DataFrame(rows).to_csv(directory / 'subject_01.csv', header=False, index=False)
    second = build_bi2014a_prepared_participant(directory, preprocessing=spec())
    np.testing.assert_array_equal(first.calibration.X, second.calibration.X)
    assert not np.array_equal(first.future.X, second.future.X)


def test_partial_terminal_raw_schedule_survives_roundtrip_and_denominator(tmp_path):
    directory, _ = subject(tmp_path, tail=2)
    prepared = build_bi2014a_prepared_participant(directory, preprocessing=spec())
    assert prepared.ledger['dropped_tail_flashes'] == 2
    assert prepared.future.event_timeline.n_events == 26
    np.testing.assert_array_equal(prepared.future.event_timeline.evidence_indices[-2:], [-1, -1])
    assert all('terminal_incomplete_repetition' in value for value in prepared.future.event_timeline.status_details[-2:])
    assert prepared.ledger['future_decisions'][-1]['terminal_censor']
    assert not prepared.future_decisions[-1].schedule_complete
    path = save_epoch_dataset(tmp_path / 'future.npz', prepared.future, compressed=False)
    loaded = load_epoch_dataset(path)
    assert loaded.y is None
    np.testing.assert_array_equal(loaded.event_timeline.event_ids, prepared.future.event_timeline.event_ids)
    np.testing.assert_array_equal(loaded.event_timeline.evidence_indices, prepared.future.event_timeline.evidence_indices)
    np.testing.assert_array_equal(loaded.event_timeline.status_details, prepared.future.event_timeline.status_details)


@pytest.mark.parametrize('reference', ['left earlobe', 'measured_channel:FZ'])
def test_reference_claim_cannot_relabel_native_or_fake_measured_channel(tmp_path, reference):
    directory, _ = subject(tmp_path)
    with pytest.raises(ValueError, match='reference'):
        build_bi2014a_subject_dataset(directory, preprocessing=spec(),
            output_channels=('P3',), observed_raw_channels=('P3',), reference_channel='CZ', source_reference=reference)


def test_guard_keeps_original_causal_and_source_none_profiles_distinct():
    causal = preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)
    assert_bi2014a_candidate_input_contract(causal)
    assert_bi2014a_candidate_input_contract(spec())
    with pytest.raises(ValueError, match='baseline'):
        assert_bi2014a_candidate_input_contract(replace(spec(), baseline_mode='mean_only'))


def test_inference_cache_cannot_decode_public_flash_codes_into_target_labels(tmp_path):
    directory, _ = subject(tmp_path)
    prepared = build_bi2014a_prepared_participant(directory, preprocessing=spec())
    future = prepared.future
    assert future.y is None
    assert set(future.event_timeline.stimulus_ids) <= set(range(12))
    assert not {'flash_code', 'raw_target_label', 'target_row', 'target_col', 'raw_is_target',
                'selection_boundary_reason'} & set(future.metadata)
    assert 'flash_schedule' not in future.provenance
    for description, decision in zip(prepared.ledger['future_decisions'], prepared.future_decisions, strict=True):
        rows = future.event_timeline.selection_ids == description['episode_id']
        np.testing.assert_array_equal(future.event_timeline.stimulus_ids[rows], decision.stimulus_indices)
    assert all(value is not None for value in prepared.ledger['parent_scoring_truth'].values())
