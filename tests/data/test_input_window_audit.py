"""Counterexamples for overlap, cached crops, and coarsened BI membership."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from data.bi2014a_candidate import BI2014A_CANDIDATE_TASK_CONTRACT
from data.candidate_task import validate_candidate_membership_metadata
from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from data.preprocess import source_epoch_support, transform_epoch_samples
from experiments.audit_bi2014a_window import audit_zip, following_onsets


def test_count_actual_onsets_instead_of_dividing_by_median_interval():
    samples = np.array([0, 200, 400, 600, 1200, 1400])
    np.testing.assert_array_equal(
        following_onsets(samples, sfreq=1000, tmax_ms=600), [2, 2, 1, 0, 1, 0])
    np.testing.assert_array_equal(
        following_onsets(samples, sfreq=1000, tmax_ms=1200), [3, 3, 3, 2, 1, 0])


@pytest.mark.parametrize("samples", [[1, 1], [2, 1], [-1, 1], [0.5, 1.5]])
def test_onset_count_rejects_ambiguous_sample_indices(samples):
    with pytest.raises(ValueError, match="integer samples"):
        following_onsets(np.array(samples), sfreq=512, tmax_ms=1200)


def test_raw_zip_audit_checks_labels_and_retains_tail_in_overlap_count(tmp_path):
    import zipfile

    codes = [60, 21, 22, 23, 24, 25, 80, 41, 42, 43, 44, 45, 21]
    table = np.zeros((14 * 128, 19))
    onsets = np.arange(13) * 128
    table[onsets, 17] = codes
    table[onsets, 18] = np.where(np.array(codes) >= 60, 2, 1)
    path = tmp_path / "subject_01.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("subject_01.csv", pd.DataFrame(table).to_csv(header=False, index=False))
    record = audit_zip(path)
    assert record["n_complete_repetitions"] == 1
    assert record["incomplete_tail_flashes"] == 1
    assert record["within_repetition_ms"]["median"] == 250
    assert record["between_repetition_ms"]["n"] == 0
    assert record["following_onsets"]["600"]["n"] == 13


def test_adjacent_epochs_share_raw_samples_even_before_filtering():
    spec = preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)
    events = np.array([[512, 0, 1], [658, 0, 1]])  # 285.15625 ms apart
    start, stop = source_epoch_support(events, 512, spec)
    assert start[1] < stop[0]
    raw = np.zeros((1, 1600))
    raw[0, events[1, 0] + 100] = 1e-5  # response to the later stimulus
    epochs = np.stack([raw[:, a:b] for a, b in zip(start, stop, strict=True)])
    transformed = transform_epoch_samples(epochs, 512, spec)
    assert np.max(np.abs(transformed[0])) > 1e-7
    assert np.max(np.abs(transformed[1])) > 1e-7


def test_cropping_long_fft_cache_does_not_remove_future_sample_dependency():
    long = preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)
    short_contract = replace(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT, tmax_ms=600)
    short = preprocessing_spec_from_contract(short_contract)
    event = np.array([[512, 0, 1]])
    start, long_stop = source_epoch_support(event, 512, long)
    _, short_stop = source_epoch_support(event, 512, short)
    values = np.zeros((1, 1, int(long_stop[0] - start[0])))
    values[..., int(round((.9 + .2) * 512))] = 1.0  # only at +900 ms
    crop = transform_epoch_samples(values, 512, long)[..., :short.n_times]
    rebuilt = transform_epoch_samples(values[..., :int(short_stop[0] - start[0])], 512, short)
    assert short.n_times == 102 and long.n_times == 179
    assert np.max(np.abs(crop)) > 1e-5
    np.testing.assert_array_equal(rebuilt, 0)


def test_coarse_three_by_three_membership_rejects_original_binary_labels():
    contract = replace(BI2014A_CANDIDATE_TASK_CONTRACT,
                       task_id="counterexample_coarsened_3x3", grid_shape=(3, 3),
                       candidate_ids=tuple(range(6)), row_candidate_ids=tuple(range(3)),
                       column_candidate_ids=tuple(range(3, 6)))
    original = np.arange(6)
    metadata = pd.DataFrame({
        "candidate_id": np.r_[original // 2, 3 + original // 2],
        "row_code": np.r_[original // 2, np.full(6, -1)],
        "col_code": np.r_[np.full(6, -1), original // 2],
        "target_row": 0, "target_col": 0, "selection_id": "s0", "repetition_index": 0,
    })
    labels = np.r_[original == 0, original == 0].astype(int)
    coarse_labels = np.r_[original // 2 == 0, original // 2 == 0].astype(int)
    assert labels.sum() == 2 and coarse_labels.sum() == 4
    with pytest.raises(ValueError, match="labels disagree with candidate-membership"):
        validate_candidate_membership_metadata(metadata, contract, labels=labels)
