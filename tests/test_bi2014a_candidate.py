from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.bi2014a_candidate import recover_bi2014a_candidates


def _write_synthetic_subject(tmp_path) -> None:
    subject_dir = tmp_path / "subject_01"
    subject_dir.mkdir()
    rows = np.zeros((1188, 19), dtype=float)
    rng = np.random.default_rng(0)
    for rep in range(99):
        row = rep % 6
        col = rep % 6
        target_row_code = 60 + row
        target_col_code = 80 + col
        nontarget_rows = [code for code in range(20, 26) if code != 20 + row][:5]
        nontarget_cols = [code for code in range(40, 46) if code != 40 + col][:5]
        codes = np.asarray(nontarget_rows + nontarget_cols + [target_row_code, target_col_code])
        rng.shuffle(codes)
        start = rep * 12
        rows[start : start + 12, 17] = codes
        rows[start : start + 12, 18] = np.where((codes >= 60) & (codes <= 85), 2, 1)
    pd.DataFrame(rows).to_csv(subject_dir / "subject_01.csv", index=False, header=False)


def test_recover_bi2014a_synthetic_schedule(tmp_path) -> None:
    _write_synthetic_subject(tmp_path)
    record = recover_bi2014a_candidates(tmp_path / "subject_01")

    assert len(record.flash_sample) == 1188
    assert np.count_nonzero(record.target_label == 2) == 198
    assert np.count_nonzero(record.target_label == 1) == 990
    assert record.n_repetitions == 99
    assert record.dropped_tail_flashes == 0
    assert set(np.unique(record.row_code).tolist()) == {-1, 0, 1, 2, 3, 4, 5}
    assert set(np.unique(record.col_code).tolist()) == {-1, 0, 1, 2, 3, 4, 5}
    assert int(record.repetition_index.max()) < 99
    target_rows = record.target_label == 2
    target_cols = target_rows
    assert np.all(record.row_code[target_rows & (record.row_code >= 0)] == record.target_row[target_rows & (record.row_code >= 0)])
    assert np.all(record.col_code[target_cols & (record.col_code >= 0)] == record.target_col[target_cols & (record.col_code >= 0)])


def test_recover_bi2014a_keeps_complete_repetitions_and_drops_tail(tmp_path) -> None:
    _write_synthetic_subject(tmp_path)
    path = tmp_path / "subject_01" / "subject_01.csv"
    table = pd.read_csv(path, header=None)
    table.loc[len(table)] = 0.0
    table.iloc[-1, 17] = 21
    table.iloc[-1, 18] = 1
    table.to_csv(path, index=False, header=False)

    record = recover_bi2014a_candidates(tmp_path / "subject_01")

    assert record.n_repetitions == 99
    assert record.dropped_tail_flashes == 1
    assert len(record.flash_sample) == 1188
    assert np.count_nonzero(record.target_label == 2) == 198


def test_recover_bi2014a_rejects_invalid_flash_structure(tmp_path) -> None:
    _write_synthetic_subject(tmp_path)
    path = tmp_path / "subject_01" / "subject_01.csv"
    table = pd.read_csv(path, header=None)
    table.iloc[0, 17] = 21
    table.to_csv(path, index=False, header=False)

    with pytest.raises(ValueError, match="invalid flash structure"):
        recover_bi2014a_candidates(tmp_path / "subject_01")
