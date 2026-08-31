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
        # Three repetitions per selection, then move to a new character.
        row = (rep // 3) % 6
        col = (rep // 3) % 6
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
    assert record.n_explicit_boundaries == 0
    assert record.dropped_tail_flashes == 0
    assert set(np.unique(record.row_code).tolist()) == {-1, 0, 1, 2, 3, 4, 5}
    assert set(np.unique(record.col_code).tolist()) == {-1, 0, 1, 2, 3, 4, 5}
    assert len(np.unique(record.selection_id)) == 33
    assert set(np.unique(record.repetition_index).tolist()) == {0, 1, 2}
    for selection in np.unique(record.selection_id):
        rows = record.selection_id == selection
        assert np.array_equal(
            np.unique(record.repetition_index[rows]), np.arange(3, dtype=np.int64)
        )
        assert all(
            np.count_nonzero(rows & (record.repetition_index == repetition)) == 12
            for repetition in range(3)
        )
    target_rows = record.target_label == 2
    target_cols = target_rows
    assert np.all(record.row_code[target_rows & (record.row_code >= 0)] == record.target_row[target_rows & (record.row_code >= 0)])
    assert np.all(record.col_code[target_cols & (record.col_code >= 0)] == record.target_col[target_cols & (record.col_code >= 0)])


def test_recover_bi2014a_splits_same_target_at_raw_level_boundary(tmp_path) -> None:
    subject_dir = tmp_path / "subject_01"
    subject_dir.mkdir()
    row, col = 2, 4
    codes = np.asarray(
        [
            *(code for code in range(20, 26) if code != 20 + row),
            60 + row,
            *(code for code in range(40, 46) if code != 40 + col),
            80 + col,
        ],
        dtype=np.int64,
    )
    raw_rows: list[np.ndarray] = []
    for repetition in range(3):
        for code in codes:
            raw = np.zeros(19, dtype=float)
            raw[17] = code
            raw[18] = 2 if code >= 60 else 1
            raw_rows.append(raw)
        if repetition == 1:
            boundary = np.zeros(19, dtype=float)
            boundary[17] = 104
            raw_rows.append(boundary)
    pd.DataFrame(raw_rows).to_csv(
        subject_dir / "subject_01.csv", index=False, header=False
    )

    record = recover_bi2014a_candidates(subject_dir)

    assert record.n_explicit_boundaries == 1
    assert len(np.unique(record.selection_id)) == 2
    assert record.repetition_index.reshape(3, 12)[:, 0].tolist() == [0, 1, 0]
    assert (
        np.unique(record.selection_boundary_reason.reshape(3, 12)[2]).tolist()
        == ["raw_level_or_restart_code_104"]
    )


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
