from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.channel import build_channel_identity
from data.epochs import EpochDataset, PreprocessingSpec
from data.events import observed_only_timeline
from transfer.bi_decision import hit_at_repetition_6x6
from transfer.bi_within_subject import bi2014a_prefix_suffix_split


def test_hit_at_repetition_6x6_target_row_and_column() -> None:
    n = 24
    selection = np.repeat("s1", n)
    repetitions = np.concatenate([np.zeros(12, dtype=np.int64), np.ones(12, dtype=np.int64)])
    # Each repetition: all six rows and all six columns.
    row_flashes = np.asarray([20, 21, 22, 23, 24, 25], dtype=np.int64)
    col_flashes = np.asarray([40, 41, 42, 43, 44, 45], dtype=np.int64)
    flash = np.tile(np.concatenate([row_flashes, col_flashes]), 2)
    logits = np.zeros(n, dtype=float)
    logits[flash == 21] = 2.0
    logits[flash == 42] = 2.0
    target_rows = np.full(n, 1, dtype=np.int64)
    target_cols = np.full(n, 2, dtype=np.int64)
    hits = hit_at_repetition_6x6(
        logits,
        flash,
        target_rows,
        target_cols,
        selection,
        repetitions,
        max_repetitions=2,
    )
    assert hits[1] == 1.0
    assert hits[2] == 1.0


def _bi_dataset() -> EpochDataset:
    n = 48
    selection = np.repeat(["s1", "s2"], 24)
    repetition = np.tile(
        np.concatenate([np.zeros(12, dtype=np.int64), np.ones(12, dtype=np.int64)]),
        2,
    )
    flash = np.tile(
        np.concatenate(
            [np.arange(20, 26, dtype=np.int64), np.arange(40, 46, dtype=np.int64)]
        ),
        4,
    )
    target_row = np.where(selection == "s1", 1, 4).astype(np.int64)
    target_col = np.where(selection == "s1", 2, 5).astype(np.int64)
    row_code = np.where((flash >= 20) & (flash <= 25), flash - 20, -1)
    col_code = np.where((flash >= 40) & (flash <= 45), flash - 40, -1)
    y = np.where(
        ((flash >= 20) & (flash <= 25)) & (row_code == target_row),
        1,
        np.where(((flash >= 40) & (flash <= 45)) & (col_code == target_col), 1, 0),
    ).astype(np.int64)
    identity = build_channel_identity(("Fz", "Cz", "Pz"), allow_missing_positions=False)
    timeline = observed_only_timeline(
        dataset_id="synthetic_bi",
        subject_ids=selection,
        stimulus_ids=flash,
        onset_times_s=np.arange(n, dtype=float),
        group_ids=selection,
        online_causal=True,
        timing_source="synthetic_bi",
    )
    return EpochDataset(
        name="synthetic_bi",
        X=np.zeros((n, 3, 128), dtype=np.float32),
        y=y,
        subject_ids=selection.astype(str),
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(3, dtype=bool),
        preprocessing=PreprocessingSpec(
            name="p300_single_subject_causal_v1",
            sfreq=128.0,
            l_freq=2.0,
            h_freq=30.0,
            tmin_ms=-200.0,
            tmax_ms=800.0,
            n_times=128,
            baseline_mode="mean_only",
            filter_phase="forward",
        ),
        event_timeline=timeline,
        metadata=pd.DataFrame(
            {
                "subject": selection,
                "flash_code": flash,
                "row_code": row_code,
                "col_code": col_code,
                "target_row": target_row,
                "target_col": target_col,
                "selection_id": selection,
                "repetition_index": repetition,
            }
        ),
        provenance={"source": "unit_test", "source_reference": "average", "source_sample_rate_hz": 128.0},
    )


def test_bi_prefix_suffix_split_requires_complete_repetitions() -> None:
    dataset = _bi_dataset()
    split = bi2014a_prefix_suffix_split(dataset, prefix_repetitions=1, test_repetitions=1)
    assert set(split.usable_selections) == {"s1", "s2"}
    assert int(split.prefix_mask.sum()) == 24
    assert int(split.suffix_mask.sum()) == 24
    assert set(np.unique(split.suffix_repetition_indices[split.suffix_mask]).tolist()) == {0}
    with pytest.raises(ValueError, match="No BI selection"):
        bi2014a_prefix_suffix_split(dataset, prefix_repetitions=2, test_repetitions=2)
