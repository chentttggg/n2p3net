from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from data.channel import build_channel_identity
from data.contract import CAUSAL_IIR_INITIAL_STATE
from data.epochs import (
    EpochDataset,
    PreprocessingSpec,
    concatenate_epoch_datasets,
    save_epoch_dataset,
)
from data.events import observed_only_timeline
from experiments.run_bi2014a_candidate import main as run_bi_candidate_main
from models.n2p3net import N2P3Net
from transfer.bi_decision import hit_at_repetition_6x6
from transfer.bi_within_subject import (
    bi2014a_calibration_decision_split,
    bi2014a_prefix_suffix_split,
)


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


def test_hit_at_repetition_6x6_tie_abstains_instead_of_choosing_row_zero() -> None:
    flashes = np.concatenate(
        [np.arange(20, 26, dtype=np.int64), np.arange(40, 46, dtype=np.int64)]
    )
    hits = hit_at_repetition_6x6(
        np.zeros(12),
        flashes,
        np.zeros(12, dtype=np.int64),
        np.zeros(12, dtype=np.int64),
        np.repeat("selection", 12),
        np.zeros(12, dtype=np.int64),
        max_repetitions=1,
    )

    assert hits[1] == 0.0


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
            name="p300_single_subject_causal_v2",
            sfreq=128.0,
            l_freq=2.0,
            h_freq=30.0,
            tmin_ms=-200.0,
            tmax_ms=800.0,
            n_times=128,
            baseline_mode="mean_only",
            filter_phase="forward",
            causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
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


def _bi_cross_decision_dataset(subject: str = "s1") -> EpochDataset:
    n_selections = 7
    repetitions_per_selection = 3
    rows: list[dict[str, object]] = []
    labels: list[int] = []
    subject_ids: list[str] = []
    onset: list[float] = []
    cursor = 0
    for selection in range(n_selections):
        target_row = selection % 6
        target_col = (selection + 2) % 6
        for repetition in range(repetitions_per_selection):
            codes = [
                *(20 + row for row in range(6) if row != target_row),
                60 + target_row,
                *(40 + col for col in range(6) if col != target_col),
                80 + target_col,
            ]
            for code in codes:
                row_code = code - 20 if 20 <= code <= 25 else code - 60 if 60 <= code <= 65 else -1
                col_code = code - 40 if 40 <= code <= 45 else code - 80 if 80 <= code <= 85 else -1
                rows.append(
                    {
                        "subject": subject,
                        "flash_code": code,
                        "row_code": row_code,
                        "col_code": col_code,
                        "target_row": target_row,
                        "target_col": target_col,
                        "selection_id": f"{subject}:selection{selection}",
                        "repetition_index": repetition,
                    }
                )
                labels.append(int(code >= 60))
                subject_ids.append(subject)
                onset.append(cursor * 1.5)
                cursor += 1
    metadata = pd.DataFrame(rows)
    identity = build_channel_identity(("Fz", "Cz", "Pz"), allow_missing_positions=False)
    timeline = observed_only_timeline(
        dataset_id=f"synthetic-bi-decisions:{subject}",
        subject_ids=np.asarray(subject_ids),
        stimulus_ids=metadata["flash_code"].to_numpy(dtype=np.int64),
        onset_times_s=np.asarray(onset),
        evidence_available_times_s=np.asarray(onset) + 0.8,
        group_ids=metadata["selection_id"].astype(str).to_numpy(),
        selection_ids=metadata["selection_id"].astype(str).to_numpy(),
        online_causal=True,
        timing_source="synthetic_causal_steady_state",
    )
    dataset = EpochDataset(
        name="synthetic-bi-decisions",
        X=np.zeros((len(rows), 3, 128), dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        subject_ids=np.asarray(subject_ids),
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(3, dtype=bool),
        preprocessing=PreprocessingSpec(
            name="p300_single_subject_causal_v2",
            filter_phase="forward",
            causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
        ),
        event_timeline=timeline,
        metadata=metadata,
        provenance={
            "source": "unit_test",
            "source_reference": "average",
            "source_sample_rate_hz": 128.0,
        },
    )

    return dataset


def test_bi_calibration_uses_earlier_known_decisions_and_tests_new_targets() -> None:
    dataset = _bi_cross_decision_dataset()
    metadata = dataset.metadata
    split = bi2014a_calibration_decision_split(
        dataset,
        calibration_selections=3,
        test_repetitions=2,
    )

    assert split.usable_subjects == ("s1",)
    assert len(split.calibration_selections_by_subject["s1"]) == 3
    assert len(split.requested_test_selections_by_subject["s1"]) == 4
    assert len(split.test_selections_by_subject["s1"]) == 4
    assert split.failed_test_selections_by_subject["s1"] == {}
    assert int(split.calibration_mask.sum()) == 3 * 3 * 12
    assert int(split.test_mask.sum()) == 4 * 2 * 12
    assert set(metadata.loc[split.calibration_mask, "selection_id"]).isdisjoint(
        set(metadata.loc[split.test_mask, "selection_id"])
    )

    first_later = "s1:selection3"
    broken_row = metadata.index[
        (metadata["selection_id"] == first_later)
        & (metadata["repetition_index"] == 0)
    ][0]
    dataset.metadata.loc[broken_row, "repetition_index"] = 99
    capped = bi2014a_calibration_decision_split(
        dataset,
        calibration_selections=3,
        test_repetitions=2,
        max_test_selections=2,
    )

    assert capped.requested_test_selections_by_subject["s1"] == (
        "s1:selection3",
        "s1:selection4",
    )
    assert capped.test_selections_by_subject["s1"] == ("s1:selection4",)
    assert capped.failed_test_selections_by_subject["s1"] == {
        "s1:selection3": "insufficient_complete_test_repetitions"
    }
    assert "s1:selection5" not in capped.test_selections_by_subject["s1"]


def test_invalid_calibration_keeps_frozen_later_decisions_as_failures() -> None:
    dataset = concatenate_epoch_datasets(
        [_bi_cross_decision_dataset("s1"), _bi_cross_decision_dataset("s2")],
        name="synthetic-bi-two-subjects",
    )
    broken_row = dataset.metadata.index[
        (dataset.metadata["selection_id"] == "s1:selection0")
        & (dataset.metadata["repetition_index"] == 0)
    ][0]
    dataset.metadata.loc[broken_row, "repetition_index"] = 99

    split = bi2014a_calibration_decision_split(
        dataset,
        calibration_selections=3,
        test_repetitions=2,
        max_test_selections=2,
    )

    assert split.usable_subjects == ("s2",)
    assert split.requested_test_selections_by_subject["s1"] == (
        "s1:selection3",
        "s1:selection4",
    )
    assert split.failed_test_selections_by_subject["s1"] == {
        "s1:selection3": "calibration_failed",
        "s1:selection4": "calibration_failed",
    }
    assert split.excluded_subjects["s1"] == "invalid_calibration_decision"
    assert not split.test_mask[np.asarray(dataset.subject_ids) == "s1"].any()


def test_bi_runner_counts_failed_requested_decision_without_replacement(
    tmp_path, monkeypatch
) -> None:
    dataset = _bi_cross_decision_dataset()
    metadata = dataset.metadata
    broken_row = metadata.index[
        (metadata["selection_id"] == "s1:selection3")
        & (metadata["repetition_index"] == 0)
    ][0]
    dataset.metadata.loc[broken_row, "repetition_index"] = 99
    cache = save_epoch_dataset(tmp_path / "bi-causal.npz", dataset)
    trunk = N2P3Net(n_channels=3, n_times=128, sfreq=128.0, tmin_s=-0.2)
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "trunk_state_dict": trunk.state_dict(),
            "training_subject_keys": [f"{dataset.name}\0other"],
            "training_subjects": ["other"],
            "holdout_subjects": ["s1"],
            "source_dataset_name": dataset.name,
            "input_channel_names": list(dataset.channel_names),
            "input_preprocessing": asdict(dataset.preprocessing),
            "input_source_reference": dataset.provenance["source_reference"],
            "config": {"pooling_mode": "ms_flatten", "training": "supervised"},
            "classifier_trained": True,
            "input_mean": [0.0, 0.0, 0.0],
            "input_std": [1.0, 1.0, 1.0],
            "training_pos_weight": 5.0,
            "training_prior": 1.0 / 6.0,
            "architecture": trunk.architecture_record(),
            "n_channels": 3,
            "n_times": 128,
            "input_sample_rate_hz": 128.0,
            "input_tmin_s": -0.2,
        },
        checkpoint,
    )
    output = tmp_path / "bi-summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_bi2014a_candidate.py",
            "--dataset-cache",
            str(cache),
            "--checkpoint",
            str(checkpoint),
            "--calibration-selections",
            "3",
            "--test-reps",
            "2",
            "--max-test-selections",
            "2",
            "--head",
            "zero_shot",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )

    run_bi_candidate_main()

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["decision_accounting"]["requested"] == 2
    assert summary["decision_accounting"]["eligible"] == 1
    assert summary["decision_accounting"]["failed"] == 1
    ledger = summary["subject_decision_ledger"][0]
    assert ledger["requested"] == 2
    assert ledger["eligible"] == 1
    assert ledger["failed_test_selections"] == {
        "s1:selection3": "insufficient_complete_test_repetitions"
    }
    assert summary["records"][0]["requested_test_selections"] == [
        "s1:selection3",
        "s1:selection4",
    ]
