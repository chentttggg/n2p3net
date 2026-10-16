from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from data.epochs import (
    PreprocessingSpec,
)
from tests.support.candidate import causal_candidate_dataset
from transfer.within_subject import causal_prefix_suffix_split, chronological_time_validation_split


def test_causal_prefix_suffix_split_is_chronological_and_complete() -> None:
    dataset = causal_candidate_dataset()
    split = causal_prefix_suffix_split(dataset, prefix_repetitions=2, test_repetitions=2)

    assert set(split.usable_groups) == {"g1", "g2", "g3"}
    assert split.candidate_vocab == (0, 1, 2, 3)
    assert split.excluded_groups == {}
    assert int(split.prefix_mask.sum()) == int(split.suffix_mask.sum()) == 3 * 4 * 2
    for group in split.usable_groups:
        rows = np.flatnonzero(split.group_ids == group)
        assert int(split.prefix_mask[rows].sum()) == 8
        assert int(split.suffix_mask[rows].sum()) == 8
        suffix_reps = split.suffix_repetition_indices[rows][split.suffix_mask[rows]]
        assert set(np.unique(suffix_reps).tolist()) == {0, 1}
        prefix_onsets = dataset.event_timeline.onset_times_s[rows][split.prefix_mask[rows]]
        suffix_onsets = dataset.event_timeline.onset_times_s[rows][split.suffix_mask[rows]]
        assert float(np.max(prefix_onsets)) < float(np.min(suffix_onsets))
        prefix_available = dataset.event_timeline.evidence_available_times_s[rows][
            split.prefix_mask[rows]
        ]
        suffix_epoch_starts = suffix_onsets + dataset.preprocessing.tmin_ms / 1000.0
        assert float(np.max(prefix_available)) < float(np.min(suffix_epoch_starts))
        prefix_evidence_times = dataset.event_timeline.evidence_available_times_s[rows][
            split.prefix_mask[rows]
        ]
        assert float(np.max(prefix_evidence_times)) < float(np.min(suffix_onsets))


def test_causal_split_supports_session_start_zero_calibration() -> None:
    dataset = causal_candidate_dataset()
    split = causal_prefix_suffix_split(dataset, prefix_repetitions=0, test_repetitions=2)

    assert not split.prefix_mask.any()
    assert int(split.suffix_mask.sum()) == 3 * 4 * 2
    assert all(
        record["prefix_available_trials"] == 0 for record in split.evidence_cost_by_group.values()
    )


def test_operational_split_allows_unequal_available_candidate_counts() -> None:
    dataset = causal_candidate_dataset()
    timeline = dataset.event_timeline
    scheduled_group = np.asarray(timeline.group_ids).astype(str)
    scheduled_candidate = np.asarray(timeline.candidate_ids).astype(str)
    remove_event = np.flatnonzero((scheduled_group == "g1") & (scheduled_candidate == "1"))[-1]
    remove_epoch = int(timeline.evidence_indices[remove_event])
    evidence = np.asarray(timeline.evidence_indices, dtype=np.int64).copy()
    evidence[remove_event] = -1
    evidence[evidence > remove_epoch] -= 1
    statuses = np.asarray(timeline.statuses).astype(str).copy()
    details = np.asarray(timeline.status_details).astype(str).copy()
    available_at = np.asarray(timeline.evidence_available_times_s, dtype=float).copy()
    statuses[remove_event] = "missing"
    details[remove_event] = "synthetic_missing_candidate_occurrence"
    available_at[remove_event] = np.nan
    keep = np.arange(dataset.n_epochs) != remove_epoch
    dataset.X = dataset.X[keep]
    dataset.y = dataset.y[keep]
    dataset.subject_ids = dataset.subject_ids[keep]
    dataset.metadata = dataset.metadata.iloc[keep].reset_index(drop=True)
    dataset.event_timeline = replace(
        timeline,
        evidence_indices=evidence,
        statuses=statuses,
        status_details=details,
        evidence_available_times_s=available_at,
    )
    dataset.validate(require_labels=True)

    split = causal_prefix_suffix_split(dataset, prefix_repetitions=2, test_repetitions=2)

    assert "g1" in split.usable_groups
    selected = split.selected_scheduled_repetitions["g1"]["suffix"]
    assert len({len(values) for values in selected.values()}) == 1

    all_split = causal_prefix_suffix_split(dataset, prefix_repetitions=0, test_repetitions=None)
    all_counts = all_split.evidence_cost_by_group["g1"]["suffix_available_trials_by_candidate"]
    assert len(set(all_counts.values())) == 2
    assert all_split.evidence_cost_by_group["g1"]["balanced_all_repetitions"] == 5
    assert set(all_split.evidence_cost_by_repetition["g1"]) == {
        "1",
        "2",
        "3",
        "4",
        "5",
    }


def test_real_time_inner_split_enforces_evidence_embargo() -> None:
    onset = np.arange(15, dtype=float)
    available = onset + 0.8
    y = np.zeros(15, dtype=np.int64)
    y[[1, 4, 7, 10, 13]] = 1

    split = chronological_time_validation_split(
        onset,
        available,
        y,
        epoch_start_offset_s=-0.2,
        min_positive_per_partition=2,
    )

    assert float(np.max(available[split.train_mask])) < float(
        np.min((onset - 0.2)[split.validation_mask])
    )
    assert split.embargo_mask is not None and split.embargo_mask.any()
    assert np.count_nonzero(y[split.train_mask]) >= 2
    assert np.count_nonzero(y[split.validation_mask]) >= 2


def test_causal_prefix_suffix_split_rejects_overlapping_epoch_evidence() -> None:
    dataset = causal_candidate_dataset()
    dataset.event_timeline = replace(
        dataset.event_timeline,
        evidence_available_times_s=dataset.event_timeline.onset_times_s + 100.0,
    )

    with pytest.raises(ValueError, match="No selection group"):
        causal_prefix_suffix_split(dataset, prefix_repetitions=2, test_repetitions=2)


def test_causal_prefix_suffix_split_rejects_zero_phase_cache() -> None:
    dataset = causal_candidate_dataset()
    dataset.preprocessing = PreprocessingSpec(
        name="offline",
        sfreq=128.0,
        l_freq=2.0,
        h_freq=30.0,
        tmin_ms=-200.0,
        tmax_ms=800.0,
        n_times=128,
        baseline_mode="mean_only",
        filter_phase="zero",
    )
    # EpochDataset.validate now refuses a forward contract whose timeline is
    # still marked acausal; this is the first leakage gate.
    with pytest.raises(ValueError, match="online_causal"):
        causal_prefix_suffix_split(dataset, prefix_repetitions=2, test_repetitions=2)
