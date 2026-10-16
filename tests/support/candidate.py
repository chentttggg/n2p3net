"""Synthetic candidate builders for contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.candidate_task import CandidateTaskContract
from data.channel import build_channel_identity
from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.epochs import EpochDataset, preprocessing_spec_from_contract
from data.events import ScheduledEventTimeline, candidate_repetition_indices, observed_only_timeline


def candidate_task_contract(
    dataset_id: str = "synthetic-BNCI2014-008",
    *,
    raw_label_codebook: bool = False,
) -> CandidateTaskContract:
    return CandidateTaskContract(
        dataset_id=dataset_id,
        task_id="p300_row_column_speller_6x6",
        population={"label": "synthetic_test_population"},
        evidence_scope={"stage": "development", "product_confirmation": False},
        membership_kind="row_column",
        grid_shape=(6, 6),
        candidate_ids=tuple(range(12)),
        row_candidate_ids=tuple(range(6)),
        column_candidate_ids=tuple(range(6, 12)),
        target_representation="row_column_intersection",
        raw_target_label_is_target=({"1": False, "2": True} if raw_label_codebook else None),
    )


def candidate_dataset(
    subject: str = "s1",
    *,
    n_selections: int = 7,
    repetitions_per_selection: int | tuple[int, ...] = 3,
) -> EpochDataset:
    reps = (
        (repetitions_per_selection,) * n_selections
        if isinstance(repetitions_per_selection, int)
        else repetitions_per_selection
    )
    rows: list[dict[str, object]] = []
    labels: list[int] = []
    subject_ids: list[str] = []
    onset: list[float] = []
    cursor = 0
    for selection in range(n_selections):
        target_row = selection % 6
        target_col = (selection + 2) % 6
        for repetition in range(reps[selection]):
            for candidate_id in range(12):
                row_code = candidate_id if candidate_id < 6 else -1
                col_code = candidate_id - 6 if candidate_id >= 6 else -1
                is_target = row_code == target_row or col_code == target_col
                rows.append(
                    {
                        "subject": subject,
                        "candidate_id": candidate_id,
                        "raw_is_target": is_target,
                        "row_code": row_code,
                        "col_code": col_code,
                        "target_row": target_row,
                        "target_col": target_col,
                        "selection_id": f"{subject}:selection{selection}",
                        "repetition_index": repetition,
                    }
                )
                labels.append(int(is_target))
                subject_ids.append(subject)
                onset.append(cursor * 1.5)
                cursor += 1
    metadata = pd.DataFrame(rows)
    identity = build_channel_identity(("Fz", "Cz", "Pz"), allow_missing_positions=False)
    timeline = observed_only_timeline(
        dataset_id=f"synthetic-BNCI2014-008:{subject}",
        subject_ids=np.asarray(subject_ids),
        stimulus_ids=metadata["candidate_id"].to_numpy(dtype=np.int64),
        onset_times_s=np.asarray(onset),
        evidence_available_times_s=np.asarray(onset) + 0.8,
        group_ids=metadata["selection_id"].astype(str).to_numpy(),
        selection_ids=metadata["selection_id"].astype(str).to_numpy(),
        online_causal=True,
        timing_source="synthetic_causal_steady_state",
    )
    dataset = EpochDataset(
        name="synthetic-BNCI2014-008-decisions",
        X=np.zeros(
            (len(rows), 3, SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times),
            dtype=np.float32,
        ),
        y=np.asarray(labels, dtype=np.int64),
        subject_ids=np.asarray(subject_ids),
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(3, dtype=bool),
        preprocessing=preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT),
        event_timeline=timeline,
        metadata=metadata,
        provenance={
            "source": "unit_test",
            "source_reference": "average",
            "source_sample_rate_hz": 128.0,
            "candidate_task_contract": candidate_task_contract().record(),
        },
    )

    return dataset


def causal_candidate_dataset() -> EpochDataset:
    one_group_candidates = np.array(
        [1, 2, 1, 3, 4, 2, 1, 3, 4, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        dtype=np.int64,
    )
    n_epochs = 3 * len(one_group_candidates)
    groups = np.repeat(np.array(["g1", "g2", "g3"]), n_epochs // 3)
    candidates = np.tile(one_group_candidates, 3)
    targets = np.repeat(np.array([2, 3, 1], dtype=np.int64), n_epochs // 3)
    repetitions = candidate_repetition_indices(candidates.astype(str), groups)
    identity = build_channel_identity(("Fz", "Cz", "Pz"), allow_missing_positions=False)
    timeline = ScheduledEventTimeline(
        event_ids=np.asarray([f"e{i}" for i in range(n_epochs)]),
        group_ids=groups,
        subject_ids=groups,
        stimulus_ids=candidates,
        onset_samples=np.arange(n_epochs, dtype=np.int64) * 100,
        onset_times_s=np.arange(n_epochs, dtype=float),
        evidence_available_times_s=np.arange(n_epochs, dtype=float) + 0.8,
        evidence_indices=np.arange(n_epochs, dtype=np.int64),
        statuses=np.repeat("available", n_epochs),
        status_details=np.repeat("", n_epochs),
        dataset_ids=np.repeat("synthetic", n_epochs),
        session_ids=np.repeat("", n_epochs),
        run_ids=groups,
        selection_ids=groups,
        complete=True,
        online_causal=True,
        timing_source="synthetic_causal",
        candidate_ids=candidates.astype(str),
        target_candidate_ids=targets.astype(str),
        repetition_indices=repetitions,
    )
    return EpochDataset(
        name="synthetic_causal_candidates",
        X=np.zeros(
            (n_epochs, 3, SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times),
            dtype=np.float32,
        ),
        y=(candidates == targets).astype(np.int64),
        subject_ids=groups.astype(str),
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(3, dtype=bool),
        preprocessing=preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT),
        event_timeline=timeline,
        metadata=pd.DataFrame({"subject": groups.astype(str)}),
        provenance={
            "source": "unit_test",
            "source_reference": "average",
            "source_sample_rate_hz": 128.0,
        },
    )
