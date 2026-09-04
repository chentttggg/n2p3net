from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.candidate_task import CandidateTaskContract
from data.channel import build_channel_identity
from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.epochs import (
    TARGET_CALIBRATION_SCOPE_SCHEMA,
    EpochDataset,
    load_epoch_dataset,
    preprocessing_spec_from_contract,
    save_epoch_dataset,
)
from data.events import observed_only_timeline
from experiments.prepare_target_calibration_cache import main
from transfer.candidate_within_subject import candidate_calibration_decision_split


def _dataset(subject: str = "s1", *, n_selections: int = 7) -> EpochDataset:
    rows: list[dict[str, object]] = []
    labels: list[int] = []
    subject_ids: list[str] = []
    onset: list[float] = []
    for selection in range(n_selections):
        target_row = selection % 6
        target_col = (selection + 2) % 6
        for repetition in range(3):
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
                onset.append(len(labels) * 1.5)
    metadata = pd.DataFrame(rows)
    identity = build_channel_identity(("Fz", "Cz", "Pz"), allow_missing_positions=False)
    timeline = observed_only_timeline(
        dataset_id=f"synthetic:{subject}",
        subject_ids=np.asarray(subject_ids),
        stimulus_ids=metadata["candidate_id"].to_numpy(dtype=np.int64),
        onset_times_s=np.asarray(onset),
        evidence_available_times_s=np.asarray(onset) + 0.8,
        group_ids=metadata["selection_id"].astype(str).to_numpy(),
        selection_ids=metadata["selection_id"].astype(str).to_numpy(),
        online_causal=True,
        timing_source="target_calibration_cache_fixture",
    )
    return EpochDataset(
        name=f"synthetic-{subject}",
        X=np.zeros(
            (len(rows), len(identity.names), SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times),
            dtype=np.float32,
        ),
        y=np.asarray(labels, dtype=np.int64),
        subject_ids=np.asarray(subject_ids),
        channel_names=identity.names,
        channel_positions_m=np.asarray(identity.coords, dtype=np.float32),
        channel_mask=np.ones(len(identity.names), dtype=bool),
        preprocessing=preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT),
        event_timeline=timeline,
        metadata=metadata,
        provenance={
            "source": "unit_test",
            "source_reference": "average",
            "candidate_task_contract": CandidateTaskContract(
                dataset_id=f"synthetic:{subject}",
                task_id="p300_row_column_speller_6x6",
                population={"label": "synthetic_test_population"},
                evidence_scope={"stage": "development", "product_confirmation": False},
                membership_kind="row_column",
                grid_shape=(6, 6),
                candidate_ids=tuple(range(12)),
                row_candidate_ids=tuple(range(6)),
                column_candidate_ids=tuple(range(6, 12)),
                target_representation="row_column_intersection",
                raw_target_label_is_target=None,
            ).record(),
        },
    )


def test_target_calibration_cache_keeps_only_calibration_rows(tmp_path: Path) -> None:
    dataset = _dataset()
    source = tmp_path / "target.npz"
    output = tmp_path / "target-cal.npz"
    save_epoch_dataset(source, dataset)

    main(
        [
            "--dataset-cache",
            str(source),
            "--output",
            str(output),
            "--calibration-selections",
            "3",
            "--test-repetitions",
            "3",
        ]
    )

    split = candidate_calibration_decision_split(
        dataset, calibration_selections=3, test_repetitions=3
    )
    expected_rows = int(np.asarray(split.calibration_mask).sum())
    adapted = load_epoch_dataset(output, require_labels=True, validation="attested")
    assert adapted.n_epochs == expected_rows == 3 * 12 * 3
    assert adapted.provenance["target_calibration_scope"]["schema"] == TARGET_CALIBRATION_SCOPE_SCHEMA
    assert adapted.provenance["target_calibration_scope"]["calibration_selections"] == 3
    assert set(adapted.metadata["selection_id"].unique()) == {
        f"s1:selection{i}" for i in range(3)
    }
    assert not bool(np.asarray(adapted.y) is None)


def test_target_calibration_cache_subset_rows_preserves_identity_contract(tmp_path: Path) -> None:
    dataset = _dataset(n_selections=5)
    source = tmp_path / "target.npz"
    output = tmp_path / "target-cal.npz"
    save_epoch_dataset(source, dataset)

    main(
        [
            "--dataset-cache",
            str(source),
            "--output",
            str(output),
            "--calibration-selections",
            "2",
            "--test-repetitions",
            "2",
            "--selection-repetition-floor",
            "2",
        ]
    )

    adapted = load_epoch_dataset(output, require_labels=True, validation="attested")
    assert adapted.identity_table is not None
    assert adapted.identity_table.local_subject_ids == ("s1",)
    assert adapted.n_epochs == 2 * 12 * 3
    assert adapted.event_timeline.n_events == adapted.n_epochs



def test_source_selection_vectors_cli_writes_attested_vectors(tmp_path: Path) -> None:
    dataset = _dataset()
    source = tmp_path / "source.npz"
    labels_path = tmp_path / "labels.npz"
    groups_path = tmp_path / "groups.npz"
    save_epoch_dataset(source, dataset)

    from experiments.prepare_source_selection_vectors import main as vectors_main

    vectors_main(
        [
            "--dataset-cache",
            str(source),
            "--output-labels",
            str(labels_path),
            "--output-groups",
            str(groups_path),
        ]
    )

    with np.load(labels_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["labels"], dataset.y)
    with np.load(groups_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["groups"], np.asarray(dataset.subject_ids, dtype=str))
