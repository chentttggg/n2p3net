from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from data.channel import build_channel_identity
from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.epochs import (
    EpochDataset,
    concatenate_epoch_datasets,
    preprocessing_spec_from_contract,
    save_epoch_dataset,
)
from data.events import observed_only_timeline
from data.identity import DatasetIdentityTable
from experiments.analyze_bi2014a_cross_decision import analyze_manifest
from experiments.build_bi2014a_cross_decision_manifest import build_manifest
from experiments.run_bi2014a_candidate import main as run_bi_candidate_main
from models.n2p3net import N2P3Net
from research.contracts import TrainingRunContract, semantic_sha256
from transfer.bi_decision import (
    bi2014a_expected_candidate_counts,
    decision_outcomes_at_repetition_6x6,
)
from transfer.bi_within_subject import (
    bi2014a_calibration_decision_split,
    bi2014a_prefix_suffix_split,
)
from transfer.checkpoint import CHECKPOINT_SCHEMA
from transfer.outcomes import (
    CandidateCoverage,
    DecisionKey,
    DecisionOutcome,
    DecisionStatus,
    build_decision_outcome_accounting,
)


def test_decision_outcomes_at_repetition_6x6_target_row_and_column() -> None:
    n = 24
    selection = np.repeat("s1", n)
    repetitions = np.concatenate([np.zeros(12, dtype=np.int64), np.ones(12, dtype=np.int64)])
    # Each repetition: all six rows and all six columns.
    row_flashes = np.asarray([20, 61, 22, 23, 24, 25], dtype=np.int64)
    col_flashes = np.asarray([40, 41, 82, 43, 44, 45], dtype=np.int64)
    flash = np.tile(np.concatenate([row_flashes, col_flashes]), 2)
    logits = np.zeros(n, dtype=float)
    logits[flash == 61] = 2.0
    logits[flash == 82] = 2.0
    target_rows = np.full(n, 1, dtype=np.int64)
    target_cols = np.full(n, 2, dtype=np.int64)
    outcomes = decision_outcomes_at_repetition_6x6(
        logits,
        flash,
        target_rows,
        target_cols,
        selection,
        repetitions,
        subject_ids=np.repeat("subject", n),
        max_repetitions=2,
    )
    assert [outcome.evidence_level for outcome in outcomes] == [1, 2]
    assert all(outcome.status == DecisionStatus.CORRECT for outcome in outcomes)


def test_decision_outcome_preserves_tie_as_primary_status() -> None:
    flashes = np.asarray([60, 21, 22, 23, 24, 25, 80, 41, 42, 43, 44, 45], dtype=np.int64)
    outcomes = decision_outcomes_at_repetition_6x6(
        np.zeros(12),
        flashes,
        np.zeros(12, dtype=np.int64),
        np.zeros(12, dtype=np.int64),
        np.repeat("selection", 12),
        np.zeros(12, dtype=np.int64),
        subject_ids=np.repeat("s1", 12),
        onset_times_s=np.arange(12, dtype=float),
        evidence_available_times_s=np.arange(12, dtype=float) + 0.8,
        max_repetitions=1,
    )

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.status == DecisionStatus.TIE
    assert outcome.predicted_candidate is None
    assert outcome.coverage.complete
    assert outcome.to_record()["timing"]["evidence_available_s"] == 11.8


def test_decision_outcome_marks_missing_candidate_incomplete() -> None:
    flashes = np.asarray([60, 21, 22, 23, 24, 25, 80, 41, 42, 43, 44], dtype=np.int64)
    outcomes = decision_outcomes_at_repetition_6x6(
        np.ones(11),
        flashes,
        np.zeros(11, dtype=np.int64),
        np.zeros(11, dtype=np.int64),
        np.repeat("selection", 11),
        np.zeros(11, dtype=np.int64),
        subject_ids=np.repeat("s1", 11),
        max_repetitions=1,
    )

    assert outcomes[0].status == DecisionStatus.INCOMPLETE
    assert outcomes[0].failure_reason == "incomplete_candidate_coverage"
    assert outcomes[0].coverage.missing_event_counts == {"column:5": 1}
    assert not outcomes[0].coverage.complete


def test_decision_outcome_rejects_target_metadata_that_disagrees_with_codes() -> None:
    flashes = np.concatenate([np.arange(20, 26, dtype=np.int64), np.arange(40, 46, dtype=np.int64)])
    outcomes = decision_outcomes_at_repetition_6x6(
        np.ones(12),
        flashes,
        np.zeros(12, dtype=np.int64),
        np.zeros(12, dtype=np.int64),
        np.repeat("selection", 12),
        np.zeros(12, dtype=np.int64),
        subject_ids=np.repeat("s1", 12),
        max_repetitions=1,
    )

    assert outcomes[0].coverage.complete
    assert outcomes[0].status == DecisionStatus.INCOMPLETE
    assert outcomes[0].failure_reason == "flash_code_target_semantics_mismatch"


def test_decision_accounting_requires_every_requested_status() -> None:
    key = DecisionKey("s1", "selection")
    fit_failure = DecisionOutcome(
        key=key,
        evidence_level=1,
        status=DecisionStatus.FIT_FAILURE,
        coverage=CandidateCoverage.from_mappings(bi2014a_expected_candidate_counts(1), {}),
        failure_reason="optimizer_non_finite",
    )
    accounting = build_decision_outcome_accounting(
        [fit_failure], requested_decisions=[key], evidence_levels=[1]
    )
    record = accounting.to_record()["by_evidence_level"]["1"]
    assert record["requested"] == record["accounted"] == 1
    assert record["fit_failure"] == 1

    with pytest.raises(ValueError, match="accounting mismatch"):
        build_decision_outcome_accounting([], requested_decisions=[key], evidence_levels=[1])


def _bi_dataset() -> EpochDataset:
    n = 48
    selection = np.repeat(["s1", "s2"], 24)
    repetition = np.tile(
        np.concatenate([np.zeros(12, dtype=np.int64), np.ones(12, dtype=np.int64)]),
        2,
    )
    candidate_flash = np.tile(
        np.concatenate([np.arange(20, 26, dtype=np.int64), np.arange(40, 46, dtype=np.int64)]),
        4,
    )
    target_row = np.where(selection == "s1", 1, 4).astype(np.int64)
    target_col = np.where(selection == "s1", 2, 5).astype(np.int64)
    row_code = np.where((candidate_flash >= 20) & (candidate_flash <= 25), candidate_flash - 20, -1)
    col_code = np.where((candidate_flash >= 40) & (candidate_flash <= 45), candidate_flash - 40, -1)
    is_target = ((row_code >= 0) & (row_code == target_row)) | (
        (col_code >= 0) & (col_code == target_col)
    )
    flash = np.where(
        is_target & (row_code >= 0),
        60 + row_code,
        np.where(is_target & (col_code >= 0), 80 + col_code, candidate_flash),
    ).astype(np.int64)
    y = is_target.astype(np.int64)
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
        X=np.zeros(
            (n, 3, SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times),
            dtype=np.float32,
        ),
        y=y,
        subject_ids=selection.astype(str),
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(3, dtype=bool),
        preprocessing=preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT),
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
        provenance={
            "source": "unit_test",
            "source_reference": "average",
            "source_sample_rate_hz": 128.0,
        },
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


def _bi_cross_decision_dataset(subject: str = "s1", *, n_selections: int = 7) -> EpochDataset:
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
        (metadata["selection_id"] == first_later) & (metadata["repetition_index"] == 0)
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
    dataset = concatenate_epoch_datasets(
        [
            _bi_cross_decision_dataset("s1"),
            _bi_cross_decision_dataset("s2"),
            _bi_cross_decision_dataset("s3", n_selections=3),
        ],
        name="synthetic-bi-runner-integration",
    )
    dataset.provenance["source_reference"] = "average"
    metadata = dataset.metadata
    broken_row = metadata.index[
        (metadata["selection_id"] == "s1:selection3") & (metadata["repetition_index"] == 0)
    ][0]
    dataset.metadata.loc[broken_row, "repetition_index"] = 99
    cache = save_epoch_dataset(tmp_path / "bi-causal.npz", dataset)
    trunk = N2P3Net(
        n_channels=3,
        n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq,
        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
    )
    checkpoint = tmp_path / "source.pt"
    source_member = tmp_path / "source.txt"
    source_member.write_text("runner source fixture", encoding="utf-8")
    source_archive = tmp_path / "source.tar.gz"
    with tarfile.open(source_archive, mode="w:gz") as archive:
        archive.add(source_member, arcname="source.txt")
    source_snapshot_sha256 = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    source_manifest = tmp_path / "source.manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema": "n2p3_source_freeze/1",
                "archive": source_archive.name,
                "archive_sha256": source_snapshot_sha256,
                "source_commit": "a" * 40,
                "member_count": 1,
                "byte_size": source_archive.stat().st_size,
            }
        ),
        encoding="utf-8",
    )
    training_identity_ledger = DatasetIdentityTable.from_source_rows(["other"], [dataset.name])
    target_identities = [
        dataset.identity_table.record_for(subject) for subject in ("s1", "s2", "s3")
    ]
    training_contract = TrainingRunContract(
        source_cache_sha256=semantic_sha256({"fixture": "synthetic-source-cache"}),
        source_identity_digest=training_identity_ledger.digest(),
        source_snapshot_sha256=source_snapshot_sha256,
        architecture=trunk.architecture_record(),
        preprocessing={
            "epoch": asdict(dataset.preprocessing),
            "channel_names": list(dataset.channel_names),
            "source_reference": "average",
        },
        optimizer={"name": "none", "scope": "untrained_unit_test_fixture"},
        validation={"mode": "none", "scope": "untrained_unit_test_fixture"},
        objective={"name": "none", "scope": "untrained_unit_test_fixture"},
        seed=0,
        training_participant_keys=training_identity_ledger.authority_keys("source"),
        holdout_participant_keys=tuple(
            sorted(
                {
                    *(
                        key
                        for identity in target_identities
                        for key in (
                            *identity.origin_subject_keys,
                            *identity.global_person_keys,
                        )
                    ),
                }
            )
        ),
    )
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "trunk_state_dict": trunk.state_dict(),
            "training_identity_ledger": training_identity_ledger.payload(),
            "training_identity_ledger_digest": training_identity_ledger.digest(),
            "training_contract": training_contract.record(),
            "training_contract_digest": training_contract.digest(),
            "source_cache_sha256": training_contract.source_cache_sha256,
            "holdout_subjects": ["s1", "s2", "s3"],
            "source_dataset_name": dataset.name,
            "input_channel_names": list(dataset.channel_names),
            "input_preprocessing": asdict(dataset.preprocessing),
            "input_source_reference": "average",
            "config": {"pooling_mode": "ms_flatten", "training": "supervised"},
            "classifier_trained": True,
            "input_mean": [0.0, 0.0, 0.0],
            "input_std": [1.0, 1.0, 1.0],
            "source_calibration": {
                "pos_weight": 5.0,
                "train_prior": 1.0 / 6.0,
                "temperature": 1.0,
                "source": "unit_test_fixture",
            },
            "architecture": trunk.architecture_record(),
            "n_channels": 3,
            "n_times": dataset.n_times,
            "input_sample_rate_hz": dataset.preprocessing.sfreq,
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
            "--arm-name",
            "zero_shot_source",
            "--training-replicate-key",
            "seed:0",
            "--partition-key",
            "partition:0",
            "--source-snapshot-manifest",
            str(source_manifest),
            "--identity-exclusion-policy",
            "source",
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
    assert summary["schema"] == "n2p3_bi2014a_cross_decision_result/2"
    assert summary["decision_accounting"]["requested"] == 4
    assert summary["decision_accounting"]["data_eligible"] == 3
    assert summary["decision_accounting"]["data_ineligible"] == 1
    assert summary["decision_accounting"]["evaluation_successful"] == 3
    assert summary["decision_accounting"]["evaluation_failed"] == 0
    assert summary["participant_accounting"] == {
        "requested": 3,
        "decision_planned": 2,
        "selection_failed": 1,
    }
    assert len(summary["decision_outcomes"]) == 8
    for repetition in ("1", "2"):
        status_counts = summary["decision_accounting"]["by_evidence_level"][repetition]
        assert status_counts["requested"] == status_counts["accounted"] == 4
        assert status_counts["tie"] == 3
        assert status_counts["incomplete"] == 1
        assert (
            sum(
                status_counts[status]
                for status in (
                    "correct",
                    "incorrect",
                    "tie",
                    "abstain",
                    "incomplete",
                    "fit_failure",
                )
            )
            == status_counts["requested"]
        )
    assert summary["decision_failures"] == [
        {
            "participant_key": summary["requested_participant_keys"][0],
            "decision_id": "s1:selection3",
            "status": "incomplete",
            "stage": "decision_eligibility",
            "reason": "insufficient_complete_test_repetitions",
        }
    ]
    assert summary["decision_plan_digest"]
    assert len(summary["decision_plan"]["entries"]) == 4
    assert summary["decision_plan"]["participant_selection_failures"] == [
        {
            "participant_key": next(
                key
                for key in summary["requested_participant_keys"]
                if key.endswith("/participant/s3")
            ),
            "stage": "decision_selection",
            "reason": "insufficient_decisions",
        }
    ]
    assert all(outcome["target_candidate"] is not None for outcome in summary["decision_outcomes"])
    s1_record = next(record for record in summary["records"] if record["subject"] == "s1")
    assert s1_record["requested_test_selections"] == [
        "s1:selection3",
        "s1:selection4",
    ]
    failed_outcomes = [
        outcome
        for outcome in summary["decision_outcomes"]
        if outcome["decision_id"] == "s1:selection3"
    ]
    assert {outcome["status"] for outcome in failed_outcomes} == {"incomplete"}
    assert {outcome["failure_reason"] for outcome in failed_outcomes} == {
        "insufficient_complete_test_repetitions"
    }
    assert all(not outcome["candidate_coverage"]["complete"] for outcome in failed_outcomes)
    s3_key = summary["decision_plan"]["participant_selection_failures"][0]["participant_key"]
    assert all(outcome["participant_key"] != s3_key for outcome in summary["decision_outcomes"])

    second_output = tmp_path / "bi-summary-reference.json"
    second_argv = list(sys.argv)
    second_argv[second_argv.index("zero_shot_source")] = "zero_shot_reference"
    second_argv[second_argv.index(str(output))] = str(second_output)
    monkeypatch.setattr("sys.argv", second_argv)
    run_bi_candidate_main()

    manifest_path = build_manifest(
        result_paths=(output, second_output),
        source_snapshot_manifest=source_manifest,
        inference_scope="conditional_frozen_models",
        planned_contrasts=(("zero_shot_source", "zero_shot_reference"),),
        output=tmp_path / "experiment.json",
        bootstrap_iterations=1000,
        evidence_level=2,
    )
    analysis = analyze_manifest(manifest_path)
    assert analysis["requested_participants"] == 3
    assert analysis["planned_contrasts"][0]["paired_unit"] == "participant"
    assert analysis["planned_contrasts"][0]["paired_bootstrap_interval"]["n_units"] == 3
    assert analysis["runs"][0]["decision_endpoints"]["denominators"]["planned_decisions"] == 4
