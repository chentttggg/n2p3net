from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pickle
import sys
import tarfile
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from data.candidate_task import CandidateTaskContract
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
from experiments.analyze_candidate_cross_decision import analyze_manifest
from experiments.build_candidate_cross_decision_manifest import build_manifest
from experiments.run_candidate_cross_decision import (
    _validate_cached_candidate_labels,
)
from experiments.run_candidate_cross_decision import (
    main as run_candidate_main,
)
from models.n2p3net import N2P3Net
from research.contracts import (
    TrainingRunContract,
    assert_promotion_evidence_gate,
    semantic_sha256,
)
from transfer.candidate_decision import (
    decision_outcomes_at_repetition,
    expected_candidate_counts,
)
from transfer.candidate_within_subject import (
    candidate_calibration_decision_split,
    candidate_prefix_suffix_split,
)
from transfer.checkpoint import CHECKPOINT_SCHEMA
from transfer.outcomes import (
    CandidateCoverage,
    DecisionKey,
    DecisionOutcome,
    DecisionStatus,
    build_decision_outcome_accounting,
)


def _task_contract(
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


def test_decision_outcomes_at_repetition_6x6_target_row_and_column() -> None:
    n = 24
    selection = np.repeat("s1", n)
    repetitions = np.concatenate([np.zeros(12, dtype=np.int64), np.ones(12, dtype=np.int64)])
    # Each repetition: all six rows and all six columns.
    candidates = np.tile(np.arange(12, dtype=np.int64), 2)
    logits = np.zeros(n, dtype=float)
    logits[candidates == 1] = 2.0
    logits[candidates == 8] = 2.0
    target_rows = np.full(n, 1, dtype=np.int64)
    target_cols = np.full(n, 2, dtype=np.int64)
    outcomes = decision_outcomes_at_repetition(
        logits,
        candidates,
        target_rows,
        target_cols,
        selection,
        repetitions,
        contract=_task_contract(),
        subject_ids=np.repeat("subject", n),
        max_repetitions=2,
    )
    assert [outcome.evidence_level for outcome in outcomes] == [1, 2]
    assert all(outcome.status == DecisionStatus.CORRECT for outcome in outcomes)


def test_decision_outcome_preserves_tie_as_primary_status() -> None:
    candidates = np.arange(12, dtype=np.int64)
    outcomes = decision_outcomes_at_repetition(
        np.zeros(12),
        candidates,
        np.zeros(12, dtype=np.int64),
        np.zeros(12, dtype=np.int64),
        np.repeat("selection", 12),
        np.zeros(12, dtype=np.int64),
        contract=_task_contract(),
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
    candidates = np.arange(11, dtype=np.int64)
    outcomes = decision_outcomes_at_repetition(
        np.ones(11),
        candidates,
        np.zeros(11, dtype=np.int64),
        np.zeros(11, dtype=np.int64),
        np.repeat("selection", 11),
        np.zeros(11, dtype=np.int64),
        contract=_task_contract(),
        subject_ids=np.repeat("s1", 11),
        max_repetitions=1,
    )

    assert outcomes[0].status == DecisionStatus.INCOMPLETE
    assert outcomes[0].failure_reason == "incomplete_candidate_coverage"
    assert outcomes[0].coverage.missing_event_counts == {"candidate:11": 1}
    assert not outcomes[0].coverage.complete


def test_decision_outcome_rejects_candidate_outside_declared_vocabulary() -> None:
    candidates = np.arange(12, dtype=np.int64)
    candidates[-1] = 99
    with pytest.raises(ValueError, match="outside the task contract"):
        decision_outcomes_at_repetition(
            np.ones(12),
            candidates,
            np.zeros(12, dtype=np.int64),
            np.zeros(12, dtype=np.int64),
            np.repeat("selection", 12),
            np.zeros(12, dtype=np.int64),
            contract=_task_contract(),
            subject_ids=np.repeat("s1", 12),
            max_repetitions=1,
        )


def test_decision_accounting_requires_every_requested_status() -> None:
    key = DecisionKey("s1", "selection")
    fit_failure = DecisionOutcome(
        key=key,
        evidence_level=1,
        status=DecisionStatus.FIT_FAILURE,
        coverage=CandidateCoverage.from_mappings(
            expected_candidate_counts(_task_contract(), 1), {}
        ),
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


def test_promotion_gate_rejects_seven_repetitions_and_accepts_eight() -> None:
    with pytest.raises(ValueError, match="test_repetitions >= 8"):
        assert_promotion_evidence_gate(7)
    assert_promotion_evidence_gate(8, primary_evidence_level=8)


def test_promotion_runner_cli_rejects_test_reps_below_eight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_candidate_cross_decision.py",
            "--dataset-cache",
            "missing.npz",
            "--arm-name",
            "arm",
            "--training-replicate-key",
            "seed:0",
            "--partition-key",
            "partition:0",
            "--source-snapshot-manifest",
            "missing.manifest.json",
            "--identity-exclusion-policy",
            "source",
            "--test-reps",
            "7",
            "--output",
            "unused.json",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        run_candidate_main()


def test_removed_bi_transfer_modules_and_clis_are_not_importable() -> None:
    assert importlib.util.find_spec("transfer.bi_decision") is None
    assert importlib.util.find_spec("transfer.bi_within_subject") is None
    assert importlib.util.find_spec("experiments.run_bi2014a_candidate") is None
    assert importlib.util.find_spec("experiments.build_bi2014a_cross_decision_manifest") is None
    assert importlib.util.find_spec("experiments.analyze_bi2014a_cross_decision") is None


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
    candidate_id = np.where(row_code >= 0, row_code, 6 + col_code)
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
                "candidate_id": candidate_id,
                "raw_is_target": is_target,
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
            "candidate_task_contract": _task_contract().record(),
        },
    )


def test_derived_bi_cache_raw_label_tampering_is_rejected() -> None:
    dataset = _bi_dataset()
    dataset.provenance["source"] = "derived_common_car_cache"
    dataset.provenance["candidate_task_contract"] = _task_contract(
        "BI2014a", raw_label_codebook=True
    ).record()
    dataset.metadata["raw_target_label"] = np.where(dataset.y == 1, 2, 1)
    dataset.metadata.loc[0, "raw_target_label"] = (
        1 if dataset.metadata.loc[0, "raw_target_label"] == 2 else 2
    )

    with pytest.raises(ValueError, match="raw_target_label disagrees"):
        _validate_cached_candidate_labels(dataset)


def test_bi_prefix_suffix_split_requires_complete_repetitions() -> None:
    dataset = _bi_dataset()
    split = candidate_prefix_suffix_split(dataset, prefix_repetitions=1, test_repetitions=1)
    assert {item.decision_id for item in split.usable_decisions} == {"s1", "s2"}
    assert int(split.prefix_mask.sum()) == 24
    assert int(split.suffix_mask.sum()) == 24
    assert set(np.unique(split.suffix_repetition_indices[split.suffix_mask]).tolist()) == {0}
    with pytest.raises(ValueError, match="No candidate decision"):
        candidate_prefix_suffix_split(dataset, prefix_repetitions=2, test_repetitions=2)


def _candidate_cross_decision_dataset(
    subject: str = "s1",
    *,
    n_selections: int = 7,
    repetitions_per_selection: int = 3,
) -> EpochDataset:
    rows: list[dict[str, object]] = []
    labels: list[int] = []
    subject_ids: list[str] = []
    onset: list[float] = []
    cursor = 0
    for selection in range(n_selections):
        target_row = selection % 6
        target_col = (selection + 2) % 6
        for repetition in range(repetitions_per_selection):
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
            "candidate_task_contract": _task_contract().record(),
        },
    )

    return dataset


def _checkpoint_payload_for_dataset(
    dataset: EpochDataset, *, source_snapshot_sha256: str
) -> dict[str, object]:
    dataset.validate(require_labels=True)
    assert dataset.identity_table is not None
    trunk = N2P3Net(
        n_channels=dataset.n_channels,
        n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq,
        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
    )
    training_identity_ledger = DatasetIdentityTable.from_source_rows(["other"], [dataset.name])
    holdouts = tuple(
        sorted(
            {
                key
                for identity in dataset.identity_table.records
                for key in (*identity.origin_subject_keys, *identity.global_person_keys)
            }
        )
    )
    training_contract = TrainingRunContract(
        source_cache_sha256=semantic_sha256({"fixture": "source-cache"}),
        source_identity_digest=training_identity_ledger.digest(),
        source_snapshot_sha256=source_snapshot_sha256,
        architecture=trunk.architecture_record(),
        preprocessing={
            "epoch": asdict(dataset.preprocessing),
            "channel_names": list(dataset.channel_names),
            "source_reference": dataset.provenance["source_reference"],
        },
        optimizer={"name": "none"},
        validation={"mode": "none"},
        objective={"name": "none"},
        seed=0,
        training_participant_keys=training_identity_ledger.authority_keys("source"),
        holdout_participant_keys=holdouts,
    )
    return {
        "schema": CHECKPOINT_SCHEMA,
        "trunk_state_dict": trunk.state_dict(),
        "training_identity_ledger": training_identity_ledger.payload(),
        "training_identity_ledger_digest": training_identity_ledger.digest(),
        "training_contract": training_contract.record(),
        "training_contract_digest": training_contract.digest(),
        "source_cache_sha256": training_contract.source_cache_sha256,
        "input_channel_names": list(dataset.channel_names),
        "input_preprocessing": asdict(dataset.preprocessing),
        "input_source_reference": dataset.provenance["source_reference"],
        "classifier_trained": True,
        "input_mean": [0.0] * dataset.n_channels,
        "input_std": [1.0] * dataset.n_channels,
        "source_calibration": {
            "pos_weight": 5.0,
            "train_prior": 1.0 / 6.0,
            "temperature": 1.0,
            "source": "unit_test_fixture",
        },
        "architecture": trunk.architecture_record(),
        "n_channels": dataset.n_channels,
        "n_times": dataset.n_times,
        "input_sample_rate_hz": dataset.preprocessing.sfreq,
        "input_tmin_s": dataset.preprocessing.tmin_ms / 1000.0,
    }


def test_generic_bnci_contract_uses_early_decisions_and_tests_new_targets() -> None:
    dataset = _candidate_cross_decision_dataset()
    metadata = dataset.metadata
    split = candidate_calibration_decision_split(
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
    capped = candidate_calibration_decision_split(
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


def test_candidate_split_returns_typed_empty_selection_failure() -> None:
    dataset = _candidate_cross_decision_dataset("s1", n_selections=4, repetitions_per_selection=7)

    split = candidate_calibration_decision_split(
        dataset,
        calibration_selections=3,
        test_repetitions=8,
    )

    assert split.usable_subjects == ()
    assert split.requested_test_selections_by_subject == {"s1": ("s1:selection3",)}
    assert split.test_selections_by_subject == {}
    assert split.failed_test_selections_by_subject == {
        "s1": {"s1:selection3": "insufficient_complete_test_repetitions"}
    }
    assert split.excluded_subjects == {"s1": "no_eligible_unknown_test_decision"}
    assert not split.calibration_mask.any()
    assert not split.test_mask.any()


def test_runner_records_all_empty_r8_partition_without_fabricated_decisions(
    tmp_path, monkeypatch
) -> None:
    dataset = _candidate_cross_decision_dataset("s1", n_selections=4, repetitions_per_selection=7)
    cache = save_epoch_dataset(tmp_path / "target.npz", dataset)
    source_member = tmp_path / "source.txt"
    source_member.write_text("runner source fixture", encoding="utf-8")
    source_archive = tmp_path / "source.tar.gz"
    with tarfile.open(source_archive, mode="w:gz") as archive:
        archive.add(source_member, arcname="source.txt")
    source_manifest = tmp_path / "source.manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema": "n2p3_source_freeze/1",
                "archive": source_archive.name,
                "archive_sha256": hashlib.sha256(source_archive.read_bytes()).hexdigest(),
                "source_commit": "a" * 40,
                "member_count": 1,
                "byte_size": source_archive.stat().st_size,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "empty-result.json"
    argv = [
        "run_candidate_cross_decision.py",
        "--dataset-cache",
        str(cache),
        "--arm-name",
        "scratch_linear",
        "--training-replicate-key",
        "seed:0",
        "--partition-key",
        "partition:empty",
        "--source-snapshot-manifest",
        str(source_manifest),
        "--identity-exclusion-policy",
        "source",
        "--calibration-selections",
        "3",
        "--test-reps",
        "8",
        "--head",
        "linear",
        "--device",
        "cpu",
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_candidate_main()

    raw = output.read_text(encoding="utf-8")
    assert "NaN" not in raw
    result = json.loads(raw, parse_constant=lambda value: pytest.fail(value))
    assert result["run_status"] == "completed_with_selection_failures"
    assert result["participant_accounting"] == {
        "requested": 1,
        "decision_planned": 0,
        "selection_failed": 1,
    }
    assert result["decision_plan"]["entries"] == []
    assert result["decision_plan"]["participant_selection_failures"][0]["reason"] == (
        "no_eligible_unknown_test_decision"
    )
    assert result["decision_outcomes"] == []
    assert result["decision_failures"] == []
    assert result["records"] == []
    assert result["binary_auc_mean"] is None
    assert result["decision_accounting"]["requested"] == 0
    assert result["decision_accounting"]["data_eligible"] == 0
    assert result["decision_accounting"]["data_ineligible"] == 0
    endpoint = result["participant_operational_endpoints"]
    assert len(endpoint) == 1
    assert endpoint[0]["planned_decisions"] == 0
    assert endpoint[0]["requested_participant_operational_hit_rate"] == 0.0

    reference_output = tmp_path / "empty-reference.json"
    reference_argv = [*argv]
    reference_argv[reference_argv.index("scratch_linear")] = "scratch_linear_reference"
    reference_argv[reference_argv.index(str(output))] = str(reference_output)
    monkeypatch.setattr(sys, "argv", reference_argv)
    run_candidate_main()
    manifest_path = build_manifest(
        result_paths=[output, reference_output],
        source_snapshot_manifest=source_manifest,
        inference_scope="conditional_frozen_models",
        planned_contrasts=[("scratch_linear", "scratch_linear_reference")],
        output=tmp_path / "empty-manifest.json",
        bootstrap_iterations=1000,
        evidence_level=8,
    )
    analysis = analyze_manifest(manifest_path)
    assert analysis["requested_participants"] == 1
    for metric in analysis["metrics"].values():
        assert metric["conditional_frozen_model_requested_participant_operational_hit_rate"] == 0.0
    for run in analysis["runs"]:
        assert run["decision_endpoints"]["denominators"] == {
            "planned_decisions": 0,
            "data_eligible_decisions": 0,
            "evaluation_successful_decisions": 0,
        }

    invalid_checkpoint = tmp_path / "invalid.pt"
    invalid_checkpoint.write_bytes(b"not a checkpoint")
    invalid_argv = [*argv]
    invalid_argv[invalid_argv.index("--arm-name") + 1] = "invalid_checkpoint"
    invalid_argv[invalid_argv.index("--output") + 1] = str(tmp_path / "must-not-exist.json")
    invalid_argv[invalid_argv.index("--dataset-cache") : invalid_argv.index("--dataset-cache")] = [
        "--checkpoint",
        str(invalid_checkpoint),
    ]
    monkeypatch.setattr(sys, "argv", invalid_argv)
    with pytest.raises((pickle.UnpicklingError, RuntimeError, ValueError)):
        run_candidate_main()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_n_times", "n_times"),
        ("wrong_reference", "source_reference"),
        ("bad_state_dict", "target trunk|declared architecture"),
        ("identity_overlap", "includes target subject"),
    ],
)
def test_empty_partition_still_rejects_invalid_checkpoint_contract(
    tmp_path, monkeypatch, mutation: str, message: str
) -> None:
    dataset = _candidate_cross_decision_dataset("s1", n_selections=4, repetitions_per_selection=7)
    cache = save_epoch_dataset(tmp_path / "target.npz", dataset)
    source_member = tmp_path / "source.txt"
    source_member.write_text("source snapshot", encoding="utf-8")
    source_archive = tmp_path / "source.tar.gz"
    with tarfile.open(source_archive, "w:gz") as archive:
        archive.add(source_member, arcname="source.txt")
    source_sha256 = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    source_manifest = tmp_path / "source.manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema": "n2p3_source_freeze/1",
                "archive": source_archive.name,
                "archive_sha256": source_sha256,
                "source_commit": "a" * 40,
                "member_count": 1,
                "byte_size": source_archive.stat().st_size,
            }
        ),
        encoding="utf-8",
    )
    payload = _checkpoint_payload_for_dataset(dataset, source_snapshot_sha256=source_sha256)
    if mutation == "wrong_n_times":
        payload["n_times"] = dataset.n_times + 1
    elif mutation == "wrong_reference":
        payload["input_source_reference"] = "nose"
        training_record = copy.deepcopy(payload["training_contract"])
        training_record["preprocessing"]["source_reference"] = "nose"
        training_contract = TrainingRunContract(**training_record)
        payload["training_contract"] = training_contract.record()
        payload["training_contract_digest"] = training_contract.digest()
    elif mutation == "bad_state_dict":
        state = dict(payload["trunk_state_dict"])
        state.pop(next(iter(state)))
        payload["trunk_state_dict"] = state
    else:
        assert dataset.identity_table is not None
        overlapping_ledger = DatasetIdentityTable((dataset.identity_table.record_for("s1"),))
        training_record = copy.deepcopy(payload["training_contract"])
        training_record["source_identity_digest"] = overlapping_ledger.digest()
        training_record["training_participant_keys"] = list(
            overlapping_ledger.authority_keys("source")
        )
        training_record["holdout_participant_keys"] = []
        training_contract = TrainingRunContract(**training_record)
        payload["training_identity_ledger"] = overlapping_ledger.payload()
        payload["training_identity_ledger_digest"] = overlapping_ledger.digest()
        payload["training_contract"] = training_contract.record()
        payload["training_contract_digest"] = training_contract.digest()
    checkpoint = tmp_path / f"{mutation}.pt"
    torch.save(payload, checkpoint)
    output = tmp_path / f"{mutation}.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_candidate_cross_decision.py",
            "--dataset-cache",
            str(cache),
            "--checkpoint",
            str(checkpoint),
            "--arm-name",
            mutation,
            "--training-replicate-key",
            "seed:0",
            "--partition-key",
            "partition:empty",
            "--source-snapshot-manifest",
            str(source_manifest),
            "--identity-exclusion-policy",
            "source",
            "--calibration-selections",
            "3",
            "--test-reps",
            "8",
            "--head",
            "zero_shot",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(ValueError, match=message):
        run_candidate_main()
    assert not output.exists()


def test_invalid_calibration_keeps_frozen_later_decisions_as_failures() -> None:
    dataset = concatenate_epoch_datasets(
        [
            _candidate_cross_decision_dataset("s1"),
            _candidate_cross_decision_dataset("s2"),
        ],
        name="synthetic-bi-two-subjects",
        provenance={
            "source_reference": "average",
            "source_sample_rate_hz": 128.0,
            "candidate_task_contract": _task_contract().record(),
        },
    )
    broken_row = dataset.metadata.index[
        (dataset.metadata["selection_id"] == "s1:selection0")
        & (dataset.metadata["repetition_index"] == 0)
    ][0]
    dataset.metadata.loc[broken_row, "repetition_index"] = 99

    split = candidate_calibration_decision_split(
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
            _candidate_cross_decision_dataset("s1", repetitions_per_selection=8),
            _candidate_cross_decision_dataset("s2", repetitions_per_selection=8),
            _candidate_cross_decision_dataset("s3", n_selections=3, repetitions_per_selection=8),
        ],
        name="synthetic-bi-runner-integration",
        provenance={
            "source_reference": "average",
            "source_sample_rate_hz": 128.0,
            "candidate_task_contract": _task_contract().record(),
        },
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
            "run_candidate_cross_decision.py",
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
            "8",
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

    run_candidate_main()

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["schema"] == "n2p3_candidate_cross_decision_result/1"
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
    assert len(summary["decision_outcomes"]) == 32
    for repetition in (str(level) for level in range(1, 9)):
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
    run_candidate_main()

    rejected_result = tmp_path / "candidate-r7.json"
    rejected = json.loads(output.read_text(encoding="utf-8"))
    rejected["test_reps"] = 7
    rejected["evaluation_contract"]["target_protocol"]["test_repetitions"] = 7
    rejected["evaluation_contract"]["decision"]["test_repetitions"] = 7
    rejected["evaluation_contract_digest"] = semantic_sha256(rejected["evaluation_contract"])
    rejected_result.write_text(json.dumps(rejected), encoding="utf-8")
    with pytest.raises(ValueError, match="test_repetitions >= 8"):
        build_manifest(
            result_paths=(rejected_result, second_output),
            source_snapshot_manifest=source_manifest,
            inference_scope="conditional_frozen_models",
            planned_contrasts=(("zero_shot_source", "zero_shot_reference"),),
            output=tmp_path / "rejected-experiment.json",
            bootstrap_iterations=1000,
            evidence_level=8,
        )

    missing_level_result = tmp_path / "candidate-missing-r8.json"
    missing_level = json.loads(output.read_text(encoding="utf-8"))
    first_key = (
        missing_level["decision_outcomes"][0]["participant_key"],
        missing_level["decision_outcomes"][0]["decision_id"],
    )
    missing_level["decision_outcomes"] = [
        outcome
        for outcome in missing_level["decision_outcomes"]
        if not (
            (outcome["participant_key"], outcome["decision_id"]) == first_key
            and outcome["evidence_level"] == 8
        )
    ]
    missing_level_result.write_text(json.dumps(missing_level), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly levels 1..8"):
        build_manifest(
            result_paths=(missing_level_result, second_output),
            source_snapshot_manifest=source_manifest,
            inference_scope="conditional_frozen_models",
            planned_contrasts=(("zero_shot_source", "zero_shot_reference"),),
            output=tmp_path / "missing-level-experiment.json",
            bootstrap_iterations=1000,
            evidence_level=8,
        )

    manifest_path = build_manifest(
        result_paths=(output, second_output),
        source_snapshot_manifest=source_manifest,
        inference_scope="conditional_frozen_models",
        planned_contrasts=(("zero_shot_source", "zero_shot_reference"),),
        output=tmp_path / "experiment.json",
        bootstrap_iterations=1000,
        evidence_level=8,
    )
    analysis = analyze_manifest(manifest_path)
    assert analysis["requested_participants"] == 3
    assert analysis["planned_contrasts"][0]["paired_unit"] == "participant"
    assert analysis["planned_contrasts"][0]["paired_bootstrap_interval"]["n_units"] == 3
    assert analysis["runs"][0]["decision_endpoints"]["denominators"]["planned_decisions"] == 4
