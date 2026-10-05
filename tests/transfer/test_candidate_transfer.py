from __future__ import annotations

import copy
import hashlib
import json
import pickle

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
from experiments.analyze_candidate_cross_decision import (
    REQUESTED_METRIC,
    analyze_manifest,
)
from experiments.build_candidate_cross_decision_manifest import build_manifest
from experiments.run_candidate_cross_decision import (
    _validate_cached_candidate_labels,
)
from experiments.run_candidate_cross_decision import (
    main as run_candidate_main,
)
from research.contracts import (
    TrainingRunContract,
    semantic_sha256,
)
from research.evaluation import verified_source_snapshot
from tests.support.candidate import candidate_dataset, candidate_task_contract
from tests.support.checkpoints import candidate_checkpoint_payload
from tests.support.io import read_json, write_json
from tests.support.snapshots import write_source_snapshot
from transfer.candidate_decision import (
    decision_outcomes_at_repetition,
    expected_candidate_counts,
)
from transfer.candidate_within_subject import (
    candidate_calibration_decision_split,
    candidate_prefix_suffix_split,
)
from transfer.outcomes import (
    CandidateCoverage,
    DecisionKey,
    DecisionOutcome,
    DecisionStatus,
    build_decision_outcome_accounting,
)
from transfer.subject_adapter import SubjectAdapter


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
        contract=candidate_task_contract(),
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
        contract=candidate_task_contract(),
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
        contract=candidate_task_contract(),
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
            contract=candidate_task_contract(),
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
            expected_candidate_counts(candidate_task_contract(), 1), {}
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


def test_promotion_runner_cli_rejects_test_reps_below_eight() -> None:
    argv = _runner_argv(
        "missing.npz", "missing.manifest.json", "unused.json", arm_name="arm", test_reps=7
    )
    with pytest.raises(SystemExit, match="2"):
        run_candidate_main(argv)


def _bi_dataset() -> EpochDataset:
    selection = np.repeat(["s1", "s2"], 24)
    n = len(selection)
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
            (n, len(identity.names), SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times),
            dtype=np.float32,
        ),
        y=y,
        subject_ids=selection.astype(str),
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(len(identity.names), dtype=bool),
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
            "candidate_task_contract": candidate_task_contract().record(),
        },
    )


def test_derived_bi_cache_raw_label_tampering_is_rejected() -> None:
    dataset = _bi_dataset()
    dataset.provenance["source"] = "derived_common_car_cache"
    dataset.provenance["candidate_task_contract"] = candidate_task_contract(
        "BI2014a", raw_label_codebook=True
    ).record()
    dataset.metadata["raw_target_label"] = np.where(dataset.y == 1, 2, 1)
    non_target_row = dataset.metadata.index[dataset.y == 0][0]
    dataset.metadata.loc[non_target_row, "raw_target_label"] = 2

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


def test_generic_bnci_contract_uses_early_decisions_and_tests_new_targets() -> None:
    dataset = candidate_dataset()
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
    assert capped.test_selections_by_subject["s1"] == (
        "s1:selection3",
        "s1:selection4",
    )
    assert capped.failed_test_selections_by_subject["s1"] == {}


@pytest.mark.parametrize(
    "repetitions,horizon,options,calibration_rows,test_rows",
    [
        pytest.param((4, 4, 4, 1, 2, 3, 4, 8), 8, {}, 144, 216, id="mixed-depth-default"),
        pytest.param(
            (4, 4, 4, 1, 2, 3, 4, 8),
            8,
            {"selection_repetition_floor": 2},
            144,
            216,
            id="mixed-depth-explicit-floor",
        ),
        pytest.param((4, 4, 4, 3, 4, 5), 4, {}, 144, 132, id="default-horizon"),
        pytest.param(
            (4, 4, 4, 3, 4, 5),
            4,
            {"selection_repetition_floor": 4},
            144,
            132,
            id="explicit-horizon",
        ),
        pytest.param((7, 7, 7, 7), 8, {}, 252, 84, id="all-test-decisions-shallow"),
    ],
)
def test_candidate_split_keeps_shallow_decisions_in_requested_denominator(
    repetitions,
    horizon,
    options,
    calibration_rows,
    test_rows,
) -> None:
    dataset = candidate_dataset(
        "s1", n_selections=len(repetitions), repetitions_per_selection=repetitions
    )
    split = candidate_calibration_decision_split(
        dataset, calibration_selections=3, test_repetitions=horizon, **options
    )
    expected = {"s1": tuple(f"s1:selection{index}" for index in range(3, len(repetitions)))}
    assert split.usable_subjects == ("s1",)
    assert split.requested_test_selections_by_subject == expected
    assert split.test_selections_by_subject == expected
    assert split.failed_test_selections_by_subject == {"s1": {}}
    assert split.excluded_subjects == {}
    assert split.calibration_mask.sum() == calibration_rows
    assert split.test_mask.sum() == test_rows


@pytest.mark.parametrize(
    "floor,message",
    [(value, "positive integer") for value in (0, -1, True, 2.0, "2")] + [(5, "must not exceed")],
)
def test_selection_floor_rejects_invalid_values(floor, message) -> None:
    dataset = candidate_dataset("s1", n_selections=5, repetitions_per_selection=4)
    with pytest.raises(ValueError, match=message):
        candidate_calibration_decision_split(
            dataset,
            calibration_selections=3,
            test_repetitions=4,
            selection_repetition_floor=floor,
        )


def test_candidate_split_typed_failure_when_no_later_decisions_exist() -> None:
    dataset = candidate_dataset("s1", n_selections=3, repetitions_per_selection=7)

    split = candidate_calibration_decision_split(
        dataset,
        calibration_selections=3,
        test_repetitions=8,
    )

    assert split.usable_subjects == ()
    assert split.requested_test_selections_by_subject == {}
    assert split.test_selections_by_subject == {}
    assert split.failed_test_selections_by_subject == {}
    assert split.excluded_subjects == {"s1": "insufficient_decisions"}
    assert not split.calibration_mask.any()
    assert not split.test_mask.any()


def _runner_argv(cache, manifest, output, **options) -> list[str]:
    options = {
        "dataset_cache": cache,
        "source_snapshot_manifest": manifest,
        "output": output,
        "training_replicate_key": "seed:0",
        "partition_key": "partition:0",
        "identity_exclusion_policy": "source",
        "calibration_selections": 3,
        "test_reps": 8,
        "head": "linear",
        "device": "cpu",
        **options,
    }
    return [
        value
        for name, value in options.items()
        for value in ("--" + name.replace("_", "-"), str(value))
    ]


def test_runner_records_all_empty_r8_partition_without_fabricated_decisions(tmp_path) -> None:
    dataset = candidate_dataset("s1", n_selections=3, repetitions_per_selection=7)
    cache = save_epoch_dataset(tmp_path / "target.npz", dataset)
    source_manifest, _, source_archive = write_source_snapshot(tmp_path)
    output = tmp_path / "empty-result.json"
    argv = _runner_argv(
        cache,
        source_manifest,
        output,
        arm_name="scratch_linear",
        partition_key="partition:empty",
    )

    run_candidate_main(argv)

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
        "insufficient_decisions"
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
    reference_argv = _runner_argv(
        cache,
        source_manifest,
        reference_output,
        arm_name="scratch_linear_reference",
        partition_key="partition:empty",
    )
    run_candidate_main(reference_argv)
    manifest_path = build_manifest(
        result_paths=[output, reference_output],
        source_snapshot_manifest=source_manifest,
        inference_scope="conditional_frozen_models",
        planned_contrasts=[("scratch_linear", "scratch_linear_reference")],
        output=tmp_path / "empty-manifest.json",
        bootstrap_iterations=1000,
        evidence_level=8,
        metric_name=REQUESTED_METRIC,
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
    invalid_argv = _runner_argv(
        cache,
        source_manifest,
        tmp_path / "must-not-exist.json",
        arm_name="invalid_checkpoint",
        partition_key="partition:empty",
        checkpoint=invalid_checkpoint,
    )
    with pytest.raises((pickle.UnpicklingError, RuntimeError, ValueError)):
        run_candidate_main(invalid_argv)


@pytest.fixture
def empty_partition_checkpoint(tmp_path):
    dataset = candidate_dataset("s1", n_selections=4, repetitions_per_selection=7)
    cache = save_epoch_dataset(tmp_path / "target.npz", dataset)
    manifest, source_sha256, _ = write_source_snapshot(tmp_path)
    payload = candidate_checkpoint_payload(dataset, source_snapshot_sha256=source_sha256)
    checkpoint = tmp_path / "source.pt"
    output = tmp_path / "result.json"
    argv = _runner_argv(
        cache,
        manifest,
        output,
        checkpoint=checkpoint,
        arm_name="zero_shot_source",
        partition_key="partition:empty",
        head="zero_shot",
    )
    return dataset, payload, checkpoint, output, argv


def _reject_empty_partition_checkpoint(case, message):
    _, payload, checkpoint, output, argv = case
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match=message):
        run_candidate_main(argv)
    assert not output.exists()


def test_empty_partition_rejects_wrong_checkpoint_time_axis(empty_partition_checkpoint):
    dataset, payload, *_ = empty_partition_checkpoint
    payload["n_times"] = dataset.n_times + 1
    _reject_empty_partition_checkpoint(empty_partition_checkpoint, "n_times")


def test_empty_partition_rejects_wrong_checkpoint_reference(empty_partition_checkpoint):
    _, payload, *_ = empty_partition_checkpoint
    payload["input_source_reference"] = "nose"
    training_record = copy.deepcopy(payload["training_contract"])
    training_record["preprocessing"]["source_reference"] = "nose"
    contract = TrainingRunContract(**training_record)
    payload.update(training_contract=contract.record(), training_contract_digest=contract.digest())
    _reject_empty_partition_checkpoint(empty_partition_checkpoint, "source_reference")


def test_empty_partition_rejects_missing_checkpoint_parameter(empty_partition_checkpoint):
    _, payload, *_ = empty_partition_checkpoint
    del payload["trunk_state_dict"][next(iter(payload["trunk_state_dict"]))]
    _reject_empty_partition_checkpoint(
        empty_partition_checkpoint, "target trunk|declared architecture"
    )


def test_empty_partition_rejects_source_target_identity_overlap(empty_partition_checkpoint):
    dataset, payload, *_ = empty_partition_checkpoint
    ledger = DatasetIdentityTable((dataset.identity_table.record_for("s1"),))
    training_record = copy.deepcopy(payload["training_contract"])
    training_record.update(
        source_identity_digest=ledger.digest(),
        training_participant_keys=list(ledger.authority_keys("source")),
        holdout_participant_keys=[],
    )
    contract = TrainingRunContract(**training_record)
    payload.update(
        training_identity_ledger=ledger.payload(),
        training_identity_ledger_digest=ledger.digest(),
        training_contract=contract.record(),
        training_contract_digest=contract.digest(),
    )
    _reject_empty_partition_checkpoint(empty_partition_checkpoint, "includes target subject")


def test_invalid_calibration_keeps_frozen_later_decisions_as_failures() -> None:
    dataset = concatenate_epoch_datasets(
        [
            candidate_dataset("s1"),
            candidate_dataset("s2"),
        ],
        name="synthetic-bi-two-subjects",
        provenance={
            "source_reference": "average",
            "source_sample_rate_hz": 128.0,
            "candidate_task_contract": candidate_task_contract().record(),
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


@pytest.mark.parametrize("separate_evaluation_snapshot", [False, True])
def test_bi_runner_counts_failed_requested_decision_without_replacement(
    tmp_path, separate_evaluation_snapshot
) -> None:
    dataset = concatenate_epoch_datasets(
        [
            candidate_dataset("s1", repetitions_per_selection=8),
            candidate_dataset("s2", repetitions_per_selection=8),
            candidate_dataset("s3", n_selections=3, repetitions_per_selection=8),
        ],
        name="synthetic-bi-runner-integration",
        provenance={
            "source_reference": "average",
            "source_sample_rate_hz": 128.0,
            "candidate_task_contract": candidate_task_contract().record(),
        },
    )
    dataset.provenance["source_reference"] = "average"
    metadata = dataset.metadata
    broken_row = metadata.index[
        (metadata["selection_id"] == "s1:selection3") & (metadata["repetition_index"] == 0)
    ][0]
    dataset.metadata.loc[broken_row, "repetition_index"] = 99
    cache = save_epoch_dataset(tmp_path / "bi-causal.npz", dataset)
    checkpoint = tmp_path / "source.pt"
    source_manifest, source_snapshot_sha256, _ = write_source_snapshot(tmp_path)
    evaluation_manifest = source_manifest
    evaluation_sha256 = source_snapshot_sha256
    if separate_evaluation_snapshot:
        evaluation_dir = tmp_path / "evaluation"
        evaluation_dir.mkdir()
        evaluation_manifest, evaluation_sha256, _ = write_source_snapshot(
            evaluation_dir, source_value=2, source_commit="b" * 40
        )
    torch.save(
        candidate_checkpoint_payload(dataset, source_snapshot_sha256=source_snapshot_sha256),
        checkpoint,
    )
    output = tmp_path / "bi-summary.json"
    argv = _runner_argv(
        cache,
        source_manifest,
        output,
        checkpoint=str(checkpoint),
        arm_name="zero_shot_source",
        max_test_selections="2",
        head="zero_shot",
    )

    if separate_evaluation_snapshot:
        argv += ["--evaluation-snapshot-manifest", str(evaluation_manifest)]
    run_candidate_main(argv)

    summary = read_json(output)
    assert summary["schema"] == "n2p3_candidate_cross_decision_result/1"
    # C1: source training and evaluation each retain their physical source identity.
    assert summary["source_snapshot_sha256"] == source_snapshot_sha256
    assert summary["evaluation_snapshot_sha256"] == evaluation_sha256
    assert summary["evaluation_contract"]["source_snapshot_sha256"] == evaluation_sha256
    assert (
        summary["source_snapshot_manifest_sha256"]
        == hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    )
    assert summary["source_commit"] == "a" * 40
    # C2: environment identity is recorded fail-closed with a self-digest.
    environment = summary["environment"]
    assert len(environment["lockfiles"]["uv.lock"]) == 64
    assert environment["environment_digest"]
    assert environment["device"] == "cpu"
    # C3: Split-BN provenance is restated from the checkpoint payload.
    splitbn = summary["splitbn"]
    assert splitbn["present"] is True
    assert splitbn["effective_target_buffer_source"] is None
    assert splitbn["buffer_estimand"] is None
    assert len(splitbn["bn_buffer_digest"]) == 64
    assert summary["decision_accounting"]["requested"] == 4
    assert summary["decision_accounting"]["data_eligible"] == 4
    assert summary["decision_accounting"]["data_ineligible"] == 0
    assert summary["decision_accounting"]["evaluation_successful"] == 3
    assert summary["decision_accounting"]["evaluation_failed"] == 1
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
            "participant_key": next(
                key
                for key in summary["requested_participant_keys"]
                if key.endswith("/participant/s1")
            ),
            "decision_id": "s1:selection3",
            "status": "incomplete",
            "stage": "decision_evaluation",
            "reason": "incomplete_candidate_coverage",
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
        "incomplete_candidate_coverage"
    }
    assert all(not outcome["candidate_coverage"]["complete"] for outcome in failed_outcomes)
    s3_key = summary["decision_plan"]["participant_selection_failures"][0]["participant_key"]
    assert all(outcome["participant_key"] != s3_key for outcome in summary["decision_outcomes"])

    second_output = tmp_path / "bi-summary-reference.json"
    second_argv = _runner_argv(
        cache,
        source_manifest,
        second_output,
        checkpoint=checkpoint,
        arm_name="zero_shot_reference",
        max_test_selections=2,
        head="zero_shot",
        evaluation_snapshot_manifest=evaluation_manifest,
    )
    run_candidate_main(second_argv)

    rejected_result = tmp_path / "candidate-r7.json"
    rejected = read_json(output)
    rejected["test_reps"] = 7
    rejected["evaluation_contract"]["target_protocol"]["test_repetitions"] = 7
    rejected["evaluation_contract"]["decision"]["test_repetitions"] = 7
    rejected["evaluation_contract_digest"] = semantic_sha256(rejected["evaluation_contract"])
    write_json(rejected_result, rejected)
    with pytest.raises(ValueError, match="test_repetitions >= 8"):
        build_manifest(
            result_paths=(rejected_result, second_output),
            source_snapshot_manifest=source_manifest,
            evaluation_snapshot_manifest=evaluation_manifest,
            inference_scope="conditional_frozen_models",
            planned_contrasts=(("zero_shot_source", "zero_shot_reference"),),
            output=tmp_path / "rejected-experiment.json",
            bootstrap_iterations=1000,
            evidence_level=8,
        )

    missing_level_result = tmp_path / "candidate-missing-r8.json"
    missing_level = read_json(output)
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
    write_json(missing_level_result, missing_level)
    with pytest.raises(ValueError, match="exactly levels 1..8"):
        build_manifest(
            result_paths=(missing_level_result, second_output),
            source_snapshot_manifest=source_manifest,
            evaluation_snapshot_manifest=evaluation_manifest,
            inference_scope="conditional_frozen_models",
            planned_contrasts=(("zero_shot_source", "zero_shot_reference"),),
            output=tmp_path / "missing-level-experiment.json",
            bootstrap_iterations=1000,
            evidence_level=8,
        )

    manifest_path = build_manifest(
        result_paths=(output, second_output),
        source_snapshot_manifest=source_manifest,
        evaluation_snapshot_manifest=evaluation_manifest,
        inference_scope="conditional_frozen_models",
        planned_contrasts=(("zero_shot_source", "zero_shot_reference"),),
        output=tmp_path / "experiment.json",
        bootstrap_iterations=1000,
        evidence_level=8,
        metric_name=REQUESTED_METRIC,
    )
    analysis = analyze_manifest(manifest_path)
    assert analysis["requested_participants"] == 3
    assert analysis["planned_contrasts"][0]["paired_unit"] == "participant"
    assert analysis["planned_contrasts"][0]["paired_bootstrap_interval"]["n_units"] == 3
    assert analysis["runs"][0]["decision_endpoints"]["denominators"]["planned_decisions"] == 4
    assert analysis["source_snapshot"]["archive_sha256"] == source_snapshot_sha256
    assert analysis["evaluation_snapshot"]["archive_sha256"] == evaluation_sha256

    if separate_evaluation_snapshot:
        with pytest.raises(ValueError, match="evaluation snapshot disagrees"):
            build_manifest(
                result_paths=(output, second_output),
                source_snapshot_manifest=source_manifest,
                inference_scope="conditional_frozen_models",
                planned_contrasts=(("zero_shot_source", "zero_shot_reference"),),
                output=tmp_path / "wrong-evaluation.json",
                bootstrap_iterations=1000,
            )
        manifest_record = read_json(manifest_path)
        manifest_record["source_snapshot"] = manifest_record["evaluation_snapshot"]
        write_json(manifest_path, manifest_record)
        with pytest.raises(ValueError, match="checkpoint source snapshot disagrees"):
            analyze_manifest(manifest_path)


def _bi_runner_linear_fixture(tmp_path, name: str) -> tuple[object, object]:
    """Two-subject dataset cache plus a valid source freeze for linear-head runs."""

    dataset = concatenate_epoch_datasets(
        [
            candidate_dataset("s1", repetitions_per_selection=8),
            candidate_dataset("s2", repetitions_per_selection=8),
        ],
        name=name,
        provenance={
            "source_reference": "average",
            "source_sample_rate_hz": 128.0,
            "candidate_task_contract": candidate_task_contract().record(),
        },
    )
    dataset.provenance["source_reference"] = "average"
    cache = save_epoch_dataset(tmp_path / f"{name}.npz", dataset)
    source_manifest, _, _ = write_source_snapshot(tmp_path)
    return cache, source_manifest


def test_candidate_runner_declares_prefix_for_chronological_refit(tmp_path) -> None:
    cache, manifest = _bi_runner_linear_fixture(tmp_path, "refit-scope")
    output = tmp_path / "refit-result.json"
    argv = _runner_argv(
        cache,
        manifest,
        output,
        arm_name="refit-scope-test",
        partition_key="scope",
        calibration_selections="1",
        epoch_selection="target_time_split",
        epochs="1",
    )
    run_candidate_main(argv)
    result = read_json(output)
    assert result["records"]
    for record in result["records"]:
        scope = record["prefix_refit"]["fit_scope"]
        assert record["prefix_refit"]["enabled"]
        assert scope["prefix_mask_explicit"]
        assert scope["refit_rows"] == record["n_prefix_after_qc"]
        assert scope["outside_prefix_rows"] == 0
        assert record["prefix_refit"]["as_mmd_losses"] == []
        assert record["prefix_refit"]["initial_fit_as_mmd_losses"] == []


def test_bi_runner_converts_subject_fit_oom_to_fit_failure_with_preserved_denominator(
    tmp_path, monkeypatch
) -> None:
    cache, source_manifest = _bi_runner_linear_fixture(tmp_path, "synthetic-bi-runner-oom")

    def _raise_oom(self, *args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(SubjectAdapter, "fit", _raise_oom)
    output = tmp_path / "bi-summary-oom.json"
    argv = _runner_argv(
        cache,
        source_manifest,
        output,
        arm_name="linear_oom_probe",
        max_test_selections="2",
    )

    run_candidate_main(argv)

    summary = read_json(output)
    assert summary["run_status"] == "completed"
    assert summary["decision_accounting"]["requested"] == 4
    assert summary["decision_accounting"]["data_eligible"] == 4
    assert summary["decision_accounting"]["data_ineligible"] == 0
    assert summary["decision_accounting"]["evaluation_successful"] == 0
    assert summary["decision_accounting"]["evaluation_failed"] == 4
    assert summary["participant_accounting"] == {
        "requested": 2,
        "decision_planned": 2,
        "selection_failed": 0,
    }
    assert summary["oom_accounting"] == {
        "oom_fit_failure_subjects": ["s1", "s2"],
        "oom_fit_failure_decisions": 4,
        "requested_decision_slots": 4,
        "ledger_decision_slots": 4,
        "requested_denominator_preserved": True,
    }
    assert len(summary["decision_outcomes"]) == 32
    assert all(outcome["status"] == "fit_failure" for outcome in summary["decision_outcomes"])
    assert {outcome["failure_reason"] for outcome in summary["decision_outcomes"]} == {
        "subject_fit:accelerator_oom"
    }
    for repetition in (str(level) for level in range(1, 9)):
        status_counts = summary["decision_accounting"]["by_evidence_level"][repetition]
        assert status_counts["fit_failure"] == 4
        assert status_counts["requested"] == status_counts["accounted"] == 4
    # No checkpoint was provided, so the Split-BN summary records absence.
    assert summary["splitbn"]["present"] is False
    assert summary["splitbn"]["bn_buffer_digest"] is None
    records = {record["subject"]: record for record in summary["records"]}
    assert set(records) == {"s1", "s2"}


def test_bi_runner_reraises_non_oom_runtime_error(tmp_path, monkeypatch) -> None:
    cache, source_manifest = _bi_runner_linear_fixture(tmp_path, "synthetic-bi-runner-reraise")

    def _raise_non_oom(self, *args, **kwargs):
        raise RuntimeError("weight tensor is not contiguous")

    monkeypatch.setattr(SubjectAdapter, "fit", _raise_non_oom)
    output = tmp_path / "bi-summary-reraise.json"
    argv = _runner_argv(
        cache,
        source_manifest,
        output,
        arm_name="linear_reraise_probe",
        max_test_selections="2",
    )

    with pytest.raises(RuntimeError, match="weight tensor is not contiguous"):
        run_candidate_main(argv)
    assert not output.exists()


def test_verified_source_snapshot_returns_full_identity(tmp_path) -> None:
    manifest, _, source_archive = write_source_snapshot(tmp_path, source_commit="b" * 40)

    identity = verified_source_snapshot(manifest)

    assert identity["archive_sha256"] == hashlib.sha256(source_archive.read_bytes()).hexdigest()
    assert identity["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert identity["source_commit"] == "b" * 40
    assert identity["byte_size"] == source_archive.stat().st_size
    assert identity["member_count"] == 1
