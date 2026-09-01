from __future__ import annotations

import copy
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from data.bi2014a_schedule import BI2014A_FLASH_SCHEDULE
from experiments.analyze_bi2014a_cross_decision import (
    MANIFEST_SCHEMA,
    POWER_PLAN_SCHEMA,
    RESULT_SCHEMA,
    analyze_manifest,
)
from research.contracts import (
    ArmContract,
    DecisionPlanContract,
    DecisionPlanEntry,
    EvaluationRunContract,
    StatisticalDesignContract,
    TrainingRunContract,
    scratch_procedure_record,
    semantic_sha256,
    training_procedure_record,
)
from transfer.bi_decision import bi2014a_expected_candidate_counts
from transfer.outcomes import (
    CandidateCoverage,
    DecisionKey,
    DecisionOutcome,
    DecisionStatus,
    build_decision_outcome_accounting,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _outcome(participant: str, *, correct: bool) -> DecisionOutcome:
    expected_counts = bi2014a_expected_candidate_counts(1)
    return DecisionOutcome(
        key=DecisionKey(participant, "decision-1"),
        evidence_level=1,
        status=DecisionStatus.CORRECT if correct else DecisionStatus.INCORRECT,
        coverage=CandidateCoverage.from_mappings(expected_counts, expected_counts),
        target_candidate="row:0|column:0",
        predicted_candidate=("row:0|column:0" if correct else "row:1|column:1"),
        onset_start_s=1.0,
        onset_end_s=2.0,
        evidence_available_s=2.25,
    )


def _result_record(
    evaluation: EvaluationRunContract,
    *,
    checkpoint_sha256: str,
    correct_by_participant: dict[str, bool],
    local_subject_by_participant: dict[str, str],
) -> dict[str, object]:
    participants = evaluation.requested_participant_keys
    plan = DecisionPlanContract(
        target_cache_sha256=evaluation.target_cache_sha256,
        target_identity_digest=evaluation.target_identity_digest,
        requested_participant_keys=participants,
        entries=tuple(
            DecisionPlanEntry(participant, "decision-1", "row:0|column:0")
            for participant in participants
        ),
    )
    paired_outcomes = tuple(
        (
            participant,
            _outcome(
                local_subject_by_participant[participant],
                correct=correct_by_participant[participant],
            ),
        )
        for participant in participants
    )
    outcomes = tuple(outcome for _, outcome in paired_outcomes)
    requested_keys = [
        DecisionKey(local_subject_by_participant[participant], "decision-1")
        for participant in participants
    ]
    global_accounting = build_decision_outcome_accounting(
        outcomes,
        requested_decisions=requested_keys,
        evidence_levels=(1,),
    ).to_record()
    global_accounting["data_eligible"] = len(participants)
    global_accounting["data_ineligible"] = 0
    global_accounting["evaluation_successful"] = len(participants)
    global_accounting["evaluation_failed"] = 0
    return {
        "schema": RESULT_SCHEMA,
        "run_status": "completed",
        "evaluation_contract": evaluation.record(),
        "evaluation_contract_digest": evaluation.digest(),
        "decision_plan": plan.record(),
        "decision_plan_digest": plan.digest(),
        "participant_accounting": {
            "requested": len(participants),
            "decision_planned": len(participants),
            "selection_failed": 0,
        },
        "requested_participant_keys": list(participants),
        "target_cache_sha256": evaluation.target_cache_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "decision_outcomes": [
            {**outcome.to_record(), "participant_key": participant}
            for participant, outcome in paired_outcomes
        ],
        "decision_accounting": global_accounting,
        "decision_failures": [],
    }


def _make_fixture(
    tmp_path: Path,
    *,
    inference_scope: str = "conditional_frozen_models",
    replicate_count: int = 2,
) -> tuple[Path, dict[str, object]]:
    participants = ("BI:P01", "BI:P02", "BI:P03")
    local_subject_by_participant = {
        "BI:P01": "s01",
        "BI:P02": "s02",
        "BI:P03": "s03",
    }
    partitions = {
        "partition:a": participants[:2],
        "partition:b": participants[2:],
    }
    replicates = tuple(f"retrain:{index}" for index in range(replicate_count))
    arms = ("adapted", "source")
    target_cache = _digest("target-cache")
    target_identity = _digest("target-identity")
    source_member = tmp_path / "source.txt"
    source_member.write_text("physical source snapshot fixture\n", encoding="utf-8")
    source_snapshot_path = tmp_path / "source-snapshot.tar.gz"
    with tarfile.open(source_snapshot_path, mode="w:gz") as archive:
        archive.add(source_member, arcname="source.txt")
    source_snapshot = hashlib.sha256(source_snapshot_path.read_bytes()).hexdigest()
    source_manifest_path = tmp_path / "source-snapshot.manifest.json"
    _write_json(
        source_manifest_path,
        {
            "schema": "n2p3_source_freeze/1",
            "archive": source_snapshot_path.name,
            "archive_sha256": source_snapshot,
            "source_commit": "a" * 40,
            "member_count": 1,
            "byte_size": source_snapshot_path.stat().st_size,
        },
    )
    metric = {
        "name": "requested_participant_operational_hit_rate",
        "evidence_level": 1,
        "replicate_aggregation": (
            "mean_across_declared_frozen_models"
            if inference_scope == "conditional_frozen_models"
            else "independent_training_replicates"
        ),
    }
    contrasts = (("adapted", "source"),)

    power_plan = None
    power_digest = None
    if inference_scope == "training_procedure":
        power_plan = {
            "schema": POWER_PLAN_SCHEMA,
            "inference_scope": "training_procedure",
            "training_replicate_keys": list(replicates),
            "planned_contrasts": [list(pair) for pair in contrasts],
            "primary_metric": metric,
            "planned_independent_replicates": replicate_count,
            "planning_parameters": {
                "method": "simulation",
                "target_power": 0.8,
                "alpha": 0.05,
                "effect_definition": "paired replicate-level subject-macro hit-rate delta",
            },
        }
        power_digest = semantic_sha256(power_plan)
    design = StatisticalDesignContract(
        inference_scope=inference_scope,  # type: ignore[arg-type]
        participant_cluster_key="participant_key",
        training_replicate_keys=replicates,
        partition_keys=tuple(partitions),
        planned_contrasts=contrasts,
        power_plan_digest=power_digest,
    )

    checkpoint_registry: dict[str, object] = {}
    checkpoint_contracts: dict[tuple[str, str], tuple[str, TrainingRunContract]] = {}
    for replicate_index, replicate in enumerate(replicates):
        for partition, holdout in partitions.items():
            checkpoint_id = f"checkpoint:{replicate}:{partition}"
            checkpoint_path = tmp_path / f"checkpoint-{_digest(checkpoint_id)[:16]}.pt"
            checkpoint_path.write_bytes(checkpoint_id.encode("ascii"))
            checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            training = TrainingRunContract(
                source_cache_sha256=_digest(f"source-cache:{replicate}:{partition}"),
                source_identity_digest=_digest(f"source-identity:{replicate}:{partition}"),
                source_snapshot_sha256=source_snapshot,
                architecture={"name": "n2p3", "kernel": 35},
                preprocessing={"sfreq": 128.0, "l_freq": 0.1, "tmax_ms": 1200.0},
                optimizer={"name": "adamw", "epochs": 4},
                validation={"kind": "participant_holdout"},
                objective={"trial_ce": 1.0},
                seed=replicate_index,
                training_participant_keys=(f"SRC:{replicate}:{partition}",),
                holdout_participant_keys=holdout,
            )
            checkpoint_registry[checkpoint_id] = {
                "checkpoint_path": checkpoint_path.name,
                "checkpoint_sha256": checkpoint_sha,
                "training_contract": training.record(),
                "training_contract_digest": training.digest(),
            }
            checkpoint_contracts[(replicate, partition)] = (
                checkpoint_id,
                training,
            )

    runs = []
    for arm in arms:
        for replicate_index, replicate in enumerate(replicates):
            for partition, requested in partitions.items():
                checkpoint_id, training = checkpoint_contracts[(replicate, partition)]
                checkpoint_sha = checkpoint_registry[checkpoint_id]["checkpoint_sha256"]  # type: ignore[index]
                evaluation = EvaluationRunContract(
                    arm_name=arm,
                    model_origin={
                        "kind": "checkpoint",
                        "checkpoint_sha256": checkpoint_sha,
                        "training_contract_digest": training.digest(),
                    },
                    target_cache_sha256=target_cache,
                    target_identity_digest=target_identity,
                    source_snapshot_sha256=source_snapshot,
                    target_protocol={
                        "estimand": "known_early_decisions_to_later_unknown_decisions",
                        "calibration_decisions": 2,
                        "calibration_selections": 2,
                        "test_repetitions": 1,
                        "split_axis": "decision_time",
                        "calibration_truth_access": "adapter_visible",
                        "test_truth_access": "scorer_only",
                    },
                    adaptation={
                        "procedure": {
                            "head": arm,
                            "normalization": "source",
                        },
                        "replicate_parameters": {
                            "training_replicate_key": replicate,
                            "random_seed": replicate_index,
                        },
                    },
                    decision={
                        "task": "BI2014a_6x6_row_column",
                        "schedule": BI2014A_FLASH_SCHEDULE.record(),
                        "schedule_digest": BI2014A_FLASH_SCHEDULE.digest(),
                        "aggregation": "cumulative_row_column_llr",
                        "tie_policy": "abstain",
                        "test_repetitions": 1,
                    },
                    requested_participant_keys=requested,
                    evidence_scope={
                        "phase": "development",
                        "estimand": "early_known_to_later_unknown_decisions",
                    },
                )
                run_id = f"{arm}-{replicate}-{partition}"
                result_path = tmp_path / f"run-{_digest(run_id)[:16]}.json"
                correct = {
                    participant: (
                        arm == "adapted" or (participant == "BI:P03" and replicate_index % 2 == 0)
                    )
                    for participant in requested
                }
                _write_json(
                    result_path,
                    _result_record(
                        evaluation,
                        checkpoint_sha256=checkpoint_sha,  # type: ignore[arg-type]
                        correct_by_participant=correct,
                        local_subject_by_participant=local_subject_by_participant,
                    ),
                )
                plan = DecisionPlanContract.from_record(
                    json.loads(result_path.read_text(encoding="utf-8"))["decision_plan"]
                )
                runs.append(
                    {
                        "run_id": run_id,
                        "result_path": result_path.name,
                        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                        "training_replicate_key": replicate,
                        "partition_key": partition,
                        "checkpoint_id": checkpoint_id,
                        "evaluation_contract": evaluation.record(),
                        "evaluation_contract_digest": evaluation.digest(),
                        "decision_plan": plan.record(),
                        "decision_plan_digest": plan.digest(),
                    }
                )

    source_procedure = training_procedure_record(next(iter(checkpoint_contracts.values()))[1])
    arm_contracts = {}
    for arm in arms:
        contract = ArmContract(
            arm_name=arm,
            model_origin_kind="checkpoint",
            source_training_procedure=source_procedure,
            adaptation_procedure={"head": arm, "normalization": "source"},
            allowed_variation_axes=("training_replicate", "participant_partition"),
        )
        arm_contracts[arm] = {"contract": contract.record(), "digest": contract.digest()}
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "source_snapshot": {
            "manifest_path": source_manifest_path.name,
            "manifest_sha256": hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
        },
        "target_dataset": {
            "target_cache_sha256": target_cache,
            "target_identity_digest": target_identity,
            "requested_participant_keys": list(participants),
        },
        "metric": metric,
        "bootstrap": {"iterations": 1000, "seed": 17, "confidence_level": 0.95},
        "checkpoint_registry": checkpoint_registry,
        "arm_contracts": arm_contracts,
        "statistical_design": design.record(),
        "statistical_design_digest": design.digest(),
        "runs": runs,
    }
    if power_plan is not None:
        manifest["power_plan_contract"] = power_plan
    manifest_path = tmp_path / "experiment.json"
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def _rewrite_result(tmp_path: Path, run: dict[str, object], result: dict[str, object]) -> None:
    path = tmp_path / str(run["result_path"])
    _write_json(path, result)
    run["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_conditional_analysis_is_dynamic_and_only_bootstraps_participants(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _make_fixture(tmp_path)
    analysis = analyze_manifest(manifest_path)

    assert analysis["requested_participants"] == 3
    assert analysis["participant_partitions"] == ["partition:a", "partition:b"]
    assert analysis["inference"]["scope"] == "conditional_frozen_models"
    assert analysis["inference"]["sampling_unit"] == "participant"
    assert analysis["inference"]["p_values_computed"] is False
    contrast = analysis["planned_contrasts"][0]
    assert contrast["paired_unit"] == "participant"
    assert contrast["paired_bootstrap_interval"]["n_units"] == 3
    assert "paired_sign_flip_p" not in json.dumps(analysis)


def test_swapped_result_artifacts_fail_the_manifest_contract(tmp_path: Path) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    runs = manifest["runs"]
    runs[0]["result_path"], runs[4]["result_path"] = (  # type: ignore[index]
        runs[4]["result_path"],  # type: ignore[index]
        runs[0]["result_path"],  # type: ignore[index]
    )
    runs[0]["result_sha256"], runs[4]["result_sha256"] = (  # type: ignore[index]
        runs[4]["result_sha256"],  # type: ignore[index]
        runs[0]["result_sha256"],  # type: ignore[index]
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="evaluation contract disagrees"):
        analyze_manifest(manifest_path)


def test_renamed_arm_with_recomputed_result_digest_still_fails(tmp_path: Path) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    run = manifest["runs"][0]  # type: ignore[index]
    result_path = tmp_path / run["result_path"]  # type: ignore[index]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["evaluation_contract"]["arm_name"] = "renamed"
    result["evaluation_contract_digest"] = semantic_sha256(result["evaluation_contract"])
    _rewrite_result(tmp_path, run, result)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="evaluation contract disagrees"):
        analyze_manifest(manifest_path)


def test_result_bytes_are_bound_by_the_manifest_sha(tmp_path: Path) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    run = manifest["runs"][0]  # type: ignore[index]
    result_path = tmp_path / run["result_path"]  # type: ignore[index]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["unregistered_mutation"] = True
    _write_json(result_path, result)

    with pytest.raises(ValueError, match="result SHA mismatch"):
        analyze_manifest(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("checkpoint_sha", "provenance mismatch"),
        ("requested_participants", "provenance mismatch"),
        ("outcome_accounting", "decision_accounting disagrees"),
    ],
)
def test_result_provenance_and_accounting_mutations_fail_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    run = manifest["runs"][0]  # type: ignore[index]
    result_path = tmp_path / run["result_path"]  # type: ignore[index]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if mutation == "checkpoint_sha":
        result["checkpoint_sha256"] = _digest("wrong-checkpoint")
    elif mutation == "requested_participants":
        result["requested_participant_keys"] = result["requested_participant_keys"][:-1]
    else:
        result["decision_accounting"]["by_evidence_level"]["1"]["correct"] += 1
    _rewrite_result(tmp_path, run, result)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match=message):
        analyze_manifest(manifest_path)


def test_target_identity_mutation_cannot_be_hidden_by_recomputing_result_digest(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    run = manifest["runs"][0]  # type: ignore[index]
    result_path = tmp_path / run["result_path"]  # type: ignore[index]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["evaluation_contract"]["target_identity_digest"] = _digest("wrong-identity")
    result["evaluation_contract_digest"] = semantic_sha256(result["evaluation_contract"])
    _rewrite_result(tmp_path, run, result)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="evaluation contract disagrees"):
        analyze_manifest(manifest_path)


def test_outcomes_cannot_shrink_the_manifest_frozen_decision_denominator(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    run = manifest["runs"][0]  # type: ignore[index]
    result_path = tmp_path / run["result_path"]  # type: ignore[index]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["decision_outcomes"] = result["decision_outcomes"][1:]
    _rewrite_result(tmp_path, run, result)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="manifest-frozen decision plan"):
        analyze_manifest(manifest_path)


def test_local_subject_alias_cannot_replace_the_authority_sampling_key(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    run = manifest["runs"][0]  # type: ignore[index]
    result_path = tmp_path / run["result_path"]  # type: ignore[index]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["decision_outcomes"][0]["participant_key"] = result["decision_outcomes"][0]["subject"]
    _rewrite_result(tmp_path, run, result)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="frozen decision plan"):
        analyze_manifest(manifest_path)


def test_same_arm_adaptation_procedure_cannot_drift_between_cells(tmp_path: Path) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    run = manifest["runs"][0]  # type: ignore[index]
    run["evaluation_contract"]["adaptation"]["procedure"]["normalization"] = "target"
    run["evaluation_contract_digest"] = semantic_sha256(run["evaluation_contract"])
    result_path = tmp_path / run["result_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["evaluation_contract"] = copy.deepcopy(run["evaluation_contract"])
    result["evaluation_contract_digest"] = run["evaluation_contract_digest"]
    _rewrite_result(tmp_path, run, result)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="adaptation disagrees"):
        analyze_manifest(manifest_path)


def test_same_arm_source_training_procedure_cannot_drift_between_cells(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    checkpoint_id = manifest["runs"][0]["checkpoint_id"]  # type: ignore[index]
    checkpoint = manifest["checkpoint_registry"][checkpoint_id]
    checkpoint["training_contract"]["optimizer"] = {"name": "sgd", "epochs": 4}
    checkpoint["training_contract_digest"] = semantic_sha256(checkpoint["training_contract"])
    for run in manifest["runs"]:
        if run["checkpoint_id"] != checkpoint_id:
            continue
        run["evaluation_contract"]["model_origin"]["training_contract_digest"] = checkpoint[
            "training_contract_digest"
        ]
        run["evaluation_contract_digest"] = semantic_sha256(run["evaluation_contract"])
        result_path = tmp_path / run["result_path"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["evaluation_contract"] = copy.deepcopy(run["evaluation_contract"])
        result["evaluation_contract_digest"] = run["evaluation_contract_digest"]
        _rewrite_result(tmp_path, run, result)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="source training procedure changed"):
        analyze_manifest(manifest_path)


def test_training_replicates_cannot_reuse_adaptation_seed(tmp_path: Path) -> None:
    manifest_path, manifest = _make_fixture(
        tmp_path, inference_scope="training_procedure", replicate_count=3
    )
    for run in manifest["runs"]:
        if run["training_replicate_key"] != "retrain:1":
            continue
        run["evaluation_contract"]["adaptation"]["replicate_parameters"]["random_seed"] = 0
        run["evaluation_contract_digest"] = semantic_sha256(run["evaluation_contract"])
        result_path = tmp_path / run["result_path"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["evaluation_contract"] = copy.deepcopy(run["evaluation_contract"])
        result["evaluation_contract_digest"] = run["evaluation_contract_digest"]
        _rewrite_result(tmp_path, run, result)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="reuses adaptation randomness"):
        analyze_manifest(manifest_path)


def test_irrelevant_model_origin_field_cannot_bypass_checkpoint_reuse_identity(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    run = manifest["runs"][0]  # type: ignore[index]
    run["evaluation_contract"]["model_origin"]["display_name"] = "alias"
    run["evaluation_contract_digest"] = semantic_sha256(run["evaluation_contract"])
    result_path = tmp_path / run["result_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["evaluation_contract"] = copy.deepcopy(run["evaluation_contract"])
    result["evaluation_contract_digest"] = run["evaluation_contract_digest"]
    _rewrite_result(tmp_path, run, result)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="invalid model origin"):
        analyze_manifest(manifest_path)


def test_training_contract_source_identity_tampering_is_detected(tmp_path: Path) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    registry = manifest["checkpoint_registry"]  # type: ignore[assignment]
    first = next(iter(registry.values()))
    first["training_contract"]["source_identity_digest"] = _digest("wrong-source")
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="semantic digest mismatch"):
        analyze_manifest(manifest_path)


def test_non_bi_candidate_coverage_is_rejected_even_when_its_summary_is_consistent(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    run = manifest["runs"][0]  # type: ignore[index]
    result_path = tmp_path / run["result_path"]  # type: ignore[index]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    coverage = CandidateCoverage.from_mappings({"row:0": 1}, {"row:0": 1})
    result["decision_outcomes"][0]["candidate_coverage"] = coverage.to_record()
    _rewrite_result(tmp_path, run, result)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="BI schedule"):
        analyze_manifest(manifest_path)


def test_typed_fit_failure_remains_in_the_operational_denominator(tmp_path: Path) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    run = manifest["runs"][0]  # type: ignore[index]
    result_path = tmp_path / run["result_path"]  # type: ignore[index]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    outcome = result["decision_outcomes"][0]
    outcome["status"] = "fit_failure"
    outcome["predicted_candidate"] = None
    outcome["failure_reason"] = "nonfinite_optimization"
    result["decision_failures"] = [
        {
            "participant_key": outcome["participant_key"],
            "decision_id": outcome["decision_id"],
            "status": "fit_failure",
            "stage": "adapter_fit",
            "reason": "nonfinite_optimization",
        }
    ]
    counts = result["decision_accounting"]["by_evidence_level"]["1"]
    counts["correct"] -= 1
    counts["fit_failure"] += 1
    counts["operational_hit_rate"] = counts["correct"] / counts["requested"]
    result["decision_accounting"]["evaluation_successful"] -= 1
    result["decision_accounting"]["evaluation_failed"] += 1
    _rewrite_result(tmp_path, run, result)
    _write_json(manifest_path, manifest)

    analysis = analyze_manifest(manifest_path)
    assert analysis["runs"][0]["decision_accounting"]["evaluation_failed"] == 1


def test_checkpoint_registry_hashes_the_physical_artifact(tmp_path: Path) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    checkpoint = next(iter(manifest["checkpoint_registry"].values()))
    checkpoint_path = tmp_path / checkpoint["checkpoint_path"]
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="checkpoint .* SHA mismatch"):
        analyze_manifest(manifest_path)


def test_training_procedure_uses_retraining_replicates_not_partitions(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _make_fixture(
        tmp_path, inference_scope="training_procedure", replicate_count=3
    )
    analysis = analyze_manifest(manifest_path)

    assert analysis["inference"]["scope"] == "training_procedure"
    assert analysis["inference"]["sampling_unit"] == "independent_training_replicate"
    contrast = analysis["planned_contrasts"][0]
    assert contrast["paired_unit"] == "independent_training_replicate"
    assert contrast["paired_bootstrap_interval"]["n_units"] == 3
    assert set(contrast["paired_unit_differences"]) == {
        "retrain:0",
        "retrain:1",
        "retrain:2",
    }


def test_training_procedure_requires_the_bound_power_plan_contract(tmp_path: Path) -> None:
    manifest_path, manifest = _make_fixture(
        tmp_path, inference_scope="training_procedure", replicate_count=3
    )
    del manifest["power_plan_contract"]
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="power_plan_contract"):
        analyze_manifest(manifest_path)


def test_same_checkpoint_cannot_masquerade_as_an_independent_retraining(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _make_fixture(
        tmp_path, inference_scope="training_procedure", replicate_count=3
    )
    runs = manifest["runs"]  # type: ignore[assignment]
    source = next(
        run
        for run in runs
        if run["training_replicate_key"] == "retrain:0"
        and run["partition_key"] == "partition:a"
        and run["evaluation_contract"]["arm_name"] == "adapted"
    )
    target = next(
        run
        for run in runs
        if run["training_replicate_key"] == "retrain:1"
        and run["partition_key"] == "partition:a"
        and run["evaluation_contract"]["arm_name"] == "adapted"
    )
    target["checkpoint_id"] = source["checkpoint_id"]
    target["evaluation_contract"]["model_origin"] = copy.deepcopy(
        source["evaluation_contract"]["model_origin"]
    )
    target["evaluation_contract_digest"] = semantic_sha256(target["evaluation_contract"])
    target_result_path = tmp_path / target["result_path"]
    target_result = json.loads(target_result_path.read_text(encoding="utf-8"))
    target_result["evaluation_contract"] = copy.deepcopy(target["evaluation_contract"])
    target_result["evaluation_contract_digest"] = target["evaluation_contract_digest"]
    target_result["checkpoint_sha256"] = source["evaluation_contract"]["model_origin"][
        "checkpoint_sha256"
    ]
    _rewrite_result(tmp_path, target, target_result)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="training seed"):
        analyze_manifest(manifest_path)


def test_scratch_origin_is_explicit_and_does_not_need_a_fake_checkpoint(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _make_fixture(tmp_path)
    runs = manifest["runs"]  # type: ignore[assignment]
    for run in runs:
        if run["evaluation_contract"]["arm_name"] != "adapted":
            continue
        initialization = {
            "architecture": "linear_scratch",
            "seed": run["evaluation_contract"]["adaptation"]["replicate_parameters"]["random_seed"],
        }
        origin = {
            "kind": "scratch",
            "initialization_contract_digest": semantic_sha256(initialization),
            "initialization": initialization,
        }
        run["checkpoint_id"] = None
        run["evaluation_contract"]["model_origin"] = origin
        run["evaluation_contract_digest"] = semantic_sha256(run["evaluation_contract"])
        result_path = tmp_path / run["result_path"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["evaluation_contract"] = copy.deepcopy(run["evaluation_contract"])
        result["evaluation_contract_digest"] = run["evaluation_contract_digest"]
        result["checkpoint_sha256"] = None
        _rewrite_result(tmp_path, run, result)
    used = {run["checkpoint_id"] for run in runs if run["checkpoint_id"] is not None}
    manifest["checkpoint_registry"] = {
        key: value for key, value in manifest["checkpoint_registry"].items() if key in used
    }
    first_adapted = next(run for run in runs if run["evaluation_contract"]["arm_name"] == "adapted")
    scratch_contract = ArmContract(
        arm_name="adapted",
        model_origin_kind="scratch",
        source_training_procedure=scratch_procedure_record(
            first_adapted["evaluation_contract"]["model_origin"]["initialization"]
        ),
        adaptation_procedure=first_adapted["evaluation_contract"]["adaptation"]["procedure"],
        allowed_variation_axes=("training_replicate", "participant_partition"),
    )
    manifest["arm_contracts"]["adapted"] = {
        "contract": scratch_contract.record(),
        "digest": scratch_contract.digest(),
    }
    _write_json(manifest_path, manifest)

    analysis = analyze_manifest(manifest_path)
    assert analysis["metrics"]["adapted"]


def test_legacy_v1_manifest_is_not_accepted(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    _write_json(path, {"schema": "n2p3_bi2014a_cross_decision_experiment/1"})
    with pytest.raises(ValueError, match="v1 is unsupported"):
        analyze_manifest(path)
