from __future__ import annotations

from dataclasses import replace

import pytest

from research.contracts import (
    MIN_PROMOTION_HIT_AT_R,
    ArmContract,
    DecisionPlanContract,
    DecisionPlanEntry,
    EvaluationRunContract,
    ParticipantSelectionFailure,
    StatisticalDesignContract,
    TrainingRunContract,
    assert_contract_digest,
    assert_promotion_evidence_gate,
    resolve_evidence_floor,
    semantic_sha256,
    training_procedure_record,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _training_contract() -> TrainingRunContract:
    return TrainingRunContract(
        source_cache_sha256=SHA_A,
        source_identity_digest=SHA_B,
        source_snapshot_sha256=SHA_C,
        architecture={"name": "n2p3", "kernel": 35},
        preprocessing={"sfreq": 128.0, "l_freq": 0.1, "tmax_ms": 1200.0},
        optimizer={"name": "adamw", "epochs": 100},
        validation={"kind": "group_disjoint", "full_refit": True},
        objective={"trial_ce": 1.0},
        seed=7,
        training_participant_keys=("BI:01", "BI:02"),
        holdout_participant_keys=("BI:03",),
    )


def test_semantic_digest_ignores_mapping_order_but_not_meaning() -> None:
    first = {"b": [2, 3], "a": {"x": 1.0}}
    second = {"a": {"x": 1.0}, "b": [2, 3]}
    assert semantic_sha256(first) == semantic_sha256(second)
    assert semantic_sha256(first) != semantic_sha256({"b": [3, 2], "a": {"x": 1.0}})


def test_training_contract_rejects_participant_overlap() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        replace(_training_contract(), holdout_participant_keys=("BI:02",))


def test_evaluation_contract_binds_checkpoint_and_resolved_protocol() -> None:
    training = _training_contract()
    evaluation = EvaluationRunContract(
        arm_name="zero_shot_source",
        model_origin={
            "kind": "checkpoint",
            "checkpoint_sha256": SHA_A,
            "training_contract_digest": training.digest(),
        },
        target_cache_sha256=SHA_B,
        target_identity_digest=SHA_C,
        source_snapshot_sha256=SHA_A,
        target_protocol={"calibration_decisions": 5, "test_repetitions": 2},
        adaptation={"head": "zero_shot", "normalization": "source"},
        decision={"aggregation": "mean", "tie_policy": "abstain"},
        requested_participant_keys=("BI:03",),
        evidence_scope={"stage": "development", "population": "BI2014a"},
    )
    changed = replace(
        evaluation, target_protocol={"calibration_decisions": 4, "test_repetitions": 2}
    )
    assert evaluation.digest() != changed.digest()
    assert_contract_digest(evaluation.record(), evaluation.digest(), name="evaluation")
    with pytest.raises(ValueError, match="digest mismatch"):
        assert_contract_digest(changed.record(), evaluation.digest(), name="evaluation")


def test_evaluation_contract_supports_real_scratch_origin_without_fake_checkpoint() -> None:
    contract = EvaluationRunContract(
        arm_name="linear_scratch",
        model_origin={
            "kind": "scratch",
            "initialization_contract_digest": SHA_A,
            "seed": 9,
        },
        target_cache_sha256=SHA_B,
        target_identity_digest=SHA_C,
        source_snapshot_sha256=SHA_A,
        target_protocol={"calibration_decisions": 5, "test_repetitions": 2},
        adaptation={"head": "linear", "normalization": "target_prefix"},
        decision={"aggregation": "mean", "tie_policy": "abstain"},
        requested_participant_keys=("BI:03",),
        evidence_scope={"stage": "development", "population": "BI2014a"},
    )
    assert contract.record()["model_origin"]["kind"] == "scratch"


def test_training_procedure_inference_requires_power_plan() -> None:
    common = dict(
        participant_cluster_key="participant_key",
        training_replicate_keys=("seed:1", "seed:2", "seed:3"),
        partition_keys=("partition:0",),
        planned_contrasts=(("joint", "single_domain"),),
    )
    conditional = StatisticalDesignContract(inference_scope="conditional_frozen_models", **common)
    assert conditional.allows_training_procedure_claims is False
    with pytest.raises(ValueError, match="power plan"):
        StatisticalDesignContract(inference_scope="training_procedure", **common)
    procedure = StatisticalDesignContract(
        inference_scope="training_procedure", power_plan_digest=SHA_A, **common
    )
    assert procedure.allows_training_procedure_claims is True


def test_decision_plan_binds_authority_key_truth_and_target_dataset() -> None:
    plan = DecisionPlanContract(
        target_cache_sha256=SHA_A,
        target_identity_digest=SHA_B,
        requested_participant_keys=("BI:03",),
        entries=(DecisionPlanEntry("BI:03", "selection:1", "row:1|column:2"),),
    )
    assert DecisionPlanContract.from_record(plan.record()).digest() == plan.digest()
    with pytest.raises(ValueError, match="unique"):
        DecisionPlanContract(
            target_cache_sha256=SHA_A,
            target_identity_digest=SHA_B,
            requested_participant_keys=("BI:03",),
            entries=(plan.entries[0], plan.entries[0]),
        )


def test_decision_plan_keeps_requested_participant_without_inventing_decision() -> None:
    plan = DecisionPlanContract(
        target_cache_sha256=SHA_A,
        target_identity_digest=SHA_B,
        requested_participant_keys=("BI:03", "BI:04"),
        entries=(DecisionPlanEntry("BI:03", "selection:1", "row:1|column:2"),),
        participant_selection_failures=(
            ParticipantSelectionFailure(
                participant_key="BI:04",
                stage="decision_selection",
                reason="insufficient_decisions",
            ),
        ),
    )
    assert plan.requested_participant_keys == ("BI:03", "BI:04")
    assert plan.participant_keys == ("BI:03",)
    with pytest.raises(ValueError, match="every requested"):
        replace(plan, participant_selection_failures=())


def test_arm_contract_freezes_source_and_target_procedures() -> None:
    training = _training_contract()
    source_procedure = training_procedure_record(training)
    arm = ArmContract(
        arm_name="zero_shot_source",
        model_origin_kind="checkpoint",
        source_training_procedure=source_procedure,
        adaptation_procedure={"head": "zero_shot", "normalization": "source"},
        allowed_variation_axes=("training_replicate", "participant_partition"),
    )
    assert ArmContract.from_record(arm.record()).digest() == arm.digest()
    changed = replace(training, optimizer={"name": "sgd", "epochs": 100})
    assert training_procedure_record(changed) != arm.source_training_procedure
    with pytest.raises(ValueError, match="exactly"):
        replace(arm, allowed_variation_axes=("training_replicate",))


def test_training_procedure_record_separates_controls_from_fit_outcomes() -> None:
    base = _training_contract()
    base = replace(
        base,
        optimizer={
            "name": "torch.optim.Adam",
            "selection_config": {"epochs": 30, "batch_size": 256, "lr": 0.001},
            "refit_config": {"epochs": 8, "batch_size": 256, "lr": 0.001},
            "selection_execution": {"fused_adam": False, "compile_mode": None},
            "refit_execution": {"fused_adam": False, "compile_mode": None},
            "selection_runtime": {"elapsed_s": 41.5},
            "refit_runtime": {"elapsed_s": 12.25},
            "optimizer_rows_per_epoch": 45712,
        },
        validation={
            "strategy": "group_disjoint_epoch_selection_then_full_source_refit",
            "group_key": "local_subject_id",
            "selected_epoch_zero_based": 7,
            "refit_epochs": 8,
            "selection_calibration_source": "source_group_validation",
            "selection_domain": None,
            "full_source_refit": True,
        },
        objective={
            "name": "weighted_binary_cross_entropy",
            "effective_pos_weight": 1.1377,
            "training_prior": 0.1667,
            "qc_ptp_uv": 100.0,
            "input_statistics_scope": "all_source_rows",
            "label_counts": [38060, 7652],
            "source_risk_selection": 0.2311,
            "source_risk_refit": 0.2298,
        },
    )
    # A different target partition reruns inner validation on its own holdout:
    # the realized epoch count, label balance, risk, and telemetry all move.
    other = replace(
        base,
        holdout_participant_keys=("BI:04",),
        optimizer={
            **base.optimizer,
            "refit_config": {"epochs": 7, "batch_size": 256, "lr": 0.001},
            "selection_runtime": {"elapsed_s": 39.75},
            "refit_runtime": {"elapsed_s": 11.5},
            "optimizer_rows_per_epoch": 43218,
        },
        validation={
            **base.validation,
            "selected_epoch_zero_based": 6,
            "refit_epochs": 7,
        },
        objective={
            **base.objective,
            "effective_pos_weight": 1.1422,
            "training_prior": 0.1663,
            "label_counts": [36011, 7207],
            "source_risk_selection": 0.2442,
            "source_risk_refit": 0.2401,
        },
    )
    assert training_procedure_record(other) == training_procedure_record(base)
    # But a genuine control change must move the procedure.
    control = replace(
        base,
        objective={**base.objective, "qc_ptp_uv": 120.0},
    )
    assert training_procedure_record(control) != training_procedure_record(base)
    assert training_procedure_record(base)["optimizer"]["selection_config"]["epochs"] == 30
    assert "refit_config" not in training_procedure_record(base)["optimizer"]


def test_training_procedure_record_binds_transfer_decision_and_effective_batch() -> None:
    base = _training_contract()
    base = replace(
        base,
        optimizer={
            "name": "torch.optim.Adam",
            "selection_config": {"epochs": 30, "batch_size": 256},
            "selection_execution": {"fused_adam": True, "compile_mode": "default"},
            "refit_execution": {"fused_adam": True, "compile_mode": "default"},
            "effective_batch_contract": {
                "requested_batch_size": 256,
                "selection_batch_size": 256,
                "refit_batch_size": 256,
                "selection_oom_retries": 0,
                "refit_oom_retries": 0,
            },
        },
        objective={
            "name": "weighted_binary_cross_entropy",
            "qc_ptp_uv": 100.0,
            "input_statistics_scope": "all_source_rows",
            "feature_transfer": {
                "schema": "n2p3_feature_domain_transfer/2",
                "source_domain_count": 2,
                "source_drop_p": 0.3,
                "source_drop_mode": "sample",
            },
            "feature_alignment_contract": {"weight": 0.05},
            "decision_loss": {
                "schema": "n2p3_decision_loss_config/1",
                "lambda_candidate": 0.2,
                "temperature": 1.0,
            },
            "source_residual_diagnostics": {"run_max_ratio": 0.1},
        },
    )
    baseline = training_procedure_record(base)

    changed_drop = replace(
        base,
        objective={
            **base.objective,
            "feature_transfer": {
                **base.objective["feature_transfer"],
                "source_drop_p": 0.5,
            },
        },
    )
    changed_decision = replace(
        base,
        objective={
            **base.objective,
            "decision_loss": {
                **base.objective["decision_loss"],
                "temperature": 0.7,
            },
        },
    )
    changed_batch = replace(
        base,
        optimizer={
            **base.optimizer,
            "effective_batch_contract": {
                **base.optimizer["effective_batch_contract"],
                "selection_batch_size": 128,
                "selection_oom_retries": 1,
            },
        },
    )
    changed_diagnostics = replace(
        base,
        objective={
            **base.objective,
            "source_residual_diagnostics": {"run_max_ratio": 0.4},
        },
    )

    assert training_procedure_record(changed_drop) != baseline
    assert training_procedure_record(changed_decision) != baseline
    assert training_procedure_record(changed_batch) != baseline
    assert training_procedure_record(changed_diagnostics) == baseline


# --------------------------------------------------------------------------
# Promotion evidence gate: the horizon is a declared parameter, not a constant.
# --------------------------------------------------------------------------


def test_resolve_evidence_floor_defaults_to_promotion_depth() -> None:
    assert resolve_evidence_floor(None) == MIN_PROMOTION_HIT_AT_R
    assert resolve_evidence_floor(8) == 8
    assert resolve_evidence_floor(2) == 2


def test_resolve_evidence_floor_rejects_values_below_one() -> None:
    for floor in (0, -1):
        with pytest.raises(ValueError, match="at least 1"):
            resolve_evidence_floor(floor)


def test_resolve_evidence_floor_rejects_non_integers() -> None:
    for floor in (True, 2.0, "2"):
        with pytest.raises(ValueError, match="must be an integer"):
            resolve_evidence_floor(floor)


def test_gate_default_horizon_is_promotion_depth() -> None:
    """Callers that declare nothing keep the promotion requirement."""

    assert_promotion_evidence_gate(8)
    with pytest.raises(ValueError, match=r"test_repetitions >= 8"):
        assert_promotion_evidence_gate(7)


def test_gate_accepts_declared_shallower_horizon() -> None:
    assert_promotion_evidence_gate(2, minimum_evidence_level=2)
    assert_promotion_evidence_gate(8, minimum_evidence_level=2)


def test_gate_rejects_test_reps_below_declared_horizon() -> None:
    with pytest.raises(ValueError, match=r"test_repetitions >= 3"):
        assert_promotion_evidence_gate(2, minimum_evidence_level=3)


def test_gate_primary_level_must_clear_declared_horizon() -> None:
    assert_promotion_evidence_gate(
        8, primary_evidence_level=2, minimum_evidence_level=2
    )
    for level in (1, 9):
        with pytest.raises(ValueError, match=r"2 <= R <= 8 \(got \d+\)"):
            assert_promotion_evidence_gate(
                8, primary_evidence_level=level, minimum_evidence_level=2
            )


def test_gate_evidence_levels_must_contain_declared_horizon() -> None:
    assert_promotion_evidence_gate(
        8,
        evidence_levels=(1, 2, 4, 8),
        primary_evidence_level=4,
        minimum_evidence_level=2,
    )
    with pytest.raises(ValueError, match="missing hit@2"):
        assert_promotion_evidence_gate(
            8, evidence_levels=(1, 4, 8), minimum_evidence_level=2
        )


def test_gate_still_requires_promotion_level_when_undeclared() -> None:
    with pytest.raises(ValueError, match="missing hit@8"):
        assert_promotion_evidence_gate(8, evidence_levels=(1, 2, 4))
