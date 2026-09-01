"""Research contracts that bind experiments, artifacts, and inference scope."""

from research.contracts import (
    ARM_CONTRACT_SCHEMA,
    DECISION_PLAN_SCHEMA,
    EVALUATION_RUN_CONTRACT_SCHEMA,
    MIN_PROMOTION_HIT_AT_R,
    STATISTICAL_DESIGN_SCHEMA,
    TRAINING_RUN_CONTRACT_SCHEMA,
    ArmContract,
    DecisionPlanContract,
    DecisionPlanEntry,
    EvaluationRunContract,
    ParticipantSelectionFailure,
    StatisticalDesignContract,
    TrainingRunContract,
    assert_promotion_evidence_gate,
    semantic_sha256,
)

__all__ = [
    "ARM_CONTRACT_SCHEMA",
    "DECISION_PLAN_SCHEMA",
    "EVALUATION_RUN_CONTRACT_SCHEMA",
    "MIN_PROMOTION_HIT_AT_R",
    "STATISTICAL_DESIGN_SCHEMA",
    "TRAINING_RUN_CONTRACT_SCHEMA",
    "ArmContract",
    "DecisionPlanContract",
    "DecisionPlanEntry",
    "EvaluationRunContract",
    "ParticipantSelectionFailure",
    "StatisticalDesignContract",
    "TrainingRunContract",
    "assert_promotion_evidence_gate",
    "semantic_sha256",
]
