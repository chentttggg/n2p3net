"""Evaluate known early decisions -> later unknown candidate-task decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.calibration import (  # noqa: E402
    calibration_data_from_model,
    fit_weighted_logit_temperature,
)
from data.artifact import FoldLocalArtifactPolicy, apply_fold_local_artifact_policy  # noqa: E402
from data.candidate_task import (  # noqa: E402
    CandidateTaskContract,
    candidate_task_contract_from_provenance,
    validate_candidate_membership_metadata,
)
from data.contract import assert_causal_p300_input_contract  # noqa: E402
from data.epochs import load_epoch_dataset, loaded_epoch_cache_attestation  # noqa: E402
from data.identity import IdentityExclusionPolicy  # noqa: E402
from data.qc_features import compute_epoch_qc_features  # noqa: E402
from models.n2p3net import N2P3Net  # noqa: E402
from research.contracts import (  # noqa: E402
    MIN_PROMOTION_HIT_AT_R,
    DecisionPlanContract,
    DecisionPlanEntry,
    ParticipantSelectionFailure,
    assert_promotion_evidence_gate,
    semantic_sha256,
)
from research.evaluation import (  # noqa: E402
    build_evaluation_run_contract,
    checkpoint_model_origin,
    scratch_model_origin,
    source_snapshot_sha256_from_archive_manifest,
)
from research.execution import ExpectedSubjectError  # noqa: E402
from train.device import get_device  # noqa: E402
from transfer.candidate_decision import (  # noqa: E402
    decision_outcomes_at_repetition,
    expected_candidate_counts,
    row_column_target,
)
from transfer.candidate_within_subject import (  # noqa: E402
    candidate_calibration_decision_split,
)
from transfer.checkpoint import (  # noqa: E402
    assert_checkpoint_target_identity_excluded,
    checkpoint_classifier_is_trained,
    checkpoint_input_stats,
    checkpoint_scores_to_llr,
    checkpoint_training_contract,
    load_checkpoint_payload,
    load_n2p3_trunk_checkpoint,
    predict_n2p3_checkpoint,
)
from transfer.cohort import read_subject_manifest, resolve_subject_scope  # noqa: E402
from transfer.outcomes import (  # noqa: E402
    CandidateCoverage,
    DecisionKey,
    DecisionOutcome,
    DecisionStatus,
    build_decision_outcome_accounting,
)
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig  # noqa: E402
from transfer.within_subject import chronological_time_validation_split  # noqa: E402


def _load_trunk(
    path: str | Path | dict[str, object],
    dataset,
    *,
    target_subject: str,
    identity_exclusion_policy: IdentityExclusionPolicy,
) -> N2P3Net:
    trunk, _ = load_n2p3_trunk_checkpoint(
        path,
        dataset,
        target_subject=target_subject,
        identity_exclusion_policy=identity_exclusion_policy,
    )
    return trunk


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _failed_decision_outcomes(
    *,
    subject: str,
    requested_decisions: tuple[str, ...],
    failure_reasons: dict[str, str],
    target_by_decision: dict[str, str],
    test_repetitions: int,
    candidate_task: CandidateTaskContract,
) -> tuple[DecisionOutcome, ...]:
    outcomes: list[DecisionOutcome] = []
    for decision in requested_decisions:
        if decision not in failure_reasons:
            continue
        for repetition in range(1, test_repetitions + 1):
            outcomes.append(
                DecisionOutcome(
                    key=DecisionKey(subject, decision),
                    evidence_level=repetition,
                    status=DecisionStatus.INCOMPLETE,
                    coverage=CandidateCoverage.from_mappings(
                        expected_candidate_counts(candidate_task, repetition), {}
                    ),
                    target_candidate=target_by_decision[decision],
                    failure_reason=failure_reasons[decision],
                )
            )
    return tuple(outcomes)


def _fit_failure_decision_outcomes(
    *,
    subject: str,
    decisions: tuple[str, ...],
    test_repetitions: int,
    failure: ExpectedSubjectError,
    target_by_decision: dict[str, str],
    candidate_task: CandidateTaskContract,
) -> tuple[DecisionOutcome, ...]:
    return tuple(
        DecisionOutcome(
            key=DecisionKey(subject, decision),
            evidence_level=level,
            status=DecisionStatus.FIT_FAILURE,
            coverage=CandidateCoverage.from_mappings(
                expected_candidate_counts(candidate_task, level), {}
            ),
            target_candidate=target_by_decision[decision],
            failure_reason=f"{failure.stage}:{failure.code.value}",
        )
        for decision in decisions
        for level in range(1, test_repetitions + 1)
    )


def _accounting_record(
    outcomes: tuple[DecisionOutcome, ...],
    *,
    subject: str,
    requested_decisions: tuple[str, ...],
    eligible_decisions: tuple[str, ...],
    test_repetitions: int,
) -> dict[str, object]:
    accounting = build_decision_outcome_accounting(
        outcomes,
        requested_decisions=[DecisionKey(subject, decision) for decision in requested_decisions],
        evidence_levels=range(1, test_repetitions + 1),
    )
    record = accounting.to_record()
    data_eligible = len(eligible_decisions)
    failed_evaluation_keys = {
        outcome.key.decision_id
        for outcome in outcomes
        if outcome.key.decision_id in eligible_decisions
        and outcome.status in {DecisionStatus.INCOMPLETE, DecisionStatus.FIT_FAILURE}
    }
    record["data_eligible"] = data_eligible
    record["data_ineligible"] = len(requested_decisions) - data_eligible
    record["evaluation_successful"] = data_eligible - len(failed_evaluation_keys)
    record["evaluation_failed"] = len(failed_evaluation_keys)
    return record


def _build_decision_plan(
    dataset,
    *,
    candidate_task: CandidateTaskContract,
    requested_subjects: tuple[str, ...],
    requested_decisions_by_subject: dict[str, tuple[str, ...]],
    participant_by_subject: dict[str, str],
    participant_failure_reasons: dict[str, str],
    target_cache_sha256: str,
) -> tuple[DecisionPlanContract, dict[str, dict[str, str]]]:
    if dataset.metadata is None or dataset.identity_table is None:
        raise ValueError("Candidate decision plan requires metadata and participant identity.")
    selection_ids = dataset.metadata["selection_id"].astype(str).to_numpy()
    subjects = np.asarray(dataset.subject_ids).astype(str)
    target_rows = dataset.metadata["target_row"].to_numpy(dtype=np.int64)
    target_cols = dataset.metadata["target_col"].to_numpy(dtype=np.int64)
    entries: list[DecisionPlanEntry] = []
    participant_failures: list[ParticipantSelectionFailure] = []
    local_targets: dict[str, dict[str, str]] = {}
    for subject in requested_subjects:
        local_targets[subject] = {}
        requested_decisions = requested_decisions_by_subject.get(subject, ())
        failure_reason = participant_failure_reasons.get(subject)
        if failure_reason is not None:
            if not failure_reason:
                raise ValueError(
                    f"requested participant {subject!r} has an empty selection failure reason."
                )
            participant_failures.append(
                ParticipantSelectionFailure(
                    participant_key=participant_by_subject[subject],
                    stage="decision_selection",
                    reason=failure_reason,
                )
            )
            continue
        if not requested_decisions:
            raise ValueError(
                f"requested participant {subject!r} has neither a decision nor a failure."
            )
        for decision in requested_decisions:
            rows = (subjects == subject) & (selection_ids == decision)
            unique_rows = np.unique(target_rows[rows])
            unique_cols = np.unique(target_cols[rows])
            if (
                len(unique_rows) != 1
                or len(unique_cols) != 1
                or not 0 <= int(unique_rows[0]) < candidate_task.n_rows
                or not 0 <= int(unique_cols[0]) < candidate_task.n_columns
            ):
                raise ValueError(
                    f"Candidate decision {subject!r}/{decision!r} lacks one valid frozen target."
                )
            target = row_column_target(int(unique_rows[0]), int(unique_cols[0]))
            local_targets[subject][decision] = target
            entries.append(
                DecisionPlanEntry(
                    participant_key=participant_by_subject[subject],
                    decision_id=decision,
                    target_candidate=target,
                )
            )
    plan = DecisionPlanContract(
        target_cache_sha256=target_cache_sha256,
        target_identity_digest=dataset.identity_table.digest(),
        requested_participant_keys=tuple(
            participant_by_subject[subject] for subject in requested_subjects
        ),
        entries=tuple(entries),
        participant_selection_failures=tuple(participant_failures),
    )
    return plan, local_targets


def _validate_cached_candidate_labels(dataset) -> CandidateTaskContract:
    if dataset.metadata is None or dataset.y is None:
        raise ValueError("Candidate cache requires metadata and binary labels.")
    contract = candidate_task_contract_from_provenance(dataset.provenance)
    validate_candidate_membership_metadata(
        dataset.metadata,
        contract,
        labels=np.asarray(dataset.y),
    )
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--training-replicate-key", required=True)
    parser.add_argument("--partition-key", required=True)
    parser.add_argument("--source-snapshot-manifest", type=Path, required=True)
    parser.add_argument(
        "--identity-exclusion-policy",
        required=True,
        choices=("source", "source_or_global", "global"),
    )
    parser.add_argument("--calibration-selections", type=int, default=5)
    parser.add_argument("--test-reps", type=int, default=MIN_PROMOTION_HIT_AT_R)
    parser.add_argument("--max-test-selections", type=int, default=None)
    parser.add_argument(
        "--head",
        choices=("auto", "zero_shot", "classifier_fine", "linear", "mlp16", "full_fine", "adapter"),
        default="auto",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--epoch-selection",
        choices=("fixed_budget", "target_time_split"),
        default="fixed_budget",
    )
    parser.add_argument(
        "--adapt-batchnorm",
        action="store_true",
        help="Experimental full-fine arm: update target BatchNorm running statistics.",
    )
    parser.add_argument(
        "--normalization",
        choices=("auto", "source", "target_prefix", "shrinkage"),
        default="auto",
    )
    parser.add_argument("--target-stat-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fold-local-qc",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--target-subjects-file",
        type=Path,
        default=None,
        help="JSON list or newline-delimited target subjects assigned to this checkpoint.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        assert_promotion_evidence_gate(args.test_reps, name="candidate promotion runner")
    except ValueError as error:
        parser.error(str(error))
    source_snapshot_sha256 = source_snapshot_sha256_from_archive_manifest(
        args.source_snapshot_manifest
    )
    if args.target_subjects_file is not None and args.max_subjects is not None:
        parser.error(
            "--target-subjects-file defines the exact denominator; do not combine it "
            "with --max-subjects"
        )

    device = torch.device(args.device) if args.device != "auto" else get_device()
    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    target_cache_sha256 = str(loaded_epoch_cache_attestation(dataset)["sha256"])
    assert_causal_p300_input_contract(dataset.preprocessing)
    candidate_task = _validate_cached_candidate_labels(dataset)
    split = candidate_calibration_decision_split(
        dataset,
        calibration_selections=args.calibration_selections,
        test_repetitions=args.test_reps,
        max_test_selections=args.max_test_selections,
    )
    manifest_subjects = (
        read_subject_manifest(args.target_subjects_file)
        if args.target_subjects_file is not None
        else None
    )
    try:
        requested_order, usable_order = resolve_subject_scope(
            np.asarray(dataset.subject_ids).astype(str),
            split.usable_subjects,
            requested_subjects=manifest_subjects,
            max_subjects=args.max_subjects,
        )
    except ValueError as error:
        parser.error(str(error))
    requested_subjects = set(requested_order)
    if dataset.identity_table is None:
        raise ValueError("Candidate cache lacks the current participant identity table.")
    participant_by_subject = {
        subject: dataset.identity_table.record_for(subject).authority_key(
            args.identity_exclusion_policy
        )
        for subject in requested_order
    }
    decision_plan, target_by_subject = _build_decision_plan(
        dataset,
        candidate_task=candidate_task,
        requested_subjects=tuple(requested_order),
        requested_decisions_by_subject=split.requested_test_selections_by_subject,
        participant_by_subject=participant_by_subject,
        participant_failure_reasons=split.excluded_subjects,
        target_cache_sha256=target_cache_sha256,
    )
    subjects = np.asarray(usable_order)

    checkpoint_payload = load_checkpoint_payload(args.checkpoint) if args.checkpoint else None
    if (
        checkpoint_payload is not None
        and checkpoint_training_contract(checkpoint_payload).source_snapshot_sha256
        != source_snapshot_sha256
    ):
        raise ValueError(
            "checkpoint TrainingRunContract source snapshot disagrees with the "
            "verified physical source freeze."
        )
    if checkpoint_payload is not None:
        if not requested_order:
            raise AssertionError("validated target scope unexpectedly has no participant")
        _load_trunk(
            checkpoint_payload,
            dataset,
            target_subject=requested_order[0],
            identity_exclusion_policy=args.identity_exclusion_policy,
        )
        for target_subject in requested_order[1:]:
            assert_checkpoint_target_identity_excluded(
                checkpoint_payload,
                dataset,
                target_subject=target_subject,
                identity_exclusion_policy=args.identity_exclusion_policy,
            )
    checkpoint_stats = (
        checkpoint_input_stats(checkpoint_payload, dataset.n_channels)
        if checkpoint_payload is not None
        else None
    )
    head_mode = args.head
    if head_mode == "auto":
        head_mode = "zero_shot" if checkpoint_payload is not None else "linear"
    if head_mode in {"zero_shot", "classifier_fine"} and checkpoint_payload is None:
        parser.error(f"--head {head_mode} requires --checkpoint")
    if head_mode in {"zero_shot", "classifier_fine"} and not checkpoint_classifier_is_trained(
        checkpoint_payload
    ):
        parser.error(f"--head {head_mode} requires a supervised classifier checkpoint")
    if head_mode == "zero_shot" and args.fold_local_qc:
        parser.error("zero_shot cannot fit target calibration QC")
    normalization = args.normalization
    if normalization == "auto":
        normalization = "source" if checkpoint_payload is not None else "target_prefix"
    if checkpoint_payload is None and normalization in {"source", "shrinkage"}:
        parser.error(f"--normalization {normalization} requires --checkpoint")
    if head_mode == "zero_shot" and normalization != "source":
        parser.error("zero_shot uses source checkpoint statistics")
    epochs = (
        0
        if head_mode == "zero_shot"
        else int(args.epochs)
        if args.epochs is not None
        else 5
        if head_mode in {"classifier_fine", "full_fine"}
        else 30
    )
    lr = (
        0.0
        if head_mode == "zero_shot"
        else float(args.lr)
        if args.lr is not None
        else 1e-4
        if head_mode in {"classifier_fine", "full_fine"}
        else 1e-3
    )
    trial_mask = (
        np.asarray(dataset.trial_channel_mask, dtype=bool)
        if dataset.trial_channel_mask is not None
        else np.broadcast_to(
            np.asarray(dataset.channel_mask, dtype=bool), dataset.X.shape[:2]
        ).copy()
    )
    qc_features = dataset.qc_features
    if qc_features is None and args.fold_local_qc:
        qc_features = compute_epoch_qc_features(
            dataset.X,
            channel_mask=np.asarray(dataset.channel_mask, dtype=bool),
            trial_channel_mask=trial_mask,
        )
    evidence = np.asarray(dataset.event_timeline.evidence_indices, dtype=np.int64)
    available_events = np.flatnonzero(evidence >= 0)
    onset_by_epoch = np.empty(dataset.n_epochs, dtype=float)
    available_by_epoch = np.empty(dataset.n_epochs, dtype=float)
    onset_by_epoch[evidence[available_events]] = np.asarray(
        dataset.event_timeline.onset_times_s, dtype=float
    )[available_events]
    available_by_epoch[evidence[available_events]] = np.asarray(
        dataset.event_timeline.evidence_available_times_s, dtype=float
    )[available_events]

    records = []
    outcomes_by_subject: dict[str, tuple[DecisionOutcome, ...]] = {}
    for subject in subjects:
        subject_rows = np.flatnonzero(dataset.subject_ids == subject)
        pre_rows = subject_rows[split.calibration_mask[subject_rows]]
        post_rows = subject_rows[split.test_mask[subject_rows]]
        if not len(pre_rows) or not len(post_rows):
            raise AssertionError(
                f"usable candidate-task subject {subject!s} lacks calibration or test rows."
            )
        local_rows = np.concatenate((pre_rows, post_rows))
        local_train = np.zeros(len(local_rows), dtype=bool)
        local_train[: len(pre_rows)] = True
        local_test = ~local_train
        local_meta = dataset.metadata.iloc[local_rows]
        local_groups = local_meta["selection_id"].astype(str).to_numpy()
        local_X = np.asarray(dataset.X[local_rows], dtype=np.float32)
        local_y = np.asarray(dataset.y[local_rows], dtype=np.int64)
        local_mask = trial_mask[local_rows]
        local_qc = None if qc_features is None else qc_features.subset(local_rows)
        artifact_audit = {"enabled": False, "scope": "known_calibration_decisions"}
        if args.fold_local_qc:
            local_X, local_mask, effective_train, artifact_audit = apply_fold_local_artifact_policy(
                FoldLocalArtifactPolicy(),
                local_X,
                local_groups,
                local_train,
                local_test,
                local_mask,
                local_qc,
                materialize_masked_data=True,
            )
            effective_train &= local_train
            before_counts = np.bincount(local_y[local_train], minlength=2)
            after_counts = np.bincount(local_y[effective_train], minlength=2)
            artifact_audit["train_label_counts_before"] = before_counts.astype(int).tolist()
            artifact_audit["train_label_counts_after"] = after_counts.astype(int).tolist()
            artifact_audit["train_label_retention"] = (
                after_counts / np.maximum(before_counts, 1)
            ).tolist()
        else:
            effective_train = local_train

        trunk = (
            _load_trunk(
                checkpoint_payload,
                dataset,
                target_subject=str(subject),
                identity_exclusion_policy=args.identity_exclusion_policy,
            )
            if args.checkpoint
            else N2P3Net(
                dataset.n_channels,
                n_times=dataset.n_times,
                sfreq=dataset.preprocessing.sfreq,
                tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
                pooling_mode="ms_flatten",
            )
        )
        suffix_X = local_X[local_test]
        suffix_mask = local_mask[local_test]
        y_post = local_y[local_test]
        if head_mode == "zero_shot":
            assert checkpoint_payload is not None and checkpoint_stats is not None
            suffix_logits = predict_n2p3_checkpoint(
                trunk,
                suffix_X,
                input_stats=checkpoint_stats,
                device=device,
                trial_channel_mask=suffix_mask,
            )
            suffix_llr, calibration_record = checkpoint_scores_to_llr(
                checkpoint_payload, suffix_logits
            )
            total_parameters = trunk.parameter_count()
            trainable_parameters = 0
            refit_record = {"enabled": False, "mode": "zero_shot"}
        else:
            adapter = SubjectAdapter(
                trunk,
                config=SubjectAdapterConfig(
                    head_kind=head_mode,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    lr=lr,
                    seed=args.seed,
                    val_group_fraction=(None if args.epoch_selection == "fixed_budget" else 0.1),
                    refit_full_prefix=args.epoch_selection == "target_time_split",
                    freeze_batchnorm_stats=not args.adapt_batchnorm,
                    input_statistics=normalization,
                    target_stat_weight=args.target_stat_weight,
                    input_mean=None if checkpoint_stats is None else checkpoint_stats[0],
                    input_std=None if checkpoint_stats is None else checkpoint_stats[1],
                ),
                device=device,
            )
            calibration_rows = local_rows[effective_train]
            inner = None
            try:
                if args.epoch_selection == "target_time_split":
                    inner = chronological_time_validation_split(
                        onset_by_epoch[calibration_rows],
                        available_by_epoch[calibration_rows],
                        local_y[effective_train],
                        epoch_start_offset_s=float(dataset.preprocessing.tmin_ms) / 1000.0,
                    )
                    adapter.fit(
                        local_X[effective_train],
                        local_y[effective_train],
                        training_mask=inner.train_mask,
                        validation_mask=inner.validation_mask,
                        trial_channel_mask=local_mask[effective_train],
                    )
                else:
                    adapter.fit(
                        local_X[effective_train],
                        local_y[effective_train],
                        trial_channel_mask=local_mask[effective_train],
                    )
                suffix_logits = adapter.predict_logit(suffix_X, trial_channel_mask=suffix_mask)
            except ExpectedSubjectError as failure:
                requested_test = split.requested_test_selections_by_subject[str(subject)]
                eligible_test = split.test_selections_by_subject[str(subject)]
                failed_test = split.failed_test_selections_by_subject[str(subject)]
                subject_outcomes = (
                    *_fit_failure_decision_outcomes(
                        subject=str(subject),
                        decisions=eligible_test,
                        test_repetitions=args.test_reps,
                        failure=failure,
                        target_by_decision=target_by_subject[str(subject)],
                        candidate_task=candidate_task,
                    ),
                    *_failed_decision_outcomes(
                        subject=str(subject),
                        requested_decisions=requested_test,
                        failure_reasons=failed_test,
                        test_repetitions=args.test_reps,
                        target_by_decision=target_by_subject[str(subject)],
                        candidate_task=candidate_task,
                    ),
                )
                outcomes_by_subject[str(subject)] = tuple(subject_outcomes)
                records.append(
                    {
                        "subject": str(subject),
                        "participant_key": participant_by_subject[str(subject)],
                        "binary_auc": None,
                        "eligible_hit_rate_by_evidence": {
                            level: 0.0 for level in range(1, args.test_reps + 1)
                        },
                        "fit_failure": failure.record(subject=str(subject)),
                        "decision_accounting": _accounting_record(
                            tuple(subject_outcomes),
                            subject=str(subject),
                            requested_decisions=requested_test,
                            eligible_decisions=eligible_test,
                            test_repetitions=args.test_reps,
                        ),
                        "decision_outcomes": [
                            {
                                **outcome.to_record(),
                                "participant_key": participant_by_subject[str(subject)],
                            }
                            for outcome in subject_outcomes
                        ],
                    }
                )
                continue
            if inner is not None:
                calibration_logits, calibration_y, calibration_source = calibration_data_from_model(
                    adapter, local_X[effective_train], local_y[effective_train]
                )
                calibration = fit_weighted_logit_temperature(
                    calibration_logits,
                    calibration_y,
                    pos_weight=float(adapter.training_pos_weight_),
                    train_prior=float(adapter.training_prior_),
                    source=calibration_source,
                )
                suffix_llr = calibration.to_llr(suffix_logits)
                calibration_record = {
                    "mode": "weighted_ce_llr_pre_refit_selection_scale",
                    "source": calibration.source,
                    "n_samples": calibration.n_samples,
                    "temperature": calibration.temperature,
                    "offset": calibration.offset,
                    "order_preserving": True,
                    "probability_calibration_valid_for_refit_model": False,
                }
            else:
                suffix_llr, calibration_record = checkpoint_scores_to_llr(
                    {
                        "source_calibration": {
                            "pos_weight": adapter.training_pos_weight_,
                            "train_prior": adapter.training_prior_,
                            "temperature": 1.0,
                            "source": "target_fixed_budget_weighted_ce_analytic",
                        }
                    },
                    suffix_logits,
                )
                calibration_record["source"] = "target_fixed_budget_weighted_ce_analytic"
            total_parameters = adapter.total_parameter_count()
            trainable_parameters = adapter.parameter_count()
            refit_record = {
                "enabled": bool(adapter.last_history.get("refit_full_prefix")),
                "selected_epochs": adapter.last_history.get("best_epoch"),
                "refit_epochs": adapter.last_history.get("refit_epochs"),
                "inner_embargo_trials": int(
                    np.count_nonzero(inner.embargo_mask)
                    if inner is not None and inner.embargo_mask is not None
                    else 0
                ),
                "epoch_selection": args.epoch_selection,
            }
        auc = None
        if len(np.unique(y_post)) == 2:
            auc = float(roc_auc_score(y_post, suffix_logits))
        meta = dataset.metadata.iloc[post_rows]
        evaluated_outcomes = decision_outcomes_at_repetition(
            suffix_llr,
            meta["candidate_id"].to_numpy(dtype=np.int64),
            meta["target_row"].to_numpy(dtype=np.int64),
            meta["target_col"].to_numpy(dtype=np.int64),
            meta["selection_id"].astype(str).to_numpy(),
            split.test_repetition_indices[post_rows],
            contract=candidate_task,
            subject_ids=np.repeat(str(subject), len(post_rows)),
            onset_times_s=onset_by_epoch[post_rows],
            evidence_available_times_s=available_by_epoch[post_rows],
            max_repetitions=args.test_reps,
        )
        requested_test = split.requested_test_selections_by_subject[str(subject)]
        eligible_test = split.test_selections_by_subject[str(subject)]
        failed_test = split.failed_test_selections_by_subject[str(subject)]
        failed_outcomes = _failed_decision_outcomes(
            subject=str(subject),
            requested_decisions=requested_test,
            failure_reasons=failed_test,
            test_repetitions=args.test_reps,
            target_by_decision=target_by_subject[str(subject)],
            candidate_task=candidate_task,
        )
        indexed_outcomes = {
            (outcome.key.decision_id, outcome.evidence_level): outcome
            for outcome in (*evaluated_outcomes, *failed_outcomes)
        }
        subject_outcomes = tuple(
            indexed_outcomes[(decision, repetition)]
            for decision in requested_test
            for repetition in range(1, args.test_reps + 1)
        )
        outcomes_by_subject[str(subject)] = subject_outcomes
        decision_accounting = _accounting_record(
            subject_outcomes,
            subject=str(subject),
            requested_decisions=requested_test,
            eligible_decisions=eligible_test,
            test_repetitions=args.test_reps,
        )
        evaluated_accounting = build_decision_outcome_accounting(
            evaluated_outcomes,
            requested_decisions=[DecisionKey(str(subject), decision) for decision in eligible_test],
            evidence_levels=range(1, args.test_reps + 1),
        )
        eligible_hit_rate_by_evidence = {
            counts.evidence_level: (
                counts.correct / len(eligible_test) if eligible_test else float("nan")
            )
            for counts in evaluated_accounting.counts_by_evidence
        }
        records.append(
            {
                "subject": str(subject),
                "participant_key": participant_by_subject[str(subject)],
                "n_prefix": int(len(pre_rows)),
                "n_prefix_after_qc": int(effective_train.sum()),
                "n_suffix": int(len(post_rows)),
                "binary_auc": auc,
                "eligible_hit_rate_by_evidence": eligible_hit_rate_by_evidence,
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "head": head_mode,
                "calibration": calibration_record,
                "prefix_refit": refit_record,
                "artifact_quality": artifact_audit,
                "calibration_selections": split.calibration_selections_by_subject[str(subject)],
                "requested_test_selections": requested_test,
                "test_selections": eligible_test,
                "failed_test_selections": failed_test,
                "decision_accounting": decision_accounting,
                "decision_outcomes": [
                    {
                        **outcome.to_record(),
                        "participant_key": participant_by_subject[str(subject)],
                    }
                    for outcome in subject_outcomes
                ],
                "suffix_predictions": {
                    "epoch_rows": post_rows.astype(int).tolist(),
                    "raw_logits": suffix_logits.tolist(),
                    "llr_scores": suffix_llr.tolist(),
                    "labels": y_post.astype(int).tolist(),
                    "candidate_ids": meta["candidate_id"].to_numpy(dtype=np.int64).tolist(),
                    "selection_ids": meta["selection_id"].astype(str).tolist(),
                    "repetition_indices": (
                        split.test_repetition_indices[post_rows].astype(int).tolist()
                    ),
                    "onset_times_s": onset_by_epoch[post_rows].tolist(),
                    "evidence_available_times_s": available_by_epoch[post_rows].tolist(),
                },
            }
        )
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)

    subject_decision_ledger = []
    for subject in sorted(requested_subjects, key=lambda value: participant_by_subject[value]):
        selection_failed = subject in split.excluded_subjects
        requested = (
            () if selection_failed else split.requested_test_selections_by_subject.get(subject, ())
        )
        eligible = () if selection_failed else split.test_selections_by_subject.get(subject, ())
        failed = (
            {} if selection_failed else split.failed_test_selections_by_subject.get(subject, {})
        )
        if subject not in outcomes_by_subject:
            outcomes_by_subject[subject] = _failed_decision_outcomes(
                subject=subject,
                requested_decisions=requested,
                failure_reasons=failed,
                test_repetitions=args.test_reps,
                target_by_decision=target_by_subject[subject],
                candidate_task=candidate_task,
            )
        subject_outcomes = outcomes_by_subject[subject]
        accounting = _accounting_record(
            subject_outcomes,
            subject=subject,
            requested_decisions=requested,
            eligible_decisions=eligible,
            test_repetitions=args.test_reps,
        )
        subject_decision_ledger.append(
            {
                "subject": subject,
                "participant_key": participant_by_subject[subject],
                **accounting,
                "failed_test_selections": failed,
                "subject_exclusion_reason": split.excluded_subjects.get(subject),
            }
        )
    data_eligible_decisions = sum(item["data_eligible"] for item in subject_decision_ledger)
    data_ineligible_decisions = sum(item["data_ineligible"] for item in subject_decision_ledger)
    evaluation_successful_decisions = sum(
        item["evaluation_successful"] for item in subject_decision_ledger
    )
    evaluation_failed_decisions = sum(item["evaluation_failed"] for item in subject_decision_ledger)
    all_outcomes = tuple(
        outcome
        for subject in sorted(requested_subjects)
        for outcome in outcomes_by_subject[subject]
    )
    global_accounting = build_decision_outcome_accounting(
        all_outcomes,
        requested_decisions=[
            DecisionKey(subject, decision)
            for subject in sorted(requested_subjects)
            if subject not in split.excluded_subjects
            for decision in split.requested_test_selections_by_subject.get(subject, ())
        ],
        evidence_levels=range(1, args.test_reps + 1),
    )
    decision_accounting = global_accounting.to_record()
    decision_accounting["data_eligible"] = data_eligible_decisions
    decision_accounting["data_ineligible"] = data_ineligible_decisions
    decision_accounting["evaluation_successful"] = evaluation_successful_decisions
    decision_accounting["evaluation_failed"] = evaluation_failed_decisions
    participant_accounting = {
        "requested": len(decision_plan.requested_participant_keys),
        "decision_planned": len(decision_plan.participant_keys),
        "selection_failed": len(decision_plan.participant_selection_failures),
    }
    participant_operational_endpoints = []
    selection_failed_keys = {
        failure.participant_key for failure in decision_plan.participant_selection_failures
    }
    for subject in sorted(requested_subjects):
        participant_key = participant_by_subject[subject]
        planned_decisions = sum(
            entry.participant_key == participant_key for entry in decision_plan.entries
        )
        correct_at_r = sum(
            outcome.key.subject_id == subject
            and outcome.evidence_level == args.test_reps
            and outcome.status is DecisionStatus.CORRECT
            for outcome in all_outcomes
        )
        if participant_key in selection_failed_keys:
            if planned_decisions:
                raise AssertionError("selection-failed participant has planned decisions")
            endpoint = 0.0
        else:
            if planned_decisions < 1:
                raise AssertionError("planned participant has no decision denominator")
            endpoint = correct_at_r / planned_decisions
        participant_operational_endpoints.append(
            {
                "participant_key": participant_key,
                "evidence_level": args.test_reps,
                "planned_decisions": planned_decisions,
                "correct_decisions": correct_at_r,
                "selection_failed": participant_key in selection_failed_keys,
                "requested_participant_operational_hit_rate": endpoint,
            }
        )
    checkpoint_sha256 = _sha256_file(args.checkpoint) if args.checkpoint else None
    model_origin = (
        checkpoint_model_origin(
            checkpoint_payload,
            checkpoint_sha256=checkpoint_sha256,
        )
        if checkpoint_payload is not None and checkpoint_sha256 is not None
        else scratch_model_origin(
            {
                "model": "N2P3Net",
                "pooling_mode": "ms_flatten",
                "head": head_mode,
                "seed": args.seed,
                "n_channels": dataset.n_channels,
                "n_times": dataset.n_times,
            }
        )
    )
    evaluation_contract = build_evaluation_run_contract(
        arm_name=args.arm_name,
        model_origin=model_origin,
        dataset=dataset,
        target_cache_sha256=target_cache_sha256,
        source_snapshot_sha256=source_snapshot_sha256,
        requested_subjects=tuple(sorted(requested_subjects)),
        identity_policy=args.identity_exclusion_policy,
        target_protocol={
            "estimand": "known_early_decisions_to_later_unknown_decisions",
            "split_axis": "decision_time",
            "calibration_truth_access": "adapter_visible",
            "test_truth_access": "scorer_only",
            "calibration_selections": args.calibration_selections,
            "test_repetitions": args.test_reps,
            "max_test_selections": args.max_test_selections,
            "epoch_selection": args.epoch_selection,
        },
        adaptation={
            "procedure": {
                "head": head_mode,
                "epochs": epochs,
                "learning_rate": lr,
                "batch_size": args.batch_size,
                "adapt_batchnorm": bool(args.adapt_batchnorm),
                "normalization": normalization,
                "target_stat_weight": args.target_stat_weight,
                "fold_local_qc": bool(args.fold_local_qc),
                "identity_exclusion_policy": args.identity_exclusion_policy,
            },
            "replicate_parameters": {
                "training_replicate_key": args.training_replicate_key,
                "random_seed": args.seed,
            },
        },
        decision={
            "candidate_task_contract": candidate_task.record(),
            "candidate_task_contract_digest": semantic_sha256(candidate_task.record()),
            "aggregation": "cumulative_candidate_membership_llr",
            "tie_policy": "abstain",
            "test_repetitions": args.test_reps,
        },
        evidence_scope={
            "stage": candidate_task.evidence_scope["stage"],
            "population": dict(candidate_task.population),
            "task": candidate_task.task_id,
            "dataset_id": candidate_task.dataset_id,
            "product_confirmation": candidate_task.evidence_scope["product_confirmation"],
        },
    )
    data_eligible_keys = {
        (participant_by_subject[subject], decision)
        for subject in requested_subjects
        for decision in split.test_selections_by_subject.get(subject, ())
    }
    decision_failures_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for outcome in all_outcomes:
        if outcome.status not in {DecisionStatus.INCOMPLETE, DecisionStatus.FIT_FAILURE}:
            continue
        participant_key = participant_by_subject[outcome.key.subject_id]
        failure_key = (participant_key, outcome.key.decision_id)
        decision_failures_by_key.setdefault(
            failure_key,
            {
                "participant_key": participant_key,
                "decision_id": outcome.key.decision_id,
                "status": outcome.status.value,
                "stage": (
                    "subject_fit"
                    if outcome.status is DecisionStatus.FIT_FAILURE
                    else "decision_evaluation"
                    if failure_key in data_eligible_keys
                    else "decision_eligibility"
                ),
                "reason": outcome.failure_reason,
            },
        )
    summary = {
        "schema": "n2p3_candidate_cross_decision_result/1",
        "run_status": (
            "completed_with_selection_failures"
            if decision_plan.participant_selection_failures
            else "completed"
        ),
        "arm_name": args.arm_name,
        "training_replicate_key": args.training_replicate_key,
        "partition_key": args.partition_key,
        "source_snapshot_manifest": str(args.source_snapshot_manifest.resolve()),
        "source_snapshot_sha256": source_snapshot_sha256,
        "evaluation_contract": evaluation_contract.record(),
        "evaluation_contract_digest": evaluation_contract.digest(),
        "decision_plan": decision_plan.record(),
        "decision_plan_digest": decision_plan.digest(),
        "requested_participant_keys": list(evaluation_contract.requested_participant_keys),
        "dataset_cache": str(Path(args.dataset_cache).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "checkpoint_sha256": checkpoint_sha256,
        "target_cache_sha256": target_cache_sha256,
        "candidate_task_contract": candidate_task.record(),
        "candidate_task_contract_digest": semantic_sha256(candidate_task.record()),
        "calibration_selections": args.calibration_selections,
        "test_reps": args.test_reps,
        "head": head_mode,
        "epochs": epochs,
        "lr": lr,
        "epoch_selection": args.epoch_selection,
        "adapt_batchnorm": bool(args.adapt_batchnorm),
        "normalization": normalization,
        "target_stat_weight": args.target_stat_weight,
        "seed": args.seed,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
        "estimand": "known_early_decisions_to_later_unknown_decisions",
        "excluded_subjects": split.excluded_subjects,
        "decision_accounting": decision_accounting,
        "participant_accounting": participant_accounting,
        "participant_operational_endpoints": participant_operational_endpoints,
        "decision_failures": list(decision_failures_by_key.values()),
        "decision_outcomes": [
            {
                **outcome.to_record(),
                "participant_key": participant_by_subject[outcome.key.subject_id],
            }
            for outcome in all_outcomes
        ],
        "binary_auc_mean": (
            float(np.mean(auc_values))
            if (
                auc_values := [
                    rec["binary_auc"] for rec in records if rec["binary_auc"] is not None
                ]
            )
            else None
        ),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"[summary] {output}", flush=True)


if __name__ == "__main__":
    main()
