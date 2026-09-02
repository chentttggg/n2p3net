"""Evaluate early BrainSync sessions against policy-constrained later sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.brainsync_contract import (  # noqa: E402
    BRAIN_SYNC_SESSION_SCHEMA,
    DEFAULT_ADULT_MIN_AGE_YEARS,
    DecisionTargetPolicy,
    PopulationScopePolicy,
    derive_brainsync_evidence_scope,
    derive_population_scope,
)
from data.contract import assert_causal_p300_input_contract  # noqa: E402
from data.epochs import load_epoch_dataset, loaded_epoch_cache_attestation  # noqa: E402
from research.evaluation import (  # noqa: E402
    build_evaluation_run_contract,
    checkpoint_model_origin,
    source_snapshot_sha256_from_archive_manifest,
)
from research.execution import ExpectedSubjectError  # noqa: E402
from train.device import get_device  # noqa: E402
from transfer.checkpoint import (  # noqa: E402
    checkpoint_classifier_is_trained,
    checkpoint_input_stats,
    checkpoint_scores_to_llr,
    checkpoint_training_contract,
    load_checkpoint_payload,
    load_n2p3_trunk_checkpoint,
    predict_n2p3_checkpoint,
)
from transfer.cohort import read_subject_manifest, resolve_subject_scope  # noqa: E402
from transfer.evaluation import candidate_decision_outcomes  # noqa: E402
from transfer.outcomes import (  # noqa: E402
    CandidateCoverage,
    DecisionKey,
    DecisionOutcome,
    DecisionStatus,
    build_decision_outcome_accounting,
)
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig  # noqa: E402
from transfer.within_subject import calibration_decision_split  # noqa: E402

RESULT_SCHEMA = "n2p3_brainsync_cross_decision_result/2"


def _failed_outcomes(
    *,
    subject: str,
    requested: tuple[str, ...],
    failures: dict[str, str],
    test_repetitions: int,
) -> tuple[DecisionOutcome, ...]:
    outcomes: list[DecisionOutcome] = []
    vocabulary = tuple(str(value) for value in range(1, 10))
    for decision in requested:
        reason = failures.get(decision)
        if reason is None:
            continue
        for level in range(1, test_repetitions + 1):
            outcomes.append(
                DecisionOutcome(
                    key=DecisionKey(subject, decision),
                    evidence_level=level,
                    status=DecisionStatus.INCOMPLETE,
                    coverage=CandidateCoverage.from_mappings(
                        {candidate: level for candidate in vocabulary}, {}
                    ),
                    failure_reason=reason,
                )
            )
    return tuple(outcomes)


def _fit_failure_outcomes(
    *,
    subject: str,
    decisions: tuple[str, ...],
    test_repetitions: int,
    failure: ExpectedSubjectError,
) -> tuple[DecisionOutcome, ...]:
    vocabulary = tuple(str(value) for value in range(1, 10))
    return tuple(
        DecisionOutcome(
            key=DecisionKey(subject, decision),
            evidence_level=level,
            status=DecisionStatus.FIT_FAILURE,
            coverage=CandidateCoverage.from_mappings(
                {candidate: level for candidate in vocabulary}, {}
            ),
            failure_reason=f"{failure.stage}:{failure.code.value}",
        )
        for decision in decisions
        for level in range(1, test_repetitions + 1)
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_population_rows(dataset) -> tuple[tuple[str, ...], tuple[float | None, ...]]:
    required = {"session", "subject", "age_years", "age_source"}
    missing = sorted(required.difference(dataset.metadata.columns))
    if missing:
        raise ValueError(
            "BrainSync cache lacks population evidence columns: " + ", ".join(missing)
        )
    subjects: list[str] = []
    ages: list[float | None] = []
    for session_id, rows in dataset.metadata.groupby("session", sort=False):
        subject_values = {str(value).strip() for value in rows["subject"]}
        if len(subject_values) != 1 or "" in subject_values:
            raise ValueError(
                f"BrainSync session {session_id!r} has inconsistent subject metadata."
            )
        sources = {str(value) for value in rows["age_source"]}
        if sources != {"session.experiment.age"}:
            raise ValueError(
                f"BrainSync session {session_id!r} has unsupported age provenance."
            )
        nonmissing_ages = {
            float(value)
            for value in rows["age_years"]
            if value is not None and not (isinstance(value, float) and np.isnan(value))
        }
        if len(nonmissing_ages) > 1:
            raise ValueError(
                f"BrainSync session {session_id!r} has inconsistent age metadata."
            )
        subjects.append(next(iter(subject_values)))
        ages.append(next(iter(nonmissing_ages)) if nonmissing_ages else None)
    return tuple(subjects), tuple(ages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--source-snapshot-manifest", type=Path, required=True)
    parser.add_argument(
        "--identity-exclusion-policy",
        required=True,
        choices=("source", "source_or_global", "global"),
    )
    parser.add_argument("--calibration-selections", type=int, default=5)
    parser.add_argument("--test-reps", type=int, default=5)
    parser.add_argument("--max-test-selections", type=int, default=None)
    parser.add_argument(
        "--target-policy",
        required=True,
        choices=tuple(policy.value for policy in DecisionTargetPolicy),
        help="Predeclared target-sequence estimand; failed requested sessions are not replaced.",
    )
    parser.add_argument(
        "--population-policy",
        choices=tuple(policy.value for policy in PopulationScopePolicy),
        default=PopulationScopePolicy.DESCRIPTIVE.value,
        help="Use adult_only only when every included session has qualifying age evidence.",
    )
    parser.add_argument(
        "--adult-min-age-years",
        type=float,
        default=DEFAULT_ADULT_MIN_AGE_YEARS,
    )
    parser.add_argument(
        "--head",
        choices=("zero_shot", "classifier_fine", "linear", "mlp16", "full_fine", "adapter"),
        default="zero_shot",
    )
    parser.add_argument(
        "--normalization",
        choices=("source", "target_prefix", "shrinkage"),
        default="source",
    )
    parser.add_argument("--target-stat-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-subjects-file", type=Path, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.target_subjects_file is not None and args.max_subjects is not None:
        parser.error("--target-subjects-file cannot be combined with --max-subjects")
    if args.head == "zero_shot" and args.normalization != "source":
        parser.error("zero_shot requires source normalization")

    source_snapshot_sha256 = source_snapshot_sha256_from_archive_manifest(
        args.source_snapshot_manifest
    )
    device = torch.device(args.device) if args.device != "auto" else get_device()
    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    assert_causal_p300_input_contract(dataset.preprocessing)
    if dataset.provenance.get("session_schema") != BRAIN_SYNC_SESSION_SCHEMA:
        raise ValueError("BrainSync cache does not declare the exact v2 session schema.")
    if dataset.provenance.get("decision_unit") != "session":
        raise ValueError("BrainSync cache must declare session as its decision unit.")
    population_subjects, population_ages = _session_population_rows(dataset)
    population_scope = derive_population_scope(
        population_subjects,
        population_ages,
        policy=args.population_policy,
        adult_min_age_years=args.adult_min_age_years,
    )
    evidence_scope = derive_brainsync_evidence_scope(
        population_scope,
        target_policy=args.target_policy,
    )
    target_cache_sha = str(loaded_epoch_cache_attestation(dataset)["sha256"])
    split = calibration_decision_split(
        dataset,
        calibration_selections=args.calibration_selections,
        test_repetitions=args.test_reps,
        target_policy=args.target_policy,
        max_test_selections=args.max_test_selections,
        candidate_vocabulary=range(1, 10),
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
    if dataset.identity_table is None:
        raise ValueError("BrainSync cache lacks the current participant identity table.")
    participant_by_subject = {
        subject: dataset.identity_table.record_for(subject).authority_key(
            args.identity_exclusion_policy
        )
        for subject in requested_order
    }
    payload = load_checkpoint_payload(args.checkpoint)
    training_contract = checkpoint_training_contract(payload)
    if training_contract.source_snapshot_sha256 != source_snapshot_sha256:
        raise ValueError(
            "checkpoint training source snapshot disagrees with the verified "
            "source snapshot manifest."
        )
    # Validate the complete checkpoint and target-exclusion contract for the
    # frozen requested cohort, including subjects with no eligible decision.
    for subject in requested_order:
        load_n2p3_trunk_checkpoint(
            payload,
            dataset,
            target_subject=subject,
            identity_exclusion_policy=args.identity_exclusion_policy,
        )
    checkpoint_stats = checkpoint_input_stats(payload, dataset.n_channels)
    if args.head in {"zero_shot", "classifier_fine", "full_fine"} and not (
        checkpoint_classifier_is_trained(payload)
    ):
        parser.error(f"--head {args.head} requires a supervised source classifier")
    epochs = (
        0
        if args.head == "zero_shot"
        else args.epochs
        if args.epochs is not None
        else 5
        if args.head in {"classifier_fine", "full_fine"}
        else 30
    )
    lr = (
        0.0
        if args.head == "zero_shot"
        else args.lr
        if args.lr is not None
        else 1e-4
        if args.head in {"classifier_fine", "full_fine"}
        else 1e-3
    )
    trial_mask = (
        np.asarray(dataset.trial_channel_mask, dtype=bool)
        if dataset.trial_channel_mask is not None
        else np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2]).copy()
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
    for subject in usable_order:
        subject_rows = np.flatnonzero(np.asarray(dataset.subject_ids).astype(str) == subject)
        calibration_rows = subject_rows[split.calibration_mask[subject_rows]]
        test_rows = subject_rows[split.test_mask[subject_rows]]
        if not len(calibration_rows) or not len(test_rows):
            continue
        trunk, _ = load_n2p3_trunk_checkpoint(
            payload,
            dataset,
            target_subject=subject,
            identity_exclusion_policy=args.identity_exclusion_policy,
        )
        suffix_X = np.asarray(dataset.X[test_rows], dtype=np.float32)
        suffix_mask = trial_mask[test_rows]
        if args.head == "zero_shot":
            raw_logits = predict_n2p3_checkpoint(
                trunk,
                suffix_X,
                input_stats=checkpoint_stats,
                device=device,
                trial_channel_mask=suffix_mask,
            )
            llr, calibration = checkpoint_scores_to_llr(payload, raw_logits)
            trainable_parameters = 0
        else:
            adapter = SubjectAdapter(
                trunk,
                config=SubjectAdapterConfig(
                    head_kind=args.head,
                    epochs=int(epochs),
                    batch_size=args.batch_size,
                    lr=float(lr),
                    seed=args.seed,
                    val_group_fraction=None,
                    refit_full_prefix=False,
                    freeze_batchnorm_stats=True,
                    input_statistics=args.normalization,
                    target_stat_weight=args.target_stat_weight,
                    input_mean=checkpoint_stats[0],
                    input_std=checkpoint_stats[1],
                ),
                device=device,
            )
            try:
                adapter.fit(
                    np.asarray(dataset.X[calibration_rows], dtype=np.float32),
                    np.asarray(dataset.y[calibration_rows], dtype=np.int64),
                    trial_channel_mask=trial_mask[calibration_rows],
                )
                raw_logits = adapter.predict_logit(
                    suffix_X, trial_channel_mask=suffix_mask
                )
            except ExpectedSubjectError as failure:
                requested = split.requested_test_groups_by_subject[subject]
                eligible = split.test_groups_by_subject[subject]
                failed = split.failed_test_groups_by_subject[subject]
                subject_outcomes = (
                    *_fit_failure_outcomes(
                        subject=subject,
                        decisions=eligible,
                        test_repetitions=args.test_reps,
                        failure=failure,
                    ),
                    *_failed_outcomes(
                        subject=subject,
                        requested=requested,
                        failures=failed,
                        test_repetitions=args.test_reps,
                    ),
                )
                outcomes_by_subject[subject] = tuple(subject_outcomes)
                records.append(
                    {
                        "subject": subject,
                        "participant_key": participant_by_subject[subject],
                        "binary_auc": None,
                        "fit_failure": failure.record(subject=subject),
                        "decision_outcomes": [
                            {
                                **outcome.to_record(),
                                "participant_key": participant_by_subject[subject],
                            }
                            for outcome in subject_outcomes
                        ],
                    }
                )
                continue
            llr, calibration = checkpoint_scores_to_llr(
                {
                    "source_calibration": {
                        "pos_weight": adapter.training_pos_weight_,
                        "train_prior": adapter.training_prior_,
                        "temperature": 1.0,
                        "source": "target_fixed_budget_weighted_ce_analytic",
                    }
                },
                raw_logits,
            )
            trainable_parameters = adapter.parameter_count()

        groups = split.group_ids[test_rows]
        truth = {group: split.truth_by_group[group] for group in np.unique(groups)}
        auc = float(roc_auc_score(dataset.y[test_rows], raw_logits))
        requested = split.requested_test_groups_by_subject[subject]
        eligible = split.test_groups_by_subject[subject]
        failed = split.failed_test_groups_by_subject[subject]
        evaluated_outcomes = candidate_decision_outcomes(
            llr,
            split.candidate_codes[test_rows],
            groups,
            truth,
            split.test_repetition_indices[test_rows],
            {group: subject for group in np.unique(groups)},
            aggregation="mean",
            max_repetitions=args.test_reps,
            candidate_vocabulary=split.candidate_vocab,
            onset_times_s=onset_by_epoch[test_rows],
            evidence_available_times_s=available_by_epoch[test_rows],
        )
        subject_outcomes = (*evaluated_outcomes, *_failed_outcomes(
            subject=subject,
            requested=requested,
            failures=failed,
            test_repetitions=args.test_reps,
        ))
        outcomes_by_subject[subject] = tuple(subject_outcomes)
        subject_accounting = build_decision_outcome_accounting(
            subject_outcomes,
            requested_decisions=[DecisionKey(subject, decision) for decision in requested],
            evidence_levels=range(1, args.test_reps + 1),
        ).to_record()
        records.append(
            {
                "subject": subject,
                "calibration_groups": split.calibration_groups_by_subject[subject],
                "requested_test_groups": requested,
                "test_groups": eligible,
                "failed_test_groups": failed,
                "binary_auc": auc,
                "trainable_parameters": trainable_parameters,
                "calibration": calibration,
                "decision_accounting": subject_accounting,
                "decision_outcomes": [
                    {
                        **outcome.to_record(),
                        "participant_key": participant_by_subject[subject],
                    }
                    for outcome in subject_outcomes
                ],
                "suffix_predictions": {
                    "epoch_rows": test_rows.tolist(),
                    "raw_logits": raw_logits.tolist(),
                    "llr_scores": llr.tolist(),
                    "labels": np.asarray(dataset.y[test_rows], dtype=int).tolist(),
                    "candidate_codes": split.candidate_codes[test_rows].tolist(),
                    "group_ids": groups.tolist(),
                    "repetition_indices": split.test_repetition_indices[test_rows].tolist(),
                },
            }
        )

    ledger = []
    for subject in requested_order:
        requested = split.requested_test_groups_by_subject.get(subject, ())
        eligible = split.test_groups_by_subject.get(subject, ())
        if subject not in outcomes_by_subject:
            outcomes_by_subject[subject] = _failed_outcomes(
                subject=subject,
                requested=requested,
                failures=split.failed_test_groups_by_subject.get(subject, {}),
                test_repetitions=args.test_reps,
            )
        subject_outcomes = outcomes_by_subject[subject]
        subject_accounting = build_decision_outcome_accounting(
            subject_outcomes,
            requested_decisions=[DecisionKey(subject, decision) for decision in requested],
            evidence_levels=range(1, args.test_reps + 1),
        ).to_record()
        ledger.append(
            {
                "subject": subject,
                "participant_key": participant_by_subject[subject],
                "requested": len(requested),
                "eligible": len(eligible),
                "failed": len(requested) - len(eligible),
                "by_evidence_level": subject_accounting["by_evidence_level"],
                "subject_exclusion_reason": split.excluded_subjects.get(subject),
                "failed_test_groups": split.failed_test_groups_by_subject.get(subject, {}),
            }
        )
    requested_decisions = sum(record["requested"] for record in ledger)
    eligible_decisions = sum(record["eligible"] for record in ledger)
    all_outcomes = tuple(
        outcome for subject in requested_order for outcome in outcomes_by_subject[subject]
    )
    global_accounting = build_decision_outcome_accounting(
        all_outcomes,
        requested_decisions=[
            DecisionKey(subject, decision)
            for subject in requested_order
            for decision in split.requested_test_groups_by_subject.get(subject, ())
        ],
        evidence_levels=range(1, args.test_reps + 1),
    )
    checkpoint_sha256 = _sha256(args.checkpoint)
    evaluation_contract = build_evaluation_run_contract(
        arm_name=args.arm_name,
        model_origin=checkpoint_model_origin(
            payload, checkpoint_sha256=checkpoint_sha256
        ),
        dataset=dataset,
        target_cache_sha256=target_cache_sha,
        source_snapshot_sha256=source_snapshot_sha256,
        requested_subjects=requested_order,
        identity_policy=args.identity_exclusion_policy,
        target_protocol={
            "decision_unit": "session",
            "calibration_selections": args.calibration_selections,
            "test_repetitions": args.test_reps,
            "max_test_selections": args.max_test_selections,
            "target_policy": split.target_policy,
            "population_policy": args.population_policy,
            "adult_min_age_years": args.adult_min_age_years,
        },
        adaptation={
            "head": args.head,
            "normalization": args.normalization,
            "target_stat_weight": args.target_stat_weight,
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": args.batch_size,
            "identity_exclusion_policy": args.identity_exclusion_policy,
        },
        decision={
            "task": "BrainSync_9_choice",
            "aggregation": "mean",
            "tie_policy": "abstain",
            "test_repetitions": args.test_reps,
        },
        evidence_scope=evidence_scope.to_dict(),
    )
    decision_accounting = global_accounting.to_record()
    decision_accounting["eligible"] = eligible_decisions
    decision_accounting["pre_evaluation_failed"] = requested_decisions - eligible_decisions
    summary = {
        "schema": RESULT_SCHEMA,
        "run_status": "completed",
        "arm_name": args.arm_name,
        "evaluation_contract": evaluation_contract.record(),
        "evaluation_contract_digest": evaluation_contract.digest(),
        "requested_participant_keys": list(
            evaluation_contract.requested_participant_keys
        ),
        "dataset_cache": str(Path(args.dataset_cache).resolve()),
        "target_cache_sha256": target_cache_sha,
        "source_snapshot_manifest": str(args.source_snapshot_manifest.resolve()),
        "source_snapshot_sha256": source_snapshot_sha256,
        "input_preprocessing": asdict(dataset.preprocessing),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "head": args.head,
        "normalization": args.normalization,
        "epochs": epochs,
        "lr": lr,
        "seed": args.seed,
        "calibration_selections": args.calibration_selections,
        "test_repetitions": args.test_reps,
        "target_policy": split.target_policy,
        "population_scope": population_scope.to_dict(),
        "requested_subjects": list(requested_order),
        "usable_subjects": len(records),
        "excluded_subjects": split.excluded_subjects,
        "decision_accounting": decision_accounting,
        "operational_subject_hit_by_repetition": {
            str(repetition): float(
                np.mean(
                    [
                        record["by_evidence_level"][str(repetition)][
                            "operational_hit_rate"
                        ]
                        for record in ledger
                    ]
                )
            )
            for repetition in range(1, args.test_reps + 1)
        },
        "operational_decision_hit_by_repetition": {
            str(counts.evidence_level): (
                counts.correct / requested_decisions if requested_decisions else 0.0
            )
            for counts in global_accounting.counts_by_evidence
        },
        "subject_decision_ledger": ledger,
        "decision_outcomes": [
            {
                **outcome.to_record(),
                "participant_key": participant_by_subject[outcome.key.subject_id],
            }
            for outcome in all_outcomes
        ],
        "records": records,
        "evidence_scope": evidence_scope.to_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {output}", flush=True)


if __name__ == "__main__":
    main()
