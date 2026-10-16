"""Compare source pretraining and target-only learning on a fixed binary suffix."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data.contract import assert_causal_p300_input_contract  # noqa: E402
from data.epochs import load_epoch_dataset, loaded_epoch_cache_attestation  # noqa: E402
from models.n2p3net import N2P3Net  # noqa: E402
from research.evaluation import (  # noqa: E402
    build_evaluation_run_contract,
    checkpoint_model_origin,
    scratch_model_origin,
    verified_source_snapshot,
)
from research.execution import ExpectedSubjectError, atomic_write_json  # noqa: E402
from transfer.binary import (  # noqa: E402
    STANDARD_CALIBRATION_TRIALS,
    binary_time_split,
    calibration_arms,
    matched_split_digests,
    paired_transfer_effects,
    resolve_calibration_splits,
    scratch_like_trunk,
    shrinkage_feature_scores,
    target_shrinkage_scores,
    validate_calibration_budgets,
    validate_matched_transfer_controls,
)
from transfer.checkpoint import (  # noqa: E402
    checkpoint_classifier_is_trained,
    checkpoint_input_stats,
    checkpoint_training_contract,
    load_checkpoint_payload,
    load_n2p3_trunk_checkpoint,
    predict_n2p3_checkpoint,
    predict_n2p3_checkpoint_with_features,
)
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig  # noqa: E402

ARMS = (
    "zero_shot",
    "classifier_fine",
    "full_fine",
    "target_only",
    "target_only_tuned",
    "target_shrinkage_lda",
    "feature_shrinkage_lda",
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-snapshot-manifest", type=Path)
    parser.add_argument("--evaluation-snapshot-manifest", type=Path)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--target-subjects", required=True)
    parser.add_argument("--calibration-trials", default=",".join(map(str, STANDARD_CALIBRATION_TRIALS)))
    parser.add_argument("--minimum-test-trials", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scratch-epochs", type=int, default=60)
    parser.add_argument("--scratch-lr", type=float, default=1e-3)
    parser.add_argument(
        "--adaptation-normalization",
        choices=("source", "target_prefix", "shrinkage"),
        default="source",
    )
    parser.add_argument("--adapt-batchnorm", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-plan", type=Path)
    parser.add_argument("--stage", choices=("B0", "B0_score", "E1", "E3", "compact", "ES", "P1", "P2", "unified"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-rule", type=Path)
    parser.add_argument("--batch-cohort", type=Path)
    args = parser.parse_args(argv)
    if args.stage:
        if args.training_plan is None:
            parser.error("training-chain stages require --training-plan with whole-study source exclusion")
        command = [sys.executable, str(ROOT / "experiments/run_cross_dataset_transfer.py"),
                   "--plan", str(args.training_plan.resolve()), "--subjects", args.target_subjects,
                   "--output", str(args.output.resolve()), "--stage", args.stage,
                   "--device", args.device, "--cpu-threads", str(args.cpu_threads),
                   "--target-seed", str(args.seed)]
        if args.dry_run:
            command.append("--dry-run")
        if args.batch_rule:
            command.extend(["--batch-rule", str(args.batch_rule.resolve())])
        if args.batch_cohort:
            command.extend(["--batch-cohort", str(args.batch_cohort.resolve())])
        subprocess.run(command, check=True)
        return
    if args.training_plan or args.dry_run or args.batch_rule or args.batch_cohort:
        parser.error("training-plan, batch-rule and dry-run require --stage")
    if args.dataset_cache is None or args.checkpoint is None or args.source_snapshot_manifest is None:
        parser.error("dataset-cache, checkpoint and source-snapshot-manifest are required")
    arms = tuple(args.arms.split(","))
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        parser.error("arms must be unique supported baseline names")
    subjects = tuple(value.strip() for value in args.target_subjects.split(",") if value.strip())
    try:
        budgets = tuple(int(value) for value in args.calibration_trials.split(","))
        validate_calibration_budgets(budgets)
        validate_matched_transfer_controls(arms, budgets)
    except ValueError as error:
        parser.error(str(error))
    if not subjects or len(set(subjects)) != len(subjects) or args.minimum_test_trials < 1:
        parser.error("unique target subjects and positive minimum test size are required")
    if args.cpu_threads < 1 or args.output.exists():
        parser.error("positive cpu threads and a new output directory are required")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    snapshot = verified_source_snapshot(args.source_snapshot_manifest)
    evaluation_snapshot = verified_source_snapshot(
        args.evaluation_snapshot_manifest or args.source_snapshot_manifest
    )
    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    assert_causal_p300_input_contract(dataset.preprocessing)
    if dataset.provenance.get("event_contract") != "binary_evidence":
        raise ValueError("an explicit binary_evidence source view is required.")
    cache_sha = str(loaded_epoch_cache_attestation(dataset)["sha256"])
    payload = load_checkpoint_payload(args.checkpoint)
    if not checkpoint_classifier_is_trained(payload):
        raise ValueError("a supervised source classifier is required.")
    if checkpoint_training_contract(payload).source_snapshot_sha256 != snapshot["archive_sha256"]:
        raise ValueError("source checkpoint and executed snapshot disagree.")
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    assert dataset.identity_table is not None
    source_stats = checkpoint_input_stats(payload, dataset.n_channels)
    timeline = dataset.event_timeline
    events = np.flatnonzero(timeline.evidence_indices >= 0)
    row_events = np.empty(dataset.n_epochs, dtype=np.int64)
    row_events[timeline.evidence_indices[events]] = events
    subject_values = np.asarray(dataset.subject_ids).astype(str)
    scope = {
        "stage": "development",
        "product_confirmation": False,
        "task": "binary_target_nontarget",
        "candidate_performance_claim": False,
    }
    protocol = {
        "kind": "single_recording_chronological_binary_calibration",
        "calibration_trials": list(budgets),
        "common_test_after_trials": max(budgets),
        "purge": "test_epoch_start_at_or_after_maximum_prefix_evidence_available",
        "selection_metadata_used": False,
    }
    # Validate every requested identity and split before starting any fit.
    splits = {}
    reference_trunks = {}
    unavailable_budgets = []
    for subject in subjects:
        trunk, _ = load_n2p3_trunk_checkpoint(payload, dataset, target_subject=subject)
        if type(trunk) is not N2P3Net:
            raise ValueError("this baseline comparison requires an N2P3Net source core.")
        if trunk.splitbn_config is not None and set(arms) & {"target_only", "target_only_tuned"}:
            raise ValueError(
                "SplitBN source arms must use a separately matched plain target-only baseline."
            )
        reference_trunks[subject] = trunk
        rows = np.flatnonzero(subject_values == subject)
        ev = row_events[rows]
        streams = set(
            zip(
                timeline.dataset_ids[ev],
                timeline.session_ids[ev],
                timeline.run_ids[ev],
                strict=True,
            )
        )
        if len(streams) != 1:
            raise ValueError(
                f"{subject}: explicitly select one recording stream before this protocol."
            )
        resolved, unavailable = resolve_calibration_splits(
            budgets,
            lambda budget, maximum, ev=ev: binary_time_split(
                    timeline.onset_times_s[ev],
                    timeline.evidence_available_times_s[ev],
                    calibration_trials=budget,
                    maximum_calibration_trials=maximum,
                    tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
                ),
            minimum_test_trials=args.minimum_test_trials,
            available_trials=len(rows),
        )
        splits[subject] = (rows, resolved)
        unavailable_budgets.extend({"participant_key": dataset.identity_table.record_for(subject).authority_key("source_or_global"),
                                    "local_subject_id": subject, "requested_calibration_trials": budget,
                                    "effective_calibration_trials": max(resolved),
                                    "status": "budget_reduced" if max(resolved) else "unavailable", "reason": reason}
                                   for budget, reason in unavailable.items())
    args.output.mkdir(parents=True)
    report = {
        "schema": "n2p3_binary_subject_transfer/2",
        "status": "running",
        "source_snapshot": snapshot,
        "evaluation_snapshot": evaluation_snapshot,
        "target_cache_sha256": cache_sha,
        "checkpoint_sha256": checkpoint_sha,
        "target_protocol": protocol,
        "evidence_scope": scope,
        "seed": args.seed,
        "budget_adjustments": unavailable_budgets,
        "cells": [],
    }
    result_path = args.output / "record.json"
    atomic_write_json(result_path, report)
    for subject, (rows, subject_splits) in splits.items():
        participant = dataset.identity_table.record_for(subject).authority_key("source_or_global")
        for budget, split in subject_splits.items():
            train_rows, test_rows = rows[split.calibration_rows], rows[split.test_rows]
            y_train, y_test = dataset.y[train_rows], dataset.y[test_rows]
            split_digests = matched_split_digests(train_rows, test_rows)
            for arm in calibration_arms(arms, budget, budgets):
                started = time.perf_counter()
                trunk = copy.deepcopy(reference_trunks[subject])
                target_only = arm.startswith("target_")
                epochs = args.scratch_epochs if arm == "target_only_tuned" else args.epochs
                lr = args.scratch_lr if arm == "target_only_tuned" else args.lr
                freeze_bn = not (
                    arm == "target_only_tuned" or (arm == "full_fine" and args.adapt_batchnorm)
                )
                if arm == "target_shrinkage_lda":
                    origin = scratch_model_origin(
                        {
                            "algorithm": "shrinkage_lda",
                            "features": "20Hz_fft_flatten",
                            "scaler_scope": "calibration_only",
                        }
                    )
                elif target_only:
                    trunk = scratch_like_trunk(trunk, seed=args.seed)
                    origin = scratch_model_origin(
                        {"architecture": trunk.architecture_record(), "seed": args.seed}
                    )
                else:
                    origin = checkpoint_model_origin(payload, checkpoint_sha256=checkpoint_sha)
                normalization = (
                    "target_prefix"
                    if target_only
                    else (
                        args.adaptation_normalization
                        if arm in {"classifier_fine", "full_fine"}
                        else "source"
                    )
                )
                adaptation = {
                    "head": arm,
                    "normalization": normalization,
                    "epochs": 0
                    if arm in {"zero_shot", "target_shrinkage_lda", "feature_shrinkage_lda"}
                    else epochs,
                    "lr": lr,
                    "batch_size": args.batch_size,
                    "calibration_trials": budget,
                    "epoch_selection": "fixed_budget",
                    "seed": args.seed,
                    "freeze_batchnorm_stats": freeze_bn,
                    "source_statistics_used": normalization in {"source", "shrinkage"},
                }
                contract = build_evaluation_run_contract(
                    arm_name=arm,
                    model_origin=origin,
                    dataset=dataset,
                    target_cache_sha256=cache_sha,
                    source_snapshot_sha256=evaluation_snapshot["archive_sha256"],
                    requested_subjects=[subject],
                    identity_policy="source_or_global",
                    target_protocol={**protocol, "maximum_calibration_trials": max(subject_splits)},
                    adaptation=adaptation,
                    decision={"kind": "none_binary_trial_evaluation"},
                    evidence_scope=scope,
                )
                cell = {
                    **split_digests,
                    "participant_key": participant,
                    "local_subject_id": subject,
                    "arm": arm,
                    "calibration_trials": budget,
                    "calibration_positives": int(y_train.sum()),
                    "test_trials": len(test_rows),
                    "purged_trials": len(split.purged_rows),
                    "evaluation_contract": contract.record(),
                    "evaluation_contract_digest": contract.digest(),
                }
                try:
                    if (arm != "zero_shot" and len(np.unique(y_train)) != 2) or len(np.unique(y_test)) != 2:
                        cell.update(status="insufficient_classes", auc=None)
                    else:
                        if arm == "target_shrinkage_lda":
                            if dataset.trial_channel_mask is not None and not bool(
                                dataset.trial_channel_mask[np.r_[train_rows, test_rows]].all()
                            ):
                                raise ValueError(
                                    "the shrinkage baseline requires fully observed channels"
                                )
                            prediction = target_shrinkage_scores(
                                dataset.X[train_rows],
                                y_train,
                                dataset.X[test_rows],
                                sfreq=dataset.preprocessing.sfreq,
                            )
                        elif arm == "feature_shrinkage_lda":
                            combined = np.r_[train_rows, test_rows]
                            _, features = predict_n2p3_checkpoint_with_features(
                                trunk,
                                dataset.X[combined],
                                input_stats=source_stats,
                                device=device,
                                trial_channel_mask=(
                                    None
                                    if dataset.trial_channel_mask is None
                                    else dataset.trial_channel_mask[combined]
                                ),
                            )
                            assert features is not None
                            prediction = shrinkage_feature_scores(
                                features[:budget], y_train, features[budget:]
                            )
                        elif arm == "zero_shot":
                            prediction = predict_n2p3_checkpoint(
                                trunk,
                                dataset.X[test_rows],
                                input_stats=source_stats,
                                device=device,
                                trial_channel_mask=(
                                    None
                                    if dataset.trial_channel_mask is None
                                    else dataset.trial_channel_mask[test_rows]
                                ),
                            )
                        else:
                            adapter = SubjectAdapter(
                                trunk,
                                device=device,
                                config=SubjectAdapterConfig(
                                    head_kind="classifier_fine"
                                    if arm == "classifier_fine"
                                    else "full_fine",
                                    epochs=epochs,
                                    lr=lr,
                                    batch_size=args.batch_size,
                                    seed=args.seed,
                                    val_group_fraction=None,
                                    refit_full_prefix=False,
                                    freeze_batchnorm_stats=freeze_bn,
                                    input_statistics=normalization,
                                    input_mean=None if target_only else source_stats[0],
                                    input_std=None if target_only else source_stats[1],
                                ),
                            )
                            adapter.fit(
                                dataset.X[train_rows],
                                y_train,
                                trial_channel_mask=(
                                    None
                                    if dataset.trial_channel_mask is None
                                    else dataset.trial_channel_mask[train_rows]
                                ),
                            )
                            prediction = adapter.predict_logit(
                                dataset.X[test_rows],
                                trial_channel_mask=(
                                    None
                                    if dataset.trial_channel_mask is None
                                    else dataset.trial_channel_mask[test_rows]
                                ),
                            )
                            cell["fit_history"] = copy.deepcopy(adapter.last_history)
                            del adapter
                        if not np.isfinite(prediction).all():
                            raise ValueError("non-finite test predictions")
                        cell.update(
                            status="completed", auc=float(roc_auc_score(y_test, prediction))
                        )
                        np.savez_compressed(
                            args.output / f"{subjects.index(subject):03d}_{budget}_{arm}.npz",
                            participant_key=np.asarray(participant),
                            train_rows=train_rows,
                            test_rows=test_rows,
                            y=y_test,
                            scores=prediction,
                        )
                except ExpectedSubjectError as error:
                    cell.update(status="fit_failure", auc=None, failure=str(error))
                cell["elapsed_sec"] = time.perf_counter() - started
                report["cells"].append(cell)
                atomic_write_json(result_path, report)
                print(
                    f"[binary] subject={subject} K={budget} arm={arm} auc={cell['auc']}", flush=True
                )
                del trunk
                gc.collect()
    report["paired_transfer_effects"] = paired_transfer_effects(report["cells"], budget_field="calibration_trials")
    report["status"] = "completed"
    report["summary"] = [
        {
            "arm": arm,
            "calibration_trials": budget,
            "valid_participants": len(values),
            "requested_participants": len(subjects),
            "subject_macro_auc": float(np.mean(values)) if values else None,
        }
        for budget in sorted({k for _, resolved in splits.values() for k in resolved})
        for arm in calibration_arms(arms, budget, budgets)
        for values in [
            [
                cell["auc"]
                for cell in report["cells"]
                if cell["arm"] == arm
                and cell["calibration_trials"] == budget
                and cell["auc"] is not None
            ]
        ]
    ]
    atomic_write_json(result_path, report)


if __name__ == "__main__":
    main()
