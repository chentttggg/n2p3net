"""Evaluate an entirely held-out study with explicit recording-order calibration."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from data.epochs import load_epoch_dataset, loaded_epoch_cache_attestation
from research.evaluation import (
    build_evaluation_run_contract,
    checkpoint_model_origin,
    scratch_model_origin,
    verified_source_snapshot,
)
from research.execution import ExpectedSubjectError, atomic_write_json
from train.runtime import cpu_thread_budget
from transfer.binary import (
    binary_recording_split,
    binary_time_split,
    fit_shrinkage_feature_model,
    scratch_like_trunk,
    target_shrinkage_features,
)
from transfer.checkpoint import (
    checkpoint_input_stats,
    checkpoint_training_identity_ledger,
    load_checkpoint_payload,
    load_n2p3_trunk_checkpoint,
    predict_n2p3_checkpoint_with_features,
)
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig

ARMS = ("zero_shot", "classifier_fine", "full_fine", "target_only", "shrinkage_lda", "feature_shrinkage_lda")


def frozen_features(base, X, mask, stats, device, batch_size=256):
    scores, features = [], []
    for start in range(0, len(X), batch_size):
        score, feature = predict_n2p3_checkpoint_with_features(
            base, X[start:start + batch_size], input_stats=stats, device=device,
            trial_channel_mask=mask[start:start + batch_size],
        )
        if feature is None:
            raise ValueError("plain source trunk features are required")
        scores.append(score)
        features.append(feature)
    return np.concatenate(scores), np.concatenate(features)


def planned_splits(dataset, subject: str, protocol: dict, budgets: tuple[int, ...]):
    rows = np.flatnonzero(np.asarray(dataset.subject_ids).astype(str) == subject)
    timeline = dataset.event_timeline
    events = np.flatnonzero(timeline.evidence_indices >= 0)
    row_events = np.empty(dataset.n_epochs, dtype=np.int64)
    if not np.array_equal(np.sort(timeline.evidence_indices[events]), np.arange(dataset.n_epochs)):
        raise ValueError("one physical event per epoch is required")
    row_events[timeline.evidence_indices[events]] = events
    ev = row_events[rows]
    onset, available = timeline.onset_times_s[ev], timeline.evidence_available_times_s[ev]
    session, run = timeline.session_ids[ev].astype(str), timeline.run_ids[ev].astype(str)
    if protocol["kind"] == "single_recording_prefix":
        if len(set(zip(session, run, strict=True))) != 1:
            raise ValueError("single-recording protocol cannot concatenate reset clocks")
        splits = {
            k: binary_time_split(
                onset,
                available,
                calibration_trials=k,
                maximum_calibration_trials=max(budgets),
                tmin_s=dataset.preprocessing.tmin_ms / 1000,
            )
            for k in budgets
        }
    elif protocol["kind"] == "last_session_holdout":
        declared_sessions, declared_runs = protocol["session_order"], protocol["run_order"]
        if not set(session) <= set(declared_sessions) or not set(run) <= set(declared_runs):
            raise ValueError("recording ids are absent from the declared original protocol order")
        pairs = [
            (s, r)
            for s in declared_sessions
            for r in declared_runs
            if bool(((session == s) & (run == r)).any())
        ]
        if len({s for s, _ in pairs}) < 2 or pairs[-1][0] != protocol["test_session"]:
            raise ValueError("required later test session or earlier calibration is missing")

        def key(pair):
            return json.dumps(pair, separators=(",", ":"))

        order = tuple(key(pair) for pair in pairs)
        tests = tuple(key(pair) for pair in pairs if pair[0] == protocol["test_session"])
        identifiers = np.asarray([key(pair) for pair in zip(session, run, strict=True)])
        splits = {
            k: binary_recording_split(
                onset,
                available,
                identifiers,
                recording_order=order,
                test_recordings=tests,
                calibration_trials=k,
                maximum_calibration_trials=max(budgets),
            )
            for k in budgets
        }
    else:
        raise ValueError("unsupported explicit target protocol")
    common = splits[budgets[0]].test_rows
    if len(common) < protocol.get("minimum_test_trials", 200):
        raise ValueError("too few independent later test rows for the declared protocol")
    for split in splits.values():
        np.testing.assert_array_equal(split.test_rows, common)
    return rows, splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--subjects", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    with cpu_thread_budget(args.cpu_threads):
        execute(args)


def execute(args):
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    plan = json.loads(args.plan.read_text())
    budgets = tuple(plan["calibration_trials"])
    if not budgets or tuple(sorted(set(budgets))) != budgets or budgets[0] < 1:
        raise ValueError("calibration budgets must be unique increasing positive integers")
    requested = args.subjects.split(",")
    if len(requested) != len(set(requested)):
        raise ValueError("requested participants must be unique")
    evaluation = verified_source_snapshot(plan["evaluation_snapshot_manifest"])
    for name, digest in json.loads(Path(plan["evaluation_snapshot_manifest"]).read_text())[
        "files"
    ].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"executed source differs from snapshot: {name}")
    dataset = load_epoch_dataset(plan["target_cache"], require_labels=True, validation="attested")
    target_sha = loaded_epoch_cache_attestation(dataset)["sha256"]
    if target_sha != plan["target_cache_sha256"]:
        raise ValueError("target cache differs from the frozen plan")
    payload = load_checkpoint_payload(plan["checkpoint"])
    checkpoint_sha = hashlib.sha256(Path(plan["checkpoint"]).read_bytes()).hexdigest()
    training_identity = checkpoint_training_identity_ledger(payload)
    if set(training_identity.authority_keys()) != set(plan["expected_source_participant_keys"]):
        raise ValueError("source ledger differs from the complete admitted source studies")
    if plan["target_study"] in plan["source_studies"]:
        raise ValueError("the entire target study must be excluded from source training")
    if set(dataset.identity_table.authority_keys()) & set(training_identity.authority_keys()):
        raise ValueError("a target-study participant is present in source training")
    source_snapshot = verified_source_snapshot(plan["source_snapshot_manifest"])
    if payload["source_snapshot_sha256"] != source_snapshot["archive_sha256"]:
        raise ValueError("source model and declared training snapshot disagree")
    splits = {}
    exclusions = []
    for subject in requested:
        load_n2p3_trunk_checkpoint(payload, dataset, target_subject=subject)
        try:
            splits[subject] = planned_splits(dataset, subject, plan["target_protocol"], budgets)
        except ValueError as error:
            exclusions.append(
                {"subject": subject, "reason": str(error), "status": "protocol_ineligible"}
            )
    if not splits:
        raise ValueError(f"no requested participants satisfy the declared protocol: {exclusions}")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "n2p3_cross_dataset_transfer/1",
        "status": "running",
        "plan": plan,
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "checkpoint_sha256": checkpoint_sha,
        "source_snapshot": source_snapshot,
        "evaluation_snapshot": evaluation,
        "requested_subjects": requested,
        "exclusions": exclusions,
        "cells": [],
    }
    atomic_write_json(args.output / "record.json", report)
    stats = checkpoint_input_stats(payload, dataset.n_channels)
    loaded_seconds = time.perf_counter()
    with (args.output / "progress.jsonl").open("x", encoding="utf-8") as progress:
        for index, (subject, (rows, subject_splits)) in enumerate(splits.items()):
            participant = dataset.identity_table.record_for(subject).authority_key()
            base, _ = load_n2p3_trunk_checkpoint(payload, dataset, target_subject=subject)
            test = rows[subject_splits[budgets[0]].test_rows]
            y_test = dataset.y[test]
            mask = (
                np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2])
                if dataset.trial_channel_mask is None
                else dataset.trial_channel_mask
            )
            zero, test_features = frozen_features(base, dataset.X[test], mask[test], stats, device)
            maximum_train = rows[subject_splits[max(budgets)].calibration_rows]
            _, train_features = frozen_features(base, dataset.X[maximum_train], mask[maximum_train], stats, device)
            feature_rows = {int(row): index for index, row in enumerate(maximum_train)}
            initial_calibration = rows[subject_splits[budgets[0]].calibration_rows]
            observed = mask[initial_calibration].all(axis=0)
            lda_observed = bool(observed.any() and mask[rows][:, observed].all())
            raw_features = (
                target_shrinkage_features(
                    dataset.X[rows][:, observed], sfreq=dataset.preprocessing.sfreq
                )
                if observed.any()
                else None
            )
            for k, split in subject_splits.items():
                train = rows[split.calibration_rows]
                y_train = dataset.y[train]
                for arm in ARMS:
                    started = time.perf_counter()
                    target_only = arm in {"target_only", "shrinkage_lda"}
                    origin = (
                        scratch_model_origin(
                            {
                                "algorithm": arm,
                                "architecture": base.architecture_record()
                                if arm == "target_only"
                                else "20Hz_waveform_lda",
                                "seed": plan["seed"],
                            }
                        )
                        if target_only
                        else checkpoint_model_origin(payload, checkpoint_sha256=checkpoint_sha)
                    )
                    adaptation = {
                        "arm": arm,
                        "seed": plan["seed"],
                        "device": str(device),
                        "epochs": 0
                        if arm in {"zero_shot", "shrinkage_lda", "feature_shrinkage_lda"}
                        else plan["target_epochs"],
                        "lr": plan["target_lr"],
                        "input_statistics": "source" if arm in {"zero_shot", "feature_shrinkage_lda"} else "calibration_only",
                        "bn_update": arm in {"full_fine", "target_only"},
                        "calibration_trials": k,
                        "selection": "fixed_recipe",
                    }
                    contract = build_evaluation_run_contract(
                        arm_name=arm,
                        model_origin=origin,
                        dataset=dataset,
                        target_cache_sha256=target_sha,
                        source_snapshot_sha256=evaluation["archive_sha256"],
                        requested_subjects=[subject],
                        identity_policy="source_or_global",
                        target_protocol={
                            **plan["target_protocol"],
                            "maximum_calibration_trials": max(budgets),
                        },
                        adaptation=adaptation,
                        decision={"kind": "binary_no_candidate_claim"},
                        evidence_scope={
                            "stage": "development",
                            "heldout_study": plan["target_study"],
                            "source_studies": plan["source_studies"],
                            "product_confirmation": False,
                        },
                    )
                    cell = {
                        "subject": subject,
                        "participant_key": participant,
                        "K": k,
                        "arm": arm,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "calibration_positive": int(y_train.sum()),
                        "evaluation_contract": contract.record(),
                        "evaluation_contract_digest": contract.digest(),
                    }
                    key = f"{index:03d}_{k}_{arm}"
                    try:
                        if arm == "shrinkage_lda" and not lda_observed:
                            cell.update(status="unobserved_lda_input", auc=None)
                        elif len(np.unique(y_test)) != 2 or (
                            arm != "zero_shot" and len(np.unique(y_train)) != 2
                        ):
                            cell.update(status="insufficient_classes", auc=None)
                        else:
                            if arm == "zero_shot":
                                scores = zero
                            elif arm == "shrinkage_lda":
                                if raw_features is None:
                                    raise ValueError("no complete observed EEG channels for LDA")
                                fitted = fit_shrinkage_feature_model(
                                    raw_features[split.calibration_rows], y_train
                                )
                                scores = fitted.decision_function(raw_features[split.test_rows])
                            elif arm == "feature_shrinkage_lda":
                                fitted = fit_shrinkage_feature_model(
                                    train_features[[feature_rows[int(row)] for row in train]], y_train
                                )
                                scores = fitted.decision_function(test_features)
                            else:
                                trunk = (
                                    scratch_like_trunk(base, seed=plan["seed"])
                                    if target_only
                                    else copy.deepcopy(base)
                                )
                                adapter = SubjectAdapter(
                                    trunk,
                                    device=device,
                                    config=SubjectAdapterConfig(
                                        head_kind="classifier_fine"
                                        if arm == "classifier_fine"
                                        else "full_fine",
                                        epochs=plan["target_epochs"],
                                        lr=plan["target_lr"],
                                        batch_size=32,
                                        seed=plan["seed"],
                                        val_group_fraction=None,
                                        refit_full_prefix=False,
                                        input_statistics="target_prefix",
                                        freeze_batchnorm_stats=arm == "classifier_fine",
                                    ),
                                )
                                adapter.fit(
                                    dataset.X[train], y_train, trial_channel_mask=mask[train]
                                )
                                scores = adapter.predict_logit(
                                    dataset.X[test], trial_channel_mask=mask[test]
                                )
                                cell["fit_history"] = copy.deepcopy(adapter.last_history)
                                torch.save(
                                    {
                                        "trunk_state_dict": adapter.trunk.state_dict(),
                                        "input_mean": adapter._input_mean_t,
                                        "input_std": adapter._input_std_t,
                                        "evaluation_contract_digest": contract.digest(),
                                    },
                                    args.output / f"{key}.pt",
                                )
                                del adapter, trunk
                            if not np.isfinite(scores).all():
                                raise ValueError("nonfinite predictions")
                            cell.update(
                                status="completed",
                                auc=float(roc_auc_score(y_test, scores)),
                                bacc_at_raw_score_zero=float(
                                    balanced_accuracy_score(y_test, scores > 0)
                                ),
                            )
                            np.savez_compressed(
                                args.output / f"{key}.npz",
                                train_rows=train,
                                test_rows=test,
                                y=y_test,
                                scores=scores,
                            )
                    except ExpectedSubjectError as error:
                        cell.update(status="fit_failure", auc=None, failure=str(error))
                    cell["elapsed_seconds"] = time.perf_counter() - started
                    atomic_write_json(args.output / f"{key}.json", cell)
                    report["cells"].append(cell)
                    progress.write(
                        json.dumps(
                            {
                                f: cell.get(f)
                                for f in ("subject", "K", "arm", "status", "auc", "elapsed_seconds")
                            }
                        )
                        + "\n"
                    )
                    progress.flush()
                    print(
                        json.dumps(
                            {f: cell.get(f) for f in ("subject", "K", "arm", "status", "auc")}
                        ),
                        flush=True,
                    )
    report.update(status="completed", evaluation_seconds=time.perf_counter() - loaded_seconds)
    atomic_write_json(args.output / "record.json", report)


if __name__ == "__main__":
    main()
