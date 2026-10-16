"""Evaluate an entirely held-out study with explicit recording-order calibration."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
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
from data.study import (
    checkpoint_study_membership,
    dataset_study_membership,
    load_study_membership_proof,
    study_ids_for_identity,
)
from research.evaluation import (
    build_evaluation_run_contract,
    scratch_model_origin,
    verified_executed_snapshot,
    verified_source_snapshot,
)
from research.execution import ExpectedSubjectError, atomic_write_json
from train.runtime import cpu_thread_budget
from transfer.binary import (
    STANDARD_CALIBRATION_TRIALS,
    calibration_arms,
    fit_shrinkage_feature_model,
    matched_split_digests,
    paired_transfer_effects,
    scratch_like_trunk,
    target_shrinkage_features,
    validate_calibration_budgets,
    validate_matched_transfer_controls,
)
from transfer.binary import (
    dataset_calibration_splits as planned_splits,
)
from transfer.checkpoint import (
    PreparedTransferCheckpoint,
    assert_checkpoint_library_exclusion,
    checkpoint_input_stats,
    checkpoint_training_identity_ledger,
    load_checkpoint_payload,
    predict_n2p3_checkpoint_with_features,
)
from transfer.result_reuse import seal_result_record, target_result_key
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


def admit_unified_prefixes(dataset, prefixes, *, requested, budgets, plan,
                          training_identity, source_membership, prefix_manifest):
    """Bind actual raw views to this target plan and preserve allocation fallback."""
    from data.epochs import select_epoch_channels, select_scheduled_events
    from data.identity import assert_target_identity_excluded

    declared = {str(entry.get("subject", entry.get("record", {}).get("subject"))): entry
                for entry in prefix_manifest["records"]}
    source_studies = set(study_ids_for_identity(training_identity, source_membership))
    adjustments, allocations = [], {}
    for subject in requested:
        if subject not in prefixes:
            raise ValueError(f"requested participant lacks raw prefix preparation: {subject}")
        entry = prefixes[subject]
        evidence = entry["manifest"]
        if evidence != declared.get(subject):
            raise ValueError("loaded target prefixes differ from the frozen manifest")
        if tuple(evidence["requested_budgets"]) != budgets:
            raise ValueError("raw prefix requested budgets differ from the target plan")
        effective = tuple(sorted(entry["calibration"]))
        validate_calibration_budgets(effective)
        maximum = max(effective)
        expected = tuple(sorted({k for k in budgets if k <= maximum} | {maximum}))
        reasons = evidence.get("adjustments", {})
        if (tuple(evidence["effective_budgets"]) != effective or maximum > max(budgets)
                or evidence["maximum_calibration_events"] != maximum or effective != expected
                or set(reasons) != {str(k) for k in budgets if k > maximum}
                or any(not isinstance(reason, str) or not reason.strip() for reason in reasons.values())):
            raise ValueError("raw prefix effective budgets lack the declared limited-data adjustments")
        if any(evidence["protocol"].get(key) != value for key, value in plan["target_protocol"].items()):
            raise ValueError("raw prefix recording protocol differs from the target plan")
        expected_identity = dataset.identity_table.record_for(subject).authority_key()
        views = [entry["test"], *(view for view in entry["calibration"].values() if view is not None)]
        for inner in entry.get("early_stopping_inner", {}).values():
            if inner.get("status") == "ready":
                views.extend((inner["training"], inner["validation"]))
        for view in views:
            if view.identity_table.authority_keys() != (expected_identity,):
                raise ValueError("actual raw prefix participant authority differs from the target cache")
            membership = dataset_study_membership(view)
            studies = set(study_ids_for_identity(view.identity_table, membership))
            if studies != {plan["target_study"]} or studies & source_studies:
                raise ValueError("actual raw prefix study differs from the held-out target study")
            assert_target_identity_excluded(training_identity, view.identity_table.record_for(subject))
        fixed = select_scheduled_events(dataset, dataset.event_timeline.subject_ids.astype(str) == subject,
                                        reason="verify_fixed_test_against_authorized_raw_prefix")
        actual = select_epoch_channels(entry["test"], dataset.channel_names)
        masks = [view.trial_channel_mask if view.trial_channel_mask is not None else
                 np.broadcast_to(view.channel_mask, view.X.shape[:2]) for view in (fixed, actual)]
        if (fixed.preprocessing != actual.preprocessing
                or fixed.provenance.get("source_reference") != actual.provenance.get("source_reference")
                or fixed.event_timeline.fingerprint() != actual.event_timeline.fingerprint()
                or not np.array_equal(fixed.X, actual.X) or not np.array_equal(fixed.y, actual.y)
                or not np.array_equal(masks[0], masks[1])
                or not np.array_equal(fixed.channel_positions_m, actual.channel_positions_m)):
            raise ValueError("actual raw prefix test differs from the frozen target cache evidence")
        allocations[subject] = {"participant_key": expected_identity,
            "requested_budgets": list(budgets), "effective_budgets": list(effective), "adjustments": reasons,
            "retained_calibration_trials": {str(k): 0 if view is None else view.n_epochs
                                             for k, view in entry["calibration"].items()},
            "fixed_test_trials": actual.n_epochs}
        adjustments.extend({"subject": subject, "participant_key": expected_identity,
            "requested_K": k, "effective_K": maximum, "status": "budget_reduced" if maximum else "unavailable",
            "reason": reasons[str(k)]} for k in budgets if k > maximum)
    return {"budget_adjustments": adjustments, "prefix_allocations": allocations}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--subjects")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--stage", choices=("B0", "B0_score", "E1", "E3", "compact", "ES", "P1", "P2", "unified"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-rule", type=Path)
    parser.add_argument("--freeze-batch-from", type=Path, nargs="+")
    parser.add_argument("--declare-batch-cohort-from", type=Path, nargs="+")
    parser.add_argument("--batch-cohort-design", type=Path)
    parser.add_argument("--batch-cohort", type=Path)
    args = parser.parse_args(argv)
    if args.declare_batch_cohort_from:
        from research.contracts import semantic_sha256
        from transfer.batch_cohort import declare_batch_cohort

        if args.output.exists() or args.batch_cohort_design is None:
            parser.error("cohort declaration requires --batch-cohort-design and a new output file")
        design = json.loads(args.batch_cohort_design.read_text(encoding="utf-8"))
        if design.get("schema") == "n2p3_four_study_development/1":
            roster = {study: item["participant_keys"] for study, item in design["datasets"].items()}
        elif design.get("schema") == "n2p3_batch_cohort_design/1":
            roster = design["participant_keys_by_study"]
        else:
            parser.error("unsupported predeclared participant roster schema")
        cohort = declare_batch_cohort([json.loads(path.read_text(encoding="utf-8"))
                                      for path in args.declare_batch_cohort_from], roster)
        cohort.pop("digest")
        cohort["design_sha256"] = hashlib.sha256(args.batch_cohort_design.read_bytes()).hexdigest()
        cohort["digest"] = semantic_sha256(cohort)
        atomic_write_json(args.output, cohort)
        return
    if args.freeze_batch_from:
        from transfer.batch_cohort import freeze_batch_cohort_rule, validate_batch_cohort

        if args.output.exists():
            parser.error("batch freeze output must be a new JSON file")
        if args.batch_cohort is None:
            parser.error("global batch freeze requires --batch-cohort declared before B0 execution")
        cohort = validate_batch_cohort(json.loads(args.batch_cohort.read_text(encoding="utf-8")))
        rows = []
        inputs = []
        policies = set()
        for path in args.freeze_batch_from:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") != "completed" or record.get("task_manifest", {}).get("stage") != "B0":
                parser.error("batch freeze requires completed B0 inner-validation records")
            if record.get("evidence_scope", "development") != "development":
                parser.error("software correctness probes cannot freeze a scientific B0 rule")
            if record.get("batch_cohort_digest") != cohort["digest"]:
                parser.error("B0 results were not executed under the declared cohort")
            rows.extend(record["batch_selection_rows"])
            inputs.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            policies.add((record.get("source_seed"), record["task_manifest"].get("target_seed"),
                          record.get("requested_device"), json.dumps(record.get("environment"), sort_keys=True)))
        if len(policies) != 1:
            parser.error("B0 records must share frozen seeds, device and runtime environment")
        frozen = freeze_batch_cohort_rule(rows, cohort)
        frozen["inputs"] = inputs
        frozen["execution_policy"] = list(next(iter(policies)))
        atomic_write_json(args.output, frozen)
        return
    if args.plan is None or not args.subjects:
        parser.error("--plan and --subjects are required for target execution")
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    if args.dry_run and args.stage is None:
        parser.error("dry-run requires an explicit training stage")
    with cpu_thread_budget(args.cpu_threads):
        execute(args)


def execute(args):
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    evidence_scope = plan.get("evidence_scope", "development")
    if evidence_scope not in {"development", "software_correctness"}:
        raise ValueError("target training-chain evidence scope must be explicit development or software_correctness")
    if "correctness_target_epochs" in plan and evidence_scope != "software_correctness":
        raise ValueError("correctness budgets cannot alter development training")
    target_seed = plan.get("target_seed", plan.get("seed", 20260906))
    for field in ("target_cache", "checkpoint", "evaluation_snapshot_manifest", "source_snapshot_manifest"):
        path = Path(plan[field])
        if not path.is_absolute():
            plan[field] = str((args.plan.parent / path).resolve())
    for field in ("batch_rule_file", "batch_cohort_file", "target_prefix_manifest"):
        if plan.get(field) is not None:
            path = Path(plan[field])
            if not path.is_absolute():
                plan[field] = str((args.plan.parent / path).resolve())
    for field in ("study_membership_proofs",):
        plan[field] = [str((args.plan.parent / path).resolve()) if not Path(path).is_absolute() else path
                       for path in plan.get(field, [])]
    budgets = tuple(plan.get("calibration_trials", STANDARD_CALIBRATION_TRIALS))
    validate_calibration_budgets(budgets)
    validate_matched_transfer_controls(ARMS, budgets)
    requested = args.subjects.split(",")
    if len(requested) != len(set(requested)):
        raise ValueError("requested participants must be unique")
    evaluation = verified_executed_snapshot(plan["evaluation_snapshot_manifest"], root=ROOT)
    dataset = getattr(args, "target_dataset", None)
    if dataset is None:
        dataset = load_epoch_dataset(plan["target_cache"], require_labels=True, validation="attested")
    target_sha = loaded_epoch_cache_attestation(dataset)["sha256"]
    if target_sha != plan["target_cache_sha256"]:
        raise ValueError("target cache differs from the frozen plan")
    payload = load_checkpoint_payload(plan["checkpoint"])
    checkpoint_sha = hashlib.sha256(Path(plan["checkpoint"]).read_bytes()).hexdigest()
    result_reuse = getattr(args, "result_reuse", None)
    source_state_key = None
    if result_reuse is not None:
        from research.contracts import semantic_sha256
        from transfer.target_experiments import _tensor_state_digest

        source_state_key = semantic_sha256({"state": _tensor_state_digest(payload["trunk_state_dict"]),
            "input_mean": payload["input_mean"], "input_std": payload["input_std"],
            "calibration": payload.get("source_calibration"), "architecture": payload["architecture"]})
    training_identity = checkpoint_training_identity_ledger(payload)
    target_membership = dataset_study_membership(dataset)
    source_membership = checkpoint_study_membership(
        payload, proofs=[load_study_membership_proof(path) for path in plan["study_membership_proofs"]])
    assert_checkpoint_library_exclusion(payload, dataset.identity_table,
        source_membership=source_membership, target_membership=target_membership)
    if set(study_ids_for_identity(training_identity, source_membership)) != set(plan["source_studies"]):
        raise ValueError("declared source studies disagree with actual producer study membership")
    if set(study_ids_for_identity(dataset.identity_table, target_membership)) != {plan["target_study"]}:
        raise ValueError("declared target study disagrees with actual producer study membership")
    if set(training_identity.authority_keys()) != set(plan["expected_source_participant_keys"]):
        raise ValueError("source ledger differs from the complete admitted source studies")
    if plan["target_study"] in plan["source_studies"]:
        raise ValueError("the entire target study must be excluded from source training")
    if set(dataset.identity_table.authority_keys()) & set(training_identity.authority_keys()):
        raise ValueError("a target-study participant is present in source training")
    source_snapshot = verified_source_snapshot(plan["source_snapshot_manifest"])
    if payload["source_snapshot_sha256"] != source_snapshot["archive_sha256"]:
        raise ValueError("source model and declared training snapshot disagree")
    prepared = PreparedTransferCheckpoint(payload, dataset)
    prepared_origin = prepared.model_origin(checkpoint_sha)
    unified = getattr(args, "stage", None) == "unified"
    scope_datasets = None
    prefixes = None
    prefix_admission = {}
    if unified:
        from data.source_records import load_target_prefixes
        from research.source_matrix import (
            validate_unified_source_factors,
            validate_unified_source_input,
        )
        from train.early_stopping import UnifiedTrainingBudget

        plan = {**plan, "unified_training_budget": UnifiedTrainingBudget(
            **plan.get("unified_training_budget", {})
        ).record()}
        selection_binding = plan.get("source_prior_selection")
        if selection_binding is None:
            if evidence_scope != "software_correctness":
                raise ValueError("unified development requires frozen source-inner prior selection evidence")
        else:
            from research.contracts import semantic_sha256

            selection_path = Path(selection_binding["path"])
            if not selection_path.is_absolute():
                selection_path = args.plan.parent / selection_path
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if (semantic_sha256(selection) != selection_binding["record_sha256"]
                    or selection.get("status") != "selected"
                    or selection.get("selected_weight") != plan.get("source_prior_weight")
                    or selection.get("unified_training_budget") != plan["unified_training_budget"]
                    or selection.get("source_budget") != plan.get("source_budget")
                    or payload.get("source_budget") != plan.get("source_budget")
                    or selection.get("source_cache_sha256") != payload["source_cache_sha256"]
                    or plan["target_study"] not in selection.get("fold", {}).get("fit_ledger", {}).get("excluded_source_datasets", [])):
                raise ValueError("frozen source-prior selection does not match the actual unified source/target plan")
        validate_unified_source_factors(payload, plan.get("source_factors"))
        validate_unified_source_input(dataset, plan.get("source_preprocessing"))
        prefix_path = Path(plan["target_prefix_manifest"])
        prefix_bytes = prefix_path.read_bytes()
        if hashlib.sha256(prefix_bytes).hexdigest() != plan["target_prefix_manifest_sha256"]:
            raise ValueError("target raw prefix manifest differs from the frozen plan")
        prefixes = getattr(args, "target_prefixes", None)
        if prefixes is None:
            prefixes = load_target_prefixes(prefix_path, subjects=requested, load_inner=False)
        prefix_admission = admit_unified_prefixes(dataset, prefixes, requested=requested, budgets=budgets,
            plan=plan, training_identity=training_identity, source_membership=source_membership,
            prefix_manifest=json.loads(prefix_bytes))
    splits = {}
    exclusions = []
    unavailable_budgets = prefix_admission.get("budget_adjustments", [])
    for subject in requested:
        prepared.assert_target_excluded(subject)
        if unified:
            continue
        try:
            missing = {}
            splits[subject] = planned_splits(dataset, subject, plan["target_protocol"], budgets, unavailable=missing)
            unavailable_budgets.extend({"subject": subject, "participant_key": dataset.identity_table.record_for(subject).authority_key(),
                                        "requested_K": k, "effective_K": max(splits[subject][1]),
                                        "status": "budget_reduced" if max(splits[subject][1]) else "unavailable", "reason": reason}
                                       for k, reason in missing.items())
        except ValueError as error:
            exclusions.append(
                {"subject": subject, "participant_key": dataset.identity_table.record_for(subject).authority_key(),
                 "study": plan["target_study"], "requested_Ks": list(budgets),
                 "reason": str(error), "status": "protocol_ineligible"}
            )
    if not splits and not getattr(args, "stage", None):
        raise ValueError(f"no requested participants satisfy the declared protocol: {exclusions}")
    if getattr(args, "stage", None):
        from transfer.target_experiments import (
            apply_correctness_budget,
            build_target_tasks,
            build_unified_scoped_tasks,
            execute_target_tasks,
            resolve_target_batch_rule,
        )

        rule_path = getattr(args, "batch_rule", None)
        batch, batch_rule = resolve_target_batch_rule(plan, rule_path=rule_path,
            requested_batch=getattr(args, "batch_size", None), dry_run=args.dry_run)
        if args.stage == "B0_score" and not (rule_path or plan.get("batch_rule") or plan.get("batch_rule_file")):
            raise ValueError("B0 final scoring requires a frozen inner-validation batch rule")
        seed = getattr(args, "target_seed", None)
        if seed is None:
            seed = target_seed
        order = plan["target_protocol"]
        if unified:
            manifest, scope_datasets = build_unified_scoped_tasks(prefixes,
                requested_subjects=requested, channel_names=dataset.channel_names,
                preprocessing=plan["source_preprocessing"], batch_size=batch, seed=seed,
                source_prior_weight=plan["source_prior_weight"],
                decision_weight=plan.get("decision_weight", 1.),
                target_max_updates=plan.get("target_max_updates"),
                unified_training_budget=plan["unified_training_budget"])
        else:
            manifest = build_target_tasks(dataset, splits, stage=args.stage, batch_size=batch, seed=seed,
                                      recording_order=order,
                                      include_historical_controls=plan.get("include_historical_controls", True),
                                      epoch_selection=plan.get("epoch_selection"),
                                      global_early_stopping=plan.get("global_early_stopping"))
        if evidence_scope == "software_correctness":
            apply_correctness_budget(manifest, epochs=plan.get("correctness_target_epochs", 1))
        manifest.update(batch_rule=batch_rule, budget_adjustments=unavailable_budgets,
                        protocol_exclusions=exclusions, requested_calibration_budgets=list(budgets))
        if unified:
            manifest["prefix_allocations"] = prefix_admission["prefix_allocations"]
        context = dict(plan=plan, target_study=plan["target_study"], target_protocol=plan["target_protocol"],
                       evidence_scope=evidence_scope,
                       source_seed=prepared.source_seed,
                       target_cache_sha256=target_sha, checkpoint_sha256=checkpoint_sha,
                       source_snapshot=source_snapshot, evaluation_snapshot=evaluation,
                       source_study_membership=source_membership, target_study_membership=target_membership,
                       requested_subjects=requested, requested_device=str(device),
                       environment={"python": platform.python_version(), "torch": torch.__version__,
                                    "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
                                    "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda"
                                                    and torch.cuda.is_available() else "cpu"),
                                    "cpu_threads": args.cpu_threads},
                       plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest())
        if args.stage == "B0" and not args.dry_run and evidence_scope == "development":
            from transfer.batch_cohort import validate_batch_cohort, verify_batch_cohort_manifest

            cohort_path = getattr(args, "batch_cohort", None) or plan.get("batch_cohort_file")
            if cohort_path is None:
                raise ValueError("B0 development execution requires a predeclared --batch-cohort")
            cohort = validate_batch_cohort(json.loads(Path(cohort_path).read_text(encoding="utf-8")))
            verify_batch_cohort_manifest(manifest, plan["target_study"], cohort)
            context["batch_cohort_digest"] = cohort["digest"]
        if args.dry_run:
            args.output.mkdir(parents=True, exist_ok=False)
            atomic_write_json(args.output / "tasks.json", {**context, **manifest})
            print(json.dumps({"stage": args.stage, "task_count": manifest["task_count"],
                              "fit_calls": manifest["fit_calls"], "dry_run": True}))
        else:
            execute_target_tasks(dataset, payload, manifest, output=args.output, context=context,
                                 device=str(device), recording_order=order,
                                 result_reuse=getattr(args, "result_reuse", None), scope_datasets=scope_datasets)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "n2p3_cross_dataset_transfer/1",
        "evidence_scope": evidence_scope,
        "status": "running",
        "plan": plan,
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "checkpoint_sha256": checkpoint_sha,
        "source_snapshot": source_snapshot,
        "evaluation_snapshot": evaluation,
        "requested_subjects": requested,
        "source_seed": prepared.source_seed,
        "target_seed": target_seed,
        "source_study_membership": source_membership,
        "target_study_membership": target_membership,
        "exclusions": exclusions,
        "calibration_trials": list(budgets),
        "budget_adjustments": unavailable_budgets,
        "cells": [],
    }
    atomic_write_json(args.output / "record.json", report)
    stats = checkpoint_input_stats(payload, dataset.n_channels)
    loaded_seconds = time.perf_counter()
    with (args.output / "progress.jsonl").open("x", encoding="utf-8") as progress:
        for index, (subject, (rows, subject_splits)) in enumerate(splits.items()):
            participant = dataset.identity_table.record_for(subject).authority_key()
            base = prepared.trunk_for_target(subject)
            test = rows[subject_splits[budgets[0]].test_rows]
            y_test = dataset.y[test]
            mask = (
                np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2])
                if dataset.trial_channel_mask is None
                else dataset.trial_channel_mask
            )
            zero, test_features = frozen_features(base, dataset.X[test], mask[test], stats, device)
            effective_maximum = max(subject_splits)
            maximum_train = rows[subject_splits[effective_maximum].calibration_rows]
            train_features = None
            if len(maximum_train):
                _, train_features = frozen_features(base, dataset.X[maximum_train], mask[maximum_train], stats, device)
            feature_rows = {int(row): index for index, row in enumerate(maximum_train)}
            first_positive = next((k for k in subject_splits if k > 0), None)
            initial_calibration = rows[subject_splits[first_positive].calibration_rows] if first_positive else rows[:0]
            observed = mask[initial_calibration].all(axis=0) if len(initial_calibration) else np.zeros(dataset.n_channels, dtype=bool)
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
                split_digests = matched_split_digests(train, test)
                for arm in calibration_arms(ARMS, k, budgets):
                    started = time.perf_counter()
                    target_only = arm in {"target_only", "shrinkage_lda"}
                    origin = (
                        scratch_model_origin(
                            {
                                "algorithm": arm,
                                "architecture": base.architecture_record()
                                if arm == "target_only"
                                else "20Hz_waveform_lda",
                                "seed": target_seed,
                            }
                        )
                        if target_only
                        else prepared_origin
                    )
                    adaptation = {
                        "arm": arm,
                        "seed": target_seed,
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
                            "maximum_calibration_trials": effective_maximum,
                            "requested_calibration_trials": list(budgets),
                        },
                        adaptation=adaptation,
                        decision={"kind": "binary_no_candidate_claim"},
                        evidence_scope={
                            "stage": evidence_scope,
                            "heldout_study": plan["target_study"],
                            "source_studies": plan["source_studies"],
                            "product_confirmation": False,
                            "source_study_membership_digest": source_membership["digest"],
                            "target_study_membership_digest": target_membership["digest"],
                        },
                    )
                    cell = {
                        **split_digests,
                        "study": plan["target_study"],
                        "source_arm": plan.get("source_arm"),
                        "source_seed": prepared.source_seed,
                        "target_seed": target_seed,
                        "evidence_scope": evidence_scope,
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
                    reuse_key = None
                    cached = None
                    if result_reuse is not None:
                        reuse_key = target_result_key(dataset_sha256=target_sha, participant_key=participant,
                            split=split_digests, architecture=base.architecture_record(), treatment=adaptation,
                            batch_size=plan.get("target_batch_size", 32),
                            source_state=None if target_only else source_state_key,
                            snapshot_sha256=evaluation["archive_sha256"], device=device)
                        cached = result_reuse.get(reuse_key)
                    try:
                        if arm == "shrinkage_lda" and not lda_observed:
                            cell.update(status="unobserved_lda_input", auc=None)
                        elif len(np.unique(y_test)) != 2 or (
                            arm != "zero_shot" and len(np.unique(y_train)) != 2
                        ):
                            cell.update(status="insufficient_classes", auc=None)
                        else:
                            if cached is not None:
                                scores = cached["scores"]
                                if cached["fit_history"] is not None:
                                    cell["fit_history"] = cached["fit_history"]
                                if cached["model_payload"] is not None:
                                    fitted_payload = cached["model_payload"]
                                    fitted_payload["evaluation_contract_digest"] = contract.digest()
                                    torch.save(fitted_payload, args.output / f"{key}.pt")
                                cell["prediction_reuse"] = {"key": reuse_key,
                                    "origin_prediction": cached["origin_prediction"],
                                    "origin_contract_digest": cached["origin_contract_digest"],
                                    "origin_model_sha256": cached["model_sha256"],
                                    "origin_result_integrity": cached["origin_result_integrity"],
                                    "fit_executed": False}
                            elif arm == "zero_shot":
                                scores = zero
                            elif arm == "shrinkage_lda":
                                if raw_features is None:
                                    raise ValueError("no complete observed EEG channels for LDA")
                                fitted = fit_shrinkage_feature_model(
                                    raw_features[split.calibration_rows], y_train
                                )
                                scores = fitted.decision_function(raw_features[split.test_rows])
                            elif arm == "feature_shrinkage_lda":
                                assert train_features is not None
                                fitted = fit_shrinkage_feature_model(
                                    train_features[[feature_rows[int(row)] for row in train]], y_train
                                )
                                scores = fitted.decision_function(test_features)
                            else:
                                trunk = (
                                    scratch_like_trunk(base, seed=target_seed)
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
                                        batch_size=plan.get("target_batch_size", 32),
                                        seed=target_seed,
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
                    artifacts = {}
                    model_path = args.output / f"{key}.pt"
                    if cell["status"] == "completed":
                        artifacts["prediction"] = args.output / f"{key}.npz"
                        if model_path.is_file():
                            artifacts["model"] = model_path
                    cell = seal_result_record(cell, files=artifacts)
                    atomic_write_json(args.output / f"{key}.json", cell)
                    if result_reuse is not None and cached is None and cell["status"] == "completed":
                        result_reuse.put(reuse_key, scores=scores, cell=cell,
                            prediction_path=args.output / f"{key}.npz", model_path=artifacts.get("model"))
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
    report["paired_transfer_effects"] = paired_transfer_effects(report["cells"], budget_field="K")
    report.update(status="completed", evaluation_seconds=time.perf_counter() - loaded_seconds)
    report = seal_result_record(report)
    atomic_write_json(args.output / "record.json", report)


if __name__ == "__main__":
    main()
