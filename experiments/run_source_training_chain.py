"""Execute or preflight the S1/E2/S2 source chain from an explicit local plan."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from baselines.deep import DeepConfig  # noqa: E402
from data.contract import SOURCE_COHORT_DATA_CONTRACTS, assert_p300_input_contract  # noqa: E402
from data.epochs import (  # noqa: E402
    load_epoch_dataset,
    loaded_epoch_cache_attestation,
    save_epoch_dataset,
    select_epoch_channels,
)
from data.study import dataset_study_membership, study_ids_for_identity  # noqa: E402
from research.contracts import TrainingRunContract, semantic_sha256  # noqa: E402
from research.evaluation import (  # noqa: E402
    build_evaluation_run_contract,
    verified_source_snapshot,
)
from research.execution import ExpectedSubjectError, atomic_write_json  # noqa: E402
from research.source_execution import (  # noqa: E402
    HostMemoryAdmission,
    SharedTrajectoryRegistry,
    gpu_task_slot,
    release_source_worker,
    resolve_source_resources,
)
from research.source_resume import SourceChainResume  # noqa: E402
from research.source_selection import (  # noqa: E402
    SOURCE_CHAIN_LAYOUTS,
    SOURCE_UPDATE_CANDIDATES,
    fit_source_trajectory,
    select_source_updates,
    source_fit_ledger,
    source_fixed_dose,
    source_identity_axes,
    source_inner_folds,
)
from train.runtime import available_cpu_threads, cpu_thread_budget  # noqa: E402
from transfer.binary import (  # noqa: E402
    STANDARD_CALIBRATION_TRIALS,
    dataset_calibration_splits,
    matched_split_digests,
)
from transfer.calibration_candidates import fit_calibration_candidate  # noqa: E402
from transfer.checkpoint import (  # noqa: E402
    PreparedTransferCheckpoint,
    checkpoint_training_contract,
    load_checkpoint_payload,
    predict_n2p3_checkpoint,
)
from transfer.recipes import TargetRecipe  # noqa: E402
from transfer.target_experiments import resolve_target_batch_rule  # noqa: E402


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _model_state_digest(trunk):
    digest = hashlib.sha256()
    for name, value in sorted(trunk.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def fit_or_reuse_source_trajectory(
    *, registry, dataset, rows, config, destination, task_id,
    source_cache_sha256, source_snapshot_sha256, excluded_source_datasets,
    device, evidence_scope, gpu_slots=None,
):
    """Reuse exact source training inputs while binding each consumer's own contract."""
    rows = np.asarray(rows, dtype=np.int64)
    fit_ledger = source_fit_ledger(dataset, rows, excluded_source_datasets=excluded_source_datasets)
    domains, people, _ = source_identity_axes(dataset)
    timeline = dataset.event_timeline
    events = np.flatnonzero(timeline.evidence_indices >= 0)
    if not np.array_equal(np.sort(timeline.evidence_indices[events]), np.arange(dataset.n_epochs)):
        raise ValueError("source trajectory reuse requires one physical event per epoch")
    row_events = np.empty(dataset.n_epochs, dtype=np.int64)
    row_events[timeline.evidence_indices[events]] = events
    training_config = asdict(config)
    training_config.pop("fixed_updates")
    training_config.pop("snapshot_updates")
    signature_record = {
        "fit_input_label_mask_sha256": fit_ledger["fit_input_label_mask_sha256"],
        "ordered_event_ids_sha256": semantic_sha256(timeline.event_ids[row_events[rows]].astype(str).tolist()),
        "ordered_participant_keys_sha256": semantic_sha256(people[rows].tolist()),
        "ordered_domains_sha256": semantic_sha256(domains[rows].tolist()),
        "preprocessing": dataset.preprocessing.record(),
        "channel_names": list(dataset.channel_names),
        "channel_positions_m": dataset.channel_positions_m.tolist(),
        "channel_mask": dataset.channel_mask.tolist(),
        "source_reference": dataset.provenance.get("source_reference"),
        "config": training_config, "device": str(torch.device(device)),
        "source_snapshot_sha256": source_snapshot_sha256, "evidence_scope": evidence_scope,
        "initialization": "scratch_N2P3NetBaseline_fixed_source_seed",
    }
    signature = semantic_sha256(signature_record)
    arguments = dict(dataset=dataset, rows=rows, config=config, destination=destination, task_id=task_id,
        source_cache_sha256=source_cache_sha256, source_snapshot_sha256=source_snapshot_sha256,
        excluded_source_datasets=excluded_source_datasets, device=device, evidence_scope=evidence_scope,
        signature=signature, signature_record=signature_record, fit_ledger=fit_ledger, gpu_slots=gpu_slots)
    if isinstance(registry, SharedTrajectoryRegistry):
        with registry.transaction(signature) as local:
            return _materialize_source_trajectory(registry=local, **arguments)
    return _materialize_source_trajectory(registry=registry, **arguments)


def _materialize_source_trajectory(*, registry, dataset, rows, config, destination, task_id,
        source_cache_sha256, source_snapshot_sha256, excluded_source_datasets, device, evidence_scope,
        signature, signature_record, fit_ledger, gpu_slots):
    requested_updates = sorted(set(config.snapshot_updates) | {int(config.fixed_updates)})
    existing = next((entry for entry in registry.get(signature, [])
                     if set(requested_updates) <= set(entry["checkpoints"])), None)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    reuse = {"schema": "n2p3_exact_source_trajectory_reuse/1", "reused": existing is not None,
             "training_signature": signature, "signature_record": signature_record,
             "requested_source_cache_sha256": source_cache_sha256,
             "requested_task": task_id, "requested_updates": requested_updates,
             "origin_task": task_id if existing is None else existing["task_id"],
             "origin_source_cache_sha256": source_cache_sha256 if existing is None else existing["source_cache_sha256"],
             "origin_checkpoints": {}}
    if existing is None:
        stored = {}
        with gpu_task_slot(gpu_slots):
            try:
                model, payloads = fit_source_trajectory(
                    dataset, rows, config=config, source_cache_sha256=source_cache_sha256,
                    source_snapshot_sha256=source_snapshot_sha256,
                    excluded_source_datasets=excluded_source_datasets,
                    device=torch.device(device), evidence_scope=evidence_scope,
                )
                physical_updates = int(model.last_history["optimizer_updates"])
                physical_draws = int(model.last_history["trial_draws"])
                for update, payload in payloads.items():
                    path = destination / f"source.u{update}.pt"
                    torch.save(payload, path)
                    checkpoints[update] = str(path.resolve())
                    stored[update] = {"path": str(path.resolve()), "sha256": _sha256(path)}
                del model, payloads
            finally:
                release_source_worker(device)
        registry.setdefault(signature, []).append({
            "task_id": task_id, "source_cache_sha256": source_cache_sha256, "checkpoints": stored,
        })
    else:
        physical_updates = physical_draws = 0
        membership = dataset_study_membership(dataset)
        for update in requested_updates:
            origin = existing["checkpoints"][update]
            if not Path(origin["path"]).is_file() or _sha256(origin["path"]) != origin["sha256"]:
                raise ValueError("cached source checkpoint changed after its original training")
            payload = load_checkpoint_payload(origin["path"])
            original = checkpoint_training_contract(payload)
            if set(original.training_participant_keys) != set(fit_ledger["training_participant_keys"]):
                raise ValueError("cached source checkpoint has different training participants")
            origin_record = {**origin, "training_contract_digest": original.digest(),
                             "source_snapshot_sha256": original.source_snapshot_sha256}
            reuse["origin_checkpoints"][str(update)] = origin_record
            provenance = {"schema": reuse["schema"], "training_signature": signature,
                          "origin_checkpoint": origin_record, "physical_fit_performed": False,
                          "physical_optimizer_updates": 0, "physical_trial_draws": 0}
            contract = TrainingRunContract(
                source_cache_sha256=source_cache_sha256,
                source_identity_digest=dataset.identity_table.digest(),
                source_snapshot_sha256=source_snapshot_sha256,
                architecture=original.architecture, preprocessing=original.preprocessing,
                optimizer={**original.optimizer, "config": asdict(config), "actual_updates": update,
                           "physical_optimizer_updates": 0, "physical_trial_draws": 0},
                validation={"strategy": "exact_fitted_source_trajectory_reuse", "fit_ledger": fit_ledger,
                            "evidence_scope": evidence_scope, "source_training_reuse": provenance},
                objective={**original.objective, "source_study_membership": membership},
                seed=config.seed, training_participant_keys=tuple(fit_ledger["training_participant_keys"]),
                holdout_participant_keys=(),
            )
            path = destination / f"source.u{update}.pt"
            torch.save({**payload, "source_cache_sha256": source_cache_sha256,
                "training_identity_ledger": fit_ledger["training_identity_ledger"],
                "training_identity_ledger_digest": fit_ledger["training_identity_ledger_digest"],
                "training_contract": contract.record(), "training_contract_digest": contract.digest(),
                "source_study_membership": membership, "source_fit_ledger": fit_ledger,
                "source_training_reuse": provenance,
                "runtime": {"physical_fit_performed": False, "origin_runtime": payload.get("runtime")},
            }, path)
            checkpoints[update] = str(path.resolve())
    result = {"checkpoints": checkpoints, "source_training_reuse": reuse, "fit_ledger": fit_ledger,
              "physical_optimizer_updates": physical_updates, "physical_trial_draws": physical_draws}
    atomic_write_json(destination / "trajectory_record.json", result)
    return result


def _chronological_rows(dataset, protocols):
    domains, people, _ = source_identity_axes(dataset)
    timeline = dataset.event_timeline
    ev = np.flatnonzero(timeline.evidence_indices >= 0)
    row_events = np.empty(dataset.n_epochs, int)
    if not np.array_equal(np.sort(timeline.evidence_indices[ev]), np.arange(dataset.n_epochs)):
        raise ValueError("chronological dose requires one physical event per epoch")
    row_events[timeline.evidence_indices[ev]] = ev
    result = {}
    for person in np.unique(people):
        rows = np.flatnonzero(people == person)
        protocol = protocols[str(domains[rows[0]])]
        events = row_events[rows]
        sessions, runs = timeline.session_ids[events].astype(str), timeline.run_ids[events].astype(str)
        if protocol["kind"] == "single_recording_prefix":
            if len(set(zip(sessions, runs, strict=True))) != 1:
                raise ValueError("fixed dose cannot infer order across recordings")
            ordering = {next(iter(zip(sessions, runs, strict=True))): 0}
        else:
            ordering = {(s, r): index for index, (s, r) in enumerate(
                (s, r) for s in protocol["session_order"] for r in protocol["run_order"])}
        if not set(zip(sessions, runs, strict=True)) <= set(ordering):
            raise ValueError("dose recordings absent from explicit protocol order")
        order = sorted(range(len(rows)), key=lambda i: (
            ordering[sessions[i], runs[i]], float(timeline.onset_times_s[events[i]])))
        result[person] = rows[order].tolist()
    return result


def _pseudo_target_cohort(dataset, validation_rows, protocols, budgets):
    domains, people, _ = source_identity_axes(dataset)
    subjects = np.asarray(dataset.subject_ids).astype(str)
    requested, failures = [], []
    for subject in sorted(set(subjects[validation_rows])):
        participant = dataset.identity_table.record_for(subject).authority_key()
        rows = np.flatnonzero(subjects == subject)
        domain = str(domains[rows[0]])
        adjustments = {}
        try:
            subject_rows, splits = dataset_calibration_splits(dataset, subject, protocols[domain], budgets,
                                                              unavailable=adjustments)
            for budget, split in splits.items():
                train, test = subject_rows[split.calibration_rows], subject_rows[split.test_rows]
                legal = set(dataset.y[test].tolist()) == {0, 1} and (
                    budget == 0 or (len(train) > 2 and set(dataset.y[train].tolist()) == {0, 1}))
                cell = {"domain": domain, "subject": subject, "participant_key": participant,
                        "actual_K": budget, "train_rows": train.tolist(), "test_rows": test.tolist(),
                        "purged_rows": subject_rows[split.purged_rows].tolist(),
                        "budget_adjustments": adjustments, **matched_split_digests(train, test)}
                if legal:
                    requested.append(cell)
                else:
                    failures.append({**cell, "reason": "missing_class_or_insufficient_fit_rows"})
        except ValueError as error:
            failures.append({"domain": domain, "subject": subject, "participant_key": participant,
                             "requested_budgets": list(budgets), "reason": str(error)})
    return requested, failures


def _score_checkpoint(path, dataset, cohort, *, cache_sha256, snapshot_sha256, recipe, output, device, evidence_scope):
    outcomes = []
    observed = (np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2])
                if dataset.trial_channel_mask is None else dataset.trial_channel_mask)
    started = time.perf_counter()
    try:
        prepared = PreparedTransferCheckpoint(path, dataset)
        payload = prepared.payload
        checkpoint_sha = _sha256(path)
        origin = prepared.model_origin(checkpoint_sha)
        preparation_error = None
    except (OSError, ValueError, RuntimeError) as error:
        preparation_error = str(error)
    preparation_seconds = time.perf_counter() - started
    for cell in cohort:
        outcome = {key: value for key, value in cell.items() if key not in {"train_rows", "test_rows", "purged_rows"}}
        outcome.update(status="fit_failure", auc=None, source_ce=None)
        try:
            if preparation_error is not None:
                raise ValueError(preparation_error)
            trunk = prepared.trunk_for_target(cell["subject"])
            source_state_digest = _model_state_digest(trunk)
            outcome["updates"] = int(payload["source_fixed_updates"])
            training = np.zeros(dataset.n_epochs, bool)
            training[cell["train_rows"]] = True
            test = np.asarray(cell["test_rows"], int)
            if cell["actual_K"]:
                fitted = fit_calibration_candidate(dataset, recipe, training, source_trunk=trunk,
                    source_input_stats=prepared.input_statistics,
                    device=device)
                scores = fitted.predict_logit(dataset.X[test], observed[test])
                outcome["adaptation_history"] = fitted.fit_history
            else:
                scores = predict_n2p3_checkpoint(trunk, dataset.X[test],
                    input_stats=prepared.input_statistics,
                    device=torch.device(device), trial_channel_mask=observed[test])
                weights = np.where(dataset.y[test] == 1, 8.0, 1.0)
                ce = np.logaddexp(0, scores) - dataset.y[test] * scores
                outcome["source_ce"] = float(weights @ ce / weights.sum())
            outcome.update(status="completed", auc=float(roc_auc_score(dataset.y[test], scores)))
            if _model_state_digest(trunk) != source_state_digest:
                raise RuntimeError("pseudo-target adaptation changed the source checkpoint state")
            outcome.update(source_model_state_sha256=source_state_digest,
                           source_input_statistics_sha256=semantic_sha256({"mean": payload["input_mean"], "std": payload["input_std"]}),
                           source_seed=payload["training_contract"]["seed"], target_seed=recipe.seed,
                           target_recipe=recipe.record(), checkpoint_sha256=checkpoint_sha,
                           checkpoint_preparation_seconds=preparation_seconds,
                           checkpoint_validation_scope="one_task_group_with_per_target_identity_checks")
            contract = build_evaluation_run_contract(
                arm_name="source_selection_calibrated" if cell["actual_K"] else "source_selection_zero",
                model_origin=origin,
                dataset=dataset, target_cache_sha256=cache_sha256, source_snapshot_sha256=snapshot_sha256,
                requested_subjects=(cell["subject"],), identity_policy="source_or_global",
                target_protocol={"scope": "source_only_pseudo_target", "actual_K": cell["actual_K"],
                                 "train_rows": cell["train_rows"], "test_rows": cell["test_rows"]},
                adaptation=recipe.record(), decision={"metric": "binary_auc", "source_ce_pos_weight": 8.0},
                evidence_scope={"stage": evidence_scope, "outer_target_used": False},
            )
            outcome.update(evaluation_contract=contract.record(), evaluation_contract_digest=contract.digest())
            name = semantic_sha256({"checkpoint": outcome["checkpoint_sha256"], "person": cell["participant_key"],
                                    "K": cell["actual_K"], "recipe": recipe.record()})[:24]
            np.savez_compressed(output / f"{name}.npz", train_rows=np.asarray(cell["train_rows"], int),
                                test_rows=test, y=dataset.y[test], scores=scores)
            outcome["prediction_path"] = str(output / f"{name}.npz")
        except (OSError, ValueError, RuntimeError, ExpectedSubjectError) as error:
            outcome.update(status="fit_failure", auc=None, source_ce=None, failure=str(error))
        outcomes.append(outcome)
    return outcomes


_E2_DATASET = None


def _initialize_e2_worker(dataset):
    global _E2_DATASET
    torch.set_num_threads(1)
    _E2_DATASET = dataset


def _score_e2_worker(arguments):
    path, cohort, options = arguments
    slots = options.pop("gpu_slots", None)
    with gpu_task_slot(slots):
        try:
            return _score_checkpoint(path, _E2_DATASET, cohort, **options)
        finally:
            release_source_worker(options["device"])


def _score_e2_candidates(dataset, path, cohort, candidates, *, workers, progress_path, checkpoint_paths=None,
                         gpu_slots=None, memory_admission=None, **options):
    if memory_admission is not None:
        with memory_admission.target_workers(workers) as effective:
            return _score_e2_candidates(dataset, path, cohort, candidates, workers=effective,
                progress_path=progress_path, checkpoint_paths=checkpoint_paths, gpu_slots=gpu_slots, **options)
    if type(workers) is not int or not 1 <= workers <= available_cpu_threads():
        raise ValueError("target_workers must be within the available CPU budget")
    if workers == 1:
        outcomes = []
        for update in candidates:
            checkpoint = path / f"source.u{update}.pt" if checkpoint_paths is None else checkpoint_paths[str(update)]
            with gpu_task_slot(gpu_slots):
                try:
                    rows = _score_checkpoint(checkpoint, dataset, cohort, **options)
                finally:
                    release_source_worker(options["device"])
            outcomes.extend({**row, "updates": update} for row in rows)
        return outcomes
    # Spawn isolates CUDA contexts and random generators from the source trainer.
    people = sorted({cell["participant_key"] for cell in cohort})
    shards = [set(people[index::workers]) for index in range(workers)]
    outcomes = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize_e2_worker, initargs=(dataset,)) as pool:
        futures = {}
        for update in candidates:
            for shard in shards:
                selected = [cell for cell in cohort if cell["participant_key"] in shard]
                if selected:
                    checkpoint = path / f"source.u{update}.pt" if checkpoint_paths is None else checkpoint_paths[str(update)]
                    arguments = (checkpoint, selected, {**options, "gpu_slots": gpu_slots})
                    futures[pool.submit(_score_e2_worker, arguments)] = update
        for future in as_completed(futures):
            outcomes.extend({**row, "updates": futures[future]} for row in future.result())
            atomic_write_json(progress_path, {"completed_cells": len(outcomes),
                "requested_cells": len(cohort) * len(candidates), "workers": workers,
                "failed_cells": sum(row["status"] != "completed" for row in outcomes)})
    return sorted(outcomes, key=lambda row: (row["updates"], row["participant_key"], row["actual_K"]))


def _save_chain_report(output, report):
    atomic_write_json(output / "source_chain_record.json", report)
    tasks = report.get("tasks", [])
    atomic_write_json(output / "worker_progress.json", {
        "pid": os.getpid(), "updated_unix": time.time(), "task_count": len(tasks),
        "completed_tasks": sum(task["status"] == "completed" for task in tasks),
        "failed_tasks": sum(task["status"] == "fit_failure" for task in tasks),
        "resumed_tasks": sum(task.get("source_training_reuse", {}).get("kind") == "completed_previous_run" for task in tasks),
        "physical_optimizer_updates": sum(task.get("physical_optimizer_updates", 0) for task in tasks),
        "last_task": tasks[-1]["key"] if tasks else None,
        "completed_selections": len(report.get("selection", [])),
    })


def source_execution_groups(plan, stages):
    groups = []
    names = [outer["name"] for outer in plan["outer_folds"]]
    layouts = tuple(plan.get("source_layouts", ("six", "common_three")))
    if len(names) != len(set(names)) or len(layouts) != len(set(layouts)):
        raise ValueError("source groups require unique outer names and layouts")
    for outer in plan["outer_folds"]:
        for layout in layouts:
            if layout not in SOURCE_CHAIN_LAYOUTS:
                raise ValueError("unsupported source layout")
            if layout == "six" and ("S1" not in stages or outer.get("family_exclusion", False)):
                continue
            groups.append((outer, layout))
    return groups


def _source_group_worker(arguments):
    plan, output, device, stages = arguments
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    with cpu_thread_budget(1), plan["_memory_admission"].source_group():
        execute_source_chain(plan, output=Path(output), dry_run=False, device=device,
                             stages=stages, evaluate_outer=False)
    return str(Path(output) / "source_chain_record.json")


def _execute_source_groups(plan, *, output, device, stages, evaluate_outer, groups):
    from research.evaluation import verified_executed_snapshot

    resources = resolve_source_resources(plan, len(groups), device)
    snapshot = verified_executed_snapshot(plan["source_snapshot_manifest"], root=ROOT)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("parallel source chain requires a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    batch, rule = resolve_target_batch_rule(plan)
    plan = {**plan, "target_batch_size": batch, "batch_rule": rule, "target_workers": resources["target_workers"]}
    report = {"schema": "n2p3_source_training_chain_run/1", "dry_run": False,
              "status": "running", "source_snapshot": snapshot, "resource_policy": resources,
              "evidence_scope": plan.get("evidence_scope", "development"), "batch_rule": rule,
              "source_seeds": plan.get("source_seeds", [20260906]),
              "target_seeds": plan.get("target_seeds", [20260906]), "groups": [],
              "tasks": [], "selection": [], "preflight_failures": [], "outer_evaluation_jobs": []}
    _save_chain_report(output, report)
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        slots = manager.BoundedSemaphore(resources["gpu_slots"]) if torch.device(device).type == "cuda" else None
        memory_admission = HostMemoryAdmission(manager, resources)
        with ProcessPoolExecutor(max_workers=resources["source_workers"], mp_context=context, max_tasks_per_child=1) as pool:
            futures = {}
            for index, (outer, layout) in enumerate(groups):
                directory = output / "groups" / f"{index:02d}_{outer['name']}_{layout}"
                child = {**plan, "outer_folds": [outer], "source_layouts": [layout],
                         "_source_group": True, "_defer_outer_plans": True, "_gpu_slots": slots,
                         "_memory_admission": memory_admission,
                         "shared_registry_directory": str(output / "trajectory_registry"),
                         "target_workers": resources["inner_workers_per_group"]}
                entry = {"outer": outer["name"], "layout": layout, "directory": str(directory), "status": "running"}
                report["groups"].append(entry)
                futures[pool.submit(_source_group_worker, (child, str(directory), device, stages))] = entry
            _save_chain_report(output, report)
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    entry = futures[future]
                    try:
                        entry.update(status="completed", record=future.result())
                    except Exception as error:
                        entry.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
                for entry in report["groups"]:
                    progress = Path(entry["directory"]) / "worker_progress.json"
                    if progress.exists():
                        entry["progress"] = json.loads(progress.read_text(encoding="utf-8"))
                _save_chain_report(output, report)
        if any(entry["status"] == "failed" for entry in report["groups"]):
            report["status"] = "failed"
            _save_chain_report(output, report)
            raise RuntimeError("source group failed; completed group records are preserved")
        for entry in report["groups"]:
            child = json.loads(Path(entry["record"]).read_text(encoding="utf-8"))
            for key in ("tasks", "selection", "preflight_failures", "outer_evaluation_jobs"):
                report[key].extend(child[key])
        report.update(task_count=len(report["tasks"]),
            optimizer_updates=sum(task["optimizer_updates"] for task in report["tasks"]),
            trial_draws=sum(task["trial_draws"] for task in report["tasks"]),
            source_failure_count=sum(task["status"] == "fit_failure" for task in report["tasks"]),
            reused_source_task_count=sum(bool(task.get("source_training_reuse", {}).get("reused")) for task in report["tasks"]),
            completed_physical_optimizer_updates=sum(task.get("physical_optimizer_updates", 0) for task in report["tasks"]),
            completed_physical_trial_draws=sum(task.get("physical_trial_draws", 0) for task in report["tasks"]))
        report["fitting_performed"] = report["completed_physical_optimizer_updates"] > 0
        _prepare_outer_plans(report, {**plan, "_gpu_slots": slots}, output, device=device, execute=evaluate_outer)
    report["status"] = "completed"
    _save_chain_report(output, report)
    return report


def execute_source_chain(plan, *, output: Path, dry_run: bool, device: str, stages: tuple[str, ...], evaluate_outer: bool = False):
    if plan.get("schema") != "n2p3_source_training_chain/1":
        raise ValueError("unsupported source training chain plan schema")
    if not dry_run and not plan.get("_source_group", False):
        groups = source_execution_groups(plan, stages)
        if groups:
            return _execute_source_groups(plan, output=output, device=device, stages=stages,
                                           evaluate_outer=evaluate_outer, groups=groups)
    workers = plan.get("target_workers", 1)
    if workers in (None, "auto"):
        workers = 1 if dry_run else available_cpu_threads()
        plan = {**plan, "target_workers": workers}
    if type(workers) is not int or not 1 <= workers <= available_cpu_threads():
        raise ValueError("target_workers must be within the available CPU budget")
    evidence_scope = plan.get("evidence_scope", "development")
    if evidence_scope not in {"development", "software_correctness"}:
        raise ValueError("source chain supports only development or software_correctness evidence")
    correctness_fields = {key for key in plan if key.startswith("correctness_")}
    if correctness_fields and evidence_scope != "software_correctness":
        raise ValueError("correctness budgets cannot be used under public development evidence")
    if evidence_scope == "software_correctness":
        candidates = tuple(plan.get("correctness_candidate_updates", [1, 2]))
        source_updates = int(plan.get("correctness_source_updates", 2))
        if not candidates or tuple(sorted(set(candidates))) != candidates or min(candidates) < 1 or max(candidates) > 10 or not 1 <= source_updates <= 10:
            raise ValueError("software correctness uses at most ten source updates")
    else:
        candidates, source_updates = SOURCE_UPDATE_CANDIDATES, 4000
    source_seeds = tuple(plan.get("source_seeds", [20260906]))
    target_seeds = tuple(plan.get("target_seeds", [20260906]))
    if not source_seeds or not target_seeds or len(set(source_seeds)) != len(source_seeds) or len(set(target_seeds)) != len(target_seeds):
        raise ValueError("source and target seeds must be independently unique")
    batch, batch_rule = resolve_target_batch_rule(plan, dry_run=dry_run)
    plan = {**plan, "target_batch_size": batch, "batch_rule": batch_rule}
    snapshot_path = Path(plan["source_snapshot_manifest"])
    snapshot = verified_source_snapshot(snapshot_path)
    if not dry_run:
        from research.evaluation import verified_executed_snapshot
        verified_executed_snapshot(snapshot_path, root=ROOT)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("source chain requires a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "n2p3_source_training_chain_run/1", "dry_run": dry_run,
              "evidence_scope": evidence_scope,
              "batch_rule": batch_rule,
              "source_snapshot": snapshot, "source_seeds": list(source_seeds),
              "target_seeds": list(target_seeds), "tasks": [], "selection": [],
              "preflight_failures": [], "outer_evaluation_jobs": [], "fitting_performed": False}
    _save_chain_report(output, report)
    trajectory_registry = (SharedTrajectoryRegistry(plan["shared_registry_directory"])
                           if plan.get("shared_registry_directory") else {})
    resume = SourceChainResume(plan.get("resume_source_run"))
    for outer in plan["outer_folds"]:
        source_cache = Path(outer["source_cache"])
        source = load_epoch_dataset(source_cache, require_labels=True, validation="attested")
        assert_p300_input_contract(source.preprocessing, SOURCE_COHORT_DATA_CONTRACTS["causal"])
        if source.provenance.get("source_reference") != "measured_channel:CZ":
            raise ValueError("source-chain caches must contain measured Cz differences")
        source_sha = loaded_epoch_cache_attestation(source)["sha256"]
        excluded = tuple(outer["excluded_source_datasets"])
        if not excluded:
            raise ValueError("outer whole-library exclusions must be explicitly named")
        source_fit_ledger(source, np.arange(source.n_epochs), excluded_source_datasets=excluded)
        domains, _, _ = source_identity_axes(source)
        if len(set(domains)) != (2 if outer.get("family_exclusion", False) else 3):
            raise ValueError("source chain requires three ordinary or two family-excluded source libraries")
        for layout in plan.get("source_layouts", ("six", "common_three")):
            if outer.get("family_exclusion", False) and layout != "common_three":
                continue
            if layout == "six" and "S1" not in stages:
                continue
            dataset = select_epoch_channels(source, SOURCE_CHAIN_LAYOUTS[layout])
            domains, _, _ = source_identity_axes(dataset)
            if layout == "common_three" and dataset.trial_channel_mask is not None and not dataset.trial_channel_mask.all():
                raise ValueError("common-three source experiments require genuinely observed channels on every row")
            all_rows = np.arange(dataset.n_epochs)
            for seed in source_seeds:
                base = DeepConfig(fixed_updates=source_updates, batch_size=512, seed=seed, val_group_frac=None,
                                  sampling="uniform_rows", compile_mode=plan.get("compile_mode", "reduce-overhead"))
                work = []
                if "S1" in stages and not outer.get("family_exclusion", False):
                    work.extend((f"S1_{statistics}_{sampling}", all_rows, replace(base, input_statistics=statistics, sampling=sampling))
                                for statistics in ("observed_rows", "observed_domain_participant")
                                for sampling in ("uniform_rows", "domain_participant"))
                if layout == "common_three" and "S2" in stages:
                    chronology = _chronological_rows(dataset, outer["source_protocols"])
                    groups = [("multi", sorted(set(domains)))] + [(f"single_{d}", [d]) for d in sorted(set(domains))]
                    if outer.get("family_exclusion", False):
                        groups = [("multi", sorted(set(domains)))]
                    for label, selected_domains in groups:
                        work.append((f"S2_full_{label}", np.flatnonzero(np.isin(domains, selected_domains)), base))
                        if outer.get("family_exclusion", False):
                            continue
                        dose = source_fixed_dose(dataset, domain_ids=selected_domains,
                            trials_per_participant=128 if label == "multi" else 384,
                            seed=seed, chronological_rows=chronology)
                        if dose["status"] != "ready":
                            report["preflight_failures"].append({"outer": outer["name"], "arm": f"S2_dose_{label}", **dose})
                        else:
                            work.append((f"S2_dose_{label}", np.asarray(dose["selected_rows"], int), base))
                if layout == "common_three" and "E2" in stages and not outer.get("family_exclusion", False):
                    for fold in source_inner_folds(dataset, seed=seed, excluded_source_datasets=excluded):
                        work.append((f"E2_{fold['axis']}_{fold['fold_id']}", np.asarray(fold["train_rows"], int),
                                     replace(base, fixed_updates=max(candidates), snapshot_updates=candidates)))
                for label, rows, config in work:
                    key = f"{outer['name']}/{layout}/seed{seed}/{label}"
                    ledger = source_fit_ledger(dataset, rows, excluded_source_datasets=excluded)
                    task = {"key": key, "status": "planned", "config": config.__dict__, "fit_ledger": ledger,
                            "source_cache": str(source_cache), "source_cache_sha256": source_sha,
                            "optimizer_updates": config.fixed_updates,
                            "trial_draws": config.fixed_updates * config.batch_size,
                            "layout_contrast": "information_and_input_dimension_change"}
                    report["tasks"].append(task)
                    if dry_run:
                        continue
                    destination = output / key
                    destination.mkdir(parents=True, exist_ok=True)
                    try:
                        acquired = resume.acquire(key, config, ledger, source_sha)
                        if acquired is None:
                            acquired = fit_or_reuse_source_trajectory(
                                registry=trajectory_registry, dataset=dataset, rows=rows, config=config,
                                destination=destination, task_id=key,
                                source_cache_sha256=source_sha, source_snapshot_sha256=snapshot["archive_sha256"],
                                excluded_source_datasets=excluded, device=device, evidence_scope=evidence_scope,
                                gpu_slots=plan.get("_gpu_slots"))
                    except RuntimeError as error:
                        task.update(status="fit_failure", failure=str(error))
                        _save_chain_report(output, report)
                        continue
                    report["fitting_performed"] |= acquired["physical_optimizer_updates"] > 0
                    task["status"] = "completed"
                    task.update({name: acquired[name] for name in (
                        "source_training_reuse", "physical_optimizer_updates", "physical_trial_draws")})
                    task["checkpoints"] = {str(update): value for update, value in acquired["checkpoints"].items()}
                    gc.collect()
                    _save_chain_report(output, report)
                    if not label.startswith("E2_"):
                        report["outer_evaluation_jobs"].append({"source_checkpoint": acquired["checkpoints"][source_updates],
                            "outer": outer["name"], "layout": layout, "source_seed": seed,
                            "source_arm": label,
                            "target_seeds": list(target_seeds), "target_cache": outer.get("target_cache"),
                            "target_protocol": outer.get("target_protocol")})
                if layout == "common_three" and "E2" in stages and not outer.get("family_exclusion", False):
                    _run_e2_selection(dataset, outer, seed, target_seeds, output, source_sha,
                                      snapshot["archive_sha256"], plan, report, dry_run, device,
                                      trajectory_registry, resume)
            del dataset
        del source
        gc.collect()
    report.update(task_count=len(report["tasks"]),
        optimizer_updates=sum(task["optimizer_updates"] for task in report["tasks"]),
        trial_draws=sum(task["trial_draws"] for task in report["tasks"]))
    report["requested_source_task_count"] = report["task_count"] + sum(
        failure.get("status") == "invalid_design" for failure in report["preflight_failures"])
    report["source_failure_count"] = sum(task.get("status") == "fit_failure" for task in report["tasks"])
    report["reused_source_task_count"] = sum(bool(task.get("source_training_reuse", {}).get("reused")) for task in report["tasks"])
    report["completed_physical_optimizer_updates"] = sum(task.get("physical_optimizer_updates", 0) for task in report["tasks"])
    report["completed_physical_trial_draws"] = sum(task.get("physical_trial_draws", 0) for task in report["tasks"])
    report["unexecutable_dose_count"] = sum(failure.get("status") == "invalid_design" for failure in report["preflight_failures"])
    report["resource_estimate"] = {
        "optimizer_updates": report["optimizer_updates"], "trial_draws": report["trial_draws"],
        "seconds": None, "seconds_status": "requires_measured_matching_device_throughput",
        "e2_refit_upper_bound_per_outer_source_seed": sum(candidates),
        "target_fit_count": sum(sum(int(cell[2] > 0) for cell in selection.get("requested_cells", []))
                                * len(candidates) for selection in report["selection"]),
    }
    if not dry_run and not plan.get("_defer_outer_plans", False):
        _prepare_outer_plans(report, plan, output, device=device, execute=evaluate_outer)
    _save_chain_report(output, report)
    return report


def _run_e2_selection(dataset, outer, seed, target_seeds, output, source_sha, snapshot_sha, plan, report, dry_run, device,
                      trajectory_registry, resume=None):
    folds = source_inner_folds(dataset, seed=seed, excluded_source_datasets=outer["excluded_source_datasets"])
    evidence_scope = plan.get("evidence_scope", "development")
    candidates = tuple(plan.get("correctness_candidate_updates", [1, 2])) if evidence_scope == "software_correctness" else SOURCE_UPDATE_CANDIDATES
    target_epochs = int(plan.get("correctness_target_epochs", 1)) if evidence_scope == "software_correctness" else 60
    if not 1 <= target_epochs <= 60:
        raise ValueError("target epoch budget is out of bounds")
    for target_seed in target_seeds:
        recipe = TargetRecipe("E2_public_fixed", batch_size=int(plan.get("target_batch_size", 32)), seed=target_seed, epochs=target_epochs)
        for axis in ("participant", "library"):
            outcomes, requested = [], []
            previous_selection = None if resume is None else resume.selection(outer["name"], seed, target_seed, axis)
            for fold in (item for item in folds if item["axis"] == axis):
                cohort, failures = _pseudo_target_cohort(dataset, fold["validation_rows"], outer["source_protocols"],
                                                         STANDARD_CALIBRATION_TRIALS)
                report["preflight_failures"].extend(failures)
                requested.extend((cell["domain"], cell["participant_key"], cell["actual_K"]) for cell in cohort)
                if dry_run:
                    continue
                path = output / f"{outer['name']}/common_three/seed{seed}/E2_{axis}_{fold['fold_id']}"
                path.mkdir(parents=True, exist_ok=True)
                if previous_selection is not None:
                    scopes = {(cell["participant_key"], cell["actual_K"]): cell for cell in cohort}
                    reused = [row for row in previous_selection["outcomes"]
                              if (row["participant_key"], row["actual_K"]) in scopes]
                    if len(reused) != len(cohort) * len(candidates):
                        raise ValueError("resumed E2 selection cohort changed")
                    for row in reused:
                        scope = scopes[row["participant_key"], row["actual_K"]]
                        if (row["target_recipe"] != recipe.record() or any(row[field] != scope[field] for field in (
                                "calibration_rows_sha256", "test_rows_sha256"))):
                            raise ValueError("resumed E2 recipe or scoring rows changed")
                        if row["status"] == "completed":
                            resume.verify(row["prediction_path"])
                        outcomes.append({**row, "resumed_from": str(resume.record_path)})
                else:
                    task_key = f"{outer['name']}/common_three/seed{seed}/E2_{axis}_{fold['fold_id']}"
                    checkpoints = next(task["checkpoints"] for task in report["tasks"] if task["key"] == task_key)
                    outcomes.extend(_score_e2_candidates(dataset, path, cohort, candidates,
                        workers=plan.get("target_workers", 1), progress_path=path / "scoring_progress.json",
                        checkpoint_paths=checkpoints, cache_sha256=source_sha, snapshot_sha256=snapshot_sha,
                        recipe=recipe, output=path, device=device, evidence_scope=evidence_scope,
                        gpu_slots=plan.get("_gpu_slots"), memory_admission=plan.get("_memory_admission")))
            if dry_run:
                report["selection"].append({"outer": outer["name"], "source_seed": seed, "target_seed": target_seed,
                    "axis": axis, "requested_cells": [list(cell) for cell in requested], "recipe": recipe.record()})
                continue
            if not requested:
                report["selection"].append({"outer": outer["name"], "source_seed": seed, "target_seed": target_seed,
                    "axis": axis, "status": "no_legal_pseudo_target_cohort", "U_ce": None, "U_cal": None, "U_zero": None})
                continue
            selected = select_source_updates(outcomes, requested_cells=requested, candidates=candidates)
            report["selection"].append({"outer": outer["name"], "source_seed": seed, "target_seed": target_seed,
                "axis": axis, **selected, "outcomes": outcomes})
            _save_chain_report(output, report)
            for updates in sorted({value for name, value in selected.items() if name in {"U_ce", "U_cal", "U_zero"} and value is not None}):
                destination = output / f"{outer['name']}/common_three/seed{seed}/refit.u{updates}.pt"
                refit_key = (outer["name"], seed, updates)
                existing_refit = next((item for item in report["tasks"] if item.get("refit_key") == list(refit_key)), None)
                if existing_refit is not None and existing_refit.get("status") == "fit_failure":
                    continue
                already_refit = existing_refit is not None
                if already_refit:
                    destination = Path(existing_refit["checkpoint"])
                if not already_refit:
                    config = DeepConfig(fixed_updates=updates, batch_size=512, seed=seed, val_group_frac=None,
                                        sampling="uniform_rows", compile_mode=plan.get("compile_mode", "reduce-overhead"))
                    try:
                        acquired = None if resume is None else resume.acquire(str(destination), config,
                            source_fit_ledger(dataset, np.arange(dataset.n_epochs),
                                excluded_source_datasets=outer["excluded_source_datasets"]),
                            source_sha, refit_key=refit_key)
                        if acquired is None:
                            acquired = fit_or_reuse_source_trajectory(
                                registry=trajectory_registry, dataset=dataset, rows=np.arange(dataset.n_epochs),
                                config=config, destination=destination.with_suffix(""), task_id=str(destination),
                                source_cache_sha256=source_sha, source_snapshot_sha256=snapshot_sha,
                                excluded_source_datasets=outer["excluded_source_datasets"], device=device, evidence_scope=evidence_scope,
                                gpu_slots=plan.get("_gpu_slots"))
                    except RuntimeError as error:
                        report["tasks"].append({"key": str(destination), "refit_key": list(refit_key),
                            "status": "fit_failure", "failure": str(error), "optimizer_updates": updates,
                            "trial_draws": updates * 512, "source_seed": seed, "config": config.__dict__})
                        _save_chain_report(output, report)
                        continue
                    destination = Path(acquired["checkpoints"][updates])
                    report["tasks"].append({"key": str(destination), "refit_key": list(refit_key),
                        "status": "completed", "checkpoint": str(destination),
                        "optimizer_updates": updates, "trial_draws": updates * 512,
                        "source_seed": seed, "fit_ledger": acquired["fit_ledger"],
                        "refit_initialization": "scratch_same_source_seed", "config": config.__dict__,
                        **{name: acquired[name] for name in (
                            "source_training_reuse", "physical_optimizer_updates", "physical_trial_draws")}})
                report["outer_evaluation_jobs"].append({"source_checkpoint": str(destination), "outer": outer["name"],
                    "layout": "common_three", "source_seed": seed, "target_seed": target_seed, "axis": axis,
                    "source_arm": "E2_" + axis,
                    "selectors": [name for name in ("U_ce", "U_cal", "U_zero") if selected[name] == updates],
                    "target_cache": outer.get("target_cache"), "target_protocol": outer.get("target_protocol")})


def _prepare_outer_plans(report, plan, output, *, device, execute):
    """Resolve projected targets only after all source selection has finished."""
    outers = {outer["name"]: outer for outer in plan["outer_folds"]}
    prepared = {}
    final_jobs = []
    grouped_jobs = {}
    workers = plan.get("target_workers", 1)
    for job in report["outer_evaluation_jobs"]:
        outer = outers[job["outer"]]
        targets = outer.get("targets") or [{"cache": outer["target_cache"],
            "protocol": outer["target_protocol"], "study": outer.get("target_study", outer["name"])}]
        payload = load_checkpoint_payload(job["source_checkpoint"])
        for target in targets:
            cache_key = (str(target["cache"]), job["layout"])
            if cache_key not in prepared:
                raw = load_epoch_dataset(target["cache"], require_labels=True, validation="attested")
                if not set(study_ids_for_identity(raw.identity_table, dataset_study_membership(raw))) <= set(outer["excluded_source_datasets"]):
                    raise ValueError("outer target library disagrees with the source exclusion ledger")
                selected = select_epoch_channels(raw, SOURCE_CHAIN_LAYOUTS[job["layout"]])
                cache_name = semantic_sha256({"parent": _sha256(target["cache"]), "layout": job["layout"]})[:20]
                cache = output / "target_views" / f"{cache_name}.npz"
                save_epoch_dataset(cache, selected)
                prepared[cache_key] = {"cache": str(cache.resolve()), "sha256": _sha256(cache),
                                      "subjects": selected.identity_table.local_subject_ids}
                prepared_checkpoint = PreparedTransferCheckpoint(payload, selected)
                for subject in selected.identity_table.local_subject_ids:
                    prepared_checkpoint.assert_target_excluded(subject)
                del raw, selected
                gc.collect()
            view = prepared[cache_key]
            for target_seed in job.get("target_seeds", [job.get("target_seed", 20260906)]):
                identifiers = job.get("selectors", [job["source_arm"]])
                for selector in identifiers:
                    name = semantic_sha256({"job": job, "target": target["study"],
                                            "seed": target_seed, "selector": selector})[:24]
                    evaluation_plan = {
                        "schema": "n2p3_cross_dataset_transfer_plan/1", "stage": plan.get("evidence_scope", "development"),
                        "evidence_scope": plan.get("evidence_scope", "development"),
                        "checkpoint": str(Path(job["source_checkpoint"]).resolve()),
                        "target_cache": view["cache"], "target_cache_sha256": view["sha256"],
                        "target_study": target["study"], "source_studies": payload["source_fit_ledger"]["actual_source_libraries"],
                        "expected_source_participant_keys": payload["training_contract"]["training_participant_keys"],
                        "source_snapshot_manifest": str(Path(plan.get("source_snapshot_manifests", {}).get(
                            payload["source_snapshot_sha256"], plan["source_snapshot_manifest"])).resolve()),
                        "evaluation_snapshot_manifest": str(Path(plan.get("evaluation_snapshot_manifest", plan["source_snapshot_manifest"])).resolve()),
                        "target_protocol": target["protocol"], "calibration_trials": list(STANDARD_CALIBRATION_TRIALS),
                        "source_seed": job["source_seed"], "target_seed": target_seed,
                        "target_batch_size": int(plan.get("target_batch_size", 32)),
                        "batch_rule": plan["batch_rule"],
                        "target_epochs": int(plan.get("correctness_target_epochs", 1)) if plan.get("evidence_scope") == "software_correctness" else 60,
                        "target_lr": 1e-3,
                        "source_arm": job["source_arm"] + ("_" + selector if "selectors" in job else ""),
                        "source_selector": selector, "source_layout": job["layout"],
                    }
                    plan_path = output / "outer_plans" / f"{name}.json"
                    atomic_write_json(plan_path, evaluation_plan)
                    result_path = output / "outer_results" / name
                    command = [str(ROOT / ".venv/Scripts/python.exe"), str(ROOT / "experiments/run_cross_dataset_transfer.py"),
                               "--plan", str(plan_path.resolve()), "--subjects", ",".join(view["subjects"]),
                               "--output", str(result_path.resolve()), "--device", device, "--cpu-threads", "1"]
                    final_jobs.append({"plan": str(plan_path.resolve()), "argv": command,
                                       "source_arm": evaluation_plan["source_arm"], "shards": []})
                    for shard in range(workers):
                        subjects = list(view["subjects"])[shard::workers]
                        if not subjects:
                            continue
                        shard_output = result_path if workers == 1 else result_path / f"shard{shard}"
                        entry = {"plan": str(plan_path.resolve()), "subjects": subjects,
                                 "output": str(shard_output), "device": device, "target_cache": view["cache"],
                                 "gpu_slots": plan.get("_gpu_slots")}
                        grouped_jobs.setdefault((view["cache"], shard), []).append(entry)
                        final_jobs[-1]["shards"].append(str(shard_output))
    report["outer_evaluation_jobs"] = final_jobs
    _save_chain_report(output, report)
    if execute:
        report["outer_execution"] = []
        if workers == 1:
            for jobs in grouped_jobs.values():
                report["outer_execution"].extend(_evaluate_outer_group(jobs))
                _save_chain_report(output, report)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [pool.submit(_evaluate_outer_group, jobs) for jobs in grouped_jobs.values()]
                for future in as_completed(futures):
                    report["outer_execution"].extend(future.result())
                    _save_chain_report(output, report)


def _evaluate_outer_group(jobs):
    from types import SimpleNamespace

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.run_cross_dataset_transfer import execute as evaluate
    from transfer.result_reuse import TargetResultReuse

    results = []
    reuse = TargetResultReuse()
    with cpu_thread_budget(1):
        dataset = load_epoch_dataset(jobs[0]["target_cache"], require_labels=True, validation="attested")
        for job in jobs:
            if job["target_cache"] != jobs[0]["target_cache"]:
                raise ValueError("outer worker group must use one attested target cache")
            result_path = Path(job["output"])
            with gpu_task_slot(job.get("gpu_slots")):
                try:
                    evaluate(SimpleNamespace(plan=Path(job["plan"]), subjects=",".join(job["subjects"]),
                        output=result_path, device=job["device"], cpu_threads=1, stage=None,
                        result_reuse=reuse, target_dataset=dataset))
                finally:
                    release_source_worker(job["device"])
            record = json.loads((result_path / "record.json").read_text())
            results.append({"output": str(result_path), "status": record["status"],
                "completed_cells": sum(cell["status"] == "completed" for cell in record["cells"]),
                "failed_cells": sum(cell["status"] != "completed" for cell in record["cells"]),
                "reused_cells": sum("prediction_reuse" in cell for cell in record["cells"])})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=("S1", "E2", "S2"), default=["S1", "E2", "S2"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluate-outer", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, help="Total CPU budget; default uses the process quota")
    parser.add_argument("--source-workers", type=int, help="Independent source groups; default automatic")
    args = parser.parse_args(argv)
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if args.cpu_threads is not None:
        plan["cpu_budget"] = args.cpu_threads
    if args.source_workers is not None:
        plan["source_workers"] = args.source_workers
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

    def resolve_path(value):
        path = Path(value)
        return str(path if path.is_absolute() else (plan_path.parent / path).resolve())

    for key in ("source_snapshot_manifest", "evaluation_snapshot_manifest", "batch_rule_file"):
        if key in plan:
            plan[key] = resolve_path(plan[key])
    for outer in plan["outer_folds"]:
        for key in ("source_cache", "target_cache"):
            if key in outer:
                outer[key] = resolve_path(outer[key])
        for target in outer.get("targets", []):
            target["cache"] = resolve_path(target["cache"])
    with cpu_thread_budget(1):
        report = execute_source_chain(plan, output=args.output,
            dry_run=args.dry_run, device=args.device, stages=tuple(args.stages), evaluate_outer=args.evaluate_outer)
    print(json.dumps({key: report[key] for key in ("task_count", "optimizer_updates", "trial_draws", "dry_run")}))


if __name__ == "__main__":
    main()
