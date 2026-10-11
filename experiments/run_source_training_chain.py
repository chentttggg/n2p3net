"""Execute or preflight the S1/E2/S2 source chain from an explicit local plan."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


from research.contracts import TrainingRunContract, semantic_sha256  # noqa: E402
from research.execution import ExpectedSubjectError, atomic_write_json  # noqa: E402


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _model_state_digest(trunk):
    import torch

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
    device, evidence_scope, gpu_slots=None, architecture=None,
    runtime_preload_budget_bytes=None,
    runtime_input_staging_gate=None,
    runtime_host_process_budget_bytes=None,
    source_program_config=None,
):
    """Reuse exact source training inputs while binding each consumer's own contract."""
    import numpy as np
    import torch

    from models.n2p3net import DEFAULT_N2P3_ARCHITECTURE
    from research.source_execution import SharedTrajectoryRegistry
    from research.source_selection import source_fit_ledger, source_identity_axes

    architecture = DEFAULT_N2P3_ARCHITECTURE if architecture is None else architecture

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
        "architecture_configuration": asdict(architecture),
        "source_snapshot_sha256": source_snapshot_sha256, "evidence_scope": evidence_scope,
        "initialization": "scratch_N2P3NetBaseline_fixed_source_seed",
    }
    if config.global_early_stopping is not None:
        # The selector's cache/row partition is producer evidence, not a
        # consumer label that can be rebound to another fitted trajectory.
        signature_record["selection_input_binding"] = {
            "source_cache_sha256": source_cache_sha256,
            "full_fit_ledger_digest": semantic_sha256(fit_ledger)}
    if source_program_config is not None:
        signature_record["source_program_config"] = source_program_config
    signature = semantic_sha256(signature_record)
    arguments = dict(dataset=dataset, rows=rows, config=config, destination=destination, task_id=task_id,
        source_cache_sha256=source_cache_sha256, source_snapshot_sha256=source_snapshot_sha256,
        excluded_source_datasets=excluded_source_datasets, device=device, evidence_scope=evidence_scope,
        signature=signature, signature_record=signature_record, fit_ledger=fit_ledger, gpu_slots=gpu_slots,
        architecture=architecture)
    arguments["runtime_preload_budget_bytes"] = runtime_preload_budget_bytes
    arguments["runtime_input_staging_gate"] = runtime_input_staging_gate
    arguments["runtime_host_process_budget_bytes"] = runtime_host_process_budget_bytes
    arguments["source_program_config"] = source_program_config
    if isinstance(registry, SharedTrajectoryRegistry):
        with registry.transaction(signature) as local:
            return _materialize_source_trajectory(registry=local, **arguments)
    return _materialize_source_trajectory(registry=registry, **arguments)


def _materialize_source_trajectory(*, registry, dataset, rows, config, destination, task_id,
        source_cache_sha256, source_snapshot_sha256, excluded_source_datasets, device, evidence_scope,
        signature, signature_record, fit_ledger, gpu_slots, architecture, runtime_preload_budget_bytes=None,
        runtime_input_staging_gate=None, runtime_host_process_budget_bytes=None, source_program_config=None):
    from contextlib import nullcontext

    import torch

    from data.study import dataset_study_membership
    from research.source_execution import (
        current_input_admission,
        gpu_task_slot,
        release_source_worker,
        save_source_checkpoint,
    )
    from research.source_selection import fit_source_trajectory
    from transfer.checkpoint import checkpoint_training_contract, load_checkpoint_payload

    selecting = config.global_early_stopping is not None
    requested_updates = [] if selecting else sorted(set(config.snapshot_updates) | {int(config.fixed_updates)})
    existing = next((entry for entry in registry.get(signature, [])
                     if set(requested_updates) <= set(entry["checkpoints"])), None)
    if selecting and existing is not None:
        requested_updates = sorted(existing["checkpoints"])
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
                    architecture=architecture,
                    runtime_preload_budget_bytes=runtime_preload_budget_bytes,
                    runtime_input_staging_gate=runtime_input_staging_gate,
                    runtime_host_process_budget_bytes=runtime_host_process_budget_bytes,
                    runtime_progress_directory=destination / "progress",
                    source_program_config=source_program_config,
                )
                physical_updates = int(model.last_history["optimizer_updates"])
                physical_draws = int(model.last_history["trial_draws"])
                for update, payload in payloads.items():
                    path = destination / f"source.u{update}.pt"
                    save_source_checkpoint(path, payload)
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
            from research.source_selection import source_architecture_record
            if semantic_sha256(original.architecture) != semantic_sha256(source_architecture_record(dataset, architecture)):
                raise ValueError("cached source checkpoint has a different model architecture")
            if set(original.training_participant_keys) != set(fit_ledger["training_participant_keys"]):
                raise ValueError("cached source checkpoint has different training participants")
            origin_record = {**origin, "training_contract_digest": original.digest(),
                             "source_snapshot_sha256": original.source_snapshot_sha256}
            reuse["origin_checkpoints"][str(update)] = origin_record
            if selecting:
                if (original.source_cache_sha256 != source_cache_sha256
                        or original.validation["fit_ledger"] != fit_ledger):
                    raise ValueError("selected trajectory reuse must preserve its original source producer binding")
                path = destination / f"source.u{update}.pt"
                admission = current_input_admission()
                with (admission.artifact_writer(path, Path(origin["path"]).stat().st_size)
                      if admission is not None else nullcontext()):
                    shutil.copyfile(origin["path"], path)
                checkpoints[update] = str(path.resolve())
                continue
            provenance = {"schema": reuse["schema"], "training_signature": signature,
                          "origin_checkpoint": origin_record, "physical_fit_performed": False,
                          "physical_optimizer_updates": 0, "physical_trial_draws": 0}
            contract = TrainingRunContract(
                source_cache_sha256=source_cache_sha256,
                source_identity_digest=dataset.identity_table.digest(),
                source_snapshot_sha256=source_snapshot_sha256,
                architecture=original.architecture, preprocessing=original.preprocessing,
                optimizer={**original.optimizer, "config": original.optimizer["config"] if selecting else asdict(config), "actual_updates": update,
                           "physical_optimizer_updates": 0, "physical_trial_draws": 0},
                validation={**original.validation, "strategy": "exact_fitted_source_trajectory_reuse", "fit_ledger": fit_ledger,
                            "evidence_scope": evidence_scope, "source_training_reuse": provenance},
                objective={**original.objective, "source_study_membership": membership},
                seed=config.seed, training_participant_keys=tuple(fit_ledger["training_participant_keys"]),
                holdout_participant_keys=(),
            )
            path = destination / f"source.u{update}.pt"
            save_source_checkpoint(path, {**payload, "source_cache_sha256": source_cache_sha256,
                "training_identity_ledger": fit_ledger["training_identity_ledger"],
                "training_identity_ledger_digest": fit_ledger["training_identity_ledger_digest"],
                "training_contract": contract.record(), "training_contract_digest": contract.digest(),
                "source_study_membership": membership, "source_fit_ledger": fit_ledger,
                "source_training_reuse": provenance,
                "runtime": {"physical_fit_performed": False, "origin_runtime": payload.get("runtime")},
            })
            checkpoints[update] = str(path.resolve())
    result = {"checkpoints": checkpoints, "source_training_reuse": reuse, "fit_ledger": fit_ledger,
              "physical_optimizer_updates": physical_updates, "physical_trial_draws": physical_draws}
    atomic_write_json(destination / "trajectory_record.json", result)
    return result


def _chronological_rows(dataset, protocols):
    import numpy as np

    from research.source_selection import source_identity_axes

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
    import numpy as np

    from research.source_selection import source_identity_axes
    from transfer.binary import dataset_calibration_splits, matched_split_digests

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
                        "budget_adjustments": adjustments, "test_budget": split.test_budget.record(),
                        **matched_split_digests(train, test)}
                if legal:
                    requested.append(cell)
                else:
                    failures.append({**cell, "reason": "missing_class_or_insufficient_fit_rows"})
        except ValueError as error:
            failures.append({"domain": domain, "subject": subject, "participant_key": participant,
                             "requested_budgets": list(budgets), "reason": str(error)})
    return requested, failures


def _score_checkpoint(path, dataset, cohort, *, cache_sha256, snapshot_sha256, recipe, output, device, evidence_scope):
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from research.comparison import evaluation_input_identity
    from research.evaluation import build_evaluation_run_contract
    from transfer.binary import matched_input_digests
    from transfer.calibration_candidates import fit_calibration_candidate
    from transfer.checkpoint import PreparedTransferCheckpoint, predict_n2p3_checkpoint

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
        test = np.asarray(cell["test_rows"], int)
        outcome.update(matched_input_digests(dataset.X, dataset.y, test, observed, dataset.preprocessing))
        outcome["comparison_identity"] = evaluation_input_identity(dataset, cell["train_rows"], test)
        try:
            if preparation_error is not None:
                raise ValueError(preparation_error)
            trunk = prepared.trunk_for_target(cell["subject"])
            source_state_digest = _model_state_digest(trunk)
            outcome["updates"] = int(payload["source_fixed_updates"])
            training = np.zeros(dataset.n_epochs, bool)
            training[cell["train_rows"]] = True
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
    import torch

    global _E2_DATASET
    torch.set_num_threads(1)
    _E2_DATASET = dataset


def _score_e2_worker(arguments):
    from research.source_execution import gpu_task_slot, release_source_worker

    path, cohort, options = arguments
    slots = options.pop("gpu_slots", None)
    with gpu_task_slot(slots):
        try:
            return _score_checkpoint(path, _E2_DATASET, cohort, **options)
        finally:
            release_source_worker(options["device"])


def _score_e2_candidates(dataset, path, cohort, candidates, *, workers, progress_path, checkpoint_paths=None,
                         gpu_slots=None, memory_admission=None, **options):
    from research.source_execution import gpu_task_slot, release_source_worker
    from train.runtime import available_cpu_threads

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


def _independent_source_requests(plan):
    pairs = plan.get("seed_pairs") or list(zip(plan["source_seeds"], plan["target_seeds"], strict=True))
    return [{"key": f"{outer['name']}/six/seed{source_seed}/P2_source_ce",
             "outer": outer["name"], "outer_target_study": outer["target_study"],
             "source_studies": outer["source_studies"], "source_seed": source_seed,
             "target_seed": target_seed, "status": "pending"}
            for outer in plan["outer_folds"] for source_seed, target_seed in pairs]


def _finish_independent_source_report(report, output):
    """Retain every requested procedure, including unavailable source inputs."""
    tasks = {task["key"]: task for task in report["tasks"]}
    for request in report["requested_source_tasks"]:
        if request["key"] in tasks:
            task = tasks[request["key"]]
            if request.get("artifact_admission_status") != "rejected" and (
                    task["status"] in {"completed", "fit_failure"} or request["status"] == "pending"):
                request["status"] = task["status"]
            if task.get("failure") is not None:
                request["failure"] = task["failure"]
    report.update(requested_source_task_count=len(report["requested_source_tasks"]),
        source_failure_count=sum(request["status"] not in {"completed", "planned", "pending"}
                                 for request in report["requested_source_tasks"]),
        completed_physical_fit_count=sum(task.get("physical_fit_count", 0) for task in tasks.values()),
        completed_logical_fit_count=sum(task.get("logical_fit_count", 0) for task in tasks.values()))
    report["unmeasured_source_fit_tasks"] = [task["key"] for task in tasks.values()
                                            if task["status"] in {"fit_failure", "running"}]
    if not report["dry_run"]:
        report.update(optimizer_updates=sum(task["optimizer_updates"] for task in tasks.values()
                                            if task["status"] == "completed"),
            trial_draws=sum(task["trial_draws"] for task in tasks.values() if task["status"] == "completed"),
            optimizer_exposure_scope="completed_source_procedures",
            physical_optimizer_exposure_scope="returned_fit_attempts")
    atomic_write_json(output / "source_seed_catalog.json", {
        "schema": "n2p3_source_seed_catalog/1", "source_program": report["source_program_record"],
        "source_snapshot": report["source_snapshot"], "source_evidence_scope": report["evidence_scope"],
        "entries": report["source_seed_catalog"], "requested": report["requested_source_tasks"],
        "optimizer_exposure_scope": report.get("optimizer_exposure_scope", "planned_upper_bound"),
        "physical_optimizer_exposure_scope": report.get("physical_optimizer_exposure_scope", "no_execution"),
        "unmeasured_source_fit_tasks": report["unmeasured_source_fit_tasks"],
        "physical_fit_count": report["completed_physical_fit_count"],
        "logical_fit_count": report["completed_logical_fit_count"]})


def _save_chain_report(output, report):
    import torch

    atomic_write_json(output / "source_chain_record.json", report)
    tasks = report.get("tasks", [])
    completed = {task["key"] for task in tasks if task["status"] in {"completed", "fit_failure"}}
    active = next((task for task in reversed(tasks) if task["status"] == "running"), None)
    remaining = [request for request in report.get("requested_source_tasks", [])
                 if request["key"] not in completed]
    atomic_write_json(output / "worker_progress.json", {
        "pid": os.getpid(), "updated_unix": time.time(), "task_count": len(tasks),
        "completed_tasks": sum(task["status"] == "completed" for task in tasks),
        "failed_tasks": sum(task["status"] == "fit_failure" for task in tasks),
        "resumed_tasks": sum(task.get("source_training_reuse", {}).get("kind") == "completed_previous_run" for task in tasks),
        "physical_optimizer_updates": sum(task.get("physical_optimizer_updates", 0) for task in tasks),
        "native_threads": torch.get_num_threads(),
        "source_input_admitted": report.get("source_input_admitted", False),
        "active_source_seed": None if active is None else active["config"]["seed"],
        "remaining_source_seed_tasks": [{"key": request["key"], "source_seed": request["source_seed"]}
                                         for request in remaining],
        "source_seed_execution_topology": "sequential_in_one_cache_group",
        "last_task": tasks[-1]["key"] if tasks else None,
        "completed_selections": len(report.get("selection", [])),
    })


def source_execution_groups(plan, stages):
    from research.source_selection import SOURCE_CHAIN_LAYOUTS

    groups = []
    names = [outer["name"] for outer in plan["outer_folds"]]
    layouts = tuple(plan.get("source_layouts", ("six",) if set(stages) in ({"P2_source"}, {"unified_source"}) else ("six", "common_three")))
    if len(names) != len(set(names)) or len(layouts) != len(set(layouts)):
        raise ValueError("source groups require unique outer names and layouts")
    for outer in plan["outer_folds"]:
        for layout in layouts:
            if layout not in SOURCE_CHAIN_LAYOUTS:
                raise ValueError("unsupported source layout")
            if layout == "six" and not {"P2_source", "unified_source"}.intersection(stages) and (
                "S1" not in stages or outer.get("family_exclusion", False)
            ):
                continue
            groups.append((outer, layout))
    return groups


def _source_group_worker(arguments):
    from contextlib import nullcontext

    from research.source_execution import install_input_admission
    from research.worker_diagnostics import diagnostic_task, worker_diagnostic_phase

    plan, output, device, stages = arguments
    install_input_admission(plan.get("_source_input_admission"))
    with diagnostic_task(output, {"phase_kind": "source_group", "outer": plan["outer_folds"][0]["name"],
            "stages": list(stages), "device": device, "native_threads": plan.get("_native_threads_per_worker", 1)}):
        worker_diagnostic_phase("import_training_runtime")
        from train.runtime import configure_spawned_worker_threads, cpu_thread_budget

        native_threads = plan.get("_native_threads_per_worker", 1)
        configure_spawned_worker_threads(native_threads)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(native_threads)
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        snapshot_directory = plan.get("verified_cache_snapshot_directory")
        if snapshot_directory is not None:
            directory = Path(snapshot_directory).resolve()
            directory.mkdir(parents=True, exist_ok=True)
        admission = nullcontext() if plan.get("_phase_reserved") else plan["_memory_admission"].source_group()
        # Resident EEG RAM is not an output filesystem claim. Cache, JSON,
        # checkpoint and progress writers admit their own actual output bytes.
        with cpu_thread_budget(native_threads), admission:
            worker_diagnostic_phase("source_input_and_memory_admission")
            if plan.get("_prepared_source_inputs") is not None:
                plan = _prepare_source_group_inputs(plan, Path(output))
            worker_diagnostic_phase("execute_source_chain")
            execute_source_chain(plan, output=Path(output), dry_run=False, device=device,
                                 stages=stages, evaluate_outer=False)
            snapshot = json.loads((Path(output) / "source_chain_record.json").read_text(encoding="utf-8"))["source_snapshot"]
            _assert_source_bytes(plan["source_snapshot_manifest"], snapshot)
    return str(Path(output) / "source_chain_record.json")


def _prepare_source_group_inputs(plan, output):
    """Materialize one complete outer view, then let all seeds share its load."""
    from data.epochs import _read_epoch_cache_record
    from experiments.prepare_multidomain_source import main as merge
    from research.source_execution import SourceInputPhases
    from research.source_resume import SourceChainResume

    phases = SourceInputPhases(output / "source_input_stages.json")
    plan = {**plan, "_source_input_phases": phases}

    inputs = plan["_prepared_source_inputs"]
    outer = dict(plan["outer_folds"][0])
    bindings = {study: {field: record[field] for field in (
        "manifest_sha256", "cache_sha256", "cache_record_sha256", "preparation_method")}
        for study, record in inputs.items()}
    previous = SourceChainResume(plan.get("resume_source_run")).prepared_source_view(outer["name"], bindings)
    if previous is not None:
        outer.update(previous)
    elif len(inputs) == 1:
        record = next(iter(inputs.values()))
        outer.update(source_cache=record["cache"], expected_source_cache_sha256=record["cache_sha256"])
    else:
        view_root = (Path(plan["source_view_working_directory"]) if plan.get("source_view_working_directory")
                     else output.parent.parent / "source_views")
        cache = view_root / output.name / "source.npz"
        argv = ["--output", str(cache), "--target-channels", "Fz,P3,Pz,P4,PO8,Oz",
            "--event-contract", "binary_evidence", "--reference-mode", "measured_channel",
            "--reference-channel", "Cz",
            "--missing-channel-policy", "mask", "--source-workers",
            str(plan.get("source_merge_workers", plan["_native_threads_per_worker"])), "--uncompressed",
            "--cpu-threads", str(plan["_native_threads_per_worker"])]
        for study in outer["source_studies"]:
            argv.extend(("--source", f"{study}={inputs[study]['cache']}"))
        if plan.get("verified_cache_snapshot_directory"):
            argv.extend(("--verified-snapshot-directory", str(plan["verified_cache_snapshot_directory"])))
        if plan.get("_source_worker_memory_bytes"):
            argv.extend(("--source-memory-budget-bytes", str(plan["_source_worker_memory_bytes"])))
        with phases.phase("whole_fold_merge_and_write"):
            production = merge(argv, retain_production=True, phases=phases)
        metadata, attestation = _read_epoch_cache_record(cache)
        loaded = {record["namespace"]: record["cache_sha256"] for record in metadata["provenance"]["sources"]}
        if loaded != {study: record["cache_sha256"] for study, record in inputs.items()}:
            raise ValueError("source merge consumed different bytes from the admitted prepared inventory")
        outer.update(source_cache=str(cache.resolve()), expected_source_cache_sha256=attestation["sha256"])
        plan = {**plan, "_resident_source_production": production}
    outer["source_input_bindings"] = bindings
    return {**plan, "outer_folds": [outer]}


def _source_preparation_worker(arguments):
    from research.worker_diagnostics import diagnostic_task, worker_diagnostic_phase

    manifest, output, profile, record_workers, staging = arguments
    from research.source_execution import install_input_admission

    install_input_admission(staging)
    with diagnostic_task(output, {"phase_kind": "source_preparation", "manifest": manifest,
                                 "profile": profile, "record_workers": record_workers}):
        worker_diagnostic_phase("import_preparation_runtime")
        from experiments.prepare_source_records import main as prepare
        from train.runtime import cpu_thread_budget

        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        with cpu_thread_budget(1):
            worker_diagnostic_phase("prepare_record_workers")
            prepare(["--manifest", manifest, "--output", output, "--preprocessing-config", profile,
                     "--workers", str(record_workers)])
        worker_diagnostic_phase("read_preparation_result")
        path = Path(output) / "preparation.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        return {"record": str(path), "status": record["status"], "record_workers": record["workers"],
                "controller_pid": os.getpid(), "native_threads_per_record_worker": 1,
                "cache_sha256": record["cache_sha256"]}


def _source_input_wait_worker(arguments):
    """Wait for one explicitly external preparation, never shrink its library."""
    from data.epochs import parse_preprocessing_spec
    from data.source_admission import PreparedSourceInventory

    directory, study, preprocessing, manifest_hashes, timeout = arguments
    inventory = PreparedSourceInventory(directory, preprocessing=parse_preprocessing_spec(preprocessing),
                                        expected_manifest_sha256=manifest_hashes)
    deadline = time.monotonic() + timeout
    while True:
        state = inventory.inspect([study])
        if study in state["ready"]:
            return {"study": study, "cache_sha256": state["ready"][study]["cache_sha256"]}
        if study in state["failed"]:
            raise ValueError(state["failed"][study])
        if time.monotonic() >= deadline:
            raise ValueError(f"preparation wait expired: {state['pending'][study]}")
        time.sleep(min(1., max(.01, deadline - time.monotonic())))


def _source_input_failure_worker(arguments):
    raise ValueError(arguments)


def _assert_source_bytes(snapshot_path, snapshot):
    """The archive was verified at ingress; compare actual runtime files at egress."""
    if _sha256(snapshot_path) != snapshot["manifest_sha256"]:
        raise ValueError("source snapshot manifest changed during its declared execution")
    files = json.loads(Path(snapshot_path).read_bytes())["files"]
    for name, expected in files.items():
        if (name.startswith(("src/", "experiments/")) and name.endswith(".py")) or name in {"pyproject.toml", "uv.lock"}:
            if _sha256(ROOT / name) != expected:
                raise ValueError(f"executed source changed during the source programme: {name}")


def _admit_source_group(path, *, child_plan, program, snapshot):
    """Admit returned producer bytes before releasing a source dependency."""
    from baselines.deep import DeepConfig
    from models.n2p3net import DEFAULT_N2P3_ARCHITECTURE, DEFAULT_N2P3_POOLING_MODE, N2P3Net
    from research.source_catalog import SourceSeedCatalog
    from train.early_stopping import GlobalEarlyStoppingPolicy
    from transfer.checkpoint import checkpoint_training_contract, load_checkpoint_payload

    _assert_source_bytes(child_plan["source_snapshot_manifest"], snapshot)
    child = json.loads(Path(path).read_text(encoding="utf-8"))
    if (child["source_snapshot"]["archive_sha256"] != snapshot["archive_sha256"]
            or child["source_program_record"] != program
            or child["evidence_scope"] != child_plan.get("evidence_scope", "development")
            or not child.get("source_input_admitted") or child["preflight_failures"]
            or not child["tasks"] or any(task["status"] != "completed" for task in child["tasks"])):
        raise ValueError("source cache-group report does not prove its complete declared programme")
    outer = child_plan["outer_folds"][0]
    if program["name"] != "independent_selection_refit":
        for task in child["tasks"]:
            for checkpoint in task["checkpoints"].values():
                payload = load_checkpoint_payload(checkpoint)
                contract = checkpoint_training_contract(payload)
                if (contract.source_cache_sha256 != task["source_cache_sha256"]
                        or contract.seed != task["config"]["seed"]
                        or contract.validation["fit_ledger"] != task["fit_ledger"]):
                    raise ValueError("source checkpoint differs from its completed task input/fit/seed binding")
        return child
    wanted = {request["key"] for request in _independent_source_requests(child_plan)}
    if {task["key"] for task in child["tasks"]} != wanted or len(child["tasks"]) != len(wanted):
        raise ValueError("independent source group did not complete every declared source seed")
    entries = child["source_seed_catalog"]
    seeds = {pair[0] for pair in program["seed_pairs"]}
    if (len(entries) != len(seeds)
            or {(entry["outer_target_study"], entry["source_seed"]) for entry in entries}
               != {(outer["target_study"], seed) for seed in seeds}):
        raise ValueError("independent source consumers require every declared source-seed catalog entry")
    scope = "software_correctness" if child["evidence_scope"] == "software_correctness" else "public_development"
    catalog = SourceSeedCatalog(entries, base_directory=Path(path).parent, evidence_scope=scope,
                                expected_source_config=program["source_config"])
    by_seed = {task["config"]["seed"]: task for task in child["tasks"]}
    if set(by_seed) != seeds or len(by_seed) != len(child["tasks"]):
        raise ValueError("independent task configuration must preserve its declared source seeds")
    for record in catalog.record():
        task = by_seed[record["source_seed"]]
        if (set(record["producer"]["source_studies"]) != set(outer["source_studies"])
                or task["source_producer"] != record["producer"]
                or task["source_input"]["preprocessing"] != program["preprocessing"]
                or task["key"] != f"{outer['name']}/six/seed{record['source_seed']}/P2_source_ce"
                or task["config"]["global_early_stopping"] != program["stopping"]
                or task["source_input_bindings"] != outer.get("source_input_bindings")
                   and child_plan.get("_prepared_source_inputs") is None):
            raise ValueError("source producer differs from its whole-library/input/programme declaration")
        if child_plan.get("_prepared_source_inputs") is not None:
            expected = {study: {field: value[field] for field in (
                "manifest_sha256", "cache_sha256", "cache_record_sha256", "preparation_method")}
                for study, value in child_plan["_prepared_source_inputs"].items()}
            if task["source_input_bindings"] != expected:
                raise ValueError("source fit consumed a different admitted prepared-source inventory")
        entry = next(item for item in entries if item["source_seed"] == record["source_seed"])
        payload = load_checkpoint_payload(entry["path"])
        config = DeepConfig(global_early_stopping=GlobalEarlyStoppingPolicy(**program["stopping"]),
            batch_size=512, seed=record["source_seed"], val_group_frac=None, sampling="uniform_rows",
            compile_mode=child_plan.get("compile_mode", "reduce-overhead"),
            source_objective=None, **record["producer"]["source_program_config"]["resolved"])
        refit_config = replace(config, global_early_stopping=None,
                               val_group_frac=None,
                               fixed_updates=payload["source_epoch_selection"]["refit_updates"])
        contract = checkpoint_training_contract(payload)
        architecture = N2P3Net.default_architecture_record(pooling_mode=DEFAULT_N2P3_POOLING_MODE,
            tmin_s=program["preprocessing"]["tmin_ms"] / 1000,
            sfreq=program["preprocessing"]["sfreq"], n_times=task["source_input"]["n_times"],
            architecture=DEFAULT_N2P3_ARCHITECTURE)
        if (payload["source_cache_sha256"] != task["source_cache_sha256"]
                or payload["source_fit_ledger"] != task["fit_ledger"]
                or payload["global_early_stopping"] != program["stopping"]
                or payload["input_preprocessing"] != task["source_input"]["preprocessing"]
                or payload["input_channel_names"] != task["source_input"]["channel_names"]
                or payload["n_times"] != task["source_input"]["n_times"]
                or semantic_sha256(task["config"]) != semantic_sha256(asdict(config))
                or semantic_sha256(contract.optimizer["config"]) != semantic_sha256(asdict(refit_config))
                or semantic_sha256(task["architecture_configuration"]) != semantic_sha256(asdict(DEFAULT_N2P3_ARCHITECTURE))
                or contract.architecture != architecture or payload["architecture"] != architecture):
            raise ValueError("source catalog does not bind the completed full-source input ledger")
    child["source_seed_catalog"] = [{**entry, "path": str((Path(path).parent / entry["path"]).resolve())}
                                     for entry in entries]
    return child


def _source_legacy_outer_preparation_worker(arguments):
    report, plan, output, device = arguments
    groups = _prepare_outer_plans(report, plan, Path(output), device=device, execute=False)
    return {"record": str(Path(output) / "source_chain_record.json"), "groups": list(groups.values())}


@contextmanager
def _source_parent_input_owner(admission):
    from research.source_execution import current_input_admission, install_input_admission

    previous = current_input_admission()
    install_input_admission(admission)
    try:
        yield
    finally:
        install_input_admission(previous)


def _execute_source_groups(plan, *, output, device, stages, evaluate_outer, groups,
                           additional_tasks=(), on_additional_finished=None, on_additional_complete=None):
    from data.epochs import parse_preprocessing_spec
    from data.source_admission import PreparedSourceInventory, bind_source_inputs
    from research.cloud_execution import CloudExecutor, CloudTask, PhaseResources
    from research.evaluation import verified_executed_snapshot
    from research.source_execution import (
        HostMemoryAdmission,
        SourceInputAdmission,
        resolve_source_resources,
        with_cloud_resource_floor,
    )
    from research.source_selection import resolve_source_program
    from transfer.target_experiments import resolve_target_batch_rule

    program = resolve_source_program(plan, stages)
    resources = resolve_source_resources(with_cloud_resource_floor(plan, device), len(groups), device)
    snapshot = verified_executed_snapshot(plan["source_snapshot_manifest"], root=ROOT)
    independent = program["name"] == "independent_selection_refit"
    prepared_directory = plan.get("prepared_source_directory")
    manifests = plan.get("source_preparation_manifests", {})
    input_wait = plan.get("wait_for_preparation_seconds", 0.)
    if not isinstance(input_wait, (int, float)) or not 0 <= input_wait < float("inf"):
        raise ValueError("preparation wait must be finite and nonnegative")
    if (prepared_directory or manifests) and not independent:
        raise ValueError("prepared source DAG requires the explicit independent source program")
    if manifests and not prepared_directory:
        raise ValueError("source preparation requires an explicit prepared_source_directory")
    required_studies = {study for outer, _ in groups for study in outer.get("source_studies", ())}
    inventory = None
    manifest_hashes = {}
    if prepared_directory:
        if any("source_cache" in outer for outer, _ in groups):
            raise ValueError("declare prepared source studies or fixed source caches, never both")
        from data.source_parsers import validate_source_manifest
        from experiments.prepare_source_records import _manifest_preprocessing

        if not isinstance(manifests, dict) or not set(manifests) <= required_studies:
            raise ValueError("preparation manifests must identify declared source dependencies")
        spec = parse_preprocessing_spec(program["preprocessing"])
        for study, path in manifests.items():
            declaration = json.loads(Path(path).read_text(encoding="utf-8"))
            validate_source_manifest(declaration)
            if declaration["study"] != study or _manifest_preprocessing(declaration, None) != spec:
                raise ValueError("source preparation manifest must explicitly declare the independent program's profile")
            manifest_hashes[study] = _sha256(path)
        inventory = PreparedSourceInventory(prepared_directory, preprocessing=spec,
                                            expected_manifest_sha256=manifest_hashes)
    declared_consumers = plan.get("ready_consumers", [])
    group_by_outer = {outer["name"]: index for index, (outer, _) in enumerate(groups)}
    names = set()
    for consumer in declared_consumers:
        if (not independent or set(consumer) != {"kind", "name", "outer", "plan", "output"}
                or consumer["kind"] not in {"binary", "candidate"}
                or not consumer["name"] or consumer["name"] in names
                or consumer["outer"] not in group_by_outer):
            raise ValueError("ready_consumers require unique explicit current binary/candidate requests and an independent outer")
        names.add(consumer["name"])
        declaration = json.loads(Path(consumer["plan"]).read_bytes())
        if (declaration.get("seed_pairs") != program["seed_pairs"]
                or declaration.get("expected_source_config") != program["source_config"]):
            raise ValueError("ready consumer must preserve the declared complete source/target pairs and source configuration")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("parallel source chain requires a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    batch, rule = resolve_target_batch_rule(plan)
    plan = {**plan, "target_batch_size": batch, "batch_rule": rule,
            "cpu_budget": resources["cpu_budget"], "target_workers": resources["target_workers"],
            "_target_native_threads_per_worker": resources["native_threads_per_worker"]}
    report = {"schema": "n2p3_source_training_chain_run/1", "dry_run": False,
              "status": "running", "source_snapshot": snapshot, "resource_policy": resources,
              "source_program": program["name"], "source_program_record": program,
              "evidence_scope": plan.get("evidence_scope", "development"), "batch_rule": rule,
              "source_seeds": plan.get("source_seeds", [20260906]),
              "target_seeds": plan.get("target_seeds", [20260906]), "groups": [],
              "preparation": [], "input_waits": [], "resource_observations": [], "consumers": [],
              "tasks": [], "selection": [], "preflight_failures": [], "outer_evaluation_jobs": []}
    if independent:
        report.update(source_seed_catalog=[], requested_source_tasks=_independent_source_requests(plan))
    group_ids = {}
    for index, (outer, layout) in enumerate(groups):
        task_id = f"source-{index:02d}-{semantic_sha256([outer['name'], layout])[:12]}"
        group_ids[index] = task_id
        directory = output / "groups" / f"{index:02d}_{outer['name']}_{layout}"
        report["groups"].append({"outer": outer["name"], "layout": layout, "task_id": task_id,
            "directory": str(directory), "status": "pending", "source_studies": outer.get("source_studies")})
    profile_path = output / "source_preprocessing.json"
    if inventory is not None:
        atomic_write_json(profile_path, program["preprocessing"])
    diagnostic_identity = {"source_snapshot": snapshot, "actual_source_root": str(ROOT),
        "resolved_plan_sha256": semantic_sha256({key: value for key, value in plan.items() if not key.startswith("_")}),
        "stages": list(stages), "device": device, "resources": resources}
    diagnostic_identity["actual_run_sha256"] = semantic_sha256(diagnostic_identity)
    diagnostics_directory = output / "worker_diagnostics"
    report["worker_diagnostics"] = {"directory": str(diagnostics_directory), "identity": diagnostic_identity}
    _save_chain_report(output, report)
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        input_admission = SourceInputAdmission(manager, capacity_bytes=resources["shared_staging_memory_bytes"],
                                               load_workers=int(plan.get("source_load_workers", 1)))
        with _source_parent_input_owner(input_admission):
            memory_admission = HostMemoryAdmission(manager, resources, input_admission=input_admission)
            phases = {phase: PhaseResources(phase, resources["native_threads_per_worker"],
                resources["source_worker_memory_bytes"] if phase in {"source", "preparation"} else resources["target_worker_memory_bytes"],
                resources["native_threads_per_worker"], device if phase != "preparation" else "cpu")
                for phase in ("source", "preparation", "calibration")}
            consumers, consumer_tasks = {}, {}
            for declaration in declared_consumers:
                index = group_by_outer[declaration["outer"]]
                if declaration["kind"] == "binary":
                    from experiments.run_erp_comparator import build_binary_cloud_tasks as builder
                else:
                    from experiments.run_candidate_ensemble_study import (
                        build_candidate_cloud_tasks as builder,
                    )
                programme = builder(declaration["plan"], declaration["output"], resources,
                    task_prefix=f"consumer-{declaration['name']}", dependencies=(group_ids[index],),
                    input_admission=input_admission)
                consumers[declaration["name"]] = (index, programme)
                report["consumers"].append({**declaration, "status": "waiting_source",
                    "source_dependency": group_ids[index], "requested_tasks": [task.task_id for task in programme.tasks]})
            _save_chain_report(output, report)
            state = inventory.inspect(required_studies) if inventory is not None else {"ready": {}, "pending": {}, "failed": {}}
            bindings = bind_source_inputs({}, state["ready"])
            input_nodes, input_results, source_plans, submitted_groups = {}, {}, {}, set()
            tasks = []
            for study in sorted(required_studies - set(state["ready"])) if inventory is not None else ():
                task_id = f"input-{semantic_sha256(study)[:16]}"
                input_nodes[study] = task_id
                if study in state["failed"]:
                    adapter, arguments = "_source_input_failure_worker", state["failed"][study]
                    claim = PhaseResources("preparation", 1, min(64 * 2**20, resources["source_worker_memory_bytes"]), 1)
                    entry = {"study": study, "task_id": task_id, "status": "pending"}
                    report["input_waits"].append(entry)
                elif study in manifests:
                    phase_threads = max(2, resources["native_threads_per_worker"])
                    entry = {"study": study, "task_id": task_id, "status": "pending", "manifest": manifests[study]}
                    report["preparation"].append(entry)
                    if phase_threads > resources["worker_cpu_budget"]:
                        adapter = "_source_input_failure_worker"
                        arguments = "source preparation needs CPU budget for its controller and a record worker alongside the coordinator"
                        claim = PhaseResources("preparation", 1, resources["source_worker_memory_bytes"], 1)
                    else:
                        adapter = "_source_preparation_worker"
                        arguments = (manifests[study], str(Path(prepared_directory) / study), str(profile_path),
                                     phase_threads - 1, input_admission)
                        claim = PhaseResources("preparation", phase_threads, resources["source_worker_memory_bytes"], 1)
                else:
                    adapter = "_source_input_wait_worker"
                    arguments = (prepared_directory, study, program["preprocessing"], manifest_hashes, input_wait)
                    claim = PhaseResources("preparation", 1, min(64 * 2**20, resources["source_worker_memory_bytes"]), 1)
                    entry = {"study": study, "task_id": task_id, "status": "pending"}
                    report["input_waits"].append(entry)
                entry["resource_reservation"] = {"phase": claim.phase, "cpu_threads": claim.cpu_threads,
                                                   "memory_bytes": claim.memory_bytes, "gpu_workers": 0}
                tasks.append(CloudTask(task_id, f"experiments.run_source_training_chain:{adapter}", arguments,
                    str(output / "input_tasks" / task_id), claim))

            def input_status():
                return {"ready": {study: {field: record[field] for field in (
                    "cache_sha256", "manifest_sha256", "n_epochs", "participants")}
                    for study, record in state["ready"].items()}, "pending": state["pending"], "failed": state["failed"]}

            def source_tasks_ready():
                additions = []
                for index, (outer, layout) in enumerate(groups):
                    if index in submitted_groups:
                        continue
                    dependencies = tuple(input_nodes[study] for study in outer.get("source_studies", ()) if study in input_nodes)
                    if any(task_id not in input_results for task_id in dependencies):
                        continue
                    child = {**plan, "outer_folds": [outer], "source_layouts": [layout],
                        "_source_group": True, "_defer_outer_plans": True, "_phase_reserved": True,
                        "_memory_admission": memory_admission,
                        "_native_threads_per_worker": resources["native_threads_per_worker"],
                        "_source_worker_memory_bytes": resources["source_worker_memory_bytes"],
                        "_source_input_admission": input_admission,
                        "_runtime_input_staging_gate": input_admission if input_admission.capacity else None,
                        "_runtime_preload_budget_bytes": resources["resident_memory_budget_bytes"],
                        "_runtime_host_process_budget_bytes": resources["source_worker_memory_bytes"],
                        "shared_registry_directory": str(output / "trajectory_registry"),
                        "target_workers": resources["inner_workers_per_group"]}
                    if inventory is not None and all(task_id not in input_results or input_results[task_id] == "completed"
                                                      for task_id in dependencies):
                        child["_prepared_source_inputs"] = {study: {field: state["ready"][study][field] for field in (
                            "cache", "cache_sha256", "cache_record_sha256", "manifest_sha256", "preparation_method")}
                            for study in outer["source_studies"]}
                    entry = report["groups"][index]
                    source_plans[group_ids[index]] = child
                    entry["resource_reservation"] = {"phase": "source", "cpu_threads": phases["source"].cpu_threads,
                        "memory_bytes": phases["source"].memory_bytes, "gpu_workers": int(str(device).startswith("cuda"))}
                    additions.append(CloudTask(group_ids[index], "experiments.run_source_training_chain:_source_group_worker",
                        (child, entry["directory"], device, stages), entry["directory"], phases["source"], dependencies))
                    submitted_groups.add(index)
                return additions

            def record_failure(outcome, error):
                failure = {"type": type(error).__name__, "message": str(error), "stage": "artifact_admission"}
                if outcome.failure is None:
                    outcome.status, outcome.failure = "failed", failure
                else:
                    outcome.supplementary_failures.append(failure)

            legacy_tasks = {}
            additional_ids = set()
            pipeline_ids = {*group_ids.values(), *input_nodes.values(),
                *(task.task_id for _, programme in consumers.values() for task in programme.tasks)}

            def register_additional(new_tasks):
                new_tasks = tuple(new_tasks)
                ids = [task.task_id for task in new_tasks]
                if (len(set(ids)) != len(ids) or set(ids) & (pipeline_ids | additional_ids | set(legacy_tasks))
                        or any(key.startswith(("consumer-binding-failure-", "outer-preparation-", "outer-evaluation-"))
                               for key in ids)):
                    raise ValueError("additional cloud tasks must have distinct pipeline task identities")
                additional_ids.update(ids)
                report.setdefault("additional_cloud_requests", []).extend({"task_id": task.task_id,
                    "adapter": task.adapter, "resources": asdict(task.resources), "depends_on": list(task.depends_on)}
                    for task in new_tasks)
                return new_tasks

            tasks.extend(register_additional(additional_tasks() if callable(additional_tasks) else additional_tasks))

            def on_finished(outcome):
                nonlocal state, bindings
                additions = []
                if outcome.task_id in input_nodes.values():
                    study = next(study for study, task_id in input_nodes.items() if task_id == outcome.task_id)
                    if outcome.status == "completed":
                        try:
                            current = inventory.inspect(required_studies)
                            if study not in current["ready"]:
                                raise ValueError(f"prepared task returned without an admitted cache: {study}")
                            bindings = bind_source_inputs(bindings, current["ready"])
                            state = current
                        except Exception as error:
                            record_failure(outcome, error)
                    input_results[outcome.task_id] = outcome.status
                    if outcome.status != "completed":
                        state["failed"][study] = outcome.failure
                        state["pending"].pop(study, None)
                    entry = next(entry for entry in report["preparation"] + report["input_waits"]
                                 if entry["task_id"] == outcome.task_id)
                    entry.update(status="completed" if outcome.status == "completed" else "failed", worker=outcome.worker)
                    if outcome.status == "completed" and isinstance(outcome.value, dict):
                        entry.update(outcome.value)
                    if outcome.failure is not None:
                        entry["failure"] = outcome.failure
                    additions.extend(source_tasks_ready())
                elif outcome.task_id in source_plans:
                    entry = next(entry for entry in report["groups"] if entry["task_id"] == outcome.task_id)
                    child = None
                    if outcome.status == "completed":
                        try:
                            expected = Path(entry["directory"]) / "source_chain_record.json"
                            if Path(outcome.value).resolve() != expected.resolve():
                                raise ValueError("source worker returned a different output record")
                            child = _admit_source_group(expected, child_plan=source_plans[outcome.task_id],
                                                       program=program, snapshot=snapshot)
                        except Exception as error:
                            record_failure(outcome, error)
                    entry.update(status="completed" if outcome.status == "completed" else "failed", worker=outcome.worker)
                    if outcome.failure is not None:
                        entry["failure"] = outcome.failure
                    if child is not None and outcome.status == "completed":
                        entry.update(record=str(Path(entry["directory"]) / "source_chain_record.json"),
                                     source_artifact_admission="completed_declared_producer_programme")
                        if independent:
                            entry["accepted_source_seed_catalog"] = child["source_seed_catalog"]
                        if not independent:
                            task_id = f"outer-preparation-{outcome.task_id}"
                            child_output = str(output / "outer_groups" / outcome.task_id)
                            legacy_tasks[task_id] = {"kind": "preparation", "outer": entry["outer"]}
                            additions.append(CloudTask(task_id,
                                "experiments.run_source_training_chain:_source_legacy_outer_preparation_worker",
                                (child, source_plans[outcome.task_id], child_output, device), child_output,
                                phases["preparation"], (outcome.task_id,)))
                    for declaration in report["consumers"]:
                        index, programme = consumers[declaration["name"]]
                        if group_ids[index] != outcome.task_id:
                            continue
                        if outcome.status == "completed":
                            try:
                                ready = programme.bind_source(child["source_seed_catalog"], plan["source_snapshot_manifest"])
                                declaration.update(status="source_admitted", source_seed_catalog=child["source_seed_catalog"],
                                    source_snapshot_manifest=plan["source_snapshot_manifest"])
                            except Exception as error:
                                declaration.update(status="binding_failed", failure={"type": type(error).__name__, "message": str(error)})
                                # The source remains a valid producer when a consumer's own declaration fails.
                                gate_id = f"consumer-binding-failure-{declaration['name']}"
                                additions.append(CloudTask(gate_id,
                                    "experiments.run_source_training_chain:_source_input_failure_worker", str(error),
                                    str(output / "consumer_failures" / gate_id),
                                    PhaseResources("preparation", 1, min(64 * 2**20, resources["source_worker_memory_bytes"]), 1),
                                    (outcome.task_id,)))
                                ready = [replace(task, depends_on=(*task.depends_on, gate_id)) for task in programme.tasks]
                        else:
                            ready = programme.tasks
                            declaration.update(status="source_dependency_failed", source_failure=outcome.failure)
                        for task in ready:
                            consumer_tasks[task.task_id] = programme
                        additions.extend(ready)
                elif outcome.task_id in consumer_tasks:
                    consumer_tasks[outcome.task_id].on_finished(outcome)
                elif outcome.task_id in legacy_tasks:
                    entry = legacy_tasks[outcome.task_id]
                    entry.update(status=outcome.status, failure=outcome.failure)
                    if entry["kind"] == "preparation" and outcome.status == "completed":
                        child = json.loads(Path(outcome.value["record"]).read_text(encoding="utf-8"))
                        report["outer_evaluation_jobs"].extend(child["outer_evaluation_jobs"])
                        if evaluate_outer:
                            for index, jobs in enumerate(outcome.value["groups"]):
                                task_id = f"outer-evaluation-{outcome.task_id}-{index}"
                                legacy_tasks[task_id] = {"kind": "evaluation", "outer": entry["outer"]}
                                additions.append(CloudTask(task_id,
                                    "experiments.run_source_training_chain:_evaluate_outer_group", jobs,
                                    str(output / "outer_execution" / task_id), phases["calibration"], (outcome.task_id,)))
                    elif entry["kind"] == "evaluation" and outcome.status == "completed":
                        report.setdefault("outer_execution", []).extend(outcome.value)
                elif outcome.task_id in additional_ids and on_additional_finished is not None:
                    additions.extend(register_additional(on_additional_finished(outcome) or ()))
                if inventory is not None:
                    atomic_write_json(output / "input_status.json", input_status())
                    atomic_write_json(output / "input_bindings.json", bindings)
                _save_chain_report(output, report)
                return additions

            previous_observation = None

            def on_observation(observation):
                nonlocal previous_observation
                merged = {**observation["resources"], **input_admission.record(),
                    "ready_source_groups": sum(entry["task_id"] in observation["running"] or
                        entry["task_id"] in observation["pending"] and all(input_results.get(input_nodes.get(study), "completed") == "completed"
                            for study in entry["source_studies"] or ()) for entry in report["groups"]),
                    "waiting_ready_source_groups": sum(entry["task_id"] in observation["pending"]
                        and all(input_results.get(input_nodes.get(study), "completed") == "completed"
                            for study in entry["source_studies"] or ()) for entry in report["groups"]),
                    "pending_source_groups": sum(entry["status"] == "pending" for entry in report["groups"]),
                    "active_pipeline_nodes": len(observation["running"]),
                    "running_tasks": observation["running"], "pending_tasks": observation["pending"]}
                if merged != previous_observation:
                    report["resource_observations"].append({"observed_unix": observation["observed_unix"], **merged})
                    previous_observation = merged
                for entry in report["groups"]:
                    if entry["task_id"] in observation["running"]:
                        entry["status"] = "running"
                    progress = Path(entry["directory"]) / "worker_progress.json"
                    if progress.exists():
                        entry["progress"] = json.loads(progress.read_text(encoding="utf-8"))
                _save_chain_report(output, report)

            tasks = source_tasks_ready() + tasks
            if inventory is not None:
                atomic_write_json(output / "input_status.json", input_status())
                atomic_write_json(output / "input_bindings.json", bindings)
            _save_chain_report(output, report)
            executor = CloudExecutor(memory_admission, max_workers=resources["worker_cpu_budget"],
                diagnostic_directory=diagnostics_directory, identity=diagnostic_identity)
            run = executor.run(tasks, on_finished=on_finished, on_observation=on_observation)
            report["worker_diagnostics"] = run.diagnostics
            report["observation_failures"] = run.observation_failures
            report["cloud_execution"] = {"status": run.status, "results": [asdict(result) for result in run.results.values()]}
            if additional_ids:
                report["additional_cloud_tasks"] = [asdict(result) for key, result in run.results.items()
                    if key in additional_ids]
            try:
                _assert_source_bytes(plan["source_snapshot_manifest"], snapshot)
                report["execution_end_source_check"] = "actual_runtime_bytes_match_the_ingress_archive"
            except ValueError as error:
                report["execution_end_source_failure"] = {"type": type(error).__name__, "message": str(error)}
            for declaration in report["consumers"]:
                _, programme = consumers[declaration["name"]]
                try:
                    result = programme.finalize(run.results, input_admission=input_admission)
                    declaration.update(status=result["status"], result=result)
                except Exception as error:
                    declaration.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
            for entry in report["groups"]:
                path = Path(entry["directory"]) / "source_chain_record.json"
                if not path.exists():
                    continue
                child = json.loads(path.read_text(encoding="utf-8"))
                for key in ("tasks", "selection", "preflight_failures"):
                    report[key].extend(child[key])
                report.setdefault("source_prior_selection", []).extend(child.get("source_prior_selection", []))
                if independent:
                    report["source_seed_catalog"].extend(entry.get("accepted_source_seed_catalog", []))
                    if entry["status"] != "completed":
                        report.setdefault("unadmitted_source_artifacts", []).append({
                            "outer": entry["outer"], "record": str(path),
                            "source_seed_catalog": child.get("source_seed_catalog", []),
                            "admission_failure": entry.get("failure")})
            report.update(task_count=len(report["tasks"]),
                optimizer_updates=sum(task["optimizer_updates"] for task in report["tasks"] if task["status"] == "completed"),
                trial_draws=sum(task["trial_draws"] for task in report["tasks"] if task["status"] == "completed"),
                source_failure_count=sum(task["status"] == "fit_failure" for task in report["tasks"]),
                reused_source_task_count=sum(bool(task.get("source_training_reuse", {}).get("reused")) for task in report["tasks"]),
                completed_physical_optimizer_updates=sum(task.get("physical_optimizer_updates", 0) for task in report["tasks"]),
                completed_physical_trial_draws=sum(task.get("physical_trial_draws", 0) for task in report["tasks"]))
            report["fitting_performed"] = report["completed_physical_optimizer_updates"] > 0
            if independent:
                failures = {entry["outer"]: entry for entry in report["groups"] if entry["status"] == "failed"}
                for request in report["requested_source_tasks"]:
                    if request["outer"] in failures:
                        request.update(status="source_group_failure", failure=failures[request["outer"]],
                                       artifact_admission_status="rejected")
                _finish_independent_source_report(report, output)
            if on_additional_complete is not None:
                try:
                    finalization = on_additional_complete(run)
                    report["additional_finalization"] = {
                        "status": "completed" if finalization is None else finalization["status"]}
                except Exception as error:
                    report["additional_finalization"] = {"status": "failed", "failure": {
                        "type": type(error).__name__, "message": str(error), "stage": "additional_artifact_finalization"}}
            report["source_input_admission"] = input_admission.record()
            failed = (run.status != "completed" or "execution_end_source_failure" in report
                      or any(entry["status"] == "failed" for entry in report["consumers"])
                      or report.get("additional_finalization", {}).get("status", "completed") != "completed")
            report["status"] = "failed" if failed else "completed"
            _save_chain_report(output, report)
    if failed:
        raise RuntimeError("source/consumer programme failed; complete requested and completed artifact records are preserved")
    return report


def execute_source_chain(plan, *, output: Path, dry_run: bool, device: str, stages: tuple[str, ...],
                         evaluate_outer: bool = False, additional_tasks=(), on_additional_finished=None,
                         on_additional_complete=None):
    """Run source and dependent consumers under one owner, with optional CloudTasks.

    Additional adapters use the same resource claims and input/artifact admission.
    A no-argument task factory runs after that owner is installed, so initial
    request ledgers also receive shared persistent-file admission.
    Their per-task hook may return successors; the final hook consumes CloudRun
    while the shared artifact owner is live and may return a report with a
    completion status. Neither hook enters a scientific plan or a source-fit child.
    """
    if not callable(additional_tasks):
        additional_tasks = tuple(additional_tasks)
    if any(callback is not None and (not additional_tasks or not callable(callback))
           for callback in (on_additional_finished, on_additional_complete)):
        raise ValueError("additional completion requires initial tasks and a callable")
    if additional_tasks and (dry_run or plan.get("_source_group", False)):
        raise ValueError("additional cloud tasks require the executing parent resource owner")
    import numpy as np

    from baselines.deep import DeepConfig
    from data.contract import SOURCE_COHORT_DATA_CONTRACTS, assert_p300_input_contract
    from data.epochs import (
        _verified_cache_snapshot_directory,
        load_epoch_dataset,
        loaded_epoch_cache_attestation,
        select_epoch_channels,
    )
    from models.n2p3net import DEFAULT_N2P3_ARCHITECTURE, N2P3ArchitectureConfig
    from research.evaluation import verified_source_snapshot
    from research.source_execution import SharedTrajectoryRegistry
    from research.source_matrix import unified_source_conditions, validate_unified_source_input
    from research.source_resume import SourceChainResume
    from research.source_selection import (
        SOURCE_CHAIN_LAYOUTS,
        SOURCE_UPDATE_CANDIDATES,
        resolve_independent_source_config,
        resolve_source_program,
        source_architecture_record,
        source_fit_ledger,
        source_fixed_dose,
        source_identity_axes,
        source_inner_folds,
        validate_independent_source_input,
        validate_source_procedure,
    )
    from train.runtime import available_cpu_threads
    from transfer.target_experiments import resolve_target_batch_rule

    if plan.get("schema") != "n2p3_source_training_chain/1":
        raise ValueError("unsupported source training chain plan schema")
    program = resolve_source_program(plan, stages)
    independent = program["name"] == "independent_selection_refit"
    if independent and evaluate_outer:
        raise ValueError("independent source checkpoints are consumed by the declared independent public panels")
    unified = "unified_source" in stages
    from train.early_stopping import GlobalEarlyStoppingPolicy, UnifiedTrainingBudget

    stopping = None
    budget = None
    if unified:
        if plan.get("global_early_stopping") is not None:
            raise ValueError("unified uses unified_training_budget; global early stopping is a separate research program")
        budget = UnifiedTrainingBudget(**plan.get("unified_training_budget", {}))
        plan = {**plan, "unified_training_budget": budget.record(),
                "calibration_trials": plan.get("calibration_trials", [0, 200, 800])}
    if not unified and plan.get("global_early_stopping") is not None:
        stopping = GlobalEarlyStoppingPolicy(**plan.get("global_early_stopping", {}))
        plan = {**plan, "global_early_stopping": stopping.record()}
        if "E2" in stages:
            raise ValueError("global stopping cannot be combined with the historical E2 update scan")
    if unified and set(stages) != {"unified_source"}:
        raise ValueError("unified source requires its own explicit stage and input plan")
    if unified and not isinstance(plan.get("source_preprocessing"), dict):
        raise ValueError("unified source requires an explicit complete source_preprocessing record")
    conditions = unified_source_conditions(**plan.get("unified_source", {})) if unified else ()
    condition_by_arm = {condition["source_arm"]: condition for condition in conditions}
    if not dry_run and not plan.get("_source_group", False):
        groups = source_execution_groups(plan, stages)
        if groups:
            return _execute_source_groups(plan, output=output, device=device, stages=stages,
                evaluate_outer=evaluate_outer, groups=groups, additional_tasks=additional_tasks,
                on_additional_finished=on_additional_finished, on_additional_complete=on_additional_complete)
    if additional_tasks:
        raise ValueError("additional cloud tasks require at least one declared source group")
    workers = plan.get("target_workers", 1)
    if workers in (None, "auto"):
        workers = 1 if dry_run else available_cpu_threads()
        plan = {**plan, "target_workers": workers}
    if type(workers) is not int or not 1 <= workers <= available_cpu_threads():
        raise ValueError("target_workers must be within the available CPU budget")
    evidence_scope = plan.get("evidence_scope", "development")
    if (unified and not dry_run and plan.get("source_prior_selection") is None
            and not all(outer.get("source_prior_selection") is not None for outer in plan["outer_folds"])
            ):
        raise ValueError("unified development requires source_prior_selection with independent source-inner raw prefixes")
    if evidence_scope == "software_correctness":
        candidates = tuple(plan.get("correctness_candidate_updates", [1, 2]))
        source_updates = int(plan.get("correctness_source_updates", 2))
        if not candidates or tuple(sorted(set(candidates))) != candidates or min(candidates) < 1 or max(candidates) > 10 or not 1 <= source_updates <= 10:
            raise ValueError("software correctness uses at most ten source updates")
    else:
        candidates, source_updates = SOURCE_UPDATE_CANDIDATES, 4000
    source_seeds = tuple(plan.get("source_seeds", [20260906]))
    target_seeds = tuple(plan.get("target_seeds", [20260906]))
    from transfer.source_objective import source_objective_recipes

    objectives = source_objective_recipes(plan.get("source_objectives")) if "P2_source" in stages else {}
    from stats.transfer import resolve_seed_pairs

    if type(plan.get("paired_source_target_seeds", True)) is not bool:
        raise ValueError("paired_source_target_seeds must be an explicit boolean")
    if (objectives or unified) and not independent and plan.get("paired_source_target_seeds", True) and set(source_seeds) != set(target_seeds):
        raise ValueError("paired source-objective experiments require matching declared source/target seeds")
    if not source_seeds or not target_seeds or len(set(source_seeds)) != len(source_seeds) or len(set(target_seeds)) != len(target_seeds):
        raise ValueError("source and target seeds must be independently unique")
    crossed_pairs = () if independent else resolve_seed_pairs(source_seeds, target_seeds)
    objective_pairs = ()
    if independent:
        objective_pairs = tuple(map(tuple, program["seed_pairs"]))
    elif objectives or unified:
        objective_pairs = resolve_seed_pairs(source_seeds, target_seeds,
            mode="paired" if plan.get("paired_source_target_seeds", True) else "crossed")
    batch, batch_rule = resolve_target_batch_rule(plan, dry_run=dry_run)
    plan = {**plan, "target_batch_size": batch, "batch_rule": batch_rule}
    snapshot_path = Path(plan["source_snapshot_manifest"])
    snapshot = verified_source_snapshot(snapshot_path)
    if not dry_run:
        from research.evaluation import verified_executed_snapshot
        verified_executed_snapshot(snapshot_path, root=ROOT)
    phases = plan.get("_source_input_phases")
    precreated = {phases.path.resolve()} if phases is not None else set()
    if output.exists() and any(path.resolve() not in precreated for path in output.iterdir()):
        raise FileExistsError("source chain requires a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "n2p3_source_training_chain_run/1", "dry_run": dry_run,
              "evidence_scope": evidence_scope,
              "source_program": program["name"], "source_program_record": program,
              "batch_rule": batch_rule,
              "source_snapshot": snapshot, "source_seeds": list(source_seeds),
              "target_seeds": list(target_seeds), "tasks": [], "selection": [],
              "preflight_failures": [], "outer_evaluation_jobs": [], "fitting_performed": False}
    if independent:
        report["source_seed_catalog"] = []
        report["requested_source_tasks"] = _independent_source_requests(plan)
    _save_chain_report(output, report)
    trajectory_registry = (SharedTrajectoryRegistry(plan["shared_registry_directory"])
                           if plan.get("shared_registry_directory") else {})
    resume = SourceChainResume(plan.get("resume_source_run"))
    for outer in plan["outer_folds"]:
        source_cache = Path(outer["source_cache"])
        from contextlib import nullcontext

        production = plan.get("_resident_source_production")
        phases = plan.get("_source_input_phases")
        if production is not None:
            if production.path != source_cache.resolve() or len(plan["outer_folds"]) != 1:
                raise ValueError("resident source production must belong to this exact whole-fold cache")
            with phases.phase("resident_writer_receipt_and_decoded_verify") if phases is not None else nullcontext():
                production_identity = production.verify()
            source = production.dataset
        else:
            production_identity = None
            snapshot_directory = _verified_cache_snapshot_directory(source_cache,
                byte_size=source_cache.stat().st_size, directory_override=plan.get("verified_cache_snapshot_directory"))
            with phases.phase("attested_resume_load") if phases is not None else nullcontext():
                source = load_epoch_dataset(source_cache, require_labels=True, validation="attested",
                                            snapshot_directory=snapshot_directory)
        input_record = None
        if independent:
            input_record = validate_independent_source_input(source, program["preprocessing"])
        elif unified:
            validate_unified_source_input(source, plan["source_preprocessing"])
        else:
            assert_p300_input_contract(source.preprocessing, SOURCE_COHORT_DATA_CONTRACTS["causal"])
        if source.provenance.get("source_reference") != "measured_channel:CZ":
            raise ValueError("source-chain caches must contain measured Cz differences")
        source_sha = (production_identity["sha256"] if production_identity is not None
                      else loaded_epoch_cache_attestation(source)["sha256"])
        if outer.get("expected_source_cache_sha256", source_sha) != source_sha:
            raise ValueError("source worker loaded different bytes from its admitted input binding")
        excluded = tuple(outer["excluded_source_datasets"])
        with phases.phase("whole_source_input_ledger") if phases is not None else nullcontext():
            source_fit_ledger(source, np.arange(source.n_epochs), excluded_source_datasets=excluded)
            domains, _, _ = source_identity_axes(source)
        from research.worker_diagnostics import worker_diagnostic_phase

        worker_diagnostic_phase("actual_source_input_admitted", source_shape=list(source.X.shape),
            source_dtype=str(source.X.dtype), source_cache_sha256=source_sha, source_cache_path=str(source_cache),
            source_program=program["name"], native_threads=plan.get("_native_threads_per_worker", 1))
        from research.source_catalog import SourceCatalog

        SourceCatalog(plan.get("source_pools")).validate_fold(outer, domains)
        report["source_input_admitted"] = True
        if production_identity is not None:
            report["source_input_production"] = production_identity
        source_config_record = None
        if independent:
            from data.study import dataset_study_membership

            source_config_record = resolve_independent_source_config(program["source_config"],
                source.identity_table, dataset_study_membership(source))
        for layout in plan.get("source_layouts", ("six",) if set(stages) in ({"P2_source"}, {"unified_source"}) else ("six", "common_three")):
            if outer.get("family_exclusion", False) and layout != "common_three" and not objectives and not unified:
                continue
            if layout == "six" and "S1" not in stages and not objectives and not unified:
                continue
            dataset = select_epoch_channels(source, SOURCE_CHAIN_LAYOUTS[layout], copy=False)
            domains, _, _ = source_identity_axes(dataset)
            if layout == "common_three" and dataset.trial_channel_mask is not None and not dataset.trial_channel_mask.all():
                raise ValueError("common-three source experiments require genuinely observed channels on every row")
            all_rows = np.arange(dataset.n_epochs)
            prior_selection = None
            if unified and (outer.get("source_prior_selection") is not None or plan.get("source_prior_selection") is not None):
                prior_selection = _select_unified_source_prior(dataset, outer, layout=layout, plan=plan, report=report,
                    output=output, source_sha=source_sha, snapshot_sha=snapshot["archive_sha256"],
                    seed_pairs=objective_pairs, registry=trajectory_registry, device=device, dry_run=dry_run, resume=resume)
                report.setdefault("source_prior_selection", []).append(prior_selection)
                if not dry_run and prior_selection["status"] != "selected":
                    report["preflight_failures"].append({"outer": outer["name"], "layout": layout,
                        "status": "source_prior_selection_pending", "reason": prior_selection.get("reason", "no_complete_legal_candidate")})
                    continue
            for seed in source_seeds:
                source_budget = None
                if budget is not None:
                    selected_epochs = (budget.reference_max_epochs if dry_run else
                                       prior_selection["source_budget"]["selected_epochs"])
                    source_budget = (None if dry_run else prior_selection["source_budget"])
                    main_updates = selected_epochs * ((len(all_rows) + 511) // 512)
                else:
                    main_updates = source_updates
                base = DeepConfig(fixed_updates=main_updates if stopping is None else None,
                                  source_budget=source_budget,
                                  global_early_stopping=stopping, batch_size=512, seed=seed, val_group_frac=None,
                                  sampling="uniform_rows", compile_mode=plan.get("compile_mode", "reduce-overhead"))
                if source_config_record is not None:
                    base = replace(base, **source_config_record["resolved"])
                work = []
                work.extend((f"P2_source_{name}", all_rows, replace(base, source_objective=objective))
                            for name, objective in objectives.items())
                work.extend((condition["source_arm"], all_rows,
                             replace(base, source_objective=condition["source_objective"])) for condition in conditions)
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
                            trials_per_participant=128 if label == "multi" else 128 * len(set(domains)),
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
                    condition = condition_by_arm.get(label)
                    architecture = (N2P3ArchitectureConfig(**condition["architecture_configuration"])
                                    if condition is not None else DEFAULT_N2P3_ARCHITECTURE)
                    key = f"{outer['name']}/{layout}/seed{seed}/{label}"
                    ledger = source_fit_ledger(dataset, rows, excluded_source_datasets=excluded)
                    updates_bound = (config.fixed_updates if stopping is None else
                                     2 * stopping.source_max_epochs * ((len(rows) + config.batch_size - 1) // config.batch_size))
                    task = {"key": key, "status": "planned", "config": asdict(config), "fit_ledger": ledger,
                            "source_cache": str(source_cache), "source_cache_sha256": source_sha,
                            "optimizer_updates": updates_bound,
                            "trial_draws": updates_bound * config.batch_size,
                            **({"budget_kind": "inner_plus_refit_upper_bound"} if stopping is not None else {}),
                            "layout_contrast": "information_and_input_dimension_change"}
                    if independent:
                        task.update(requested_optimizer_updates_upper_bound=updates_bound,
                                    requested_trial_draws_upper_bound=updates_bound * config.batch_size)
                        if not dry_run:
                            task.update(optimizer_updates=None, trial_draws=None, budget_kind="actual_pending")
                    task["architecture_configuration"] = asdict(architecture)
                    if input_record is not None:
                        task["source_input"] = input_record
                        task["source_input_bindings"] = outer.get("source_input_bindings")
                        task["source_program_config"] = source_config_record
                    if budget is not None:
                        task["budget_kind"] = ("reference_selected_exposure_upper_bound" if dry_run
                                               else "frozen_reference_exposure")
                    if condition is not None:
                        task["source_factors"] = condition["source_factors"]
                    report["tasks"].append(task)
                    if dry_run:
                        continue
                    task["status"] = "running"
                    _save_chain_report(output, report)
                    destination = output / key
                    destination.mkdir(parents=True, exist_ok=True)
                    try:
                        worker_diagnostic_phase("source_fit_or_verified_reuse", source_seed=seed, task_key=key,
                            resolved_config=asdict(config), source_cache_sha256=source_sha, failure=None,
                            physical_optimizer_updates=None, physical_trial_draws=None)
                        acquired = resume.acquire(key, config, ledger, source_sha,
                            architecture_record=source_architecture_record(dataset, architecture))
                        if acquired is None:
                            acquired = fit_or_reuse_source_trajectory(
                                registry=trajectory_registry, dataset=dataset, rows=rows, config=config,
                                destination=destination, task_id=key,
                                source_cache_sha256=source_sha, source_snapshot_sha256=snapshot["archive_sha256"],
                                excluded_source_datasets=excluded, device=device, evidence_scope=evidence_scope,
                                gpu_slots=plan.get("_gpu_slots"), architecture=architecture,
                                runtime_preload_budget_bytes=plan.get("_runtime_preload_budget_bytes"),
                                runtime_input_staging_gate=plan.get("_runtime_input_staging_gate"),
                                runtime_host_process_budget_bytes=plan.get("_runtime_host_process_budget_bytes"),
                                source_program_config=source_config_record)
                    except RuntimeError as error:
                        worker_diagnostic_phase("source_fit_python_failure", failure={"type": type(error).__name__, "message": str(error)},
                                                source_seed=seed, task_key=key)
                        task.update(status="fit_failure", failure=str(error))
                        if independent:
                            task["budget_kind"] = "actual_unmeasured_after_fit_failure"
                        _save_chain_report(output, report)
                        continue
                    report["fitting_performed"] |= acquired["physical_optimizer_updates"] > 0
                    worker_diagnostic_phase("source_fit_returned", source_seed=seed, task_key=key,
                        physical_optimizer_updates=acquired["physical_optimizer_updates"],
                        physical_trial_draws=acquired["physical_trial_draws"])
                    task.update({name: acquired[name] for name in (
                        "source_training_reuse", "physical_optimizer_updates", "physical_trial_draws")})
                    task["checkpoints"] = {str(update): value for update, value in acquired["checkpoints"].items()}
                    if independent:
                        from transfer.checkpoint import load_checkpoint_payload

                        checkpoint = next(iter(acquired["checkpoints"].values()))
                        payload = load_checkpoint_payload(checkpoint)
                        producer = validate_source_procedure(payload, source_seed=seed,
                            outer_target_study=outer["target_study"],
                            evidence_scope=("software_correctness" if evidence_scope == "software_correctness"
                                            else "public_development"), kind="selection_refit",
                            expected_source_config=program["source_config"])
                        selection = payload["source_epoch_selection"]
                        task.update(optimizer_updates=selection["total_optimizer_updates"],
                            trial_draws=selection["total_trial_draws"], budget_kind="actual_inner_plus_refit",
                            source_producer=producer, source_epoch_selection=selection,
                            physical_fit_count=0 if acquired["physical_optimizer_updates"] == 0 else 2,
                            logical_fit_count=2)
                        entry = {"outer_target_study": outer["target_study"], "source_seed": seed,
                            "path": str(Path(checkpoint).resolve()), "sha256": _sha256(checkpoint),
                            "procedure": {"kind": "selection_refit"}}
                        task["source_catalog_entry"] = entry
                        report["source_seed_catalog"].append(entry)
                    task["status"] = "completed"
                    gc.collect()
                    _save_chain_report(output, report)
                    if not independent and not label.startswith("E2_") and outer.get("role", "evaluation") != "deployment":
                        selected_checkpoint = (next(iter(acquired["checkpoints"].values())) if stopping is not None
                                               else acquired["checkpoints"][config.fixed_updates])
                        report["outer_evaluation_jobs"].append({"source_checkpoint": selected_checkpoint,
                            "outer": outer["name"], "layout": layout, "source_seed": seed,
                            "source_arm": label,
                            "analysis_seed_pairs": [list(pair) for pair in (
                                objective_pairs if label.startswith("P2_source_") or unified else crossed_pairs)],
                            **({"analysis_source_arms": [f"P2_source_{name}" for name in objectives]}
                               if label.startswith("P2_source_") else {}),
                            **({"analysis_source_arms": list(condition_by_arm), "source_factors": condition["source_factors"]}
                               if condition is not None else {}),
                            **({"source_prior_weight": prior_selection["selected_weight"],
                                "unified_training_budget": budget.record(),
                                "source_budget": prior_selection["source_budget"],
                                "source_prior_selection": {"record_sha256": semantic_sha256(prior_selection),
                                    "path": str((output / outer["name"] / layout / "source_prior_selection/selection.json").resolve())}}
                               if prior_selection is not None and prior_selection["status"] == "selected" else {}),
                            "target_seeds": ([seed] if (label.startswith("P2_source_") or unified)
                                and plan.get("paired_source_target_seeds", True) else list(target_seeds)),
                            "target_stage": "unified" if unified else "P2" if label.startswith("P2_source_") else None,
                            "target_cache": outer.get("target_cache"),
                            "target_protocol": outer.get("target_protocol")})
                if layout == "common_three" and "E2" in stages and not outer.get("family_exclusion", False):
                    _run_e2_selection(dataset, outer, seed, target_seeds, output, source_sha,
                                      snapshot["archive_sha256"], plan, report, dry_run, device,
                                      trajectory_registry, resume)
            del dataset
        del source
        gc.collect()
    report.update(task_count=len(report["tasks"]),
        optimizer_updates=sum(task["optimizer_updates"] for task in report["tasks"]
                              if not independent or dry_run or task["status"] == "completed"),
        trial_draws=sum(task["trial_draws"] for task in report["tasks"]
                        if not independent or dry_run or task["status"] == "completed"))
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
    if not dry_run and not independent and not plan.get("_defer_outer_plans", False):
        _prepare_outer_plans(report, plan, output, device=device, execute=evaluate_outer)
    if independent:
        _finish_independent_source_report(report, output)
    _save_chain_report(output, report)
    return report


def _run_e2_selection(dataset, outer, seed, target_seeds, output, source_sha, snapshot_sha, plan, report, dry_run, device,
                      trajectory_registry, resume=None):
    import numpy as np

    from baselines.deep import DeepConfig
    from research.source_selection import (
        SOURCE_UPDATE_CANDIDATES,
        select_source_updates,
        source_fit_ledger,
        source_inner_folds,
    )
    from transfer.binary import STANDARD_CALIBRATION_TRIALS
    from transfer.recipes import TargetRecipe

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
                                gpu_slots=plan.get("_gpu_slots"),
                                runtime_preload_budget_bytes=plan.get("_runtime_preload_budget_bytes"),
                                runtime_input_staging_gate=plan.get("_runtime_input_staging_gate"),
                                runtime_host_process_budget_bytes=plan.get("_runtime_host_process_budget_bytes"))
                    except RuntimeError as error:
                        report["tasks"].append({"key": str(destination), "refit_key": list(refit_key),
                            "status": "fit_failure", "failure": str(error), "optimizer_updates": updates,
                            "trial_draws": updates * 512, "source_seed": seed, "config": asdict(config)})
                        _save_chain_report(output, report)
                        continue
                    destination = Path(acquired["checkpoints"][updates])
                    report["tasks"].append({"key": str(destination), "refit_key": list(refit_key),
                        "status": "completed", "checkpoint": str(destination),
                        "optimizer_updates": updates, "trial_draws": updates * 512,
                        "source_seed": seed, "fit_ledger": acquired["fit_ledger"],
                        "refit_initialization": "scratch_same_source_seed", "config": asdict(config),
                        **{name: acquired[name] for name in (
                            "source_training_reuse", "physical_optimizer_updates", "physical_trial_draws")}})
                report["outer_evaluation_jobs"].append({"source_checkpoint": str(destination), "outer": outer["name"],
                    "layout": "common_three", "source_seed": seed, "target_seed": target_seed, "axis": axis,
                    "source_arm": "E2_" + axis,
                    "selectors": [name for name in ("U_ce", "U_cal", "U_zero") if selected[name] == updates],
                    "target_cache": outer.get("target_cache"), "target_protocol": outer.get("target_protocol")})


@lru_cache(maxsize=2)
def _source_prior_prefix(path, digest, subject):
    from data.source_records import load_target_prefixes

    if _sha256(path) != digest:
        raise ValueError("source-prior raw prefix manifest changed after planning")
    return load_target_prefixes(path, subjects=(subject,), load_inner=False)[subject]


def _score_source_prior_person(arguments):
    """One isolated person/seed owns its RNG, checkpoint and all lambda candidates."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from data.epochs import select_epoch_channels
    from research.source_execution import gpu_task_slot, release_source_worker
    from research.source_matrix import validate_unified_source_input
    from train.early_stopping import UnifiedTrainingBudget
    from train.runtime import cpu_thread_budget
    from transfer.calibration_candidates import (
        calibration_decision_groups,
        fit_calibration_candidate,
    )
    from transfer.checkpoint import PreparedTransferCheckpoint, load_checkpoint_payload
    from transfer.recipes import unified_target_recipes

    args = arguments
    outcomes = []
    with cpu_thread_budget(args.get("cpu_threads", 1)), gpu_task_slot(args.get("gpu_slots")):
        try:
            budget = UnifiedTrainingBudget(**args["unified_training_budget"])
            entry = _source_prior_prefix(args["prefix_manifest"], args["prefix_manifest_sha256"], args["subject"])
            original = entry["calibration"][args["K"]]
            calibration = select_epoch_channels(original, args["channel_names"])
            query = select_epoch_channels(entry["test"], args["channel_names"])
            validate_unified_source_input(calibration, args["preprocessing"])
            validate_unified_source_input(query, args["preprocessing"])
            payload = load_checkpoint_payload(args["checkpoint"])
            prepared = PreparedTransferCheckpoint(payload, calibration)
            source_trunk = prepared.trunk_for_target(calibration.identity_table.local_subject_ids[0])
            source_state = _model_state_digest(source_trunk)
            groups, _, _, _ = calibration_decision_groups(calibration, np.ones(calibration.n_epochs, bool))
            for weight in args["candidates"]:
                cell = {"participant_key": args["participant_key"], "source_seed": args["source_seed"],
                        "target_seed": args["target_seed"], "source_prior_weight": weight,
                        "checkpoint_sha256": _sha256(args["checkpoint"]),
                        "membership_available": bool(groups), "decision_factor": 1 if groups else None}
                try:
                    recipes = unified_target_recipes(membership_available=bool(groups), source_prior_weight=weight,
                        batch_size=args["batch_size"], seed=args["target_seed"], epochs=budget.target_epochs,
                        decision_weight=args["decision_weight"])
                    recipe = next(recipe for recipe in recipes if recipe.name == "unified_q1")
                    fitted = fit_calibration_candidate(calibration, recipe, np.ones(calibration.n_epochs, bool),
                        source_trunk=source_trunk, source_input_stats=prepared.input_statistics, device=args["device"])
                    fit_budget = fitted.fit_history["fit_budget"]
                    selection = {"mode": "fixed_full_calibration", "selected_epochs": budget.target_epochs,
                        "total_optimizer_updates": fit_budget["updates"], "physical_optimizer_updates": fit_budget["updates"],
                        "total_trial_draws": fit_budget["draws"], "physical_trial_draws": fit_budget["draws"]}
                    mask = query.trial_channel_mask
                    if mask is None:
                        mask = np.broadcast_to(query.channel_mask, query.X.shape[:2])
                    scores = fitted.predict_logit(query.X, mask)
                    if _model_state_digest(source_trunk) != source_state:
                        raise RuntimeError("source-prior adaptation changed its shared source checkpoint")
                    prediction = Path(args["output"]) / "predictions" / (semantic_sha256(cell)[:24] + ".npz")
                    prediction.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(prediction, scores=scores, y=query.y)
                    cell.update(status="completed", auc=float(roc_auc_score(query.y, scores)),
                        prediction_path=str(prediction.resolve()), prediction_sha256=_sha256(prediction),
                        target_recipe=recipe.record(), epoch_selection=selection)
                except (ValueError, RuntimeError) as error:
                    cell.update(status="fit_failed", auc=None, failure={"type": type(error).__name__, "message": str(error)})
                outcomes.append(cell)
        finally:
            release_source_worker(args["device"])
    return outcomes


def _select_unified_source_prior(dataset, outer, *, layout, plan, report, output, source_sha,
                                 snapshot_sha, seed_pairs, registry, device, dry_run, resume=None):
    """Select one reference exposure and one lambda per outer fold, never per main seed."""
    import numpy as np

    from baselines.deep import DeepConfig
    from data.source_records import load_target_prefixes
    from models.n2p3net import N2P3ArchitectureConfig
    from research.source_matrix import unified_source_conditions
    from research.source_selection import (
        select_source_prior_weight,
        source_architecture_record,
        source_identity_axes,
        source_prior_development_fold,
    )
    from train.early_stopping import UnifiedTrainingBudget
    from transfer.checkpoint import load_checkpoint_payload

    declaration = outer.get("source_prior_selection", plan.get("source_prior_selection", {}))
    if set(declaration) - {"prefix_manifests", "candidates", "calibration_trials", "development_domains", "excluded_development_domains"}:
        raise ValueError("source-prior split settings belong to unified_training_budget")
    from research.source_catalog import study_names

    domains, _, _ = source_identity_axes(dataset)
    development_domains = declaration.get("development_domains")
    excluded_development = declaration.get("excluded_development_domains", {})
    if development_domains is not None:
        development_domains = study_names(development_domains, field="development_domains")
        if (not isinstance(excluded_development, dict)
                or set(development_domains) & set(excluded_development)
                or set(development_domains) | set(excluded_development) != set(domains)
                or any(not isinstance(reason, str) or not reason.strip() for reason in excluded_development.values())):
            raise ValueError("development support must partition actual source domains with explicit exclusion reasons")
    elif excluded_development:
        raise ValueError("development exclusions require explicit development_domains")
    budget = UnifiedTrainingBudget(**plan.get("unified_training_budget", {}))
    candidates = tuple(declaration.get("candidates", (0., 1e-3, 1e-2, .1)))
    select_source_prior_weight([], requested_cells=[], candidates=candidates)
    calibration_k = int(declaration.get("calibration_trials", 200))
    if calibration_k != 200 and plan.get("evidence_scope") != "software_correctness":
        raise ValueError("source-prior development uses the predeclared K200 estimand")
    admitted = set(dataset.identity_table.authority_keys())
    locations, eligible, inputs = {}, set(), []
    # Keep only identities and paths in the parent: a whole source pool of C_k
    # tensors must not remain resident while reference training is running.
    for name in declaration.get("prefix_manifests", []):
        path = Path(name)
        digest = _sha256(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        inputs.append({"path": str(path.resolve()), "sha256": digest})
        for raw_entry in manifest["records"]:
            subject = str(raw_entry.get("subject", raw_entry.get("record", {}).get("subject")))
            loaded = load_target_prefixes(path, subjects=(subject,), load_inner=False)
            entry = loaded[subject]
            key = entry["test"].identity_table.authority_keys()[0]
            if key not in admitted or key in locations:
                raise ValueError("source-prior prefix participant is outside the admitted source or repeated")
            locations[key] = {"prefix_manifest": str(path.resolve()), "prefix_manifest_sha256": digest, "subject": subject}
            calibration = entry["calibration"].get(calibration_k)
            if (calibration is not None and set(np.unique(calibration.y)) == {0, 1}
                    and set(np.unique(entry["test"].y)) == {0, 1}):
                eligible.add(key)
            del loaded, entry, calibration
    destination = output / outer["name"] / layout / "source_prior_selection"
    record = {"schema": "n2p3_source_prior_selection/2", "outer": outer["name"], "layout": layout,
        "candidates": list(candidates), "calibration_trials": calibration_k,
        "unified_training_budget": budget.record(), "source_cache_sha256": source_sha,
        "source_snapshot_sha256": snapshot_sha, "input_manifests": inputs,
        "development_support": {"domains": sorted(set(domains) if development_domains is None else development_domains),
                                "excluded_domains": excluded_development},
        "reference_factors": {"R": 1, "S": 1, "M": 1},
        "reference_decision": "Q1_when_membership_available_otherwise_trial_plus_prior", "outcomes": []}
    try:
        fold = source_prior_development_fold(dataset, seed=budget.development_seed,
            people_per_domain=budget.development_people_per_domain, eligible_people=eligible,
            development_domains=development_domains,
            excluded_source_datasets=outer["excluded_source_datasets"])
    except ValueError as error:
        record.update(status="pending", selected_weight=None, reason=str(error))
        atomic_write_json(destination / "selection.json", record)
        return record
    record["fold"] = fold
    heldout = fold["validation_participant_keys"]
    if dry_run:
        record.update(status="planned", selected_weight=None, requested_fits=len(heldout) * len(candidates))
        atomic_write_json(destination / "selection.json", record)
        return record
    condition = next(item for item in unified_source_conditions(**plan.get("unified_source", {}))
                     if item["source_factors"] == record["reference_factors"])
    architecture = N2P3ArchitectureConfig(**condition["architecture_configuration"])
    source_seed = target_seed = budget.development_seed
    config = DeepConfig(global_early_stopping=budget.reference_policy(), batch_size=512, seed=source_seed,
        val_group_frac=None, sampling="uniform_rows", source_objective=condition["source_objective"],
        compile_mode=plan.get("compile_mode", "reduce-overhead"))
    task_id = f"{outer['name']}/{layout}/source_prior_reference"
    acquired = (None if resume is None else resume.acquire(task_id, config, fold["fit_ledger"], source_sha,
        architecture_record=source_architecture_record(dataset, architecture)))
    if acquired is None:
        acquired = fit_or_reuse_source_trajectory(registry=registry, dataset=dataset, rows=np.asarray(fold["train_rows"]),
            config=config, destination=destination / "reference", task_id=task_id,
            source_cache_sha256=source_sha, source_snapshot_sha256=snapshot_sha,
            excluded_source_datasets=outer["excluded_source_datasets"], device=device,
            evidence_scope=plan.get("evidence_scope", "development"), architecture=architecture, gpu_slots=plan.get("_gpu_slots"),
            runtime_preload_budget_bytes=plan.get("_runtime_preload_budget_bytes"),
                                runtime_input_staging_gate=plan.get("_runtime_input_staging_gate"),
                                runtime_host_process_budget_bytes=plan.get("_runtime_host_process_budget_bytes"))
    checkpoint = next(iter(acquired["checkpoints"].values()))
    payload = load_checkpoint_payload(checkpoint)
    record["source_budget"] = {"mode": "reference_selected_exposure", "selected_epochs": payload["source_epoch_selection"]["selected_epochs"],
        "reference_checkpoint_sha256": _sha256(checkpoint), "reference_policy": budget.reference_policy().record(),
        "reference_fit_ledger_sha256": semantic_sha256(fold["fit_ledger"])}
    report["fitting_performed"] = report.get("fitting_performed", False) or acquired["physical_optimizer_updates"] > 0
    report["tasks"].append({"key": task_id, "status": "completed", "role": "source_prior_development",
        "optimizer_updates": payload["source_epoch_selection"]["total_optimizer_updates"],
        "trial_draws": payload["source_epoch_selection"]["total_trial_draws"],
        "config": asdict(config), "architecture_configuration": asdict(architecture),
        "checkpoints": {str(key): value for key, value in acquired["checkpoints"].items()},
        **{name: acquired[name] for name in ("fit_ledger", "source_training_reuse", "physical_optimizer_updates", "physical_trial_draws")}})
    jobs = [{**locations[person], "checkpoint": checkpoint, "participant_key": person,
        "source_seed": source_seed, "target_seed": target_seed, "K": calibration_k, "candidates": candidates,
        "channel_names": dataset.channel_names, "preprocessing": plan["source_preprocessing"],
        "unified_training_budget": budget.record(), "decision_weight": plan.get("decision_weight", 1.),
        "batch_size": plan.get("target_batch_size", 512), "device": device, "output": str(destination),
        "gpu_slots": plan.get("_gpu_slots")} for person in heldout]
    for job in jobs:
        job["cpu_threads"] = plan.get("_native_threads_per_worker", 1)
    from contextlib import nullcontext

    requested_workers = min(plan.get("target_workers", 1), len(jobs))
    admission = plan.get("_memory_admission")
    with (admission.target_workers(requested_workers) if admission is not None else nullcontext(requested_workers)) as workers:
        record["target_workers"] = workers
        if workers == 1:
            for job in jobs:
                record["outcomes"].extend(_score_source_prior_person(job))
                atomic_write_json(destination / "selection.json", record)
        else:
            from research.source_execution import current_input_admission, install_input_admission

            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=install_input_admission,
                    initargs=(current_input_admission(), os.environ.get("N2P3_CLOUD_TASK_ID"))) as pool:
                futures = [pool.submit(_score_source_prior_person, job) for job in jobs]
                for future in as_completed(futures):
                    record["outcomes"].extend(future.result())
                    atomic_write_json(destination / "selection.json", record)
    record.update(select_source_prior_weight(record["outcomes"],
        requested_cells=[(person, source_seed, target_seed) for person in heldout], candidates=candidates,
        participant_domains=fold["participant_domains"]))
    record["target_cost"] = {field: sum(row.get("epoch_selection", {}).get(field, 0) for row in record["outcomes"])
        for field in ("total_optimizer_updates", "total_trial_draws", "physical_optimizer_updates", "physical_trial_draws")}
    atomic_write_json(destination / "selection.json", record)
    return record


def _prepare_outer_plans(report, plan, output, *, device, execute):
    """Resolve projected targets only after all source selection has finished."""
    from data.epochs import (
        concatenate_epoch_datasets,
        load_epoch_dataset,
        save_epoch_dataset,
        select_epoch_channels,
    )
    from data.study import dataset_study_membership, study_ids_for_identity
    from research.source_selection import SOURCE_CHAIN_LAYOUTS
    from transfer.binary import STANDARD_CALIBRATION_TRIALS
    from transfer.checkpoint import PreparedTransferCheckpoint, load_checkpoint_payload

    outers = {outer["name"]: outer for outer in plan["outer_folds"]}
    prepared = {}
    final_jobs = []
    grouped_jobs = {}
    workers = plan.get("target_workers", 1)
    native_threads = plan.get("_target_native_threads_per_worker", 1)
    for job in report["outer_evaluation_jobs"]:
        outer = outers[job["outer"]]
        unified = job.get("target_stage") == "unified"
        targets = outer.get("targets") or [{**({"prefix_manifest": outer["target_prefix_manifest"]}
                                              if unified else {"cache": outer["target_cache"]}),
            "protocol": outer["target_protocol"], "study": outer.get("target_study", outer["name"])}]
        payload = load_checkpoint_payload(job["source_checkpoint"])
        for target in targets:
            input_path = target["prefix_manifest"] if unified else target["cache"]
            cache_key = (str(input_path), job["layout"], unified)
            if cache_key not in prepared:
                if unified:
                    from data.source_records import load_target_prefixes
                    from research.source_matrix import validate_unified_source_input

                    prefixes = load_target_prefixes(input_path, load_inner=False)
                    tests = [entry["test"] for entry in prefixes.values()]
                    raw = concatenate_epoch_datasets(tests, name="unified_fixed_raw_tests",
                        provenance={"source_reference": tests[0].provenance["source_reference"]})
                    validate_unified_source_input(raw, plan["source_preprocessing"])
                else:
                    raw = load_epoch_dataset(target["cache"], require_labels=True, validation="attested")
                if not set(study_ids_for_identity(raw.identity_table, dataset_study_membership(raw))) <= set(outer["excluded_source_datasets"]):
                    raise ValueError("outer target library disagrees with the source exclusion ledger")
                selected = select_epoch_channels(raw, SOURCE_CHAIN_LAYOUTS[job["layout"]])
                cache_name = semantic_sha256({"parent": _sha256(input_path), "layout": job["layout"]})[:20]
                cache = output / "target_views" / f"{cache_name}.npz"
                save_epoch_dataset(cache, selected)
                prepared[cache_key] = {"cache": str(cache.resolve()), "sha256": _sha256(cache),
                                      "subjects": selected.identity_table.local_subject_ids}
                if unified:
                    prepared[cache_key].update(target_prefix_manifest=str(Path(input_path).resolve()),
                        target_prefix_manifest_sha256=_sha256(input_path))
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
                        **{field: view[field] for field in ("target_prefix_manifest", "target_prefix_manifest_sha256") if field in view},
                        "target_study": target["study"], "source_studies": payload["source_fit_ledger"]["actual_source_libraries"],
                        "expected_source_participant_keys": payload["training_contract"]["training_participant_keys"],
                        "source_snapshot_manifest": str(Path(plan.get("source_snapshot_manifests", {}).get(
                            payload["source_snapshot_sha256"], plan["source_snapshot_manifest"])).resolve()),
                        "evaluation_snapshot_manifest": str(Path(plan.get("evaluation_snapshot_manifest", plan["source_snapshot_manifest"])).resolve()),
                        "target_protocol": target["protocol"], "calibration_trials": list(plan.get("calibration_trials", STANDARD_CALIBRATION_TRIALS)),
                        "source_seed": job["source_seed"], "target_seed": target_seed,
                        **{field: job[field] for field in ("analysis_seed_pairs", "analysis_source_arms", "source_factors") if field in job},
                        **{field: plan[field] for field in ("source_preprocessing", "source_prior_weight", "decision_weight", "target_max_updates") if field in plan},
                        **{field: job[field] for field in ("source_prior_weight", "source_prior_selection", "source_budget", "unified_training_budget") if field in job},
                        "target_batch_size": int(plan.get("target_batch_size", 32)),
                        "batch_rule": plan["batch_rule"],
                        "target_epochs": (job["unified_training_budget"]["target_epochs"] if unified
                            else int(plan.get("correctness_target_epochs", 1)) if plan.get("evidence_scope") == "software_correctness" else 60),
                        "target_lr": 1e-3,
                        **({"global_early_stopping": plan["global_early_stopping"]}
                           if plan.get("global_early_stopping") is not None else {}),
                        "source_arm": job["source_arm"] + ("_" + selector if "selectors" in job else ""),
                        "source_selector": selector, "source_layout": job["layout"],
                    }
                    plan_path = output / "outer_plans" / f"{name}.json"
                    atomic_write_json(plan_path, evaluation_plan)
                    result_path = output / "outer_results" / name
                    command = [sys.executable, str(ROOT / "experiments/run_cross_dataset_transfer.py"),
                               "--plan", str(plan_path.resolve()), "--subjects", ",".join(view["subjects"]),
                               "--output", str(result_path.resolve()), "--device", device,
                               "--cpu-threads", str(native_threads)]
                    if job.get("target_stage"):
                        command.extend(("--stage", job["target_stage"]))
                    final_jobs.append({"plan": str(plan_path.resolve()), "argv": command,
                                       "source_arm": evaluation_plan["source_arm"], "shards": [],
                                       "shard_subjects": []})
                    for shard in range(workers):
                        subjects = list(view["subjects"])[shard::workers]
                        if not subjects:
                            continue
                        shard_output = result_path if workers == 1 else result_path / f"shard{shard}"
                        entry = {"plan": str(plan_path.resolve()), "subjects": subjects,
                                 "output": str(shard_output), "device": device, "target_cache": view["cache"],
                                 "cpu_threads": native_threads,
                                 "target_stage": job.get("target_stage"),
                                 "gpu_slots": plan.get("_gpu_slots")}
                        grouped_jobs.setdefault((view["cache"], shard), []).append(entry)
                        final_jobs[-1]["shards"].append(str(shard_output))
                        final_jobs[-1]["shard_subjects"].append(subjects)
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

    return grouped_jobs

def _evaluate_outer_group(jobs):
    from types import SimpleNamespace

    from data.epochs import load_epoch_dataset
    from research.source_execution import gpu_task_slot, release_source_worker
    from train.runtime import cpu_thread_budget

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.run_cross_dataset_transfer import execute as evaluate
    from transfer.result_reuse import TargetResultReuse

    results = []
    reuse = TargetResultReuse()
    threads = jobs[0].get("cpu_threads", 1)
    with cpu_thread_budget(threads):
        dataset = load_epoch_dataset(jobs[0]["target_cache"], require_labels=True, validation="attested")
        target_prefixes = None
        if jobs[0].get("target_stage") == "unified":
            from data.source_records import load_target_prefixes

            first_plan = json.loads(Path(jobs[0]["plan"]).read_text(encoding="utf-8"))
            target_prefixes = load_target_prefixes(first_plan["target_prefix_manifest"],
                subjects=tuple(sorted({subject for job in jobs for subject in job["subjects"]})), load_inner=False)
        for job in jobs:
            if job["target_cache"] != jobs[0]["target_cache"]:
                raise ValueError("outer worker group must use one attested target cache")
            result_path = Path(job["output"])
            with gpu_task_slot(job.get("gpu_slots")):
                try:
                    evaluate(SimpleNamespace(plan=Path(job["plan"]), subjects=",".join(job["subjects"]),
                        output=result_path, device=job["device"], cpu_threads=threads,
                        stage=job.get("target_stage"), dry_run=False,
                        result_reuse=reuse, target_dataset=dataset, target_prefixes=target_prefixes))
                finally:
                    release_source_worker(job["device"])
            record = json.loads((result_path / "record.json").read_text(encoding="utf-8"))
            results.append({"output": str(result_path), "status": record["status"],
                "completed_cells": sum(cell["status"] == "completed" for cell in record["cells"]),
                "failed_cells": sum(cell["status"] != "completed" for cell in record["cells"]),
                "reused_cells": sum("prediction_reuse" in cell for cell in record["cells"])})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=("S1", "E2", "S2", "P2_source", "unified_source"), default=["S1", "E2", "S2"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluate-outer", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, help="Total CPU budget; default uses the process quota")
    parser.add_argument("--source-workers", type=int, help="Independent source groups; default automatic")
    args = parser.parse_args(argv)
    from train.runtime import cpu_thread_budget

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

    for key in ("source_snapshot_manifest", "evaluation_snapshot_manifest", "batch_rule_file",
                "prepared_source_directory", "source_view_working_directory", "verified_cache_snapshot_directory"):
        if key in plan:
            plan[key] = resolve_path(plan[key])
    if "source_preparation_manifests" in plan:
        plan["source_preparation_manifests"] = {study: resolve_path(path)
            for study, path in plan["source_preparation_manifests"].items()}
    if "ready_consumers" in plan:
        plan["ready_consumers"] = [{**consumer, "plan": resolve_path(consumer["plan"]),
                                    "output": resolve_path(consumer["output"])}
                                   for consumer in plan["ready_consumers"]]
    if plan.get("resume_source_run") is not None:
        plan["resume_source_run"] = {**plan["resume_source_run"],
                                    "inventory": resolve_path(plan["resume_source_run"]["inventory"])}
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
