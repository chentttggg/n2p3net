"""The legacy source-pool scientific adapter under the shared resource owner."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import time
from pathlib import Path

from data.epochs import parse_preprocessing_spec
from data.source_admission import PreparedSourceInventory, bind_source_inputs
from research.cloud_execution import CloudExecutor, CloudTask, PhaseResources
from research.execution import atomic_write_json
from research.source_execution import (
    CLOUD_RESOURCE_CONFIG,
    HostMemoryAdmission,
    resolve_source_resources,
    source_phase_resources,
)
from train.runtime import available_cpu_threads, cpu_thread_budget


def wait_source_pool_input(arguments):
    directory, study, preprocessing_json, timeout = arguments
    inventory = PreparedSourceInventory(directory,
        preprocessing=parse_preprocessing_spec(json.loads(preprocessing_json)))
    deadline = time.monotonic()+timeout
    while True:
        state = inventory.inspect([study])
        if study in state["ready"]:
            return state["ready"][study]
        if study in state["failed"]:
            raise ValueError(state["failed"][study])
        if time.monotonic() >= deadline:
            raise ValueError(f"preparation wait expired: {state['pending'][study]}")
        time.sleep(min(1., max(.01, deadline-time.monotonic())))


def prepare_source_pool_dose(arguments):
    import numpy as np

    from experiments.run_source_pool_matrix import source_cache
    from research.source_matrix import freeze_matched_dose
    from research.source_selection import source_identity_axes

    tasks, prepared, preprocessing_json, output = arguments
    studies = sorted({study for task in tasks for study in task["studies"]})
    counts = {}
    for study in studies:
        data = source_cache(prepared[study]["cache"], prepared[study]["cache_sha256"], preprocessing_json)
        _, people, _ = source_identity_axes(data)
        counts[study] = {person: np.bincount(data.y[people == person], minlength=2).tolist()
                         for person in sorted(set(people))}
        del data, people
        source_cache.cache_clear()
    result = freeze_matched_dose(counts, tasks=tasks)
    atomic_write_json(Path(output)/"matched_dose.json", result)
    return result


def execute_original_source_pool_group(arguments):
    """Keep the original actual scientific fit/seed/dose implementation."""
    from experiments.run_source_pool_matrix import execute_group

    records = execute_group(arguments)
    output = Path(arguments[2])
    return [{"id": row["task"]["id"], "status": row["status"],
        "record": str(output/row["task"]["id"]/"result.json"),
        "record_sha256": hashlib.sha256((output/row["task"]["id"]/"result.json").read_bytes()).hexdigest()}
        for row in records]


def execute_cloud_source_pool(*, groups, tasks, dose_tasks, prepared_directory, initial,
                              output, snapshot, config, preprocessing_json, workers,
                              threads_per_worker, device, wait_timeout, initial_bindings=None):
    """Schedule ready original groups through central floor and actual leases."""
    groups = list(groups)
    if not groups or threads_per_worker < 2:
        raise ValueError("cloud source pools require ready group declarations and one native plus one inter-op CPU thread")
    floor = json.loads(CLOUD_RESOURCE_CONFIG.read_text(encoding="utf8"))["cloud_resource_floor"]
    resource_plan = {"resource_scope": "cloud", "cloud_resource_floor": floor,
        "cpu_budget": available_cpu_threads(), "source_workers": workers, "gpu_workers": workers,
        "target_workers": floor["gpu_workers"], "source_native_threads_per_worker": threads_per_worker-1}
    resources = resolve_source_resources(resource_plan, len(groups), device)
    output = Path(output)
    atomic_write_json(output/"resource_policy.json", {"resource_scope": "cloud",
        "owner": "source_family_resource_owner", **resources})
    prepared, failed_inputs = dict(initial["ready"]), dict(initial["failed"])
    bindings = bind_source_inputs(dict(initial_bindings or {}), prepared)
    dose_studies = {study for task in dose_tasks for study in task["studies"]}
    required = {study for group in groups for task in group for study in task["studies"]} | dose_studies
    scheduled, group_tasks, results = set(), {}, {}
    dose, dose_scheduled = None, False
    phases = source_phase_resources(resources, device)
    preparation = PhaseResources("preparation", 1, min(64*2**20, resources["source_worker_memory_bytes"]), 1)
    dose_preparation = PhaseResources("preparation", 1, resources["source_worker_memory_bytes"], 1)
    waiting = {f"pool-input-{index}": study for index, study in enumerate(sorted(required-set(prepared)))}
    initial_tasks = [CloudTask(key, "research.source_pool_cloud:wait_source_pool_input",
        (str(prepared_directory), study, preprocessing_json, wait_timeout),
        str(output/"cloud/input"/study), preparation) for key, study in waiting.items()]

    def ready_tasks():
        nonlocal dose_scheduled
        additions = []
        if dose_tasks and not dose_scheduled and dose_studies <= set(prepared):
            dose_scheduled = True
            additions.append(CloudTask("pool-dose", "research.source_pool_cloud:prepare_source_pool_dose",
                (dose_tasks, dict(prepared), preprocessing_json, str(output)), str(output/"cloud/dose"), dose_preparation))
        for index, group in enumerate(groups):
            if index in scheduled or not set(group[0]["studies"]) <= set(prepared):
                continue
            if group[0]["dose"] == "matched" and dose is None:
                continue
            key = f"pool-source-{index:04d}"
            scheduled.add(index)
            group_tasks[key] = group
            arguments = (group, dict(prepared), str(output), snapshot, config, dose,
                phases.native_threads, preprocessing_json, device)
            additions.append(CloudTask(key, "research.source_pool_cloud:execute_original_source_pool_group",
                arguments, str(output/"cloud"/key), phases))
        return additions

    def on_finished(outcome):
        nonlocal dose, bindings
        if outcome.task_id in waiting:
            study = waiting[outcome.task_id]
            if outcome.status == "completed":
                prepared[study] = outcome.value
                bindings = bind_source_inputs(bindings, {study: outcome.value})
                atomic_write_json(output/"input_bindings.json", bindings)
            else:
                failed_inputs[study] = outcome.failure
        elif outcome.task_id == "pool-dose":
            if outcome.status == "completed":
                dose = outcome.value
            else:
                failed_inputs["matched_dose"] = outcome.failure
        elif outcome.task_id in group_tasks:
            if outcome.status == "completed":
                results.update({row["id"]: row for row in outcome.value})
            else:
                for task in group_tasks[outcome.task_id]:
                    path = output/task["id"]/"result.json"
                    row = {"id": task["id"], "status": "failed", "failure": outcome.failure}
                    if path.is_file():
                        row["retained_result_record"] = {"path": str(path), "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "observation_only_not_admission": True,
                            "completion_owner": "execute_group/fit_or_reuse_source_trajectory"}
                    results[task["id"]] = row
        return ready_tasks()

    with cpu_thread_budget(resources["coordinator_cpu_threads"]), multiprocessing.get_context("spawn").Manager() as manager:
        admission = HostMemoryAdmission(manager, resources)

        def on_observation(observation):
            unscheduled_ready = sum(set(group[0]["studies"]) <= set(prepared)
                and (group[0]["dose"] != "matched" or dose is not None) for index, group in enumerate(groups)
                if index not in scheduled)
            running = set(observation.get("running", ()))
            pending = set(observation.get("pending", ()))
            admitting = set(observation.get("admitting", ()))
            owned_ready = (running | pending | admitting) & set(group_tasks)
            waiting_ready = len((pending-running-admitting) & set(group_tasks))+unscheduled_ready
            ready = len(owned_ready)+unscheduled_ready
            active = observation["resources"]["source_groups"]
            atomic_write_json(output/"progress.json", {"status": "running", "resource_scope": "cloud",
                "owner": "source_family_resource_owner", "requested_tasks": len(tasks),
                "results": [results[key] for key in sorted(results)], "actual_source_group_claims": active,
                "waiting_ready_source_groups": waiting_ready, "actual_ready_source_groups": ready,
                "dependency_tail_below_gpu_floor": ready < floor["gpu_workers"],
                "resource_observation": observation["resources"], "failed_inputs": failed_inputs})

        executor = CloudExecutor(admission, max_workers=resources["worker_cpu_budget"],
            diagnostic_directory=output/"worker_diagnostics", identity={"source_snapshot": snapshot,
                "resource_scope": "cloud", "resources": resources, "original_scientific_adapter": "execute_group"})
        try:
            run = executor.run([*initial_tasks, *ready_tasks()], on_finished=on_finished, on_observation=on_observation)
        except BaseException as error:
            failure = {"type": type(error).__name__, "error": str(error), "stage": "source_pool_controller"}
            for task in tasks:
                if task["id"] not in results:
                    results[task["id"]] = {"id": task["id"], "status": "failed", "failure": failure}
            record = {"status": "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                "resource_scope": "cloud", "owner": "source_family_resource_owner", "requested_tasks": len(tasks),
                "results": [results[task["id"]] for task in tasks], "resource_observation": admission.record(),
                "failed_inputs": failed_inputs, "controller_failure": failure}
            try:
                atomic_write_json(output/"progress.json", record)
            except BaseException as report_error:
                error.add_note(f"terminal source-pool report failed: {report_error}")
            raise
        for task in tasks:
            if task["id"] not in results:
                results[task["id"]] = {"id": task["id"], "status": "failed",
                    "reason": "source_dependencies_not_completed", "failed_inputs": failed_inputs}
        status = "completed" if run.status == "completed" and all(row["status"] == "completed" for row in results.values()) else "failed"
        atomic_write_json(output/"progress.json", {"status": status, "resource_scope": "cloud",
            "owner": "source_family_resource_owner", "requested_tasks": len(tasks),
            "results": [results[task["id"]] for task in tasks], "active": [], "pending": [],
            "resource_observation": admission.record(), "failed_inputs": failed_inputs,
            "worker_diagnostics": run.diagnostics, "observation_failures": run.observation_failures})
    if status != "completed":
        raise RuntimeError("source matrix has unresolved failures; complete requested and actual artifact records are preserved")
    return resources
