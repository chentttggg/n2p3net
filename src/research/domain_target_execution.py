"""Domain target groups using the shared CPU, host-memory and GPU leases."""
from __future__ import annotations

import hashlib
import multiprocessing
import time
from pathlib import Path

from research.cloud_execution import CloudExecutor, CloudTask, target_phase_resources
from research.execution import atomic_write_json
from research.source_execution import (
    HostMemoryAdmission,
    resolve_source_resources,
    with_cloud_resource_floor,
)
from train.runtime import cpu_thread_budget


def evaluate_domain_target_group(jobs):
    from experiments.run_source_training_chain import _evaluate_outer_group

    return _evaluate_outer_group(jobs)


def execute_cloud_domain_targets(groups, *, plan, source_policy, output, progress, started, device):
    groups = list(groups)
    try:
        return _run_cloud_domain_targets(groups, plan=plan, source_policy=source_policy,
            output=output, progress=progress, started=started, device=device)
    except BaseException as error:
        if progress.get("status") not in {"failed", "interrupted"}:
            failure = {"type": type(error).__name__, "error": str(error)}
            outcomes = progress.get("target_group_outcomes", [{"index": index, "status": "pending", "jobs": group}
                for index, group in enumerate(groups)])
            for row in outcomes:
                if row["status"] == "pending":
                    row.update(status="failed", failure=failure)
                    progress["failed_groups"].append({"index": row["index"], "failure": failure})
            progress.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                requested_target_group_count=len(groups), requested_target_job_count=sum(map(len, groups)),
                target_group_outcomes=outcomes, controller_failure=failure, elapsed_seconds=time.time() - started)
            try:
                atomic_write_json(Path(output) / "matrix.json", progress)
            except BaseException as report_error:
                error.add_note(f"terminal domain matrix report failed: {report_error}")
        raise


def _run_cloud_domain_targets(groups, *, plan, source_policy, output, progress, started, device):
    """Keep ready cache groups parallel and retain the full requested ledger."""
    groups, output = list(groups), Path(output)
    native = source_policy.get("target_native_threads_per_worker",
        max(1, source_policy["worker_cpu_budget"] // source_policy["target_workers"] - 1))
    resource_plan = {**plan, "cpu_budget": source_policy["cpu_budget"],
        "source_workers": source_policy["source_workers"], "target_workers": source_policy["target_workers"],
        "gpu_workers": source_policy["gpu_slots"], "target_native_threads_per_worker": native}
    resources = resolve_source_resources(with_cloud_resource_floor(resource_plan, device), max(1, len(groups)), device)
    phase = target_phase_resources(resources, device)
    tasks = [CloudTask(f"domain-target-{index:04d}", "research.domain_target_execution:evaluate_domain_target_group",
        group, str(output / "target_workers" / f"group{index}"), phase) for index, group in enumerate(groups)]
    if any(job["cpu_threads"] != native for group in groups for job in group):
        raise ValueError("domain target jobs must use the native width held by their shared owner")
    indices = {task.task_id: index for index, task in enumerate(tasks)}
    outcomes = {index: {"index": index, "status": "pending", "jobs": group}
                for index, group in enumerate(groups)}
    progress.update(resource_owner="source_family_resource_owner", resource_policy=resources,
        requested_target_group_count=len(groups), requested_target_job_count=sum(map(len, groups)))

    def write():
        progress.update(target_group_outcomes=[outcomes[index] for index in sorted(outcomes)],
            elapsed_seconds=time.time() - started)
        atomic_write_json(output / "matrix.json", progress)

    def retained_records(group):
        records = []
        for job in group:
            path = Path(job["output"]) / "record.json"
            if path.is_file():
                records.append({"path": str(path), "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "observation_only_not_admission": True})
        return records

    def on_finished(result):
        index = indices[result.task_id]
        row = outcomes[index]
        row.update(status=result.status, supplementary_failures=result.supplementary_failures)
        if result.status == "completed":
            row["results"] = result.value
            progress["completed_groups"].append({"index": index, "results": result.value})
        else:
            row.update(failure=result.failure, retained_result_records=retained_records(groups[index]))
            progress["failed_groups"].append({"index": index, "failure": result.failure})
        write()
        return ()

    def on_observation(observation):
        ready = (set(observation.get("running", ())) | set(observation.get("pending", ()))
                 | set(observation.get("admitting", ()))) & indices.keys()
        progress.update(resource_observation=observation["resources"], actual_ready_target_groups=len(ready),
            dependency_tail_below_gpu_floor=len(ready) < resources["cloud_resource_floor"]["gpu_workers"])
        write()

    write()
    with cpu_thread_budget(resources["coordinator_cpu_threads"]), multiprocessing.get_context("spawn").Manager() as manager:
        admission = HostMemoryAdmission(manager, resources)
        executor = CloudExecutor(admission, max_workers=resources["target_workers"],
            diagnostic_directory=output / "target_worker_diagnostics",
            identity={"program": progress["program"], "resource_scope": "cloud", "resources": resources,
                "original_scientific_adapter": "experiments.run_source_training_chain:_evaluate_outer_group"})
        try:
            run = executor.run(tasks, on_finished=on_finished, on_observation=on_observation)
        except BaseException as error:
            failure = {"type": type(error).__name__, "error": str(error)}
            for index, row in outcomes.items():
                if row["status"] == "pending":
                    row.update(status="failed", failure=failure, retained_result_records=retained_records(groups[index]))
                    progress["failed_groups"].append({"index": index, "failure": failure})
            progress.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                resource_observation=admission.record(), controller_failure=failure)
            try:
                write()
            except BaseException as report_error:
                error.add_note(f"terminal domain matrix report failed: {report_error}")
            raise
        for index, row in outcomes.items():
            if row["status"] == "pending":
                row.update(status="failed", failure={"reason": "target_group_not_completed"})
                progress["failed_groups"].append({"index": index, "failure": row["failure"]})
        failures = sum(row["failed_cells"] for group in progress["completed_groups"] for row in group["results"])
        progress.update(status="failed" if failures or progress["failed_groups"] or run.status != "completed" else "completed",
            failed_cells=failures, resource_observation=admission.record(), worker_diagnostics=run.diagnostics,
            observation_failures=run.observation_failures)
        write()
    if progress["status"] != "completed":
        raise RuntimeError("target matrix contains failures; complete requested groups and completed results remain available")
    return resources
