"""One owned CPU programme actor, using the existing cloud family resource owner.

The project capacity remains CUDA-capable with its central CPU/GPU floor. The
actual NumPy source programme is one CPU phase, with zero GPU work or GPU claim.
Its internal threads retain the original same-process publication registries.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

from research.cloud_execution import CloudExecutor, CloudTask, PhaseResources
from research.execution import atomic_write_json
from research.source_execution import (
    CLOUD_RESOURCE_CONFIG,
    HostMemoryAdmission,
    resolve_source_resources,
    with_cloud_resource_floor,
)


def programme_requirements(resources):
    if (resources.get("schema") != "public_population_source_resources/3"
            or resources.get("resource_scope") != "cloud" or resources.get("device") != "cpu"):
        raise ValueError("source fitting requires the owned cloud CPU programme declaration")
    names = ("source_workers", "fold_workers", "native_threads_per_worker", "cpu_threads_total",
        "source_worker_reservation_bytes", "fold_worker_reservation_bytes", "shared_programme_reservation_bytes")
    if any(type(resources.get(name)) is not int or resources[name] < 1 for name in names):
        raise ValueError("cloud programme CPU widths and complete RAM reservations must be positive integers")
    concurrency = max(resources["source_workers"], resources["fold_workers"])
    native = resources["native_threads_per_worker"]
    if concurrency > 13 or concurrency * native + 1 > resources["cpu_threads_total"]:
        raise ValueError("the whole source programme must fit its pool and central controller CPU budget")
    headroom = resources.get("memory_headroom_fraction")
    if isinstance(headroom, bool) or not isinstance(headroom, (float, int)) or not .05 <= headroom < 1:
        raise ValueError("the cloud programme requires at least five percent effective host headroom")
    memory = max(resources["source_workers"] * resources["source_worker_reservation_bytes"],
        resources["fold_workers"] * resources["fold_worker_reservation_bytes"])
    memory += resources["shared_programme_reservation_bytes"]
    return {"concurrency": concurrency, "native_threads": native,
        "programme_cpu_threads": concurrency * native, "programme_memory_bytes": memory,
        "ready_gpu_tasks": 0, "programme_gpu_claim": 0}


def resolve_programme_owner(resources, *, cpu_count=None, free_device_bytes=None, free_host_bytes=None):
    """Resolve real project capacity, then declare this actual CPU-only phase.

    The optional observations are the existing resolver's metadata seam, used
    for controls only. A real cloud launch uses its actual runtime observations.
    """
    requested = programme_requirements(resources)
    central = json.loads(CLOUD_RESOURCE_CONFIG.read_text(encoding="utf8"))
    maximum = max(resources["source_worker_reservation_bytes"], resources["fold_worker_reservation_bytes"])
    plan = {**central, "resource_scope": "cloud", "cpu_budget": resources["cpu_threads_total"],
        "source_workers": requested["concurrency"], "target_workers": central["cloud_resource_floor"]["gpu_workers"],
        "source_worker_memory_gib": maximum / 2**30, "shared_staging_memory_gib": 0,
        "host_memory_fraction": 1 - resources["memory_headroom_fraction"]}
    if resources.get("cloud_resource_floor") is not None:
        plan["cloud_resource_floor"] = resources["cloud_resource_floor"]
    # This is project capacity admission, identical to the existing cloud owner.
    # Actual work below has PhaseResources.device == 'cpu', so claims no GPU.
    capacity = resolve_source_resources(with_cloud_resource_floor(plan, "cuda"), 13, "cuda",
        cpu_count=cpu_count, free_device_bytes=free_device_bytes, free_host_bytes=free_host_bytes)
    if (capacity["cpu_budget"] != resources["cpu_threads_total"]
            or capacity["source_workers"] < requested["concurrency"]
            or requested["programme_cpu_threads"] + capacity["coordinator_cpu_threads"] > capacity["cpu_budget"]
            or requested["programme_memory_bytes"] > capacity["host_memory_budget_bytes"] - capacity["shared_staging_memory_bytes"]):
        raise ValueError("the actual shared owner cannot admit the complete source programme phase")
    phase = PhaseResources("source", requested["programme_cpu_threads"], requested["programme_memory_bytes"],
        requested["native_threads"], "cpu")
    return capacity, phase


def require_programme_phase(resources, capacity, managed):
    requested = programme_requirements(resources)
    phase = managed.task.resources
    if (managed.admission.cpu_budget != capacity["cpu_budget"]
            or managed.admission.gpu_worker_limit != capacity["gpu_slots"]
            or resources["cpu_threads_total"] != capacity["cpu_budget"]
            or phase.phase != "source" or phase.device != "cpu"
            or phase.native_threads != requested["native_threads"]
            or phase.cpu_threads < requested["programme_cpu_threads"]
            or phase.memory_bytes < requested["programme_memory_bytes"]):
        raise ValueError("the actual cloud programme phase does not cover its internal source/fold pools")
    return {"project_capacity_device": "cuda", "actual_programme_device": "cpu",
        "actual_programme_phase": asdict(phase), "ready_gpu_tasks": 0, "programme_gpu_claim": 0,
        "internal_source_workers": resources["source_workers"], "internal_fold_workers": resources["fold_workers"],
        "member_leases": "covered by the outer aggregate phase; no nested double reservation"}


def execute_population_programme(arguments):
    """The only actual scientific entry runs in one original managed process."""
    from experiments.run_public_population_source import main

    return main(arguments["argv"], cloud_owner_resources=arguments["capacity"])


def run_population_source_cloud(*, argv, resources, output, identity):
    """Controller entry. Invoke only after the user supplies cloud and authorizes it."""
    output = Path(output).resolve()
    controller = output.with_name(output.name + ".cloud")
    if output.exists() or controller.exists():
        raise FileExistsError("the cloud source programme requires new scientific and controller output directories")
    capacity, phase = resolve_programme_owner(resources)
    # Claim the controller path once, so simultaneous launches cannot replace
    # each other's terminal receipt while the scientific actor owns `output`.
    controller.mkdir(parents=True, exist_ok=False)
    identity = {**identity, "programme": "full_13_person_13_fold_population_source",
        "resource_owner": "source_family_resource_owner", "project_capacity_device": "cuda",
        "actual_programme_device": "cpu", "programme_phase": asdict(phase), "ready_gpu_tasks": 0,
        "programme_gpu_claim": 0, "internal_source_workers": resources["source_workers"],
        "internal_fold_workers": resources["fold_workers"], "independent_acceptance_issued": False}
    task = CloudTask("population-source-complete", "research.population_source_cloud:execute_population_programme",
        {"argv": list(argv), "capacity": capacity}, str(output), phase)

    def acknowledge(outcome):
        if outcome.status != "completed":
            return ()
        reference = outcome.value
        content = Path(reference["path"]).read_bytes()
        if len(content) != reference["bytes"] or hashlib.sha256(content).hexdigest() != reference["sha256"]:
            raise ValueError("the retired actor's actual source publication changed")
        record = json.loads(content)
        if (record["schema"] != "complete_original_population_source_execution/1"
                or record["complete"] is not True or len(record["persons"]) != 13 or len(record["folds"]) != 13):
            raise ValueError("cloud completion requires the actual complete thirteen-person/thirteen-fold publication")
        return ()

    with get_context("spawn").Manager() as manager:
        admission = HostMemoryAdmission(manager, capacity)
        executor = CloudExecutor(admission, max_workers=1, diagnostic_directory=controller, identity=identity)
        run = executor.run((task,), on_finished=acknowledge)
        receipt = {"schema": "owned_cloud_population_source_programme/1", "identity": identity,
            "capacity": capacity, "results": {key: asdict(value) for key, value in run.results.items()},
            "status": "complete_pending_independent_acceptance" if run.status == "completed" else "failed",
            "diagnostics": run.diagnostics, "observation_failures": run.observation_failures,
            "settled_resource_observation": admission.record(), "source_requested": 13, "fold_requested": 13,
            "ready_gpu_tasks": 0, "programme_gpu_claim": 0, "independent_acceptance_issued": False}
        atomic_write_json(controller / "SOURCE_CLOUD_RECEIPT.json", receipt)
    if run.status != "completed":
        raise RuntimeError("the owned cloud source programme failed; original outputs and owner diagnostics retained")
    return run.results[task.task_id].value
