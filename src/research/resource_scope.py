"""Execution scope observations, separate from the source-family admission owner."""
from __future__ import annotations

import sys
from dataclasses import asdict

LOCAL_RESOURCE_SCOPES = ("local_development", "software_correctness")


def _managed_worker():
    return getattr(sys.modules.get("research.cloud_execution"), "_CURRENT_WORKER_PHASES", None)


def require_standalone_resource_scope(plan):
    """A direct standalone CLI cannot turn a cloud declaration into admission."""
    declared = plan.get("resource_scope", "local_development")
    managed = _managed_worker()
    if managed is not None:
        if declared == "cloud" or plan.get("cloud_resource_floor") is not None:
            from research.source_execution import with_cloud_resource_floor

            # Scope is explicitly cloud here. Query its shared policy without
            # selecting or initializing a worker's actual compute device.
            floor = with_cloud_resource_floor({"resource_scope": "cloud",
                "cloud_resource_floor": plan.get("cloud_resource_floor")}, "cuda")["cloud_resource_floor"]
            if (managed.admission.cpu_budget < floor["cpu_threads"]
                    or managed.admission.gpu_worker_limit < floor["gpu_workers"]):
                raise ValueError("the actual source-family owner cannot meet the central cloud resource floor")
        return
    if declared not in LOCAL_RESOURCE_SCOPES or plan.get("cloud_resource_floor") is not None:
        raise ValueError("standalone execution is local/development; cloud execution requires the source-family resource owner")


def entrypoint_resource_record(*, entrypoint, requested_device, effective_device,
                               requested_cpu_threads=None, declared_scope="local_development", execution_performed=True,
                               configured_compute_workers=1):
    """Record effective resources; device choice never identifies a cloud host.

    A real source-family worker can retain its owner phase in the observation.
    The observation itself grants no resource admission or data permission.
    """
    import torch
    from threadpoolctl import threadpool_info

    from train.runtime import available_cpu_threads

    if declared_scope not in LOCAL_RESOURCE_SCOPES:
        raise ValueError("direct CLI resource scope must be local_development or software_correctness")
    managed = _managed_worker()
    phase = asdict(managed.task.resources) if managed is not None else None
    return {"schema": "n2p3_entrypoint_effective_resources/1",
        "entrypoint": str(entrypoint), "entrypoint_default_scope": declared_scope,
        "resource_scope": "source_family_managed" if managed is not None else declared_scope,
        "execution_performed": bool(execution_performed),
        "requested_device": str(requested_device), "effective_device": str(effective_device),
        "requested_cpu_threads": requested_cpu_threads, "effective_torch_threads": torch.get_num_threads(),
        "effective_torch_interop_threads": torch.get_num_interop_threads(),
        "available_process_cpu_threads": available_cpu_threads(),
        "effective_native_pools": [{key: value[key] for key in ("user_api", "internal_api", "prefix", "num_threads")
            if key in value} for value in threadpool_info()],
        "configured_compute_workers": configured_compute_workers, "observed_cuda_initialized": torch.cuda.is_initialized(),
        "source_family_task_id": managed.task.task_id if managed is not None else None,
        "source_family_phase": phase,
        "owner_cpu_budget": managed.admission.cpu_budget if managed is not None else None,
        "owner_gpu_slots": managed.admission.gpu_worker_limit if managed is not None else None,
        "observation_only_not_cloud_admission": True}
