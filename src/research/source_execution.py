"""Resource admission and process-safe source trajectory reuse."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from research.execution import atomic_write_json
from train.runtime import available_cpu_threads


@contextmanager
def process_file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            while True:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class SharedTrajectoryRegistry:
    """Lock one semantic signature while fitting or publishing its checkpoints."""

    def __init__(self, directory):
        self.directory = Path(directory)

    @contextmanager
    def transaction(self, signature):
        if len(signature) != 64 or any(value not in "0123456789abcdef" for value in signature):
            raise ValueError("trajectory registry signature must be SHA-256")
        with process_file_lock(self.directory / f"{signature}.lock"):
            path = self.directory / f"{signature}.json"
            entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            for entry in entries:
                entry["checkpoints"] = {int(key): value for key, value in entry["checkpoints"].items()}
            local = {signature: entries}
            yield local
            atomic_write_json(path, local[signature])


@contextmanager
def gpu_task_slot(semaphore):
    if semaphore is None:
        yield
        return
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def release_source_worker(device):
    if torch.device(device).type == "cuda":
        torch.compiler.reset()
        from train.device import release_device_memory

        release_device_memory(torch.device(device))


def available_host_memory_bytes():
    import psutil

    limits = [psutil.virtual_memory().available]
    try:
        quota = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        used = int(Path("/sys/fs/cgroup/memory.current").read_text())
        if quota != "max":
            limits.append(max(0, int(quota) - used))
    except (OSError, ValueError):
        pass
    return min(limits)


class HostMemoryAdmission:
    """Reserve whole process groups before loading datasets or spawning children."""

    def __init__(self, manager, resources):
        self.state = manager.Namespace()
        self.state.free = resources["host_memory_budget_bytes"]
        self.state.groups = 0
        self.state.child_cpus = 0
        self.lock = manager.RLock()
        self.group_bytes = resources["source_worker_memory_bytes"]
        self.target_bytes = resources["target_worker_memory_bytes"]
        self.target_limit = resources["target_workers"]
        self.cpu_budget = resources["cpu_budget"]

    @contextmanager
    def source_group(self):
        while True:
            with self.lock:
                if (self.state.free >= self.group_bytes
                        and self.state.groups + self.state.child_cpus < self.cpu_budget):
                    self.state.free -= self.group_bytes
                    self.state.groups += 1
                    break
            time.sleep(0.1)
        try:
            yield
        finally:
            with self.lock:
                self.state.free += self.group_bytes
                self.state.groups -= 1

    @contextmanager
    def target_workers(self, requested):
        with self.lock:
            desired = min(self.target_limit, max(requested, self.cpu_budget // max(1, self.state.groups)))
            cpu_free = self.cpu_budget - self.state.groups - self.state.child_cpus + 1
            allowed = min(desired, self.state.free // self.target_bytes, cpu_free)
            workers = int(allowed) if allowed >= 2 else 1
            reservation = workers * self.target_bytes if workers > 1 else 0
            self.state.free -= reservation
            self.state.child_cpus += workers - 1
        try:
            yield workers
        finally:
            with self.lock:
                self.state.free += reservation
                self.state.child_cpus -= workers - 1


def resolve_source_resources(plan, group_count, device, *, cpu_count=None, free_device_bytes=None, free_host_bytes=None):
    available = available_cpu_threads() if cpu_count is None else cpu_count
    if group_count < 1 or available < 1:
        raise ValueError("task count and available CPU budget must be positive")
    if plan.get("cpu_budget") is not None and (type(plan["cpu_budget"]) is not int or plan["cpu_budget"] < 1):
        raise ValueError("CPU budget must be a positive integer")
    budget = min(available, plan.get("cpu_budget") or available)
    requested = plan.get("source_workers", "auto")
    if requested not in (None, "auto") and (type(requested) is not int or requested < 1):
        raise ValueError("source_workers must be auto or a positive integer")
    groups = min(group_count, budget, group_count if requested in (None, "auto") else requested)
    memory_gib = float(plan.get("gpu_task_memory_gib", 3.0))
    if not 0 < memory_gib < float("inf") or budget < 1:
        raise ValueError("resource budgets must be positive and finite")
    if free_host_bytes is None:
        free_host_bytes = available_host_memory_bytes()
    host_budget = int(free_host_bytes * 0.85)
    group_bytes = int(float(plan.get("source_worker_memory_gib", 6.0)) * 2**30)
    target_bytes = int(float(plan.get("target_worker_memory_gib", 4.0)) * 2**30)
    if min(group_bytes, target_bytes) <= 0 or host_budget < max(group_bytes, target_bytes):
        raise ValueError("available host memory is below the declared worker requirement")
    groups = min(groups, host_budget // group_bytes)
    if torch.device(device).type == "cuda":
        if free_device_bytes is None:
            free_device_bytes = torch.cuda.mem_get_info(torch.device(device))[0]
        slots = max(1, min(budget, int(0.8 * free_device_bytes / (memory_gib * 2**30))))
    else:
        slots = budget
    target_request = plan.get("target_workers", "auto")
    if target_request not in (None, "auto") and (type(target_request) is not int or target_request < 1):
        raise ValueError("target_workers must be auto or a positive integer")
    target = min(host_budget // target_bytes, slots if target_request in (None, "auto") else min(target_request, slots))
    return {"policy": "automatic_resource_admission", "available_cpu_threads": available,
            "cpu_budget": budget, "requested_source_workers": requested or "auto",
            "source_groups": group_count, "source_workers": groups,
            "gpu_slots": slots, "target_workers": target,
            "inner_workers_per_group": min(target, max(1, (budget - groups) // min(groups, 4))),
            "native_threads_per_worker": 1, "compiler_threads_per_worker": 1,
            "gpu_task_memory_gib": memory_gib, "free_device_bytes": free_device_bytes,
            "available_host_memory_bytes": free_host_bytes, "host_memory_budget_bytes": host_budget,
            "source_worker_memory_bytes": group_bytes, "target_worker_memory_bytes": target_bytes}
