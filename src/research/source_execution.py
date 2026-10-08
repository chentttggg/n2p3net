"""Resource admission and process-safe source trajectory reuse."""
from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Condition, RLock
from zipfile import ZipFile

from research.execution import atomic_write_json


class SourceInputPhases:
    """Record actual cache preparation/ingress costs without changing its identity."""

    def __init__(self, path):
        import psutil

        self.path = Path(path)
        self.process = psutil.Process()
        self.lock = RLock()
        self.completed = []
        self.active = {}
        self.next_id = 0

    def _write(self):
        atomic_write_json(self.path, {"schema": "n2p3_source_input_phases/1", "pid": os.getpid(),
            "updated_unix": time.time(), "active": list(self.active.values()), "completed": self.completed,
            "cpu_accounting_scope": "whole_process_including_other_active_threads"})

    @contextmanager
    def phase(self, name):
        wall = time.perf_counter()
        cpu = self.process.cpu_times()
        before_cpu = cpu.user + cpu.system
        before_rss = self.process.memory_info().rss
        with self.lock:
            token = self.next_id
            self.next_id += 1
            self.active[token] = {"name": name, "started_unix": time.time(), "rss_before_bytes": before_rss}
            self._write()
        try:
            yield
        except BaseException:
            self._finish(token, wall, before_cpu, before_rss, "failed")
            raise
        else:
            self._finish(token, wall, before_cpu, before_rss, "completed")

    def _finish(self, token, wall, before_cpu, before_rss, status):
        cpu = self.process.cpu_times()
        with self.lock:
            stage = self.active.pop(token)
            self.completed.append({**stage, "status": status, "wall_seconds": time.perf_counter() - wall,
                "process_cpu_seconds": cpu.user + cpu.system - before_cpu, "rss_before_bytes": before_rss,
                "rss_after_bytes": self.process.memory_info().rss})
            self._write()


class PreparedSourceThreadMemory:
    """Bound concurrently decoded source temporaries beside final resident rows."""

    def __init__(self, *, budget_bytes, resident_bytes, temporary_bytes, disk_bytes=None, snapshot_bytes=()):
        self.capacity = budget_bytes - resident_bytes
        if self.capacity < max(temporary_bytes, default=0):
            raise ValueError("source memory budget cannot hold the complete final view and one admitted source temporary")
        if disk_bytes is not None and disk_bytes < max(snapshot_bytes, default=0):
            raise ValueError("source snapshot filesystem cannot hold one admitted input cache")
        self.free = self.capacity
        self.disk_free = disk_bytes
        self.peak_reserved = 0
        self.peak_snapshot = 0
        self.disk_capacity = disk_bytes
        self.condition = Condition()

    @contextmanager
    def reserve(self, nbytes, snapshot_bytes=0):
        with self.condition:
            self.condition.wait_for(lambda: self.free >= nbytes and (self.disk_free is None or self.disk_free >= snapshot_bytes))
            self.free -= nbytes
            if self.disk_free is not None:
                self.disk_free -= snapshot_bytes
                self.peak_snapshot = max(self.peak_snapshot, self.disk_capacity - self.disk_free)
            self.peak_reserved = max(self.peak_reserved, self.capacity - self.free)
        try:
            yield
        finally:
            with self.condition:
                self.free += nbytes
                if self.disk_free is not None:
                    self.disk_free += snapshot_bytes
                self.condition.notify_all()


def filesystem_uses_host_memory(directory):
    """Identify tmpfs/ramfs backing the actual snapshot or output directory."""
    import psutil

    directory = Path(directory).resolve(strict=True)
    mounts = []
    for partition in psutil.disk_partitions(all=True):
        mount = Path(partition.mountpoint)
        if directory == mount or mount in directory.parents:
            mounts.append((len(mount.parts), partition.fstype))
    return bool(mounts and max(mounts)[1].lower() in {"tmpfs", "ramfs"})



class PreparedSourceStorage:
    """Own final EEG storage and source decode/snapshot capacity for a caller.

    Resource decisions share one implementation; scientific adapters still
    produce each source's verified channels, reference and identity themselves.
    """

    def __init__(self, inputs, *, n_channels, output_path, snapshot_directory=None, budget_bytes=None):
        import numpy as np

        from data.epochs import _read_epoch_cache_record, _verified_cache_snapshot_directory

        self.names = [name for name, _ in inputs]
        shapes = [_read_epoch_cache_record(path)[0]["shape"] for _, path in inputs]
        if len({shape[2] for shape in shapes}) != 1:
            raise ValueError("source caches require one physical time-grid before allocation")
        final_shape = (sum(shape[0] for shape in shapes), n_channels, shapes[0][2])
        self.snapshot_directories, self.snapshot_in_memory = [], []
        self.temporary_costs, self.snapshot_costs, non_eeg_bytes = [], [], []
        for index, (_, path) in enumerate(inputs):
            with ZipFile(path) as archive:
                total = sum(info.file_size for info in archive.infolist())
                eeg = archive.getinfo("X.npy").file_size
            non_eeg_bytes.append(total - eeg)
            snapshot_bytes = path.stat().st_size
            directory = _verified_cache_snapshot_directory(path, byte_size=snapshot_bytes,
                directory_override=snapshot_directory)
            in_memory = filesystem_uses_host_memory(directory)
            self.snapshot_directories.append(directory)
            self.snapshot_in_memory.append(in_memory)
            self.snapshot_costs.append(snapshot_bytes)
            projected = shapes[index][0] * n_channels * shapes[index][2] * 4
            self.temporary_costs.append(eeg + projected + 2 * (total - eeg) + 32 * 2**20
                + (snapshot_bytes if in_memory else 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_in_memory = filesystem_uses_host_memory(output_path.parent)
        final_eeg_bytes = int(np.prod(final_shape)) * 4
        # Twice decoded metadata allows for merged Unicode widths and writer
        # materialization. This remains a capacity estimate, not a hard RSS cap.
        self.output_estimate_bytes = final_eeg_bytes + 2 * sum(non_eeg_bytes) + 32 * 2**20
        self.resident_bytes = final_eeg_bytes + 2 * sum(non_eeg_bytes)
        if self.output_in_memory:
            self.resident_bytes += self.output_estimate_bytes
        self.budget_bytes = int(available_host_memory_bytes() * .8) if budget_bytes is None else budget_bytes
        directory = self.snapshot_directories[0]
        if any(path.stat().st_dev != directory.stat().st_dev for path in self.snapshot_directories):
            raise ValueError("source production requires one verified snapshot filesystem or an explicit snapshot directory")
        reserve = max(64 * 2**20, int(max(self.snapshot_costs) * .05))
        disk_bytes = max(0, shutil.disk_usage(directory).free - reserve)
        self.memory = PreparedSourceThreadMemory(budget_bytes=self.budget_bytes, resident_bytes=self.resident_bytes,
            temporary_bytes=self.temporary_costs, disk_bytes=disk_bytes, snapshot_bytes=self.snapshot_costs)
        # Admit the complete resident view and at least one source before the
        # first large allocation. Threads receive only disjoint ordered rows.
        self.X = np.empty(final_shape, dtype=np.float32)
        offsets = np.cumsum([0, *[shape[0] for shape in shapes]])
        self.destinations = [self.X[offsets[index]:offsets[index + 1]] for index in range(len(inputs))]

    def reserve(self, index):
        return self.memory.reserve(self.temporary_costs[index], self.snapshot_costs[index])

    def record(self):
        return {"schema": "n2p3_source_input_capacity/1", "budget_bytes": self.budget_bytes,
            "resident_estimate_bytes": self.resident_bytes,
            "source_temporary_estimates_bytes": dict(zip(self.names, self.temporary_costs, strict=True)),
            "peak_admitted_temporary_bytes": self.memory.peak_reserved,
            "peak_admitted_snapshot_bytes": self.memory.peak_snapshot,
            "snapshot_in_host_memory": dict(zip(self.names, self.snapshot_in_memory, strict=True)),
            "output_in_host_memory": self.output_in_memory, "output_archive_estimate_bytes": self.output_estimate_bytes,
            "estimate_basis": "final float32 EEG plus twice decoded non-EEG; original X plus projected X plus twice non-EEG and 32MiB per source; tmpfs snapshot/output additionally charged",
            "estimated_bytes_are_not_measured_RSS": True}


class SourceInputLease:
    """Release normalized host bytes only after transfer or fit cleanup."""

    def __init__(self, owner, phase, nbytes):
        self.owner, self.phase, self.nbytes = owner, phase, nbytes
        self.active = True

    def release_unused(self, retained_bytes=0):
        if not self.active:
            return
        if not 0 <= retained_bytes <= self.nbytes:
            raise ValueError("retained source staging bytes exceed the admitted lease")
        with self.owner.condition:
            self.owner.state.free += self.nbytes - retained_bytes
            self.nbytes = retained_bytes
            if retained_bytes == 0:
                if self.phase == "nn":
                    self.owner.state.nn_active -= 1
                else:
                    self.owner.state.load_active -= 1
                self.active = False
            self.owner.condition.notify_all()


class SourceInputAdmission:
    """Shared staging bytes with NN priority and separate cache-I/O capacity.

    Complete source views and source-builder working sets retain their whole
    group reservation throughout fitting. This pool admits extra normalized
    matrices; cache-I/O leases use zero extra bytes on disk-backed snapshots.
    """

    def __init__(self, manager, *, capacity_bytes, load_workers):
        if capacity_bytes < 0 or load_workers != 1:
            raise ValueError("source input admission requires nonnegative bytes and one group I/O worker; parallel groups need a shared snapshot filesystem budget")
        self.capacity = capacity_bytes
        self.load_workers = load_workers
        self.condition = manager.Condition()
        self.state = manager.Namespace()
        self.state.free = capacity_bytes
        self.state.nn_waiting = self.state.nn_active = self.state.load_active = 0
        self.state.peak_bytes = self.state.peak_loaders = self.state.peak_nn = 0

    @contextmanager
    def reserve(self, phase, nbytes=0):
        if phase not in {"load", "nn"} or type(nbytes) is not int or not 0 <= nbytes <= self.capacity:
            raise ValueError("source input phase exceeds its shared staging byte capacity")
        with self.condition:
            if phase == "nn":
                self.state.nn_waiting += 1
                self.condition.notify_all()
            try:
                self.condition.wait_for(lambda: self.state.free >= nbytes and (
                    phase == "nn" or self.state.nn_waiting == 0 and self.state.load_active < self.load_workers))
                self.state.free -= nbytes
                if phase == "nn":
                    self.state.nn_active += 1
                    self.state.peak_nn = max(self.state.peak_nn, self.state.nn_active)
                else:
                    self.state.load_active += 1
                    self.state.peak_loaders = max(self.state.peak_loaders, self.state.load_active)
                self.state.peak_bytes = max(self.state.peak_bytes, self.capacity - self.state.free)
            finally:
                if phase == "nn":
                    self.state.nn_waiting -= 1
                self.condition.notify_all()
        lease = SourceInputLease(self, phase, nbytes)
        try:
            yield lease
        finally:
            lease.release_unused()

    def record(self):
        with self.condition:
            return {"shared_staging_capacity_bytes": self.capacity, "free_staging_bytes": self.state.free,
                "nn_waiting": self.state.nn_waiting, "nn_active": self.state.nn_active,
                "load_active": self.state.load_active, "load_worker_capacity": self.load_workers,
                "peak_staging_bytes": self.state.peak_bytes, "peak_loaders": self.state.peak_loaders,
                "peak_nn_staging": self.state.peak_nn, "ready_nn_priority": True}


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
    import torch

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
            # Persisted inactive disk pages are reclaimable. tmpfs/shmem is
            # working memory and must never be counted as reclaimable capacity.
            stat = dict(line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
            reclaimable = min(int(stat.get("inactive_file", 0)),
                              max(0, int(stat.get("file", 0)) - int(stat.get("shmem", 0))))
            limits.append(max(0, int(quota) - used + reclaimable))
    except (OSError, ValueError):
        pass
    return min(limits)


class HostMemoryAdmission:
    """Reserve whole process groups before loading datasets or spawning children."""

    def __init__(self, manager, resources):
        self.state = manager.Namespace()
        self.state.free = resources["host_memory_budget_bytes"] - resources.get("shared_staging_memory_bytes", 0)
        self.state.groups = 0
        self.state.child_cpus = 0
        self.coordinator_threads = resources.get("coordinator_cpu_threads", 0)
        self.state.cpu_threads = self.coordinator_threads
        self.state.preparation_groups = 0
        self.lock = manager.RLock()
        self.group_bytes = resources["source_worker_memory_bytes"]
        self.target_bytes = resources["target_worker_memory_bytes"]
        self.target_limit = resources["target_workers"]
        self.cpu_budget = resources["cpu_budget"]
        self.gpu_worker_limit = resources["gpu_slots"]
        self.native_threads = resources["native_threads_per_worker"]
        self.source_limit = resources["source_workers"]

    def reserve_phase(self, phase, *, cpu_threads, memory_bytes):
        """Reserve a ready DAG node without parking another process in a wait."""
        if phase not in {"source", "preparation"}:
            raise ValueError("unknown source pipeline phase")
        if (type(cpu_threads) is not int or cpu_threads < 1
                or type(memory_bytes) is not int or memory_bytes < 1):
            raise ValueError("source phases require positive CPU and memory budgets")
        with self.lock:
            if (self.state.free < memory_bytes or self.state.cpu_threads + cpu_threads > self.cpu_budget
                    or phase == "source" and self.state.groups >= min(self.source_limit, self.gpu_worker_limit)):
                return None
            self.state.free -= memory_bytes
            self.state.cpu_threads += cpu_threads
            if phase == "source":
                self.state.groups += 1
            else:
                self.state.preparation_groups += 1
        return {"phase": phase, "cpu_threads": cpu_threads, "memory_bytes": memory_bytes}

    def release_phase(self, reservation):
        with self.lock:
            self.state.free += reservation["memory_bytes"]
            self.state.cpu_threads -= reservation["cpu_threads"]
            if reservation["phase"] == "source":
                self.state.groups -= 1
            else:
                self.state.preparation_groups -= 1

    def record(self):
        with self.lock:
            return {"reserved_cpu_threads": self.state.cpu_threads,
                    "coordinator_cpu_threads": self.coordinator_threads,
                    "free_cpu_threads": self.cpu_budget - self.state.cpu_threads,
                    "free_worker_memory_bytes": self.state.free,
                    "source_groups": self.state.groups,
                    "preparation_groups": self.state.preparation_groups,
                    "nested_target_workers": self.state.child_cpus}

    @contextmanager
    def source_group(self):
        while True:
            with self.lock:
                if (self.state.free >= self.group_bytes
                        and self.state.cpu_threads + self.native_threads <= self.cpu_budget):
                    self.state.free -= self.group_bytes
                    self.state.groups += 1
                    self.state.cpu_threads += self.native_threads
                    break
            time.sleep(0.1)
        try:
            yield
        finally:
            with self.lock:
                self.state.free += self.group_bytes
                self.state.groups -= 1
                self.state.cpu_threads -= self.native_threads

    @contextmanager
    def target_workers(self, requested):
        with self.lock:
            desired = min(self.target_limit, max(requested, self.cpu_budget // max(1, self.state.groups)))
            cpu_free = (self.cpu_budget - self.state.cpu_threads) // self.native_threads + 1
            task_free = self.gpu_worker_limit - self.state.groups - self.state.child_cpus + 1
            allowed = min(desired, self.state.free // self.target_bytes, cpu_free, task_free)
            workers = int(allowed) if allowed >= 2 else 1
            reservation = workers * self.target_bytes if workers > 1 else 0
            self.state.free -= reservation
            self.state.child_cpus += workers - 1
            self.state.cpu_threads += (workers - 1) * self.native_threads
        try:
            yield workers
        finally:
            with self.lock:
                self.state.free += reservation
                self.state.child_cpus -= workers - 1
                self.state.cpu_threads -= (workers - 1) * self.native_threads


def source_group_input_status(groups, state):
    """A fold is ready only when every declared source study is admitted."""
    ready, pending, failed = [], [], []
    for outer, layout in groups:
        dependencies = set(outer["source_studies"])
        errors = {study: state["failed"][study] for study in sorted(dependencies & set(state["failed"]))}
        if errors:
            failed.append((outer, layout, errors))
        elif dependencies <= set(state["ready"]):
            ready.append((outer, layout))
        else:
            pending.append((outer, layout, sorted(dependencies - set(state["ready"]))))
    return {"ready": ready, "pending": pending, "failed": failed}


def resolve_source_resources(plan, group_count, device, *, cpu_count=None, free_device_bytes=None, free_host_bytes=None):
    import torch

    from train.runtime import available_cpu_threads

    available = available_cpu_threads() if cpu_count is None else cpu_count
    if group_count < 1 or available < 1:
        raise ValueError("task count and available CPU budget must be positive")
    if plan.get("cpu_budget") is not None and (type(plan["cpu_budget"]) is not int or plan["cpu_budget"] < 1):
        raise ValueError("CPU budget must be a positive integer")
    budget = min(available, plan.get("cpu_budget") or available)
    coordinator_threads = int(budget > 1)
    worker_cpu_budget = budget - coordinator_threads
    requested = plan.get("source_workers", "auto")
    if requested not in (None, "auto") and (type(requested) is not int or requested < 1):
        raise ValueError("source_workers must be auto or a positive integer")
    groups = min(group_count, worker_cpu_budget, group_count if requested in (None, "auto") else requested)
    memory_gib = float(plan.get("gpu_task_memory_gib", 3.0))
    if not 0 < memory_gib < float("inf") or budget < 1:
        raise ValueError("resource budgets must be positive and finite")
    if free_host_bytes is None:
        free_host_bytes = available_host_memory_bytes()
    host_fraction = plan.get("host_memory_fraction", .85)
    if not isinstance(host_fraction, (float, int)) or isinstance(host_fraction, bool) or not 0 < host_fraction <= .95:
        raise ValueError("host_memory_fraction must be in (0, .95]")
    host_budget = int(free_host_bytes * host_fraction)
    staging_bytes = int(float(plan.get("shared_staging_memory_gib", 0)) * 2**30)
    if staging_bytes < 0:
        raise ValueError("shared staging memory must be nonnegative")
    group_budget = host_budget - staging_bytes
    group_bytes = int(float(plan.get("source_worker_memory_gib", 6.0)) * 2**30)
    target_bytes = int(float(plan.get("target_worker_memory_gib", 4.0)) * 2**30)
    if min(group_bytes, target_bytes) <= 0 or group_budget < max(group_bytes, target_bytes):
        raise ValueError("available host memory is below the declared worker requirement")
    groups = min(groups, group_budget // group_bytes)
    if torch.device(device).type == "cuda":
        if free_device_bytes is None:
            free_device_bytes = torch.cuda.mem_get_info(torch.device(device))[0]
        slots = max(1, min(worker_cpu_budget, int(0.8 * free_device_bytes / (memory_gib * 2**30))))
    else:
        slots = worker_cpu_budget
    requested_gpu = plan.get("gpu_workers")
    if requested_gpu is not None:
        if type(requested_gpu) is not int or requested_gpu < 1:
            raise ValueError("gpu_workers must be positive")
        slots = min(slots, requested_gpu)
    target_request = plan.get("target_workers", "auto")
    if target_request not in (None, "auto") and (type(target_request) is not int or target_request < 1):
        raise ValueError("target_workers must be auto or a positive integer")
    target = min(host_budget // target_bytes, slots if target_request in (None, "auto") else min(target_request, slots))
    floor = plan.get("cloud_resource_floor")
    if floor is not None:
        if (not isinstance(floor, dict) or set(floor) != {"cpu_threads", "gpu_workers"}
                or any(type(value) is not int or value < 1 for value in floor.values())):
            raise ValueError("cloud_resource_floor requires positive cpu_threads and gpu_workers")
        if (budget < floor["cpu_threads"] or slots < floor["gpu_workers"]
                or target < floor["gpu_workers"] or groups < min(group_count, floor["gpu_workers"])):
            raise ValueError("resources cannot meet declared cloud CPU/GPU worker floor; do not silently serialize")
    native_threads = max(1, worker_cpu_budget // max(groups, target))
    return {"policy": "automatic_resource_admission", "available_cpu_threads": available,
            "cpu_budget": budget, "worker_cpu_budget": worker_cpu_budget,
            "coordinator_cpu_threads": coordinator_threads,
            "requested_source_workers": requested or "auto",
            "source_groups": group_count, "source_workers": groups,
            "gpu_slots": slots, "target_workers": target,
            "inner_workers_per_group": min(target, max(1, (budget - groups) // min(groups, 4))),
            "native_threads_per_worker": native_threads, "compiler_threads_per_worker": 1,
            "cloud_resource_floor": floor,
            "gpu_task_memory_gib": memory_gib, "free_device_bytes": free_device_bytes,
            "resident_memory_budget_bytes": (int(free_device_bytes * .90 / slots)
                                             if torch.device(device).type == "cuda" else None),
            "available_host_memory_bytes": free_host_bytes, "host_memory_budget_bytes": host_budget,
            "host_memory_fraction": host_fraction, "shared_staging_memory_bytes": staging_bytes,
            "source_worker_memory_bytes": group_bytes, "target_worker_memory_bytes": target_bytes}
