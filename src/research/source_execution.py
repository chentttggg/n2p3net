"""Resource admission and process-safe source trajectory reuse."""
from __future__ import annotations

import json
import os
import shutil
import stat
import time
import uuid
from contextlib import contextmanager
from multiprocessing.managers import SyncManager
from pathlib import Path
from threading import Condition, RLock
from zipfile import ZipFile

from research.execution import atomic_write_json

_PROCESS_INPUT_ADMISSION = None


def install_input_admission(admission, task_id=None):
    global _PROCESS_INPUT_ADMISSION
    _PROCESS_INPUT_ADMISSION = admission
    if task_id is not None:
        os.environ["N2P3_CLOUD_TASK_ID"] = task_id


def current_input_admission():
    return _PROCESS_INPUT_ADMISSION


class _CheckpointByteCounter:
    """Count the actual Torch archive without retaining a second payload."""

    def __init__(self):
        self.nbytes = 0

    def write(self, data):
        self.nbytes += len(data)
        return len(data)

    def flush(self):
        pass


def save_source_checkpoint(path, payload):
    """Admit serialized model/ledger bytes independently of resident EEG RAM."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    admission = current_input_admission()
    if admission is None:
        torch.save(payload, path)
        return
    counter = _CheckpointByteCounter()
    torch.save(payload, counter)
    # Both passes use Torch's file-like archive layout. Counting stores no
    # tensors or serialized buffer; the existing writer settles actual partials.
    with admission.artifact_writer(path, counter.nbytes), path.open("wb") as stream:
        torch.save(payload, stream)


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
    """A normalized-input lease; remote transactions own its accounting."""

    def __init__(self, owner, phase, nbytes, token):
        self.owner, self.phase, self.nbytes, self.token = owner, phase, nbytes, token
        self.active = True

    def release_unused(self, retained_bytes=0):
        if self.active:
            self.owner._ledger.release(self.token, retained_bytes)
            self.nbytes, self.active = retained_bytes, retained_bytes != 0


def _named_output_bytes(path, directory=False):
    path = Path(path)
    if not path.exists():
        return 0
    if directory:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return path.stat().st_size


def _external_file_identity(info):
    return {name: getattr(info, name) for name in (
        "st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


class _SourceInputState:
    """Short atomic transactions execute in the existing Manager server.

    A native client never owns the server's lock. Waiting remains in the
    client, so a dead waiter cannot strand a server thread or claim bytes
    after task retirement.
    """

    def __init__(self, capacity, load_workers):
        self.capacity, self.load_workers = capacity, load_workers
        self.lock = RLock()
        self.leases, self.filesystems, self.retired_tasks = {}, {}, set()
        self.nested_artifacts = {}
        self.reclaimed_paths = {}
        self.free = capacity
        self.nn_waiting = self.nn_active = self.load_active = 0
        self.peak_bytes = self.peak_loaders = self.peak_nn = 0

    def _identity_allowed(self, identity):
        if identity["task_id"] in self.retired_tasks:
            raise RuntimeError("input allocation requested after the owned task tree retired")

    def _peak(self):
        self.peak_bytes = max(self.peak_bytes, self.capacity - self.free)
        self.peak_loaders = max(self.peak_loaders, self.load_active)
        self.peak_nn = max(self.peak_nn, self.nn_active)

    def _held_persistent(self, identity):
        # This task's output writer cannot settle until its own nested input
        # operation returns. Waiting for that active allocation to release
        # would wait for ourselves, including a frozen subprocess descendant.
        return [row for row in self.leases.values() if row["kind"] == "persistent_artifact" and (
            row["committed"] or row["pid"] == identity["pid"]
            or identity["task_id"] is not None and row["task_id"] == identity["task_id"])]

    def _filesystem(self, directory, key, nbytes, uses_host_memory):
        safety = max(64 * 2**20, int(nbytes * .05))
        state = self.filesystems.setdefault(key, {"directory": directory,
            "baseline_free_bytes": shutil.disk_usage(directory).free, "reserved_bytes": 0,
            "safety_bytes": safety, "peak_reserved_bytes": 0, "uses_host_memory": uses_host_memory,
            "reclaimed_bytes": 0, "deferred_reclaim_upper_bound_bytes": 0})
        state["safety_bytes"] = max(state["safety_bytes"], safety)
        return state

    def _unsettled_disk_bytes(self, key):
        return sum(row["disk_bytes"] for row in self.leases.values()
            if (row["kind"] == "anonymous_snapshot" or row["kind"] == "persistent_artifact"
                and not row["committed"]) and row["filesystem"] == key)

    def _filesystem_room(self, state, directory, key):
        physical = shutil.disk_usage(directory).free - state["safety_bytes"]
        if state["deferred_reclaim_upper_bound_bytes"]:
            # Deferred deletion is only an allocation upper bound. Actual free
            # has not paid for unmaterialized snapshots/writers yet; subtract
            # their whole peaks until settlement, even after a partial write.
            physical -= self._unsettled_disk_bytes(key)
        return min(state["baseline_free_bytes"] + state["reclaimed_bytes"]
            + state["deferred_reclaim_upper_bound_bytes"] - state["safety_bytes"] - state["reserved_bytes"], physical)

    def try_phase(self, phase, nbytes, token, identity):
        with self.lock:
            self._identity_allowed(identity)
            persistent = sum(row["nbytes"] for row in self._held_persistent(identity))
            if self.capacity - persistent < nbytes:
                raise ValueError("input staging cannot fit after transient leases retire; persistent files retain their budget")
            if phase == "nn" and token not in self.leases:
                self.leases[token] = {"kind": "waiting_nn", "phase": phase, "nbytes": 0, **identity}
                self.nn_waiting += 1
            if self.free < nbytes or phase == "load" and (self.nn_waiting or self.load_active >= self.load_workers):
                return False
            if phase == "nn":
                self.nn_waiting -= 1
                self.nn_active += 1
            else:
                self.load_active += 1
            self.free -= nbytes
            self.leases[token] = {"kind": "process", "phase": phase, "nbytes": nbytes, **identity}
            self._peak()
            return True

    def try_resident_input(self, nbytes, token, identity):
        with self.lock:
            self._identity_allowed(identity)
            if self.free < nbytes:
                return False
            self.free -= nbytes
            self.leases[token] = {"kind": "resident_input", "phase": "resident", "nbytes": nbytes, **identity}
            self._peak()
            return True

    def try_snapshot(self, directory, key, nbytes, ram_bytes, ram_backed, token, identity):
        with self.lock:
            self._identity_allowed(identity)
            state = self._filesystem(directory, key, nbytes, ram_backed)
            persistent = self._held_persistent(identity)
            disk_held = sum(row["disk_bytes"] for row in persistent if row["filesystem"] == key)
            ram_held = sum(row["nbytes"] for row in persistent)
            if (state["baseline_free_bytes"] + state["reclaimed_bytes"] + state["deferred_reclaim_upper_bound_bytes"]
                    - state["safety_bytes"] - disk_held < nbytes
                    or self.capacity - ram_held < ram_bytes):
                raise ValueError("snapshot cannot fit after transient leases retire; persistent files retain their budget")
            room = self._filesystem_room(state, directory, key)
            if room < nbytes or self.free < ram_bytes or self.load_active >= self.load_workers or self.nn_waiting:
                # Unowned external storage pressure cannot be cured by waiting
                # when this owner has no live allocation which can release it.
                pending = any(row["kind"] == "anonymous_snapshot" and row["filesystem"] == key
                    or row["kind"] == "persistent_artifact" and row["filesystem"] == key and not row["committed"]
                    for row in self.leases.values())
                if room < nbytes and not pending:
                    raise ValueError("snapshot filesystem has no live allocation capable of releasing the required space")
                return False
            self.free -= ram_bytes
            self.load_active += 1
            state["reserved_bytes"] += nbytes
            state["peak_reserved_bytes"] = max(state["peak_reserved_bytes"], state["reserved_bytes"])
            self.leases[token] = {"kind": "anonymous_snapshot", "phase": "load", "nbytes": ram_bytes,
                "disk_bytes": nbytes, "filesystem": key, **identity}
            self._peak()
            return True

    def release(self, token, retained_bytes=0):
        with self.lock:
            row = self.leases.get(token)
            if row is None:
                return
            if type(retained_bytes) is not int or not 0 <= retained_bytes <= row["nbytes"]:
                raise ValueError("retained input bytes exceed the admitted lease")
            if row["kind"] == "waiting_nn":
                self.nn_waiting -= 1
            elif row["kind"] == "persistent_artifact":
                raise ValueError("persistent files require actual filesystem settlement")
            else:
                self.free += row["nbytes"] - retained_bytes
                if retained_bytes:
                    row["nbytes"] = retained_bytes
                    return
                if row["phase"] == "nn":
                    self.nn_active -= 1
                elif row["phase"] == "load":
                    self.load_active -= 1
                if row["kind"] == "anonymous_snapshot":
                    self.filesystems[row["filesystem"]]["reserved_bytes"] -= row["disk_bytes"]
            del self.leases[token]

    def reserve_artifact(self, path, paths, directory, key, nbytes, ram_bytes, ram_backed, identity):
        with self.lock:
            self._identity_allowed(identity)
            covering = next((row for row in self.leases.values()
                if row["kind"] == "persistent_artifact" and row["directory"] and not row["committed"]
                and row["pid"] == identity["pid"] and row["task_id"] == identity["task_id"]
                and row["filesystem"] == key and row["uses_host_memory"] == ram_backed
                and all(Path(item).is_relative_to(row["path"]) for item in paths)), None)
            if covering is not None:
                # The inner writer knows its actual encoded size before writing.
                # Grow the same outer lease to cover surviving bytes plus this
                # atomic temporary; a large late-bound scientific state must be
                # admitted here, never discovered only at final settlement.
                pending = 0
                for row in self.nested_artifacts.values():
                    if row["covering_token"] != covering["token"]:
                        continue
                    materialized = max(0, sum(_named_output_bytes(item, row["directory"])
                        for item in row["paths"]) - row["baseline_bytes"])
                    pending += max(0, row["disk_bytes"] - materialized)
                actual = max(0, _named_output_bytes(covering["path"], True) - covering["baseline_bytes"])
                required = actual + pending + nbytes
                addition = max(0, required - covering["disk_bytes"])
                if addition:
                    parent = str(Path(covering["path"]).parent)
                    state = self._filesystem(parent, key, required, ram_backed)
                    additional_ram = addition if covering["uses_host_memory"] else 0
                    if self._filesystem_room(state, parent, key) < addition or self.free < additional_ram:
                        raise ValueError("nested artifact actual peak exceeds the remaining shared filesystem/RAM budget")
                    self.free -= additional_ram
                    state["reserved_bytes"] += addition
                    state["peak_reserved_bytes"] = max(state["peak_reserved_bytes"], state["reserved_bytes"])
                    covering["disk_bytes"] += addition
                    covering["nbytes"] += additional_ram
                    self._peak()
                token = uuid.uuid4().hex
                self.nested_artifacts[token] = {"covering_token": covering["token"], "paths": paths,
                    "directory": directory, "disk_bytes": nbytes, **identity,
                    "baseline_bytes": sum(_named_output_bytes(item, directory) for item in paths)}
                return {"token": token, "owns_transaction": False}
            matches = [(token, row) for token, row in self.leases.items()
                       if row["kind"] == "persistent_artifact" and row["path"] == path]
            token, previous = matches[0] if matches else (uuid.uuid4().hex, None)
            if previous and not previous["committed"]:
                raise ValueError("a named artifact writer already owns this output path")
            if previous and previous["directory"] != directory:
                raise ValueError("named writer retry changed its output kind")
            parent = str(Path(path).parent)
            state = self._filesystem(parent, key, nbytes, ram_backed)
            room = self._filesystem_room(state, parent, key)
            if room < nbytes or self.free < ram_bytes:
                raise ValueError("named working artifact exceeds the remaining shared filesystem/RAM budget")
            if previous:
                baseline = previous["baseline_bytes"] + sum(_named_output_bytes(item, directory)
                    for item in paths if item not in previous["paths"])
                paths = list(dict.fromkeys([*previous["paths"], *paths]))
            else:
                baseline = sum(_named_output_bytes(item, directory) for item in paths)
            self.free -= ram_bytes
            state["reserved_bytes"] += nbytes
            state["peak_reserved_bytes"] = max(state["peak_reserved_bytes"], state["reserved_bytes"])
            self.leases[token] = {"kind": "persistent_artifact", "token": token, "path": path, "paths": paths,
                "directory": directory, "baseline_bytes": baseline, "phase": "artifact", "filesystem": key,
                "uses_host_memory": ram_backed,
                "nbytes": ram_bytes + (previous["nbytes"] if previous else 0),
                "disk_bytes": nbytes + (previous["disk_bytes"] if previous else 0), "committed": False, **identity}
            self._peak()
            return {"token": token, "owns_transaction": True}

    def settle_artifact(self, token):
        with self.lock:
            row = self.leases[token]
            if any(item["covering_token"] == token for item in self.nested_artifacts.values()):
                raise ValueError("outer artifact writer ended with pending nested writers")
            total = sum(_named_output_bytes(path, row["directory"]) for path in row["paths"])
            actual = max(0, total - row["baseline_bytes"])
            difference = row["disk_bytes"] - actual
            filesystem = self.filesystems[row["filesystem"]]
            filesystem["reserved_bytes"] -= difference
            filesystem["peak_reserved_bytes"] = max(filesystem["peak_reserved_bytes"], filesystem["reserved_bytes"])
            if row["uses_host_memory"]:
                self.free += difference
                row["nbytes"] = actual
            row.update(disk_bytes=actual, committed=True, surviving_total_bytes=total)
            row["paths"] = [row["path"], *(path for path in row["paths"][1:] if Path(path).exists())]
            self._peak()
            if difference < 0:
                raise ValueError("actual working artifact exceeds its pre-admitted estimate; surviving bytes remain charged")
            return {"path": row["path"], "surviving_total_bytes": total,
                    "baseline_bytes": row["baseline_bytes"], "charged_bytes": actual}

    def settle_nested_artifact(self, token):
        with self.lock:
            self.nested_artifacts.pop(token)

    def remove_external_file(self, path, expected_identity, ram_backed, identity):
        """Delete one verified external file, never an owner-managed output."""
        with self.lock:
            self._identity_allowed(identity)
            if path in self.reclaimed_paths:
                raise ValueError("external file path was already reclaimed by this owner")
            target = Path(path)
            info = target.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("external reclamation requires a regular file, never a symlink or directory")
            if _external_file_identity(info) != expected_identity:
                raise ValueError("external file identity changed after archive verification")
            if any(row["kind"] == "persistent_artifact" and any(
                target == Path(item) or row["directory"] and target.is_relative_to(item)
                for item in row["paths"]) for row in self.leases.values()):
                raise ValueError("external file overlaps a named artifact; retain its original settlement accounting")
            parent, key = str(target.parent), str(info.st_dev)
            state = self._filesystem(parent, key, 0, ram_backed)
            before = shutil.disk_usage(parent).free
            # st_blocks is the allocated byte authority on Linux. Sparse bytes,
            # other hardlinks, and platforms lacking this field earn no credit.
            blocks = getattr(info, "st_blocks", None)
            allocated = blocks * 512 if blocks is not None else 0
            reclaimable = allocated if info.st_nlink == 1 else 0
            receipt = {"path": path, "filesystem": key, "original_identity": expected_identity,
                "allocated_bytes": allocated, "physical_blocks_known": blocks is not None,
                "link_count": info.st_nlink, "free_before_bytes": before, "free_after_bytes": None,
                "observed_free_delta_bytes": None, "credited_bytes": 0,
                "deferred_allocation_upper_bound_bytes": reclaimable,
                "removed": True, **identity}
            target.unlink()
            # The unlink is irreversible. Commit its fact and path guard before
            # the next sensor can fail; unobserved space remains only a D bound.
            self.reclaimed_paths[path] = receipt
            state["deferred_reclaim_upper_bound_bytes"] += reclaimable
            try:
                after = shutil.disk_usage(parent).free
            except OSError as error:
                receipt["probe_failure"] = {"type": type(error).__name__, "message": str(error)}
                error.external_file_removal = dict(receipt)
                raise
            credited = min(reclaimable, max(0, after - before))
            state["reclaimed_bytes"] += credited
            state["deferred_reclaim_upper_bound_bytes"] -= credited
            receipt.update(free_after_bytes=after, observed_free_delta_bytes=after - before,
                credited_bytes=credited, deferred_allocation_upper_bound_bytes=reclaimable - credited)
            return dict(receipt)

    def retire_task(self, task_id):
        with self.lock:
            self.retired_tasks.add(task_id)
            for token, row in list(self.nested_artifacts.items()):
                if row["task_id"] == task_id:
                    del self.nested_artifacts[token]
            failures = []
            for token, row in list(self.leases.items()):
                if row["task_id"] != task_id:
                    continue
                if row["kind"] == "persistent_artifact":
                    if not row["committed"]:
                        try:
                            self.settle_artifact(token)
                        except Exception as error:
                            failures.append(error)
                else:
                    self.release(token)
            if failures:
                raise failures[0]

    def record(self):
        with self.lock:
            return {"shared_staging_capacity_bytes": self.capacity, "free_staging_bytes": self.free,
                "resident_input_bytes": sum(row["nbytes"] for row in self.leases.values()
                    if row["kind"] == "resident_input"),
                "nn_waiting": self.nn_waiting, "nn_active": self.nn_active, "load_active": self.load_active,
                "load_worker_capacity": self.load_workers, "peak_staging_bytes": self.peak_bytes,
                "peak_loaders": self.peak_loaders, "peak_nn_staging": self.peak_nn, "ready_nn_priority": True,
                "filesystem_pools": {key: {**row, "live_unsettled_reserved_bytes": self._unsettled_disk_bytes(key)}
                    for key, row in self.filesystems.items()},
                "external_file_reclaims": [dict(row) for row in self.reclaimed_paths.values()],
                "pending_nested_artifacts": [dict(row) for row in self.nested_artifacts.values()],
                "active_leases": [dict(row) for row in self.leases.values()]}


class SourceInputAdmission:
    """One process-safe owner for staging, snapshots and persistent outputs.

    Construct it with the existing SyncManager after this module is imported;
    all phases and nested spawn/forkserver workers share its one ledger.
    """

    def __init__(self, manager, *, capacity_bytes, load_workers):
        if type(capacity_bytes) is not int or capacity_bytes < 0 or type(load_workers) is not int or load_workers < 1:
            raise ValueError("source input admission requires nonnegative bytes and positive I/O capacity")
        self.capacity, self.load_workers = capacity_bytes, load_workers
        self._ledger = manager.SourceInputState(capacity_bytes, load_workers)

    @staticmethod
    def _identity():
        return {"pid": os.getpid(), "task_id": os.environ.get("N2P3_CLOUD_TASK_ID")}

    def try_resident_input(self, nbytes):
        """Adopt shared reader bytes without blocking or occupying a decode slot."""
        if type(nbytes) is not int or not 0 <= nbytes <= self.capacity:
            raise ValueError("resident input bytes exceed the shared staging capacity")
        token = uuid.uuid4().hex
        if not self._ledger.try_resident_input(nbytes, token, self._identity()):
            return None
        return SourceInputLease(self, "resident", nbytes, token)

    @contextmanager
    def reserve(self, phase, nbytes=0):
        if phase not in {"load", "nn"} or type(nbytes) is not int or not 0 <= nbytes <= self.capacity:
            raise ValueError("source input phase exceeds its shared staging byte capacity")
        token, identity = uuid.uuid4().hex, self._identity()
        try:
            while not self._ledger.try_phase(phase, nbytes, token, identity):
                time.sleep(.05)
            lease = SourceInputLease(self, phase, nbytes, token)
            try:
                yield lease
            finally:
                lease.release_unused()
        finally:
            self._ledger.release(token)

    @contextmanager
    def snapshot(self, directory, nbytes):
        directory = Path(directory).resolve(strict=True)
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("snapshot bytes must be a nonnegative integer")
        key = str(directory.stat().st_dev)
        ram_backed = filesystem_uses_host_memory(directory)
        ram_bytes = nbytes if ram_backed else 0
        if ram_bytes > self.capacity:
            raise ValueError("anonymous snapshot exceeds the shared RAM staging pool; use a disk snapshot directory")
        token, identity = uuid.uuid4().hex, self._identity()
        try:
            while not self._ledger.try_snapshot(str(directory), key, nbytes, ram_bytes, ram_backed, token, identity):
                time.sleep(.05)
            yield
        finally:
            self._ledger.release(token)

    def reserve_artifact(self, path, nbytes, *, temporary_paths=(), directory=False):
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("persistent file estimates must be nonnegative integer bytes")
        paths = [str(path), *(str(Path(item).resolve()) for item in temporary_paths)]
        if any(Path(item).parent.stat().st_dev != path.parent.stat().st_dev for item in paths):
            raise ValueError("one named writer transaction must stay on one filesystem")
        ram_backed = filesystem_uses_host_memory(path.parent)
        ram_bytes = nbytes if ram_backed else 0
        if ram_bytes > self.capacity:
            raise ValueError("persistent working artifact exceeds shared RAM capacity; use a disk working directory")
        return self._ledger.reserve_artifact(str(path), paths, directory, str(path.parent.stat().st_dev),
            nbytes, ram_bytes, ram_backed, self._identity())

    def settle_artifact(self, token):
        return self._ledger.settle_artifact(token)

    def remove_external_file(self, path, *, expected_stat):
        """Reclaim an archived external file through the same disk-budget owner.

        The caller owns lossless archive verification and cleanup authorization.
        This method rejects named output overlap and rechecks the original inode
        identity before unlinking. Disk credit cannot enlarge fixed RAM staging.
        A post-unlink probe OSError carries its confirmed external_file_removal
        receipt so the caller can preserve the deletion journal before reraising.
        """
        path = Path(path)
        path = path.parent.resolve(strict=True) / path.name
        return self._ledger.remove_external_file(str(path), _external_file_identity(expected_stat),
            filesystem_uses_host_memory(path.parent), self._identity())

    @contextmanager
    def artifact_writer(self, path, nbytes, *, temporary_paths=(), directory=False):
        allocation = self.reserve_artifact(path, nbytes, temporary_paths=temporary_paths, directory=directory)
        if not allocation["owns_transaction"]:
            try:
                yield None
            finally:
                self._ledger.settle_nested_artifact(allocation["token"])
            return
        token = allocation["token"]
        primary = None
        try:
            yield token
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                self.settle_artifact(token)
            except Exception as error:
                if primary is None:
                    raise
                primary.add_note(f"named artifact settlement failed: {error}")

    def artifact_directory(self, directory, nbytes):
        return self.artifact_writer(directory, nbytes, directory=True)

    def write_json(self, path, payload):
        return atomic_write_json(path, payload, input_admission=self)

    def retire_task(self, task_id):
        return self._ledger.retire_task(task_id)

    def record(self):
        return self._ledger.record()
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

    if torch.device(device).type == "cuda" and torch.cuda.is_initialized():
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


class _HostMemoryState:
    """CPU/host/GPU transactions share the original owner in its Manager."""

    def __init__(self, resources):
        self.resources = resources
        self.lock = RLock()
        coordinator = resources.get("coordinator_cpu_threads", 0)
        self.state = {"free": resources["host_memory_budget_bytes"] - resources.get("shared_staging_memory_bytes", 0),
            "cpu_threads": coordinator, "groups": 0, "child_cpus": 0,
            "preparation_groups": 0, "calibration_groups": 0, "gpu_calibration_groups": 0,
            "gpu_capable_calibration_actors": 0,
            "evaluation_groups": 0, "gpu_groups": 0, "nested_gpu_workers": 0}
        self.phases, self.nested, self.retired_tasks = {}, {}, set()
        self.target_gpu_waiters = []

    def try_activate_target_gpu(self, task_id):
        """Upgrade the same resident target actor before its first CUDA work."""
        with self.lock:
            if task_id in self.retired_tasks:
                raise RuntimeError("target GPU requested after task retirement")
            rows = [row for row in self.phases.values() if row["task_id"] == task_id]
            if len(rows) != 1 or rows[0]["phase"] != "calibration" or not rows[0]["gpu_capable"]:
                raise ValueError("target GPU activation requires its one calibration reservation")
            row = rows[0]
            if row["gpu_workers"]:
                return dict(row)
            if any(item["task_id"] == task_id for item in self.nested.values()):
                raise ValueError("target GPU activation cannot overlap nested worker reservations")
            if task_id not in self.target_gpu_waiters:
                self.target_gpu_waiters.append(task_id)
            if (self.target_gpu_waiters[0] != task_id
                    or self.state["gpu_groups"] + self.state["nested_gpu_workers"] >= self.resources["gpu_slots"]
                    or self.state["gpu_calibration_groups"] >= self.resources["target_workers"]):
                return None
            self.state["gpu_groups"] += 1
            self.state["gpu_calibration_groups"] += 1
            row["gpu_workers"] = 1
            self.target_gpu_waiters.remove(task_id)
            return dict(row)

    def reserve_phase(self, phase, cpu_threads, memory_bytes, gpu_workers, task_id, gpu_capable):
        with self.lock:
            if task_id is not None and task_id in self.retired_tasks:
                raise RuntimeError("phase allocation requested after the owned task tree retired")
            state, resources = self.state, self.resources
            if (gpu_workers and self.target_gpu_waiters
                    or state["free"] < memory_bytes or state["cpu_threads"] + cpu_threads > resources["cpu_budget"]
                    or gpu_workers and state["gpu_groups"] + state["nested_gpu_workers"] >= resources["gpu_slots"]
                    or phase == "source" and state["groups"] >= resources["source_workers"]
                    or gpu_capable and state["gpu_capable_calibration_actors"] >= resources["target_workers"]
                    or phase == "calibration" and gpu_workers and state["gpu_calibration_groups"] >= resources["target_workers"]):
                return None
            state["free"] -= memory_bytes
            state["cpu_threads"] += cpu_threads
            state["gpu_groups"] += gpu_workers
            state["groups" if phase == "source" else f"{phase}_groups"] += 1
            if phase == "calibration":
                state["gpu_calibration_groups"] += gpu_workers
                state["gpu_capable_calibration_actors"] += int(gpu_capable)
            reservation = {"token": uuid.uuid4().hex, "phase": phase, "cpu_threads": cpu_threads,
                "memory_bytes": memory_bytes, "gpu_workers": gpu_workers, "task_id": task_id,
                "gpu_capable": gpu_capable}
            self.phases[reservation["token"]] = reservation
            return dict(reservation)

    def release_phase(self, token):
        with self.lock:
            reservation = self.phases.pop(token, None)
            if reservation is None:
                return
            task_id = reservation["task_id"]
            if task_id is not None:
                self.retired_tasks.add(task_id)
                if task_id in self.target_gpu_waiters:
                    self.target_gpu_waiters.remove(task_id)
                for nested_token, row in list(self.nested.items()):
                    if row["task_id"] == task_id:
                        self.release_nested(nested_token)
            state = self.state
            state["free"] += reservation["memory_bytes"]
            state["cpu_threads"] -= reservation["cpu_threads"]
            state["gpu_groups"] -= reservation["gpu_workers"]
            phase = reservation["phase"]
            state["groups" if phase == "source" else f"{phase}_groups"] -= 1
            if phase == "calibration":
                state["gpu_calibration_groups"] -= reservation["gpu_workers"]
                state["gpu_capable_calibration_actors"] -= int(reservation["gpu_capable"])

    def reserve_nested(self, requested, gpu_backed, task_id):
        with self.lock:
            if type(requested) is not int or requested < 1:
                raise ValueError("nested worker request must contain at least one ready job")
            if task_id is not None and task_id in self.retired_tasks:
                raise RuntimeError("nested allocation requested after the owned task tree retired")
            state, resources = self.state, self.resources
            native = resources["native_threads_per_worker"]
            # Never grant more nested workers than the caller has jobs; idle grants block other phases.
            desired = min(resources["target_workers"], requested)
            cpu_free = (resources["cpu_budget"] - state["cpu_threads"]) // native + 1
            gpu_free = (resources["gpu_slots"] - state["gpu_groups"] - state["nested_gpu_workers"] + 1
                        if gpu_backed else desired)
            if gpu_backed and self.target_gpu_waiters:
                gpu_free = 1  # Existing parent work continues; new slots go to queued targets.
            allowed = min(desired, state["free"] // resources["target_worker_memory_bytes"], cpu_free, gpu_free)
            workers = int(allowed) if allowed >= 2 else 1
            memory = workers * resources["target_worker_memory_bytes"] if workers > 1 else 0
            gpu_workers = (workers - 1) * int(gpu_backed)
            state["free"] -= memory
            state["child_cpus"] += workers - 1
            state["nested_gpu_workers"] += gpu_workers
            state["cpu_threads"] += (workers - 1) * native
            reservation = {"token": uuid.uuid4().hex, "workers": workers, "memory_bytes": memory,
                "gpu_workers": gpu_workers, "cpu_threads": (workers - 1) * native, "task_id": task_id}
            self.nested[reservation["token"]] = reservation
            return dict(reservation)

    def release_nested(self, token):
        with self.lock:
            row = self.nested.pop(token, None)
            if row is not None:
                self.state["free"] += row["memory_bytes"]
                self.state["child_cpus"] -= row["workers"] - 1
                self.state["nested_gpu_workers"] -= row["gpu_workers"]
                self.state["cpu_threads"] -= row["cpu_threads"]

    def record(self):
        with self.lock:
            state = self.state
            return {"reserved_cpu_threads": state["cpu_threads"],
                "coordinator_cpu_threads": self.resources.get("coordinator_cpu_threads", 0),
                "free_cpu_threads": self.resources["cpu_budget"] - state["cpu_threads"],
                "free_worker_memory_bytes": state["free"], "source_groups": state["groups"],
                "preparation_groups": state["preparation_groups"], "calibration_groups": state["calibration_groups"],
                "gpu_calibration_groups": state["gpu_calibration_groups"], "evaluation_groups": state["evaluation_groups"],
                "gpu_capable_calibration_actors": state["gpu_capable_calibration_actors"],
                "active_gpu_groups": state["gpu_groups"], "nested_target_workers": state["child_cpus"],
                "nested_gpu_workers": state["nested_gpu_workers"],
                "waiting_target_gpu_tasks": list(self.target_gpu_waiters)}


class HostMemoryAdmission:
    """Reserve complete groups through atomic methods in the existing Manager."""

    def __init__(self, manager, resources, *, input_admission=None):
        self.input_admission = input_admission
        self.coordinator_threads = resources.get("coordinator_cpu_threads", 0)
        self.group_bytes = resources["source_worker_memory_bytes"]
        self.target_bytes = resources["target_worker_memory_bytes"]
        self.target_limit = resources["target_workers"]
        self.cpu_budget = resources["cpu_budget"]
        self.gpu_worker_limit = resources["gpu_slots"]
        self.source_cpu_threads = resources.get("source_cpu_threads_per_worker", resources["native_threads_per_worker"])
        self.source_limit = resources["source_workers"]
        self.gpu_backed = resources.get("gpu_backed_workers", True)
        self._ledger = manager.HostMemoryState(resources)

    def reserve_phase(self, phase, *, cpu_threads, memory_bytes, gpu_workers=None, task_id=None, gpu_capable=None):
        if phase not in {"source", "preparation", "calibration", "evaluation"}:
            raise ValueError("unknown source pipeline phase")
        if type(cpu_threads) is not int or cpu_threads < 1 or type(memory_bytes) is not int or memory_bytes < 1:
            raise ValueError("source phases require positive CPU and memory budgets")
        gpu_workers = int(self.gpu_backed and phase in {"source", "calibration"}) if gpu_workers is None else gpu_workers
        if gpu_workers not in (0, 1):
            raise ValueError("one cache-group task claims zero or one actual GPU worker")
        gpu_capable = (phase == "calibration" and bool(gpu_workers)) if gpu_capable is None else gpu_capable
        if (type(gpu_capable) is not bool or gpu_capable and phase != "calibration"
                or phase == "calibration" and gpu_workers and not gpu_capable):
            raise ValueError("GPU-capable actor capacity belongs to actual CUDA target tasks")
        return self._ledger.reserve_phase(phase, cpu_threads, memory_bytes, gpu_workers, task_id, gpu_capable)

    def release_phase(self, reservation):
        self._ledger.release_phase(reservation["token"])

    def record(self):
        return self._ledger.record()

    @contextmanager
    def source_group(self):
        reservation = None
        while reservation is None:
            reservation = self.reserve_phase("source", cpu_threads=self.source_cpu_threads, memory_bytes=self.group_bytes,
                task_id=os.environ.get("N2P3_CLOUD_TASK_ID"))
            if reservation is None:
                time.sleep(.05)
        try:
            yield
        finally:
            self.release_phase(reservation)

    @contextmanager
    def target_workers(self, requested, *, device=None):
        if type(requested) is not int or requested < 1:
            raise ValueError("nested worker request must contain at least one ready job")
        gpu_backed = self.gpu_backed if device is None else str(device).startswith("cuda")
        reservation = self._ledger.reserve_nested(requested, gpu_backed, os.environ.get("N2P3_CLOUD_TASK_ID"))
        try:
            yield reservation["workers"]
        finally:
            self._ledger.release_nested(reservation["token"])


SyncManager.register("SourceInputState", _SourceInputState)
SyncManager.register("HostMemoryState", _HostMemoryState)


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


CLOUD_RESOURCE_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "cloud_training_resources.json"


def _resolved_cloud_resource_floor(plan, device):
    """Resolve the shared CUDA floor; explicit overrides may raise its minimum."""
    import torch

    def validate(floor):
        if (not isinstance(floor, dict) or set(floor) != {"cpu_threads", "gpu_workers"}
                or any(type(value) is not int or value < 1 for value in floor.values())):
            raise ValueError("cloud_resource_floor requires positive cpu_threads and gpu_workers")

    declared = plan.get("cloud_resource_floor")
    if declared is not None:
        validate(declared)
    exempt = (plan.get("resource_scope") in {"local_development", "software_correctness"}
              or plan.get("evidence_scope") == "software_correctness")
    if torch.device(device).type != "cuda" or exempt:
        return declared
    minimum = json.loads(CLOUD_RESOURCE_CONFIG.read_text(encoding="utf-8"))["cloud_resource_floor"]
    validate(minimum)
    if declared is None:
        return minimum
    if any(declared[axis] < value for axis, value in minimum.items()):
        raise ValueError("declared floor is below the central cloud resource floor; "
                         "use an explicit local/software scope for a probe")
    return declared


def with_cloud_resource_floor(plan, device):
    """Apply the central floor to CUDA plans while preserving explicit probe scope.

    Omitted/null floors use the configuration minimum. Cloud overrides may require
    more capacity; they cannot lower the minimum. The input plan is never mutated.
    """

    floor = _resolved_cloud_resource_floor(plan, device)
    if floor is plan.get("cloud_resource_floor"):
        return plan
    return {**plan, "cloud_resource_floor": floor}


def source_phase_resources(resources, device):
    """Bind the source worker's actual native/inter-op widths to its CPU claim."""
    from research.cloud_execution import PhaseResources

    native = resources.get("source_native_threads_per_worker", resources["native_threads_per_worker"])
    interop = resources.get("source_interop_threads_per_worker")
    cpu = resources.get("source_cpu_threads_per_worker", native)
    return PhaseResources("source", cpu, resources["source_worker_memory_bytes"], native, str(device),
                          interop_threads=interop)


def resolve_source_resources(plan, group_count, device, *, cpu_count=None, free_device_bytes=None, free_host_bytes=None):
    import torch

    from train.runtime import available_cpu_threads

    floor = _resolved_cloud_resource_floor(plan, device)
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
    group_bytes = int(float(plan.get("source_worker_memory_gib", 6.0)) * 2**30)
    target_bytes = int(float(plan.get("target_worker_memory_gib", 4.0)) * 2**30)
    replay_inputs = plan.get("evaluation_design") == "independent_prefix" or any(
        row.get("kind") == "candidate" for row in plan.get("ready_consumers", ()))
    staging_policy = "declared" if "shared_staging_memory_gib" in plan else "not_requested"
    if "shared_staging_memory_gib" in plan:
        staging_bytes = int(float(plan["shared_staging_memory_gib"]) * 2**30)
    elif replay_inputs:
        configured = json.loads(CLOUD_RESOURCE_CONFIG.read_text(encoding="utf-8"))["shared_staging_memory_gib"]
        worker_memory = max(group_bytes, target_bytes)
        if floor is not None:
            # Reserve either full worker phase before sizing the shared input
            # pool. A source dependency tail may contain fewer ready groups.
            worker_memory = max(min(group_count, floor["gpu_workers"]) * group_bytes,
                                floor["gpu_workers"] * target_bytes)
        staging_bytes = min(int(configured * 2**30), int(host_budget * .15),
            max(0, host_budget - worker_memory))
        staging_policy = "replay_inputs_from_available_host_memory"
    else:
        staging_bytes = 0
    if staging_bytes < 0 or replay_inputs and staging_bytes == 0:
        raise ValueError("shared replay staging needs positive available bytes; other staging must be nonnegative")
    group_budget = host_budget - staging_bytes
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
        if torch.device(device).type == "cuda":
            slots = min(slots, requested_gpu)
    target_request = plan.get("target_workers", "auto")
    if target_request not in (None, "auto") and (type(target_request) is not int or target_request < 1):
        raise ValueError("target_workers must be auto or a positive integer")
    target = min(group_budget // target_bytes, slots if target_request in (None, "auto") else min(target_request, slots))
    if floor is not None:
        if torch.device(device).type != "cuda":
            raise ValueError("declared cloud GPU worker floor requires a CUDA device; do not run it on CPU")
        if (budget < floor["cpu_threads"] or slots < floor["gpu_workers"]
                or target < floor["gpu_workers"] or groups < min(group_count, floor["gpu_workers"])):
            raise ValueError("resources cannot meet declared cloud CPU/GPU worker floor; do not silently serialize")
    native_threads = max(1, worker_cpu_budget // max(groups, target))
    source_native = plan.get("source_native_threads_per_worker")
    source_concurrency = min(groups, slots) if torch.device(device).type == "cuda" else groups
    legacy_scope = (plan.get("resource_scope") in {"local_development", "software_correctness"}
                    or plan.get("evidence_scope") == "software_correctness")
    if source_native is None and torch.device(device).type == "cuda" and not legacy_scope:
        source_native = worker_cpu_budget // source_concurrency - 1
    if source_native is not None:
        if type(source_native) is not int or source_native < 1:
            raise ValueError("source native threads must be a positive integer with room for one inter-op thread")
        if (source_native + 1) * source_concurrency > worker_cpu_budget:
            raise ValueError("source native/inter-op claims cannot meet the declared source worker capacity")
    target_native = plan.get("target_native_threads_per_worker")
    if target_native is not None:
        if type(target_native) is not int or target_native < 1:
            raise ValueError("target native threads must be a positive integer")
        # This opt-in policy uses one immutable inter-op thread per fresh actor.
        # Keep the requested GPU capacity; reject an impossible CPU topology.
        if (target_native + 1) * target > worker_cpu_budget:
            raise ValueError("target native/inter-op claims cannot meet the declared target worker capacity")
    return {"policy": "automatic_resource_admission", "available_cpu_threads": available,
            "cpu_budget": budget, "worker_cpu_budget": worker_cpu_budget,
            "max_cache_group_processes": worker_cpu_budget,
            "coordinator_cpu_threads": coordinator_threads,
            "requested_source_workers": requested or "auto",
            "source_groups": group_count, "source_workers": groups,
            "gpu_slots": slots, "target_workers": target, "gpu_backed_workers": torch.device(device).type == "cuda",
            "inner_workers_per_group": min(target, max(1, (budget - groups) // min(groups, 4))),
            "native_threads_per_worker": native_threads, "compiler_threads_per_worker": 1,
            "cloud_resource_floor": floor,
            "gpu_task_memory_gib": memory_gib, "free_device_bytes": free_device_bytes,
            "resident_memory_budget_bytes": (int(free_device_bytes * .90 / slots)
                                             if torch.device(device).type == "cuda" else None),
            "available_host_memory_bytes": free_host_bytes, "host_memory_budget_bytes": host_budget,
            "host_memory_fraction": host_fraction, "shared_staging_memory_bytes": staging_bytes,
            "shared_staging_policy": staging_policy,
            "source_worker_memory_bytes": group_bytes, "target_worker_memory_bytes": target_bytes,
            **({"source_native_threads_per_worker": source_native, "source_interop_threads_per_worker": 1,
                "source_cpu_threads_per_worker": source_native + 1} if source_native is not None else {}),
            **({"target_native_threads_per_worker": target_native} if target_native is not None else {})}
