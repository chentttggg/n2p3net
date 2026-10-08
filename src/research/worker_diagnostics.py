"""Persistent process facts for source workers, without training or array copies."""
from __future__ import annotations

import faulthandler
import hashlib
import json
import os
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import psutil

from research.execution import atomic_write_json

CGROUP_FILES = ("cpu.max", "memory.current", "memory.max", "memory.peak", "memory.events",
                "memory.events.local", "memory.pressure", "memory.swap.current", "memory.swap.max",
                "pids.current", "pids.max", "pids.events", "cgroup.events")
_WORKER_DIAGNOSTICS = None


def system_snapshot():
    """Allowlisted resource facts only; no environment or scientific inputs."""
    try:
        process = psutil.Process()
        memory = process.memory_info()
        resources = {"pid": process.pid, "create_time": process.create_time(), "rss_bytes": memory.rss,
                     "OS_thread_count": process.num_threads(),
                     "host_available_memory_bytes": psutil.virtual_memory().available, "status": "read"}
    except (psutil.Error, OSError) as error:
        resources = {"pid": os.getpid(), "status": "unavailable",
                     "error": {"type": type(error).__name__, "message": str(error)}}
    cgroup = {}
    if sys.platform == "linux":
        for name in CGROUP_FILES:
            try:
                cgroup[name] = {"status": "read", "value": (Path("/sys/fs/cgroup") / name).read_text().strip()}
            except OSError as error:
                cgroup[name] = {"status": "unavailable", "error": str(error)}
    return {**resources, "cgroup": cgroup}


class WorkerDiagnostics:
    def __init__(self, directory, identity):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        process = psutil.Process()
        self.pid, self.created = process.pid, process.create_time()
        self.stem = f"worker_{self.pid}_{int(self.created * 1e6)}"
        self.state_path = self.directory / f"{self.stem}.state.json"
        self.events_path = self.directory / f"{self.stem}.events.jsonl"
        self.stderr_path = self.directory / f"{self.stem}.stderr.log"
        self.stderr = self.stderr_path.open("a", encoding="utf-8", buffering=1)
        # File descriptor 2 includes native library writes. Python stderr and
        # faulthandler use the same durable worker file; stdout stays unchanged.
        os.dup2(self.stderr.fileno(), 2)
        sys.stderr = self.stderr
        faulthandler.enable(file=self.stderr, all_threads=True)
        self.process_state = {"schema": "n2p3_source_worker_diagnostic/1", "identity": identity,
            "pid": self.pid, "create_time": self.created, "parent_pid": process.ppid(),
            "executable": sys.executable, "stderr_path": str(self.stderr_path),
            "events_path": str(self.events_path), "faulthandler_enabled": faulthandler.is_enabled()}
        self.state = dict(self.process_state)
        self.task_sequence = 0
        self.update("worker_started", status="idle")

    def update(self, phase, **facts):
        self.state.update(phase=phase, updated_utc=datetime.now(UTC).isoformat(), updated_unix=time.time(),
                          resources=system_snapshot(), **facts)
        try:
            atomic_write_json(self.state_path, self.state)
            with self.events_path.open("a", encoding="utf-8", buffering=1) as stream:
                stream.write(json.dumps(self.state, ensure_ascii=True, allow_nan=False) + "\n")
        except (OSError, ValueError, RuntimeError) as error:
            self.record_persistence_failure(error)

    def record_persistence_failure(self, error):
        observation = {"type": type(error).__name__, "message": str(error),
                       "phase": self.state["phase"], "observed_utc": datetime.now(UTC).isoformat(),
                       "primary_failure": self.state.get("failure")}
        self.process_state["diagnostic_persistence_failure"] = observation
        self.state["diagnostic_persistence_failure"] = observation

    @contextmanager
    def task(self, output, role):
        output = Path(output).resolve()
        if self.directory.is_relative_to(output):
            raise ValueError("worker diagnostic files must live outside the scientific worker output")
        self.state = dict(self.process_state)
        self.task_sequence += 1
        self.update("task_started", status="running", output=str(output), role=role,
                    task_sequence=self.task_sequence)
        try:
            yield self
        except BaseException as error:
            try:
                traceback.print_exc(file=self.stderr)
                self.stderr.flush()
            except OSError as diagnostic_error:
                self.record_persistence_failure(diagnostic_error)
            self.update("python_exception", status="failed", failure={"type": type(error).__name__, "message": str(error)})
            raise
        else:
            self.update("task_returned", status="completed")


def initialize_worker_diagnostics(directory, identity):
    global _WORKER_DIAGNOSTICS
    _WORKER_DIAGNOSTICS = WorkerDiagnostics(directory, identity)


@contextmanager
def diagnostic_task(output, role):
    if _WORKER_DIAGNOSTICS is None:
        yield None
    else:
        with _WORKER_DIAGNOSTICS.task(output, role) as observer:
            yield observer


def worker_diagnostic_phase(phase, **facts):
    if _WORKER_DIAGNOSTICS is not None:
        _WORKER_DIAGNOSTICS.update(phase, **facts)


class PoolFailureDiagnostics:
    """Record the first parent observation, not an inferred native root cause."""

    def __init__(self, directory, identity):
        self.identity = {**identity, "pool_instance_id": uuid.uuid4().hex,
                         "parent_pid": os.getpid(), "parent_create_time": psutil.Process().create_time()}
        self.directory = Path(directory).resolve() / self.identity["pool_instance_id"]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.processes = {}
        self.process_identities = {}
        self.first = None
        self.events = []

    def remember_processes(self, executor):
        processes = (getattr(executor, "_processes", None) or {}).copy()
        for pid, process in processes.items():
            if pid not in self.processes:
                self.processes[pid] = process
                identity = {"pid": pid, "create_time": None, "identity_source": "executor_process_handle"}
                try:
                    actual = psutil.Process(pid)
                    identity.update(create_time=actual.create_time(), parent_pid=actual.ppid(), executable=actual.exe())
                except psutil.Error as error:
                    identity["identity_read_error"] = {"type": type(error).__name__, "message": str(error)}
                self.process_identities[pid] = identity

    def capture(self, error, task, *, stage):
        states = []
        excluded_states = []
        for path in sorted(self.directory.glob("*.state.json")):
            saved = _read_json_observation(path)
            record = saved.pop("record", None)
            observation = {"state_path": saved.pop("path"), "state_sha256": saved.pop("sha256", None),
                           "state_status": saved.pop("status"), **saved}
            if record is None:
                states.append(observation)
                continue
            observation["worker"] = record
            identity = record.get("identity")
            if not isinstance(identity, dict) or (record.get("output") is not None and not isinstance(record["output"], str)):
                observation.update(state_status="unreadable", error={"type": "ValueError", "message": "invalid worker identity or output"})
                states.append(observation)
                continue
            if identity.get("pool_instance_id") != self.identity["pool_instance_id"]:
                observation["exclusion"] = "different_pool_instance"
                excluded_states.append(observation)
                continue
            output = Path(record["output"]) if record.get("output") else None
            phases = {}
            if output is not None:
                for name in ("source_input_stages.json", "worker_progress.json"):
                    phases[name] = _read_json_observation(output / name)
            observation["saved_phase_state"] = phases
            states.append(observation)
        process_observations = []
        for pid, process in self.processes.items():
            alive = process.is_alive()
            process_observations.append({**self.process_identities[pid], "exit_code": process.exitcode, "alive": alive})
        event = {"schema": "n2p3_source_pool_failure_observation/1", "observed_utc": datetime.now(UTC).isoformat(),
            "identity": self.identity, "parent_pid": os.getpid(), "stage": stage, "task": task,
            "exception": {"type": type(error).__name__, "message": str(error)},
            "process_exit_observations": process_observations,
            "worker_states": states, "excluded_worker_states": excluded_states, "parent_resources": system_snapshot(),
            "native_first_cause": "unknown", "observation_order": len(self.events),
            "relationship": "first_parent_observation" if self.first is None else "same_pool_failure_followup"}
        if self.first is None:
            self.first = event
        self.events.append(event)
        persistence_errors = []
        destinations = [(self.directory / "pool_failure_observations.json", self.events)]
        if len(self.events) == 1:
            destinations.insert(0, (self.directory / "first_pool_failure.json", event))
        for path, payload in destinations:
            try:
                atomic_write_json(path, payload)
            except (OSError, ValueError, RuntimeError) as diagnostic_error:
                persistence_errors.append({"path": str(path), "type": type(diagnostic_error).__name__,
                                           "message": str(diagnostic_error)})
        return {"directory": str(self.directory), "relationship": event["relationship"],
                "persistence_errors": persistence_errors,
                "first_observed_utc": self.first["observed_utc"], "native_first_cause": "unknown"}


def _read_json_observation(path):
    """Hash and parse the same bytes; failed diagnostic reads stay observations."""
    observation = {"path": str(path)}
    try:
        payload = path.read_bytes()
    except OSError as error:
        return {**observation, "status": "unavailable", "error": {"type": type(error).__name__, "message": str(error)}}
    observation["sha256"] = hashlib.sha256(payload).hexdigest()
    try:
        record = json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)
        if not isinstance(record, dict):
            raise ValueError("diagnostic record must be a JSON object")
    except (UnicodeError, ValueError) as error:
        return {**observation, "status": "unreadable", "error": {"type": type(error).__name__, "message": str(error)}}
    return {**observation, "status": "read", "record": record}


def _reject_json_constant(value):
    raise ValueError(f"nonfinite diagnostic JSON constant: {value}")
