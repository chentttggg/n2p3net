"""Small process-owned pause/resource seam for local scientific programs."""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from uuid import uuid4

import psutil


class LocalProgramStoppedError(RuntimeError):
    """An interrupted attempt remains unfinished, never numeric unavailable."""


class LocalResourceBlockedError(LocalProgramStoppedError):
    """Current memory cannot safely admit even the remaining single actor."""


def _identified_process(pid, created):
    """Return only the same process birth; access errors remain blocking errors."""
    try:
        process = psutil.Process(pid)
        return process if abs(process.create_time()-created) < .01 else None
    except psutil.NoSuchProcess:
        return None


def actual_alive(pid, created):
    """A reused PID or Linux zombie cannot own a live controller/worker lease."""
    process = _identified_process(pid, created)
    try:
        return (process is not None and process.is_running()
                and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def atomic_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.writing")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05*(attempt+1))


def read_control(path, *, run_id, controller_pid, controller_created, plan_file_sha256=None, max_age=10.):
    """Identity and freshness are one tuple; partial/locked writes are retried."""
    deadline = time.monotonic()+3.
    while True:
        try:
            record = json.loads(Path(path).read_bytes())
            actual = (record["run_id"], record["controller_pid"], record["controller_created"])
            if actual != (run_id, controller_pid, controller_created):
                raise LocalProgramStoppedError("control belongs to another controller attempt")
            if plan_file_sha256 is not None and record.get("plan_file_sha256") != plan_file_sha256:
                raise LocalProgramStoppedError("control belongs to another prepared program")
            if not actual_alive(controller_pid, controller_created):
                raise LocalProgramStoppedError("controller no longer exists or its PID was reused")
            now, updated = time.time(), record["updated_time"]
            if (type(updated) not in (int, float) or not math.isfinite(updated)
                    or updated > now+8*math.ulp(now)):
                raise LocalProgramStoppedError("controller heartbeat timestamp is invalid or in the future")
            if now-updated > max_age:
                raise LocalProgramStoppedError("controller heartbeat is stale")
            return record
        except (PermissionError, FileNotFoundError, json.JSONDecodeError, KeyError):
            if time.monotonic() >= deadline:
                raise LocalProgramStoppedError("controller record is unreadable") from None
            time.sleep(.05)
        except psutil.NoSuchProcess as error:
            raise LocalProgramStoppedError("controller no longer exists") from error


class LocalProgramControl:
    def __init__(self, path, owner, progress):
        self.path, self.owner, self.progress = Path(path), dict(owner), progress

    def checkpoint(self, phase):
        while True:
            record = read_control(self.path, **self.owner)
            if record.get("abort"):
                raise LocalProgramStoppedError("controller requested abort")
            cpu = psutil.cpu_percent(interval=.1)
            if not record.get("pause") and cpu < record["worker_cpu_threshold_percent"]:
                self.progress(phase)
                return
            self.progress("paused_"+phase)
            time.sleep(.5)


class OwnedProcessTree:
    """Track a launcher plus actual descendants, retaining each PID/birth pair."""

    def __init__(self, pid, created):
        self.pid, self.created = pid, created
        self.known = {pid: created}
        self.known_lock = RLock()

    def _alive(self):
        try:
            # Discover descendants even when the same-birth launcher is already
            # a zombie: a live child still owns resources and blocks resume.
            process = _identified_process(self.pid, self.created)
            if process is not None:
                for child in process.children(recursive=True):
                    with self.known_lock:
                        self.known[child.pid] = child.create_time()
        except psutil.NoSuchProcess:
            pass
        result = []
        with self.known_lock:
            known = tuple(self.known.items())
        for pid, created in known:
            try:
                if actual_alive(pid, created):
                    result.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                pass
        return result

    def sample(self):
        rows = []
        for process in self._alive():
            try:
                rows.append({"pid": process.pid, "created": process.create_time(),
                             "rss_bytes": process.memory_info().rss,
                             "cpu_seconds": sum(process.cpu_times()[:2])})
            except psutil.NoSuchProcess:
                pass
        return {"actual_processes": rows, "actual_process_pids": [row["pid"] for row in rows],
                "rss_bytes": sum(row["rss_bytes"] for row in rows),
                "cpu_seconds": sum(row["cpu_seconds"] for row in rows)}

    def stop(self):
        for process in reversed(self._alive()):
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, remaining = psutil.wait_procs(self._alive(), timeout=3.)
        for process in remaining:
            # Recheck the original birth after the bounded wait.
            try:
                if actual_alive(process.pid, self.known[process.pid]):
                    process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(remaining, timeout=3.)
        return self.sample()


class LocalResourceBudget:
    """Hysteresis and existing-worker growth consume actual process-tree RSS."""

    def __init__(self, limits, *, historical_peak=0):
        self.limits = dict(limits)
        self.peak = max(historical_peak, limits["worker_rss_estimate_bytes"])
        self.quiet, self.cpu_paused = 0, True

    def sample(self, cpu, available_memory, workers):
        limits = self.limits
        self.peak = max([self.peak, *(row["rss_bytes"] for row in workers)])
        self.quiet = self.quiet+1 if cpu < limits["resume_cpu_threshold_percent"] else 0
        if cpu >= limits["worker_cpu_threshold_percent"]:
            self.cpu_paused = True
        elif self.cpu_paused and self.quiet >= limits["required_quiet_samples"]:
            self.cpu_paused = False
        estimated = self.peak*limits["growth_safety_factor"]
        existing_growth = sum(max(0., estimated-row["rss_bytes"]) for row in workers)
        available = available_memory-existing_growth-limits["memory_reserve_bytes"]
        memory_paused = bool(workers) and available < 0
        paused = self.cpu_paused or memory_paused
        launch = (not paused and cpu < limits["launch_cpu_threshold_percent"]
                  and len(workers) < limits["max_workers"] and available >= estimated)
        return {"pause": paused, "cpu_pause": self.cpu_paused, "memory_pause": memory_paused,
                "worker_peak_rss_bytes": self.peak, "estimated_worker_peak_bytes": estimated,
                "existing_worker_growth_bytes": existing_growth,
                "available_for_new_worker_bytes": available, "launch": launch,
                "memory_action": "reduce_concurrency" if memory_paused and len(workers) > 1
                else "resource_blocked" if memory_paused else None}


def assert_no_owned_live(directory):
    """Resume must account for recorded descendants after a launcher exits."""
    directory = Path(directory)
    members = set()
    for path in directory.glob("worker.*.handle.json"):
        record = json.loads(path.read_bytes())
        members.add((record["worker_pid"], record["worker_created"]))
    for path in directory.glob("worker.*.identity.json"):
        record = json.loads(path.read_bytes())
        members.add((record["actual_worker_pid"], record["actual_worker_created"]))
    history = directory/"resources.jsonl"
    if history.exists():
        for line in history.read_text().splitlines():
            row = json.loads(line)
            for worker in row["workers"]:
                members.update((value["pid"], value["created"]) for value in worker["actual_processes"])
    for pid, created in members:
        if actual_alive(pid, created):
            raise LocalProgramStoppedError("prior owned worker or descendant remains live")


class LocalResourceMonitor:
    """One owner keeps control fresh while parent model admission is busy."""

    def __init__(self, directory, owner, limits, snapshot, *, historical_peak=0):
        self.directory, self.owner, self.snapshot = Path(directory), dict(owner), snapshot
        self.budget = LocalResourceBudget(limits, historical_peak=historical_peak)
        self.stop_event, self.publish_lock, self.latest_lock = Event(), Lock(), Lock()
        self.latest, self.failure = None, None
        self.controller_tree = OwnedProcessTree(owner["controller_pid"], owner["controller_created"])
        self.thread = Thread(target=self._run, name="scientific-program-control", daemon=True)

    def start(self):
        self.thread.start()

    def current(self):
        with self.latest_lock:
            if self.failure is not None:
                raise LocalProgramStoppedError("resource monitor failed: "+self.failure["detail"])
            return self.latest

    def _run(self):
        try:
            sequence = 0
            while not self.stop_event.is_set():
                cpu, memory = psutil.cpu_percent(interval=.1), psutil.virtual_memory()
                actors, pending = self.snapshot()
                details = [{"subject": subject, **tree.sample()} for subject, tree in actors]
                details = [row for row in details if row["actual_processes"]]
                decision = self.budget.sample(cpu, memory.available, details)
                sequence += 1
                measurement = self.owner | {"sequence": sequence, "time": time.time(),
                    "host_cpu_percent": cpu, "available_memory_bytes": memory.available,
                    "workers": details, "pending_subjects": pending,
                    "controller_process_tree": self.controller_tree.sample(), **decision}
                with self.publish_lock:
                    if self.stop_event.is_set():
                        return
                    atomic_record(self.directory/"control.json", self.owner | {
                        "updated_time": time.time(), "pause": decision["pause"], "abort": False,
                        "worker_cpu_threshold_percent": self.budget.limits["worker_cpu_threshold_percent"]})
                    with (self.directory/"resources.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(measurement)+"\n")
                    atomic_record(self.directory/"progress.json", {"status": "running", **measurement})
                with self.latest_lock:
                    self.latest = measurement
                self.stop_event.wait(.4)
        except BaseException as error:
            with self.latest_lock:
                self.failure = {"type": type(error).__name__, "detail": str(error), "traceback": traceback.format_exc()}

    def stop(self):
        # Serialize the last heartbeat write before disabling publication. A
        # late sensor cannot revive the controller after terminal abort.
        with self.publish_lock:
            self.stop_event.set()
        self.thread.join(timeout=3.)
        return not self.thread.is_alive()


@contextmanager
def score_checkpoints(module, control, completed):
    """Only execution checkpoints wrap the original unchanged score owners."""
    projection, resolution = module._project_diffuse_arrays, module._conditional_score_resolution

    @wraps(projection)
    def project(*args, **kwargs):
        control.checkpoint("candidate_diffuse_projection")
        return projection(*args, **kwargs)

    @wraps(resolution)
    def resolve(*args, **kwargs):
        control.checkpoint("candidate_conditional_resolution")
        result = resolution(*args, **kwargs)
        completed(result)
        return result

    module._project_diffuse_arrays, module._conditional_score_resolution = project, resolve
    try:
        yield
    finally:
        module._project_diffuse_arrays, module._conditional_score_resolution = projection, resolution
