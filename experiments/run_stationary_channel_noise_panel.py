"""Prepare or execute the fixed sixteen-cell corrected stationary P1 program."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from threading import Lock
from uuid import uuid4

import psutil

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

from research.local_program_control import (  # noqa: E402
    LocalProgramControl,
    LocalResourceBlockedError,
    LocalResourceMonitor,
    OwnedProcessTree,
    assert_no_owned_live,
    atomic_record,
)
from research.stationary_panel import (  # noqa: E402
    ARMS,
    admit_cell,
    load_plan,
    prepare,
    read,
    reduce_program,
    sha,
    verify,
    worker,
)


def _owner(record):
    return {name: record[name] for name in ("run_id", "controller_pid", "controller_created", "plan_file_sha256")}


def run(plan_file):
    """Two ready local processes, fresh identity tuple, scoped cleanup."""
    directory, config = Path(plan_file).parent, load_plan(plan_file)
    verify(config["source_files"])
    verify(config["original_files"])
    own, run_id = psutil.Process(), uuid4().hex
    plan_sha = sha(plan_file)
    owner = {"run_id": run_id, "controller_pid": own.pid, "controller_created": own.create_time(), "plan_file_sha256": plan_sha}
    lease = directory/"controller.lease.json"
    if lease.exists():
        old = read(lease)
        try:
            process = psutil.Process(old["controller_pid"])
            if abs(process.create_time()-old["controller_created"]) < .01:
                raise RuntimeError("another live controller owns this program")
        except psutil.NoSuchProcess:
            pass
        assert_no_owned_live(directory)
        lease.rename(directory/f"controller.expired.{old['run_id']}.json")
    pending, active, logs, trees = [], {}, [], []
    for subject in config["participants"]:
        if any(not (directory/"cells"/f"{subject}_{arm}"/"receipt.json").exists() for arm in ARMS):
            pending.append(subject)
        else:
            for arm in ARMS:
                admit_cell(config, subject, arm, directory)
    resources, status, code = config["resources"], "running", 0
    historical = [json.loads(line) for line in (directory/"resources.jsonl").read_text().splitlines()
                  if line.strip()] if (directory/"resources.jsonl").exists() else []
    historical_peak = max((row["worker_peak_rss_bytes"] for row in historical
        if row["plan_file_sha256"] == plan_sha), default=0)
    state_lock = Lock()
    def snapshot():
        with state_lock:
            return [(subject, item["tree"]) for subject, item in active.items()], list(pending)
    monitor = LocalResourceMonitor(directory, owner, resources, snapshot, historical_peak=historical_peak)
    failure, cleanup_errors, cleanup = None, [], []
    def heartbeat(paused, abort=False):
        atomic_record(directory/"control.json", owner | {"updated_time": time.time(),
            "pause": paused, "abort": abort, "worker_cpu_threshold_percent": resources["worker_cpu_threshold_percent"]})
    with lease.open("x", encoding="utf-8") as stream:
        json.dump(owner, stream)
    monitor.start()
    try:
        launched_sequence = 0
        while pending or active:
            for subject, item in list(active.items()):
                result = item["process"].poll()
                if result is not None:
                    if item["tree"].sample()["actual_processes"]:
                        continue
                    item["log"].close()
                    if result:
                        raise subprocess.CalledProcessError(result, item["process"].args)
                    for arm in ARMS:
                        admit_cell(config, subject, arm, directory)
                    with state_lock:
                        del active[subject]
            decision = monitor.current()
            if decision is not None and decision["memory_action"] == "reduce_concurrency":
                # Reschedule the newest unsealed actor. Its completed science
                # cells survive, and its unsealed attempt is preserved on retry.
                subject = next((subject for subject in reversed(active) if any(
                    not (directory/"cells"/f"{subject}_{arm}"/"receipt.json").exists() for arm in ARMS)), None)
                if subject is not None:
                    item = active[subject]
                    stopped = item["tree"].stop()
                    item["process"].wait(timeout=3.)
                    if stopped["actual_processes"]:
                        raise LocalResourceBlockedError("memory reduction did not stop the owned actor")
                    item["log"].close()
                    atomic_record(directory/f"resource_reschedule.{subject}.{run_id}.{time.time_ns()}.json", owner | {
                        "subject": subject, "resource_snapshot": decision, "owned_tree_remaining": stopped,
                        "scientific_failure_evidence_created": False})
                    with state_lock:
                        del active[subject]
                        pending.insert(0, subject)
                    continue
            if decision is not None and decision["memory_action"] == "resource_blocked":
                raise LocalResourceBlockedError("current RAM cannot admit the declared remaining actor growth")
            if (pending and decision is not None and decision["sequence"] > launched_sequence
                    and decision["launch"] and len(active) < resources["max_workers"]):
                with state_lock:
                    subject = pending.pop(0)
                launched_sequence = decision["sequence"]
                log = (directory/f"worker.{subject}.{run_id}.log").open("w", encoding="utf-8")
                logs.append(log)
                env = os.environ.copy()
                for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    env[variable] = str(resources["native_threads"])
                env["PYTHONPATH"] = str(Path(config["source_root"])/"src")+os.pathsep+config["source_root"]
                command = [sys.executable, str(Path(config["source_root"])/"experiments/run_stationary_channel_noise_panel.py"),
                    "--phase", "worker", "--output", str(directory), "--subject", subject, "--run-id", run_id]
                process = subprocess.Popen(command, cwd=config["source_root"], env=env, stdout=log, stderr=subprocess.STDOUT)
                created = psutil.Process(process.pid).create_time()
                tree = OwnedProcessTree(process.pid, created)
                trees.append(tree)
                with state_lock:
                    active[subject] = {"process": process, "created": created, "log": log, "tree": tree}
                atomic_record(directory/f"worker.{subject}.{run_id}.handle.json", owner | {
                    "worker_pid": process.pid, "worker_created": created, "command": command,
                    "plan_file_sha256": plan_sha, "launch_cpu_percent": decision["host_cpu_percent"],
                    "launch_available_memory_bytes": decision["available_memory_bytes"],
                    "estimated_worker_peak_bytes": decision["estimated_worker_peak_bytes"]})
            time.sleep(.1)
        reduce_program(plan_file)
        status = "completed"
    except BaseException as error:
        status = "resource_blocked" if isinstance(error, LocalResourceBlockedError) else "execution_failed_or_interrupted"
        code = 1
        failure = {"type": type(error).__name__, "detail": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        if not monitor.stop():
            cleanup_errors.append({"stage": "control_thread_join", "detail": "control thread is still alive; publication disabled"})
        if monitor.failure is not None and failure is None:
            failure = monitor.failure
            status, code = "control_monitor_failed", 1
        try:
            heartbeat(True, abort=True)
        except BaseException as error:
            cleanup_errors.append({"type": type(error).__name__, "detail": str(error), "stage": "stop_heartbeat"})
        # Only the subprocesses whose PID/birth tuple this controller created.
        for tree in trees:
            try:
                cleanup.append({"launcher_pid": tree.pid, "launcher_created": tree.created,
                                "remaining": tree.stop()})
            except BaseException as error:
                cleanup_errors.append({"type": type(error).__name__, "detail": str(error), "stage": "owned_tree_cleanup"})
        for item in active.values():
            try:
                item["process"].wait(timeout=3)
            except subprocess.TimeoutExpired:
                cleanup_errors.append({"stage": "launcher_wait", "pid": item["process"].pid})
        for log in logs:
            if not log.closed:
                log.close()
        remaining = sum(len(row["remaining"]["actual_processes"]) for row in cleanup)
        if remaining or cleanup_errors:
            status, code = "cleanup_failed", 1
        atomic_record(directory/f"terminal.{run_id}.json", owner | {"status": status, "exit_code": code,
            "finished_time": time.time(), "original_failure": failure, "owned_tree_cleanup": cleanup,
            "cleanup_errors": cleanup_errors, "owned_processes_remaining": remaining,
            "control_thread_stopped": not monitor.thread.is_alive(), "control_thread_failure": monitor.failure})
        atomic_record(directory/"terminal.json", read(directory/f"terminal.{run_id}.json"))
        atomic_record(directory/"progress.json", {"status": status, "run_id": run_id,
            "plan_file_sha256": plan_sha, "owned_processes_remaining": remaining,
            "result_sha256": sha(directory/"result.json") if (directory/"result.json").exists() else None})
        if code and failure is None:
            raise RuntimeError("program did not finish with an empty owned process tree")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "worker", "run", "reduce"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--original")
    parser.add_argument("--capsule")
    parser.add_argument("--baseline")
    parser.add_argument("--subject")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.phase == "prepare":
        if not all((args.original, args.capsule, args.baseline)):
            parser.error("prepare needs original, capsule and baseline")
        prepare(args.original, args.capsule, args.baseline, Path(__file__).resolve().parents[1], output)
        return
    config = load_plan(output/"plan.json")
    if Path(__file__).resolve() != Path(config["source_root"])/"experiments/run_stationary_channel_noise_panel.py":
        raise ValueError("execute using the actual frozen program source")
    if args.phase == "run":
        run(output/"plan.json")
    elif args.phase == "reduce":
        reduce_program(output/"plan.json")
    else:
        handle = output/f"worker.{args.subject}.{args.run_id}.handle.json"
        deadline = time.monotonic()+10.
        while not handle.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        if not handle.exists():
            raise RuntimeError("sealed worker handle was not published by controller")
        record = read(handle)
        if record["run_id"] != args.run_id or record["plan_file_sha256"] != sha(output/"plan.json"):
            raise ValueError("worker handle belongs to another prepared controller attempt")
        ancestry = [psutil.Process(), *psutil.Process().parents()]
        if not any(process.pid == record["worker_pid"] and abs(process.create_time()-record["worker_created"]) < .01
                   for process in ancestry):
            raise ValueError("actual Python worker is not descended from its sealed launcher")
        atomic_record(output/f"worker.{args.subject}.{args.run_id}.identity.json", _owner(record) | {
            "actual_worker_pid": os.getpid(), "actual_worker_created": psutil.Process().create_time(),
            "sealed_launcher_pid": record["worker_pid"], "sealed_launcher_created": record["worker_created"],
            "worker_source_sha256": sha(__file__)})
        def progress(phase):
            atomic_record(output/f"progress.{args.subject}.json", {
                "phase": phase, "time": time.time(), "subject": args.subject, "run_id": args.run_id, "worker_pid": os.getpid()})
        worker(output/"plan.json", args.subject, LocalProgramControl(output/"control.json", _owner(record), progress))


if __name__ == "__main__":
    main()
