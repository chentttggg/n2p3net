"""Run the immutable sixteen-cell stationary program through the shared cloud owner.

Preparation must bind the new source/score plan, exact-SHA input relocation,
sealed Windows original-loader exports, and an accepted cloud controller first.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from uuid import uuid4

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]

from research import cloud_execution  # noqa: E402
from research.cloud_execution import CloudExecutor, CloudTask, PhaseResources  # noqa: E402
from research.local_program_control import (  # noqa: E402
    actual_alive,
    assert_no_owned_live,
    atomic_record,
)
from research.source_execution import (  # noqa: E402
    HostMemoryAdmission,
    process_file_lock,
    resolve_source_resources,
)
from research.stationary_panel import (  # noqa: E402
    ARMS,
    admit_cell,
    cloud_input_owner,
    load_plan,
    read,
    reduce_program,
    sha,
    worker_native_threads,
)


def validate_cloud_declaration(config, resources):
    """Four native5+interop1 claims plus coordinator1 fit the fixed CPU25 budget."""
    declaration = config["cloud_execution"]
    expected = {"cpu_budget": 25, "worker_capacity": 4, "native_threads": 5,
                "interop_threads": 1, "coordinator_cpu_threads": 1, "gpu_claim_per_actor": 0}
    if any(type(declaration.get(key)) is not int or declaration[key] != value for key, value in expected.items()):
        raise ValueError("stationary cloud requires CPU25, four CPU actors, native5/interop1 and GPU claim0")
    if (resources["cpu_budget"] != 25 or resources["coordinator_cpu_threads"] != 1
            or resources["source_workers"] < 4 or resources["target_workers"] < 4
            or worker_native_threads(config) != 5):
        raise ValueError("resolved resources cannot satisfy the declared stationary cloud capacity")
    if resources["gpu_backed_workers"]:
        raise ValueError("stationary cloud scoring is CPU work and must claim zero GPU workers")
    subjects = [f"A{index:02d}" for index in range(1, 9)]
    if (list(config["participants"]) != subjects or len(config["requested_cells"]) != 16
            or config["seed"] != 20260918 or config["fits_allowed"] != 0
            or config["response_fits_allowed"] != 0):
        raise ValueError("stationary cloud changed the original sixteen-cell zero-fit scope")
    return declaration


class StationaryCloudProgramme:
    """The shared executor owns process lifetime; this owner admits every arm."""
    def __init__(self, plan_file, config, resources, input_owner):
        self.plan_file, self.config, self.resources = Path(plan_file), config, resources
        self.input_owner, self.acknowledged = input_owner, set()
        declaration = validate_cloud_declaration(config, resources)
        phase = PhaseResources("evaluation", 6, resources["target_worker_memory_bytes"], 5,
                               "cpu", interop_threads=1)
        self.tasks = tuple(CloudTask(f"stationary:{subject}", "research.stationary_panel:cloud_worker",
            {"task_id": f"stationary:{subject}", "subject": subject, "plan_file": str(self.plan_file),
             "input_owner": declaration["input_owner"], "native_threads": 5, "interop_threads": 1},
            str(self.plan_file.parent/"cells"), phase) for subject in config["participants"])

    def on_finished(self, outcome):
        if outcome.status != "completed":
            return ()  # Keep the shared owner's original failure and full request.
        if outcome.task_id not in {task.task_id for task in self.tasks}:
            raise ValueError("stationary ACK does not belong to a declared subject task")
        retired = outcome.worker or {}
        if (retired.get("alive") is not False or retired.get("exit_code") != 0
                or retired.get("process_tree_retirement", {}).get("active_owned_processes_after_exit") != 0):
            raise ValueError("stationary ACK requires a settled successful cloud actor tree")
        subject = outcome.task_id.removeprefix("stationary:")
        if (subject not in self.config["participants"] or outcome.value != {
                "subject": subject, "plan_file_sha256": sha(self.plan_file), "arms": list(ARMS)}):
            raise ValueError("stationary task result differs from its complete requested subject")
        for arm in ARMS:
            admit_cell(self.config, subject, arm, self.plan_file.parent, input_owner=self.input_owner)
        self.acknowledged.add(subject)
        return ()

    def finish(self, run):
        expected = {task.task_id for task in self.tasks}
        complete = (run.status == "completed" and set(run.results) == expected
                    and self.acknowledged == set(self.config["participants"]))
        receipt = {"schema": "p1_stationary_cloud_queue/1", "status": "completed" if complete else "failed",
            "plan_file_sha256": sha(self.plan_file), "requested_cells": self.config["requested_cells"],
            "requested_tasks": sorted(expected), "acknowledged_subjects": sorted(self.acknowledged),
            "results": {key: asdict(value) for key, value in run.results.items()},
            "diagnostics": run.diagnostics, "observation_failures": run.observation_failures,
            "fits": 0, "response_fits": 0, "gpu_claim_per_actor": 0}
        if not complete:
            atomic_record(self.plan_file.parent/"cloud_queue.json", receipt)
            return receipt
        # New-law metrics are reduced only after every original requested cell
        # has a valid sealed receipt and the executor has settled every tree.
        try:
            from train.runtime import cpu_thread_budget

            with cpu_thread_budget(self.resources["coordinator_cpu_threads"]):
                reduce_program(self.plan_file, input_owner=self.input_owner)
        except BaseException as error:
            receipt.update(status="failed", reduction_failure={"type": type(error).__name__, "message": str(error)})
            atomic_record(self.plan_file.parent/"cloud_queue.json", receipt)
            raise
        atomic_record(self.plan_file.parent/"cloud_queue.json", receipt)
        return receipt


def accepted_controller(declaration):
    """An unresolved late-orphan controller is a HOLD, never an executable plan."""
    descriptor = declaration["controller_binding"]
    if sha(descriptor["path"]) != descriptor["sha256"]:
        raise ValueError("cloud owner controller review binding changed")
    receipt = read(descriptor["path"])
    if (receipt.get("status") != "accepted" or receipt.get("late_orphan_gate_accepted") is not True
            or receipt.get("cloud_execution_sha256") != sha(cloud_execution.__file__)):
        raise ValueError("HOLD: stationary cloud needs the accepted late-orphan-safe controller")


@contextmanager
def programme_attempt(plan_file):
    """One scientific output owner; the shared controller owns historical trees."""
    plan_file = Path(plan_file)
    directory, lease = plan_file.parent, plan_file.parent/"controller.lease.json"
    def reject_live_controller():
        if lease.exists():
            previous = read(lease)
            if actual_alive(previous["controller_pid"], previous["controller_created"]):
                raise RuntimeError("another live controller owns this stationary program")
    reject_live_controller()
    seen_attempts = {path.name for path in directory.glob("controller.*.*.json")}
    # Reuse the shared filesystem transaction lock rather than another OS lock
    # implementation. Recheck the lease after lock acquisition to close races.
    with process_file_lock(directory/"stationary.programme.lock"):
        reject_live_controller()
        if {path.name for path in directory.glob("controller.*.*.json")} != seen_attempts:
            raise RuntimeError("another stationary programme attempt won the output lease")
        assert_no_owned_live(directory)  # Old local attempts retain their gate.
        historical_gate = getattr(cloud_execution, "assert_no_prior_owned_live", None)
        if historical_gate is None:
            raise RuntimeError("HOLD: shared cloud controller has no historical owned-tree admission API")
        # API contract: Path -> None on a verified empty prior owned tree;
        # any live/unknown/changed ownership raises. It never edits science.
        if historical_gate(directory/"cloud_diagnostics") is not None:
            raise RuntimeError("HOLD: historical owned-tree API returned an unrecognised admission result")
        previous = read(lease) if lease.exists() else None
        if previous is not None:
            lease.rename(directory/f"controller.expired.{previous['run_id']}.json")
        process = psutil.Process()
        owner = {"schema": "p1_stationary_programme_attempt/1", "run_id": uuid4().hex,
            "controller_pid": process.pid, "controller_created": process.create_time(),
            "plan_file_sha256": sha(plan_file), "source_sha256": sha(__file__),
            "previous_attempt": previous}
        with lease.open("x", encoding="utf-8") as stream:
            json.dump(owner, stream, allow_nan=False)
        try:
            yield owner
        finally:
            # Keep the attempt evidence at its own immutable name. A crash
            # before this leaves the lease for the next historical-tree gate.
            primary = sys.exception()
            try:
                lease.rename(directory/f"controller.finished.{owner['run_id']}.json")
            except BaseException as lease_error:
                if primary is None:
                    raise
                primary.add_note(f"programme lease archival failed: {lease_error}")


def run(plan_file):
    plan_file = Path(plan_file).resolve()
    raw = read(plan_file)
    declaration = raw["cloud_execution"]
    accepted_controller(declaration)
    inputs = cloud_input_owner(declaration["input_owner"])
    config = load_plan(plan_file, input_owner=inputs)
    if Path(__file__).resolve() != Path(config["source_root"])/"experiments/run_stationary_cloud_panel.py":
        raise ValueError("execute using the new frozen stationary cloud source")
    with programme_attempt(plan_file) as attempt:
        return _run_attempt(plan_file, config, inputs, attempt)


def _run_attempt(plan_file, config, inputs, attempt):
    declaration = config["cloud_execution"]
    resources = resolve_source_resources(declaration["resource_plan"], 8, "cpu")
    programme = StationaryCloudProgramme(plan_file, config, resources, inputs)
    atomic_record(plan_file.parent/f"cloud_resources.{attempt['run_id']}.json", resources)
    try:
        with get_context("spawn").Manager() as manager:
            admission = HostMemoryAdmission(manager, resources)
            executor = CloudExecutor(admission, max_workers=4,
                diagnostic_directory=plan_file.parent/"cloud_diagnostics",
                identity={"plan_file_sha256": sha(plan_file), "program": config["program"], "attempt": attempt})
            result = executor.run(programme.tasks, on_finished=programme.on_finished)
        return programme.finish(result)
    except BaseException as error:
        # An infrastructure/contract interruption preserves its actual primary
        # and all sealed/partial science. Never manufacture numerical evidence.
        try:
            failure = {"schema": "p1_stationary_cloud_execution_failure/1", "status": "failed",
                "plan_file_sha256": sha(plan_file), "requested_cells": config["requested_cells"],
                "acknowledged_subjects": sorted(programme.acknowledged),
                "original_failure": {"type": type(error).__name__, "message": str(error),
                                     "traceback": traceback.format_exc()},
                "sealed_cell_receipts": {str(path): sha(path) for path in sorted(
                    (plan_file.parent/"cells").glob("*/receipt.json"))}}
            atomic_record(plan_file.parent/"cloud_execution_failure.json", failure)
        except BaseException as journal_error:
            error.add_note(f"cloud failure journal could not be published: {journal_error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    return 0 if run(args.plan)["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
