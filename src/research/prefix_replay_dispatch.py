"""Retain shared input leases through the actual reader and reducer lifetimes.

Scientific task builders own the complete replay DAG. This owner only queues
completed preparation artifacts, adopts bytes without waiting in a callback,
and releases the holder after every declared reader has actually retired.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from research.cloud_execution import (
    CloudResult,
    CloudRun,
    CloudTask,
    _failure,
    assert_no_prior_owned_live,
)
from research.contracts import semantic_sha256
from research.prefix_replay_inputs import PrefixReplayInputHolder


@dataclass
class _PendingInputs:
    outcome: CloudResult
    request_sha: str
    build_readers: object
    capsule_key: str | None


@dataclass
class _ReaderGroup:
    holder: PrefixReplayInputHolder
    readers: set[str]


def _retired_reader(outcome):
    if not isinstance(outcome, CloudResult):
        raise TypeError("replay retirement requires an actual CloudExecutor result")
    worker = outcome.worker
    if worker is None:
        # These controller outcomes have no launched scientific reader.
        stage = (outcome.failure or {}).get("stage")
        if outcome.status == "blocked" or outcome.status == "failed" and stage == "spawn":
            return
        raise ValueError("replay reader lacks actual process retirement")
    if (worker.get("alive") is not False or type(worker.get("exit_code")) is not int
            or worker.get("process_tree_retirement", {}).get("active_owned_processes_after_exit") != 0):
        raise ValueError("replay input bytes remain owned until the full reader tree retires")


class PrefixReplayInputPool:
    """One parent owner; all methods run on the CloudExecutor callback thread."""

    def __init__(self, input_admission, *, release_backing_aliases=None, validate_reader=None):
        self._admission = input_admission
        self._release_backing_aliases = release_backing_aliases
        self._validate_reader = validate_reader
        self._pending, self._groups, self._readers, self._observed = {}, {}, {}, {}
        self._task_ids = set()
        self.failures = {}

    def _close_holder(self, task_id, holder):
        if self._release_backing_aliases is not None:
            self._release_backing_aliases(task_id)
        holder.close()

    def _reject_preparation(self, task_id, pending, error):
        failure = _failure(error, "artifact_admission")
        self.failures[task_id] = failure
        if pending.outcome.failure is None:
            pending.outcome.status, pending.outcome.failure = "failed", failure
        else:
            pending.outcome.supplementary_failures.append(failure)

    def _reader_tasks(self, tasks):
        tasks = tuple(tasks)
        if not tasks or any(not isinstance(task, CloudTask) for task in tasks):
            raise ValueError("shared replay inputs require their declared reader task frontier")
        ids = [task.task_id for task in tasks]
        if len(set(ids)) != len(ids) or set(ids) & self._task_ids:
            raise ValueError("shared replay input tasks repeat a reader identity")
        if any(task.resources.phase != "evaluation" or task.resources.device != "cpu"
               or type(task.resources.native_threads) is not int or task.resources.native_threads < 1
               or task.resources.interop_threads is not None and (
                   type(task.resources.interop_threads) is not int or task.resources.interop_threads < 1)
               or type(task.resources.cpu_threads) is not int
               or task.resources.cpu_threads != task.resources.native_threads
                   + (task.resources.interop_threads or 0) for task in tasks):
            raise ValueError("shared replay readers require a fully accounted CPU evaluation programme")
        if self._validate_reader is not None:
            for task in tasks:
                self._validate_reader(task)
        return tasks, ids

    def add_readers(self, preparation_task_id, tasks):
        """Register newly unlocked reducers before retiring the last predecessor."""
        tasks, ids = self._reader_tasks(tasks)
        group = self._groups[preparation_task_id]
        group.readers.update(ids)
        self._readers.update((key, preparation_task_id) for key in ids)
        self._task_ids.update(ids)
        return tasks

    def queue_preparation(self, outcome, request, build_readers, *, capsule_key=None):
        if not isinstance(outcome, CloudResult):
            raise TypeError("shared replay preparation requires an actual CloudExecutor result")
        task_id = request["preparation_task_id"]
        if outcome.task_id != task_id or task_id in self._observed:
            raise ValueError("shared replay preparation changed or repeated its original task")
        self._observed[task_id] = outcome
        self._pending[task_id] = _PendingInputs(outcome, semantic_sha256(request), build_readers, capsule_key)
        return self.adopt_pending()

    def adopt_pending(self):
        """Try each ready preparation once; a full byte pool never blocks ACK."""
        additions = []
        for task_id, pending in tuple(self._pending.items()):
            holder = None
            try:
                value = pending.outcome.value
                descriptor = value if pending.capsule_key is None else value[pending.capsule_key]
                options = {} if pending.capsule_key is None else {"result_key": pending.capsule_key}
                holder = PrefixReplayInputHolder.from_prepared(descriptor, request_sha=pending.request_sha,
                    prep_outcome=pending.outcome, admission=self._admission, **options)
                if holder is None:
                    continue
                tasks, ids = self._reader_tasks(pending.build_readers(holder))
            except BaseException as error:
                if holder is not None:
                    self._close_holder(task_id, holder)
                del self._pending[task_id]
                if not isinstance(error, Exception):
                    raise
                # A deferred rejection belongs to its preparation, never the
                # unrelated completed reader whose release enabled the retry.
                self._reject_preparation(task_id, pending, error)
                continue
            self._groups[task_id] = _ReaderGroup(holder, set(ids))
            self._readers.update((key, task_id) for key in ids)
            self._task_ids.update(ids)
            del self._pending[task_id]
            additions.extend(tasks)
        return tuple(additions)

    def finished_reader(self, outcome):
        _retired_reader(outcome)
        group_id = self._readers[outcome.task_id]
        group = self._groups[group_id]
        self._observed[outcome.task_id] = outcome
        group.readers.remove(outcome.task_id)
        del self._readers[outcome.task_id]
        if not group.readers:
            self._close_holder(group_id, group.holder)
            del self._groups[group_id]
        return self.adopt_pending()

    def retire_after_run(self, run):
        """Release unpublished readers only after this executor's settled exit."""
        if not isinstance(run, CloudRun) or any(run.results.get(key) is not value
                for key, value in self._observed.items()):
            raise ValueError("shared replay cleanup requires its actual originating CloudExecutor run")
        directory = Path(run.diagnostics["directory"])
        receipts = list(directory.glob(".cloud_controller.*.json"))
        if len(receipts) != 1:
            raise ValueError("shared replay cleanup lacks its unique controller ownership receipt")
        record = json.loads(receipts[0].read_bytes())
        if record["identity"]["pool_instance_id"] != run.diagnostics["identity"]["pool_instance_id"]:
            raise ValueError("shared replay cleanup changed its actual controller identity")
        assert_no_prior_owned_live(directory)
        for task_id, group in self._groups.items():
            self._close_holder(task_id, group.holder)
        for task_id, pending in self._pending.items():
            self._reject_preparation(task_id, pending, ValueError(
                "shared replay input bytes could not be admitted before the executor retired"))
        self._groups.clear()
        self._readers.clear()
        self._pending.clear()

    def record(self):
        return {"pending_preparations": list(self._pending),
                "resident_preparations": list(self._groups), "active_readers": list(self._readers),
                "failed_preparations": dict(self.failures)}
