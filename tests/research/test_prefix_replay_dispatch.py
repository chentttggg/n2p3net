"""Actual worker exits drive shared input adoption, dependencies and release."""
from __future__ import annotations

import hashlib
import json
import os
from multiprocessing import get_context
from pathlib import Path

import pytest

from research.cloud_execution import CloudExecutor, CloudTask, PhaseResources
from research.contracts import semantic_sha256
from research.source_execution import HostMemoryAdmission, SourceInputAdmission
from tests.support.prefix_replay_inputs import prepared_prefix_inputs


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    return prepared_prefix_inputs(tmp_path_factory.mktemp("dispatcher-prefix-inputs"))


def fixture_preparation_adapter(arguments):
    """Return an already authorized scientific fixture, with a real OS exit."""
    return arguments["artifact"]


def reader_adapter(arguments):
    from research.prefix_replay_inputs import open_prepared_inputs

    with open_prepared_inputs(arguments["inputs"], expected_request_sha=arguments["request_sha"],
                              expected_capsule_sha=arguments["capsule_sha"]) as borrowed:
        assert borrowed.views.scoring_truth == {}
        value = float(borrowed.views.calibration[200].raw.samples.sum())
        if arguments.get("native_exit"):
            os._exit(79)
        return value


def _two_preparations(prepared, root):
    _, _, original_request, original_artifact = prepared
    original_path = Path(original_artifact["path"])
    original = json.loads(original_path.read_bytes())
    tasks, requests, totals = [], {}, []
    for index in range(2):
        task_id = f"prepare:{index:03d}"
        request = {**original_request, "preparation_task_id": task_id}
        payload = {**original, "request": request, "request_sha256": semantic_sha256(request)}
        directory = root / task_id.replace(":", "_")
        directory.mkdir()
        os.link(original_path.parent / "arrays.bin", directory / "arrays.bin")
        capsule = directory / "capsule.json"
        content = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        capsule.write_bytes(content)
        artifact = {"path": str(capsule), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        tasks.append(CloudTask(task_id, f"{__name__}:fixture_preparation_adapter", {"artifact": artifact},
            str(directory), PhaseResources("preparation", 1, 2**30, 1)))
        requests[task_id] = request
        totals.append(payload["blob"]["bytes"] + len(content))
    assert totals[0] == totals[1]
    return tasks, requests, totals[0]


def _run(prepared, root, *, native_exit=False, adoption_error=False, corrupt_pending=False, dynamic=False,
         release_aliases=False):
    from research.prefix_replay_dispatch import PrefixReplayInputPool

    tasks, requests, capacity = _two_preparations(prepared, root)
    resources = {"host_memory_budget_bytes": 4 * 2**30, "shared_staging_memory_bytes": capacity,
        "source_worker_memory_bytes": 2**30, "target_worker_memory_bytes": 2**30,
        "source_workers": 1, "target_workers": 1, "gpu_slots": 1, "cpu_budget": 3 if dynamic else 2,
        "coordinator_cpu_threads": 1, "native_threads_per_worker": 1}
    events, staged_tasks, aliases = [], {}, {}
    with get_context("spawn").Manager() as manager:
        inputs = SourceInputAdmission(manager, capacity_bytes=capacity, load_workers=1)
        admission = HostMemoryAdmission(manager, resources, input_admission=inputs)
        def release_backing_aliases(task_id):
            # The old physical bytes must disappear before the ledger admits
            # the next preparation, including aliases held by actual tasks.
            assert inputs.record()["resident_input_bytes"] == capacity
            descriptor = aliases[task_id]
            descriptor.pop("payload", None)
            if isinstance(descriptor.get("capsule"), bytes):
                descriptor.pop("capsule")
            events.append((task_id + ":aliases_released", capacity, {}))

        options = {"release_backing_aliases": release_backing_aliases} if release_aliases else {}
        pool = PrefixReplayInputPool(inputs, **options)
        later = {}

        def build(task_id, holder):
            if adoption_error:
                raise OSError("injected after resident allocation, before task publication")
            if release_aliases and task_id.endswith("001"):
                assert "payload" not in aliases["prepare:000"]
                assert not isinstance(aliases["prepare:000"].get("capsule"), bytes)
            descriptor = holder.descriptor
            aliases[task_id] = descriptor
            common = {"inputs": descriptor, "request_sha": semantic_sha256(requests[task_id]),
                      "capsule_sha": tasks[int(task_id[-3:])].arguments["artifact"]["sha256"]}
            phase = PhaseResources("evaluation", 2, 2**30, 1, interop_threads=1) if dynamic else (
                PhaseResources("evaluation", 1, 2**30, 1))
            reader = CloudTask(task_id + ":reader", f"{__name__}:reader_adapter",
                {**common, "native_exit": native_exit and task_id.endswith("000")},
                str(root / (task_id.replace(":", "_") + "_reader")), phase)
            reducer = CloudTask(task_id + ":reducer", f"{__name__}:reader_adapter", common,
                str(root / (task_id.replace(":", "_") + "_reducer")),
                phase, (reader.task_id,))
            staged_tasks.update((task.task_id, task) for task in (reader, reducer))
            if dynamic:
                later[reader.task_id] = task_id, reducer
                return (reader,)
            return reader, reducer

        def callback(outcome):
            events.append((outcome.task_id, inputs.record()["resident_input_bytes"], pool.record()))
            if outcome.task_id in requests:
                additions = pool.queue_preparation(outcome, requests[outcome.task_id],
                    lambda holder: build(outcome.task_id, holder))
                if corrupt_pending and outcome.task_id == "prepare:001":
                    # Replace this staged fixture's hardlink; keep the first
                    # holder's authorized bytes and original MAT unchanged.
                    blob = Path(outcome.value["path"]).parent / "arrays.bin"
                    corrupt = blob.with_suffix(".invalid")
                    corrupt.write_bytes(blob.read_bytes() + b"x")
                    corrupt.replace(blob)
                return additions
            unlocked = ()
            if outcome.task_id in later:
                group_id, reducer = later.pop(outcome.task_id)
                pool.add_readers(group_id, (reducer,))
                unlocked = (reducer,)
            additions = (*unlocked, *pool.finished_reader(outcome))
            if outcome.task_id.endswith(":reader"):
                assert inputs.record()["resident_input_bytes"] == capacity
            return additions

        run = CloudExecutor(admission, max_workers=1, diagnostic_directory=root / "diagnostics",
            identity={"scope": "software_correctness", "fixture": "shared_input_lifetime"},
            poll_seconds=.02).run(tasks, on_finished=callback)
        pool.retire_after_run(run)
        final = inputs.record()
    return run, events, final, staged_tasks, capacity


def test_actual_pending_adoption_retries_after_last_reader_and_reducer_retire(prepared, tmp_path):
    run, events, final, _, capacity = _run(prepared, tmp_path)
    assert run.status == "completed" and len(run.results) == 6
    before_reader = next(row for row in events if row[0] == "prepare:001")
    assert before_reader[1] == capacity
    first_reducer = next(row for row in events if row[0] == "prepare:000:reducer")
    assert first_reducer[2]["pending_preparations"] == ["prepare:001"]
    assert final["resident_input_bytes"] == final["load_active"] == final["nn_active"] == 0
    assert final["free_staging_bytes"] == capacity
    assert all(value.worker["alive"] is False and value.worker["exit_code"] == 0
               and value.worker["process_tree_retirement"]["active_owned_processes_after_exit"] == 0
               for value in run.results.values())
    values = [value.value for key, value in run.results.items() if not key.startswith("prepare:")
              or key.endswith((":reader", ":reducer"))]
    assert len(values) == 4 and len(set(values)) == 1


def test_actual_native_reader_failure_and_blocked_reducer_release_and_preserve_roster(prepared, tmp_path):
    run, _, final, staged, capacity = _run(prepared, tmp_path, native_exit=True)
    failed = run.results["prepare:000:reader"]
    assert failed.status == "failed" and failed.worker["exit_code"] == 79
    assert failed.worker["process_tree_retirement"]["active_owned_processes_after_exit"] == 0
    assert run.results["prepare:000:reducer"].status == "blocked"
    assert run.results["prepare:001:reducer"].status == "completed"
    assert len(staged) == 4 and len(run.results) == 6 and run.status == "failed"
    assert final["free_staging_bytes"] == capacity and final["resident_input_bytes"] == 0


def test_actual_callback_oserror_before_publication_releases_resident_claim(prepared, tmp_path):
    run, _, final, staged, capacity = _run(prepared, tmp_path, adoption_error=True)
    assert run.status == "failed" and len(run.results) == 2 and not staged
    assert all(value.worker["exit_code"] == 0 and value.failure["stage"] == "artifact_admission"
               for value in run.results.values())
    assert final["resident_input_bytes"] == 0 and final["free_staging_bytes"] == capacity


def test_deferred_bad_input_does_not_change_completed_peer_reducer(prepared, tmp_path):
    run, _, final, staged, capacity = _run(prepared, tmp_path, corrupt_pending=True)
    assert run.results["prepare:000:reducer"].status == "completed"
    assert run.results["prepare:001"].status == "failed"
    assert run.results["prepare:001"].worker["exit_code"] == 0
    assert run.results["prepare:001"].failure["stage"] == "artifact_admission"
    assert len(staged) == 2 and len(run.results) == 4
    assert final["resident_input_bytes"] == 0 and final["free_staging_bytes"] == capacity


def test_dynamic_reducer_frontier_retains_holder_with_explicit_interop_budget(prepared, tmp_path):
    run, _, final, _, capacity = _run(prepared, tmp_path, dynamic=True)
    assert run.status == "completed" and len(run.results) == 6
    assert final["resident_input_bytes"] == 0 and final["free_staging_bytes"] == capacity


def test_task_aliases_are_released_before_pending_inputs_take_the_same_budget(prepared, tmp_path):
    run, events, final, tasks, capacity = _run(prepared, tmp_path, release_aliases=True)
    assert run.status == "completed" and len(run.results) == 6
    assert [row[0] for row in events if row[0].endswith(":aliases_released")] == [
        "prepare:000:aliases_released", "prepare:001:aliases_released"]
    assert all("payload" not in task.arguments["inputs"]
               and not isinstance(task.arguments["inputs"].get("capsule"), bytes) for task in tasks.values())
    assert final["free_staging_bytes"] == capacity and final["resident_input_bytes"] == 0
