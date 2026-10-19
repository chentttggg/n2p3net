"""A failed actor cannot promote an old JSON record around trajectory admission."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from research import source_execution
from research import source_pool_cloud as owner
from research.cloud_execution import CloudResult, CloudRun


def test_old_self_declared_completed_result_stays_unadmitted_after_worker_failure(tmp_path, monkeypatch):
    class Manager:
        HostMemoryState = source_execution._HostMemoryState
        def __enter__(self):
            return self
        def __exit__(self, *error):
            return False
    monkeypatch.setattr(owner.multiprocessing, "get_context", lambda *args: SimpleNamespace(Manager=Manager))
    monkeypatch.setattr(owner, "available_cpu_threads", lambda: 25)
    resolver = source_execution.resolve_source_resources
    monkeypatch.setattr(owner, "resolve_source_resources", lambda plan, count, device:
        resolver(plan, count, device, cpu_count=25, free_device_bytes=32 * 2**30, free_host_bytes=128 * 2**30))
    class Executor:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, tasks, *, on_finished, on_observation):
            task, = tasks
            outcome = CloudResult(task.task_id, "failed", failure={"type": "InputBindingMismatch", "message": "software stale-input control"})
            on_finished(outcome)
            return CloudRun({task.task_id: outcome}, {}, [])
    monkeypatch.setattr(owner, "CloudExecutor", Executor)
    task = {"id": "one", "studies": ["a"], "channels": None, "excluded": ["held"], "dose": "full", "seed": 7}
    snapshot = {"archive_sha256": "d" * 64}
    prepared = {"a": {"cache": "not-opened.npz", "cache_sha256": "a" * 64,
        "cache_record_sha256": "b" * 64, "manifest_sha256": "c" * 64,
        "preparation_method": {"scope": "software_correctness"}}}
    retained = tmp_path / "one/result.json"
    retained.parent.mkdir()
    old = {"task": task, "status": "completed", "source_snapshot": snapshot,
        "input_caches": {"a": "old-input-digest"}, "trajectory": {"checkpoints": {"4": "preserved-not-opened.pt"}}}
    retained.write_text(json.dumps(old), encoding="utf8")
    original = retained.read_bytes()
    with pytest.raises(RuntimeError, match="complete requested"):
        owner.execute_cloud_source_pool(groups=[[task]], tasks=(task,), dose_tasks=(),
            prepared_directory="not-opened", initial={"ready": prepared, "pending": {}, "failed": {}},
            output=tmp_path, snapshot=snapshot, config={"fixed_updates": 4}, preprocessing_json="{}",
            workers=4, threads_per_worker=6, device="cuda", wait_timeout=1)
    record = json.loads((tmp_path / "progress.json").read_bytes())
    assert record["requested_tasks"] == len(record["results"]) == 1
    row = record["results"][0]
    assert row["status"] == "failed"
    observation = row["retained_result_record"]
    assert observation["observation_only_not_admission"] is True
    assert observation["sha256"] == hashlib.sha256(original).hexdigest()
    assert retained.read_bytes() == original


def test_controller_interruption_preserves_completed_rows_and_requested_denominator(tmp_path, monkeypatch):
    import torch

    class Manager:
        HostMemoryState = source_execution._HostMemoryState
        def __enter__(self):
            return self
        def __exit__(self, *error):
            return False
    monkeypatch.setattr(owner.multiprocessing, "get_context", lambda *args: SimpleNamespace(Manager=Manager))
    monkeypatch.setattr(owner, "available_cpu_threads", lambda: 25)
    resolver = source_execution.resolve_source_resources
    monkeypatch.setattr(owner, "resolve_source_resources", lambda plan, count, device:
        resolver(plan, count, device, cpu_count=25, free_device_bytes=32 * 2**30, free_host_bytes=128 * 2**30))
    class Executor:
        def __init__(self, admission, **kwargs):
            self.admission = admission
        def run(self, tasks, *, on_finished, on_observation):
            first, second = tasks
            assert torch.get_num_threads() == 1
            token = self.admission.reserve_phase("source", cpu_threads=first.resources.cpu_threads,
                memory_bytes=first.resources.memory_bytes, gpu_workers=1, task_id=first.task_id)
            assert token is not None
            self.admission.release_phase(token)
            on_finished(CloudResult(first.task_id, "completed", [{"id": "first", "status": "completed"}]))
            raise KeyboardInterrupt("synthetic controller interruption")
    monkeypatch.setattr(owner, "CloudExecutor", Executor)
    tasks = [{"id": name, "studies": ["a"], "channels": None, "excluded": [name], "dose": "full", "seed": 7}
             for name in ("first", "second")]
    ready = {"cache": "not-opened.npz", "cache_sha256": "a" * 64,
        "cache_record_sha256": "b" * 64, "manifest_sha256": "c" * 64,
        "preparation_method": {"scope": "software_correctness"}}
    with pytest.raises(KeyboardInterrupt, match="synthetic controller interruption"):
        owner.execute_cloud_source_pool(groups=[[task] for task in tasks], tasks=tasks, dose_tasks=(),
            prepared_directory="not-opened", initial={"ready": {"a": ready}, "pending": {}, "failed": {}},
            output=tmp_path, snapshot={"archive_sha256": "d" * 64}, config={"fixed_updates": 4},
            preprocessing_json="{}", workers=4, threads_per_worker=6, device="cuda", wait_timeout=1)
    record = json.loads((tmp_path / "progress.json").read_bytes())
    assert record["status"] == "interrupted"
    assert record["requested_tasks"] == len(record["results"]) == 2
    assert [row["id"] for row in record["results"]] == ["first", "second"]
    assert [row["status"] for row in record["results"]] == ["completed", "failed"]
    assert record["resource_observation"]["reserved_cpu_threads"] == 1
    assert record["resource_observation"]["active_gpu_groups"] == 0
