"""Explicit local scope and shared cloud admission controls; no data or fits."""
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from research import resource_scope as scope
from research import source_execution
from research import source_pool_cloud as cloud
from research.cloud_execution import CloudResult, CloudRun, PhaseResources


def cli(name):
    return importlib.import_module("experiments." + name)


def test_cuda_single_fit_is_explicit_local_and_not_a_cloud_capacity_claim(monkeypatch):
    monkeypatch.setattr(scope, "_managed_worker", lambda: None)
    value = scope.entrypoint_resource_record(entrypoint="local_probe", requested_device="cuda:0",
        effective_device="cuda:0", requested_cpu_threads=2, execution_performed=False)
    assert value["resource_scope"] == "local_development"
    assert value["configured_compute_workers"] == 1
    assert value["source_family_phase"] is None
    assert value["observation_only_not_cloud_admission"] is True
    assert value["execution_performed"] is False


@pytest.mark.parametrize("plan", [{"resource_scope": "cloud"}, {"cloud_resource_floor": {"cpu_threads": 25, "gpu_workers": 4}}])
def test_standalone_plan_cannot_self_declare_cloud_around_the_owner(monkeypatch, plan):
    monkeypatch.setattr(scope, "_managed_worker", lambda: None)
    with pytest.raises(ValueError, match="requires the source-family resource owner"):
        scope.require_standalone_resource_scope(plan)


def test_managed_scope_uses_actual_phase_and_actual_owner_capacity(monkeypatch):
    managed = SimpleNamespace(task=SimpleNamespace(task_id="owned-task",
        resources=PhaseResources("source", 6, 1024, 5, "cuda", interop_threads=1)),
        admission=SimpleNamespace(cpu_budget=25, gpu_worker_limit=4))
    monkeypatch.setattr(scope, "_managed_worker", lambda: managed)
    scope.require_standalone_resource_scope({"resource_scope": "cloud"})
    value = scope.entrypoint_resource_record(entrypoint="owned_adapter", requested_device="cuda", effective_device="cuda")
    assert value["resource_scope"] == "source_family_managed"
    assert value["source_family_phase"]["cpu_threads"] == 6
    assert value["owner_gpu_slots"] == 4
    managed.admission.cpu_budget = 2
    with pytest.raises(ValueError, match="central cloud resource floor"):
        scope.require_standalone_resource_scope({"resource_scope": "cloud"})


@pytest.mark.parametrize("name,arguments", [
    ("run_pretrain", ["--source-cache", "unused", "--source-snapshot-manifest", "unused", "--checkpoint", "unused"]),
    ("run_pretrain_supervised", ["--source-cache", "unused", "--source-snapshot-manifest", "unused", "--checkpoint", "unused"]),
    ("run_binary_subject_transfer", ["--target-subjects", "unused", "--output", "unused"]),
    ("probe_checkpoint_masks", ["--cache", "unused", "--checkpoint", "unused", "--output", "unused"]),
])
def test_direct_cli_rejects_cloud_mode_before_original_data_access(monkeypatch, name, arguments):
    entry = cli(name)
    if hasattr(entry, "load_epoch_dataset"):
        monkeypatch.setattr(entry, "load_epoch_dataset", lambda *args, **kwargs: pytest.fail("cloud scope reached EEG loading"))
    if name == "run_pretrain":
        monkeypatch.setattr(sys, "argv", [name, *arguments, "--resource-scope", "cloud"])
        call = entry.main
    else:
        def call():
            return entry.main([*arguments, "--resource-scope", "cloud"])
    with pytest.raises(SystemExit) as error:
        call()
    assert error.value.code == 2


def test_cross_dataset_and_promotion_cloud_claims_do_not_reach_data_or_dispatch(tmp_path, monkeypatch):
    path = tmp_path/"cloud-plan.json"
    path.write_text(json.dumps({"resource_scope": "cloud"}), encoding="utf8")
    transfer = cli("run_cross_dataset_transfer")
    with pytest.raises(ValueError, match="source-family resource owner"):
        transfer.execute(SimpleNamespace(plan=path, device="cpu", cpu_threads=1))
    promotion = cli("run_candidate_promotion_matrix")
    monkeypatch.setattr(promotion, "load_plan", lambda *args, **kwargs: pytest.fail("unowned cloud reached DAG dispatch"))
    with pytest.raises(ValueError, match="source-family resource owner"):
        promotion.main(["--plan", str(path)])


def ready(study):
    return {"cache": f"unused-{study}.npz", "cache_sha256": study*64,
        "cache_record_sha256": "c"*64, "manifest_sha256": "d"*64,
        "preparation_method": {"scope": "software_correctness"}}


def test_explicit_cloud_cli_dispatches_to_shared_owner_and_never_direct_pool(tmp_path, monkeypatch):
    entry = cli("run_source_pool_matrix")
    task = {"id": "one", "studies": ["a"], "channels": None, "excluded": ["held"], "dose": "full", "seed": 7}
    monkeypatch.setattr(entry, "verified_executed_snapshot", lambda *args, **kwargs: {"archive_sha256": "a"*64})
    monkeypatch.setattr(entry, "select_source_matrix_tasks", lambda **kwargs: (task,))
    monkeypatch.setattr(entry, "source_matrix_tasks", lambda *args: (task,))
    monkeypatch.setattr(entry, "matched_dose_tasks", lambda *args: ())
    monkeypatch.setattr(entry, "available_cpu_threads", lambda: 32)
    monkeypatch.setattr(entry, "PreparedSourceInventory", lambda *args, **kwargs:
        SimpleNamespace(inspect=lambda studies: {"ready": {"a": ready("a")}, "pending": {}, "failed": {}}))
    monkeypatch.setattr(entry, "ProcessPoolExecutor", lambda *args, **kwargs: pytest.fail("cloud used the unowned legacy pool"))
    calls = []
    monkeypatch.setattr(cloud, "execute_cloud_source_pool", lambda **kwargs: calls.append(kwargs))
    entry.main(["--prepared", "unused", "--output", str(tmp_path/"run"), "--snapshot", "unused",
        "--workers", "4", "--threads-per-worker", "6", "--resource-scope", "cloud", "--device", "cuda"])
    assert len(calls) == 1
    assert calls[0]["tasks"] == (task,)
    assert calls[0]["workers"] == 4
    assert calls[0]["device"] == "cuda"
    assert json.loads((tmp_path/"run/plan.json").read_bytes())["resource_owner"] == "source_family_resource_owner"


@pytest.mark.parametrize("failed_input", [False, True])
def test_shared_owner_central_floor_ready_tail_and_complete_requested_ledger(tmp_path, monkeypatch, failed_input):
    class Manager:
        HostMemoryState = source_execution._HostMemoryState
        def __enter__(self):
            return self
        def __exit__(self, *error):
            return False
    monkeypatch.setattr(cloud.multiprocessing, "get_context", lambda *args: SimpleNamespace(Manager=Manager))
    monkeypatch.setattr(cloud, "available_cpu_threads", lambda: 25)
    resolver = source_execution.resolve_source_resources
    captured = []
    def resolve(plan, count, device):
        assert plan["resource_scope"] == "cloud"
        return resolver(plan, count, device, cpu_count=25,
            free_device_bytes=32*2**30, free_host_bytes=128*2**30)
    monkeypatch.setattr(cloud, "resolve_source_resources", resolve)

    class Executor:
        def __init__(self, admission, **kwargs):
            self.admission = admission
        def run(self, tasks, *, on_finished, on_observation):
            pending, outcomes, held = list(tasks), {}, []
            while pending:
                task = pending.pop(0)
                captured.append(task)
                if task.adapter.endswith("wait_source_pool_input"):
                    outcome = CloudResult(task.task_id, "failed", failure={"reason": "synthetic missing source"})
                else:
                    assert task.adapter.endswith("execute_original_source_pool_group")
                    assert (task.resources.cpu_threads, task.resources.native_threads, task.resources.interop_threads) == (6, 5, 1)
                    token = self.admission.reserve_phase("source", cpu_threads=6,
                        memory_bytes=task.resources.memory_bytes, gpu_workers=1, task_id=task.task_id)
                    assert token is not None
                    held.append(token)
                    outcome = CloudResult(task.task_id, "completed",
                        [{"id": row["id"], "status": "completed"} for row in task.arguments[0]])
                outcomes[task.task_id] = outcome
                pending.extend(on_finished(outcome))
            on_observation({"resources": self.admission.record(), "running": [], "pending": []})
            for token in held:
                self.admission.release_phase(token)
            return CloudRun(outcomes, {"software_stub": True}, [])
    monkeypatch.setattr(cloud, "CloudExecutor", Executor)
    tasks = [{"id": f"task-{study}", "studies": [study], "channels": None,
        "excluded": ["held"], "dose": "full", "seed": 7} for study in ("a", "b")]
    initial = {"ready": {"a": ready("a"), **({} if failed_input else {"b": ready("b")})},
        "pending": {"b": "waiting"} if failed_input else {}, "failed": {}}
    tmp_path.mkdir(exist_ok=True)
    arguments = dict(groups=[[task] for task in tasks], tasks=tasks, dose_tasks=(), prepared_directory="unused",
        initial=initial, output=tmp_path, snapshot={"archive_sha256": "a"*64}, config={"fixed_updates": 4},
        preprocessing_json="{}", workers=4, threads_per_worker=6, device="cuda", wait_timeout=1)
    if failed_input:
        with pytest.raises(RuntimeError, match="complete requested"):
            cloud.execute_cloud_source_pool(**arguments)
    else:
        resources = cloud.execute_cloud_source_pool(**arguments)
        assert resources["cloud_resource_floor"] == {"cpu_threads": 25, "gpu_workers": 4}
        assert resources["source_workers"] == 2  # Two actual groups; no invented fits.
    result = json.loads((tmp_path/"progress.json").read_bytes())
    assert result["requested_tasks"] == len(result["results"]) == 2
    assert sum(row["status"] == "completed" for row in result["results"]) == (1 if failed_input else 2)
    assert result["resource_observation"]["reserved_cpu_threads"] == 1
    assert result["resource_observation"]["active_gpu_groups"] == 0
    assert all(task.arguments[0][0]["seed"] == 7 for task in captured if task.adapter.endswith("execute_original_source_pool_group"))


def test_observation_keeps_queued_ready_groups_distinct_from_a_dependency_tail(tmp_path, monkeypatch):
    class Manager:
        HostMemoryState = source_execution._HostMemoryState
        def __enter__(self):
            return self
        def __exit__(self, *error):
            return False
    monkeypatch.setattr(cloud.multiprocessing, "get_context", lambda *args: SimpleNamespace(Manager=Manager))
    monkeypatch.setattr(cloud, "available_cpu_threads", lambda: 25)
    resolver = source_execution.resolve_source_resources
    monkeypatch.setattr(cloud, "resolve_source_resources", lambda plan, count, device:
        resolver(plan, count, device, cpu_count=25, free_device_bytes=32*2**30, free_host_bytes=128*2**30))
    class StopAfterObservationError(Exception):
        pass
    class Executor:
        def __init__(self, admission, **kwargs):
            self.admission = admission
        def run(self, tasks, *, on_finished, on_observation):
            ids = [task.task_id for task in tasks]
            assert len(ids) == 6
            state = self.admission.record()
            state["source_groups"] = 2
            on_observation({"resources": state, "running": ids[:2], "pending": ids[2:], "admitting": []})
            raise StopAfterObservationError
    monkeypatch.setattr(cloud, "CloudExecutor", Executor)
    tasks = [{"id": f"t{i}", "studies": ["a"], "channels": None,
        "excluded": [f"held{i}"], "dose": "full", "seed": 7} for i in range(6)]
    with pytest.raises(StopAfterObservationError):
        cloud.execute_cloud_source_pool(groups=[[task] for task in tasks], tasks=tasks, dose_tasks=(),
            prepared_directory="unused", initial={"ready": {"a": ready("a")}, "pending": {}, "failed": {}},
            output=tmp_path, snapshot={"archive_sha256": "a"*64}, config={"fixed_updates": 4},
            preprocessing_json="{}", workers=4, threads_per_worker=6, device="cuda", wait_timeout=1)
    observation = json.loads((tmp_path/"progress.json").read_bytes())
    assert observation["actual_source_group_claims"] == 2
    assert observation["waiting_ready_source_groups"] == 4
    assert observation["actual_ready_source_groups"] == 6
    assert observation["dependency_tail_below_gpu_floor"] is False


def test_matched_dose_preparation_unlocks_only_its_declared_source_groups(tmp_path, monkeypatch):
    class Manager:
        HostMemoryState = source_execution._HostMemoryState
        def __enter__(self):
            return self
        def __exit__(self, *error):
            return False
    monkeypatch.setattr(cloud.multiprocessing, "get_context", lambda *args: SimpleNamespace(Manager=Manager))
    monkeypatch.setattr(cloud, "available_cpu_threads", lambda: 25)
    resolver = source_execution.resolve_source_resources
    monkeypatch.setattr(cloud, "resolve_source_resources", lambda plan, count, device:
        resolver(plan, count, device, cpu_count=25, free_device_bytes=32*2**30, free_host_bytes=128*2**30))
    seen = []
    class Executor:
        def __init__(self, admission, **kwargs):
            self.admission = admission
        def run(self, tasks, *, on_finished, on_observation):
            tasks = list(tasks)
            assert len(tasks) == 2
            original = next(task for task in tasks if task.adapter.endswith("execute_original_source_pool_group"))
            prepare = next(task for task in tasks if task.adapter.endswith("prepare_source_pool_dose"))
            assert original.arguments[0][0]["dose"] == "full"
            assert prepare.resources.phase == "preparation" and prepare.resources.device == "cpu"
            first = CloudResult(original.task_id, "completed", [{"id": "full", "status": "completed"}])
            assert on_finished(first) == []
            dose = {"software_source_only_dose": [3, 1]}
            prepared = CloudResult(prepare.task_id, "completed", dose)
            additions = on_finished(prepared)
            assert len(additions) == 1
            matched = additions[0]
            assert matched.arguments[0][0]["dose"] == "matched"
            assert matched.arguments[5] is dose
            assert matched.arguments[4] == {"fixed_updates": 4}
            final = CloudResult(matched.task_id, "completed", [{"id": "matched", "status": "completed"}])
            assert on_finished(final) == []
            seen.extend((original, prepare, matched))
            return CloudRun({row.task_id: row for row in (first, prepared, final)}, {}, [])
    monkeypatch.setattr(cloud, "CloudExecutor", Executor)
    tasks = [{"id": dose, "studies": ["a"], "channels": None, "excluded": ["held"],
        "dose": dose, "seed": 7} for dose in ("full", "matched")]
    cloud.execute_cloud_source_pool(groups=[[task] for task in tasks], tasks=tasks, dose_tasks=(tasks[1],),
        prepared_directory="unused", initial={"ready": {"a": ready("a")}, "pending": {}, "failed": {}},
        output=tmp_path, snapshot={"archive_sha256": "a"*64}, config={"fixed_updates": 4},
        preprocessing_json="{}", workers=4, threads_per_worker=6, device="cuda", wait_timeout=1)
    assert len(seen) == 3
    final = json.loads((tmp_path/"progress.json").read_bytes())
    assert final["status"] == "completed"
    assert [row["id"] for row in final["results"]] == ["full", "matched"]
