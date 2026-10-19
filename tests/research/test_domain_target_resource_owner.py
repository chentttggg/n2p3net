"""Actual CLI routing and shared target leases without EEG or model fits."""
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research import domain_target_execution as owner
from research import source_execution
from research.cloud_execution import CloudResult, CloudRun


def source_policy():
    return {"target_workers": 4, "source_workers": 4, "cpu_budget": 25,
            "worker_cpu_budget": 24, "gpu_slots": 4, "native_threads_per_worker": 6}


def entrypoint():
    return importlib.import_module("experiments.run_domain_transfer_matrix")


@pytest.mark.parametrize("scope", ["cloud", "local_development"])
def test_actual_domain_cli_keeps_plan_seed_shards_and_parallel_route(tmp_path, monkeypatch, scope):
    entry = entrypoint()
    plan = {"resource_scope": scope, "outer_folds": [{"name": "outer"}]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf8")
    target_plan = tmp_path / "target.json"
    payload = {"source_arm": "P2_source_ce", "checkpoint": str(tmp_path / "source.pt"),
        "target_cache": "unused.npz", "source_seed": 7, "target_seed": 11,
        "target_study": "a", "target_protocol": "binary"}
    target_plan.write_text(json.dumps(payload), encoding="utf8")
    target_bytes = target_plan.read_bytes()
    source = {"resource_policy": source_policy(), "tasks": [{"key": "outer/a",
        "checkpoints": {"100": payload["checkpoint"]}}], "outer_evaluation_jobs": [{"plan": str(target_plan),
        "argv": ["--subjects", "p1,p2,p3,p4"], "shards": ["old0", "old1"],
        "shard_subjects": [["p1", "p3"], ["p2", "p4"]]}]}
    captured, direct = [], []
    monkeypatch.setattr(entry, "execute_source_chain", lambda plan, **kwargs: source)
    monkeypatch.setattr(owner, "execute_cloud_domain_targets", lambda groups, **kwargs: captured.append((groups, kwargs)))

    class Future:
        def __init__(self, jobs):
            self.jobs = jobs
        def result(self):
            return [{"failed_cells": 0, "output": job["output"]} for job in self.jobs]
    class Pool:
        def __init__(self, **kwargs):
            assert scope == "local_development", "cloud reached the unowned direct pool"
            assert kwargs["max_workers"] == 4
        def __enter__(self):
            return self
        def __exit__(self, *error):
            return False
        def submit(self, adapter, group):
            direct.append(group)
            return Future(group)
    monkeypatch.setattr(entry, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(entry, "as_completed", lambda futures: list(futures))
    output = tmp_path / "run"
    entry.main(["--plan", str(path), "--output", str(output), "--device", "cuda"])
    groups = captured[0][0] if scope == "cloud" else direct
    assert len(groups) == 2
    assert [job["target_stage"] for group in groups for job in group] == ["P1", "P2", "P1", "P2"]
    assert [job["subjects"] for group in groups for job in group] == [["p1", "p3"], ["p1", "p3"], ["p2", "p4"], ["p2", "p4"]]
    assert {job["cpu_threads"] for group in groups for job in group} == ({5} if scope == "cloud" else {6})
    assert target_plan.read_bytes() == target_bytes
    assert (payload["source_seed"], payload["target_seed"]) == (7, 11)
    assert len(json.loads((output / "target_jobs.json").read_bytes())["jobs"]) == 4
    if scope == "cloud":
        assert captured[0][1]["plan"]["cloud_resource_floor"] == {"cpu_threads": 25, "gpu_workers": 4}
        assert direct == []
    else:
        assert captured == []


def test_scientific_adapter_delegates_original_group_without_changing_result(monkeypatch):
    import experiments.run_source_training_chain as original

    jobs, result = [{"plan": "source-seed7-target-seed11.json", "subjects": ["held"]}], [{"failed_cells": 0}]
    seen = []
    monkeypatch.setattr(original, "_evaluate_outer_group", lambda values: seen.append(values) or result)
    assert owner.evaluate_domain_target_group(jobs) is result
    assert seen[0] is jobs


@pytest.mark.parametrize("mode", ["success", "group_failure", "interrupt"])
def test_ready_parallel_capacity_complete_ledger_and_lease_return(tmp_path, monkeypatch, mode):
    class Manager:
        HostMemoryState = source_execution._HostMemoryState
        def __enter__(self):
            return self
        def __exit__(self, *error):
            return False
    monkeypatch.setattr(owner.multiprocessing, "get_context", lambda *args: SimpleNamespace(Manager=Manager))
    resolver = source_execution.resolve_source_resources
    monkeypatch.setattr(owner, "resolve_source_resources", lambda plan, count, device:
        resolver(plan, count, device, cpu_count=25, free_device_bytes=32 * 2**30, free_host_bytes=128 * 2**30))
    held_resources = []

    class Executor:
        def __init__(self, admission, **kwargs):
            self.admission = admission
            assert kwargs["max_workers"] == 4
        def run(self, tasks, *, on_finished, on_observation):
            tasks = list(tasks)
            assert len(tasks) == 6
            assert torch.get_num_threads() == 1  # Controller stays inside its own claim.
            ids = [task.task_id for task in tasks]
            tokens = []
            for task in tasks[:4]:
                phase = task.resources
                assert (phase.phase, phase.cpu_threads, phase.native_threads, phase.interop_threads) == ("calibration", 6, 5, 1)
                assert phase.defer_gpu is True
                token = self.admission.reserve_phase("calibration", cpu_threads=6, memory_bytes=phase.memory_bytes,
                    gpu_workers=0, gpu_capable=True, task_id=task.task_id)
                assert token is not None
                assert self.admission._ledger.try_activate_target_gpu(task.task_id) is not None
                tokens.append(token)
            assert self.admission.reserve_phase("calibration", cpu_threads=6, memory_bytes=tasks[4].resources.memory_bytes,
                gpu_workers=0, gpu_capable=True, task_id=tasks[4].task_id) is None
            state = self.admission.record()
            assert state["reserved_cpu_threads"] == 25 and state["active_gpu_groups"] == 4
            held_resources.append(state)
            on_observation({"resources": state, "running": ids[:4], "pending": ids[4:], "admitting": []})
            observation = json.loads((tmp_path / "matrix.json").read_bytes())
            assert observation["actual_ready_target_groups"] == 6
            assert observation["dependency_tail_below_gpu_floor"] is False
            for token in tokens:
                self.admission.release_phase(token)
            result = {}
            for index, task in enumerate(tasks):
                if mode == "interrupt" and index == 1:
                    raise KeyboardInterrupt("software controller interruption")
                if mode == "group_failure" and index == 1:
                    retained = Path(task.arguments[0]["output"]) / "record.json"
                    retained.parent.mkdir(parents=True)
                    retained.write_text('{"status":"running"}', encoding="utf8")
                    outcome = CloudResult(task.task_id, "failed", failure={"type": "SyntheticWorkerError", "message": "software failure"})
                else:
                    outcome = CloudResult(task.task_id, "completed", [{"output": task.arguments[0]["output"],
                        "completed_cells": 2, "failed_cells": 0, "reused_cells": 1}])
                result[task.task_id] = outcome
                assert on_finished(outcome) == ()
            return CloudRun(result, {"software_stub": True}, [])
    monkeypatch.setattr(owner, "CloudExecutor", Executor)
    groups = [[{"plan": "source-seed7-target-seed11.json", "subjects": [f"p{index}"],
        "output": str(tmp_path / f"group{index}"), "cpu_threads": 5, "device": "cuda"}] for index in range(6)]
    progress = {"program": "P1_P2", "status": "running_targets", "completed_groups": [], "failed_groups": []}
    arguments = dict(plan={"resource_scope": "cloud"}, source_policy=source_policy(),
        output=tmp_path, progress=progress, started=time.time(), device="cuda")
    if mode == "success":
        resources = owner.execute_cloud_domain_targets(groups, **arguments)
        assert resources["target_workers"] == 4 and resources["target_native_threads_per_worker"] == 5
    elif mode == "group_failure":
        with pytest.raises(RuntimeError, match="complete requested groups"):
            owner.execute_cloud_domain_targets(groups, **arguments)
    else:
        with pytest.raises(KeyboardInterrupt, match="software controller interruption"):
            owner.execute_cloud_domain_targets(groups, **arguments)
    record = json.loads((tmp_path / "matrix.json").read_bytes())
    assert record["requested_target_group_count"] == record["requested_target_job_count"] == len(record["target_group_outcomes"]) == 6
    assert record["status"] == {"success": "completed", "group_failure": "failed", "interrupt": "interrupted"}[mode]
    assert sum(row["status"] == "completed" for row in record["target_group_outcomes"]) == {"success": 6, "group_failure": 5, "interrupt": 1}[mode]
    assert [row["jobs"] for row in record["target_group_outcomes"]] == groups
    assert record["resource_observation"]["reserved_cpu_threads"] == 1
    assert record["resource_observation"]["active_gpu_groups"] == 0
    assert len(held_resources) == 1
    if mode == "group_failure":
        assert record["target_group_outcomes"][1]["retained_result_records"][0]["observation_only_not_admission"] is True


def test_prelaunch_admission_failure_leaves_a_terminal_complete_requested_record(tmp_path, monkeypatch):
    resolver = source_execution.resolve_source_resources
    monkeypatch.setattr(owner, "resolve_source_resources", lambda plan, count, device:
        resolver(plan, count, device, cpu_count=24, free_device_bytes=32 * 2**30, free_host_bytes=128 * 2**30))
    monkeypatch.setattr(owner.multiprocessing, "get_context", lambda *args: pytest.fail("invalid capacity reached process creation"))
    groups = [[{"plan": "source-seed7-target-seed11.json", "subjects": ["held"],
        "output": str(tmp_path / "one"), "cpu_threads": 5, "device": "cuda"}]]
    progress = {"program": "P1_P2", "status": "running_targets", "completed_groups": [], "failed_groups": []}
    with pytest.raises(ValueError, match="CPU|cpu"):
        owner.execute_cloud_domain_targets(groups, plan={"resource_scope": "cloud"}, source_policy=source_policy(),
            output=tmp_path, progress=progress, started=time.time(), device="cuda")
    result = json.loads((tmp_path / "matrix.json").read_bytes())
    assert result["status"] == "failed"
    assert result["requested_target_job_count"] == result["requested_target_group_count"] == len(result["target_group_outcomes"]) == 1
    assert result["target_group_outcomes"][0]["status"] == "failed"
