"""Source native/inter-op widths match actual workers and the admitted CPU claim."""

import json
import os
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from research import source_execution as owner


def public_plan(**updates):
    return {"cpu_budget": 25, "source_workers": 4, "target_workers": 4, "gpu_workers": 4,
            "source_load_workers": 4, "source_worker_memory_gib": 2., "target_worker_memory_gib": 2.,
            "shared_staging_memory_gib": .125, **updates}


def resources(plan, device="cuda"):
    return owner.resolve_source_resources(plan, 4, device, cpu_count=25,
        free_device_bytes=32 * 2**30, free_host_bytes=88 * 2**30)


@pytest.mark.parametrize("explicit", [False, True])
def test_public_source_five_native_one_interop_accounts_four_actors_and_coordinator(explicit):
    plan = public_plan(**({"source_native_threads_per_worker": 5} if explicit else {}))
    result = resources(plan)
    assert result["source_workers"] == result["gpu_slots"] == 4
    assert result["source_native_threads_per_worker"] == 5
    assert result["source_interop_threads_per_worker"] == 1
    assert result["source_cpu_threads_per_worker"] == 6
    phase = owner.source_phase_resources(result, "cuda")
    assert (phase.cpu_threads, phase.native_threads, phase.interop_threads) == (6, 5, 1)
    assert phase.memory_bytes == 2 * 2**30 and phase.device == "cuda"
    manager = SimpleNamespace(HostMemoryState=owner._HostMemoryState)
    admission = owner.HostMemoryAdmission(manager, result)
    original = admission.record()
    with ExitStack() as stack:
        for _ in range(4):
            stack.enter_context(admission.source_group())
        live = admission.record()
        assert live["source_groups"] == live["active_gpu_groups"] == 4
        assert live["reserved_cpu_threads"] == 25 and live["free_cpu_threads"] == 0
        assert admission.reserve_phase("source", cpu_threads=6, memory_bytes=2 * 2**30) is None
    assert admission.record() == original


@pytest.mark.parametrize("native", [0, -1, True, 1.5, "5", 6])
def test_invalid_or_overbudget_source_native_does_not_reduce_four_worker_capacity(native):
    with pytest.raises(ValueError, match="source native|source native/inter-op"):
        resources(public_plan(source_native_threads_per_worker=native))


@pytest.mark.parametrize("device,scope", [("cpu", {}), ("cpu", {"evidence_scope": "software_correctness"}),
                                         ("cuda", {"evidence_scope": "software_correctness"}),
                                         ("cuda", {"resource_scope": "local_development"})])
def test_undeclared_cpu_and_software_source_paths_keep_original_thread_policy(device, scope):
    result = resources(public_plan(**scope), device)
    assert result["native_threads_per_worker"] == 6
    assert "source_native_threads_per_worker" not in result
    phase = owner.source_phase_resources(result, device)
    assert phase.cpu_threads == phase.native_threads == 6 and phase.interop_threads is None


def test_actual_e2_caller_keeps_general_native_threads_and_releases_nested_admission(tmp_path, monkeypatch):
    from experiments import run_source_training_chain as runner

    result = resources(public_plan(source_native_threads_per_worker=5))
    admission = owner.HostMemoryAdmission(SimpleNamespace(HostMemoryState=owner._HostMemoryState), result)
    original = admission.record()
    scored, released = [], []

    def score(checkpoint, dataset, cohort, **options):
        scored.append((checkpoint, dataset, cohort, options))
        assert admission.record()["reserved_cpu_threads"] == 7
        return [{"participant_key": "software-person", "actual_K": 0, "status": "completed"}]

    monkeypatch.setattr(runner, "_score_checkpoint", score)
    monkeypatch.setattr(owner, "release_source_worker", released.append)
    dataset = object()
    cohort = [{"participant_key": "software-person"}]
    with admission.source_group():
        outcomes = runner._score_e2_candidates(dataset, tmp_path, cohort, (3,), workers=1,
            progress_path=tmp_path / "unused-progress.json", memory_admission=admission, device="cpu")
        assert admission.native_threads == result["native_threads_per_worker"] == 6
        assert admission.source_cpu_threads == 6
    assert outcomes == [{"participant_key": "software-person", "actual_K": 0,
                         "status": "completed", "updates": 3}]
    assert scored == [(tmp_path / "source.u3.pt", dataset, cohort, {"device": "cpu"})]
    assert released == ["cpu"]
    assert admission.record() == original
    assert not (tmp_path / "source.u3.pt").exists()


def test_actual_source_worker_reapplies_declared_interop_before_first_tensor(tmp_path):
    script = '''
import json, pathlib, sys, torch
from experiments import run_source_training_chain as runner
output = pathlib.Path(sys.argv[1])
runner._assert_source_bytes = lambda *args: None
def execute(plan, *, output, **kwargs):
    native, interop = torch.get_num_threads(), torch.get_num_interop_threads()
    assert (native, interop) == (5, 1), (native, interop)
    tensor = torch.arange(4.)
    assert tensor.sum().item() == 6.
    (output / "source_chain_record.json").write_text(json.dumps({"source_snapshot": {}}))
    print(json.dumps({"native_before_first_tensor": native, "interop_before_first_tensor": interop}))
runner.execute_source_chain = execute
plan = {"outer_folds": [{"name": "software-thread-call"}], "_phase_reserved": True,
        "_native_threads_per_worker": 5, "_source_interop_threads_per_worker": 1,
        "source_snapshot_manifest": "unused-software-snapshot"}
output.mkdir()
runner._source_group_worker((plan, str(output), "cpu", ("P2_source",)))
'''
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-B", "-c", script, str(tmp_path / "source-worker")],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"native_before_first_tensor": 5, "interop_before_first_tensor": 1}


def test_actual_cli_passes_source_phase_native_and_interop_to_child(tmp_path, monkeypatch):
    import research.cloud_execution as execution
    import research.evaluation as evaluation
    from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract
    from experiments import run_source_training_chain as runner
    from tests.support.source_programs import independent_source_plan

    spec = replace(preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT), baseline_mode="none")
    plan = independent_source_plan(tmp_path / "unread-source.npz", spec, tmp_path / "unused-snapshot.json")
    plan.update(public_plan(source_native_threads_per_worker=5))
    originals = owner.resolve_source_resources
    monkeypatch.setattr(owner, "resolve_source_resources", lambda current, count, device:
        originals(current, count, device, cpu_count=25, free_device_bytes=32 * 2**30,
                  free_host_bytes=88 * 2**30))
    monkeypatch.setattr(evaluation, "verified_executed_snapshot", lambda *args, **kwargs: {"archive_sha256": "a" * 64})
    groups = [(dict(plan["outer_folds"][0], name=f"fixed-cache-{index}", target_study=study,
                   excluded_source_datasets=[study]), "six")
              for index, study in enumerate(("BI2014a", "BI2015a", "BNCI2014-008", "Lee2019"))]
    plan["outer_folds"] = [outer for outer, _ in groups]
    captured = []
    class CapturedTasksError(Exception):
        pass
    def run(executor, tasks, **kwargs):
        captured.extend(task for task in tasks if task.resources.phase == "source")
        raise CapturedTasksError
    monkeypatch.setattr(execution.CloudExecutor, "run", run)
    with pytest.raises(CapturedTasksError):
        runner._execute_source_groups(plan, output=tmp_path / "unused-fit", device="cuda",
            stages=("P2_source",), evaluate_outer=False, groups=groups)
    assert len(captured) == 4
    for task in captured:
        assert (task.resources.cpu_threads, task.resources.native_threads, task.resources.interop_threads) == (6, 5, 1)
        child = task.arguments[0]
        assert child["_native_threads_per_worker"] == 5 and child["_source_interop_threads_per_worker"] == 1
    assert not (tmp_path / "unread-source.npz").exists()
