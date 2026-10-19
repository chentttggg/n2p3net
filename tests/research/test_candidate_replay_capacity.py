"""Reject impossible saved-artifact replay capacity before any fit can start."""

import json
from multiprocessing import get_context

import pytest

from experiments.run_candidate_ensemble_study import CandidateCloudProgramme
from research.source_execution import SourceInputAdmission, resolve_source_resources


@pytest.mark.parametrize("native", (1, 4))
def test_prefix_replay_rejects_cpu_budget_missing_interop_before_output(tmp_path, native):
    plan = {"evaluation_design": "independent_prefix", "records": [], "source_checkpoints": {}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    resources = resolve_source_resources({"cpu_budget": native + 1, "source_workers": 1,
        "target_workers": 1, "evidence_scope": "software_correctness",
        "source_worker_memory_gib": .125, "target_worker_memory_gib": .125},
        1, "cpu", cpu_count=8, free_host_bytes=2**30)
    output = tmp_path / "consumer"
    with get_context("spawn").Manager() as manager:
        inputs = SourceInputAdmission(manager, capacity_bytes=2**20, load_workers=1)
        with pytest.raises(ValueError, match="replay.*CPU"):
            CandidateCloudProgramme(path, output, resources, task_prefix="capacity-control",
                dependencies=(), input_admission=inputs)
    assert not output.exists()


@pytest.mark.parametrize("native", (1, 4))
def test_standalone_default_admits_original_native_plus_interop(tmp_path, monkeypatch, native):
    import experiments.run_candidate_ensemble_study as runner
    import research.source_execution as source_execution
    from research.cloud_execution import target_phase_resources
    from research.source_execution import _HostMemoryState

    class StopBeforeDispatchError(Exception):
        pass

    plan = {"workers": 1, "cpu_threads_per_worker": native, "records": [{}],
        "evidence_scope": "software_correctness", "resume_binding": None,
        "evaluation_design": "independent_prefix"}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(runner, "_prepare_candidate_plan", lambda *args, **kwargs: (plan, {}, {}))

    def inspect_capacity(resource_plan, count, device):
        resources = resolve_source_resources(resource_plan, count, device,
            cpu_count=16, free_host_bytes=8 * 2**30)
        target = target_phase_resources(resources, device)
        assert target.native_threads == native and target.interop_threads == 1
        ledger = _HostMemoryState(resources)
        claim = ledger.reserve_phase("evaluation", native + 1, 2**20, 0, "replay-control", False)
        assert claim is not None, "standalone default cannot admit its original replay native/inter-op programme"
        assert ledger.record()["reserved_cpu_threads"] <= resources["cpu_budget"]
        raise StopBeforeDispatchError

    monkeypatch.setattr(source_execution, "resolve_source_resources", inspect_capacity)
    with pytest.raises(StopBeforeDispatchError):
        runner.run_study(path, tmp_path / "output")
