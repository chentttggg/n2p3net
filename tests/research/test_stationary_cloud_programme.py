"""Complete requests, explicit resource claims and scientific ACK boundaries."""
from __future__ import annotations

import json
from dataclasses import replace
from multiprocessing import get_context
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import run_stationary_cloud_panel as entry
from research import stationary_panel as panel
from research.cloud_execution import CloudExecutor, CloudResult, CloudRun, PhaseResources
from research.local_program_control import atomic_record
from research.source_execution import HostMemoryAdmission


def _config():
    people = {f"A{i:02d}": {} for i in range(1, 9)}
    return {"participants": people,
        "requested_cells": [{"subject": subject, "arm": arm} for subject in people for arm in panel.ARMS],
        "seed": 20260918, "fits_allowed": 0, "response_fits_allowed": 0,
        "resources": {"native_threads": 5},
        "cloud_execution": {"cpu_budget": 25, "worker_capacity": 4, "native_threads": 5,
            "interop_threads": 1, "coordinator_cpu_threads": 1, "gpu_claim_per_actor": 0, "input_owner": {}}}


def _resources():
    return {"cpu_budget": 25, "coordinator_cpu_threads": 1, "source_workers": 4,
            "target_workers": 4, "target_worker_memory_bytes": 100, "gpu_backed_workers": False}


def _programme(tmp_path):
    plan = tmp_path/"plan.json"
    atomic_record(plan, _config())
    return entry.StationaryCloudProgramme(plan, _config(), _resources(), object())


def _outcome(programme, subject):
    return CloudResult(f"stationary:{subject}", "completed", {
        "subject": subject, "plan_file_sha256": panel.sha(programme.plan_file), "arms": list(panel.ARMS)},
        worker={"alive": False, "exit_code": 0, "process_tree_retirement": {"active_owned_processes_after_exit": 0}})


def test_ack_rejects_live_or_unsettled_successful_worker(tmp_path):
    programme = _programme(tmp_path)
    outcome = _outcome(programme, "A01")
    outcome.worker["process_tree_retirement"]["active_owned_processes_after_exit"] = 1
    with pytest.raises(ValueError, match="settled successful"):
        programme.on_finished(outcome)
    assert programme.acknowledged == set()


def test_eight_tasks_keep_sixteen_cells_and_four_cpu_only_phase_claims(tmp_path):
    programme = _programme(tmp_path)
    assert len(programme.tasks) == 8
    assert len(programme.config["requested_cells"]) == 16
    for task in programme.tasks:
        assert task.resources.cpu_threads == 6
        assert task.resources.native_threads == 5
        assert task.resources.interop_threads == 1
        assert task.resources.device == "cpu" and not task.resources.defer_gpu
        assert task.depends_on == ()
    assert 4 * programme.tasks[0].resources.cpu_threads + 1 == 25


@pytest.mark.parametrize("change", [{"cpu_budget": 24}, {"target_workers": 3}, {"gpu_backed_workers": True}])
def test_cloud_capacity_never_silently_downgrades(change):
    with pytest.raises(ValueError):
        entry.validate_cloud_declaration(_config(), _resources() | change)


def test_serial_ack_of_both_arms_precedes_reduce_and_partial_keeps_complete_request(tmp_path, monkeypatch):
    programme = _programme(tmp_path)
    calls = []
    monkeypatch.setattr(entry, "admit_cell", lambda config, subject, arm, *_args, **_kwargs:
                        calls.append((subject, arm)))
    monkeypatch.setattr(entry, "reduce_program", lambda *_args, **_kwargs: calls.append("reduce"))
    outcomes = {task.task_id: _outcome(programme, task.arguments["subject"]) for task in programme.tasks}
    for outcome in outcomes.values():
        programme.on_finished(outcome)
    assert len(calls) == 16 and "reduce" not in calls
    receipt = programme.finish(CloudRun(outcomes, {}, []))
    assert receipt["status"] == "completed" and calls[-1] == "reduce"


def test_ack_contract_failure_keeps_partial_and_primary_in_all_sixteen_requests(tmp_path, monkeypatch):
    programme = _programme(tmp_path)
    def reject(config, subject, arm, *_args, **_kwargs):
        if subject == "A02" and arm == "private8":
            raise ValueError("actual producer mismatch")
    monkeypatch.setattr(entry, "admit_cell", reject)
    reductions = []
    monkeypatch.setattr(entry, "reduce_program", lambda *_args, **_kwargs: reductions.append("reduce"))
    completed = _outcome(programme, "A01")
    programme.on_finished(completed)
    wrong = _outcome(programme, "A02")
    with pytest.raises(ValueError, match="actual producer"):
        programme.on_finished(wrong)
    assert programme.acknowledged == {"A01"}
    wrong.status, wrong.failure = "failed", {"type": "ValueError", "message": "actual producer mismatch"}
    native = CloudResult("stationary:A03", "failed", failure={"type": "NativeExit", "message": "exit7"})
    programme.on_finished(native)
    receipt = programme.finish(CloudRun({row.task_id: row for row in (completed, wrong, native)}, {}, []))
    assert receipt["status"] == "failed" and len(receipt["requested_cells"]) == 16
    assert receipt["results"]["stationary:A03"]["failure"] == native.failure
    assert reductions == [] and not (tmp_path/"result.json").exists()


def test_complete_processes_without_every_scientific_ack_cannot_reduce(tmp_path, monkeypatch):
    programme = _programme(tmp_path)
    monkeypatch.setattr(entry, "reduce_program", lambda *_args, **_kwargs: pytest.fail("unacknowledged reduction"))
    outcomes = {task.task_id: _outcome(programme, task.arguments["subject"]) for task in programme.tasks}
    receipt = programme.finish(CloudRun(outcomes, {}, []))
    assert receipt["status"] == "failed" and receipt["acknowledged_subjects"] == []


def test_unaccepted_or_different_controller_is_hold(tmp_path):
    binding = tmp_path/"controller.json"
    atomic_record(binding, {"status": "accepted", "late_orphan_gate_accepted": True,
                            "cloud_execution_sha256": "a"*64})
    with pytest.raises(ValueError, match="HOLD"):
        entry.accepted_controller({"controller_binding": {"path": str(binding), "sha256": panel.sha(binding)}})


def test_cross_host_export_retains_original_plan_identity_and_rejects_changed_bytes(tmp_path, monkeypatch):
    export = tmp_path/"exports"/"A01"
    export.mkdir(parents=True)
    atomic_record(tmp_path/"plan.json", {"role": "new_cloud_score_plan"})
    script = tmp_path/"source"/"experiments"/"export_frozen_p1_state.py"
    script.parent.mkdir(parents=True)
    script.write_text("# new source does not re-sign old export")
    geometry = panel.noise_owner.NoiseGeometry(("Cz",), 128., "V", "native", "raw_identity", 1)
    response = panel.noise_owner.FrozenERPResponse(np.ones((1, 1)), np.ones((1, 1)), geometry,
        "a"*64, "fixed", {"fit_scope": "analytic_reference"})
    response_record = json.loads(json.dumps(response.record()))
    np.savez(export/"response.npz", mean=response.mean, coefficient_factor=response.coefficient_factor)
    old_noise = tmp_path/"old_noise.json"
    atomic_record(old_noise, {"state": {"actual_old_primitive": 7}, "state_sha256": "b"*64})
    literal = r"C:\original\old_noise.json"
    original = {"path": literal, "sha256": panel.sha(old_noise)}
    loaders = {"channel": {"path": "original_capsule/channel.py", "sha256": "c"*64}}
    body = {"subject": "A01", "plan_file_sha256": "d"*64, "export_script_sha256": "e"*64,
            "loader_sources": loaders, "response": response_record,
            "response_arrays_sha256": panel.sha(export/"response.npz"), "fit_calls": [], "score_calls": 0,
            "original_input_digest": "f"*64,
            "noises": {arm: {"state": {"actual_old_primitive": 7}, "state_sha256": "b"*64,
                             "original_file": original} for arm in panel.ARMS}}
    atomic_record(export/"export.json", body)
    authority = {key: body[key] for key in ("plan_file_sha256", "export_script_sha256", "loader_sources")}
    authority.update(export_json_sha256=panel.sha(export/"export.json"),
                     response_arrays_sha256=panel.sha(export/"response.npz"))
    config = {"source_root": str(tmp_path/"source"), "original_loader_sources": loaders,
        "participants": {"A01": {"response_record": response_record, "geometry": geometry.record(),
            "noise_files": {arm: original for arm in panel.ARMS}, "input_manifest": {"sha256": "f"*64}}}}
    # This small response/transport fixture isolates _export_state. Actual
    # fixed-plan/held-authority gates are covered by the real read-only probe.
    monkeypatch.setattr(panel, "WINDOWS_EXPORT_PLAN_SHA256", "d"*64)
    monkeypatch.setattr(panel, "original_export_plan", lambda *_: {"original_loader_sources": loaders,
        "source_root": r"C:\original\source", "source_files": {
            r"C:\original\source\experiments\export_frozen_p1_state.py": "e"*64}})
    owner = SimpleNamespace(resolve=lambda path: old_noise if path == literal else path,
                            export_authority=lambda _: authority)
    admitted, _ = panel._export_state(config, "A01", export, input_owner=owner)
    assert admitted["plan_file_sha256"] == "d"*64 != panel.sha(tmp_path/"plan.json")
    assert admitted["fit_calls"] == [] and admitted["score_calls"] == 0
    (export/"export.json").write_bytes((export/"export.json").read_bytes()+b" ")
    with pytest.raises(ValueError, match="sealed Windows authority"):
        panel._export_state(config, "A01", export, input_owner=owner)


def _synthetic_subject(arguments):
    if arguments["subject"] == "A02":
        raise RuntimeError("original actor failure")
    return {"subject": arguments["subject"], "plan_file_sha256": panel.sha(arguments["plan_file"]),
            "arms": list(panel.ARMS)}


def test_actual_executor_ack_failure_never_masks_original_actor_failure(tmp_path, monkeypatch):
    programme = _programme(tmp_path)
    def reject(*_args, **_kwargs):
        raise ValueError("actual ACK contract failure")
    monkeypatch.setattr(entry, "admit_cell", reject)
    # A software fixture uses two lightweight CPU claims. It verifies the real
    # executor callback/primary-failure path, not scientific CPU25 throughput.
    tasks = [replace(task, adapter="tests.research.test_stationary_cloud_programme:_synthetic_subject",
                     resources=PhaseResources("evaluation", 1, 128*2**20, 1))
             for task in programme.tasks[:2]]
    resources = {"host_memory_budget_bytes": 256*2**20, "shared_staging_memory_bytes": 0,
        "source_worker_memory_bytes": 128*2**20, "target_worker_memory_bytes": 128*2**20,
        "source_workers": 2, "target_workers": 2, "gpu_slots": 2,
        "native_threads_per_worker": 1, "cpu_budget": 3, "coordinator_cpu_threads": 1}
    with get_context("spawn").Manager() as manager:
        admission = HostMemoryAdmission(manager, resources)
        initial = admission.record()
        run = CloudExecutor(admission, max_workers=2, diagnostic_directory=tmp_path/"diagnostics",
            identity={"software_scope": "ACK_primary_failure"}, poll_seconds=.1).run(
                tasks, on_finished=programme.on_finished)
        assert admission.record() == initial
    assert run.results["stationary:A01"].failure["stage"] == "artifact_admission"
    assert run.results["stationary:A01"].failure["message"] == "actual ACK contract failure"
    assert run.results["stationary:A02"].failure["message"] == "original actor failure"
    assert all(row.worker["alive"] is False for row in run.results.values())
    receipt = programme.finish(run)
    assert receipt["status"] == "failed" and len(receipt["requested_cells"]) == 16
