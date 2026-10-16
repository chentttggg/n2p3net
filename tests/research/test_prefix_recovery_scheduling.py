"""Original replay and new fits share one resource owner without a build barrier."""
import time
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import pytest

import experiments.run_candidate_ensemble_study as runner
from research.cloud_execution import CloudExecutor, CloudTask, PhaseResources
from research.source_execution import (
    HostMemoryAdmission,
    SourceInputAdmission,
    resolve_source_resources,
)
from tests.support.prefix_recovery import (
    completed_prefix_producer,
    write_json,
)
from transfer.ensemble import EnsembleProgram


@pytest.fixture(scope="module")
def original_prefix_run(tmp_path_factory):
    return completed_prefix_producer(tmp_path_factory.mktemp("scheduled-prefix-producer"))


def replay_waits_for_fresh_start(arguments):
    """A liveness witness: this independent numerical task cannot finish first."""
    marker = Path(arguments["fresh_started"])
    waiting = Path(arguments["replay_waiting"])
    waiting.write_text(str(time.time()))
    deadline = time.monotonic() + 30
    while not marker.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("fresh ready work was serialized behind original numerical replay")
        time.sleep(.02)

    def forbidden(*args, **kwargs):
        raise AssertionError("original admission must never fit")

    runner.fit_calibration_prefix_programs = forbidden
    runner.fit_calibration_ensemble = forbidden
    import research.calibration_selection as selection
    selection.fit_nested_prefix_selection = forbidden
    return runner.candidate_participant_admission_adapter(arguments)


def fresh_marks_start(arguments):
    Path(arguments["fresh_started"]).write_text(str(time.time()))
    return runner.candidate_participant_adapter(arguments)


def deliberate_admission_failure(arguments):
    import os

    os._exit(79)


def completed_metadata_actor(arguments):
    """No numerical work: exercise retirement and the physical metadata ingress."""
    return arguments["result_path"]


def recovery_plan(original_prefix_run, tmp_path):
    _, plan, original, _, _, declaration = original_prefix_run
    path = tmp_path / "plan.json"
    native = plan["cpu_threads_per_worker"]
    current = {**plan, "prefix_recovery": declaration, "workers": 2, "cpu_budget": max(4, 2*native+2),
        "source_workers": 2, "target_workers": 2, "source_worker_memory_gib": .625,
        "target_worker_memory_gib": .625, "shared_staging_memory_gib": .0625,
        "prefix_replay_graph_memory_gib": .625, "prefix_replay_worker_memory_gib": .625}
    write_json(path, current)
    resources = resolve_source_resources(current, len(current["records"]), "cpu")
    return path, original, resources


@pytest.mark.parametrize("native_threads", (1, 4))
def test_actual_recovery_dag_keeps_original_readonly_and_starts_fresh_without_replay_barrier(
    original_prefix_run, tmp_path, native_threads,
):
    if native_threads != 1:
        producer_root = tmp_path / "original-native4"
        producer_root.mkdir()
        original_prefix_run = completed_prefix_producer(producer_root, native_threads=native_threads)
    path, original, resources = recovery_plan(original_prefix_run, tmp_path)
    before = (original.output / "result.json").read_bytes()
    observed = []
    with get_context("spawn").Manager() as manager:
        inputs = SourceInputAdmission(manager, capacity_bytes=resources["shared_staging_memory_bytes"], load_workers=1)
        admission = HostMemoryAdmission(manager, resources, input_admission=inputs)
        programme = runner.build_candidate_cloud_tasks(path, tmp_path / "actual-dag", resources, input_admission=inputs)
        initial = tuple(programme.tasks)
        assert len(initial) == 2
        executed = CloudExecutor(admission, max_workers=2, diagnostic_directory=programme.output / "diagnostics",
            identity={"scope": "actual_mixed_recovery_dag"}, poll_seconds=.05).run(
                initial, on_finished=programme.on_finished, on_observation=observed.append)
        report = programme.finalize(executed.results, input_admission=inputs, cloud_run=executed)
        assert inputs.record()["resident_input_bytes"] == 0
    assert report["status"] == "completed", report.get("execution_failures")
    assert report["decision_denominator_complete"] and {row["record_index"] for row in report["participants"]} == {0, 1}
    assert len(report["fit_execution"]) == 2
    assert report["fit_execution"]["candidate:000"]["worker"]["exit_code"] == 0
    assert report["participant_execution"]["0"]["role"] == "original_producer_independent_worker_replay"
    assert any(all(task.task_id in snapshot["running"] for task in initial) for snapshot in observed)
    assert (original.output / "result.json").read_bytes() == before
    assert all(result.worker is not None and result.worker["exit_code"] == 0 for result in executed.results.values())
    if native_threads == 4:
        roles = {task.adapter.rsplit(":", 1)[-1]: task.resources for task in programme.tasks
                 if task.resources.phase == "evaluation"}
        assert roles["ordinary_prefix_replay_adapter"].native_threads == 4
        assert roles["ordinary_prefix_replay_adapter"].cpu_threads == 5
        assert roles["selection_chunk_adapter"].native_threads == 1
        assert roles["selection_chunk_adapter"].cpu_threads == 2
        assert roles["candidate_participant_admission_adapter"].native_threads == 4
        assert roles["candidate_participant_admission_adapter"].interop_threads == 1
        assert roles["candidate_participant_admission_adapter"].cpu_threads == 5


def test_real_shared_owner_starts_fresh_fit_before_original_replay_completes_and_admits_both(
        original_prefix_run, tmp_path, monkeypatch):
    path, original, resources = recovery_plan(original_prefix_run, tmp_path)
    original_bytes = (original.output / "result.json").read_bytes()

    def main_must_not_replay(*args, **kwargs):
        raise AssertionError("main callback performed the heavy numerical checker")

    monkeypatch.setattr(runner, "_admit_candidate_participant", main_must_not_replay)
    with get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=resources["shared_staging_memory_bytes"], load_workers=2)
        admission = HostMemoryAdmission(manager, resources, input_admission=owner)
        # Keep the explicit legacy worker admission compatibility path beside
        # the default shared replay DAG exercised above.
        programme = runner.build_candidate_cloud_tasks(path, tmp_path / "run", resources)
        assert not programme.admitted and programme.execution_workers == resources["max_cache_group_processes"]
        assert programme.execution_workers >= 2
        assert len(programme.tasks) == 2 and not list(programme.output.glob("recovered_participant_*.json"))
        marker, waiting = tmp_path / "fresh.started", tmp_path / "replay.waiting"
        tasks = []
        for task in programme.tasks:
            adapter = ("replay_waits_for_fresh_start" if task.resources.phase == "evaluation" else "fresh_marks_start")
            tasks.append(replace(task, adapter=f"tests.research.test_prefix_recovery_scheduling:{adapter}",
                arguments={**task.arguments, "fresh_started": str(marker), "replay_waiting": str(waiting)}))
        observed = []
        executed = CloudExecutor(admission, max_workers=programme.execution_workers,
            diagnostic_directory=programme.output / "diagnostics", identity={"scope": "software_correctness"},
            poll_seconds=.02).run(tasks, on_finished=programme.on_finished, on_observation=observed.append)
        report = programme.finalize(executed.results, input_admission=owner)
        final_resources = admission.record()
    assert marker.exists() and waiting.exists() and executed.status == "completed"
    assert len(executed.results) == 3 and report["status"] == "completed" and report["decision_denominator_complete"]
    assert {row["record_index"] for row in report["participants"]} == {0, 1}
    assert len(report["fit_execution"]) == 2 and len(report["admission_execution"]) == 2
    assert report["participant_execution"]["0"]["role"] == "original_producer_independent_worker_replay"
    assert any(len(row["running"]) == 2 for row in observed)
    assert final_resources["reserved_cpu_threads"] == resources["coordinator_cpu_threads"]
    assert final_resources["evaluation_groups"] == final_resources["calibration_groups"] == 0
    assert (original.output / "result.json").read_bytes() == original_bytes
    for outcome in executed.results.values():
        assert outcome.worker["exit_code"] == 0


def test_fit_os_zero_then_actual_evaluation_failure_keeps_full_failed_denominator(
        original_prefix_run, tmp_path):
    path, _, resources = recovery_plan(original_prefix_run, tmp_path)
    with get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=resources["shared_staging_memory_bytes"], load_workers=2)
        admission = HostMemoryAdmission(manager, resources, input_admission=owner)
        programme = runner.build_candidate_cloud_tasks(path, tmp_path / "run", resources)

        def callback(outcome):
            additions = programme.on_finished(outcome)
            return tuple(replace(task, adapter="tests.research.test_prefix_recovery_scheduling:deliberate_admission_failure")
                         for task in additions)

        executed = CloudExecutor(admission, max_workers=programme.execution_workers,
            diagnostic_directory=programme.output / "diagnostics", identity={"scope": "software_correctness"},
            poll_seconds=.02).run(programme.tasks, on_finished=callback)
        report = programme.finalize(executed.results, input_admission=owner)
    fitted = report["fit_execution"]["candidate:001"]
    evaluated = report["admission_execution"]["candidate:001:admission"]
    assert fitted["status"] == "completed" and fitted["worker"]["exit_code"] == 0
    assert evaluated["status"] == "failed" and evaluated["worker"]["exit_code"] != 0
    assert report["status"] == "failed" and not report["decision_denominator_complete"]
    assert len(report["requested_records"]) == 2 and report["unresolved_record_indexes"] == [1]
    missing = [row for row in report["requested_pipeline_cells"] if row["record_index"] == 1]
    assert missing and all(row["status"] == "failed" and row["failure"] is not None for row in missing)


def test_admission_ipc_digest_rejects_resealed_receipt_tampering(original_prefix_run, tmp_path):
    path, _, resources = recovery_plan(original_prefix_run, tmp_path)
    programme = runner.build_candidate_cloud_tasks(path, tmp_path / "run", resources)
    task = next(task for task in programme.tasks if task.resources.phase == "evaluation")
    with get_context("spawn").Manager() as manager:
        admission = HostMemoryAdmission(manager, resources)
        executed = CloudExecutor(admission, max_workers=1, diagnostic_directory=programme.output / "diagnostics",
            identity={"scope": "software_correctness"}, poll_seconds=.02).run((task,))
    outcome = executed.results[task.task_id]
    assert outcome.status == "completed", outcome.failure
    target = Path(task.output) / "admission.json"
    record = runner._read_sealed_record(target)
    record["participant"]["requested"] = []
    runner._sealed_record(target, record)
    with pytest.raises(ValueError, match="return digest"):
        programme.on_finished(outcome)
    assert not programme.admitted


def test_progress_write_failure_preserves_fit_and_eight_person_failed_report(tmp_path, monkeypatch):
    """Actual callback failure before ACK must never publish an unscheduled evaluator."""
    plan = {"evidence_scope": "software_correctness", "evaluation_design": "independent_prefix",
        "records": [{"sha256": f"{index:064x}"} for index in range(8)], "budgets": [0, 200, 800, 2000],
        "seed_pairs": [[20260918, 20260918]], "calibration_strategies": ["fixed", "grouped_temperature"],
        "mixtures": [EnsembleProgram.equal_weights("raw", ("raw",)).record()],
        "prefix_selection": {"output_name": "selected"}}
    output = tmp_path / "run"
    output.mkdir()
    source = runner._executing_source()
    resources = {"cpu_budget": 2, "coordinator_cpu_threads": 1, "host_memory_budget_bytes": 2**30,
        "shared_staging_memory_bytes": 0, "source_workers": 1, "target_workers": 1, "gpu_slots": 1,
        "native_threads_per_worker": 1, "source_worker_memory_bytes": 2**28, "target_worker_memory_bytes": 2**28}
    tasks = [CloudTask(f"candidate:{index:03d}",
        "tests.research.test_prefix_recovery_scheduling:completed_metadata_actor",
        {"record_index": index, "result_path": str(output / f"actor_{index}" / f"participant_{index:03d}" / "result.json")},
        str(output / f"actor_{index}"), PhaseResources("calibration", 1, 2**28, 1, "cpu"),
        () if index == 0 else ("candidate:000",)) for index in range(8)]
    first = tasks[0]
    group = Path(first.output)
    directory = group / "participant_000"
    directory.mkdir(parents=True)
    write_json(group / "plan.json", plan)
    write_json(group / "executed_source.json", source)
    scopes = write_json(directory / "scopes.json", {"source_sha256": plan["records"][0]["sha256"]})
    write_json(directory / "result.json", {"status": "completed", "scope": "metadata_only"})
    runner._sealed_record(directory / "request_ledger.json", {"execution_source": source, "record_index": 0,
        "record": plan["records"][0], "plan_digest": runner.semantic_sha256(runner._plan_authority(plan)),
        "native_threads": 1, "scopes_sha256": scopes["sha256"]})
    # Reuse the production callback/finalizer without constructing scientific fits.
    programme = runner.CandidateCloudProgramme.__new__(runner.CandidateCloudProgramme)
    programme.output, programme.resources = output, resources
    programme.plan, programme.normalized, programme.source = plan, plan, source
    programme.plan_path = group / "plan.json"
    programme.task_roster, programme.tasks = tuple(tasks), list(tasks)
    programme.recovery_binding = {"scope": "software_correctness", "reused_record_indexes": []}
    programme.input_admission = None
    programme.current_context = programme._write_admission_context("current", plan, source)
    programme.admitted, programme.admission_failures = {}, {}
    programme.evaluation_tasks, programme.fit_outcomes = {}, {}
    programme.replay_enabled, programme.replay_pool = False, None
    programme.replay_states, programme.preparation_failures = {}, {}
    writer = runner._write_candidate_json
    failed_writes = []

    def fail_first_progress(path, record, owner=None):
        if Path(path) == output / "progress.json" and not failed_writes:
            failed_writes.append(path)
            raise OSError("injected progress publication failure")
        return writer(path, record, owner)

    monkeypatch.setattr(runner, "_write_candidate_json", fail_first_progress)
    with get_context("spawn").Manager() as manager:
        admission = HostMemoryAdmission(manager, resources)
        executed = CloudExecutor(admission, max_workers=1, diagnostic_directory=output / "diagnostics",
            identity={"scope": "software_correctness"}, poll_seconds=.02).run(tasks, on_finished=programme.on_finished)
        report = programme.finalize(executed.results)
    assert len(failed_writes) == 1 and executed.status == report["status"] == "failed"
    fitted = report["fit_execution"][first.task_id]
    failure = executed.results[first.task_id].failure
    assert fitted["status"] == "completed" and fitted["worker"]["exit_code"] == 0 and fitted["failure"] is None
    assert failure == {"type": "OSError", "stage": "artifact_admission", "message": "injected progress publication failure"}
    assert not programme.evaluation_tasks and not report["admission_execution"]
    assert len(executed.results) == len(report["requested_records"]) == 8
    assert report["unresolved_record_indexes"] == list(range(8)) and not report["decision_denominator_complete"]
    requests = report["requested_pipeline_cells"]
    assert len(requests) == 8 * 4 * 3 and {row["record_index"] for row in requests} == set(range(8))
    assert all(row["status"] == "failed" and row["failure"] == failure for row in requests if row["record_index"] == 0)
    assert all(row["status"] == "blocked" for row in requests if row["record_index"] != 0)
    assert report["execution_failures"][0]["failure"] == failure


def test_initial_old_and_fresh_ready_tasks_share_the_declared_phase_owner(original_prefix_run, tmp_path):
    path, _, resources = recovery_plan(original_prefix_run, tmp_path)
    programme = runner.build_candidate_cloud_tasks(path, tmp_path / "run", resources)
    replay = next(task for task in programme.tasks if task.resources.phase == "evaluation")
    fresh = next(task for task in programme.tasks if task.resources.phase == "calibration")
    assert replay.resources.device == "cpu" and replay.resources.native_threads == replay.resources.interop_threads == 1
    assert replay.resources.cpu_threads == 2
    assert replay.arguments["request"]["producer_task"]["arguments"]["record_index"] == 0
    assert fresh.arguments["record_index"] == 1 and not fresh.depends_on
    assert not programme.admitted and not (programme.output / "result.json").exists()


def test_existing_owner_admits_two_cpu_replays_and_four_gpu_capable_fits_within_cpu25():
    """Transaction feasibility; this does not allocate GPU or claim utilization."""
    resources = {"cpu_budget": 25, "coordinator_cpu_threads": 1, "host_memory_budget_bytes": 48 * 2**30,
        "shared_staging_memory_bytes": 8 * 2**30, "source_workers": 4, "target_workers": 4,
        "gpu_slots": 4, "gpu_backed_workers": True, "native_threads_per_worker": 4,
        "source_worker_memory_bytes": 4 * 2**30, "target_worker_memory_bytes": 4 * 2**30}
    with get_context("spawn").Manager() as manager:
        admission = HostMemoryAdmission(manager, resources)
        claims = [admission.reserve_phase("calibration", cpu_threads=5, memory_bytes=4 * 2**30,
            gpu_workers=0, task_id=f"fresh:{index}", gpu_capable=True) for index in range(4)]
        claims += [admission.reserve_phase("evaluation", cpu_threads=2, memory_bytes=4 * 2**30,
            gpu_workers=0, task_id=f"replay:{index}", gpu_capable=False) for index in range(2)]
        assert all(claim is not None for claim in claims)
        actual = admission.record()
        assert actual["reserved_cpu_threads"] == 25 and actual["gpu_capable_calibration_actors"] == 4
        assert actual["evaluation_groups"] == 2 and actual["calibration_groups"] == 4
        for claim in claims:
            admission.release_phase(claim)
        assert admission.record()["reserved_cpu_threads"] == 1
