"""Actual candidate CLI replay DAG preserves full scientific requests."""
import json
import os
import shutil
from copy import deepcopy
from dataclasses import replace
from multiprocessing import get_context

import pytest

import experiments.run_candidate_ensemble_study as runner
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from research.calibration_selection import CalibrationProgramCandidate
from research.cloud_execution import CloudExecutor
from research.contracts import semantic_sha256
from research.source_execution import (
    HostMemoryAdmission,
    SourceInputAdmission,
    resolve_source_resources,
)
from tests.support.candidate_programs import raw_plan
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram


def replay_plan(tmp_path):
    path = raw_plan(tmp_path)
    plan = json.loads(path.read_bytes())
    raw = CalibrationExpertProgram.from_record(plan["experts"][0])
    program = EnsembleProgram.equal_weights("raw", (raw.expert_id,))
    candidate = CalibrationProgramCandidate("raw-fixed", (raw,), program)
    plan.update(evaluation_design="independent_prefix", seed_pairs=[[20260918, 20260918]],
        budgets=[0, 200], source_seed_catalog=[], calibration_strategies=["fixed", "grouped_temperature"],
        authorization_preprocessing=preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT).record(),
        source_worker_memory_gib=.625, target_worker_memory_gib=.625,
        prefix_replay_graph_memory_gib=.625, prefix_replay_worker_memory_gib=.625,
        prefix_selection={"schema": "n2p3_prefix_selection_plan/1", "output_name": "selected",
            "baseline_name": candidate.name, "minimum_selection_rows": 1, "candidates": [candidate.record()]})
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def selection_saved_native_exit(arguments):
    original = runner._sealed_record

    def save_and_exit(path, record):
        original(path, record)
        if path.parent.name == "selection_fit_receipts" and path.name.endswith(".saved.json"):
            os._exit(73)

    runner._sealed_record = save_and_exit
    return runner.candidate_participant_adapter(arguments)


def _run(tmp_path, plan_path, callback=None, initial_adapter=None):
    plan = json.loads(plan_path.read_bytes())
    resources = resolve_source_resources({**plan, "cpu_budget": 2, "source_workers": 1,
        "target_workers": 1, "source_worker_memory_gib": .625, "target_worker_memory_gib": .625,
        "shared_staging_memory_gib": .0625}, 1, "cpu")
    with get_context("spawn").Manager() as manager:
        inputs = SourceInputAdmission(manager, capacity_bytes=resources["shared_staging_memory_bytes"], load_workers=1)
        admission = HostMemoryAdmission(manager, resources, input_admission=inputs)
        programme = runner.build_candidate_cloud_tasks(plan_path, tmp_path / "run", resources, input_admission=inputs)
        if initial_adapter is not None:
            programme.tasks[0] = replace(programme.tasks[0], adapter=initial_adapter)
        run = CloudExecutor(admission, max_workers=1, diagnostic_directory=tmp_path / "diagnostics",
            identity={"scope": "actual_candidate_replay_dag_software"}, poll_seconds=.05).run(
                programme.tasks, on_finished=(lambda outcome: callback(programme, outcome)) if callback else programme.on_finished)
        report = programme.finalize(run.results, input_admission=inputs, cloud_run=run)
        resources_after, inputs_after = admission.record(), inputs.record()
    return programme, run, report, resources_after, inputs_after


@pytest.fixture(scope="module")
def completed_actual_replay_dag(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("completed-actual-replay")
    path = replay_plan(tmp_path)
    return _run(tmp_path, path)


def test_actual_full_candidate_dag_runs_one_shared_preparation_and_private_parent_tail(completed_actual_replay_dag):
    programme, run, report, resources, inputs = completed_actual_replay_dag
    assert run.status == "completed", {key: value.failure for key, value in run.results.items()}
    assert report["status"] == "completed" and report["decision_denominator_complete"]
    assert len(programme.replay_states) == 1
    assert set(report["metrics"]) == {f"raw@{strategy}@K{budget}" for budget in (0, 200)
        for strategy in ("fixed", "grouped_temperature")} | {"selected@nested_selection@K0", "selected@nested_selection@K200"}
    assert inputs["resident_input_bytes"] == 0 and resources["reserved_cpu_threads"] == 1
    assert not programme.replay_pool.record()["active_readers"]
    assert len(report["fit_execution"]) == 1
    assert all(value.worker["process_tree_retirement"]["active_owned_processes_after_exit"] == 0
        for value in run.results.values())
    for state in programme.replay_states.values():
        descriptor = state["reader_arguments"]["shared_inputs"]
        assert not isinstance(descriptor.get("payload"), bytes)
        assert not isinstance(descriptor.get("capsule"), bytes)


@pytest.mark.parametrize("foreign_anchor", ["participant_directory", "producer_source"])
def test_actual_complete_reducer_owner_cannot_cross_its_original_participant_anchor(
        completed_actual_replay_dag, tmp_path, foreign_anchor):
    from research.prefix_replay_execution import CompletedParticipantReplay

    original, original_run, _, _, _ = completed_actual_replay_dag
    actual_fit = original_run.results[original.task_roster[0].task_id]
    copied_directory = tmp_path / "same-byte-participant"
    with get_context("spawn").Manager() as manager:
        resources = original.resources
        inputs = SourceInputAdmission(manager, capacity_bytes=resources["shared_staging_memory_bytes"], load_workers=1)
        admission = HostMemoryAdmission(manager, resources, input_admission=inputs)
        programme = runner.build_candidate_cloud_tasks(original.original_plan_path, tmp_path / "foreign-replay",
            resources, input_admission=inputs)
        # Replay the existing sealed packages without invoking any fit actor.
        preparation = programme._make_replay_preparation(original.task_roster[0], actual_fit,
            original.current_context, original.normalized, original.source, "new_producer", source_root=runner.ROOT)
        request = deepcopy(preparation.arguments["request"])
        if foreign_anchor == "participant_directory":
            shutil.copytree(request["participant_directory"], copied_directory)
            request["participant_directory"] = str(copied_directory)
        else:
            # This does not alter the frozen compiler or any package byte.
            request["producer_source"]["src/research/source_execution.py"] = "f" * 64
        preparation = replace(preparation, arguments={**preparation.arguments, "request": request})
        programme.tasks = [preparation]
        programme.task_roster = original.task_roster
        programme.replay_states[preparation.task_id]["request"] = request
        accepted_owners = []

        def callback(outcome):
            ready = programme.on_finished(outcome)
            for task in ready:
                if task.adapter.endswith(":candidate_participant_admission_adapter"):
                    accepted_owners.append(CompletedParticipantReplay.from_reduced(
                        task.arguments["replay_request"], task.arguments["reducer_outcomes"]))
            return ready

        run = CloudExecutor(admission, max_workers=1, diagnostic_directory=tmp_path / "diagnostics",
            identity={"scope": "actual_reducer_original_anchor_software"}, poll_seconds=.05).run(
                programme.tasks, on_finished=callback)
        assert len(accepted_owners) == 1
        assert all(not task.adapter.endswith(":candidate_participant_adapter") for task in programme.tasks)
        final = run.results[original.task_roster[0].task_id + ":admission"]
        assert final.status == "failed", final
        assert "original participant/source anchor" in final.failure["message"]
        report = programme.finalize(run.results, input_admission=inputs, cloud_run=run)
        assert report["status"] == "failed" and report["decision_denominator_complete"]
        assert inputs.record()["resident_input_bytes"] == 0


def test_partial_actor_saved_selector_without_original_summary_keeps_other_sealed_outputs(tmp_path):
    path = replay_plan(tmp_path)
    programme, run, report, _, inputs = _run(tmp_path, path,
        initial_adapter="tests.research.test_candidate_replay_dag:selection_saved_native_exit")
    original = run.results[programme.task_roster[0].task_id]
    assert original.worker["exit_code"] == 73
    assert report["status"] == "failed" and report["decision_denominator_complete"]
    participant = report["participants"][0]
    assert participant["status"] == "partial" and participant["predictions"]
    assert all(row["calibration_strategy"] != "nested_selection" for row in participant["predictions"])
    assert len(participant["selection_inflight_programmes"]) == 1
    assert participant["selection_fits"] == []
    assert inputs["resident_input_bytes"] == 0


def test_actual_bad_node_blocks_reducers_and_keeps_requested_pipeline_denominator(tmp_path):
    path = replay_plan(tmp_path)
    changed = False

    def callback(programme, outcome):
        nonlocal changed
        ready = programme.on_finished(outcome)
        mutated = []
        for task in ready:
            if not changed and task.adapter.endswith(":selection_chunk_adapter"):
                requests = deepcopy(task.arguments["node_requests"])
                requests[0]["request"]["prefix_receipt_sha256"] = "f" * 64
                requests[0]["request_sha256"] = semantic_sha256(requests[0]["request"])
                task = replace(task, arguments={**task.arguments, "node_requests": requests})
                changed = True
            mutated.append(task)
        return tuple(mutated)

    programme, run, report, _, inputs = _run(tmp_path, path, callback=callback)
    assert changed and report["status"] == "failed"
    assert report["decision_denominator_complete"]
    assert len(report["requested_pipeline_cells"]) == 6 * len(report["decision_plan"]["entries"])
    assert programme.fit_outcomes[programme.task_roster[0].task_id].status == "completed"
    assert any(value.status == "blocked" and value.worker is None for value in run.results.values())
    assert inputs["resident_input_bytes"] == 0


@pytest.mark.parametrize("phase", ["initial_readers", "registered_reducer"])
def test_actual_progress_error_after_build_keeps_fit_fact_and_unpublished_roster(tmp_path, phase):
    path = replay_plan(tmp_path)
    raised = False

    def callback(programme, outcome):
        original = programme._write_admission_progress

        def fail_after_build():
            nonlocal raised
            if not raised:
                for state in programme.replay_states.values():
                    ready = bool(state["tasks"]) if phase == "initial_readers" else any(
                        task.adapter.endswith(":ordinary_reducer_adapter") or task.adapter.endswith(":selection_reducer_adapter")
                        for task in state["tasks"].values())
                    if ready:
                        raised = True
                        raise OSError("actual replay progress write failed after task construction")
            return original()

        programme._write_admission_progress = fail_after_build
        try:
            return programme.on_finished(outcome)
        finally:
            programme._write_admission_progress = original

    programme, run, report, _, inputs = _run(tmp_path, path, callback=callback)
    assert raised and report["status"] == "failed"
    fitted = programme.fit_outcomes[programme.task_roster[0].task_id]
    assert fitted.status == "completed" and fitted.worker["exit_code"] == 0
    assert report["requested_pipeline_cells"]
    assert inputs["resident_input_bytes"] == 0
    assert all(value.worker is None or value.worker["process_tree_retirement"]["active_owned_processes_after_exit"] == 0
        for value in run.results.values())


def test_actual_completed_preparation_without_adoption_closes_as_failed_full_requested(tmp_path):
    path = replay_plan(tmp_path)

    def callback(programme, outcome):
        class DeferredResidentCapacity:
            def try_resident_input(self, nbytes):
                return None

        programme.replay_pool._admission = DeferredResidentCapacity()
        return programme.on_finished(outcome)

    programme, run, report, _, inputs = _run(tmp_path, path, callback=callback)
    assert report["status"] == "failed" and report["decision_denominator_complete"]
    assert len(report["requested_pipeline_cells"]) == 6 * len(report["decision_plan"]["entries"])
    assert all(not row["predictions"] for row in report["participants"])
    assert programme.fit_outcomes[programme.task_roster[0].task_id].status == "completed"
    preparation = next(value for key, value in run.results.items() if key.endswith(":replay_prepare"))
    assert preparation.worker["exit_code"] == 0 and preparation.worker["alive"] is False
    assert not programme.replay_pool.record()["pending_preparations"]
    assert inputs["resident_input_bytes"] == 0


def test_default_standalone_cloud_caller_resolves_replay_staging_and_full_roster(tmp_path):
    path = replay_plan(tmp_path)
    plan = json.loads(path.read_bytes())
    assert "shared_staging_memory_gib" not in plan
    report = runner.run_study(path, tmp_path / "default-run")
    assert report["status"] == "completed" and report["decision_denominator_complete"]
    assert len(report["fit_execution"]) == 1 and report["replay_execution"]
    assert report["replay_inputs"]["resident_preparations"] == []
