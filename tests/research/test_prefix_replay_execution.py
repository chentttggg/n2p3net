"""Real CloudExecutor ingress from a frozen old compiler to current reducers."""
from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from research import prefix_replay_execution as owner
from research.cloud_execution import CloudExecutor, CloudTask, PhaseResources
from research.contracts import semantic_sha256
from research.prefix_replay_inputs import PrefixReplayInputHolder
from research.source_execution import HostMemoryAdmission, SourceInputAdmission

_BUILD_OLD_CASE = '''
import hashlib, importlib.util, json, pathlib, sys
root, original = map(pathlib.Path, sys.argv[1:3])
native = int(sys.argv[3])
spec = importlib.util.spec_from_file_location("transfer.ensemble_views", original)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from dataclasses import replace
from tests.support.independent_prefix import independent_prefix_case
from research.calibration_selection import CalibrationProgramCandidate, fit_nested_prefix_selection
from research.nested_selection_replay import InnerReplayRecorder
from research.prefix_inner_replay import persist_prefix_inner_replay, unavailable_prefix_expert_scores
from research.prefix_selection_replay import replay_prefix_selection
from research.contracts import semantic_sha256
from research.execution import ExpectedSubjectError
from train.runtime import cpu_thread_budget
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import FittedEnsemble, fit_calibration_prefix_programs
case = root / "case"
case.mkdir()
_, _, mat, preprocessing, experts, old_views = independent_prefix_case(case)
profiles = (experts[1].profile,)
with cpu_thread_budget(1):
    views = module.prepare_bnci_independent_prefixes(mat, source_sha256=old_views.ledger["source_sha256"],
        budgets=(0,400), profiles=profiles, authorization_preprocessing=preprocessing)
    raw = replace(experts[0], n_basis=1)
    experts = (raw, experts[1])
    program = EnsembleProgram.equal_weights("real-old-compiler", tuple(x.expert_id for x in experts))
    candidates = (CalibrationProgramCandidate("fixed", experts, program),
        CalibrationProgramCandidate("grouped", experts, program, "grouped_temperature"))
    context = {"requested_Ks":[400], "requested_to_actual_K":{"400":400},
        "test_scope":views.ledger["test_scope"], "requested_decision_ids":[v.record_id for v in views.decisions]}
    selection = fit_nested_prefix_selection(views.calibration[400], candidates, baseline_name="fixed",
        prefix_compiler=views.prefix_compiler, minimum_selection_rows=1, context=context)
    for view in views.decisions:
        selection.score(view)
    participant = root / "participant"
    participant.mkdir()
    selection.save(participant / "selection")
    replay = replay_prefix_selection(participant/"selection", prefix=views.calibration[400],
        prefix_compiler=views.prefix_compiler, future_views=views.decisions, expected_graph_sha256=selection.graph_sha256)
    summary = {"package":"selection", "K":400, "graph_sha256":selection.graph_sha256,
        "selection_sha256":semantic_sha256(selection.record), "producer_count":replay["producer_count"],
        "exact_score_checks":replay["exact_score_checks"], "selected_candidate":selection.record["selected_candidate"],
        "final_refit_scope":selection.record["final_refit_scope"]}
    (root/"original_summary.json").write_text(json.dumps({"record":summary,"record_sha256":semantic_sha256(summary)}),encoding="utf-8")
    with cpu_thread_budget(native):
        recorder = InnerReplayRecorder()
        programs = fit_calibration_prefix_programs(views.calibration[400], experts, program,
            strategies=("fixed","grouped_temperature"), inner_replay_recorder=recorder,
            inner_replay_context=context)
        for view in views.decisions:
            for strategy, value in FittedEnsemble.score_programs(programs,view).items():
                if isinstance(value,ExpectedSubjectError):
                    value = unavailable_prefix_expert_scores(views.calibration[400].record(),
                        [x.record() for x in experts], strategy, value.record(subject=view.participant_key), view)
                recorder.record_prefix_future_score(strategy,view=view,evidence=value)
        persist_prefix_inner_replay(views.calibration[400],programs,participant/"ordinary",recorder=recorder,
            context=context)
    (root/"scopes.json").write_text(json.dumps(views.ledger),encoding="utf-8")
    record = json.loads((participant/"ordinary"/"prefix_inner_replay.json").read_bytes())
    prefix = views.calibration[400]
    inputs = {"schema":"n2p3_prefix_replay_input_request/1", "preparation_task_id":"prepare-real-old",
        "original_source_sha256":old_views.ledger["source_sha256"],
        "producer_source_sha256":hashlib.sha256(original.read_bytes()).hexdigest(),
        "participant_key":prefix.target_contract["participant_key"], "source_relative_path":mat.name,
        "requested_Ks":[0,400], "actual_Ks":[0,400], "profiles":[p.record() for p in profiles],
        "authorization_preprocessing":preprocessing.record(), "observation_response_seconds":1.2,
        "pre_roll_seconds":.2}
    (root/"old_case.json").write_text(json.dumps({"input_request":inputs,"graph_sha256":selection.graph_sha256,
        "prefix_record_sha256":record["record_sha256"],"context":context,"mat":str(mat)}),encoding="utf-8")
'''


@pytest.fixture(scope="module", params=(1, 4))
def old_case(tmp_path_factory, request):
    native = request.param
    root = tmp_path_factory.mktemp("real-prefix-execution")
    repository = Path(__file__).resolve().parents[2]
    frozen = root / "original_ensemble_views.py"
    frozen.write_bytes(subprocess.check_output(["git", "show", "c432d3d:src/transfer/ensemble_views.py"],
        cwd=repository))
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(repository / "src"), str(repository)]),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    subprocess.run([sys.executable, "-B", "-c", _BUILD_OLD_CASE, str(root), str(frozen), str(native)],
        cwd=repository, env=environment, check=True)
    facts = json.loads((root / "old_case.json").read_bytes())
    consumer = {name: owner.artifact_receipt(repository / name)["sha256"] for name in owner._CONSUMER_PATHS}
    producer = {**consumer, "src/transfer/ensemble_views.py": owner.artifact_receipt(frozen)["sha256"]}
    package_files = {}
    saved_seals = {}
    for name in ("selection", "ordinary"):
        package_files[name] = [{**owner.artifact_receipt(path), "path": path.relative_to(root / "participant").as_posix()}
            for path in sorted((root / "participant" / name).rglob("*")) if path.is_file()]
        seal = {"request": {"package": name, "K": 400}, "files": package_files[name]}
        if name == "selection":
            seal["graph_sha256"] = facts["graph_sha256"]
        saved_seals[name] = owner._sealed(root / (name + ".saved.json"), seal)
    ledger = owner._sealed(root / "participant" / "request_ledger.json", {
        "native_threads": native, "execution_source": producer,
        "record": {"sha256": owner.artifact_receipt(facts["mat"])["sha256"]},
        "scopes_sha256": owner.artifact_receipt(root / "scopes.json")["sha256"],
        "participant_key": facts["input_request"]["participant_key"]})
    request = {"schema": owner.SCHEMA, "preparation_task_id": "prepare-real-old",
        "producer_ledger": ledger, "ordinary_native_threads": native,
        "participant_directory": str(root / "participant"), "producer_module": owner.artifact_receipt(frozen),
        "mat": owner.artifact_receipt(facts["mat"]), "scopes": owner.artifact_receipt(root / "scopes.json"),
        "input_request": facts["input_request"], "producer_source": producer, "consumer_source": consumer,
        "selection_packages": [{"package": "selection", "graph_sha256": facts["graph_sha256"],
            "original_request": {"package": "selection", "K": 400}, "source_checkpoints": {},
            "files": package_files["selection"], "saved_seal": saved_seals["selection"],
            "replayed_summary": owner.artifact_receipt(root / "original_summary.json")}],
        "ordinary_packages": [{"package": "ordinary", "prefix_record_sha256": facts["prefix_record_sha256"],
            "original_request": {"package": "ordinary", "K": 400}, "context": facts["context"],
            "source_checkpoints": {}, "files": package_files["ordinary"], "saved_seal": saved_seals["ordinary"]}]}
    return root, request


def _task(root, task_id, adapter, arguments, dependencies=()):
    output = root / task_id
    native = owner.replay_native_threads(arguments["request"], ordinary=adapter.startswith("ordinary_"))
    return CloudTask(task_id, "research.prefix_replay_execution:" + adapter,
        {**arguments, "output": str(output), "output_estimate_bytes": 2**28}, str(output),
        PhaseResources("evaluation", native + 1, 2**30, native, "cpu", interop_threads=1), tuple(dependencies))


@pytest.fixture(scope="module")
def executed(old_case):
    root, request = old_case
    manager = multiprocessing.Manager()
    resources = {"host_memory_budget_bytes": 4 * 2**30, "shared_staging_memory_bytes": 2**30,
        "cpu_budget": 9, "coordinator_cpu_threads": 1, "gpu_slots": 4, "source_workers": 2,
        "target_workers": 2, "native_threads_per_worker": 1,
        "source_worker_memory_bytes": 2**30, "target_worker_memory_bytes": 2**30}
    inputs = SourceInputAdmission(manager, capacity_bytes=2**30, load_workers=2)
    admission = HostMemoryAdmission(manager, resources, input_admission=inputs)
    executor = CloudExecutor(admission, max_workers=2, diagnostic_directory=root / "diagnostics",
        identity={"scope": "controlled_frozen_prefix_execution"}, poll_seconds=.05)
    holder, original_graph, parked_graph = None, None, None
    try:
        prep = _task(root, request["preparation_task_id"], "prefix_replay_prepare_adapter", {"request": request})
        initial = executor.run((prep,))
        outcome = initial.results[prep.task_id]
        assert outcome.status == "completed", outcome.failure
        manifest = owner.read_prefix_preparation(prep, outcome, request)
        holder = PrefixReplayInputHolder.from_prepared(outcome.value["capsule"],
            request_sha=manifest["input_request_sha256"], prep_outcome=outcome, admission=inputs, result_key="capsule")
        assert holder is not None
        common = {"request": request, "shared_inputs": holder.descriptor,
            "input_request_sha256": manifest["input_request_sha256"], "capsule_sha256": manifest["capsule"]["sha256"]}
        job = manifest["selection_jobs"][0]
        original_graph = root / "participant" / "selection" / "prefix_selection.json"
        parked_graph = original_graph.with_name("original_graph_not_available_to_readers.json")
        original_graph.rename(parked_graph)
        tasks = [_task(root, f"chunk-{index}", "selection_chunk_adapter", {**common,
            "package": "selection", "node_requests": job["node_requests"][start:start + 2]})
            for index, start in enumerate(range(0, len(job["node_requests"]), 2))]
        ordinary = _task(root, "ordinary-worker", "ordinary_prefix_replay_adapter", {**common, "package": "ordinary"})
        run = executor.run((*tasks, ordinary))
        assert run.status == "completed", {k: v.failure for k, v in run.results.items()}
        selection_reducer = _task(root, "selection-reducer", "selection_reducer_adapter", {**common,
            "job": job, "node_outcomes": [(task, run.results[task.task_id]) for task in reversed(tasks)]})
        ordinary_reducer = _task(root, "ordinary-reducer", "ordinary_reducer_adapter", {**common,
            "package": "ordinary", "ordinary_outcome": (ordinary, run.results[ordinary.task_id])})
        final = executor.run((selection_reducer, ordinary_reducer))
        assert final.status == "completed", {k: v.failure for k, v in final.results.items()}
        actual = [(task, final.results[task.task_id]) for task in (selection_reducer, ordinary_reducer)]
        completed = owner.CompletedParticipantReplay.from_reduced(request, actual)
        yield request, manifest, completed, actual, admission, inputs, {
            "root": root, "common": common, "job": job, "executor": executor,
            "prep_task": prep, "prep_outcome": outcome}
    finally:
        if parked_graph is not None and parked_graph.exists():
            parked_graph.rename(original_graph)
        if holder is not None:
            holder.close()
        manager.shutdown()


def test_actual_old_compiler_cloud_chunks_and_private_admission_close_the_original_rosters(executed):
    request, manifest, completed, actual, admission, inputs, _ = executed
    completed.assert_request(request)
    selected, ordinary = completed.selection("selection"), completed.ordinary("ordinary")
    assert manifest["compiled_input_source_sha256"] == request["producer_module"]["sha256"]
    assert manifest["compiled_input_source_sha256"] != request["consumer_source"]["src/transfer/ensemble_views.py"]
    assert set(manifest["input_request"]["actual_Ks"]) > {0, 400}
    assert selected["summary"]["zero_new_fits"] and selected["summary"]["actual_K"] == 400
    assert selected["summary"]["exact_score_checks"] > 0 and ordinary["summary"]["exact_score_checks"] > 0
    assert selected["input_consumer"]["future_truth_present"] is False
    assert ordinary["input_consumer"]["future_truth_present"] is False
    assert all(outcome.worker["exit_code"] == 0 and not outcome.worker["alive"] for _, outcome in actual)
    assert ordinary["native_threads"] == request["ordinary_native_threads"]
    assert selected["native_threads"] == 1
    assert ordinary["interop_threads"] == selected["interop_threads"] == 1
    for task, _ in actual:
        expected = 1 if "selection_reducer" in task.adapter else request["ordinary_native_threads"]
        assert task.resources.native_threads == expected and task.resources.cpu_threads == expected + 1
        assert task.resources.interop_threads == 1
    assert admission.record()["reserved_cpu_threads"] == 1
    assert inputs.record()["resident_input_bytes"] > 0


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "exit", "tree", "mapping", "request", "artifact", "native"])
def test_completed_admission_rejects_incomplete_or_unrelated_actual_facts(executed, mutation):
    request, _, _, actual, _, _, _ = executed
    facts = [(task, replace(outcome, worker=dict(outcome.worker))) for task, outcome in actual]
    if mutation == "missing":
        facts.pop()
    elif mutation == "duplicate":
        facts.append(facts[0])
    elif mutation == "exit":
        facts[0][1].worker["exit_code"] = 1
    elif mutation == "tree":
        facts[0][1].worker["process_tree_retirement"] = {"active_owned_processes_after_exit": 1}
    elif mutation == "mapping":
        facts[0] = facts[0][0], {"status": "completed", "accepted": True}
    elif mutation == "request":
        request = {**request, "participant_directory": str(Path(request["participant_directory"]).parent)}
    elif mutation == "native":
        task, outcome = facts[-1]
        facts[-1] = replace(task, resources=replace(task.resources, native_threads=2, cpu_threads=3)), outcome
    else:
        facts[0][1].value = {**facts[0][1].value, "sha256": "f" * 64}
    with pytest.raises((ValueError, TypeError)):
        owner.CompletedParticipantReplay.from_reduced(request, facts)


def test_request_cannot_relabel_the_original_native_programme(old_case):
    _, request = old_case
    altered = {**request, "ordinary_native_threads": 2}
    with pytest.raises(ValueError, match="original source-bound ledger"):
        owner._validate_request(altered)


@pytest.mark.parametrize("mutation", ("native", "interop"))
def test_actual_worker_rejects_a_mismatched_numeric_programme(executed, mutation):
    _, _, _, _, _, _, runtime = executed
    task = _task(runtime["root"], "wrong-ordinary-" + mutation, "ordinary_prefix_replay_adapter",
        {**runtime["common"], "package": "ordinary"})
    if mutation == "native":
        task = replace(task, resources=replace(task.resources, cpu_threads=3, native_threads=2))
        expected = "worker environment"
    else:
        task = replace(task, resources=replace(task.resources,
            cpu_threads=task.resources.native_threads + 2, interop_threads=2))
        expected = "numeric pools"
    result = runtime["executor"].run((task,)).results[task.task_id]
    assert result.status == "failed" and expected in result.failure["message"]
    assert result.worker["exit_code"] == 0 and result.worker["alive"] is False
    assert result.worker["process_tree_retirement"]["active_owned_processes_after_exit"] == 0
    assert not (Path(task.output) / "replay.json").exists()


def test_completed_result_cannot_mutate_another_private_ingress(executed):
    _, _, completed, _, _, _, _ = executed
    changed = completed.selection("selection")
    changed["selection"]["baseline"] = "caller-poison"
    assert completed.selection("selection")["selection"]["baseline"] == "fixed"
    with pytest.raises(TypeError):
        owner.CompletedParticipantReplay()


def test_failed_actual_node_blocks_reducer_and_preserves_shared_inputs_until_retirement(executed):
    request, _, _, _, admission, inputs, runtime = executed
    bad_request = deepcopy(runtime["job"]["node_requests"][0])
    bad_request["request"]["prefix_receipt_sha256"] = "f" * 64
    bad_request["request_sha256"] = semantic_sha256(bad_request["request"])
    bad = _task(runtime["root"], "failed-node", "selection_chunk_adapter", {
        **runtime["common"], "package": "selection", "node_requests": [bad_request]})
    reducer = _task(runtime["root"], "blocked-reducer", "selection_reducer_adapter", {
        **runtime["common"], "job": runtime["job"], "node_outcomes": []}, dependencies=(bad.task_id,))
    before = inputs.record()["resident_input_bytes"]
    result = runtime["executor"].run((bad, reducer))
    failed, blocked = result.results[bad.task_id], result.results[reducer.task_id]
    assert failed.status == "failed" and "externally sealed producer" in failed.failure["message"]
    assert failed.worker["alive"] is False
    assert failed.worker["process_tree_retirement"]["active_owned_processes_after_exit"] == 0
    assert blocked.status == "blocked" and blocked.worker is None
    assert not (Path(reducer.output) / "admission.json").exists()
    assert inputs.record()["resident_input_bytes"] == before > 0
    assert admission.record()["reserved_cpu_threads"] == 1
    with pytest.raises(ValueError):
        owner.CompletedParticipantReplay.from_reduced(request, ((reducer, blocked),))


def test_actual_preparation_accepts_canonical_tuple_equivalent_original_request(executed):
    request, original, _, _, _, _, runtime = executed
    expected = deepcopy(request)
    expected["input_request"]["requested_Ks"] = tuple(expected["input_request"]["requested_Ks"])
    assert semantic_sha256(expected) == semantic_sha256(request)
    admitted = owner.read_prefix_preparation(runtime["prep_task"], runtime["prep_outcome"], expected)
    assert semantic_sha256(admitted) == semantic_sha256(original)


@pytest.mark.parametrize("field", ["request_sha256", "capsule_sha256", "producer_source_sha256"])
def test_completed_reducer_rejects_another_prepared_source_even_with_same_numerical_rows(executed, field):
    request, _, _, actual, _, _, runtime = executed
    task, outcome = actual[0]
    record = owner._read_sealed(outcome.value)
    record["input_consumer"][field] = "f" * 64
    foreign = runtime["root"] / ("foreign-" + field)
    foreign.mkdir()
    receipt = owner._sealed(foreign / "admission.json", record)
    altered_task = replace(task, output=str(foreign), arguments={**task.arguments, "output": str(foreign)})
    altered_outcome = replace(outcome, value=receipt)
    with pytest.raises(ValueError, match="unique original preparation"):
        owner.CompletedParticipantReplay.from_reduced(request, ((altered_task, altered_outcome), actual[1]))
