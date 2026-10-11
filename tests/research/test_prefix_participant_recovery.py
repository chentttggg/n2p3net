"""Completed prefix models retain their producer; remaining original indexes fit once."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

import experiments.run_candidate_ensemble_study as runner
from research.calibration_selection import CalibrationProgramCandidate
from research.cloud_execution import CloudResult
from research.contracts import semantic_sha256
from research.prefix_recovery import _reviewed_transition, load_prefix_recovery
from tests.research.test_candidate_cloud_execution import resources
from tests.support.independent_prefix import independent_prefix_case
from train.runtime import cpu_thread_budget
from transfer.ensemble import EnsembleProgram


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def freeze_runtime(directory):
    """A real physical source freeze also works in an extracted, git-free test source."""
    directory.mkdir()
    paths = [*(runner.ROOT / "src").rglob("*.py"), *(runner.ROOT / "experiments").glob("*.py"),
             runner.ROOT / "pyproject.toml", runner.ROOT / "uv.lock"]
    files = {path.relative_to(runner.ROOT).as_posix(): path.read_bytes() for path in paths}
    archive = directory / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))
    return write_json(directory / "source.manifest.json", {"schema": "n2p3_source_freeze/1",
        "archive": archive.name, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_commit": "0" * 40, "working_tree_snapshot": True, "source_commit_role": "base_commit_only",
        "byte_size": archive.stat().st_size, "member_count": len(files),
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}})


@pytest.fixture(scope="module")
def original_prefix_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("completed-prefix-producer")
    with cpu_thread_budget(1):
        _, fields, path, preprocessing, experts, views = independent_prefix_case(root)
    second = root / "A02.mat"
    fields = {**fields, "X": np.random.default_rng(40).normal(0, 4, fields["X"].shape).astype(np.float32)}
    savemat(second, {"data": fields})
    raw = experts[0]
    ensemble = EnsembleProgram.equal_weights("raw", (raw.expert_id,))
    candidate = CalibrationProgramCandidate("raw_fixed", (raw,), ensemble)
    source_manifest = freeze_runtime(root / "source")
    plan = {"schema": runner.SCHEMA, "evidence_scope": "software_correctness",
        "evaluation_design": "independent_prefix", "seed_pairs": [[20260918, 20260918]],
        "records": [{"path": str(path), "sha256": views.ledger["source_sha256"]},
                    {"path": str(second), "sha256": hashlib.sha256(second.read_bytes()).hexdigest()}],
        "budgets": [0, 200], "experts": [raw.record()], "mixtures": [ensemble.record()],
        "baseline_mixture": "raw", "authorization_preprocessing": preprocessing.record(),
        "observation_response_seconds": 308 / 256, "pre_roll_seconds": .2, "device": "cpu",
        "workers": 1, "cpu_threads_per_worker": 1, "source_checkpoints": {}, "source_seed_catalog": [],
        "source_snapshot_manifest": source_manifest["path"], "calibration_strategies": ["fixed", "grouped_temperature"],
        "prefix_selection": {"schema": "n2p3_prefix_selection_plan/1", "output_name": "selected",
            "baseline_name": candidate.name, "minimum_selection_rows": 3, "candidates": [candidate.record()]}}
    plan_receipt = write_json(root / "input_plan.json", plan)
    actual_resources = resources(Path(plan_receipt["path"]))
    programme = runner.build_candidate_cloud_tasks(plan_receipt["path"], root / "failed_parent", actual_resources)
    task = programme.tasks[0]
    with cpu_thread_budget(1):
        runner.run_study(programme.plan_path, task.output, _actor_index=0, _native_threads=1)
        completed = CloudResult(task.task_id, "completed", str(Path(task.output) / "participant_000/result.json"))
        programme.on_finished(completed)
    failed = CloudResult(programme.tasks[1].task_id, "failed", failure={"type": "SoftwareFixtureInterruption"})
    programme.finalize({completed.task_id: completed, failed.task_id: failed})
    source = json.loads((programme.output / "executed_source.json").read_bytes())
    request = write_json(root / "queue_request.json", {"schema": "n2p3_dataset_queue_request/2", "dataset_id": "software",
        "argv": ["python", str(runner.ROOT / "experiments/run_candidate_ensemble_study.py"),
                 "--plan", plan_receipt["path"], "--output", str(programme.output)],
        "cwd": str(runner.ROOT), "output_dir": str(programme.output), "plan": plan_receipt, "source": source_manifest})
    child = {"pid": 1, "create_time": 1., "process_group": 1}
    actual_exit = write_json(root / "queue_exit.json", {"schema": "n2p3_dataset_queue_exit/2", "dataset_id": "software",
        "request_sha256": request["sha256"], "child": child, "exit_code": 1, "delegated_process_exit_code": 1,
        "science_execution_started": True, "resource_tree": {"active_owned_processes_after_exit": 0}})
    terminal = write_json(root / "queue_terminal.json", {"schema": "n2p3_dataset_queue_terminal/2", "dataset_id": "software",
        "request_sha256": request["sha256"], "child": child, "exit_code": 1, "process_status": "failed",
        "external_exit_receipt_sha256": actual_exit["sha256"], "missing_result_files": [],
        "artifact_facts": [{"path": name, "sha256": hashlib.sha256((programme.output / name).read_bytes()).hexdigest()}
                           for name in ("plan.json", "result.json")]})
    transition = write_json(root / "transition.json", {"schema": "n2p3_prefix_reviewed_source_transition/1",
        "review_status": "accepted", "producer_source_digest": semantic_sha256(source),
        "consumer_source_digest": semantic_sha256(source), "changes": []})
    group = Path(task.output)
    participant = group / "participant_000"
    declaration = {"schema": "n2p3_prefix_participant_recovery/1", "original_run": str(programme.output),
        "original_request": request, "original_exit": actual_exit, "original_terminal": terminal,
        "original_executed_source_sha256": hashlib.sha256((programme.output / "executed_source.json").read_bytes()).hexdigest(),
        "original_source_archive_sha256": json.loads(Path(source_manifest["path"]).read_bytes())["archive_sha256"],
        "reviewed_transition": transition, "participants": [{"record_index": 0,
            "request_ledger_sha256": hashlib.sha256((participant / "request_ledger.json").read_bytes()).hexdigest(),
            "result_sha256": hashlib.sha256((participant / "result.json").read_bytes()).hexdigest(),
            "actor_plan_sha256": hashlib.sha256((group / "plan.json").read_bytes()).hexdigest(),
            "actor_source_sha256": hashlib.sha256((group / "executed_source.json").read_bytes()).hexdigest()}]}
    return root, plan, programme, actual_resources, source, declaration


def forbid_fit(*args, **kwargs):
    raise AssertionError("completed producer must be replayed with zero new fits")


def test_recovery_replays_original_prefix_and_selection_with_zero_fit_and_keeps_missing_denominator(
        original_prefix_run, tmp_path, monkeypatch):
    _, plan, original, actual_resources, _, declaration = original_prefix_run
    before = (original.output / "result.json").read_bytes()
    path = tmp_path / "new_plan.json"
    write_json(path, {**plan, "prefix_recovery": declaration})
    monkeypatch.setattr(runner, "fit_calibration_prefix_programs", forbid_fit)
    monkeypatch.setattr(runner, "fit_calibration_ensemble", forbid_fit)
    import research.calibration_selection as selection
    monkeypatch.setattr(selection, "fit_nested_prefix_selection", forbid_fit)
    with cpu_thread_budget(1):
        consumer = runner.build_candidate_cloud_tasks(path, tmp_path / "consumer", actual_resources)
    assert [task.arguments["record_index"] for task in consumer.tasks] == [1]
    assert len(consumer.task_roster) == 2 and set(consumer.admitted) == {"candidate:000"}
    recovered = consumer.admitted["candidate:000"]
    assert recovered["status"] == "completed" and len(recovered["selection_fits"]) == 2
    assert all(row["score_source_digest"] == recovered["execution_binding"]["producer_source_digest"]
               for row in recovered["predictions"])
    failure = CloudResult(consumer.tasks[0].task_id, "failed", failure={"type": "SoftwareFixtureInterruption"})
    report = consumer.finalize({failure.task_id: failure})
    assert report["status"] == "failed" and not report["decision_denominator_complete"]
    assert len(report["requested_records"]) == 2 and report["unresolved_record_indexes"] == [1]
    assert {row["record_index"] for row in report["requested_pipeline_cells"]} == {0, 1}
    assert report["prefix_recovery"]["original_failed_status"] == "failed"
    assert (original.output / "result.json").read_bytes() == before


def test_only_remaining_original_index_fits_and_full_programme_finalizes(original_prefix_run, tmp_path):
    _, plan, original, actual_resources, _, declaration = original_prefix_run
    before = (original.output / "result.json").read_bytes()
    path = tmp_path / "new_plan.json"
    write_json(path, {**plan, "prefix_recovery": declaration})
    with cpu_thread_budget(1):
        consumer = runner.build_candidate_cloud_tasks(path, tmp_path / "consumer", actual_resources)
        task = consumer.tasks[0]
        assert task.arguments["record_index"] == 1
        runner.run_study(consumer.plan_path, task.output, _actor_index=1, _native_threads=1)
        outcome = CloudResult(task.task_id, "completed", str(Path(task.output) / "participant_001/result.json"))
        consumer.on_finished(outcome)
        report = consumer.finalize({task.task_id: outcome})
    assert report["status"] == "completed" and report["decision_denominator_complete"]
    assert {row["record_index"] for row in report["participants"]} == {0, 1}
    assert report["requested_budgets"] == [0, 200] and report["seed_pairs"] == [[20260918, 20260918]]
    assert len(report["metrics"]) == 6 and report["prefix_selection"] == plan["prefix_selection"]
    assert report["participant_execution"]["0"]["role"] == "original_producer_zero_fit_parent_replay"
    assert report["participant_execution"]["1"]["role"] == "new_producer"
    assert not (consumer.output / "participant_groups/participant_000").exists()
    assert (original.output / "result.json").read_bytes() == before


@pytest.mark.parametrize("mutation", ["records", "budget", "seed", "candidate", "future_policy", "duplicate", "ledger", "producer_source"])
def test_recovery_rejects_changed_programme_or_original_authority_before_model_ingress(
        original_prefix_run, tmp_path, mutation):
    _, _, original, _, source, declaration = original_prefix_run
    plan = json.loads(json.dumps({**original.normalized, "prefix_recovery": declaration}))
    recovery = plan["prefix_recovery"]
    if mutation == "records":
        plan["records"] = plan["records"][:1]
    elif mutation == "budget":
        plan["budgets"] = [200]
    elif mutation == "seed":
        plan["seed_pairs"][0][1] += 1
    elif mutation == "candidate":
        plan["prefix_selection"]["candidates"][0]["name"] = "other"
    elif mutation == "future_policy":
        plan["observation_response_seconds"] += .01
    elif mutation == "duplicate":
        recovery["participants"] *= 2
    elif mutation == "ledger":
        recovery["participants"][0]["request_ledger_sha256"] = "a" * 64
    else:
        recovery["original_executed_source_sha256"] = "a" * 64
    with pytest.raises(ValueError):
        load_prefix_recovery(plan, source, output=tmp_path / "consumer", consumer_root=runner.ROOT)


@pytest.mark.parametrize("mutation", ["success_exit", "active_tree", "child", "external_exit"])
def test_recovery_requires_actual_failed_exit_and_retired_original_tree(original_prefix_run, tmp_path, mutation):
    _, _, original, _, source, declaration = original_prefix_run
    plan = json.loads(json.dumps({**original.normalized, "prefix_recovery": declaration}))
    receipt = plan["prefix_recovery"]["original_exit"]
    exit_record = json.loads(Path(receipt["path"]).read_bytes())
    if mutation == "success_exit":
        exit_record["exit_code"] = 0
    elif mutation == "active_tree":
        exit_record["resource_tree"]["active_owned_processes_after_exit"] = 1
    elif mutation == "child":
        exit_record["child"]["create_time"] += 1
    else:
        plan["prefix_recovery"]["original_terminal"]["sha256"] = "a" * 64
    if mutation != "external_exit":
        plan["prefix_recovery"]["original_exit"] = write_json(tmp_path / "changed_exit.json", exit_record)
    with pytest.raises(ValueError):
        load_prefix_recovery(plan, source, output=tmp_path / "consumer", consumer_root=runner.ROOT)


def test_reviewed_transition_rejects_scientific_code_change_even_when_listed(tmp_path):
    name = "src/transfer/continuous_decision.py"
    producer, consumer = {name: "a" * 64}, {name: "b" * 64}
    receipt = write_json(tmp_path / "transition.json", {"schema": "n2p3_prefix_reviewed_source_transition/1",
        "review_status": "accepted", "producer_source_digest": semantic_sha256(producer),
        "consumer_source_digest": semantic_sha256(consumer), "changes": [{"path": name,
            "producer_sha256": producer[name], "consumer_sha256": consumer[name], "role": None}]})
    with pytest.raises(ValueError, match="reviewed admission/resource"):
        _reviewed_transition(receipt, producer, consumer)


def test_recovery_output_boundary_precedes_any_original_write(original_prefix_run, monkeypatch):
    root, plan, original, actual_resources, _, declaration = original_prefix_run
    path = root / "boundary_plan.json"
    write_json(path, {**plan, "prefix_recovery": declaration})
    output = original.output / "forbidden_new_consumer"
    with pytest.raises(ValueError, match="read-only"):
        runner.build_candidate_cloud_tasks(path, output, actual_resources)
    assert not output.exists()


@pytest.mark.parametrize("boundary", ["original_run", "producer_checkout"])
def test_remaining_actor_rejects_original_output_before_preparation(tmp_path, monkeypatch, boundary):
    original_run = tmp_path / "original_run"
    producer_checkout = tmp_path / "producer_checkout"
    original_run.mkdir()
    producer_checkout.mkdir()
    request = write_json(tmp_path / "request.json", {"cwd": str(producer_checkout)})
    plan = tmp_path / "recovery_plan.json"
    write_json(plan, {"prefix_recovery": {"original_run": str(original_run),
        "original_request": request, "participants": [{"record_index": 0}]}})
    output = {"original_run": original_run, "producer_checkout": producer_checkout}[boundary] / "remaining_actor"

    def forbidden_preparation(*args, **kwargs):
        raise AssertionError("entered preparation before original-output admission")

    monkeypatch.setattr(runner, "_prepare_candidate_plan", forbidden_preparation)
    with pytest.raises(ValueError, match="read-only"):
        runner.run_study(plan, output, _actor_index=1, execution="cloud")
    assert not output.exists()


@pytest.mark.parametrize("mode", ["inline", "reused_actor"])
def test_recovery_cannot_refit_completed_actor_or_fall_back_to_inline(original_prefix_run, tmp_path, mode):
    _, plan, _, _, _, declaration = original_prefix_run
    path = tmp_path / "plan.json"
    write_json(path, {**plan, "prefix_recovery": declaration})
    with pytest.raises(ValueError, match="only remaining original participants"):
        runner.run_study(path, tmp_path / "output", execution="inline" if mode == "inline" else "cloud",
                         _actor_index=None if mode == "inline" else 0)
    assert not (tmp_path / "output").exists()
