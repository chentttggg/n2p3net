"""Completed prefix models retain their producer; remaining original indexes fit once."""
import io
import json
from pathlib import Path

import pytest
import torch

import experiments.run_candidate_ensemble_study as runner
from research.cloud_execution import CloudResult
from research.contracts import semantic_sha256
from research.prefix_recovery import _reviewed_transition, load_prefix_recovery
from tests.support.prefix_recovery import completed_prefix_producer, write_json
from train.runtime import cpu_thread_budget
from transfer.ensemble_fit import FrozenSourceCheckpoint


@pytest.fixture(scope="module")
def original_prefix_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("completed-prefix-producer")
    return completed_prefix_producer(root)


@pytest.fixture(scope="module")
def original_neural_prefix_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("completed-neural-prefix-producer")
    return completed_prefix_producer(root, neural=True)




def forbid_fit(*args, **kwargs):
    raise AssertionError("completed producer must be replayed with zero new fits")


def test_neural_normalized_catalog_recovers_actual_source_models_without_reopening_records_as_paths(
        original_neural_prefix_run, tmp_path, monkeypatch):
    _, plan, original, actual_resources, source, declaration = original_neural_prefix_run
    before = (original.output / "result.json").read_bytes()
    assert "path" in plan["source_seed_catalog"][0]
    record = original.normalized["source_seed_catalog"][0]
    assert "path" not in record and record["checkpoint_sha256"] == plan["source_seed_catalog"][0]["sha256"]
    path = tmp_path / "new_plan.json"
    write_json(path, {**plan, "prefix_recovery": declaration})
    canonical = runner.SourceSeedCatalog
    catalog_calls = []

    def catalog_owner(entries, **kwargs):
        catalog_calls.append(entries)
        return canonical(entries, **kwargs)

    monkeypatch.setattr(runner, "SourceSeedCatalog", catalog_owner)
    monkeypatch.setattr(runner, "fit_calibration_prefix_programs", forbid_fit)
    monkeypatch.setattr(runner, "fit_calibration_ensemble", forbid_fit)
    import research.calibration_selection as selection
    monkeypatch.setattr(selection, "fit_nested_prefix_selection", forbid_fit)
    with cpu_thread_budget(1):
        consumer = runner.build_candidate_cloud_tasks(path, tmp_path / "consumer", actual_resources)
    assert len(catalog_calls) == 1 and "path" in catalog_calls[0][0]
    assert consumer.normalized["source_seed_catalog"] == original.normalized["source_seed_catalog"]
    assert [task.arguments["record_index"] for task in consumer.tasks] == [1] and len(consumer.task_roster) == 2
    recovered = consumer.admitted["candidate:000"]
    assert recovered["status"] == "completed" and len(recovered["selection_fits"]) == 2
    nn = [row for row in recovered["predictions"] if row["mixture"] == "NN"
          and row["calibration_strategy"] == "fixed" and row["evidence"]["log_probability"] is not None]
    assert nn and {row["K"] for row in nn} == {0, 200}
    assert all(row["score_source_digest"] == semantic_sha256(source) for row in recovered["predictions"])
    checkpoint = consumer.source_snapshots[("BNCI2014-008", 20260918)]
    assert isinstance(checkpoint, FrozenSourceCheckpoint) and checkpoint.digest == record["checkpoint_sha256"]
    # The fresh worker callback uses the same snapshots and full parent gate.
    callback = CloudResult(consumer.task_roster[0].task_id, "completed",
        str(Path(original.task_roster[0].output) / "participant_000/result.json"))
    original.on_finished(callback)
    assert len(catalog_calls) == 1 and original.admitted[callback.task_id]["status"] == "completed"
    assert (original.output / "result.json").read_bytes() == before


@pytest.mark.parametrize("mutation", ["missing", "wrong_seed", "wrong_bytes", "foreign_study"])
def test_neural_parent_rejects_missing_or_changed_snapshot_before_raw_and_model_ingress(
        original_neural_prefix_run, tmp_path, monkeypatch, mutation):
    _, _, original, _, source, _ = original_neural_prefix_run
    task = original.task_roster[0]
    snapshots = dict(original.source_snapshots)
    checkpoint = snapshots[("BNCI2014-008", 20260918)]
    if mutation == "missing":
        snapshots = {}
    elif mutation == "wrong_seed":
        snapshots = {("BNCI2014-008", 20260919): checkpoint}
    elif mutation == "foreign_study":
        snapshots = {("BI2014a", 20260918): checkpoint}
    else:
        payload = torch.load(io.BytesIO(checkpoint.content), weights_only=True)
        payload["training_contract"]["seed"] += 1
        payload["training_contract_digest"] = semantic_sha256(payload["training_contract"])
        altered = tmp_path / "different_source.pt"
        torch.save(payload, altered)
        snapshots[("BNCI2014-008", 20260918)] = FrozenSourceCheckpoint.from_path(altered)
    monkeypatch.setattr(runner, "prepare_bnci_independent_prefixes", forbid_fit)
    outcome = CloudResult(task.task_id, "completed", str(Path(task.output) / "participant_000/result.json"))
    with pytest.raises(ValueError, match="declared canonical source snapshots"):
        runner._admit_candidate_participant(task, outcome, original.normalized, source, source_snapshots=snapshots)


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
