"""Actual spawned candidate outputs retain scientific requests across native exit."""

import hashlib
import json
import os
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import pytest

import experiments.run_candidate_ensemble_study as runner
from research.cloud_execution import CloudExecutor, CloudResult
from research.source_execution import (
    HostMemoryAdmission,
    SourceInputAdmission,
    resolve_source_resources,
)
from tests.support.candidate_programs import raw_plan, study_plan
from transfer.ensemble import EnsembleProgram


def interrupted_candidate_adapter(arguments):
    """Inject native death in the child after a real saved probability receipt."""
    original = runner._prediction_saved

    def save_and_exit(directory, prediction):
        original(directory, prediction)
        if arguments.get("exit_after_available_probability") and prediction["evidence"]["log_probability"] is None:
            return
        os._exit(73)

    runner._prediction_saved = save_and_exit
    return runner.candidate_participant_adapter(arguments)


def resources(plan_path):
    plan = json.loads(plan_path.read_bytes())
    return resolve_source_resources({**plan, "cpu_budget": len(plan["records"]) + 1,
        "source_workers": len(plan["records"]), "target_workers": len(plan["records"]),
        "source_worker_memory_gib": .25, "target_worker_memory_gib": .25}, len(plan["records"]), "cpu")


def test_actual_native_exit_preserves_sealed_model_probability_and_all_requested_decisions(tmp_path):
    plan_path = raw_plan(tmp_path, records=2)
    actual_resources = resources(plan_path)
    with get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=actual_resources["shared_staging_memory_bytes"], load_workers=2)
        admission = HostMemoryAdmission(manager, actual_resources, input_admission=owner)
        programme = runner.build_candidate_cloud_tasks(plan_path, tmp_path / "run", actual_resources, input_admission=owner)
        programme.tasks[0] = replace(programme.tasks[0],
            adapter="tests.research.test_candidate_cloud_execution:interrupted_candidate_adapter")
        executed = CloudExecutor(admission, max_workers=2, diagnostic_directory=tmp_path / "diagnostics",
            identity={"scope": "software_correctness"}, poll_seconds=.05).run(programme.tasks, on_finished=programme.on_finished)
        report = programme.finalize(executed.results, input_admission=owner)
        assert report["status"] == "failed" and report["decision_denominator_complete"]
        assert report["unresolved_record_indexes"] == []
        by_index = {participant["record_index"]: participant for participant in report["participants"]}
        assert by_index[0]["status"] == "partial" and len(by_index[0]["predictions"]) == 1
        assert len(by_index[0]["artifact_receipts"]) == 1
        assert by_index[1]["status"] == "completed" and len(by_index[1]["predictions"]) == 2
        assert sum(len(participant["requested"]) for participant in report["participants"]) == 4
        assert len(report["requested_pipeline_cells"]) == 4
        assert report["metrics"]["raw"]["requested_participant_count"] == 2
        failure = executed.results[programme.tasks[0].task_id]
        assert failure.failure["stage"] == "process_exit" and not failure.supplementary_failures
        artifacts = [record for record in owner.record()["active_leases"] if record["kind"] == "persistent_artifact"]
        assert artifacts and all(record["committed"] for record in artifacts)
        assert any(record["directory"] and record["disk_bytes"] > 0 for record in artifacts)


@pytest.mark.parametrize("violation", ["request_and_prediction", "fit_request", "source", "model", "path"])
def test_scientific_admission_rejects_self_consistent_omissions_and_external_artifacts(tmp_path, violation):
    plan_path = raw_plan(tmp_path)
    programme = runner.build_candidate_cloud_tasks(plan_path, tmp_path / "run", resources(plan_path))
    task = programme.tasks[0]
    runner.run_study(programme.plan_path, task.output, _actor_index=0, _native_threads=task.resources.native_threads)
    directory = Path(task.output) / "participant_000"
    outcome = CloudResult(task.task_id, "completed", str(directory / "result.json"))
    if violation in {"request_and_prediction", "fit_request"}:
        ledger = runner._read_sealed_record(directory / "request_ledger.json")
        if violation == "fit_request":
            ledger["fit_requests"].pop()
        else:
            removed = ledger["pipeline_requests"].pop()
            result = json.loads((directory / "result.json").read_bytes())
            result["predictions"] = [value for value in result["predictions"]
                if value["evidence"]["decision_id"] != removed["requested"]["decision_id"]]
            runner.atomic_write_json(directory / "result.json", result)
            for path in (directory / "prediction_receipts").glob("*.json"):
                if runner._read_sealed_record(path)["evidence"]["decision_id"] == removed["requested"]["decision_id"]:
                    path.unlink()
        runner._sealed_record(directory / "request_ledger.json", ledger)
    elif violation == "source":
        source_path = Path(task.output) / "executed_source.json"
        source = json.loads(source_path.read_bytes())
        source["experiments/run_candidate_ensemble_study.py"] = "0" * 64
        runner.atomic_write_json(source_path, source)
    elif violation == "model":
        model = next(directory.glob("seed*/expert_000.npz"))
        model.write_bytes(model.read_bytes() + b"changed")
    else:
        outcome.value = str(tmp_path / "external" / "result.json")
    with pytest.raises(ValueError):
        programme.on_finished(outcome)
    assert not programme.admitted and task.task_id in programme.admission_failures


def test_unbound_source_consumer_registers_one_seed_requests_before_any_fit(tmp_path):
    path = raw_plan(tmp_path)
    plan = json.loads(path.read_bytes())
    # The actual source expert comes from the existing neural fixture; a missing
    # producer must remain unresolved rather than create a guessed participant.
    (tmp_path / "neural").mkdir()
    fixture = json.loads(study_plan(tmp_path / "neural").read_bytes())
    neural = fixture["experts"][-1]
    neural["recipe"]["initialization"] = "source"
    plan.update(evaluation_design="independent_prefix", seed_pairs=[[20260918, 20260918]], budgets=[0, 200],
        source_seed_catalog=[], experts=[neural], mixtures=[EnsembleProgram.equal_weights("scratch", ("scratch",)).record()],
        baseline_mixture="scratch", calibration_strategies=["fixed"])
    path.write_text(json.dumps(plan), encoding="utf-8")
    programme = runner.build_candidate_cloud_tasks(path, tmp_path / "run", resources(path), dependencies=("source:BNCI",))
    requests = json.loads((programme.output / "consumer_requests.json").read_bytes())["requests"]
    assert len(requests) == 2 and all(value["actual_K"] is None and value["physical_fit_attempts"] is None for value in requests)
    assert programme.normalized is None and programme.tasks[0].depends_on == ("source:BNCI",)
    blocked = CloudResult(programme.tasks[0].task_id, "blocked", failure={"stage": "dependency_failed"})
    programme.on_finished(blocked)
    report = programme.finalize({blocked.task_id: blocked})
    assert report["unresolved_record_indexes"] == [0] and not report["decision_denominator_complete"]
    assert report["participants"] == [] and len(report["requested_pipeline_cells"]) == 2


def test_actual_actor_second_index_keeps_original_roster_authority(tmp_path):
    path = raw_plan(tmp_path, records=2)
    programme = runner.build_candidate_cloud_tasks(path, tmp_path / "run", resources(path))
    task = programme.tasks[1]
    runner.run_study(programme.plan_path, task.output, _actor_index=1, _native_threads=task.resources.native_threads)
    expected = Path(task.output) / "participant_001" / "result.json"
    outcome = CloudResult(task.task_id, "completed", str(expected))
    programme.on_finished(outcome)
    assert programme.admitted[task.task_id]["record_index"] == 1
    ledger = runner._read_sealed_record(expected.parent / "request_ledger.json")
    assert ledger["record_index"] == 1 and ledger["record"] == programme.normalized["records"][1]


def test_independent_one_seed_native_exit_keeps_unscored_actual_budget_panels(tmp_path):
    path = raw_plan(tmp_path)
    plan = json.loads(path.read_bytes())
    import torch

    from data.bnci2014_008_candidate import BNCI2014_008_SOURCE_REFERENCE
    from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract
    from tests.support.checkpoints import write_checkpoint
    from transfer.calibration_candidates import waveform_lda_recipe
    from transfer.ensemble_fit import CalibrationExpertProgram
    from transfer.ensemble_views import SelectionEpochProfile, prepare_bnci_independent_prefixes
    from transfer.recipes import TargetRecipe

    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    profile = SelectionEpochProfile.from_preprocessing(replace(spec, baseline_mode="none"),
        source_sample_rate_hz=256., channel_names=("Fz", "P3", "Pz", "P4", "PO8", "Oz"),
        reference=BNCI2014_008_SOURCE_REFERENCE, reference_channel="Cz")
    views = prepare_bnci_independent_prefixes(plan["records"][0]["path"], source_sha256=plan["records"][0]["sha256"],
        budgets=(0, 200), profiles=(profile,), authorization_preprocessing=spec)
    source_path = write_checkpoint(tmp_path / "source.pt", views.calibration[200].trial_datasets[profile.digest],
                                   source_snapshot_sha256="b" * 64)
    payload = torch.load(source_path, weights_only=True)
    payload["training_contract"]["seed"] = 20260918
    payload["training_contract_digest"] = runner.semantic_sha256(payload["training_contract"])
    torch.save(payload, source_path)
    recipe = TargetRecipe("NN", initialization="source", input_statistics="source", epochs=1, batch_size=128,
                          bn_buffers="calibration", dropout="off")
    neural = (CalibrationExpertProgram("NN", "trial", recipe, profile),
        CalibrationExpertProgram("scratch", "trial", replace(recipe, name="scratch", initialization="scratch",
                                                              input_statistics="target_prefix"), profile),
        CalibrationExpertProgram("waveform", "trial", waveform_lda_recipe(), profile))
    plan["experts"] += [expert.record() for expert in neural]
    plan["mixtures"] = [EnsembleProgram.equal_weights("NN", ("NN",)).record(),
        EnsembleProgram.equal_weights("strict", ("raw", "NN", "scratch", "waveform")).record()]
    plan["baseline_mixture"] = "NN"
    plan.update(evaluation_design="independent_prefix", seed_pairs=[[20260918, 20260918]], budgets=[0, 200],
        source_seed_catalog=[{"outer_target_study": "BNCI2014-008", "source_seed": 20260918, "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "procedure": {"kind": "fixed_recipe", "selection": "not_applicable"}}],
        calibration_strategies=["fixed", "grouped_temperature"], temperature_expert_ids=["NN"],
        authorization_preprocessing=spec.record())
    path.write_text(json.dumps(plan), encoding="utf-8")
    actual_resources = resources(path)
    with get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=actual_resources["shared_staging_memory_bytes"], load_workers=1)
        admission = HostMemoryAdmission(manager, actual_resources, input_admission=owner)
        programme = runner.build_candidate_cloud_tasks(path, tmp_path / "run", actual_resources, input_admission=owner)
        programme.tasks[0] = replace(programme.tasks[0],
            adapter="tests.research.test_candidate_cloud_execution:interrupted_candidate_adapter",
            arguments={**programme.tasks[0].arguments, "exit_after_available_probability": True})
        executed = CloudExecutor(admission, max_workers=1, diagnostic_directory=tmp_path / "diagnostics",
            identity={"scope": "software_correctness"}, poll_seconds=.05).run(programme.tasks, on_finished=programme.on_finished)
        report = programme.finalize(executed.results, input_admission=owner)
    assert report["status"] == "failed" and report["decision_denominator_complete"]
    assert set(report["metrics"]) == {f"{mixture}@{strategy}@K{budget}" for mixture in ("NN", "strict")
                                     for strategy in ("fixed", "grouped_temperature") for budget in (0, 200)}
    assert report["metrics"]["NN@fixed@K200"]["requested_participant_count"] == 1
    assert report["metrics"]["NN@fixed@K200"]["probability_coverage"] == 0
    predictions = report["participants"][0]["predictions"]
    available = [row for row in predictions if row["evidence"]["log_probability"] is not None]
    assert len(available) == 1 and available[0]["saved_replay_exact"]
    assert len(predictions) > len(available)
    assert any(row["evidence"]["failure_kind"] == "incomplete" and
        "partial_decision_after_fixed_Kmax_and_original_epoch_support_purge" in row["evidence"]["failure_reason"]
        for row in predictions)
    assert all(row["K"] == 0 and row["calibration_strategy"] == "fixed" for row in predictions)
    # Prefix replay seals the actual complete strategy roster before the first
    # prediction is admitted.  K0 grouped-temperature has already returned its
    # named unavailable result, even when a native exit interrupts predictions.
    fits = report["participants"][0]["fits"]
    assert len(fits) == 8 and {row["calibration_strategy"] for row in fits} == {"fixed", "grouped_temperature"}
    grouped = [row for row in fits if row["calibration_strategy"] == "grouped_temperature"]
    assert len(grouped) == 4 and all(not row["fit_available"] for row in grouped)
    assert all(row["temperature_calibration"]["status"] == "unavailable"
        and row["temperature_calibration"]["failure"]["error_code"] == "insufficient_calibration_groups" for row in grouped)
    assert {row["actual_K"] for row in report["requested_pipeline_cells"]} == {0, 200}
    assert all(row["status"] == "not_started" for row in report["requested_pipeline_cells"] if row["actual_K"] == 200)
    failure = executed.results[programme.tasks[0].task_id]
    assert failure.worker["exit_code"] == 73 and failure.failure["stage"] == "process_exit"
    assert not failure.supplementary_failures
