"""Temperature-plus-epsilon uses whole-algorithm OOF and distinct producer roles."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from research.calibration_selection import CalibrationProgramCandidate
from research.contracts import semantic_sha256
from research.nested_selection_replay import InnerReplayRecorder
from research.prefix_inner_replay import persist_prefix_inner_replay, replay_prefix_inner_replay
from tests.research.test_pool_probability_prefix import case as pool_case
from train.runtime import cpu_thread_budget
from transfer.ensemble_fit import FittedEnsemble, fit_calibration_prefix_programs
from transfer.pool_calibration import TEMPERATURE_POOL_STRATEGY


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    return pool_case.__wrapped__(tmp_path_factory)


def test_temperature_subset_full_pool_scope_and_same_run_hierarchical_replay(case, tmp_path, monkeypatch):
    from transfer import ensemble_fit as fit_owner

    *_, specs, program, views = case
    prefix = views.calibration[800]
    future = next(view for view in views.decisions if view.samples is not None)
    recorder = InnerReplayRecorder()
    context = {"test_scope": views.ledger["test_scope"], "requested_decision_ids": [future.record_id]}
    with cpu_thread_budget(1):
        programs = fit_calibration_prefix_programs(prefix, specs, program,
            strategies=("fixed", "grouped_temperature", TEMPERATURE_POOL_STRATEGY),
            temperature_expert_ids=(specs[0].expert_id,), inner_replay_recorder=recorder,
            inner_replay_context=context)
        model = programs[TEMPERATURE_POOL_STRATEGY]
        assert isinstance(model, FittedEnsemble)
        assert model.temperature_record["strategy"] == "grouped_temperature"
        assert model.scorer.pool_calibration["base_probability_strategy"] == "grouped_temperature"
        assert model.oof_probability_record["captured_expert_ids"] == (specs[0].expert_id,)
        assert model.pool_oof_record["captured_expert_ids"] == program.expert_ids
        assert model.pool_oof_record["base_temperature_OOF_sha256"]
        for row in model.pool_oof_record["rows"]:
            assert set(row["expert_evidence"]) == set(program.expert_ids)
            assert row["held_record_id"] not in row["training_record_ids"]
            inner_temperature = row["inner_temperature_calibration"]
            assert inner_temperature["temperature_expert_ids"] == (specs[0].expert_id,)
            assert inner_temperature["fixed_control_expert_ids"] == (specs[1].expert_id,)
            assert len(inner_temperature["OOF_groups"]) == 5
            for inner in inner_temperature["OOF_groups"]:
                assert row["held_record_id"] not in inner["training_record_ids"]
                assert row["held_record_id"] != inner["held_record_id"]
                assert inner["held_record_id"] not in inner["training_record_ids"]
        deep = [use for uses in recorder.uses.values() for use in uses
                if use["phase"] == "temperature_oof_inner" and use.get("pool_held_record_id")]
        assert deep
        assert all(use["temperature_held_record_id"] != use["pool_held_record_id"]
                   and not {use["temperature_held_record_id"], use["pool_held_record_id"]}
                   & set(use["training_record_ids"]) for use in deep)
        shared = FittedEnsemble.score_programs(programs, future)
        for name, evidence in shared.items():
            recorder.record_prefix_future_score(name, view=future, evidence=evidence)
        expected = {name: fitted.scorer.score(shared[name]).record() for name, fitted in programs.items()}
        persist_prefix_inner_replay(prefix, programs, tmp_path / "package", recorder=recorder)

    def no_fit(*args, **kwargs):
        raise AssertionError("temperature-pool replay must not refit EEG")

    monkeypatch.setattr(fit_owner, "_fit_expert", no_fit)
    with cpu_thread_budget(1):
        restored = FittedEnsemble.load_programs(tmp_path / "package")
        assert {name: fitted.score(future).record() for name, fitted in restored.items()} == expected
        replay = replay_prefix_inner_replay(tmp_path / "package", prefix=prefix, future_views=(future,))
    assert replay["scores"][0]["strategies"] == expected
    assert replay["producer_count"] == len(recorder.producers)
    assert replay["exact_score_checks"] > 6
    manifest_path = tmp_path / "package/prefix_inner_replay.json"
    original_manifest = manifest_path.read_bytes()
    payload = json.loads(original_manifest)
    producer = next(producer for producer in payload["record"]["inner_producers"]
                    if any(use["phase"] == "temperature_oof_inner" and use.get("pool_held_record_id")
                           for use in producer["uses"]))
    use = next(use for use in producer["uses"] if use["phase"] == "temperature_oof_inner"
               and use.get("pool_held_record_id"))
    use["temperature_held_record_id"] = use["pool_held_record_id"]
    payload["record_sha256"] = semantic_sha256(payload["record"])
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with cpu_thread_budget(1), pytest.raises(ValueError, match="OOF held row"):
        replay_prefix_inner_replay(tmp_path / "package")
    manifest_path.write_bytes(original_manifest)


def test_temperature_tail_declares_four_group_minimum_before_any_learner(case, monkeypatch):
    from transfer import ensemble_fit as fit_owner

    *_, specs, program, views = case
    parent = views.calibration[800]

    def no_fit(*args, **kwargs):
        raise AssertionError("structural four-group admission must precede the EEG learner")

    monkeypatch.setattr(fit_owner, "_fit_expert", no_fit)
    with cpu_thread_budget(1):
        earlier = views.prefix_compiler.before_group(parent, parent.complete_groups[3])
        assert len(earlier.complete_groups) == 3
        programs = fit_calibration_prefix_programs(earlier, specs, program,
            strategies=(TEMPERATURE_POOL_STRATEGY,), temperature_expert_ids=(specs[0].expert_id,))
    assert programs[TEMPERATURE_POOL_STRATEGY].stage == "pool_probability"


def test_temperature_tail_failure_preserves_actual_partial_hierarchy(case, tmp_path, monkeypatch):
    from transfer import ensemble_fit as fit_owner

    *_, specs, program, views = case
    prefix = views.calibration[800]
    future = next(view for view in views.decisions if view.samples is not None)
    actual_score = fit_owner._FittedExpert.score
    held_ids = {group.view.record_id for group in prefix.complete_groups}

    def missing_control(expert, view, **kwargs):
        if expert.program.expert_id == specs[1].expert_id and view.record_id in held_ids:
            return expert.unavailable(view, "declared fixed-control OOF failure")
        return actual_score(expert, view, **kwargs)

    monkeypatch.setattr(fit_owner._FittedExpert, "score", missing_control)
    recorder = InnerReplayRecorder()
    context = {"test_scope": views.ledger["test_scope"], "requested_decision_ids": [future.record_id]}
    with cpu_thread_budget(1):
        programs = fit_calibration_prefix_programs(prefix, specs, program,
            strategies=("fixed", TEMPERATURE_POOL_STRATEGY), temperature_expert_ids=(specs[0].expert_id,),
            inner_replay_recorder=recorder, inner_replay_context=context)
        assert programs["fixed"].fit_available
        assert programs[TEMPERATURE_POOL_STRATEGY].stage == "pool_probability"
        assert "declared fixed-control OOF failure" in programs[TEMPERATURE_POOL_STRATEGY].detail
        scores = FittedEnsemble.score_programs(programs, future)
        for strategy, value in scores.items():
            if strategy == TEMPERATURE_POOL_STRATEGY:
                from research.prefix_inner_replay import unavailable_prefix_expert_scores

                value = unavailable_prefix_expert_scores(prefix.record(), [spec.record() for spec in specs], strategy,
                    programs[strategy].record(subject=future.participant_key), future)
            recorder.record_prefix_future_score(strategy, view=future, evidence=value)
        persist_prefix_inner_replay(prefix, programs, tmp_path / "package", recorder=recorder)

    def no_fit(*args, **kwargs):
        raise AssertionError("unavailable pool programme replay must preserve its actual partial hierarchy")

    monkeypatch.setattr(fit_owner, "_fit_expert", no_fit)
    with cpu_thread_budget(1):
        replay = replay_prefix_inner_replay(tmp_path / "package", prefix=prefix, future_views=(future,))
    assert replay["producer_count"] == len(recorder.producers)
    assert replay["scores"][0]["strategies"][TEMPERATURE_POOL_STRATEGY]["log_probability"] is None


def test_predeclared_temperature_tail_final_future_and_parent_reject_changed_probability(case, tmp_path, monkeypatch):
    from experiments import run_candidate_ensemble_study as runner
    from transfer import ensemble_fit as fit_owner

    _, path, preprocessing, specs, program, _ = case
    candidate = CalibrationProgramCandidate("predeclared-tail", specs, program, TEMPERATURE_POOL_STRATEGY,
        temperature_expert_ids=(specs[0].expert_id,))
    plan = {"schema": "n2p3_candidate_ensemble_study/1", "evaluation_design": "independent_prefix",
        "evidence_scope": "software_correctness", "records": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
        "budgets": [800], "seed_pairs": [[0, 21]], "experts": [spec.record() for spec in specs],
        "mixtures": [program.record()], "baseline_mixture": program.name,
        "source_checkpoints": {}, "source_seed_catalog": [], "calibration_strategies": ["fixed"],
        "authorization_preprocessing": preprocessing.record(), "observation_response_seconds": 308 / 256,
        "pre_roll_seconds": .2, "device": "cpu", "workers": 1, "cpu_threads_per_worker": 1,
        "prefix_selection": {"schema": "n2p3_prefix_selection_plan/1", "output_name": "selected-tail",
            "baseline_name": candidate.name, "minimum_selection_rows": 1, "candidates": [candidate.record()]}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    actual_fit, fit_calls = fit_owner._fit_expert, []

    def captured_fit(*args, **kwargs):
        fitted = actual_fit(*args, **kwargs)
        fit_calls.append({"expert_id": args[0].expert_id, "labels": args[1].n_epochs})
        return fitted

    monkeypatch.setattr(fit_owner, "_fit_expert", captured_fit)
    with cpu_thread_budget(1):
        result = runner.run_study(plan_path, tmp_path / "run", execution="inline")
    assert result["status"] == "completed"
    participant = result["participants"][0]
    selected = [row for row in participant["predictions"] if row["calibration_strategy"] == "nested_selection"]
    assert len(selected) == len(participant["requested"])
    assert all(row["selected_candidate"]["strategy"] == TEMPERATURE_POOL_STRATEGY for row in selected)
    assert any(row["evidence"]["log_probability"] is not None for row in selected)
    directory = tmp_path / "run/participant_000"
    from research.selection_graph_store import SelectionGraphStore

    graph = SelectionGraphStore(directory / "source0_target21_K800_selection/prefix_selection.json").graph
    assert graph["selection"]["final_selection"]["minimum_training_groups"] == 4
    normalized = json.loads((tmp_path / "run/plan.json").read_bytes())
    source = json.loads((tmp_path / "run/executed_source.json").read_bytes())
    task = SimpleNamespace(output=str(tmp_path / "run"), arguments={"record_index": 0},
                           resources=SimpleNamespace(native_threads=1))
    outcome = SimpleNamespace(status="completed", value=str(directory / "result.json"))

    def no_fit(*args, **kwargs):
        raise AssertionError("published temperature-tail parent must not refit EEG")

    monkeypatch.setattr(fit_owner, "_fit_expert", no_fit)
    with cpu_thread_budget(1):
        admitted = runner._admit_candidate_participant(task, outcome, normalized, source)
    assert admitted["status"] == "completed" and len(admitted["requested"]) == len(selected)
    final_node = next(node for node in graph["nodes"] if node["fit_key"] == graph["selection"]["final_refit_key"])
    final_manifest = json.loads((directory / "source0_target21_K800_selection" / final_node["package"] /
        "ensemble.json").read_bytes())["record"]
    (tmp_path / "temperature_tail_readout.json").write_text(json.dumps({"status": admitted["status"],
        "requested": len(admitted["requested"]), "selected_predictions": len(selected),
        "selected_strategy": TEMPERATURE_POOL_STRATEGY, "actual_software_expert_fit_calls": len(fit_calls),
        "actual_fit_inputs": fit_calls, "graph_nodes": len(graph["nodes"]),
        "exact_score_checks": participant["selection_fits"][0]["exact_score_checks"],
        "final_pool_state": final_manifest["pool_calibration"],
        "final_temperature_state": final_manifest["temperature_calibration"],
        "source_digest": semantic_sha256(source)}, indent=2), encoding="utf-8")
    from research.calibration_selection import FittedPrefixSelection
    from research.prefix_selection_replay import _SelectionReplayAdmission

    selection_directory = directory / "source0_target21_K800_selection"
    with cpu_thread_budget(1):
        loaded = FittedPrefixSelection.load(selection_directory, expected_selection_sha256=semantic_sha256(graph["selection"]))
    changed_graph = json.loads(json.dumps(graph))
    for row in changed_graph["selection"]["candidates"]:
        row["temperature_expert_ids"] = [specs[1].expert_id]
    changed_graph["selection"]["selected_candidate"]["temperature_expert_ids"] = [specs[1].expert_id]
    for row in changed_graph["selection"]["outer_rows"]:
        row["selected_candidate"]["temperature_expert_ids"] = [specs[1].expert_id]
    for row in changed_graph["nodes"]:
        row["candidate"]["temperature_expert_ids"] = [specs[1].expert_id]
    with pytest.raises(ValueError, match="temperature subset"):
        FittedPrefixSelection(loaded.final, changed_graph["selection"])
    graph_path = selection_directory / "prefix_selection.json"
    original_graph = graph_path.read_bytes()
    graph_path.write_text(json.dumps({"record": changed_graph, "record_sha256": semantic_sha256(changed_graph)}), encoding="utf-8")
    with cpu_thread_budget(1), pytest.raises(ValueError, match="temperature subset"):
        _SelectionReplayAdmission(selection_directory).load()
    graph_path.write_bytes(original_graph)
    receipt = next(path for path in (directory / "prediction_receipts").glob("*.json")
        if runner._read_sealed_record(path)["calibration_strategy"] == "nested_selection"
        and runner._read_sealed_record(path)["evidence"]["log_probability"] is not None)
    changed = runner._read_sealed_record(receipt)
    original_receipt = receipt.read_bytes()
    changed["evidence"]["log_probability"] = np.log(np.full(36, 1 / 36)).tolist()
    runner._sealed_record(receipt, changed)
    with cpu_thread_budget(1), pytest.raises(ValueError, match="nested prefix prediction"):
        runner._admit_candidate_participant(task, outcome, normalized, source)
    receipt.write_bytes(original_receipt)
