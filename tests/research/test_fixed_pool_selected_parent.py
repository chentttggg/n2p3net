"""Predeclared fixed-tail final prediction reaches the real public parent."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from research.calibration_selection import CalibrationProgramCandidate
from research.contracts import semantic_sha256
from tests.research.test_pool_probability_prefix import case as pool_case
from train.runtime import cpu_thread_budget
from transfer.pool_calibration import POOL_STRATEGY


def test_predeclared_fixed_tail_final_future_and_public_parent(tmp_path, tmp_path_factory, monkeypatch):
    from experiments import run_candidate_ensemble_study as runner
    from transfer import ensemble_fit as fit_owner

    _, path, preprocessing, specs, program, _ = pool_case.__wrapped__(tmp_path_factory)
    candidate = CalibrationProgramCandidate("declared-tail", specs, program, POOL_STRATEGY)
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
    fit = fit_owner._fit_expert
    fits = []

    def actual_fit(*args, **kwargs):
        fitted = fit(*args, **kwargs)
        fits.append({"expert_id": args[0].expert_id, "labels": args[1].n_epochs,
                     "source_only": kwargs.get("source_only", False)})
        return fitted

    monkeypatch.setattr(fit_owner, "_fit_expert", actual_fit)
    with cpu_thread_budget(1):
        result = runner.run_study(plan_path, tmp_path / "run", execution="inline")
    assert result["status"] == "completed"
    participant = result["participants"][0]
    selected = [row for row in participant["predictions"] if row["calibration_strategy"] == "nested_selection"]
    assert len(selected) == len(participant["requested"])
    assert all(row["selected_candidate"]["strategy"] == POOL_STRATEGY for row in selected)
    assert any(row["evidence"]["log_probability"] is not None for row in selected)
    assert all(row["saved_replay_exact"] for row in selected)
    directory = tmp_path / "run/participant_000"
    from research.selection_graph_store import SelectionGraphStore

    graph = SelectionGraphStore(directory / "source0_target21_K800_selection/prefix_selection.json").graph
    selection = graph["selection"]
    assert selection["selected_candidate"]["strategy"] == POOL_STRATEGY
    assert selection["final_selection"]["minimum_training_groups"] == 2
    assert len(selection["final_selection"]["requested_held_record_ids"]) == 4
    assert selection["final_refit_scope"]["actual_K"] == 800
    final_node = next(node for node in graph["nodes"] if node["fit_key"] == selection["final_refit_key"])
    final_programme = json.loads((directory / "source0_target21_K800_selection" / final_node["package"] /
        "temperature_programs.json").read_bytes())["record"]["programs"][POOL_STRATEGY]
    assert final_programme["status"] == "completed" and final_programme["pool_calibration"]
    (tmp_path / "selected_tail_readout.json").write_text(json.dumps({"status": result["status"],
        "requested": len(participant["requested"]), "selected_predictions": len(selected),
        "selected_strategy": POOL_STRATEGY, "pool_calibration": final_programme["pool_calibration"],
        "actual_software_expert_fit_calls": len(fits), "actual_fit_inputs": fits,
        "graph_nodes": len(graph["nodes"]), "exact_score_checks": participant["selection_fits"][0]["exact_score_checks"]}), encoding="utf-8")

    def no_fit(*args, **kwargs):
        raise AssertionError("published selected-tail parent admission must not refit EEG")

    monkeypatch.setattr(fit_owner, "_fit_expert", no_fit)
    receipt = next(path for path in (directory / "prediction_receipts").glob("*.json")
        if runner._read_sealed_record(path)["calibration_strategy"] == "nested_selection"
        and runner._read_sealed_record(path)["evidence"]["log_probability"] is not None)
    changed = runner._read_sealed_record(receipt)
    changed["evidence"]["log_probability"] = np.log(np.full(36, 1 / 36)).tolist()
    runner._sealed_record(receipt, changed)
    normalized = json.loads((tmp_path / "run/plan.json").read_bytes())
    source = json.loads((tmp_path / "run/executed_source.json").read_bytes())
    assert semantic_sha256(source) == result["source_digest"]
    task = SimpleNamespace(output=str(tmp_path / "run"), arguments={"record_index": 0},
                           resources=SimpleNamespace(native_threads=1))
    outcome = SimpleNamespace(status="completed", value=str(directory / "result.json"))
    with cpu_thread_budget(1), pytest.raises(ValueError, match="nested prefix prediction"):
        runner._admit_candidate_participant(task, outcome, normalized, source)
