"""Actual complete-pool OOF, prefix selection and model-only probability replay."""

import hashlib
import json
from dataclasses import replace

import pytest

from research.calibration_selection import CalibrationProgramCandidate
from research.contracts import semantic_sha256
from research.nested_selection_replay import InnerReplayRecorder
from research.prefix_inner_replay import persist_prefix_inner_replay, replay_prefix_inner_replay
from tests.support.independent_prefix import independent_prefix_case
from train.runtime import cpu_thread_budget
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import FittedEnsemble, fit_calibration_prefix_programs
from transfer.ensemble_views import prepare_bnci_independent_prefixes
from transfer.pool_calibration import POOL_STRATEGY


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    _, fields, path, preprocessing, experts, _ = independent_prefix_case(tmp_path_factory.mktemp("pool-prefix"))
    spec = experts[1]
    # The nonunit weights are part of this exact programme's learning target.
    other = replace(spec, expert_id="other-toeplitz", recipe=replace(spec.recipe, name="other-bin", erp_bin_width=32))
    program = EnsembleProgram("declared-pool", (spec.expert_id, other.expert_id), (.25, .75))
    with cpu_thread_budget(1):
        views = prepare_bnci_independent_prefixes(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            budgets=(200, 800), profiles=(spec.profile,), authorization_preprocessing=preprocessing)
    return fields, path, preprocessing, (spec, other), program, views


def test_actual_pool_oof_all_experts_scope_and_zero_fit_saved_input_replay(case, tmp_path, monkeypatch):
    from transfer import ensemble_fit as fit_owner

    *_, specs, program, views = case
    prefix = views.calibration[800]
    future = next(view for view in views.decisions if view.samples is not None)
    context = {"test_scope": views.ledger["test_scope"], "requested_decision_ids": [future.record_id]}
    recorder = InnerReplayRecorder()
    with cpu_thread_budget(1):
        programs = fit_calibration_prefix_programs(prefix, specs, program,
            strategies=("fixed", POOL_STRATEGY), inner_replay_recorder=recorder, inner_replay_context=context)
        model = programs[POOL_STRATEGY]
        assert isinstance(model, FittedEnsemble)
        oof = model.pool_oof_record
        assert model.oof_probability_record is None
        assert oof["actual_K"] == 800 and oof["complete_group_labels"] == 720
        assert model.temperature_record["strategy"] == "fixed"
        assert model.probability_strategy == POOL_STRATEGY
        assert model.scorer.pool_calibration["risk_recipe"] == "equal_requested_complete_selection/1"
        assert len(oof["rows"]) == model.scorer.pool_calibration["requested_rows"] == 6
        assert len(recorder.producers) == 6
        for row in oof["rows"]:
            assert row["held_record_id"] not in row["training_record_ids"]
            assert len(row["training_record_ids"]) == 5
            assert set(row["expert_evidence"]) == set(program.expert_ids)
            assert row["phase"] == "pool_probability_oof_inner" and row["risk_mass"] == 1.
        shared = FittedEnsemble.score_programs(programs, future)
        for name, evidence in shared.items():
            recorder.record_prefix_future_score(name, view=future, evidence=evidence)
        expected = model.scorer.score(shared[POOL_STRATEGY]).record()
        persist_prefix_inner_replay(prefix, programs, tmp_path / "package", recorder=recorder)

    def no_fit(*args, **kwargs):
        raise AssertionError("probability replay must not fit EEG")

    monkeypatch.setattr(fit_owner, "_fit_expert", no_fit)
    with cpu_thread_budget(1):
        restored = FittedEnsemble.load_programs(tmp_path / "package")
        assert restored[POOL_STRATEGY].score(future).record() == expected
        replay = replay_prefix_inner_replay(tmp_path / "package", prefix=prefix, future_views=(future,))
    assert replay["scores"][0]["strategies"][POOL_STRATEGY] == expected
    assert replay["producer_count"] == 6 and replay["exact_score_checks"] == 8
    manifest = json.loads((tmp_path / "package/ensemble.json").read_bytes())
    assert manifest["record"]["schema"] in {"n2p3_fitted_candidate_ensemble/2", "n2p3_fitted_candidate_ensemble/3"}
    # Alter epsilon and coherently reseal this outer programme; actual model/OOF binding must still reject.
    probability_path = tmp_path / "package/temperature_programs.json"
    changed = json.loads(probability_path.read_bytes())
    state = changed["record"]["programs"][POOL_STRATEGY]["pool_calibration"]
    state["epsilon"] = .5 if state["epsilon"] != .5 else .25
    changed["record_sha256"] = semantic_sha256(changed["record"])
    probability_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="pool calibration"):
        FittedEnsemble.load_programs(tmp_path / "package")


def test_one_group_probability_program_is_explicitly_unavailable(case):
    *_, specs, program, views = case
    with cpu_thread_budget(1):
        fitted = fit_calibration_prefix_programs(views.calibration[200], specs, program,
            strategies=("fixed", POOL_STRATEGY))
    assert fitted["fixed"].fit_available
    assert fitted[POOL_STRATEGY].stage == "pool_probability"


def test_complete_pool_cannot_learn_epsilon_by_dropping_an_unavailable_expert(case, monkeypatch):
    from transfer import ensemble_fit as fit_owner

    *_, specs, program, views = case
    score = fit_owner._FittedExpert.score

    def missing(expert, view, **kwargs):
        if expert.program.expert_id == "other-toeplitz":
            return expert.unavailable(view, "declared held expert failure")
        return score(expert, view, **kwargs)

    monkeypatch.setattr(fit_owner._FittedExpert, "score", missing)
    with cpu_thread_budget(1):
        fitted = fit_calibration_prefix_programs(views.calibration[800], specs, program,
            strategies=("fixed", POOL_STRATEGY))
    assert isinstance(fitted["fixed"], FittedEnsemble)
    assert fitted[POOL_STRATEGY].stage == "pool_probability"
    assert "declared held expert failure" in fitted[POOL_STRATEGY].detail


def test_nested_probability_candidate_has_two_group_warmup_and_public_parent_replay(case, tmp_path):
    from experiments.run_candidate_ensemble_study import run_study

    _, path, preprocessing, specs, program, _ = case
    baseline = CalibrationProgramCandidate("fixed", specs, program)
    tail = CalibrationProgramCandidate("tail", specs, program, POOL_STRATEGY)
    plan = {"schema": "n2p3_candidate_ensemble_study/1", "evaluation_design": "independent_prefix",
        "evidence_scope": "software_correctness", "records": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
        "budgets": [800], "seed_pairs": [[0, 21]], "experts": [spec.record() for spec in specs],
        "mixtures": [program.record()], "baseline_mixture": program.name,
        "source_checkpoints": {}, "source_seed_catalog": [], "calibration_strategies": ["fixed"],
        "authorization_preprocessing": preprocessing.record(), "observation_response_seconds": 308 / 256,
        "pre_roll_seconds": .2, "device": "cpu", "workers": 1, "cpu_threads_per_worker": 1,
        "prefix_selection": {"schema": "n2p3_prefix_selection_plan/1", "output_name": "selected",
            "baseline_name": "fixed", "minimum_selection_rows": 1, "candidates": [baseline.record(), tail.record()]}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with cpu_thread_budget(1):
        result = run_study(plan_path, tmp_path / "run", execution="inline")
    assert result["status"] == "completed"
    participant = result["participants"][0]
    graph_path = tmp_path / "run/participant_000/source0_target21_K800_selection/prefix_selection.json"
    from research.selection_graph_store import SelectionGraphStore

    selected = SelectionGraphStore(graph_path).graph["selection"]
    assert selected["final_selection"]["minimum_training_groups"] == 2
    assert len(selected["final_selection"]["requested_held_record_ids"]) == 4
    assert selected["final_refit_scope"]["actual_K"] == 800
    assert selected["outer_scores_used_for_final_selection"] is False
    for outer in selected["outer_rows"]:
        for rows in outer["selection"]["candidate_selection_rows"].values():
            assert all(row["held_record_id"] != outer["held_record_id"] for row in rows)
    chosen = [row for row in participant["predictions"] if row["calibration_strategy"] == "nested_selection"]
    assert len(chosen) == len(participant["requested"])
    assert all(row["saved_replay_exact"] for row in chosen)
