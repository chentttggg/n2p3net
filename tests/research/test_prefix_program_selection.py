"""Actual prefix scopes, whole-program selection, and saved graph boundaries."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from tests.support.independent_prefix import independent_prefix_case
from train.runtime import cpu_thread_budget
from transfer.ensemble_views import prepare_bnci_independent_prefixes


@pytest.fixture(scope="module")
def scope_case(tmp_path_factory):
    root, fields, path, spec, experts, _ = independent_prefix_case(
        tmp_path_factory.mktemp("prefix-selection-scope"))
    with cpu_thread_budget(1):
        views = prepare_bnci_independent_prefixes(path,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            budgets=(0, 200, 800), profiles=(experts[1].profile,),
            authorization_preprocessing=spec)
    return root, fields, path, spec, experts, views


def test_before_group_rebuilds_real_prefix_and_preserves_full_final_dose(scope_case):
    *_, experts, views = scope_case
    parent = views.calibration[800]
    held = parent.complete_groups[4]
    with cpu_thread_budget(1):
        earlier = views.prefix_compiler.before_group(parent, held)
    assert earlier.budget == len(earlier.raw.target_mask) == 480
    assert earlier.raw.right_censored
    assert earlier.target_contract["raw_scope"][1] <= held.view.raw_scope[0]
    assert len(earlier.complete_groups) == 4
    data = earlier.trial_datasets[experts[1].profile.digest]
    assert data.n_epochs == 480
    assert parent.budget == 800 and len(parent.raw.target_mask) == 800
    assert parent.trial_datasets[experts[1].profile.digest].n_epochs == 796
    # Filtering all C and then selecting rows is a different numerical input.
    assert not np.array_equal(data.X, parent.trial_datasets[experts[1].profile.digest].X[:480])


def test_before_first_group_has_no_target_observations(scope_case):
    *_, views = scope_case
    parent = views.calibration[200]
    earlier = views.prefix_compiler.before_group(parent, parent.complete_groups[0])
    assert earlier.budget == 0 and earlier.raw is None
    assert not earlier.trial_datasets and not earlier.complete_groups


def test_prefix_compiler_rejects_forged_parent_actual_samples(scope_case):
    *_, views = scope_case
    parent = views.calibration[800]
    changed = parent.raw.samples.copy()
    changed[10, 0] += 1e-6
    forged = replace(parent, raw=replace(parent.raw, samples=changed))
    with pytest.raises(ValueError, match="original.*prefix|prefix.*owner|actual.*parent"):
        views.prefix_compiler.before_group(forged, forged.complete_groups[3])


def test_prefix_compiler_rejects_forged_held_and_future_group(scope_case):
    *_, views = scope_case
    parent = views.calibration[800]
    group = parent.complete_groups[3]
    forged = replace(group, target_candidate=group.view.decision.candidate_vocabulary[0],
                     labels=group.view.decision.membership[:, 0])
    with pytest.raises(ValueError, match="held.*owner|held.*parent|group.*authority"):
        views.prefix_compiler.before_group(parent, forged)
    with pytest.raises(ValueError, match="held.*owner|held.*parent|group.*authority"):
        views.prefix_compiler.before_group(views.calibration[200], group)


def test_compiler_constructor_requires_original_source_bytes_not_claimed_record(scope_case):
    from transfer.ensemble_views import CalibrationPrefixCompiler

    _, _, path, spec, experts, views = scope_case
    with pytest.raises(ValueError, match="actual original verified MAT bytes"):
        CalibrationPrefixCompiler(views.prefix_compiler._record,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_relative_path=path.name,
            budgets=(200,), profiles=(experts[1].profile,), authorization_preprocessing=spec)
    corrupted = bytearray(path.read_bytes())
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="actual original verified MAT bytes"):
        CalibrationPrefixCompiler(bytes(corrupted), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_relative_path=path.name, budgets=(200,), profiles=(experts[1].profile,),
            authorization_preprocessing=spec)


def _candidates(experts, *, grouped=False):
    from research.calibration_selection import CalibrationProgramCandidate
    from transfer.ensemble import EnsembleProgram

    spec = replace(experts[0], n_basis=1)
    roster = EnsembleProgram.equal_weights("whole-prefix-raw", (spec.expert_id,))
    return (CalibrationProgramCandidate("baseline", (spec,), roster),
            CalibrationProgramCandidate("alternative", (replace(spec, ridge=10.),), roster,
                                        "grouped_temperature" if grouped else "fixed"))


def test_whole_prefix_selection_uses_real_h_v_and_final_partial_labels(scope_case):
    from research.calibration_selection import fit_nested_prefix_selection

    *_, experts, views = scope_case
    with cpu_thread_budget(1):
        fitted = fit_nested_prefix_selection(views.calibration[800], _candidates(experts),
            baseline_name="baseline", prefix_compiler=views.prefix_compiler, minimum_selection_rows=1)
    assert len(fitted.record["outer_rows"]) == 6
    assert fitted.record["final_refit_scope"]["actual_K"] == 800
    assert fitted.final.fit_receipts[0]["label_use"]["raw"]["used_labels"] == 800
    for outer in fitted.record["outer_rows"]:
        held = next(group for group in views.calibration[800].complete_groups
                    if group.view.record_id == outer["held_record_id"])
        assert outer["refit_scope"]["input"]["target_contract"]["raw_scope"][1] <= held.view.raw_scope[0]
        for rows in outer["selection"]["candidate_selection_rows"].values():
            assert all(row["held_record_id"] != held.view.record_id for row in rows)
    assert fitted.record["outer_rows"][0]["status"] == "unavailable"
    assert fitted.record["outer_rows"][0]["evidence"]["failure_kind"] == "not_available"


def test_one_complete_group_runs_fixed_baseline_without_claiming_selection(scope_case):
    from research.calibration_selection import fit_nested_prefix_selection

    *_, experts, views = scope_case
    with cpu_thread_budget(1):
        fitted = fit_nested_prefix_selection(views.calibration[200], _candidates(experts, grouped=True),
            baseline_name="baseline", prefix_compiler=views.prefix_compiler, minimum_selection_rows=1)
    assert fitted.record["selected_candidate"]["name"] == "baseline"
    assert fitted.record["final_selection"]["decision"]["candidate_metrics"] == {}
    assert fitted.final.fit_receipts[0]["label_use"]["raw"]["used_labels"] == 200
    assert len(fitted.record["outer_rows"]) == 1


def test_actual_highdose_model_cannot_rebind_to_lowdose_prefix(scope_case):
    from research.nested_selection_replay import InnerReplayRecorder
    from research.prefix_inner_replay import _check_final_authority, prefix_native_fit_inputs
    from transfer.ensemble import EnsembleProgram
    from transfer.ensemble_fit import FittedEnsemble, _FittedExpert, fit_calibration_prefix_programs

    *_, experts, views = scope_case
    spec = replace(experts[0], n_basis=1)
    roster = EnsembleProgram.equal_weights("native-fit-input", (spec.expert_id,))
    recorder = InnerReplayRecorder()
    with cpu_thread_budget(1):
        low = fit_calibration_prefix_programs(views.calibration[200], (spec,), roster,
            strategies=("fixed",), inner_replay_recorder=recorder)["fixed"]
        high = fit_calibration_prefix_programs(views.calibration[800], (spec,), roster,
            strategies=("fixed",))["fixed"]
    from research.prefix_inner_replay import prefix_input_identity

    parent = views.calibration[200]
    record = {"prefix_input": prefix_input_identity(parent), "prefix_record_sha256":
        __import__("research.contracts", fromlist=["semantic_sha256"]).semantic_sha256(parent.record()),
        "actual_K": 200, "participant_key": parent.target_contract["participant_key"],
        "expert_programs": [spec.record()], "ensemble_program": roster.record(),
        "source_checkpoint_digests": {}, "source_checkpoint_roles": {spec.expert_id: {"digest": None, "role": None}},
        "fit_context": {}, "native_fit_inputs": prefix_native_fit_inputs(parent, (spec,))}
    _check_final_authority(record, {"fixed": low})
    native = replace(high._experts[0].predictor, provenance={**high._experts[0].predictor.provenance,
        "ensemble_fit_binding": low._experts[0].predictor.provenance["ensemble_fit_binding"]})
    expert = _FittedExpert(spec, native, native.digest(), low._experts[0].fit_digest, None, 0.)
    forged = FittedEnsemble(low.scorer, (expert,), low.fit_receipts, low.temperature_record,
                            target_contract=low._target_contract)
    with pytest.raises(ValueError, match="native continuous fitter input/dose"):
        _check_final_authority(record, {"fixed": forged})


def test_same_run_graph_saves_and_zero_fit_replays_actual_models(scope_case, tmp_path, monkeypatch):
    import transfer.ensemble_fit as fit_owner
    from research.calibration_selection import FittedPrefixSelection, fit_nested_prefix_selection
    from research.contracts import semantic_sha256
    from research.prefix_selection_replay import replay_prefix_selection

    *_, experts, views = scope_case
    future = next(view for view in views.decisions if view.samples is not None)
    context = {"test_scope": views.ledger["test_scope"], "requested_decision_ids": [future.record_id]}
    with cpu_thread_budget(1):
        selected = fit_nested_prefix_selection(views.calibration[200], _candidates(experts, grouped=True),
            baseline_name="baseline", prefix_compiler=views.prefix_compiler,
            minimum_selection_rows=1, context=context)
        expected = selected.score(future)
        package = selected.save(tmp_path / "whole-prefix")

        def no_fit(*args, **kwargs):
            raise AssertionError("saved graph replay attempted a new numerical fit")

        monkeypatch.setattr(fit_owner, "_fit_expert", no_fit)
        restored = FittedPrefixSelection.load(package, expected_selection_sha256=semantic_sha256(selected.record))
        assert restored.score(future).record() == expected.record()
        replay = replay_prefix_selection(package, prefix=views.calibration[200],
            prefix_compiler=views.prefix_compiler, future_views=(future,),
            expected_selection_sha256=semantic_sha256(selected.record), expected_graph_sha256=selected.graph_sha256)
    assert replay["zero_new_fits"] and replay["exact_score_checks"] == 2
    assert replay["outer_requested"] == 1 and replay["actual_K"] == 200


def test_grouped_candidate_keeps_original_inner_program_inside_h(scope_case, tmp_path):
    from research.calibration_selection import fit_nested_prefix_selection
    from research.prefix_selection_replay import replay_prefix_selection

    *_, experts, views = scope_case
    future = next(view for view in views.decisions if view.samples is not None)
    with cpu_thread_budget(1):
        selected = fit_nested_prefix_selection(views.calibration[800], _candidates(experts, grouped=True),
            baseline_name="baseline", prefix_compiler=views.prefix_compiler, minimum_selection_rows=1,
            context={"test_scope": views.ledger["test_scope"], "requested_decision_ids": [future.record_id]})
        trace = selected.record["final_selection"]
        assert len(trace["requested_held_record_ids"]) == 3
        for row in trace["candidate_selection_rows"]["alternative"]:
            node = selected._fits.nodes[row["fit_key"]]
            grouped = node["programs"]["grouped_temperature"]
            assert grouped.temperature_record["strategy"] == "grouped_temperature"
            assert grouped.oof_probability_record["actual_K"] == node["prefix"].budget
            assert grouped.oof_probability_record["selection_performed"] is False
            assert len(node["prefix"].complete_groups) >= 3
            assert all(item["held_record_id"] != row["held_record_id"]
                       for item in grouped.oof_probability_record["rows"])
        selected.score(future)
        package = selected.save(tmp_path / "grouped-graph")
        replay = replay_prefix_selection(package, prefix=views.calibration[800],
            prefix_compiler=views.prefix_compiler, future_views=(future,),
            expected_graph_sha256=selected.graph_sha256)
    assert replay["zero_new_fits"] and replay["producer_count"] == selected.record["fit_calls"]


def test_public_caller_captures_replays_and_parent_rejects_changed_selected_probability(scope_case, tmp_path):
    import json
    from types import SimpleNamespace

    from experiments.run_candidate_ensemble_study import (
        _admit_candidate_participant,
        _read_sealed_record,
        _sealed_record,
        run_study,
    )
    from research.calibration_selection import CalibrationProgramCandidate
    from transfer.ensemble import EnsembleProgram

    _, _, path, spec, experts, _ = scope_case
    raw = replace(experts[0], n_basis=1)
    primary = EnsembleProgram.equal_weights("raw_primary", (raw.expert_id,))
    fixed = CalibrationProgramCandidate("fixed", (raw,), primary)
    learned = CalibrationProgramCandidate("temperature", (raw,), primary, "grouped_temperature")
    plan = {"schema": "n2p3_candidate_ensemble_study/1", "evidence_scope": "software_correctness",
        "evaluation_design": "independent_prefix", "seed_pairs": [[0, 20260918]],
        "records": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
        "budgets": [0, 200], "experts": [raw.record()], "mixtures": [primary.record()],
        "baseline_mixture": primary.name, "authorization_preprocessing": spec.record(),
        "observation_response_seconds": 308 / 256, "pre_roll_seconds": .2, "device": "cpu",
        "workers": 1, "cpu_threads_per_worker": 1, "source_checkpoints": {}, "source_seed_catalog": [],
        "calibration_strategies": ["fixed"], "prefix_selection": {"schema": "n2p3_prefix_selection_plan/1",
            "output_name": "selected", "candidates": [fixed.record(), learned.record()],
            "baseline_name": "fixed", "minimum_selection_rows": 1}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with cpu_thread_budget(1):
        report = run_study(plan_path, tmp_path / "run", execution="inline")
    assert report["status"] == "completed"
    assert "selected@nested_selection@K200" in report["metrics"]
    participant = report["participants"][0]
    selected = [row for row in participant["predictions"] if row["calibration_strategy"] == "nested_selection"]
    assert len(selected) == 2 * len(participant["requested"])
    assert len(participant["selection_fits"]) == 2
    assert all(row["selected_candidate"]["name"] == "fixed" for row in selected)
    assert participant["selection_fits"][1]["final_refit_scope"]["actual_K"] == 200
    directory = tmp_path / "run" / "participant_000"
    target = next(path for path in (directory / "prediction_receipts").glob("*.json")
        if _read_sealed_record(path)["calibration_strategy"] == "nested_selection"
        and _read_sealed_record(path)["evidence"]["log_probability"] is not None)
    changed = _read_sealed_record(target)
    changed["evidence"]["log_probability"] = np.log(np.ones(36) / 36).tolist()
    _sealed_record(target, changed)
    normalized = json.loads((tmp_path / "run" / "plan.json").read_bytes())
    source = json.loads((tmp_path / "run" / "executed_source.json").read_bytes())
    task = SimpleNamespace(output=str(tmp_path / "run"), arguments={"record_index": 0},
                           resources=SimpleNamespace(native_threads=1))
    outcome = SimpleNamespace(status="completed", value=str(directory / "result.json"))
    with cpu_thread_budget(1), pytest.raises(ValueError, match="nested prefix prediction"):
        _admit_candidate_participant(task, outcome, normalized, source)


def test_final_expected_subject_error_keeps_complete_saved_graph_and_future_roster(scope_case, tmp_path):
    from research.calibration_selection import (
        CalibrationProgramCandidate,
        FittedPrefixSelection,
        fit_nested_prefix_selection,
    )
    from research.contracts import semantic_sha256
    from research.execution import ExpectedSubjectError
    from research.prefix_selection_replay import replay_prefix_selection
    from transfer.ensemble import EnsembleProgram

    *_, experts, views = scope_case
    spec = replace(experts[0], n_basis=1)
    candidate = CalibrationProgramCandidate("explicit-grouped-baseline", (spec,),
        EnsembleProgram.equal_weights("unavailable-grouped", (spec.expert_id,)), "grouped_temperature")
    context = {"test_scope": views.ledger["test_scope"],
               "requested_decision_ids": [view.record_id for view in views.decisions]}
    with cpu_thread_budget(1):
        selected = fit_nested_prefix_selection(views.calibration[200], (candidate,),
            baseline_name=candidate.name, prefix_compiler=views.prefix_compiler, minimum_selection_rows=1,
            context=context)
        assert isinstance(selected.final, ExpectedSubjectError)
        evidence = [selected.score(view) for view in views.decisions]
        assert all(value.log_probability is None for value in evidence)
        package = selected.save(tmp_path / "expected-final-failure")
        restored = FittedPrefixSelection.load(package, expected_selection_sha256=semantic_sha256(selected.record))
        assert isinstance(restored.final, ExpectedSubjectError)
        replay = replay_prefix_selection(package, prefix=views.calibration[200],
            prefix_compiler=views.prefix_compiler, future_views=views.decisions,
            expected_graph_sha256=selected.graph_sha256)
    assert len(replay["scores"]) == len(views.decisions)
    assert replay["zero_new_fits"] and replay["outer_requested"] == 1
