"""Opt-in source/scratch families retain the actual prefix programme owners."""

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.io import savemat

from data.bnci2014_008_candidate import BNCI2014_008_CHANNELS, BNCI2014_008_SOURCE_REFERENCE
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from research import calibration_selection as selection
from research.contracts import semantic_sha256
from tests.support.checkpoints import write_checkpoint
from tests.support.independent_prefix import independent_prefix_case
from train.runtime import cpu_thread_budget
from transfer.calibration_candidates import waveform_lda_recipe
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram, FrozenSourceCheckpoint
from transfer.ensemble_views import SelectionEpochProfile, prepare_bnci_independent_prefixes
from transfer.recipes import TargetRecipe, erp_target_recipe


def declared_plan(profile):
    source = CalibrationExpertProgram("pretrained-A", "trial", TargetRecipe(
        "source-treatment", initialization="source", input_statistics="source", epochs=1,
        batch_size=128, bn_buffers="calibration", dropout="off", seed=21,
        trial_loss_reduction="calibration_balanced_fixed_mass"), profile)
    scratch = replace(source, expert_id="random-B", recipe=replace(source.recipe,
        name="scratch-treatment", initialization="scratch", input_statistics="target_prefix"))
    xdawn = CalibrationExpertProgram("spatial-C", "trial", erp_target_recipe(
        "xdawn_erp", bin_width=16, seed=21), profile)
    waveform = CalibrationExpertProgram("waveform-control", "trial", waveform_lda_recipe(seed=21), profile)
    ordinary = EnsembleProgram.equal_weights("ordinary-source", (source.expert_id,))
    return {"schema": "n2p3_candidate_ensemble_study/1", "evaluation_design": "independent_prefix",
            "experts": [item.record() for item in (source, scratch, xdawn, waveform)],
            "mixtures": [ordinary.record()], "baseline_mixture": ordinary.name,
            "seed_pairs": [[11, 21]], "budgets": [0, 200, 800],
            "source_seed_catalog": [{"external_authority": "preserve exact original declaration"}],
            "prefix_selection": {"historical": "old programme kept only in original plan"}}


def transform(plan, **overrides):
    factory = getattr(selection, "transfer_scratch_prefix_plan", None)
    assert factory is not None, "the explicit fixed transfer/scratch plan factory is missing"
    arguments = {"source_expert_id": "pretrained-A", "scratch_expert_id": "random-B",
                 "xdawn_expert_id": "spatial-C", "baseline_name": "scratch_nn_fixed",
                 "output_name": "transfer-scratch-selected", "minimum_selection_rows": 1}
    return factory(plan, **{**arguments, **overrides})


@pytest.fixture(scope="module")
def plan_case():
    preprocessing = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    profile = SelectionEpochProfile.from_preprocessing(replace(preprocessing, baseline_mode="none"),
        source_sample_rate_hz=256., channel_names=BNCI2014_008_CHANNELS, reference=BNCI2014_008_SOURCE_REFERENCE)
    return declared_plan(profile)


def test_explicit_family_changes_only_selection_and_preserves_original_plan(plan_case):
    original = copy.deepcopy(plan_case)
    transformed = transform(plan_case)
    assert plan_case == original
    assert {key: value for key, value in transformed.items() if key != "prefix_selection"} == {
        key: value for key, value in original.items() if key != "prefix_selection"}
    candidates = tuple(selection.CalibrationProgramCandidate.from_record(row)
                       for row in transformed["prefix_selection"]["candidates"])
    assert [candidate.name for candidate in candidates] == [
        "source_nn_fixed", "scratch_nn_fixed", "source_nn_xdawn_fixed", "scratch_nn_xdawn_fixed"]
    assert [candidate.ensemble_program.expert_ids for candidate in candidates] == [
        ("pretrained-A",), ("random-B",), ("pretrained-A", "spatial-C"), ("random-B", "spatial-C")]
    assert all(candidate.strategy == "fixed" and candidate.temperature_expert_ids is None for candidate in candidates)
    assert candidates[2].ensemble_program.weights == candidates[3].ensemble_program.weights == (.5, .5)
    assert selection._prefix_common_training_groups(candidates) == 1
    transformed["source_seed_catalog"][0]["external_authority"] = "changed derived plan"
    assert plan_case == original


@pytest.mark.parametrize("violation", ["seed", "epochs", "lr", "profile", "wrong_source",
                                      "wrong_scratch", "wrong_xdawn", "duplicate", "unknown"])
def test_family_rejects_unmatched_or_guessed_programmes(plan_case, violation):
    plan = copy.deepcopy(plan_case)
    programmes = [CalibrationExpertProgram.from_record(record) for record in plan["experts"]]
    if violation in {"seed", "epochs", "lr"}:
        recipe = programmes[1].recipe
        programmes[1] = replace(programmes[1], recipe=replace(recipe, **{
            violation: getattr(recipe, violation) + 1}))
    elif violation == "profile":
        profile = programmes[1].profile
        programmes[1] = replace(programmes[1], profile=replace(profile,
            preprocessing=replace(profile.preprocessing, baseline_mode="mean_only")))
    elif violation == "wrong_source":
        programmes[0] = replace(programmes[0], recipe=programmes[1].recipe)
    elif violation == "wrong_scratch":
        programmes[1] = replace(programmes[1], recipe=programmes[0].recipe)
    elif violation == "wrong_xdawn":
        programmes[2] = replace(programmes[2], recipe=erp_target_recipe("toeplitz_lda"))
    elif violation == "duplicate":
        programmes.append(programmes[0])
    plan["experts"] = [programme.record() for programme in programmes]
    with pytest.raises(ValueError):
        transform(plan, **({"source_expert_id": "NN"} if violation == "unknown" else {}))


@pytest.mark.parametrize("arguments", [{"baseline_name": "future-winner"}, {"output_name": "ordinary-source"},
                                       {"output_name": " "}, {"minimum_selection_rows": 0},
                                       {"minimum_selection_rows": True}, {"scratch_expert_id": "pretrained-A"}])
def test_family_requires_explicit_legal_baseline_output_and_row_policy(plan_case, arguments):
    with pytest.raises(ValueError):
        transform(plan_case, **arguments)


def test_same_target_statistics_are_also_a_legal_matched_programme(plan_case):
    plan = copy.deepcopy(plan_case)
    source = CalibrationExpertProgram.from_record(plan["experts"][0])
    plan["experts"][0] = replace(source, recipe=replace(source.recipe, input_statistics="target_prefix")).record()
    assert len(transform(plan)["prefix_selection"]["candidates"]) == 4


def test_xdawn_retains_its_own_declared_numerical_profile(plan_case):
    plan = copy.deepcopy(plan_case)
    xdawn = CalibrationExpertProgram.from_record(plan["experts"][2])
    profile = replace(xdawn.profile, preprocessing=replace(xdawn.profile.preprocessing, baseline_mode="mean_only"))
    plan["experts"][2] = replace(xdawn, profile=profile, recipe=replace(xdawn.recipe, seed=0)).record()
    candidates = tuple(selection.CalibrationProgramCandidate.from_record(row)
        for row in transform(plan)["prefix_selection"]["candidates"])
    assert candidates[2].expert_programs[1].profile.digest == profile.digest
    assert candidates[3].expert_programs[1].record() == plan["experts"][2]
    assert candidates[0].expert_programs[0].profile.digest != profile.digest


@pytest.mark.parametrize("field,value", [("schema", "other-plan/1"), ("evaluation_design", "calibration_groups")])
def test_family_retains_original_prefix_authority_as_its_only_input_route(plan_case, field, value):
    with pytest.raises(ValueError):
        transform({**plan_case, field: value})


@pytest.fixture(scope="module")
def actual_case(tmp_path_factory):
    root, fields, path, preprocessing, experts, _ = independent_prefix_case(
        tmp_path_factory.mktemp("transfer-scratch-native"))
    plan = declared_plan(experts[1].profile)
    xdawn = CalibrationExpertProgram.from_record(plan["experts"][2])
    own_profile = replace(xdawn.profile, preprocessing=replace(
        xdawn.profile.preprocessing, baseline_mode="mean_only"))
    plan["experts"][2] = replace(xdawn, profile=own_profile, recipe=replace(xdawn.recipe, seed=0)).record()
    with cpu_thread_budget(1):
        views = prepare_bnci_independent_prefixes(path,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), budgets=(0, 200, 800),
            profiles=(experts[1].profile, own_profile), authorization_preprocessing=preprocessing)
        checkpoint = write_checkpoint(root / "external-source.pt",
            views.calibration[200].trial_datasets[experts[1].profile.digest], source_snapshot_sha256="b" * 64)
    payload = torch.load(checkpoint, weights_only=True)
    payload["training_contract"]["seed"] = 11
    payload["training_contract"]["optimizer"]["config"] = {
        "pos_weight": payload["source_calibration"]["pos_weight"]}
    payload["training_contract_digest"] = semantic_sha256(payload["training_contract"])
    torch.save(payload, checkpoint)
    source = FrozenSourceCheckpoint.from_path(checkpoint)
    plan.update(evidence_scope="software_correctness",
        records=[{"path": str(path), "sha256": views.ledger["source_sha256"]}],
        source_seed_catalog=[{"outer_target_study": "BNCI2014-008", "source_seed": 11,
            "path": str(checkpoint), "sha256": source.digest,
            "procedure": {"kind": "fixed_recipe", "selection": "not_applicable"}}],
        authorization_preprocessing=preprocessing.record(), observation_response_seconds=308 / 256,
        pre_roll_seconds=.2, device="cpu", workers=1, cpu_threads_per_worker=1,
        source_checkpoints={}, calibration_strategies=["fixed"])
    return root, fields, path, preprocessing, views, source, plan


def test_actual_family_cli_fits_all_programmes_and_replays_graph_parent(actual_case, tmp_path, monkeypatch):
    from experiments import run_candidate_ensemble_study as runner
    from research.prefix_selection_replay import replay_prefix_selection
    from transfer import ensemble_fit as fit_owner
    from transfer.binary import scratch_like_trunk
    from transfer.calibration_candidates import _state_digest
    from transfer.ensemble_fit import FittedEnsemble

    _, _, _, _, views, source, original = actual_case
    plan = transform(original)
    plan_path = tmp_path / "new-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    actual_fits = []
    fit = fit_owner.fit_calibration_candidate

    def observe_fit(*args, **kwargs):
        fitted = fit(*args, **kwargs)
        if fitted.recipe.initialization in {"source", "scratch"}:
            actual_fits.append((fitted.recipe, fitted.fit_history,
                               _state_digest(dict(kwargs["source_trunk"].state_dict())),
                               _state_digest(dict(scratch_like_trunk(kwargs["source_trunk"],
                                   seed=fitted.recipe.seed).state_dict()))))
        return fitted

    monkeypatch.setattr(fit_owner, "fit_calibration_candidate", observe_fit)
    with cpu_thread_budget(1):
        report = runner.run_study(plan_path, tmp_path / "run", execution="inline")
    assert report["status"] == "completed"
    assert {recipe.initialization for recipe, *_ in actual_fits} == {"source", "scratch"}
    for recipe, history, donor, fresh in actual_fits:
        assert donor != fresh
        assert history["initial_state_sha256"] == (donor if recipe.initialization == "source" else fresh)
        assert recipe.seed == 21
    participant = report["participants"][0]
    predictions = [row for row in participant["predictions"] if row["calibration_strategy"] == "nested_selection"]
    assert len(predictions) == 3 * len(participant["requested"])
    assert all(row["saved_replay_exact"] for row in predictions)
    zero = [row for row in predictions if row["K"] == 0]
    assert zero and all(row["selected_candidate"]["name"] == "scratch_nn_fixed" for row in zero)
    assert all(row["evidence"]["log_probability"] is None for row in zero)
    ordinary_zero = [row for row in participant["predictions"]
        if row["K"] == 0 and row["mixture"] == "ordinary-source"]
    assert len(ordinary_zero) == len(participant["requested"])
    assert any(row["evidence"]["log_probability"] is not None for row in ordinary_zero)
    directory = tmp_path / "run" / "participant_000"
    graphs = {}
    for budget in (200, 800):
        package = directory / f"source11_target21_K{budget}_selection"
        from research.selection_graph_store import SelectionGraphStore

        graph = SelectionGraphStore(package / "prefix_selection.json").graph
        graphs[budget] = graph
        record = graph["selection"]
        assert record["final_refit_scope"]["actual_K"] == budget
        assert record["outer_scores_used_for_final_selection"] is False
        assert len(record["outer_rows"]) == len(views.calibration[budget].complete_groups)
        assert len(record["final_selection"]["requested_held_record_ids"]) == (0 if budget == 200 else 5)
        final_node = next(node for node in graph["nodes"] if node["fit_key"] == record["final_refit_key"])
        final = FittedEnsemble.load(package / final_node["package"])
        assert final.fit_receipts[0]["label_use"]["raw"]["used_labels"] == budget
        assert final.fit_receipts[0]["label_use"]["trial"][final._experts[0].program.profile.digest]["used_labels"] == (196 if budget == 200 else 796)
    assert graphs[200]["selection"]["selected_candidate"]["name"] == "scratch_nn_fixed"
    assert graphs[200]["selection"]["final_selection"]["decision"]["candidate_metrics"] == {}
    assert graphs[200]["selection"]["outer_rows"][0]["status"] == "unavailable"
    assert {node["candidate"]["name"] for node in graphs[800]["nodes"]} == {
        "source_nn_fixed", "scratch_nn_fixed", "source_nn_xdawn_fixed", "scratch_nn_xdawn_fixed"}
    first_models = {}
    for node in graphs[800]["nodes"]:
        if node["scope"]["actual_K"] != 120:
            continue
        fitted = FittedEnsemble.load(directory / "source11_target21_K800_selection" / node["package"])
        first_models[node["candidate"]["name"]] = fitted
        for expert in fitted._experts:
            assert expert.source_digest == (source.digest if expert.program.recipe.initialization == "source" else None)
    assert first_models["source_nn_fixed"]._experts[0].predictor.record["architecture"] == first_models[
        "scratch_nn_fixed"]._experts[0].predictor.record["architecture"]
    for outer in graphs[800]["selection"]["outer_rows"]:
        for rows in outer["selection"]["candidate_selection_rows"].values():
            assert all(row["held_record_id"] != outer["held_record_id"] for row in rows)

    def no_new_fit(*args, **kwargs):
        raise AssertionError("family graph replay must not fit EEG again")

    monkeypatch.setattr(fit_owner, "_fit_expert", no_new_fit)
    package = directory / "source11_target21_K800_selection"
    with cpu_thread_budget(1):
        replay = replay_prefix_selection(package, prefix=views.calibration[800],
            prefix_compiler=views.prefix_compiler, future_views=views.decisions,
            source_checkpoints={name: source for name in ("pretrained-A", "random-B")})
    assert replay["zero_new_fits"] and replay["exact_score_checks"] > 0
    # The real parent recomputes selected mixtures instead of trusting resealed probabilities.
    receipt = next(path for path in (directory / "prediction_receipts").glob("*.json")
        if runner._read_sealed_record(path)["calibration_strategy"] == "nested_selection"
        and runner._read_sealed_record(path)["evidence"]["log_probability"] is not None)
    changed = runner._read_sealed_record(receipt)
    changed["evidence"]["log_probability"] = np.log(np.ones(36) / 36).tolist()
    runner._sealed_record(receipt, changed)
    normalized = json.loads((tmp_path / "run" / "plan.json").read_bytes())
    executed = json.loads((tmp_path / "run" / "executed_source.json").read_bytes())
    task = SimpleNamespace(output=str(tmp_path / "run"), arguments={"record_index": 0},
                           resources=SimpleNamespace(native_threads=1))
    outcome = SimpleNamespace(status="completed", value=str(directory / "result.json"))
    with cpu_thread_budget(1), pytest.raises(ValueError, match="nested prefix prediction"):
        runner._admit_candidate_participant(task, outcome, normalized, executed,
            source_snapshots={("BNCI2014-008", 11): source})


def test_held_truth_change_cannot_enter_earlier_source_scratch_programmes(actual_case, tmp_path):
    _, fields, _, preprocessing, views, source, plan = actual_case
    prefix = views.calibration[800]
    held = prefix.complete_groups[4]
    changed_fields = copy.deepcopy(fields)
    start = held.view.raw_scope[0]
    stop = held.view.raw_scope[1]
    stimuli = changed_fields["y_stim"][start:stop]
    changed_fields["y"][start:stop] = np.where(stimuli == 0, 0., np.where((stimuli == 6) | (stimuli == 8), 2., 1.))
    changed_path = tmp_path / "changed-held" / "A01.mat"
    changed_path.parent.mkdir()
    savemat(changed_path, {"data": changed_fields})
    profiles = tuple(CalibrationExpertProgram.from_record(row).profile for row in plan["experts"])
    with cpu_thread_budget(1):
        changed_views = prepare_bnci_independent_prefixes(changed_path,
            source_sha256=hashlib.sha256(changed_path.read_bytes()).hexdigest(), budgets=(800,),
            profiles=profiles, authorization_preprocessing=preprocessing)
    before = views.prefix_compiler.before_group(prefix, held)
    changed_before = changed_views.prefix_compiler.before_group(changed_views.calibration[800],
        changed_views.calibration[800].complete_groups[4])
    np.testing.assert_array_equal(before.raw.target_mask, changed_before.raw.target_mask)
    np.testing.assert_array_equal(before.raw.samples, changed_before.raw.samples)
    candidates = tuple(selection.CalibrationProgramCandidate.from_record(row)
        for row in transform(plan)["prefix_selection"]["candidates"])
    fitted = []
    with cpu_thread_budget(1):
        for actual, compiler in ((before, views.prefix_compiler), (changed_before, changed_views.prefix_compiler)):
            result = selection.fit_nested_prefix_selection(actual, candidates, baseline_name="scratch_nn_fixed",
                prefix_compiler=compiler, minimum_selection_rows=1,
                source_checkpoints={name: source for name in ("pretrained-A", "random-B")})
            fitted.append(result)
    assert fitted[0].record["selected_candidate"] == fitted[1].record["selected_candidate"]
    for left, right in zip(fitted[0].final._experts, fitted[1].final._experts, strict=True):
        if left.program.recipe.initialization in {"source", "scratch"}:
            assert left.predictor.record["architecture"] == right.predictor.record["architecture"]
        for name, tensor in left.predictor._state.items():
            assert torch.equal(tensor, right.predictor._state[name])


def test_explicit_source_baseline_keeps_frozen_source_k0(actual_case):
    _, _, _, _, views, source, plan = actual_case
    family = transform(plan, baseline_name="source_nn_fixed")["prefix_selection"]
    candidates = tuple(selection.CalibrationProgramCandidate.from_record(row) for row in family["candidates"])
    with cpu_thread_budget(1):
        result = selection.fit_nested_prefix_selection(views.calibration[0], candidates,
            baseline_name=family["baseline_name"], prefix_compiler=views.prefix_compiler,
            minimum_selection_rows=family["minimum_selection_rows"],
            source_checkpoints={name: source for name in ("pretrained-A", "random-B")})
    assert result.record["selected_candidate"]["name"] == "source_nn_fixed"
    assert result.record["final_selection"]["decision"]["candidate_metrics"] == {}
    assert result.record["final_refit_scope"]["actual_K"] == 0
    assert not result.final.fit_receipts
    assert result.final._experts[0].source_digest == source.digest


def test_declared_family_ties_and_missing_candidates_keep_requested_rows(actual_case):
    *_, views, _, plan = actual_case
    family = transform(plan)["prefix_selection"]
    groups = views.calibration[800].complete_groups[1:3]
    outcomes = []
    for group in groups:
        probability = np.full(36, .4 / 35)
        probability[group.view.decision.candidate_vocabulary.index(group.target_candidate)] = .6
        identity = CandidateExpertIdentity("complete-software", "a" * 64, "b" * 64, None, "c" * 64, "software", False)
        counts = tuple(map(int, group.view.decision.membership.sum(axis=0)))
        outcomes.append(CandidateEvidence(group.view.decision.candidate_vocabulary, np.log(probability),
            group.view.participant_key, group.view.decision_id, group.view.physical_support_digest,
            (identity,), group.view.decision.repetitions, counts, counts))
    metrics = selection._metrics(groups, outcomes)
    names = tuple(row["name"] for row in family["candidates"])
    winner, _ = selection.select_calibration_candidate(dict.fromkeys(names, metrics),
        baseline_name=family["baseline_name"], tie_order=names)
    assert winner == names[0]
    missing = selection._metrics(groups, outcomes[:1])
    roster = dict.fromkeys(names, metrics)
    roster[names[0]] = missing
    winner, record = selection.select_calibration_candidate(roster,
        baseline_name=family["baseline_name"], tie_order=names)
    assert winner == names[1]
    assert not record["comparisons"][names[0]]["eligible"]
    assert len(missing.cells) == 2 and missing.probability_coverage == .5
