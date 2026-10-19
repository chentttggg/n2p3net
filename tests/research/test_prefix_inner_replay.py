"""RED/green checks for same-run CalibrationPrefix producer replay."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from research.contracts import semantic_sha256
from research.execution import ExpectedSubjectError, SubjectFailureCode
from research.nested_selection_replay import InnerReplayRecorder
from research.prefix_inner_replay import (
    persist_prefix_inner_replay,
    replay_prefix_inner_replay,
)
from tests.support.independent_prefix import independent_prefix_case
from train.runtime import cpu_thread_budget
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import fit_calibration_prefix, fit_calibration_prefix_programs
from transfer.ensemble_views import prepare_bnci_independent_prefixes


def _raw_spatial_spec(case):
    from transfer.ensemble_views import observed_spatial_projection

    _, _, _, _, experts, views = case
    channels = ("Fz", "P3", "Pz", "P4", "PO8", "Oz")
    native = views.calibration[200].target_contract["channel_names"]
    matrix, _ = observed_spatial_projection(native, channels, reference_channel="Cz")
    declaration = {"mode": "observed_measured_reference", "slot_channels": list(channels),
        "reference_channel": "Cz", "observed_only_for_raw_covariance": True,
        "native_sample_rate_hz": 256., "matrix": matrix.tolist()}
    return replace(experts[0], expert_id="raw6", spatial=declaration)


@pytest.fixture(scope="module")
def independent_case(tmp_path_factory):
    return independent_prefix_case(tmp_path_factory.mktemp("prefix-replay"))


@pytest.fixture(autouse=True)
def one_native_thread():
    with cpu_thread_budget(1):
        yield


def _prefix_case(independent_case):
    _, _, _, _, experts, views = independent_case
    return experts, views.calibration[200], views


def test_prefix_inner_replay_persists_same_run_package_and_denominator(independent_case, tmp_path):
    experts, prefix, views = _prefix_case(independent_case)
    roster = EnsembleProgram.equal_weights("prefix", tuple(spec.expert_id for spec in experts))
    recorder = InnerReplayRecorder()
    context = {"source_seed": 11, "target_seed": experts[1].recipe.seed, "requested_Ks": [200, 5000],
               "requested_to_actual_K": {"0": 0, "200": 200, "5000": 200}}
    programs = fit_calibration_prefix_programs(prefix, experts, roster,
        strategies=("fixed", "grouped_temperature"), inner_replay_recorder=recorder,
        inner_replay_context=context)
    directory = persist_prefix_inner_replay(prefix, programs, tmp_path / "prefix", recorder=recorder,
        context=context)
    record = json.loads((directory / "prefix_inner_replay.json").read_bytes())["record"]
    assert record["actual_K"] == 200
    assert record["requested_Ks"] == [200, 5000]
    assert record["requested_to_actual_K"]["5000"] == 200
    assert record["programs"]["grouped_temperature"]["status"] == "unavailable"
    assert (directory / "temperature_programs.json").is_file()
    assert (directory / "ensemble.json").is_file()
    result = replay_prefix_inner_replay(directory, prefix=prefix)
    assert result["status"] == "replayed"


@pytest.mark.parametrize("field", ["prefix_input", "requested_Ks", "source_checkpoint_roles"])
def test_prefix_inner_replay_rejects_manifest_authority_mutation(independent_case, tmp_path, field):
    experts, prefix, views = _prefix_case(independent_case)
    roster = EnsembleProgram.equal_weights("prefix", tuple(spec.expert_id for spec in experts))
    recorder = InnerReplayRecorder()
    context = {"requested_Ks": [200], "requested_to_actual_K": {"200": 200}}
    programs = fit_calibration_prefix_programs(prefix, experts, roster, strategies=("fixed",),
        inner_replay_recorder=recorder, inner_replay_context=context)
    directory = persist_prefix_inner_replay(prefix, programs, tmp_path / field, recorder=recorder,
        context=context)
    path = directory / "prefix_inner_replay.json"
    payload = json.loads(path.read_bytes())
    if field == "prefix_input":
        payload["record"][field]["target_contract"]["available_sample"] += 1
    elif field == "requested_Ks":
        payload["record"][field] = [201]
    else:
        payload["record"][field]["toeplitz"]["digest"] = "f" * 64
    payload["record_sha256"] = semantic_sha256(payload["record"])
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|identity|alias|source|authority|denominator"):
        replay_prefix_inner_replay(directory, prefix=prefix)


def test_prefix_inner_replay_rejects_temperature_package_mutation(independent_case, tmp_path):
    experts, prefix, _ = _prefix_case(independent_case)
    roster = EnsembleProgram.equal_weights("prefix", tuple(spec.expert_id for spec in experts))
    recorder = InnerReplayRecorder()
    programs = fit_calibration_prefix_programs(prefix, experts, roster, strategies=("fixed",),
        inner_replay_recorder=recorder)
    directory = persist_prefix_inner_replay(prefix, programs, tmp_path / "temperature", recorder=recorder)
    path = directory / "temperature_programs.json"
    payload = json.loads(path.read_bytes())
    payload["record"]["programs"]["fixed"]["temperature_calibration"]["strategy"] = "changed"
    payload["record_sha256"] = semantic_sha256(payload["record"])
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="temperature"):
        replay_prefix_inner_replay(directory, prefix=prefix)


def test_prefix_inner_replay_refuses_truthful_future_rows(independent_case, tmp_path):
    experts, prefix, views = _prefix_case(independent_case)
    roster = EnsembleProgram.equal_weights("prefix", tuple(spec.expert_id for spec in experts))
    recorder = InnerReplayRecorder()
    programs = fit_calibration_prefix_programs(prefix, experts, roster, strategies=("fixed",),
        inner_replay_recorder=recorder)
    directory = persist_prefix_inner_replay(prefix, programs, tmp_path / "truth", recorder=recorder)
    with pytest.raises(ValueError, match="truth|target"):
        replay_prefix_inner_replay(directory, prefix=prefix, future_views=({"decision_id": "x", "truth": "a"},))


@pytest.fixture(scope="module")
def grouped_case(independent_case, tmp_path_factory):
    _, _, path, authorization, experts, views = independent_case
    prepared = prepare_bnci_independent_prefixes(path, source_sha256=views.ledger["source_sha256"],
        budgets=(400,), profiles=(experts[1].profile,), authorization_preprocessing=authorization)
    prefix = prepared.calibration[400]
    specs = (experts[1],)
    roster = EnsembleProgram.equal_weights("grouped-prefix", (specs[0].expert_id,))
    context = {"requested_Ks": [400], "requested_to_actual_K": {"400": 400},
        "test_scope": prepared.ledger["test_scope"],
        "requested_decision_ids": [view.record_id for view in prepared.decisions]}
    recorder = InnerReplayRecorder()
    with cpu_thread_budget(1):
        programs = fit_calibration_prefix_programs(prefix, specs, roster,
            strategies=("fixed", "grouped_temperature"), inner_replay_recorder=recorder, inner_replay_context=context)
        for strategy, model in programs.items():
            for view in prepared.decisions:
                recorder.record_prefix_future_score(strategy, view=view, evidence=model.score_experts(view))
        directory = persist_prefix_inner_replay(prefix, programs, tmp_path_factory.mktemp("grouped") / "package",
            recorder=recorder, context=context)
    return prefix, prepared, specs, roster, programs, recorder, context, directory


@pytest.mark.parametrize("mutation", ["expert", "ensemble", "alias", "raw_cutoff", "available",
    "label_use", "source_seed", "target_seed", "source_digest", "inner_roster", "held_row", "oof_rows", "path"])
def test_prefix_relationship_rejects_resigned_manifest_attacks(grouped_case, mutation):
    prefix, _, _, _, _, _, context, directory = grouped_case
    path = directory / "prefix_inner_replay.json"
    before = path.read_bytes()
    payload = json.loads(before)
    record = payload["record"]
    if mutation == "expert":
        record["expert_programs"][0]["noise_order"] += 1
    elif mutation == "ensemble":
        record["ensemble_program"]["name"] = "foreign-name"
    elif mutation == "alias":
        record["requested_Ks"] = record["requested_denominator"]["requested_Ks"] = [9999]
        record["requested_to_actual_K"] = {"9999": 400}
    elif mutation == "raw_cutoff":
        record["prefix_input"]["target_contract"]["raw_scope"][1] += 1
    elif mutation == "available":
        record["prefix_input"]["target_contract"]["available_sample"] += 1
    elif mutation == "label_use":
        record["prefix_input"]["label_use"]["raw"]["used_labels"] -= 1
    elif mutation in {"source_seed", "target_seed"}:
        record[mutation] = 42
    elif mutation == "source_digest":
        record["source_checkpoint_roles"]["toeplitz"]["digest"] = "a" * 64
    elif mutation == "inner_roster":
        record["inner_producers"] = []
    elif mutation == "held_row":
        producer = record["inner_producers"][0]
        producer["oof_rows"][0]["held_record_id"] = producer["training_record_ids"][0]
    elif mutation == "oof_rows":
        for producer in record["inner_producers"]:
            producer["uses"], producer["oof_rows"] = [], []
    else:
        record["inner_producers"][0]["package"] = "../outside-model"
    payload["record_sha256"] = semantic_sha256(record)
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            replay_prefix_inner_replay(directory, prefix=prefix, context=context)
    finally:
        path.write_bytes(before)


def test_grouped_prefix_captures_actual_misses_and_replays_every_future_row(grouped_case):
    prefix, prepared, _, _, _, recorder, context, directory = grouped_case
    assert len(recorder.producers) == len(prefix.complete_groups) == 3
    assert all(not use["cache_hit"] for rows in recorder.uses.values() for use in rows)
    replay = replay_prefix_inner_replay(directory, prefix=prefix, future_views=iter(prepared.decisions), context=context,
        expected_record_sha256=recorder.prefix_receipt_sha256)
    assert len(replay["scores"]) == len(prepared.decisions)
    assert replay["exact_score_checks"] == 3 + 2 * len(prepared.decisions)
    assert all(set(row["strategies"]) == {"fixed", "grouped_temperature"} for row in replay["scores"])


def test_prefix_input_label_and_sample_mutations_are_independently_rejected(grouped_case):
    prefix, _, _, _, _, _, context, directory = grouped_case
    # Strip complete groups so the ordinary prefix constructor admits a raw
    # mutation; replay must still bind the actual samples and label bytes.
    for raw in (replace(prefix.raw, samples=np.asarray(prefix.raw.samples) * 1.01),
                replace(prefix.raw, target_mask=~prefix.raw.target_mask)):
        changed = replace(prefix, raw=raw, complete_groups=(), trial_datasets={})
        with pytest.raises(ValueError):
            replay_prefix_inner_replay(directory, prefix=changed, context=context)


def test_direct_grouped_prefix_recorder_uses_same_fitter(independent_case, tmp_path):
    _, _, path, authorization, experts, views = independent_case
    prefix = prepare_bnci_independent_prefixes(path, source_sha256=views.ledger["source_sha256"],
        budgets=(400,), profiles=(experts[1].profile,), authorization_preprocessing=authorization).calibration[400]
    roster = EnsembleProgram.equal_weights("direct-grouped", (experts[1].expert_id,))
    recorder = InnerReplayRecorder()
    fitted = fit_calibration_prefix(prefix, (experts[1],), roster, calibration_strategy="grouped_temperature",
                                   inner_replay_recorder=recorder)
    directory = persist_prefix_inner_replay(prefix, {"grouped_temperature": fitted}, tmp_path / "direct", recorder=recorder)
    assert replay_prefix_inner_replay(directory, prefix=prefix)["exact_score_checks"] == 3


def test_grouped_inner_failure_preserves_fixed_and_actual_failed_scope(independent_case, tmp_path, monkeypatch):
    import transfer.ensemble_fit as owner

    _, _, path, authorization, experts, views = independent_case
    prefix = prepare_bnci_independent_prefixes(path, source_sha256=views.ledger["source_sha256"],
        budgets=(400,), profiles=(experts[1].profile,), authorization_preprocessing=authorization).calibration[400]
    roster = EnsembleProgram.equal_weights("failed-inner", (experts[1].expert_id,))
    error = ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE, stage="inner", detail="synthetic rank failure")

    def fail_inner(*args, **kwargs):
        raise error

    monkeypatch.setattr(owner, "fit_calibration_ensemble", fail_inner)
    recorder = InnerReplayRecorder()
    programs = fit_calibration_prefix_programs(prefix, (experts[1],), roster,
        strategies=("fixed", "grouped_temperature"), inner_replay_recorder=recorder)
    directory = persist_prefix_inner_replay(prefix, programs, tmp_path / "failed", recorder=recorder)
    record = json.loads((directory / "prefix_inner_replay.json").read_bytes())["record"]
    assert record["programs"]["grouped_temperature"]["status"] == "unavailable"
    assert len(record["inner_producers"]) == 1
    assert record["inner_producers"][0]["status"] == "unavailable"
    assert replay_prefix_inner_replay(directory, prefix=prefix)["producer_count"] == 1


def test_grouped_raised_oof_score_preserves_fixed_and_actual_partial_producer(independent_case, tmp_path, monkeypatch):
    from transfer.ensemble_fit import FittedEnsemble

    _, _, path, authorization, experts, views = independent_case
    prefix = prepare_bnci_independent_prefixes(path, source_sha256=views.ledger["source_sha256"],
        budgets=(400,), profiles=(experts[1].profile,), authorization_preprocessing=authorization).calibration[400]
    roster = EnsembleProgram.equal_weights("raised-inner-score", (experts[1].expert_id,))
    original = FittedEnsemble.score_experts

    def raised_inner(self, view):
        if self._target_contract is None:
            raise ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE, stage="inner_score", detail="actual raised score")
        return original(self, view)

    monkeypatch.setattr(FittedEnsemble, "score_experts", raised_inner)
    recorder = InnerReplayRecorder()
    programs = fit_calibration_prefix_programs(prefix, (experts[1],), roster,
        strategies=("fixed", "grouped_temperature"), inner_replay_recorder=recorder)
    directory = persist_prefix_inner_replay(prefix, programs, tmp_path / "raised", recorder=recorder)
    replay = replay_prefix_inner_replay(directory, prefix=prefix)
    assert programs["fixed"].fit_available
    assert isinstance(programs["grouped_temperature"], ExpectedSubjectError)
    assert replay["producer_count"] == 1 and replay["exact_score_checks"] == 0
    assert len(replay["preserved_oof_score_failures"]) == 1
    assert replay["preserved_oof_score_failures"][0]["failure"]["stage"] == "inner_score"


@pytest.mark.parametrize("axis", ["matrix", "channel_order", "reference", "sample_rate", "source_sha"])
def test_spatial_record_json_equivalence_keeps_real_changes_visible(independent_case, axis):
    spec = _raw_spatial_spec(independent_case)
    original = spec.record()
    exported = json.loads(json.dumps(original))
    assert original != exported  # Frozen nested tuples differ in native Python equality.
    assert semantic_sha256(original) == semantic_sha256(exported)
    if axis == "matrix":
        exported["spatial"]["matrix"][0][0] += .01
    elif axis == "channel_order":
        exported["spatial"]["slot_channels"].reverse()
    elif axis == "reference":
        exported["spatial"]["reference_channel"] = "Fz"
    elif axis == "sample_rate":
        exported["spatial"]["native_sample_rate_hz"] = 128.
    else:
        exported["spatial"]["source_sha256"] = "f" * 64
    assert semantic_sha256(original) != semantic_sha256(exported)


def test_actual_independent_prefix_spatial_caller_admits_json_roundtrip(independent_case, tmp_path):
    from experiments.run_candidate_ensemble_study import run_study
    from transfer.ensemble_fit import FittedEnsemble

    _, _, path, authorization, _, views = independent_case
    spec = _raw_spatial_spec(independent_case)
    mixture = EnsembleProgram.equal_weights("raw6", (spec.expert_id,))
    plan = {"schema": "n2p3_candidate_ensemble_study/1", "evidence_scope": "software_correctness",
        "evaluation_design": "independent_prefix", "seed_pairs": [[11, 21]],
        "records": [{"path": str(path), "sha256": views.ledger["source_sha256"]}],
        "budgets": [200], "experts": [spec.record()], "mixtures": [mixture.record()], "baseline_mixture": mixture.name,
        "authorization_preprocessing": authorization.record(), "observation_response_seconds": 308 / 256,
        "pre_roll_seconds": .2, "device": "cpu", "workers": 1, "cpu_threads_per_worker": 1,
        "source_checkpoints": {}, "source_seed_catalog": [], "calibration_strategies": ["fixed"]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    report = run_study(plan_path, tmp_path / "run")
    assert report["status"] == "completed" and len(report["participants"][0]["predictions"]) == len(views.decisions)
    assert any(row["evidence"]["log_probability"] is not None for row in report["participants"][0]["predictions"])
    package = next((tmp_path / "run").rglob("ensemble.json")).parent
    model = FittedEnsemble.load_programs(package)["fixed"]._experts[0].predictor
    receipt = model.provenance["raw_spatial_execution"][0]
    assert receipt["rank"] == 6 and receipt["native_sample_rate_hz"] == 256.
    assert tuple(receipt["native_channels"]) == tuple(views.calibration[200].target_contract["channel_names"])
    assert model.channels == ("Fz", "P3", "Pz", "P4", "PO8", "Oz")
