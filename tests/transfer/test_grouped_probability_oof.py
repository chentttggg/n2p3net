"""Actual synthetic C-only fits and immutable OOF/final-program persistence."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from data.preprocess import RawSampleScope
from models.candidate_evidence import CandidateEvidence
from research.contracts import semantic_sha256
from tests.support.checkpoints import write_checkpoint
from tests.transfer.test_ensemble_fit import programs, selections
from train.runtime import cpu_thread_budget
from transfer.calibration_state import FrozenTrialPredictor, _numeric_state_digest
from transfer.continuous_decision import CalibrationRecord
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import (
    FittedEnsemble,
    fit_calibration_ensemble,
    fit_calibration_prefix_programs,
)
from transfer.ensemble_views import CalibrationPrefix, build_calibration_epoch_dataset


@pytest.fixture(autouse=True)
def one_native_thread():
    with cpu_thread_budget(1):
        yield


@pytest.fixture(scope="module")
def captured_case():
    import transfer.ensemble_fit as owner

    groups = selections()
    specs = (replace(programs()[0], noise_order=0), programs()[1])
    program = EnsembleProgram.equal_weights("actual-oof", tuple(spec.expert_id for spec in specs))
    real_fit, real_attach = owner.fit_calibration_ensemble, owner._attach_oof_probability
    inner_fits, all_c = [], []

    def observe_fit(calibration, *args, **kwargs):
        calibration = tuple(calibration)
        result = real_fit(calibration, *args, **kwargs)
        inner_fits.append((calibration, result))
        return result

    def observe_attach(fitted, temperature, record):
        all_c.append(fitted)
        return real_attach(fitted, temperature, record)

    with pytest.MonkeyPatch.context() as patch, cpu_thread_budget(1):
        patch.setattr(owner, "fit_calibration_ensemble", observe_fit)
        patch.setattr(owner, "_attach_oof_probability", observe_attach)
        fitted = real_fit(groups[:3], specs, program, calibration_strategy="grouped_temperature")
    assert len(inner_fits) == 3 and len(all_c) == 1
    return groups, specs, program, fitted, inner_fits, all_c[0]


def test_real_oof_producers_match_saved_probabilities_and_fit_scopes(captured_case):
    groups, specs, program, fitted, inner_fits, baseline = captured_case
    record = fitted.oof_probability_record
    assert record["actual_K"] == record["complete_group_labels"] == 12
    assert record["selection_performed"] is False
    assert record["captured_expert_ids"] == program.expert_ids
    for index, ((training, producer), row) in enumerate(zip(inner_fits, record["rows"], strict=True)):
        held = groups[index]
        assert tuple(group.view.record_id for group in training) == row["training_record_ids"]
        assert held.view.record_id not in row["training_record_ids"]
        assert groups[3].view.record_id not in row["training_record_ids"]
        assert row["training_receipts_sha256"] == semantic_sha256(producer.fit_receipts)
        actual = producer.score_experts(held.view)
        for spec in specs:
            persisted = CandidateEvidence.from_record(row["expert_evidence"][spec.expert_id])
            assert persisted.record() == actual[spec.expert_id].record()
            np.testing.assert_array_equal(persisted.log_probability, actual[spec.expert_id].log_probability)
    assert [entry["model_digest"] for entry in record["final_base_experts"]] == [
        expert.model_digest for expert in baseline._experts]
    assert record["final_training_receipts_sha256"] == semantic_sha256(baseline.fit_receipts)
    assert baseline.oof_probability_record is None
    with pytest.raises(TypeError):
        record["rows"][0]["expert_evidence"][specs[0].expert_id]["log_probability"][0] = 0.
    with pytest.raises(AttributeError):
        fitted.oof_probability_record = None


def test_actual_oof_and_predictions_replay_in_existing_ensemble_package(captured_case, tmp_path):
    groups, _, _, fitted, _, _ = captured_case
    directory = fitted.save(tmp_path / "package")
    restored = FittedEnsemble.load(directory)
    assert restored.oof_probability_record == fitted.oof_probability_record
    assert restored.temperature_record == fitted.temperature_record
    assert restored.score(groups[3].view).record() == fitted.score(groups[3].view).record()
    assert set(path.name for path in directory.iterdir()) == {"ensemble.json", "expert_000.npz", "expert_001.pt"}


@pytest.mark.parametrize("field", ["probabilities", "actual_K", "native", "source", "programme", "temperature"])
def test_resigned_oof_edits_cannot_replace_actual_model_binding(captured_case, tmp_path, field):
    _, specs, _, fitted, _, _ = captured_case
    directory = fitted.save(tmp_path / "package")
    manifest = directory / "ensemble.json"
    payload = json.loads(manifest.read_bytes())
    oof = payload["record"]["OOF_probability"]
    if field == "probabilities":
        oof["rows"][0]["expert_evidence"][specs[0].expert_id]["log_probability"] = np.log([.5, .5]).tolist()
    elif field == "actual_K":
        oof["actual_K"] += 4
    elif field == "native":
        oof["native_observation"]["reference"] = "left earlobe"
    elif field == "source":
        oof["source_checkpoint_sha256"] = {specs[0].expert_id: "f"*64}
    elif field == "programme":
        oof["final_program"]["weights"] = [.6, .4]
    else:
        name = specs[0].expert_id
        oof["temperature_calibration"]["temperatures"][name] *= 1.01
        payload["record"]["temperature_calibration"] = oof["temperature_calibration"]
    payload["record_sha256"] = semantic_sha256(payload["record"])
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="OOF|recipe and fit authority"):
        FittedEnsemble.load(directory)


def synthetic_prefix(groups, *, partial_label=False):
    shifted = []
    cutoff = 600*(len(groups)-1)+551
    complete_stop = cutoff
    if partial_label:
        cutoff += 64
    samples = np.zeros((cutoff, 3))
    onsets, labels = [], []
    for index, group in enumerate(groups):
        scope = (600*index, 600*index+551)
        view = replace(group.view, raw_scope=scope, available_sample=scope[1])
        shifted.append(replace(group, view=view))
        samples[slice(*scope)] = view.samples
        onsets.extend(view.onsets+scope[0])
        labels.extend(group.labels)
    if partial_label:
        onsets.append(complete_stop)
        labels.append(False)
        samples[complete_stop:] = np.random.default_rng(848).normal(0, 1e-6, (64, 3))
    first = shifted[0].view
    budget = len(labels)
    target = {"participant_key": first.participant_key, "candidate_vocabulary": first.decision.candidate_vocabulary,
        "source_sha256": first.source_sha256, "source_dataset_id": first.source_dataset_id,
        "source_subject_id": first.source_subject_id, "raw_scope": (0, cutoff),
        "sample_rate_hz": first.sample_rate_hz, "channel_names": first.channel_names,
        "channel_types": first.channel_types, "reference": first.reference, "signal_unit": first.signal_unit}
    raw = CalibrationRecord(f"{first.participant_key}:C{budget}:{first.source_sha256}", samples,
        np.asarray(onsets, dtype=np.int64), np.asarray(labels, dtype=bool), right_censored=True)
    use = {"raw": {"requested_labels": budget, "used_labels": budget,
        "read_sample_support": (0, cutoff), "available_sample": cutoff, "right_censored": True,
        "next_unknown_flash_sample": cutoff,
        "authorization_scope": RawSampleScope("synthetic_C", 0, cutoff, tuple(range(budget))).record()}, "trial": {}}
    return CalibrationPrefix(budget, raw, {}, tuple(shifted), target, use)


def test_prefix_k_shared_fixed_grouped_package_and_temperature_binding(tmp_path):
    groups = selections()
    prefix = synthetic_prefix(groups[:3], partial_label=True)
    spec = replace(programs()[0], noise_order=0)
    programme = EnsembleProgram.equal_weights("prefix-OOF", (spec.expert_id,))
    fitted = fit_calibration_prefix_programs(prefix, (spec,), programme, strategies=("fixed", "grouped_temperature"))
    fixed, grouped = fitted["fixed"], fitted["grouped_temperature"]
    assert fixed._experts is grouped._experts
    assert fixed.oof_probability_record == grouped.oof_probability_record
    assert grouped.oof_probability_record["actual_K"] == prefix.budget == 13
    assert grouped.oof_probability_record["complete_group_labels"] == 12
    assert grouped.oof_probability_record["fit_authority_sha256"] == semantic_sha256(prefix.record())
    directory = FittedEnsemble.save_programs(tmp_path / "prefix", fitted, participant_key=fixed.participant_key)
    restored = FittedEnsemble.load_programs(directory)
    assert restored["fixed"]._experts is restored["grouped_temperature"]._experts
    assert restored["fixed"].oof_probability_record == grouped.oof_probability_record
    for name in fitted:
        assert restored[name].score(groups[3].view).record() == fitted[name].score(groups[3].view).record()
    manifest = directory / "temperature_programs.json"
    payload = json.loads(manifest.read_bytes())
    payload["record"]["programs"]["grouped_temperature"]["temperature_calibration"]["temperatures"][spec.expert_id] *= 1.01
    payload["record_sha256"] = semantic_sha256(payload["record"])
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="actual temperature programme"):
        FittedEnsemble.load_programs(directory)


def test_source_role_binding_and_changed_actual_final_numbers_rejected(tmp_path):
    groups = selections()
    scratch = programs()[-1]
    source = replace(scratch, expert_id="source", recipe=replace(scratch.recipe, name="source", initialization="source"))
    data = build_calibration_epoch_dataset(groups[:3], source.profile)
    checkpoint = write_checkpoint(tmp_path / "source.pt", data, source_snapshot_sha256="c"*64)
    programme = EnsembleProgram.equal_weights("source-OOF", (source.expert_id,))
    fitted = fit_calibration_ensemble(groups[:3], (source,), programme, source_checkpoints={source.expert_id: checkpoint},
                                    calibration_strategy="grouped_temperature")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    record = fitted.oof_probability_record
    assert record["source_checkpoint_sha256"] == {source.expert_id: checkpoint_sha}
    for row in record["rows"]:
        assert row["expert_evidence"][source.expert_id]["expert_identities"][0]["source_digest"] == checkpoint_sha
    directory = fitted.save(tmp_path / "package")
    model = directory / "expert_000.pt"
    model_payload = torch.load(model, weights_only=True)
    state = model_payload["state"]
    key = next(key for key, tensor in state.items() if key.startswith("trunk.") and tensor.dtype.is_floating_point)
    state[key].view(-1)[0] += .125
    model_payload["record"]["class_calibration"]["fit_contract"]["frozen_state_digest"] = _numeric_state_digest(state)
    changed = FrozenTrialPredictor(model_payload["record"], state)
    torch.save(model_payload, model)
    manifest = directory / "ensemble.json"
    payload = json.loads(manifest.read_bytes())
    payload["record"]["experts"][0]["model_file_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
    payload["record"]["experts"][0]["model_digest"] = changed.digest()
    payload["record_sha256"] = semantic_sha256(payload["record"])
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="actual final numerical state"):
        FittedEnsemble.load(directory)


def test_prefix_oof_subset_preserves_unavailable_fixed_control_and_requested_roster(tmp_path):
    groups = selections()
    prefix = synthetic_prefix(groups[:3], partial_label=True)
    raw, trial = replace(programs()[0], noise_order=0), programs()[1]
    programme = EnsembleProgram.equal_weights("subset-OOF", (raw.expert_id, trial.expert_id))
    result = fit_calibration_prefix_programs(prefix, (raw, trial), programme,
        strategies=("fixed", "grouped_temperature"), temperature_expert_ids=(raw.expert_id,))
    fixed, grouped = result["fixed"], result["grouped_temperature"]
    assert fixed._experts is grouped._experts
    assert not fixed.fit_available and not grouped.fit_available
    assert grouped.oof_probability_record["captured_expert_ids"] == (raw.expert_id,)
    assert len(grouped.oof_probability_record["final_base_experts"]) == 2
    assert grouped.temperature_record["fixed_control_expert_ids"] == (trial.expert_id,)
    assert grouped.temperature_record["temperatures"][trial.expert_id] == 1.
    directory = FittedEnsemble.save_programs(tmp_path / "subset", result, participant_key=fixed.participant_key)
    restored = FittedEnsemble.load_programs(directory)
    for name, fitted in result.items():
        assert restored[name].oof_probability_record == fitted.oof_probability_record
        raw_evidence = restored[name].score_experts(groups[3].view)
        assert raw_evidence[raw.expert_id].probability_available
        assert not raw_evidence[trial.expert_id].probability_available
        pooled = restored[name].score(groups[3].view)
        assert not pooled.probability_available
        assert len(pooled.expert_identities) == 2
