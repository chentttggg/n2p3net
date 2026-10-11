"""Actual expert fitting, shared physical input and frozen label-free replay."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from data.identity import origin_subject_key
from research.contracts import semantic_sha256
from tests.support.checkpoints import write_checkpoint
from tests.support.ensemble_fixtures import programs, selections
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram, FittedEnsemble, fit_calibration_ensemble
from transfer.ensemble_views import (
    CalibrationSelection,
    build_calibration_epoch_dataset,
)
from transfer.kernel_prior import KernelPriorSpec


def test_fixed_prefix_raw_ensemble_preserves_program_support_and_exact_replay(tmp_path):
    groups = selections()
    spec = replace(programs()[0], conditioning_samples=32)
    mixture = EnsembleProgram.equal_weights("conditional", (spec.expert_id,))
    fitted = fit_calibration_ensemble(groups[:3], (spec,), mixture)
    before = fitted.score(groups[3].view)
    assert before.expert_identities[0].score_kind == "continuous_conditional_tail_log_density_fixed_candidate_prior/1"
    assert before.expert_identities[0].numerical_input_digest != groups[3].view.raw_input_digest
    package = fitted.save(tmp_path / "conditional")
    restored = FittedEnsemble.load(package)
    after = restored.score(groups[3].view)
    np.testing.assert_array_equal(before.log_probability, after.log_probability)
    assert before.record() == after.record()


@pytest.mark.parametrize("prior_factory", [KernelPriorSpec.calibration_g, KernelPriorSpec.calibration_trace_isotropic])
def test_calibration_g_program_saved_package_and_wrong_prior_relation(tmp_path, monkeypatch, prior_factory):
    import transfer.continuous_decision as continuous
    from transfer.ensemble_fit import _FittedExpert

    compiled = []
    original_compiler = continuous.compile_kernel_prior

    def capture(*args, **kwargs):
        result = original_compiler(*args, **kwargs)
        compiled.append(result)
        return result

    monkeypatch.setattr(continuous, "compile_kernel_prior", capture)

    groups = selections()
    spec = CalibrationExpertProgram("raw_g", "continuous_ar", noise_order=1, n_basis=2,
        conditioning_samples=32, kernel_prior=prior_factory(9.))
    assert "ridge" not in spec.record()
    assert CalibrationExpertProgram.from_record(spec.record()).prior_spec == spec.prior_spec
    mixture = EnsembleProgram.equal_weights("g", (spec.expert_id,))
    fitted = fit_calibration_ensemble(groups[:3], (spec,), mixture, calibration_strategy="grouped_temperature")
    before = fitted.score(groups[3].view)
    path = fitted.save(tmp_path/"g")
    assert FittedEnsemble.load(path).score(groups[3].view).record() == before.record()
    actual = fitted._experts[0]
    changed = replace(spec, kernel_prior=prior_factory(8.))
    with pytest.raises(ValueError, match="actual numerical model"):
        _FittedExpert(changed, actual.predictor, actual.model_digest, actual.fit_digest, None, 0.)
    assert actual.predictor.kernel_prior.rank_ledger["observed_rows"] == 3*551
    assert fitted.temperature_record["final_experts_refitted_on_all_authorized_groups"] is True
    assert len(compiled) == 4
    for prior, row in zip(compiled[1:], fitted.temperature_record["OOF_groups"], strict=True):
        assert prior.rank_ledger["observed_rows"] == 2*551
        assert row["held_record_id"] not in {r["record_id"] for r in prior.provenance["records"]}
        if prior.spec.kind == "calibration_trace_isotropic":
            assert prior.derived_precision < actual.predictor.kernel_prior.derived_precision


def test_continuous_g_and_explicit_ridge_programs_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually"):
        CalibrationExpertProgram("g", "continuous_ar", conditioning_samples=32,
            kernel_prior=KernelPriorSpec.calibration_g(9.), ridge=1.)


def test_actual_raw_support_rejects_a_relabelled_program_even_when_shell_binding_agrees():
    from transfer.ensemble_fit import _FittedExpert

    groups = selections()
    spec = replace(programs()[0], conditioning_samples=32)
    mixture = EnsembleProgram.equal_weights("conditional", (spec.expert_id,))
    fitted = fit_calibration_ensemble(groups[:3], (spec,), mixture)
    actual = fitted._experts[0]
    changed = replace(spec, conditioning_samples=31)
    with pytest.raises(ValueError, match="actual numerical model"):
        _FittedExpert(changed, actual.predictor, actual.model_digest, actual.fit_digest, actual.source_digest, 0.)


def test_grouped_temperature_keeps_conditional_program_through_inner_fits_and_reload(tmp_path):
    groups = selections()
    spec = replace(programs()[0], conditioning_samples=32)
    mixture = EnsembleProgram.equal_weights("conditional", (spec.expert_id,))
    fitted = fit_calibration_ensemble(groups[:3], (spec,), mixture, calibration_strategy="grouped_temperature")
    assert fitted.temperature_record["strategy"] == "grouped_temperature"
    before = fitted.score(groups[3].view)
    package = fitted.save(tmp_path / "conditional_oof")
    after = FittedEnsemble.load(package).score(groups[3].view)
    assert before.record() == after.record()
    assert all(expert.program.conditioning_samples == 32 for expert in fitted._experts)


@pytest.fixture(scope="module")
def frozen_raw_classic():
    groups = selections()
    specs = programs()[:2]
    program = EnsembleProgram.equal_weights("raw-and-toeplitz", tuple(spec.expert_id for spec in specs))
    return groups, specs, program, fit_calibration_ensemble(groups[:3], specs, program)


def test_unresolved_continuous_prediction_preserves_trial_peer_and_declared_roster(monkeypatch, frozen_raw_classic):
    from transfer.continuous_decision import ContinuousDecisionModel, KernelPosteriorResolutionError

    groups, specs, program, fitted = frozen_raw_classic
    view = groups[3].view
    before = fitted.score_experts(view)

    def unresolved(*args, **kwargs):
        raise KernelPosteriorResolutionError({"kernel_dimensions": 4, "unresolved_held_direction": True})

    monkeypatch.setattr(ContinuousDecisionModel, "score", unresolved)
    after = fitted.score_experts(view)
    assert tuple(after) == program.expert_ids
    raw = after[specs[0].expert_id]
    assert not raw.probability_available and raw.failure_kind == "not_available"
    reason = json.loads(raw.failure_reason)
    assert reason["stage"] == "read_only_prediction"
    assert reason["numerical_evidence"]["unresolved_held_direction"]
    assert raw.expected_event_counts == before[specs[0].expert_id].expected_event_counts
    assert after[specs[1].expert_id].record() == before[specs[1].expert_id].record()
    mixture = fitted.scorer.score(after)
    assert not mixture.probability_available
    assert {identity.expert_id for identity in mixture.expert_identities} == set(program.expert_ids)

    def invalid_contract(*args, **kwargs):
        raise ValueError("actual artifact/input contract differs")

    monkeypatch.setattr(ContinuousDecisionModel, "score", invalid_contract)
    with pytest.raises(ValueError, match="artifact/input contract"):
        fitted.score_experts(view)


def test_incomplete_physical_groups_retain_outcomes_after_contract_validation(frozen_raw_classic):
    groups, _, _, fitted = frozen_raw_classic
    view = groups[3].view
    missing = view.decision.evidence_indices.copy()
    missing[0] = -1
    for decision in (replace(view.decision, timeline_complete=False), replace(view.decision, evidence_indices=missing)):
        output = fitted.score_experts(replace(view, decision=decision))
        assert all(value.failure_kind == "incomplete" and not value.probability_available for value in output.values())
    with pytest.raises(ValueError, match="reference"):
        fitted.score(replace(view, decision=replace(view.decision, evidence_indices=missing), reference="left earlobe"))
    reordered = replace(view.decision, candidate_vocabulary=tuple(reversed(view.decision.candidate_vocabulary)),
                         membership=view.decision.membership[:, ::-1])
    with pytest.raises(ValueError, match="vocabulary"):
        fitted.score(replace(view, decision=reordered))


def test_raw_calibration_rejects_mixed_native_reference(frozen_raw_classic):
    groups, specs, program, _ = frozen_raw_classic
    other = replace(groups[2], view=replace(groups[2].view, reference="left earlobe"))
    with pytest.raises(ValueError, match="layout/reference"):
        fit_calibration_ensemble((groups[0], groups[1], other), specs, program)


@pytest.mark.parametrize("changed", ["raw_order", "trial_kind", "fit_participant"])
def test_model_bytes_bind_recipe_and_fit_authority(frozen_raw_classic, tmp_path, changed):
    _, _, _, fitted = frozen_raw_classic
    path = tmp_path / "package"
    fitted.save(path)
    manifest = path / "ensemble.json"
    payload = json.loads(manifest.read_bytes())
    if changed == "raw_order":
        payload["record"]["experts"][0]["program"]["noise_order"] = 16
    elif changed == "trial_kind":
        payload["record"]["experts"][1]["program"]["recipe"]["initialization"] = "scratch"
        payload["record"]["experts"][1]["program"]["recipe"].pop("erp_bin_width")
    else:
        payload["record"]["training_receipts"][0]["participant_key"] = origin_subject_key("fixture", "other")
    payload["record_sha256"] = semantic_sha256(payload["record"])
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="recipe and fit authority"):
        FittedEnsemble.load(path)


def test_frozen_programme_temperature_and_trial_state_cannot_be_mutated(frozen_raw_classic):
    _, _, _, fitted = frozen_raw_classic
    with pytest.raises(TypeError):
        fitted.temperature_record["temperatures"]["toeplitz"] = 100
    with pytest.raises(AttributeError):
        fitted.scorer.program = EnsembleProgram.equal_weights("other", ("toeplitz",))
    with pytest.raises(AttributeError):
        fitted.scorer = fitted.scorer
    with pytest.raises(AttributeError):
        fitted.participant_key = origin_subject_key("fixture", "other")
    trial = fitted._experts[1].predictor
    assert not hasattr(trial, "state")
    with pytest.raises(TypeError):
        trial.record["kind"] = "n2p3"


def test_real_four_expert_fit_scores_without_held_labels_and_reloads(tmp_path):
    groups = selections()
    specs = programs()
    program = EnsembleProgram.equal_weights("four-scratch", tuple(spec.expert_id for spec in specs))
    fitted = fit_calibration_ensemble(groups[:3], specs, program)
    held = groups[3].view
    expert = fitted.score_experts(held)
    mixed = fitted.score(held)
    assert set(expert) == set(program.expert_ids)
    assert mixed.probability_available
    np.testing.assert_allclose(mixed.probability, np.mean([value.probability for value in expert.values()], axis=0))
    assert len({value.physical_support_digest for value in expert.values()}) == 1
    assert len({identity.numerical_input_digest for value in expert.values()
                for identity in value.expert_identities}) >= 3
    path = tmp_path / "personal"
    fitted.save(path)
    restored = FittedEnsemble.load(path)
    np.testing.assert_array_equal(restored.score(held).log_probability, mixed.log_probability)
    changed_truth = CalibrationSelection(held, np.array([False, True, True, False]), "b")
    np.testing.assert_array_equal(restored.score(changed_truth.view).log_probability, mixed.log_probability)
    with pytest.raises(ValueError, match="calibration"):
        restored.score(groups[0].view)
    with pytest.raises(ValueError, match="another participant"):
        restored.score(replace(held, participant_key=origin_subject_key("fixture", "other"), source_subject_id="other"))
    model = path / "expert_000.npz"
    model.write_bytes(model.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="model bytes"):
        FittedEnsemble.load(path)


def test_temperature_uses_only_actual_grouped_oof_and_survives_save(tmp_path):
    groups = selections()
    specs = programs()[1:2]
    program = EnsembleProgram.equal_weights("toeplitz-temperature", ("toeplitz",))
    fitted = fit_calibration_ensemble(groups[:3], specs, program, calibration_strategy="grouped_temperature")
    record = fitted.temperature_record
    assert record["strategy"] == "grouped_temperature"
    assert len(record["OOF_groups"]) == 3
    for row in record["OOF_groups"]:
        assert row["held_record_id"] not in row["training_record_ids"]
        assert "record3" not in row["training_record_ids"]
    assert .05 <= record["temperatures"]["toeplitz"] <= 20
    prediction = fitted.score(groups[3].view)
    assert prediction.expert_identities[0].probability_calibrated
    path = tmp_path / "calibrated"
    fitted.save(path)
    np.testing.assert_array_equal(FittedEnsemble.load(path).score(groups[3].view).log_probability,
                                  prediction.log_probability)
    with pytest.raises(ValueError, match="three complete groups"):
        fit_calibration_ensemble(groups[:1], specs, program, calibration_strategy="grouped_temperature")


def test_source_admission_and_matched_scratch_never_inherit_source_values(tmp_path):
    groups = selections()
    scratch = programs()[-1]
    source = replace(scratch, expert_id="source", recipe=replace(scratch.recipe, name="source", initialization="source"))
    data = build_calibration_epoch_dataset(groups[:3], source.profile)
    checkpoint = write_checkpoint(tmp_path / "source.pt", data, source_snapshot_sha256="c" * 64)
    specs = (source, scratch)
    program = EnsembleProgram.equal_weights("source-and-scratch", ("source", "scratch"))
    fitted = fit_calibration_ensemble(groups[:3], specs, program,
        source_checkpoints={"source": checkpoint, "scratch": checkpoint})
    before = fitted.score_experts(groups[3].view)
    assert before["source"].expert_identities[0].source_digest is not None
    assert before["scratch"].expert_identities[0].source_digest is None
    payload = torch.load(checkpoint, weights_only=True)
    for value in payload["trunk_state_dict"].values():
        if value.dtype.is_floating_point:
            value.add_(1.)
    payload["input_mean"] = [99.] * data.n_channels
    altered = tmp_path / "altered.pt"
    torch.save(payload, altered)
    fresh = fit_calibration_ensemble(groups[:3], specs, program,
        source_checkpoints={"source": altered, "scratch": altered}).score_experts(groups[3].view)
    np.testing.assert_array_equal(fresh["scratch"].log_probability, before["scratch"].log_probability)
    assert not np.array_equal(fresh["source"].log_probability, before["source"].log_probability)
