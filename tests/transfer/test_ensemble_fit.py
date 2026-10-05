"""Actual expert fitting, shared physical input and frozen label-free replay."""

from dataclasses import replace

import numpy as np
import pytest

from data.candidate_task import MembershipDecision
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from data.identity import origin_subject_key
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram, FittedEnsemble, fit_calibration_ensemble
from transfer.ensemble_views import (
    CalibrationSelection,
    PredictionDecisionView,
    SelectionEpochProfile,
)
from transfer.recipes import TargetRecipe, erp_target_recipe


def selections():
    groups = []
    for index in range(4):
        decision = MembershipDecision.direct(("a", "b", "b", "a"), ("a", "b"),
            event_ids=tuple(f"group{index}/flash{row}" for row in range(4)),
            repetition_indices=np.array([0, 0, 1, 1]), repetitions=2)
        onsets = np.array([51, 115, 179, 243])
        samples = np.random.default_rng(index + 70).normal(0, 1e-6, (551, 3))
        labels = np.array([True, False, False, True])
        response = np.exp(-((np.arange(307) / 256 - .32) / .08) ** 2)
        for onset, target in zip(onsets, labels, strict=True):
            samples[onset:onset + len(response), 2] += target * response * 2e-6
        view = PredictionDecisionView(record_id=f"record{index}", participant_key=origin_subject_key("fixture", "one"),
            decision_id=f"selection{index}", samples=samples, onsets=onsets, decision=decision,
            source_sha256="a" * 64, raw_scope=(1000 + 600 * index, 1551 + 600 * index),
            sample_rate_hz=256., channel_names=("Fz", "Cz", "Pz"), channel_types=("EEG",) * 3,
            reference="right earlobe", source_relative_path="software-fixture.mat",
            source_dataset_id="fixture", source_subject_id="one", available_sample=1551 + 600 * index)
        groups.append(CalibrationSelection(view, labels, "a"))
    return groups


def programs():
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    none = SelectionEpochProfile.from_preprocessing(replace(spec, baseline_mode="none"),
        source_sample_rate_hz=256., channel_names=("Fz", "Cz", "Pz"), reference="right earlobe")
    baseline = SelectionEpochProfile.from_preprocessing(spec, source_sample_rate_hz=256.,
        channel_names=("Fz", "Cz", "Pz"), reference="right earlobe")
    return (CalibrationExpertProgram("raw_ar8", "continuous_ar", noise_order=8, n_basis=2),
            CalibrationExpertProgram("toeplitz", "trial", erp_target_recipe("toeplitz_lda", bin_width=8), none),
            CalibrationExpertProgram("xdawn", "trial", erp_target_recipe("xdawn_erp", bin_width=8), baseline),
            CalibrationExpertProgram("scratch", "trial", TargetRecipe("scratch", initialization="scratch",
                epochs=1, batch_size=32, bn_buffers="calibration", dropout="off"), baseline))


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
