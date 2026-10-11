"""Frozen predictors must replay the real fitted models, including input rounding."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from research.contracts import semantic_sha256
from tests.support.datasets import binary_dataset
from transfer.calibration_candidates import fit_calibration_candidate, waveform_lda_recipe
from transfer.calibration_state import FrozenTrialPredictor, _numeric_state_digest
from transfer.recipes import TargetRecipe, erp_target_recipe


@pytest.mark.parametrize("family", ["waveform_lda", "toeplitz_lda", "xdawn_erp", "scratch"])
def test_frozen_numeric_predictor_matches_actual_fit_and_reload(family, tmp_path):
    data = binary_dataset(count=40)
    if family == "toeplitz_lda":
        data = replace(data, preprocessing=replace(data.preprocessing, baseline_mode="none"))
    if family == "waveform_lda":
        recipe = waveform_lda_recipe()
    elif family == "scratch":
        recipe = TargetRecipe("scratch", initialization="scratch", epochs=1, batch_size=32,
                              bn_buffers="calibration", dropout="off")
    else:
        recipe = erp_target_recipe(family, bin_width=8)
    fitted = fit_calibration_candidate(data, recipe, np.ones(data.n_epochs, dtype=bool))
    mask = np.broadcast_to(data.channel_mask, data.X.shape[:2])
    expected = fitted.predict_logit(data.X, mask)
    frozen = FrozenTrialPredictor.from_fitted(fitted)
    path = tmp_path / "model.pt"
    frozen.save(path)
    restored = FrozenTrialPredictor.load(path)
    np.testing.assert_allclose(frozen.predict_logit(data.X, mask), expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(restored.predict_logit(data.X, mask), frozen.predict_logit(data.X, mask))
    # Ingress rejects actual corruption before any inference occurs.
    saved = torch.load(path, weights_only=True)
    key = "input_std" if family == "scratch" else "erp.coefficient" if fitted.erp else "feature_scale"
    saved["state"][key].fill_(float("nan"))
    torch.save(saved, path)
    with pytest.raises(ValueError, match="nonfinite"):
        FrozenTrialPredictor.load(path)


def test_fit_binding_accepts_json_array_freeze_and_rejects_real_epoch_change():
    """Tuple-frozen metadata must retain the same source input contract."""
    from transfer.ensemble_views import SelectionEpochProfile

    data = binary_dataset(count=40)
    recipe = TargetRecipe("source_fixture", initialization="scratch", epochs=1, batch_size=32,
                          bn_buffers="calibration", dropout="off")
    fitted = fit_calibration_candidate(data, recipe, np.ones(data.n_epochs, dtype=bool))
    predictor = FrozenTrialPredictor.from_fitted(fitted)
    profile_input = replace(data.preprocessing, filter_phase="zero", zero_phase_edge_policy="mne_reflect_limited",
        causal_iir_initial_state="not_applicable")
    profile = SelectionEpochProfile.from_preprocessing(profile_input, source_sample_rate_hz=profile_input.sfreq,
        channel_names=data.channel_names, reference="Cz")
    record = dict(predictor.record)
    source_contract = {"preprocessing": {"epoch": profile.preprocessing.record(),
        "channel_names": list(profile.channel_names), "source_reference": "measured_channel:Cz"},
        "optimizer": {"config": {"pos_weight": 1.}}}
    record["selection_epoch_capability"] = profile.execution_capability()
    record["source_calibration"] = {"pos_weight": 1., "train_prior": .5, "temperature": 1.}
    record["class_calibration"] = {"origin": "source_mixture", "pos_weight": 1., "prior": .5,
        "temperature": 1., "raw_score_kind": "source_weighted_CE_logit",
        "fit_contract": {"training_contract": source_contract,
                          "training_contract_digest": semantic_sha256(source_contract),
                          "frozen_state_digest": _numeric_state_digest(fitted.state)}}
    restored = FrozenTrialPredictor(record, fitted.state)
    restored.with_fit_binding({"scope": "fixture"})
    changed = copy.deepcopy(record)
    changed["class_calibration"]["fit_contract"]["training_contract"]["preprocessing"]["epoch"]["h_freq"] = 29.0
    changed["class_calibration"]["fit_contract"]["training_contract_digest"] = semantic_sha256(
        changed["class_calibration"]["fit_contract"]["training_contract"])
    with pytest.raises(ValueError, match="source inference capability"):
        FrozenTrialPredictor(changed, fitted.state)
