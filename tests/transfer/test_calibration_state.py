"""Frozen predictors must replay the real fitted models, including input rounding."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from tests.support.datasets import binary_dataset
from transfer.calibration_candidates import fit_calibration_candidate, waveform_lda_recipe
from transfer.calibration_state import FrozenTrialPredictor
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
