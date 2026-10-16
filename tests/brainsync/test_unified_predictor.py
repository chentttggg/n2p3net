from __future__ import annotations

import json

import numpy as np
import pytest

from data.artifact import FoldLocalArtifactPolicy
from data.brainsync import load_brainsync_inference_session, load_brainsync_sessions
from experiments.brainsync_model_task import inference_record, main
from tests.support.brainsync import write_bids_session
from train.early_stopping import UnifiedTrainingBudget
from transfer.calibrated import (
    fit_calibrated_predictor,
    load_calibrated_predictor,
    preflight_calibrated_predictor,
    save_calibrated_predictor,
)
from transfer.recipes import TargetRecipe


@pytest.mark.parametrize("quality", [False, True], ids=["plain", "fold-local-quality"])
def test_unified_full_calibration_reloads_identical_masked_scores_and_membership(tmp_path, quality):
    calibration = write_bids_session(tmp_path / "cal", session_id="cal", target=3,
        started_utc="2026-09-14T10:00:00+08:00", blocks=(tuple(range(1, 10)) * 5,))
    testing = write_bids_session(tmp_path / "test", session_id="test", target=0,
        started_utc="2026-09-16T10:00:00+08:00", blocks=(tuple(range(1, 10)) * 5,))
    data = load_brainsync_sessions((calibration,))
    stopping = UnifiedTrainingBudget(target_epochs=2).record()
    recipe = TargetRecipe("unified_scratch", initialization="scratch", batch_size=16,
        bn_buffers="calibration", bn_affine="trainable", dropout="off", epochs=2,
        seed=20260906, candidate_loss_weight=1.0)
    quality_options = {"epoch_quality_policy": FoldLocalArtifactPolicy(min_clean_epochs=2)} if quality else {}
    plan = preflight_calibrated_predictor(data, selector_mode="unified", transfer_policy=recipe,
                                         unified_training_budget=stopping, **quality_options)
    assert plan["execution_ready"], plan["execution_blockers"]
    assert not plan["inner_selection_planned"] and not plan["fallback_planned"]
    predictor = fit_calibrated_predictor(data, selector_mode="unified", transfer_policy=recipe,
                                         unified_training_budget=stopping, **quality_options)
    assert predictor.metadata["selection"]["mode"] == "fixed_unified_program"
    assert predictor.metadata["selection"]["fit_execution"]["fit_calls"] == 1
    assert predictor.trunk.observed_channel_norm
    if quality:
        selection = predictor.metadata["selection"]
        assert "inner_history" not in selection
        assert selection["refit_history"]["epoch_quality"]["requested_training_rows"] == 45
        assert json.loads(json.dumps(predictor.epoch_quality.record())) == predictor.metadata["epoch_quality"]["model"]
    later = load_brainsync_inference_session(testing)
    before = predictor.predict_logit(later)
    restored = load_calibrated_predictor(save_calibrated_predictor(tmp_path / "unified.pt", predictor))
    np.testing.assert_array_equal(before, restored.predict_logit(later))
    result = inference_record(restored, later)
    assert result["score_kind"] == "membership_llr_sum/1"
    assert result["requested_repetitions"] == 5
    assert result["target_labels_used"] is False
    if not quality:
        assert result["by_repetition"][-1]["counts"] == {str(i): 5 for i in range(1, 10)}
    else:
        assert restored.epoch_quality.record() == predictor.epoch_quality.record()
        assert result["scheduled_trials"] == 45




def test_unified_cli_fits_full_calibration_once_including_short_recording(tmp_path):
    calibration = write_bids_session(tmp_path / "cal", session_id="cal", target=3,
        started_utc="2026-09-14T10:00:00+08:00", blocks=(tuple(range(1, 10)) * 5,))
    validation = write_bids_session(tmp_path / "val", session_id="val", target=5,
        started_utc="2026-09-15T10:00:00+08:00", blocks=(tuple(range(1, 10)),))
    dataset = load_brainsync_sessions((calibration, validation))
    recipe = TargetRecipe("unified_scratch", initialization="scratch", batch_size=64,
        bn_buffers="calibration", bn_affine="trainable", dropout="off", epochs=1,
        seed=20260906, candidate_loss_weight=1.)
    stopping = UnifiedTrainingBudget(target_epochs=1).record()
    plan = preflight_calibrated_predictor(dataset, selector_mode="unified", transfer_policy=recipe,
        unified_training_budget=stopping, selection_protocol="brainsync_operational_r5")
    assert plan["execution_ready"], plan["execution_blockers"]
    assert plan["selection_protocol"] == "fixed_full_calibration"
    assert not plan["inner_selection_planned"]
    recipe_path, policy_path = tmp_path / "recipe.json", tmp_path / "policy.json"
    recipe_path.write_text(json.dumps(recipe.record()), encoding="utf-8")
    policy_path.write_text(json.dumps(stopping), encoding="utf-8")
    output = tmp_path / "dry-run"
    artifact = tmp_path / "person.pt"
    arguments = ["--task", "calibrate", "--session-dir", str(calibration), "--session-dir", str(validation),
                 "--selector-mode", "unified", "--calibration-recipe", str(recipe_path),
                 "--training-budget", str(policy_path), "--model-artifact", str(artifact)]
    main([*arguments, "--output-dir", str(output), "--dry-run"])
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["execution_ready"] and summary["selection_protocol"] == plan["selection_protocol"]
    assert not artifact.exists()
    completed = tmp_path / "completed"
    main([*arguments, "--output-dir", str(completed)])
    result = json.loads((completed / "summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    selection = result["internal_validation"]
    assert selection["mode"] == "fixed_unified_program"
    assert selection["scoring_protocol"] == plan["selection_protocol"]
    assert selection["fit_execution"]["fit_calls"] == 1
    restored = load_calibrated_predictor(artifact)
    assert restored.metadata["selection"]["scoring_protocol"] == plan["selection_protocol"]


def test_unified_cli_rejects_explicit_old_recipe_metric(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--task", "calibrate", "--selector-mode", "unified", "--selection-protocol", "binary_auc",
              "--model-artifact", str(tmp_path / "unused.pt"), "--output-dir", str(tmp_path / "unused")])
    assert error.value.code == 2
    assert not (tmp_path / "unused").exists()
