from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from data.brainsync import load_brainsync_inference_session, load_brainsync_sessions
from data.domain import adapt_observed_montage
from experiments.brainsync_model_task import inference_record
from experiments.brainsync_model_task import main as model_task
from experiments.fit_brainsync_eog import fit_brainsync_eog_sessions
from tests.test_brainsync import _write_bids_session, _write_current_checkpoint
from transfer.binary import target_shrinkage_scores
from transfer.calibrated import (
    CalibrationRecipe,
    fit_calibrated_predictor,
    load_calibrated_predictor,
    save_calibrated_predictor,
)


@pytest.fixture
def sessions(tmp_path):
    roots = []
    for index, target in enumerate((1, 3, 5)):
        roots.append(_write_bids_session(
            tmp_path / f"cal-{index}", session_id=f"cal-{index}", target=target,
            started_utc=f"2026-09-06T08:{index * 3:02d}:00Z",
            blocks=(tuple(range(1, 10)) * 3,),
        ))
    testing = _write_bids_session(
        tmp_path / "test", session_id="new-test", target=7,
        started_utc="2026-09-06T08:12:00Z", blocks=(tuple(range(1, 10)) * 3,),
    )
    return roots, testing


def test_lda_persistence_matches_shared_estimator_and_ignores_test_labels(tmp_path, sessions):
    roots, testing = sessions
    calibration = load_brainsync_sessions(roots)
    target = load_brainsync_inference_session(testing)
    predictor = fit_calibrated_predictor(calibration)
    expected = target_shrinkage_scores(calibration.X, calibration.y, target.X,
                                       sfreq=calibration.preprocessing.sfreq)
    actual = predictor.predict_logit(target)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-6)
    artifact = save_calibrated_predictor(tmp_path / "model.pt", predictor)
    restored = load_calibrated_predictor(artifact)
    np.testing.assert_array_equal(restored.predict_logit(target), actual)
    assert not restored.metadata["source_pretraining_used"]
    assert restored.metadata["selection"]["validation_auc"] is None
    result = inference_record(restored, target, repetitions=2)
    assert result["target_labels_used"] is False
    assert set(result["candidate_scores"]) == set(map(str, range(1, 10)))
    changed = replace(target, y=np.zeros(target.n_epochs, dtype=np.int64))
    np.testing.assert_array_equal(restored.predict_logit(changed), actual)
    with pytest.raises(FileExistsError):
        save_calibrated_predictor(artifact, restored)


def test_source_selection_refits_full_calibration_and_round_trips(tmp_path, sessions):
    roots, testing = sessions
    dataset = load_brainsync_sessions(roots)
    source = _write_current_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="f" * 64)
    predictor = fit_calibrated_predictor(
        dataset, source_checkpoint=source,
        recipes=(CalibrationRecipe("small", 1, 1e-4), CalibrationRecipe("larger", 2, 1e-3)),
        batch_size=16,
    )
    selection = predictor.metadata["selection"]
    assert selection["training_sessions"] == ["cal-0", "cal-1"]
    assert selection["validation_sessions"] == ["cal-2"]
    assert selection["refit_rows"] == dataset.n_epochs
    assert selection["fit_scope"]["training_rows"] == dataset.n_epochs
    assert selection["fit_scope"]["validation_rows"] == 0
    assert selection["fit_scope"]["prefix_mask_explicit"]
    assert predictor.metadata["source_pretraining_used"]
    np.testing.assert_array_equal(predictor.state["input_std"].numpy(), np.ones((1, 7, 1)))
    target = load_brainsync_inference_session(testing)
    before = predictor.predict_logit(target)
    saved = save_calibrated_predictor(tmp_path / "adapted.pt", predictor)
    restored = load_calibrated_predictor(saved)
    np.testing.assert_array_equal(restored.predict_logit(target), before)
    assert torch.load(saved, weights_only=True)["metadata"]["model_kind"] == "source_initialized_n2p3"


def test_predictor_rejects_identity_time_domain_reuse_and_corruption(tmp_path, sessions):
    roots, testing = sessions
    calibration = load_brainsync_sessions(roots)
    predictor = fit_calibrated_predictor(calibration)
    with pytest.raises(ValueError, match="reused"):
        predictor.validate_inference_scope(load_brainsync_inference_session(roots[0]))
    target = load_brainsync_inference_session(testing)
    early = copy.deepcopy(target)
    early.metadata["session_start_timestamp_s"] = 0.0
    with pytest.raises(ValueError, match="before calibration"):
        predictor.validate_inference_scope(early)
    wrong_reference = copy.deepcopy(target)
    wrong_reference.provenance["source_reference"] = "unrelated-reference"
    with pytest.raises(ValueError, match="reference"):
        predictor.predict_logit(wrong_reference)
    wrong_simulation = copy.deepcopy(target)
    wrong_simulation.provenance["simulation_status"] = "declared_mock"
    with pytest.raises(ValueError, match="simulation status"):
        predictor.predict_logit(wrong_simulation)
    wrong_person_dir = _write_bids_session(
        tmp_path / "other", session_id="other-person", target=2,
        subject_id="P002", started_utc="2026-09-06T09:00:00Z",
    )
    with pytest.raises(ValueError, match="different participant"):
        predictor.predict_logit(load_brainsync_inference_session(wrong_person_dir))
    artifact = save_calibrated_predictor(tmp_path / "model.pt", predictor)
    payload = torch.load(artifact, weights_only=True)
    payload["state"]["coefficient"][0] += 1
    torch.save(payload, tmp_path / "corrupt.pt")
    with pytest.raises(ValueError, match="state digest"):
        load_calibrated_predictor(tmp_path / "corrupt.pt")


@pytest.mark.parametrize("ocular", [False, True])
def test_cli_calibrate_then_infer_with_explicit_reference(tmp_path, sessions, ocular):
    roots, testing = sessions
    artifact = tmp_path / "personal.pt"
    calibration_output = tmp_path / "calibration-result"
    arguments = ["--task", "calibrate", "--model-artifact", str(artifact),
                 "--output-dir", str(calibration_output), "--reference-channel", "Cz"]
    for root in roots:
        arguments.extend(["--session-dir", str(root)])
    if ocular:
        arguments.append("--ocular-correction")
    model_task(arguments)
    result = json.loads((calibration_output / "summary.json").read_text())
    assert result["status"] == "completed"
    assert result["model_kind"] == "target_only_shrinkage_lda"
    assert result["ocular_correction_applied"] is ocular
    assert load_calibrated_predictor(artifact).metadata["input_contract"]["n_channels"] == 6
    pending = json.loads((testing / "session.json").read_text())
    pending["target_label"] = {"status": "pending"}
    pending["experiment"]["thought_digit"] = None
    pending["experiment"]["target_label_status"] = "pending_post_experiment_confirmation"
    (testing / "session.json").write_text(json.dumps(pending), encoding="utf-8")
    inference_output = tmp_path / "inference-result"
    model_task(["--task", "infer", "--model-artifact", str(artifact),
                "--output-dir", str(inference_output), "--session-dir", str(testing)])
    inference = json.loads((inference_output / "summary.json").read_text())
    assert inference["status"] in {"predicted", "abstained"}
    assert inference["requested_repetitions"] == 3
    assert inference["target_labels_used"] is False
    assert len(inference["by_repetition"]) == 3
    inspected = tmp_path / "inspected"
    model_task(["--task", "inspect", "--model-artifact", str(artifact), "--output-dir", str(inspected)])
    inspected_record = json.loads((inspected / "summary.json").read_text())
    assert inspected_record["participant_key"] == result["participant_key"]
    assert inspected_record["subject_id"] == "P001"
    assert inspected_record["calibration_trials"] == result["calibration_trials"]


def test_requested_repetition_shortage_abstains_without_shrinking_candidates(sessions):
    roots, testing = sessions
    predictor = fit_calibrated_predictor(load_brainsync_sessions(roots))
    output = inference_record(predictor, load_brainsync_inference_session(testing), repetitions=4)
    assert output["prediction"] is None
    assert output["status"] == "abstained"
    assert output["reason"] == "incomplete_candidate_coverage"
    assert len(output["candidate_scores"]) == 9


def test_incompatible_source_is_rejected_instead_of_silent_lda_fallback(tmp_path, sessions):
    roots, _ = sessions
    data = load_brainsync_sessions(roots)
    projected = adapt_observed_montage(
        data, tuple(name for name in data.channel_names if name != "CZ"), reference_channel="Cz"
    )
    source = _write_current_checkpoint(tmp_path / "source.pt", projected, source_snapshot_sha256="f" * 64)
    with pytest.raises(ValueError, match="source checkpoint is incompatible"):
        fit_calibrated_predictor(data, source_checkpoint=source)


def test_neural_eog_is_frozen_from_inner_training_sessions_only(tmp_path, sessions):
    roots, testing = sessions
    ocular = fit_brainsync_eog_sessions(tuple(roots[:2]), reference_channel="Cz")
    calibration = load_brainsync_sessions(roots, ocular_model=ocular)
    source = _write_current_checkpoint(tmp_path / "source.pt", calibration, source_snapshot_sha256="e" * 64)
    predictor = fit_calibrated_predictor(
        calibration, source_checkpoint=source, recipes=(CalibrationRecipe("probe", 1, 1e-4),)
    )
    assert predictor.metadata["input_request"]["ocular_model"] == ocular.record()
    assert predictor.metadata["selection"]["refit_rows"] == calibration.n_epochs
    later = load_brainsync_inference_session(testing, ocular_model=ocular)
    assert np.isfinite(predictor.predict_logit(later)).all()
    contaminated = fit_brainsync_eog_sessions(tuple(roots), reference_channel="Cz")
    with pytest.raises(ValueError, match="outside the declared calibration budget"):
        fit_calibrated_predictor(
            load_brainsync_sessions(roots, ocular_model=contaminated), source_checkpoint=source,
            recipes=(CalibrationRecipe("probe", 1, 1e-4),),
        )


def test_cli_failure_is_structured_and_does_not_create_a_model(tmp_path, sessions):
    roots, _ = sessions
    artifact, output = tmp_path / "model.pt", tmp_path / "failed-task"
    with pytest.raises(SystemExit) as stopped:
        model_task(["--task", "calibrate", "--session-dir", str(roots[0]),
                    "--source-checkpoint", str(tmp_path / "source.pt"),
                    "--model-artifact", str(artifact), "--output-dir", str(output)])
    assert stopped.value.code == 1
    result = json.loads((output / "summary.json").read_text())
    assert result["status"] == "failed"
    assert result["error"]["type"] == "ValueError"
    assert "at least two" in result["error"]["message"]
    assert not artifact.exists()


def test_one_session_fixed_lda_calibrates_and_infers_without_validation_claim(tmp_path, sessions):
    roots, testing = sessions
    artifact = tmp_path / "one-session.pt"
    calibration_output = tmp_path / "one-session-calibration"
    model_task(["--task", "calibrate", "--session-dir", str(roots[0]),
                "--model-artifact", str(artifact), "--output-dir", str(calibration_output)])
    summary = json.loads((calibration_output / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["model_kind"] == "target_only_shrinkage_lda"
    assert len(summary["calibration_sessions"]) == 1
    assert summary["target_trials"] == 3
    assert summary["selected_recipe"] is None
    assert summary["internal_validation"] == {
        "mode": "fixed_fallback_no_recipe_selection", "validation_auc": None,
    }
    manifest = json.loads((testing / "session.json").read_text())
    manifest["target_label"] = {"status": "pending"}
    manifest["experiment"]["thought_digit"] = None
    manifest["experiment"]["target_label_status"] = "pending_post_experiment_confirmation"
    (testing / "session.json").write_text(json.dumps(manifest), encoding="utf-8")
    inference_output = tmp_path / "one-session-inference"
    model_task(["--task", "infer", "--session-dir", str(testing),
                "--model-artifact", str(artifact), "--output-dir", str(inference_output)])
    result = json.loads((inference_output / "summary.json").read_text())
    assert result["status"] in {"predicted", "abstained"}
    assert result["target_labels_used"] is False
    assert result["requested_repetitions"] == 3
    assert set(result["candidate_scores"]) == set(map(str, range(1, 10)))
