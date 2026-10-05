"""Versioned personal QC follows the same calibration and artifact lifecycle."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from data.artifact import FoldLocalArtifactModel, FoldLocalArtifactPolicy
from data.artifact_sidecar import artifact_model_from_record
from data.brainsync import load_brainsync_sessions
from data.continuous_qc import save_continuous_qc
from data.personal_quality import PersonalEpochQualityModel, PersonalEpochQualityPolicy
from data.qc_features import compute_epoch_qc_features
from experiments.brainsync_model_task import main as model_task
from experiments.run_brainsync_recovery import quality_policy_from_plan
from research.contracts import semantic_sha256
from tests.support.brainsync import write_bids_session
from tests.support.io import read_json
from tests.support.quality import (
    fit_raw_quality,
    masked_bids_calibration,
    personal_dataset,
    quality_pair,
)
from transfer.calibrated import (
    CALIBRATED_PREDICTOR_SCHEMA,
    CalibratedPredictor,
    CalibrationRecipe,
    fit_calibrated_predictor,
    load_calibrated_predictor,
    preflight_calibrated_predictor,
    save_calibrated_predictor,
)
from transfer.calibration_candidates import (
    CalibrationCandidateFitCache,
    fit_calibration_candidate,
    select_calibration_candidate,
    waveform_lda_recipe,
)
from transfer.quality import (
    apply_epoch_quality,
    epoch_quality_from_record,
    fit_epoch_quality,
    load_epoch_quality,
    save_epoch_quality,
    validate_personal_epoch_raw_quality,
)


@pytest.mark.parametrize("policy,model_type", [
    (FoldLocalArtifactPolicy(), FoldLocalArtifactModel),
    (PersonalEpochQualityPolicy(), PersonalEpochQualityModel),
])
def test_quality_json_dispatch_roundtrip_preserves_algorithm_and_digest(tmp_path, policy, model_type):
    dataset = personal_dataset()
    model = fit_epoch_quality(dataset, np.arange(dataset.n_epochs) < 40, policy)
    record = model.record()
    restored = load_epoch_quality(save_epoch_quality(tmp_path / "quality.json", model))
    assert isinstance(restored, model_type)
    assert restored.record() == record
    assert semantic_sha256(restored.record()) == semantic_sha256(record)
    np.testing.assert_array_equal(
        restored.transform(dataset.X).trial_channel_mask, model.transform(dataset.X).trial_channel_mask,
    )


@pytest.mark.parametrize("changed", [
    {"method": "personal_epoch_signal_evidence"}, {"method": "unknown"},
    {"ptp_normalization": "absolute_volts"}, {"schema": "unknown/1"},
])
def test_public_loader_never_reinterprets_threshold_semantics(changed):
    dataset = personal_dataset()
    record = fit_epoch_quality(dataset, np.arange(dataset.n_epochs) < 40, FoldLocalArtifactPolicy()).record()
    record.update(changed)
    with pytest.raises(ValueError, match="method|normalization"):
        artifact_model_from_record(record)


def test_dispatch_rejects_missing_unknown_or_mismatched_method():
    for record in ({}, {"method": "unknown"}, {"method": "personal_epoch_signal_evidence", "schema": "wrong/1"}):
        with pytest.raises(ValueError):
            epoch_quality_from_record(record)


def test_recovery_plan_dispatch_keeps_legacy_defaults_and_personal_method():
    for policy in (FoldLocalArtifactPolicy(), PersonalEpochQualityPolicy()):
        assert quality_policy_from_plan(json.loads(json.dumps(asdict(policy)))) == policy
    with pytest.raises(ValueError, match="method"):
        quality_policy_from_plan({"method": "unknown"})


def test_personal_frozen_scope_remains_bound_to_fit_values_and_inner_training():
    dataset = personal_dataset()
    training = np.arange(dataset.n_epochs) < 40
    frozen = fit_epoch_quality(dataset, training, PersonalEpochQualityPolicy())
    with pytest.raises(ValueError, match="outside.*training"):
        select_calibration_candidate(dataset, (waveform_lda_recipe(),), calibration_mask=training,
            epoch_quality_model=frozen, recording_order=("session-a", "session-b", "session-c"))
    changed = dataset.X.copy()
    changed[0, 0, 0] += 1e-6
    with pytest.raises(ValueError, match="fit data digest"):
        fit_calibration_candidate(replace(dataset, X=changed), waveform_lda_recipe(), training,
                                  epoch_quality_model=frozen)


def test_candidate_cache_binds_training_qc_features_without_using_held_out_features():
    dataset, _, policy = quality_pair()
    dataset = replace(dataset, qc_features=compute_epoch_qc_features(dataset.X, channel_mask=dataset.channel_mask))
    training = np.arange(dataset.n_epochs) < 40
    cache = CalibrationCandidateFitCache()

    def fit(data):
        return cache.fit(data, waveform_lda_recipe(), training, phase="inner", epoch_quality_policy=policy)

    first = fit(dataset)
    held_out = copy.deepcopy(dataset)
    held_out.qc_features.relative_ptp[-1, 0] += 1.
    assert fit(held_out).fit_history["fit_reuse"]["cache_hit"]
    metadata_only = replace(dataset, metadata=dataset.metadata.assign(operator_note="unchanged signal"))
    assert fit(metadata_only).fit_history["fit_reuse"]["cache_hit"]
    changed = copy.deepcopy(dataset)
    changed.qc_features.relative_ptp[0, 0] += 1.
    other = fit(changed)
    assert not other.fit_history["fit_reuse"]["cache_hit"]
    assert first.fit_history["fit_reuse"]["fit_key_sha256"] != other.fit_history["fit_reuse"]["fit_key_sha256"]


def _continuous_bids_calibration(tmp_path):
    dataset = masked_bids_calibration(tmp_path)
    roots = dataset.provenance["session_dirs"]
    return load_brainsync_sessions(roots, continuous_qc_model=fit_raw_quality(roots[0]))


def _quality_predictor(dataset, policy):
    return fit_calibrated_predictor(
        dataset,
        selector_mode="no_transfer",
        recipes=(
            CalibrationRecipe(
                "waveform",
                1,
                1e-3,
                initialization="waveform_lda",
                bn_buffers="source",
                bn_affine="frozen",
                dropout="off",
            ),
        ),
        epoch_quality_policy=policy,
    )


@pytest.mark.parametrize(
    "build_dataset,policy",
    [
        (masked_bids_calibration, FoldLocalArtifactPolicy()),
        (_continuous_bids_calibration, PersonalEpochQualityPolicy()),
    ],
    ids=["public-quality", "personal-quality"],
)
def test_calibrated_models_reload_the_declared_quality_and_all_unavailable_predictions(
    tmp_path, build_dataset, policy
):
    dataset = build_dataset(tmp_path)
    predictor = _quality_predictor(dataset, policy)
    path = save_calibrated_predictor(tmp_path / "model.pt", predictor)
    assert torch.load(path, weights_only=True)["schema"] == CALIBRATED_PREDICTOR_SCHEMA
    restored = load_calibrated_predictor(path)
    assert restored.epoch_quality.record() == predictor.epoch_quality.record()
    np.testing.assert_array_equal(restored.predict_logit(dataset), predictor.predict_logit(dataset))
    flat = replace(dataset, X=np.zeros_like(dataset.X), qc_features=None)
    assert np.isnan(restored.predict_logit(flat)).all()


def test_public_quality_remains_readable_in_legacy_model_packages(tmp_path):
    predictor = _quality_predictor(masked_bids_calibration(tmp_path), FoldLocalArtifactPolicy())
    path = save_calibrated_predictor(tmp_path / "model.pt", predictor)
    payload = torch.load(path, weights_only=True)
    for schema in ("n2p3_calibrated_predictor/1", "n2p3_calibrated_predictor/3"):
        payload["schema"] = schema
        old_path = tmp_path / (schema.rsplit("/", 1)[1] + ".pt")
        torch.save(payload, old_path)
        assert (
            load_calibrated_predictor(old_path).epoch_quality.record()
            == predictor.epoch_quality.record()
        )


def test_personal_quality_requires_raw_evidence_and_current_model_package(tmp_path):
    dataset = _continuous_bids_calibration(tmp_path)
    predictor = _quality_predictor(dataset, PersonalEpochQualityPolicy())
    without_raw = replace(
        dataset, provenance={**dataset.provenance, "continuous_artifact_processing": {}}
    )
    with pytest.raises(ValueError, match="continuous QC /4"):
        apply_epoch_quality(without_raw, predictor.epoch_quality)
    metadata = copy.deepcopy(predictor.metadata)
    metadata["input_contract"]["continuous_artifact_processing"] = {}
    with pytest.raises(ValueError, match="continuous QC /4"):
        CalibratedPredictor(metadata, predictor.state)
    payload = torch.load(
        save_calibrated_predictor(tmp_path / "model.pt", predictor), weights_only=True
    )
    for schema in ("n2p3_calibrated_predictor/1", "n2p3_calibrated_predictor/3"):
        payload["schema"] = schema
        old_path = tmp_path / (schema.rsplit("/", 1)[1] + ".pt")
        torch.save(payload, old_path)
        with pytest.raises(ValueError, match="legacy.*new epoch quality"):
            load_calibrated_predictor(old_path)


def test_cli_epoch_qc_preflight_and_fit_use_the_same_personal_policy(tmp_path, monkeypatch):
    roots = [write_bids_session(tmp_path / f"cal-{i}", session_id=f"cal-{i}", target=i + 1,
        started_utc=f"2026-09-06T08:{i * 3:02d}:00Z", blocks=(tuple(range(1, 10)) * 5,)) for i in range(2)]
    profile = save_continuous_qc(tmp_path / "continuous.json", fit_raw_quality(roots[0]))
    args = ["--task", "calibrate", "--epoch-qc", "--reference-mode", "native",
            "--continuous-qc-model", str(profile),
            *[value for root in roots for value in ("--session-dir", str(root))]]
    output = tmp_path / "preflight"

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight fitted a QC model")

    with monkeypatch.context() as context:
        context.setattr(PersonalEpochQualityPolicy, "fit", forbidden)
        model_task([*args, "--dry-run", "--model-artifact", str(tmp_path / "dry.pt"), "--output-dir", str(output)])
    plan = read_json(output / "summary.json")
    assert plan["planned_epoch_quality"]["method"] == "personal_epoch_signal_evidence"
    assert plan["planned_epoch_quality"]["fitted"] is False
    assert not (tmp_path / "dry.pt").exists()
    fitted_output = tmp_path / "fitted"
    model_task([*args, "--model-artifact", str(tmp_path / "personal.pt"), "--output-dir", str(fitted_output)])
    summary = read_json(fitted_output / "summary.json")
    assert summary["epoch_qc_method"] == plan["planned_epoch_quality"]["method"]
    assert summary["epoch_qc_schema"] == "n2p3_personal_epoch_quality/1"
    assert summary["status"] == "completed"


def test_brainsync_new_epoch_policy_rejects_missing_continuous_quality_at_all_fit_entries(tmp_path):
    root = write_bids_session(tmp_path / "cal", session_id="cal", target=1,
        started_utc="2026-09-06T08:00:00Z", blocks=(tuple(range(1, 10)) * 5,))
    dataset = load_brainsync_sessions([root])
    policy = PersonalEpochQualityPolicy()
    training = np.ones(dataset.n_epochs, dtype=bool)
    for call in (
        lambda: fit_epoch_quality(dataset, training, policy),
        lambda: fit_calibration_candidate(dataset, waveform_lda_recipe(), training, epoch_quality_policy=policy),
        lambda: fit_calibrated_predictor(dataset, epoch_quality_policy=policy),
    ):
        with pytest.raises(ValueError, match="requires applied.*continuous QC /4"):
            call()
    plan = preflight_calibrated_predictor(dataset, epoch_quality_policy=policy)
    assert not plan["execution_ready"]
    assert any("--continuous-qc-model" in reason for reason in plan["execution_blockers"])
    output = tmp_path / "dry"
    model_task(["--task", "calibrate", "--epoch-qc", "--dry-run", "--session-dir", str(root),
                "--model-artifact", str(tmp_path / "missing.pt"), "--output-dir", str(output)])
    assert not read_json(output / "summary.json")["execution_ready"]
    assert not (tmp_path / "missing.pt").exists()


def test_new_epoch_policy_requires_raw_constant_detection_and_intact_profile(tmp_path):
    root = write_bids_session(tmp_path / "cal", session_id="cal", target=1,
        started_utc="2026-09-06T08:00:00Z", blocks=(tuple(range(1, 10)) * 5,))
    model = fit_raw_quality(root)
    record = model.record()
    valid = {"status": "applied", "model": record, "model_digest": semantic_sha256(record)}
    validate_personal_epoch_raw_quality(valid, PersonalEpochQualityPolicy())
    legacy = replace(model, spec=replace(model.spec, flatline_min_duration_s=None)).record()
    invalid = [
        {}, {**valid, "status": "not_applied"}, {**valid, "model_digest": "0" * 64},
        {"status": "applied", "model": legacy, "model_digest": semantic_sha256(legacy)},
    ]
    for processing in invalid:
        with pytest.raises(ValueError, match="continuous QC /4"):
            validate_personal_epoch_raw_quality(processing, PersonalEpochQualityPolicy())
        validate_personal_epoch_raw_quality(processing, FoldLocalArtifactPolicy())
